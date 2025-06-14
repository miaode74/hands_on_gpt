#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import PretrainedConfig
from typing import List


# In[2]:


import math
import struct
import inspect
import time
from typing import Any, Optional, Tuple, List
import numpy as np
from torch import nn 
from transformers import PreTrainedModel 
import torch


# In[3]:


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)).type_as(x)


# In[4]:


def precompute_pos_cis(dim: int, end: int = int(32*1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0,dim,2)[: (dim//2)].float() / dim))

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t,freqs).float()

    pos_cis = torch.polar(torch.ones_like(freqs),freqs)
    return pos_cis 


# In[5]:


def apply_rotray_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis,x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i==1 or i == ndim -1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1,2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1,2))
    pos_cis  = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# In[6]:


def repeat_kv(x: torch.Tensor, n_rep: int):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:,:,None,:].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs,slen,n_kv_heads * n_rep, head_dim)
    )


# In[7]:


class LMConfig(PretrainedConfig):
    model_type = "miaodeeai"
    def __init__(self,
                 dim: int = 512,
                 n_layers: int = 1,
                 n_heads: int = 8,
                 n_kv_heads: int = 2,
                 vocab_size: int = 6400,
                 hidden_dim: int = None,
                 multiple_of: int = 64,
                 norm_eps: float = 1e-5,
                 max_seq_len: int = 8192,
                 rope_theta: int = 1e6,
                 dropout: float = 0.0,
                 flash_attn: bool = True,
                 ###底下的是使用 MoE 的时候才需要的参数
                 use_moe: bool = False,
                 num_experts_per_tok: int =2,
                 num_routed_experts: int=4,
                 n_shared_experts: bool = True,
                 scoring_func: str = 'softmax',
                 aux_loss_alpha: float = 0.1,
                 seq_aux: bool = True,
                 norm_topk_prob: bool= True,
                 **kwargs,
                 ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim 
        self.multiple_of = multiple_of
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.norm_eps = norm_eps
### 这里是moe相关的参数
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.num_routed_experts = num_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        super().__init__(**kwargs)


# In[8]:


import json
import random
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os 
import ast


# In[9]:


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, data_path: str):
        samples = []
        with open(data_path,'r', encoding = 'utf-8') as f:
            for line_num, line in enumerate(f,1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples 
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        sample = self.samples[index]

        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        inputs_ids = encoding['input_ids'].squeeze()
        loss_mask = (inputs_ids != self.tokenizer.pad_token_id)
        X = torch.tensor(inputs_ids[:-1], dtype=torch.long)
        Y = torch.tensor(inputs_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X,Y,loss_mask
    


# In[10]:


# xq,xk = torch.randn((2,16,4,64)), torch.randn((2,16,4,64))
# pos_cis = precompute_pos_cis(64,16)
# print(f"pos_cis shape: {pos_cis.shape}, pos_cis[0,0]: {pos_cis[0,0]}")


# In[11]:


# xq_rope, xk_rope = apply_rotray_emb(xq, xk, pos_cis)
# print(f"xq_rope shape: {xq_rope.shape}, xk_rope shape: {xk_rope.shape}")


# In[12]:


from typing import Any, Optional, Tuple, List 
import torch.nn as nn
import math 
import torch
import  torch.nn.functional as F

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep ==1:
        return x
    return (
        x[:,:,:,None,:].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# In[13]:


class Attention(nn.Module):
    def __init__(self,args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % args.n_kv_heads == 0
        
        self.n_local_heads = args.n_heads 
        self.n_local_kv_heads = args.n_kv_heads 
        self.n_rep = self.n_local_heads // self.n_local_kv_heads 
        self.head_dim = args.dim // args.n_heads 

        # q,k,v, o projection

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        mask = torch.full((1,1,args.max_seq_len, args.max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask, persistent=False)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                use_cache=False):
        bsz, seq_len, _ = x.shape

        ####Forward Q,K,V && RoPE #### 
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotray_emb(xq, xk, pos_cis)

        ###E KV Cache ####
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1,2),
            repeat_kv(xk, self.n_rep).transpose(1,2),
            repeat_kv(xv, self.n_rep).transpose(1,2)
        )

        #### Scaled Dot Production ####
        if self.flash and seq_len !=1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(xq,xk,xv,
                                                    attn_mask = None,
                                                    dropout_p = dropout_p,
                                                    is_causal = True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:,:,:seq_len,:seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1,2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = config.dim * 4
            hidden_dim = int(2* hidden_dim /3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of -1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return x


# In[16]:


# ffn = FeedForward(LMConfig(n_layers=2))
# x = torch.randn((4,16,512)) # batch_size, seq_len, embed_dim
# output = ffn(x)
# print(f"输入张量x 的形状 {x.shape}, 输出张量 output 的形状: {output.shape}")


# In[17]:


class GPTBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.attention = Attention(config)
        
        self.layer_id = layer_id 
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn , past_kv= self.attention(self.attention_norm(x),pos_cis,past_key_value= past_key_value, use_cache=use_cache)

        h = x + h_attn 
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv



# In[ ]:


from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class cqxGPT(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig):

        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size , self.n_layers = params.vocab_size, params.n_layers
        self.token_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([GPTBlock(1,params) for layer in range(self.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.token_embeddings.weight = self.output.weight
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(dim=params.dim//params.n_heads, theta = params.rope_theta),
            persistent=False
        )
        self.OUT = CausalLMOutputWithPast()
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        past_key_values = past_key_values or [None] * self.n_layers
        start_pos = args.get('start_pos', 0)
        h = self.dropout(self.token_embeddings(input_ids))

        pos_cis = self.pos_cis[start_pos: start_pos + input_ids.size(1)]
        past_kvs = []

        for l , layer  in enumerate(self.layers):
            print(f"第 {l} 层的输入张量 h 的形状: {h.shape}, pos_cis 的形状: {pos_cis.shape}")

            h, past_kv = layer(h, pos_cis=pos_cis, past_key_value=past_key_values[l], use_cache=use_cache)
            print(f"finished layer {l}, output h 的形状: {h.shape}, size_cache_k 的形状: {past_kv[0].shape},size_cache_v = {past_kv[1].shape}")
            past_kvs.append(past_kv)

        print(f"forward operation completed, num_kv_cache = {len(past_kvs)}")
        logits = self.output(self.norm(h))
        aux_loss = 0
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("past_key_values", past_kvs)
        return self.OUT 
    
    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, eos_token_id= 2, max_new_tokens: int = 512, temperature=0.75, top_p=0.90, stream=False, rp=1, use_cache=True, pad_token_id=0, **args):

        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, pad_token_id, **args)
        
        generated = []

        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:,-1:] for tokens in out]
            print(f"new token list : {tokens_list}")
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad 
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat([seq, torch.full((1,max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)], dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)
    def _stream(self,  input_ids,  eos_token_id,  max_new_tokens,  temperature,  top_p,  rp,  use_cache,  **args):
        start,  first_seq,  past_kvs = input_ids.shape[1],  True,  None
        new_token_idx = 0 #  new token 计数器
        while input_ids.shape[1] < max_new_tokens - 1:
            print(f'gernerating new token: idx = {start + new_token_idx}')
            if first_seq or not use_cache: # 若第一次生成序列 or 无 KV Cache,  每次生成传入整个 token id 序列
                out,  first_seq = self(input_ids,  past_key_values=past_kvs,  use_cache=use_cache,  **args),  False
            else: # 若非第一次生成 and 有 KV Cache, 每次传入最后一个 token id 与 KV Cache 进行推理加速
                out = self(input_ids[:,  -1:],  past_key_values=past_kvs,  use_cache=use_cache, 
                           start_pos=input_ids.shape[1] - 1,  **args)
            logits,  past_kvs = out.logits[:,  -1,  :],  out.past_key_values # logits.shape: (batch_size,  seq_len,  embed_dim), 获取最后一位 logits
            logits[:,  list(set(input_ids.tolist()[0]))] /= rp # 对生成 token 进行惩罚, 降低后续重复生成几率
            logits /= (temperature + 1e-9) # 调整温度, 控制生成多样性
            if top_p is not None and top_p < 1.0: # top-p 采样
                sorted_logits,  sorted_indices = torch.sort(logits,  descending=True,  dim=-1)
                sorted_probs = F.softmax(sorted_logits,  dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs,  dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:,  1:] = sorted_indices_to_remove[:,  :-1].clone()
                sorted_indices_to_remove[:,  0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1,  sorted_indices,  sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits,  dim=-1),  num_samples=1) # 从保留的 token 中采样
            input_ids = torch.cat((input_ids,  input_ids_next),  dim=1)
            new_token_idx += 1
            yield input_ids[:,  start:]
            if input_ids_next.item() == eos_token_id:
                break



# In[21]:


# In[22]:


import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler 
from contextlib import nullcontext

from transformers import AutoTokenizer


# In[23]:


warnings.filterwarnings("ignore")


# In[24]:


class args:
    epochs: int =1
    batch_size: int = 2
    learning_rate: float = 5e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    use_wandb = False
    wandb_project = "cqxGPT"
    num_workers: int = 8
    ddp: bool = True
    accumulation_steps: int = 1
    grad_clip: float = 1.0 
    warmup_iters: int=0
    log_interval: int = 1
    # save_interval: int = 1000
    local_rank: int = 0
    dim: int = 512
    n_layers: int = 2
    max_seq_len: int = 512
    use_moe: bool = False 
    data_path: str = "/cpfs/user/boyuan/verl_workspace/hw1_code/hw1-code/pretrain/minimind/dataset/pretrain_hq.jsonl"


# In[25]:


args.device


# In[39]:


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("/cpfs/user/boyuan/verl_workspace/hw1_code/hw1-code/pretrain/minimind/models", trust_remote_code=True)
    model = cqxGPT(lm_config).to(args.device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M trainable parameters.")
    return model, tokenizer


# In[40]:


# lm_config = LMConfig(args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
# model, tokenizer = init_model(lm_config)


# In[41]:


# train_dataset = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)

# train_loader = DataLoader(
#     train_dataset,
#     batch_size = args.batch_size,
#     pin_memory=True,
#     drop_last=False,
#     shuffle=False,
#     num_workers=8,
# )

# print(f"Train dataset size: {len(train_dataset)}, Train loader size: {len(train_loader)}, Vocab size: {tokenizer.vocab_size}, Max sequence length: {lm_config.max_seq_len}")


# In[42]:


# loader = iter(train_loader)


# In[ ]:


def get_lr(current_step: int, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "bfloat16"))
# optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) 

# device_type = "cuda" if "cuda" in args.device else "cpu"
# ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)


# In[49]:


def train_epoch(epoch):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X,Y,loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        with ctx:
            res = model(X,use_cache=True)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)),
                            Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
        
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            print(f"Epoch: [{epoch+1}/{args.epochs}] {step}/{iter_per_epoch},  loss:{loss.item() * args.accumulation_steps :.3f} lr{optimizer.param_groups[-1]['lr']}, epoch time: {spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60}")


# In[55]:

if __name__ == "main":
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)