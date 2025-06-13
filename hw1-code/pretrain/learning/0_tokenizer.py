import os 
import random
import json
random.seed(42)
from tokenizers import (decoders,models,normalizers,pre_tokenizers,processors,trainers,Tokenizer)

def train_tokenizer():
    def read_texts_from_jsonl(file_path):
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
    
    data_path = "/cpfs/user/boyuan/verl_workspace/hw1_code/hw1-code/pretrain/dataset/pretrain_hq.jsonl"
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special_tokens = ["<|endoftext|>","<|im_start|>","<|im_end|>"]

    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    tokenizer_dir = "/cpfs/user/boyuan/verl_workspace/hw1_code/hw1-code/pretrain/models/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "add_tokens_decoder":{
            "0":{
                "content" : "<|endoftext|>",
                "lstrip" : False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
                },
    },
    "additional_special_tokens": [],
    "bos_token": "<|im_start|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "legacy": True,
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "sp_model_kwargs":{},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<|endoftext|>",
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
    }
    with open(os.path.join(tokenizer_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print("Tokenizer training complete and saved to", tokenizer_dir)
    
def eval_tokenizer():
    # from transformers import AutoTokenizer 
    from transformers import PreTrainedTokenizerFast
    # tokenizer = AutoTokenizer.from_pretrained("/cpfs/user/boyuan/verl_workspace/hw1_code/hw1-code/pretrain/models/")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="/cpfs/user/boyuan/verl_workspace/hw1_code/hw1-code/pretrain/models/tokenizer.json")


    tokenizer.chat_template=""""chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
"""

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，充满了幽默感。"},
        {"role": "user", "content": "请问我女朋友又爱吃香菜又爱吃榴莲会发生什么事？"},
        {"role": "assistant", "content": "那简直是世界上最可爱的女朋友了！她的味蕾一定是个冒险家，敢于挑战所有的味道组合！香菜和榴莲的结合，可能会让她的味觉体验达到巅峰！"},
        {"role": "user", "content": "你能帮我写一首关于香菜和榴莲的诗吗？"},
        {"role": "assistant", "content": "香菜飘香榴莲甜，\n味蕾交织如梦幻。\n一口清新一口浓，\n两种风味共此生。"}
    ]
    new_prompt = tokenizer.apply_chat_template(messages,tokenize=False)
    print("New Prompt:", new_prompt)

    actual_vocab_size = len(tokenizer)
    print("Actual Vocab Size:", actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print("Model Inputs:", len(model_inputs['input_ids']))

    inputs_ids = model_inputs['input_ids']
    response = tokenizer.decode(inputs_ids, skip_special_tokens=False)
    print("If Decoded Response equal? :", response == new_prompt)

if __name__ == "__main__":
    # train_tokenizer()
    eval_tokenizer()
    print("Tokenizer training and evaluation completed successfully.")
