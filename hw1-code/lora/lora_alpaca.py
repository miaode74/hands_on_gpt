#!/usr/bin/env python
"""Minimal‑change LoRA fine‑tuning script for Alpaca on BLOOMZ‑560M.
------------------------------------------------------------------
* Keeps your original IMDb‑LoRA training pipeline almost intact.
* All Alpaca‑specific logic (prompt building + mask) lives in a few
  helper functions – no need to touch the LoRA/Trainer boilerplate.
* **Fixed multi‑GPU padding error** by padding to a fixed length at
  preprocessing time and switching to `DefaultDataCollator`.

Usage
-----
Single‑GPU (auto batch‑size):
$ python alpaca_lora_finetune.py

8‑GPU DDP:
$ torchrun --nproc_per_node 8 alpaca_lora_finetune.py

Adjust the variables in the CONFIG section as you like.
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

# ============================== CONFIG ==============================
MODEL_NAME: str = "bigscience/bloomz-560m"   # backbone
DATASET_NAME: str = "tatsu-lab/alpaca"        # HF dataset
OUTPUT_DIR: str = "./peft_alpaca_outputs"     # where checkpoints go
CUT_OFF_LEN: int = 512                        # max token length
TRAIN_ON_INPUTS: bool = False                 # mask the prompts?
ADD_EOS_TOKEN: bool = True                    # append <eos> if missing

LORA_R: int = 4
LORA_ALPHA: int = 1
LORA_DROPOUT: float = 0.05
TARGET_MODULES: List[str] = ["query_key_value"]  # BLOOM style

LEARNING_RATE: float = 3e-4   # more conservative for multi‑GPU
NUM_EPOCHS: int = 2
# ====================================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# ------------------------- Prompt helpers ------------------------- #

def build_prompt(record: Dict[str, str]) -> str:
    """Compose Alpaca style instruction → input → response prompt."""
    if record["input"]:
        return (
            f"### Instruction:\n{record['instruction']}\n\n"
            f"### Input:\n{record['input']}\n\n"
            f"### Response:\n{record['output']}"
        )
    return (
        f"### Instruction:\n{record['instruction']}\n\n"
        f"### Response:\n{record['output']}"
    )


def tokenize(text: str, add_eos_token: bool = True) -> Dict[str, List[int]]:
    """Tokenise + pad to fixed length so every worker gets equal tensors."""
    out = tokenizer(
        text,
        truncation=True,
        max_length=CUT_OFF_LEN,
        padding="max_length",  # <-- fixes variable‑length batches
    )

    if add_eos_token and out["input_ids"][ -1] != tokenizer.eos_token_id:
        # ensure EOS fits: overwrite last token
        out["input_ids"][-1] = tokenizer.eos_token_id

    out["labels"] = out["input_ids"].copy()  # will be masked later if needed
    return out


def generate_and_tokenize_prompt(dp):
    full_prompt = build_prompt(dp)
    tok = tokenize(full_prompt, add_eos_token=ADD_EOS_TOKEN)

    if not TRAIN_ON_INPUTS:
        user_prompt = build_prompt({**dp, "output": ""})
        user_tok = tokenize(user_prompt, add_eos_token=ADD_EOS_TOKEN)
        user_len = len(user_tok["input_ids"]) - (1 if ADD_EOS_TOKEN else 0)
        tok["labels"][:user_len] = [-100] * user_len  # mask
    return tok

# --------------------------- Data prep ---------------------------- #
print("Loading Alpaca dataset …")
raw_ds = load_dataset(DATASET_NAME, split="train")
print("Tokenising … (this can take a few minutes)")
train_ds = raw_ds.map(generate_and_tokenize_prompt, remove_columns=raw_ds.column_names, num_proc=8)

# --------------------------- LoRA setup --------------------------- #
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="lora_only",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------- Trainer setup -------------------------- #
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    auto_find_batch_size=True,
    report_to=["none"],  # change to "wandb" if you wish
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
)

collator = DefaultDataCollator(return_tensors="pt")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=collator,
)

# ---------------------------- Train! ------------------------------ #
if __name__ == "__main__":
    trainer.train()

    # save LoRA adapters only (tiny files)
    lora_path = os.path.join(OUTPUT_DIR, "lora_adapters")
    model.save_pretrained(lora_path)
    print(f"LoRA weights saved to {lora_path}")

    # -------- inference demo --------
    prompt = "### Instruction:\n给出一个含有二次函数的数学应用题并提供详细解答"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )
    print("\n=== Sample output ===\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
