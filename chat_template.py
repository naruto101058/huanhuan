import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, BitsAndBytesConfig, GenerationConfig
import torch
from peft import LoraConfig,get_peft_model,PeftModel
from trl import SFTConfig, SFTTrainer
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

dataset = load_dataset( "json" , data_files= "data/huanhuan.json" )
example = dataset["train"][0]
local_dir = "/root/autodl-tmp/huanhuan/model/qwen/models--Qwen--Qwen3.5-4B"
tokenizer = AutoTokenizer.from_pretrained(local_dir,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    # 使用 apply_chat_template 确保遵循 Qwen 的标准格式，并且特殊 token 被正确处理
    messages = [
        {"role": "system", "content": "你现在要扮演皇帝的女人甄嬛"},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    
    # 将消息转换为带特殊 token 的完整输入文本
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    response_text = example["output"] + "<|im_end|>\n"
    print(prompt)
    print(response_text)
    # 分别对 instruction (不计入 loss) 和 response 进行 tokenize
    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(response_text, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
process_func(example)