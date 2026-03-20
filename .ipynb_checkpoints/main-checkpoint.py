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
class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]
# DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"



def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # 恢复使用 bfloat16 (RTX 3090 支持)
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # 恢复使用 flash_attention_2 (RTX 3090 支持)，确保 torch_dtype 为 bfloat16
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True,dtype=torch.bfloat16,attn_implementation="flash_attention_2",quantization_config=bnb_config)
    model.enable_input_require_grads()
    def process_func(example):
        MAX_LENGTH = 384
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer("<|im_start|>system\n你现在要扮演皇帝的女人甄嬛<|im_end|>\n"+"<|im_start|>user\n"+example["instruction"]+"<|im_end|>\n"+"<|im_start|>assistant\n")
        response = tokenizer(example["output"]+"<|im_end|>\n")
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

    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
    # model = get_peft_model(model, peft_config)
    dataset = load_dataset( "json" , data_files= "data/huanhuan.json" )
    # 划分训练集和测试集，test_size=0.1 表示测试集占 10%
    dataset = dataset["train"].train_test_split(test_size=0.1)
    train_data = dataset["train"].map(process_func, remove_columns=dataset["train"].column_names)
    test_data = dataset["test"].map(process_func, remove_columns=dataset["test"].column_names)
    print(tokenizer.decode(train_data[0]["input_ids"]))
    decoded_labels = [token_id for token_id in train_data[0]["labels"] if token_id >= 0]
    print(tokenizer.decode(decoded_labels))
    # class AutocastSFTTrainer(SFTTrainer):
    #     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #         # 解决 bf16 开启时，部分模型层仍是 float32 导致的 mat1 and mat2 dtype mismatch 问题
    #         with autocast(dtype=torch.bfloat16 if self.args.bf16 else torch.float16):
    #             return super().compute_loss(model, inputs, return_outputs, **kwargs)
    training_args = SFTConfig(
        output_dir="./output/qwen_4b_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True, # 必须与 bnb_4bit_compute_dtype 一致，开启混合精度训练
        gradient_checkpointing_kwargs={'use_reentrant': True}, # 修复 SavedTensorHooks 导致的 INTERNAL ASSERT FAILED 报错
        bf16=True, # 开启 bfloat16 混合精度 (RTX 3090 支持)
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    #加载Lora模型
    p_model = PeftModel.from_pretrained(model, model_id=training_args.output_dir)
    value = "你是谁"
    ipt = tokenizer(f"<|im_start|>user\n{value}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
    output = tokenizer.decode(p_model.generate(**ipt, do_sample=False)[0], skip_special_tokens=True)
    print(output)
    # merge_model = p_model.merge_and_unload()
    # merge_model.save_pretrained("./output/qwen_4b_lora_merge")
 
if __name__ == "__main__":
    main()
