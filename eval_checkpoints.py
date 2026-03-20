import torch
import glob
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import PeftModel

def main():
    # 1. 基础配置：保持与训练时完全一致
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
    )
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True, cache_dir="model/qwen")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", 
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        cache_dir="model/qwen"
    )

    # 2. 准备测试数据（复用 main.py 的逻辑）
    def process_func(example):
        MAX_LENGTH = 384
        input_ids, attention_mask, labels = [], [], []
        messages = [
            {"role": "system", "content": "你现在要扮演皇帝的女人甄嬛"},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        response_text = example["output"] + "<|im_end|>\n"
        
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

    print("加载并处理测试数据...")
    dataset = load_dataset("json", data_files="data/huanhuan.json")
    # 【重要】确保这里的 seed=42 和 main.py 里一致
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    test_data = dataset["test"].map(process_func, remove_columns=dataset["test"].column_names)

    # 3. 查找所有的 checkpoint 目录
    output_dir = "./tuning/qwen_4b_lora"
    checkpoints = glob.glob(f"{output_dir}/checkpoint-*")
    
    if not checkpoints:
        print(f"未在 {output_dir} 找到任何 checkpoint！")
        return
        
    # 按照 step 数字大小排序
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

    # 4. 加载第一个 checkpoint 以初始化 PeftModel
    model = PeftModel.from_pretrained(base_model, checkpoints[0])
    
    # 初始化一个 Trainer 专门用于评估
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./eval_temp",
            per_device_eval_batch_size=8,  # 根据你的显存（如 RTX 3090）可适当调大
            bf16=True,
            report_to="none",              # 关闭 wandb 等日志上报
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
        eval_dataset=test_data,
    )

    # 5. 遍历评估所有 checkpoint
    eval_results = {}
    for ckpt in checkpoints:
        print(f"\n========== 正在评估: {ckpt} ==========")
        # 动态加载不同的 LoRA 权重，避免重复加载庞大的 Base Model
        model.load_adapter(ckpt, adapter_name="default")
        
        # 运行评估
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        eval_results[ckpt] = eval_loss
        print(f">>> {ckpt} eval_loss: {eval_loss:.4f}")

    # 6. 打印最终汇总结果
    print("\n" + "="*40)
    print("📊 所有 Checkpoint 的 Eval Loss 汇总：")
    for ckpt, loss in eval_results.items():
        step = ckpt.split("-")[-1]
        print(f"Step {step}: {loss:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
