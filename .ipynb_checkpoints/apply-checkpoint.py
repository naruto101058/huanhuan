import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, BitsAndBytesConfig, GenerationConfig
import torch
from peft import LoraConfig,get_peft_model,PeftModel

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',trust_remote_code=True,cache_dir="model/qwen")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", 
                                            trust_remote_code=True,
                                            dtype=torch.bfloat16,
                                            # attn_implementation="flash_attention_2",
                                            # quantization_config=bnb_config,
                                            cache_dir="model/qwen")

p_model = PeftModel.from_pretrained(model, model_id="./tuning/qwen_4b_lora/checkpoint-795")

p_model.eval()
while True:
    value = input("请输入：")
    ipt = tokenizer(f"<|im_start|>user\n{value}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
    out = p_model.generate(
        **ipt,
        do_sample=False,
        max_new_tokens=64,   # 最多生成 128 个新token
        # min_new_tokens=1,     # 可选：至少生成 1 个
    )
    output = tokenizer.decode(out[0], skip_special_tokens=True)
    print(output)