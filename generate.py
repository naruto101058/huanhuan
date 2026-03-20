from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, BitsAndBytesConfig, GenerationConfig
import torch
import os
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True).to("cuda")
print(tokenizer.special_tokens_map)
print("结束ID:",tokenizer.eos_token_id)
prompt = "九尾，我可以变成为尾兽模式么？"
message = [
    {"role": "system", "content": "你是火影世界中的九尾九喇嘛。帮助你的人柱力解决一切问题,请你回答的时候一定要使用中文"},
    {"role": "user", "content": prompt}
]
input_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
print("输入>",input_text)
model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
print(model_inputs)
generate_ids = model_inputs.input_ids

# def generate_next_token(model, generate_ids, temperature=0.7, debug=False):
#     print("input_ids:", generate_ids) if debug else None
#     logits = model.forward(generate_ids).logits
#     print("logits: ", logits) if debug else None
    
#     if temperature > 0:
#         probs = torch.softmax(logits[:, -1] / temperature, dim = -1)
#         print("probs: ", len(probs[0]), probs) if debug else None
#         next_token = torch.multinomial(probs, num_samples=1)  # 按照概率采样得到下一个token,按照概率随机抽一个token，概率大的token被抽中的概率大，但是小概率token也有可能被抽中
#     else:
#         next_token = torch.argmax(logits[:, -1], dim=-1)
    
#     print("next_id: ", next_token, ", token: ", tokenizer.decode(next_token)) if debug else None
#     return next_token.reshape(-1, 1)
# # next_token = generate_next_token(model, generate_ids, temperature=0.7, debug=True)
# # generate_ids = torch.cat([generate_ids, next_token], dim=-1)
# # print("输出>",tokenizer.batch_decode(generate_ids[:,len(model_inputs.input_ids[0]):], skip_special_tokens=True)[0])
# max_new_tokens = 10000000000
# for _ in range(max_new_tokens):
#     next_token = generate_next_token(model, generate_ids, temperature=0.7, debug=False)
#     generate_ids = torch.cat([generate_ids, next_token], dim=-1)
#     if next_token.item() == tokenizer.eos_token_id:
#         break
#     print(tokenizer.decode(next_token.item(), skip_special_tokens=False), end="")


def generate_next_token(model, input_ids, past_key_values=None, temperature=0.7, debug=False):
    """
    input_ids: 第一次传完整prompt；后续传 shape=[B,1] 的 next_token
    past_key_values: KV cache
    """
    if debug:
        print("input_ids shape:", tuple(input_ids.shape))
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    logits = outputs.logits  # [B, T, V]
    past = outputs.past_key_values
    next_token_logits = logits[:, -1, :]  # [B, V]
    if temperature and temperature > 0:
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B,1]
    else:
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B,1]

    if debug:
        # tokenizer.decode 需要 1d/list；这里只打印 batch=1 的情况
        print("next_id:", next_token[0].item(), "token:", tokenizer.decode([next_token[0].item()]))
    return next_token, past
# ====== generation loop ======
model.eval()
max_new_tokens = 10000000000
past = None
# 只用于保存最终输出（占用很小），模型计算用 past 来避免每步重算
# 如果你不需要保留全部 ids，也可以不 cat（见下面注释）
with torch.inference_mode():  # 等价于 no_grad + 更激进的推理优化
    # 第一次：喂完整 prompt，建立 KV cache
    next_token, past = generate_next_token(
        model, generate_ids, past_key_values=None, temperature=0.7, debug=False
    )
    generate_ids = torch.cat([generate_ids, next_token], dim=-1)
    if next_token.item() != tokenizer.eos_token_id:
        print(tokenizer.decode(next_token.item(), skip_special_tokens=False), end="")
    # 后续：每次只喂 1 个 token + past
    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        next_token, past = generate_next_token(
            model, next_token, past_key_values=past, temperature=0.7, debug=False
        )
        # 保存结果（可选）
        generate_ids = torch.cat([generate_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        print(tokenizer.decode(next_token.item(), skip_special_tokens=False), end="")