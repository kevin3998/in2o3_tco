import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 定义模型路径
model_path = "/model/models/qwen3-14b-in2o3-tco-merged"  # <-- 替换成你合并后模型的实际路径

# 2. 加载 Tokenizer 和模型
#    device_map="auto" 会自动将模型分配到可用的硬件上 (如 GPU)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # or torch.float16，根据你的硬件调整
    device_map="auto"
)

# 3. 准备输入 (Prompt)
#    注意：这里的 prompt 格式需要和你微调时使用的格式保持一致！
#    例如，如果微调时使用了 Alpaca 格式，这里也应该遵循。
prompt = "ITO设计中的基质材料都有哪些？简要介绍一下"
# chat_template = model.chat_template # 如果你的模型有聊天模板，可以使用
# messages = [{"role": "user", "content": prompt}]
# formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False) # 使用模板格式化

# 4. 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 5. 生成输出
#    可以调整 max_new_tokens, temperature, top_p, do_sample 等参数来控制生成质量
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 6. 解码并打印结果
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)