import os
import sys
import torch
import transformers
import peft
import huggingface_hub
import logging

# --- 0. 日志和基本配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 配置路径 (请仔细检查，确保这里的三个路径都是正确的) ---
BASE_MODEL_PATH = "/Users/chenlintao/Desktop/models/Qwen3-14B"
LORA_ADAPTER_PATH = "../results/qwen3-14b-lora-bf16_lr-2e-4/final_checkpoint"
MERGED_MODEL_OUTPUT_PATH = "/Users/chenlintao/Desktop/in2o3_tco/models/qwen3-14b-merged-diagnostics"

# --- 2. 内建诊断：检查环境和路径 (这是解决问题的关键!) ---
print("=" * 50)
print("--- 1. 开始执行内置诊断程序 ---")
print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"Transformers 版本: {transformers.__version__}")
print(f"PEFT 版本: {peft.__version__}")
print(f"Huggingface Hub 版本: {huggingface_hub.__version__}")
print("\n" + "-" * 50)

print(f"检查基础模型路径: {BASE_MODEL_PATH}")
if not os.path.isdir(BASE_MODEL_PATH):
    raise FileNotFoundError(f"错误: 基础模型路径不存在 -> {BASE_MODEL_PATH}")
print("✅ 基础模型路径... OK")
print("-" * 50)

print(f"检查LoRA适配器路径: {LORA_ADAPTER_PATH}")
if not os.path.isdir(LORA_ADAPTER_PATH):
    raise FileNotFoundError(f"错误: LoRA适配器路径不存在 -> {LORA_ADAPTER_PATH}")
print("✅ LoRA适配器路径... OK")
print("-" * 50)

print("LoRA适配器文件夹内容:")
try:
    files_in_lora_dir = os.listdir(LORA_ADAPTER_PATH)
    for f in files_in_lora_dir:
        print(f"  - {f}")
    if "adapter_config.json" not in files_in_lora_dir:
        print("\n[警告] 文件夹中未找到 'adapter_config.json'!\n")
    if not any(f.endswith((".bin", ".safetensors")) for f in files_in_lora_dir):
        print("\n[警告] 文件夹中未找到模型权重文件 (.bin or .safetensors)!\n")
except Exception as e:
    print(f"错误: 无法列出文件夹内容: {e}")

print("=" * 50)
print("--- 诊断程序结束 ---")
print("=" * 50 + "\n")

# --- 3. 主程序：加载并合并模型 ---
try:
    logging.info("--- 2. 开始执行模型合并 ---")

    logging.info(f"加载基础模型: {BASE_MODEL_PATH}")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    logging.info("基础模型和分词器加载完成。")

    logging.info(f"加载PEFT LoRA适配器: {LORA_ADAPTER_PATH}")
    # 使用最标准、最直接的方式加载适配器
    model = peft.PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH,
        is_trainable=False,
    )
    logging.info("LoRA适配器加载完成。")

    logging.info("开始合并权重...")
    model = model.merge_and_unload()
    logging.info("权重合并完成。")

    logging.info(f"保存合并后的模型到: {MERGED_MODEL_OUTPUT_PATH}")
    os.makedirs(MERGED_MODEL_OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(MERGED_MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_OUTPUT_PATH)

    print("\n" + "=" * 50)
    print("✅✅✅ 模型合并成功！✅✅✅")
    print(f"合并后的模型已保存至: {MERGED_MODEL_OUTPUT_PATH}")
    print("=" * 50)

except Exception as e:
    logging.error("程序在执行期间发生致命错误！")
    # 重新引发异常以打印完整的错误堆栈信息
    raise e