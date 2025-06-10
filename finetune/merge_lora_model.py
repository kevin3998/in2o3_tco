# 7_merge_lora_model.py
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =================================================================================
# === 1. 参数配置区 ===
# =================================================================================
class MergeConfig:
    # --- 路径配置 ---
    # 基础模型路径（您之前下载的原始Qwen2-14B模型）
    base_model_path = "/Users/chenlintao/Desktop/models/Qwen3-14B"

    # LoRA适配器路径（您微调后得到的最佳断点）
    # 请确保这个路径指向您训练输出的 final_best_checkpoint 文件夹
    lora_adapter_path = "../results/qwen3-14b-lora-bf16/final_best_checkpoint"

    # 合并后新模型的保存路径
    merged_model_output_path = "../models/qwen3-14b-in2o3-tco-merged"


# =================================================================================

def main():
    config = MergeConfig()
    logging.info("--- 开始合并LoRA适配器与基础模型 ---")

    # 确保输出目录存在
    os.makedirs(config.merged_model_output_path, exist_ok=True)

    # 加载基础模型和Tokenizer
    logging.info(f"加载基础模型: {config.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True)

    # 加载LoRA适配器并与基础模型合并
    logging.info(f"加载LoRA适配器: {config.lora_adapter_path}")
    # PeftModel会自动将LoRA适配器加载到基础模型之上
    model = PeftModel.from_pretrained(base_model, config.lora_adapter_path)

    logging.info("开始合并权重...")
    # merge_and_unload() 会将LoRA权重合并到模型自身的权重中，并卸载PEFT模块
    model = model.merge_and_unload()
    logging.info("权重合并完成。")

    # 保存合并后的完整模型和Tokenizer
    logging.info(f"保存合并后的模型到: {config.merged_model_output_path}")
    model.save_pretrained(config.merged_model_output_path)
    tokenizer.save_pretrained(config.merged_model_output_path)

    logging.info("✅ 模型合并完成！现在您可以使用新模型进行推理部署了。")


if __name__ == "__main__":
    main()