# merge_and_sample_datasets.py
import json
import random
import os
import argparse
import logging

# --- 日志设置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# =================================================================================
# === 1. 参数配置区 ===
# =================================================================================
class Config:
    # --- 输入文件路径 ---
    # 包含简单问答数据的文件
    SIMPLE_QA_FILE_PATH = "../data/generated_qa_pairs/qa_pairs_FR.jsonl"
    # 包含CoT问答数据的文件
    COT_QA_FILE_PATH = "../data/generated_qa_pairs/qa_pairs_CoT.jsonl"

    # --- 输出文件路径 ---
    # 最终用于微调的混合数据集
    MERGED_OUTPUT_PATH = "../data/generated_qa_pairs/finetune_training_set_mixed_final.jsonl"

    # --- 采样与配比设置 ---
    # 从简单问答数据中随机抽取的数量 (按70/30比例计算得出)
    SIMPLE_QA_SAMPLE_SIZE = 2851
    # 使用全部的CoT数据，如果您想减少，也可以修改这个数字
    COT_QA_SAMPLE_SIZE = 1222

    # 用于随机抽样的种子，保证每次运行结果一致
    RANDOM_SEED = 42


# =================================================================================

def merge_and_sample(config: Config):
    """
    Loads, samples, merges, shuffles, and saves the datasets.
    """
    logging.info("开始处理数据集...")

    # --- 读取数据 ---
    try:
        logging.info(f"从 '{config.SIMPLE_QA_FILE_PATH}' 读取简单问答数据...")
        with open(config.SIMPLE_QA_FILE_PATH, 'r', encoding='utf-8') as f:
            simple_qa_lines = f.readlines()
        logging.info(f"成功读取 {len(simple_qa_lines)} 条简单问答数据。")

        logging.info(f"从 '{config.COT_QA_FILE_PATH}' 读取CoT问答数据...")
        with open(config.COT_QA_FILE_PATH, 'r', encoding='utf-8') as f:
            cot_qa_lines = f.readlines()
        logging.info(f"成功读取 {len(cot_qa_lines)} 条CoT问答数据。")

    except FileNotFoundError as e:
        logging.error(f"错误：找不到输入文件。请检查路径配置。 {e}")
        return

    # --- 数据采样 ---
    if len(simple_qa_lines) < config.SIMPLE_QA_SAMPLE_SIZE:
        logging.warning(
            f"简单问答数据总量 ({len(simple_qa_lines)}) 小于期望的采样数 ({config.SIMPLE_QA_SAMPLE_SIZE})。将使用所有简单问答数据。"
        )
        sampled_simple_qa = simple_qa_lines
    else:
        logging.info(f"从简单问答数据中随机采样 {config.SIMPLE_QA_SAMPLE_SIZE} 条...")
        random.seed(config.RANDOM_SEED)
        sampled_simple_qa = random.sample(simple_qa_lines, config.SIMPLE_QA_SAMPLE_SIZE)

    if len(cot_qa_lines) < config.COT_QA_SAMPLE_SIZE:
        logging.warning(
            f"CoT数据总量 ({len(cot_qa_lines)}) 小于期望的采样数 ({config.COT_QA_SAMPLE_SIZE})。将使用所有CoT数据。"
        )
        sampled_cot_qa = cot_qa_lines
    else:
        # 如果CoT数据也想采样，可以使用同样逻辑，这里默认全取
        sampled_cot_qa = cot_qa_lines[:config.COT_QA_SAMPLE_SIZE]

    logging.info(f"采样完成。简单问答: {len(sampled_simple_qa)} 条, CoT问答: {len(sampled_cot_qa)} 条。")

    # --- 合并与打乱 ---
    logging.info("合并并打乱所有数据...")
    merged_data = sampled_simple_qa + sampled_cot_qa
    random.shuffle(merged_data)
    logging.info(f"数据打乱完成。总数据集大小: {len(merged_data)} 条。")

    # --- 保存最终文件 ---
    try:
        os.makedirs(os.path.dirname(config.MERGED_OUTPUT_PATH), exist_ok=True)
        with open(config.MERGED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for line in merged_data:
                f.write(line)
        logging.info(f"✅ 处理完成！最终的混合训练集已保存至: {config.MERGED_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"写入输出文件时发生错误: {e}")


def main():
    config = Config()
    merge_and_sample(config)


if __name__ == "__main__":
    main()