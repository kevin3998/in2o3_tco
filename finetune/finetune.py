import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
import logging

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =================================================================================
# === 1. 参数配置区 ===
# =================================================================================
class BaseConfig:
    # -- 模型与数据路径 --
    model_path = "/Users/chenlintao/Desktop/models/Qwen3-14B"
    dataset_path = "../data/finetune_sets/finetune_training_set_mixed_final.jsonl"

    # -- LoRA 配置 --
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05

    # -- 训练参数 --
    num_train_epochs = 3
    learning_rate = 2e-4
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.05
    logging_steps = 1
    save_strategy = "steps"
    save_steps = 10

    # -- SFT特定参数 --
    max_seq_length = 1024

    # -- 其他 --
    validation_set_size = 0.2

    # --- 断点续训开关 ---
    # True: 尝试从最新的断点恢复训练
    # False: 从头开始一次全新的训练
    resume_from_checkpoint = False  # <-- 首次训练建议设为False


class BF16LoRAConfig(BaseConfig):
    # 为128GB统一内存优化的配置
    output_dir = "../results/qwen3-14b-lora-bf16_lr-2e-4"
    batch_size = 4
    gradient_accumulation_steps = 8
    optimizer = "adamw_torch"


# --- 辅助函数 ---
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main():
    config = BF16LoRAConfig()
    logging.info(
        f"--- 启动 BF16 LoRA 微调模式 (为 {config.batch_size * config.gradient_accumulation_steps} 有效批次大小优化) ---")

    if not torch.backends.mps.is_available(): return
    logging.info("检测到MPS设备，将使用 'mps' 进行训练。")

    full_dataset = load_dataset("json", data_files=config.dataset_path, split="train")
    split_dataset = full_dataset.train_test_split(test_size=config.validation_set_size, seed=42)
    train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']
    logging.info(f"数据集加载完成。训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条。")

    model = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 device_map="auto")
    if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    peft_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                             target_modules=find_all_linear_names(model), bias="none", task_type="CAUSAL_LM")

    # === 在查找断点前，先判断输出目录是否存在 ===
    last_checkpoint = None
    if config.resume_from_checkpoint:
        # 只有当输出目录存在时，才尝试在其中寻找断点
        if os.path.isdir(config.output_dir):
            last_checkpoint = get_last_checkpoint(config.output_dir)

        if last_checkpoint:
            logging.info(f"检测到断点，将从 '{last_checkpoint}' 恢复训练。")
        else:
            logging.info("配置了断点续训，但未找到任何断点，将从头开始训练。")
    # ======================================================

    training_args = SFTConfig(
        output_dir=config.output_dir,
        max_seq_length=config.max_seq_length,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optimizer,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        group_by_length=True,
        report_to="tensorboard",
        save_total_limit=3,
        load_best_model_at_end=True,  # 建议开启，以便自动保存最佳模型
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        use_cpu=False,
        weight_decay=0.01
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.tokenizer = tokenizer

    logging.info("所有配置完成，开始或恢复模型微调...")

    # 将找到的断点路径（或None）传递给 .train() 方法
    trainer.train(resume_from_checkpoint=last_checkpoint)

    logging.info("✅ 微调流程全部完成！")

    final_output_dir = os.path.join(config.output_dir, "final_checkpoint")
    logging.info(f"训练完成，将最终的LoRA适配器保存在 {final_output_dir}")
    trainer.save_model(final_output_dir)


if __name__ == "__main__":
    main()