# finetune/train.py

import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import Config
from transformers import Trainer


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path="train_log.jsonl"):
        self.log_path = log_path
        if os.path.exists(log_path):
            os.remove(log_path)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            record = {
                "step": state.global_step,
                "loss": logs["loss"]
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


def train():
    # 加载分词器和基础模型
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        trust_remote_code=True,
        load_in_4bit=True,  # 启用 QLoRA 低资源微调
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto"
    )

    # PEFT 配置
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # 可根据模型结构修改
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载 JSONL 数据集
    dataset = load_dataset("json", data_files={"train": Config.dataset_path})["train"]

    def format_example(example):
        messages = example["messages"]
        prompt = messages[0]["content"]
        response = messages[1]["content"]
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        return tokenizer(full_prompt, truncation=True, max_length=Config.max_seq_length)

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        num_train_epochs=Config.num_train_epochs,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        save_total_limit=2,
        fp16=True,
        lr_scheduler_type="cosine",
        warmup_steps=Config.warmup_steps,
        logging_dir="./logs",
        report_to=["tensorboard"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[LossLoggerCallback("train_log.jsonl")]  #  日志记录
    )

    # 开始训练
    trainer.train()
    model.save_pretrained(Config.output_dir)
    tokenizer.save_pretrained(Config.output_dir)

if __name__ == "__main__":
    train()
