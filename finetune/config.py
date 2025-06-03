class Config:
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    output_dir = "./checkpoints/lora_membrane"
    dataset_path = "./data/fine_tune/structured_extraction_filtered.jsonl"
    max_seq_length = 2048
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 2e-5
    num_train_epochs = 3
    save_steps = 100
    logging_steps = 10
    warmup_steps = 100

    use_lora = True
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
