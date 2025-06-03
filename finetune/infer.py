# finetune/infer.py

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from config import Config
from peft import PeftModel

def load_model(model_path=None):
    model_path = model_path or Config.output_dir
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        trust_remote_code=True,
        device_map="auto",
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer

def infer(text, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    prompt = f"<|im_start|>user\n请从以下膜材料文献中抽取材料设计、制备方法、性能指标和应用场景等信息：\n{text}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|im_start|>assistant\n")[-1].strip()
    return response

# 示例调用
if __name__ == "__main__":
    sample_text = "本研究开发了一种基于PVDF的中空纤维膜，采用相转化法制备，通量达到150 L/m²·h，适用于染料废水处理。"
    result = infer(sample_text)
    print("模型抽取结果：\n", result)
