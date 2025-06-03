import json

# 输入路径为你的抽取结果 JSON 文件
input_path = "../data/extracted_json/structured_extraction_111.json"
# 输出路径为转换为微调格式后的 JSONL 文件
output_path = "../data/finetune_data/structured_extraction_filtered.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    extracted_data = json.load(f)

num_samples = 0
with open(output_path, "w", encoding="utf-8") as f_out:
    for item in extracted_data:
        # 检查 input 文本是否为空
        input_text = item.get("input", "").strip()
        if not input_text:
            continue  # 跳过 input 为空的样本

        # 检查 output 字段是否有效
        output_obj = item.get("output", None)
        if output_obj is None:
            continue  # 跳过 output 字段缺失的样本

        # 如果 output 为列表（如多重抽取），确保列表不为空
        if isinstance(output_obj, list):
            if not output_obj:
                continue  # 列表为空则跳过
            # 可选：进一步检查列表中是否存在有效信息，比如是否存在有效的 "Material"
            valid_entries = [entry for entry in output_obj
                             if isinstance(entry, dict) and
                             entry.get("Material", "").strip() and
                             entry.get("Details", {})]
            if not valid_entries:
                continue  # 如果所有条目均无有效信息，则跳过
            # 如果有多个条目，统一输出整个列表
            output_obj = valid_entries
        # 如果 output 为字典，检查是否包含必要信息
        elif isinstance(output_obj, dict):
            if not output_obj.get("Material", "").strip() or not output_obj.get("Details", {}):
                continue
        else:
            continue  # 非预期数据格式，跳过

        # 将 output 转为格式化的字符串
        output_str = json.dumps(output_obj, ensure_ascii=False, indent=2)

        # 组装成微调需要的 messages 格式
        jsonl_obj = {
            "messages": [
                {
                    "role": "user",
                    "content": "请从以下膜材料文献中抽取材料设计、制备方法、性能指标和应用场景等信息：\n" + input_text
                },
                {
                    "role": "assistant",
                    "content": output_str
                }
            ]
        }
        f_out.write(json.dumps(jsonl_obj, ensure_ascii=False) + "\n")
        num_samples += 1

print(f"✅ 已转换为微调格式，筛选后样本总数：{num_samples}，保存至：{output_path}")
