import regex as re
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os


# Reuse your existing PromptManager if it's in a separate file and importable
# Otherwise, include it here. For this example, I'll assume it's available.
# from extraction_framework import PromptManager # Assuming your class is here

# If PromptManager is in the same file as the original script, no need to re-declare if running together.
# For a standalone script, you'd copy the PromptManager class definition here.
class PromptManager:  # Copied from your original for completeness if run standalone
    def __init__(self):
        self.templates = {}

    def add_prompt(self, task_type: str, language: str, template: str):  # Changed 'domain' to 'task_type'
        self.templates[(task_type, language)] = template

    def get_prompt(self, task_type: str, language: str, json_data_str: str) -> str:  # Changed 'text' to 'json_data_str'
        template = self.templates.get((task_type, language))
        if not template:
            raise ValueError(f"Prompt template not found for task_type '{task_type}' and language '{language}'")
        return template.format(json_data_string=json_data_str)


class QAFromStructuredDataProcessor:
    def __init__(self, client: OpenAI, prompt_manager: PromptManager, model="gpt-4"):  # Removed DomainConfig
        self.client = client
        self.prompt_manager = prompt_manager
        self.model = model

    def _parse_qa_response(self, content: str, structured_input_json: dict, paper_meta: dict, paper_context: str) -> \
    List[Dict]:
        try:
            json_str = re.sub(r"[\x00-\x1F]", "", content)  # Remove control characters

            # Try to find the main JSON structure (LLMs sometimes wrap in ```json ... ```)
            match = re.search(r"\{\s*\"qa_pairs\"\s*:\s*\[.*\]\s*\}", json_str, re.DOTALL)
            if match:
                json_str_to_parse = match.group(0)
            else:
                json_str_to_parse = json_str  # Fallback

            raw_data = json.loads(json_str_to_parse, strict=False)

            # Basic key standardization for the top-level key
            qa_pair_list = []
            for key, value in raw_data.items():
                if key.lower().replace("_", "") == "qapairs":
                    qa_pair_list = value
                    break
            if not qa_pair_list and "qa_pairs" in raw_data:  # Fallback to exact key
                qa_pair_list = raw_data["qa_pairs"]

            if not isinstance(qa_pair_list, list):
                print(f"⚠️ Expected 'qa_pairs' to be a list, but got {type(qa_pair_list)}. Content: {content[:200]}")
                return []

            generated_qa_data = []
            for qa_entry in qa_pair_list:
                if isinstance(qa_entry, dict) and "question" in qa_entry and "answer" in qa_entry:
                    # Basic cleaning of question and answer
                    question = str(qa_entry["question"]).strip()
                    answer = str(qa_entry["answer"]).strip()

                    # Filter out pairs with "Not specified" answers or very short/generic answers if desired
                    if answer.lower() == "not specified" or len(answer) < 3:  # Basic filter
                        continue

                    generated_qa_data.append({
                        "meta": paper_meta,  # Original paper metadata
                        "question": question,
                        "answer": answer,
                        "context": paper_context,  # Full text of the paper as context
                        "source_structured_data": structured_input_json  # The JSON this QA was generated from
                    })
                else:
                    print(f"⚠️ Invalid QA entry format found in LLM response: {str(qa_entry)[:100]}")

            return generated_qa_data
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败 for QA response: {e}\n原始模型输出 (前500字符): {content[:500]}")
            return []
        except Exception as e:
            print(f"❌ _parse_qa_response 中发生其他错误: {e}")
            return []

    def generate_qa_for_single_item(self, structured_item_output: dict, paper_meta: dict, paper_context: str,
                                    task_type: str, language: str) -> List[Dict]:
        """
        Generates QA pairs for a single structured data item.
        'structured_item_output' is the content of the "output" key from your original script.
        """
        if not structured_item_output or not structured_item_output.get("Details"):  # Basic check
            print(
                f"ℹ️ Structured data for {paper_meta.get('doi', 'N/A')} is empty or lacks 'Details'. Skipping QA generation.")
            return []

        json_data_str = json.dumps(structured_item_output, indent=2, ensure_ascii=False)
        prompt = self.prompt_manager.get_prompt(task_type, language, json_data_str)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert AI assistant skilled in creating Question-Answer pairs from structured data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # Slightly higher temperature for more diverse questions, but still factual
                    stream=False
                )
                # Make sure choices and message exist
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    if content:
                        parsed_qa_pairs = self._parse_qa_response(content, structured_item_output, paper_meta,
                                                                  paper_context)
                        return parsed_qa_pairs
                    else:
                        print(f"⚠️ LLM returned empty content for {paper_meta.get('doi', 'N/A')}.")
                        return []  # Return empty if content is None or empty
                else:
                    print(f"⚠️ LLM response structure unexpected for {paper_meta.get('doi', 'N/A')}.")
                    return []


            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(
                        f"QA生成请求失败 for {paper_meta.get('doi', 'N/A')}，{delay}s后重试（{attempt + 1}/{max_retries}）")
                    time.sleep(delay)
                    continue
                print(f"❌ QA生成最终请求失败 for {paper_meta.get('doi', 'N/A')}：{str(e)}")
                return []
        return []  # Should be unreachable if loop finishes, but as a fallback

    def process_structured_data_to_qa(self, structured_data_entries: List[Dict], task_type: str, language: str,
                                      checkpoint_path="qa_checkpoint.json") -> List[Dict]:
        """
        Processes a list of structured data entries (output from your first script) to generate QA pairs.
        Each entry in 'structured_data_entries' is expected to have "meta", "input" (original text), 
        and "output" (the structured JSON for QA generation) keys.
        """
        default_checkpoint = {
            "processed_dois_for_qa": [],  # Use a different key to avoid conflict with original checkpoint
            "qa_results": [],
            "total_elapsed_qa": 0.0,
            "total_processed_qa": 0
        }

        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                checkpoint = {**default_checkpoint, **checkpoint}  # Merge with default
                print(
                    f"▶ QA检查点已加载 | 累计处理 {checkpoint['total_processed_qa']} 条结构化数据 | 累计用时 {checkpoint['total_elapsed_qa']:.1f}s")
                processed_dois_for_qa = set(checkpoint["processed_dois_for_qa"])
                qa_results = checkpoint["qa_results"]
                total_elapsed_qa = checkpoint["total_elapsed_qa"]
                total_processed_qa = checkpoint["total_processed_qa"]
            except Exception as e:
                print(f"⚠ QA检查点加载失败: {str(e)}，重新开始处理")
                processed_dois_for_qa, qa_results, total_elapsed_qa, total_processed_qa = set(), [], 0.0, 0
        else:
            processed_dois_for_qa, qa_results, total_elapsed_qa, total_processed_qa = set(), [], 0.0, 0

        total_items = len(structured_data_entries)
        try:
            for idx, item_entry in enumerate(structured_data_entries):
                session_start_time = time.time()
                # Assuming 'doi' is in 'meta' and can be used as a unique identifier
                entry_doi = item_entry.get("meta", {}).get("doi", f"item_{idx}")

                if entry_doi in processed_dois_for_qa:
                    continue

                structured_output_for_qa = item_entry.get("output")
                original_paper_context = item_entry.get("input", "")
                paper_meta = item_entry.get("meta", {})

                if not structured_output_for_qa:
                    print(f"ℹ️ 条目 {entry_doi} 没有 'output' 字段，跳过QA生成。")
                    processed_dois_for_qa.add(entry_doi)  # Mark as processed to avoid re-check
                    continue

                try:
                    new_qa_pairs = self.generate_qa_for_single_item(
                        structured_output_for_qa, paper_meta, original_paper_context, task_type, language
                    )
                    qa_results.extend(new_qa_pairs)
                    processed_dois_for_qa.add(entry_doi)

                    session_elapsed = time.time() - session_start_time
                    total_elapsed_qa += session_elapsed
                    total_processed_qa += 1

                    # ETA calculation
                    avg_time_per_item = total_elapsed_qa / total_processed_qa if total_processed_qa > 0 else 0
                    eta_seconds = avg_time_per_item * (total_items - total_processed_qa)

                    print(
                        f"\r✔ QA生成进度 [{total_processed_qa}/{total_items}] | 本次用时 {session_elapsed:.1f}s | 总用时 {total_elapsed_qa:.1f}s | 预计剩余 {eta_seconds / 3600:.1f}h",
                        end="",
                        flush=True
                    )

                    if total_processed_qa % 5 == 0:  # Save checkpoint periodically
                        self._save_qa_checkpoint(checkpoint_path, processed_dois_for_qa, qa_results, total_elapsed_qa,
                                                 total_processed_qa)

                except KeyboardInterrupt:
                    print("\n⚠ 用户中断QA生成，保存进度...")
                    self._save_qa_checkpoint(checkpoint_path, processed_dois_for_qa, qa_results, total_elapsed_qa,
                                             total_processed_qa)
                    raise
                except Exception as e_item:  # Catch error for a single item and continue
                    print(f"\n⚠ QA生成中处理条目 {entry_doi} 失败: {str(e_item)}")
                    # Optionally save checkpoint here too if you want to be very robust
                    continue  # Continue with the next item

            self._cleanup_checkpoint(checkpoint_path)  # Clean up successful completion

        except Exception as e_main:  # Catch any wider loop exception
            print(f"\n⚠ QA生成主流程发生未处理异常: {str(e_main)}")
            self._save_qa_checkpoint(checkpoint_path, processed_dois_for_qa, qa_results, total_elapsed_qa,
                                     total_processed_qa)
            raise

        print(f"\n✅ QA生成全部处理完成 | 总用时 {total_elapsed_qa / 3600:.1f}小时 | 生成 {len(qa_results)} 条QA对")
        return qa_results

    def _save_qa_checkpoint(self, path: str, dois: set, results: list, elapsed: float, processed: int):
        checkpoint = {
            "processed_dois_for_qa": list(dois),
            "qa_results": results,
            "total_elapsed_qa": elapsed,
            "total_processed_qa": processed
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ QA检查点保存失败: {str(e)}")

    def _cleanup_checkpoint(self, path: str):
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"\n✔ QA检查点文件 {os.path.basename(path)} 已清理")
            except Exception as e:
                print(f"⚠ QA检查点清理失败: {str(e)}")


# main_qa_generation.py

# Assume PromptManager is defined in this file or imported
# Assume QAFromStructuredDataProcessor is defined in this file or imported


PROMPT_FOR_QA_FROM_STRUCTURED_JSON = """
You are an AI assistant specialized in creating high-quality question-answer (QA) pairs from structured JSON data. This data was extracted from scientific papers on Transparent Conductive Oxides (TCOs).
Your goal is to generate QA pairs that can be used for fine-tuning other language models.

**Input Data:**
You will be provided with a JSON object representing structured information for a single material entry. Here is an example of the input JSON structure you will receive:
```json
{{
  "Material Name": "In₂O₃:Sn (5 at%)",
  "Details": {{
    "Design": {{
      "Base Material": "In₂O₃",
      "Dopant(s)": "Sn 5 at%",
      "Crystal Structure": "Cubic bixbyite"
    }},
    "Fabrication": {{
      "Method": "Magnetron sputtering",
      "Substrate Temperature": "350 °C",
      "Annealing Conditions": "Not specified"
    }},
    "Properties": {{
      "Electrical Resistivity": "2.1e-4 Ω·cm",
      "Carrier Concentration": "8.5e20 cm⁻³",
      "Hall Mobility": "30 cm²/Vs",
      "Optical Transmittance": "85 % at 550 nm",
      "Band Gap": "3.75 eV"
    }},
    "Application": {{
      "Device Type": "OLED anode",
      "Performance in Device": "Luminance Efficiency 45 cd/A"
    }}
  }}
}}
"""

if __name__ == "__main__":
    # ========== 配置 ==========
    # 输入：你第一个脚本（extraction_framework.py）生成的结构化JSON文件路径
    structured_data_input_path = "structured_info.json"  # 你的结构化数据输出文件

    # 输出：生成的QA对将保存到这里
    qa_output_path = "generated_qa_pairs_from_structured_111.jsonl"  # 使用 .jsonl
    qa_checkpoint_path = "qa_generation_checkpoint.json"

    task_type = "structured_to_qa"  # 新的task_type用于PromptManager
    language = "en"
    # model = "gpt-4" # 或者你的 DeepSeek 模型
    model = "DeepSeek-R1-671B"

    # 初始化OpenAI客户端 (与你原脚本一致)
    client = OpenAI(api_key="sk-MzAxLTExMzc5NzE5ODU0LTE3NDc4Nzc1MjM2MTI=", base_url="https://api.scnet.cn/api/llm/v1")

    # 设置新的Prompt模板
    prompt_mgr_qa = PromptManager()
    prompt_mgr_qa.add_prompt(task_type, language, PROMPT_FOR_QA_FROM_STRUCTURED_JSON)  # 使用上面定义的PROMPT

    # 初始化QA生成器
    qa_processor = QAFromStructuredDataProcessor(client, prompt_mgr_qa, model=model)

    # ========== 读取已抽取的结构化数据 ==========
    print(f"正在从 {structured_data_input_path} 读取结构化数据...")
    try:
        with open(structured_data_input_path, "r", encoding="utf-8") as f:
            structured_entries = json.load(f)  # 这是一个包含多个条目（每个条目对应一篇文献的提取）的列表
        print(f"成功读取 {len(structured_entries)} 条结构化数据条目。")
    except Exception as e:
        print(f"❌ 读取结构化数据文件失败: {e}")
        structured_entries = []

    if not structured_entries:
        print("没有可处理的结构化数据，脚本将退出。")
    else:
        # ========== 处理结构化数据生成QA对 ==========
        all_generated_qa_pairs = qa_processor.process_structured_data_to_qa(
            structured_entries,
            task_type,
            language,
            checkpoint_path=qa_checkpoint_path
        )

        # ========== 保存QA对结果 ==========
        print(f"\n准备保存 {len(all_generated_qa_pairs)} 条QA对到 {qa_output_path}...")
        try:
            # 保存为 JSONL 格式，每行一个QA对，更适合LLM微调
            with open(qa_output_path, "w", encoding="utf-8") as f:
                for qa_pair_item in all_generated_qa_pairs:
                    # qa_pair_item 已经包含了 meta, question, answer, context, source_structured_data
                    f.write(json.dumps(qa_pair_item, ensure_ascii=False) + "\n")
            print(f"✔ QA对已保存至 {qa_output_path}")
        except Exception as e:
            print(f"❌ 保存QA对失败: {e}")
