# 3.extraction_framework.py
import regex as re
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os

class PromptManager:
    def __init__(self):
        self.templates = {}

    def add_prompt(self, domain: str, language: str, template: str):
        self.templates[(domain, language)] = template

    def get_prompt(self, domain: str, language: str, text: str) -> str:
        template = self.templates.get((domain, language))
        if not template:
            raise ValueError(f"Prompt template not found for domain '{domain}' and language '{language}'")
        return template.format(text=text)


class DomainConfig:
    def __init__(self, domain_name: str, keyword_groups: dict, blacklist: dict, field_mapping: dict):  # 新增字段映射
        self.domain = domain_name
        self.keyword_groups = keyword_groups
        self.blacklist = blacklist
        self.field_mapping = field_mapping

        self.patterns = {
            cat: re.compile(f"{group['en']}|{group['zh']}", re.IGNORECASE)
            for cat, group in self.keyword_groups.items()
        }
        self.material_pattern = re.compile(
            f"{self.keyword_groups['materials']['en']}|{self.keyword_groups['materials']['zh']}",
            re.IGNORECASE | re.UNICODE
        )
        self.blacklist_pattern = re.compile(
            f"{blacklist['en']}|{blacklist['zh']}",
            re.IGNORECASE
        )

    def is_domain_related(self, text: str) -> bool:
        return self.material_pattern.search(text) is not None and not self.blacklist_pattern.search(text)

    def count_keywords(self, text: str) -> int:
        return sum(len(pattern.findall(text)) for pattern in self.patterns.values())


class PaperProcessor:
    def __init__(self, client: OpenAI, prompt_manager: PromptManager, config: DomainConfig, model="gpt-4"):
        self.client = client
        self.prompt_manager = prompt_manager
        self.config = config
        self.model = model

    def process_single_paper(self, paper: dict, domain: str, language: str) -> List[Dict]:
        full_text = self._get_full_text(paper)
        prompt = self.prompt_manager.get_prompt(domain, language, full_text)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert academic extraction assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    stream=False
                )
                parsed = self._parse_response(response.choices[0].message.content, full_text, paper)
                return self._standardize_fields(parsed, language)  # 新增字段标准化
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"请求失败，{delay}s后重试（{attempt + 1}/{max_retries}）")
                    time.sleep(delay)
                    continue
                print(f"❌ 最终请求失败：{str(e)}")
                return []

    # 新增字段标准化方法（使用field_mapping）
    def _standardize_fields(self, data: List[Dict], lang: str) -> List[Dict]:
        mapping = self.config.field_mapping.get(lang, {})
        reverse_mapping = {v: k for k, vals in mapping.items() for v in vals}
        for entry in data:
            details = entry["output"]["Details"]
            for old_key in list(details.keys()):
                if new_key := reverse_mapping.get(old_key):
                    details[new_key] = details.pop(old_key)
        return data

    def process_papers_with_checkpoint(self, papers: List[Dict], domain: str, language: str,
                                       checkpoint_path="checkpoint.json") -> List[Dict]:
        default_checkpoint = {
            "processed_dois": [],
            "results": [],
            "total_elapsed": 0.0,
            "total_processed": 0
        }

        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                checkpoint = {**default_checkpoint, **checkpoint}
                print(
                    f"▶ 检查点已加载 | 累计处理 {checkpoint['total_processed']} 篇 | 累计用时 {checkpoint['total_elapsed']:.1f}s")
                processed_dois = set(checkpoint["processed_dois"])
                results = checkpoint["results"]
                total_elapsed = checkpoint["total_elapsed"]
                total_processed = checkpoint["total_processed"]
            except Exception as e:
                print(f"⚠ 检查点加载失败: {str(e)}，重新开始处理")
                processed_dois = set()
                results = []
                total_elapsed = 0.0
                total_processed = 0
        else:
            processed_dois = set()
            results = []
            total_elapsed = 0.0
            total_processed = 0

        total = len(papers)
        try:
            for idx, paper in enumerate(papers):
                session_start = time.time()
                doi = paper.get("doi", f"paper_{idx}")
                if doi in processed_dois:
                    continue
                try:
                    new_results = self.process_single_paper(paper, domain, language)
                    results.extend(new_results)
                    processed_dois.add(doi)

                    session_elapsed = time.time() - session_start
                    total_elapsed += session_elapsed
                    total_processed += 1
                    eta = (total_elapsed / total_processed) * (total - total_processed) if total_processed else 0

                    print(
                        f"\r✔ 进度 [{total_processed}/{total}] | 本次用时 {session_elapsed:.1f}s | 总用时 {total_elapsed:.1f}s | 预计剩余 {eta / 3600:.1f}h",
                        end="",
                        flush=True
                    )

                    # 定期保存检查点（保持原逻辑）
                    if total_processed % 5 == 0:
                        self._save_checkpoint(checkpoint_path, processed_dois, results, total_elapsed, total_processed)

                except KeyboardInterrupt:
                    print("\n⚠ 用户中断，保存进度...")
                    self._save_checkpoint(checkpoint_path, processed_dois, results, total_elapsed, total_processed)
                    raise
                except Exception as e:
                    print(f"\n⚠ 文献处理失败: {doi} | 错误: {str(e)}")
                    self._save_checkpoint(checkpoint_path, processed_dois, results, total_elapsed, total_processed)
                    continue

            self._cleanup_checkpoint(checkpoint_path)

        except Exception as e:
            print(f"\n⚠ 未处理异常: {str(e)}")
            self._save_checkpoint(checkpoint_path, processed_dois, results, total_elapsed, total_processed)
            raise

        print(f"\nAll processing completed, a total of {len(results)} valid information")
        print(f"\n✅ 全部处理完成 | 总用时 {total_elapsed / 3600:.1f}小时")
        return results

    def _save_checkpoint(self, path: str, dois: set, results: list, elapsed: float, processed: int):
        checkpoint = {
            "processed_dois": list(dois),
            "results": results,
            "total_elapsed": elapsed,
            "total_processed": processed
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ 检查点保存失败: {str(e)}")

    def _cleanup_checkpoint(self, path: str):
        """安全删除检查点文件"""
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"\n✔ 检查点文件 {os.path.basename(path)} 已清理")
            except Exception as e:
                print(f"⚠ 检查点清理失败: {str(e)}")

    def _parse_response(self, content: str, input_text: str, paper_meta: dict) -> List[Dict]:
        try:
            json_str = re.sub(r"[\x00-\x1F]", "", content)
            blocks = re.findall(r"\{(?:[^{}]|(?R))*\}", json_str, re.DOTALL)

            parsed_data = []
            for raw_block in blocks:
                try:
                    raw_data = json.loads(raw_block, strict=False)
                    standardized = self._recursive_standardize_keys(raw_data)
                    entries = standardized.get("output") or standardized.get("Output") or []
                    if not entries:
                        print("⚠️ 模型返回中未检测到有效 output 字段。")
                    if not isinstance(entries, list):
                        entries = [entries]

                    for entry in entries:
                        material = self._extract_material(entry)
                        clean_material = self._clean_material_name(material)

                        details = entry.get("Details", entry.get("details", {}))
                        details = self._standardize_details(details)

                        parsed_data.append({
                            "meta": {
                                "title": paper_meta.get("title", ""),
                                "doi": paper_meta.get("doi", ""),
                                "source": paper_meta.get("journal", ""),
                                "year": paper_meta.get("year", "")
                            },
                            "input": input_text,
                            "output": {
                                "Material": clean_material,
                                "Details": details
                            }
                        })
                except Exception as e:
                    print(f"解析失败: {e}\n{raw_block[:200]}")
            return parsed_data
        except Exception as e:
            print(f"总解析失败: {e}")
            return []

    def _recursive_standardize_keys(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {self._format_key(k): self._recursive_standardize_keys(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._recursive_standardize_keys(item) for item in data]
        return data

    def _format_key(self, key: str) -> str:
        if key.lower().replace(" ", "") == "materialname":
            return "Material"
        return key[0].upper() + key[1:].lower() if key else key

    def _extract_material(self, entry: dict) -> str:
        """从 entry 中多路径尝试提取材料名称"""
        search_paths = [
            ["Material"],
            ["Design", "Material"],
            ["material"],
            ["Design", "material"],
            ["Composition", "Base"],
            ["Composition"]
        ]
        for path in search_paths:
            try:
                current = entry
                for key in path:
                    current = current[key]
                if current:
                    if isinstance(current, list):
                        current = current[0]
                    return str(current).split("(")[0].split(",")[0].strip()
            except Exception:
                continue
        return "Unknown"

    def _find_material_in_text(self, text: str) -> Optional[str]:
        # 从DomainConfig动态获取材料关键词
        material_pattern = self.config.material_pattern
        match = material_pattern.search(text)
        return match.group() if match else None

    def _clean_material_name(self, material: str) -> str:
        """材料名称清洗"""
        # 允许Unicode下标、希腊字母和化学符号
        cleaned = re.sub(
            r"[^\w\s\-()₀₁₂₃₄₅₆₇₈₉αβγδεζηθικλμνξπρσςτυφχψωΔ]",
            "",
            material.split("(")[0].split(",")[0].split(";")[0]
        ).strip()
        cleaned = re.sub(r"\s*\([^)]+\)", "", material)  # 移除括号内容
        cleaned = re.sub(r"\b\w+\d+$", "", cleaned)
        return cleaned if cleaned else "Unknown"

    def _standardize_details(self, details: dict) -> dict:
        """标准化详情字段结构"""
        required_sections = ["Design", "Fabrication", "Performance", "Application"]
        return {
            section: details.get(section, {})
            for section in required_sections
        }

    def _get_full_text(self, paper: Dict) -> str:
        if fulltext := paper.get("fulltext", ""):
            return fulltext.strip()
        if abstract := paper.get("abstract_as_paragraph", ""):
            return abstract.strip()
        return ""

    def _split_paragraphs(self, text: str) -> List[str]:
        split_pattern = r"(?:\n{2,}|(?<=[\.。；;!？?])\s{1,})"
        return [chunk.strip() for chunk in re.split(split_pattern, text) if chunk.strip()]

    def _detect_keywords(self, text: str) -> Dict:
        result = {"total": 0}
        for pattern in self.config.patterns.values():
            result["total"] += len(pattern.findall(text))
        return result

    def _get_entries(self, data: dict) -> List[dict]:
        """提取有效条目列表"""
        if "Output" in data and isinstance(data["Output"], list):
            return data["Output"]
        elif "output" in data and isinstance(data["output"], list):
            return data["output"]
        return [data]

# 示例配置：膜材料
tco_keywords = {
    "materials": {
        "en": r"(?i)\b(In₂O₃|ITO|AZO|FTO|TCO|transparent conductive oxide|doped semiconductor)\b",
        "zh": r"(氧化铟锡|掺锡氧化铟|铝掺杂氧化锌|氟掺杂氧化锡|透明导电氧化物|掺杂半导体)"
    },
    "properties": {
        "en": r"(?i)\b(resistivity|carrier concentration|mobility|transmittance|band gap|work function)\b",
        "zh": r"(电阻率|载流子浓度|迁移率|透光率|带隙|功函数)"
    },
}
tco_blacklist = {
    "en": r"\b(graphene|perovskite|quantum dot|organic semiconductor|CNT)\b",
    "zh": r"(石墨烯|钙钛矿|量子点|有机半导体|碳纳米管)"
}


tco_prompt_en = """Your task is to meticulously analyze the following scientific text about Transparent Conductive Oxides (TCOs).
Your primary goal is to extract structured information for **EVERY distinct doped TCO material composition** discussed in the text.

**Output Format Instructions:**
- Your entire response MUST be a single, valid JSON object.
- This JSON object must contain one top-level key: "output".
- The value of "output" MUST be a list of JSON objects. Each object in this list represents one distinct TCO material composition found.
- Adhere strictly to the field names and nested structure shown in the example below.
- All string values within the JSON must be properly escaped.

**Handling Missing Information:**
- If information for any specific field (e.g., "Crystal Structure", "Substrate Temperature", "Band Gap") is explicitly searched for but not found in the text for a given material, you MUST use the string value "Not specified" for that field.
- Do NOT omit field keys if the information is simply not present; use "Not specified" instead.

**Extraction Details for Each Material Composition:**

1.  **`Material Name`**: (String) Provide the full chemical name or common abbreviation of the TCO material, including dopants and their concentrations if specified (e.g., "In₂O₃:Sn (5 at%)", "ZnO:Al (2 wt%)", "IGZO").
2.  **`Details`**: (Object) A nested object containing the following four sections:
    a.  **`Design`**: (Object) Information about the material's composition and intended structure.
        -   `"Base Material"`: (String) The undoped base oxide material (e.g., "In₂O₃", "ZnO").
        -   `"Dopant(s)"`: (String) List all dopant elements and their concentrations if available (e.g., "Sn 5 at%", "Al 2 wt%, Ga 1 wt%"). If multiple, combine into a single descriptive string.
        -   `"Crystal Structure"`: (String) The reported crystal structure (e.g., "Cubic bixbyite", "Wurtzite", "Amorphous").
    b.  **`Fabrication`**: (Object) Details about how the material was synthesized or deposited.
        -   `"Method"`: (String) The primary fabrication or deposition technique (e.g., "Magnetron sputtering", "Sol-gel spin coating", "Pulsed Laser Deposition").
        -   `"Substrate Temperature"`: (String) Key temperature of the substrate during deposition, including units (e.g., "350 °C", "Room Temperature").
        -   `"Annealing Conditions"`: (String) Post-deposition annealing details if specified (e.g., "500 °C in N₂ for 1 hour", "Rapid Thermal Annealing at 400 °C").
    c.  **`Properties`**: (Object) Key reported physical and chemical performance metrics. Provide values along with their units as a single string.
        -   `"Electrical Resistivity"`: (String) e.g., "2.1e-4 Ω·cm".
        -   `"Carrier Concentration"`: (String) e.g., "8.5e20 cm⁻³".
        -   `"Hall Mobility"`: (String) e.g., "35 cm²/Vs".
        -   `"Optical Transmittance"`: (String) Include percentage and wavelength/range if specified, e.g., "85 % at 550 nm", ">90 % (Visible spectrum)".
        -   `"Band Gap"`: (String) e.g., "3.7 eV".
        -   *Include other relevant properties if clearly stated and quantifiable.*
    d.  **`Application`**: (Object) Mentioned uses or device integrations of the material.
        -   `"Device Type"`: (String) The specific device where the TCO is used (e.g., "Solar cell electrode", "OLED anode", "Thin film transistor channel").
        -   `"Performance in Device"`: (String) Key device performance metric directly related to the TCO material, if reported (e.g., "Solar cell efficiency 15%", "OLED external quantum efficiency 20%").

**Example JSON Output Structure:**
```json
{{
  "output": [
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
    }},
    {{
      "Material Name": "ZnO:Al (2 wt%)",
      "Details": {{
        "Design": {{
          "Base Material": "ZnO",
          "Dopant(s)": "Al 2 wt%",
          "Crystal Structure": "Wurtzite"
        }},
        "Fabrication": {{
          "Method": "Spray pyrolysis",
          "Substrate Temperature": "450 °C",
          "Annealing Conditions": "400 °C in vacuum for 30 min"
        }},
        "Properties": {{
          "Electrical Resistivity": "5.0e-4 Ω·cm",
          "Carrier Concentration": "Not specified",
          "Hall Mobility": "15 cm²/Vs",
          "Optical Transmittance": ">90 % in visible range",
          "Band Gap": "3.3 eV"
        }},
        "Application": {{
          "Device Type": "Transparent heater",
          "Performance in Device": "Achieved 150 °C at 5V"
        }}
      }}
    }}
  ]
}}
"""

from openai import OpenAI

# 初始化
input_path = "./output/extracted_papers_pycharm.json"
output_path = "./output/structured_extraction.json"
checkpoint_path = "checkpoint.json"

domain = "In2O3"
language = "en"
model = "DeepSeek-R1-671B"  # 可换为 "DeepSeek-R1-671B"

client = OpenAI(api_key="sk-MzAxLTExMzc5NzE5ODU0LTE3NDc4Nzc1MjM2MTI=", base_url="https://api.scnet.cn/api/llm/v1")

prompt_mgr = PromptManager()
prompt_mgr.add_prompt(domain, language,tco_prompt_en)

tco_field_mapping_en = {
    "base_material": ["Base Material", "base_material", "Matrix Material"],
    "dopants": ["Dopant(s)", "Dopant", "Dopants", "Dopant Type/Concentration"],
    "crystal_structure": ["Crystal Structure", "Crystal structure", "structure"],
    "method": ["Method", "Fabrication Method", "Deposition method", "Technique"],
    "substrate_temperature": ["Substrate Temperature", "Substrate Temp", "Growth Temperature", "Deposition Temperature"],
    "annealing_conditions": ["Annealing Conditions", "Annealing", "Post-annealing treatment"],
    "electrical_resistivity": ["Electrical Resistivity", "Resistivity", "ρ"],
    "carrier_concentration": ["Carrier Concentration", "Carrier Density", "n_e", "n_h"],
    "hall_mobility": ["Hall Mobility", "Mobility", "μ_H"],
    "optical_transmittance": ["Optical Transmittance", "Transmittance", "Transparency", "T%"],
    "band_gap": ["Band Gap", "Optical Bandgap", "Eg"],
    "device_type": ["Device Type", "Application Device", "Device"],
    "performance_in_device": ["Performance in Device", "Device Performance", "Application Metric"]
}
config = DomainConfig(
    domain,
    keyword_groups=tco_keywords,
    blacklist=tco_blacklist,
    field_mapping=tco_field_mapping_en  # 新增
)
processor = PaperProcessor(client, prompt_mgr, config, model=model)

# ========== 读取文献数据并处理 ==========
with open(input_path, "r", encoding="utf-8") as f:
    papers = json.load(f)

results = processor.process_papers_with_checkpoint(papers, domain, language, checkpoint_path=check6ed4xpoint_path)

# ========== 保存结果 ==========
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✔ 抽取结果已保存至 {output_path}")