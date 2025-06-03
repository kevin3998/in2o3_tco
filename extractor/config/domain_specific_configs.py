# src/config/domain_specific_configs.py
import regex as re
from typing import Dict, Any

class DomainConfig:
    def __init__(self, domain_name: str, keyword_groups: dict, blacklist: dict, field_mapping: dict):
        self.domain = domain_name
        self.keyword_groups = keyword_groups
        self.blacklist = blacklist
        self.field_mapping = field_mapping

        self.patterns = {
            cat: re.compile(f"{group.get('en', '')}|{group.get('zh', '')}", re.IGNORECASE)
            for cat, group in self.keyword_groups.items() if isinstance(group, dict)
        }
        materials_group = self.keyword_groups.get('materials', {})
        self.material_pattern = re.compile(
            f"{materials_group.get('en', '')}|{materials_group.get('zh', '')}",
            re.IGNORECASE | re.UNICODE
        ) if materials_group.get('en') or materials_group.get('zh') else None

        blacklist_en = self.blacklist.get('en', '')
        blacklist_zh = self.blacklist.get('zh', '')
        self.blacklist_pattern = re.compile(
            f"{blacklist_en}|{blacklist_zh}",
            re.IGNORECASE
        ) if blacklist_en or blacklist_zh else None

    def is_domain_related(self, text: str) -> bool:
        if not self.material_pattern: return True
        is_related = self.material_pattern.search(text) is not None
        if self.blacklist_pattern:
            is_blacklisted = self.blacklist_pattern.search(text) is not None
            return is_related and not is_blacklisted
        return is_related

    def count_keywords(self, text: str) -> int:
        return sum(len(pattern.findall(text)) for pattern in self.patterns.values())

# --- Config for Membrane Domain (Existing) ---
MEMBRANE_KEYWORDS = {
    "materials": {"en": r"(?i)\b(PVDF|PES|PSf|polysulfone|polyethersulfone|polyimide|ceramic\s+membrane|TFC|Nafion)\b", "zh": r"(聚偏氟乙烯|聚醚砜|聚砜|聚酰亚胺|陶瓷膜|薄膜复合膜|全氟磺酸)"},
    "structure": {"en": r"(?i)\b(hollow\s+fiber|spiral\s+wound|flat\s+sheet)\b", "zh": r"(中空纤维|卷式膜|平板膜)"},
    "process": {"en": r"(?i)\b(phase\s+inversion|electrospinning|interfacial\s+polymerization|sol-gel)\b", "zh": r"(相转化法|静电纺丝|界面聚合|溶胶-凝胶)"},
    "application": {"en": r"(?i)\b(water\s+treatment|desalination|wastewater\s+reclamation|brackish\s+water)\b", "zh": r"(水处理|海水淡化|废水回用|苦咸水)"},
    "metrics": {"en": r"(?i)\b(flux|rejection\s+rate|MWCO|porosity|contact\s+angle|mechanical\s+strength)\b", "zh": r"(通量|截留率|截留分子量|孔隙率|接触角|机械强度)"}
}
MEMBRANE_BLACKLIST = {
    "en": r"\b(cell membrane|biofilm|medical membrane|dialysis)\b",
    "zh": r"(细胞膜|生物膜|医用膜|透析)"
}
MEMBRANE_FIELD_MAPPING = {
    "en": {
        "Design": ["Design", "design", "Material Design"], "Fabrication": ["Fabrication", "fabrication", "Preparation", "Synthesis"], "Performance": ["Performance", "performance characteristics", "Performance Metrics"], "Application": ["Application", "application field", "Usage"],
        "MaterialName": ["Material", "Material Name", "Base Material"], "WaterFlux": ["Water Flux", "Flux", "Permeability"], # ... etc.
    }, "zh": { # (膜材料的中文映射 - 保持或按需修改)
        "Design": ["材料设计", "设计"], "Fabrication": ["制备方法", "制备", "合成"], # ... etc.
    }
}
MEMBRANE_CONFIG = DomainConfig(
    domain_name="membrane",
    keyword_groups=MEMBRANE_KEYWORDS,
    blacklist=MEMBRANE_BLACKLIST,
    field_mapping=MEMBRANE_FIELD_MAPPING
)

# --- Config for In2O3 TCO Domain (New) ---
IN2O3_TCO_KEYWORDS = {
    "materials": {
        "en": r"(?i)\b(In2O3|Indium\s*Oxide|Sn-doped\s*In2O3|ITO|Indium\s*Tin\s*Oxide|W-doped\s*In2O3|IWO|Mo-doped\s*In2O3|IMO|Ti-doped\s*In2O3|ITiO|Zr-doped\s*In2O3|IZrO|Hf-doped\s*In2O3|IHO|IZO|Indium\s*Zinc\s*Oxide|AZO|Aluminum\s*doped\s*ZnO|FTO|Fluorine\s*doped\s*SnO2|GZO|Gallium\s*doped\s*ZnO|TCO|Transparent\s*Conducting\s*Oxide)\b",
        "zh": r"(氧化铟|掺锡氧化铟|ITO|掺钨氧化铟|IWO|掺钼氧化铟|IMO|氧化锌|掺铝氧化锌|AZO|掺镓氧化锌|GZO|透明导电氧化物)"
    },
    "properties": {
        "en": r"(?i)\b(resistivity|sheet\s*resistance|conductivity|carrier\s*concentration|mobility|Hall\s*mobility|optical\s*transmittance|transparency|band\s*gap|work\s*function|figure\s*of\s*merit|FoM|haze)\b",
        "zh": r"(电阻率|方阻|电导率|载流子浓度|迁移率|霍尔迁移率|透光率|透明度|带隙|功函数|品质因数|雾度)"
    },
    "fabrication": {
        "en": r"(?i)\b(sputtering|DC\s*sputtering|RF\s*sputtering|magnetron\s*sputtering|PLD|pulsed\s*laser\s*deposition|ALD|atomic\s*layer\s*deposition|CVD|chemical\s*vapor\s*deposition|sol-gel|spray\s*pyrolysis|annealing|substrate|thin\s*film|deposition|target|precursor)\b",
        "zh": r"(溅射|直流溅射|射频溅射|磁控溅射|脉冲激光沉积|原子层沉积|化学气相沉积|溶胶凝胶|喷雾热解|退火|衬底|薄膜|沉积|靶材|前驱体)"
    },
    "applications": {
        "en": r"(?i)\b(solar\s*cell|photovoltaic|transparent\s*electrode|display|LCD|OLED|LED|touch\s*panel|touch\s*screen|low-emissivity|low-E\s*glass|gas\s*sensor|thin\s*film\s*transistor|TFT|heater)\b",
        "zh": r"(太阳能电池|光伏|透明电极|显示器|液晶显示|有机发光二极管|发光二极管|触摸屏|低辐射玻璃|气体传感器|薄膜晶体管|加热器)"
    }
}
IN2O3_TCO_BLACKLIST = { # Initially empty, add terms if they cause issues
    "en": r"\b(example_of_a_blacklisted_tco_term)\b", # Placeholder
    "zh": r"\b(黑名单示例)\b"  # Placeholder
}
IN2O3_TCO_FIELD_MAPPING = { # Standard Internal Key : [Possible LLM output variations for TCOs]
    "en": {
        "Design": ["Design", "Material Design", "Composition Design"],
        "Fabrication": ["Fabrication", "Deposition", "Synthesis", "Thin Film Growth"],
        "Performance": ["Performance", "Properties", "Optoelectronic Properties", "Electrical Properties", "Optical Properties"],
        "Application": ["Application", "Device Application", "Usage"],
        # Specific fields within categories (examples)
        "MaterialName": ["Material", "Material Name", "TCO Material", "Composition"], # Will be further cleaned by clean_material_name
        "HostMaterial": ["Host Material", "Base Oxide"],
        "Dopant": ["Dopant", "Dopant Element", "Doping Element"],
        "DopingConcentration": ["Doping Concentration", "Dopant Concentration", "at.%", "wt.%"], # Value and unit often together
        "DepositionMethod": ["Deposition Method", "Fabrication Method", "Technique"],
        "AnnealingTemperature": ["Annealing Temperature", "Post-Annealing Temperature"],
        "AnnealingAtmosphere": ["Annealing Atmosphere", "Annealing Ambient"],
        "FilmThickness": ["Film Thickness", "Thickness"],
        "Resistivity": ["Resistivity", "Electrical Resistivity"],
        "SheetResistance": ["Sheet Resistance", "Rs"],
        "CarrierConcentration": ["Carrier Concentration", "Carrier Density", "Ne"],
        "Mobility": ["Mobility", "Hall Mobility", "μH"],
        "OpticalTransmittance": ["Optical Transmittance", "Transmittance", "Transparency"],
        "BandGap": ["Band Gap", "Optical Band Gap", "Eg"],
        "WorkFunction": ["Work Function", "Φ"],
        "FigureOfMerit": ["Figure of Merit", "FoM"]
    },
    "zh": { # Add Chinese mappings if your source documents or prompts are in Chinese
        "Design": ["设计", "材料设计", "组分设计"],
        "Fabrication": ["制备", "薄膜生长", "沉积方法"],
        "Performance": ["性能", "光电性能", "电学性能", "光学性能"],
        "Application": ["应用", "器件应用"],
        "Resistivity": ["电阻率"],
        "OpticalTransmittance": ["透光率"],
        # ... other relevant Chinese mappings
    }
}
IN2O3_TCO_CONFIG = DomainConfig(
    domain_name="in2o3_tco",
    keyword_groups=IN2O3_TCO_KEYWORDS,
    blacklist=IN2O3_TCO_BLACKLIST,
    field_mapping=IN2O3_TCO_FIELD_MAPPING
)

# --- Global Dictionary of Domain Configurations ---
DOMAIN_CONFIGURATIONS: Dict[str, DomainConfig] = {
    "membrane": MEMBRANE_CONFIG,
    "in2o3_tco": IN2O3_TCO_CONFIG,
}

def get_domain_config(domain_name: str) -> DomainConfig:
    config = DOMAIN_CONFIGURATIONS.get(domain_name)
    if not config:
        raise ValueError(f"DomainConfig not found for domain: {domain_name}")
    return config