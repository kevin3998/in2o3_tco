# src/config/domain_specific_configs.py
import regex as re
from typing import Dict, Any

class DomainConfig:
    # ... (DomainConfig class definition remains the same as you have it) ...
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
        ) if materials_group and (materials_group.get('en') or materials_group.get('zh')) else None

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


# --- Config for In2O3 TCO Domain (Refined) ---
IN2O3_TCO_KEYWORDS = { # Keep your existing keywords, ensure they are comprehensive
    "materials": {
        "en": r"(?i)\b(In2O3|Indium\s*Oxide|Sn-doped\s*In2O3|ITO|Indium\s*Tin\s*Oxide|W-doped\s*In2O3|IWO|Mo-doped\s*In2O3|IMO|Ti-doped\s*In2O3|ITiO|Zr-doped\s*In2O3|IZrO|Hf-doped\s*In2O3|IHO|IZO|Indium\s*Zinc\s*Oxide|AZO|Aluminum\s*doped\s*ZnO|FTO|Fluorine\s*doped\s*SnO2|GZO|Gallium\s*doped\s*ZnO|TCO|Transparent\s*Conducting\s*Oxide|Ce-doped\s*In2O3|ICO|In2O3:H|Hydrogenated\s*In2O3)\b",
        "zh": r"(氧化铟|掺锡氧化铟|ITO|掺钨氧化铟|IWO|掺钼氧化铟|IMO|氧化锌|掺铝氧化锌|AZO|掺镓氧化锌|GZO|透明导电氧化物|掺铈氧化铟|ICO|氢化氧化铟)"
    },
    # ... (keep other keyword groups: properties, fabrication, applications - ensure they are comprehensive)
     "properties": {
        "en": r"(?i)\b(resistivity|sheet\s*resistance|conductivity|carrier\s*concentration|mobility|Hall\s*mobility|optical\s*transmittance|transparency|band\s*gap|work\s*function|figure\s*of\s*merit|FoM|haze|crystal\s*structure|crystallinity|grain\s*size|surface\s*roughness)\b",
        "zh": r"(电阻率|方阻|电导率|载流子浓度|迁移率|霍尔迁移率|透光率|透明度|带隙|功函数|品质因数|雾度|晶体结构|结晶度|晶粒尺寸|表面粗糙度)"
    },
    "fabrication": {
        "en": r"(?i)\b(sputtering|DC\s*sputtering|RF\s*sputtering|magnetron\s*sputtering|PLD|pulsed\s*laser\s*deposition|ALD|atomic\s*layer\s*deposition|CVD|chemical\s*vapor\s*deposition|sol-gel|spray\s*pyrolysis|annealing|substrate|thin\s*film|deposition|target|precursor|reactive\s*plasma\s*deposition|RPD)\b",
        "zh": r"(溅射|直流溅射|射频溅射|磁控溅射|脉冲激光沉积|原子层沉积|化学气相沉积|溶胶凝胶|喷雾热解|退火|衬底|薄膜|沉积|靶材|前驱体|反应等离子体沉积)"
    },
    "applications": {
        "en": r"(?i)\b(solar\s*cell|photovoltaic|transparent\s*electrode|display|LCD|OLED|LED|touch\s*panel|touch\s*screen|low-emissivity|low-E\s*glass|gas\s*sensor|thin\s*film\s*transistor|TFT|heater|optoelectronic)\b",
        "zh": r"(太阳能电池|光伏|透明电极|显示器|液晶显示|有机发光二极管|发光二极管|触摸屏|低辐射玻璃|气体传感器|薄膜晶体管|加热器|光电器件)"
    }
}
IN2O3_TCO_BLACKLIST = {
    "en": r"\b(non_tco_oxide_example)\b", # Adjust as needed
}

# This FIELD_MAPPING is CRUCIAL. Keys are the Pydantic schema aliases (PascalCase).
# Values are lists of variations the LLM might output.
IN2O3_TCO_FIELD_MAPPING = {
    "en": {
        # Top-level categories for Details object (as defined in Pydantic schemas)
        "Design": ["Design", "Material Design", "Composition Design", "Compositional Details"],
        "Fabrication": ["Fabrication", "Deposition", "Synthesis", "Thin Film Growth", "Preparation"],
        "Performance": ["Performance", "Properties", "Optoelectronic Properties", "Electrical Properties", "Optical Properties", "Structural Properties", "Characterization"],
        "Application": ["Application", "Device Application", "Usage", "Potential Application"],

        # Design sub-fields (matching aliases in TCODesignSchema & TCODopantSchema)
        "HostMaterial": ["HostMaterial", "Host Material", "Base Oxide", "Matrix Material"],
        "PrimaryDopant": ["PrimaryDopant", "Primary Dopant"], # This key expects an object
            "Element": ["Element", "DopantElement", "DopingElement"], # Sub-key of PrimaryDopant & CoDopants
            "Concentration_text": ["Concentration_text", "DopantConcentrationText", "Concentration (text)"], # Sub-key
        "CoDopants": ["CoDopants", "Co-dopants", "Additional Dopants"], # This key expects a list of objects
        "TargetStoichiometry": ["TargetStoichiometry", "Target Stoichiometry", "Stoichiometric Ratio", "Nominal Composition"],
        "MaterialDescriptionSource": ["MaterialDescriptionSource", "Source of Material", "Target Source"],

        # Fabrication sub-fields (matching aliases in TCOFabricationSchema, TCODepositionParamsSchema, TCOAnnealingSchema)
        "DepositionMethod": ["DepositionMethod", "Deposition Technique", "Fabrication Method"],
        "SubstrateMaterial": ["SubstrateMaterial", "Substrate", "Underlayer"],
        "PrecursorMaterialsText": ["PrecursorMaterialsText", "Precursors", "Starting Materials"],
        "TargetMaterialText": ["TargetMaterialText", "Target Material", "Sputtering Target", "Target Composition for Deposition"],
        "DepositionParameters": ["DepositionParameters", "Deposition Conditions"], # Expects an object
            "BasePressure": ["BasePressure", "Base Pressure"],
            "WorkingPressure": ["WorkingPressure", "Working Pressure", "Sputtering Pressure", "Deposition Pressure"],
            "DepositionTemperature": ["DepositionTemperature", "Substrate Temperature", "Growth Temperature", "T_sub", "SubstrateTemp"],
            "GasAtmosphere": ["GasAtmosphere", "Deposition Atmosphere", "Process Gases", "Sputtering Gas"],
            "GasFlowRates": ["GasFlowRates", "Gas Flow Rate"],
            "DepositionPower": ["DepositionPower", "Sputtering Power", "RF Power", "DC Power"],
            "DepositionTime": ["DepositionTime", "Growth Time"],
            "DepositionRate": ["DepositionRate", "Growth Rate"],
        "DepositionParametersTextSummary": ["DepositionParametersTextSummary", "General Deposition Info"],
        "AnnealingConditions": ["AnnealingConditions", "Post-Annealing Treatment"], # Expects an object
            "Temperature": ["Temperature", "Annealing Temperature", "Anneal Temp"], # Sub-key
            "Atmosphere": ["Atmosphere", "Annealing Atmosphere", "Anneal Ambient"], # Sub-key
            "Duration": ["Duration", "Annealing Duration", "Annealing Time"], # Sub-key
        "AnnealingConditionsTextSummary": ["AnnealingConditionsTextSummary", "General Annealing Info"],
        "FilmThicknessText": ["FilmThicknessText", "Film Thickness", "Thickness"], # For string like "150 nm"
        "FilmThicknessValue": ["FilmThicknessValue"], # For numeric part if LLM separates
        "FilmThicknessUnit": ["FilmThicknessUnit"],   # For unit part if LLM separates

        # Performance sub-fields (matching aliases in various TCOPerformance sub-schemas)
        "ElectricalProperties": ["ElectricalProperties", "Electrical Properties"], # Expects object
            "Resistivity": ["Resistivity", "Electrical Resistivity", "ρ"],
            "SheetResistance": ["SheetResistance", "Sheet Resistance", "Rs", "R_sheet"],
            "Conductivity": ["Conductivity", "Electrical Conductivity", "σ"],
            "CarrierConcentration": ["CarrierConcentration", "Carrier Density", "Carrier Conc.", "Ne", "Nh"],
            "CarrierType": ["CarrierType", "Type", "ConductionType"],
            "HallMobility": ["HallMobility", "Mobility", "μH", "Hall Mobility value"],
        "OpticalProperties": ["OpticalProperties", "Optical Properties"], # Expects object
            "AverageTransmittance": ["AverageTransmittance", "Average Transmittance", "Mean Transmittance"],
            "TransmittanceAt550nm": ["TransmittanceAt550nm", "Transmittance at 550nm"],
            "WavelengthRangeTransmittance": ["WavelengthRangeTransmittance", "Transmittance Range"],
            "OpticalBandGapText": ["OpticalBandGapText", "Optical Band Gap", "Band Gap", "Eg"],
            "BandGapDeterminationMethod": ["BandGapDeterminationMethod", "Tauc Plot Method"],
            "OpticalTransmittanceDescription": ["OpticalTransmittanceDescription", "Optical Transmittance", "Transparency (general description)"], # New field for general text
            "Haze": ["Haze", "Haze Factor"],
        "StructuralProperties": ["StructuralProperties", "Structural Properties", "Crystallographic Properties"], # Expects object
            "CrystalStructure": ["CrystalStructure", "Crystalline Structure", "Phase", "Phase Composition"],
            "PreferredOrientation": ["PreferredOrientation", "Crystal Orientation"],
            "GrainSize": ["GrainSize", "Average Grain Size"],
        "OtherPerformanceMetrics": ["OtherPerformanceMetrics", "Other Properties"], # Expects object
            "WorkFunctionText": ["WorkFunctionText", "Work Function", "Φ"],
            "FigureOfMeritValue": ["FigureOfMeritValue", "Figure of Merit (value)"],
            "FigureOfMeritType": ["FigureOfMeritType", "Figure of Merit (type)", "FoM Type"],
            "SurfaceRoughnessRMS": ["SurfaceRoughnessRMS", "RMS Roughness", "Roughness (RMS)"],
            "Stability": ["Stability", "Material Stability"],
        "GeneralPerformanceSummary": ["GeneralPerformanceSummary", "Performance Summary"],


        # Application sub-fields
        "PotentialApplicationArea": ["PotentialApplicationArea", "Application Area", "Potential Use"],
        "DemonstratedInDevice": ["DemonstratedInDevice", "Device Integration"], # Expects object
            "DeviceType": ["DeviceType", "Type of Device"],
            "TCORoleInDevice": ["TCORoleInDevice", "Role in Device"],
            "DevicePerformanceMetricName": ["DevicePerformanceMetricName", "Device Metric"],
            "DevicePerformanceMetricValue": ["DevicePerformanceMetricValue", "Device Performance"],

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
    "in2o3_tco": IN2O3_TCO_CONFIG,
}

def get_domain_config(domain_name: str) -> DomainConfig:
    config = DOMAIN_CONFIGURATIONS.get(domain_name)
    if not config:
        raise ValueError(f"DomainConfig not found for domain: {domain_name}")
    return config