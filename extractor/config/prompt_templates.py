"""
Module: prompt_templates
Functionality: Stores and manages prompt templates used for LLM interactions.
               Provides functions to load these templates into a PromptManager
               instance, allowing for domain-specific and language-specific prompts.
"""
from extractor.utils.general_utils import PromptManager # Corrected import path

# --- Define prompt templates here ---

MEMBRANE_PROMPT_EN = """You are given the cleaned full text of a membrane materials paper (excluding references and keywords).
Please extract structured information for **each material mentioned in the text**. If multiple membrane materials are described (e.g., PVDF, PES, PSF), generate a separate entry for each.

For each material, extract and return the following 4 categories of information (omit any missing category):
1. **Design** (e.g., material name, composition, structure, composite ratio, additives)
2. **Fabrication** (e.g., method, steps, key parameters)
3. **Performance** (e.g., water flux, rejection rate, hydrophilicity, antifouling, porosity, contact angle, mechanical strength)
4. **Application** (e.g., application field, influent water quality, treatment targets)

Strictly implement the following requirements:
1. Return a strict JSON object in the following format for the entire paper, containing a list in the "output" field where each item corresponds to one material:
{{
  "output": [ // List of materials found in the paper
    {{
      "Material": "PVDF", // The primary material this entry is about
      "Details": {{
        "Design": {{
          "Material": "PVDF", // Can be more specific e.g. "PVDF-co-HFP"
          "Structure": "hollow fiber",
          "Composite Ratio": "80/20 wt%",
          "Additives": "GO nanoparticles (0.5 wt%)"
        }},
        "Fabrication": {{
          "Method": "Non-solvent induced phase separation (NIPS)",
          "Key Parameters": "Polymer concentration: 18 wt%, Casting solution temperature: 25°C, Coagulation bath: water at 40°C"
        }},
        "Performance": {{
          "Water Flux": "150 L/m²·h (at 1 bar)",
          "Rejection Rate": "95% for BSA (Bovine Serum Albumin)",
          "Contact Angle": "75°",
          "Porosity": "80%"
        }},
        "Application": {{
          "Scenario": "Textile wastewater treatment",
          "Influent Water Quality": {{ // This can be a nested object
            "COD": "500 mg/L",
            "Dye concentration": "50 mg/L (Reactive Blue 19)"
          }},
          "Treatment Target": "Decolorization rate >90%, COD removal > 80%"
        }}
      }}
    }},
    {{
      "Material": "PES", // Second material entry if found
      "Details": {{ ... }}
    }}
    // ... more material entries if applicable
  ]
}}
Only return the JSON. Do not include any extra explanations or markdown backticks.
2. The main key of the returned JSON object MUST be "output", and its value MUST be a list of material entries.
3. Inside each material entry, "Material" should be the identified main material name, and "Details" should contain the 4 categories.
4. If a category or sub-field is not mentioned, omit it from the "Details" or its sub-object.
Input Text:
{text}
""" # Note: The prompt's example JSON should be as close as possible to what you desire.
      # The {text} placeholder will be filled by PromptManager.

# This would be in your src/config/prompt_templates.py


# This would be in your src/config/prompt_templates.py

# --- In2O3 TCO Prompt (Refined and Complete) ---
# --- In2O3 TCO Prompt (FINAL CORRECTED VERSION) ---
IN2O3_TCO_PROMPT_EN = """You are an expert academic extraction assistant specializing in materials science. You will be given the full text of a paper on Indium Oxide (In2O3) based Transparent Conducting Oxides (TCOs).
Your task is to extract structured information for **each distinct TCO composition or sample that has unique properties or fabrication conditions described in the text**.

For each distinct material/sample, you MUST extract the following categories of information. If information for a category or a specific sub-field is not present, omit that key from the output.

1.  **Design**:
    * `HostMaterial`: (e.g., "In2O3").
    * `PrimaryDopant`: This MUST BE A SINGLE JSON OBJECT describing the main dopant. Include its `Element` (e.g., "Sn") and `Concentration_text` (e.g., "5 at.%", "Target: 10 wt.% SnO2"). If no specific primary dopant is mentioned, this can be an empty object `{{}}`.
    * `CoDopants`: This MUST BE A LIST OF JSON OBJECTS if co-dopants are present. Each object must have `Element` and `Concentration_text`. If none, provide an empty list `[]`.
    * `TargetStoichiometry`: (e.g., "In(2-x)SnxO3", "Target: In2O3 with 1 wt% WO3").
    * `MaterialDescriptionSource`: (e.g., "Commercial ITO target (99.99% purity)").

2.  **Fabrication**:
    * `DepositionMethod`: (e.g., "DC Magnetron Sputtering").
    * `SubstrateMaterial`: (e.g., "Glass (Corning Eagle)").
    * `TargetMaterialText`: For sputtering/PLD, extract the full description (e.g., "Ceramic target of In2O3 with 10 wt.% SnO2").
    * `DepositionParameters`: This MUST BE A JSON OBJECT. Extract specific key-value pairs like `BasePressure`, `WorkingPressure`, `DepositionTemperature`, `GasAtmosphere`, `GasFlowRates`, `DepositionPower`.
    * `DepositionParametersTextSummary`: Use this field ONLY if specific parameters for the `DepositionParameters` object are NOT available, but a general description (e.g., "films were deposited at low temperature") is. If specific parameters are extracted, OMIT this summary field.
    * `AnnealingConditions`: This MUST BE A JSON OBJECT with keys like `Temperature`, `Atmosphere`, `Duration`.
    * `AnnealingConditionsTextSummary`: Use this field ONLY if specific parameters for the `AnnealingConditions` object are NOT available. If specific parameters are extracted, OMIT this summary field.
    * `FilmThicknessText`: Extract the full thickness description as text (e.g., "approx. 150 nm", "100-1200 nm").

3.  **Performance**:
    * **CRITICAL:** All performance metrics MUST be nested within their corresponding sub-objects (`ElectricalProperties`, `OpticalProperties`, etc.) as shown in the example.
    * `ElectricalProperties`: A JSON object containing:
        * `Resistivity`: (e.g., "3.5 x 10^-4 Ω·cm").
        * `SheetResistance`: (e.g., "10 Ω/sq").
        * `CarrierConcentration`: (e.g., "6.2 x 10^20 cm⁻³").
        * `HallMobility`: (e.g., "45 cm²/Vs").
        * `CarrierType`: (e.g., "n-type").
    * `OpticalProperties`: A JSON object containing:
        * `AverageTransmittance`: For average values (e.g., ">90% in visible range (400-700nm)").
        * `TransmittanceAt550nm`: For specific values (e.g., "92% at 550 nm").
        * `OpticalBandGapText`: Extract the full text value (e.g., "3.75 eV (Tauc plot)").
        * `OpticalTransmittanceDescription`: Use for general textual descriptions (e.g., "High in visible and near-infrared regions") if specific values are not available.
        * `Haze`: (e.g., "< 1%").
    * `StructuralProperties`: A JSON object containing:
        * `CrystalStructure`: (e.g., "Cubic bixbyite", "Amorphous").
        * `PreferredOrientation`: (e.g., "(222)").
        * `GrainSize`: (e.g., "30-50 nm").
    * `OtherPerformanceMetrics`: A JSON object containing:
        * `WorkFunctionText`: (e.g., "4.8 eV (UPS)").
        * `FigureOfMeritValue`: (e.g., "15 x 10^-3 Ω⁻¹").
        * `SurfaceRoughnessRMS`: (e.g., "0.5 nm (RMS)").

4.  **Application**:
    * `PotentialApplicationArea`: (e.g., "Transparent electrode for perovskite solar cells").
    * `DevicePerformance`: If used in a device and performance is reported (e.g., "Solar cell efficiency (PCE): 18.5%").

**Strictly implement the following requirements:**
1.  **JSON Format:** Return a single strict JSON object. The root of this object MUST be an `"output"` key, and its value MUST be a list of material entries. Do not include any extra text or markdown backticks.
2.  **Data Specificity:** Extract data that corresponds ONLY to a specific, unique material or sample's synthesis and characterization as described in the paper. If the text contains general review tables listing typical properties for a class of materials (e.g., a summary of different TCOs), **DO NOT** use that data to populate the fields for a specific experimental sample mentioned elsewhere.
3.  **Field Nesting:** All performance metrics MUST be nested inside their respective sub-objects (`ElectricalProperties`, `OpticalProperties`, `StructuralProperties`, `OtherPerformanceMetrics`) as shown in the example below. Do not place fields like `Resistivity` or `HallMobility` directly under the `Performance` object.
4.  **Omit Missing Information:** If a category, sub-object, or field is not mentioned in the text for a specific material, omit that key entirely from the JSON output. Do not use `null` or empty strings for missing information.

**Example of Desired Output Structure:**
{{
  "output": [
    {{
      "MaterialName": "ICO:H film (post-annealed)",
      "Details": {{
        "Design": {{
          "HostMaterial": "In2O3",
          "PrimaryDopant": {{ "Element": "Ce", "Concentration_text": "3 wt % CeO2 in target" }},
          "CoDopants": [{{ "Element": "H", "Concentration_text": "1.3 at.%" }}],
          "MaterialDescriptionSource": "ICO pellets (Sumitomo Metal Mining)"
        }},
        "Fabrication": {{
          "DepositionMethod": "dc arc-discharge ion plating (IP)",
          "SubstrateMaterial": "Corning Eagle XG glass",
          "DepositionParameters": {{
            "DepositionTemperature": "150 °C",
            "WorkingPressure": "0.45 Pa",
            "GasAtmosphere": "Ar, O2 (8–13 vol %), H2 (1.0 vol %)"
          }},
          "AnnealingConditions": {{
            "Temperature": "200 °C",
            "Duration": "30 min",
            "Atmosphere": "air"
          }},
          "FilmThicknessText": "100-nm-thick"
        }},
        "Performance": {{
          "ElectricalProperties": {{
            "HallMobility": "130–145 cm2 V-1 s-1",
            "CarrierType": "n-type"
          }},
          "StructuralProperties": {{
            "CrystalStructure": "bixbyte In2O3 polycrystalline structure"
          }},
          "OtherPerformanceMetrics": {{
            "SurfaceRoughnessRMS": "0.296 nm"
          }}
        }},
        "Application": {{
          "PotentialApplicationArea": "Transparent electrodes for solar cells"
        }}
      }}
    }}
  ]
}}

Input Text:
{text}
"""

# Remember to update load_prompts in src/config/prompt_templates.py
# def load_prompts(prompt_manager: PromptManager):
#     prompt_manager.add_prompt(domain="membrane", language="en", template=MEMBRANE_PROMPT_EN)
#     prompt_manager.add_prompt(domain="in2o3_tco", language="en", template=IN2O3_TCO_PROMPT_EN)
#     return prompt_manager


def load_prompts(prompt_manager: PromptManager):
    prompt_manager.add_prompt(domain="membrane", language="en", template=MEMBRANE_PROMPT_EN)
    prompt_manager.add_prompt(domain="in2o3_tco", language="en", template=IN2O3_TCO_PROMPT_EN) # Added new prompt
    # prompt_manager.add_prompt(domain="another_domain", language="en", template=ANOTHER_PROMPT_EN)
    return prompt_manager