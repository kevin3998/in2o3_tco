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
IN2O3_TCO_PROMPT_EN = """You are given the cleaned full text of a paper on Indium Oxide (In2O3) based materials, doped to become Transparent Conducting Oxides (TCOs).
Please extract structured information for **each distinct TCO composition or sample described with unique properties/fabrication conditions in the text**.

For each distinct In2O3-based TCO material/sample, extract and return the following categories of information (omit any missing category or sub-field if the information is not present):

1.  **Design**:
    * HostMaterial: (e.g., "In2O3").
    * PrimaryDopant: This MUST BE A SINGLE JSON OBJECT describing the main dopant. Include its "Element" (e.g., "Sn") and "Concentration_text" (e.g., "5 at.%"). If multiple elements seem equally primary (e.g., "Ti and W co-doped In2O3"), select ONE as PrimaryDopant and list other(s) under "CoDopants". If no specific primary dopant, this can be an empty object {{}} or omitted.
    * CoDopants: This MUST BE A LIST OF JSON OBJECTS if co-dopants are present. Each object should have "Element" and "Concentration_text". If none, provide an empty list [] or omit. DO NOT use a simple string here.
    * TargetStoichiometry: (e.g., "In(2-x)SnxO3", "Target: In2O3 with 1 wt% WO3").
    * MaterialDescriptionSource: (e.g., "Commercial ITO target (99.99% purity)").

2.  **Fabrication**:
    * DepositionMethod: (e.g., "DC Magnetron Sputtering").
    * SubstrateMaterial: (e.g., "Glass (Corning Eagle)").
    * PrecursorMaterialsText: (For sol-gel, CVD, ALD etc.).
    * TargetMaterialText: (For sputtering, PLD etc., e.g., "ITO ceramic target (90 wt% In2O3, 10 wt% SnO2)").
    * DepositionParameters: This SHOULD BE A JSON OBJECT with specific keys like "BasePressure", "WorkingPressure", "DepositionTemperature", "GasAtmosphere", "GasFlowRates", "DepositionPower".
    * DepositionParametersTextSummary: If ONLY a general descriptive statement about deposition (e.g., "low-temperature deposition") is found AND specific parameters for the `DepositionParameters` object are NOT available, put the statement here. If specific parameters are extracted into the `DepositionParameters` object, omit this summary field or leave it null.
    * AnnealingConditions: This SHOULD BE A JSON OBJECT with specific keys like "Temperature", "Atmosphere", "Duration".
    * AnnealingConditionsTextSummary: If ONLY a general descriptive statement about annealing is found AND specific parameters for the `AnnealingConditions` object are NOT available, put the statement here. If specific parameters are extracted into `AnnealingConditions` object, omit this summary field or leave it null.
    * FilmThicknessText: (e.g., "150 nm", including measurement method if specified).

3.  **Performance**:
    * Resistivity: (e.g., "3.5 x 10^-4 Ω·cm").
    * SheetResistance: (e.g., "10 Ω/sq").
    * CarrierConcentration: (e.g., "6.2 x 10^20 cm⁻³").
    * CarrierType: (e.g., "n-type", "p-type", if mentioned alongside carrier concentration or Hall mobility).
    * HallMobility: (e.g., "45 cm²/Vs").
    * AverageTransmittance: (For average values over a range, e.g., ">90% in visible range (400-700nm)").
    * TransmittanceAt550nm: (For specific wavelength values, e.g., "92% at 550 nm").
    * OpticalTransmittanceDescription: If only a general textual description of transmittance is available (e.g., "High in visible and near-infrared regions", "Decreases with Sn doping"), put it here. Prioritize extracting specific values into AverageTransmittance or TransmittanceAt550nm if possible.
    * OpticalBandGapText: (Eg) (e.g., "3.75 eV (Tauc plot)").
    * WorkFunctionText: (Φ) (e.g., "4.8 eV (UPS)").
    * FigureOfMerit: (Type and value, e.g., "Haacke's FoM: 15 x 10^-3 Ω⁻¹").
    * Haze: (e.g., "< 1%").
    * CrystalStructure: (e.g., "Cubic bixbyite In2O3", "Amorphous", "Preferred (222) orientation").
    * GrainSize: (e.g., "30-50 nm").
    * SurfaceRoughnessRMS: (e.g., "0.5 nm (RMS)").

4.  **Application**:
    * PotentialApplicationArea: (e.g., "Transparent electrode for perovskite solar cells").
    * DevicePerformance: (If TCO used in a device and its performance reported, e.g., "Solar cell efficiency (PCE): 18.5%").

Strictly implement the following requirements:
1.  Return a strict JSON object. The root of this object MUST contain an "output" key, and its value MUST be a list. Each item in the "output" list corresponds to one TCO material/sample.
    Example Structure:
    {{
      "output": [
        {{
          "MaterialName": "Sn-doped In2O3 (10 wt% SnO2 in target)", // Example: Descriptive material name
          "Details": {{
            "Design": {{
              "HostMaterial": "In2O3",
              "PrimaryDopant": {{ "Element": "Sn", "Concentration_text": "Target: 10 wt.% SnO2" }},
              "CoDopants": [], // Empty list if no co-dopants
              "TargetStoichiometry": "In2O3 with 10 wt.% SnO2 in target"
            }},
            "Fabrication": {{
              "DepositionMethod": "DC Magnetron Sputtering",
              "SubstrateMaterial": "Glass",
              "TargetMaterialText": "Ceramic target of In2O3 with 10 wt.% SnO2",
              "DepositionParameters": {{ // Populated object
                "WorkingPressure": "0.4 Pa",
                "DepositionTemperature": "Room Temperature",
                "GasAtmosphere": "Ar (20 sccm), O2 (0.1 sccm)",
                "DepositionPower": "100W"
              }},
              // DepositionParametersTextSummary would be omitted or null here
              "AnnealingConditions": {{ // Populated object
                "Temperature": "300°C",
                "Atmosphere": "N2",
                "Duration": "1 hour"
              }},
              "FilmThicknessText": "180 nm"
            }},
            "Performance": {{
              "Resistivity": "2.5 x 10^-4 Ω·cm",
              "CarrierConcentration": "7.5 x 10^20 cm⁻³",
              "CarrierType": "n-type",
              "HallMobility": "33 cm²/Vs",
              "AverageTransmittance": ">85% (400-800 nm)",
              "OpticalBandGapText": "3.85 eV",
              "CrystalStructure": "Polycrystalline cubic bixbyite"
            }},
            "Application": {{
              "PotentialApplicationArea": "Transparent electrodes for solar cells"
            }}
          }}
        }},
        {{
          "MaterialName": "In2O3 film (Low Temp RPD)",
          "Details": {{
            "Design": {{
              "HostMaterial": "In2O3",
              "PrimaryDopant": {{}} // No specific dopant mentioned as primary
            }},
            "Fabrication": {{
              "DepositionMethod": "Reactive Plasma Deposition (RPD)",
              "SubstrateMaterial": "Glass",
              "DepositionParameters": {{}}, // No specific parameters, summary below
              "DepositionParametersTextSummary": "Films were prepared by RPD at substrate temperatures below 60°C using Ar, O2, and H2O vapor.",
              "AnnealingConditions": {{ // Only some specific annealing info
                "Temperature": "250°C",
                "Atmosphere": "Vacuum"
              }},
              // AnnealingConditionsTextSummary could be "Vacuum annealing was performed." if Temperature/Atmosphere were not extractable
              "FilmThicknessText": "220 nm"
            }},
            "Performance": {{
              "OpticalTransmittanceDescription": "High transmittance in visible and NIR." // General description
            }},
            "Application": {{}} // Empty if no specific application details
          }}
        }}
      ]
    }}
Only return the JSON. Do not include any extra explanations or markdown backticks.
2.  Inside each entry in the "output" list, use "MaterialName" as the descriptive name. The "Details" object holds the 4 categories.
3.  If a category or sub-field is not mentioned, omit it or use null if appropriate for an optional field that has no value.
4.  For `DepositionParameters` and `AnnealingConditions`: prioritize populating their structured JSON objects with specific key-value data. If such specific data is NOT available and only a general textual description exists, place that text in `DepositionParametersTextSummary` or `AnnealingConditionsTextSummary` respectively, and the corresponding structured object (`DepositionParameters` or `AnnealingConditions`) should then be an empty object `{{}}` or omitted.
5.  For `PrimaryDopant`, provide a single JSON object. For `CoDopants`, provide a LIST of JSON objects; if none, use an empty list `[]`.
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