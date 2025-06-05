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
IN2O3_TCO_PROMPT_EN = """You are given the cleaned full text of a paper on In2O3-based Transparent Conducting Oxides (TCOs), potentially with various dopants.
Please extract structured information for **each distinct TCO composition or sample described with unique properties/fabrication conditions in the text**. If multiple TCOs are detailed (e.g., In2O3:Sn (ITO) with different Sn doping levels, or In2O3:W vs In2O3:Mo), generate a separate entry for each.

For each distinct TCO material/sample, extract and return the following categories of information (omit any missing category or sub-field if the information is not present):

1.  **Design**:
    * HostMaterial: The main oxide matrix (e.g., "In2O3").
    * PrimaryDopant: This MUST BE A SINGLE JSON OBJECT describing the main dopant element chosen to characterize this specific sample. Include its "element" (e.g., "Sn") and "concentration_text" (e.g., "5 at.%"). If the source implies multiple elements are "primary" (e.g., "Ti and W co-doped In2O3"), select ONE element as the PrimaryDopant (e.g., the first mentioned, or the one with higher concentration if specified). The other dopant(s) should then be listed under "CoDopants". If no specific dopant is highlighted as primary for an undoped or complex sample, this field can be an empty object {{}} or omitted.
    * DopingConcentration: (This information should primarily be captured within the PrimaryDopant object's "concentration_text" field. This separate field can be omitted if redundant or used for an overall doping level if distinct from PrimaryDopant).
    * CoDopants: This MUST BE A LIST OF JSON OBJECTS if co-dopants or other significant elemental additions are present and are not chosen as the PrimaryDopant. Each object in the list should describe one co-dopant, including its "element" (e.g., "H", "Ce") and "concentration_text" (e.g., "from H2 annealing", "2 wt.% CeO2 target precursor"). If no co-dopants are mentioned or applicable, provide an empty list [] or omit the "CoDopants" field entirely. DO NOT put a simple string here; it must be a list of objects or omitted/empty list.
    * TargetStoichiometry: The intended chemical formula or target composition (e.g., "In(2-x)SnxO3", "In2O3:Sn (10 at.%)", "Nominal composition In2O3 with 1 wt% WO3").
 
2.  **Fabrication**:
    * DepositionMethod (e.g., "DC Magnetron Sputtering", "Pulsed Laser Deposition (PLD)", "Sol-gel spin coating", "Atomic Layer Deposition (ALD)").
    * SubstrateMaterial (e.g., "Corning Eagle Glass", "PET flexible substrate", "Si wafer with SiO2 layer").
    * PrecursorMaterialsText (For methods like sol-gel, CVD, ALD; list as text, e.g., "Indium nitrate, Tin (IV) chloride pentahydrate in ethanol").
    * TargetMaterialText (For methods like sputtering, PLD; describe target, e.g., "ITO ceramic target (90 wt% In2O3, 10 wt% SnO2, 99.99% purity)").
    * DepositionParameters: This SHOULD BE A JSON OBJECT. Populate with specific key-value parameters if available (e.g., BasePressure, WorkingPressure, DepositionTemperature, GasAtmosphere, GasFlowRates, DepositionPower, DepositionTime, DepositionRate).
    * DepositionParametersTextSummary: If you find only a general descriptive statement about deposition conditions (e.g., "films were deposited at various oxygen partial pressures", "low-temperature deposition process was employed") and CANNOT extract specific key-value parameters for the `DepositionParameters` object, then put that descriptive statement here as a string. In such a case, `DepositionParameters` object can be an empty object {{}} or omitted. DO NOT put general statements into the `DepositionParameters` object itself.
    * AnnealingConditions: This SHOULD BE A JSON OBJECT. Populate with specific key-value parameters if available (e.g., Temperature, Atmosphere, Duration, RampRate).
    * AnnealingConditionsTextSummary: If you find only a general descriptive statement about annealing (e.g., "samples were annealed under reducing conditions to improve conductivity") and CANNOT extract specific key-value parameters for the `AnnealingConditions` object, then put that descriptive statement here as a string. In such a case, `AnnealingConditions` object can be an empty object {{}} or omitted.
    * FilmThicknessText (e.g., "150 nm", "approximately 200 nm thick", including measurement method if specified, e.g., "180 nm (measured by ellipsometry)").

3.  **Performance**: (Extract specific values with units and conditions where possible)
    * Resistivity (e.g., "3.5 x 10^-4 Ω·cm", "200 μΩ·cm at 300K")
    * SheetResistance (e.g., "10 Ω/sq for 100nm film", "100 Ω/□")
    * CarrierConcentration (e.g., "6.2 x 10^20 cm⁻³", specify n-type or p-type if mentioned)
    * HallMobility (e.g., "45 cm²/Vs at room temperature")
    * OpticalTransmittance (e.g., "Average > 90% in visible range (400-700nm)", "92% at 550 nm wavelength")
    * OpticalBandGapText (Eg) (e.g., "3.75 eV", "Eg = 4.1 eV (determined by Tauc plot)")
    * WorkFunctionText (Φ) (e.g., "4.8 eV", "Work function of 4.65 eV measured by UPS")
    * FigureOfMerit (FoM) (e.g., "Haacke's FoM: 15 x 10^-3 Ω⁻¹", specify type and value)
    * Haze (e.g., "< 1% for optimized films")
    * CrystalStructure (e.g., "Cubic bixbyite In2O3 phase identified by XRD", "Amorphous structure", "Preferred (222) orientation")
    * GrainSize (e.g., "Average grain size of 30-50 nm from SEM")
    * SurfaceRoughnessRMS (e.g., "RMS roughness of 0.5 nm measured by AFM over 1x1 μm² area")
    * Other relevant optoelectronic, structural, or morphological properties.

4.  **Application**:
    * PotentialApplicationArea (e.g., "Transparent electrode for perovskite solar cells", "Active channel layer in thin film transistors (TFTs)", "Gas sensing material for NO2 detection")
    * DevicePerformance (If the TCO was used in a specific device and its performance reported, e.g., "Solar cell achieved Power Conversion Efficiency (PCE) of 18.5%", "TFT on-/off ratio of 10^6")

Strictly implement the following requirements:
1.  Return a strict JSON object in the following format for the entire paper, containing a list in the "output" field where each item corresponds to one TCO material/sample:
    {{
      "output": [
        {{
          "MaterialName": "Sn-doped In2O3 (10 wt% Sn)",
          "Details": {{
            "Design": {{
              "HostMaterial": "In2O3",
              "PrimaryDopant": {{ "element": "Sn", "concentration_text": "10 wt.%" }},
              "CoDopants": [], // Example: No co-dopants specified for this entry
              "TargetStoichiometry": "Target: ITO (90 wt% In2O3, 10 wt% SnO2)"
            }},
            "Fabrication": {{
              "DepositionMethod": "RF Magnetron Sputtering",
              "SubstrateMaterial": "Corning 7059 glass",
              "TargetMaterialText": "ITO ceramic target (90 wt% In2O3, 10 wt% SnO2)",
              "DepositionParameters": {{ 
                "DepositionPower": "120W",
                "GasAtmosphere": "Argon",
                "WorkingPressure": "0.5 Pa",
                "DepositionTemperature": "200°C"
              }},
              // "DepositionParametersTextSummary": null, // Omit if DepositionParameters object is well-populated
              "AnnealingConditions": {{ 
                "Temperature": "400°C",
                "Atmosphere": "Forming gas (5% H2 / 95% N2)",
                "Duration": "30 min"
              }},
              // "AnnealingConditionsTextSummary": null, // Omit if AnnealingConditions object is well-populated
              "FilmThicknessText": "200 nm"
            }},
            "Performance": {{
              "Resistivity": "2.1 x 10^-4 Ω·cm",
              "SheetResistance": "10.5 Ω/sq",
              "CarrierConcentration": "8.5 x 10^20 cm⁻³ (n-type)",
              "HallMobility": "35 cm²/Vs",
              "OpticalTransmittance": "Average >88% in 400-700 nm range, Peak 91% at 550 nm",
              "OpticalBandGapText": "3.9 eV (from Tauc plot)",
              "FigureOfMerit": "Haacke FoM: 12.5x10⁻³ Ω⁻¹"
            }},
            "Application": {{
              "PotentialApplicationArea": "Transparent conductive electrode for flexible OLEDs"
            }}
          }}
        }},
        {{
          "MaterialName": "In2O3:H (Hydrogenated In2O3)",
          "Details": {{
            "Design": {{
              "HostMaterial": "In2O3",
              "PrimaryDopant": {{ "element": "H", "concentration_text": "Interstitial doping during H2 plasma treatment" }},
              "CoDopants": [] // Explicitly empty if none
            }},
            "Fabrication": {{
              "DepositionMethod": "Sputtering followed by H2 plasma treatment",
              "SubstrateMaterial": "Quartz",
              "DepositionParameters": {{}}, // Example: No specific deposition parameters listed, only summary
              "DepositionParametersTextSummary": "Films were initially deposited by sputtering, then subjected to hydrogen plasma.",
              "AnnealingConditions": {{ // Example: Specific annealing parameters
                "Temperature": "300°C",
                "Atmosphere": "Vacuum (10^-5 Torr)",
                "Duration": "1 hour"
              }}
              // "AnnealingConditionsTextSummary": null, // Omitted as AnnealingConditions object is populated
            }},
            "Performance": {{
              "Resistivity": "5.0 x 10^-4 Ω·cm",
              "CarrierConcentration": "7.0 x 10^20 cm⁻³",
              "OpticalTransmittance": "Average 85% in visible region"
            }},
            "Application": {{
              "PotentialApplicationArea": "High mobility TFTs"
            }}
          }}
        }},
        {{
          "MaterialName": "Ti and W co-doped In2O3 (IWTO)",
          "Details": {{
            "Design": {{
              "HostMaterial": "In2O3",
              "PrimaryDopant": {{ "element": "W", "concentration_text": "1.1 wt% WO3 target component" }}, // LLM to pick one as primary
              "CoDopants": [ // Other dopant(s) listed here
                {{ "element": "Ti", "concentration_text": "0.05 wt% TiO2 target component" }}
              ],
              "TargetStoichiometry": "Target: In2O3 with 1.1 wt% WO3 and 0.05 wt% TiO2"
            }},
            "Fabrication": {{
              "DepositionMethod": "Sputtering",
              "DepositionParametersTextSummary": "Films prepared by co-sputtering from composite targets."
            }},
            "Performance": {{
              "Resistivity": "4.5 x 10^-4 Ω·cm"
            }},
            "Application": {{}} // Example: No specific application details found
          }}
        }}
        // ... more TCO material/sample entries if applicable
      ]
    }}
Only return the JSON. Do not include any extra explanations or markdown backticks.
2.  The main key of the returned JSON object MUST be "output", and its value MUST be a list.
3.  Inside each entry in the "output" list, use "MaterialName" as the descriptive name for the TCO sample (e.g., "In2O3:W (3 at.%)", "ITO commercial target"). The "Details" object should contain the 4 categories.
4.  For `PrimaryDopant`, provide a single JSON object. For `CoDopants`, provide a LIST of JSON objects; if no co-dopants, use an empty list `[]` or omit the "CoDopants" field.
5.  If specific key-value parameters are found for `DepositionParameters` or `AnnealingConditions`, populate their respective JSON objects. If only general descriptive text is found for these aspects, place it in `DepositionParametersTextSummary` or `AnnealingConditionsTextSummary` respectively, and ensure the corresponding object (`DepositionParameters` or `AnnealingConditions`) is empty `{{}}` or omitted.
6.  If a category or sub-field is not mentioned for a specific sample, omit it from its "Details" or sub-object. Be precise with units and conditions if reported.
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