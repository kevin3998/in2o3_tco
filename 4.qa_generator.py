# qa_pair_generator.py
import json
import time
import os
import regex as re
import argparse
import logging
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()]
)


# --- 1. Centralized Configuration ---
class Config:
    # --- Modify paths and settings here for direct runs, or use command-line arguments ---
    DEFAULT_INPUT_PATH = "data/extracted_json/pydantic/structured2.json"
    DEFAULT_OUTPUT_PATH = "data/generated_qa_pairs/qa_pairs_for_finetuning.jsonl"
    DEFAULT_CHECKPOINT_PATH = "data/generated_qa_pairs/checkpoint_qa_gen.json"
    DEFAULT_DOMAIN = "in2o3_tco"  # Options: "in2o3_tco", "membrane"
    DEFAULT_MODEL = "DeepSeek-R1-671B"


# --- 2. LLM Client Initialization ---
def get_llm_client() -> OpenAI:
    """Initializes and returns the OpenAI client, loading credentials from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables (.env file).")
        raise ValueError("API key is missing.")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        logging.info(f"OpenAI client initialized for base_url: {base_url if base_url else 'default'}")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        raise


# --- 3. Domain-Specific Prompt Templates ---
QA_GENERATION_PROMPT_FOR_IN2O3_TCO = """
You are a highly specialized AI assistant tasked with creating sophisticated question-answer (QA) pairs from structured data about **Indium Oxide (In2O3) based materials, which are doped to become Transparent Conducting Oxides (TCOs)**. These QA pairs will be used to fine-tune another language model, aiming to develop its reasoning and predictive abilities specifically for **doped In2O3 systems**.
**Your input ("Contextual Information") consists of:**
1.  `MaterialName`: The specific name of the In2O3-based TCO material/sample (e.g., "Sn-doped In2O3", "ITO", "In2O3:W").
2.  `Details`: A structured JSON object containing detailed information about this specific In2O3-based material, categorized under "Design", "Fabrication", "Performance", and "Application".
**Your Goal:**
Generate exactly {num_qa_pairs} diverse and high-quality QA pairs. Each QA pair must be:
* **Strictly Grounded:** The answer MUST be directly derivable ONLY from the provided "Details" JSON for the given "MaterialName" (which is an In2O3-based TCO). Do NOT use any external knowledge or make assumptions.
* **Reasoning-Oriented:** Prioritize questions that require connecting information across different parts of the "Details" (e.g., linking a "Fabrication" parameter of a doped In2O3 film to a "Performance" outcome).
* **Predictive (Simulated):** Formulate questions that ask to "predict" a property of the doped In2O3 given other properties/conditions, where the answer is explicitly available in the "Details".
* **Insightful:** Go beyond simple fact retrieval where possible, aiming for questions a researcher or engineer studying In2O3 TCOs might ask to understand the material's behavior and potential.
* **Clear and Natural:** Phrased in clear, concise, and natural English.
**Contextual Information to be Used:**
Material Name: {material_name}
Details (structured JSON for an In2O3-based TCO):
{details_json_str}
**Types of Question-Answer Pairs to Generate (aim for a diverse mix, focusing on types 2-6):**
**1. Advanced Factual Retrieval (Multi-Point for Doped In2O3):**
   * Objective: Questions that require retrieving and combining multiple specific facts about the {material_name}.
   * Example Question: "For the In2O3-based TCO named {material_name}, what are the reported values for its Resistivity, primary Dopant Concentration, and Optical Transmittance in the visible range, based on the provided details?"
   * Example Answer: "For {material_name}, the reported Resistivity is [value from Details.Performance.ElectricalProperties.Resistivity], the primary Dopant Concentration is [value from Details.Design.PrimaryDopant.concentration_text], and Optical Transmittance is [value from Details.Performance.OpticalProperties.AverageTransmittance]."
**2. Relationship Identification within Doped In2O3 Systems (Simple Reasoning):**
   * Objective: Questions that ask about observed relationships or associations between parameters for {material_name} as presented in the "Details".
   * Example Question: "According to the details for {material_name}, what DepositionMethod for the In2O3 film is associated with achieving a FilmThickness of [value from Details.Fabrication.FilmThicknessText]?"
   * Example Answer: "The details for {material_name} associate the DepositionMethod '[value from Details.Fabrication.DepositionMethod]' with a FilmThickness of [value from Details.Fabrication.FilmThicknessText]."
   * Example Question: "Is there a reported link between the AnnealingAtmosphere of [value from Details.Fabrication.AnnealingConditions.Atmosphere] and the resulting CarrierConcentration of [value from Details.Performance.ElectricalProperties.CarrierConcentration] for the In2O3-based TCO {material_name} in this data?"
   * Example Answer: "Yes, for {material_name}, an AnnealingAtmosphere of [value] is reported alongside a CarrierConcentration of [value]." OR "The provided details list an AnnealingAtmosphere of [value] and a CarrierConcentration of [value] for {material_name}, but do not explicitly state a direct causal link between them for this specific entry."
**3. "Predictive" QA for Doped In2O3 Properties (Based on Stated Facts):**
   * Objective: Frame questions as if predicting an outcome for {material_name}, where the outcome is a performance metric explicitly stated in the "Details" for a given set of design/fabrication conditions also stated in the "Details".
   * Example Question: "Given the In2O3-based TCO, {material_name}, designed with a PrimaryDopant element of [element from Details.Design.PrimaryDopant.element] and fabricated with a SubstrateMaterial of [value from Details.Fabrication.SubstrateMaterial], what is its expected SheetResistance based on these details?"
   * Example Answer: "Based on these details, for {material_name} with a PrimaryDopant of [element] and SubstrateMaterial of [value], the expected SheetResistance is [value from Details.Performance.ElectricalProperties.SheetResistance]."
**4. Condition/Formulation Recommendation for Doped In2O3 (Inverse Problem):**
   * Objective: Ask for design or fabrication parameters of the In2O3-based system that are associated with achieving specific (reported) performance metrics.
   * Example Question: "To obtain an In2O3-based film like {material_name} with an OpticalBandGapText reported as [value from Details.Performance.OpticalProperties.OpticalBandGapText], what AnnealingTemperature and AnnealingDuration are indicated in its fabrication details?"
   * Example Answer: "The fabrication details for achieving an OpticalBandGapText of [value] for {material_name} (an In2O3-based TCO) indicate an AnnealingTemperature of '[value from Details.Fabrication.AnnealingConditions.Temperature]' and Duration of '[value from Details.Fabrication.AnnealingConditions.Duration]'."
**5. Rationale/Association Explanation for Doped In2O3 (Grounded in Data):**
   * Objective: Ask "why" a certain design choice (e.g., specific dopant for In2O3) or fabrication step *might be associated* with a particular performance characteristic, if the "Details" provide elements for a plausible link. The answer should state the association found in the data.
   * Example Question: "What characteristic of the HostMaterial (which should be In2O3 or a variation) in {material_name} might be associated with its reported high OpticalTransmittance of [value from Details.Performance.OpticalProperties.AverageTransmittance], based on the provided data context?"
   * Example Answer: "In the context of {material_name}, its In2O3-based HostMaterial is reported alongside a high OpticalTransmittance of [value]. (The provided data might not offer further explanation for this inherent property of In2O3, but the association is noted)."
**6. Scenario Adaptability Assessment for Doped In2O3:**
   * Objective: Pose questions about the suitability of {material_name} (based on its reported properties) for an application mentioned or related to "Details.Application".
   * Example Question: "Considering the reported HallMobility of [value] and WorkFunctionText of [value] for the In2O3-based TCO {material_name}, how well would it meet the specific requirements for use as a '[value from Details.Application.PotentialApplicationArea]' which typically needs [e.g., high mobility and a specific work function alignment]?"
   * Example Answer: "With a HallMobility of [value] and WorkFunctionText of [value], {material_name} (an In2O3-based TCO) shows [e.g., strong potential / certain challenges] for use as a '[application]' requiring [e.g., high mobility and specific work function alignment]. Its [specific property] is particularly [advantageous/a concern]."

**Output Format:**
Return a strict JSON list of objects. Each object MUST have a "question" key and an "answer" key.

**Example of a single QA pair in the list:**
{{

  "question": "For the In2O3-based material {material_name}, which was fabricated using [DepositionMethod from Details] on a [SubstrateMaterial from Details], what is the resulting Resistivity stated in its performance details?",
  "answer": "When the In2O3-based material {material_name} is fabricated using [DepositionMethod] on a [SubstrateMaterial], the stated Resistivity is [value from Details.Performance.ElectricalProperties.Resistivity]."
}}
Now, generate exactly {num_qa_pairs} diverse question-answer pairs based on the "Contextual Information" about the In2O3-based TCO {material_name} and these detailed guidelines. Prioritize generating questions of types 2 through 6. Ensure all references to the material correctly imply or state its In2O3-based nature.
"""



PROMPT_REGISTRY = {
    "in2o3_tco": QA_GENERATION_PROMPT_FOR_IN2O3_TCO,
}


# --- 4. Core QA Generation Logic ---
def generate_qa_for_material_entry(client: OpenAI, model_name: str, material_name: str, details: dict, domain: str,
                                   num_qa_pairs: int = 7) -> list:
    details_json_str = json.dumps(details, indent=2, ensure_ascii=False)

    prompt_template = PROMPT_REGISTRY.get(domain)
    if not prompt_template:
        raise ValueError(f"No prompt template found for domain: '{domain}'")

    prompt = prompt_template.format(
        material_name=material_name,
        details_json_str=details_json_str,
        num_qa_pairs=num_qa_pairs
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an AI assistant that generates question-answer pairs from structured data according to specific instructions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Slightly increased for more creative yet grounded questions
                stream=False
            )
            content = response.choices[0].message.content

            match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                logging.warning(
                    f"Could not find a JSON list in LLM response for {material_name}. Trying to find a single object.")
                match_obj = re.search(r'\{\s*".*?"\s*:\s*".*?"\s*\}', content, re.DOTALL)
                if match_obj:
                    json_str = f"[{match_obj.group(0)}]"
                else:
                    logging.error(f"No valid JSON found at all for {material_name}. Content: {content[:200]}")
                    json_str = "[]"  # Default to empty list string on failure

            qa_pairs = json.loads(json_str)
            if isinstance(qa_pairs, list) and all(
                    isinstance(item, dict) and "question" in item and "answer" in item for item in qa_pairs):
                return qa_pairs
            else:
                logging.warning(
                    f"LLM response for {material_name} not in expected QA list format after parsing. Parsed object type: {type(qa_pairs)}")
                return []
        except json.JSONDecodeError as e:
            logging.error(
                f"JSONDecodeError for {material_name} (Attempt {attempt + 1}/{max_retries}): {e}. Content: {content[:500] if 'content' in locals() else 'N/A'}")
        except Exception as e:
            logging.error(f"Error generating QA for {material_name} (Attempt {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** (attempt + 1))
    return []


def process_structured_data_for_qa(client: OpenAI, model_name: str, domain: str, structured_data_list: List[Dict],
                                   checkpoint_path: str) -> List[Dict]:
    all_qa_pairs = []
    processed_entry_identifiers = set()
    total_elapsed_qa_gen = 0.0
    start_from_index = 0

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            all_qa_pairs = checkpoint_data.get("all_qa_pairs", [])
            processed_entry_identifiers = set(checkpoint_data.get("processed_entry_identifiers", []))
            total_elapsed_qa_gen = checkpoint_data.get("total_elapsed_qa_gen", 0.0)
            start_from_index = checkpoint_data.get("last_processed_entry_index", -1) + 1
            logging.info(
                f"▶ QA Gen Checkpoint loaded. Resuming from entry index {start_from_index}. Already have {len(all_qa_pairs)} QA pairs.")
        except Exception as e:
            logging.warning(f"⚠ QA Gen Checkpoint loading failed: {e}. Starting from scratch.")
            # Re-initialize to default empty state
            all_qa_pairs, processed_entry_identifiers, total_elapsed_qa_gen, start_from_index = [], set(), 0.0, 0

    total_material_entries = len(structured_data_list)
    logging.info(f"Total material entries to process for QA generation: {total_material_entries}")

    if total_material_entries == 0:
        return all_qa_pairs

    last_processed_idx = -1
    try:
        for current_idx, extracted_entry in enumerate(structured_data_list):
            if current_idx < start_from_index:
                last_processed_idx = current_idx
                continue

            # ... (Logic to parse `extracted_entry` as before, to get doi, material_name, details) ...
            meta_source_paper = extracted_entry.get("meta_source_paper", {})
            extracted_material_data = extracted_entry.get("extracted_material_data", {})
            doi = meta_source_paper.get("doi", f"unknown_doi_idx_{current_idx}")
            material_name = extracted_material_data.get("MaterialName")
            details = extracted_material_data.get("Details")
            if not material_name or not isinstance(details, dict):
                logging.warning(
                    f"⚠️ Skipping entry at index {current_idx} due to missing 'MaterialName' or invalid 'Details' (DOI: {doi}).")
                continue

            entry_identifier = f"{doi}::{material_name}"
            if entry_identifier in processed_entry_identifiers:
                logging.info(f"ℹ️ Skipping already processed entry: {entry_identifier}")
                last_processed_idx = current_idx
                continue

            session_start_time = time.time()
            logging.info(f"\nProcessing entry {current_idx + 1}/{total_material_entries}: {material_name} (DOI: {doi})")

            generated_pairs = generate_qa_for_material_entry(client, model_name, material_name, details, domain)

            if generated_pairs:
                for pair in generated_pairs:
                    pair["source_doi"] = doi
                    pair["source_material_name"] = material_name
                all_qa_pairs.extend(generated_pairs)
                processed_entry_identifiers.add(entry_identifier)
                logging.info(f"✔ Generated {len(generated_pairs)} QA pairs for {material_name}.")
            else:
                logging.warning(f"❌ No QA pairs generated or generation failed for {material_name}.")

            last_processed_idx = current_idx
            # ... (ETA calculation and print logic as before) ...
            session_elapsed = time.time() - session_start_time
            total_elapsed_qa_gen += session_elapsed
            entries_processed_this_session = (current_idx - start_from_index + 1)
            avg_time_per_entry = total_elapsed_qa_gen / entries_processed_this_session if entries_processed_this_session > 0 else 0
            eta_seconds = avg_time_per_entry * (total_material_entries - (current_idx + 1))
            print(
                f"Progress: [{current_idx + 1}/{total_material_entries}] | Session: {session_elapsed:.2f}s | Total QA Time: {total_elapsed_qa_gen:.2f}s | ETA: {eta_seconds / 3600:.2f}h",
                end='\r')

            if (current_idx + 1) % 5 == 0:
                save_qa_checkpoint(checkpoint_path, all_qa_pairs, list(processed_entry_identifiers),
                                   total_elapsed_qa_gen, last_processed_idx)

    except KeyboardInterrupt:
        logging.warning("\n⚠ User interrupted. Saving QA generation checkpoint...")
    except Exception as e:
        import traceback
        logging.error(f"\n⚠ An unexpected error occurred during QA generation: {e}")
        traceback.print_exc()
    finally:
        save_qa_checkpoint(checkpoint_path, all_qa_pairs, list(processed_entry_identifiers), total_elapsed_qa_gen,
                           last_processed_idx if last_processed_idx != -1 else len(structured_data_list) - 1)
        print(f"\nQA Generation process ended. Total QA pairs in list: {len(all_qa_pairs)}")

    return all_qa_pairs


def save_qa_checkpoint(path: str, qa_pairs: list, processed_ids_list: list, elapsed_time: float, last_idx: int):
    # This function remains largely the same
    checkpoint_content = {
        "all_qa_pairs": qa_pairs,
        "processed_entry_identifiers": processed_ids_list,
        "total_elapsed_qa_gen": elapsed_time,
        "last_processed_entry_index": last_idx
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_content, f, indent=2, ensure_ascii=False)
        logging.info(f"\n✔ QA Gen Checkpoint saved to {path}")
    except Exception as e:
        logging.warning(f"⚠ Failed to save QA Gen checkpoint: {e}")


# --- 5. Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate Question-Answer pairs from structured JSON data for LLM fine-tuning.")
    parser.add_argument("--input_file", type=str, default=Config.DEFAULT_INPUT_PATH,
                        help=f"Path to the input structured JSON file. Default: {Config.DEFAULT_INPUT_PATH}")
    parser.add_argument("--output_file", type=str, default=Config.DEFAULT_OUTPUT_PATH,
                        help=f"Path to save the output QA pairs JSONL file. Default: {Config.DEFAULT_OUTPUT_PATH}")
    parser.add_argument("--checkpoint_file", type=str, default=Config.DEFAULT_CHECKPOINT_PATH,
                        help=f"Path for the checkpoint file. Default: {Config.DEFAULT_CHECKPOINT_PATH}")
    parser.add_argument("--domain", type=str, default=Config.DEFAULT_DOMAIN, choices=PROMPT_REGISTRY.keys(),
                        help=f"The domain for QA generation. Default: {Config.DEFAULT_DOMAIN}")
    parser.add_argument("--model_name", type=str, default=Config.DEFAULT_MODEL,
                        help=f"Name of the LLM to use for QA generation. Default: {Config.DEFAULT_MODEL}")

    args = parser.parse_args()
    logging.info(f"Starting QA pair generation with configuration: {args}")

    try:
        client = get_llm_client()
        with open(args.input_file, "r", encoding="utf-8") as f:
            structured_data_list = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from input file: {args.input_file}")
        return
    except Exception as e:
        logging.error(f"An error occurred during setup: {e}")
        return

    if not isinstance(structured_data_list, list):
        logging.error(f"Error: Expected input file {args.input_file} to contain a JSON list.")
        return

    final_qa_pairs = process_structured_data_for_qa(client, args.model_name, args.domain, structured_data_list,
                                                    args.checkpoint_file)

    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for qa_pair in final_qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
        logging.info(
            f"\n✅ QA pair generation complete. {len(final_qa_pairs)} pairs saved to {args.output_file} in JSONL format.")
    except Exception as e:
        logging.error(f"Error writing output file {args.output_file}: {e}")

    if os.path.exists(args.checkpoint_file):
        logging.info(f"ⓘ QA Gen Checkpoint {args.checkpoint_file} retained. Review results before manual deletion.")


if __name__ == "__main__":
    main()