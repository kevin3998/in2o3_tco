import json
import time
import os
import regex as re  # Keep regex for parsing LLM output if needed for QA generation itself
from openai import OpenAI
from typing import List, Dict, Any

# --- Configuration ---
# Path to the JSON file output by your 'extractor' script
INPUT_STRUCTURED_DATA_PATH = "data/extracted_json/json_load_directly/structured_extraction.json"  # Or your actual output path from extractor
# Path to save the generated QA pairs
OUTPUT_QA_PAIRS_PATH = "data/generated_qa_pairs/qa_pairs_from_structured.jsonl"  # Outputting as JSONL is common for fine-tuning data
CHECKPOINT_PATH_QA_GEN = "data/generated_qa_pairs/checkpoint_qa.json"

# Initialize client (ensure API key and base_url are correct and active)
try:
    # Replace with your actual client initialization if different
    client = OpenAI(api_key="sk-MzAxLTExMzc5NzE5ODU0LTE3NDc4Nzc1MjM2MTI=", base_url="https://api.scnet.cn/api/llm/v1")
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}. Please check API key and base URL.")
    exit(1)

model_for_qa_generation = "DeepSeek-R1-671B"  # Or your preferred model

# --- Prompt for QA Generation (QA_GENERATION_PROMPT_TEMPLATE_ADVANCED - keep as is) ---
# This prompt template is intended for your QA generation script
# (the one that takes structured JSON as input and outputs QA pairs)

QA_GENERATION_PROMPT_FOR_IN2O3_TCO_REASONING_PREDICTION = """
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


def generate_qa_for_material_entry(material_name: str, details: dict, num_qa_pairs: int = 7) -> list:
    # This function remains largely the same as your provided version
    # It takes material_name and details, formats them into the prompt, calls the LLM, and parses the response.
    details_json_str = json.dumps(details, indent=2, ensure_ascii=False)
    prompt = QA_GENERATION_PROMPT_FOR_IN2O3_TCO_REASONING_PREDICTION.format(
        material_name=material_name,
        details_json_str=details_json_str,
        num_qa_pairs=num_qa_pairs
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_for_qa_generation,
                messages=[
                    {"role": "system",
                     "content": "You are an AI assistant that generates question-answer pairs from structured data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Adjust temperature as needed
                stream=False
            )
            content = response.choices[0].message.content

            # Robust parsing of LLM response (list of QA pairs)
            match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)  # Looks for a JSON list
            if match:
                json_str = match.group(0)
            else:  # Fallback if no clear list, try finding at least one object (less ideal)
                match_obj = re.search(r'\{\s*".*?"\s*:\s*".*?"\s*\}', content, re.DOTALL)
                if match_obj:
                    json_str = f"[{match_obj.group(0)}]"  # Wrap single object in a list
                else:
                    json_str = content  # Last resort

            qa_pairs = json.loads(json_str)
            if isinstance(qa_pairs, list) and all(
                    isinstance(item, dict) and "question" in item and "answer" in item for item in qa_pairs):
                return qa_pairs
            else:
                print(
                    f"⚠️ LLM response for {material_name} not in expected QA list format after parsing. Raw content snippet: {content[:200]}")
                return []

        except json.JSONDecodeError as e:
            print(
                f"❌ JSONDecodeError for {material_name} (Attempt {attempt + 1}/{max_retries}): {e}. Content snippet: {content[:500] if 'content' in locals() else 'Content not available'}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"❌ Error generating QA for {material_name} (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        # If loop finishes due to retries or an unrecoverable error in the last attempt
        if attempt == max_retries - 1:
            return []  # Return empty list on final failure
    return []  # Should be reached only if max_retries is 0 or less


def process_structured_data_for_qa(structured_data_list: List[Dict], checkpoint_path: str) -> List[Dict]:
    all_qa_pairs = []
    processed_entry_identifiers = set()  # To track processed (DOI, MaterialName) to avoid duplicates if input has them
    total_elapsed_qa_gen = 0.0
    start_from_index = 0  # For resuming from checkpoint

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            all_qa_pairs = checkpoint_data.get("all_qa_pairs", [])
            processed_entry_identifiers = set(checkpoint_data.get("processed_entry_identifiers", []))
            total_elapsed_qa_gen = checkpoint_data.get("total_elapsed_qa_gen", 0.0)
            start_from_index = checkpoint_data.get("last_processed_entry_index", -1) + 1
            print(
                f"▶ QA Gen Checkpoint loaded. Resuming from entry index {start_from_index}. Already have {len(all_qa_pairs)} QA pairs.")
        except Exception as e:
            print(f"⚠ QA Gen Checkpoint loading failed: {e}. Starting from scratch.")
            all_qa_pairs = []
            processed_entry_identifiers = set()
            total_elapsed_qa_gen = 0.0
            start_from_index = 0

    # structured_data_list is the list of objects from your extractor's output JSON
    # Each 'item' in this list corresponds to one extracted material entry

    total_material_entries_to_process = len(structured_data_list)
    print(
        f"Total material entries from extractor output to process for QA generation: {total_material_entries_to_process}")

    if total_material_entries_to_process == 0:
        print("No structured material entries found in the input file.")
        return all_qa_pairs

    last_successfully_processed_idx = -1

    try:
        for current_idx, extracted_entry in enumerate(structured_data_list):
            if current_idx < start_from_index:
                last_successfully_processed_idx = current_idx
                continue

            if not isinstance(extracted_entry, dict):
                print(f"⚠️ Skipping item at index {current_idx} as it's not a dictionary.")
                continue

            # --- MODIFICATION START: Accessing data from the new input format ---
            meta_source_paper = extracted_entry.get("meta_source_paper", {})
            extracted_material_data = extracted_entry.get("extracted_material_data", {})

            doi = meta_source_paper.get("doi", f"unknown_doi_idx_{current_idx}")
            material_name = extracted_material_data.get("MaterialName")
            details = extracted_material_data.get("Details")
            # --- MODIFICATION END ---

            if not material_name or not isinstance(details, dict):
                print(
                    f"⚠️ Skipping entry at index {current_idx} due to missing 'MaterialName' or invalid 'Details' (DOI: {doi}).")
                continue

            # Create a unique identifier for checkpointing based on DOI and MaterialName
            entry_identifier = f"{doi}::{material_name}"
            if entry_identifier in processed_entry_identifiers:
                print(f"ℹ️ Skipping already processed entry: {entry_identifier}")
                last_successfully_processed_idx = current_idx
                continue

            session_start_time = time.time()
            # Using material_name for display, could use title from meta_source_paper if preferred
            print(
                f"\nProcessing entry {current_idx + 1}/{total_material_entries_to_process}: {material_name} (DOI: {doi})")

            generated_pairs = generate_qa_for_material_entry(
                material_name,
                details  # Pass the "Details" dictionary
            )

            if generated_pairs:
                # Add source information to each generated QA pair
                for pair in generated_pairs:
                    pair["source_doi"] = doi
                    pair["source_material_name"] = material_name
                    # You can add other meta_source_paper info if needed
                    # pair["source_paper_title"] = meta_source_paper.get("title")
                all_qa_pairs.extend(generated_pairs)
                processed_entry_identifiers.add(entry_identifier)
                print(f"✔ Generated {len(generated_pairs)} QA pairs for {material_name}.")
            else:
                print(f"❌ No QA pairs generated or generation failed for {material_name}.")

            last_successfully_processed_idx = current_idx
            session_elapsed = time.time() - session_start_time
            total_elapsed_qa_gen += session_elapsed

            entries_processed_this_session = (current_idx - start_from_index + 1)
            avg_time_per_entry = total_elapsed_qa_gen / entries_processed_this_session if entries_processed_this_session > 0 else 0
            eta_seconds = avg_time_per_entry * (total_material_entries_to_process - (current_idx + 1))

            print(
                f"Progress: [{current_idx + 1}/{total_material_entries_to_process}] | Session: {session_elapsed:.2f}s | Total QA Time: {total_elapsed_qa_gen:.2f}s | ETA: {eta_seconds / 3600:.2f}h")

            if (current_idx + 1) % 5 == 0:  # Save checkpoint periodically
                save_qa_checkpoint(checkpoint_path, all_qa_pairs, list(processed_entry_identifiers),
                                   total_elapsed_qa_gen, last_successfully_processed_idx)

    except KeyboardInterrupt:
        print("\n⚠ User interrupted. Saving QA generation checkpoint...")
    except Exception as e:
        import traceback
        print(f"\n⚠ An unexpected error occurred during QA generation: {e}")
        traceback.print_exc()
    finally:
        save_qa_checkpoint(checkpoint_path, all_qa_pairs, list(processed_entry_identifiers), total_elapsed_qa_gen,
                           last_successfully_processed_idx if 'last_successfully_processed_idx' in locals() else -1)
        print(f"\nQA Generation process ended. Total QA pairs in list: {len(all_qa_pairs)}")

    return all_qa_pairs


def save_qa_checkpoint(path: str, qa_pairs: list, processed_ids_list: list, elapsed_time: float, last_idx: int):
    # This function remains largely the same, ensure key for processed_ids matches
    checkpoint_content = {
        "all_qa_pairs": qa_pairs,
        "processed_entry_identifiers": processed_ids_list,  # Changed key name for clarity
        "total_elapsed_qa_gen": elapsed_time,
        "last_processed_entry_index": last_idx
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_content, f, indent=2, ensure_ascii=False)
        print(f"\n✔ QA Gen Checkpoint saved to {path}")
    except Exception as e:
        print(f"⚠ Failed to save QA Gen checkpoint: {e}")


def main():
    if not os.path.exists(INPUT_STRUCTURED_DATA_PATH):
        print(f"Error: Input file not found at {INPUT_STRUCTURED_DATA_PATH}")
        return

    try:
        with open(INPUT_STRUCTURED_DATA_PATH, "r", encoding="utf-8") as f:
            # structured_data_list is the direct output from your extractor script
            # which is a list of material entry objects
            structured_data_list = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from input file {INPUT_STRUCTURED_DATA_PATH}: {e}")
        return
    except Exception as e:
        print(f"Error reading input file {INPUT_STRUCTURED_DATA_PATH}: {e}")
        return

    if not isinstance(structured_data_list, list):  # Ensure the loaded data is a list
        print(
            f"Error: Expected input file {INPUT_STRUCTURED_DATA_PATH} to contain a JSON list. Got: {type(structured_data_list)}")
        return

    if not structured_data_list:
        print("No structured data loaded from input file. File might be empty or contain an empty list.")
        # Allow script to proceed to save an empty output if input is empty list
        # return

    print(f"Loaded {len(structured_data_list)} structured material entries from {INPUT_STRUCTURED_DATA_PATH}")

    final_qa_pairs = process_structured_data_for_qa(structured_data_list, CHECKPOINT_PATH_QA_GEN)

    output_dir = os.path.dirname(OUTPUT_QA_PAIRS_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save in JSONL format for fine-tuning
    try:
        with open(OUTPUT_QA_PAIRS_PATH, "w", encoding="utf-8") as f:
            for qa_pair in final_qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
        print(
            f"\n✅ QA pair generation complete. {len(final_qa_pairs)} pairs saved to {OUTPUT_QA_PAIRS_PATH} in JSONL format.")
    except Exception as e:
        print(f"Error writing output file {OUTPUT_QA_PAIRS_PATH}: {e}")

    if os.path.exists(CHECKPOINT_PATH_QA_GEN):
        print(f"ⓘ QA Gen Checkpoint {CHECKPOINT_PATH_QA_GEN} retained. Review results before manual deletion.")


if __name__ == "__main__":
    main()