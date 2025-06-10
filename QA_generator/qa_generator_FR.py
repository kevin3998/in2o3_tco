# qa_generator_FR.py (MODIFIED FOR DIVERSE QUESTION GENERATION)
import json
import time
import os
import regex as re
import argparse
import logging
from typing import List, Dict, Any, Iterator
from openai import OpenAI
from dotenv import load_dotenv

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()]
)


# --- 2. LLM Client Initialization ---
def get_llm_client() -> OpenAI:
    """Initializes and returns the OpenAI client, loading credentials from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables (.env file).")
        raise ValueError("OPENAI_API_KEY not found.")

    return OpenAI(api_key=api_key, base_url=base_url)


# --- 3. Checkpoint Management ---
def save_checkpoint(checkpoint_path: str, data: dict):
    """Saves progress to a checkpoint file."""
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logging.debug(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")


def load_checkpoint(checkpoint_path: str) -> dict:
    """Loads progress from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not load checkpoint {checkpoint_path}: {e}. Starting fresh.")
    return {"processed_entries": 0, "qa_pairs": []}


# --- 4. Core QA Generation Logic ---

# --- MODIFICATION: New, improved prompt for generating diverse questions ---
BATCH_QA_DIVERSE_GENERATION_PROMPT = """
You are an AI assistant that creates high-quality, DIVERSE question-answer pairs for fine-tuning a materials science language model.
Based on the provided information about a specific material, you will generate a series of natural-sounding questions.

**Context:**
- **Material Name:** "{material_name}"
- **Fields to Process (JSON format):**
{fields_json}

**Task:**
For EACH field in the JSON above, generate one clear and natural-sounding question in English.
To create a robust dataset, you MUST VARY the style of your questions. Use a mix of the styles shown in the examples below. Do not use the same style for every question.

**Desired Question Styles & Examples:**
- **Interrogative (Standard Question):** "What was the deposition temperature used for the {material_name} films?"
- **Declarative (Statement of Need):** "I'm looking for the deposition temperature of the {material_name}." or "Tell me the deposition temperature used for {material_name}."
- **Imperative (Command):** "Find the deposition temperature for the {material_name}."
- **Keyword-based (Short Query):** "Deposition temperature for {material_name}?"

**Output Format:**
Respond with a single, valid JSON array of objects. Each object in the array must have three keys: "field_name_human", "question", and "answer".
The "field_name_human" and "answer" must exactly match the values from the input JSON. Do NOT include any other text, explanations, or markdown formatting like ```json outside of the JSON array.

**Generated JSON:**
"""


def _camel_to_human(name: str) -> str:
    """Converts CamelCase to human-readable 'camel case'."""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()


def _recursive_generate_qa(
        data_dict: Dict,
        material_name: str,
        path: List[str]
) -> Iterator[Dict[str, str]]:
    """
    Recursively traverses the data dictionary to generate QA pairs.
    It now uses the path to create more context-aware questions.
    """
    for key, value in data_dict.items():
        if isinstance(value, dict) and value:
            yield from _recursive_generate_qa(value, material_name, path + [key])
        elif isinstance(value, list) and value:
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    list_item_path = path + [f"{key} (item {i + 1})"]
                    yield from _recursive_generate_qa(item, material_name, list_item_path)
        elif isinstance(value, (str, int, float)) and str(value).strip():
            category_path_str = " -> ".join(path)
            field_name_human = _camel_to_human(key)

            if "unit" in field_name_human.lower() or "summary" in field_name_human.lower():
                continue

            yield {
                "material_name": material_name,
                "category_path": category_path_str,
                "field_name_human": field_name_human,
                "field_value": str(value)
            }


# --- MODIFICATION: Rewritten function for batch processing ---
def process_structured_data_for_qa(
        client: OpenAI,
        model_name: str,
        domain: str,
        structured_data: List[Dict],
        checkpoint_path: str
) -> List[Dict[str, Any]]:
    """
    Processes a list of structured data entries to generate QA pairs using an LLM
    in a batch mode (one LLM call per material).
    """
    checkpoint = load_checkpoint(checkpoint_path)
    all_qa_pairs = checkpoint.get("qa_pairs", [])
    start_index = checkpoint.get("processed_entries", 0)

    total_entries = len(structured_data)
    logging.info(
        f"Starting QA generation in BATCH mode. Total entries: {total_entries}. Already processed: {start_index}.")

    for i, entry in enumerate(structured_data[start_index:], start=start_index):
        try:
            material_data = entry.get("extracted_material_data", {})
            material_name = material_data.get("MaterialName", "this material")
            details = material_data.get("Details", {})

            # 1. Gather all potential fields for the current material entry.
            potential_qa_items = list(_recursive_generate_qa(details, material_name, []))

            if not potential_qa_items:
                logging.info(f"No fields to process for material '{material_name}'. Skipping.")
                continue

            # Create a lookup map to correlate LLM output with original data.
            field_map = {item['field_name_human']: item for item in potential_qa_items}

            # 2. Format the fields for the batch prompt.
            fields_for_prompt = [
                {
                    "category_path": item["category_path"],
                    "field_name_human": item["field_name_human"],
                    "field_value": item["field_value"]
                } for item in potential_qa_items
            ]
            fields_json_str = json.dumps(fields_for_prompt, indent=2)
            # Use the new diverse prompt template
            prompt = BATCH_QA_DIVERSE_GENERATION_PROMPT.format(
                material_name=material_name,
                fields_json=fields_json_str
            )

            # 3. Make a single LLM call for the entire material.
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,  # Slightly increase temperature for more creativity/diversity
                    max_tokens=4096,
                    stream=False
                )

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    response_content = response.choices[0].message.content.strip()

                    # 4. Parse the JSON response from the LLM.
                    try:
                        # Attempt to find the JSON array in the response to handle potential markdown ```json
                        match = re.search(r'\[.*\]', response_content, re.DOTALL)
                        if not match:
                            raise json.JSONDecodeError("No JSON array found in the response.", response_content, 0)

                        generated_pairs = json.loads(match.group(0))

                        if not isinstance(generated_pairs, list):
                            logging.warning(f"LLM output for '{material_name}' was not a list. Skipping.")
                            continue

                        # 5. Correlate generated pairs with original data and append.
                        for pair in generated_pairs:
                            question = pair.get("question")
                            answer = pair.get("answer")
                            field_name = pair.get("field_name_human")

                            if not all([question, answer, field_name]):
                                logging.warning(f"Skipping incomplete pair from LLM for '{material_name}': {pair}")
                                continue

                            original_item = field_map.get(field_name)
                            if original_item:
                                qa_pair = {
                                    "source_doi": entry.get("meta_source_paper", {}).get("doi"),
                                    "material_name": original_item["material_name"],
                                    "context": f"Regarding the material '{original_item['material_name']}', specifically its {original_item['category_path']}.",
                                    "question": question,
                                    "answer": original_item["field_value"]  # Use original answer as ground truth
                                }
                                all_qa_pairs.append(qa_pair)
                            else:
                                logging.warning(
                                    f"LLM returned question for unknown field '{field_name}' in material '{material_name}'.")

                    except json.JSONDecodeError:
                        logging.error(
                            f"Failed to decode JSON from LLM for material '{material_name}'. Response:\n{response_content}")
                        continue
                else:
                    finish_reason = response.choices[0].finish_reason if response.choices else "unknown reason"
                    logging.warning(
                        f"LLM returned no content for material '{material_name}'. Finish reason: {finish_reason}. Skipping.")

            except Exception as llm_e:
                logging.error(f"LLM call failed for material {material_name}: {llm_e}")
                continue

            # 6. Log progress and save checkpoints.
            progress = (i + 1) / total_entries * 100
            logging.info(f"Progress: {progress:.2f}% ({i + 1}/{total_entries}) - Processed material '{material_name}'")

            if (i + 1) % 5 == 0:
                save_checkpoint(checkpoint_path, {"processed_entries": i + 1, "qa_pairs": all_qa_pairs})

        except Exception as e:
            logging.error(f"Error processing entry {i}: {e}", exc_info=True)
            continue

    save_checkpoint(checkpoint_path, {"processed_entries": total_entries, "qa_pairs": all_qa_pairs})
    return all_qa_pairs


# --- 5. Main Execution Block ---
def main():
    """Main function to run the QA pair generation pipeline."""
    IDE_RUN_CONFIG = True

    args = None
    if IDE_RUN_CONFIG:
        logging.info("--- RUNNING IN IDE MODE: Using parameters defined in the script. ---")

        class Args:
            pass

        args = Args()
        args.input_file = "../data/extracted_json/pydantic/structured_info.json"
        args.output_file = "../data/generated_qa_pairs/qa_pairs_FR.jsonl"
        args.checkpoint_file = "../data/generated_qa_pairs/checkpoint_qa_gen_v3_batch_diverse.json"
        args.domain = "in2o3_tco"
        args.model_name = "DeepSeek-R1-671B"

    else:
        logging.info("--- RUNNING IN COMMAND-LINE MODE: Parsing arguments from terminal. ---")
        parser = argparse.ArgumentParser(
            description="Generate Question-Answer pairs from structured JSON data for LLM fine-tuning.")
        parser.add_argument("--input_file", type=str, help="Path to the structured JSON input file.")
        parser.add_argument("--output_file", type=str, help="Path to save the generated QA pairs (JSONL).")
        parser.add_argument("--checkpoint_file", type=str, help="Path for the checkpoint file.")
        parser.add_argument("--domain", type=str, help="Domain of the data.")
        parser.add_argument("--model_name", type=str, help="LLM to use for generating questions.")
        parser.set_defaults(
            input_file="../data/extracted_json/pydantic/structured_info.json",
            output_file="../data/generated_qa_pairs/qa_pairs_FR.jsonl",
            checkpoint_file="../data/generated_qa_pairs/checkpoint_qa_gen_v3_batch_diverse.json",
            domain="in2o3_tco",
            model_name="DeepSeek-R1-671B"
        )
        args = parser.parse_args()

    try:
        client = get_llm_client()
        logging.info(f"Loading structured data from: {args.input_file}")
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

    final_qa_pairs = process_structured_data_for_qa(
        client,
        args.model_name,
        args.domain,
        structured_data_list,
        args.checkpoint_file
    )

    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for qa_pair in final_qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
        logging.info(
            f"\n✅ QA pair generation complete. {len(final_qa_pairs)} pairs saved to {args.output_file} in JSONL format.")
    except Exception as e:
        logging.error(f"Error writing output file {args.output_file}: {e}")

    # Cleanup checkpoint on successful completion
    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)
        logging.info(f"Checkpoint file {args.checkpoint_file} removed.")


if __name__ == "__main__":
    main()