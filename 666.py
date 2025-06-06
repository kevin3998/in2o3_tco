# extract_specific_data.py
import json
import argparse
import os
import logging

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def extract_data_fields(input_file_path: str, output_file_path: str, target_key: str = "extracted_material_data"):
    """
    Extracts a specific key's value from each entry in a list within a JSON file
    and saves these extracted parts to a new JSON file.

    Args:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to save the output JSON file.
        target_key (str): The key whose value needs to be extracted from each entry.
                          Defaults to "extracted_material_data".
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            source_data_list = json.load(f_in)
    except FileNotFoundError:
        logging.error(f"错误：输入文件未找到 '{input_file_path}'")
        return
    except json.JSONDecodeError:
        logging.error(f"错误：无法解析输入文件中的JSON '{input_file_path}'")
        return
    except Exception as e:
        logging.error(f"读取输入文件时发生未知错误: {e}")
        return

    if not isinstance(source_data_list, list):
        logging.error(f"错误：输入文件 '{input_file_path}' 的顶层结构应为一个列表。实际得到: {type(source_data_list)}")
        return

    extracted_data_items = []
    entries_processed = 0
    entries_with_target_key = 0

    for i, entry in enumerate(source_data_list):
        entries_processed += 1
        if isinstance(entry, dict):
            target_data = entry.get(target_key)
            if target_data is not None: # Check if key exists
                if isinstance(target_data, dict): # Ensure the target data itself is a dict as expected
                    extracted_data_items.append(target_data)
                    entries_with_target_key += 1
                else:
                    logging.warning(f"条目 {i} 中 '{target_key}' 的值不是预期的字典类型，而是 {type(target_data)}。已跳过。")
            else:
                logging.warning(f"条目 {i} 中未找到目标键 '{target_key}'。已跳过。")
        else:
            logging.warning(f"列表中的条目 {i} 不是字典类型，而是 {type(entry)}。已跳过。")

    logging.info(f"共处理 {entries_processed} 个条目。")
    logging.info(f"找到并提取了 {entries_with_target_key} 个 '{target_key}' 部分。")

    if not extracted_data_items:
        logging.warning(f"未提取到任何 '{target_key}' 数据，输出文件将为空列表。")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"已创建输出目录: {output_dir}")
        except OSError as e:
            logging.error(f"创建输出目录 '{output_dir}' 失败: {e}")
            return


    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(extracted_data_items, f_out, indent=2, ensure_ascii=False)
        logging.info(f"提取的数据已成功保存至: '{output_file_path}'")
    except IOError as e:
        logging.error(f"写入输出文件 '{output_file_path}' 失败: {e}")
    except Exception as e:
        logging.error(f"保存输出文件时发生未知错误: {e}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="从结构化提取输出的JSON文件中提取所有 'extracted_material_data' 部分。"
    # )
    # parser.add_argument(
    #     "input_file",
    #     type=str,
    #     help="包含结构化提取结果的输入JSON文件路径。"
    # )
    # parser.add_argument(
    #     "output_file",
    #     type=str,
    #     help="用于保存提取出的 'extracted_material_data' 列表的输出JSON文件路径。"
    # )
    # parser.add_argument(
    #     "--key",
    #     type=str,
    #     default="extracted_material_data",
    #     help="要从每个条目中提取的目标键名 (默认为: 'extracted_material_data')."
    # )

    # args = parser.parse_args()
    # extract_data_fields(args.input_file, args.output_file, args.key)


    # PyCharm直接运行时可以修改这里的默认值，或通过命令行参数覆盖
    # Pycharm run example:
    SCRIPT_INPUT_FILE = "data/extracted_json/pydantic/structured2.json"
    SCRIPT_OUTPUT_FILE = "data/extracted_json/pydantic/material_data_only2.json"
    SCRIPT_TARGET_KEY = "extracted_material_data"
    extract_data_fields(SCRIPT_INPUT_FILE, SCRIPT_OUTPUT_FILE, SCRIPT_TARGET_KEY)

