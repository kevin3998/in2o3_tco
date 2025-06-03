import os
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.core.periodic_table import Element  # 用于检查元素符号

# --- 配置区 ---
MP_API_KEY = os.environ.get("MP_API_KEY")
if not MP_API_KEY:
    MP_API_KEY = input("请输入您的 Materials Project API 密钥: ")

ELEMENTS_TO_INCLUDE = ["In", "O"]
STRICTLY_THESE_ELEMENTS = False

# 更新：请求更全面的属性列表
PROPERTIES_TO_FETCH = [
    # 基本识别信息
    "material_id", "formula_pretty", "chemsys",
    "composition", "composition_reduced", "formula_anonymous",
    "elements", "nelements", "nsites",
    "volume", "density", "density_atomic",
    # 结构与对称性 (symmetry 会包含晶系、空间群等)
    "symmetry",
    "structure",  # 完整的结构对象 (可能非常详细，在CSV/Excel中会是其字典表示)
    # 热力学稳定性
    "formation_energy_per_atom", "energy_per_atom", "uncorrected_energy_per_atom",
    "energy_above_hull", "is_stable",
    "equilibrium_reaction_energy_per_atom", "decomposes_to",
    # 电子结构
    "band_gap", "is_gap_direct", "is_metal",
    "cbm", "vbm", "efermi",  # 导带底、价带顶、费米能级
    "bandstructure",  # 能带结构摘要或对象 (可能详细)
    "dos",  # 态密度摘要或对象 (可能详细)
    # "dos_energy_up", "dos_energy_down", # 通常包含在 dos 对象内，但如果API直接提供也可尝试
    "es_source_calc_id",  # 电子结构计算的源ID
    # 磁性
    "is_magnetic", "ordering", "total_magnetization",
    "total_magnetization_normalized_vol", "total_magnetization_normalized_formula_units",
    "num_magnetic_sites", "num_unique_magnetic_sites", "types_of_magnetic_species",
    # 力学性质
    "bulk_modulus", "shear_modulus", "universal_anisotropy", "homogeneous_poisson",
    # 光谱 (XAS - X射线吸收谱)
    "xas",  # XAS 数据摘要或对象 (可能详细)
    # 其他能量相关 (某些可能已弃用或包含在上述能量中)
    # "e_total", "e_ionic", "e_electronic", # 经常与 energy_per_atom 等相关
    # 表面/界面相关性质 (可能与异质结构材料的筛选相关)
    "weighted_surface_energy_EV_PER_ANG2", "weighted_surface_energy",
    "weighted_work_function", "surface_anisotropy", "shape_factor",
    "has_reconstructed",
    # 通用信息
    "theoretical", "last_updated", "origins", "warnings",
    "task_ids", "deprecated", "deprecation_reasons",
    "possible_species", "has_props",  # has_props 指示是否有其他计算性质如介电、压电等
    "database_IDs",
    "builder_meta",  # 构建此条目的元数据
]
# 移除重复项并确保列表元素唯一
PROPERTIES_TO_FETCH = sorted(list(set(PROPERTIES_TO_FETCH)))

OUTPUT_CSV_FILE = "in_o_materials_comprehensive_data.csv"
OUTPUT_EXCEL_FILE = "in_o_materials_comprehensive_data.xlsx"


# --- 脚本执行区 ---
def fetch_materials_data(api_key, elements_to_include, properties_to_fetch, strictly_these_elements):
    """
    从 Materials Project 数据库获取材料数据。
    """
    print("正在连接到 Materials Project 数据库...")
    try:
        with MPRester(api_key) as mpr:
            print("连接成功！")

            search_kwargs = {"fields": properties_to_fetch}

            if strictly_these_elements:
                if not elements_to_include or len(elements_to_include) == 0:
                    print("错误：当 strictly_these_elements 为 True 时，elements_to_include 不能为空。")
                    return None
                search_kwargs["chemsys"] = "-".join(sorted(elements_to_include))
            elif elements_to_include:
                search_kwargs["all_elements"] = elements_to_include
            else:
                print("警告：未指定任何元素查询条件，可能会返回大量数据或出错。")
                return None

            print(f"查询参数: {search_kwargs}")
            print(f"请求 {len(properties_to_fetch)} 个字段。请注意，部分字段可能返回复杂或大量数据。")

            docs = mpr.materials.summary.search(**search_kwargs)

            if docs:
                print(f"成功检索到 {len(docs)} 条相关材料信息。")
                data_list = []
                for i, doc in enumerate(docs):
                    print(f"  正在处理第 {i + 1}/{len(docs)} 条材料: {getattr(doc, 'material_id', 'N/A')}", end='\r')
                    try:
                        data_dict = doc.model_dump(exclude_none=True)  # For Pydantic v2+, exclude_none 减少空值
                    except AttributeError:
                        data_dict = doc.dict(exclude_none=True)  # For Pydantic v1

                    # 从 'symmetry' 字段提取详细信息
                    symmetry_data = data_dict.get("symmetry")
                    if symmetry_data and isinstance(symmetry_data, dict):
                        data_dict['crystal_system'] = symmetry_data.get('crystal_system')
                        data_dict['spacegroup_symbol'] = symmetry_data.get('symbol')
                        data_dict['spacegroup_number'] = symmetry_data.get('number')
                    else:
                        data_dict['crystal_system'] = None
                        data_dict['spacegroup_symbol'] = None
                        data_dict['spacegroup_number'] = None

                    # 处理 'elements' 字段
                    if 'elements' in data_dict and isinstance(data_dict['elements'], list):
                        str_elements = []
                        for el_obj in data_dict['elements']:
                            if hasattr(el_obj, 'symbol'):
                                str_elements.append(el_obj.symbol)
                            elif isinstance(el_obj, str):
                                str_elements.append(el_obj)
                            else:
                                str_elements.append(str(el_obj))
                        data_dict['elements'] = str_elements

                    # 对于其他可能为复杂对象的字段（如 structure, bandstructure, dos, xas），
                    # model_dump() 会将其转换为字典。它们在CSV中会是嵌套的JSON字符串或类似形式。
                    # 这里不需要额外处理，除非你想提取特定子属性。
                    # 例如，如果想提取晶格常数：
                    # structure_data = data_dict.get("structure")
                    # if structure_data and isinstance(structure_data, dict) and "lattice" in structure_data:
                    #     lattice = structure_data["lattice"]
                    #     data_dict["lattice_a"] = lattice.get("a")
                    #     data_dict["lattice_b"] = lattice.get("b")
                    #     data_dict["lattice_c"] = lattice.get("c")
                    #     data_dict["lattice_alpha"] = lattice.get("alpha")
                    #     data_dict["lattice_beta"] = lattice.get("beta")
                    #     data_dict["lattice_gamma"] = lattice.get("gamma")

                    data_list.append(data_dict)
                print("\n所有材料信息处理完毕。")
                return data_list
            else:
                print("未找到符合条件的材料信息。")
                return []

    except Exception as e:
        print(f"\n发生错误: {e}")
        print("请检查您的 API 密钥是否正确、网络连接是否正常，以及请求的字段和参数是否有效。")
        print("也请确保您的 pymatgen 库是最新版本。")
        return None


def save_data_to_files(data, csv_filename, excel_filename):
    if not data:
        print("没有数据可以保存。")
        return

    print(f"准备将 {len(data)} 条数据保存到文件...")
    try:
        df = pd.DataFrame(data)

        if 'elements' in df.columns:
            df['elements'] = df['elements'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)

        # 将所有列转换为字符串，以避免因复杂嵌套对象导致写入Excel错误
        # 对于非常复杂或嵌套的字段，它们在CSV/Excel中可能显示为JSON字符串
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].astype(str)

        print(f"正在保存到 CSV 文件: {csv_filename} ...")
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"数据已成功保存到 CSV 文件: {csv_filename}")

        try:
            print(f"正在保存到 Excel 文件: {excel_filename} ...")
            df.to_excel(excel_filename, index=False)
            print(f"数据已成功保存到 Excel 文件: {excel_filename}")
        except Exception as e_excel:
            print(f"保存到 Excel 时发生错误: {e_excel}")
            print("Excel 文件可能未成功保存。CSV文件应已保存。")
            print("这通常发生在数据包含非常长或复杂（如深度嵌套）的字符串时，超出了Excel单元格的限制。")

    except Exception as e:
        print(f"保存文件时发生一般性错误: {e}")


if __name__ == "__main__":
    print("--- 开始从 Materials Project 提取数据 (尝试获取更全面的信息) ---")
    print("警告：请求大量字段（尤其是结构、能带、DOS等）可能会导致较长的下载时间和较大的文件。")
    print("对于异质结构，本脚本主要获取体相材料的属性，部分表面性质可能相关，但不会直接给出界面数据。")

    materials_data = fetch_materials_data(
        api_key=MP_API_KEY,
        elements_to_include=ELEMENTS_TO_INCLUDE,
        properties_to_fetch=PROPERTIES_TO_FETCH,
        strictly_these_elements=STRICTLY_THESE_ELEMENTS
    )

    if materials_data is not None:
        save_data_to_files(materials_data, OUTPUT_CSV_FILE, OUTPUT_EXCEL_FILE)

        if materials_data:
            print("\n--- 数据预览 (部分关键列，前3条) ---")
            df_preview = pd.DataFrame(materials_data)
            preview_cols = ["material_id", "formula_pretty", "formation_energy_per_atom", "band_gap", "crystal_system",
                            "spacegroup_symbol"]
            # 确保预览的列存在
            actual_preview_cols = [col for col in preview_cols if col in df_preview.columns]
            if 'elements' in df_preview.columns and df_preview['elements'].apply(lambda x: isinstance(x, list)).any():
                df_preview['elements'] = df_preview['elements'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else x)

            print(df_preview[actual_preview_cols].head(3))
        else:
            print("未检索到数据，不进行预览。")

    print("\n--- 脚本执行完毕 ---")