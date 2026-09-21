#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理SPE数据，结合溶剂分类和浓度信息生成表格
将浓度信息加入到溶剂数据中，格式为 (solvent, ratio)
"""

import json
import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


def load_solvent_classification(file_path: str) -> Dict[str, str]:
    """
    加载溶剂分类文件，建立溶剂名称到类别的映射
    
    Args:
        file_path: 溶剂分类JSON文件路径
        
    Returns:
        溶剂名称到类别的映射字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        classification_data = json.load(f)
    
    # 建立溶剂名称到类别的映射
    solvent_to_category = {}
    for category, solvents in classification_data.items():
        for solvent in solvents:
            # 将溶剂名称转换为小写，便于匹配
            solvent_lower = solvent.lower().strip()
            solvent_to_category[solvent_lower] = category
    
    return solvent_to_category


def normalize_solvent_name(solvent_name: str) -> str:
    """
    标准化溶剂名称，便于匹配
    
    Args:
        solvent_name: 原始溶剂名称
        
    Returns:
        标准化后的溶剂名称
    """
    if not solvent_name or solvent_name.lower() in ['none', 'null', '']:
        return ''
    
    # 转换为小写并去除多余空格
    normalized = solvent_name.lower().strip()
    
    # 移除常见的单位信息
    normalized = re.sub(r'\s*\d+\s*(ml|ml/l|mol/l|m|mm|g/l|%|v/v|w/w)\b', '', normalized)
    normalized = re.sub(r'\s*\d+\.\d+\s*(ml|ml/l|mol/l|m|mm|g/l|%|v/v|w/w)\b', '', normalized)
    
    # 移除pH相关信息
    normalized = re.sub(r'\s*\(ph\s*\d+(\.\d+)?\)', '', normalized)
    
    # 移除浓度信息
    normalized = re.sub(r'\s*\d+(\.\d+)?\s*mol/l', '', normalized)
    normalized = re.sub(r'\s*\d+(\.\d+)?\s*m', '', normalized)
    
    return normalized.strip()


def find_solvent_category(solvent_name: str, solvent_mapping: Dict[str, str]) -> Optional[str]:
    """
    根据溶剂名称查找对应的类别
    
    Args:
        solvent_name: 溶剂名称
        solvent_mapping: 溶剂名称到类别的映射
        
    Returns:
        对应的类别，如果找不到则返回None
    """
    if not solvent_name:
        return None
    
    # 首先尝试直接匹配
    normalized_name = normalize_solvent_name(solvent_name)
    if normalized_name in solvent_mapping:
        return solvent_mapping[normalized_name]
    
    # 尝试部分匹配
    for mapped_solvent, category in solvent_mapping.items():
        if normalized_name in mapped_solvent or mapped_solvent in normalized_name:
            return category
    
    # 尝试关键词匹配
    keywords = normalized_name.split()
    for keyword in keywords:
        if keyword in solvent_mapping:
            return solvent_mapping[keyword]
    
    return None


def convert_ratio_to_decimal(ratio: Any) -> Optional[float]:
    """
    将各种格式的ratio转换为0-1之间的小数
    
    支持的格式：
    1. 比例格式 (X:Y): 转换为 val1 / (val1 + val2)
    2. 百分比格式: 提取百分比数字并除以100
    3. 数值格式: 如果已经是0-1之间就保留，如果>1且<=100，除以100
    
    返回: 0-1之间的小数，如果无法转换则返回None
    """
    if ratio is None:
        return None
    
    if not isinstance(ratio, str):
        # 如果是数字类型
        try:
            val = float(ratio)
            if 0 <= val <= 1:
                return val
            elif 1 < val <= 100:
                # 可能是百分比，转换为小数
                return val / 100.0
            else:
                return None
        except:
            return None
    
    ratio_str = str(ratio).strip()
    
    # 空字符串或'none'
    if ratio_str == '' or ratio_str.lower() == 'none':
        return None
    
    # 1. 尝试解析比例格式 (X:Y)
    ratio_pattern = r'^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$'
    match = re.match(ratio_pattern, ratio_str)
    if match:
        val1 = float(match.group(1))
        val2 = float(match.group(2))
        total = val1 + val2
        if total > 0:
            # 返回第一个值占总和的比例
            return val1 / total
        else:
            return None
    
    # 2. 尝试解析百分比格式
    percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
    match = re.search(percent_pattern, ratio_str)
    if match:
        percent_val = float(match.group(1))
        if 0 <= percent_val <= 100:
            return percent_val / 100.0
        else:
            return None
    
    # 3. 尝试解析纯数字
    try:
        val = float(ratio_str)
        if 0 <= val <= 1:
            return val
        elif 1 < val <= 100:
            # 可能是百分比，转换为小数
            return val / 100.0
        else:
            return None
    except:
        pass
    
    # 4. 尝试从包含冒号的其他格式中提取数字
    if ':' in ratio_str:
        numbers = re.findall(r'\d+(?:\.\d+)?', ratio_str)
        if len(numbers) >= 2:
            try:
                val1 = float(numbers[0])
                val2 = float(numbers[1])
                total = val1 + val2
                if total > 0:
                    return val1 / total
            except:
                pass
    
    return None


def extract_step_solvents_with_ratio(step_data: Any, solvent_mapping: Dict[str, str]) -> List[Tuple[str, Optional[float]]]:
    """
    从步骤数据中提取溶剂信息和对应的浓度
    
    Args:
        step_data: 步骤数据（可能是列表或字典）
        solvent_mapping: 溶剂名称到类别的映射
        
    Returns:
        列表，每个元素是 (溶剂类别, 浓度值) 的元组
    """
    solvents_with_ratio = []
    
    if isinstance(step_data, list):
        for item in step_data:
            if isinstance(item, dict):
                kind = item.get('Kind')
                ratio = item.get('Ratio')
                
                if kind and kind.lower() not in ['none', 'null', '']:
                    # 查找溶剂类别
                    category = find_solvent_category(kind, solvent_mapping)
                    if category:
                        # 转换ratio为小数
                        ratio_decimal = convert_ratio_to_decimal(ratio)
                        solvents_with_ratio.append((category, ratio_decimal))
                    else:
                        # 如果找不到类别，使用原始名称
                        ratio_decimal = convert_ratio_to_decimal(ratio)
                        solvents_with_ratio.append((kind, ratio_decimal))
    elif isinstance(step_data, dict):
        kind = step_data.get('Kind')
        ratio = step_data.get('Ratio')
        
        if kind and kind.lower() not in ['none', 'null', '']:
            # 查找溶剂类别
            category = find_solvent_category(kind, solvent_mapping)
            if category:
                # 转换ratio为小数
                ratio_decimal = convert_ratio_to_decimal(ratio)
                solvents_with_ratio.append((category, ratio_decimal))
            else:
                # 如果找不到类别，使用原始名称
                ratio_decimal = convert_ratio_to_decimal(ratio)
                solvents_with_ratio.append((kind, ratio_decimal))
    
    return solvents_with_ratio


def format_solvent_with_ratio(solvent: str, ratio: Optional[float]) -> str:
    """
    格式化溶剂和浓度信息
    
    Args:
        solvent: 溶剂名称或类别
        ratio: 浓度值（0-1之间的小数）或None
        
    Returns:
        格式化后的字符串，如 "(unknown, 0.1)" 或 "(nitric acid+water, null)"
    """
    if ratio is None:
        return f"({solvent}, null)"
    else:
        return f"({solvent}, {ratio})"


def process_method_data(method: Dict[str, Any], solvent_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    处理单个方法的数据，提取溶剂和浓度信息
    
    Args:
        method: 方法数据
        solvent_mapping: 溶剂名称到类别的映射
        
    Returns:
        处理后的方法数据
    """
    cas_mp = method.get('casMp', '')
    
    # 解析CasMp字段
    cas_mp_list = []
    if cas_mp:
        # 移除方括号和引号，然后按逗号分割
        cas_mp_clean = cas_mp.strip("[]'\"")
        cas_mp_list = [item.strip().strip("'\"") for item in cas_mp_clean.split(',')]
    
    extracted_response = method.get('extracted_response', {})
    
    # 处理各个步骤
    steps = ['Sample loading', 'Condition', 'Wash', 'Elute', 'Reconstitute']
    step_formatted = {}
    
    for step in steps:
        step_data = extracted_response.get(step, [])
        solvents_with_ratio = extract_step_solvents_with_ratio(step_data, solvent_mapping)
        
        if solvents_with_ratio:
            # 格式化每个溶剂和浓度
            formatted_solvents = []
            for solvent, ratio in solvents_with_ratio:
                formatted = format_solvent_with_ratio(solvent, ratio)
                formatted_solvents.append(formatted)
            
            # 用分号连接多个溶剂
            step_formatted[step] = '; '.join(formatted_solvents)
        else:
            step_formatted[step] = ''
    
    return {
        'CasMp': cas_mp_list,
        **step_formatted
    }


def count_empty_steps(row: pd.Series) -> int:
    """
    计算一行中空步骤的数量
    
    Args:
        row: DataFrame的一行
        
    Returns:
        空步骤的数量
    """
    steps = ['Sample loading', 'Condition', 'Wash', 'Elute', 'Reconstitute']
    empty_count = 0
    
    for step in steps:
        value = row[step]
        if pd.isna(value) or value == '':
            empty_count += 1
    
    return empty_count


def main():
    """主函数"""
    # 文件路径
    base_dir = Path(__file__).parent.parent
    classification_file = base_dir / 'data' / 'spe_kind_classification_reclassified.json'
    methods_file = base_dir / 'data' / 'merged_extracted_content_classified_v2.json'
    output_file = base_dir / 'data' / 'spe_solvent_ratio.csv'
    
    print("正在加载溶剂分类数据...")
    solvent_mapping = load_solvent_classification(str(classification_file))
    print(f"加载了 {len(solvent_mapping)} 个溶剂分类映射")
    
    print("正在加载方法数据...")
    with open(methods_file, 'r', encoding='utf-8') as f:
        methods_data = json.load(f)
    print(f"加载了 {len(methods_data)} 个方法")
    
    print("正在处理数据...")
    processed_data = []
    
    for i, method in enumerate(methods_data):
        if i % 1000 == 0:
            print(f"已处理 {i}/{len(methods_data)} 个方法")
        
        processed_method = process_method_data(method, solvent_mapping)
        processed_data.append(processed_method)
    
    print("正在生成CSV文件...")
    
    # 创建DataFrame
    df_data = []
    for item in processed_data:
        cas_mp_list = item['CasMp']
        if cas_mp_list:
            # 为每个CasMp创建一行
            for cas_mp in cas_mp_list:
                row = {
                    'CasMp': f"['{cas_mp}']",
                    'Sample loading': item['Sample loading'],
                    'Condition': item['Condition'],
                    'Wash': item['Wash'],
                    'Elute': item['Elute'],
                    'Reconstitute': item['Reconstitute']
                }
                df_data.append(row)
        else:
            # 如果没有CasMp，创建一个空行
            row = {
                'CasMp': '',
                'Sample loading': item['Sample loading'],
                'Condition': item['Condition'],
                'Wash': item['Wash'],
                'Elute': item['Elute'],
                'Reconstitute': item['Reconstitute']
            }
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    print(f"处理完成！共生成 {len(df)} 行数据")
    
    # 数据清洗：仅保留空步骤数小于等于2的方法
    print("\n正在进行数据清洗...")
    df['empty_steps'] = df.apply(count_empty_steps, axis=1)
    before_clean = len(df)
    df_cleaned = df[df['empty_steps'] <= 2].copy()
    after_clean = len(df_cleaned)
    removed_count = before_clean - after_clean
    
    print(f"清洗前: {before_clean} 行")
    print(f"清洗后: {after_clean} 行")
    print(f"清洗掉了 {removed_count} 个方法（{removed_count/before_clean*100:.2f}%）")
    
    # 删除辅助列
    df_cleaned = df_cleaned.drop(columns=['empty_steps'])
    
    # 将空字符串替换为NaN，然后保存到CSV文件
    df_cleaned = df_cleaned.replace('', pd.NA)
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8', na_rep='')
    
    print(f"\n处理完成！结果已保存到: {output_file}")
    print(f"总共生成了 {len(df_cleaned)} 行数据")
    
    # 显示前几行作为示例
    print("\n前5行数据示例:")
    print(df_cleaned.head())


if __name__ == "__main__":
    main()




