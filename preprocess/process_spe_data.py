#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理SPE数据，结合溶剂分类和方法步骤信息生成表格
"""

import json
import pandas as pd
import re
from typing import Dict, List, Any, Optional


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


def extract_step_solvents(step_data: Any) -> List[str]:
    """
    从步骤数据中提取溶剂信息
    
    Args:
        step_data: 步骤数据（可能是列表或字典）
        
    Returns:
        溶剂名称列表
    """
    solvents = []
    
    if isinstance(step_data, list):
        for item in step_data:
            if isinstance(item, dict) and 'Kind' in item:
                kind = item['Kind']
                if kind and kind.lower() not in ['none', 'null', '']:
                    solvents.append(kind)
    elif isinstance(step_data, dict):
        if 'Kind' in step_data:
            kind = step_data['Kind']
            if kind and kind.lower() not in ['none', 'null', '']:
                solvents.append(kind)
    
    return solvents


def process_method_data(method: Dict[str, Any], solvent_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    处理单个方法的数据
    
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
    step_categories = {}
    
    for step in steps:
        step_data = extracted_response.get(step, [])
        solvents = extract_step_solvents(step_data)
        
        # 将溶剂名称映射到类别
        categories = []
        for solvent in solvents:
            category = find_solvent_category(solvent, solvent_mapping)
            if category:
                categories.append(category)
        
        # 去重并保持顺序
        unique_categories = []
        for cat in categories:
            if cat not in unique_categories:
                unique_categories.append(cat)
        
        step_categories[step] = '; '.join(unique_categories) if unique_categories else ''
    
    return {
        'CasMp': cas_mp_list,
        **step_categories
    }


def main():
    """主函数"""
    # 文件路径
    classification_file = '/Users/suziyang/Documents/others/环境污染物project/data/spe_kind_classification_reclassified.json'
    methods_file = '/Users/suziyang/Documents/others/环境污染物project/data/deepseek_extraction/merged_extracted_content_classified_v2.json'
    output_file = '/Users/suziyang/Documents/others/环境污染物project/result/spe_processed_data_new.csv'
    
    print("正在加载溶剂分类数据...")
    solvent_mapping = load_solvent_classification(classification_file)
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
    
    # 将空字符串替换为NaN，然后保存到CSV文件
    df = df.replace('', pd.NA)
    df.to_csv(output_file, index=False, encoding='utf-8', na_rep='')
    
    print(f"处理完成！结果已保存到: {output_file}")
    print(f"总共生成了 {len(df)} 行数据")
    
    # 显示前几行作为示例
    print("\n前5行数据示例:")
    print(df.head())


if __name__ == "__main__":
    main()
