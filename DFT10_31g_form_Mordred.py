#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Mordred descriptors for candidate molecules
读取 candidate_molecules-31g.xlsx，计算35个指定描述符，保存到新文件
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import os
import warnings

warnings.filterwarnings('ignore')

# ================= Configuration =================
# 35个指定的描述符（与建模一致）
SELECTED_BITS = [
    906, 1336, 334, 1310, 1348, 1406, 1437, 345, 218, 230,
    449, 510, 299, 3, 1547, 566, 1418, 1069, 232, 264,
    1076, 1077, 322, 1313, 1294, 1064, 1329, 1146, 1477, 1359,
    805, 1394, 1358, 1573, 485
]

# 输入输出文件
INPUT_FILE = 'candidate_molecules-31g.xlsx'
OUTPUT_FILE = 'candidate_molecules-31g_with_Md.xlsx'
DESCRIPTOR_INFO_FILE = 'candidate_mordred_descriptor_info-31g.xlsx'
SELECTED_OUTPUT_FILE = 'candidate_molecules-31g_35descriptors.xlsx'


def calculate_mordred_descriptors(smiles_list, compound_names):
    """
    为SMILES列表计算Mordred描述符

    Parameters:
    -----------
    smiles_list : list
        SMILES字符串列表
    compound_names : list
        化合物名称列表

    Returns:
    --------
    descriptor_df : pandas.DataFrame
        包含所有描述符的DataFrame
    descriptor_info : pandas.DataFrame
        描述符信息
    invalid_compounds : list
        无效SMILES的列表
    """
    print("初始化Mordred计算器...")
    calc = Calculator(descriptors, ignore_3D=True)

    print(f"总描述符数量: {len(calc.descriptors)}")

    # 获取描述符名称
    descriptor_names = [str(d) for d in calc.descriptors]

    # 初始化结果字典
    results = {f'Md_{i}': [] for i in range(len(descriptor_names))}
    valid_names = []
    invalid_compounds = []

    print(f"\n处理 {len(smiles_list)} 个化合物...")

    for idx, (name, smiles) in enumerate(zip(compound_names, smiles_list)):
        print(f"  [{idx + 1}/{len(smiles_list)}] {name}")

        # 转换SMILES为分子
        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            print(f"    ✗ 无效SMILES: {smiles[:60]}...")
            invalid_compounds.append((name, smiles))
            # 为这个化合物添加NaN值
            for i in range(len(descriptor_names)):
                results[f'Md_{i}'].append(np.nan)
            valid_names.append(name)
        else:
            print(f"    ✓ 分子有效，计算描述符中...")
            try:
                desc_values = calc(mol)

                # 存储描述符值
                for i, value in enumerate(desc_values):
                    try:
                        if value is None or isinstance(value, str):
                            results[f'Md_{i}'].append(np.nan)
                        else:
                            results[f'Md_{i}'].append(float(value))
                    except (ValueError, TypeError):
                        results[f'Md_{i}'].append(np.nan)

                valid_names.append(name)
                print(f"    ✓ 描述符计算完成")
            except Exception as e:
                print(f"    ✗ 计算错误: {e}")
                invalid_compounds.append((name, smiles))
                for i in range(len(descriptor_names)):
                    results[f'Md_{i}'].append(np.nan)
                valid_names.append(name)

    print(f"\n处理完成。无效化合物: {len(invalid_compounds)}")

    # 创建DataFrame
    descriptor_df = pd.DataFrame(results)
    descriptor_df.insert(0, 'Name', valid_names)

    # 将NaN值填充为0
    descriptor_cols = [col for col in descriptor_df.columns if col != 'Name']
    nan_count = descriptor_df[descriptor_cols].isna().sum().sum()
    descriptor_df[descriptor_cols] = descriptor_df[descriptor_cols].fillna(0)
    print(f"✓ 填充了 {nan_count} 个NaN值为0")

    # 创建描述符信息DataFrame
    descriptor_info = pd.DataFrame({
        'Descriptor_Code': [f'Md_{i}' for i in range(len(descriptor_names))],
        'Descriptor_Name': descriptor_names,
        'Description': [str(d.__class__.__doc__).strip()[:200] + '...' if d.__class__.__doc__ else 'N/A'
                        for d in calc.descriptors]
    })

    return descriptor_df, descriptor_info, invalid_compounds


def extract_selected_descriptors(descriptor_df, selected_bits):
    """
    提取指定的35个描述符

    Parameters:
    -----------
    descriptor_df : pandas.DataFrame
        包含所有描述符的DataFrame
    selected_bits : list
        指定的描述符索引列表

    Returns:
    --------
    filtered_df : pandas.DataFrame
        只包含指定描述符的DataFrame
    """
    selected_descriptors = [f'Md_{bit}' for bit in selected_bits]

    # 检查哪些描述符可用
    available = []
    missing = []

    for desc in selected_descriptors:
        if desc in descriptor_df.columns:
            available.append(desc)
        else:
            missing.append(desc)

    print(f"需要的描述符: {len(selected_descriptors)}")
    print(f"找到的描述符: {len(available)}")

    if missing:
        print(f"缺失的描述符: {len(missing)}")
        for desc in missing[:10]:
            print(f"  - {desc}")
        if len(missing) > 10:
            print(f"  ... 还有 {len(missing) - 10} 个")

    # 创建筛选后的DataFrame
    filtered_df = pd.DataFrame()
    filtered_df['Name'] = descriptor_df['Name']

    for desc in selected_descriptors:
        if desc in descriptor_df.columns:
            filtered_df[desc] = descriptor_df[desc]
        else:
            print(f"警告: 添加缺失列 {desc} 并填充0")
            filtered_df[desc] = 0.0

    return filtered_df, available, missing


def main():
    """主函数"""
    print("=" * 70)
    print("候选分子 Mordred 描述符计算 (DFT-31g)")
    print("=" * 70)

    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        print("请确保 candidate_molecules-31g.xlsx 在当前目录")
        return None

    # 读取候选分子文件
    print(f"\n读取输入文件: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    print(f"✓ 成功加载 {len(df)} 个候选分子")
    print(f"  列名: {list(df.columns)}")

    # 获取化合物名称和SMILES
    if 'No.' in df.columns:
        compound_names = df['No.'].astype(str).tolist()
        print(f"使用 'No.' 列作为化合物名称")
    elif 'Name' in df.columns:
        compound_names = df['Name'].tolist()
        print(f"使用 'Name' 列作为化合物名称")
    else:
        compound_names = [f'Candidate_{i + 1}' for i in range(len(df))]
        print(f"创建化合物名称")

    # 获取SMILES
    if 'Structure' in df.columns:
        smiles_col = 'Structure'
    elif 'SMILES' in df.columns:
        smiles_col = 'SMILES'
    else:
        print("错误: 找不到 'Structure' 或 'SMILES' 列")
        return None

    smiles_list = df[smiles_col].tolist()
    print(f"使用 '{smiles_col}' 列作为SMILES")

    # 显示候选分子信息
    print("\n候选分子列表:")
    for i, (name, smiles) in enumerate(zip(compound_names, smiles_list)):
        smiles_short = smiles[:60] + "..." if len(str(smiles)) > 60 else smiles
        print(f"  {i + 1}. {name}: {smiles_short}")

    # 生成Mordred描述符
    print("\n" + "=" * 70)
    print("生成 Mordred 描述符")
    print("=" * 70)

    descriptor_df, descriptor_info, invalid_compounds = calculate_mordred_descriptors(
        smiles_list, compound_names
    )

    if len(invalid_compounds) > 0:
        print(f"\n⚠️ 无效SMILES ({len(invalid_compounds)}):")
        for name, smiles in invalid_compounds:
            print(f"  - {name}: {smiles}")

    # 合并原始数据与描述符
    print("\n合并描述符与原始数据...")
    result_df = df.copy()

    # 添加描述符列
    for col in descriptor_df.columns:
        if col != 'Name':
            result_df[col] = descriptor_df[col]

    # 保存完整描述符结果
    print(f"\n保存完整描述符结果到: {OUTPUT_FILE}")
    result_df.to_excel(OUTPUT_FILE, index=False)
    print(f"✓ 保存了 {len(descriptor_df.columns) - 1} 个描述符")

    # 提取指定的35个描述符
    print("\n" + "=" * 70)
    print("提取指定的35个描述符")
    print("=" * 70)

    selected_df, available, missing = extract_selected_descriptors(descriptor_df, SELECTED_BITS)

    # 保存35个描述符结果
    print(f"\n保存35个描述符结果到: {SELECTED_OUTPUT_FILE}")
    selected_df.to_excel(SELECTED_OUTPUT_FILE, index=False)
    print(f"✓ 保存了 {len(selected_df.columns) - 1} 个指定描述符")

    # 保存描述符信息
    print(f"\n保存描述符信息到: {DESCRIPTOR_INFO_FILE}")
    descriptor_info.to_excel(DESCRIPTOR_INFO_FILE, index=False)
    print(f"✓ 描述符信息保存成功")

    # 生成汇总报告
    summary_file = 'candidate_descriptor_summary-31g.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("候选分子 MORDRED 描述符计算汇总 (DFT-31g)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"输入文件: {INPUT_FILE}\n")
        f.write(f"总候选分子数: {len(df)}\n")
        f.write(f"无效SMILES: {len(invalid_compounds)}\n")
        f.write(f"总Mordred描述符数: {descriptor_df.shape[1] - 1}\n")
        f.write(f"指定的35个描述符: {len(SELECTED_BITS)}\n")
        f.write(f"找到的描述符: {len(available)}\n")
        f.write(f"缺失的描述符: {len(missing)}\n\n")

        f.write("输出文件:\n")
        f.write(f"  1. {OUTPUT_FILE} - 完整描述符数据\n")
        f.write(f"  2. {SELECTED_OUTPUT_FILE} - 35个指定描述符\n")
        f.write(f"  3. {DESCRIPTOR_INFO_FILE} - 描述符信息\n")
        f.write(f"  4. {summary_file} - 本汇总文件\n\n")

        if invalid_compounds:
            f.write("无效化合物:\n")
            f.write("-" * 40 + "\n")
            for name, smiles in invalid_compounds:
                f.write(f"  {name}: {smiles[:80]}...\n")
            f.write("\n")

        if missing:
            f.write("缺失的指定描述符:\n")
            f.write("-" * 40 + "\n")
            for desc in missing[:20]:
                f.write(f"  {desc}\n")
            if len(missing) > 20:
                f.write(f"  ... 还有 {len(missing) - 20} 个\n")

    # 打印汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"总候选分子数: {len(df)}")
    print(f"无效SMILES: {len(invalid_compounds)}")
    print(f"总Mordred描述符数: {descriptor_df.shape[1] - 1}")
    print(f"指定的35个描述符: {len(SELECTED_BITS)}")
    print(f"找到的描述符: {len(available)}")
    print(f"缺失的描述符: {len(missing)}")

    print(f"\n输出文件保存到当前目录:")
    print(f"  1. {OUTPUT_FILE}")
    print(f"  2. {SELECTED_OUTPUT_FILE}")
    print(f"  3. {DESCRIPTOR_INFO_FILE}")
    print(f"  4. {summary_file}")

    # 显示35个描述符的预览
    print("\n" + "=" * 70)
    print("35个指定描述符预览 (前5行，前5列)")
    print("=" * 70)

    # 选择要显示的列
    display_cols = ['Name'] + [f'Md_{bit}' for bit in SELECTED_BITS[:5]]
    available_display = [col for col in display_cols if col in selected_df.columns]

    if available_display:
        print(selected_df[available_display].head().to_string(index=False))
    else:
        print("无法显示预览")

    print("\n" + "=" * 70)
    print("脚本执行完成!")
    print("=" * 70)

    return selected_df


def check_descriptor_values(df):
    """检查描述符值是否正常"""
    print("\n" + "=" * 70)
    print("描述符值检查")
    print("=" * 70)

    # 获取所有描述符列
    desc_cols = [col for col in df.columns if col.startswith('Md_')]

    if len(desc_cols) == 0:
        print("没有找到描述符列")
        return

    # 检查每列是否有变化
    print(f"检查 {len(desc_cols)} 个描述符列...")

    constant_cols = []
    for col in desc_cols:
        if df[col].std() == 0 or df[col].nunique() == 1:
            constant_cols.append(col)

    if constant_cols:
        print(f"⚠️ 发现 {len(constant_cols)} 个常数列（所有值相同）:")
        for col in constant_cols[:10]:
            print(f"  - {col}: {df[col].iloc[0]}")
        if len(constant_cols) > 10:
            print(f"  ... 还有 {len(constant_cols) - 10} 个")
    else:
        print("✓ 所有描述符列都有变化，计算正常")

    # 显示描述符值范围
    print(f"\n描述符值范围:")
    print(f"  最小值: {df[desc_cols].min().min():.6f}")
    print(f"  最大值: {df[desc_cols].max().max():.6f}")
    print(f"  均值: {df[desc_cols].mean().mean():.6f}")
    print(f"  标准差: {df[desc_cols].std().mean():.6f}")


if __name__ == "__main__":
    # 检查依赖包
    try:
        from rdkit import Chem
        from mordred import Calculator, descriptors

        print("✓ Mordred 和 RDKit 已安装")
    except ImportError as e:
        print("❌ 错误: 未安装所需包!")
        print("请安装 Mordred 和 RDKit:")
        print("  pip install mordred rdkit-pypi")
        import sys

        sys.exit(1)

    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 输入文件 '{INPUT_FILE}' 不存在!")
        print("请确保 candidate_molecules-31g.xlsx 在当前目录")
        import sys

        sys.exit(1)

    # 运行主函数
    result = main()

    # 检查描述符值
    if result is not None:
        check_descriptor_values(result)