#!/usr/bin/env python3
"""
Step 4: Verify Selected Candidates - Table Output
验证10个候选分子并输出表格
(Updated for new_output/more/ directory and Step 3 changes)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors
from tabulate import tabulate
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
BASE_DIR = Path(__file__).parent

# 修改为 new_output/more（与 Step 3 保持一致）
OUTPUT_DIR = BASE_DIR / 'new_output' / 'more'
RESULT_DIR = OUTPUT_DIR / 'results'
STEPWISE_DIR = OUTPUT_DIR / 'stepwise_results'
DESCRIPTOR_DIR = OUTPUT_DIR / 'descriptors'
LOG_DIR = OUTPUT_DIR / 'logs'

# 确保目录存在
for dir_path in [RESULT_DIR, STEPWISE_DIR, DESCRIPTOR_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 您选择的10个分子
SELECTED_IDS = [
    'NPBS016549',
    'NPBS074565',
    'NPBS135282',
    'NPBS135489',
    'NPBS138346',
    'NPBS186088',
    'NPBS188867',
    'NPBS190307',
    'NPBS203027',
    'NPBS213314'
]

# ==================== 显色子结构完整列表（用于显示） ====================

CHROMOPHORE_CATEGORIES = {
    'Anthraquinone': '醌类',
    'Quinone': '醌类',
    'Flavonoid': '类黄酮',
    'Anthocyanin': '类黄酮',
    'Carotenoid': '类胡萝卜素',
    'Xanthophyll': '类胡萝卜素',
    'Proanthocyanidin': '单宁类',
    'Galloyl': '单宁类',
    'Ellagic_Acid': '单宁类',
    'Catechin': '单宁类',
    'Tannin': '单宁类',
    'Caffeic_Acid': '咖啡类',
    'Chlorogenic_Acid': '咖啡类',
    'Caffeine': '咖啡类',
    'Trigonelline': '咖啡类',
    'Betalain': '其他',
    'Curcuminoid': '其他',
    'Indigo': '其他',
    'Coumarin': '其他',
    'Porphyrin': '其他'
}

# 关键生物危害说明
CRITICAL_BIOHAZARD_NAMES = {
    'Mutagenic_alert': '对硝基苯酚结构',
    'Carcinogenic_alert': '叠氮/三氮烯结构',
    'Reproductive_toxin': '含氮杂环片段'
}


# ==================== 日志类 ====================

class Logger:
    """同时输出到控制台和日志文件"""

    def __init__(self, log_dir, script_name="step04"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{script_name}_{timestamp}.log"

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Step 4 运行日志 - 验证10个候选分子\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message, print_to_console=True):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")

        if print_to_console:
            print(message)

    def log_section(self, title, print_to_console=True):
        timestamp = datetime.now().strftime("%H:%M:%S")
        separator = "=" * 60
        log_line = f"\n[{timestamp}] {separator}\n[{timestamp}] {title}\n[{timestamp}] {separator}"

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")

        if print_to_console:
            print(f"\n{separator}")
            print(title)
            print(separator)

    def log_table(self, title, headers, data, print_to_console=True):
        """记录表格到日志"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{title}\n")
            f.write("-" * 40 + "\n")
            f.write(" | ".join(headers) + "\n")
            f.write("-" * 40 + "\n")
            for row in data[:20]:  # 只记录前20行
                f.write(" | ".join(str(x) for x in row) + "\n")
            if len(data) > 20:
                f.write(f"... 共 {len(data)} 行\n")
            f.write("-" * 40 + "\n")

    def close(self):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        print(f"\n✅ 日志已保存: {self.log_file}")


# ==================== 结构验证函数 ====================

def validate_smiles(smiles):
    """验证SMILES并返回分子信息"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'valid': False, 'error': 'Invalid SMILES'}

        mw = Descriptors.MolWt(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        n_rings = mol.GetRingInfo().NumRings()

        atoms = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atoms[symbol] = atoms.get(symbol, 0) + 1

        return {
            'valid': True,
            'mw': round(mw, 2),
            'heavy_atoms': heavy_atoms,
            'n_rings': n_rings,
            'atoms': atoms
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def get_chromophore_category(chrom_name):
    """获取显色子结构类别"""
    return CHROMOPHORE_CATEGORIES.get(chrom_name, '未分类')


def parse_biohazard_alerts(critical_str, info_str):
    """解析生物安全警报为可读格式"""
    critical_parts = []
    info_parts = []

    if isinstance(critical_str, str) and critical_str != 'None':
        for alert in critical_str.split('; '):
            if alert in CRITICAL_BIOHAZARD_NAMES:
                critical_parts.append(f"⚠️ {CRITICAL_BIOHAZARD_NAMES[alert]}")
            else:
                critical_parts.append(f"⚠️ {alert}")

    if isinstance(info_str, str) and info_str != 'None':
        for alert in info_str.split('; '):
            info_parts.append(f"ℹ️ {alert}")

    if not critical_parts and not info_parts:
        return '✅ Safe'

    result = []
    if critical_parts:
        result.append('关键: ' + ', '.join(critical_parts))
    if info_parts:
        result.append('信息: ' + ', '.join(info_parts))

    return '; '.join(result)


# ==================== 主函数 ====================

def main():
    # 初始化日志
    logger = Logger(LOG_DIR, "step04")

    logger.log_section("Step 4: 验证10个候选分子")
    logger.log(f"(Using {OUTPUT_DIR} directory)")
    logger.log(f"输出目录: {OUTPUT_DIR}")
    logger.log(f"日志目录: {LOG_DIR}")

    # ============================================================
    # 1. 加载数据（优先使用 Step 3 的最终结果）
    # ============================================================
    logger.log_section("1. 加载数据")

    # 按优先级顺序尝试加载
    possible_files = [
        RESULT_DIR / 'final_selected_candidates.csv',  # Step 3 最终结果
        RESULT_DIR / 'final_filtered_candidates.csv',  # Step 3 综合筛选
        RESULT_DIR / 'candidates_has_chromophore.csv',  # Step 3 含显色子结构
        STEPWISE_DIR / 'step03_06_final_complete.csv',  # Step 3 分步结果
        RESULT_DIR / 'top_100_final_candidates.csv',  # Step 3 Top 100
        BASE_DIR / 'new_output' / 'results' / 'screening_results_200_700nm.csv',  # 上级目录
    ]

    df = None
    for file_path in possible_files:
        if file_path.exists():
            df = pd.read_csv(file_path)
            logger.log(f"  ✅ 从 {file_path.name} 加载: {len(df):,} 个分子")
            break

    if df is None:
        logger.log("❌ 找不到数据文件！请先运行 Step 3")
        logger.log(f"   搜索路径: {RESULT_DIR}")
        logger.close()
        return

    # ============================================================
    # 2. 提取选择的分子
    # ============================================================
    logger.log_section(f"2. 提取 {len(SELECTED_IDS)} 个选择的分子")

    selected_df = df[df['Identifier'].isin(SELECTED_IDS)].copy()
    logger.log(f"  ✅ 找到 {len(selected_df)} 个匹配的分子")

    # 按选择顺序排序
    selected_df['Order'] = selected_df['Identifier'].map({id_: i for i, id_ in enumerate(SELECTED_IDS)})
    selected_df = selected_df.sort_values('Order').reset_index(drop=True)

    # 检查缺失的ID
    missing_ids = set(SELECTED_IDS) - set(selected_df['Identifier'].tolist())
    if missing_ids:
        logger.log(f"  ⚠️ 未找到的ID: {missing_ids}")

    # 如果有缺失的ID，补充空行
    if len(selected_df) < len(SELECTED_IDS):
        logger.log(f"  ⚠️ 只有 {len(selected_df)} 个分子被找到，将用空行补充")

    # ============================================================
    # 3. 生成表格数据
    # ============================================================
    logger.log_section("📊 表格1: 10个候选分子筛选信息")

    table1_data = []
    for idx, row in selected_df.iterrows():
        smiles = row.get('SMILES', '')
        mol_info = validate_smiles(smiles) if smiles else {'valid': False}

        # 获取显色子结构
        chrom = row.get('Chromophore_Matches', '')
        if isinstance(chrom, str) and chrom == 'None':
            chrom = '-'
        elif isinstance(chrom, str) and len(chrom) > 25:
            chrom = chrom[:25] + '...'

        # 获取生物安全（关键+信息）
        critical = row.get('Critical_Biohazard_Alerts', '')
        info = row.get('Info_Biohazard_Alerts', '')
        bio_display = parse_biohazard_alerts(critical, info)
        if len(bio_display) > 30:
            bio_display = bio_display[:30] + '...'

        # 判断是否通过综合筛选
        passed_final = (
                row.get('Has_Chromophore', False) == True and
                row.get('Has_Critical_Biohazard', True) == False
        )

        # 波长是否在200-700nm
        wavelength = row.get('Predicted_bo', 0)
        in_wavelength = 200 <= wavelength <= 700

        # SAscore参考
        sa_ref = row.get('SA_Score_Reference', 0)
        sa_display = f"{sa_ref:.2f}" if sa_ref != 0 else '-'

        table1_data.append([
            idx + 1,
            row.get('Identifier', ''),
            f"{wavelength:.2f}",
            '✅' if in_wavelength else '❌',
            '✅' if row.get('Has_Chromophore', False) else '❌',
            '✅' if not row.get('Has_Critical_Biohazard', True) else '❌',
            sa_display,
            '✅' if passed_final else '❌',
            '✅' if mol_info.get('valid', False) else '❌'
        ])

    # 补充缺失的ID
    if len(table1_data) < len(SELECTED_IDS):
        for i in range(len(table1_data), len(SELECTED_IDS)):
            table1_data.append([
                i + 1,
                SELECTED_IDS[i],
                '-',
                '-',
                '-',
                '-',
                '-',
                '-',
                '❌'
            ])

    headers1 = ['#', 'ID', 'λmax', '200-700', 'Chrom', 'Safe', 'SA(ref)', 'Passed', 'Valid']
    print(tabulate(table1_data, headers=headers1, tablefmt='grid', stralign='left'))
    logger.log_table("表格1: 候选分子筛选信息", headers1, table1_data)

    # ============================================================
    # 4. 表格2: 显色子结构详情
    # ============================================================
    logger.log_section("📊 表格2: 显色子结构详情")

    table2_data = []
    for idx, row in selected_df.iterrows():
        chrom = row.get('Chromophore_Matches', '')
        if isinstance(chrom, str) and chrom != 'None' and chrom != '':
            chrom_list = chrom.split('; ')
            chrom_details = []
            for c in chrom_list:
                category = get_chromophore_category(c)
                chrom_details.append(f"{c}({category})")
            chrom_display = ', '.join(chrom_details)
        else:
            chrom_display = '-'

        table2_data.append([
            idx + 1,
            row.get('Identifier', ''),
            chrom_display[:50] + '...' if len(chrom_display) > 50 else chrom_display,
        ])

    if len(table2_data) < len(SELECTED_IDS):
        for i in range(len(table2_data), len(SELECTED_IDS)):
            table2_data.append([
                i + 1,
                SELECTED_IDS[i],
                '-'
            ])

    headers2 = ['#', 'ID', '显色子结构 (类别)']
    print(tabulate(table2_data, headers=headers2, tablefmt='grid', stralign='left'))
    logger.log_table("表格2: 显色子结构详情", headers2, table2_data)

    # ============================================================
    # 5. 表格3: 生物安全详情
    # ============================================================
    logger.log_section("📊 表格3: 生物安全详情")
    logger.log("  关键警报: 对硝基苯酚、叠氮/三氮烯、含氮杂环片段")
    logger.log("  信息警报: 迈克尔受体、皮肤致敏剂、PAINS")

    table3_data = []
    for idx, row in selected_df.iterrows():
        critical = row.get('Critical_Biohazard_Alerts', 'None')
        info = row.get('Info_Biohazard_Alerts', 'None')

        # 解析关键警报
        critical_display = []
        if isinstance(critical, str) and critical != 'None':
            for alert in critical.split('; '):
                if alert in CRITICAL_BIOHAZARD_NAMES:
                    critical_display.append(CRITICAL_BIOHAZARD_NAMES[alert])
                else:
                    critical_display.append(alert)
        critical_str = ', '.join(critical_display) if critical_display else '无'

        # 解析信息警报
        info_display = []
        if isinstance(info, str) and info != 'None':
            info_display = info.split('; ')
        info_str = ', '.join(info_display) if info_display else '无'

        has_critical = row.get('Has_Critical_Biohazard', False)
        status = '❌ 有关键警报' if has_critical else '✅ 安全'

        table3_data.append([
            idx + 1,
            row.get('Identifier', ''),
            status,
            critical_str[:40] + '...' if len(critical_str) > 40 else critical_str,
            info_str[:40] + '...' if len(info_str) > 40 else info_str,
        ])

    if len(table3_data) < len(SELECTED_IDS):
        for i in range(len(table3_data), len(SELECTED_IDS)):
            table3_data.append([
                i + 1,
                SELECTED_IDS[i],
                '-',
                '-',
                '-'
            ])

    headers3 = ['#', 'ID', '状态', '关键警报', '信息警报']
    print(tabulate(table3_data, headers=headers3, tablefmt='grid', stralign='left'))
    logger.log_table("表格3: 生物安全详情", headers3, table3_data)

    # ============================================================
    # 6. 表格4: 综合评分详情
    # ============================================================
    logger.log_section("📊 表格4: 综合评分详情")
    logger.log("  权重: 波长(二值) 35% + 相似度 25% + 显色子 20% + 生物安全 20%")

    table4_data = []
    for idx, row in selected_df.iterrows():
        table4_data.append([
            idx + 1,
            row.get('Identifier', ''),
            f"{row.get('Combined_Score', 0):.4f}",
            f"{row.get('Wavelength_Score', 0):.1f}",
            f"{row.get('Similarity_Score', 0):.3f}",
            f"{row.get('Chromophore_Score', 0):.1f}",
            f"{row.get('Biosafety_Score', 0):.1f}",
        ])

    if len(table4_data) < len(SELECTED_IDS):
        for i in range(len(table4_data), len(SELECTED_IDS)):
            table4_data.append([
                i + 1,
                SELECTED_IDS[i],
                '-',
                '-',
                '-',
                '-',
                '-'
            ])

    headers4 = ['#', 'ID', '综合分', '波长', '相似度', '显色子', '生物安全']
    print(tabulate(table4_data, headers=headers4, tablefmt='grid', stralign='left'))
    logger.log_table("表格4: 综合评分详情", headers4, table4_data)

    # ============================================================
    # 7. 表格5: 结构验证信息
    # ============================================================
    logger.log_section("📊 表格5: 分子结构验证")

    table5_data = []
    for idx, row in selected_df.iterrows():
        smiles = row.get('SMILES', '')
        mol_info = validate_smiles(smiles) if smiles else {'valid': False}

        if mol_info.get('valid', False):
            atoms_str = '; '.join([f"{k}:{v}" for k, v in mol_info['atoms'].items()])
            table5_data.append([
                idx + 1,
                row.get('Identifier', ''),
                mol_info.get('mw', 0),
                mol_info.get('heavy_atoms', 0),
                mol_info.get('n_rings', 0),
                atoms_str[:40] + '...' if len(atoms_str) > 40 else atoms_str,
            ])
        else:
            table5_data.append([
                idx + 1,
                row.get('Identifier', ''),
                '-',
                '-',
                '-',
                'Invalid'
            ])

    if len(table5_data) < len(SELECTED_IDS):
        for i in range(len(table5_data), len(SELECTED_IDS)):
            table5_data.append([
                i + 1,
                SELECTED_IDS[i],
                '-',
                '-',
                '-',
                'Not Found'
            ])

    headers5 = ['#', 'ID', 'MW', 'Heavy Atoms', 'Rings', 'Elements']
    print(tabulate(table5_data, headers=headers5, tablefmt='grid', stralign='left'))
    logger.log_table("表格5: 分子结构验证", headers5, table5_data)

    # ============================================================
    # 8. 筛选状态汇总
    # ============================================================
    logger.log_section("📊 表格6: 筛选状态汇总")

    in_candidates = len(selected_df)
    has_critical = selected_df[selected_df['Has_Critical_Biohazard'] == True].shape[
        0] if 'Has_Critical_Biohazard' in selected_df.columns else 0
    has_chrom = selected_df[selected_df['Has_Chromophore'] == True].shape[
        0] if 'Has_Chromophore' in selected_df.columns else 0
    in_wavelength = selected_df[(selected_df['Predicted_bo'] >= 200) & (selected_df['Predicted_bo'] <= 700)].shape[
        0] if 'Predicted_bo' in selected_df.columns else 0

    # 显色子结构统计
    chrom_counts = {}
    chrom_by_category = {}
    for val in selected_df['Chromophore_Matches']:
        if isinstance(val, str) and val not in ['None', 'Unknown', '']:
            for chrom in val.split('; '):
                chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
                category = get_chromophore_category(chrom)
                if category not in chrom_by_category:
                    chrom_by_category[category] = 0
                chrom_by_category[category] += 1

    table6_data = [
        ['总检查数', len(SELECTED_IDS)],
        ['在候选列表中', f"{in_candidates}/{len(SELECTED_IDS)}"],
        ['200-700nm范围内', f"{in_wavelength}/{in_candidates}" if in_candidates > 0 else "N/A"],
        ['含显色子结构', f"{has_chrom}/{in_candidates}" if in_candidates > 0 else "N/A"],
        ['无关键生物危害', f"{in_candidates - has_critical}/{in_candidates}" if in_candidates > 0 else "N/A"],
        ['', ''],
        ['显色子结构分布 (按类别)', ''],
    ]

    for category, count in sorted(chrom_by_category.items(), key=lambda x: x[1], reverse=True):
        table6_data.append([f'  {category}', count])

    table6_data.append(['', ''])
    table6_data.append(['显色子结构分布 (详细)', ''])

    for chrom, count in sorted(chrom_counts.items(), key=lambda x: x[1], reverse=True):
        category = get_chromophore_category(chrom)
        table6_data.append([f'  {chrom} ({category})', count])

    headers6 = ['指标', '数值']
    print(tabulate(table6_data, headers=headers6, tablefmt='grid', stralign='left'))
    logger.log_table("表格6: 筛选状态汇总", headers6, table6_data)

    # ============================================================
    # 9. 保存到文件
    # ============================================================
    logger.log_section("💾 保存结果")

    if len(selected_df) > 0:
        output_df = selected_df.copy()

        # 选择输出列
        out_cols = ['Identifier', 'Predicted_bo', 'Has_Chromophore', 'Chromophore_Matches',
                    'Has_Critical_Biohazard', 'Critical_Biohazard_Alerts', 'Info_Biohazard_Alerts',
                    'SA_Score_Reference', 'Nearest_Neighbor_Similarity',
                    'Wavelength_Score', 'Similarity_Score', 'Chromophore_Score',
                    'Biosafety_Score', 'Combined_Score', 'SMILES']

        out_cols = [col for col in out_cols if col in output_df.columns]

        output_df = output_df[out_cols].copy()

        # 格式化数值列
        if 'Predicted_bo' in output_df.columns:
            output_df['Predicted_bo'] = output_df['Predicted_bo'].round(2)
        if 'SA_Score_Reference' in output_df.columns:
            output_df['SA_Score_Reference'] = output_df['SA_Score_Reference'].round(2)
        if 'Nearest_Neighbor_Similarity' in output_df.columns:
            output_df['Nearest_Neighbor_Similarity'] = output_df['Nearest_Neighbor_Similarity'].round(4)
        if 'Combined_Score' in output_df.columns:
            output_df['Combined_Score'] = output_df['Combined_Score'].round(4)

        # 保存CSV
        csv_path = RESULT_DIR / 'selected_10_candidates_verified.csv'
        output_df.to_csv(csv_path, index=False)
        logger.log(f"  ✅ CSV: {csv_path}")

        # 保存Excel
        try:
            excel_path = RESULT_DIR / 'selected_10_candidates_verified.xlsx'
            output_df.to_excel(excel_path, index=False)
            logger.log(f"  ✅ Excel: {excel_path}")
        except Exception as e:
            logger.log(f"  ⚠️ Excel导出失败 (需要安装openpyxl): pip install openpyxl")
    else:
        logger.log("  ⚠️ 没有找到任何选择的分子，无法保存")

    # ============================================================
    # 10. 最终推荐
    # ============================================================
    logger.log_section("💡 最终推荐")

    if len(selected_df) == 0:
        logger.log("⚠️ 没有找到任何选择的分子，无法生成推荐")
    else:
        # 筛选安全且通过综合筛选的
        safe_passed = selected_df[
            (selected_df['Has_Chromophore'] == True) &
            (selected_df['Has_Critical_Biohazard'] == False)
            ]

        if len(safe_passed) > 0:
            logger.log("\n✅ 最优先推荐 (含显色子结构 + 无关键生物危害):")
            for idx, row in safe_passed.iterrows():
                sa_ref = row.get('SA_Score_Reference', 0)
                logger.log(f"  {row['Identifier']}: λmax={row['Predicted_bo']:.2f} nm, SA(参考)={sa_ref:.2f}")

        # 综合评分最高
        if 'Combined_Score' in selected_df.columns:
            top_combined = selected_df.nlargest(3, 'Combined_Score')
            logger.log("\n⭐ 综合评分最高 (Top 3):")
            for idx, row in top_combined.iterrows():
                status = "✅ Safe" if row.get('Has_Critical_Biohazard', True) == False else "⚠️ Has Alert"
                logger.log(
                    f"  {row['Identifier']}: 综合分={row['Combined_Score']:.4f}, λmax={row['Predicted_bo']:.2f} nm, {status}")

        # 波长最好的
        if 'Predicted_bo' in selected_df.columns:
            top_wavelength = selected_df.nlargest(3, 'Predicted_bo')
            logger.log("\n📈 波长最好 (Top 3):")
            for idx, row in top_wavelength.iterrows():
                status = "✅ Safe" if row.get('Has_Critical_Biohazard', True) == False else "⚠️ Has Alert"
                logger.log(f"  {row['Identifier']}: λmax={row['Predicted_bo']:.2f} nm, {status}")

    # ============================================================
    # 11. 完成
    # ============================================================
    logger.log("\n" + "=" * 80)
    logger.log("✅ 验证完成！")
    logger.log("=" * 80)

    # 关闭日志
    logger.close()


if __name__ == "__main__":
    main()