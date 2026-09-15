#!/usr/bin/env python3
"""
Step 1: Calculate Mordred descriptors AND MACCS fingerprints for NPBS dataset
WITH STANDARDIZATION AND MACCS FINGERPRINTS (NO DEDUPLICATION)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from mordred import Calculator, descriptors
import time
from pathlib import Path
from tqdm import tqdm
import pickle

# Suppress RDKit warnings
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ==================== Configuration ====================
BASE_DIR = Path(__file__).parent

MODEL_DIR = BASE_DIR.parent / '03_modeling' / '03_modeling' / 'DNN_model' / 'dnn_model_35_result' / 'dnn_model_result'

OUTPUT_DIR = BASE_DIR / 'new_output'
DESCRIPTOR_DIR = OUTPUT_DIR / 'descriptors'
RESULT_DIR = OUTPUT_DIR / 'results'
LOG_DIR = OUTPUT_DIR / 'logs'
FINGERPRINT_DIR = OUTPUT_DIR / 'fingerprints'
STEPWISE_DIR = OUTPUT_DIR / 'stepwise_results'

for dir_path in [OUTPUT_DIR, DESCRIPTOR_DIR, RESULT_DIR, LOG_DIR, FINGERPRINT_DIR, STEPWISE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 35描述符索引 (1-based)
SELECTED_DESCRIPTORS = [
    906, 1336, 334, 1310, 1348, 1406, 1437, 345, 218, 230,
    449, 510, 299, 3, 1547, 566, 1418, 1069, 232, 264,
    1076, 1077, 322, 1313, 1294, 1064, 1329, 1146, 1477, 1359,
    805, 1394, 1358, 1573, 485
]

BATCH_SIZE = 100
INPUT_FILE = 'NPBS_Data.csv'
DESCRIPTOR_FILE = 'NPBS_Data_with_descriptors.csv'
PROCESSED_FILE = 'NPBS_Data_processed.csv'
INVALID_LOG_FILE = 'invalid_smiles_log.xlsx'
PREPROCESS_REPORT = 'preprocessing_report.csv'
MACCS_FINGERPRINT_RDKIT = 'maccs_fingerprints_rdkit.pkl'


# ==================== 数据预处理函数 ====================

def standardize_smiles(smiles):
    """
    SMILES标准化 - 保留立体化学信息
    仅标准化，不去重
    """
    try:
        if not isinstance(smiles, str) or pd.isna(smiles):
            return None

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        # 取最大片段（去除盐/溶剂）
        if '.' in smiles.strip():
            fragments = Chem.GetMolFrags(mol, asMols=True)
            if fragments:
                mol = max(fragments, key=lambda m: m.GetNumAtoms())

        Chem.SanitizeMol(mol)
        # 保留立体化学信息
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        return None


def identify_metal(smiles):
    """识别金属元素 - 返回(是否含金属, 金属元素列表)"""
    if not isinstance(smiles, str):
        return False, []

    metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
              'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
              'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Hf',
              'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']

    found_metals = []
    for metal in metals:
        if f'[{metal}' in smiles or f'[{metal.lower()}' in smiles:
            found_metals.append(metal)

    return len(found_metals) > 0, found_metals


# ==================== MACCS指纹计算 ====================

def calculate_maccs_fingerprint(smiles):
    """计算单个分子的MACCS指纹"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.tolist()
    except:
        return None


def calculate_maccs_fingerprints_batch(smiles_list):
    """批量计算MACCS指纹"""
    fingerprints = []
    valid_indices = []

    for i, smiles in enumerate(tqdm(smiles_list, desc="Calculating MACCS fingerprints")):
        fp = calculate_maccs_fingerprint(smiles)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)

    return fingerprints, valid_indices


# ==================== Mordred描述符计算函数 ====================

def smiles_to_mol(smiles):
    """Convert SMILES to RDKit molecule"""
    try:
        if isinstance(smiles, str):
            smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
        return mol
    except:
        return None


def setup_calculator():
    """Setup Mordred calculator with selected descriptors"""
    selected_indices = [idx - 1 for idx in SELECTED_DESCRIPTORS]
    selected_names = [f'Md_{idx}' for idx in SELECTED_DESCRIPTORS]

    full_calc = Calculator(descriptors, ignore_3D=True)
    selected_desc_list = [full_calc.descriptors[idx] for idx in selected_indices if idx < len(full_calc.descriptors)]

    calc = Calculator(selected_desc_list, ignore_3D=True)
    return calc, selected_names


def calc_for_molecule(mol, calculator):
    """Calculate descriptors for a single molecule"""
    try:
        values = calculator(mol)
        return [float(v) if v is not None and not isinstance(v, str) else np.nan for v in values]
    except:
        return None


# ==================== 预处理主函数（不去重） ====================

def preprocess_no_dedup(df):
    """
    预处理 - 标准化 + 金属识别，但不去重
    """
    print("\n" + "=" * 50)
    print("Data Preprocessing: Standardization ONLY")
    print("(NO DEDUPLICATION - all molecules retained)")
    print("=" * 50)

    original_count = len(df)
    print(f"原始分子数: {original_count}")

    print("\n1. 标准化SMILES (保留立体化学)...")
    df['Canonical_SMILES'] = df['SMILES'].apply(standardize_smiles)

    invalid_mask = df['Canonical_SMILES'].isna()
    invalid_count = invalid_mask.sum()
    print(f"   标准化失败: {invalid_count} (已移除)")
    df = df[~invalid_mask].copy()

    print("\n2. 识别金属元素...")
    df['Has_Metal'], df['Metal_Elements'] = zip(*df['Canonical_SMILES'].apply(identify_metal))
    metal_count = df['Has_Metal'].sum()
    print(f"   含金属分子: {metal_count}")
    print(f"   非金属分子: {len(df) - metal_count}")

    print("\n3. ⚠️ 注意: 未进行去重，所有有效分子均保留")

    # 输出中间步骤
    print("\n4. 输出中间结果...")

    all_molecules_df = df[['Identifier', 'Original_SMILES', 'Canonical_SMILES', 'Has_Metal', 'Metal_Elements']].copy()
    all_molecules_df.to_csv(STEPWISE_DIR / 'step00_all_molecules.csv', index=False)
    print(f"   所有分子: {STEPWISE_DIR / 'step00_all_molecules.csv'}")

    non_metal_df = df[~df['Has_Metal']][['Identifier', 'Original_SMILES', 'Canonical_SMILES']].copy()
    non_metal_df.to_csv(STEPWISE_DIR / 'step01_non_metal.csv', index=False)
    print(f"   非金属分子: {STEPWISE_DIR / 'step01_non_metal.csv'}")

    metal_df = df[df['Has_Metal']][['Identifier', 'Original_SMILES', 'Canonical_SMILES', 'Metal_Elements']].copy()
    metal_df.to_csv(STEPWISE_DIR / 'step01_metal_removed.csv', index=False)
    print(f"   含金属分子: {STEPWISE_DIR / 'step01_metal_removed.csv'}")

    # 保存预处理报告
    report = {
        '原始分子数': original_count,
        '标准化失败': invalid_count,
        '有效分子数': len(df),
        '含金属分子': metal_count,
        '非金属分子': len(df) - metal_count,
        '有效率(%)': round((len(df) / original_count) * 100, 2),
        '去重': '否 (全部保留)',
        '标准化策略': '保留立体化学'
    }

    report_df = pd.DataFrame([report])
    report_path = LOG_DIR / PREPROCESS_REPORT
    report_df.to_csv(report_path, index=False)
    print(f"\n预处理报告已保存: {report_path}")
    print(report_df)

    return df


# ==================== Main函数 ====================

def main():
    print("=" * 70)
    print("Step 1: Descriptor Calculation + MACCS Fingerprints")
    print("(Standardization + MACCS - NO DEDUPLICATION)")
    print("=" * 70)

    input_path = BASE_DIR / INPUT_FILE
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found in current directory")
        return

    print(f"Model directory: {MODEL_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"\nReading {INPUT_FILE}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} molecules")
    print(f"Columns: {list(df.columns)}")

    # 保存原始SMILES以备后用
    df['Original_SMILES'] = df['SMILES']

    # ========== 数据预处理（不去重） ==========
    df = preprocess_no_dedup(df)

    identifiers = df['Identifier'].tolist()
    smiles_list = df['Canonical_SMILES'].tolist()

    # ========== 计算MACCS指纹 ==========
    print("\n" + "=" * 50)
    print("MACCS Fingerprint Calculation")
    print("=" * 50)

    fingerprints, valid_fp_indices = calculate_maccs_fingerprints_batch(smiles_list)
    print(f"MACCS指纹计算成功: {len(fingerprints)}")

    rdkit_fps = []
    for fp_list in fingerprints:
        fp = DataStructs.CreateFromBitString(''.join(str(b) for b in fp_list))
        rdkit_fps.append(fp)

    fp_rdkit_path = FINGERPRINT_DIR / MACCS_FINGERPRINT_RDKIT
    with open(fp_rdkit_path, 'wb') as f:
        pickle.dump({
            'fingerprints': rdkit_fps,
            'identifiers': [identifiers[i] for i in valid_fp_indices],
            'smiles': [smiles_list[i] for i in valid_fp_indices]
        }, f)
    print(f"MACCS指纹已保存 (RDKit格式): {fp_rdkit_path}")

    # ========== Mordred描述符计算 ==========
    print("\n" + "=" * 50)
    print("Mordred Descriptor Calculation")
    print("=" * 50)

    calculator, selected_names = setup_calculator()
    print(f"Selected {len(selected_names)} descriptors")

    all_results = {name: [] for name in selected_names}
    all_valid_ids = []
    all_valid_smiles = []
    all_invalid = []

    start_time = time.time()

    for i in tqdm(range(0, len(smiles_list), BATCH_SIZE), desc="Processing descriptors"):
        batch_end = min(i + BATCH_SIZE, len(smiles_list))
        batch_smiles = smiles_list[i:batch_end]
        batch_ids = identifiers[i:batch_end]

        batch_results = {name: [] for name in selected_names}
        batch_valid_ids = []
        batch_valid_smiles = []

        for identifier, smiles in zip(batch_ids, batch_smiles):
            mol = smiles_to_mol(smiles)
            if mol is None:
                all_invalid.append((identifier, smiles, "Invalid SMILES"))
                continue

            values = calc_for_molecule(mol, calculator)
            if values is None:
                all_invalid.append((identifier, smiles, "Calculation failed"))
                continue

            for j, name in enumerate(selected_names):
                batch_results[name].append(values[j])
            batch_valid_ids.append(identifier)
            batch_valid_smiles.append(smiles)

        for name in selected_names:
            all_results[name].extend(batch_results[name])
        all_valid_ids.extend(batch_valid_ids)
        all_valid_smiles.extend(batch_valid_smiles)

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Valid: {len(all_valid_ids):,}")
    print(f"Invalid: {len(all_invalid):,}")

    # ========== 保存描述符文件 ==========
    descriptor_df = pd.DataFrame(all_results)
    descriptor_df.insert(0, 'SMILES', all_valid_smiles)
    descriptor_df.insert(0, 'Identifier', all_valid_ids)

    descriptor_df[selected_names] = descriptor_df[selected_names].fillna(0)

    out_path = DESCRIPTOR_DIR / DESCRIPTOR_FILE
    descriptor_df.to_csv(out_path, index=False)
    print(f"Saved descriptors to: {out_path}")

    # 保存预处理后的完整数据（包含金属信息）
    processed_path = DESCRIPTOR_DIR / PROCESSED_FILE
    df.to_csv(processed_path, index=False)
    print(f"Saved processed data to: {processed_path}")

    if all_invalid:
        invalid_df = pd.DataFrame(all_invalid, columns=['Identifier', 'SMILES', 'Error'])
        invalid_df.to_excel(LOG_DIR / INVALID_LOG_FILE, index=False)
        print(f"Saved invalid log to: {LOG_DIR / INVALID_LOG_FILE}")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total molecules (after standardization): {len(df)}")
    print(f"  - Valid for descriptors: {len(descriptor_df)}")
    print(f"  - MACCS fingerprints: {len(fingerprints)}")
    print(f"  - Invalid molecules: {len(all_invalid)}")
    print(f"  - Metal-containing: {df['Has_Metal'].sum()}")
    print(f"  - Non-metal: {(~df['Has_Metal']).sum()}")

    print("\n✅ 处理策略:")
    print("   - SMILES标准化: 是 (保留立体化学)")
    print("   - 去重: 否 (所有分子全部保留)")
    print("   - MACCS指纹: 是")

    print("\nNext: python 02_run_screening.py")


if __name__ == "__main__":
    main()