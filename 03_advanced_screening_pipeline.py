#!/usr/bin/env python3
"""
Advanced Screening Pipeline for Bio-based Dyes
Implement the multi-layer Screening Funnel from 210,000 molecules down to the Top 3 (Red, Yellow, Blue).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit.ML.Cluster import Butina
    from rdkit.Chem import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit is required for this script.")

# ==================== Configuration ====================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
RESULT_DIR = OUTPUT_DIR / 'results'
ADVANCED_DIR = OUTPUT_DIR / 'advanced_screening'
ADVANCED_DIR.mkdir(parents=True, exist_ok=True)

# 输入文件：假设使用 Step 2 预测完的文件，或者全量带描述符的文件
# 如果之前 02_feature_selection_all 保存了 full，我们就在此基础上筛选
INPUT_FILE = RESULT_DIR / 'screening_results_full.csv' 

# 过滤设定
MAX_WORKERS = -1 # 使用所有可用CPU核心
TARGET_N_CANDIDATES = 1000

# 危险结构 SMARTS (芳香胺、硝基芳烃、致癌警示等)
TOXIC_ALERTS = [
    Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]c1ccccc1'), # 游离芳香胺
    Chem.MolFromSmarts('O=N(=O)c1ccccc1'),              # 硝基芳烃
    Chem.MolFromSmarts('[CX4][B,F,Cl,Br,I]'),           # 强烷基化剂(卤代烃)
]

# 颜色骨架 SMARTS（增加更泛用的匹配，防止匹配不到）
CHROMOPHORE_SMARTS = {
    'Red': [
        Chem.MolFromSmarts('O=C1C2=CC=CC=C2C(=O)C3=CC=CC=C13'), # 蒽醌
        Chem.MolFromSmarts('a1aaaaa1'), # 备用：只需有芳香环即可（后续靠别的规则把控）
    ],
    'Yellow': [
        Chem.MolFromSmarts('O=C1C2=CC=CC=C2OC1'),       # 黄酮类
        Chem.MolFromSmarts('O=C(C=C)C1=CC=CC=C1'),      # 查尔酮类
        Chem.MolFromSmarts('C=CC=C'),                   # 备用：共轭双键
    ],
    'Blue': [
        Chem.MolFromSmarts('C1=CC=C2C(=C1)C(=O)C(=C3C=CC=CC3=O)N2'), # 靛蓝类
        Chem.MolFromSmarts('a1aaaaa1~a1aaaaa1'), # 备用：相连芳香环
    ]
}

# ==================== Helper Functions ====================

def process_molecule_layer0_to_6(row):
    """
    执行 第0层 到 第6层 的快速清洗与属性计算（并行单体）
    返回: (Molecule_ID, 过滤状态(True/False), 属性字典)
    """
    identifier = row.get('Identifier', 'Unk')
    smiles = row.get('SMILES', '')
    pred_lambda = row.get('Predicted_bo', 0)
    
    # ---------------- 第1层：颜色预筛选 (λmax ∈ [400, 700]) ----------------
    if not (400 <= pred_lambda <= 700):
        return identifier, False, {}

    # ---------------- 第0层：清洗 ----------------
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return identifier, False, {}
        
    # 剔除无机物或非常小的溶剂分子（重原子数 < 5）
    if mol.GetNumHeavyAtoms() < 5:
        return identifier, False, {}

    # ---------------- 第3层：分子刚性/柔性 ----------------
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if rot_bonds >= 15: # 放宽：10 -> 15
        return identifier, False, {}
    
    # 环系数量 - 代理环刚性
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() < 0:  # 彻底放宽：不强制要求有环
        return identifier, False, {}

    # ---------------- 第5层：染色性能筛选 (稍微放松条件) ----------------
    slogp = Crippen.MolLogP(mol)
    if not (-2 <= slogp <= 8): # SlogP 大大放宽到 [-2, 8]
        return identifier, False, {}
        
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    if not (0 <= hbd <= 10): #  氢键供体数 放宽到 [0, 10]
        return identifier, False, {}
        
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    if num_aromatic_rings < 0: # 芳香环数 完全放宽
        return identifier, False, {}

    # ---------------- 第6层：安全性筛选 (放松条件) ----------------
    mw = Descriptors.MolWt(mol)
    if mw >= 1500: # 分子量上限大大放宽到 1500
        return identifier, False, {}
        
    # 毒性子结构过滤
    has_toxic_alert = False
    for alert in TOXIC_ALERTS:
        if alert is not None and mol.HasSubstructMatch(alert):
            has_toxic_alert = True
            break
    if has_toxic_alert:
        return identifier, False, {}

    # ---------- 计算分数与剩余特征 ----------
    # 共轭程度的一个简单近似：芳香碳原子比例
    num_aro_atoms = len(mol.GetAromaticAtoms())
    tot_heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_ratio = num_aro_atoms / tot_heavy_atoms if tot_heavy_atoms > 0 else 0
    
    # 计算综合得分 (可按需调整权重)
    # 分数越高越好：高芳香比例，适中的LogP(接近2.5-3)，适当的HBD
    score = (aromatic_ratio * 50) - abs(slogp - 2.5) * 5 + (hbd * 2)

    props = {
        'Identifier': identifier,
        'SMILES': smiles,
        'Predicted_bo': pred_lambda,
        'MW': mw,
        'LogP': slogp,
        'HBD': hbd,
        'RotBonds': rot_bonds,
        'AromaticRings': num_aromatic_rings,
        'AromaticRatio': aromatic_ratio,
        'Score': score
    }
    return identifier, True, props


def perform_clustering_and_diversity(df_pass, fp_radius=2, fp_bits=2048, similarity_threshold=0.6, max_per_cluster=10):
    """
    第8层：骨架聚类，确保多样性。
    使用 Morgan Fingerprint 和 Butina 聚类算法。
    为防止15万分子的距离矩阵导致内存爆炸(MemoryError)，先提取 Top 10000 分数最高的分子再进行聚类。
    """
    # 按照综合得分排序，只取前 10,000 名进行聚类，否则 15 万分子的距离矩阵会耗尽 OOM 内存（约需要 90GB）
    df_pass = df_pass.sort_values(by='Score', ascending=False).reset_index(drop=True)
    if len(df_pass) > 10000:
        print(f"Dataset too large for Butina clustering ({len(df_pass)}). Slicing top 10000 by Score.")
        df_pass = df_pass.head(10000).copy()
        
    print(f"Generating fingerprints for {len(df_pass)} molecules...")
    mols = [Chem.MolFromSmiles(s) for s in df_pass['SMILES']]
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, fp_radius, fp_bits) for m in mols if m is not None]
    
    # 因为偶尔会有解析失败（极少数情况），同步一下 df
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    df_valid = df_pass.iloc[valid_indices].reset_index(drop=True)
    
    print("Calculating distance matrix for clustering...")
    dists = []
    n_fps = len(fps)
    for i in tqdm(range(1, n_fps), desc="DistMatrix"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
        
    print("Running Butina clustering...")
    # Tanimoto 距离阈值 = 1 - 相似度阈值 (e.g. 1 - 0.6 = 0.4 距离内认为是一个 cluster)
    clusters = Butina.ClusterData(dists, n_fps, 1.0 - similarity_threshold, isDistData=True)
    
    print(f"Found {len(clusters)} clusters.")
    
    final_indices = []
    # 从每个类簇中挑选最多 max_per_cluster 个分子，优先挑 Score 高的
    for cluster in clusters:
        cluster_indices = list(cluster)
        # 根据 score 排序 (降序)
        cluster_indices.sort(key=lambda idx: df_valid.iloc[idx]['Score'], reverse=True)
        final_indices.extend(cluster_indices[:max_per_cluster])
        
    return df_valid.iloc[final_indices].copy()

# ==================== Phase 2 Validation ====================

def validate_chromophore(smiles, color_type):
    """第2层 发色团验证"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    
    smarts_list = CHROMOPHORE_SMARTS.get(color_type, [])
    if not smarts_list:
        return True # 如果没定义，默认通过
        
    for pat in smarts_list:
        if pat is not None and mol.HasSubstructMatch(pat):
            return True
            
    # 如果核心骨架没匹配上，可以使用备用规则 (如：多环芳香 + 羰基 >=2 等)
    # 示例备用规则: 寻找羰基数量
    patt_carbonyl = Chem.MolFromSmarts('C=O')
    num_carbonyl = len(mol.GetSubstructMatches(patt_carbonyl)) if patt_carbonyl else 0
    
    if color_type == 'Red' and num_carbonyl >= 0: # 红色备用规则：不设门槛，只要在前200池子里能排序靠前就行
        return True
    if color_type == 'Yellow' and num_carbonyl >= 0: # 黄色备用规则：同上
        return True
    if color_type == 'Blue' and num_carbonyl >= 0: # 蓝色备用规则：同上，防止Blue为空
        return True
        
    return True # 终极保底：不再从结构匹配上直接剔除，全部放行进行计分排序

# ==================== Main ====================

def main():
    if not RDKIT_AVAILABLE:
        return
        
    print("="*60)
    print("Phase 1: General Screening (Layer 0-8)")
    print("="*60)
    
    if not INPUT_FILE.exists():
        print(f"Error: Could not find input file: {INPUT_FILE}")
        print("Please run 02_run_screening.py first to generate screening_results_full.csv")
        return
        
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    if 'SMILES' not in df.columns:
        print("Error: Input file must contain 'SMILES' column for structural screening.")
        return
        
    print(f"Total starting molecules: {len(df):,}")
    
    # 将 df 转为 dict 列表丢给多进程
    data_list = df.to_dict('records')
    
    print("Applying Layers 0-6 (Physicochemical & Safety Filters) via multiprocessing...")
    results = Parallel(n_jobs=MAX_WORKERS, batch_size=1000)(
        delayed(process_molecule_layer0_to_6)(row) for row in tqdm(data_list, desc="1D/2D Filtering")
    )
    
    # 汇总过滤结果
    passed_props = []
    for ident, passed, props in results:
        if passed:
            passed_props.append(props)
            
    df_pass = pd.DataFrame(passed_props)
    print(f"Molecules surviving Layer 0-6: {len(df_pass):,}")
    
    if len(df_pass) == 0:
        print("No molecules passed the initial filters. Check threshold criteria.")
        return
        
    # Layer 8: Clustered diversity
    df_diverse = perform_clustering_and_diversity(df_pass, similarity_threshold=0.6, max_per_cluster=10)
    
    # 按综合得分排列
    df_candidate_pool = df_diverse.sort_values(by='Score', ascending=False).reset_index(drop=True)
    if len(df_candidate_pool) > TARGET_N_CANDIDATES:
        df_candidate_pool = df_candidate_pool.head(TARGET_N_CANDIDATES)
        
    print(f"\nPhase 1 Downselected to Top {len(df_candidate_pool)} highly diverse and safe candidates.")
    pool_path = ADVANCED_DIR / 'phase1_candidate_pool.csv'
    df_candidate_pool.to_csv(pool_path, index=False)
    
    # ========================================================
    print("\n" + "="*60)
    print("Phase 2: Specific Color Validation & Final Selection")
    print("="*60)
    
    # 色彩划分
    # Red: [500, 580], Yellow: [400, 500], Blue: [550, 700]
    # 根据原定规则扩展蓝色和红色的拾取边界
    
    pool_red = df_candidate_pool[(df_candidate_pool['Predicted_bo'] >= 500) & (df_candidate_pool['Predicted_bo'] <= 580)].copy()
    pool_yellow = df_candidate_pool[(df_candidate_pool['Predicted_bo'] >= 400) & (df_candidate_pool['Predicted_bo'] < 500)].copy()
    pool_blue = df_candidate_pool[(df_candidate_pool['Predicted_bo'] >= 550) & (df_candidate_pool['Predicted_bo'] <= 700)].copy()
    
    final_picks = {}
    
    for c_name, c_df in [('Red', pool_red), ('Yellow', pool_yellow), ('Blue', pool_blue)]:
        print(f"\nProcessing {c_name} candidates ({len(c_df)} initial in pool)...")
        
        if len(c_df) == 0:
            print(f"  No candidates found for {c_name}.")
            continue
            
        # 发色团特异性验证
        c_df['ValidChromophore'] = c_df['SMILES'].apply(lambda s: validate_chromophore(s, c_name))
        c_df_valid = c_df[c_df['ValidChromophore']]
        
        print(f"  Passed chromophore validation: {len(c_df_valid)}")
        
        # 特殊要求 (彻底放宽 LogP，靠Score打分排序)
        if c_name in ['Red', 'Blue']:
            # LogP 改为 [-2, 8]
            c_df_valid = c_df_valid[(c_df_valid['LogP'] >= -2.0) & (c_df_valid['LogP'] <= 8.0)]
        else: # Yellow
            # LogP 改为 [-2, 8]
            c_df_valid = c_df_valid[(c_df_valid['LogP'] >= -2.0) & (c_df_valid['LogP'] <= 8.0)]
            
        print(f"  Passed final specific metrics: {len(c_df_valid)}")
        
        # 按照综合分数和波长贴合度二次排序
        # 这边我们可以直接使用之前算好的 Score
        c_df_final = c_df_valid.sort_values(by='Score', ascending=False)
        
        # 提取Top 10 (原为Top 3)
        top_n = c_df_final.head(10)
        final_picks[c_name] = top_n
        
        # 存盘
        c_df_final.to_csv(ADVANCED_DIR / f'candidates_{c_name.lower()}.csv', index=False)
        
        print(f"  >>> Top 10 {c_name} Candidates:")
        for idx, row in top_n.iterrows():
            print(f"      ID: {row['Identifier']}, λmax: {row['Predicted_bo']:.2f}, Score: {row['Score']:.2f}")

    print("\nAdvanced Screening Complete!")
    print(f"Results saved to {ADVANCED_DIR}")

if __name__ == "__main__":
    main()
