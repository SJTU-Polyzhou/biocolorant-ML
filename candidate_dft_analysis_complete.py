# -*- coding: utf-8 -*-
"""
候选分子 DFT vs DNN 预测值综合分析
=====================================
功能模块：
1. 统计指标：R², MAE, RMSE, MedAE, MaxAE, Bias, Pct10/20/30
2. 6种AD分类方法（含独立散点图）
3. Bland-Altman一致性分析（含3张表格）
4. LOO敏感性分析（含4张图 + 合并图）
5. 描述符空间聚类分析（KMeans + PCA + t-SNE）
6. 带/不带分子名称两套图
7. 完整结果表格输出（含评判标准）

参考：external_validation_pipeline_multi_ad.py 框架

评判标准：
- Within AD: Distance ≤ Threshold
- Outside AD: Distance > Threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import chi2
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ============================================================
# 1. Configuration
# ============================================================

# Get current script directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Current directory: {CURRENT_DIR}")

# ========== 图表控制开关 ==========
GENERATE_BOTH_VERSIONS = True  # True: 同时生成带名称和不带名称两套图
LOO_SENSITIVITY = True  # True: 启用LOO敏感性分析
CLUSTER_ANALYSIS = True  # True: 启用描述符空间聚类分析（使用所有描述符）
CLUSTER_ANALYSIS_35 = True  # True: 启用35个描述符的聚类分析（新增）

print(f"\nChart settings:")
print(f"  Generate both versions (with/without names): {GENERATE_BOTH_VERSIONS}")
print(f"  LOO Sensitivity Analysis: {LOO_SENSITIVITY}")
print(f"  Clustering Analysis (all descriptors): {CLUSTER_ANALYSIS}")
print(f"  Clustering Analysis (35 descriptors): {CLUSTER_ANALYSIS_35}")

# 35个指定的描述符（与建模一致）
SELECTED_BITS = [
    906, 1336, 334, 1310, 1348, 1406, 1437, 345, 218, 230,
    449, 510, 299, 3, 1547, 566, 1418, 1069, 232, 264,
    1076, 1077, 322, 1313, 1294, 1064, 1329, 1146, 1477, 1359,
    805, 1394, 1358, 1573, 485
]

# ========== AD方法配置 ==========
AD_METHODS = {
    'euclidean_nearest': True,
    'euclidean_center': True,
    'mahalanobis': True,
    'leverage': True,
    'knn_density': True,
    'isolation_forest': True,
}

# ========== 聚类分析参数 ==========
N_CLUSTERS = 5
CLUSTER_THRESHOLD_PERCENTILE = 95
PCA_RANDOM_STATE = 42
TSNE_RANDOM_STATE = 42
TSNE_PERPLEXITY = 30
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

# ========== 文件配置 ==========
CANDIDATE_FILE = 'candidate_molecules-31g.xlsx'
DESC_FILE = 'candidate_molecules-31g_with_Md.xlsx'
TRAIN_DESC_FILE = 'data_w_Md_bo.xlsx'
OUTPUT_DIR = 'candidate_dft_complete_analysis'

# 颜色配置
WITHIN_AD_COLOR = '#1F4E79'
OUTSIDE_AD_COLOR = '#D95F02'
ALL_DATA_COLOR = '#4C72B0'
ALL_DATA_ALPHA = 0.5
CLUSTER_CMAP = 'viridis'
BACKGROUND_COLOR = 'white'
BLACK = '#000000'


# ============================================================
# 2. Helper functions
# ============================================================

def setup_cjche_style_larger():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.title_fontsize'] = 11
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.grid'] = False


def setup_cjche_style_cluster():
    """聚类分析专用样式"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.facecolor'] = BACKGROUND_COLOR
    plt.rcParams['axes.facecolor'] = BACKGROUND_COLOR
    plt.rcParams['axes.grid'] = False


def save_figure_tiff(fig, filepath, dpi=300):
    fig.savefig(filepath, format='tiff', dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")


def set_axes_border(ax, linewidth=1.5):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


def set_axes_border_cluster(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(BLACK)
    ax.tick_params(axis='both', colors=BLACK)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
        for col in df.columns:
            if col.lower() == name.lower():
                return col
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None


def compute_metrics(y_true, y_pred):
    """计算所有指标：R², MAE, RMSE, MedAE, MaxAE, Bias"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        return {
            'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MedAE': np.nan,
            'MaxAE': np.nan, 'Bias': np.nan,
            'Pct10': np.nan, 'Pct20': np.nan, 'Pct30': np.nan,
            'N': 0
        }

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    maxae = np.max(abs_errors)
    bias = np.mean(errors)

    pct10 = np.mean(abs_errors <= 10) * 100
    pct20 = np.mean(abs_errors <= 20) * 100
    pct30 = np.mean(abs_errors <= 30) * 100

    return {
        'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MedAE': medae,
        'MaxAE': maxae, 'Bias': bias,
        'Pct10': pct10, 'Pct20': pct20, 'Pct30': pct30,
        'N': len(y_true)
    }


# ============================================================
# 3. Data loading
# ============================================================

def load_candidate_data():
    """加载候选分子数据（DFT和DNN值）"""
    if not os.path.exists(CANDIDATE_FILE):
        print(f"错误: 找不到候选文件 {CANDIDATE_FILE}")
        return None

    df = pd.read_excel(CANDIDATE_FILE, engine='openpyxl')
    print(f"✓ 加载候选文件: {len(df)} 个分子")

    id_col = find_column(df, ['No.', 'ID', '编号'])
    dft_col = find_column(df, ['31g-DFTλmax/nm', 'DFT', '31g', 'Closest-to-visible'])
    dnn_col = find_column(df, ['DNN-test', 'DNN', 'Predicted', 'Pred'])

    if None in [id_col, dft_col, dnn_col]:
        print("错误: 缺少必要列")
        return None

    ids = df[id_col].tolist()
    dft_vals = [float(x) if pd.notna(x) else None for x in df[dft_col]]
    dnn_vals = [float(x) if pd.notna(x) else None for x in df[dnn_col]]

    valid = [i for i, (d, n) in enumerate(zip(dft_vals, dnn_vals)) if d is not None and n is not None]
    if len(valid) < len(ids):
        print(f"  过滤了 {len(ids) - len(valid)} 个含缺失值的样本")
        ids = [ids[i] for i in valid]
        dft_vals = [dft_vals[i] for i in valid]
        dnn_vals = [dnn_vals[i] for i in valid]

    print(f"  有效样本: {len(ids)}")

    return {
        'ids': ids,
        'dft': np.array(dft_vals),
        'dnn': np.array(dnn_vals),
        'df': df
    }


def load_candidate_descriptors():
    """加载候选分子的所有描述符（原函数，不变）"""
    if not os.path.exists(DESC_FILE):
        print(f"警告: 描述符文件 {DESC_FILE} 不存在")
        return None, None

    df = pd.read_excel(DESC_FILE, engine='openpyxl')
    desc_cols = [col for col in df.columns if col.startswith('Md_')]
    print(f"✓ 加载候选描述符: {len(df)} 个分子, {len(desc_cols)} 个描述符")
    return df[desc_cols].values, df


def load_candidate_descriptors_35():
    """新增：加载候选分子的35个选定描述符"""
    if not os.path.exists(DESC_FILE):
        print(f"警告: 描述符文件 {DESC_FILE} 不存在")
        return None, None

    df = pd.read_excel(DESC_FILE, engine='openpyxl')
    selected_cols = [f'Md_{bit}' for bit in SELECTED_BITS]
    available_cols = [col for col in selected_cols if col in df.columns]
    if len(available_cols) < len(SELECTED_BITS):
        missing = set(SELECTED_BITS) - set([int(col.split('_')[1]) for col in available_cols])
        print(f"  警告: 缺少 {len(missing)} 个描述符列: {missing}")
    desc_cols = available_cols
    print(f"✓ 加载候选35个描述符: {len(df)} 个分子, {len(desc_cols)} 个描述符")
    return df[desc_cols].values, df


def load_training_descriptors():
    """加载训练集描述符（原函数，不变）"""
    if not os.path.exists(TRAIN_DESC_FILE):
        print(f"警告: 训练集描述符文件 {TRAIN_DESC_FILE} 不存在")
        return None, None

    df_all = pd.read_excel(TRAIN_DESC_FILE, engine='openpyxl')
    train_col = find_column(df_all, ['Training set/Testing set', 'Set', 'Type'])

    if train_col is not None:
        train_mask = df_all[train_col].astype(str).str.lower().str.contains('training')
        df_train = df_all[train_mask].copy()
        print(f"✓ 训练集: {len(df_train)} 个分子")
    else:
        print("警告: 未找到训练集标识列，使用全部数据")
        df_train = df_all.copy()

    desc_cols = [col for col in df_train.columns if col.startswith('Md_')]
    print(f"✓ 加载训练集描述符: {len(df_train)} 个分子, {len(desc_cols)} 个描述符")
    return df_train[desc_cols].values, df_train


def load_training_descriptors_35():
    """新增：加载训练集的35个选定描述符"""
    if not os.path.exists(TRAIN_DESC_FILE):
        print(f"警告: 训练集描述符文件 {TRAIN_DESC_FILE} 不存在")
        return None, None

    df_all = pd.read_excel(TRAIN_DESC_FILE, engine='openpyxl')
    train_col = find_column(df_all, ['Training set/Testing set', 'Set', 'Type'])

    if train_col is not None:
        train_mask = df_all[train_col].astype(str).str.lower().str.contains('training')
        df_train = df_all[train_mask].copy()
        print(f"✓ 训练集: {len(df_train)} 个分子")
    else:
        print("警告: 未找到训练集标识列，使用全部数据")
        df_train = df_all.copy()

    selected_cols = [f'Md_{bit}' for bit in SELECTED_BITS]
    available_cols = [col for col in selected_cols if col in df_train.columns]
    if len(available_cols) < len(SELECTED_BITS):
        missing = set(SELECTED_BITS) - set([int(col.split('_')[1]) for col in available_cols])
        print(f"  警告: 缺少 {len(missing)} 个描述符列: {missing}")
    desc_cols = available_cols
    print(f"✓ 加载训练集35个描述符: {len(df_train)} 个分子, {len(desc_cols)} 个描述符")
    return df_train[desc_cols].values, df_train


# ============================================================
# 4. AD Methods
# ============================================================

def ad_euclidean_nearest(train_X, test_X, percentile=95):
    dist = cdist(test_X, train_X, metric='euclidean')
    min_dist = np.min(dist, axis=1)
    train_dist = cdist(train_X, train_X, metric='euclidean')
    np.fill_diagonal(train_dist, np.nan)
    threshold = np.nanpercentile(train_dist, percentile)
    return min_dist <= threshold, min_dist, threshold, 'Euclidean-Nearest'


def ad_euclidean_center(train_X, test_X, percentile=95):
    center = np.mean(train_X, axis=0)
    dist = np.linalg.norm(test_X - center, axis=1)
    train_center = np.mean(train_X, axis=0)
    train_dist = np.linalg.norm(train_X - train_center, axis=1)
    threshold = np.percentile(train_dist, percentile)
    return dist <= threshold, dist, threshold, 'Euclidean-Center'


def ad_mahalanobis(train_X, test_X, alpha=0.05):
    center = np.mean(train_X, axis=0)
    cov = np.cov(train_X.T)
    try:
        inv_cov = np.linalg.pinv(cov)
    except:
        inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    dist = np.array([mahalanobis(x, center, inv_cov) for x in test_X])
    threshold = np.sqrt(chi2.ppf(1 - alpha, train_X.shape[1]))
    return dist <= threshold, dist, threshold, 'Mahalanobis'


def ad_leverage(train_X, test_X):
    n_train = train_X.shape[0]
    p = train_X.shape[1]
    H_test = test_X @ np.linalg.pinv(train_X.T @ train_X) @ test_X.T
    leverage = np.diag(H_test)
    threshold = 3 * (p + 1) / n_train
    return leverage <= threshold, leverage, threshold, 'Leverage'


def ad_knn_density(train_X, test_X, k=5, percentile=95):
    knn = NearestNeighbors(n_neighbors=min(k, len(train_X)))
    knn.fit(train_X)
    train_dist, _ = knn.kneighbors(train_X)
    train_mean_dist = np.mean(train_dist, axis=1)
    threshold = np.percentile(train_mean_dist, percentile)
    test_dist, _ = knn.kneighbors(test_X)
    test_mean_dist = np.mean(test_dist, axis=1)
    return test_mean_dist <= threshold, test_mean_dist, threshold, f'KNN-k={k}'


def ad_isolation_forest(train_X, test_X, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    iso_forest.fit(train_X)
    pred = iso_forest.predict(test_X)
    scores = iso_forest.decision_function(test_X)
    threshold = 0.0
    return pred == 1, scores, threshold, 'IsolationForest'


def apply_all_ad_methods(train_X, test_X):
    results = {}

    if train_X is None or test_X is None:
        return results

    method_configs = [
        ('euclidean_nearest', ad_euclidean_nearest, {}),
        ('euclidean_center', ad_euclidean_center, {}),
        ('mahalanobis', ad_mahalanobis, {'alpha': 0.05}),
        ('leverage', ad_leverage, {}),
        ('knn_density', ad_knn_density, {'k': 5, 'percentile': 95}),
        ('isolation_forest', ad_isolation_forest, {'contamination': 0.05}),
    ]

    print("\n应用AD方法...")
    for method_name, method_func, kwargs in method_configs:
        if AD_METHODS.get(method_name, False):
            try:
                ad_flags, distances, threshold, display_name = method_func(train_X, test_X, **kwargs)
                results[method_name] = {
                    'flags': ad_flags,
                    'distances': distances,
                    'threshold': threshold,
                    'display_name': display_name,
                    'n_in': np.sum(ad_flags),
                    'n_out': np.sum(~ad_flags)
                }
                print(f"  ✓ {display_name}: Within AD {np.sum(ad_flags)}, Outside AD {np.sum(~ad_flags)}")
            except Exception as e:
                print(f"  ✗ {method_name} 失败: {e}")

    return results


# ============================================================
# 5. Bland-Altman Analysis
# ============================================================

def bland_altman_analysis(y1, y2):
    mean_vals = (y1 + y2) / 2
    diff_vals = y1 - y2
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    return {
        'mean': mean_vals,
        'diff': diff_vals,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'loa_upper': loa_upper,
        'loa_lower': loa_lower,
        'n': len(y1)
    }


# ============================================================
# 6. LOO Sensitivity Analysis
# ============================================================

def loo_sensitivity_analysis(ids, exp_vals, pred_vals, output_dir):
    print("\n" + "=" * 60)
    print("LOO Sensitivity Analysis")
    print("=" * 60)

    n_samples = len(exp_vals)
    print(f"  Total samples: {n_samples}")
    print(f"  Performing LOO analysis for {n_samples} samples...")

    original_metrics = compute_metrics(exp_vals, pred_vals)

    loo_results = []

    for i in range(n_samples):
        idx_keep = np.ones(n_samples, dtype=bool)
        idx_keep[i] = False

        exp_loo = exp_vals[idx_keep]
        pred_loo = pred_vals[idx_keep]

        metrics_loo = compute_metrics(exp_loo, pred_loo)

        delta_r2 = original_metrics['R2'] - metrics_loo['R2']
        delta_mae = original_metrics['MAE'] - metrics_loo['MAE']
        delta_rmse = original_metrics['RMSE'] - metrics_loo['RMSE']
        delta_bias = original_metrics['Bias'] - metrics_loo['Bias']

        loo_results.append({
            'Sample_ID': ids[i],
            'Index': i + 1,
            'Experimental': exp_vals[i],
            'Predicted': pred_vals[i],
            'Error': pred_vals[i] - exp_vals[i],
            'Abs_Error': np.abs(pred_vals[i] - exp_vals[i]),
            'R2_without': metrics_loo['R2'],
            'MAE_without': metrics_loo['MAE'],
            'RMSE_without': metrics_loo['RMSE'],
            'Bias_without': metrics_loo['Bias'],
            'Delta_R2': delta_r2,
            'Delta_MAE': delta_mae,
            'Delta_RMSE': delta_rmse,
            'Delta_Bias': delta_bias,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_samples} samples")

    print(f"✓ LOO analysis completed!")

    df_loo = pd.DataFrame(loo_results)

    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    loo_path = os.path.join(tables_dir, 'Table_LOO_Sensitivity_Analysis.csv')
    df_loo.to_csv(loo_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved LOO analysis: {loo_path}")

    print("\n" + "-" * 60)
    print("Top 5 samples with largest influence on R²:")
    print("-" * 60)
    df_sorted_r2 = df_loo.sort_values('Delta_R2', ascending=False)
    print(f"{'Sample':<12} {'R2_without':<12} {'Delta_R2':<12} {'Error':<12}")
    for _, row in df_sorted_r2.head(5).iterrows():
        print(f"{row['Sample_ID']:<12} {row['R2_without']:<12.4f} {row['Delta_R2']:<12.4f} {row['Error']:<12.2f}")

    print("\nTop 5 samples with largest influence on MAE:")
    print("-" * 60)
    df_sorted_mae = df_loo.sort_values('Delta_MAE', ascending=False)
    print(f"{'Sample':<12} {'MAE_without':<12} {'Delta_MAE':<12} {'Abs_Error':<12}")
    for _, row in df_sorted_mae.head(5).iterrows():
        print(f"{row['Sample_ID']:<12} {row['MAE_without']:<12.4f} {row['Delta_MAE']:<12.4f} {row['Abs_Error']:<12.2f}")

    return df_loo


# ============================================================
# 7. LOO Plotting functions
# ============================================================

def create_loo_delta_r2_fig14(df_loo):
    fig, ax = plt.subplots(figsize=(14 / 2.54, 8 / 2.54))
    x = np.arange(len(df_loo))
    colors = ['red' if d > 0 else 'blue' for d in df_loo['Delta_R2']]
    ax.bar(x, df_loo['Delta_R2'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('ΔR² (R²_all - R²_without)', fontsize=12)
    ax.set_title('LOO Sensitivity: Impact on R²', fontsize=12)
    ax.set_xticks(x[::max(1, len(x) // 20)])
    ax.set_xticklabels(df_loo['Sample_ID'].values[::max(1, len(x) // 20)], rotation=45, ha='right', fontsize=9)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    return fig


def create_loo_delta_mae_fig15(df_loo):
    fig, ax = plt.subplots(figsize=(14 / 2.54, 8 / 2.54))
    x = np.arange(len(df_loo))
    colors = ['red' if d > 0 else 'blue' for d in df_loo['Delta_MAE']]
    ax.bar(x, df_loo['Delta_MAE'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('ΔMAE (MAE_all - MAE_without)', fontsize=12)
    ax.set_title('LOO Sensitivity: Impact on MAE', fontsize=12)
    ax.set_xticks(x[::max(1, len(x) // 20)])
    ax.set_xticklabels(df_loo['Sample_ID'].values[::max(1, len(x) // 20)], rotation=45, ha='right', fontsize=9)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    return fig


def create_loo_influence_scatter_fig16(df_loo, show_names=True):
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    sizes = np.abs(df_loo['Delta_R2']) * 500 + 20
    colors = np.abs(df_loo['Error'])

    scatter = ax.scatter(df_loo['Experimental'], df_loo['Predicted'],
                         s=sizes, c=colors, cmap='coolwarm_r', alpha=0.7,
                         edgecolors='black', linewidth=0.8)

    min_val = min(df_loo['Experimental'].min(), df_loo['Predicted'].min()) - 10
    max_val = max(df_loo['Experimental'].max(), df_loo['Predicted'].max()) + 10
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)

    if show_names:
        offsets = [(5, -8), (8, -5), (2, -12), (12, -3), (-5, -10), (5, -15)]
        for idx, row in df_loo.iterrows():
            offset_idx = idx % len(offsets)
            ox, oy = offsets[offset_idx]
            ax.annotate(str(row['Sample_ID']), (row['Experimental'], row['Predicted']),
                        xytext=(ox, oy), textcoords='offset points',
                        fontsize=7, alpha=0.8, ha='left', va='top')

    ax.set_xlabel('DFT calculated λmax (nm)', fontsize=12)
    ax.set_ylabel('DNN predicted λmax (nm)', fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error (nm)', fontsize=11)

    textstr = 'Circle size = |ΔR²| influence'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    return fig


def create_loo_cumulative_fig17(df_loo):
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    df_sorted = df_loo.sort_values('Delta_R2', ascending=False)
    cumulative_r2 = np.cumsum(df_sorted['Delta_R2'].values)
    cumulative_mae = np.cumsum(df_sorted['Delta_MAE'].values)

    x_cum = np.arange(len(df_sorted))

    ax.plot(x_cum, cumulative_r2, 'b-', linewidth=2, label='Cumulative ΔR²')
    ax.plot(x_cum, cumulative_mae, 'r-', linewidth=2, label='Cumulative ΔMAE')

    ax.set_xlabel('Samples sorted by influence', fontsize=12)
    ax.set_ylabel('Cumulative change', fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    return fig


# ============================================================
# 8. NEW: Figure 18/20 - Combined Scatter + Influence Plot
# ============================================================

def create_fig18_combined(ids, exp_vals, pred_vals, abs_errors, df_loo, global_stats, target_dir, show_names=True):
    """
    Figure 18: Combined scatter plot with influence size
    Merges Fig1 (scatter) and Fig16 (influence scatter) information
    """
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    # 气泡图：点的大小表示影响力
    if df_loo is not None and len(df_loo) > 0:
        sizes = np.abs(df_loo['Delta_R2']) * 500 + 20
        colors = np.abs(df_loo['Error'])
        exp_vals_loo = df_loo['Experimental'].values
        pred_vals_loo = df_loo['Predicted'].values

        scatter = ax.scatter(exp_vals_loo, pred_vals_loo,
                             s=sizes, c=colors, cmap='coolwarm_r', alpha=0.7,
                             edgecolors='black', linewidth=0.8)

        if show_names:
            offsets = [(5, -8), (8, -5), (2, -12), (12, -3), (-5, -10), (5, -15)]
            for idx, row in df_loo.iterrows():
                offset_idx = idx % len(offsets)
                ox, oy = offsets[offset_idx]
                ax.annotate(str(row['Sample_ID']), (row['Experimental'], row['Predicted']),
                            xytext=(ox, oy), textcoords='offset points',
                            fontsize=7, alpha=0.8, ha='left', va='top')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Absolute Error (nm)', fontsize=11)
    else:
        norm = Normalize(vmin=0, vmax=np.max(abs_errors))
        scatter = ax.scatter(exp_vals, pred_vals, c=abs_errors, cmap=plt.cm.coolwarm_r, norm=norm,
                             s=40, alpha=0.7, edgecolors='k', linewidth=0.8)
        if show_names:
            for i, cid in enumerate(ids):
                ax.annotate(str(cid), (exp_vals[i], pred_vals[i]),
                            xytext=(5, -8), textcoords='offset points',
                            fontsize=7, alpha=0.8, ha='left', va='top')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Absolute error (nm)', fontsize=11)

    # 对角线
    min_val = min(min(exp_vals), min(pred_vals)) - 10
    max_val = max(max(exp_vals), max(pred_vals)) + 10
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Exp. λmax (nm)', fontsize=12)
    ax.set_ylabel('Pre. λmax (nm)', fontsize=12)
    ax.tick_params(labelsize=11)

    # ===== 合并统计信息和影响力说明（精简版：R² + MAE + Circle size） =====
    if df_loo is not None and len(df_loo) > 0:
        textstr = '\n'.join((
            f'$R^2$ = {global_stats["R2"]:.4f}',
            f'MAE = {global_stats["MAE"]:.4f} nm',
            f'Circle size = |ΔR²|'
        ))
    else:
        textstr = '\n'.join((
            f'$R^2$ = {global_stats["R2"]:.4f}',
            f'MAE = {global_stats["MAE"]:.4f} nm'
        ))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    set_axes_border(ax)
    plt.tight_layout()
    return fig


# ============================================================
# 9. Descriptor Space Clustering Analysis (所有描述符)
# ============================================================

def cluster_analysis(train_desc, cand_desc, candidate_ids, output_dir, fig_prefix='FigA'):
    """描述符空间聚类分析 - 使用所有描述符（原有功能，不变）"""
    print("\n" + "=" * 60)
    print("Descriptor Space Clustering Analysis (All Descriptors)")
    print("=" * 60)

    if train_desc is None or cand_desc is None:
        print("跳过聚类分析（描述符数据缺失）")
        return

    # 标准化
    X_all = np.vstack([train_desc, cand_desc])
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    X_train = X_all_scaled[:len(train_desc)]
    X_candidates = X_all_scaled[len(train_desc):]

    print(f"  Training set: {X_train.shape}")
    print(f"  Candidate set: {X_candidates.shape}")

    # KMeans聚类
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    cluster_labels = kmeans.fit_predict(X_train)
    cluster_centers = kmeans.cluster_centers_

    print(f"✓ KMeans clustering completed: {N_CLUSTERS} clusters")

    # 计算每个簇的统计信息
    cluster_distances = {c: [] for c in range(N_CLUSTERS)}
    for i, point in enumerate(X_train):
        c = cluster_labels[i]
        dist = np.linalg.norm(point - cluster_centers[c])
        cluster_distances[c].append(dist)

    cluster_thresholds = {}
    cluster_stats = []
    for c in range(N_CLUSTERS):
        if len(cluster_distances[c]) > 0:
            cluster_thresholds[c] = np.percentile(cluster_distances[c], CLUSTER_THRESHOLD_PERCENTILE)
            cluster_stats.append({
                'Cluster': c,
                'Number_of_Samples': len(cluster_distances[c]),
                'Mean_Distance': np.mean(cluster_distances[c]),
                'Std_Distance': np.std(cluster_distances[c]),
                'Min_Distance': np.min(cluster_distances[c]),
                'Max_Distance': np.max(cluster_distances[c]),
                'Threshold_95%': cluster_thresholds[c]
            })
        else:
            cluster_thresholds[c] = np.inf
            cluster_stats.append({
                'Cluster': c,
                'Number_of_Samples': 0,
                'Mean_Distance': np.nan,
                'Std_Distance': np.nan,
                'Min_Distance': np.nan,
                'Max_Distance': np.nan,
                'Threshold_95%': np.inf
            })
        print(f"  Cluster {c}: {len(cluster_distances[c])} samples, threshold={cluster_thresholds[c]:.4f}")

    # 评估候选分子
    candidate_results = []
    for i, cand in enumerate(X_candidates):
        dists = [np.linalg.norm(cand - center) for center in cluster_centers]
        nearest_cluster = np.argmin(dists)
        min_dist = dists[nearest_cluster]

        if min_dist <= cluster_thresholds[nearest_cluster]:
            reliability = "Within AD"
            color = WITHIN_AD_COLOR
        else:
            reliability = "Outside AD"
            color = OUTSIDE_AD_COLOR

        candidate_results.append({
            'figure_id': i + 1,
            'npbs_id': candidate_ids[i] if i < len(candidate_ids) else f'Cand_{i + 1}',
            'nearest_cluster': nearest_cluster,
            'distance': min_dist,
            'threshold': cluster_thresholds[nearest_cluster],
            'reliability': reliability,
            'color': color
        })

    # 统计
    within_count = sum(r['reliability'] == 'Within AD' for r in candidate_results)
    outside_count = sum(r['reliability'] == 'Outside AD' for r in candidate_results)
    print(f"\n  Reliability summary:")
    print(f"    Within AD: {within_count} molecules")
    print(f"    Outside AD: {outside_count} molecules")

    # ===== 保存聚类结果表格 =====
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # 表格1: 候选分子聚类详细结果（含评判标准）
    mapping_df = pd.DataFrame({
        'Figure_ID': [r['figure_id'] for r in candidate_results],
        'NPBS_ID': [r['npbs_id'] for r in candidate_results],
        'Reliability': [r['reliability'] for r in candidate_results],
        'Nearest_Cluster': [r['nearest_cluster'] for r in candidate_results],
        'Distance_to_Center': [r['distance'] for r in candidate_results],
        'Threshold': [r['threshold'] for r in candidate_results],
        'Criteria': [
            f'Distance ({r["distance"]:.3f}) ≤ Threshold ({r["threshold"]:.3f})' if r['reliability'] == 'Within AD'
            else f'Distance ({r["distance"]:.3f}) > Threshold ({r["threshold"]:.3f})'
            for r in candidate_results
        ]
    })
    mapping_df.to_csv(f'{tables_dir}/{fig_prefix}_cluster_mapping.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {tables_dir}/{fig_prefix}_cluster_mapping.csv")

    # 表格2: 簇统计信息
    cluster_df = pd.DataFrame(cluster_stats)
    cluster_df.to_csv(f'{tables_dir}/{fig_prefix}_cluster_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {tables_dir}/{fig_prefix}_cluster_statistics.csv")

    # ===== 生成聚类可视化图 =====
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    setup_cjche_style_cluster()

    legend_elements = [
        Patch(facecolor=WITHIN_AD_COLOR, edgecolor=BLACK, label='Within AD'),
        Patch(facecolor=OUTSIDE_AD_COLOR, edgecolor=BLACK, label='Outside AD'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=BLACK, markersize=8, label='Candidate')
    ]

    # PCA
    pca = PCA(n_components=2, random_state=PCA_RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_cand_pca = pca.transform(X_candidates)
    var_ratio = pca.explained_variance_ratio_

    # Fig A1: PCA - clusters
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    sc = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP,
                    alpha=ALL_DATA_ALPHA, s=25, edgecolors=BLACK, linewidth=0.5)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_pca):
            ax.scatter(X_cand_pca[idx, 0], X_cand_pca[idx, 1], c=r['color'],
                       s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
    ax.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}1_PCA_clusters.tiff')
    plt.close()

    # Fig A2: PCA - uniform
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=ALL_DATA_COLOR, alpha=ALL_DATA_ALPHA,
               s=20, edgecolors=BLACK, linewidth=0.3)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_pca):
            ax.scatter(X_cand_pca[idx, 0], X_cand_pca[idx, 1], c=r['color'],
                       s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
    ax.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}2_PCA_uniform.tiff')
    plt.close()

    # Fig A3: PCA - highlight
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='#E8E8E8', alpha=0.6, s=15, edgecolors=BLACK, linewidth=0.2)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_pca):
            ax.scatter(X_cand_pca[idx, 0], X_cand_pca[idx, 1], c=r['color'],
                       s=90, marker='*', edgecolors=BLACK, linewidth=1.0, zorder=5)
    ax.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}3_PCA_highlight.tiff')
    plt.close()

    # t-SNE
    X_combined = np.vstack([X_train, X_candidates])
    perplexity = min(TSNE_PERPLEXITY, len(X_combined) - 1)
    tsne = TSNE(n_components=2, random_state=TSNE_RANDOM_STATE, perplexity=perplexity)
    X_combined_tsne = tsne.fit_transform(X_combined)
    X_train_tsne = X_combined_tsne[:len(X_train)]
    X_cand_tsne = X_combined_tsne[len(X_train):]

    # Fig A4: t-SNE - clusters
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP,
               alpha=ALL_DATA_ALPHA, s=25, edgecolors=BLACK, linewidth=0.5)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_tsne):
            ax.scatter(X_cand_tsne[idx, 0], X_cand_tsne[idx, 1], c=r['color'],
                       s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}4_tSNE_clusters.tiff')
    plt.close()

    # Fig A5: t-SNE - space
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=ALL_DATA_COLOR, alpha=0.3,
               s=20, edgecolors=BLACK, linewidth=0.2)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_tsne):
            ax.scatter(X_cand_tsne[idx, 0], X_cand_tsne[idx, 1], c=r['color'],
                       s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}5_tSNE_space.tiff')
    plt.close()

    # Fig A6: Distribution
    fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))
    bars = ax.bar(['Within AD', 'Outside AD'], [within_count, outside_count],
                  color=[WITHIN_AD_COLOR, OUTSIDE_AD_COLOR], alpha=0.7,
                  edgecolor=BLACK, linewidth=0.8)
    for bar, count in zip(bars, [within_count, outside_count]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count),
                ha='center', va='bottom', fontsize=9, color=BLACK)
    ax.set_ylabel('Number of molecules')
    ax.set_xlabel('Reliability level')
    ax.tick_params(axis='both', colors=BLACK)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}6_distribution.tiff')
    plt.close()

    setup_cjche_style_larger()

    return candidate_results


# ============================================================
# 9b. Descriptor Space Clustering Analysis (35个描述符)
# ============================================================

def cluster_analysis_35(train_desc, cand_desc, candidate_ids, output_dir, fig_prefix='FigB'):
    """描述符空间聚类分析 - 使用35个选定描述符（新增）"""
    print("\n" + "=" * 60)
    print("Descriptor Space Clustering Analysis (35 Selected Descriptors)")
    print("=" * 60)

    if train_desc is None or cand_desc is None:
        print("跳过聚类分析（描述符数据缺失）")
        return

    # 标准化
    X_all = np.vstack([train_desc, cand_desc])
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    X_train = X_all_scaled[:len(train_desc)]
    X_candidates = X_all_scaled[len(train_desc):]

    print(f"  Training set: {X_train.shape}")
    print(f"  Candidate set: {X_candidates.shape}")

    # KMeans聚类
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    cluster_labels = kmeans.fit_predict(X_train)
    cluster_centers = kmeans.cluster_centers_

    print(f"✓ KMeans clustering completed: {N_CLUSTERS} clusters")

    # 计算每个簇的统计信息
    cluster_distances = {c: [] for c in range(N_CLUSTERS)}
    for i, point in enumerate(X_train):
        c = cluster_labels[i]
        dist = np.linalg.norm(point - cluster_centers[c])
        cluster_distances[c].append(dist)

    cluster_thresholds = {}
    cluster_stats = []
    for c in range(N_CLUSTERS):
        if len(cluster_distances[c]) > 0:
            cluster_thresholds[c] = np.percentile(cluster_distances[c], CLUSTER_THRESHOLD_PERCENTILE)
            cluster_stats.append({
                'Cluster': c,
                'Number_of_Samples': len(cluster_distances[c]),
                'Mean_Distance': np.mean(cluster_distances[c]),
                'Std_Distance': np.std(cluster_distances[c]),
                'Min_Distance': np.min(cluster_distances[c]),
                'Max_Distance': np.max(cluster_distances[c]),
                'Threshold_95%': cluster_thresholds[c]
            })
        else:
            cluster_thresholds[c] = np.inf
            cluster_stats.append({
                'Cluster': c,
                'Number_of_Samples': 0,
                'Mean_Distance': np.nan,
                'Std_Distance': np.nan,
                'Min_Distance': np.nan,
                'Max_Distance': np.nan,
                'Threshold_95%': np.inf
            })
        print(f"  Cluster {c}: {len(cluster_distances[c])} samples, threshold={cluster_thresholds[c]:.4f}")

    # 评估候选分子
    candidate_results = []
    for i, cand in enumerate(X_candidates):
        dists = [np.linalg.norm(cand - center) for center in cluster_centers]
        nearest_cluster = np.argmin(dists)
        min_dist = dists[nearest_cluster]

        if min_dist <= cluster_thresholds[nearest_cluster]:
            reliability = "Within AD"
            color = WITHIN_AD_COLOR
        else:
            reliability = "Outside AD"
            color = OUTSIDE_AD_COLOR

        candidate_results.append({
            'figure_id': i + 1,
            'npbs_id': candidate_ids[i] if i < len(candidate_ids) else f'Cand_{i + 1}',
            'nearest_cluster': nearest_cluster,
            'distance': min_dist,
            'threshold': cluster_thresholds[nearest_cluster],
            'reliability': reliability,
            'color': color
        })

    # 统计
    within_count = sum(r['reliability'] == 'Within AD' for r in candidate_results)
    outside_count = sum(r['reliability'] == 'Outside AD' for r in candidate_results)
    print(f"\n  Reliability summary:")
    print(f"    Within AD: {within_count} molecules")
    print(f"    Outside AD: {outside_count} molecules")

    # ===== 保存聚类结果表格 =====
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # 表格1: 候选分子聚类详细结果（含评判标准）
    mapping_df = pd.DataFrame({
        'Figure_ID': [r['figure_id'] for r in candidate_results],
        'NPBS_ID': [r['npbs_id'] for r in candidate_results],
        'Reliability': [r['reliability'] for r in candidate_results],
        'Nearest_Cluster': [r['nearest_cluster'] for r in candidate_results],
        'Distance_to_Center': [r['distance'] for r in candidate_results],
        'Threshold': [r['threshold'] for r in candidate_results],
        'Criteria': [
            f'Distance ({r["distance"]:.3f}) ≤ Threshold ({r["threshold"]:.3f})' if r['reliability'] == 'Within AD'
            else f'Distance ({r["distance"]:.3f}) > Threshold ({r["threshold"]:.3f})'
            for r in candidate_results
        ]
    })
    mapping_df.to_csv(f'{tables_dir}/{fig_prefix}_cluster_mapping.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {tables_dir}/{fig_prefix}_cluster_mapping.csv")

    # 表格2: 簇统计信息
    cluster_df = pd.DataFrame(cluster_stats)
    cluster_df.to_csv(f'{tables_dir}/{fig_prefix}_cluster_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {tables_dir}/{fig_prefix}_cluster_statistics.csv")

    # ===== 生成聚类可视化图 =====
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    setup_cjche_style_cluster()

    legend_elements = [
        Patch(facecolor=WITHIN_AD_COLOR, edgecolor=BLACK, label='Within AD'),
        Patch(facecolor=OUTSIDE_AD_COLOR, edgecolor=BLACK, label='Outside AD'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=BLACK, markersize=8, label='Candidate')
    ]

    # PCA
    pca = PCA(n_components=2, random_state=PCA_RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_cand_pca = pca.transform(X_candidates)
    var_ratio = pca.explained_variance_ratio_

    # Fig B1: PCA - clusters
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    sc = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP,
                    alpha=ALL_DATA_ALPHA, s=25, edgecolors=BLACK, linewidth=0.5)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_pca):
            ax.scatter(X_cand_pca[idx, 0], X_cand_pca[idx, 1], c=r['color'],
                       s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
    ax.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}1_PCA_clusters.tiff')
    plt.close()

    # Fig B2: PCA - uniform
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=ALL_DATA_COLOR, alpha=ALL_DATA_ALPHA,
               s=20, edgecolors=BLACK, linewidth=0.3)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_pca):
            ax.scatter(X_cand_pca[idx, 0], X_cand_pca[idx, 1], c=r['color'],
                       s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
    ax.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}2_PCA_uniform.tiff')
    plt.close()

    # Fig B3: PCA - highlight
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='#E8E8E8', alpha=0.6, s=15, edgecolors=BLACK, linewidth=0.2)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_pca):
            ax.scatter(X_cand_pca[idx, 0], X_cand_pca[idx, 1], c=r['color'],
                       s=90, marker='*', edgecolors=BLACK, linewidth=1.0, zorder=5)
    ax.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}3_PCA_highlight.tiff')
    plt.close()

    # t-SNE
    X_combined = np.vstack([X_train, X_candidates])
    perplexity = min(TSNE_PERPLEXITY, len(X_combined) - 1)
    tsne = TSNE(n_components=2, random_state=TSNE_RANDOM_STATE, perplexity=perplexity)
    X_combined_tsne = tsne.fit_transform(X_combined)
    X_train_tsne = X_combined_tsne[:len(X_train)]
    X_cand_tsne = X_combined_tsne[len(X_train):]

    # Fig B4: t-SNE - clusters
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP,
               alpha=ALL_DATA_ALPHA, s=25, edgecolors=BLACK, linewidth=0.5)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_tsne):
            ax.scatter(X_cand_tsne[idx, 0], X_cand_tsne[idx, 1], c=r['color'],
                       s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}4_tSNE_clusters.tiff')
    plt.close()

    # Fig B5: t-SNE - space
    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=ALL_DATA_COLOR, alpha=0.3,
               s=20, edgecolors=BLACK, linewidth=0.2)
    for r in candidate_results:
        idx = r['figure_id'] - 1
        if idx < len(X_cand_tsne):
            ax.scatter(X_cand_tsne[idx, 0], X_cand_tsne[idx, 1], c=r['color'],
                       s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}5_tSNE_space.tiff')
    plt.close()

    # Fig B6: Distribution
    fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))
    bars = ax.bar(['Within AD', 'Outside AD'], [within_count, outside_count],
                  color=[WITHIN_AD_COLOR, OUTSIDE_AD_COLOR], alpha=0.7,
                  edgecolor=BLACK, linewidth=0.8)
    for bar, count in zip(bars, [within_count, outside_count]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count),
                ha='center', va='bottom', fontsize=9, color=BLACK)
    ax.set_ylabel('Number of molecules')
    ax.set_xlabel('Reliability level')
    ax.tick_params(axis='both', colors=BLACK)
    set_axes_border_cluster(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{figures_dir}/{fig_prefix}6_distribution.tiff')
    plt.close()

    setup_cjche_style_larger()

    return candidate_results


# ============================================================
# 10. Main plotting function
# ============================================================

def create_dft_plots(data, ad_results, output_dir=OUTPUT_DIR):
    """创建DFT vs DNN对比图 + AD分析 + Bland-Altman"""

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    figures_no_names_dir = os.path.join(output_dir, 'figures_no_names')
    tables_dir = os.path.join(output_dir, 'tables')

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    if GENERATE_BOTH_VERSIONS:
        os.makedirs(figures_no_names_dir, exist_ok=True)

    setup_cjche_style_larger()

    ids = data['ids']
    dft_vals = data['dft']
    dnn_vals = data['dnn']
    errors = dnn_vals - dft_vals
    abs_errors = np.abs(errors)

    # 全局统计
    global_stats = compute_metrics(dft_vals, dnn_vals)
    print("\n" + "=" * 60)
    print("DFT vs DNN 统计结果")
    print("=" * 60)
    for key in ['R2', 'MAE', 'RMSE', 'MedAE', 'MaxAE', 'Bias', 'Pct10', 'Pct20', 'Pct30']:
        value = global_stats[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ===== Bland-Altman分析 =====
    ba_results = bland_altman_analysis(dft_vals, dnn_vals)
    print("\n" + "=" * 60)
    print("Bland-Altman Analysis (DFT vs DNN)")
    print("=" * 60)
    print(f"  Mean difference: {ba_results['mean_diff']:.4f} nm")
    print(f"  Std difference: {ba_results['std_diff']:.4f} nm")
    print(f"  95% LoA: [{ba_results['loa_lower']:.4f}, {ba_results['loa_upper']:.4f}] nm")

    # ===== 保存 Bland-Altman 分析表格 =====
    os.makedirs(tables_dir, exist_ok=True)

    # 表格1: Bland-Altman 汇总统计
    ba_summary_df = pd.DataFrame({
        'Metric': ['Mean Difference', 'Std Deviation', '95% LoA Lower', '95% LoA Upper', 'N'],
        'Value': [
            ba_results['mean_diff'],
            ba_results['std_diff'],
            ba_results['loa_lower'],
            ba_results['loa_upper'],
            ba_results['n']
        ],
        'Unit': ['nm', 'nm', 'nm', 'nm', 'samples']
    })
    ba_summary_path = os.path.join(tables_dir, 'Table_Bland_Altman_Summary.csv')
    ba_summary_df.to_csv(ba_summary_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存 Bland-Altman 汇总表: {ba_summary_path}")

    # 表格2: Bland-Altman 详细数据（每个分子的均值和差值）
    ba_detailed_df = pd.DataFrame({
        'ID': ids,
        'DFT_calculated_λmax': dft_vals,
        'DNN_predicted_λmax': dnn_vals,
        'Average_DFT_DNN': ba_results['mean'],
        'Difference_DFT_minus_DNN': ba_results['diff']
    })
    ba_detailed_path = os.path.join(tables_dir, 'Table_Bland_Altman_Detailed.csv')
    ba_detailed_df.to_csv(ba_detailed_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存 Bland-Altman 详细数据: {ba_detailed_path}")

    # 表格3: Bland-Altman 95% LoA 解释
    ba_interpretation_df = pd.DataFrame({
        'Description': [
            'Mean Difference',
            'Standard Deviation of Differences',
            '95% Limits of Agreement (LoA)',
            'Interpretation'
        ],
        'Value': [
            f"{ba_results['mean_diff']:.4f} nm",
            f"{ba_results['std_diff']:.4f} nm",
            f"[{ba_results['loa_lower']:.4f}, {ba_results['loa_upper']:.4f}] nm",
            f"95% of differences fall between {ba_results['loa_lower']:.4f} and {ba_results['loa_upper']:.4f} nm"
        ]
    })
    ba_interpretation_path = os.path.join(tables_dir, 'Table_Bland_Altman_Interpretation.csv')
    ba_interpretation_df.to_csv(ba_interpretation_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存 Bland-Altman 解释表: {ba_interpretation_path}")

    # 数据范围
    dft_min, dft_max = min(dft_vals), max(dft_vals)
    dnn_min, dnn_max = min(dnn_vals), max(dnn_vals)
    data_min = min(dft_min, dnn_min)
    data_max = max(dft_max, dnn_max)
    margin = (data_max - data_min) * 0.1
    tick_min = int(data_min / 50) * 50
    tick_max = int(data_max / 50) * 50 + 50

    # 按DFT排序
    sorted_idx = np.argsort(dft_vals)
    ids_sorted = [ids[i] for i in sorted_idx]
    dft_sorted = dft_vals[sorted_idx]
    dnn_sorted = dnn_vals[sorted_idx]
    errors_sorted = errors[sorted_idx]
    abs_errors_sorted = abs_errors[sorted_idx]

    x = np.arange(len(ids))
    width = 0.35
    max_val_plot = max(max(dft_sorted), max(dnn_sorted))

    # ===== 先运行LOO分析以获取df_loo（用于Fig20） =====
    df_loo = None
    if LOO_SENSITIVITY and len(ids) > 1:
        df_loo = loo_sensitivity_analysis(ids, dft_vals, dnn_vals, output_dir)

    # ===== 版本列表 =====
    versions = [
        (figures_dir, True, 'with_names'),
        (figures_no_names_dir, False, 'no_names')
    ] if GENERATE_BOTH_VERSIONS else [(figures_dir, True, 'with_names')]

    for target_dir, show_names, version_name in versions:
        if not os.path.exists(target_dir):
            continue
        print(f"\n  Generating {version_name} version...")

        # ========== Figure 1: 柱状对比图 ==========
        fig, ax = plt.subplots(figsize=(14 / 2.54, 10 / 2.54))

        ax.bar(x - width / 2, dft_sorted, width, label='DFT calculated',
               color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.bar(x + width / 2, dnn_sorted, width, label='DNN predicted',
               color='#D95F02', alpha=0.8, edgecolor='black', linewidth=1.2)

        for i, (dft, dnn) in enumerate(zip(dft_sorted, dnn_sorted)):
            ax.plot([i - width / 2, i + width / 2], [dft, dnn], 'gray', linewidth=1.0, alpha=0.5)
            ax.text(i - width / 2, dft + max_val_plot * 0.02, f'{dft:.1f}',
                    ha='center', va='bottom', fontsize=7, rotation=70)
            ax.text(i + width / 2, dnn + max_val_plot * 0.02, f'{dnn:.1f}',
                    ha='center', va='bottom', fontsize=7, rotation=70)

        ax.set_ylim(0, max_val_plot * 1.12)
        ax.set_xlabel('Candidate Molecule', fontsize=12)
        ax.set_ylabel('λmax (nm)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10, frameon=False, loc='upper left')
        set_axes_border(ax)
        plt.tight_layout()
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig1_Comparison_Bar.tiff'))
        plt.close(fig)

        # ========== Figure 2: 基础散点图 ==========
        fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))
        norm = Normalize(vmin=0, vmax=np.max(abs_errors))
        sc = ax.scatter(dft_vals, dnn_vals, c=abs_errors, cmap=plt.cm.coolwarm_r, norm=norm,
                        s=20, alpha=0.7, edgecolors='k', linewidth=0.8)

        ax.plot([data_min - margin, data_max + margin],
                [data_min - margin, data_max + margin], 'k--', linewidth=1.5, alpha=0.7)

        if show_names:
            for i, cid in enumerate(ids):
                ax.annotate(str(cid), (dft_vals[i], dnn_vals[i]),
                            xytext=(5, -8), textcoords='offset points',
                            fontsize=7, alpha=0.8, ha='left', va='top')

        ax.set_xlim(data_min - margin, data_max + margin)
        ax.set_ylim(data_min - margin, data_max + margin)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(tick_min, tick_max + 1, 50))
        ax.set_yticks(np.arange(tick_min, tick_max + 1, 50))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('DFT calculated λmax (nm)', fontsize=12)
        ax.set_ylabel('DNN predicted λmax (nm)', fontsize=12)
        ax.tick_params(labelsize=11)

        cbar = plt.colorbar(sc)
        cbar.set_label('Absolute error (nm)', fontsize=11)

        textstr = '\n'.join((f'$R^2$ = {global_stats["R2"]:.4f}',
                             f'MAE = {global_stats["MAE"]:.2f} nm'))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        set_axes_border(ax)
        plt.tight_layout()
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig2_Scatter_Basic.tiff'))
        plt.close(fig)

        # ========== Figures 3-8: 各AD方法的散点图 ==========
        fig_idx = 3
        for method_name, ad_data in ad_results.items():
            if len(ad_data['flags']) != len(ids):
                continue

            fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

            colors_ad = ['#2C7BB6' if flag else '#D7191C' for flag in ad_data['flags']]
            ax.scatter(dft_vals, dnn_vals, c=colors_ad, s=20, alpha=0.7,
                       edgecolors='k', linewidth=0.8)

            ax.plot([data_min - margin, data_max + margin],
                    [data_min - margin, data_max + margin], 'k--', linewidth=1.5, alpha=0.7)

            if show_names:
                for i, cid in enumerate(ids):
                    ax.annotate(str(cid), (dft_vals[i], dnn_vals[i]),
                                xytext=(5, -8), textcoords='offset points',
                                fontsize=7, alpha=0.8, ha='left', va='top')

            ax.set_xlim(data_min - margin, data_max + margin)
            ax.set_ylim(data_min - margin, data_max + margin)
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(tick_min, tick_max + 1, 50))
            ax.set_yticks(np.arange(tick_min, tick_max + 1, 50))
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('DFT calculated λmax (nm)', fontsize=12)
            ax.set_ylabel('DNN predicted λmax (nm)', fontsize=12)
            ax.tick_params(labelsize=11)

            # 图例：使用 "Within AD" 和 "Outside AD"
            legend_elements = [
                Patch(facecolor='#2C7BB6', label=f'Within AD ({ad_data["n_in"]})'),
                Patch(facecolor='#D7191C', label=f'Outside AD ({ad_data["n_out"]})')
            ]
            legend = ax.legend(handles=legend_elements, fontsize=9, frameon=False, loc='upper left')

            # 方法/阈值文本紧挨图例下方，黑色，无边框
            legend_box = legend.get_window_extent().transformed(ax.transAxes.inverted())
            legend_bottom = legend_box.y0
            legend_left = legend_box.x0
            info_text = f'Method: {ad_data["display_name"]}\nThreshold: {ad_data["threshold"]:.3f}'
            ax.text(legend_left, legend_bottom - 0.02, info_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='left',
                    color='black')

            set_axes_border(ax)
            plt.tight_layout()
            save_figure_tiff(fig, os.path.join(target_dir, f'Fig{fig_idx}_Scatter_{method_name}.tiff'))
            plt.close(fig)
            fig_idx += 1

        # ========== Figure 9: 误差分析 ==========
        fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))
        colors_err = ['red' if e > 0 else 'green' for e in errors_sorted]
        ax.bar(x, errors_sorted, color=colors_err, alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.axhline(y=0, color='black', linewidth=1.5)
        ax.axhline(y=20, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='±20 nm')
        ax.axhline(y=-20, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Candidate Molecule', fontsize=12)
        ax.set_ylabel('Prediction Error (DNN - DFT) (nm)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10, frameon=False, loc='lower right')
        for i, err in enumerate(errors_sorted):
            va = 'bottom' if err >= 0 else 'top'
            ax.text(i, err + (3 if err >= 0 else -3), f'{err:.1f}', ha='center', va=va, fontsize=7)
        set_axes_border(ax)
        plt.tight_layout()
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig9_Error_Analysis.tiff'))
        plt.close(fig)

        # ========== Figure 10: 绝对误差 ==========
        fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))
        ax.bar(x, abs_errors_sorted, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Candidate Molecule', fontsize=12)
        ax.set_ylabel('Absolute Prediction Error (nm)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=9)
        mean_ae = np.mean(abs_errors)
        ax.axhline(y=mean_ae, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_ae:.1f} nm')
        ax.legend(fontsize=10, frameon=False, loc='upper left')
        for i, err in enumerate(abs_errors_sorted):
            ax.text(i, err + 2, f'{err:.1f}', ha='center', va='bottom', fontsize=7)
        set_axes_border(ax)
        plt.tight_layout()
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig10_Absolute_Error.tiff'))
        plt.close(fig)

        # ========== Figure 11: 相对误差 ==========
        fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))
        rel_errors_sorted = [abs_errors_sorted[i] / dft_sorted[i] * 100 if dft_sorted[i] != 0 else 0
                             for i in range(len(dft_sorted))]
        ax.bar(x, rel_errors_sorted, color='lightgreen', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Candidate Molecule', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=9)
        mean_re = np.mean(rel_errors_sorted)
        ax.axhline(y=mean_re, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_re:.1f}%')
        ax.legend(fontsize=10, frameon=False, loc='upper left')
        for i, err in enumerate(rel_errors_sorted):
            ax.text(i, err + 1, f'{err:.1f}%', ha='center', va='bottom', fontsize=7)
        set_axes_border(ax)
        plt.tight_layout()
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig11_Relative_Error.tiff'))
        plt.close(fig)

        # ========== Figure 12: AD箱线图 ==========
        if ad_results:
            fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))
            first_method = list(ad_results.keys())[0]
            ad_data = ad_results[first_method]
            ad_in = ad_data['flags']
            ad_out = ~ad_in

            data_box = [abs_errors[ad_in], abs_errors[ad_out]]
            bp = ax.boxplot(data_box, labels=['Within AD', 'Outside AD'],
                            patch_artist=True, showmeans=True, meanline=True)

            colors_box = ['#2C7BB6', '#D7191C']
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Absolute Prediction Error (nm)', fontsize=12)
            ax.tick_params(labelsize=11)

            stats_in = compute_metrics(dft_vals[ad_in], dnn_vals[ad_in])
            stats_out = compute_metrics(dft_vals[ad_out], dnn_vals[ad_out])

            textstr = (f'Within AD (N={stats_in["N"]}): MAE={stats_in["MAE"]:.1f}, RMSE={stats_in["RMSE"]:.1f}\n'
                       f'Outside AD (N={stats_out["N"]}): MAE={stats_out["MAE"]:.1f}, RMSE={stats_out["RMSE"]:.1f}')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            set_axes_border(ax)
            plt.tight_layout()
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig12_AD_Boxplot.tiff'))
            plt.close(fig)

        # ========== Figure 13: RMSE对比 ==========
        fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))

        if ad_results:
            first_method = list(ad_results.keys())[0]
            ad_data = ad_results[first_method]
            ad_in = ad_data['flags']
            stats_in = compute_metrics(dft_vals[ad_in], dnn_vals[ad_in])
            stats_out = compute_metrics(dft_vals[~ad_in], dnn_vals[~ad_in])

            rmse_values = {'All': global_stats['RMSE'],
                           'Within AD': stats_in['RMSE'],
                           'Outside AD': stats_out['RMSE']}
            colors_bar = ['#2ca02c', '#2C7BB6', '#D7191C']
        else:
            rmse_values = {'All': global_stats['RMSE']}
            colors_bar = ['#2ca02c']

        bars = ax.bar(rmse_values.keys(), rmse_values.values(),
                      color=colors_bar[:len(rmse_values)],
                      edgecolor='black', linewidth=1.2, alpha=0.7, width=0.6)

        for bar, value in zip(bars, rmse_values.values()):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('RMSE (nm)', fontsize=12)
        ax.tick_params(labelsize=11)
        plt.xticks(rotation=0, ha='center')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        plt.tight_layout()
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig13_RMSE_Comparison.tiff'))
        plt.close(fig)

        # ========== Figure 14: Bland-Altman Plot ==========
        fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

        ax.scatter(ba_results['mean'], ba_results['diff'], s=30, alpha=0.7,
                   edgecolors='black', linewidth=0.8, color='steelblue')

        ax.axhline(y=ba_results['mean_diff'], color='red', linestyle='-', linewidth=1.5,
                   label=f'Mean diff = {ba_results["mean_diff"]:.2f} nm')

        ax.axhline(y=ba_results['loa_upper'], color='orange', linestyle='--', linewidth=1.5,
                   label=f'95% LoA: [{ba_results["loa_lower"]:.2f}, {ba_results["loa_upper"]:.2f}]')
        ax.axhline(y=ba_results['loa_lower'], color='orange', linestyle='--', linewidth=1.5)

        if show_names:
            for i, cid in enumerate(ids):
                ax.annotate(str(cid), (ba_results['mean'][i], ba_results['diff'][i]),
                            xytext=(5, -8), textcoords='offset points',
                            fontsize=7, alpha=0.8, ha='left', va='top')

        ax.set_xlabel('Average of DFT and DNN (nm)', fontsize=12)
        ax.set_ylabel('Difference (DFT - DNN) (nm)', fontsize=12)
        ax.tick_params(labelsize=11)

        # 图例放在左上角
        legend = ax.legend(fontsize=9, frameon=False, loc='upper left')

        # 统计信息紧挨图例下方，无边框，黑色
        legend_box = legend.get_window_extent().transformed(ax.transAxes.inverted())
        legend_bottom = legend_box.y0
        legend_left = legend_box.x0

        n = ba_results['n']
        textstr = f'N = {n}\nMean ± SD: {ba_results["mean_diff"]:.2f} ± {ba_results["std_diff"]:.2f} nm'
        ax.text(legend_left, legend_bottom - 0.02, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                color='black')

        set_axes_border(ax)
        plt.tight_layout()
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig14_Bland_Altman.tiff'))
        plt.close(fig)

        # ========== Figure 15: AD方法重叠热图 ==========
        if len(ad_results) > 1:
            fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))

            method_names = []
            overlap_matrix = []

            for m1 in ad_results.keys():
                method_names.append(ad_results[m1]['display_name'])
                row = []
                for m2 in ad_results.keys():
                    if len(ad_results[m1]['flags']) == len(ad_results[m2]['flags']):
                        overlap = np.sum(ad_results[m1]['flags'] & ad_results[m2]['flags'])
                        total = np.sum(ad_results[m1]['flags'])
                        row.append(overlap / total * 100 if total > 0 else 0)
                    else:
                        row.append(0)
                overlap_matrix.append(row)

            im = ax.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=100)

            ax.set_xticks(np.arange(len(method_names)))
            ax.set_yticks(np.arange(len(method_names)))
            ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(method_names, fontsize=9)

            for i in range(len(method_names)):
                for j in range(len(method_names)):
                    ax.text(j, i, f'{overlap_matrix[i][j]:.0f}%',
                            ha='center', va='center', color='black' if overlap_matrix[i][j] < 50 else 'white',
                            fontsize=8)

            ax.set_xlabel('AD Method', fontsize=12)
            ax.set_ylabel('AD Method', fontsize=12)
            cbar = plt.colorbar(im)
            cbar.set_label('Overlap (%)', fontsize=11)

            set_axes_border(ax)
            plt.tight_layout()
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig15_AD_Overlap_Heatmap.tiff'))
            plt.close(fig)

        # ========== LOO 图表 ==========
        if df_loo is not None:
            # Figure 16
            fig = create_loo_delta_r2_fig14(df_loo)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig16_LOO_Delta_R2.tiff'))
            plt.close(fig)

            # Figure 17
            fig = create_loo_delta_mae_fig15(df_loo)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig17_LOO_Delta_MAE.tiff'))
            plt.close(fig)

            # Figure 18
            fig = create_loo_influence_scatter_fig16(df_loo, show_names)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig18_LOO_Influence_Scatter.tiff'))
            plt.close(fig)

            # Figure 19
            fig = create_loo_cumulative_fig17(df_loo)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig19_LOO_Cumulative_Influence.tiff'))
            plt.close(fig)

            # ===== NEW: Figure 20 - Combined scatter + influence =====
            fig = create_fig18_combined(ids, dft_vals, dnn_vals, abs_errors, df_loo, global_stats, target_dir,
                                        show_names)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig20_Combined_Scatter_Influence.tiff'))
            plt.close(fig)

    # ===== 保存 LOO 表格（只保存一次） =====
    if df_loo is not None:
        loo_summary = {
            'Metric': ['R2', 'MAE', 'RMSE', 'MedAE', 'Bias'],
            'Original': [
                global_stats['R2'], global_stats['MAE'], global_stats['RMSE'],
                global_stats['MedAE'], global_stats['Bias']
            ],
            'Mean_LOO': [
                df_loo['R2_without'].mean(),
                df_loo['MAE_without'].mean(),
                df_loo['RMSE_without'].mean(),
                df_loo['Bias_without'].mean() if 'Bias_without' in df_loo.columns else np.nan,
                df_loo['Bias_without'].mean() if 'Bias_without' in df_loo.columns else np.nan
            ],
            'Std_LOO': [
                df_loo['R2_without'].std(),
                df_loo['MAE_without'].std(),
                df_loo['RMSE_without'].std(),
                df_loo['Bias_without'].std() if 'Bias_without' in df_loo.columns else np.nan,
                df_loo['Bias_without'].std() if 'Bias_without' in df_loo.columns else np.nan
            ],
            'Max_Delta': [
                df_loo['Delta_R2'].max(),
                df_loo['Delta_MAE'].max(),
                df_loo['Delta_RMSE'].max(),
                df_loo['Delta_Bias'].max() if 'Delta_Bias' in df_loo.columns else np.nan,
                df_loo['Delta_Bias'].max() if 'Delta_Bias' in df_loo.columns else np.nan
            ]
        }
        df_summary_loo = pd.DataFrame(loo_summary)
        loo_summary_path = os.path.join(tables_dir, 'Table_LOO_Summary.csv')
        df_summary_loo.to_csv(loo_summary_path, index=False, encoding='utf-8-sig')

    # ===== 保存结果表格 - All AD Methods（含评判标准） =====
    result_df = pd.DataFrame({
        'ID': ids,
        'DFT_calculated_λmax': dft_vals,
        'DNN_predicted_λmax': dnn_vals,
        'Prediction_Error': errors,
        'Absolute_Error': abs_errors
    })

    # 添加所有AD方法的结果
    for method_name, ad_data in ad_results.items():
        if len(ad_data['flags']) == len(ids):
            # AD分类标签
            result_df[f'AD_{method_name}'] = ad_data['flags']
            result_df[f'AD_{method_name}_dist'] = ad_data['distances']
            result_df[f'AD_{method_name}_threshold'] = ad_data['threshold']
            # 标签
            result_df[f'AD_{method_name}_label'] = ['Within AD' if f else 'Outside AD' for f in ad_data['flags']]
            # 评判标准
            result_df[f'AD_{method_name}_criteria'] = [
                f'Distance ({d:.3f}) ≤ Threshold ({ad_data["threshold"]:.3f})' if f
                else f'Distance ({d:.3f}) > Threshold ({ad_data["threshold"]:.3f})'
                for f, d in zip(ad_data['flags'], ad_data['distances'])
            ]

    # 保存完整结果表格（含所有AD方法）
    result_path = os.path.join(tables_dir, 'dft_dnn_comparison_all_AD.csv')
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存完整结果表格 (含所有AD方法): {result_path}")

    # 保存简版结果表格（仅第一个AD方法）
    result_df_simple = pd.DataFrame({
        'ID': ids,
        'DFT_calculated_λmax': dft_vals,
        'DNN_predicted_λmax': dnn_vals,
        'Prediction_Error': errors,
        'Absolute_Error': abs_errors
    })

    if ad_results:
        first_method = list(ad_results.keys())[0]
        ad_data = ad_results[first_method]
        result_df_simple['AD_inside'] = ad_data['flags']
        result_df_simple['AD_distance'] = ad_data['distances']
        result_df_simple['AD_threshold'] = ad_data['threshold']
        result_df_simple['AD_label'] = ['Within AD' if f else 'Outside AD' for f in ad_data['flags']]
        result_df_simple['AD_criteria'] = [
            f'Distance ({d:.3f}) ≤ Threshold ({ad_data["threshold"]:.3f})' if f
            else f'Distance ({d:.3f}) > Threshold ({ad_data["threshold"]:.3f})'
            for f, d in zip(ad_data['flags'], ad_data['distances'])
        ]

    result_simple_path = os.path.join(tables_dir, 'dft_dnn_comparison_with_AD.csv')
    result_df_simple.to_csv(result_simple_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存简版结果表格: {result_simple_path}")

    # ===== 分组统计表 =====
    stats_rows = [{
        'Group': 'All',
        'N': global_stats['N'],
        'R2': global_stats['R2'],
        'MAE': global_stats['MAE'],
        'RMSE': global_stats['RMSE'],
        'MedAE': global_stats['MedAE'],
        'MaxAE': global_stats['MaxAE'],
        'Bias': global_stats['Bias'],
        '%≤10nm': global_stats['Pct10'],
        '%≤20nm': global_stats['Pct20'],
        '%≤30nm': global_stats['Pct30']
    }]

    if ad_results:
        first_method = list(ad_results.keys())[0]
        ad_data = ad_results[first_method]
        ad_in = ad_data['flags']
        ad_out = ~ad_in

        stats_in = compute_metrics(dft_vals[ad_in], dnn_vals[ad_in])
        stats_out = compute_metrics(dft_vals[ad_out], dnn_vals[ad_out])

        stats_rows.append({
            'Group': 'Within AD',
            'N': stats_in['N'],
            'R2': stats_in['R2'],
            'MAE': stats_in['MAE'],
            'RMSE': stats_in['RMSE'],
            'MedAE': stats_in['MedAE'],
            'MaxAE': stats_in['MaxAE'],
            'Bias': stats_in['Bias'],
            '%≤10nm': stats_in['Pct10'],
            '%≤20nm': stats_in['Pct20'],
            '%≤30nm': stats_in['Pct30']
        })

        stats_rows.append({
            'Group': 'Outside AD',
            'N': stats_out['N'],
            'R2': stats_out['R2'],
            'MAE': stats_out['MAE'],
            'RMSE': stats_out['RMSE'],
            'MedAE': stats_out['MedAE'],
            'MaxAE': stats_out['MaxAE'],
            'Bias': stats_out['Bias'],
            '%≤10nm': stats_out['Pct10'],
            '%≤20nm': stats_out['Pct20'],
            '%≤30nm': stats_out['Pct30']
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(tables_dir, 'AD_group_statistics.csv')
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存分组统计: {stats_path}")

    # ===== AD方法对比汇总 =====
    if ad_results:
        print("\n" + "=" * 60)
        print("AD Method Comparison Summary")
        print("=" * 60)
        print(
            f"{'Method':<25} {'Within AD':<12} {'Outside AD':<12} {'MAE_in':<10} {'MAE_out':<10} {'RMSE_in':<10} {'RMSE_out':<10}")
        print("-" * 90)

        for method_name, ad_data in ad_results.items():
            if len(ad_data['flags']) == len(ids):
                ad_in = ad_data['flags']
                ad_out = ~ad_in
                stats_in = compute_metrics(dft_vals[ad_in], dnn_vals[ad_in])
                stats_out = compute_metrics(dft_vals[ad_out], dnn_vals[ad_out])
                print(f"{ad_data['display_name']:<25} {ad_data['n_in']:<12} {ad_data['n_out']:<12} "
                      f"{stats_in['MAE']:<10.2f} {stats_out['MAE']:<10.2f} "
                      f"{stats_in['RMSE']:<10.2f} {stats_out['RMSE']:<10.2f}")

    return global_stats


# ============================================================
# 11. Main function
# ============================================================

def main():
    print("=" * 80)
    print("候选分子 DFT vs DNN 综合分析")
    print("功能: 统计指标 + AD分类 + Bland-Altman + LOO + 聚类分析")
    print(f"生成两套图: {GENERATE_BOTH_VERSIONS}")
    print(f"LOO敏感性分析: {LOO_SENSITIVITY}")
    print(f"聚类分析 (所有描述符): {CLUSTER_ANALYSIS}")
    print(f"聚类分析 (35个描述符): {CLUSTER_ANALYSIS_35}")
    print("=" * 80)

    # 1. 加载数据
    data = load_candidate_data()
    if data is None:
        return

    # 2. 加载描述符
    # 2a. 加载所有描述符（用于AD分类）
    print("\n" + "-" * 60)
    print("加载所有描述符 (用于AD分类)")
    print("-" * 60)
    cand_desc_all, df_cand = load_candidate_descriptors()
    train_desc_all, df_train = load_training_descriptors()

    # 2b. 加载35个选定描述符（新增，用于聚类分析）
    print("\n" + "-" * 60)
    print("加载35个选定描述符 (用于聚类分析)")
    print("-" * 60)
    cand_desc_35, _ = load_candidate_descriptors_35()
    train_desc_35, _ = load_training_descriptors_35()

    # 3. AD分类（使用所有描述符）
    print("\n" + "=" * 80)
    print("AD分类 (使用所有描述符)")
    print("=" * 80)

    ad_results = {}
    if train_desc_all is not None and cand_desc_all is not None:
        if cand_desc_all.shape[0] != len(data['ids']):
            print(f"警告: 描述符样本数 ({cand_desc_all.shape[0]}) 与数据样本数 ({len(data['ids'])}) 不一致")
            min_samples = min(cand_desc_all.shape[0], len(data['ids']))
            cand_desc_all = cand_desc_all[:min_samples]
            data['ids'] = data['ids'][:min_samples]
            data['dft'] = data['dft'][:min_samples]
            data['dnn'] = data['dnn'][:min_samples]

        ad_results = apply_all_ad_methods(train_desc_all, cand_desc_all)
    else:
        print("跳过AD分类")

    # 4. 生成DFT对比图
    print("\n" + "=" * 80)
    print("生成DFT对比图")
    print("=" * 80)

    global_stats = create_dft_plots(data, ad_results, OUTPUT_DIR)

    # 5. 聚类分析（使用所有描述符）- 原有功能，不变
    cluster_results_all = None
    if CLUSTER_ANALYSIS and train_desc_all is not None and cand_desc_all is not None:
        print("\n" + "=" * 80)
        print("聚类分析 (使用所有描述符)")
        print("=" * 80)

        # 获取候选分子ID
        if df_cand is not None:
            if 'Name' in df_cand.columns:
                candidate_ids = df_cand['Name'].tolist()
            elif 'No.' in df_cand.columns:
                candidate_ids = df_cand['No.'].astype(str).tolist()
            else:
                candidate_ids = [f'Cand_{i + 1}' for i in range(len(cand_desc_all))]
        else:
            candidate_ids = [f'Cand_{i + 1}' for i in range(len(cand_desc_all))]

        if len(candidate_ids) > len(cand_desc_all):
            candidate_ids = candidate_ids[:len(cand_desc_all)]

        cluster_results_all = cluster_analysis(train_desc_all, cand_desc_all, candidate_ids, OUTPUT_DIR,
                                               fig_prefix='FigA')

    # 6. 聚类分析（使用35个选定描述符）- 新增功能
    cluster_results_35 = None
    if CLUSTER_ANALYSIS_35 and train_desc_35 is not None and cand_desc_35 is not None:
        print("\n" + "=" * 80)
        print("聚类分析 (使用35个选定描述符)")
        print("=" * 80)

        # 获取候选分子ID
        if df_cand is not None:
            if 'Name' in df_cand.columns:
                candidate_ids_35 = df_cand['Name'].tolist()
            elif 'No.' in df_cand.columns:
                candidate_ids_35 = df_cand['No.'].astype(str).tolist()
            else:
                candidate_ids_35 = [f'Cand_{i + 1}' for i in range(len(cand_desc_35))]
        else:
            candidate_ids_35 = [f'Cand_{i + 1}' for i in range(len(cand_desc_35))]

        if len(candidate_ids_35) > len(cand_desc_35):
            candidate_ids_35 = candidate_ids_35[:len(cand_desc_35)]

        cluster_results_35 = cluster_analysis_35(train_desc_35, cand_desc_35, candidate_ids_35, OUTPUT_DIR,
                                                 fig_prefix='FigB')

    # 7. 打印总结
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print(f"输出目录: {OUTPUT_DIR}/")
    if GENERATE_BOTH_VERSIONS:
        print("  figures/          - TIFF图 (带分子名称)")
        print("  figures_no_names/ - TIFF图 (不带分子名称)")
    else:
        print("  figures/          - TIFF图")
    print("  tables/           - CSV结果表格")

    if CLUSTER_ANALYSIS and cluster_results_all:
        within_count = sum(r['reliability'] == 'Within AD' for r in cluster_results_all)
        outside_count = sum(r['reliability'] == 'Outside AD' for r in cluster_results_all)
        print(f"\n  聚类分析结果 (所有描述符):")
        print(f"    Within AD: {within_count} 个分子")
        print(f"    Outside AD: {outside_count} 个分子")

    if CLUSTER_ANALYSIS_35 and cluster_results_35:
        within_count = sum(r['reliability'] == 'Within AD' for r in cluster_results_35)
        outside_count = sum(r['reliability'] == 'Outside AD' for r in cluster_results_35)
        print(f"\n  聚类分析结果 (35个描述符):")
        print(f"    Within AD: {within_count} 个分子")
        print(f"    Outside AD: {outside_count} 个分子")

    print("\n统计摘要:")
    for key in ['R2', 'MAE', 'RMSE', 'MedAE', 'MaxAE', 'Bias']:
        value = global_stats[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("=" * 80)


if __name__ == '__main__':
    main()