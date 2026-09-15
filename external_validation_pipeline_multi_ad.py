# -*- coding: utf-8 -*-
"""
External Validation Pipeline: Prediction + Validation + Multiple AD Methods
Supports 6 AD methods with comparative analysis
Style consistent with dnn_model_35_result_more

DNN模型参数直接从训练脚本中集成
描述符文件从 descriptor_results/ 子目录读取
修复：处理空数组导致的 r2_score 错误
新增：留一敏感性分析 (LOO Sensitivity Analysis)

图表控制：
- GENERATE_BOTH_VERSIONS = True: 同时生成带名称和不带名称两套图
- GENERATE_BOTH_VERSIONS = False: 仅生成带名称版本

Fig2-7修改：
- 方法/阈值文本紧挨图例下方
- 去除白色边框
- 文本改为黑色
- 图例显示 "Within AD" / "Outside AD"

新增：
- 表格添加评判标准列
- Fig18: 合并Fig1和Fig16信息（散点图 + 气泡图 + 合并文本框）
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ============================================================
# 1. Configuration
# ============================================================

# Get current script directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Current directory: {CURRENT_DIR}")

# ========== 图表控制开关 ==========
GENERATE_BOTH_VERSIONS = True  # True: 同时生成带名称和不带名称两套图, False: 仅生成带名称

print(f"\nChart settings:")
print(f"  Generate both versions (with/without names): {GENERATE_BOTH_VERSIONS}")

# 35 descriptors used by the model
SELECTED_BITS = [
    906, 1336, 334, 1310, 1348, 1406, 1437, 345, 218, 230,
    449, 510, 299, 3, 1547, 566, 1418, 1069, 232, 264,
    1076, 1077, 322, 1313, 1294, 1064, 1329, 1146, 1477, 1359,
    805, 1394, 1358, 1573, 485
]

# DNN Model parameters (直接从训练脚本复制)
FINAL_PARAMS = {
    'hidden_layer_sizes': (80, 70, 60, 50, 40, 30),
    'activation': 'tanh',
    'solver': 'adam',
    'alpha': 0.1,
    'batch_size': 8,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.00015,
    'max_iter': 15000,
    'early_stopping': True,
    'validation_fraction': 0.22,
    'n_iter_no_change': 40,
    'tol': 5e-07,
    'random_state': 123
}

# File configuration
ORIG_FILE = os.path.join(CURRENT_DIR, 'my_test_dyes.xlsx')

# 描述符文件在 descriptor_results 子目录中
DESC_DIR = os.path.join(CURRENT_DIR, 'descriptor_results')
DESC_FILE_FULL = os.path.join(DESC_DIR, 'my_test_dyes_with_Md.xlsx')
DESC_FILE_MODEL = os.path.join(DESC_DIR, 'my_test_dyes_model_descriptors.xlsx')

if os.path.exists(DESC_FILE_FULL):
    DESC_FILE = DESC_FILE_FULL
    print(f"✓ Using full descriptor file: {os.path.basename(DESC_FILE)}")
elif os.path.exists(DESC_FILE_MODEL):
    DESC_FILE = DESC_FILE_MODEL
    print(f"✓ Using model descriptor file (35 features): {os.path.basename(DESC_FILE)}")
else:
    DESC_FILE = DESC_FILE_FULL
    print(f"Warning: No descriptor file found in {DESC_DIR}!")

TRAIN_DESC_FILE = os.path.join(CURRENT_DIR, 'data_w_Md_bo.xlsx')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'external_validation_results_multi_ad')

print(f"\nFile paths:")
print(f"  Original file: {os.path.basename(ORIG_FILE)}")
print(f"  Descriptor file: {os.path.basename(DESC_FILE)}")
print(f"  Training descriptor file: {os.path.basename(TRAIN_DESC_FILE)}")
print(f"\nFile existence:")
print(f"  Original file exists: {os.path.exists(ORIG_FILE)}")
print(f"  Descriptor file exists: {os.path.exists(DESC_FILE)}")
print(f"  Training descriptor file exists: {os.path.exists(TRAIN_DESC_FILE)}")

# AD method selection
AD_METHODS = {
    'euclidean_nearest': True,
    'euclidean_center': True,
    'mahalanobis': True,
    'leverage': True,
    'knn_density': True,
    'isolation_forest': True,
}

# LOO Sensitivity Analysis settings
LOO_SENSITIVITY = True


# ============================================================
# 2. Plotting style
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


def save_figure_tiff(fig, filepath, dpi=300):
    fig.savefig(filepath, format='tiff', dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")


def set_axes_border(ax, linewidth=1.5):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


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


# ============================================================
# 3. Prediction module
# ============================================================

def load_or_train_model():
    print("\n" + "=" * 60)
    print("Training DNN model from training data...")
    print("=" * 60)

    if not os.path.exists(TRAIN_DESC_FILE):
        print(f"Error: Training descriptor file not found: {TRAIN_DESC_FILE}")
        return None, None, None

    df_all = pd.read_excel(TRAIN_DESC_FILE, engine='openpyxl')
    print(f"✓ Loaded training data: {len(df_all)} molecules")

    train_col = find_column(df_all, ['Training set/Testing set', 'Set', 'Type'])
    if train_col is not None:
        train_mask = df_all[train_col].astype(str).str.lower().str.contains('training')
        df_train = df_all[train_mask].copy()
        print(f"✓ Training set: {len(df_train)} molecules")
    else:
        print("Warning: Training set column not found, using all data")
        df_train = df_all.copy()

    desc_cols = [f'Md_{i}' for i in SELECTED_BITS]
    for c in desc_cols:
        if c not in df_train.columns:
            df_train[c] = 0.0

    target_col = find_column(df_train, ['bo', 'target', 'y', 'λmax'])
    if target_col is None:
        print("Error: Target column not found in training data")
        return None, None, None

    X_train_raw = df_train[desc_cols].values.astype(float)
    y_train_raw = df_train[target_col].values.astype(float)

    print(f"✓ X_train shape: {X_train_raw.shape}")
    print(f"✓ y_train shape: {y_train_raw.shape}")

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()

    print("\nTraining DNN model...")
    print(f"  Hidden layers: {FINAL_PARAMS['hidden_layer_sizes']}")
    print(f"  Activation: {FINAL_PARAMS['activation']}")
    print(f"  Max iterations: {FINAL_PARAMS['max_iter']}")

    model = MLPRegressor(**FINAL_PARAMS)
    model.fit(X_train, y_train_scaled)
    print("✓ Model training completed!")

    return model, scaler_X, scaler_y


def predict_external(model, scaler_X, scaler_y):
    print("\n" + "=" * 80)
    print("Step 1: DNN Model Prediction")
    print("=" * 80)

    if not os.path.exists(DESC_FILE):
        print(f"Error: Descriptor file not found: {DESC_FILE}")
        return None, None, None

    df_desc = pd.read_excel(DESC_FILE, engine='openpyxl')
    print(f"✓ Loaded descriptor file: {len(df_desc)} molecules")
    print(f"  File: {os.path.basename(DESC_FILE)}")

    desc_cols = [f'Md_{i}' for i in SELECTED_BITS]
    existing_cols = [c for c in desc_cols if c in df_desc.columns]
    missing_cols = [c for c in desc_cols if c not in df_desc.columns]

    if missing_cols:
        print(f"  Warning: {len(missing_cols)} descriptor columns missing, filling with zeros")
        for c in missing_cols:
            df_desc[c] = 0.0

    print(f"  Using {len(existing_cols)} descriptor columns")

    X_raw = df_desc[desc_cols].values.astype(float)
    X_scaled = scaler_X.transform(X_raw)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    print(f"✓ Prediction completed, range: {y_pred.min():.1f} ~ {y_pred.max():.1f} nm")

    if not os.path.exists(ORIG_FILE):
        print(f"Warning: Original file not found: {ORIG_FILE}")
        id_col = find_column(df_desc, ['Name', 'ID', 'Compound'])
        ids = df_desc[id_col].values if id_col else np.arange(1, len(y_pred) + 1)
        exp_vals = np.full(len(ids), np.nan)
    else:
        df_orig = pd.read_excel(ORIG_FILE, engine='openpyxl')
        id_col = find_column(df_orig, ['ID', 'No.', '编号', 'Compound', 'Name'])
        exp_col = find_column(df_orig, ['Exp', 'Experimental', 'λmax_exp', '实验值', 'λmax'])

        ids = df_orig[id_col].values if id_col else np.arange(1, len(df_orig) + 1)
        exp_vals = np.array([float(x) if pd.notna(x) else np.nan for x in df_orig[exp_col]]) if exp_col else np.full(
            len(df_orig), np.nan)

        if len(ids) == len(y_pred):
            pred_vals = y_pred
        else:
            min_len = min(len(ids), len(y_pred))
            pred_vals = np.full(len(ids), np.nan)
            pred_vals[:min_len] = y_pred[:min_len]

        df_orig['DNN_pred'] = pred_vals
        output_file = os.path.join(CURRENT_DIR, 'my_test_dyes_with_pred.xlsx')
        df_orig.to_excel(output_file, index=False)
        print(f"✓ Saved predictions to: {output_file}")

    valid = ~np.isnan(exp_vals) & ~np.isnan(pred_vals)
    return np.array(ids)[valid], exp_vals[valid], pred_vals[valid]


# ============================================================
# 4. Multiple AD Methods
# ============================================================

def load_training_descriptors():
    if not os.path.exists(TRAIN_DESC_FILE):
        print(f"Warning: Training descriptor file not found: {TRAIN_DESC_FILE}")
        return None

    df_all = pd.read_excel(TRAIN_DESC_FILE, engine='openpyxl')
    print(f"✓ Loaded {len(df_all)} molecules from {TRAIN_DESC_FILE}")

    train_col = find_column(df_all, ['Training set/Testing set', 'Set', 'Type'])

    if train_col is None:
        print("Warning: Training set column not found, using all data as training set")
        df_train = df_all.copy()
    else:
        train_mask = df_all[train_col].astype(str).str.lower().str.contains('training')
        df_train = df_all[train_mask].copy()
        print(f"  Training set: {len(df_train)} molecules")
        print(f"  Testing set: {len(df_all) - len(df_train)} molecules")

    desc_cols = [f'Md_{i}' for i in SELECTED_BITS]
    for c in desc_cols:
        if c not in df_train.columns:
            df_train[c] = 0.0

    print(f"✓ Loaded training descriptors: {len(df_train)} molecules")
    return df_train[desc_cols].values


def load_external_descriptors():
    if not os.path.exists(DESC_FILE):
        print(f"Warning: External descriptor file not found: {DESC_FILE}")
        return None

    df = pd.read_excel(DESC_FILE, engine='openpyxl')
    desc_cols = [f'Md_{i}' for i in SELECTED_BITS]
    for c in desc_cols:
        if c not in df.columns:
            df[c] = 0.0

    print(f"✓ Loaded external descriptors: {len(df)} molecules")
    return df[desc_cols].values


# ---------- AD Method 1: Euclidean distance to nearest neighbor ----------
def ad_euclidean_nearest(train_X, test_X, percentile=95):
    dist = cdist(test_X, train_X, metric='euclidean')
    min_dist = np.min(dist, axis=1)

    train_dist = cdist(train_X, train_X, metric='euclidean')
    np.fill_diagonal(train_dist, np.nan)
    threshold = np.nanpercentile(train_dist, percentile)

    return min_dist <= threshold, min_dist, threshold, 'Euclidean-Nearest'


# ---------- AD Method 2: Euclidean distance to center ----------
def ad_euclidean_center(train_X, test_X, percentile=95):
    center = np.mean(train_X, axis=0)
    dist = np.linalg.norm(test_X - center, axis=1)

    train_center = np.mean(train_X, axis=0)
    train_dist = np.linalg.norm(train_X - train_center, axis=1)
    threshold = np.percentile(train_dist, percentile)

    return dist <= threshold, dist, threshold, 'Euclidean-Center'


# ---------- AD Method 3: Mahalanobis distance ----------
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


# ---------- AD Method 4: Leverage ----------
def ad_leverage(train_X, test_X):
    n_train = train_X.shape[0]
    p = train_X.shape[1]

    H_test = test_X @ np.linalg.pinv(train_X.T @ train_X) @ test_X.T
    leverage = np.diag(H_test)

    threshold = 3 * (p + 1) / n_train

    return leverage <= threshold, leverage, threshold, 'Leverage'


# ---------- AD Method 5: KNN density ----------
def ad_knn_density(train_X, test_X, k=5, percentile=95):
    knn = NearestNeighbors(n_neighbors=min(k, len(train_X)))
    knn.fit(train_X)

    train_dist, _ = knn.kneighbors(train_X)
    train_mean_dist = np.mean(train_dist, axis=1)
    threshold = np.percentile(train_mean_dist, percentile)

    test_dist, _ = knn.kneighbors(test_X)
    test_mean_dist = np.mean(test_dist, axis=1)

    return test_mean_dist <= threshold, test_mean_dist, threshold, f'KNN-k={k}'


# ---------- AD Method 6: Isolation Forest ----------
def ad_isolation_forest(train_X, test_X, contamination=0.05):
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    iso_forest.fit(train_X)

    pred = iso_forest.predict(test_X)
    scores = iso_forest.decision_function(test_X)

    threshold = 0.0

    return pred == 1, scores, threshold, 'IsolationForest'


def apply_all_ad_methods(train_X, test_X):
    results = {}

    if train_X is None or test_X is None:
        return results

    n_samples = min(len(test_X), 1000)
    if len(test_X) > n_samples:
        test_X_subset = test_X[:n_samples]
    else:
        test_X_subset = test_X

    method_configs = [
        ('euclidean_nearest', ad_euclidean_nearest, {}),
        ('euclidean_center', ad_euclidean_center, {}),
        ('mahalanobis', ad_mahalanobis, {'alpha': 0.05}),
        ('leverage', ad_leverage, {}),
        ('knn_density', ad_knn_density, {'k': 5, 'percentile': 95}),
        ('isolation_forest', ad_isolation_forest, {'contamination': 0.05}),
    ]

    print("\nApplying AD methods...")
    for method_name, method_func, kwargs in method_configs:
        if AD_METHODS.get(method_name, False):
            try:
                ad_flags, distances, threshold, display_name = method_func(train_X, test_X_subset, **kwargs)
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
                print(f"  ✗ {method_name} failed: {e}")

    return results


# ============================================================
# 5. Metrics calculation
# ============================================================

def compute_metrics(y_true, y_pred):
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

        pct_change_r2 = (delta_r2 / abs(original_metrics['R2']) * 100) if original_metrics['R2'] != 0 else 0
        pct_change_mae = (delta_mae / original_metrics['MAE'] * 100) if original_metrics['MAE'] != 0 else 0

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
            'Pct_Change_R2': pct_change_r2,
            'Pct_Change_MAE': pct_change_mae
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
# 7. Plotting functions
# ============================================================

def create_scatter_fig1(ids, exp_vals, pred_vals, abs_errors, global_stats,
                        target_dir, show_names=True):
    """Figure 1: Basic scatter plot"""
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))
    norm = Normalize(vmin=0, vmax=np.max(abs_errors))
    sc = ax.scatter(exp_vals, pred_vals, c=abs_errors, cmap=plt.cm.coolwarm_r, norm=norm,
                    s=20, alpha=0.7, edgecolors='k', linewidth=0.8)

    min_val = min(min(exp_vals), min(pred_vals)) - 10
    max_val = max(max(exp_vals), max(pred_vals)) + 10
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)

    if show_names:
        for i, cid in enumerate(ids):
            offset_x = 5 + np.random.randint(-2, 2)
            offset_y = -8 + np.random.randint(-2, 2)
            ax.annotate(str(cid), (exp_vals[i], pred_vals[i]),
                        xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=7, alpha=0.8, ha='left', va='top')

    ax.set_xlabel('Exp. λmax (nm)', fontsize=12)
    ax.set_ylabel('Pre. λmax (nm)', fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute error (nm)', fontsize=11)

    textstr = '\n'.join((f'$R^2$ = {global_stats["R2"]:.4f}',
                         f'MAE = {global_stats["MAE"]:.4f} nm',
                         f'RMSE = {global_stats["RMSE"]:.4f} nm'))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    set_axes_border(ax)
    plt.tight_layout()
    return fig


def create_scatter_ad(ids, exp_vals, pred_vals, ad_data, min_val, max_val,
                      target_dir, show_names=True):
    """Figures 2-7: Scatter plots with AD method"""
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    colors = ['#2C7BB6' if flag else '#D7191C' for flag in ad_data['flags'][:len(ids)]]
    ax.scatter(exp_vals, pred_vals, c=colors, s=20, alpha=0.7, edgecolors='k', linewidth=0.8)

    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)

    if show_names:
        for i, cid in enumerate(ids):
            offset_x = 5 + np.random.randint(-2, 2)
            offset_y = -8 + np.random.randint(-2, 2)
            ax.annotate(str(cid), (exp_vals[i], pred_vals[i]),
                        xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=7, alpha=0.8, ha='left', va='top')

    ax.set_xlabel('Exp. λmax (nm)', fontsize=12)
    ax.set_ylabel('Pre. λmax (nm)', fontsize=12)
    ax.tick_params(labelsize=11)

    # ===== 图例：使用 "Within AD" 和 "Outside AD" =====
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
    return fig


def create_bar_chart_fig8(ids, exp_vals, pred_vals):
    """Figure 8: Bar chart comparison"""
    fig, ax = plt.subplots(figsize=(14 / 2.54, 10 / 2.54))

    x = np.arange(len(ids))
    width = 0.35

    ax.bar(x - width / 2, exp_vals, width, label='Experimental',
           color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.bar(x + width / 2, pred_vals, width, label='DNN predicted',
           color='#D95F02', alpha=0.8, edgecolor='black', linewidth=1.2)

    for i, (exp, pred) in enumerate(zip(exp_vals, pred_vals)):
        ax.plot([i - width / 2, i + width / 2], [exp, pred], 'gray', linewidth=1.0, alpha=0.5)
        ax.text(i - width / 2, exp + 3, f'{exp:.1f}',
                ha='center', va='bottom', fontsize=7, rotation=45)
        ax.text(i + width / 2, pred + 3, f'{pred:.1f}',
                ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('External Molecule', fontsize=12)
    ax.set_ylabel('λmax (nm)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 900)
    ax.legend(fontsize=10, frameon=False, loc='upper left')
    set_axes_border(ax)
    plt.tight_layout()
    return fig


def create_error_analysis_fig9(ids, errors):
    """Figure 9: Error analysis"""
    fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))
    x = np.arange(len(ids))
    colors_err = ['red' if e > 0 else 'green' for e in errors]
    ax.bar(x, errors, color=colors_err, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.axhline(y=20, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='±20 nm')
    ax.axhline(y=-20, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('External Molecule', fontsize=12)
    ax.set_ylabel('Prediction Error (DNN - Exp) (nm)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10, frameon=False, loc='lower right')
    for i, err in enumerate(errors):
        va = 'bottom' if err >= 0 else 'top'
        ax.text(i, err + (3 if err >= 0 else -3), f'{err:.1f}',
                ha='center', va=va, fontsize=7, rotation=45)
    set_axes_border(ax)
    plt.tight_layout()
    return fig


def create_absolute_error_fig10(ids, abs_errors):
    """Figure 10: Absolute error"""
    fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))
    x = np.arange(len(ids))
    ax.bar(x, abs_errors, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('External Molecule', fontsize=12)
    ax.set_ylabel('Absolute Prediction Error (nm)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=9)
    mean_ae = np.mean(abs_errors)
    ax.axhline(y=mean_ae, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_ae:.1f} nm')
    ax.legend(fontsize=10, frameon=False, loc='upper left')
    for i, err in enumerate(abs_errors):
        ax.text(i, err + 2, f'{err:.1f}', ha='center', va='bottom', fontsize=7, rotation=45)
    set_axes_border(ax)
    plt.tight_layout()
    return fig


def create_mae_distribution_fig11(abs_errors):
    """Figure 11: MAE distribution"""
    fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))
    bin_width = 50
    max_error = np.max(abs_errors)
    max_bin = int(np.ceil(max_error / bin_width) * bin_width)
    bins = np.arange(0, max_bin + bin_width, bin_width)
    counts, _ = np.histogram(abs_errors, bins=bins)
    x_pos = bins[:-1] + bin_width * 0.5
    ax.bar(x_pos, counts, width=bin_width * 0.6, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Absolute Error (nm)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(np.arange(0, max_bin + bin_width, bin_width))
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    return fig


def create_rmse_comparison_fig12(global_stats, ad_results, exp_vals, pred_vals):
    """Figure 12: RMSE comparison"""
    fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))

    rmse_values = {'All': global_stats['RMSE']}
    for method_name, ad_data in ad_results.items():
        if len(ad_data['flags']) == len(exp_vals):
            ad_in = ad_data['flags']
            if np.sum(ad_in) > 0:
                stats_in = compute_metrics(exp_vals[ad_in], pred_vals[ad_in])
                rmse_values[f'Within AD ({ad_data["display_name"]})'] = stats_in['RMSE']

    colors_bar = ['#2ca02c'] + ['#2C7BB6'] * (len(rmse_values) - 1)
    bars = ax.bar(rmse_values.keys(), rmse_values.values(), color=colors_bar[:len(rmse_values)],
                  edgecolor='black', linewidth=1.2, alpha=0.7, width=0.6)

    for bar, value in zip(bars, rmse_values.values()):
        if not np.isnan(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9, rotation=45)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('RMSE (nm)', fontsize=12)
    ax.tick_params(labelsize=11)
    plt.xticks(rotation=45, ha='right')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    return fig


def create_ad_overlap_heatmap_fig13(ad_results):
    """Figure 13: AD overlap heatmap"""
    if len(ad_results) <= 1:
        return None

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
                    ha='center', va='center', color='black' if overlap_matrix[i][j] < 50 else 'white', fontsize=8)

    ax.set_xlabel('AD Method', fontsize=12)
    ax.set_ylabel('AD Method', fontsize=12)

    cbar = plt.colorbar(im)
    cbar.set_label('Overlap (%)', fontsize=11)

    set_axes_border(ax)
    plt.tight_layout()
    return fig


def create_loo_delta_r2_fig14(df_loo):
    """Figure 14: LOO Delta R²"""
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
    """Figure 15: LOO Delta MAE"""
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
    """Figure 16: LOO influence scatter plot"""
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

    ax.set_xlabel('Experimental λmax (nm)', fontsize=12)
    ax.set_ylabel('Predicted λmax (nm)', fontsize=12)
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
    """Figure 17: LOO cumulative influence"""
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
# 8. NEW: Figure 18 - Combined Scatter + Influence Plot
# ============================================================

def create_fig18_combined(ids, exp_vals, pred_vals, abs_errors, df_loo, global_stats, target_dir, show_names=True):
    """
    Figure 18: Combined scatter plot with influence size
    Merges Fig1 (scatter) and Fig16 (influence scatter) information
    """
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    # 气泡图：点的大小表示影响力
    if df_loo is not None and len(df_loo) > 0:
        # 使用LOO的Delta_R2作为影响力
        sizes = np.abs(df_loo['Delta_R2']) * 500 + 20
        colors = np.abs(df_loo['Error'])
        exp_vals_loo = df_loo['Experimental'].values
        pred_vals_loo = df_loo['Predicted'].values

        scatter = ax.scatter(exp_vals_loo, pred_vals_loo,
                             s=sizes, c=colors, cmap='coolwarm_r', alpha=0.7,
                             edgecolors='black', linewidth=0.8)

        # 标注分子名称
        if show_names:
            offsets = [(5, -8), (8, -5), (2, -12), (12, -3), (-5, -10), (5, -15)]
            for idx, row in df_loo.iterrows():
                offset_idx = idx % len(offsets)
                ox, oy = offsets[offset_idx]
                ax.annotate(str(row['Sample_ID']), (row['Experimental'], row['Predicted']),
                            xytext=(ox, oy), textcoords='offset points',
                            fontsize=7, alpha=0.8, ha='left', va='top')

        # 颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('Absolute Error (nm)', fontsize=11)
    else:
        # 如果没有LOO数据，回退到普通散点图
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

    # 合并统计信息和影响力说明
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
# 9. Main plotting function
# ============================================================

def create_external_plots(ids, exp_vals, pred_vals, ad_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    figures_no_names_dir = os.path.join(output_dir, 'figures_no_names')
    tables_dir = os.path.join(output_dir, 'tables')

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    if GENERATE_BOTH_VERSIONS:
        os.makedirs(figures_no_names_dir, exist_ok=True)

    setup_cjche_style_larger()

    errors = pred_vals - exp_vals
    abs_errors = np.abs(errors)

    # Global statistics
    global_stats = compute_metrics(exp_vals, pred_vals)
    print("\n" + "=" * 60)
    print("External Validation Statistics")
    print("=" * 60)
    for key in ['R2', 'MAE', 'RMSE', 'MedAE', 'MaxAE', 'Bias', 'Pct10', 'Pct20', 'Pct30']:
        value = global_stats[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    min_val = min(min(exp_vals), min(pred_vals)) - 10
    max_val = max(max(exp_vals), max(pred_vals)) + 10

    # ===== 生成所有图表（两套） =====
    versions = [
        (figures_dir, True, 'with_names'),
        (figures_no_names_dir, False, 'no_names')
    ] if GENERATE_BOTH_VERSIONS else [(figures_dir, True, 'with_names')]

    # 先运行LOO分析以获取df_loo（用于Fig18）
    df_loo = None
    if LOO_SENSITIVITY and len(ids) > 1:
        df_loo = loo_sensitivity_analysis(ids, exp_vals, pred_vals, output_dir)

    for target_dir, show_names, version_name in versions:
        if not os.path.exists(target_dir):
            continue
        print(f"\n  Generating {version_name} version...")

        # Figure 1
        fig = create_scatter_fig1(ids, exp_vals, pred_vals, abs_errors, global_stats, target_dir, show_names)
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig1_Scatter_Basic.tiff'))
        plt.close(fig)

        # Figures 2-7
        fig_idx = 2
        for method_name, ad_data in ad_results.items():
            if len(ad_data['flags']) != len(ids):
                continue
            fig = create_scatter_ad(ids, exp_vals, pred_vals, ad_data, min_val, max_val,
                                    target_dir, show_names)
            save_figure_tiff(fig, os.path.join(target_dir, f'Fig{fig_idx}_Scatter_{method_name}.tiff'))
            plt.close(fig)
            fig_idx += 1

        # Figure 8
        fig = create_bar_chart_fig8(ids, exp_vals, pred_vals)
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig8_Comparison_Bar.tiff'))
        plt.close(fig)

        # Figure 9
        fig = create_error_analysis_fig9(ids, errors)
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig9_Error_Analysis.tiff'))
        plt.close(fig)

        # Figure 10
        fig = create_absolute_error_fig10(ids, abs_errors)
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig10_Absolute_Error.tiff'))
        plt.close(fig)

        # Figure 11
        fig = create_mae_distribution_fig11(abs_errors)
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig11_MAE_Distribution.tiff'))
        plt.close(fig)

        # Figure 12
        fig = create_rmse_comparison_fig12(global_stats, ad_results, exp_vals, pred_vals)
        save_figure_tiff(fig, os.path.join(target_dir, 'Fig12_RMSE_Comparison.tiff'))
        plt.close(fig)

        # Figure 13
        fig = create_ad_overlap_heatmap_fig13(ad_results)
        if fig is not None:
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig13_AD_Overlap_Heatmap.tiff'))
            plt.close(fig)

        # ===== LOO 图表 =====
        if df_loo is not None:
            # Figure 14
            fig = create_loo_delta_r2_fig14(df_loo)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig14_LOO_Delta_R2.tiff'))
            plt.close(fig)

            # Figure 15
            fig = create_loo_delta_mae_fig15(df_loo)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig15_LOO_Delta_MAE.tiff'))
            plt.close(fig)

            # Figure 16
            fig = create_loo_influence_scatter_fig16(df_loo, show_names)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig16_LOO_Influence_Scatter.tiff'))
            plt.close(fig)

            # Figure 17
            fig = create_loo_cumulative_fig17(df_loo)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig17_LOO_Cumulative_Influence.tiff'))
            plt.close(fig)

            # ===== NEW: Figure 18 - Combined scatter + influence =====
            fig = create_fig18_combined(ids, exp_vals, pred_vals, abs_errors, df_loo, global_stats, target_dir,
                                        show_names)
            save_figure_tiff(fig, os.path.join(target_dir, 'Fig18_Combined_Scatter_Influence.tiff'))
            plt.close(fig)

    # ===== 保存 LOO 表格（只保存一次） =====
    if df_loo is not None:
        loo_path = os.path.join(tables_dir, 'Table_LOO_Sensitivity_Analysis.csv')
        df_loo.to_csv(loo_path, index=False, encoding='utf-8-sig')

        summary_loo = {
            'Metric': ['R2', 'MAE', 'RMSE', 'MedAE', 'Bias'],
            'Original': [
                global_stats['R2'],
                global_stats['MAE'],
                global_stats['RMSE'],
                global_stats['MedAE'],
                global_stats['Bias']
            ],
            'Mean_LOO': [
                df_loo['R2_without'].mean(),
                df_loo['MAE_without'].mean(),
                df_loo['RMSE_without'].mean(),
                df_loo['MedAE_without'].mean() if 'MedAE_without' in df_loo.columns else np.nan,
                df_loo['Bias_without'].mean()
            ],
            'Std_LOO': [
                df_loo['R2_without'].std(),
                df_loo['MAE_without'].std(),
                df_loo['RMSE_without'].std(),
                df_loo['MedAE_without'].std() if 'MedAE_without' in df_loo.columns else np.nan,
                df_loo['Bias_without'].std()
            ],
            'Max_Delta': [
                df_loo['Delta_R2'].max(),
                df_loo['Delta_MAE'].max(),
                df_loo['Delta_RMSE'].max(),
                df_loo['Delta_Bias'].max() if 'Delta_Bias' in df_loo.columns else np.nan,
                df_loo['Delta_Bias'].max() if 'Delta_Bias' in df_loo.columns else np.nan
            ]
        }
        df_summary_loo = pd.DataFrame(summary_loo)
        loo_summary_path = os.path.join(tables_dir, 'Table_LOO_Summary.csv')
        df_summary_loo.to_csv(loo_summary_path, index=False, encoding='utf-8-sig')

    # ===== 保存结果表格（含评判标准） =====
    result_df = pd.DataFrame({
        'ID': ids,
        'Experimental_λmax': exp_vals,
        'DNN_predicted_λmax': pred_vals,
        'Prediction_Error': errors,
        'Absolute_Error': abs_errors
    })

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

    result_path = os.path.join(tables_dir, 'external_validation_all_AD.csv')
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved detailed results with criteria: {result_path}")

    # ===== Group statistics table =====
    stats_rows = []

    stats_rows.append({
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
    })

    for method_name, ad_data in ad_results.items():
        if len(ad_data['flags']) == len(exp_vals):
            ad_in = ad_data['flags']
            ad_out = ~ad_in

            n_in = np.sum(ad_in)
            n_out = np.sum(ad_out)

            if n_in > 0:
                stats_in = compute_metrics(exp_vals[ad_in], pred_vals[ad_in])
                stats_rows.append({
                    'Group': f'Within AD ({ad_data["display_name"]})',
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
            else:
                stats_rows.append({
                    'Group': f'Within AD ({ad_data["display_name"]})',
                    'N': 0,
                    'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                    'MedAE': np.nan, 'MaxAE': np.nan, 'Bias': np.nan,
                    '%≤10nm': np.nan, '%≤20nm': np.nan, '%≤30nm': np.nan
                })

            if n_out > 0:
                stats_out = compute_metrics(exp_vals[ad_out], pred_vals[ad_out])
                stats_rows.append({
                    'Group': f'Outside AD ({ad_data["display_name"]})',
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
            else:
                stats_rows.append({
                    'Group': f'Outside AD ({ad_data["display_name"]})',
                    'N': 0,
                    'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                    'MedAE': np.nan, 'MaxAE': np.nan, 'Bias': np.nan,
                    '%≤10nm': np.nan, '%≤20nm': np.nan, '%≤30nm': np.nan
                })

    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(tables_dir, 'AD_group_statistics_all.csv')
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved group statistics: {stats_path}")

    # Print AD method comparison summary
    print("\n" + "=" * 60)
    print("AD Method Comparison Summary")
    print("=" * 60)
    print(
        f"{'Method':<25} {'Within AD':<12} {'Outside AD':<12} {'MAE_in':<10} {'MAE_out':<10} {'RMSE_in':<10} {'RMSE_out':<10}")
    print("-" * 90)

    for method_name, ad_data in ad_results.items():
        if len(ad_data['flags']) == len(exp_vals):
            ad_in = ad_data['flags']
            ad_out = ~ad_in

            n_in = np.sum(ad_in)
            n_out = np.sum(ad_out)

            if n_in > 0:
                stats_in = compute_metrics(exp_vals[ad_in], pred_vals[ad_in])
                mae_in_str = f"{stats_in['MAE']:.2f}"
                rmse_in_str = f"{stats_in['RMSE']:.2f}"
            else:
                mae_in_str = "N/A"
                rmse_in_str = "N/A"

            if n_out > 0:
                stats_out = compute_metrics(exp_vals[ad_out], pred_vals[ad_out])
                mae_out_str = f"{stats_out['MAE']:.2f}"
                rmse_out_str = f"{stats_out['RMSE']:.2f}"
            else:
                mae_out_str = "N/A"
                rmse_out_str = "N/A"

            print(f"{ad_data['display_name']:<25} {n_in:<12} {n_out:<12} "
                  f"{mae_in_str:<10} {mae_out_str:<10} "
                  f"{rmse_in_str:<10} {rmse_out_str:<10}")

    return global_stats


# ============================================================
# 10. Main function
# ============================================================

def main():
    print("=" * 80)
    print("External Validation Pipeline: Prediction + Validation + Multiple AD Methods")
    print("DNN Model parameters integrated from training script")
    print("Descriptor files read from descriptor_results/ directory")
    print(f"Generate both versions (with/without names): {GENERATE_BOTH_VERSIONS}")
    print("=" * 80)

    # Step 1: Load or train model
    model, scaler_X, scaler_y = load_or_train_model()
    if model is None:
        return

    # Step 2: Predict
    ids, exp_vals, pred_vals = predict_external(model, scaler_X, scaler_y)
    if ids is None:
        return

    print(f"\n✓ Prediction completed: {len(ids)} valid samples")

    # Step 3: Multiple AD methods
    print("\n" + "=" * 80)
    print("Step 2: Multiple AD Classification")
    print("=" * 80)

    train_X = load_training_descriptors()
    test_X = load_external_descriptors()

    ad_results = {}
    if train_X is not None and test_X is not None:
        n_samples = min(len(test_X), len(ids))
        test_X = test_X[:n_samples]
        ids = ids[:n_samples]
        exp_vals = exp_vals[:n_samples]
        pred_vals = pred_vals[:n_samples]

        ad_results = apply_all_ad_methods(train_X, test_X)
    else:
        print("Skipped AD classification (training or external descriptors missing)")

    # Step 4: Validation plots
    print("\n" + "=" * 80)
    print("Step 3: Validation Plots")
    print("=" * 80)

    global_stats = create_external_plots(ids, exp_vals, pred_vals, ad_results, OUTPUT_DIR)

    # Step 5: Summary
    print("\n" + "=" * 80)
    print("External Validation Completed!")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}/")
    if GENERATE_BOTH_VERSIONS:
        print("  figures/          - TIFF figures (with molecule names)")
        print("  figures_no_names/ - TIFF figures (without molecule names)")
    else:
        print("  figures/          - TIFF figures (with molecule names)")
    print("  tables/           - CSV result tables (with criteria)")
    print("\nStatistics Summary:")
    for key in ['R2', 'MAE', 'RMSE', 'MedAE', 'MaxAE', 'Bias']:
        value = global_stats[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("=" * 80)


if __name__ == '__main__':
    main()