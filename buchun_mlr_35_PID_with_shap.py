# -*- coding: utf-8 -*-
"""
包含SHAP分析的完整MLR模型脚本 - 增强版 (35描述符)
CJChE格式版本 - 所有图表输出为TIFF 300 dpi
所有图表已去标题，符合CJChE要求
35个Mordred描述符版本
轴标签简化为 Exp. λmax (nm) / Pre. λmax (nm)
新增功能（与70MLR完全一致）：
- 使用MAE（平均绝对误差）
- 增加中位绝对误差 (MedAE)、最大绝对误差 (MaxAE)、预测偏差 (Bias)
- train-Q²_LOO-CV（仅训练集432个样本）
- VIF计算
- 10次重复10折CV获取置信区间
- 学习曲线（10折交叉验证，3张独立图）
- 字体大一号，边框加粗，无网格线
- 所有图表无标题
- 输出路径为 mlr_model_35_result_more
- 所有描述符使用真实名称
- MAE柱状图使用整数刻度
- SHAP图增加至10+张
- 威廉姆斯图四种版本（区分/不区分数据集，两种阈值）
- 增加预测误差在±10、±20、±30 nm内的化合物百分比
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, KFold
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
import shap
import warnings
from scipy import stats
import time

warnings.filterwarnings('ignore')


def setup_cjche_style_larger():
    """设置绘图样式 - 字体大一号，边框加粗，无网格，无标题"""
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


def setup_cjche_style_heatmap():
    """热力图专用样式 - 高度增加，纵轴标签清晰，无标题"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.grid'] = False


# 应用样式
setup_cjche_style_larger()


def save_figure_tiff(fig, filepath, dpi=300):
    """统一保存图片为TIFF格式，300 dpi"""
    fig.savefig(filepath, format='tiff', dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")


def load_descriptor_mapping(info_file_path='mordred_descriptor_info.xlsx'):
    """加载描述符真实名称映射"""
    if not os.path.exists(info_file_path):
        print(f"  ⚠️ 警告: {info_file_path} 不存在，将使用默认 Md_XXX 格式")
        return None

    df_info = pd.read_excel(info_file_path)
    mapping = {}
    for _, row in df_info.iterrows():
        code = row['Descriptor_Code']  # 直接使用，已经是 'Md_906' 格式
        name = row['Descriptor_Name']
        if '.' in name:
            name = name.split('.')[-1]
        mapping[code] = name
    return mapping


def calculate_vif(X, feature_names):
    """计算VIF（方差膨胀因子）"""
    n_features = X.shape[1]
    vif_data = []

    for i in range(n_features):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)

        model = LinearRegression()
        model.fit(X_others, y)
        r_squared = model.score(X_others, y)

        vif = 1 / (1 - r_squared)
        vif_data.append({
            'Feature': feature_names[i],
            'VIF': vif
        })

    return pd.DataFrame(vif_data)


def repeated_cv_confidence_mlr(X_train, y_train, n_repeats=10, n_folds=10):
    """重复CV获取置信区间（10次重复，每次10折交叉验证）"""
    results = {
        'R2_train': [], 'R2_cv': [],
        'RMSE_train': [], 'RMSE_cv': [],
        'MAE_train': [], 'MAE_cv': [],
        'MedAE_train': [], 'MedAE_cv': [],
        'MaxAE_train': [], 'MaxAE_cv': [],
        'Bias_train': [], 'Bias_cv': []
    }

    print(f"  Running {n_repeats} repeats of {n_folds}-fold CV...")

    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=repeat)

        fold_results = {
            'R2_train': [], 'R2_cv': [],
            'RMSE_train': [], 'RMSE_cv': [],
            'MAE_train': [], 'MAE_cv': [],
            'MedAE_train': [], 'MedAE_cv': [],
            'MaxAE_train': [], 'MaxAE_cv': [],
            'Bias_train': [], 'Bias_cv': []
        }

        for train_idx, cv_idx in kf.split(X_train):
            X_tr_fold, X_cv_fold = X_train[train_idx], X_train[cv_idx]
            y_tr_fold, y_cv_fold = y_train[train_idx], y_train[cv_idx]

            model = LinearRegression()
            model.fit(X_tr_fold, y_tr_fold)

            y_tr_pred = model.predict(X_tr_fold)
            y_cv_pred = model.predict(X_cv_fold)

            fold_results['R2_train'].append(r2_score(y_tr_fold, y_tr_pred))
            fold_results['R2_cv'].append(r2_score(y_cv_fold, y_cv_pred))
            fold_results['RMSE_train'].append(np.sqrt(mean_squared_error(y_tr_fold, y_tr_pred)))
            fold_results['RMSE_cv'].append(np.sqrt(mean_squared_error(y_cv_fold, y_cv_pred)))
            fold_results['MAE_train'].append(mean_absolute_error(y_tr_fold, y_tr_pred))
            fold_results['MAE_cv'].append(mean_absolute_error(y_cv_fold, y_cv_pred))
            fold_results['MedAE_train'].append(median_absolute_error(y_tr_fold, y_tr_pred))
            fold_results['MedAE_cv'].append(median_absolute_error(y_cv_fold, y_cv_pred))
            fold_results['MaxAE_train'].append(np.max(np.abs(y_tr_fold - y_tr_pred)))
            fold_results['MaxAE_cv'].append(np.max(np.abs(y_cv_fold - y_cv_pred)))
            fold_results['Bias_train'].append(np.mean(y_tr_pred - y_tr_fold))
            fold_results['Bias_cv'].append(np.mean(y_cv_pred - y_cv_fold))

        for key in fold_results:
            results[key].append(np.mean(fold_results[key]))

        if (repeat + 1) % 2 == 0:
            print(f"    Completed {repeat + 1}/{n_repeats} repeats")

    confidence_results = {}
    for key in results:
        mean_val = np.mean(results[key])
        std_val = np.std(results[key])
        ci_95 = 1.96 * std_val / np.sqrt(len(results[key]))
        confidence_results[key] = {
            'mean': mean_val,
            'std': std_val,
            'ci_95': ci_95,
            'ci_lower': mean_val - ci_95,
            'ci_upper': mean_val + ci_95,
            'values': results[key]
        }

    return confidence_results


def train_loo_cv_mlr(X_train, y_train):
    """仅对训练集进行LOO-CV计算Q²及额外指标（包含误差百分比）"""
    loo = LeaveOneOut()
    y_true_loo = []
    y_pred_loo = []

    print("  Running LOO-CV on training set (432 samples)...")

    for i, (train_idx, test_idx) in enumerate(loo.split(X_train)):
        X_train_loo, X_test_loo = X_train[train_idx], X_train[test_idx]
        y_train_loo, y_test_loo = y_train[train_idx], y_train[test_idx]

        model_loo = LinearRegression()
        model_loo.fit(X_train_loo, y_train_loo)

        y_pred_loo_value = model_loo.predict(X_test_loo.reshape(1, -1))[0]

        y_pred_loo.append(y_pred_loo_value)
        y_true_loo.append(y_test_loo[0])

        if (i + 1) % 50 == 0:
            print(f"    Completed {i + 1}/{len(X_train)} samples")

    y_true_loo = np.array(y_true_loo)
    y_pred_loo = np.array(y_pred_loo)

    y_mean = np.mean(y_true_loo)
    Q2_train = 1 - np.sum((y_true_loo - y_pred_loo) ** 2) / np.sum((y_true_loo - y_mean) ** 2)
    MAE_LOO_train = mean_absolute_error(y_true_loo, y_pred_loo)
    RMSE_LOO_train = np.sqrt(mean_squared_error(y_true_loo, y_pred_loo))
    MedAE_LOO_train = median_absolute_error(y_true_loo, y_pred_loo)
    MaxAE_LOO_train = np.max(np.abs(y_true_loo - y_pred_loo))
    Bias_LOO_train = np.mean(y_pred_loo - y_true_loo)

    # 误差百分比
    abs_err_loo = np.abs(y_true_loo - y_pred_loo)
    pct10_loo = np.mean(abs_err_loo <= 10) * 100
    pct20_loo = np.mean(abs_err_loo <= 20) * 100
    pct30_loo = np.mean(abs_err_loo <= 30) * 100

    return {
        'Q2_train_LOO': Q2_train,
        'MAE_LOO_train': MAE_LOO_train,
        'RMSE_LOO_train': RMSE_LOO_train,
        'MedAE_LOO_train': MedAE_LOO_train,
        'MaxAE_LOO_train': MaxAE_LOO_train,
        'Bias_LOO_train': Bias_LOO_train,
        'Pct10_LOO_train': pct10_loo,
        'Pct20_LOO_train': pct20_loo,
        'Pct30_LOO_train': pct30_loo,
        'y_true': y_true_loo,
        'y_pred': y_pred_loo
    }


def plot_learning_curve_separate_mlr(X_train, y_train, figures_dir):
    """绘制学习曲线 - 3张独立图，10折交叉验证"""
    print("  Generating learning curves with 10-fold CV...")

    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores_r2 = []
    cv_scores_r2 = []
    train_scores_rmse = []
    cv_scores_rmse = []
    train_scores_mae = []
    cv_scores_mae = []

    for size in train_sizes:
        n_samples = int(len(X_train) * size)
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        train_r2 = []
        cv_r2 = []
        train_rmse = []
        cv_rmse = []
        train_mae = []
        cv_mae = []

        for train_idx, val_idx in kf.split(X_subset):
            X_tr, X_cv = X_subset[train_idx], X_subset[val_idx]
            y_tr, y_cv = y_subset[train_idx], y_subset[val_idx]

            model_lc = LinearRegression()
            model_lc.fit(X_tr, y_tr)

            y_tr_pred = model_lc.predict(X_tr)
            y_cv_pred = model_lc.predict(X_cv)

            train_r2.append(r2_score(y_tr, y_tr_pred))
            cv_r2.append(r2_score(y_cv, y_cv_pred))
            train_rmse.append(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
            cv_rmse.append(np.sqrt(mean_squared_error(y_cv, y_cv_pred)))
            train_mae.append(mean_absolute_error(y_tr, y_tr_pred))
            cv_mae.append(mean_absolute_error(y_cv, y_cv_pred))

        train_scores_r2.append(np.mean(train_r2))
        cv_scores_r2.append(np.mean(cv_r2))
        train_scores_rmse.append(np.mean(train_rmse))
        cv_scores_rmse.append(np.mean(cv_rmse))
        train_scores_mae.append(np.mean(train_mae))
        cv_scores_mae.append(np.mean(cv_mae))

    # ===== R²学习曲线 (独立图) =====
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))
    ax.plot(train_sizes * len(X_train), train_scores_r2, 'b-', linewidth=2, label='Training R²')
    ax.plot(train_sizes * len(X_train), cv_scores_r2, 'r-', linewidth=2, label='CV R² (10-fold)')
    ax.fill_between(train_sizes * len(X_train),
                    np.array(train_scores_r2) - np.std(train_scores_r2),
                    np.array(train_scores_r2) + np.std(train_scores_r2),
                    alpha=0.2, color='blue')
    ax.fill_between(train_sizes * len(X_train),
                    np.array(cv_scores_r2) - np.std(cv_scores_r2),
                    np.array(cv_scores_r2) + np.std(cv_scores_r2),
                    alpha=0.2, color='red')
    ax.set_xlabel('Training set size', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.set_ylim([0, 1.05])
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig7_learning_curve_r2.tiff'), dpi=300)
    plt.close()
    print("  ✓ Learning curve R² saved")

    # ===== RMSE学习曲线 (独立图) =====
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))
    ax.plot(train_sizes * len(X_train), train_scores_rmse, 'b-', linewidth=2, label='Training RMSE')
    ax.plot(train_sizes * len(X_train), cv_scores_rmse, 'r-', linewidth=2, label='CV RMSE (10-fold)')
    ax.fill_between(train_sizes * len(X_train),
                    np.array(train_scores_rmse) - np.std(train_scores_rmse),
                    np.array(train_scores_rmse) + np.std(train_scores_rmse),
                    alpha=0.2, color='blue')
    ax.fill_between(train_sizes * len(X_train),
                    np.array(cv_scores_rmse) - np.std(cv_scores_rmse),
                    np.array(cv_scores_rmse) + np.std(cv_scores_rmse),
                    alpha=0.2, color='red')
    ax.set_xlabel('Training set size', fontsize=12)
    ax.set_ylabel('RMSE (nm)', fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig8_learning_curve_rmse.tiff'), dpi=300)
    plt.close()
    print("  ✓ Learning curve RMSE saved")

    # ===== MAE学习曲线 (独立图) =====
    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))
    ax.plot(train_sizes * len(X_train), train_scores_mae, 'b-', linewidth=2, label='Training MAE')
    ax.plot(train_sizes * len(X_train), cv_scores_mae, 'r-', linewidth=2, label='CV MAE (10-fold)')
    ax.fill_between(train_sizes * len(X_train),
                    np.array(train_scores_mae) - np.std(train_scores_mae),
                    np.array(train_scores_mae) + np.std(train_scores_mae),
                    alpha=0.2, color='blue')
    ax.fill_between(train_sizes * len(X_train),
                    np.array(cv_scores_mae) - np.std(cv_scores_mae),
                    np.array(cv_scores_mae) + np.std(cv_scores_mae),
                    alpha=0.2, color='red')
    ax.set_xlabel('Training set size', fontsize=12)
    ax.set_ylabel('MAE (nm)', fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig9_learning_curve_mae.tiff'), dpi=300)
    plt.close()
    print("  ✓ Learning curve MAE saved")

    learning_curve_data = pd.DataFrame({
        'Training_Set_Size': train_sizes * len(X_train),
        'Training_R2': train_scores_r2,
        'CV_R2': cv_scores_r2,
        'Training_RMSE': train_scores_rmse,
        'CV_RMSE': cv_scores_rmse,
        'Training_MAE': train_scores_mae,
        'CV_MAE': cv_scores_mae
    })

    return learning_curve_data


def main():
    # ==============================================================================================
    cd_excel = 'origin_data/'
    excel_name = 'data_w_Md_bo.xlsx'

    # ========== 输出到 mlr_model_35_result_more ==========
    output_dir = 'mlr_model_35_result_more'
    os.makedirs(output_dir, exist_ok=True)

    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    shap_dir = os.path.join(output_dir, 'shap_analysis')
    os.makedirs(shap_dir, exist_ok=True)
    # ==============================================================================================

    print(f"\n{'=' * 80}")
    print("MLR Model - Multiple Linear Regression with 35 descriptors (Enhanced)")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # ----------------------------------加载数据--------------------------------------------------
    print("\nLoading data and building MLR model")
    print("-" * 60)

    df = pd.read_excel(f'{cd_excel}{excel_name}')

    train_df = df[df['Training set/Testing set'] == 'Training set'].copy()
    test_df = df[df['Training set/Testing set'] == 'Testing set'].copy()
    total_df = pd.concat([train_df, test_df], ignore_index=True)

    selected_bits = [
        906, 1336, 334, 1310, 1348,
        1406, 1437, 345, 218, 230,
        449, 510, 299, 3, 1547,
        566, 1418, 1069, 232, 264,
        1076, 1077, 322, 1313, 1294,
        1064, 1329, 1146, 1477, 1359,
        805, 1394, 1358, 1573, 485
    ]

    # 生成特征名称列表
    feature_names = [f'Md_{bit}' for bit in selected_bits]

    # 加载描述符真实名称映射
    mapping = load_descriptor_mapping('mordred_descriptor_info.xlsx')
    if mapping:
        display_names = [mapping.get(f, f) for f in feature_names]
        print(f"✓ Loaded {len(mapping)} descriptor name mappings")
        # 打印前5个示例验证
        print("  Example mappings:")
        for i in range(min(5, len(feature_names))):
            print(f"    {feature_names[i]} -> {display_names[i]}")
    else:
        display_names = feature_names.copy()
        print("  Using default Md_XXX names")

    X_train = train_df[[f'Md_{bit}' for bit in selected_bits]].values
    y_train = train_df['bo'].values
    X_test = test_df[[f'Md_{bit}' for bit in selected_bits]].values
    y_test = test_df['bo'].values
    total_X = total_df[[f'Md_{bit}' for bit in selected_bits]].values
    total_y = total_df['bo'].values

    print(f"Training set samples: {len(X_train)}")
    print(f"Testing set samples: {len(X_test)}")
    print(f"Total samples: {len(total_X)}")
    print(f"Number of features: {len(selected_bits)}")

    # ----------------------------------训练MLR模型--------------------------------------------------
    print("\nTraining MLR model...")
    print("-" * 60)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    total_y_pred = model.predict(total_X)

    # 计算残差
    train_residuals = np.abs(y_train - y_train_pred)
    test_residuals = np.abs(y_test - y_test_pred)
    total_residuals = np.abs(total_y - total_y_pred)

    # 计算误差百分比
    pct10_train = np.mean(train_residuals <= 10) * 100
    pct20_train = np.mean(train_residuals <= 20) * 100
    pct30_train = np.mean(train_residuals <= 30) * 100
    pct10_test = np.mean(test_residuals <= 10) * 100
    pct20_test = np.mean(test_residuals <= 20) * 100
    pct30_test = np.mean(test_residuals <= 30) * 100
    pct10_total = np.mean(total_residuals <= 10) * 100
    pct20_total = np.mean(total_residuals <= 20) * 100
    pct30_total = np.mean(total_residuals <= 30) * 100

    # 计算性能指标
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_total = r2_score(total_y, total_y_pred)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmse_total = np.sqrt(mean_squared_error(total_y, total_y_pred))

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mae_total = mean_absolute_error(total_y, total_y_pred)

    medae_train = median_absolute_error(y_train, y_train_pred)
    medae_test = median_absolute_error(y_test, y_test_pred)
    medae_total = median_absolute_error(total_y, total_y_pred)

    maxae_train = np.max(np.abs(y_train - y_train_pred))
    maxae_test = np.max(np.abs(y_test - y_test_pred))
    maxae_total = np.max(np.abs(total_y - total_y_pred))

    bias_train = np.mean(y_train_pred - y_train)
    bias_test = np.mean(y_test_pred - y_test)
    bias_total = np.mean(total_y_pred - total_y)

    overfit = r2_train - r2_test

    print(f"\nModel Performance:")
    print(f"  R² (Training): {r2_train:.4f}")
    print(f"  R² (Testing): {r2_test:.4f}")
    print(f"  R² (Total): {r2_total:.4f}")
    print(f"  RMSE (Training): {rmse_train:.2f} nm")
    print(f"  RMSE (Testing): {rmse_test:.2f} nm")
    print(f"  RMSE (Total): {rmse_total:.2f} nm")
    print(f"  MAE (Training): {mae_train:.2f} nm")
    print(f"  MAE (Testing): {mae_test:.2f} nm")
    print(f"  MAE (Total): {mae_total:.2f} nm")
    print(f"  MedAE (Training): {medae_train:.2f} nm")
    print(f"  MedAE (Testing): {medae_test:.2f} nm")
    print(f"  MedAE (Total): {medae_total:.2f} nm")
    print(f"  MaxAE (Training): {maxae_train:.2f} nm")
    print(f"  MaxAE (Testing): {maxae_test:.2f} nm")
    print(f"  MaxAE (Total): {maxae_total:.2f} nm")
    print(f"  Bias (Training): {bias_train:.2f} nm")
    print(f"  Bias (Testing): {bias_test:.2f} nm")
    print(f"  Bias (Total): {bias_total:.2f} nm")
    print(f"  % within ±10 nm (Training): {pct10_train:.1f}%")
    print(f"  % within ±20 nm (Training): {pct20_train:.1f}%")
    print(f"  % within ±30 nm (Training): {pct30_train:.1f}%")
    print(f"  % within ±10 nm (Testing): {pct10_test:.1f}%")
    print(f"  % within ±20 nm (Testing): {pct20_test:.1f}%")
    print(f"  % within ±30 nm (Testing): {pct30_test:.1f}%")
    print(f"  % within ±10 nm (Total): {pct10_total:.1f}%")
    print(f"  % within ±20 nm (Total): {pct20_total:.1f}%")
    print(f"  % within ±30 nm (Total): {pct30_total:.1f}%")
    print(f"  Overfitting: {overfit:.4f}")

    intercept = model.intercept_
    coefficients = model.coef_

    # ----------------------------------LOO-CV验证（全数据集）------------------------------------------
    print(f"\nLOO-CV Validation (Full dataset)")
    print("-" * 60)

    loo = LeaveOneOut()
    y_true_loo = []
    y_pred_loo = []

    print("Running LOO-CV validation on full dataset...")

    for train_index, test_index in loo.split(total_X):
        X_train_loo, X_test_loo = total_X[train_index], total_X[test_index]
        y_train_loo, y_test_loo = total_y[train_index], total_y[test_index]

        model.fit(X_train_loo, y_train_loo)
        y_pred_loo.append(model.predict(X_test_loo)[0])
        y_true_loo.append(y_test_loo[0])

    y_true_loo = np.array(y_true_loo)
    y_pred_loo = np.array(y_pred_loo)
    loo_residuals = np.abs(y_true_loo - y_pred_loo)

    y_mean = np.mean(y_true_loo)
    Q2_total = 1 - np.sum((y_true_loo - y_pred_loo) ** 2) / np.sum((y_true_loo - y_mean) ** 2)
    MAE_LOO_total = mean_absolute_error(y_true_loo, y_pred_loo)
    RMSE_LOO_total = np.sqrt(mean_squared_error(y_true_loo, y_pred_loo))
    MedAE_LOO_total = median_absolute_error(y_true_loo, y_pred_loo)
    MaxAE_LOO_total = np.max(np.abs(y_true_loo - y_pred_loo))
    Bias_LOO_total = np.mean(y_pred_loo - y_true_loo)

    pct10_loo_total = np.mean(loo_residuals <= 10) * 100
    pct20_loo_total = np.mean(loo_residuals <= 20) * 100
    pct30_loo_total = np.mean(loo_residuals <= 30) * 100

    print(f"\nLOO-CV Results (Full dataset):")
    print(f"  Q² (LOO-CV): {Q2_total:.4f}")
    print(f"  MAE (LOO-CV): {MAE_LOO_total:.4f} nm")
    print(f"  RMSE (LOO-CV): {RMSE_LOO_total:.4f} nm")
    print(f"  MedAE (LOO-CV): {MedAE_LOO_total:.4f} nm")
    print(f"  MaxAE (LOO-CV): {MaxAE_LOO_total:.4f} nm")
    print(f"  Bias (LOO-CV): {Bias_LOO_total:.4f} nm")
    print(f"  % within ±10 nm (LOO-CV): {pct10_loo_total:.1f}%")
    print(f"  % within ±20 nm (LOO-CV): {pct20_loo_total:.1f}%")
    print(f"  % within ±30 nm (LOO-CV): {pct30_loo_total:.1f}%")

    # ----------------------------------Train-LOO-CV（仅训练集）----------------------------------------
    print(f"\nTrain-LOO-CV (Training set only, 432 samples)")
    print("-" * 60)

    train_loo_results = train_loo_cv_mlr(X_train, y_train)

    print(f"\nTrain-LOO-CV Results:")
    print(f"  Q² (Train-LOO-CV): {train_loo_results['Q2_train_LOO']:.4f}")
    print(f"  MAE (Train-LOO-CV): {train_loo_results['MAE_LOO_train']:.4f} nm")
    print(f"  RMSE (Train-LOO-CV): {train_loo_results['RMSE_LOO_train']:.4f} nm")
    print(f"  MedAE (Train-LOO-CV): {train_loo_results['MedAE_LOO_train']:.4f} nm")
    print(f"  MaxAE (Train-LOO-CV): {train_loo_results['MaxAE_LOO_train']:.4f} nm")
    print(f"  Bias (Train-LOO-CV): {train_loo_results['Bias_LOO_train']:.4f} nm")
    print(f"  % within ±10 nm (Train-LOO-CV): {train_loo_results['Pct10_LOO_train']:.1f}%")
    print(f"  % within ±20 nm (Train-LOO-CV): {train_loo_results['Pct20_LOO_train']:.1f}%")
    print(f"  % within ±30 nm (Train-LOO-CV): {train_loo_results['Pct30_LOO_train']:.1f}%")

    # ----------------------------------VIF计算（使用真实名称）----------------------------------------
    print(f"\nVIF Calculation")
    print("-" * 60)

    vif_df = calculate_vif(X_train, display_names)
    vif_df = vif_df.sort_values('VIF', ascending=False)

    print(f"VIF statistics:")
    print(f"  Mean VIF: {vif_df['VIF'].mean():.4f}")
    print(f"  Max VIF: {vif_df['VIF'].max():.4f}")
    print(f"  Min VIF: {vif_df['VIF'].min():.4f}")
    print(f"  Features with VIF > 10: {len(vif_df[vif_df['VIF'] > 10])}")
    print(f"  Features with VIF > 5: {len(vif_df[vif_df['VIF'] > 5])}")

    vif_path = os.path.join(tables_dir, 'TableS1_vif_results.xlsx')
    vif_df.to_excel(vif_path, index=False)
    print(f"✓ VIF results saved to: {vif_path}")

    # ----------------------------------重复CV（10折）----------------------------------------
    print(f"\nRepeated CV (10 repeats, 10-fold CV for confidence intervals)")
    print("-" * 60)

    cv_results = repeated_cv_confidence_mlr(X_train, y_train, n_repeats=10, n_folds=10)

    print("\nRepeated CV Results (10 repeats, 10-fold CV on training set):")
    print("-" * 50)
    for key in ['R2_train', 'R2_cv', 'MAE_train', 'MAE_cv', 'RMSE_train', 'RMSE_cv',
                'MedAE_train', 'MedAE_cv', 'MaxAE_train', 'MaxAE_cv', 'Bias_train', 'Bias_cv']:
        result = cv_results[key]
        print(f"  {key}:")
        print(f"    Mean: {result['mean']:.4f}")
        print(f"    Std: {result['std']:.4f}")
        print(f"    95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print()

    cv_summary = []
    for key in ['R2_train', 'R2_cv', 'MAE_train', 'MAE_cv', 'RMSE_train', 'RMSE_cv',
                'MedAE_train', 'MedAE_cv', 'MaxAE_train', 'MaxAE_cv', 'Bias_train', 'Bias_cv']:
        result = cv_results[key]
        cv_summary.append({
            'Metric': key,
            'Mean': result['mean'],
            'Std': result['std'],
            'CI_Lower_95': result['ci_lower'],
            'CI_Upper_95': result['ci_upper']
        })

    cv_summary_df = pd.DataFrame(cv_summary)
    cv_summary_path = os.path.join(tables_dir, 'TableS2_repeated_cv_confidence_intervals.xlsx')
    cv_summary_df.to_excel(cv_summary_path, index=False)
    print(f"✓ Repeated CV summary saved to: {cv_summary_path}")

    cv_detailed = pd.DataFrame({
        'Repeat': range(1, 11),
        'R2_train': cv_results['R2_train']['values'],
        'R2_cv': cv_results['R2_cv']['values'],
        'MAE_train': cv_results['MAE_train']['values'],
        'MAE_cv': cv_results['MAE_cv']['values'],
        'RMSE_train': cv_results['RMSE_train']['values'],
        'RMSE_cv': cv_results['RMSE_cv']['values'],
        'MedAE_train': cv_results['MedAE_train']['values'],
        'MedAE_cv': cv_results['MedAE_cv']['values'],
        'MaxAE_train': cv_results['MaxAE_train']['values'],
        'MaxAE_cv': cv_results['MaxAE_cv']['values'],
        'Bias_train': cv_results['Bias_train']['values'],
        'Bias_cv': cv_results['Bias_cv']['values']
    })
    cv_detailed_path = os.path.join(tables_dir, 'TableS3_repeated_cv_detailed.xlsx')
    cv_detailed.to_excel(cv_detailed_path, index=False)
    print(f"✓ Detailed CV results saved to: {cv_detailed_path}")

    # ----------------------------------生成并保存计算结果表格----------------------------------------
    print(f"\nGenerating result tables...")
    print("-" * 60)

    # 1. 模型性能汇总表 (包含新指标)
    performance_df = pd.DataFrame({
        'Dataset': ['Training', 'Testing', 'Total', 'LOO-CV (Full)', 'Train-LOO-CV'],
        'R²/Q²': [r2_train, r2_test, r2_total, Q2_total, train_loo_results['Q2_train_LOO']],
        'MAE': [mae_train, mae_test, mae_total, MAE_LOO_total, train_loo_results['MAE_LOO_train']],
        'RMSE': [rmse_train, rmse_test, rmse_total, RMSE_LOO_total, train_loo_results['RMSE_LOO_train']],
        'MedAE': [medae_train, medae_test, medae_total, MedAE_LOO_total, train_loo_results['MedAE_LOO_train']],
        'MaxAE': [maxae_train, maxae_test, maxae_total, MaxAE_LOO_total, train_loo_results['MaxAE_LOO_train']],
        'Bias': [bias_train, bias_test, bias_total, Bias_LOO_total, train_loo_results['Bias_LOO_train']],
        '% ≤10 nm': [pct10_train, pct10_test, pct10_total, pct10_loo_total, train_loo_results['Pct10_LOO_train']],
        '% ≤20 nm': [pct20_train, pct20_test, pct20_total, pct20_loo_total, train_loo_results['Pct20_LOO_train']],
        '% ≤30 nm': [pct30_train, pct30_test, pct30_total, pct30_loo_total, train_loo_results['Pct30_LOO_train']],
        'Samples': [len(y_train), len(y_test), len(total_y), len(total_y), len(y_train)]
    })

    performance_path = os.path.join(tables_dir, 'Table1_model_performance_summary.xlsx')
    performance_df.to_excel(performance_path, index=False)
    print(f"✓ Model performance summary saved to: {performance_path}")

    # 2. 预测值与实验值对比表
    all_predictions = pd.DataFrame({
        'PID': total_df['PID'].values if 'PID' in total_df.columns else range(1, len(total_y) + 1),
        'Compound_ID': range(1, len(total_y) + 1),
        'Dataset': total_df['Training set/Testing set'].values if 'Training set/Testing set' in total_df.columns else [
                                                                                                                          'Unknown'] * len(
            total_y),
        'Experimental': total_y,
        'Predicted': total_y_pred,
        'Residual': total_y - total_y_pred,
        'Absolute_Residual': total_residuals
    })

    if 'Compound' in total_df.columns:
        all_predictions.insert(2, 'Compound_Name', total_df['Compound'].values)

    predictions_path = os.path.join(tables_dir, 'Table2_predictions_comparison.xlsx')
    all_predictions.to_excel(predictions_path, index=False)
    print(f"✓ Predictions comparison saved to: {predictions_path}")

    # 3. 回归系数表（使用真实名称）
    coefficients_df = pd.DataFrame({
        'Descriptor': display_names,
        'Coefficient': coefficients,
        'Absolute_Coefficient': np.abs(coefficients)
    }).sort_values('Absolute_Coefficient', ascending=False)

    coefficients_df = pd.concat([
        pd.DataFrame({
            'Descriptor': ['Intercept'],
            'Coefficient': [intercept],
            'Absolute_Coefficient': [np.abs(intercept)]
        }),
        coefficients_df
    ], ignore_index=True)

    coefficients_path = os.path.join(tables_dir, 'Table3_regression_coefficients.xlsx')
    coefficients_df.to_excel(coefficients_path, index=False)
    print(f"✓ Regression coefficients saved to: {coefficients_path}")

    # 4. LOO-CV详细结果表
    loo_results_df = pd.DataFrame({
        'PID': total_df['PID'].values if 'PID' in total_df.columns else range(1, len(y_true_loo) + 1),
        'Sample_Index': list(range(1, len(y_true_loo) + 1)),
        'Experimental': y_true_loo,
        'Predicted_LOO': y_pred_loo,
        'Residual': y_true_loo - y_pred_loo,
        'Absolute_Residual': loo_residuals
    })

    if 'Compound' in total_df.columns:
        loo_results_df.insert(2, 'Compound_Name', total_df['Compound'].values)

    loo_path = os.path.join(tables_dir, 'Table4_loo_cv_detailed_results.xlsx')
    loo_results_df.to_excel(loo_path, index=False)
    print(f"✓ LOO-CV detailed results saved to: {loo_path}")

    # ----------------------------------绘制散点图（无网格，无标题）-----------------------------------
    print(f"\nGenerating scatter plots...")
    print("-" * 60)

    cmap = plt.cm.coolwarm_r

    # 训练集散点图
    fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54))
    norm = Normalize(vmin=0, vmax=np.max(train_residuals))
    sc = plt.scatter(y_train, y_train_pred,
                     c=train_residuals, cmap=cmap, norm=norm,
                     alpha=0.7, edgecolors='k', s=20)

    min_val = min(min(y_train), min(y_train_pred))
    max_val = max(max(y_train), max(y_train_pred))
    plt.plot([min_val, max_val], [min_val, max_val],
             color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.xlabel('Exp. λmax (nm)', fontsize=12)
    plt.ylabel('Pre. λmax (nm)', fontsize=12)
    plt.tick_params(labelsize=11)

    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute error (nm)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    textstr = '\n'.join((
        f'$R^2$ = {r2_train:.4f}',
        f'MAE = {mae_train:.4f} nm',
        f'RMSE = {rmse_train:.4f} nm'
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig1_training_set_scatter.tiff'), dpi=300)
    plt.close()

    # 测试集散点图
    fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54))
    norm = Normalize(vmin=0, vmax=np.max(test_residuals))
    sc = plt.scatter(y_test, y_test_pred,
                     c=test_residuals, cmap=cmap, norm=norm,
                     alpha=0.7, edgecolors='k', s=20)

    min_val = min(min(y_test), min(y_test_pred))
    max_val = max(max(y_test), max(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val],
             color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.xlabel('Exp. λmax (nm)', fontsize=12)
    plt.ylabel('Pre. λmax (nm)', fontsize=12)
    plt.tick_params(labelsize=11)

    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute error (nm)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    textstr = '\n'.join((
        f'$R^2$ = {r2_test:.4f}',
        f'MAE = {mae_test:.4f} nm',
        f'RMSE = {rmse_test:.4f} nm'
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig2_testing_set_scatter.tiff'), dpi=300)
    plt.close()

    # 总数据集散点图
    fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54))
    norm_total = Normalize(vmin=0, vmax=np.max(total_residuals))

    train_scatter = plt.scatter(y_train, y_train_pred,
                                c=train_residuals, cmap=cmap, norm=norm_total,
                                alpha=0.7, edgecolors='k', s=20,
                                marker='o', label='Training set')

    test_scatter = plt.scatter(y_test, y_test_pred,
                               c=test_residuals, cmap=cmap, norm=norm_total,
                               alpha=0.7, edgecolors='k', s=20,
                               marker='s', label='Testing set')

    min_val = min(min(total_y), min(total_y_pred))
    max_val = max(max(total_y), max(total_y_pred))
    plt.plot([min_val, max_val], [min_val, max_val],
             color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.xlabel('Exp. λmax (nm)', fontsize=12)
    plt.ylabel('Pre. λmax (nm)', fontsize=12)
    plt.tick_params(labelsize=11)
    plt.legend(fontsize=10, frameon=False)

    cbar = plt.colorbar(train_scatter)
    cbar.set_label('Absolute error (nm)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    textstr = '\n'.join((
        f'$R^2$ (Train) = {r2_train:.4f}',
        f'$R^2$ (Test) = {r2_test:.4f}',
        f'MAE (Train) = {mae_train:.4f} nm',
        f'MAE (Test) = {mae_test:.4f} nm'
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig3_combined_dataset_scatter.tiff'), dpi=300)
    plt.close()

    # LOO-CV散点图（全数据集）
    fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54))
    norm = Normalize(vmin=0, vmax=np.max(loo_residuals))

    sc = plt.scatter(y_true_loo, y_pred_loo,
                     c=loo_residuals, cmap=cmap, norm=norm,
                     alpha=0.7, edgecolors='k', s=20)

    min_val = min(min(y_true_loo), min(y_pred_loo))
    max_val = max(max(y_true_loo), max(y_pred_loo))
    plt.plot([min_val, max_val], [min_val, max_val],
             color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.xlabel('Exp. λmax (nm)', fontsize=12)
    plt.ylabel('Pre. λmax (nm)', fontsize=12)
    plt.tick_params(labelsize=11)

    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute error (nm)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    textstr = '\n'.join((
        f'$Q^2$ = {Q2_total:.4f}',
        f'MAE = {MAE_LOO_total:.4f} nm',
        f'RMSE = {RMSE_LOO_total:.4f} nm'
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig4_loo_cv_scatter.tiff'), dpi=300)
    plt.close()

    # Train-LOO-CV散点图
    fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54))

    train_loo_y_true = train_loo_results['y_true']
    train_loo_y_pred = train_loo_results['y_pred']
    train_loo_residuals = np.abs(train_loo_y_true - train_loo_y_pred)

    norm = Normalize(vmin=0, vmax=np.max(train_loo_residuals))
    sc = plt.scatter(train_loo_y_true, train_loo_y_pred,
                     c=train_loo_residuals, cmap=cmap, norm=norm,
                     alpha=0.7, edgecolors='k', s=20)

    min_val = min(min(train_loo_y_true), min(train_loo_y_pred))
    max_val = max(max(train_loo_y_true), max(train_loo_y_pred))
    plt.plot([min_val, max_val], [min_val, max_val],
             color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.xlabel('Exp. λmax (nm)', fontsize=12)
    plt.ylabel('Pre. λmax (nm)', fontsize=12)
    plt.tick_params(labelsize=11)

    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute error (nm)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    textstr = '\n'.join((
        f'$Q^2$ = {train_loo_results["Q2_train_LOO"]:.4f}',
        f'MAE = {train_loo_results["MAE_LOO_train"]:.4f} nm',
        f'RMSE = {train_loo_results["RMSE_LOO_train"]:.4f} nm'
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'FigS5_train_loo_cv_scatter.tiff'), dpi=300)
    plt.close()

    # ----------------------------------其他分析图--------------------------------------------
    print(f"\nGenerating additional analysis plots...")
    print("-" * 60)

    # MAE分布柱状图 - 整数刻度
    fig = plt.figure(figsize=(12 / 2.54, 8 / 2.54))

    all_absolute_errors = np.concatenate([total_residuals, loo_residuals])
    max_error = np.max(all_absolute_errors)

    # 使用整数刻度
    bin_width = 50
    max_bin = int(np.ceil(max_error / bin_width) * bin_width)
    bins = np.arange(0, max_bin + bin_width, bin_width)

    mlr_counts, _ = np.histogram(total_residuals, bins=bins)
    loo_counts, _ = np.histogram(loo_residuals, bins=bins)

    x_pos = bins[:-1] + bin_width * 0.25
    plt.bar(x_pos, mlr_counts, width=bin_width * 0.35, alpha=0.7, color='#1f77b4',
            edgecolor='black', linewidth=1.2, label='MLR model')
    plt.bar(x_pos + bin_width * 0.35, loo_counts, width=bin_width * 0.35, alpha=0.7, color='#ff7f0e',
            edgecolor='black', linewidth=1.2, label='LOO-CV')

    plt.xlabel('Absolute error (nm)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(fontsize=10, frameon=False)
    plt.xticks(np.arange(0, max_bin + bin_width, bin_width), fontsize=10)
    plt.tick_params(labelsize=11)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig5_mae_distribution_histogram.tiff'), dpi=300)
    plt.close()
    print("✓ MAE distribution histogram saved (integer ticks)")

    # RMSE对比柱状图
    fig = plt.figure(figsize=(12 / 2.54, 8 / 2.54))

    rmse_values = {
        'Training': rmse_train,
        'Testing': rmse_test,
        'Total': rmse_total,
        'LOO-CV': RMSE_LOO_total,
        'Train-LOO-CV': train_loo_results['RMSE_LOO_train']
    }

    bars = plt.bar(rmse_values.keys(), rmse_values.values(),
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                   edgecolor='black', linewidth=1.2, alpha=0.7)

    for bar, value in zip(bars, rmse_values.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=11)

    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('RMSE (nm)', fontsize=12)
    plt.xticks(rotation=15, fontsize=11)
    plt.tick_params(labelsize=11)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'FigS6_rmse_comparison_bar.tiff'), dpi=300)
    plt.close()
    print("✓ RMSE comparison bar chart saved")

    # ----------------------------------威廉姆斯图（四种版本）------------------------------------------
    print(f"\nGenerating Williams plots (four versions)")
    print("-" * 60)

    residuals = total_y - total_y_pred
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    H = total_X @ np.linalg.pinv(total_X.T @ total_X) @ total_X.T
    leverages = np.diag(H)

    # 区分训练集和测试集（用于前两个版本）
    is_train = np.zeros(len(total_y), dtype=bool)
    if 'PID' in total_df.columns:
        train_pids = train_df['PID'].values
        for i, pid in enumerate(total_df['PID'].values):
            if pid in train_pids:
                is_train[i] = True
    else:
        is_train[:len(train_df)] = True

    train_leverages = leverages[is_train]
    train_std_residuals = std_residuals[is_train]
    test_leverages = leverages[~is_train]
    test_std_residuals = std_residuals[~is_train]

    n_features = total_X.shape[1]
    n_train = len(train_df)
    n_full = len(total_X)

    h_star_full = 3 * (n_features + 1) / n_full
    h_star_train = 3 * (n_features + 1) / n_train

    # ============================================================
    # 画法1：全部数据阈值 - 区分训练/测试
    # ============================================================
    print(f"\n  [Version 1] Full dataset threshold (split): h* = {h_star_full:.4f}")

    fig1, ax1 = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    sc1 = ax1.scatter(train_leverages, train_std_residuals,
                      c=np.abs(train_std_residuals), cmap=cmap,
                      alpha=0.7, edgecolors='k', s=20, marker='o', label='Training set')
    sc2 = ax1.scatter(test_leverages, test_std_residuals,
                      c=np.abs(test_std_residuals), cmap=cmap,
                      alpha=0.7, edgecolors='k', s=20, marker='s', label='Testing set')

    ax1.axhline(y=3, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='±3σ')
    ax1.axhline(y=-3, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.axvline(x=h_star_full, color='blue', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'h* = {h_star_full:.4f}')

    ax1.set_xlabel('Leverage (h)', fontsize=12)
    ax1.set_ylabel('Standardized residual', fontsize=12)
    ax1.tick_params(labelsize=11)
    ax1.legend(fontsize=9, frameon=False, loc='upper right')

    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('|Standardized residual|', fontsize=11)
    cbar1.ax.tick_params(labelsize=10)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig1, os.path.join(figures_dir, 'Fig6a_williams_plot_full.tiff'), dpi=300)
    plt.close()
    print("  ✓ Williams plot (full data threshold, split) saved")

    # ============================================================
    # 画法2：训练集阈值 - 区分训练/测试
    # ============================================================
    print(f"\n  [Version 2] Training set threshold (split): h* = {h_star_train:.4f}")

    fig2, ax2 = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    sc3 = ax2.scatter(train_leverages, train_std_residuals,
                      c=np.abs(train_std_residuals), cmap=cmap,
                      alpha=0.7, edgecolors='k', s=20, marker='o', label='Training set')
    sc4 = ax2.scatter(test_leverages, test_std_residuals,
                      c=np.abs(test_std_residuals), cmap=cmap,
                      alpha=0.7, edgecolors='k', s=20, marker='s', label='Testing set')

    ax2.axhline(y=3, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='±3σ')
    ax2.axhline(y=-3, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.axvline(x=h_star_train, color='blue', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'h* = {h_star_train:.4f}')

    ax2.set_xlabel('Leverage (h)', fontsize=12)
    ax2.set_ylabel('Standardized residual', fontsize=12)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=9, frameon=False, loc='upper right')

    cbar2 = plt.colorbar(sc3, ax=ax2)
    cbar2.set_label('|Standardized residual|', fontsize=11)
    cbar2.ax.tick_params(labelsize=10)

    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig2, os.path.join(figures_dir, 'Fig6b_williams_plot_train.tiff'), dpi=300)
    plt.close()
    print("  ✓ Williams plot (training set threshold, split) saved")

    # ============================================================
    # 画法3：全部数据阈值 - 不区分数据集 (统一用圈)
    # ============================================================
    print(f"\n  [Version 3] Full dataset threshold (no split): h* = {h_star_full:.4f}")

    fig3, ax3 = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    sc5 = ax3.scatter(leverages, std_residuals,
                      c=np.abs(std_residuals), cmap=cmap,
                      alpha=0.7, edgecolors='k', s=20, marker='o')

    ax3.axhline(y=3, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='±3σ')
    ax3.axhline(y=-3, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax3.axvline(x=h_star_full, color='blue', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'h* = {h_star_full:.4f}')

    ax3.set_xlabel('Leverage (h)', fontsize=12)
    ax3.set_ylabel('Standardized residual', fontsize=12)
    ax3.tick_params(labelsize=11)
    ax3.legend(fontsize=9, frameon=False, loc='upper right')

    cbar3 = plt.colorbar(sc5, ax=ax3)
    cbar3.set_label('|Standardized residual|', fontsize=11)
    cbar3.ax.tick_params(labelsize=10)

    for spine in ax3.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig3, os.path.join(figures_dir, 'Fig6c_williams_plot_full_nospilt.tiff'), dpi=300)
    plt.close()
    print("  ✓ Williams plot (full data threshold, no split) saved")

    # ============================================================
    # 画法4：训练集阈值 - 不区分数据集 (统一用圈)
    # ============================================================
    print(f"\n  [Version 4] Training set threshold (no split): h* = {h_star_train:.4f}")

    fig4, ax4 = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    sc6 = ax4.scatter(leverages, std_residuals,
                      c=np.abs(std_residuals), cmap=cmap,
                      alpha=0.7, edgecolors='k', s=20, marker='o')

    ax4.axhline(y=3, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='±3σ')
    ax4.axhline(y=-3, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.axvline(x=h_star_train, color='blue', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'h* = {h_star_train:.4f}')

    ax4.set_xlabel('Leverage (h)', fontsize=12)
    ax4.set_ylabel('Standardized residual', fontsize=12)
    ax4.tick_params(labelsize=11)
    ax4.legend(fontsize=9, frameon=False, loc='upper right')

    cbar4 = plt.colorbar(sc6, ax=ax4)
    cbar4.set_label('|Standardized residual|', fontsize=11)
    cbar4.ax.tick_params(labelsize=10)

    for spine in ax4.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig4, os.path.join(figures_dir, 'Fig6d_williams_plot_train_nospilt.tiff'), dpi=300)
    plt.close()
    print("  ✓ Williams plot (training set threshold, no split) saved")

    # ============================================================
    # 异常点统计
    # ============================================================
    outliers_full = (np.abs(std_residuals) > 3) | (leverages > h_star_full)
    outliers_train = (np.abs(std_residuals) > 3) | (leverages > h_star_train)

    print(f"\n  Outlier statistics:")
    print(f"    Full data threshold (h* = {h_star_full:.4f}): {np.sum(outliers_full)} outliers")
    print(f"    Training set threshold (h* = {h_star_train:.4f}): {np.sum(outliers_train)} outliers")

    if np.any(outliers_full):
        outlier_indices = np.where(outliers_full)[0]
        outlier_df = pd.DataFrame({
            'Sample_ID': outlier_indices + 1,
            'Dataset': ['Training' if is_train[i] else 'Testing' for i in outlier_indices],
            'Leverage': leverages[outliers_full],
            'Standardized_Residual': std_residuals[outliers_full],
            'Residual': residuals[outliers_full]
        })
        outlier_path = os.path.join(tables_dir, 'TableS4_outliers_detection.xlsx')
        outlier_df.to_excel(outlier_path, index=False)
        print(f"  ✓ Outliers detected ({np.sum(outliers_full)}), results saved to: {outlier_path}")

    # ----------------------------------学习曲线（3张独立图，10折）----------------------------------------
    print(f"\nGenerating learning curves (10-fold CV, 3 separate figures)...")
    print("-" * 60)

    learning_curve_data = plot_learning_curve_separate_mlr(X_train, y_train, figures_dir)

    lc_path = os.path.join(tables_dir, 'TableS5_learning_curve_data.xlsx')
    learning_curve_data.to_excel(lc_path, index=False)
    print(f"✓ Learning curve data saved to: {lc_path}")

    # ----------------------------------重复CV置信区间图（10折）--------------------------------------------
    print(f"\nGenerating repeated CV confidence interval plots (10-fold)...")
    print("-" * 60)

    # 重复CV箱线图
    fig, axes = plt.subplots(2, 3, figsize=(24 / 2.54, 16 / 2.54))
    axes = axes.flatten()

    metrics = ['R2_train', 'R2_cv', 'MAE_train', 'MAE_cv', 'RMSE_train', 'RMSE_cv']
    titles = ['Training R²', 'CV R² (10-fold)', 'Training MAE (nm)', 'CV MAE (nm)',
              'Training RMSE (nm)', 'CV RMSE (nm)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[i]
        data = cv_results[metric]['values']

        bp = ax.boxplot(data, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.7, linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.3))

        mean_val = cv_results[metric]['mean']
        ax.plot(1, mean_val, 'r*', markersize=12, label=f'Mean: {mean_val:.4f}')

        ax.errorbar(1, mean_val,
                    yerr=[[mean_val - cv_results[metric]['ci_lower']],
                          [cv_results[metric]['ci_upper'] - mean_val]],
                    fmt='none', ecolor='red', capsize=5, capthick=2)

        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_xticklabels(['10 repeats'], fontsize=10)
        ax.legend(fontsize=9, loc='best')
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'FigS7_repeated_cv_boxplots.tiff'), dpi=300)
    plt.close()
    print("✓ Repeated CV boxplots saved")

    # 重复CV置信区间图（误差棒图）
    fig, ax = plt.subplots(figsize=(14 / 2.54, 10 / 2.54))

    metrics_display = ['R² Train', 'R² CV (10-fold)', 'MAE Train', 'MAE CV', 'RMSE Train', 'RMSE CV']
    x_pos = np.arange(len(metrics_display))
    means = []
    lower_ci = []
    upper_ci = []

    for metric in metrics:
        result = cv_results[metric]
        means.append(result['mean'])
        lower_ci.append(result['ci_lower'])
        upper_ci.append(result['ci_upper'])

    ax.errorbar(x_pos, means,
                yerr=[np.array(means) - np.array(lower_ci), np.array(upper_ci) - np.array(means)],
                fmt='o', color='black', capsize=5, capthick=2, markersize=10,
                ecolor='red', elinewidth=2)

    for i, (x, y, l, u) in enumerate(zip(x_pos, means, lower_ci, upper_ci)):
        ax.annotate(f'{y:.4f}', (x, y), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=10)
        ax.annotate(f'CI: [{l:.4f}, {u:.4f}]', (x, (l + u) / 2),
                    xytext=(0, -28), textcoords='offset points',
                    ha='center', fontsize=9, color='red')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_display, fontsize=11, rotation=15)
    ax.set_ylabel('Value', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'FigS8_repeated_cv_confidence_intervals.tiff'), dpi=300)
    plt.close()
    print("✓ Repeated CV confidence intervals plot saved")

    # ----------------------------------VIF柱状图（使用真实名称）------------------------------------
    print(f"\nGenerating VIF bar plot...")
    print("-" * 60)

    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))

    vif_sorted = vif_df.sort_values('VIF', ascending=True)
    colors_vif = ['red' if v > 10 else 'orange' if v > 5 else 'steelblue'
                  for v in vif_sorted['VIF'].values]

    ax.barh(range(len(vif_sorted)), vif_sorted['VIF'].values,
            color=colors_vif, edgecolor='black', alpha=0.8, linewidth=1.2)

    ax.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='VIF = 5')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='VIF = 10')

    ax.set_yticks(range(len(vif_sorted)))
    ax.set_yticklabels(vif_sorted['Feature'].values, fontsize=7)
    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    for i, (idx, row) in enumerate(vif_sorted.iterrows()):
        ax.text(row['VIF'] + 0.5, i, f'{row["VIF"]:.2f}',
                va='center', fontsize=7)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'FigS9_vif_bar_plot.tiff'), dpi=300)
    plt.close()
    print("✓ VIF bar plot saved")

    # ----------------------------------SHAP分析部分（使用真实名称）----------------------------------
    print(f"\n{'=' * 70}")
    print("SHAP Analysis (using real descriptor names)")
    print("=" * 70)

    print("1. Computing SHAP values...")

    try:
        explainer = shap.LinearExplainer(model, total_X, feature_names=display_names)
        shap_values = explainer.shap_values(total_X)

        print(f"   ✓ SHAP values computed, shape: {shap_values.shape}")
        print(f"   ✓ Expected value (base value): {explainer.expected_value:.4f}")

        shap_sum = np.sum(shap_values, axis=1) + explainer.expected_value
        pred_diff = np.mean(np.abs(shap_sum - model.predict(total_X)))
        print(f"   ✓ SHAP verification - mean prediction difference: {pred_diff:.6f}")

    except Exception as e:
        print(f"   ✗ LinearExplainer failed: {e}")
        print("   Trying KernelExplainer...")

        try:
            background = shap.sample(total_X, min(50, len(total_X)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(total_X)

            print(f"   ✓ SHAP values computed using KernelExplainer")
            print(f"   ✓ Shape: {shap_values.shape}")
            print(f"   ✓ Expected value: {explainer.expected_value:.4f}")

        except Exception as e2:
            print(f"   ✗ KernelExplainer also failed: {e2}")
            print("   Skipping SHAP analysis...")
            return model, None, None

    # 保存SHAP值
    shap_df = pd.DataFrame(shap_values, columns=display_names)
    shap_df.to_csv(f'{shap_dir}/shap_values.csv', index=False)
    print(f"   ✓ SHAP values saved to {shap_dir}/shap_values.csv")

    # 特征重要性（使用真实名称）
    importance_df = pd.DataFrame({
        'feature': display_names,
        'coefficient': coefficients,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    importance_df.to_csv(f'{shap_dir}/feature_importance.csv', index=False)
    print(f"   ✓ Feature importance saved to {shap_dir}/feature_importance.csv")

    # 获取Top 20特征
    top_20_indices = importance_df.head(20).index.tolist()
    top_20_display = [display_names[i] for i in top_20_indices]
    shap_values_top20 = shap_values[:, top_20_indices]
    total_X_top20 = total_X[:, top_20_indices]

    # 生成SHAP图表
    print("\n2. Generating SHAP plots (TIFF, 300 dpi)...")

    shap_style_dir = f'{shap_dir}/plots'
    os.makedirs(shap_style_dir, exist_ok=True)

    # ===== SHAP蜂群图 - 35个特征 =====
    print("  Generating FigS1: SHAP beeswarm (35 features)...")
    try:
        plt.figure(figsize=(20 / 2.54, 18 / 2.54))
        shap.summary_plot(shap_values, total_X, feature_names=display_names,
                          show=False, max_display=35, plot_type="dot",
                          color_bar=True, cmap=plt.cm.coolwarm)
        plt.tight_layout()
        save_figure_tiff(plt.gcf(), f'{shap_style_dir}/FigS1_shap_beeswarm_35features.tiff')
        plt.close()
        print("    ✓ Saved")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # ===== SHAP蜂群图 - Top 20 =====
    print("  Generating FigS2: SHAP beeswarm (Top 20)...")
    try:
        plt.figure(figsize=(16 / 2.54, 14 / 2.54))
        shap.summary_plot(shap_values_top20, total_X_top20, feature_names=top_20_display,
                          show=False, max_display=20, plot_type="dot",
                          color_bar=True, cmap=plt.cm.coolwarm)
        plt.tight_layout()
        save_figure_tiff(plt.gcf(), f'{shap_style_dir}/FigS2_shap_beeswarm_top20.tiff')
        plt.close()
        print("    ✓ Saved")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # ===== SHAP热力图 - 35个特征（高度增加）=====
    print("  Generating FigS3: SHAP heatmap (35 features, height increased)...")
    try:
        setup_cjche_style_heatmap()
        fig_height_cm = len(display_names) * 0.8 + 5
        fig_height_inch = fig_height_cm / 2.54
        plt.figure(figsize=(28 / 2.54, fig_height_inch))
        shap.plots.heatmap(shap.Explanation(values=shap_values,
                                            base_values=np.full(shap_values.shape[0], explainer.expected_value),
                                            data=total_X, feature_names=display_names), max_display=35, show=False)
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        save_figure_tiff(plt.gcf(), f'{shap_style_dir}/FigS3_shap_heatmap_35features.tiff')
        plt.close()
        setup_cjche_style_larger()
        print("    ✓ Saved")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        setup_cjche_style_larger()

    # ===== SHAP热力图 - Top 20（高度增加）=====
    print("  Generating FigS4: SHAP heatmap (Top 20, height increased)...")
    try:
        setup_cjche_style_heatmap()
        fig_height_cm = len(top_20_display) * 0.8 + 4
        fig_height_inch = fig_height_cm / 2.54
        plt.figure(figsize=(24 / 2.54, fig_height_inch))
        shap.plots.heatmap(shap.Explanation(values=shap_values_top20,
                                            base_values=np.full(shap_values_top20.shape[0], explainer.expected_value),
                                            data=total_X_top20, feature_names=top_20_display), max_display=20,
                           show=False)
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=9)
        plt.tight_layout()
        save_figure_tiff(plt.gcf(), f'{shap_style_dir}/FigS4_shap_heatmap_top20.tiff')
        plt.close()
        setup_cjche_style_larger()
        print("    ✓ Saved")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        setup_cjche_style_larger()

    # ===== SHAP特征重要性 - 35个特征 =====
    print("  Generating FigS5: SHAP importance bar (35 features)...")
    try:
        fig_height_cm = len(importance_df) * 0.4 + 3
        fig_height_inch = fig_height_cm / 2.54
        plt.figure(figsize=(16 / 2.54, fig_height_inch))
        y_pos = np.arange(len(importance_df))
        plt.barh(y_pos, importance_df['mean_abs_shap'].values,
                 color='steelblue', edgecolor='black', alpha=0.8, linewidth=1.2)
        plt.yticks(y_pos, importance_df['feature'].values, fontsize=7)
        plt.xlabel('mean(|SHAP value|)', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tick_params(labelsize=11)
        plt.tight_layout()
        save_figure_tiff(plt.gcf(), f'{shap_style_dir}/FigS5_shap_importance_bar_35features.tiff')
        plt.close()
        print("    ✓ Saved")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # ===== SHAP特征重要性 - Top 20 =====
    print("  Generating FigS6: SHAP importance bar (Top 20)...")
    try:
        plt.figure(figsize=(14 / 2.54, 12 / 2.54))
        top_20_df = importance_df.head(20)
        y_pos = np.arange(len(top_20_df))
        plt.barh(y_pos, top_20_df['mean_abs_shap'].values,
                 color='steelblue', edgecolor='black', alpha=0.8, linewidth=1.2)
        plt.yticks(y_pos, top_20_df['feature'].values, fontsize=8)
        plt.xlabel('mean(|SHAP value|)', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tick_params(labelsize=11)
        plt.tight_layout()
        save_figure_tiff(plt.gcf(), f'{shap_style_dir}/FigS6_shap_importance_bar_top20.tiff')
        plt.close()
        print("    ✓ Saved")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # ===== SHAP瀑布图 =====
    print("  Generating FigS7: SHAP waterfall...")
    try:
        plt.figure(figsize=(14 / 2.54, 10 / 2.54))
        shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                              base_values=explainer.expected_value,
                                              data=total_X[0],
                                              feature_names=display_names), max_display=15, show=False)
        plt.tight_layout()
        save_figure_tiff(plt.gcf(), f'{shap_style_dir}/FigS7_shap_waterfall.tiff')
        plt.close()
        print("    ✓ Saved")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # ===== SHAP依赖图（前5个特征）=====
    print("  Generating FigS8: SHAP dependence plots (Top 5 features)...")
    top_5_features = importance_df.head(5)['feature'].tolist()
    top_5_indices = importance_df.head(5).index.tolist()

    for i, (feature, idx) in enumerate(zip(top_5_features, top_5_indices)):
        print(f"    {i + 1}. {feature}")
        try:
            fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54))
            shap.dependence_plot(idx, shap_values, total_X,
                                 feature_names=display_names,
                                 interaction_index='auto', show=False)
            plt.tight_layout()
            safe_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_')
            save_figure_tiff(plt.gcf(), f'{shap_style_dir}/FigS8_dependence_{safe_name}.tiff')
            plt.close()
            print(f"      ✓ Saved")
        except Exception as e:
            print(f"      ✗ Failed: {e}")

    # ===== 系数与SHAP相关性分析 =====
    print("\n   Generating FigS9: Coefficient vs SHAP correlation...")
    try:
        mean_shap = np.abs(shap_values).mean(axis=0)
        corr = np.corrcoef(np.abs(coefficients), mean_shap)[0, 1]

        fig = plt.figure(figsize=(12 / 2.54, 10 / 2.54))

        sizes = mean_shap / mean_shap.max() * 500 + 50

        scatter = plt.scatter(np.abs(coefficients), mean_shap,
                              s=sizes, alpha=0.6, c=mean_shap,
                              cmap='viridis', edgecolors='black', linewidth=0.5)

        z = np.polyfit(np.abs(coefficients), mean_shap, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(np.abs(coefficients)), max(np.abs(coefficients)), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=1.2, label=f'Trend line (slope={z[0]:.4f})')

        top_indices = np.argsort(mean_shap)[-5:][::-1]
        for idx in top_indices:
            plt.annotate(display_names[idx],
                         (np.abs(coefficients[idx]), mean_shap[idx]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.8,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.2))

        plt.xlabel('|Coefficient|', fontsize=12)
        plt.ylabel('Mean |SHAP value|', fontsize=12)
        plt.tick_params(labelsize=11)
        plt.legend(loc='best', fontsize=9, frameon=False)
        plt.colorbar(scatter, label='Mean |SHAP|')
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        save_figure_tiff(fig, f'{shap_style_dir}/FigS9_coefficient_shap_correlation.tiff')
        plt.close()
        print(f"   ✓ Correlation plot saved (r = {corr:.4f})")
    except Exception as e:
        print(f"   ✗ Error generating correlation plot: {e}")

    # ===== 特征值 vs SHAP值图（前10个特征）=====
    print("\n   Generating FigS10: Feature value vs SHAP plots (top 10 features)...")
    top_10_features = importance_df.head(10)['feature'].tolist()
    top_10_indices = importance_df.head(10).index.tolist()

    for i, (feature, idx) in enumerate(zip(top_10_features, top_10_indices)):
        try:
            feature_vals = total_X[:, idx]
            shap_vals = shap_values[:, idx]

            fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

            colors = ['#FF6B9D' if val > 0 else '#4A90E2' if val < 0 else '#7F7F7F' for val in shap_vals]

            scatter = ax.scatter(feature_vals, shap_vals, c=colors, alpha=0.7, s=15,
                                 edgecolors='black', linewidth=0.5)

            try:
                sort_idx = np.argsort(feature_vals)
                sorted_x = feature_vals[sort_idx]
                sorted_y = shap_vals[sort_idx]

                window = max(3, len(sorted_x) // 10)
                if window > 1:
                    moving_avg = np.convolve(sorted_y, np.ones(window) / window, mode='valid')
                    moving_x = sorted_x[window - 1:len(sorted_x)]
                    ax.plot(moving_x, moving_avg, "k-", alpha=0.8, linewidth=1.0, label='Moving average')
            except:
                pass

            ax.set_xlabel(f'{feature}', fontsize=10)
            ax.set_ylabel('SHAP value', fontsize=10)
            ax.tick_params(labelsize=10)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#FF6B9D', edgecolor='black', label='Positive SHAP (increases prediction)'),
                Patch(facecolor='#4A90E2', edgecolor='black', label='Negative SHAP (decreases prediction)')
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=8, frameon=False)

            plt.tight_layout()

            safe_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_')
            save_figure_tiff(fig, f'{shap_style_dir}/FigS10_feature_vs_shap_{safe_name}.tiff')
            plt.close()

            print(f"   ✓ {i + 1}. {feature}")
        except Exception as e:
            print(f"   ✗ Error generating feature vs SHAP plot for {feature}: {e}")

    # 生成报告
    print("\n6. Generating analysis report...")

    try:
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MLR MODEL WITH SHAP ANALYSIS - COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("MODEL PERFORMANCE:")
        report_lines.append("-" * 50)
        report_lines.append(f"R² (Training): {r2_train:.4f}")
        report_lines.append(f"R² (Testing): {r2_test:.4f}")
        report_lines.append(f"R² (Total): {r2_total:.4f}")
        report_lines.append(f"Q² (LOO-CV Full): {Q2_total:.4f}")
        report_lines.append(f"Q² (Train-LOO-CV): {train_loo_results['Q2_train_LOO']:.4f}")
        report_lines.append(f"MAE (Training): {mae_train:.4f} nm")
        report_lines.append(f"MAE (Testing): {mae_test:.4f} nm")
        report_lines.append(f"MAE (Total): {mae_total:.4f} nm")
        report_lines.append(f"MAE (LOO-CV Full): {MAE_LOO_total:.4f} nm")
        report_lines.append(f"MAE (Train-LOO-CV): {train_loo_results['MAE_LOO_train']:.4f} nm")
        report_lines.append(f"RMSE (Training): {rmse_train:.2f} nm")
        report_lines.append(f"RMSE (Testing): {rmse_test:.2f} nm")
        report_lines.append(f"RMSE (Total): {rmse_total:.2f} nm")
        report_lines.append(f"RMSE (LOO-CV Full): {RMSE_LOO_total:.2f} nm")
        report_lines.append(f"RMSE (Train-LOO-CV): {train_loo_results['RMSE_LOO_train']:.2f} nm")
        report_lines.append(f"MedAE (Training): {medae_train:.4f} nm")
        report_lines.append(f"MedAE (Testing): {medae_test:.4f} nm")
        report_lines.append(f"MedAE (Total): {medae_total:.4f} nm")
        report_lines.append(f"MedAE (LOO-CV Full): {MedAE_LOO_total:.4f} nm")
        report_lines.append(f"MedAE (Train-LOO-CV): {train_loo_results['MedAE_LOO_train']:.4f} nm")
        report_lines.append(f"MaxAE (Training): {maxae_train:.4f} nm")
        report_lines.append(f"MaxAE (Testing): {maxae_test:.4f} nm")
        report_lines.append(f"MaxAE (Total): {maxae_total:.4f} nm")
        report_lines.append(f"MaxAE (LOO-CV Full): {MaxAE_LOO_total:.4f} nm")
        report_lines.append(f"MaxAE (Train-LOO-CV): {train_loo_results['MaxAE_LOO_train']:.4f} nm")
        report_lines.append(f"Bias (Training): {bias_train:.4f} nm")
        report_lines.append(f"Bias (Testing): {bias_test:.4f} nm")
        report_lines.append(f"Bias (Total): {bias_total:.4f} nm")
        report_lines.append(f"Bias (LOO-CV Full): {Bias_LOO_total:.4f} nm")
        report_lines.append(f"Bias (Train-LOO-CV): {train_loo_results['Bias_LOO_train']:.4f} nm")
        report_lines.append(f"Overfitting: {overfit:.4f}")
        report_lines.append("")
        report_lines.append("PREDICTION ACCURACY WITHIN THRESHOLDS:")
        report_lines.append("-" * 50)
        report_lines.append(
            f"Training set: ≤10 nm: {pct10_train:.1f}%, ≤20 nm: {pct20_train:.1f}%, ≤30 nm: {pct30_train:.1f}%")
        report_lines.append(
            f"Testing set:  ≤10 nm: {pct10_test:.1f}%, ≤20 nm: {pct20_test:.1f}%, ≤30 nm: {pct30_test:.1f}%")
        report_lines.append(
            f"Total set:    ≤10 nm: {pct10_total:.1f}%, ≤20 nm: {pct20_total:.1f}%, ≤30 nm: {pct30_total:.1f}%")
        report_lines.append(
            f"LOO-CV:       ≤10 nm: {pct10_loo_total:.1f}%, ≤20 nm: {pct20_loo_total:.1f}%, ≤30 nm: {pct30_loo_total:.1f}%")
        report_lines.append(
            f"Train-LOO-CV: ≤10 nm: {train_loo_results['Pct10_LOO_train']:.1f}%, ≤20 nm: {train_loo_results['Pct20_LOO_train']:.1f}%, ≤30 nm: {train_loo_results['Pct30_LOO_train']:.1f}%")
        report_lines.append("")

        report_lines.append("VIF STATISTICS:")
        report_lines.append("-" * 50)
        report_lines.append(f"Mean VIF: {vif_df['VIF'].mean():.4f}")
        report_lines.append(f"Max VIF: {vif_df['VIF'].max():.4f}")
        report_lines.append(f"Min VIF: {vif_df['VIF'].min():.4f}")
        report_lines.append(f"Features with VIF > 10: {len(vif_df[vif_df['VIF'] > 10])}")
        report_lines.append("")

        report_lines.append("REPEATED CV RESULTS (10 repeats, 10-fold CV):")
        report_lines.append("-" * 50)
        for key in ['R2_train', 'R2_cv', 'MAE_train', 'MAE_cv', 'RMSE_train', 'RMSE_cv',
                    'MedAE_train', 'MedAE_cv', 'MaxAE_train', 'MaxAE_cv', 'Bias_train', 'Bias_cv']:
            result = cv_results[key]
            report_lines.append(
                f"  {key}: Mean={result['mean']:.4f}, 95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        report_lines.append("")

        report_lines.append("WILLIAMS PLOT OUTLIER STATISTICS:")
        report_lines.append("-" * 50)
        report_lines.append(f"Full data threshold (h* = {h_star_full:.4f}): {np.sum(outliers_full)} outliers")
        report_lines.append(f"Training set threshold (h* = {h_star_train:.4f}): {np.sum(outliers_train)} outliers")
        report_lines.append("")

        report_lines.append("SHAP ANALYSIS SUMMARY:")
        report_lines.append("-" * 50)
        report_lines.append(f"Number of samples: {len(total_X)}")
        report_lines.append(f"Number of features: {len(display_names)}")
        report_lines.append(f"Model intercept: {intercept:.4f}")
        if 'explainer' in locals():
            report_lines.append(f"Expected SHAP value: {explainer.expected_value:.4f}")
        report_lines.append("")

        report_lines.append("TOP 20 FEATURES BY SHAP IMPORTANCE:")
        report_lines.append("-" * 50)

        for i, (idx, row) in enumerate(importance_df.head(20).iterrows()):
            report_lines.append(f"{i + 1}. {row['feature']}")
            report_lines.append(f"   Coefficient: {row['coefficient']:.6f}")
            report_lines.append(f"   Mean |SHAP|: {row['mean_abs_shap']:.6f}")
            report_lines.append(f"   Mean SHAP: {row['mean_shap']:.6f}")
            if i < 19:
                report_lines.append("")

        report_lines.append("")
        if 'corr' in locals():
            report_lines.append("CORRELATION ANALYSIS:")
            report_lines.append("-" * 50)
            report_lines.append(f"Correlation between |coefficient| and mean |SHAP|: {corr:.4f}")
            report_lines.append("")

        report_lines.append("FILES GENERATED (CJChE Format):")
        report_lines.append("-" * 50)
        report_lines.append(f"Output directory: {output_dir}")
        report_lines.append("")
        report_lines.append("Tables:")
        report_lines.append(f"  {tables_dir}/Table1_model_performance_summary.xlsx")
        report_lines.append(f"  {tables_dir}/Table2_predictions_comparison.xlsx")
        report_lines.append(f"  {tables_dir}/Table3_regression_coefficients.xlsx")
        report_lines.append(f"  {tables_dir}/Table4_loo_cv_detailed_results.xlsx")
        report_lines.append(f"  {tables_dir}/TableS1_vif_results.xlsx")
        report_lines.append(f"  {tables_dir}/TableS2_repeated_cv_confidence_intervals.xlsx")
        report_lines.append(f"  {tables_dir}/TableS3_repeated_cv_detailed.xlsx")
        report_lines.append(f"  {tables_dir}/TableS4_outliers_detection.xlsx")
        report_lines.append(f"  {tables_dir}/TableS5_learning_curve_data.xlsx")
        report_lines.append("")
        report_lines.append("Figures (TIFF, 300 dpi):")
        report_lines.append(f"  {figures_dir}/Fig1_training_set_scatter.tiff")
        report_lines.append(f"  {figures_dir}/Fig2_testing_set_scatter.tiff")
        report_lines.append(f"  {figures_dir}/Fig3_combined_dataset_scatter.tiff")
        report_lines.append(f"  {figures_dir}/Fig4_loo_cv_scatter.tiff")
        report_lines.append(f"  {figures_dir}/Fig5_mae_distribution_histogram.tiff")
        report_lines.append(f"  {figures_dir}/Fig6a_williams_plot_full.tiff")
        report_lines.append(f"  {figures_dir}/Fig6b_williams_plot_train.tiff")
        report_lines.append(f"  {figures_dir}/Fig6c_williams_plot_full_nospilt.tiff")
        report_lines.append(f"  {figures_dir}/Fig6d_williams_plot_train_nospilt.tiff")
        report_lines.append(f"  {figures_dir}/Fig7_learning_curve_r2.tiff")
        report_lines.append(f"  {figures_dir}/Fig8_learning_curve_rmse.tiff")
        report_lines.append(f"  {figures_dir}/Fig9_learning_curve_mae.tiff")
        report_lines.append(f"  {figures_dir}/FigS5_train_loo_cv_scatter.tiff")
        report_lines.append(f"  {figures_dir}/FigS6_rmse_comparison_bar.tiff")
        report_lines.append(f"  {figures_dir}/FigS7_repeated_cv_boxplots.tiff")
        report_lines.append(f"  {figures_dir}/FigS8_repeated_cv_confidence_intervals.tiff")
        report_lines.append(f"  {figures_dir}/FigS9_vif_bar_plot.tiff")
        report_lines.append("")
        report_lines.append("SHAP analysis (using real descriptor names):")
        report_lines.append(f"  {shap_dir}/shap_values.csv")
        report_lines.append(f"  {shap_dir}/feature_importance.csv")
        report_lines.append(f"  {shap_dir}/shap_analysis_report.txt")
        report_lines.append(f"  {shap_dir}/plots/FigS1_shap_beeswarm_35features.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS2_shap_beeswarm_top20.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS3_shap_heatmap_35features.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS4_shap_heatmap_top20.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS5_shap_importance_bar_35features.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS6_shap_importance_bar_top20.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS7_shap_waterfall.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS8_dependence_*.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS9_coefficient_shap_correlation.tiff")
        report_lines.append(f"  {shap_dir}/plots/FigS10_feature_vs_shap_*.tiff")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("ANALYSIS COMPLETED SUCCESSFULLY!")
        report_lines.append("Features:")
        report_lines.append("  ✓ Font size increased (labels: 12pt, ticks: 11pt)")
        report_lines.append("  ✓ Border width increased (axes: 1.5pt)")
        report_lines.append("  ✓ Grid lines removed")
        report_lines.append("  ✓ Heatmap height increased")
        report_lines.append("  ✓ All chart titles removed")
        report_lines.append("  ✓ All descriptors use real names")
        report_lines.append("  ✓ Learning curves as 3 separate figures")
        report_lines.append("  ✓ MAE histogram with integer ticks")
        report_lines.append("  ✓ Output directory: mlr_model_35_result_more")
        report_lines.append("  ✓ CV: 10 repeats of 10-fold cross-validation")
        report_lines.append("  ✓ Williams plot: four versions (split/no-split, two thresholds)")
        report_lines.append("  ✓ Training set: circles, Testing set: squares (split versions)")
        report_lines.append("  ✓ Color map indicates |standardized residual|")
        report_lines.append("  ✓ Added MedAE, MaxAE, and Bias metrics")
        report_lines.append("  ✓ Added prediction accuracy within ±10, ±20, ±30 nm")
        report_lines.append("=" * 80)

        with open(f'{shap_dir}/shap_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print("\n" + "=" * 70)
        print("SHAP Analysis Completed!")
        print("=" * 70)
        print(f"\nAll results saved to: {output_dir}/")

    except Exception as e:
        print(f"   ✗ Error generating report: {e}")

    print(f"\n{'=' * 70}")
    print("All plots have been successfully generated and saved!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    return model, shap_values, importance_df


if __name__ == '__main__':
    model, shap_values, importance_df = main()