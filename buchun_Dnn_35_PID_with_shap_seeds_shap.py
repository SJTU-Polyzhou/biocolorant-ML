# -*- coding: utf-8 -*-
"""
包含SHAP分析的完整DNN模型脚本 - 增强版 (多随机种子 + SHAP稳定性)
CJChE格式版本 - 所有图表输出为TIFF 300 dpi
所有图表已去标题，符合CJChE要求
35个Mordred描述符版本
轴标签简化为 Exp. λmax (nm) / Pre. λmax (nm)

新增功能：
- 使用MAE（平均绝对误差）
- 增加中位绝对误差 (MedAE)、最大绝对误差 (MaxAE)、预测偏差 (Bias)
- train-Q²_LOO-CV（仅训练集432个样本）
- VIF计算
- 10次重复10折CV获取置信区间
- 学习曲线（10折交叉验证）
- 字体大一号，边框加粗，无网格线
- 热力图高度增加，纵轴标签清晰
- 所有图表无标题
- 输出路径为 dnn_model_35_result_more_seeds_shap
- 所有描述符使用真实名称
- 学习曲线为3张独立图
- MAE柱状图使用整数刻度
- SHAP图增加至10+张，并增加多随机种子稳定性分析
- VIF柱状图两种画法（固定大小 + 自适应大小）
- 威廉姆斯图四种版本（区分/不区分数据集，两种阈值）
- 增加预测误差在±10、±20、±30 nm内的化合物百分比
- 多随机种子性能评估（3个种子：42, 123, 2023）
- SHAP稳定性分析（排名热图、平均排名、Spearman相关系数等）
- 所有SHAP相关输出保存在 more_shap_analysis 目录下
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings
from matplotlib.colors import Normalize
import shap
import time
from scipy.stats import spearmanr

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

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
    """
    加载描述符真实名称映射
    Excel中 Descriptor_Code 格式为 'Md_906'
    """
    if not os.path.exists(info_file_path):
        print(f"  ⚠️ 警告: {info_file_path} 不存在，将使用默认 Md_XXX 格式")
        return None

    df_info = pd.read_excel(info_file_path)
    mapping = {}
    for _, row in df_info.iterrows():
        code = row['Descriptor_Code']  # 格式为 'Md_906'
        name = row['Descriptor_Name']
        if '.' in name:
            name = name.split('.')[-1]
        mapping[code] = name  # 键为 'Md_906'，值为真实名称
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


def repeated_cv_confidence(model_params, X_train, y_train, n_repeats=10, n_folds=10):
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

            scaler_X_fold = StandardScaler()
            X_tr_fold_scaled = scaler_X_fold.fit_transform(X_tr_fold)
            X_cv_fold_scaled = scaler_X_fold.transform(X_cv_fold)

            scaler_y_fold = StandardScaler()
            y_tr_fold_scaled = scaler_y_fold.fit_transform(y_tr_fold.reshape(-1, 1)).ravel()

            model = MLPRegressor(**model_params)
            model.fit(X_tr_fold_scaled, y_tr_fold_scaled)

            y_tr_pred_scaled = model.predict(X_tr_fold_scaled)
            y_cv_pred_scaled = model.predict(X_cv_fold_scaled)

            y_tr_pred = scaler_y_fold.inverse_transform(y_tr_pred_scaled.reshape(-1, 1)).ravel()
            y_cv_pred = scaler_y_fold.inverse_transform(y_cv_pred_scaled.reshape(-1, 1)).ravel()

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


def train_loo_cv(X_train, y_train, model_params):
    """仅对训练集进行LOO-CV计算Q²及额外指标"""
    loo = LeaveOneOut()
    y_true_loo = []
    y_pred_loo = []

    print("  Running LOO-CV on training set (432 samples)...")

    for i, (train_idx, test_idx) in enumerate(loo.split(X_train)):
        X_train_loo, X_test_loo = X_train[train_idx], X_train[test_idx]
        y_train_loo, y_test_loo = y_train[train_idx], y_train[test_idx]

        scaler_X_loo = StandardScaler()
        X_train_loo_scaled = scaler_X_loo.fit_transform(X_train_loo)
        X_test_loo_scaled = scaler_X_loo.transform(X_test_loo)

        scaler_y_loo = StandardScaler()
        y_train_loo_scaled = scaler_y_loo.fit_transform(y_train_loo.reshape(-1, 1)).ravel()

        model_loo = MLPRegressor(**model_params)
        model_loo.fit(X_train_loo_scaled, y_train_loo_scaled)

        y_pred_loo_scaled = model_loo.predict(X_test_loo_scaled)
        y_pred_loo_value = scaler_y_loo.inverse_transform(y_pred_loo_scaled.reshape(-1, 1)).ravel()[0]

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

    # 计算误差百分比
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


def plot_learning_curve_separate(X_train, y_train, model_params, figures_dir):
    """绘制学习曲线 - 3张独立图，无网格，无标题，10折交叉验证"""
    print("  Generating learning curves with 10-fold CV...")

    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores_r2 = []
    cv_scores_r2 = []
    train_scores_rmse = []
    cv_scores_rmse = []
    train_scores_mae = []
    cv_scores_mae = []

    scaler_X_lc = StandardScaler()
    X_train_scaled = scaler_X_lc.fit_transform(X_train)
    scaler_y_lc = StandardScaler()
    y_train_scaled = scaler_y_lc.fit_transform(y_train.reshape(-1, 1)).ravel()

    for size in train_sizes:
        n_samples = int(len(X_train_scaled) * size)
        indices = np.random.choice(len(X_train_scaled), n_samples, replace=False)
        X_subset = X_train_scaled[indices]
        y_subset = y_train_scaled[indices]

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

            model_lc = MLPRegressor(**model_params)
            model_lc.fit(X_tr, y_tr)

            y_tr_pred = model_lc.predict(X_tr)
            y_cv_pred = model_lc.predict(X_cv)

            y_tr_pred_orig = scaler_y_lc.inverse_transform(y_tr_pred.reshape(-1, 1)).ravel()
            y_cv_pred_orig = scaler_y_lc.inverse_transform(y_cv_pred.reshape(-1, 1)).ravel()
            y_tr_orig = scaler_y_lc.inverse_transform(y_tr.reshape(-1, 1)).ravel()
            y_cv_orig = scaler_y_lc.inverse_transform(y_cv.reshape(-1, 1)).ravel()

            train_r2.append(r2_score(y_tr_orig, y_tr_pred_orig))
            cv_r2.append(r2_score(y_cv_orig, y_cv_pred_orig))
            train_rmse.append(np.sqrt(mean_squared_error(y_tr_orig, y_tr_pred_orig)))
            cv_rmse.append(np.sqrt(mean_squared_error(y_cv_orig, y_cv_pred_orig)))
            train_mae.append(mean_absolute_error(y_tr_orig, y_tr_pred_orig))
            cv_mae.append(mean_absolute_error(y_cv_orig, y_cv_pred_orig))

        train_scores_r2.append(np.mean(train_r2))
        cv_scores_r2.append(np.mean(cv_r2))
        train_scores_rmse.append(np.mean(train_rmse))
        cv_scores_rmse.append(np.mean(cv_rmse))
        train_scores_mae.append(np.mean(train_mae))
        cv_scores_mae.append(np.mean(cv_mae))

    # R²学习曲线
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
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig8_learning_curve_r2.tiff'), dpi=300)
    plt.close()
    print("  ✓ Learning curve R² saved")

    # RMSE学习曲线
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
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig9_learning_curve_rmse.tiff'), dpi=300)
    plt.close()
    print("  ✓ Learning curve RMSE saved")

    # MAE学习曲线
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
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig10_learning_curve_mae.tiff'), dpi=300)
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


def evaluate_seed_performance(X_train, y_train, X_test, y_test, total_X, total_y,
                              scaler_y, final_params, seeds, output_dir, tables_dir):
    """
    评估多个随机种子的模型性能，并返回训练好的模型列表。
    """
    print(f"\n{'=' * 70}")
    print("Multi-Seed Performance Evaluation")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    seed_results = []
    models = []  # 存储训练好的模型

    for seed in seeds:
        print(f"\n  Training model with random_state={seed}...")

        # 更新模型参数
        params = final_params.copy()
        params['random_state'] = seed

        # 每个种子独立标准化
        scaler_X_seed = StandardScaler()
        X_train_scaled = scaler_X_seed.fit_transform(X_train)
        X_test_scaled = scaler_X_seed.transform(X_test)
        total_X_scaled = scaler_X_seed.transform(total_X)

        scaler_y_seed = StandardScaler()
        y_train_scaled = scaler_y_seed.fit_transform(y_train.reshape(-1, 1)).ravel()

        # 训练模型
        model_seed = MLPRegressor(**params)
        start_time = time.time()
        model_seed.fit(X_train_scaled, y_train_scaled)
        training_time = time.time() - start_time

        # 保存模型
        models.append(model_seed)

        # 预测
        y_train_pred_scaled = model_seed.predict(X_train_scaled)
        y_test_pred_scaled = model_seed.predict(X_test_scaled)
        total_y_pred_scaled = model_seed.predict(total_X_scaled)

        y_train_pred = scaler_y_seed.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        y_test_pred = scaler_y_seed.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
        total_y_pred = scaler_y_seed.inverse_transform(total_y_pred_scaled.reshape(-1, 1)).ravel()

        # 计算指标
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

        # 误差百分比
        train_residuals = np.abs(y_train - y_train_pred)
        test_residuals = np.abs(y_test - y_test_pred)
        total_residuals = np.abs(total_y - total_y_pred)

        pct10_train = np.mean(train_residuals <= 10) * 100
        pct20_train = np.mean(train_residuals <= 20) * 100
        pct30_train = np.mean(train_residuals <= 30) * 100

        pct10_test = np.mean(test_residuals <= 10) * 100
        pct20_test = np.mean(test_residuals <= 20) * 100
        pct30_test = np.mean(test_residuals <= 30) * 100

        pct10_total = np.mean(total_residuals <= 10) * 100
        pct20_total = np.mean(total_residuals <= 20) * 100
        pct30_total = np.mean(total_residuals <= 30) * 100

        # 存储结果
        seed_results.append({
            'Seed': seed,
            'Training_Time': training_time,
            'R2_Train': r2_train,
            'R2_Test': r2_test,
            'R2_Total': r2_total,
            'Overfit': overfit,
            'RMSE_Train': rmse_train,
            'RMSE_Test': rmse_test,
            'RMSE_Total': rmse_total,
            'MAE_Train': mae_train,
            'MAE_Test': mae_test,
            'MAE_Total': mae_total,
            'MedAE_Train': medae_train,
            'MedAE_Test': medae_test,
            'MedAE_Total': medae_total,
            'MaxAE_Train': maxae_train,
            'MaxAE_Test': maxae_test,
            'MaxAE_Total': maxae_total,
            'Bias_Train': bias_train,
            'Bias_Test': bias_test,
            'Bias_Total': bias_total,
            'Pct10_Train': pct10_train,
            'Pct20_Train': pct20_train,
            'Pct30_Train': pct30_train,
            'Pct10_Test': pct10_test,
            'Pct20_Test': pct20_test,
            'Pct30_Test': pct30_test,
            'Pct10_Total': pct10_total,
            'Pct20_Total': pct20_total,
            'Pct30_Total': pct30_total
        })

        print(f"    Seed={seed}: R²_Train={r2_train:.4f}, R²_Test={r2_test:.4f}, MAE_Test={mae_test:.2f} nm")

    # 转换为DataFrame
    seed_results_df = pd.DataFrame(seed_results)

    # 计算统计信息
    print(f"\n  Summary Statistics across {len(seeds)} seeds:")
    print("-" * 50)
    for metric in ['R2_Train', 'R2_Test', 'R2_Total', 'Overfit',
                   'RMSE_Test', 'MAE_Test', 'MedAE_Test', 'MaxAE_Test']:
        mean_val = seed_results_df[metric].mean()
        std_val = seed_results_df[metric].std()
        print(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}")

    # 保存结果
    seed_results_path = os.path.join(tables_dir, 'TableS6_multi_seed_performance.xlsx')
    seed_results_df.to_excel(seed_results_path, index=False)
    print(f"\n  ✓ Multi-seed results saved to: {seed_results_path}")

    # 绘制多种子性能对比图
    plot_multi_seed_comparison(seed_results_df, output_dir)

    return seed_results_df, models  # 返回模型列表


def plot_multi_seed_comparison(seed_results_df, output_dir):
    """绘制多随机种子性能对比图"""
    figures_dir = os.path.join(output_dir, 'figures')

    print("\n  Generating multi-seed comparison plots...")

    # 定义要绘制的指标
    metrics = [
        ('R2_Train', 'R² (Training)', 'R²'),
        ('R2_Test', 'R² (Testing)', 'R²'),
        ('R2_Total', 'R² (Total)', 'R²'),
        ('MAE_Train', 'MAE (Training)', 'MAE (nm)'),
        ('MAE_Test', 'MAE (Testing)', 'MAE (nm)'),
        ('MAE_Total', 'MAE (Total)', 'MAE (nm)'),
        ('RMSE_Train', 'RMSE (Training)', 'RMSE (nm)'),
        ('RMSE_Test', 'RMSE (Testing)', 'RMSE (nm)'),
        ('RMSE_Total', 'RMSE (Total)', 'RMSE (nm)'),
        ('Overfit', 'Overfitting (R²_Train - R²_Test)', 'Overfitting')
    ]

    seeds = seed_results_df['Seed'].values

    # 创建多指标对比图（子图布局）
    fig, axes = plt.subplots(2, 5, figsize=(28 / 2.54, 12 / 2.54))
    axes = axes.flatten()

    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        values = seed_results_df[metric].values

        # 柱状图
        bars = ax.bar(range(len(seeds)), values,
                      color='steelblue', edgecolor='black',
                      alpha=0.7, linewidth=1.2, width=0.6)

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(1, abs(val)),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        # 添加均值线
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Mean: {mean_val:.4f}')

        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f'S{int(s)}' for s in seeds], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9, frameon=False)
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # 隐藏多余的子图
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'FigS10_multi_seed_comparison.tiff'), dpi=300)
    plt.close()
    print("  ✓ Multi-seed comparison plot saved")

    # 绘制种子性能箱线图汇总
    fig, ax = plt.subplots(figsize=(14 / 2.54, 10 / 2.54))

    # 选择主要指标绘制箱线图
    plot_metrics = ['R2_Train', 'R2_Test', 'R2_Total',
                    'MAE_Train', 'MAE_Test', 'MAE_Total',
                    'RMSE_Train', 'RMSE_Test', 'RMSE_Total']
    metric_labels = ['R² (Train)', 'R² (Test)', 'R² (Total)',
                     'MAE (Train)', 'MAE (Test)', 'MAE (Total)',
                     'RMSE (Train)', 'RMSE (Test)', 'RMSE (Total)']

    # 准备数据
    data_to_plot = []
    for metric in plot_metrics:
        data_to_plot.append(seed_results_df[metric].values)

    # 绘制箱线图
    bp = ax.boxplot(data_to_plot, patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.5))

    # 着色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#1f77b4', '#ff7f0e', '#2ca02c',
              '#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(metric_labels, fontsize=10, rotation=15)
    ax.set_ylabel('Value', fontsize=12)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'FigS11_multi_seed_summary_boxplot.tiff'), dpi=300)
    plt.close()
    print("  ✓ Multi-seed summary boxplot saved")


def shap_stability_analysis(models, X_explain, feature_names, display_names,
                            scaler_y, output_dir, seeds, n_top=10):
    """
    对多个模型（不同随机种子）计算SHAP特征重要性，评估排序稳定性。
    结果保存在 output_dir/more_shap_analysis/ 下。
    输出：
    - shap_rank_stability.xlsx: 特征排名稳定性汇总
    - shap_importance_matrix.xlsx: 每个模型的特征重要性分数
    - shap_rank_matrix.xlsx: 每个模型的特征排名
    - shap_spearman_matrix.xlsx: 种子间Spearman相关系数矩阵
    - shap_top_k_overlap.xlsx: Top-K特征重叠率
    """
    import shap
    from scipy.stats import spearmanr
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 创建输出目录
    out_dir = os.path.join(output_dir, 'more_shap_analysis')
    os.makedirs(out_dir, exist_ok=True)

    n_models = len(models)
    n_features = X_explain.shape[1]
    all_importances = []  # 每个模型的重要性向量

    print("\n" + "=" * 70)
    print("SHAP Stability Analysis for Multiple Seeds")
    print("=" * 70)

    # 对每个模型计算 SHAP 重要性（使用相同的解释数据集）
    for idx, model in enumerate(models):
        print(f"  Computing SHAP for model {idx + 1}/{n_models} (seed={seeds[idx]})...")

        def predict_fn(x):
            # x 已经是标准化后的特征
            pred_scaled = model.predict(x)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            return pred

        # 使用 KernelExplainer（对MLP通用）
        # 背景集：从解释数据中随机抽样 100 个样本加速
        background = shap.sample(X_explain, 100, random_state=42)
        explainer = shap.KernelExplainer(predict_fn, background)

        # 对全部解释数据计算 SHAP
        shap_values = explainer.shap_values(X_explain, nsamples=50)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        all_importances.append(mean_abs_shap)

    # 构建 DataFrame
    imp_df = pd.DataFrame(all_importances, columns=display_names)
    # 添加种子标识 (使用 seed42, seed123, seed2023 格式)
    imp_df.insert(0, 'Seed', [f'seed{s}' for s in seeds])

    # 排名（数值越大越重要，所以降序排名）
    rank_df = imp_df.copy()
    for idx in range(n_models):
        rank_df.iloc[idx, 1:] = imp_df.iloc[idx, 1:].rank(axis=0, method='min', ascending=False).astype(int)

    # 计算稳定性指标
    mean_rank = rank_df.iloc[:, 1:].mean(axis=0)
    std_rank = rank_df.iloc[:, 1:].std(axis=0)

    # 两两 Spearman 相关系数
    n_models = len(all_importances)
    spearman_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            spearman_matrix[i, j] = spearmanr(all_importances[i], all_importances[j])[0]

    # 创建Spearman矩阵DataFrame (使用 seed42, seed123, seed2023 格式)
    spearman_df = pd.DataFrame(spearman_matrix,
                               index=[f'seed{s}' for s in seeds],
                               columns=[f'seed{s}' for s in seeds])

    avg_spearman = np.mean(spearman_matrix[np.triu_indices_from(spearman_matrix, k=1)])

    # Top-K 重叠率
    top_k_overlap = []
    overlap_pairs = []
    for i in range(n_models):
        top_i = set(rank_df.iloc[i, 1:][rank_df.iloc[i, 1:] <= n_top].index)
        for j in range(i + 1, n_models):
            top_j = set(rank_df.iloc[j, 1:][rank_df.iloc[j, 1:] <= n_top].index)
            overlap = len(top_i & top_j) / n_top
            top_k_overlap.append(overlap)
            overlap_pairs.append((f'seed{seeds[i]}', f'seed{seeds[j]}', overlap))
    avg_top_overlap = np.mean(top_k_overlap)

    # 创建Top-K重叠率DataFrame
    overlap_df = pd.DataFrame(overlap_pairs, columns=['Seed_1', 'Seed_2', f'Top_{n_top}_Overlap'])

    print(f"\nStability Metrics:")
    print(f"  Average Spearman correlation: {avg_spearman:.4f}")
    print(f"  Average Top-{n_top} overlap: {avg_top_overlap:.4f}")
    print(f"  Mean rank std (across features): {np.mean(std_rank):.2f}")

    # ---- 保存表格 ----
    # 1. 特征排名稳定性汇总
    stability_df = pd.DataFrame({
        'Feature': display_names,
        'Mean_Rank': mean_rank,
        'Std_Rank': std_rank
    }).sort_values('Mean_Rank')
    stability_df.to_excel(os.path.join(out_dir, 'shap_rank_stability.xlsx'), index=False)

    # 2. 每个模型的特征重要性分数
    imp_df.to_excel(os.path.join(out_dir, 'shap_importance_matrix.xlsx'), index=False)

    # 3. 每个模型的特征排名
    rank_df.to_excel(os.path.join(out_dir, 'shap_rank_matrix.xlsx'), index=False)

    # 4. Spearman相关系数矩阵
    spearman_df.to_excel(os.path.join(out_dir, 'shap_spearman_matrix.xlsx'))

    # 5. Top-K重叠率
    overlap_df.to_excel(os.path.join(out_dir, 'shap_top_k_overlap.xlsx'), index=False)

    print(f"\n  ✓ Tables saved to: {out_dir}")
    print(f"    - shap_rank_stability.xlsx: Feature ranking stability summary")
    print(f"    - shap_importance_matrix.xlsx: SHAP importance scores for each seed")
    print(f"    - shap_rank_matrix.xlsx: Feature rankings for each seed")
    print(f"    - shap_spearman_matrix.xlsx: Spearman correlation matrix between seeds")
    print(f"    - shap_top_k_overlap.xlsx: Top-{n_top} feature overlap between seeds")

    # ---- 可视化（使用与主代码一致的样式：无标题、统一字体大小、边框加粗） ----

    # 1. 排名热图（使用coolwarm_r，与主代码一致）
    fig, ax = plt.subplots(figsize=(14, max(6, n_features * 0.4)))

    # 使用coolwarm_r，与主代码散点图颜色映射一致
    im = ax.imshow(rank_df.iloc[:, 1:].values, cmap='coolwarm_r', aspect='auto', interpolation='nearest')

    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(display_names, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(n_models))
    ax.set_yticklabels([f'seed{s}' for s in seeds], fontsize=11)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Random Seeds', fontsize=12)

    # 边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 刻度样式
    ax.tick_params(labelsize=11)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank (lower = more important)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(out_dir, 'Fig_rank_heatmap.tiff'))
    plt.close()

    # 2. 平均排名 + 误差条（Top 20）
    sorted_idx = np.argsort(mean_rank)
    top_n_show = min(20, n_features)
    top_features = [display_names[i] for i in sorted_idx[:top_n_show]]
    top_mean_rank = mean_rank.iloc[sorted_idx[:top_n_show]]
    top_std_rank = std_rank.iloc[sorted_idx[:top_n_show]]

    fig, ax = plt.subplots(figsize=(10, top_n_show * 0.4))
    y_pos = np.arange(top_n_show)

    ax.barh(y_pos, top_mean_rank, xerr=top_std_rank, align='center',
            color='steelblue', ecolor='black', capsize=3, edgecolor='black', linewidth=1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Rank (± std)', fontsize=12)

    # 边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(out_dir, 'Fig_avg_rank_top20.tiff'))
    plt.close()

    # 3. 重要性箱线图（所有特征）
    fig, ax = plt.subplots(figsize=(14, 6))

    # 使用与主代码一致的箱线图样式
    imp_df_no_seed = imp_df.iloc[:, 1:]
    imp_df_no_seed.boxplot(ax=ax, rot=90, fontsize=10, patch_artist=True,
                           boxprops=dict(linewidth=1.5),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.5))

    ax.set_ylabel('Mean |SHAP value|', fontsize=12)

    # 边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(out_dir, 'Fig_importance_boxplot.tiff'))
    plt.close()

    # 4. Spearman相关系数矩阵热图
    fig, ax = plt.subplots(figsize=(6, 5))

    # 使用coolwarm_r，与主代码一致
    im = ax.imshow(spearman_matrix, cmap='coolwarm_r', vmin=0, vmax=1)

    # 修改标签：使用 seed42, seed123, seed2023 格式
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels([f'seed{seeds[i]}' for i in range(n_models)], fontsize=11)
    ax.set_yticklabels([f'seed{seeds[i]}' for i in range(n_models)], fontsize=11)

    # 添加数值标签
    for i in range(n_models):
        for j in range(n_models):
            ax.text(j, i, f'{spearman_matrix[i, j]:.2f}',
                    ha='center', va='center', color='black', fontsize=10)

    # 边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(labelsize=11)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spearman correlation', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(out_dir, 'Fig_spearman_matrix.tiff'))
    plt.close()

    print(f"✓ SHAP stability analysis results saved to: {out_dir}")
    return imp_df, rank_df


def main():
    # ================= 路径配置 =================
    cd_excel = 'origin_data/'
    excel_name = 'data_w_Md_bo.xlsx'

    # 输出根目录
    output_dir = os.path.join('03_modeling', 'DNN_model', 'dnn_model_35_result_more_seeds_shap')
    os.makedirs(output_dir, exist_ok=True)

    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # SHAP分析输出目录
    shap_dir = os.path.join(output_dir, 'more_shap_analysis')
    os.makedirs(shap_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("DNN Model - Deep Neural Network with 35 descriptors (Enhanced)")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")

    # ================= 1. 数据准备 =================
    print("Step 1: Data loading and preprocessing")
    print("-" * 60)

    file_path = f'{cd_excel}{excel_name}'
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None, None, None

    df = pd.read_excel(file_path, engine='openpyxl')
    train_df = df[df['Training set/Testing set'] == 'Training set'].copy()
    test_df = df[df['Training set/Testing set'] == 'Testing set'].copy()
    total_df = pd.concat([train_df, test_df], ignore_index=True)

    selected_bits = [
        906, 1336, 334, 1310, 1348, 1406, 1437, 345, 218, 230,
        449, 510, 299, 3, 1547, 566, 1418, 1069, 232, 264,
        1076, 1077, 322, 1313, 1294, 1064, 1329, 1146, 1477, 1359,
        805, 1394, 1358, 1573, 485
    ]

    feature_names = [f'Md_{bit}' for bit in selected_bits]

    # ===== 加载描述符真实名称映射 =====
    mapping = load_descriptor_mapping('mordred_descriptor_info.xlsx')
    if mapping:
        display_names = [mapping.get(f, f) for f in feature_names]
        print(f"✓ Loaded {len(mapping)} descriptor name mappings")
        print("  Example mappings:")
        for i in range(min(5, len(feature_names))):
            print(f"    {feature_names[i]} -> {display_names[i]}")
    else:
        display_names = feature_names.copy()
        print("  Using default Md_XXX names")

    print(f"Data loaded successfully:")
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Testing set: {len(test_df)} samples")
    print(f"  Total samples: {len(total_df)} samples")
    print(f"  Number of descriptors: {len(selected_bits)}")

    X_train_raw = train_df[[f'Md_{bit}' for bit in selected_bits]].values
    y_train = train_df['bo'].values.reshape(-1, 1)
    X_test_raw = test_df[[f'Md_{bit}' for bit in selected_bits]].values
    y_test = test_df['bo'].values
    total_X_raw = total_df[[f'Md_{bit}' for bit in selected_bits]].values
    total_y = total_df['bo'].values

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)
    total_X = scaler_X.transform(total_X_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f"\nFeature standardization completed")
    print(f"Target standardization completed")

    # ================= 2. 最终模型参数 =================
    print("\nStep 2: Setting final model parameters")
    print("-" * 60)

    final_params = {
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

    print("Final model parameters:")
    for k, v in final_params.items():
        print(f"  {k}: {v}")

    # ================= 2.5 多随机种子评估 =================
    print(f"\nStep 2.5: Multi-seed performance evaluation")
    print("-" * 60)

    seeds = [42, 123, 2023]
    print(f"Seeds to evaluate: {seeds}")

    # 执行多种子评估，并获取模型列表
    seed_results_df, seed_models = evaluate_seed_performance(
        X_train_raw, y_train.ravel(),
        X_test_raw, y_test,
        total_X_raw, total_y.ravel(),
        scaler_y, final_params, seeds, output_dir, tables_dir
    )

    # ================= 2.6 SHAP稳定性分析（多随机种子） =================
    print(f"\nStep 2.6: SHAP stability analysis across seeds")
    print("-" * 60)
    # 使用标准化后的全数据集作为解释数据
    X_explain = total_X  # 已标准化
    shap_stability_analysis(
        seed_models, X_explain, feature_names, display_names,
        scaler_y, output_dir, seeds, n_top=10
    )

    # ================= 3. 训练最终模型 =================
    print("\nStep 3: Training final model (with seed=123)")
    print("-" * 60)

    model = MLPRegressor(**final_params)
    start_time = time.time()
    model.fit(X_train, y_train_scaled)
    training_time = time.time() - start_time

    print(f"Model training completed! Time: {training_time:.2f} seconds")

    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)
    total_y_pred_scaled = model.predict(total_X)

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    total_y_pred = scaler_y.inverse_transform(total_y_pred_scaled.reshape(-1, 1)).ravel()

    train_residuals = np.abs(y_train.ravel() - y_train_pred)
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

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_total = r2_score(total_y, total_y_pred)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmse_total = np.sqrt(mean_squared_error(total_y, total_y_pred))

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mae_total = mean_absolute_error(total_y, total_y_pred)

    medae_train = median_absolute_error(y_train.ravel(), y_train_pred)
    medae_test = median_absolute_error(y_test, y_test_pred)
    medae_total = median_absolute_error(total_y, total_y_pred)

    maxae_train = np.max(np.abs(y_train.ravel() - y_train_pred))
    maxae_test = np.max(np.abs(y_test - y_test_pred))
    maxae_total = np.max(np.abs(total_y - total_y_pred))

    bias_train = np.mean(y_train_pred - y_train.ravel())
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

    # ================= 4. LOO-CV验证（全数据集） =================
    print(f"\nStep 4: LOO-CV Validation (Full dataset)")
    print("-" * 60)

    loo = LeaveOneOut()
    y_true_loo = []
    y_pred_loo = []

    print("Running LOO-CV validation on full dataset, this may take some time...")

    for i, (train_index, test_index) in enumerate(loo.split(total_X)):
        X_train_loo, X_test_loo = total_X[train_index], total_X[test_index]
        y_train_loo, y_test_loo = total_y[train_index], total_y[test_index]

        y_train_loo_scaled = scaler_y.transform(y_train_loo.reshape(-1, 1)).ravel()

        model_loo = MLPRegressor(**final_params)
        model_loo.fit(X_train_loo, y_train_loo_scaled)

        y_pred_loo_scaled = model_loo.predict(X_test_loo)
        y_pred_loo_value = scaler_y.inverse_transform(y_pred_loo_scaled.reshape(-1, 1)).ravel()[0]

        y_pred_loo.append(y_pred_loo_value)
        y_true_loo.append(y_test_loo[0])

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(total_y)} samples")

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

    # ================= 5. Train-LOO-CV =================
    print(f"\nStep 5: Train-LOO-CV (Training set only, 432 samples)")
    print("-" * 60)

    train_loo_results = train_loo_cv(X_train_raw, y_train.ravel(), final_params)

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

    # ================= 6. VIF计算 =================
    print(f"\nStep 6: VIF Calculation")
    print("-" * 60)

    vif_df = calculate_vif(X_train_raw, display_names)
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

    # ================= 7. 重复CV (10折) =================
    print(f"\nStep 7: Repeated CV (10 repeats, 10-fold CV for confidence intervals)")
    print("-" * 60)

    cv_results = repeated_cv_confidence(
        final_params,
        X_train_raw,
        y_train.ravel(),
        n_repeats=10,
        n_folds=10
    )

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

    # ================= 8. 保存表格 =================
    print(f"\nStep 8: Generating result tables")
    print("-" * 60)

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

    all_predictions = pd.DataFrame({
        'PID': total_df['PID'].values if 'PID' in total_df.columns else range(1, len(total_y) + 1),
        'Compound_ID': range(1, len(total_y) + 1),
        'Dataset': total_df['Training set/Testing set'].values,
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

    final_params_df = pd.DataFrame(list(final_params.items()), columns=['Parameter', 'Value'])
    final_params_df.to_excel(os.path.join(tables_dir, 'Table3_final_model_parameters.xlsx'), index=False)
    print(f"✓ Final model parameters saved")

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

    # ================= 9. 保存模型 =================
    joblib.dump(model, os.path.join(output_dir, 'final_dnn_model.pkl'))
    joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))
    print(f"\n✓ Model and scalers saved")

    # ================= 10. 散点图 =================
    print(f"\nStep 9: Generating scatter plots")
    print("-" * 60)

    cmap = plt.cm.coolwarm_r

    # 训练集散点图
    fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54))
    norm = Normalize(vmin=0, vmax=np.max(train_residuals))
    sc = plt.scatter(y_train.ravel(), y_train_pred,
                     c=train_residuals, cmap=cmap, norm=norm,
                     alpha=0.7, edgecolors='k', s=20)

    min_val = min(min(y_train.ravel()), min(y_train_pred))
    max_val = max(max(y_train.ravel()), max(y_train_pred))
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
    print("✓ Training set scatter plot saved")

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
    print("✓ Testing set scatter plot saved")

    # 总数据集散点图
    fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54))
    norm = Normalize(vmin=0, vmax=np.max(total_residuals))

    train_scatter = plt.scatter(y_train.ravel(), y_train_pred,
                                c=train_residuals, cmap=cmap, norm=norm,
                                alpha=0.7, edgecolors='k', s=20,
                                marker='o', label='Training set')

    test_scatter = plt.scatter(y_test, y_test_pred,
                               c=test_residuals, cmap=cmap, norm=norm,
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
    print("✓ Combined dataset scatter plot saved")

    # LOO-CV散点图
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
    print("✓ LOO-CV scatter plot saved")

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
    print("✓ Train-LOO-CV scatter plot saved")

    # ================= 11. MAE分布柱状图 =================
    print(f"\nStep 10: Generating MAE distribution histogram")
    print("-" * 60)

    fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))

    all_absolute_errors = np.concatenate([total_residuals, loo_residuals])
    max_error = np.max(all_absolute_errors)

    bin_width = 50
    max_bin = int(np.ceil(max_error / bin_width) * bin_width)
    bins = np.arange(0, max_bin + bin_width, bin_width)

    dnn_counts, _ = np.histogram(total_residuals, bins=bins)
    loo_counts, _ = np.histogram(loo_residuals, bins=bins)

    x_pos = bins[:-1] + bin_width * 0.25
    bars1 = ax.bar(x_pos, dnn_counts, width=bin_width * 0.35, alpha=0.7, color='#1f77b4',
                   edgecolor='black', linewidth=1.2, label='DNN model')
    bars2 = ax.bar(x_pos + bin_width * 0.35, loo_counts, width=bin_width * 0.35, alpha=0.7, color='#ff7f0e',
                   edgecolor='black', linewidth=1.2, label='LOO-CV')

    ax.set_xlabel('Absolute error (nm)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.set_xticks(np.arange(0, max_bin + bin_width, bin_width))
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'Fig5_mae_distribution_histogram.tiff'), dpi=300)
    plt.close()
    print("✓ MAE distribution histogram saved (integer ticks)")

    # ================= 12. RMSE对比柱状图 =================
    print(f"\nStep 11: Generating RMSE comparison bar chart")
    print("-" * 60)

    fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))

    rmse_values = {
        'Training': rmse_train,
        'Testing': rmse_test,
        'Total': rmse_total,
        'LOO-CV': RMSE_LOO_total,
        'Train-LOO-CV': train_loo_results['RMSE_LOO_train']
    }

    bars = ax.bar(rmse_values.keys(), rmse_values.values(),
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                  edgecolor='black', linewidth=1.2, alpha=0.7, width=0.6)

    for bar, value in zip(bars, rmse_values.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{value:.2f}', ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('RMSE (nm)', fontsize=12)
    ax.set_xticklabels(rmse_values.keys(), rotation=15, fontsize=11)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(figures_dir, 'FigS6_rmse_comparison_bar.tiff'), dpi=300)
    plt.close()
    print("✓ RMSE comparison bar chart saved")

    # ================= 13. 威廉姆斯图 (四种版本) =================
    print(f"\nStep 12: Generating Williams plots (four versions)")
    print("-" * 60)

    residuals = total_y - total_y_pred
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    H = total_X @ np.linalg.pinv(total_X.T @ total_X) @ total_X.T
    leverages = np.diag(H)

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

    # 画法1
    print("\n  [Version 1] Full dataset threshold (split): h* = {:.4f}".format(h_star_full))
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

    # 画法2
    print("\n  [Version 2] Training set threshold (split): h* = {:.4f}".format(h_star_train))
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

    # 画法3
    print("\n  [Version 3] Full dataset threshold (no split): h* = {:.4f}".format(h_star_full))
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

    # 画法4
    print("\n  [Version 4] Training set threshold (no split): h* = {:.4f}".format(h_star_train))
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
    # 输出异常点信息
    # ============================================================
    outliers_full = (np.abs(std_residuals) > 3) | (leverages > h_star_full)
    outliers_train = (np.abs(std_residuals) > 3) | (leverages > h_star_train)

    print(f"\n  Outlier statistics:")
    print(f"    Full data threshold (h* = {h_star_full:.4f}): {np.sum(outliers_full)} outliers")
    print(f"    Training set threshold (h* = {h_star_train:.4f}): {np.sum(outliers_train)} outliers")

    if np.any(outliers_full):
        outlier_indices = np.where(outliers_full)[0]

        # 构建异常点DataFrame，优先使用PID
        outlier_df = pd.DataFrame({
            'PID': [total_df['PID'].values[i] if 'PID' in total_df.columns else f'Sample_{i + 1}' for i in
                    outlier_indices],
            'Dataset': ['Training' if is_train[i] else 'Testing' for i in outlier_indices],
            'Leverage': leverages[outliers_full],
            'Standardized_Residual': std_residuals[outliers_full],
            'Residual': residuals[outliers_full],
            'Experimental_λmax': total_y[outliers_full],
            'Predicted_λmax': total_y_pred[outliers_full]
        })

        # 如果存在Compound列，添加到DataFrame中（放在PID后面）
        if 'Compound' in total_df.columns:
            outlier_df.insert(1, 'Compound', [total_df['Compound'].values[i] for i in outlier_indices])

        # 重新排列列顺序：PID优先
        cols = ['PID']
        if 'Compound' in total_df.columns:
            cols.append('Compound')
        cols.extend(['Dataset', 'Leverage', 'Standardized_Residual', 'Residual',
                     'Experimental_λmax', 'Predicted_λmax'])
        outlier_df = outlier_df[cols]

        outlier_path = os.path.join(tables_dir, 'TableS4_outliers_detection.xlsx')
        outlier_df.to_excel(outlier_path, index=False)
        print(f"  ✓ Outliers detected ({np.sum(outliers_full)}), results saved to: {outlier_path}")

        # 打印异常点摘要
        print(f"\n  Outlier summary (first 5):")
        print(outlier_df.head(5).to_string(index=False))

    # ================= 14. 损失曲线 =================
    print(f"\nStep 13: Generating loss curve")
    print("-" * 60)

    if hasattr(model, 'loss_curve_'):
        fig, ax = plt.subplots(figsize=(10 / 2.54, 7 / 2.54))
        iterations = range(1, len(model.loss_curve_) + 1)
        ax.plot(iterations, model.loss_curve_, 'b-', linewidth=2, alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.tick_params(labelsize=11)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        plt.tight_layout()
        save_figure_tiff(fig, os.path.join(figures_dir, 'Fig7_loss_curve.tiff'), dpi=300)
        plt.close()
        print("✓ Loss curve saved")

    # ================= 15. 学习曲线 =================
    print(f"\nStep 14: Generating learning curves (10-fold CV, 3 separate figures)")
    print("-" * 60)

    learning_curve_data = plot_learning_curve_separate(X_train_raw, y_train.ravel(), final_params, figures_dir)

    lc_path = os.path.join(tables_dir, 'TableS5_learning_curve_data.xlsx')
    learning_curve_data.to_excel(lc_path, index=False)
    print(f"✓ Learning curve data saved to: {lc_path}")

    # ================= 16. 重复CV置信区间图 =================
    print(f"\nStep 15: Generating repeated CV confidence interval plots (10-fold)")
    print("-" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(24 / 2.54, 16 / 2.54))
    axes = axes.flatten()

    metrics_cv = ['R2_train', 'R2_cv', 'MAE_train', 'MAE_cv', 'RMSE_train', 'RMSE_cv']
    titles = ['Training R²', 'CV R² (10-fold)', 'Training MAE (nm)', 'CV MAE (nm)',
              'Training RMSE (nm)', 'CV RMSE (nm)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (metric, title, color) in enumerate(zip(metrics_cv, titles, colors)):
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

    # 置信区间误差棒图
    fig, ax = plt.subplots(figsize=(14 / 2.54, 10 / 2.54))
    metrics_display = ['R² Train', 'R² CV (10-fold)', 'MAE Train', 'MAE CV', 'RMSE Train', 'RMSE CV']
    x_pos = np.arange(len(metrics_display))
    means = []
    lower_ci = []
    upper_ci = []
    for metric in metrics_cv:
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

    # ================= 17. VIF柱状图 =================
    print(f"\nStep 16: Generating VIF bar plots (two versions)")
    print("-" * 60)

    vif_sorted = vif_df.sort_values('VIF', ascending=True)

    # 固定大小
    print("\n  [Version 1] Fixed size (16cm x 14cm)")
    fig1, ax1 = plt.subplots(figsize=(16 / 2.54, 14 / 2.54))
    colors_vif1 = ['red' if v > 10 else 'orange' if v > 5 else 'steelblue'
                   for v in vif_sorted['VIF'].values]
    ax1.barh(range(len(vif_sorted)), vif_sorted['VIF'].values,
             color=colors_vif1, edgecolor='black', alpha=0.8, linewidth=1.2)
    ax1.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='VIF = 5')
    ax1.axvline(x=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='VIF = 10')
    ax1.set_yticks(range(len(vif_sorted)))
    ax1.set_yticklabels(vif_sorted['Feature'].values, fontsize=8)
    ax1.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
    ax1.legend(fontsize=10, frameon=False)
    ax1.tick_params(labelsize=11)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    for i, (idx, row) in enumerate(vif_sorted.iterrows()):
        ax1.text(row['VIF'] + 0.5, i, f'{row["VIF"]:.2f}',
                 va='center', fontsize=7)
    plt.tight_layout()
    save_figure_tiff(fig1, os.path.join(figures_dir, 'FigS9a_vif_bar_plot_fixed.tiff'), dpi=300)
    plt.close()
    print("  ✓ VIF bar plot (fixed size) saved")

    # 自适应大小
    print("\n  [Version 2] Adaptive size")
    max_label_len = max([len(label) for label in vif_sorted['Feature'].values])
    label_width_inch = max_label_len * 0.12 + 1.5
    bar_height = 0.5
    n_features = len(vif_sorted)
    fig_height_inch = max(6, n_features * bar_height + 1.0)
    fig_width_inch = max(6, label_width_inch + 2)
    print(f"    Number of features: {n_features}")
    print(f"    Max label length: {max_label_len} characters")
    print(f"    Figure size: {fig_width_inch:.1f} x {fig_height_inch:.1f} inches")

    fig2, ax2 = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
    colors_vif2 = ['red' if v > 10 else 'orange' if v > 5 else 'steelblue'
                   for v in vif_sorted['VIF'].values]
    bars = ax2.barh(range(len(vif_sorted)), vif_sorted['VIF'].values,
                    color=colors_vif2, edgecolor='black', alpha=0.8, linewidth=1.2,
                    height=bar_height * 0.98)
    ax2.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='VIF = 5')
    ax2.axvline(x=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='VIF = 10')
    ax2.set_yticks(range(len(vif_sorted)))
    ax2.set_yticklabels(vif_sorted['Feature'].values, fontsize=11)
    ax2.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
    ax2.legend(fontsize=10, frameon=False)
    ax2.tick_params(labelsize=11)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    for i, (idx, row) in enumerate(vif_sorted.iterrows()):
        ax2.text(row['VIF'] + 0.5, i, f'{row["VIF"]:.2f}',
                 va='center', fontsize=9)
    ax2.set_xlim(0, vif_sorted['VIF'].max() * 1.15)
    plt.tight_layout()
    save_figure_tiff(fig2, os.path.join(figures_dir, 'FigS9b_vif_bar_plot_adaptive.tiff'), dpi=300)
    plt.close()
    print("  ✓ VIF bar plot (adaptive size) saved")

    # ================= 18. 最终模型SHAP分析 (单个模型) =================
    print(f"\n{'=' * 70}")
    print("Step 17: SHAP Analysis for Final Model (seed=123)")
    print("Generating: 35 features + Top 20 features (using real descriptor names)")
    print("Output to: more_shap_analysis")
    print("=" * 70)

    print("1. Computing SHAP values for final model...")

    try:
        background_size = min(50, len(total_X))
        background_indices = np.random.choice(len(total_X), background_size, replace=False)
        background = total_X[background_indices]

        def model_predict(x):
            x_scaled = x
            y_pred_scaled = model.predict(x_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            return y_pred

        explainer = shap.KernelExplainer(model_predict, background)

        n_samples = min(100, len(total_X))
        sample_indices = np.random.choice(len(total_X), n_samples, replace=False)
        X_sample = total_X[sample_indices]

        print(f"  Computing SHAP for {n_samples} samples (this may take a while)...")
        shap_values = explainer.shap_values(X_sample, nsamples=100)

        print(f"  ✓ SHAP values computed, shape: {shap_values.shape}")
        print(f"  ✓ Expected value (base value): {explainer.expected_value:.4f}")

        # 保存SHAP值（使用 more_shap_analysis 目录）
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(os.path.join(shap_dir, 'shap_values_final_model.csv'), index=False)
        print(f"  ✓ SHAP values saved to {shap_dir}/shap_values_final_model.csv")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'display_name': display_names,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0),
            'mean_shap': shap_values.mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        importance_df.to_csv(os.path.join(shap_dir, 'feature_importance_final_model.csv'), index=False)
        print(f"  ✓ Feature importance saved to {shap_dir}/feature_importance_final_model.csv")

        top_20_indices = importance_df.head(20).index.tolist()
        top_20_features = [feature_names[i] for i in top_20_indices]
        top_20_display = [display_names[i] for i in top_20_indices]
        shap_values_top20 = shap_values[:, top_20_indices]
        X_sample_top20 = X_sample[:, top_20_indices]

        print("\n2. Generating SHAP plots (TIFF, 300 dpi)...")

        shap_style_dir = os.path.join(shap_dir, 'plots')
        os.makedirs(shap_style_dir, exist_ok=True)

        # ---- SHAP蜂群图 35 features ----
        print("  Generating FigS1: SHAP beeswarm (35 features)...")
        try:
            plt.figure(figsize=(20 / 2.54, 18 / 2.54))
            shap.summary_plot(shap_values, X_sample, feature_names=display_names,
                              show=False, max_display=35, plot_type="dot",
                              color_bar=True, cmap=plt.cm.coolwarm)
            plt.tight_layout()
            save_figure_tiff(plt.gcf(), os.path.join(shap_style_dir, 'FigS1_shap_beeswarm_35features.tiff'))
            plt.close()
            print("    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # ---- SHAP蜂群图 Top20 ----
        print("  Generating FigS2: SHAP beeswarm (Top 20)...")
        try:
            plt.figure(figsize=(16 / 2.54, 14 / 2.54))
            shap.summary_plot(shap_values_top20, X_sample_top20, feature_names=top_20_display,
                              show=False, max_display=20, plot_type="dot",
                              color_bar=True, cmap=plt.cm.coolwarm)
            plt.tight_layout()
            save_figure_tiff(plt.gcf(), os.path.join(shap_style_dir, 'FigS2_shap_beeswarm_top20.tiff'))
            plt.close()
            print("    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # ---- SHAP热力图 35 features ----
        print("  Generating FigS3: SHAP heatmap (35 features, height increased)...")
        try:
            setup_cjche_style_heatmap()
            fig_height_cm = len(display_names) * 0.8 + 5
            fig_height_inch = fig_height_cm / 2.54
            plt.figure(figsize=(28 / 2.54, fig_height_inch))

            explanation = shap.Explanation(
                values=shap_values,
                base_values=np.full(shap_values.shape[0], explainer.expected_value),
                data=X_sample,
                feature_names=display_names
            )
            shap.plots.heatmap(explanation, max_display=35, show=False)

            ax = plt.gca()
            n_labels = len(ax.get_yticklabels())
            if n_labels <= len(display_names):
                ax.set_yticklabels(display_names[:n_labels], fontsize=8)
            else:
                ax.set_yticklabels(display_names + [''] * (n_labels - len(display_names)), fontsize=8)

            plt.tight_layout()
            save_figure_tiff(plt.gcf(), os.path.join(shap_style_dir, 'FigS3_shap_heatmap_35features.tiff'))
            plt.close()
            setup_cjche_style_larger()
            print("    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            setup_cjche_style_larger()

        # ---- SHAP热力图 Top20 ----
        print("  Generating FigS4: SHAP heatmap (Top 20, height increased)...")
        try:
            setup_cjche_style_heatmap()
            fig_height_cm = len(top_20_display) * 0.8 + 4
            fig_height_inch = fig_height_cm / 2.54
            plt.figure(figsize=(24 / 2.54, fig_height_inch))

            explanation_top20 = shap.Explanation(
                values=shap_values_top20,
                base_values=np.full(shap_values_top20.shape[0], explainer.expected_value),
                data=X_sample_top20,
                feature_names=top_20_display
            )
            shap.plots.heatmap(explanation_top20, max_display=20, show=False)

            ax = plt.gca()
            n_labels = len(ax.get_yticklabels())
            if n_labels <= len(top_20_display):
                ax.set_yticklabels(top_20_display[:n_labels], fontsize=9)
            else:
                ax.set_yticklabels(top_20_display + [''] * (n_labels - len(top_20_display)), fontsize=9)

            plt.tight_layout()
            save_figure_tiff(plt.gcf(), os.path.join(shap_style_dir, 'FigS4_shap_heatmap_top20.tiff'))
            plt.close()
            setup_cjche_style_larger()
            print("    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            setup_cjche_style_larger()

        # ---- SHAP重要性条形图 35 features ----
        print("  Generating FigS5: SHAP importance bar (35 features)...")
        try:
            fig_height_cm = len(importance_df) * 0.4 + 3
            fig_height_inch = fig_height_cm / 2.54
            fig, ax = plt.subplots(figsize=(16 / 2.54, fig_height_inch))
            y_pos = np.arange(len(importance_df))
            ax.barh(y_pos, importance_df['mean_abs_shap'].values,
                    color='steelblue', edgecolor='black', alpha=0.8,
                    linewidth=1.2, height=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(importance_df['display_name'].values, fontsize=11)
            ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
            ax.invert_yaxis()
            ax.tick_params(labelsize=11)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            plt.tight_layout()
            save_figure_tiff(fig, os.path.join(shap_style_dir, 'FigS5_shap_importance_bar_35features.tiff'))
            plt.close()
            print("    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # ---- SHAP重要性条形图 Top20 ----
        print("  Generating FigS6: SHAP importance bar (Top 20)...")
        try:
            fig, ax = plt.subplots(figsize=(14 / 2.54, 12 / 2.54))
            top_20_importance = importance_df.head(20)
            y_pos = np.arange(len(top_20_importance))
            ax.barh(y_pos, top_20_importance['mean_abs_shap'].values,
                    color='steelblue', edgecolor='black', alpha=0.8,
                    linewidth=1.2, height=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_20_importance['display_name'].values, fontsize=11)
            ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
            ax.invert_yaxis()
            ax.tick_params(labelsize=11)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            plt.tight_layout()
            save_figure_tiff(plt.gcf(), os.path.join(shap_style_dir, 'FigS6_shap_importance_bar_top20.tiff'))
            plt.close()
            print("    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # ---- SHAP瀑布图 ----
        print("  Generating FigS7: SHAP waterfall...")
        try:
            plt.figure(figsize=(14 / 2.54, 10 / 2.54))
            shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                                  base_values=explainer.expected_value,
                                                  data=X_sample[0],
                                                  feature_names=display_names),
                                 max_display=15, show=False)
            plt.tight_layout()
            save_figure_tiff(plt.gcf(), os.path.join(shap_style_dir, 'FigS7_shap_waterfall.tiff'))
            plt.close()
            print("    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # ---- SHAP依赖图 Top5 ----
        print("  Generating FigS8: SHAP dependence plots (Top 5 features)...")
        top_5_display = importance_df.head(5)['display_name'].tolist()
        top_5_features = importance_df.head(5)['feature'].tolist()
        for i, (feature, display_name) in enumerate(zip(top_5_features, top_5_display)):
            print(f"    {i + 1}. {display_name}")
            try:
                feature_idx = feature_names.index(feature)
                fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))
                shap.dependence_plot(feature_idx, shap_values, X_sample,
                                     feature_names=display_names,
                                     interaction_index='auto', show=False)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                plt.tight_layout()
                safe_name = display_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_')
                save_figure_tiff(fig, os.path.join(shap_style_dir, f'FigS8_dependence_{safe_name}.tiff'))
                plt.close()
                print(f"      ✓ Saved")
            except Exception as e:
                print(f"      ✗ Failed: {e}")

        # ---- SHAP重要性分布图 ----
        print("\n   Generating FigS9: SHAP importance distribution...")
        try:
            mean_shap = importance_df['mean_abs_shap'].values
            fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
            x_vals = np.arange(len(mean_shap))
            sizes = mean_shap / mean_shap.max() * 500 + 50
            scatter = ax.scatter(x_vals, mean_shap, s=sizes, alpha=0.6,
                                 c=mean_shap, cmap='viridis',
                                 edgecolors='black', linewidth=0.5)
            top_10_indices = np.argsort(mean_shap)[-10:][::-1]
            for idx in top_10_indices:
                ax.annotate(importance_df.iloc[idx]['display_name'][:20],
                            (idx, mean_shap[idx]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
            ax.set_xlabel('Feature index (sorted by importance)', fontsize=12)
            ax.set_ylabel('Mean |SHAP value|', fontsize=12)
            ax.tick_params(labelsize=11)
            plt.colorbar(scatter, label='Mean |SHAP|')
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            plt.tight_layout()
            save_figure_tiff(fig, os.path.join(shap_style_dir, 'FigS9_shap_importance_distribution.tiff'))
            plt.close()
            print("   ✓ SHAP importance distribution plot saved")
        except Exception as e:
            print(f"   ✗ Error generating plot: {e}")

        # ================= 生成报告 =================
        print("\n3. Generating analysis report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DNN MODEL WITH SHAP ANALYSIS - COMPREHENSIVE REPORT")
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
        report_lines.append("CROSS-VALIDATION (10 repeats, 10-fold CV):")
        report_lines.append("-" * 50)
        for key in ['R2_train', 'R2_cv', 'MAE_train', 'MAE_cv', 'RMSE_train', 'RMSE_cv',
                    'MedAE_train', 'MedAE_cv', 'MaxAE_train', 'MaxAE_cv', 'Bias_train', 'Bias_cv']:
            result = cv_results[key]
            report_lines.append(
                f"{key}: {result['mean']:.4f} ± {result['std']:.4f} 95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        report_lines.append("")
        report_lines.append("MULTI-SEED PERFORMANCE (3 seeds):")
        report_lines.append("-" * 50)
        for metric in ['R2_Train', 'R2_Test', 'R2_Total', 'MAE_Train', 'MAE_Test', 'MAE_Total']:
            mean_val = seed_results_df[metric].mean()
            std_val = seed_results_df[metric].std()
            report_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        report_lines.append("")
        report_lines.append("SHAP STABILITY METRICS (across seeds):")
        report_lines.append("-" * 50)
        # 从稳定性分析中读取（如果已保存）
        stability_file = os.path.join(shap_dir, 'shap_rank_stability.xlsx')
        if os.path.exists(stability_file):
            stab_df = pd.read_excel(stability_file)
            report_lines.append("  Top 5 most stable features (lowest std rank):")
            top_stable = stab_df.sort_values('Std_Rank').head(5)
            for _, row in top_stable.iterrows():
                report_lines.append(
                    f"    {row['Feature']}: mean rank = {row['Mean_Rank']:.2f}, std = {row['Std_Rank']:.2f}")
        report_lines.append("")
        report_lines.append("WILLIAMS PLOT OUTLIER STATISTICS:")
        report_lines.append("-" * 50)
        report_lines.append(f"Full data threshold (h* = {h_star_full:.4f}): {np.sum(outliers_full)} outliers")
        report_lines.append(f"Training set threshold (h* = {h_star_train:.4f}): {np.sum(outliers_train)} outliers")
        report_lines.append("")
        report_lines.append("FILES GENERATED:")
        report_lines.append("-" * 50)
        report_lines.append(f"Output directory: {output_dir}")
        report_lines.append("")
        report_lines.append("Model files:")
        report_lines.append(f"  {output_dir}/final_dnn_model.pkl")
        report_lines.append(f"  {output_dir}/scaler_X.pkl")
        report_lines.append(f"  {output_dir}/scaler_y.pkl")
        report_lines.append("")
        report_lines.append("Tables:")
        report_lines.append(f"  {tables_dir}/Table1_model_performance_summary.xlsx")
        report_lines.append(f"  {tables_dir}/Table2_predictions_comparison.xlsx")
        report_lines.append(f"  {tables_dir}/Table3_final_model_parameters.xlsx")
        report_lines.append(f"  {tables_dir}/Table4_loo_cv_detailed_results.xlsx")
        report_lines.append(f"  {tables_dir}/TableS1_vif_results.xlsx")
        report_lines.append(f"  {tables_dir}/TableS2_repeated_cv_confidence_intervals.xlsx")
        report_lines.append(f"  {tables_dir}/TableS3_repeated_cv_detailed.xlsx")
        report_lines.append(f"  {tables_dir}/TableS4_outliers_detection.xlsx")
        report_lines.append(f"  {tables_dir}/TableS5_learning_curve_data.xlsx")
        report_lines.append(f"  {tables_dir}/TableS6_multi_seed_performance.xlsx")
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
        report_lines.append(f"  {figures_dir}/Fig7_loss_curve.tiff")
        report_lines.append(f"  {figures_dir}/Fig8_learning_curve_r2.tiff")
        report_lines.append(f"  {figures_dir}/Fig9_learning_curve_rmse.tiff")
        report_lines.append(f"  {figures_dir}/Fig10_learning_curve_mae.tiff")
        report_lines.append(f"  {figures_dir}/FigS5_train_loo_cv_scatter.tiff")
        report_lines.append(f"  {figures_dir}/FigS6_rmse_comparison_bar.tiff")
        report_lines.append(f"  {figures_dir}/FigS7_repeated_cv_boxplots.tiff")
        report_lines.append(f"  {figures_dir}/FigS8_repeated_cv_confidence_intervals.tiff")
        report_lines.append(f"  {figures_dir}/FigS9a_vif_bar_plot_fixed.tiff")
        report_lines.append(f"  {figures_dir}/FigS9b_vif_bar_plot_adaptive.tiff")
        report_lines.append(f"  {figures_dir}/FigS10_multi_seed_comparison.tiff")
        report_lines.append(f"  {figures_dir}/FigS11_multi_seed_summary_boxplot.tiff")
        report_lines.append("")
        report_lines.append("SHAP analysis (final model, in more_shap_analysis):")
        report_lines.append(f"  {shap_dir}/shap_values_final_model.csv")
        report_lines.append(f"  {shap_dir}/feature_importance_final_model.csv")
        report_lines.append(f"  {shap_dir}/shap_analysis_report.txt")
        report_lines.append(f"  {shap_style_dir}/FigS1_shap_beeswarm_35features.tiff")
        report_lines.append(f"  {shap_style_dir}/FigS2_shap_beeswarm_top20.tiff")
        report_lines.append(f"  {shap_style_dir}/FigS3_shap_heatmap_35features.tiff")
        report_lines.append(f"  {shap_style_dir}/FigS4_shap_heatmap_top20.tiff")
        report_lines.append(f"  {shap_style_dir}/FigS5_shap_importance_bar_35features.tiff")
        report_lines.append(f"  {shap_style_dir}/FigS6_shap_importance_bar_top20.tiff")
        report_lines.append(f"  {shap_style_dir}/FigS7_shap_waterfall.tiff")
        report_lines.append(f"  {shap_style_dir}/FigS8_dependence_*.tiff")
        report_lines.append(f"  {shap_style_dir}/FigS9_shap_importance_distribution.tiff")
        report_lines.append("")
        report_lines.append("SHAP stability analysis (across seeds, in more_shap_analysis):")
        report_lines.append(f"  {shap_dir}/shap_rank_stability.xlsx")
        report_lines.append(f"  {shap_dir}/shap_importance_matrix.xlsx")
        report_lines.append(f"  {shap_dir}/shap_rank_matrix.xlsx")
        report_lines.append(f"  {shap_dir}/shap_spearman_matrix.xlsx")
        report_lines.append(f"  {shap_dir}/shap_top_k_overlap.xlsx")
        report_lines.append(f"  {shap_dir}/Fig_rank_heatmap.tiff")
        report_lines.append(f"  {shap_dir}/Fig_avg_rank_top20.tiff")
        report_lines.append(f"  {shap_dir}/Fig_importance_boxplot.tiff")
        report_lines.append(f"  {shap_dir}/Fig_spearman_matrix.tiff")
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
        report_lines.append("  ✓ Output directory: dnn_model_35_result_more_seeds_shap")
        report_lines.append("  ✓ CV: 10 repeats of 10-fold cross-validation")
        report_lines.append("  ✓ All bar charts: consistent bar width and label size (11pt)")
        report_lines.append("  ✓ Williams plot: four versions (split/no-split, two thresholds)")
        report_lines.append("  ✓ Color map indicates |standardized residual|")
        report_lines.append("  ✓ VIF plot: two versions (fixed + adaptive)")
        report_lines.append("  ✓ Added MedAE, MaxAE, and Bias metrics")
        report_lines.append("  ✓ Added prediction accuracy within ±10, ±20, ±30 nm")
        report_lines.append("  ✓ Multi-seed performance evaluation (3 seeds: 42, 123, 2023)")
        report_lines.append("  ✓ SHAP stability analysis across seeds (ranking consistency)")
        report_lines.append("  ✓ All SHAP outputs stored in 'more_shap_analysis'")
        report_lines.append("=" * 80)

        with open(os.path.join(shap_dir, 'shap_analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print("\n" + "=" * 70)
        print("SHAP Analysis Completed!")
        print("=" * 70)

    except Exception as e:
        print(f"  ✗ SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("All plots have been successfully generated and saved!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    return model, shap_values, importance_df


if __name__ == '__main__':
    model, shap_values, importance_df = main()