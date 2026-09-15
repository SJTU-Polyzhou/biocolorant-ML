# -*- coding: utf-8 -*-
"""
多种机器学习模型对比分析脚本：以MLR为基准，对比树模型与DNN等
基于35个Mordred描述符版本
包含模型：MLR (基准), Random Forest, Gradient Boosting, SVM, DNN (MLP)
CJChE格式版本 - 所有图表输出为TIFF 300 dpi
所有图表已去标题，符合CJChE要求
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def setup_cjche_style():
    """设置符合CJChE要求的绘图样式"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.title_fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

def save_figure_tiff(fig, filepath, dpi=300):
    """统一保存图片为TIFF格式，300 dpi"""
    fig.savefig(filepath, format='tiff', dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

# 应用CJChE样式
setup_cjche_style()

def evaluate_model(model, X_train, y_train, X_test, y_test, total_X, total_y, scale=False):
    """
    训练模型并评估其性能
    参数 scale: 是否对特征进行标准化（对SVM和DNN必要）
    """
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        total_X_scaled = scaler.transform(total_X)
    else:
        total_X_scaled = total_X.copy()

    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    total_y_pred = model.predict(total_X_scaled)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_total = r2_score(total_y, total_y_pred)
    
    aae_train = mean_absolute_error(y_train, y_train_pred)
    aae_test = mean_absolute_error(y_test, y_test_pred)
    aae_total = mean_absolute_error(total_y, total_y_pred)
    
    # LOO-CV Validation
    loo = LeaveOneOut()
    y_true_loo = []
    y_pred_loo = []

    for train_index, test_index in loo.split(total_X):
        X_train_loo, X_test_loo = total_X[train_index], total_X[test_index]
        y_train_loo, y_test_loo = total_y[train_index], total_y[test_index]
        
        if scale:
            loo_scaler = StandardScaler()
            X_train_loo = loo_scaler.fit_transform(X_train_loo)
            X_test_loo = loo_scaler.transform(X_test_loo)

        model.fit(X_train_loo, y_train_loo)
        y_pred_loo.append(model.predict(X_test_loo)[0])
        y_true_loo.append(y_test_loo[0])

    y_true_loo = np.array(y_true_loo)
    y_pred_loo = np.array(y_pred_loo)
    y_mean = np.mean(y_true_loo)
    Q2 = 1 - np.sum((y_true_loo - y_pred_loo) ** 2) / np.sum((y_true_loo - y_mean) ** 2)
    aae_loo = mean_absolute_error(y_true_loo, y_pred_loo)
    
    return {
        'R2_Train': r2_train,
        'R2_Test': r2_test,
        'R2_Total': r2_total,
        'Q2_LOO': Q2,
        'AAE_Train': aae_train,
        'AAE_Test': aae_test,
        'AAE_Total': aae_total,
        'AAE_LOO': aae_loo
    }

def main():
    cd_excel = 'origin_data/'
    excel_name = 'data_w_Md_bo.xlsx'
    output_dir = 'mlr_vs_ml_compare_result/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_excel(os.path.join(cd_excel, excel_name))
    train_df = df[df['Training set/Testing set'] == 'Training set'].copy()
    test_df = df[df['Training set/Testing set'] == 'Testing set'].copy()
    total_df = pd.concat([train_df, test_df], ignore_index=True)

    # 选用35个Mordred描述符
    selected_bits = [
        906, 1336, 334, 1310, 1348, 1406, 1437, 345, 218, 230,
        449, 510, 299, 3, 1547, 566, 1418, 1069, 232, 264,
        1076, 1077, 322, 1313, 1294, 1064, 1329, 1146, 1477, 1359,
        805, 1394, 1358, 1573, 485
    ]

    X_train = train_df[[f'Md_{bit}' for bit in selected_bits]].values
    y_train = train_df['bo'].values
    X_test = test_df[[f'Md_{bit}' for bit in selected_bits]].values
    y_test = test_df['bo'].values
    total_X = total_df[[f'Md_{bit}' for bit in selected_bits]].values
    total_y = total_df['bo'].values

    print("\nTuning Random Forest hyperparameters using GridSearchCV...")
    rf_param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    }
    rf_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Best RF Parameters: {rf_search.best_params_}")
    print(f"Best RF CV R2: {rf_search.best_score_:.4f}\n")

    # 提取Dnn_35_PID_with_shap.py中的调参结果
    dnn_params = {
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
    
    # 包装目标值标准化
    dnn_model_scaled_y = TransformedTargetRegressor(
        regressor=MLPRegressor(**dnn_params),
        transformer=StandardScaler()
    )
    
    svr_model_scaled_y = TransformedTargetRegressor(
        regressor=SVR(C=10.0, epsilon=0.1),
        transformer=StandardScaler()
    )

    # 定义要对比的模型 (设定 random_state 保证可重现)
    # 注意：SVM和MLP需要数据标准化
    models = {
        'MLR': (LinearRegression(), False),
        'RandomForest': (best_rf, False),
        'GradientBoosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
        'SVR': (svr_model_scaled_y, True),
        'DNN (MLP)': (dnn_model_scaled_y, True)
    }

    results = []

    print(f"={'=' * 70}")
    print("Machine Learning Models Comparison (Baseline: MLR)")
    print("=" * 70)

    for name, (model, scale_needed) in models.items():
        print(f"Evaluating {name} (Scaling: {scale_needed})...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, total_X, total_y, scale=scale_needed)
        metrics['Model'] = name
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    # 重新排列列顺序
    cols = ['Model', 'R2_Train', 'R2_Test', 'R2_Total', 'Q2_LOO', 'AAE_Train', 'AAE_Test', 'AAE_Total', 'AAE_LOO']
    results_df = results_df[cols]
    
    # 保存结果表格
    table_path = os.path.join(output_dir, 'Table1_ML_models_comparison.xlsx')
    results_df.to_excel(table_path, index=False)
    print(f"\n✓ Comparison results saved to: {table_path}")

    # ================= 绘制对比图表 =================
    model_names = results_df['Model']
    x = np.arange(len(model_names))
    width = 0.15 # 共有4根柱子，需缩小柱宽
    
    # 1. R2 和 Q2 比较图
    print("\nPlotting R²/Q² comparison chart...")
    fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54)) # 适度加宽图表以容纳更多模型名称
    
    ax.bar(x - 1.5*width, results_df['R2_Train'], width, label='Train $R^2$', color='#1f77b4', edgecolor='black', alpha=0.8)
    ax.bar(x - 0.5*width, results_df['R2_Test'], width, label='Test $R^2$', color='#ff7f0e', edgecolor='black', alpha=0.8)
    ax.bar(x + 0.5*width, results_df['R2_Total'], width, label='Total $R^2$', color='#2ca02c', edgecolor='black', alpha=0.8)
    ax.bar(x + 1.5*width, results_df['Q2_LOO'], width, label='LOO $Q^2$', color='#d62728', edgecolor='black', alpha=0.8)

    ax.set_ylabel('$R^2$ / $Q^2$ value', fontsize=10)
    ax.set_xticks(x)
    # 倾斜字体以防遮挡
    ax.set_xticklabels(model_names, fontsize=9, rotation=15)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    
    # 设置y轴范围（根据数据动态调整）
    min_r2 = min(results_df[['R2_Train', 'R2_Test', 'R2_Total', 'Q2_LOO']].min())
    ax.set_ylim([max(0, min_r2 - 0.1), 1.0])

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(output_dir, 'Fig1_ML_R2_Q2_comparison.tiff'))
    plt.close()

    # 2. AAE 比较图
    print("Plotting AAE comparison chart...")
    fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54))
    
    ax.bar(x - 1.5*width, results_df['AAE_Train'], width, label='Train AAE', color='#1f77b4', edgecolor='black', alpha=0.8)
    ax.bar(x - 0.5*width, results_df['AAE_Test'], width, label='Test AAE', color='#ff7f0e', edgecolor='black', alpha=0.8)
    ax.bar(x + 0.5*width, results_df['AAE_Total'], width, label='Total AAE', color='#2ca02c', edgecolor='black', alpha=0.8)
    ax.bar(x + 1.5*width, results_df['AAE_LOO'], width, label='LOO AAE', color='#d62728', edgecolor='black', alpha=0.8)

    ax.set_ylabel('AAE (nm)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9, rotation=15)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    # 将Legend放置在右上角
    ax.legend(fontsize=8, frameon=False, loc='upper right')

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(output_dir, 'Fig2_ML_AAE_comparison.tiff'))
    plt.close()

    print(f"\n{'=' * 70}")
    print("Analysis Completed Successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()