# -*- coding: utf-8 -*-
"""
多种线性模型的对比分析脚本 (Linear Regression, Ridge, Lasso, ElasticNet)
基于70个Mordred描述符版本
CJChE格式版本 - 所有图表输出为TIFF 300 dpi
所有图表已去标题，符合CJChE要求
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')


def setup_cjche_style():
    """设置符合CJChE要求的绘图样式"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.title_fontsize'] = 9
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


def evaluate_model(model, X_train, y_train, X_test, y_test, total_X, total_y):
    """训练模型并评估其性能"""
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    total_y_pred = model.predict(total_X)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_total = r2_score(total_y, total_y_pred)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mae_total = mean_absolute_error(total_y, total_y_pred)

    # LOO-CV Validation
    loo = LeaveOneOut()
    y_true_loo = []
    y_pred_loo = []

    for train_index, test_index in loo.split(total_X):
        X_train_loo, X_test_loo = total_X[train_index], total_X[test_index]
        y_train_loo, y_test_loo = total_y[train_index], total_y[test_index]
        model.fit(X_train_loo, y_train_loo)
        y_pred_loo.append(model.predict(X_test_loo)[0])
        y_true_loo.append(y_test_loo[0])

    y_true_loo = np.array(y_true_loo)
    y_pred_loo = np.array(y_pred_loo)
    y_mean = np.mean(y_true_loo)
    Q2 = 1 - np.sum((y_true_loo - y_pred_loo) ** 2) / np.sum((y_true_loo - y_mean) ** 2)
    mae_loo = mean_absolute_error(y_true_loo, y_pred_loo)

    return {
        'R2_Train': r2_train,
        'R2_Test': r2_test,
        'R2_Total': r2_total,
        'Q2_LOO': Q2,
        'MAE_Train': mae_train,
        'MAE_Test': mae_test,
        'MAE_Total': mae_total,
        'MAE_LOO': mae_loo
    }


def main():
    cd_excel = 'origin_data/'
    excel_name = 'data_w_Md_bo.xlsx'
    output_dir = 'lr_compare_result_70/'
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    df = pd.read_excel(os.path.join(cd_excel, excel_name))
    train_df = df[df['Training set/Testing set'] == 'Training set'].copy()
    test_df = df[df['Training set/Testing set'] == 'Testing set'].copy()
    total_df = pd.concat([train_df, test_df], ignore_index=True)

    # 选用70个Mordred描述符
    selected_bits = [
        355, 987, 993, 1076, 1329,
        1069, 1547, 354, 1159, 131,
        1491, 411, 1406, 1495, 1550,
        157, 194, 1376, 1064, 448,
        816, 1418, 451, 830, 1318,
        265, 258, 770, 512, 1313,
        1507, 1407, 1453, 1265, 635,
        1162, 238, 552, 920, 49,
        349, 244, 1263, 63, 545,
        563, 1066, 464, 1353, 787,
        1408, 358, 275, 334, 918,
        1260, 465, 360, 381, 1138,
        327, 1424, 1140, 1437, 452,
        779, 1246, 221, 147, 135
    ]

    X_train = train_df[[f'Md_{bit}' for bit in selected_bits]].values
    y_train = train_df['bo'].values
    X_test = test_df[[f'Md_{bit}' for bit in selected_bits]].values
    y_test = test_df['bo'].values
    total_X = total_df[[f'Md_{bit}' for bit in selected_bits]].values
    total_y = total_df['bo'].values

    # 定义要对比的模型
    models = {
        'MLR': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

    results = []

    print(f"=" * 70)
    print("Linear Models Comparison Analysis (70 Descriptors)")
    print("=" * 70)

    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, total_X, total_y)
        metrics['Model'] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    # 重新排列列顺序
    cols = ['Model', 'R2_Train', 'R2_Test', 'R2_Total', 'Q2_LOO', 'MAE_Train', 'MAE_Test', 'MAE_Total', 'MAE_LOO']
    results_df = results_df[cols]

    # 保存结果表格
    table_path = os.path.join(output_dir, 'Table1_models_comparison.xlsx')
    results_df.to_excel(table_path, index=False)
    print(f"\n✓ Comparison results saved to: {table_path}")

    # 打印结果用于调试
    print("\n" + "=" * 70)
    print("Results Summary:")
    print(results_df.round(4))
    print("=" * 70)

    # ================= 绘制对比图表 =================
    model_names = results_df['Model']
    x = np.arange(len(model_names))
    width = 0.2

    # 低饱和度配色方案 - 莫兰迪色系
    color_train = '#6B7B8D'  # 灰蓝
    color_test = '#D4A373'  # 灰杏/浅陶土
    color_total = '#8CA96F'  # 灰绿
    color_loo = '#C88B7A'  # 灰红/陶土红

    # 1. R2 和 Q2 比较图
    print("\nPlotting R²/Q² comparison chart...")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    ax.bar(x - 1.5 * width, results_df['R2_Train'], width,
           label='Training', color=color_train,
           edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x - 0.5 * width, results_df['R2_Test'], width,
           label='Testing', color=color_test,
           edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x + 0.5 * width, results_df['R2_Total'], width,
           label='Total', color=color_total,
           edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x + 1.5 * width, results_df['Q2_LOO'], width,
           label='LOO', color=color_loo,
           edgecolor='black', linewidth=0.8, alpha=0.85)

    ax.set_ylabel('$R^2$ / $Q^2$', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc='upper right', handlelength=1.5)

    min_val = min(results_df[['R2_Train', 'R2_Test', 'R2_Total', 'Q2_LOO']].min())
    max_val = max(results_df[['R2_Train', 'R2_Test', 'R2_Total', 'Q2_LOO']].max())
    y_min = max(0.7, min_val - 0.05)
    y_max = min(1.0, max_val + 0.05)
    ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(output_dir, 'Fig1_R2_Q2_comparison.tiff'))
    plt.close()

    # 2. MAE 比较图
    print("Plotting MAE comparison chart...")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    ax.bar(x - 1.5 * width, results_df['MAE_Train'], width,
           label='Training', color=color_train,
           edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x - 0.5 * width, results_df['MAE_Test'], width,
           label='Testing', color=color_test,
           edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x + 0.5 * width, results_df['MAE_Total'], width,
           label='Total', color=color_total,
           edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x + 1.5 * width, results_df['MAE_LOO'], width,
           label='LOO', color=color_loo,
           edgecolor='black', linewidth=0.8, alpha=0.85)

    ax.set_ylabel('MAE (nm)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc='upper left', handlelength=1.5)

    min_mae = min(results_df[['MAE_Train', 'MAE_Test', 'MAE_Total', 'MAE_LOO']].min())
    max_mae = max(results_df[['MAE_Train', 'MAE_Test', 'MAE_Total', 'MAE_LOO']].max())
    y_min = max(0, min_mae - 2)
    y_max = max_mae + 3
    ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    save_figure_tiff(fig, os.path.join(output_dir, 'Fig2_MAE_comparison.tiff'))
    plt.close()

    # ================= 生成分析报告 =================
    print("\nGenerating analysis report...")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("LINEAR MODELS COMPARISON ANALYSIS - COMPREHENSIVE REPORT (70 Descriptors)")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("MODELS COMPARED:")
    report_lines.append("-" * 50)
    for name in model_names:
        report_lines.append(f"  - {name}")
    report_lines.append("")

    report_lines.append("PERFORMANCE SUMMARY:")
    report_lines.append("-" * 50)

    for idx, row in results_df.iterrows():
        report_lines.append(f"\n{row['Model']}:")
        report_lines.append(f"  R² (Training): {row['R2_Train']:.4f}")
        report_lines.append(f"  R² (Testing):  {row['R2_Test']:.4f}")
        report_lines.append(f"  R² (Total):    {row['R2_Total']:.4f}")
        report_lines.append(f"  Q² (LOO-CV):   {row['Q2_LOO']:.4f}")
        report_lines.append(f"  MAE (Training): {row['MAE_Train']:.4f} nm")
        report_lines.append(f"  MAE (Testing):  {row['MAE_Test']:.4f} nm")
        report_lines.append(f"  MAE (Total):    {row['MAE_Total']:.4f} nm")
        report_lines.append(f"  MAE (LOO-CV):   {row['MAE_LOO']:.4f} nm")

    report_lines.append("")
    report_lines.append("BEST MODEL BY METRIC:")
    report_lines.append("-" * 50)

    for metric in ['R2_Total', 'Q2_LOO', 'MAE_Total', 'MAE_LOO']:
        if 'R2' in metric or 'Q2' in metric:
            best_idx = results_df[metric].idxmax()
            best_val = results_df.loc[best_idx, metric]
            report_lines.append(f"  Best {metric}: {results_df.loc[best_idx, 'Model']} ({best_val:.4f})")
        else:
            best_idx = results_df[metric].idxmin()
            best_val = results_df.loc[best_idx, metric]
            report_lines.append(f"  Best {metric}: {results_df.loc[best_idx, 'Model']} ({best_val:.4f} nm)")

    report_lines.append("")
    report_lines.append("FILES GENERATED (CJChE Format):")
    report_lines.append("-" * 50)
    report_lines.append(f"  {output_dir}/Table1_models_comparison.xlsx")
    report_lines.append(f"  {output_dir}/Fig1_R2_Q2_comparison.tiff")
    report_lines.append(f"  {output_dir}/Fig2_MAE_comparison.tiff")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("ANALYSIS COMPLETED SUCCESSFULLY!")
    report_lines.append("All figures meet CJChE requirements:")
    report_lines.append("  ✓ TIFF format")
    report_lines.append("  ✓ 300 dpi resolution")
    report_lines.append("  ✓ Arial font, 10 pt (labels), 9 pt (ticks/legend)")
    report_lines.append("  ✓ Inward ticks")
    report_lines.append("  ✓ Line width: 0.25-1.5 pt")
    report_lines.append("=" * 70)

    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Analysis report saved to: {output_dir}/analysis_report.txt")

    print(f"\n{'=' * 70}")
    print("Analysis Completed Successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()