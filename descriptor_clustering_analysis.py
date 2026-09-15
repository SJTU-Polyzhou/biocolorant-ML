# -*- coding: utf-8 -*-
"""
Descriptor Space Clustering Analysis - 使用预计算的候选分子描述符
直接读取 calculate_candidate_descriptors.py 生成的描述符文件
图中不显示数字标签，通过表格识别
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

# ================= Configuration =================
# 文件路径
TRAIN_DATA_PATH = 'data_w_Md_bo.xlsx'                    # 训练数据
CANDIDATE_DESCRIPTORS_PATH = 'candidate_molecules_35descriptors.xlsx'  # 候选分子35个描述符

# 35个Mordred描述符索引
SELECTED_BITS = [
    906, 1336, 334, 1310, 1348, 1406, 1437, 345, 218, 230,
    449, 510, 299, 3, 1547, 566, 1418, 1069, 232, 264,
    1076, 1077, 322, 1313, 1294, 1064, 1329, 1146, 1477, 1359,
    805, 1394, 1358, 1573, 485
]

# 聚类参数
N_CLUSTERS = 5
CLUSTER_THRESHOLD_PERCENTILE = 95

# 输出目录
OUTPUT_DIR = 'descriptor_clustering_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/tables', exist_ok=True)

# 颜色配置
ALL_DATA_COLOR = '#4C72B0'
ALL_DATA_ALPHA = 0.5
WITHIN_AD_COLOR = '#1F4E79'      # 深蓝色 - Within AD
OUTSIDE_AD_COLOR = '#D95F02'      # 橙色 - Outside AD
CLUSTER_CMAP = 'viridis'
BACKGROUND_COLOR = 'white'
BLACK = '#000000'


def setup_cjche_style():
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


def save_figure_tiff(fig, filepath):
    fig.savefig(filepath, format='tiff', dpi=300, bbox_inches='tight',
                facecolor=BACKGROUND_COLOR, edgecolor='none')
    print(f"  ✓ Saved: {filepath}")


def set_axes_border(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(BLACK)
    ax.tick_params(axis='both', colors=BLACK)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)


setup_cjche_style()


def main():
    print("=" * 80)
    print("描述符空间聚类分析 (使用预计算描述符)")
    print("=" * 80)

    # ========== Step 1: 加载训练数据 ==========
    print("\n[Step 1] 加载训练数据")
    print("-" * 60)

    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"错误: 找不到训练数据文件 {TRAIN_DATA_PATH}")
        return

    df_train = pd.read_excel(TRAIN_DATA_PATH, engine='openpyxl')
    print(f"✓ 加载训练数据: {len(df_train)} 个分子")

    # 提取训练集描述符
    descriptor_cols = [f'Md_{bit}' for bit in SELECTED_BITS]
    X_train_raw = df_train[descriptor_cols].values
    print(f"✓ 训练集描述符矩阵: {X_train_raw.shape}")

    # ========== Step 2: 加载候选分子描述符 ==========
    print("\n[Step 2] 加载候选分子描述符")
    print("-" * 60)

    if not os.path.exists(CANDIDATE_DESCRIPTORS_PATH):
        print(f"错误: 找不到候选分子描述符文件 {CANDIDATE_DESCRIPTORS_PATH}")
        print("请先运行 calculate_candidate_descriptors.py 生成此文件")
        return

    df_cand = pd.read_excel(CANDIDATE_DESCRIPTORS_PATH, engine='openpyxl')
    print(f"✓ 加载候选分子描述符: {len(df_cand)} 个分子")

    # 获取候选分子编号
    if 'Name' in df_cand.columns:
        candidate_ids = df_cand['Name'].tolist()
    else:
        candidate_ids = [f'Candidate_{i+1}' for i in range(len(df_cand))]

    # 提取候选分子描述符
    X_candidates_raw = df_cand[descriptor_cols].values
    print(f"✓ 候选分子描述符矩阵: {X_candidates_raw.shape}")

    # 检查描述符值是否正常
    print(f"\n  候选分子描述符统计:")
    print(f"    最小值: {X_candidates_raw.min():.6f}")
    print(f"    最大值: {X_candidates_raw.max():.6f}")
    print(f"    均值: {X_candidates_raw.mean():.6f}")
    print(f"    标准差: {X_candidates_raw.std():.6f}")

    # 检查是否所有候选分子描述符都相同
    if len(X_candidates_raw) > 1:
        first_row = X_candidates_raw[0]
        all_same = np.all([np.allclose(row, first_row) for row in X_candidates_raw])
        if all_same:
            print("  ⚠️ 警告: 所有候选分子描述符完全相同！请检查 calculate_candidate_descriptors.py 的计算结果")
        else:
            print("  ✓ 候选分子描述符各不相同，计算正常")

    # ========== Step 3: 标准化 ==========
    print("\n[Step 3] 数据标准化")
    print("-" * 60)

    # 合并训练集和候选分子一起标准化
    X_all_raw = np.vstack([X_train_raw, X_candidates_raw])
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all_raw)

    X_train = X_all[:len(X_train_raw)]
    X_candidates = X_all[len(X_train_raw):]

    print(f"✓ 标准化完成")
    print(f"  训练集: {X_train.shape}")
    print(f"  候选分子: {X_candidates.shape}")

    # ========== Step 4: KMeans聚类 ==========
    print("\n[Step 4] KMeans聚类分析")
    print("-" * 60)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train)
    cluster_centers = kmeans.cluster_centers_

    print(f"✓ 聚类完成: {N_CLUSTERS} 个簇")

    # 计算每个簇的距离阈值
    cluster_distances = {c: [] for c in range(N_CLUSTERS)}
    for i, point in enumerate(X_train):
        c = cluster_labels[i]
        dist = np.linalg.norm(point - cluster_centers[c])
        cluster_distances[c].append(dist)

    cluster_thresholds = {}
    for c in range(N_CLUSTERS):
        if len(cluster_distances[c]) > 0:
            cluster_thresholds[c] = np.percentile(cluster_distances[c], CLUSTER_THRESHOLD_PERCENTILE)
        else:
            cluster_thresholds[c] = np.inf
        print(f"  簇 {c}: {len(cluster_distances[c])} 个样本, 阈值={cluster_thresholds[c]:.4f}")

    # ========== Step 5: 评估候选分子 ==========
    print("\n[Step 5] 评估候选分子AD (Applicability Domain)")
    print("-" * 60)

    candidate_results = []
    for i, cand in enumerate(X_candidates):
        dists = [np.linalg.norm(cand - center) for center in cluster_centers]
        nearest_cluster = np.argmin(dists)
        min_dist = dists[nearest_cluster]

        if min_dist <= cluster_thresholds[nearest_cluster]:
            ad_status = "Within AD"
            color = WITHIN_AD_COLOR
            status_label = "Within AD"
        else:
            ad_status = "Outside AD"
            color = OUTSIDE_AD_COLOR
            status_label = "Outside AD"

        candidate_results.append({
            'figure_id': i + 1,
            'npbs_id': candidate_ids[i],
            'nearest_cluster': nearest_cluster,
            'distance': min_dist,
            'threshold': cluster_thresholds[nearest_cluster],
            'ad_status': ad_status,
            'status_label': status_label,
            'color': color
        })

    # 打印结果
    print("\n评估结果:")
    print("-" * 85)
    print(f"{'ID':<4} {'NPBS ID':<18} {'AD Status':<15} {'Distance':<12} {'Threshold':<12} {'Cluster':<8}")
    print("-" * 85)
    for r in candidate_results:
        print(f"  {r['figure_id']:<4} {r['npbs_id']:<18} {r['status_label']:<15} {r['distance']:.4f}     {r['threshold']:.4f}     {r['nearest_cluster']:<8}")

    # ========== Step 6: PCA可视化 ==========
    print("\n[Step 6] PCA降维可视化")
    print("-" * 60)

    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_cand_pca = pca.transform(X_candidates)

    var_ratio = pca.explained_variance_ratio_
    print(f"  PCA方差解释率: PC1={var_ratio[0]:.3f}, PC2={var_ratio[1]:.3f}")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor=WITHIN_AD_COLOR, edgecolor=BLACK, label='Within AD'),
        Patch(facecolor=OUTSIDE_AD_COLOR, edgecolor=BLACK, label='Outside AD'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=BLACK, markersize=8, label='Candidate')
    ]

    # 图1: PCA - 训练集聚类分布（不显示数字标签）
    fig, ax = plt.subplots(figsize=(12/2.54, 10/2.54))
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP,
               alpha=ALL_DATA_ALPHA, s=25, edgecolors=BLACK, linewidth=0.5)

    for r in candidate_results:
        idx = r['figure_id'] - 1
        x, y = X_cand_pca[idx]
        ax.scatter(x, y, c=r['color'], s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
        # 不添加数字标签

    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{OUTPUT_DIR}/figures/Fig1_PCA_clusters.tiff')
    plt.close()

    # 图2: PCA - 统一颜色（不显示数字标签）
    fig, ax = plt.subplots(figsize=(12/2.54, 10/2.54))
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=ALL_DATA_COLOR, alpha=ALL_DATA_ALPHA,
               s=20, edgecolors=BLACK, linewidth=0.3)

    for r in candidate_results:
        idx = r['figure_id'] - 1
        x, y = X_cand_pca[idx]
        ax.scatter(x, y, c=r['color'], s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
        # 不添加数字标签

    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{OUTPUT_DIR}/figures/Fig2_PCA_uniform.tiff')
    plt.close()

    # 图3: PCA - 高亮候选分子（不显示数字标签）
    fig, ax = plt.subplots(figsize=(12/2.54, 10/2.54))
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='#E8E8E8', alpha=0.6, s=15, edgecolors=BLACK, linewidth=0.2)

    for r in candidate_results:
        idx = r['figure_id'] - 1
        x, y = X_cand_pca[idx]
        ax.scatter(x, y, c=r['color'], s=90, marker='*', edgecolors=BLACK, linewidth=1.0, zorder=5)
        # 不添加数字标签

    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{OUTPUT_DIR}/figures/Fig3_PCA_highlight.tiff')
    plt.close()

    # ========== Step 7: t-SNE可视化 ==========
    print("\n[Step 7] t-SNE降维可视化")
    print("-" * 60)

    X_combined = np.vstack([X_train, X_candidates])
    perplexity = min(30, len(X_combined) - 1)
    print(f"  计算t-SNE (perplexity={perplexity})...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_combined_tsne = tsne.fit_transform(X_combined)
    X_train_tsne = X_combined_tsne[:len(X_train)]
    X_cand_tsne = X_combined_tsne[len(X_train):]

    # 图4: t-SNE - 聚类颜色（不显示数字标签）
    fig, ax = plt.subplots(figsize=(12/2.54, 10/2.54))
    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=cluster_labels, cmap=CLUSTER_CMAP,
               alpha=ALL_DATA_ALPHA, s=25, edgecolors=BLACK, linewidth=0.5)

    for r in candidate_results:
        idx = r['figure_id'] - 1
        x, y = X_cand_tsne[idx]
        ax.scatter(x, y, c=r['color'], s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
        # 不添加数字标签

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{OUTPUT_DIR}/figures/Fig4_tSNE_clusters.tiff')
    plt.close()

    # 图5: t-SNE - 描述符空间（不显示数字标签）
    fig, ax = plt.subplots(figsize=(12/2.54, 10/2.54))
    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=ALL_DATA_COLOR, alpha=0.3,
               s=20, edgecolors=BLACK, linewidth=0.2)

    for r in candidate_results:
        idx = r['figure_id'] - 1
        x, y = X_cand_tsne[idx]
        ax.scatter(x, y, c=r['color'], s=80, marker='*', edgecolors=BLACK, linewidth=0.8, zorder=5)
        # 不添加数字标签

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    set_axes_border(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{OUTPUT_DIR}/figures/Fig5_tSNE_space.tiff')
    plt.close()

    # 图6: AD分布柱状图
    fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))
    within_count = sum(r['ad_status'] == 'Within AD' for r in candidate_results)
    outside_count = sum(r['ad_status'] == 'Outside AD' for r in candidate_results)

    bars = ax.bar(['Within AD', 'Outside AD'], [within_count, outside_count],
                  color=[WITHIN_AD_COLOR, OUTSIDE_AD_COLOR], alpha=0.7,
                  edgecolor=BLACK, linewidth=0.8)

    for bar, count in zip(bars, [within_count, outside_count]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count),
                ha='center', va='bottom', fontsize=9, color=BLACK)

    ax.set_ylabel('Number of molecules')
    ax.set_xlabel('AD Status')
    ax.tick_params(axis='both', colors=BLACK)
    set_axes_border(ax)
    plt.tight_layout()
    save_figure_tiff(fig, f'{OUTPUT_DIR}/figures/Fig6_AD_distribution.tiff')
    plt.close()

    # ========== Step 8: 保存结果表格 ==========
    print("\n[Step 8] 保存结果表格")
    print("-" * 60)

    mapping_df = pd.DataFrame({
        'Figure_ID': [r['figure_id'] for r in candidate_results],
        'NPBS_ID': [r['npbs_id'] for r in candidate_results],
        'AD_Status': [r['status_label'] for r in candidate_results],
        'Nearest_Cluster': [r['nearest_cluster'] for r in candidate_results],
        'Distance_to_Center': [r['distance'] for r in candidate_results],
        'Threshold': [r['threshold'] for r in candidate_results]
    })
    mapping_df.to_csv(f'{OUTPUT_DIR}/tables/figure_id_to_npbs_mapping.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 保存: {OUTPUT_DIR}/tables/figure_id_to_npbs_mapping.csv")

    # 簇统计
    cluster_stats = []
    for c in range(N_CLUSTERS):
        cluster_stats.append({
            'Cluster': c,
            'Number_of_Samples': len(cluster_distances[c]),
            'Mean_Distance': np.mean(cluster_distances[c]) if cluster_distances[c] else 0,
            'Std_Distance': np.std(cluster_distances[c]) if cluster_distances[c] else 0,
            'Threshold_95%': cluster_thresholds[c]
        })
    cluster_df = pd.DataFrame(cluster_stats)
    cluster_df.to_csv(f'{OUTPUT_DIR}/tables/cluster_statistics.csv', index=False)
    print(f"✓ 保存: {OUTPUT_DIR}/tables/cluster_statistics.csv")

    # ========== 最终报告 ==========
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print(f"\n📊 候选分子AD (Applicability Domain) 统计:")
    print(f"  ✅ Within AD: {within_count} 个分子")
    print(f"  ⚠️ Outside AD: {outside_count} 个分子")

    print("\n📋 Figure ID 到 NPBS ID 映射:")
    for r in candidate_results:
        print(f"    {r['figure_id']} → {r['npbs_id']} ({r['status_label']})")

    print(f"\n📁 输出目录: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == '__main__':
    main()