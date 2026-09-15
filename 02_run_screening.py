#!/usr/bin/env python3
"""
Step 2: Run high-throughput screening
ONLY FILTER 200-700nm WAVELENGTH RANGE
(No chromophore, metal, biosafety, SAscore, or nearest neighbor similarity)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import time
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
BASE_DIR = Path(__file__).parent

MODEL_DIR = BASE_DIR.parent / '03_modeling' / '03_modeling' / 'DNN_model' / 'dnn_model_35_result' / 'dnn_model_result'

OUTPUT_DIR = BASE_DIR / 'new_output'
DESCRIPTOR_DIR = OUTPUT_DIR / 'descriptors'
RESULT_DIR = OUTPUT_DIR / 'results'
STEPWISE_DIR = OUTPUT_DIR / 'stepwise_results'
LOG_DIR = OUTPUT_DIR / 'logs'

for dir_path in [STEPWISE_DIR, RESULT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 35个特征的严格顺序
SELECTED_BITS = [
    906, 1336, 334, 1310, 1348,
    1406, 1437, 345, 218, 230,
    449, 510, 299, 3, 1547,
    566, 1418, 1069, 232, 264,
    1076, 1077, 322, 1313, 1294,
    1064, 1329, 1146, 1477, 1359,
    805, 1394, 1358, 1573, 485
]
REQUIRED_FEATURE_COLS = [f'Md_{bit}' for bit in SELECTED_BITS]

# Model paths
MODEL_PATH = MODEL_DIR / 'final_dnn_model.pkl'
if not MODEL_PATH.exists():
    if (MODEL_DIR / 'final_model.pkl').exists():
        MODEL_PATH = MODEL_DIR / 'final_model.pkl'
    elif (MODEL_DIR / 'final_model_seed2.pkl').exists():
        MODEL_PATH = MODEL_DIR / 'final_model_seed2.pkl'

SCALER_X_PATH = MODEL_DIR / 'scaler_X.pkl'
SCALER_Y_PATH = MODEL_DIR / 'scaler_y.pkl'

# File names
DESCRIPTOR_FILE = 'NPBS_Data_with_descriptors.csv'
PROCESSED_FILE = 'NPBS_Data_processed.csv'
SCREENING_RESULTS_FILE = 'screening_results_full.csv'
FILTERED_RESULTS_FILE = 'screening_results_200_700nm.csv'

# 波长范围
WAVELENGTH_MIN = 200
WAVELENGTH_MAX = 700


# ==================== 日志函数 ====================

class Logger:
    """同时输出到控制台和日志文件"""

    def __init__(self, log_dir, script_name="step02"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{script_name}_{timestamp}.log"

        # 写入日志头
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Step 2 运行日志\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message, print_to_console=True):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")

        # 输出到控制台
        if print_to_console:
            print(message)

    def log_section(self, title, print_to_console=True):
        """记录章节分隔"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        separator = "=" * 60
        log_line = f"\n[{timestamp}] {separator}\n[{timestamp}] {title}\n[{timestamp}] {separator}"

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")

        if print_to_console:
            print(f"\n{separator}")
            print(title)
            print(separator)

    def log_dataframe(self, df, name, max_rows=10):
        """记录DataFrame信息"""
        self.log(f"{name}: {len(df):,} 行, {len(df.columns)} 列")
        if len(df) > 0:
            self.log(f"  列名: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
            self.log(f"  前{min(max_rows, len(df))}行预览:")
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(df.head(min(max_rows, len(df))).to_string() + "\n\n")

    def close(self):
        """关闭日志"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        self.log(f"日志已保存: {self.log_file}")


# ==================== 模型函数 ====================

def load_model():
    """Load model and scalers"""
    print(f"  加载模型: {MODEL_PATH.name}")
    model = joblib.load(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return model, scaler_X, scaler_y


def prepare_features(df):
    """Extract feature matrix with STRICT column order"""
    missing_cols = [col for col in REQUIRED_FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing {len(missing_cols)} descriptor columns.")
    X = df[REQUIRED_FEATURE_COLS].values
    return X


def predict(model, scaler_X, scaler_y, X):
    """Make predictions"""
    X_scaled = scaler_X.transform(X)
    y_scaled = model.predict(X_scaled)
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def calculate_applicability_domain(X, scaler_X):
    """计算适用域状态"""
    try:
        X_scaled = scaler_X.transform(X)
        distances = np.linalg.norm(X_scaled, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        is_within = distances < (mean_dist + 3 * std_dist)
        return ['Within' if w else 'Outside' for w in is_within]
    except:
        return ['Within'] * len(X)


# ==================== Main函数 ====================

def main():
    # 初始化日志
    logger = Logger(LOG_DIR, "step02")

    logger.log_section("Step 2: Wavelength Screening (200-700nm ONLY)")
    logger.log("(Simplified: predictions + wavelength filter only)")
    logger.log(f"输出目录: {OUTPUT_DIR}")
    logger.log(f"日志目录: {LOG_DIR}")

    start_time_total = time.time()

    # ========== 1. 检查文件 ==========
    logger.log_section("1. 检查输入文件")

    desc_path = DESCRIPTOR_DIR / DESCRIPTOR_FILE
    if not desc_path.exists():
        error_msg = f"Error: Descriptor file not found: {desc_path}"
        logger.log(error_msg)
        logger.log("Please run 01_calculate_descriptors.py first")
        logger.close()
        print(error_msg)
        return

    processed_path = DESCRIPTOR_DIR / PROCESSED_FILE
    if not processed_path.exists():
        logger.log(f"Warning: Processed data not found at {processed_path}")
        logger.log("Using descriptor file only")
        df_desc = pd.read_csv(desc_path)
        df = df_desc.copy()
        df['Canonical_SMILES'] = df['SMILES']
        logger.log(f"  从描述符文件加载: {len(df):,} 个分子")
    else:
        logger.log(f"加载预处理数据: {processed_path}")
        df = pd.read_csv(processed_path)
        logger.log(f"  预处理数据: {len(df):,} 个分子")

        logger.log(f"加载描述符数据: {desc_path}")
        df_desc = pd.read_csv(desc_path)
        logger.log(f"  描述符数据: {len(df_desc):,} 个分子")

        df = pd.merge(df, df_desc, on='Identifier', how='inner')
        logger.log(f"  合并后: {len(df):,} 个分子")

    if not MODEL_PATH.exists():
        error_msg = f"Error: Model not found: {MODEL_PATH}"
        logger.log(error_msg)
        logger.close()
        print(error_msg)
        return

    logger.log(f"模型路径: {MODEL_PATH}")
    logger.log(f"Scaler_X路径: {SCALER_X_PATH}")
    logger.log(f"Scaler_Y路径: {SCALER_Y_PATH}")

    # ========== 2. 准备特征 ==========
    logger.log_section("2. 准备特征")

    X = prepare_features(df)
    logger.log(f"特征矩阵形状: {X.shape}")
    logger.log(f"特征数量: {X.shape[1]}")

    # ========== 3. 加载模型 ==========
    logger.log_section("3. 加载模型")

    model, scaler_X, scaler_y = load_model()
    logger.log(f"模型加载成功: {MODEL_PATH.name}")

    # ========== 4. 预测 ==========
    logger.log_section("4. 进行预测")

    predict_start = time.time()
    y_pred = predict(model, scaler_X, scaler_y, X)
    predict_time = time.time() - predict_start

    logger.log(f"预测完成，耗时: {predict_time:.1f}s")
    logger.log(f"预测值范围: {y_pred.min():.2f} - {y_pred.max():.2f} nm")
    logger.log(f"预测值平均值: {np.mean(y_pred):.2f} nm")
    logger.log(f"预测值中位数: {np.median(y_pred):.2f} nm")

    # ========== 5. 计算适用域 ==========
    logger.log_section("5. 计算适用域")

    applicability = calculate_applicability_domain(X, scaler_X)
    within_count = sum(1 for x in applicability if x == 'Within')
    logger.log(f"适用域内: {within_count:,} ({within_count / len(applicability) * 100:.1f}%)")
    logger.log(f"适用域外: {len(applicability) - within_count:,}")

    # ========== 6. 创建结果 ==========
    logger.log_section("6. 创建结果DataFrame")

    result_df = pd.DataFrame({
        'Identifier': df['Identifier'],
        'SMILES': df['Canonical_SMILES'],
        'Predicted_bo': y_pred,
        'Applicability_Domain': applicability
    })

    result_df = result_df.sort_values('Predicted_bo', ascending=False).reset_index(drop=True)
    result_df.insert(0, 'Rank', range(1, len(result_df) + 1))

    logger.log(f"结果DataFrame: {len(result_df):,} 行")

    # ========== 7. 筛选200-700nm ==========
    logger.log_section(f"7. 筛选 {WAVELENGTH_MIN}-{WAVELENGTH_MAX} nm 范围")

    filtered_df = result_df[
        (result_df['Predicted_bo'] >= WAVELENGTH_MIN) &
        (result_df['Predicted_bo'] <= WAVELENGTH_MAX)
        ].copy()

    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df.insert(0, 'Rank_200_700', range(1, len(filtered_df) + 1))

    logger.log(f"总分子数: {len(result_df):,}")
    logger.log(f"200-700nm范围内: {len(filtered_df):,} ({len(filtered_df) / len(result_df) * 100:.2f}%)")

    # ========== 8. 波长分布统计 ==========
    logger.log_section("8. 波长分布统计 (200-700nm范围内)")

    dist_200_300 = (filtered_df['Predicted_bo'] < 300).sum()
    dist_300_400 = ((filtered_df['Predicted_bo'] >= 300) & (filtered_df['Predicted_bo'] < 400)).sum()
    dist_400_500 = ((filtered_df['Predicted_bo'] >= 400) & (filtered_df['Predicted_bo'] < 500)).sum()
    dist_500_600 = ((filtered_df['Predicted_bo'] >= 500) & (filtered_df['Predicted_bo'] < 600)).sum()
    dist_600_700 = ((filtered_df['Predicted_bo'] >= 600) & (filtered_df['Predicted_bo'] <= 700)).sum()

    logger.log(f"  200-300nm: {dist_200_300:,}")
    logger.log(f"  300-400nm: {dist_300_400:,}")
    logger.log(f"  400-500nm: {dist_400_500:,}")
    logger.log(f"  500-600nm: {dist_500_600:,}")
    logger.log(f"  600-700nm: {dist_600_700:,}")

    # ========== 9. 保存结果 ==========
    logger.log_section("9. 保存结果")

    # 1. 完整结果（所有分子）
    full_path = RESULT_DIR / SCREENING_RESULTS_FILE
    result_df.to_csv(full_path, index=False)
    logger.log(f"完整结果（所有分子）: {full_path}")
    logger.log(f"  文件大小: {full_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 2. 200-700nm筛选结果
    filtered_path = RESULT_DIR / FILTERED_RESULTS_FILE
    filtered_df.to_csv(filtered_path, index=False)
    logger.log(f"200-700nm筛选结果: {filtered_path}")
    logger.log(f"  文件大小: {filtered_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 3. Top 100 候选（200-700nm范围内）
    top_100 = filtered_df.head(100)
    top_100_path = RESULT_DIR / 'top_100_candidates_200_700nm.csv'
    top_100.to_csv(top_100_path, index=False)
    logger.log(f"Top 100 候选 (200-700nm): {top_100_path}")

    # 4. 按波长区间分别保存
    bands = [
        (200, 300, '200_300nm'),
        (300, 400, '300_400nm'),
        (400, 500, '400_500nm'),
        (500, 600, '500_600nm'),
        (600, 700, '600_700nm'),
    ]

    for low, high, name in bands:
        band_df = filtered_df[
            (filtered_df['Predicted_bo'] >= low) &
            (filtered_df['Predicted_bo'] < high)
            ].copy()
        if len(band_df) > 0:
            band_path = RESULT_DIR / f'candidates_{name}.csv'
            band_df.to_csv(band_path, index=False)
            logger.log(f"  {name}: {len(band_df):,} 个分子 -> {band_path}")

    # 5. 保存到stepwise_results
    stepwise_path = STEPWISE_DIR / 'step02_wavelength_200_700nm_result.csv'
    filtered_df.to_csv(stepwise_path, index=False)
    logger.log(f"分步结果保存: {stepwise_path}")

    # ========== 10. 统计汇总 ==========
    logger.log_section("10. 运行统计汇总")

    total_time = time.time() - start_time_total
    logger.log(f"总耗时: {total_time:.1f}s")
    logger.log(f"总分子数: {len(result_df):,}")
    logger.log(f"200-700nm: {len(filtered_df):,} ({len(filtered_df) / len(result_df) * 100:.2f}%)")
    logger.log(f"预测范围: {y_pred.min():.2f} - {y_pred.max():.2f} nm")
    logger.log(f"平均预测: {np.mean(y_pred):.2f} nm")
    logger.log(f"中位数预测: {np.median(y_pred):.2f} nm")
    logger.log(f"适用域内: {within_count:,} ({within_count / len(applicability) * 100:.1f}%)")

    logger.log(f"\n输出目录: {RESULT_DIR}")
    logger.log(f"分步结果: {STEPWISE_DIR}")
    logger.log(f"日志目录: {LOG_DIR}")

    # 关闭日志
    logger.close()

    # 控制台输出总结
    print("\n" + "=" * 70)
    print(f"✅ 筛选完成！总耗时: {total_time:.1f}s")
    print("=" * 70)
    print(f"\n筛选统计:")
    print(f"  总分子数: {len(result_df):,}")
    print(f"  200-700nm: {len(filtered_df):,} ({len(filtered_df) / len(result_df) * 100:.2f}%)")
    print(f"  预测范围: {y_pred.min():.2f} - {y_pred.max():.2f} nm")
    print(f"  平均预测: {np.mean(y_pred):.2f} nm")
    print(f"  中位数预测: {np.median(y_pred):.2f} nm")
    print(f"\n日志文件: {LOG_DIR}")


if __name__ == "__main__":
    main()