import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取处理后的数据
df = pd.read_excel('bo_raw_04.xlsx')

# 获取唯一PID列表
unique_pids = df['PID'].unique().tolist()

# 设置随机种子保证可复现性
np.random.seed(42)

# 按4:1比例划分训练测试集
train_pids, test_pids = train_test_split(
    unique_pids,
    test_size=0.2,  # 20%测试集
    random_state=42
)

# 创建分组标签列
df.insert(0, 'Training set/Testing set',
          df['PID'].apply(lambda x: 'Training set' if x in train_pids else 'Testing set'))

# 保存结果
df.to_excel('data_w_Md_bo.xlsx', index=False)

# 统计数量
train_count = len(train_pids)
test_count = len(test_pids)

print(f"数据集划分完成！\n"
      f"训练集样本数: {train_count} ({train_count/len(unique_pids):.1%})\n"
      f"测试集样本数: {test_count} ({test_count/len(unique_pids):.1%})")
