import os
import numpy as np
import glob

# 设置目录路径
data_dir = '/data2/jian/LangSplat/data/lerf_ovs/figurines/language_features'

# 获取所有 .npy 文件路径
npy_files = glob.glob(os.path.join(data_dir, '*.npy'))

# 检查文件数量
print(f"Found {len(npy_files)} npy files.")

# 遍历所有文件并打印 shape
for file_path in npy_files:
    try:
        data = np.load(file_path)
        print(f"{os.path.basename(file_path)}: shape = {data.shape}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
