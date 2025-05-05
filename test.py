import os
import numpy as np

def check_npy_shapes(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = np.load(file_path)
                print(f"{file_name}: shape = {data.shape}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

# 使用示例
# folder_path = "/data2/jian/LangSplat/data/3dovs/bed/language_features"  # ← 修改为你的文件夹路径
# check_npy_shapes(folder_path)


f = "/data2/jian/LangSplat/output/bed_1/train/ours_None/renders_npy"
check_npy_shapes(f)