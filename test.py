import numpy as np
from PIL import Image

# 读取图像
mask_path = "/data2/jian/LangSplat/data/3dovs/bed/segmentations/00/banana.png"
mask = Image.open(mask_path).convert("L")  # 转为灰度图

# 转为 numpy 数组
mask_np = np.array(mask)

# 打印基本信息
print("Shape:", mask_np.shape)
print("Data type:", mask_np.dtype)
print("Min value:", np.min(mask_np))
print("Max value:", np.max(mask_np))
print("Unique values:", np.unique(mask_np))
