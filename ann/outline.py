import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from ann.utils import downsample
import torch

def get_edge_points(mask_path):
    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"Original image shape: {image.shape}")

    # 假设 downsample 处理后返回的是 torch.Tensor
    mask = downsample(image)
    print(f"Downsampled mask shape: {mask.shape}")

    # 如果是 torch.Tensor，需要转换为 numpy 数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()  # 将 tensor 转换为 numpy 数组

    print(f"Converted mask shape: {mask.shape}")

    # 二值化
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    print(f"Binary mask shape: {binary_mask.shape}")

    # 提取轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 提取所有轮廓上的点
    all_points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]  # contour.shape = (N, 1, 2)
            all_points.append([x, y])

    edge_coords = np.array(all_points)  # 形状为 (N, 2)
    return edge_coords


def restore_contour_image(edge_coords, image_shape):
    """
    从边缘点还原图像（轮廓）。
    
    :param edge_coords: 边缘点坐标数组，形状为 (N, 2)
    :param image_shape: 原图像的形状 (height, width)
    :return: 还原后的图像（黑色背景，白色轮廓）
    """
    # 创建一个与原图相同大小的全黑图像
    contour_image = np.zeros(image_shape, dtype=np.uint8)

    # 将边缘点绘制到图像上（白色轮廓）
    for point in edge_coords:
        x, y = point
        contour_image[y, x] = 255  # 在对应的 (x, y) 点上设为白色

    return contour_image

if __name__ == "__main__":

    # 路径到 mask 图片
    mask_path = "/data2/jian/LangSplat/data/3dovs/bed/segmentations/00/white sheet.png"

    if not os.path.exists(mask_path):
        print(f"File does not exist at: {mask_path}")
    else:
        print(f"File exists at: {mask_path}")

    edge_coords = get_edge_points(mask_path)


    # 打印或保存
    print("边缘点数量:", len(edge_coords))
    print("前5个边缘点坐标:", edge_coords[:10])
    print("后5个边缘点坐标:", edge_coords[10:])
    # 如需保存：
    # np.save("banana_edge_coords.npy", edge_coords)
    image_shape = (1080, 1440)  
    restored_image = restore_contour_image(edge_coords, image_shape)
    output_path = "/data2/jian/LangSplat/restored_contour.png"  # 请根据需要修改路径
    cv2.imwrite(output_path, restored_image)
    print(f"恢复的轮廓图像已保存到 {output_path}")