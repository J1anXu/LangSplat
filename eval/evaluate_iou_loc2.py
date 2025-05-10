#!/usr/bin/env python
from __future__ import annotations

import json
import os

import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from PIL import Image
from skimage.io import imsave
import sys
# 为了方便debug
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result

def threshold_mask(mask, threshold=127):
    """
    对降采样后的mask进行阈值化处理
    将大于threshold的像素值设置为255，小于threshold的设置为0
    """
    # 确保 mask 是 8 位单通道图像（0-255的灰度图）
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()  # 转换为 numpy 数组

    # 使用 OpenCV 对图像进行阈值化处理
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    return binary_mask


def downsample(image):

    # 如果 image 是 PIL.Image 对象，将其转换为 NumPy 数组
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 如果 image 是 PyTorch 张量，则将其转换为 NumPy 数组
    elif isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # 如果是 GPU 上的 tensor，先移动到 CPU，再转换为 numpy 数组

    # 检查图像是否是 NumPy 数组
    if isinstance(image, np.ndarray):
        # 判断图像维度：如果是 RGB 图像（3 通道），则 shape 是 (height, width, 3)
        if len(image.shape) == 3:
            orig_h, orig_w = image.shape[0], image.shape[1]
        elif len(image.shape) == 2:
            orig_h, orig_w = image.shape[0], image.shape[1]
        else:
            raise ValueError("Unsupported image format")
    else:
        raise ValueError(f"Expected image to be a numpy.ndarray or torch.Tensor, but got {type(image)}")

    orig_w, orig_h = image.shape[1], image.shape[0]
    if orig_h > 1080:
        print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
            "If this is not desired, please explicitly specify '--resolution/-r' as 1")

        global_down = orig_h / 1080
    else:
        global_down = 1

        
    scale = float(global_down)
    resolution = (int( orig_w  / scale), int(orig_h / scale))
    
    image = cv2.resize(image, resolution)
    image = torch.from_numpy(image)
    return image

def downsample_rgb(image):
    # 记录输入图像的设备
    device = image.device if isinstance(image, torch.Tensor) else None
    
    # 如果是 torch.Tensor，则转换为 numpy.ndarray
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # 转换到 CPU 并转换为 NumPy 数组
        
        # 如果是 CHW 格式 (通道优先)，则将其转换为 HWC 格式 (高度，宽度，通道)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # 转换为 HWC

    # 确保 image 是一个 numpy.ndarray
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected a numpy.ndarray, but got {type(image)}")

    # 获取原始图像的宽高
    orig_h, orig_w = image.shape[0], image.shape[1]
    
    # 判断是否需要缩放至1080P
    if orig_h > 1080:
        print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n"
              "If this is not desired, please explicitly specify '--resolution/-r' as 1")
        global_down = orig_h / 1080
    else:
        global_down = 1

    # 缩放比例
    scale = float(global_down)
    resolution = (int(orig_w / scale), int(orig_h / scale))
    
    # 使用 INTER_CUBIC 插值方法进行降采样
    image_resized = cv2.resize(image, resolution, interpolation=cv2.INTER_CUBIC)
    
    # 将图片转为 PyTorch 张量
    image_resized = torch.from_numpy(image_resized).float()

    # 如果原输入是 Tensor，发送到相同的设备
    if device is not None:
        image_resized = image_resized.to(device)
    
    return image_resized


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(dataset_path, dataset_name) -> Dict:
    # Step1. 构造label路径(img_paths)
    # segmentations_path = os.path.join(dataset_path, 'segmentations')
    # # 获取 segmentations_path 下所有子文件夹的名称

    json_path = os.path.expanduser('~/LangSplat/eval/3dovs_feature_map_order.json')
    with open(json_path, 'r') as f:
        config = json.load(f)
    # 提取 file_order 列表 这是featuremap的真实顺序
    file_order = config.get(dataset_name, [])

    segmentations_path = f"{dataset_path}/segmentations"
    temp_save_path = os.path.expanduser('~/LangSplat/temp_ovs')
    os.makedirs(temp_save_path, exist_ok=True)
    label_names = [
        name for name in os.listdir(segmentations_path) if os.path.isdir(os.path.join(segmentations_path, name))
    ]
    print(label_names)
    # 构造完整路径
    image_paths = []
    folder_path = os.path.join(dataset_path, 'images')
    for label in label_names:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 如果文件名匹配并且是jpg格式（不区分大小写）
            if filename.lower() == f"{label}.jpg":
                image_paths.append(os.path.join(folder_path, filename))

    print(image_paths)


    # Step4.读取蒙版  构造gt_ann dict
    gt_ann = {}
    for label in label_names:
        mask_folder = os.path.join(segmentations_path, label)
        dict_dict = {}

        for mask_name in os.listdir(mask_folder):
            mask_path = os.path.join(mask_folder, mask_name)
            if os.path.isfile(mask_path) and mask_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 去掉文件扩展名
                mask_name_wo_ext = os.path.splitext(mask_name)[0]  
                # 灰度图
                mask = Image.open(mask_path).convert('L') 


                # # 保存图像到本地，指定保存路径和文件格式
                # save_path = os.path.join(temp_save_path, f"{label}_{mask_name}.png")
                # mask.save(save_path)
                # 降采样（Resize）
                mask_downsapled = downsample(mask)
                # 降采样把二值采样成多个值了
                mask_downsapled = threshold_mask(mask_downsapled)

                dict_dict[mask_name_wo_ext]={
                    'mask': mask_downsapled,
                }

        gt_ann[label] = dict_dict
    return gt_ann, mask_downsapled.shape, image_paths

def check_binary_mask(mask, name="mask"):
    unique_vals = np.unique(mask)
    if np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [1]) or np.array_equal(unique_vals, [0, 1]):
        print(f"{name} is binary.")
        return True
    else:
        print(f"⚠️ {name} is NOT binary. Unique values: {unique_vals}")
        return False


def activate_stream(sem_map,  # 语义图
                    image,    # 输入图像，用于显示和处理激活图。
                    clip_model, # CLIP 模型实例，用于获取语义映射的最大激活值。
                    image_name: Path = None, #  保存图像和热力图的路径。
                    img_ann: Dict = None,  # 图像标注（annotations），包括每个语义对应的掩膜（mask）。
                    thresh : float = 0.5, # 用于计算掩膜的阈值（0.5表示只有超过50%激活值的区域才被认为是目标）。
                    colormap_options = None,
                    idx = None,
                    scene_name = None):
    
    # sem_map 是输入的语义图（semantic map），通常包含了每个像素在不同语义类别下的激活值。
    # sem_map:[3, 1080, 1440, 512]
    # valid_map:[3, 6, 1080, 1440]
    valid_map = clip_model.get_max_across(sem_map) #[head,h,w,c]->[head,prompt,h,w] # 3xkx832x1264
    # sem_map是作者提供的,他又没说输出是按什么顺序,所以顺序乱了很正常


    n_head, n_prompt, h, w = valid_map.shape
    image = downsample_rgb(image)
    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []

    # 遍历所有语义
    for k in range(n_prompt):
        iou_lv = np.zeros(n_head)
        mask_lv = np.zeros((n_head, h, w))
        # 遍历所有level
        for i in range(n_head):

            # 取出第i个层级的第k个level的map
            np_relev = valid_map[i][k].cpu().numpy()

            # 平滑
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            # 保存热力图heatmap
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            # 限定在0-1范围,再在最后添加一个维度使之变成(N, H, W, 1)
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)

            # 再对p_i做了归一化处理,范围缩放到[0, 1],colormaps.apply_colormap 将该值映射为一个彩色图像，采用 turbo 调色板
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))

            # 创建一个掩码（mask）,其值为 True 或 False，表示 valid_map[i][k] 中小于 0.5 的位置,mask是一个二维矩阵
            mask = (valid_map[i][k] < 0.5).squeeze()

            # 对于掩码中值为 True 的区域，将合成图像 valid_composited 中的对应像素值设置为原始图像 image 中的像素值的 30%（即进行一些颜色混合，减少亮度）
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask 
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1) # 0-1 范围的热力图，可用于生成 mask

            # 热力图转为二值 mask + 平滑
            mask_pred_uint8 = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred_uint8 = smooth(mask_pred_uint8)
            mask_lv[i] = mask_pred_uint8

            # 保存 GT 掩码（可视化用）
            mask_gt_unit8 = img_ann[clip_model.positives[k]]['mask']

            # 如果是 torch.Tensor，转换为 numpy 数组
            if isinstance(mask_gt_unit8, torch.Tensor):
                mask_npy = mask_gt_unit8.cpu().numpy()  # 转换为 numpy 数组
            else:
                mask_npy = mask_gt_unit8

            mask_gt_unit8 = mask_npy.astype(np.uint8)      
            mask_gt_binary_unit8 = (mask_gt_unit8 > 0).astype(np.uint8)  # 非0视为前景
            mask_gt_image = (mask_gt_binary_unit8.astype(np.uint8) * 255)  # 将True映射为255，False映射为0
            mask_pred_image = (mask_pred_uint8.astype(np.uint8) * 255)

            # 保存为图片
            save_dir = f'./mask_res/{scene_name}/{clip_model.positives[k]}_lv{i}'
            os.makedirs(save_dir, exist_ok=True)

            imsave(os.path.join(save_dir, f'mask_gt_{clip_model.positives[k]}_lv{i}.png'), mask_gt_image)
            imsave(os.path.join(save_dir, f'mask_pred_{clip_model.positives[k]}_lv{i}.png'), mask_pred_image)  # 保存为mask_pred.png


            # 计算 IoU（交并比）
            intersection = np.sum(np.logical_and(mask_gt_binary_unit8, mask_pred_uint8))
            union = np.sum(np.logical_or(mask_gt_unit8, mask_pred_uint8))
            iou = np.sum(intersection) / np.sum(union)

            # 第i个level在第k个prompt词下的交并比
            iou_lv[i] = iou

        
        score_lv = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lv[i] = score

        # 选择score 最高的那个level
        chosen_lv = torch.argmax(score_lv)
        print(f"{clip_model.positives[k]}_{idx:0>5},  choose lv{chosen_lv}")

        # 这个level所有语义的交并比
        chosen_iou_list.append(iou_lv[chosen_lv])

        # 被选择的level
        chosen_lvl_list.append(chosen_lv.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lv[chosen_lv], save_path)

    return chosen_iou_list, chosen_lvl_list

def evaluate(feat_dir, output_path, ae_ckpt_path, dataset_path, mask_thresh, encoder_hidden_dims, decoder_hidden_dims, logger, dataset_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # dataset_path = '/data2/jian/LangSplat/data/3dovs/bed'
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(dataset_path, dataset_name)
    
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    
    # 每个level下的每个图片只对应一个大的语义热力图
    # feat_dir[0] = '/data2/jian/LangSplat/output/bed_1/train/ours_None/renders_npy'
    # eval_index_list[0] = 10
    print(feat_dir)
    for i in range(len(feat_dir)):
        # 加载一个level下所有的渲染出来的语义图npy路径,类似于/data2/jian/LangSplat/output/bed_1/train/ours_None/renders_npy/00.npy
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        print(feat_paths_lvl)
        print(eval_index_list)
        # eval_index_list 里边有的才去加载,这里只加载 10 4 0 30 23对应的语义图
        for j, idx in enumerate(eval_index_list):
            # j是在eval_index_list里的index idx其实就是10 4 0 30 23
            print(f'compressed_sem_feats[i][j].shape = {compressed_sem_feats[i][j].shape}')
            print(f'np.load(feat_paths_lvl[idx]).shape = {np.load(feat_paths_lvl[idx]).shape}')
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all, chosen_lvl_list = [], []
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx:0>5}' # f'{idx:0>5}' = 00000
        scene_name = os.path.basename(output_path)
        image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)
        image_path_j = os.path.expanduser(image_paths[j])

        rgb_img = cv2.imread(image_path_j)[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1)           # 3x832x1264x512
        
        img_ann = gt_ann[f'{idx:02d}']
        keys = list(img_ann.keys())
        clip_model.set_positives(keys)
        c_iou_list, c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options,idx = idx, scene_name = scene_name)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    # # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    # total_bboxes = 0
    # for img_ann in gt_ann.values():
    #     total_bboxes += len(list(img_ann.keys()))
    # acc = acc_num / total_bboxes
    # logger.info("Localization accuracy: " + f'{acc:.4f}')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    feat_dir = [os.path.join(args.feat_dir, dataset_name+f"_{i}", "train/ours_None/renders_npy") for i in range(1,4)]
    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)
    dataset_path = os.path.join(args.dataset_path, dataset_name)

    evaluate(feat_dir, output_path, ae_ckpt_path, dataset_path, mask_thresh, args.encoder_dims, args.decoder_dims, logger, dataset_name)