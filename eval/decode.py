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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result

def eval_gt_lerfdata(dataset_path) -> Dict:
    # Step1. 构造label路径(img_paths)
    # segmentations_path = os.path.join(dataset_path, 'segmentations')
    # # 获取 segmentations_path 下所有子文件夹的名称
    segmentations_path = os.path.expanduser('~/LangSplat/data/3dovs/bed/segmentations')

    label_names = [
        name for name in os.listdir(segmentations_path) if os.path.isdir(os.path.join(segmentations_path, name))
    ]
    print(label_names)
    # 构造完整路径
    image_paths = [
        os.path.join(os.path.join(dataset_path, 'images'), f"{folder}.jpg") for folder in label_names
    ]

    # 打印成你想要的格式
    for i, path in enumerate(image_paths):
        print(f"# {i} ='{path}'")


    # Step4.读取蒙版  构造gt_ann dict
    gt_ann = {}
    mask_target_size = (1440,1080)  
    img_target_size = (1080,1440)
    for label in label_names:
        mask_folder = os.path.join(segmentations_path, label)
        dict_dict = {}

        for mask_name in os.listdir(mask_folder):
            mask_path = os.path.join(mask_folder, mask_name)
            if os.path.isfile(mask_path) and mask_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 去掉文件扩展名
                mask_name_wo_ext = os.path.splitext(mask_name)[0]  
                # 灰度图
                mask_img = Image.open(mask_path).convert('L')  
                # 降采样（Resize）
                print(f"Original size: {mask_img.size} target size: {mask_target_size}")
                mask_img = mask_img.resize(mask_target_size, resample=Image.NEAREST)
                mask_array = np.array(mask_img)
                print(f"Label: {label}, Mask: {mask_name_wo_ext}, Size: {mask_array.shape}")

                dict_dict[mask_name_wo_ext]={
                    'mask': mask_array,
                }

        gt_ann[label] = dict_dict


    return gt_ann, img_target_size, image_paths


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


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)
            # mask_pred范围是[0, 1]
            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred

            # mask_gt的范围是[0, 255]
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)

            # calculate iou
            mask_gt_bin = mask_gt / 255.0
            mask_pred_bin = mask_pred


            intersection = np.sum(np.logical_and(mask_gt_bin, mask_pred_bin))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

            # 保存 mask_gt 和 mask_pred
            save_root = Path.home() / "LangSplat" / "temp"
            save_root.mkdir(parents=True, exist_ok=True)

            # 构建保存路径，包括类别名和 head 编号
            prefix = f"{clip_model.positives[k]}_{i:02d}"

            gt_path = save_root / f"{prefix}_GT.png"
            pred_path = save_root / f"{prefix}_PRED.png"

            mask_gt_img = (mask_gt).astype(np.uint8)
            mask_pred_img = (mask_pred * 255).astype(np.uint8)

            cv2.imwrite(str(gt_path), mask_gt_img)
            cv2.imwrite(str(pred_path), mask_pred_img)
            print(f"Saved GT mask to {gt_path}")

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list

def evaluate(feat_dir, output_path, ae_ckpt_path, dataset_path, mask_thresh, encoder_hidden_dims, decoder_hidden_dims):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo", # 它指定使用哪种颜色映射来将数值数据转换为颜色。
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    print(f"dataset_path: {dataset_path}")

    gt_ann, image_shape, image_paths = eval_gt_lerfdata(dataset_path)

    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]


    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # TODO 读取所有特征,叠起来快速计算
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        for j in range(len(feat_paths_lvl)):
            feature = np.load(feat_paths_lvl[j])
            feature = torch.from_numpy(feature).float().to(device)
            print(f"feature shape: {feature.shape}")
            with torch.no_grad():
                h, w, _ = feature.shape
                feat_flatten = feature.flatten(0, 1)
                restored_feat = model.decode(feat_flatten)
                restored_feat = restored_feat.view(h, w, -1)   
                print(f"restored_feat shape: {restored_feat.shape}")
                np.save(os.path.join(output_path[i], f"{j:05d}.npy"), restored_feat.cpu().numpy())
                print(f"save restored_feat to {os.path.join(output_path[i], f'{j:05d}.npy')}")

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
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument('--dataset_path', type=str, default=None)
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
    # 注意这里改回去
    feat_dir = [os.path.join(args.feat_dir, dataset_name+f"_{i}", "train/ours_None/renders_npy") for i in range(1,4)]
    output_path = [os.path.join(args.feat_dir, dataset_name+f"_{i}", "train/ours_None/renders_decoded_npy") for i in range(1,4)]
    for path in output_path:
        os.makedirs(path, exist_ok=True)

    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    dataset_path = os.path.join(args.dataset_path, dataset_name)

    evaluate(feat_dir, output_path, ae_ckpt_path, dataset_path, mask_thresh, args.encoder_dims, args.decoder_dims)