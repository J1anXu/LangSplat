# -*- coding:utf-8 -*-
# 
# Author: 
# Time: 

import torch
import fnmatch
import numpy as np
import os

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#  交替方向乘子法（ADMM） 优化算法，通常用于解决带有约束和正则化的优化问题。
class ADMM:
    def __init__(self, gsmodel, rho, device):
        self.gsmodel = gsmodel 
        self.device = device # 指定张量存储的位置（CPU 或 GPU）
        self.init_rho = rho # 标量，用于 ADMM 中的惩罚项
        # u和z是ADMM算法中的辅助变量（对偶变量）
        self.u = {}
        self.z = {}
        self.rho = self.init_rho
        opacity = self.gsmodel.get_opacity
        self.u = torch.zeros(opacity.shape).to(device)
        self.z = torch.Tensor(opacity.data.cpu().clone().detach()).to(device)

    # 注意,paper中的a就是这里的opacity
    def update(self, threshold, update_u = True):
        #  a + λ
        z = self.gsmodel.get_opacity + self.u
        # z ← proxh(a + λ)  裁剪辅助变量z  Update z via Eq. 16;  这里相当于是h 
        self.z = torch.Tensor(self.prune_z(z, threshold)).to(self.device)
        # λ = λ + a − z.  Update λ via Eq. 17
        if update_u: 
            with torch.no_grad():
                diff =  self.gsmodel.get_opacity - self.z
                self.u += diff

    #  该方法根据不同的策略（由 opt 参数控制）来更新 z：
    def prune_z(self, z, threshold):
        z_update = self.metrics_sort(z, threshold)  
        return z_update

    # opacity 和 z之间的差异就是损失,z是辅助不用关注
    def get_admm_loss(self, loss): 
        return 0.5 * self.rho * (torch.norm(self.gsmodel.get_opacity - self.z + self.u, p=2)) ** 2

    def adjust_rho(self, epoch, epochs, factor=5): # 根据训练的当前进度（epoch 和 epochs）调整 rho 值。通常，rho 会随着训练的进展而增大，从而增加对约束的惩罚
        if epoch > int(0.85 * epochs):
            self.rho = factor * self.init_rho
    
    def metrics_sort(self, z, threshold): # 根据透明度的排序值来更新 z。它通过将透明度按升序排序并应用一个阈值来选择透明度值
        index = int(threshold * len(z))
        z_sort = {}
        z_update = torch.zeros(z.shape)
        z_sort, _ = torch.sort(z, 0)
        z_threshold = z_sort[index-1]
        z_update= ((z > z_threshold) * z)  
        return z_update
    
    def metrics_sample(self, z, opt): # 该方法根据透明度值的相对权重进行随机采样。首先，将透明度值归一化为概率分布，并按此分布随机选择样本。
        index = int((1 - opt.pruning_threshold) * len(z))
        prob = z / torch.sum(z)
        prob = prob.reshape(-1).cpu().numpy()
        indices = torch.tensor(np.random.choice(len(z), index, p = prob, replace=False))
        expand_indices = torch.zeros(z.shape[0] - len(indices)).int()
        indices = torch.cat((indices, expand_indices),0).to(self.device)
        z_update = torch.zeros(z.shape).to(self.device)
        z_update[indices] = z[indices]
        return z_update

    def metrics_imp_score(self, z, imp_score, opt): # 该方法基于重要性分数（imp_score）更新 z。重要性分数低于某个阈值的透明度值会被置为 0，从而对重要性较低的部分进行稀疏化。
        index = int(opt.pruning_threshold * len(z))
        imp_score_sort = {}
        imp_score_sort, _ = torch.sort(imp_score, 0)
        imp_score_threshold = imp_score_sort[index-1]
        indices = imp_score < imp_score_threshold 
        z[indices == 1] = 0  
        return z        



def get_unactivate_opacity(gaussians):
    opacity = gaussians._opacity[:, 0]
    scores = opacity
    return scores

def get_pruning_mask(scores, threshold):        
    scores_sorted, _ = torch.sort(scores, 0)
    threshold_idx = int(threshold * len(scores_sorted))
    abs_threshold = scores_sorted[threshold_idx - 1]
    mask = (scores <= abs_threshold).squeeze()
    return mask

def check_grad_leakage(model, optimizer=None):
    """
    适用于非 nn.Module 的模型，自行遍历属性检查参数梯度状态。
    """
    print("\n=== 🚨 检查梯度泄露和优化器参数（自定义模型） ===")
    leak_found = False

    for name in dir(model):
        param = getattr(model, name)
        if isinstance(param, torch.Tensor) and param.requires_grad is not None:
            if param.grad is not None and not param.requires_grad:
                print(f"⚠️ 参数 '{name}' 有 grad，但 requires_grad=False！可能泄露。")
                leak_found = True
            elif param.requires_grad:
                print(f"✅ 参数 '{name}' 设置为可训练 (requires_grad=True)。")
            else:
                print(f"🔒 参数 '{name}' 不可训练 (requires_grad=False)。")

    # 检查 optimizer 参数组
    if optimizer is not None:
        print("\n=== 🔍 检查 Optimizer 中的参数 ===")
        for i, group in enumerate(optimizer.param_groups):
            for p in group['params']:
                if not p.requires_grad:
                    print(f"⚠️ Optimizer param_group[{i}] 中包含 requires_grad=False 的参数")
                    leak_found = True

    if not leak_found:
        print("✅ 没发现泄露或异常参数。")
    raise("=== 检查完成 ===\n")
