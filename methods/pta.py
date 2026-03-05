"""
Builds upon: https://github.com/MPI-Lab/PTA
Corresponding paper: https://neurips.cc/virtual/2025/loc/san-diego/poster/117876
"""
from tkinter import NO

import torch

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy


@ADAPTATION_REGISTRY.register()
class PTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.softmax_entropy = Entropy()

    def loss_calculation(self, test_input):
        outputs, attn = self.model(*test_input, return_attn=True)

        entropys = self.softmax_entropy(outputs)
        pred = outputs.argmax(1)
        conf = outputs.softmax(dim=-1).max(dim=-1)[0]

        # calculate frequency
        class_counts = torch.unique(pred, return_counts=True)
        unique_classes = class_counts[0]    
        counts = class_counts[1]
        pred_slash = torch.tensor([1 / len(unique_classes) for _ in range(len(unique_classes))], device=self.device)
        observe = counts / counts.sum()
        
        z = pred_slash - observe
        z_tan = torch.tanh(z)
        Z = torch.zeros_like(pred, dtype=torch.float, device=self.device)
        for i, label in enumerate(unique_classes):
            class_mask = (pred == label)
            Z[class_mask] = z_tan[i]
        
        perform_update = True
        if (Z > 0).any().item() and (Z < 0).any().item():
            positive_indices = torch.where(Z > 0)[0]
            negative_indices = torch.where(Z < 0)[0]

            # rank weight for pos
            rank_conf = quantile_rank(conf[positive_indices])
            rank_bias = quantile_rank(Z[positive_indices], True)
            total_weights = rank_bias * rank_conf
           
            # attn regularization
            pos = attn[positive_indices]
            neg = attn[negative_indices]

            loss_attn = mmd_rbf_single_kernel(pos, neg)
            loss_ent = (total_weights.detach() * entropys[positive_indices]).mean(0)
            loss_ent_neg = entropys[negative_indices].mean(0)
            loss_total = loss_ent - 0.5 * loss_ent_neg + 1.0 * loss_attn
        else:
            perform_update = False
            loss_total = None
        
        return outputs, loss_total, perform_update

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.amp.autocast('cuda'):
                outputs, loss, perform_update = self.loss_calculation(x)
            if perform_update:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss, perform_update = self.loss_calculation(x)
            if perform_update:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs
    
    def configure_model(self):
        if self.cfg.MODEL.ARCH == "cav-mae-ft":
            self.saf_name = 'module.blocks_u.0.attn.qkv'
        else:
            raise ValueError(f"Unsupported model architecture: {self.cfg.MODEL.ARCH}")
        
        self.model.train()
        self.model.requires_grad_(False)
        for name, module in self.model.named_modules():
            if name == self.saf_name:
                module.requires_grad_(True)

    def collect_params(self):
        params = []
        names = []
        for name, module in self.model.named_modules():
            if name == self.saf_name:
                for np, p in module.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f'{name}.{np}')
        return params, names


def quantile_rank(x, descending=False):
    n = x.size(0)
    
    sorted_x, sorted_indices = torch.sort(x, descending=descending)
    ranks = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)

    diff = torch.cat([
        torch.tensor([True], device=x.device),
        sorted_x[1:] != sorted_x[:-1],
        torch.tensor([True], device=x.device)
    ])
    unique_indices = torch.where(diff)[0]
    
    for i in range(len(unique_indices) - 1):
        start = unique_indices[i]
        end = unique_indices[i + 1]
        if start == end:
            continue
        segment_ranks = ranks[start:end]
        ranks[start:end] = segment_ranks.mean()
    
    restored_ranks = torch.empty_like(ranks)
    restored_ranks[sorted_indices] = ranks
    
    normalized_ranks = restored_ranks / n
    return normalized_ranks


def mmd_rbf_single_kernel(source, target, sigma=None, block_size=16):
    total = torch.cat([source, target], dim=0)
    
    N = total.size(0)
    b1 = source.size(0)
    total_flat = total.view(N, -1) # (N, C*H*W)
    
    l2_distance = torch.zeros(N, N, device=source.device)
    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            i_end = min(i + block_size, N)
            j_end = min(j + block_size, N)
            block_i = total_flat[i:i_end]
            block_j = total_flat[j:j_end]
            l2_distance[i:i_end, j:j_end] = torch.cdist(block_i, block_j, p=2) ** 2

    if sigma is None:
        # median of pairwise distances
        with torch.no_grad():
            triu_indices = torch.triu_indices(N, N, offset=1)
            median_sq_dist = torch.median(l2_distance[triu_indices[0], triu_indices[1]])
            sigma = torch.sqrt(0.5 * median_sq_dist)
           
    kernel = torch.exp(-l2_distance / (sigma ** 2))
    
    XX = kernel[:b1, :b1]
    YY = kernel[b1:, b1:]
    XY = kernel[:b1, b1:]
    
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd