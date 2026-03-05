import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy


class LowRankAdapter(nn.Module):
    def __init__(self, dim=768, r=64):
        super().__init__()
        self.down = nn.Linear(dim, r, bias=False)
        self.up = nn.Linear(r, dim, bias=False)
        self.act = nn.ReLU()
        
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(3))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


def calculate_redundancy(embed: torch.Tensor) -> torch.Tensor:
    normed_embed = F.normalize(embed, p=2, dim=0)  # (B, D)
    sim_matrix = torch.matmul(normed_embed.transpose(0, 1), normed_embed).abs() # (D, D)
    
    D = sim_matrix.size(0)
    mask = ~torch.eye(D, dtype=torch.bool, device=embed.device)
    off_diag_values = sim_matrix[mask]
    return off_diag_values.mean()


def calculate_redundancy_with_filter(embed1: torch.Tensor, embed2: torch.Tensor, threshold_ratio: float = 0.05):
    joint_embed = torch.cat([embed1, embed2], dim=0) # (2B, D)
    std = joint_embed.std(dim=0) # (D,)
    max_std = std.max()
    
    active_mask = std > (max_std * threshold_ratio)
    num_kept = active_mask.sum().item()
    
    if num_kept < 2:
        return 0.0, 0.0
    
    def _calculate_redundancy(e):
        e_filtered = e[:, active_mask]
        e_normed = F.normalize(e_filtered, p=2, dim=0) # (B, D)
        sim_matrix = torch.matmul(e_normed.transpose(0, 1), e_normed) # (D, D)
        
        D = sim_matrix.size(0)
        mask = ~torch.eye(D, dtype=torch.bool, device=e.device)
        off_diag_values = sim_matrix[mask]
        
        return torch.sqrt((off_diag_values ** 2).mean()).item()

    score1 = _calculate_redundancy(embed1)
    score2 = _calculate_redundancy(embed2)
    
    return score1, score2


@ADAPTATION_REGISTRY.register()
class DASP(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.softmax_entropy = Entropy()
        self.delta = cfg.DASP.DELTA
        self.lambda_ent = cfg.DASP.LAMBDA_ENT
        self.lambda_kl = cfg.DASP.LAMBDA_KL

    def loss_calculation(self, test_input, eps: float = 1e-7):
        # z - tokens after encoder, h - embeddings before classifier
        _, z_a, z_v, h_a, h_v = self.model.module.forward(*test_input, 'tta')
        
        # r_a = calculate_redundancy(h_a)
        # r_v = calculate_redundancy(h_v)
        
        # variance filtering
        r_a, r_v = calculate_redundancy_with_filter(h_a, h_v)
        
        perform_update = True
        if abs(r_a - r_v) > self.delta:
            if r_a > r_v:
                z_a = z_a + self.model.internal_adapter_a(z_a)
                z_a = z_a.detach()
                z_a = z_a + self.model.external_adapter_a(z_a)
                
                logits_ref = self.model.module.forward_decoder(z_v)
                z_v = z_v + self.model.internal_adapter_v(z_v)
                logits_ply = self.model.module.forward_decoder(z_v)
            else:
                z_v = z_v + self.model.internal_adapter_v(z_v)
                z_v = z_v.detach()
                z_v = z_v + self.model.external_adapter_v(z_v)
                
                logits_ref = self.model.module.forward_decoder(z_a)
                z_a = z_a + self.model.internal_adapter_a(z_a)
                logits_ply = self.model.module.forward_decoder(z_a)
        else:
            perform_update = False
        
        loss_kl = F.kl_div(
            F.log_softmax(logits_ply, dim=1),
            F.softmax(logits_ref, dim=1),
            reduction='batchmean'
        ) if perform_update else 0.0
        
        outputs_2 = self.model.module.forward_decoder(torch.cat((z_a, z_v), dim=1))
        
        loss_ent = self.softmax_entropy(outputs_2).mean()
        probs = outputs_2.softmax(1).clamp(eps, 1 - eps)
        p_mean = probs.mean(0)
        loss_div = (p_mean * p_mean.log()).sum()
        loss_total = loss_div + self.lambda_ent * loss_ent + self.lambda_kl * loss_kl

        return outputs_2, loss_total, perform_update

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
        internal_adapter_a = LowRankAdapter(dim=768, r=32).to(self.device)
        internal_adapter_v = LowRankAdapter(dim=768, r=32).to(self.device)
        
        external_adapter_a = nn.Linear(768, 768, bias=False).to(self.device)
        nn.init.zeros_(external_adapter_a.weight)
        external_adapter_v = nn.Linear(768, 768, bias=False).to(self.device)
        nn.init.zeros_(external_adapter_v.weight)
        
        self.model.add_module('external_adapter_a', external_adapter_a)
        self.model.add_module('external_adapter_v', external_adapter_v)
        self.model.add_module('internal_adapter_a', internal_adapter_a)
        self.model.add_module('internal_adapter_v', internal_adapter_v)
        
        self.model.train()
        self.model.requires_grad_(False)

        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
        
    def collect_params(self):
        params = []
        names = []
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                assert param.requires_grad, 'param not requires grad'
                params.append(param)
                names.append(name)
        return params, names