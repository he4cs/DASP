"""
Builds upon: https://github.com/XLearning-SCU/2024-ICLR-READ
Corresponding paper: https://openreview.net/pdf?id=TPZRq4FALB
"""
import torch
import math

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

@ADAPTATION_REGISTRY.register()
class READ(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.softmax_entropy = Entropy()

    def loss_calculation(self, test_input):
        outputs = self.model(*test_input)
        
        p_sum = outputs.softmax(dim=-1).sum(dim=-2)
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()

        pred = outputs.softmax(dim=-1)
        pred_max = pred.max(dim=-1)[0]
        gamma = math.exp(-1)
        t = torch.ones(outputs.shape[0], device=outputs.device) * gamma
        loss_ra = (pred_max * (1 - pred_max.log() + t.log())).mean()
        
        loss = loss_ra - loss_bal
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.amp.autocast('cuda'):
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
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