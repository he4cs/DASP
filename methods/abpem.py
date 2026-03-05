"""
Builds upon: https://github.com/YushengZhao/ABPEM
Corresponding paper: https://arxiv.org/abs/2503.02221
"""
import torch

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


def kl_div(p_mean, p_std, q_mean, q_std):
    return torch.log(q_std / p_std) + (p_std ** 2 + (p_mean - q_mean) ** 2) / (2 * q_std ** 2) - 0.5


@ADAPTATION_REGISTRY.register()
class ABPEM(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.lambda_ent = cfg.ABPEM.LAMBDA_ENT
        self.lambda_bal = cfg.ABPEM.LAMBDA_BAL
        self.lambda_attn = cfg.ABPEM.LAMBDA_ATTN
        self.prime_ent_k = cfg.ABPEM.PRIME_ENT_K

    def loss_calculation(self, test_input):
        outputs, attn = self.model.module(*test_input, return_attn=True)

        p_sum = outputs.softmax(dim=-1).mean(dim=-2)
        loss_bal = - (p_sum * p_sum.log()).sum()

        pred = outputs.softmax(dim=-1)
        ent_all = -pred * torch.log(pred + 1e-6)
        ent_idx = torch.sort(pred, dim=-1, descending=True)[1]
        prime_ent = torch.gather(ent_all, dim=-1, index=ent_idx[:, :self.prime_ent_k]).sum(dim=-1)
        loss_ra = prime_ent.mean()

        attn = attn.mean(1)
        attn_a2a = attn[:, :512, :512]
        attn_v2v = attn[:, 512:, 512:]
        attn_a2v = attn[:, :512, 512:]
        attn_v2a = attn[:, 512:, :512]
        attn_a2a_mean = attn_a2a.mean(dim=(1, 2)).detach()
        attn_v2v_mean = attn_v2v.mean(dim=(1, 2)).detach()
        attn_a2v_mean = attn_a2v.mean(dim=(1, 2))
        attn_v2a_mean = attn_v2a.mean(dim=(1, 2))
        attn_a2a_std = attn_a2a.std(dim=(1, 2)).detach()
        attn_v2v_std = attn_v2v.std(dim=(1, 2)).detach()
        attn_a2v_std = attn_a2v.std(dim=(1, 2))
        attn_v2a_std = attn_v2a.std(dim=(1, 2))
        loss_attn = kl_div(attn_v2a_mean, attn_v2a_std, attn_a2a_mean, attn_a2a_std).mean() + kl_div(attn_a2v_mean, attn_a2v_std, attn_v2v_mean, attn_v2v_std).mean()

        loss = loss_ra * self.lambda_ent - loss_bal * self.lambda_bal + loss_attn * self.lambda_attn
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