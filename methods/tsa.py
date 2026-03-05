"""
Builds upon: https://github.com/chenmc1996/Uni-Modal-Distribution-Shift
Corresponding paper: https://openreview.net/pdf?id=6EZMWeV5sH
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


def FixMatchLoss(logits_s, logits_w, threshold=0.95):
    # Pseudo-labeling for unlabeled data
    with torch.no_grad():
        # Get the maximum probabilities and the pseudo-labels
        probs_u = F.softmax(logits_w, dim=-1)
        max_probs, pseudo_labels = torch.max(probs_u, dim=-1)

        # Create a mask for confident pseudo-labels
        mask = max_probs.ge(threshold).float()

    # Unsupservised loss for unlabeled data
    loss_u = (F.cross_entropy(logits_s, pseudo_labels) * mask).mean()
    return loss_u, mask


@ADAPTATION_REGISTRY.register()
class TSA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.loss_coff = cfg.TSA.LOSS_COEFF
        self.gumbel_softmax_tau = cfg.TSA.GUMBEL_SOFTMAX_TAU
        self.conf_thre = cfg.TSA.CONF_THRE

    def loss_calculation(self, test_input):
        a, v = self.model.module.forward_encoder(*test_input)

        B = a.size(0)
        choice_probs = F.gumbel_softmax(self.model.weight_adapter.expand(
            B, -1), tau=self.gumbel_softmax_tau, hard=False)  # Temperature can be adjusted
        selection = choice_probs[:, 0].view(-1, 1, 1)

        a_adapter = self.model.a_adapter + torch.eye(768).to(self.device)
        a_adapted = torch.einsum('bij,jk->bik', a, a_adapter)

        v_adapter = self.model.v_adapter + torch.eye(768).to(self.device)
        v_adapted = torch.einsum('bij,jk->bik', v, v_adapter)
        
        # for inference
        concated_a = torch.cat([a, a_adapted], dim=0)
        concated_v = torch.cat([v_adapted, v], dim=0)

        # for adaptation
        a = selection * a + (1 - selection) * a_adapted
        v = (1 - selection) * v + selection * v_adapted

        outputs_1 = self.model.module.forward_decoder(torch.cat((a, v), dim=1))

        loss_self = FixMatchLoss(outputs_1, outputs_1.detach(), threshold=self.conf_thre)[0].mean()
        p_sum = outputs_1.softmax(dim=-1).sum(dim=-2)
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()
        loss = self.loss_coff * loss_self - loss_bal
        
        with torch.no_grad():
            outputs_2 = self.model.module.forward_decoder(torch.cat((concated_a, concated_v), dim=1))
            outputs_2 = outputs_2.chunk(2)
            outputs_2 = outputs_2[0] + outputs_2[1]

        return outputs_2, loss

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
        self.model.register_parameter('a_adapter', nn.Parameter(torch.zeros(768, 768)))
        self.model.register_parameter('v_adapter', nn.Parameter(torch.zeros(768, 768)))
        self.model.register_parameter('weight_adapter', nn.Parameter(torch.Tensor([0, 0])))
        self.model.to(self.device)
        
        self.model.train()
        self.model.requires_grad_(False)
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad_(True)

    def collect_params(self):
        params = []
        names = []
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                params.append(param)
                names.append(name)
        return params, names