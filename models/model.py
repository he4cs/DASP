import torch
import logging
from typing import Union

from models import CAVMAEFT

logger = logging.getLogger(__name__)

def get_model(cfg, num_classes: int, device: Union[str, torch.device]):
    """
    Setup the pre-defined model architecture and restore the corresponding pre-trained weights
    Input:
        cfg: Configurations
        num_classes: Number of classes
        device: The device to put the loaded model
    Return:
        model: The pre-trained model
    """
    
    if cfg.MODEL.ARCH == "cav-mae-ft":
        base_model = CAVMAEFT(label_dim=num_classes, modality_specific_depth=11)
        mdl_weight = torch.load(cfg.MODEL.CKPT_PATH, map_location='cpu')
        if not isinstance(base_model, torch.nn.DataParallel):
            base_model = torch.nn.DataParallel(base_model)
        miss, unexpected = base_model.load_state_dict(mdl_weight, strict=False)
        logger.info(f'load cav-mae finetuned weights from {cfg.MODEL.CKPT_PATH}')
        logger.info(f'missing keys: {miss}, unexpected keys: {unexpected}')

    return base_model.to(device)