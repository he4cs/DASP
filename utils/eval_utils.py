import torch
import numpy as np
import logging
from typing import Union


logger = logging.getLogger(__name__)


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 print_every: int,
                 device: Union[str, torch.device]):
    
    num_correct = 0.
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            output = model(tuple(inp.to(device) for inp in data[:-1]))
            
            labels = data[-1].to(device)
            preds = output.argmax(1)
            num_correct += (preds == labels).float().sum()
            
            # track progress
            num_samples += labels.size(0)
            if print_every > 0 and (i+1) % print_every == 0:
                logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} accuracy = {num_correct / num_samples:.2%}")

    accuracy = num_correct.item() / num_samples
    return accuracy, num_samples
