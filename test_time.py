import os
import torch
import logging
import numpy as np

import methods
from models.model import get_model
from utils.misc import print_memory_info
from utils.eval_utils import get_accuracy
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",  # reset the model state after the adaptation to a domain
                      "continual",         # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",           # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",     # consecutive test samples are likely to originate from different domains
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model
    base_model = get_model(cfg, num_classes, device)
    
    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes).to(device)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")
    
    domain_sequence = cfg.CORRUPTION.DOMAIN_SEQUENCE
    assert isinstance(domain_sequence, list) and len(domain_sequence) > 0, f"Domain sequence must be a non-empty list."
    logger.info(f"Using {cfg.CORRUPTION.DATASET}_c with the following domain sequence: {domain_sequence}")
    
    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["vggsound", "ks50"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    accs = []
    accs_5 = []
    
    # start evaluation
    for i_dom, domain_name in enumerate(domain_seq_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except AttributeError:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(
                setting=cfg.SETTING,
                adaptation=cfg.MODEL.ADAPTATION,
                dataset_name=cfg.CORRUPTION.DATASET,
                data_root_dir=cfg.DATA_DIR,
                domain_name=domain_name,
                domain_names_all=domain_sequence,
                severity=severity,
                num_examples=cfg.CORRUPTION.NUM_EX,
                rng_seed=cfg.RNG_SEED,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
            )
            
            # evaluate the model
            acc, num_samples = get_accuracy(
                model,
                data_loader=test_data_loader,
                print_every=cfg.PRINT_EVERY,
                device=device
            )

            accs.append(acc)
            if severity == 5:
                accs_5.append(acc)

            logger.info(f"{cfg.CORRUPTION.DATASET} accuracy % [{domain_name}-{severity}][#samples={num_samples}]: {acc:.2%}")

    if len(accs_5) > 0:
        logger.info(f"mean accuracy: {np.mean(accs):.2%}, mean accuracy at 5: {np.mean(accs_5):.2%}")
    else:
        logger.info(f"mean accuracy: {np.mean(accs):.2%}")

    if cfg.TEST.DEBUG:
        print_memory_info()


if __name__ == '__main__':
    evaluate('Evaluation.')