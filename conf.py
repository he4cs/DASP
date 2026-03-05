import os
import sys
import argparse
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - see README.md for more information
_C.SETTING = "continual"

# Data directory
_C.DATA_DIR = "./data"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Output directory
_C.SAVE_DIR = "./output"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# GPU device id to use
_C.GPU_ID = '0'

# Enables printing intermediate results every x batches.
# Default -1 corresponds to no intermediate results
_C.PRINT_EVERY = -1

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 1

# Deterministic experiments.
_C.DETERMINISM = False

# Precision
_C.MIXED_PRECISION = True

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Model architecture
_C.MODEL.ARCH = 'cav-mae-ft'

# Path to a specific checkpoint
_C.MODEL.CKPT_PATH = "ckpt/cav-mae-ft-vgg.pth"

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

# Reset the model after a certain amount of update steps (e.g., used in RDumb)
_C.MODEL.RESET_AFTER_NUM_UPDATES = 0

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'vggsound' # Choose from: ['vggsound', 'ks50']

# Domain sequence for evaluation
_C.CORRUPTION.DOMAIN_SEQUENCE = [
    {'video': 'gaussian_noise'},
    {'video': 'shot_noise'},
    {'audio': 'gaussian_noise'},
    {'video': 'impulse_noise'},
    {'video': 'defocus_blur'},
    {'audio': 'traffic'},
    {'video': 'glass_blur'},
    {'video': 'motion_blur'},
    {'video': 'zoom_blur'},
    {'audio': 'crowd'},
    {'video': 'snow'},
    {'video': 'frost'},
    {'video': 'fog'},
    {'audio': 'rain'},
    {'video': 'brightness'},
    {'video': 'contrast'},
    {'video': 'elastic_transform'},
    {'audio': 'wind'},
    {'video': 'pixelate'},
    {'video': 'jpeg_compression'}
]

_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate. If num_ex != -1, each sequence is sub-sampled to the specified amount
_C.CORRUPTION.NUM_EX = -1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-4

# Optimizer choices: Adam, AdamW, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta1 for Adam based optimizers
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# --------------------------------- EATA options ---------------------------- #
_C.EATA = CfgNode()

# Fisher alpha. If set to 0.0, EATA becomes ETA and no EWC regularization is used
_C.EATA.FISHER_ALPHA = 2000.0

# Diversity margin
_C.EATA.D_MARGIN = 0.05
_C.EATA.MARGIN_E0 = 0.4  # Will be multiplied by: EATA.MARGIN_E0 * math.log(num_classes)

# --------------------------------- SAR options ---------------------------- #
_C.SAR = CfgNode()

# Threshold e_m for model recovery scheme
_C.SAR.RESET_CONSTANT_EM = 0.2

# --------------------------------- ABPEM options ---------------------------- #
_C.ABPEM = CfgNode()

_C.ABPEM.LAMBDA_ENT = 1.0
_C.ABPEM.LAMBDA_BAL = 1.0
_C.ABPEM.LAMBDA_ATTN = 1.0
_C.ABPEM.PRIME_ENT_K = 8

# --------------------------------- TSA options --------------------------- #
_C.TSA = CfgNode()

_C.TSA.LOSS_COEFF = 0.5
_C.TSA.GUMBEL_SOFTMAX_TAU = 1e-3
_C.TSA.CONF_THRE = 0.95

# ------------------------------- DASP options ----------------------------- #
_C.DASP = CfgNode()

_C.DASP.DELTA = 0.05
_C.DASP.LAMBDA_ENT = 0.5
_C.DASP.LAMBDA_KL = 1.0

# ------------------------------- Testing options ------------------------- #
_C.TEST = CfgNode()

# Number of workers for test data loading
_C.TEST.NUM_WORKERS = 4

# Batch size for evaluation (and updates)
_C.TEST.BATCH_SIZE = 128

# If the batch size is 1, a sliding window approach can be applied by setting window length > 1
_C.TEST.WINDOW_LENGTH = 1

# Debuging mode
_C.TEST.DEBUG = False

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL.ADAPTATION}_{cfg.CORRUPTION.DATASET}_{current_time}")
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            # enforce determinism
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(data_root_dir: str, dataset_name: str):
    # map dataset name to data directory name
    mapping = {"vggsound": "VGGSound", "ks50": "Kinetics50"}
    assert dataset_name in mapping.keys(),\
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(mapping.keys())}"
    return os.path.join(data_root_dir, mapping[dataset_name])


def get_num_classes(dataset_name: str):
    mapping = {"vggsound": 309, "ks50": 50}
    assert dataset_name in mapping.keys(), \
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(mapping.keys())}"
    return mapping[dataset_name]
