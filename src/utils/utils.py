import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from src.logging import utils as logging


def is_special_token(token: str):
    return token.startswith("<") and token.endswith(">")


def set_random_seed(seed):
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_exp_dir(config) -> str:
    exp_dir = os.path.join(config.save_dir, config.exp_name)
    os.makedirs(exp_dir)
    cfg_path = os.path.join(exp_dir, "config.yaml")
    OmegaConf.save(config=config, f=cfg_path, resolve=True)
    logging.info(f"Successfully created experiment directory: {exp_dir}")
    return exp_dir
