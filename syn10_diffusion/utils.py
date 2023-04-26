from typing import Optional
from pathlib import Path
import yaml

from syn10_diffusion import models


def seed_all(seed=313):
    """
    Seed torch and numpy with the given seed, even if not imported.

    Args:
        seed (int): Seed value for random number generation.
    """
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def parse_config(config_path: Optional[str]):
    if config_path is None:
        return {}
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_models():
    return {
        "UnetTest": models.UnetTest,
        "prod": models.UnetProd
    }
