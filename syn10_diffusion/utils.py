from typing import Optional
import math
from pathlib import Path
import yaml


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


def parse_config(config_path: Optional[str]) -> dict:
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error parsing config file {path.resolve()}: {e}")

    return config


def validate_config(config: dict):
    def check_attn_resolutions(image_size: int, channel_mult: list, attn_resolutions: list):
        if len(attn_resolutions) > 0:
            if not all([math.log2(res).is_integer() for res in attn_resolutions]):
                raise ValueError(f"Attn resolutions must be power of 2, got {attn_resolutions}")
            assert attn_resolutions == sorted(attn_resolutions, reverse=True)
            resolutions = [image_size]
            for _ in range(len(channel_mult) - 1):
                image_size //= 2
                resolutions.append(image_size)
            provided_but_not_used = set(attn_resolutions) - set(resolutions)
            if provided_but_not_used:
                raise ValueError(f"Attn resolutions {provided_but_not_used} are not used")

    assert config["image_size"] > 0
    assert config["image_channels"] > 0
    assert config["num_epochs"] > 0
    assert config["num_diffusion_timesteps"] > 0
    assert config["s"] > 0.0
    assert config["lambda_variational"] > 0.0
    assert config["norm_channels"] > 0
    assert config["model_channels"] > 0
    assert config["num_resnet_blocks"] > 0
    assert config["lr"] > 0.0
    assert config["grad_clip"] > 0.0
    assert config["train_batch_size"] > 0
    assert config["checkpoint_freq"] > 0
    assert config["tensorboard_freq"] > 0
    assert config["sample_batch_size"] > 0
    assert config["guidance"] > 0.0
    assert 0.0 < config["ema_decay"] < 1.0
    assert config["ema_delay"] >= 0
    assert config["image_max_value"] > config["image_min_value"] >= 0
    assert config["num_classes"] > 1
    assert 1.0 > config["max_beta"] > 0.0
    assert 0.0 <= config["dropout"] < 1.0
    assert 1.0 > config["p_uncond"] >= 0.0
    assert config["t_embed_mult"] >= 1
    assert len(config["channel_mult"]) > 0
    assert config["channel_mult"][0] == 1
    assert config["channel_mult"] == sorted(config["channel_mult"])
    image_size_channel_mult_mod = config["image_size"] % (2 ** (len(config["channel_mult"]) - 1))
    image_size_channel_mult_div = config["image_size"] // (2 ** (len(config["channel_mult"]) - 1))
    assert image_size_channel_mult_mod == 0 and image_size_channel_mult_div >= 3
    assert config["model_input_channels"] == config["image_channels"]
    assert 2 * config["model_input_channels"] == config["model_output_channels"] > 0
    assert config["model_channels"] % config["head_channels"] == 0
    assert int(config["model_channels"] * config["y_embed_mult"]) > 0
    assert config["model_channels"] % config["norm_channels"] == 0
    check_attn_resolutions(config["image_size"], config["channel_mult"], config["attn_resolutions"])




