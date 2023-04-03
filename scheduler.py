# Denoising Diffusion Probabilistic Model
# Scheduler Module

import numpy as np


def cosine_scheduler(timesteps: np.ndarray, out_dim: int, max_period: int = 10000) -> np.ndarray:
    """ Cosine scheduler for the diffusion process.

    Args:
        timesteps: A 1D array of timesteps.
        out_dim: The output dimension.
        max_period: The maximum period of the cosine function.

    Returns:  A 2D array of shape (len(timesteps), out_dim) containing the timestep embeddings.
    """
    assert len(timesteps.shape) == 1
    half_dim = out_dim // 2
    exponents = np.arange(half_dim) / (half_dim - 1)
    frequencies = np.exp(-np.log(max_period) * exponents)
    angles = timesteps[:, None] * frequencies[None]
    cos, sin = np.cos(angles), np.sin(angles)
    timestep_embeddings = np.concatenate((cos, sin), axis=1)
    assert timestep_embeddings.shape == (len(timesteps), out_dim)
    return timestep_embeddings


