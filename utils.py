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
