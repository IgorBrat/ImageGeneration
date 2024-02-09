import torch
import numpy as np


def set_seeds(seed: int = 42):
    """
    Set seeds for debugging purposes
    :param seed: seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
