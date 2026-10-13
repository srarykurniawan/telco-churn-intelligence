import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except ImportError:  # pragma: no cover
        pass
