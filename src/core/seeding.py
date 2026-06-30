import random
import numpy as np
import torch
import os

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds across all libraries for reproducibility

    *Args:   
        seed: int seed value - store in config and will be logged by mlflow
        deterministic: When true - forces cuDNN  into deterministic mode - performance cost but guarantees reproducibility on GPU
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True finds the fastest conv algorithm per input shape
        # non-deterministic but faster, when you dont care about exact reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True