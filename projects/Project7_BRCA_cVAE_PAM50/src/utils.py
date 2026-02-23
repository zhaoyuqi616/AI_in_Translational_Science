import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs(out_dir: str):
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
