import random
import os
import numpy as np

DEFAULT_SEED = 42

def seed_everything(seed: int = DEFAULT_SEED):
    """Seed all pseudo-random number generators for strict reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    return seed
