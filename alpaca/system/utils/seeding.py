import random

import numpy as np
import torch
import tensorflow as tf

def set_seed(seed: int) -> None:
    """
    Sets the seed for all supported frameworks.
    """
    random.seed(seed)
    np.random.seed(0) 
    torch.manual_seed(seed)
    tf.keras.utils.set_random_seed(seed)
