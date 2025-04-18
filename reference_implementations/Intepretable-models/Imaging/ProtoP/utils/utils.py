import random
import numpy as np
import torch   

def set_random_seed(rseed):
    """
    Set random seed for reproducibility.
    """
    random.seed(rseed)
    np.random.seed(rseed)
    torch.manual_seed(rseed)   
    torch.cuda.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)
