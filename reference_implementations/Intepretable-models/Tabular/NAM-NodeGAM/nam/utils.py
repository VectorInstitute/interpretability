import random

import torch
import numpy as np
from torch.utils.data import DataLoader

def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def get_full_data(data_dl: DataLoader):
    """
    """
    all_data = torch.stack([x for batch in data_dl for x in batch[0]])
    return all_data


def calculate_n_units(data,
                      n_basis_functions: int,
                      units_multiplier: int):
    """
    """
    num_unique_vals = [ len(np.unique(data[:, i])) for i in range(data.shape[1])]
    return [
        min(n_basis_functions, i * units_multiplier) for i in num_unique_vals
    ]

def random_seed(seed_value, use_cuda):
    """
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) 
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def get_device():
    """
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device