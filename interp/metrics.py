from typing import Tuple
import torch
import numpy as np
from torch.nn import functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

def feature_loss(fnn_out,
                 lambda_=0.):
    """
    """
    return lambda_ * (fnn_out ** 2).sum() / fnn_out.shape[1]

def penalized_cross_entropy(logits,
                            truth,
                            fnn_out,
                            feature_penalty=0.):
    """
    """
    return F.binary_cross_entropy_with_logits(logits.view(-1), truth.view(-1)) + feature_loss(fnn_out, feature_penalty)

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)