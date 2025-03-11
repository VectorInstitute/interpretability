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

def get_multiclass_roc_auc_score(y_probs: np.array,
                                 y_true: np.array) -> float:
    """
    """
    num_classes = y_true.shape[1]
    per_class_score = [
                        roc_auc_score(y_true[:, i], y_probs[:, i]) 
                        for i in range(num_classes)
    ]
    return np.mean(np.array(per_class_score))

def aggregate_auc_predictions(predictions: np.array,
                              labels: np.array) -> Tuple[int, int]:
    """
    """
    device_id = torch.cuda.current_device()
    # Initialize lists to gather predictions and labels from all devices
    predictions = torch.tensor(predictions).cuda(device_id)
    labels = torch.tensor(labels).cuda(device_id)

    gathered_preds = [
        torch.zeros_like(predictions) for _ in range(dist.get_world_size())
    ]
    gathered_labels = [
        torch.zeros_like(labels) for _ in range(dist.get_world_size())
    ]
    # Use all_gather to collect all data across devices
    dist.all_gather(gathered_preds, predictions)
    dist.all_gather(gathered_labels, labels)

    # Concatenate data across devices
    all_preds = torch.cat(gathered_preds)
    all_labels = torch.cat(gathered_labels)

    # Convert to numpy for AUC calculation
    return all_preds.cpu().numpy(), all_labels.cpu().numpy()

def get_dist_auc(y_probs: np.array,
                 y_true: np.array) -> float:
    """
    """
    all_preds, all_labels = aggregate_auc_predictions(y_probs, y_true)
     # Calculate AUC on rank 0
    if dist.get_rank() == 0:
        auc = get_multiclass_roc_auc_score(all_preds, all_labels)
        return auc
    return -1