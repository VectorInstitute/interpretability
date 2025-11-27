import torch
import numpy as np
from typing import Tuple
import torch.distributed as dist
from sklearn.metrics import roc_auc_score


def get_multiclass_roc_auc_score(y_probs: np.array, y_true: np.array) -> float:
    """ """
    num_classes = y_true.shape[1]
    per_class_score = [
        roc_auc_score(y_true[:, i], y_probs[:, i]) for i in range(num_classes)
    ]
    return np.mean(np.array(per_class_score))


def aggregate_auc_predictions(
    predictions: np.array, labels: np.array
) -> Tuple[int, int]:
    """ """
    device_id = torch.cuda.current_device()
    # Initialize lists to gather predictions and labels from all devices
    predictions = torch.tensor(predictions).cuda(device_id)
    labels = torch.tensor(labels).cuda(device_id)

    gathered_preds = [
        torch.zeros_like(predictions) for _ in range(dist.get_world_size())
    ]
    gathered_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
    # Use all_gather to collect all data across devices
    dist.all_gather(gathered_preds, predictions)
    dist.all_gather(gathered_labels, labels)

    # Concatenate data across devices
    all_preds = torch.cat(gathered_preds)
    all_labels = torch.cat(gathered_labels)

    # Convert to numpy for AUC calculation
    return all_preds.cpu().numpy(), all_labels.cpu().numpy()


def get_dist_auc(y_probs: np.array, y_true: np.array) -> float:
    """ """
    all_preds, all_labels = aggregate_auc_predictions(y_probs, y_true)
    # Calculate AUC on rank 0
    if dist.get_rank() == 0:
        auc = get_multiclass_roc_auc_score(all_preds, all_labels)
        return auc
    return -1
