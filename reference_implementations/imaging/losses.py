#  extracted from: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py


import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from utils.utils import get_roc_auc_score
from utils.utils import prototype_heatmap
import matplotlib.pyplot as plt

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


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



def prototypical_loss(original_image,inputs, target, n_support):
    """
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of belonging to a class c. Loss and accuracy are then computed
    and returned.

    Adjusted for multi-label datasets like NIH.

    Args:
    - input: the model output for a batch of samples (batch_size x feature_dim).
    - target: binary matrix (batch_size x num_classes), where each row is a multi-hot vector.
    - n_support: number of samples to use when computing barycentres for each class.
    """
    k = 0
    validation_estimated = []
    validation_true = []

    target_cpu = target
    input_cpu = inputs

    # Find active classes in the batch
    active_classes = torch.nonzero(target_cpu.sum(0)).squeeze(1)
    n_classes = len(active_classes)

    prototypes = []
    query_idxs = []

    # Compute prototypes for each active class
    
    for c in active_classes:
        # Get indices for samples belonging to class `c`
        
        class_idxs = torch.nonzero(target_cpu[:, c]).squeeze(1)

        # Separate support and query samples
        support_idxs = class_idxs[:n_support]
        query_idxs_c = class_idxs[n_support:]

        if len(support_idxs) > 0:
            # Compute class prototype
            class_prototype = input_cpu[support_idxs].mean(0)
            prototypes.append(class_prototype)
        else:
            print("No support samples for class {}".format(c))
        # Add query indices
        query_idxs.extend(query_idxs_c.tolist())
    
    if not prototypes or not query_idxs:
        # Handle edge case where no valid prototypes or queries are available
        
        return torch.tensor(0.0, device=inputs.device), 0.0  # Loss and AUC as placeholders

    prototypes = torch.stack(prototypes)  # Shape: (n_classes, feature_dim)
    query_samples = input_cpu[query_idxs]  # Shape: (n_query, feature_dim)

    query_samples = F.normalize(query_samples, p=2, dim=1)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    # Compute distances from queries to prototypes
    dists = euclidean_dist(query_samples, prototypes)
    
    # prototype_heatmap(original_image[query_idxs],query_samples,prototypes)
    # Compute log probabilities
    log_p_y = F.log_softmax(-dists, dim=1)

    # Multi-label target construction for queries
    query_targets = target_cpu[query_idxs][:, active_classes]

    
    
    loss_val = F.binary_cross_entropy_with_logits(-dists, query_targets.float())
    # Multi-label accuracy
    preds = (log_p_y > 0).float()  # Threshold at 0
    
    
    # Compute AUC
    validation_estimated = torch.exp(log_p_y).detach().cpu().numpy()
    validation_true = query_targets.detach().cpu().numpy()
    
    auc_val = get_roc_auc_score(validation_true, validation_estimated)

    return loss_val, auc_val,query_targets,torch.exp(log_p_y)
