#  extracted from: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py


import torch
from torch.nn import functional as F
from utils.utils import prototype_heatmap



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






# Compute prototype representations
def compute_prototypes(support_set, model):
    with torch.no_grad():
        support_images = torch.stack([s[0] for s in support_set]).to(model.device)
        support_labels = torch.tensor([s[1] for s in support_set], device=model.device)
        unique_classes = torch.unique(support_labels)
        features = model(support_images)
        features = F.normalize(model(support_images), p=2, dim=1)
        prototypes = torch.stack([
            features[support_labels == cls].mean(dim=0) 
            for cls in torch.unique(support_labels)
        ])
    return prototypes, unique_classes


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
    



# Compute classification loss and accuracy
def compute_loss_and_accuracy(query_set, prototypes, model,unique_classes,visualize_heatmap):
    query_images = torch.stack([q[0] for q in query_set]).to(model.device)
    query_labels = torch.tensor([q[1] for q in query_set], device=model.device)
    query_features = model(query_images)
    query_features = F.normalize(model(query_images), p=2, dim=1)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    distances = euclidean_dist(query_features, prototypes)
    # Map query labels to indices in unique_classes (since they may be non-sequential)
    class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
    mapped_labels = torch.tensor([class_to_index[label.item()] for label in query_labels], device=model.device)

    loss = F.cross_entropy(-distances, mapped_labels)
    # Compute accuracy using mapped indices
    pred_indices = torch.argmin(distances, dim=1)  # Closest prototype index
    pred_labels = torch.tensor([unique_classes[idx].item() for idx in pred_indices], device=model.device)
    assigned_prototypes = prototypes[pred_indices]
    accuracy = (pred_labels == query_labels).float().mean().item()
    if visualize_heatmap:
        # Visualize the prototype heatmap for the a query image
        prototype_heatmap(query_images[0],query_features[0].unsqueeze(0),prototypes[0].unsqueeze(0))
    return loss, accuracy
