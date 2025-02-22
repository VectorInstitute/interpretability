<<<<<<< Updated upstream
'''
TODO: Add metrics and loss functions
'''
=======
import torch.nn.functional as F

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
