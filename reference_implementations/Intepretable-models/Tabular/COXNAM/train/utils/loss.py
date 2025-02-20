import torch


# def cox_loss(risk_scores, duration, event, model, l2_lambda=0.01, l1_lambda=0.01, mini_batch_indices=None):
#     """
#     Compute the Cox loss with L2 and L1 regularization.
    
#     Parameters:
#         risk_scores (Tensor): Model output (log hazards).
#         duration (Tensor): Survival times.
#         event (Tensor): Event indicators (1 if event occurred, 0 if censored).
#         model (nn.Module): The CoxNAM model to apply L2 and L1 regularization.
#         l2_lambda (float): L2 regularization strength.
#         l1_lambda (float): L1 regularization strength.
#         mini_batch_indices (Tensor, optional): Indices for mini-batch sampling.
    
#     Returns:
#         Tensor: Computed loss value.
#     """
#     if risk_scores.ndim > 1:
#         risk_scores = risk_scores.squeeze()

#     if mini_batch_indices is not None:
#         risk_scores = risk_scores[mini_batch_indices]
#         duration = duration[mini_batch_indices]
#         event = event[mini_batch_indices]

#     # Filter only observed events (C(i) == 1)
#     event_filter = event == 1
#     risk_scores = risk_scores[event_filter]
#     duration = duration[event_filter]

#     # Sort by descending risk score
#     sorted_indices = torch.argsort(risk_scores, descending=True)
#     sorted_risk_scores = risk_scores[sorted_indices]
#     sorted_duration = duration[sorted_indices]

#     epsilon = 1e-6  # Small value for numerical stability

#     # Compute risk set sum (sum over survival times ≥ current time)
#     exp_risk_scores = torch.exp(sorted_risk_scores)
#     cumsum_exp_risk_scores = torch.cumsum(exp_risk_scores, dim=0) + epsilon

#     # Compute negative log-partial likelihood
#     log_likelihood = torch.sum(sorted_risk_scores - torch.log(cumsum_exp_risk_scores))

#     # Compute L2 regularization term
#     l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
    
#     # Compute L1 regularization term
#     l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
    
#     # Final loss
#     total_loss = -log_likelihood + l2_lambda * l2_penalty + l1_lambda * l1_penalty

#     # Debugging: Ensure no NaNs in loss
#     if torch.isnan(total_loss) or torch.isinf(total_loss):
#         print("Warning: NaN detected in Cox loss. Returning zero loss.")
#         return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#     return total_loss


def cox_loss(risk_scores, duration, event, model, l2_lambda=0.01, l1_lambda=0.01, mini_batch_indices=None):
    """
    Compute the Cox loss with L2 and L1 regularization, using observed events sorted by descending duration.
    
    Parameters:
        risk_scores (Tensor): Model output (log hazards).
        duration (Tensor): Survival times.
        event (Tensor): Event indicators (1 if event occurred, 0 if censored).
        model (nn.Module): The CoxNAM model (used for regularization).
        l2_lambda (float): L2 regularization strength.
        l1_lambda (float): L1 regularization strength.
        mini_batch_indices (Tensor, optional): Indices for mini-batch sampling.
    
    Returns:
        Tensor: Computed loss value.
    """
    # Ensure risk_scores is a 1D tensor.
    if risk_scores.ndim > 1:
        risk_scores = risk_scores.squeeze()

    # If mini-batch indices are provided, select the corresponding entries.
    if mini_batch_indices is not None:
        risk_scores = risk_scores[mini_batch_indices]
        duration = duration[mini_batch_indices]
        event = event[mini_batch_indices]

    # Filter to include only observed events.
    observed_mask = (event == 1)
    risk_scores_obs = risk_scores[observed_mask]
    duration_obs = duration[observed_mask]

    epsilon = 1e-6  # For numerical stability

    # Sort the observed events by descending duration.
    # This ordering ensures that the cumulative sum of exp(risk) represents the risk set:
    # all subjects with durations >= current event's duration.
    sorted_indices = torch.argsort(duration_obs, descending=True)
    sorted_risk_scores = risk_scores_obs[sorted_indices]

    # Compute cumulative sum of exponentiated risk scores.
    exp_risk_scores = torch.exp(sorted_risk_scores)
    cumsum_exp_risk_scores = torch.cumsum(exp_risk_scores, dim=0) + epsilon

    # Compute the log partial likelihood.
    log_likelihood = torch.sum(sorted_risk_scores - torch.log(cumsum_exp_risk_scores))

    # Compute regularization penalties.
    l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
    l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())

    # Final loss: negative log-likelihood with regularization.
    total_loss = -log_likelihood + l2_lambda * l2_penalty + l1_lambda * l1_penalty

    # Debug: Return zero loss if NaN/Inf is detected.
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("Warning: NaN detected in Cox loss. Returning zero loss.")
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    return total_loss
