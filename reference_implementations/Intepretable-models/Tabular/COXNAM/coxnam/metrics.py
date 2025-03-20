import torch
import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

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

def compute_td_auc(model, X_train_tensor, X_test_tensor, duration_train_tensor, duration_test_tensor, event_train_tensor, event_test_tensor, epoch=None, plot=True):
    """
    Computes Time-Dependent AUC (TD-AUC) at multiple time points during model evaluation.

    Parameters:
    - model: PyTorch model (CoxNAM or any survival model that outputs risk scores).
    - X_train_tensor: PyTorch tensor of training covariates.
    - X_test_tensor: PyTorch tensor of test covariates.
    - duration_train_tensor: PyTorch tensor of training survival times.
    - duration_test_tensor: PyTorch tensor of test survival times.
    - event_train_tensor: PyTorch tensor of training event indicators (1=event, 0=censored).
    - event_test_tensor: PyTorch tensor of test event indicators.
    - epoch: (Optional) Current epoch number for logging.
    - plot: (Optional, default=True) Whether to plot TD-AUC over time.

    Returns:
    - mean_td_auc: Mean Time-Dependent AUC across all time points.
    """

    # Move model to correct device
    device = next(model.parameters()).device
    X_test_tensor = X_test_tensor.to(device)

    # Convert PyTorch tensors to NumPy arrays (move to CPU before conversion)
    duration_train = duration_train_tensor.cpu().numpy()
    event_train = event_train_tensor.cpu().numpy()
    duration_test = duration_test_tensor.cpu().numpy()
    event_test = event_test_tensor.cpu().numpy()

    # Convert event & time into structured survival format
    y_train = np.array([(e, t) for e, t in zip(event_train, duration_train)], dtype=[('event', bool), ('time', float)])
    y_test = np.array([(e, t) for e, t in zip(event_test, duration_test)], dtype=[('event', bool), ('time', float)])

    # Ensure model is in evaluation mode
    model.eval()

    # Get risk scores from the model (move input to the same device as the model)
    with torch.no_grad():
        risk_scores = model(X_test_tensor).detach().cpu().numpy().flatten()

    # Dynamically ensure time_grid stays within observed range
    min_time = duration_test.min() + (duration_test.max() - duration_test.min()) * 0.01  # Slight buffer above min
    max_time = duration_test.max() - (duration_test.max() - duration_test.min()) * 0.01  # Slight buffer below max
    time_grid = np.linspace(min_time, max_time, num=10)


    # Compute TD-AUC
    td_auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, time_grid)

    # Log results
    if epoch is not None:
        print(f"[Epoch {epoch}] Mean TD-AUC: {np.mean(td_auc):.3f}")

    # Plot TD-AUC over time
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(time_grid, td_auc, marker="o", linestyle="-", label="Time-Dependent AUC")
        plt.xlabel("Time (Days)")
        plt.ylabel("AUC")
        plt.title("Time-Dependent AUC (TD-AUC) Over Time")
        plt.legend()
        plt.grid()
        plt.savefig("td_auc.png")

    return np.mean(td_auc)
