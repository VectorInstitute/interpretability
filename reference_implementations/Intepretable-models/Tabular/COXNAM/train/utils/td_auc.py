import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
import torch


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
