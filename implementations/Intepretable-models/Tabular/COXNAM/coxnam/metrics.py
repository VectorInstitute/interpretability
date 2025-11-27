import torch
import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc


def compute_td_auc(
    model,
    X_train_tensor,
    X_test_tensor,
    duration_train_tensor,
    duration_test_tensor,
    event_train_tensor,
    event_test_tensor,
    epoch=None,
    plot=True,
):
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
    y_train = np.array(
        [(e, t) for e, t in zip(event_train, duration_train)],
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        [(e, t) for e, t in zip(event_test, duration_test)],
        dtype=[("event", bool), ("time", float)],
    )

    # Ensure model is in evaluation mode
    model.eval()

    # Get risk scores from the model (move input to the same device as the model)
    with torch.no_grad():
        risk_scores = model(X_test_tensor).detach().cpu().numpy().flatten()

    # Dynamically ensure time_grid stays within observed range
    min_time = (
        duration_test.min() + (duration_test.max() - duration_test.min()) * 0.01
    )  # Slight buffer above min
    max_time = (
        duration_test.max() - (duration_test.max() - duration_test.min()) * 0.01
    )  # Slight buffer below max
    time_grid = np.linspace(min_time, max_time, num=10)

    # Compute TD-AUC
    td_auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, time_grid)

    # Log results
    if epoch is not None:
        print(f"[Epoch {epoch}] Mean TD-AUC: {np.mean(td_auc):.3f}")

    # Plot TD-AUC over time
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(
            time_grid, td_auc, marker="o", linestyle="-", label="Time-Dependent AUC"
        )
        plt.xlabel("Time (Days)")
        plt.ylabel("AUC")
        plt.title("Time-Dependent AUC (TD-AUC) Over Time")
        plt.legend()
        plt.grid()
        plt.savefig("td_auc.png")

    return np.mean(td_auc)


def compute_td_concordance_index(
    model, X, durations, events, time_point, time_grid, H0
):
    """
    Optimized version of time-dependent concordance index (TD-CI).

    Uses NumPy vectorization instead of nested loops for efficiency.

    Args:
        model: Trained CoxNAM model.
        X: Feature tensor for evaluation.
        durations: Numpy array (or list) of event times.
        events: Numpy array (or list) of event indicators (1 for event, 0 for censored).
        time_point: The time at which to compute survival probabilities.
        time_grid: Array of time points used for baseline hazard estimation.
        H0: Array of cumulative baseline hazard values corresponding to time_grid.

    Returns:
        td_c_index: The time-dependent concordance index (float). Returns None if no comparable pairs.
    """
    # Ensure model is in evaluation mode
    model.eval()
    with torch.no_grad():
        risk_scores = model(X).cpu().numpy().flatten()

    # Interpolate to get the cumulative hazard at time_point
    H0_t = np.interp(time_point, time_grid, H0)

    # Compute survival probabilities: S(t|x) = exp(-H0(t) * exp(h(x)))
    S_pred = np.exp(-H0_t * np.exp(risk_scores))

    # Convert durations & events to numpy if they are tensors
    durations = (
        durations.cpu().numpy().flatten()
        if isinstance(durations, torch.Tensor)
        else np.array(durations).flatten()
    )
    events = (
        events.cpu().numpy().flatten()
        if isinstance(events, torch.Tensor)
        else np.array(events).flatten()
    )

    # Select only individuals with observed events before time_point
    event_mask = (events == 1) & (durations <= time_point)

    if np.sum(event_mask) == 0:
        print("No comparable pairs found for the specified time_point.")
        return None

    event_times = durations[event_mask]
    survival_probs_event = S_pred[event_mask]

    # Sort by event times
    sorted_indices = np.argsort(event_times)
    event_times = event_times[sorted_indices]
    survival_probs_event = survival_probs_event[sorted_indices]

    # Select only individuals who have not yet experienced the event
    risk_mask = durations > np.expand_dims(
        event_times, axis=1
    )  # Compare each event time to all other durations
    num_comparable = np.sum(risk_mask, axis=1)

    if np.sum(num_comparable) == 0:
        return None  # No valid comparisons

    # Get the survival probabilities for those at risk
    survival_probs_risk = (
        S_pred[np.newaxis, :] * risk_mask
    )  # Masked array where survival probabilities exist only for those at risk

    # Compare survival probabilities
    correct_order = (
        survival_probs_event[:, np.newaxis] < survival_probs_risk
    )  # Lower survival probability = higher risk
    tie_cases = survival_probs_event[:, np.newaxis] == survival_probs_risk

    # Count correct pairs
    num_correct = np.sum(correct_order, axis=1) + 0.5 * np.sum(tie_cases, axis=1)

    # Compute the final time-dependent concordance index
    td_c_index = np.sum(num_correct) / np.sum(num_comparable)
    return td_c_index
