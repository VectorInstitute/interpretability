import torch
import numpy as np


def compute_baseline_survival(
    coxnam_model, X_train, durations_train, events_train, time_grid=None
):
    """
    Estimate the baseline cumulative hazard H0(t) using the Breslow estimator and compute S0(t)=exp(-H0(t)).

    Args:
        coxnam_model: Trained CoxNAM model.
        X_train: Tensor of training features.
        durations_train: Tensor or numpy array of training event times.
        events_train: Tensor or numpy array of training event indicators (1 for event, 0 for censored).
        time_grid: Optional array of times at which to estimate the baseline survival.
                   If None, uses unique event times from durations_train.

    Returns:
        time_grid: Array of time points.
        H0: Array of cumulative baseline hazard values at each time.
        S0: Array of baseline survival probabilities at each time.
    """
    coxnam_model.eval()
    with torch.no_grad():
        risk_scores = coxnam_model(X_train)  # log-risk scores (shape: [n_samples, 1])
    # Convert to relative risk: exp(log-risk)
    risk_scores = torch.exp(risk_scores).cpu().numpy().flatten()

    # Ensure durations and events are numpy arrays
    durations = (
        durations_train.cpu().numpy().flatten()
        if isinstance(durations_train, torch.Tensor)
        else np.array(durations_train).flatten()
    )
    events = (
        events_train.cpu().numpy().flatten()
        if isinstance(events_train, torch.Tensor)
        else np.array(events_train).flatten()
    )

    # Use unique event times where event==1 if no time grid is provided
    if time_grid is None:
        time_grid = np.sort(np.unique(durations[events == 1]))

    H0 = np.zeros_like(time_grid, dtype=float)
    cumulative_hazard = 0.0

    # For each time point, calculate the increment of the cumulative hazard
    for i, t in enumerate(time_grid):
        # Count the number of events that occur exactly at time t
        d_t = np.sum((durations == t) & (events == 1))
        if d_t == 0:
            dH0 = 0.0
        else:
            # Risk set: indices of individuals who have not yet experienced the event by time t
            risk_set = risk_scores[durations >= t]
            # Breslow estimator: dH0 = (number of events at time t) / (sum of relative risks in the risk set)
            dH0 = d_t / (risk_set.sum() if risk_set.sum() > 0 else 1)
        cumulative_hazard += dH0
        H0[i] = cumulative_hazard

    # Baseline survival function: S0(t) = exp(-H0(t))
    S0 = np.exp(-H0)
    return time_grid, H0, S0


def compute_final_survival_probabilities(coxnam_model, X_new, time_grid, H0):
    """
    Compute the final survival probability curve for each new sample.

    For each sample with features x and predicted log-risk \hat{h}(x),
    the survival function is computed as:
        S(t|x) = exp(-H0(t) * exp(\hat{h}(x)))

    Args:
        coxnam_model: Trained CoxNAM model.
        X_new: Tensor of new features (e.g., test set).
        time_grid: Array of time points (should match those from compute_baseline_survival).
        H0: Array of baseline cumulative hazard values corresponding to time_grid.

    Returns:
        survival_probabilities: A 2D numpy array of shape (n_samples, len(time_grid)),
                                where each row contains the survival probabilities over time.
    """
    coxnam_model.eval()
    with torch.no_grad():
        risk_scores_new = coxnam_model(X_new)  # log-risk scores for new samples
    # Compute relative risks for new samples
    risk_new = torch.exp(risk_scores_new).cpu().numpy().flatten()  # shape: (n_samples,)

    # For each new sample, compute survival probability curve:
    # S(t|x) = exp(-H0(t) * exp(\hat{h}(x))) = exp(-H0(t) * risk)
    survival_probabilities = np.array([np.exp(-H0 * r) for r in risk_new])
    return survival_probabilities  # shape: (n_samples, len(time_grid))
