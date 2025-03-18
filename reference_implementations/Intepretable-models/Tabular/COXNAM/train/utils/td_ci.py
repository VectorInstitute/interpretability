import torch
import numpy as np


def compute_td_concordance_index(model, X, durations, events, time_point, time_grid, H0):
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
    durations = durations.cpu().numpy().flatten() if isinstance(durations, torch.Tensor) else np.array(durations).flatten()
    events = events.cpu().numpy().flatten() if isinstance(events, torch.Tensor) else np.array(events).flatten()

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
    risk_mask = durations > np.expand_dims(event_times, axis=1)  # Compare each event time to all other durations
    num_comparable = np.sum(risk_mask, axis=1)

    if np.sum(num_comparable) == 0:
        return None  # No valid comparisons

    # Get the survival probabilities for those at risk
    survival_probs_risk = S_pred[np.newaxis, :] * risk_mask  # Masked array where survival probabilities exist only for those at risk

    # Compare survival probabilities
    correct_order = (survival_probs_event[:, np.newaxis] < survival_probs_risk)  # Lower survival probability = higher risk
    tie_cases = (survival_probs_event[:, np.newaxis] == survival_probs_risk)

    # Count correct pairs
    num_correct = np.sum(correct_order, axis=1) + 0.5 * np.sum(tie_cases, axis=1)

    # Compute the final time-dependent concordance index
    td_c_index = np.sum(num_correct) / np.sum(num_comparable)
    return td_c_index