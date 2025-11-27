import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------
# Plot Shape Functions for Interpretability
# ---------------------------
def plot_shape_functions_and_distributions(
    model: torch.nn.Module, X: np.ndarray, feature_names: list
):
    """Plot the shape functions and feature distributions for interpretability
    Parameters:
        model (torch.nn.Module): Trained CoxNAM model
        X (np.ndarray): Training set features
        feature_names (list): List of feature names
    """
    num_features = X.shape[1]

    # Dynamically determine grid size (balanced aspect ratio)
    ncols = min(4, int(np.ceil(np.sqrt(num_features))))  # Up to 4 columns
    nrows = int(np.ceil(num_features / ncols))  # Compute rows

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for i in range(num_features):
        ax = axes[i]
        feature_network = model.feature_networks[i]
        feature_network.eval()

        feature_values = X[:, i]
        feature_mean = np.mean(feature_values)
        feature_std = np.std(feature_values)

        # Define range for plotting shape functions
        sample_inputs = np.linspace(-3, 3, 100).reshape(-1, 1)
        with torch.no_grad():
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32)
            shape_values = feature_network(sample_inputs_tensor).numpy()

        # Convert sample inputs back to original scale
        orig_sample_inputs = sample_inputs * feature_std + feature_mean

        # Plot histogram
        ax.hist(feature_values, bins=30, alpha=0.7, color="b", density=True)

        # Plot shape function
        ax2 = ax.twinx()
        ax2.plot(orig_sample_inputs, shape_values, color="r")

        ax.set_xlabel("Feature Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax2.set_ylabel("Shape Function", fontsize=8)
        ax.set_title(feature_names[i], fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True)

    # Hide any unused subplots
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color="b", lw=3, label="Distribution"),
        plt.Line2D([0], [0], color="r", lw=3, label="Shape Function"),
    ]

    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig("framingham_shape_functions.png", dpi=300)
    plt.show()


def plot_baseline_survival(time_grid, S0):
    """
    Plot the baseline survival function and cumulative hazard.
    Parameters:
        time_grid: Array of time points.
        S0: Array of baseline survival probabilities at each time.
        H0: Array of cumulative baseline hazard values at each time.
    Returns:
        None
    """
    print("Baseline survival function computed.")

    # Optionally, plot the baseline survival function
    plt.figure(figsize=(6, 4))
    plt.step(time_grid, S0, where="post")
    plt.xlabel("Time")
    plt.ylabel("Baseline Survival Probability S0(t)")
    plt.title("Estimated Baseline Survival Function")
    plt.grid(True)
    plt.savefig("baseline_survival_function.png", dpi=300)
    plt.close()
    print("Baseline survival plot saved to baseline_survival_function.png")


def plot_survival_curve(survival_probs, time_grid):
    """
    Plot survival curves for test samples.
    Parameters:
        survival_probs: 2D numpy array of survival probabilities for each test sample.
        time_grid: Array of time points.
    Returns:
        None
    """
    # For example, plot survival curves for the first 5 test samples:
    plt.figure(figsize=(8, 6))
    for i in range(min(5, survival_probs.shape[0])):
        plt.step(
            time_grid, survival_probs[i, :], where="post", label=f"Test sample {i+1}"
        )
    plt.xlabel("Time")
    plt.ylabel("Survival Probability S(t|x)")
    plt.title("Survival Curves for Test Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_survival_curves_framingham.png", dpi=300)
    plt.close()
    print("Test survival curves plot saved to test_survival_curves.png")
