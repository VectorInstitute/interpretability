import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from model import CoxNAM  
from utils.loss import cox_loss 
from utils.surv_utils import compute_baseline_survival, compute_final_survival_probabilities

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(42)  # Ensures CUDA ops use the same seed

# ---------------------------
# Load and Preprocess Data (with Train-Test Split)
# ---------------------------
def load_and_prepare_data(file_path="../datasets/framingham.csv", test_size=0.2):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Use median imputation to replace NaNs
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Define survival time and event indicator
    duration = df_imputed['age'].values  # Using age as survival time
    event = df_imputed['TenYearCHD'].values  # Event indicator (1 = CHD occurred, 0 = censored)

    # Drop non-predictor columns
    X = df_imputed.drop(columns=['age', 'TenYearCHD'])

    # Split into train & test sets (80% train, 20% test)
    X_train, X_test, duration_train, duration_test, event_train, event_test = train_test_split(
        X, duration, event, test_size=test_size, random_state=42, stratify=event
    )

    # Normalize features separately for train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train, transform on train
    X_test_scaled = scaler.transform(X_test)  # Transform test using train statistics

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    duration_train_tensor = torch.tensor(duration_train, dtype=torch.float32)
    duration_test_tensor = torch.tensor(duration_test, dtype=torch.float32)
    event_train_tensor = torch.tensor(event_train, dtype=torch.float32)
    event_test_tensor = torch.tensor(event_test, dtype=torch.float32)

    return (X_train_tensor, X_test_tensor, duration_train_tensor, duration_test_tensor,
            event_train_tensor, event_test_tensor, X_train, X_test, df_imputed)

# ---------------------------
# Train CoxNAM Model
# ---------------------------
def train_model(X_tensor, duration_tensor, event_tensor, num_epochs=150, batch_size=128, l1_lambda=0.01):
    num_samples = X_tensor.shape[0]
    num_features = X_tensor.shape[1]
    input_dim = 1  # Each feature is a scalar
    hidden_units = [32,16,8]  # Define MLP structure
    
    torch.manual_seed(42)

    # Initialize the model
    coxnam_model = CoxNAM(num_features, input_dim, hidden_units)
    optimizer = optim.Adam(coxnam_model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        coxnam_model.train()
        
        generator = torch.Generator().manual_seed(42)
        permutation = torch.randperm(num_samples, generator=generator)

        epoch_loss = 0.0

        for i in range(0, num_samples, batch_size):
            mini_batch_indices = permutation[i:i+batch_size]
            optimizer.zero_grad()

            # Forward pass on the full dataset to compute risk scores
            risk_scores_full = coxnam_model(X_tensor)

            loss = cox_loss(risk_scores_full, duration_tensor, event_tensor, coxnam_model, mini_batch_indices=mini_batch_indices)


            # Check for numerical issues
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, mini-batch {i//batch_size}")

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(mini_batch_indices)

        avg_loss = epoch_loss / num_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    print("Training complete!")
    return coxnam_model

# ---------------------------
# Evaluate Model Performance
# ---------------------------
def evaluate_model(coxnam_model, X_test_tensor, duration_test, event_test):
    coxnam_model.eval()
    with torch.no_grad():
        risk_scores_test = coxnam_model(X_test_tensor).numpy().flatten()

    # Calculate the concordance index (C-index) on test data
    c_index = concordance_index(duration_test, -risk_scores_test, event_test)  # Use negative risk scores
    print(f"📊 Test C-index: {c_index:.4f}")
    return c_index




# ---------------------------
# Plot Shape Functions for Interpretability
# ---------------------------
def plot_shape_functions_and_distributions(model, X, feature_names):
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
        ax.hist(feature_values, bins=30, alpha=0.7, color='b', density=True)

        # Plot shape function
        ax2 = ax.twinx()
        ax2.plot(orig_sample_inputs, shape_values, color='r')

        ax.set_xlabel('Feature Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax2.set_ylabel('Shape Function', fontsize=8)
        ax.set_title(feature_names[i], fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True)

    # Hide any unused subplots
    for j in range(num_features, len(axes)):
        axes[j].axis('off')

    handles = [plt.Line2D([0], [0], color='b', lw=3, label='Distribution'),
               plt.Line2D([0], [0], color='r', lw=3, label='Shape Function')]

    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig('framingham_shape_functions.png', dpi=300)
    plt.show()
    
# ---------------------------
# Main Function
# ---------------------------
def main():
    (X_train_tensor, X_test_tensor, duration_train_tensor, duration_test_tensor,
     event_train_tensor, event_test_tensor, X_train, X_test, df) = load_and_prepare_data("../datasets/framingham.csv")

    # Train model on training set
    coxnam_model = train_model(X_train_tensor, duration_train_tensor, event_train_tensor)

    # Evaluate on test set
    evaluate_model(coxnam_model, X_test_tensor, duration_test_tensor.numpy(), event_test_tensor.numpy())

    # Plot shape functions
    feature_names = X_train.columns.tolist()
    # Compute baseline survival function using training data
    time_grid, H0, S0 = compute_baseline_survival(coxnam_model, X_train_tensor, duration_train_tensor, event_train_tensor)
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

    # Compute final survival probabilities for the test set
    survival_probs_test = compute_final_survival_probabilities(coxnam_model, X_test_tensor, time_grid, H0)
    # For example, plot survival curves for the first 5 test samples:
    plt.figure(figsize=(8, 6))
    for i in range(min(5, survival_probs_test.shape[0])):
        plt.step(time_grid, survival_probs_test[i, :], where="post", label=f"Test sample {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability S(t|x)")
    plt.title("Survival Curves for Test Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_survival_curves_framingham.png", dpi=300)
    plt.close()
    print("Test survival curves plot saved to test_survival_curves.png")
    plot_shape_functions_and_distributions(coxnam_model, X_train.to_numpy(), feature_names)
    
    
 

if __name__ == "__main__":
    main()