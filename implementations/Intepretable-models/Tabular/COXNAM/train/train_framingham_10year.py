import os
import yaml
import torch
import ultraimport
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

coxnam = ultraimport.create_ns_package('coxnam', '__dir__/../coxnam')
from coxnam.loss import cox_loss
from coxnam.model import CoxNAM

# ---------------------------
# Load and Preprocess Data (with Fixed Survival Time)
# ---------------------------
def load_and_prepare_data(file_path: str="../../datasets/framingham.csv",
                          test_size: float=0.2) -> tuple:
    """
    Load the Framingham dataset and perform preprocessing steps:
    - Impute missing values using median imputation
    - Split into train and test sets
    - Normalize features using StandardScaler
    - Convert to PyTorch tensors
    Parameters:
        file_path (str): Path to the Framingham dataset CSV file
        test_size (float): Fraction of data to reserve for test set
    Returns:
        tuple: data, duration and event tensors for train and test sets,
        original train and test dataframes
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Use median imputation to replace NaNs
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Use a fixed survival time of 10 years for every subject
    duration = np.full(df_imputed.shape[0], 10.0)

    # Event indicator (1 = CHD occurred within 10 years, 0 = censored)
    event = df_imputed['TenYearCHD'].values

    # Keep all predictors, including age (since age is now a predictor, not the survival time)
    X = df_imputed.drop(columns=['TenYearCHD'])

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
            event_train_tensor, event_test_tensor, X, df_imputed)

# ---------------------------
# Train CoxNAM Model
# ---------------------------
def train_model(config: dict,
                X_tensor: torch.tensor,
                duration_tensor: torch.tensor,
                event_tensor: torch.tensor,
                num_epochs: int=50,
                batch_size: int=128) -> torch.nn.Module:
    """
    Train the CoxNAM model using the Cox partial likelihood loss.
    Parameters:
        X_tensor (torch.tensor): Normalized feature tensor
        duration_tensor (torch.tensor): Duration tensor
        event_tensor (torch.tensor): Event tensor
        num_epochs (int): Number of training epochs
        batch_size (int): Mini-batch size
    Returns:
        torch.nn.Module: Trained CoxNAM model
    """
    num_samples = X_tensor.shape[0]
    num_features = X_tensor.shape[1]
    input_dim = config['train_params']['input_dim']  # Each feature is treated as a scalar input
    hidden_units = config['train_params']['hidden_units']  # Define the MLP structure
    
    # Initialize the CoxNAM model
    coxnam_model = CoxNAM(num_features, input_dim, hidden_units)
    optimizer = optim.Adam(coxnam_model.parameters(), lr=config['optimizer']['lr'])

    for epoch in range(config['train_params']['num_epochs']):
        coxnam_model.train()
        
        generator = torch.Generator().manual_seed(config['seed'])
        permutation = torch.randperm(num_samples, generator=generator)

        epoch_loss, batch_size = 0.0, config['train_params']['batch_size']

        for i in range(0, num_samples, batch_size):
            mini_batch_indices = permutation[i:i+batch_size]
            optimizer.zero_grad()

            # Forward pass on the full dataset to compute risk scores
            risk_scores_full = coxnam_model(X_tensor)

            loss = cox_loss(risk_scores_full, duration_tensor,
                            event_tensor, coxnam_model,
                            mini_batch_indices=mini_batch_indices)

            # Check for numerical issues
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, mini-batch {i//batch_size}")

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(mini_batch_indices)

        avg_loss = epoch_loss / num_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    print("CoxNAM training complete!")
    return coxnam_model

# ---------------------------
# Evaluate Model Performance
# ---------------------------
def evaluate_model(coxnam_model: torch.nn.Module,
                   X_test_tensor: torch.tensor,
                   duration_test: np.ndarray,
                   event_test:  np.ndarray) -> float:
    """ Evaluate the model on the test set using concordance index (C-index)
    Parameters:
        coxnam_model (torch.nn.Module): Trained CoxNAM model
        X_test_tensor (torch.tensor): Normalized feature tensor for test set
        duration_test (np.ndarray): Duration tensor for test set
        event_test (np.ndarray): Event tensor for test set
    Returns:
        float: C-index on the test set
    """
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
def plot_shape_functions_and_distributions(model: torch.nn.Module,
                                           X: np.ndarray,
                                           feature_names: list):
    """
    Plot the shape functions learned by the CoxNAM model along with feature value distributions.
    Parameters:
        model (torch.nn.Module): Trained CoxNAM model
        X (np.ndarray): Original feature matrix
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

        # Plot histogram of the feature values
        ax.hist(feature_values, bins=30, alpha=0.7, color='b', density=True)

        # Plot the learned shape function (using a secondary y-axis)
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
    
    # Save the figure and display it
    plt.savefig('framingham_shape_functions.png', dpi=300)
    plt.show()

# ---------------------------
# Main Function
# ---------------------------
def main():
    """ Main function to load data, train the CoxNAM model, evaluate on test set, and plot shape functions.
    """

    #Get coxnam model config
    yaml_file = os.path.join(os.path.dirname(__file__), 'coxnam.yaml')
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    data_file = os.path.join(config['data_dir'], config['framingham_file'])
    (X_train_tensor, X_test_tensor, duration_train_tensor, duration_test_tensor,
     event_train_tensor, event_test_tensor, X_train, df_imputed) = load_and_prepare_data(data_file)

    #Set seeds for reproducibility
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)  # Ensures CUDA ops use the same seed

    # Train the CoxNAM model using the fixed 10-year survival time
    coxnam_model = train_model(config, X_train_tensor,
                               duration_train_tensor, event_train_tensor)

    # Evaluate on test set
    evaluate_model(coxnam_model, X_test_tensor, 
                   duration_test_tensor.numpy(), event_test_tensor.numpy())

    # Plot shape functions for interpretability
    feature_names = X_train.columns.tolist()
    plot_shape_functions_and_distributions(coxnam_model, X_train.to_numpy(), feature_names)
    
if __name__ == "__main__":
    main()