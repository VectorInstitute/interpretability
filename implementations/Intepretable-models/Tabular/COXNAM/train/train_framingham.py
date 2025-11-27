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
from coxnam.surv_utils import *
from coxnam.plots import *

# ---------------------------
# Load and Preprocess Data (with Train-Test Split)
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
def train_model(X_tensor: torch.tensor,
                duration_tensor: torch.tensor,
                event_tensor: torch.tensor,
                num_epochs: int=150,
                batch_size: int=128):
    """ Train the CoxNAM model on the training set
    Parameters:
        X_tensor (torch.tensor): Training set features
        duration_tensor (torch.tensor): Training set survival times
        event_tensor (torch.tensor): Training set event indicators
        num_epochs (int): Number of training epochs
        batch_size (int): Mini-batch size
    Returns:
        torch.nn.Module: Trained CoxNAM model
    """
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

            loss = cox_loss(risk_scores_full, duration_tensor, event_tensor,
                            coxnam_model, mini_batch_indices=mini_batch_indices)


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
def evaluate_model(coxnam_model: torch.nn.Module,
                   X_test_tensor: torch.Tensor,
                   duration_test: np.ndarray,
                   event_test: np.ndarray) -> float:
    """ Evaluate the model on the test set using concordance index (C-index)
    Parameters:
        coxnam_model (torch.nn.Module): Trained CoxNAM model
        X_test_tensor (torch.Tensor): Test set features
        duration_test (np.ndarray): Test set survival times
        event_test (np.ndarray): Test set event indicators
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
# Main Function
# ---------------------------
def main():
    """ Main function to load data, train model, evaluate and plot shape functions
    """
    #Get coxnam model config
    yaml_file = os.path.join(os.path.dirname(__file__), 'coxnam.yaml')
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    data_file = os.path.join(config['data_dir'], config['framingham_file'])
    (X_train_tensor, X_test_tensor, duration_train_tensor, duration_test_tensor,
     event_train_tensor, event_test_tensor, X_train, X_test, df) = load_and_prepare_data(data_file)
    
    # Set seeds for reproducibility
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)  # Ensures CUDA ops use the same seed

    # Train model on training set
    coxnam_model = train_model(X_train_tensor, duration_train_tensor,
                               event_train_tensor,
                               num_epochs=config['train_params']['num_epochs'],
                               batch_size=config['train_params']['batch_size'])

    # Evaluate on test set
    evaluate_model(coxnam_model, X_test_tensor, duration_test_tensor.numpy(), event_test_tensor.numpy())

    # Plot shape functions
    feature_names = X_train.columns.tolist()

    # Compute baseline survival function using training data
    time_grid, H0, S0 = compute_baseline_survival(coxnam_model, X_train_tensor, duration_train_tensor, event_train_tensor)
    plot_baseline_survival(time_grid, S0)

    # Compute final survival probabilities for the test set
    survival_probs_test = compute_final_survival_probabilities(coxnam_model, X_test_tensor, time_grid, H0)
    plot_survival_curve(survival_probs_test, time_grid)

    plot_shape_functions_and_distributions(coxnam_model, X_train.to_numpy(), feature_names)

if __name__ == "__main__":
    main()
