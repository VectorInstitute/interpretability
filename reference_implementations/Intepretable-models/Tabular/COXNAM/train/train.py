import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines.datasets import load_rossi
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

from coxnam import CoxNAM, cox_loss

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() \
                      else "mps" if torch.backends.mps.is_available()\
                      else "cpu")

def load_and_prepare_data() -> tuple:
    """
    Load the Rossi dataset and prepare it for training the Cox-NAM model.

    Returns:
    - X_tensor: PyTorch tensor of data
    - y_tensor: PyTorch tensor of target variable
    - duration_tensor: PyTorch tensor of duration feature
    - event_tensor: PyTorch tensor of event feature
    - X: DataFrame of features
    - df: Original DataFrame
    """
    # Load the dataset
    df = load_rossi()
    print(df.info())

    # Separate features and target variables
    X = df.drop(columns=['week', 'arrest'])
    y = df['arrest']

    # Convert DataFrame to NumPy array
    X_numpy = X.to_numpy()
    y_numpy = y.to_numpy()

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numpy)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_numpy, dtype=torch.float32).to(device)

    # Use real duration and event data from the dataset
    duration = df['week'].values
    event = df['arrest'].values

    # Convert to PyTorch tensors
    duration_tensor = torch.tensor(duration, dtype=torch.float32).to(device)
    event_tensor = torch.tensor(event, dtype=torch.float32).to(device)

    return X_tensor, y_tensor, duration_tensor, event_tensor, X, df

def train_model(X_tensor: torch.tensor,
                duration_tensor: torch.tensor,
                event_tensor: torch.tensor,
                num_epochs: int = 50,
                batch_size: int = 64) -> torch.nn.Module:
    """
    Train the Cox-NAM model on the provided data.

    Parameters:
    - X_tensor: PyTorch tensor of data
    - duration_tensor: PyTorch tensor of duration feature
    - event_tensor: PyTorch tensor of event feature
    - num_epochs: Number of epochs to train the model
    - batch_size: Batch size for training

    Returns:
    - coxnam_model: Trained Cox-NAM
    """
    # Define the Cox-NAM model
    num_features = X_tensor.shape[1]
    input_dim = 1
    hidden_units = [32,16]  # Define the hidden units for each feature network
    coxnam_model = CoxNAM(num_features, input_dim, hidden_units,dropout_rate=0.2).to(device)

    # Define optimizer
    optimizer = optim.Adam(coxnam_model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        coxnam_model.train()
        permutation = torch.randperm(X_tensor.size()[0])
        for i in range(0, X_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices]
            batch_duration = duration_tensor[indices]
            batch_event = event_tensor[indices]

            optimizer.zero_grad()
            risk_scores = coxnam_model(batch_x)
            loss = cox_loss(risk_scores, batch_duration, batch_event, model=coxnam_model,l1_lambda=0.01, l2_lambda=0.01)

            # Print loss to debug
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {i//batch_size}")
            
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Training complete
    print("Training complete!")
    return coxnam_model

def evaluate_model(coxnam_model: torch.nn.Module,
                   X_tensor: torch.tensor,
                   duration: np.ndarray,
                   event: np.ndarray) -> float:
    """ Evaluate the Cox-NAM model using the C-index.

    Parameters:
    - coxnam_model: Trained Cox-NAM model
    - X_tensor: PyTorch tensor of data
    - duration: Pandas Series of duration feature
    - event: Pandas Series of event feature
    Returns:
    - c_index: C-index of the model
    """
    # Evaluate the model
    coxnam_model.eval()
    with torch.no_grad():
        risk_scores_test = coxnam_model(X_tensor).cpu().numpy().flatten()  # Flatten to ensure 1D array

    # Calculate C-index using lifelines
    c_index = concordance_index(duration, -risk_scores_test, event)
    print(f"C-index: {c_index:.4f}")
    return c_index

def plot_shape_functions_and_distributions(model: torch.nn.Module,
                                           X: pd.DataFrame,
                                           feature_names: list) -> None:
    """
    Plot the learned shape functions and feature distributions for a CoxNAM model.
    Parameters:
    - model: Trained CoxNAM model
    - X: DataFrame of features
    - feature_names: List of feature names
    Returns:
    - None
    """
    num_features = X.shape[1]
    
    # Determine grid size for plotting
    ncols = min(4, int(np.ceil(np.sqrt(num_features))))
    nrows = int(np.ceil(num_features / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()
    
    for i in range(num_features):
        ax = axes[i]
        feature_network = model.feature_networks[i]
        feature_network.eval()
        
        feature_values = X[:, i]
        feature_mean = np.mean(feature_values)
        feature_std = np.std(feature_values)
        
        # Create a range of sample inputs in the standardized space
        sample_inputs = np.linspace(-3, 3, 100).reshape(-1, 1)
        with torch.no_grad():
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32).to(device)
            shape_values = feature_network(sample_inputs_tensor).cpu().numpy()
        
        # Convert sample inputs back to the original feature scale
        orig_sample_inputs = sample_inputs * feature_std + feature_mean
        
        # Plot histogram of the feature values
        ax.hist(feature_values, bins=30, alpha=0.7, color='b', density=True)
        
        # Plot the learned shape function on a secondary y-axis
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
    plt.savefig('rossi_shape_functions.png', dpi=300)


def local_explanation(coxnam_model: torch.nn.Module,
                      x_instance: torch.tensor) -> tuple:
    """
    Compute the local explanation for a single instance using a trained CoxNAM model.
    Parameters:
    - coxnam_model: Trained CoxNAM model
    - x_instance: 1D tensor of shape (num_features,)
    
    Returns:
    - contributions: List of feature contributions
    - total_risk: Total risk score for the instance
    """
    # Ensure model is in evaluation mode
    coxnam_model.eval()
    
    contributions = []
    
    # Loop over each feature network
    for i, feature_network in enumerate(coxnam_model.feature_networks):
        # Pass the single feature through its network
        # Reshape x_i to (1,1) if your feature network expects a 2D input
        x_i = x_instance[i].view(1, 1)
        
        with torch.no_grad():
            feature_contrib = feature_network(x_i).item()
        
        contributions.append(feature_contrib)
    
    # The overall predicted log-hazard is the sum of all contributions
    total_risk = sum(contributions)
    
    return contributions, total_risk


def plot_local_explanation_bar(coxnam_model: torch.nn.Module, 
                               x_instance: torch.tensor, 
                               feature_names: list,
                               sort_features: bool = False,
                               title: str ="Local Explanation: Feature Contributions"):
    """
    Plots the local (instance-level) feature contributions for a CoxNAM model
    as a horizontal bar chart.

    Parameters:
    coxnam_model : A trained CoxNAM model that has a list of feature_networks.
    x_instance : A 1D tensor of shape (num_features,) containing the feature values for 
        the specific instance you want to explain.
    feature_names : A list of feature names corresponding to the order of features in x_instance.
    sort_features : Whether to sort features by their contribution (descending). Default is False.
    title: Title of the plot.

    Returns:
    None
    """

    coxnam_model.eval()  # Set model to evaluation mode
    
    # 1. Compute the contribution of each feature network for the single instance
    contributions = []
    with torch.no_grad():
        for i, feature_network in enumerate(coxnam_model.feature_networks):
            # Reshape single feature value (float) to a 2D tensor (1,1)
            x_i = x_instance[i].view(1, 1)
            feature_contrib = feature_network(x_i).item()
            contributions.append(feature_contrib)
    
    # 2. (Optional) Sort the features by contribution for a more organized plot
    if sort_features:
        # Sort by absolute contribution in descending order (largest impact first)
        sorted_indices = np.argsort(np.abs(contributions))[::-1]
        sorted_contributions = [contributions[idx] for idx in sorted_indices]
        sorted_features = [feature_names[idx] for idx in sorted_indices]
    else:
        sorted_contributions = contributions
        sorted_features = feature_names
    
    # 3. Plot the horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    y_positions = np.arange(len(sorted_features))
    
    ax.barh(y_positions, sorted_contributions, align='center', color='skyblue')
    ax.axvline(x=0, color='black', linewidth=1)  # Vertical line at x=0 for reference
    
    # Set labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Contribution to Log-Hazard (Risk)")
    ax.set_ylabel("Features")
    ax.set_title(title)
    
    # By default, matplotlib puts the first element at the bottom, 
    # invert so top bar corresponds to the first element in the list
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('rossi_local_explanation.png', dpi=300)

def main():
    """
    Main function to load data, train the model, evaluate the model, and plot the shape functions
    """
    X_tensor, y_tensor, duration_tensor, event_tensor, X, df = load_and_prepare_data()
    coxnam_model = train_model(X_tensor, duration_tensor, event_tensor)
    evaluate_model(coxnam_model, X_tensor, df['week'].values, df['arrest'].values)
    feature_names = X.columns.tolist()
    plot_shape_functions_and_distributions(coxnam_model, X.to_numpy(), feature_names)


    # Example usage:
    i = 14  # Some index in your dataset
    single_x = X_tensor[i]  # shape: (num_features,)
    contribs, total_risk_score = local_explanation(coxnam_model, single_x)
    print("Feature contributions:", contribs)
    print("Total risk score:", total_risk_score)
    # print probability of recividsm based on risk score
    print("Probability of recidivism:", 1 - np.exp(-np.exp(total_risk_score)))
    print("True event status:", event_tensor[i].item())
    print("True duration:", duration_tensor[i].item())
    
    # Print single_x values for each feature unnormalized and names
    print("Feature names and values:")
    for name, value in zip(feature_names, X.iloc[i].values):
        print(f"{name}: {value}")
    

    plot_local_explanation_bar(coxnam_model, single_x, feature_names, sort_features=True)


if __name__ == "__main__":
    main()
