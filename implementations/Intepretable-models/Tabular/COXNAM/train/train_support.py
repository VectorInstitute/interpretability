import os
import yaml
from tqdm import tqdm

import torch
import ultraimport
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

coxnam = ultraimport.create_ns_package("coxnam", "__dir__/../coxnam")
from coxnam.metrics import compute_td_auc  # noqa: E402
from coxnam.model import CoxNAM  # noqa: E402
from coxnam.loss import cox_loss  # noqa: E402
from coxnam.metrics import compute_td_concordance_index  # noqa: E402, F811
from coxnam.plots import plot_baseline_survival  # noqa: E402
from coxnam.surv_utils import (  # noqa: E402
    compute_baseline_survival,
)


def load_and_prepare_data(
    file_path: str, test_size: int, device: torch.device
) -> tuple:
    """Load and prepare the dataset for training CoxNAM model.
    Parameters:
        file_path (str): Path to the dataset CSV file.
        test_size (int): Test split size.
        device (torch.device): Device to use for tensor operations.

    Returns:
        tuple: A tuple containing training and testing tensors for features and target variables,
               the processed feature dataframe, and the raw dataframe.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    print("Dataset head:")
    print(df.head())
    print(f"Number of rows with missing values: {df.isnull().any(axis=1).sum()}")

    # Target variables
    duration = df["d.time"].values.astype(float)
    event = df["death"].values.astype(float)

    # Drop target columns from features
    X = df.drop(columns=["d.time", "death"])

    # Identify categorical & numerical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Impute missing values in categorical columns BEFORE encoding
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    # Impute missing values in specific numerical columns with provided values
    # According to the HBiostat Repository (https://hbiostat.org/data/repo/supportdesc, Professor Frank Harrell)
    # the following default values have been found to be useful in imputing missing baseline physiologic data:
    impute_values = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9,  # in thousands
        "urine": 2502,
    }
    for col, value in impute_values.items():
        if col in numerical_cols:
            X.loc[:, col] = X[col].fillna(value)

    # One-hot encoding for categorical variables
    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    X_categorical = encoder.fit_transform(X[categorical_cols])
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
    X_categorical = pd.DataFrame(
        X_categorical, columns=categorical_feature_names, index=X.index
    )

    # Impute missing values in numerical columns BEFORE scaling
    num_imputer = SimpleImputer(strategy="median")
    X_numerical_imputed = pd.DataFrame(
        num_imputer.fit_transform(X[numerical_cols]),
        columns=numerical_cols,
        index=X.index,
    )

    # Combine numerical and categorical features
    X_processed = pd.concat([X_numerical_imputed, X_categorical], axis=1)

    # Drop features due to multicollinearity
    X_processed = X_processed.drop(
        columns=["hospdead", "dzgroup_Coma", "surv6m", "surv2m", "dzclass_Coma"]
    )

    # Train-test split
    (
        X_train,
        X_test,
        duration_train,
        duration_test,
        event_train,
        event_test,
    ) = train_test_split(
        X_processed,
        duration,
        event,
        test_size=test_size,
        random_state=42,
        stratify=event,
    )

    # Ensure numerical_cols only includes columns that still exist
    numerical_cols = [col for col in numerical_cols if col in X_train.columns]

    # Scale numerical features AFTER imputation
    scaler = StandardScaler()
    X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    duration_train_tensor = torch.tensor(duration_train, dtype=torch.float32).to(device)
    duration_test_tensor = torch.tensor(duration_test, dtype=torch.float32).to(device)
    event_train_tensor = torch.tensor(event_train, dtype=torch.float32).to(device)
    event_test_tensor = torch.tensor(event_test, dtype=torch.float32).to(device)

    return (
        X_train_tensor,
        X_test_tensor,
        duration_train_tensor,
        duration_test_tensor,
        event_train_tensor,
        event_test_tensor,
        X_processed,
        df,
    )


def train_model(
    X_tensor: torch.tensor,
    duration_tensor: torch.tensor,
    event_tensor: torch.tensor,
    hidden_units: list,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float,
    l1_lambda: float,
    l2_lambda: float,
    device: torch.device,
    model_save_path: str,
    patience: int = 10,
) -> torch.nn.Module:
    """Train the CoxNAM model.
    Parameters:
        X_tensor: Training features tensor.
        duration_tensor: Training survival times.
        event_tensor: Training event indicators.
        hidden_units: List of hidden units for each layer of the model.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for optimizer.
        dropout_rate: Dropout rate for model.
        l1_lambda: L1 regularization lambda.
        l2_lambda: L2 regularization lambda.
        device: Torch device.
        model_save_path: Path to save the trained model.
        patience: Early stopping patience.
    Returns:
        CoxNAM: Trained CoxNAM model.
    """
    num_samples = X_tensor.shape[0]
    num_features = X_tensor.shape[1]
    input_dim = 1  # Each feature is a scalar

    # Instantiate the CoxNAM model
    coxnam_model = CoxNAM(
        num_features, input_dim, hidden_units, dropout_rate=dropout_rate
    ).to(device)
    optimizer = optim.Adam(coxnam_model.parameters(), lr=learning_rate)

    best_td_auc = -np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        coxnam_model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0

        # Process mini-batches with a tqdm progress bar
        for i in tqdm(
            range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}", leave=False
        ):
            mini_batch_indices = permutation[i : i + batch_size].to(device)
            optimizer.zero_grad()
            risk_scores_full = coxnam_model(X_tensor)
            loss = cox_loss(
                risk_scores_full,
                duration_tensor,
                event_tensor,
                coxnam_model,
                mini_batch_indices=mini_batch_indices,
                l1_lambda=l1_lambda,
                l2_lambda=l2_lambda,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss at batch {i}. Skipping update.")
                continue

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(mini_batch_indices)

        avg_loss = epoch_loss / num_samples
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

        # Compute time-dependent AUC using training data (or a validation split if available)
        td_auc_score = compute_td_auc(
            coxnam_model,
            X_tensor,
            X_tensor,
            duration_tensor,
            duration_tensor,
            event_tensor,
            event_tensor,
            epoch=epoch,
            plot=False,
        )
        print(f"Epoch [{epoch+1}/{num_epochs}] - TD-AUC: {td_auc_score:.4f}")

        if td_auc_score > best_td_auc:
            best_td_auc = td_auc_score
            patience_counter = 0
            torch.save(coxnam_model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} with TD-AUC: {td_auc_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining completed. Best TD-AUC: {best_td_auc:.4f}")
    # Load and return the best model
    coxnam_model.load_state_dict(torch.load(model_save_path, map_location=device))
    return coxnam_model


def evaluate_model(
    coxnam_model: torch.nn.Module,
    X_test_tensor: torch.tensor,
    duration_test: torch.tensor,
    event_test: torch.tensor,
    device: torch.device,
    model_save_path: str,
):
    """Evaluate the CoxNAM model on the test set.
    Parameters:
        coxnam_model: Trained CoxNAM model.
        X_test_tensor: Test features tensor.
        duration_test: Test survival times.
        event_test: Test event indicators.
        device: Torch device.
        model_save_path: Path to saved model.
    Returns:
        float: Test C-index.
    """
    if os.path.exists(model_save_path):
        print("Loading saved model for evaluation...")
        coxnam_model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("No saved model found. Using current model in memory.")
    coxnam_model = coxnam_model.to(device)
    coxnam_model.eval()

    with torch.no_grad():
        risk_scores_test = coxnam_model(X_test_tensor).cpu().numpy().flatten()

    if isinstance(duration_test, torch.Tensor):
        duration_test = duration_test.cpu().numpy()
    if isinstance(event_test, torch.Tensor):
        event_test = event_test.cpu().numpy()

    c_index = concordance_index(duration_test, -risk_scores_test, event_test)
    print(f"📊 Test C-index: {c_index:.4f}")
    return c_index


def plot_shape_functions_and_distributions(
    model: torch.nn.Module,
    X: np.ndarray,
    feature_names: list,
    device: torch.device,
    output_plot: str,
):
    """
    Plot shape functions and feature distributions for interpretability.
    Parameters:
        model: Trained CoxNAM model.
        X: NumPy array of features.
        feature_names: List of feature names.
        device: Torch device.
        output_plot: Path to save the plot.
    """
    num_features = X.shape[1]
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

        sample_inputs = np.linspace(-3, 3, 100).reshape(-1, 1)
        with torch.no_grad():
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32).to(
                device
            )
            shape_values = feature_network(sample_inputs_tensor).cpu().numpy()

        orig_sample_inputs = sample_inputs * feature_std + feature_mean

        ax.hist(feature_values, bins=30, alpha=0.7, color="b", density=True)
        ax2 = ax.twinx()
        ax2.plot(orig_sample_inputs, shape_values, color="r")

        ax.set_xlabel("Feature Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax2.set_ylabel("Shape Function", fontsize=8)
        ax.set_title(feature_names[i], fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True)

    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color="b", lw=3, label="Distribution"),
        plt.Line2D([0], [0], color="r", lw=3, label="Shape Function"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to {output_plot}")


def main():
    """
    Main function to train and evaluate the CoxNAM model.
    """

    # Get coxnam model config
    yaml_file = os.path.join(os.path.dirname(__file__), "coxnam.yaml")
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(config["seed"])

    # Determine device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load and prepare the data
    data_file = os.path.join(config["data_dir"], config["support_file"])
    (
        X_train_tensor,
        X_test_tensor,
        duration_train_tensor,
        duration_test_tensor,
        event_train_tensor,
        event_test_tensor,
        X_imputed,
        df_raw,
    ) = load_and_prepare_data(
        file_path=data_file, test_size=config["test_size"], device=device
    )

    if config["pretrained"] and os.path.exists(config["output_file"]):
        print("Bypassing training and loading pretrained model.")
        coxnam_model = CoxNAM(
            X_train_tensor.shape[1],
            input_dim=1,
            hidden_units=config["train_params"]["hidden_units"],
            dropout_rate=config["train_params"]["dropout_rate"],
        ).to(device)
        coxnam_model.load_state_dict(
            torch.load(config["output_file"], map_location=device)
        )
    else:
        print("Training model from scratch...")
        coxnam_model = train_model(
            X_train_tensor,
            duration_train_tensor,
            event_train_tensor,
            hidden_units=config["train_params"]["hidden_units"],
            num_epochs=config["train_params"]["num_epochs"],
            batch_size=config["train_params"]["batch_size"],
            learning_rate=config["optimizer"]["lr"],
            dropout_rate=config["train_params"]["dropout_rate"],
            l1_lambda=config["train_params"]["l1_lambda"],
            l2_lambda=config["train_params"]["l2_lambda"],
            device=device,
            model_save_path=config["output_file"],
            patience=config["train_params"][
                "patience"
            ],  # You can adjust patience here if needed
        )

    # Evaluate the model
    evaluate_model(
        coxnam_model,
        X_test_tensor,
        duration_test_tensor,
        event_test_tensor,
        device=device,
        model_save_path=config["output_file"],
    )

    # Compute baseline survival function using training data
    time_grid, H0, S0 = compute_baseline_survival(
        coxnam_model, X_train_tensor, duration_train_tensor, event_train_tensor
    )
    print("Baseline survival function computed.")

    td_ci = compute_td_concordance_index(
        coxnam_model,
        X_test_tensor,
        duration_test_tensor,
        event_test_tensor,
        time_point=1024,
        time_grid=time_grid,
        H0=H0,
    )
    print(f"📊 Time-dependent C-index at time 1024 days: {td_ci:.4f}")

    # Plot shape functions if requested
    if config["plot"]:
        output_plot = f"""shape_functions_hidden{config['train_params']['hidden_units']}
        _lr{config['optimizer']['lr']}_dropout{config['train_params']['dropout_rate']}.png"""
        feature_names = X_imputed.columns.tolist()
        plot_shape_functions_and_distributions(
            coxnam_model,
            X_imputed.to_numpy(),
            feature_names,
            device=device,
            output_plot=output_plot,
        )
        plot_baseline_survival(time_grid, S0)


if __name__ == "__main__":
    main()
