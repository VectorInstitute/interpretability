import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import argparse
import os
from tqdm import tqdm

from model import CoxNAM
from utils.loss import cox_loss  
from utils.td_auc import compute_td_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Train or load CoxNAM model with configurable settings.")
    parser.add_argument("--pretrained", action="store_true", help="Bypass training and load pretrained model if available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--file_path", type=str, default="../datasets/support2.csv", help="Path to dataset CSV file")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--hidden_units", type=int, nargs='+', default=[64, 32, 16],
                        help="Hidden units for each layer of the CoxNAM model")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for model")
    parser.add_argument("--l1_lambda", type=float, default=0.02, help="L1 regularization lambda")
    parser.add_argument("--l2_lambda", type=float, default=0.01, help="L2 regularization lambda")
    parser.add_argument("--model_save_path", type=str, default="coxnam_model_epoch.pth", help="Path to save/load model")
    parser.add_argument("--plot", type=bool, default=True, help="Generate plots for shape functions (default: True)")
    return parser.parse_args()


def load_and_prepare_data(file_path, test_size, device):
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

    # One-hot encoding for categorical variables
    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    X_categorical = encoder.fit_transform(X[categorical_cols])
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
    X_categorical = pd.DataFrame(X_categorical, columns=categorical_feature_names, index=X.index)

    # Impute missing values in numerical columns BEFORE scaling
    num_imputer = SimpleImputer(strategy="median")
    X_numerical_imputed = pd.DataFrame(num_imputer.fit_transform(X[numerical_cols]), 
                                       columns=numerical_cols, index=X.index)

    # Combine numerical and categorical features
    X_processed = pd.concat([X_numerical_imputed, X_categorical], axis=1)
    
    # Drop features due to multicollinearity
    X_processed = X_processed.drop(columns=['hospdead', 'dzgroup_Coma', 'surv6m', 'surv2m', 'dzclass_Coma'])

    # Train-test split
    X_train, X_test, duration_train, duration_test, event_train, event_test = train_test_split(
        X_processed, duration, event, test_size=test_size, random_state=42, stratify=event
    )

    # Ensure numerical_cols only includes columns that still exist
    numerical_cols = [col for col in numerical_cols if col in X_train.columns]

    # Scale numerical features AFTER imputation
    scaler = StandardScaler()
    X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_test_tensor  = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    duration_train_tensor = torch.tensor(duration_train, dtype=torch.float32).to(device)
    duration_test_tensor  = torch.tensor(duration_test, dtype=torch.float32).to(device)
    event_train_tensor    = torch.tensor(event_train, dtype=torch.float32).to(device)
    event_test_tensor     = torch.tensor(event_test, dtype=torch.float32).to(device)

    return (X_train_tensor, X_test_tensor, duration_train_tensor, duration_test_tensor,
            event_train_tensor, event_test_tensor, X_processed, df)


def train_model(X_tensor, duration_tensor, event_tensor, hidden_units, num_epochs, batch_size,
                learning_rate, dropout_rate, l1_lambda, l2_lambda, device, model_save_path, patience=10):
    num_samples = X_tensor.shape[0]
    num_features = X_tensor.shape[1]
    input_dim = 1  # Each feature is a scalar

    # Instantiate the CoxNAM model
    coxnam_model = CoxNAM(num_features, input_dim, hidden_units, dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(coxnam_model.parameters(), lr=learning_rate)

    best_td_auc = -np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        coxnam_model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0

        # Process mini-batches with a tqdm progress bar
        for i in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}", leave=False):
            mini_batch_indices = permutation[i:i + batch_size].to(device)
            optimizer.zero_grad()
            risk_scores_full = coxnam_model(X_tensor)
            loss = cox_loss(risk_scores_full, duration_tensor, event_tensor, coxnam_model,
                            mini_batch_indices=mini_batch_indices, l1_lambda=l1_lambda, l2_lambda=l2_lambda)
            
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
            coxnam_model, X_tensor, X_tensor, duration_tensor,
            duration_tensor, event_tensor, event_tensor, epoch=epoch, plot=False)
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


def evaluate_model(coxnam_model, X_test_tensor, duration_test, event_test, device, model_save_path):
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


def plot_shape_functions_and_distributions(model, X, feature_names, device, output_plot):
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
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32).to(device)
            shape_values = feature_network(sample_inputs_tensor).cpu().numpy()

        orig_sample_inputs = sample_inputs * feature_std + feature_mean

        ax.hist(feature_values, bins=30, alpha=0.7, color='b', density=True)
        ax2 = ax.twinx()
        ax2.plot(orig_sample_inputs, shape_values, color='r')

        ax.set_xlabel('Feature Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax2.set_ylabel('Shape Function', fontsize=8)
        ax.set_title(feature_names[i], fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True)

    for j in range(num_features, len(axes)):
        axes[j].axis('off')

    handles = [plt.Line2D([0], [0], color='b', lw=3, label='Distribution'),
               plt.Line2D([0], [0], color='r', lw=3, label='Shape Function')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to {output_plot}")


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() 
                            else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load and prepare the data
    (X_train_tensor, X_test_tensor, duration_train_tensor, duration_test_tensor,
     event_train_tensor, event_test_tensor, X_imputed, df_raw) = load_and_prepare_data(
        file_path=args.file_path, test_size=args.test_size, device=device)

    if args.pretrained and os.path.exists(args.model_save_path):
        print("Bypassing training and loading pretrained model.")
        coxnam_model = CoxNAM(X_train_tensor.shape[1], input_dim=1,
                              hidden_units=args.hidden_units,
                              dropout_rate=args.dropout_rate).to(device)
        coxnam_model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    else:
        print("Training model from scratch...")
        coxnam_model = train_model(
            X_train_tensor, duration_train_tensor, event_train_tensor,
            hidden_units=args.hidden_units,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            l1_lambda=args.l1_lambda,
            l2_lambda=args.l2_lambda,
            device=device,
            model_save_path=args.model_save_path,
            patience=10  # You can adjust patience here if needed
        )

    # Evaluate the model
    evaluate_model(coxnam_model, X_test_tensor, duration_test_tensor, event_test_tensor,
                   device=device, model_save_path=args.model_save_path)

    # Plot shape functions if requested
    if args.plot:
        output_plot = f"shape_functions_hidden{args.hidden_units}_lr{args.learning_rate}_dropout{args.dropout_rate}.png"
        feature_names = X_imputed.columns.tolist()
        plot_shape_functions_and_distributions(coxnam_model, X_imputed.to_numpy(),
                                               feature_names, device=device, output_plot=output_plot)


if __name__ == "__main__":
    main()
