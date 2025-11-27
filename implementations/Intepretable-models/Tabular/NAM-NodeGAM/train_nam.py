import os
import yaml
from typing import Tuple

import torch
import ultraimport
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader, TensorDataset

nam = ultraimport.create_ns_package("nam", "__dir__/nam")

from nam.model import NeuralAdditiveModel, ExULayer, ReLULayer  # noqa: E402
from nam.data_utils import *  # noqa: E402, F403
from nam.utils import *  # noqa: E402, F403

# Some parts were extracted from: https://github.com/kherud/neural-additive-models-pt/tree/master


def get_data_loaders(X: torch.tensor, y: torch.tensor, batch: int) -> Tuple[DataLoader]:
    """Get train, test and validation data loaders
    Parameters:
        X: Dataset of features.
        y: labels.
        batch: batch size.

    Returns:
        train_dl: DataLoader for training data.
        test_dl: DataLoader for testing data.
        val_dl: DataLoader for validation data.
    """
    train_test_split = split_dataset(X, y, n_splits=1, stratified=True)  # noqa: F405

    (x_train, y_train), (x_test, y_test) = next(train_test_split)

    # Get train val split
    train_val_split = split_dataset(x_train, y_train, n_splits=1, stratified=True)  # noqa: F405

    (x_train, y_train), (x_val, y_val) = next(train_val_split)

    train_dl = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch, shuffle=True
    )
    test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size=batch, shuffle=True)
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=batch, shuffle=True)

    return train_dl, test_dl, val_dl


def get_predictions(
    model: torch.nn.Module, X: DataLoader, device: torch.device
) -> Tuple:
    """Get predictions from model
    Parameters:
        - model: Model to get predictions from.
        - X: DataLoader with data.

    Returns:
        x_true: True labels.
        x_probs: Predicted probabilities.
        x_preds: Predicted labels.
    """

    model.eval()
    x_true, x_probs = [], []
    with torch.set_grad_enabled(False):
        for _, (features, labels) in enumerate(X):
            features, labels = features.to(device), labels.to(device)
            logits, _ = model.forward(features)
            prob = torch.sigmoid(logits)

            x_true.extend(labels.tolist())
            x_probs.extend(prob.tolist())

    threshold = 0.5
    x_preds = list(map(lambda x: x >= threshold, x_probs))

    return x_true, x_probs, x_preds


def train_and_predict(
    model: torch.nn.Module,
    cfgs: dict,
    train: DataLoader,
    test: DataLoader,
    val: DataLoader,
    device: torch.device,
) -> dict:
    """Train and predict using NAM model.
    Parameters:
        model: NAM model.
        cfgs: Configuration for training.
        train: DataLoader for training data.
        test: DataLoader for testing data.
        val: DataLoader for validation data.
        device: Device to use for training.

    Returns:
        Dictionary with scores.
    """
    y_tr_true, y_tr_preds = [], []
    y_te_true, y_te_probs, y_tr_preds = [], [], []
    y_val_true, y_val_probs, y_val_preds = [], [], []
    auc, f1, precision, recall = [], [], [], []

    classifier_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, cfgs["scheduler"]["step_size"], cfgs["scheduler"]["step_size"]
    )
    optimizer.zero_grad()

    for epoch in range(cfgs["train_params"]["epochs"]):
        model.train()
        print("Training on epoch: ", epoch)
        for _, (features, labels) in enumerate(train):
            features, labels = features.to(device), labels.to(device)
            logits, _ = model.forward(features)
            prob = torch.sigmoid(logits)
            loss = classifier_criterion(logits, labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            y_tr_true.extend(labels.tolist())
            y_tr_preds.extend(prob.tolist())

        scheduler.step()
        y_val_true, y_val_probs, y_val_preds = get_predictions(model, val, device)
        y_te_true, y_te_probs, y_te_preds = get_predictions(model, test, device)

        val_auc = roc_auc_score(y_val_true, y_val_probs)
        train_auc = roc_auc_score(y_tr_true, y_tr_preds)

        print(
            f"\n AUC scores for epoch {epoch}: \
              Training AUC: {train_auc}, Validation AUC: {val_auc}"
        )

        auc.append(roc_auc_score(y_te_true, y_te_probs))
        f1.append(f1_score(y_te_true, y_te_preds))
        precision.append(precision_score(y_te_true, y_te_preds))
        recall.append(recall_score(y_te_true, y_te_preds))

    return {
        "f1": (np.mean(f1), np.std(f1)),
        "auc": (np.mean(auc), np.std(auc)),
        "precision": (np.mean(precision), np.std(precision)),
        "recall": (np.mean(recall), np.std(recall)),
    }


def main():
    """Main function to train NAM model on US_130 dataset."""
    # Get nam model config
    yaml_file = os.path.join(os.path.dirname(__file__), "nam.yaml")
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    data_file = os.path.join(config["data_dir"], config["data_file"])
    data = pd.read_csv(data_file)
    df = process_us_130_csv(data)  # noqa: F405

    # Separate training data and labels
    X, y = df.drop("readmitted_binarized", axis=1), df["readmitted_binarized"]
    X, y = (
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32),
    )

    # Get train test val split
    train_dl, test_dl, val_dl = get_data_loaders(
        X, y, config["train_params"]["batch_size"]
    )

    print("------------------------------")
    print("Training NAM on US_130 dataset")
    print("------------------------------")

    # Setup
    device = get_device()  # noqa: F405
    print(f"Using device: {device}")
    random_seed(42, True)  # noqa: F405

    shallow_units = calculate_n_units(  # noqa: F405
        get_full_data(train_dl),  # noqa: F405
        config["nam"]["n_basis_functions"],
        config["nam"]["units_multiplier"],
    )

    shallow_layer = ExULayer if config["nam"]["shallow_layer"] == "exu" else ReLULayer
    hidden_layer = ExULayer if config["nam"]["hidden_layer"] == "exu" else ReLULayer

    model = NeuralAdditiveModel(
        input_size=X.shape[-1],
        shallow_units=shallow_units,
        shallow_layer=shallow_layer,
        hidden_layer=hidden_layer,
        hidden_dropout=config["nam"]["dropout"],
        feature_dropout=config["nam"]["feature_dropout"],
    )
    model = model.to(device)
    nam_scores = train_and_predict(model, config, train_dl, test_dl, val_dl, device)

    print("\nScores of NAM model:")
    print(
        f"F1: Mean: {nam_scores['f1'][0]} \t  \
          Std. Deviation: {nam_scores['f1'][1]}"
    )
    print(
        f"AUC: Mean: {nam_scores['auc'][0]} \t \
          Std. Deviation: {nam_scores['auc'][1]}"
    )
    print(
        f"Precision: Mean: {nam_scores['precision'][0]} \t \
          Std. Deviation: {nam_scores['precision'][1]}"
    )
    print(
        f"Recall: Mean: {nam_scores['recall'][0]} \t \
          Std. Deviation: {nam_scores['recall'][1]}"
    )


if __name__ == "__main__":
    main()
