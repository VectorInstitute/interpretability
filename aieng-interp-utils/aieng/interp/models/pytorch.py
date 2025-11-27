"""PyTorch model training utilities."""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch computations.

    Returns
    -------
    torch.device
        CUDA device if available, MPS if on Apple Silicon, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def random_seed(seed_value: int, use_cuda: bool = False) -> None:
    """
    Set random seed for reproducibility across numpy, torch, and Python's random module.

    Parameters
    ----------
    seed_value : int
        The seed value to set.
    use_cuda : bool, default=False
        Whether CUDA is being used (enables additional CUDA-specific seeding).
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def initialize_weights(m: nn.Module) -> None:
    """
    Initialize weights of linear layers using Xavier (Glorot) initialization.

    Parameters
    ----------
    m : nn.Module
        The module to initialize.
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0
) -> None:
    """
    Fill tensor with values from a truncated normal distribution.

    Values are drawn from a normal distribution and truncated to [-2, 2] standard deviations.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to fill with truncated normal values (modified in-place).
    mean : float, default=0.0
        Mean of the normal distribution.
    std : float, default=1.0
        Standard deviation of the normal distribution.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    print_every: int = 10,
) -> None:
    """
    Train a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.
    train_loader : DataLoader
        DataLoader containing training data.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model weights.
    num_epochs : int
        Number of training epochs.
    device : torch.device
        Device to train on (CPU, CUDA, or MPS).
    print_every : int, default=10
        Print training loss every N epochs.
    """
    model.apply(initialize_weights)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % print_every == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def train_or_load_model(
    model: nn.Module,
    model_path: str,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    print_every: int = 10,
) -> nn.Module:
    """
    Train a model or load from checkpoint if it exists.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train or load.
    model_path : str
        Path to save/load the model checkpoint.
    train_loader : DataLoader
        DataLoader containing training data.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model weights.
    num_epochs : int
        Number of training epochs.
    device : torch.device
        Device to train on (CPU, CUDA, or MPS).
    print_every : int, default=10
        Print training loss every N epochs.

    Returns
    -------
    nn.Module
        The trained or loaded model.
    """
    if os.path.isfile(model_path):
        print(f"Loading pre-trained model from: {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)
    else:
        print("Training new model...")
        train_pytorch_model(
            model, train_loader, criterion, optimizer, num_epochs, device, print_every
        )
        print(f"Finished training. Saving model to: {model_path}")
        dir_path = os.path.dirname(model_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(model.state_dict(), model_path)

    return model


def calculate_n_units(
    data: np.ndarray, n_basis_functions: int, units_multiplier: int
) -> list[int]:
    """
    Calculate the number of units for each feature based on unique values.

    Used in Neural Additive Models (NAM) and similar architectures to determine
    network width based on feature cardinality.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_samples, n_features).
    n_basis_functions : int
        Maximum number of basis functions per feature.
    units_multiplier : int
        Multiplier for the number of unique values.

    Returns
    -------
    list of int
        Number of units for each feature.
    """
    num_unique_vals = [len(np.unique(data[:, i])) for i in range(data.shape[1])]
    return [min(n_basis_functions, i * units_multiplier) for i in num_unique_vals]


def get_full_data(data_loader: DataLoader) -> torch.Tensor:
    """
    Extract all data from a DataLoader and stack into a single tensor.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader containing the data.

    Returns
    -------
    torch.Tensor
        All data stacked into a single tensor.
    """
    all_data = torch.stack([x for batch in data_loader for x in batch[0]])
    return all_data
