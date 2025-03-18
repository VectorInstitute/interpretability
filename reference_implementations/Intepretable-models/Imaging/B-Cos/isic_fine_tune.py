#!/usr/bin/env python
import os
import yaml
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from bcos.isic_data import ISICDataset
from bcos.minimal_bcos_resnet import resnet50, BcosConv2d

# -----------------------------------------------------------------------------
# 1. Load the Pretrained B‑cos ResNet-50 Model (with 6 Channels)
# -----------------------------------------------------------------------------

def get_model(num_classes: int = 2,
              freeze_features: bool = False) -> nn.Module:
    """
    Loads a pretrained B‑cos ResNet-50 model (6-channel input) and modifies it for fine-tuning.
    Fixes state_dict mismatch by removing the fc.weight from pretrained weights.
    
    Parameters:
    - num_classes (int): Number of output classes.
    - freeze_features (bool): If True, freezes the feature extractor layers.

    Returns:
    - model (nn.Module): The modified B‑cos ResNet-50 model.
    """

    # Load model (original has num_classes=1000)
    model = resnet50(pretrained=True, num_classes=1000,
                     in_chans=6, long_version=True)

    # Freeze feature extractor layers if specified
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Load pretrained state dictionary
    state_dict = model.state_dict()

    # REMOVE the final `fc.weight` from the state_dict to avoid shape mismatch
    if "fc.weight" in state_dict:
        del state_dict["fc.weight"]

    # ✅ Use BcosConv2d (same as original)
    in_channels = model.fc.in_channels
    model.fc = BcosConv2d(in_channels, num_classes,
                          kernel_size=1, bias=False)  

    # Load modified state_dict (now without fc.weight)
    model.load_state_dict(state_dict, strict=False)

    return model

def get_optimizer(model: nn.Module,
                  learning_rate: float,
                  freeze_features: bool = False) -> optim.Optimizer:
    """
    Creates an optimizer with different learning rates for the feature extractor and the final layer.
    
    Parameters:
    - model (nn.Module): The model to optimize.
    - learning_rate (float): The base learning rate.
    - freeze_features (bool): If True, freezes the feature extractor layers.

    Returns:
    - optimizer (optim.Optimizer): The optimizer.
    """

    if freeze_features:
        return optim.Adam(model.parameters(), lr=learning_rate)
    
    # Apply different learning rates
    params = [
        {"params": model.fc.parameters(), "lr": learning_rate * 3},
        {"params": [param for name, param in model.named_parameters() if "fc" not in name], "lr": learning_rate}
    ]
    return optim.Adam(params)

# -----------------------------------------------------------------------------
# 2. Create Data Loaders
# -----------------------------------------------------------------------------
def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Returns the training and test data loaders.
    
    Args:
        cfg: Configuration dictionary containing dataset paths and parameters.
    
    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    """

    # Define the dataset directories and CSV files.
    train_root, test_root = cfg['train_dir'], cfg['test_dir']
    train_csv = os.path.join(train_root, cfg['train_csv'])
    test_csv = os.path.join(test_root, cfg['test_csv'])
    
    # Define data transforms.
    input_size, batch_size = cfg['input_size'], cfg['train_params']['batch_size']
    num_workers = cfg['train_params']['num_workers']
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset instances.
    train_dataset = ISICDataset(root_dir=train_root, csv_file=train_csv, transform=transform)
    test_dataset = ISICDataset(root_dir=test_root, csv_file=test_csv, transform=transform)
    
    # Create data loaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# 3. Training and Evaluation Functions
# -----------------------------------------------------------------------------

def check_early_stopping(patience: int, best_loss: float,
                         current_loss: float, counter: int):
    """
    Checks if early stopping criteria are met.

    Parameters:
    - patience (int): Number of epochs to wait for improvement.
    - best_loss (float): Best loss achieved so far.
    - current_loss (float): Current loss value.
    - counter (int): Number of epochs since last improvement.

    Returns:
    - stop (bool): Whether to stop training.
    - best_loss (float): Updated best loss.
    - counter (int): Updated counter
    """

    if current_loss < best_loss:
        best_loss = current_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        return True, best_loss, counter
    return False, best_loss, counter

def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion,
                    optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """
    Trains the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on.
    
    Returns:
        train_loss (float): The training loss for the epoch.
        train_acc (float): The training accuracy for the epoch.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_model_with_early_stopping(model: nn.Module,
                                    train_loader: DataLoader,
                                    test_loader: DataLoader,
                                    criterion: torch.nn.Module,
                                    optimizer: torch.optim.Optimizer,
                                    device: torch.device,
                                    num_epochs: int,
                                    patience: int):
    """
    Trains the model with early stopping.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on.
        num_epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait for improvement before stopping.
    
    Returns:
        None
    """

    best_loss = float('inf')
    counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f'Training for epoch: {epoch}')
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader,
                                       criterion, device)

        print(f"Epoch [{epoch}/{num_epochs}]: \
              Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%\n")

        stop, best_loss, counter = check_early_stopping(patience, best_loss,
                                                        test_loss, counter)
        if stop:
            print(f"Early stopping at epoch {epoch}")
            break

def evaluate(model: nn.Module,
            dataloader: DataLoader,
            criterion: torch.nn.Module,
            device: torch.device) -> Tuple[float, float]:

    """Evaluates the model on the test set.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to train on.
    
    Returns:
        None
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# -----------------------------------------------------------------------------
# 4. Main Fine-Tuning Function
# -----------------------------------------------------------------------------
def main():
    """Main function to fine-tune the B‑cos ResNet-50 model on the ISIC dataset."""

    #Get config
    yaml_file = os.path.join(os.path.dirname(__file__), 'bcos.yaml')
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Get data loaders
    train_loader, test_loader = get_dataloaders(config)

    #Get model and define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    model = get_model(num_classes=config['num_classes'],
                      freeze_features=config['freeze_features'])
    model = model.to(device)

    optimizer = get_optimizer(model,
                              config['optimizer']['learning_rate'],
                              config['freeze_features'])
    
    # Train the model
    train_with_early_stopping = True
    num_epochs = config['train_params']['num_epochs']
    patience = config['train_params']['patience']

    if config['train_params']['early_stopping']:
        print("Training with early stopping...")
        train_model_with_early_stopping(model, train_loader, test_loader,
                                        criterion, optimizer, device,
                                        num_epochs, patience)
    else:
        for epoch in range(1, num_epochs + 1):
            _, train_acc = train_one_epoch(model, train_loader,
                                           criterion, optimizer, device)
            _, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"Epoch [{epoch}/{num_epochs}]: \
                  Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%\n")
    
    #Save finetuned model
    os.makedirs(config['output_dir'], exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__),
                             f'{config["output_dir"]}/{config["output_file"]}')
    torch.save(model.state_dict(), model_path)
    print("Model saved!")

if __name__ == "__main__":
    main()