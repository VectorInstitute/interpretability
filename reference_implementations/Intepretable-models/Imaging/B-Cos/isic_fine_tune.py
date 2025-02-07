#!/usr/bin/env python
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define constants
MODEL_SAVE_PATH = "model/finetuned_weights/fine_tuned_isic_bcos_resnet50.pth"

# -----------------------------------------------------------------------------
# 1. Define a Custom Dataset for ISIC 2016 with 6-Channel Inputs
# -----------------------------------------------------------------------------
class ISICDataset(Dataset):
    """
    A custom dataset class for the ISIC skin lesion dataset.

    Args:
        root_dir (str): Directory with all the images.
        csv_file (str): Path to the CSV file with image names and labels.
        transform (callable, optional): Optional transform to be applied on a sample.
        extension (str, optional): File extension of the images. Default is ".jpg".

    Attributes:
        root_dir (str): Directory with all the images.
        csv_file (str): Path to the CSV file with image names and labels.
        transform (callable, optional): Optional transform to be applied on a sample.
        extension (str): File extension of the images.
        samples (list): List of tuples containing image names and labels.
        label_to_index (dict): Dictionary mapping label names to integer indices.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the sample (image and label) at the given index.
    """
    def __init__(self, root_dir: str, csv_file: str, transform=None, extension=".jpg"):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.extension = extension
        self.samples = []

        # Read CSV file and store (image_name, label)
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                image_name, label = row[0].strip(), row[1].strip().lower()

                # Convert numerical labels to string format
                if label in ["0", "0.0"]:
                    label = "benign"
                elif label in ["1", "1.0"]:
                    label = "malignant"

                self.samples.append((image_name, label))

        # Ensure we have data
        if not self.samples:
            raise ValueError(f"Dataset at {csv_file} is empty or incorrectly formatted!")

        # Label encoding
        self.label_to_index = {"benign": 0, "malignant": 1}

    def __len__(self):
        return len(self.samples)  # ✅ Ensure this is defined

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]

        # Ensure label exists in dictionary
        if label not in self.label_to_index:
            raise ValueError(f"Unknown label '{label}' in dataset! Check CSV formatting.")

        img_path = os.path.join(self.root_dir, image_name + self.extension)
        image = Image.open(img_path).convert("RGB")  # Open image as RGB

        if self.transform:
            image = self.transform(image)  # Image shape: (3, H, W)

        # Generate 6-channel input: [RGB, 1-RGB]
        inverted_image = 1.0 - image
        image_6ch = torch.cat([image, inverted_image], dim=0)  # Shape: (6, H, W)

        label_idx = self.label_to_index[label]  # Convert to integer label
        return image_6ch, label_idx




# -----------------------------------------------------------------------------
# 2. Load the Pretrained B‑cos ResNet-50 Model (with 6 Channels)
# -----------------------------------------------------------------------------
from model.minimal_bcos_resnet import resnet50
from model.minimal_bcos_resnet import BcosConv2d 

def get_model(num_classes: int = 2, freeze_features: bool = False) -> nn.Module:
    """
    Loads a pretrained B‑cos ResNet-50 model (6-channel input) and modifies it for fine-tuning.
    
    Fixes state_dict mismatch by **removing the fc.weight from pretrained weights**.
    
    Parameters:
    - num_classes (int): Number of output classes.
    - freeze_features (bool): If True, freezes the feature extractor layers.
    """
    # Load model (original has num_classes=1000)
    model = resnet50(pretrained=True, num_classes=1000, in_chans=6, long_version=True)

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
    model.fc = BcosConv2d(in_channels, num_classes, kernel_size=1, bias=False)  

    # Load modified state_dict (now without fc.weight)
    model.load_state_dict(state_dict, strict=False)

    return model

def get_optimizer(model: nn.Module, learning_rate: float, freeze_features: bool) -> optim.Optimizer:
    """
    Creates an optimizer with different learning rates for the feature extractor and the final layer.
    
    Parameters:
    - model (nn.Module): The model to optimize.
    - learning_rate (float): The base learning rate.
    - freeze_features (bool): If True, freezes the feature extractor layers.
    """
    if freeze_features:
        return optim.Adam(model.parameters(), lr=learning_rate)
    else:
        # Apply different learning rates
        params = [
            {"params": model.fc.parameters(), "lr": learning_rate * 3},
            {"params": [param for name, param in model.named_parameters() if "fc" not in name], "lr": learning_rate}
        ]
        return optim.Adam(params)

# -----------------------------------------------------------------------------
# 3. Create Data Loaders
# -----------------------------------------------------------------------------
def get_dataloaders(batch_size: int = 32):
    """
        Returns the training and test data loaders.
        
        Args:
            batch_size (int): The batch size for the data loaders.
        
        Returns:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    """
    # Define the dataset directories and CSV files.
    train_root = "/home/dhaneshr/datasets/ISIC_2016/train_set"
    test_root = "/home/dhaneshr/datasets/ISIC_2016/test_set"
    train_csv = os.path.join(train_root, "train_gt.csv")
    test_csv = os.path.join(test_root, "test_gt.csv")
    
    # Define data transforms.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# 4. Training and Evaluation Functions
# -----------------------------------------------------------------------------


def check_early_stopping(patience: int, best_loss: float, current_loss: float, counter: int):
    if current_loss < best_loss:
        best_loss = current_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        return True, best_loss, counter
    return False, best_loss, counter

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device):
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

def train_model_with_early_stopping(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, criterion, optimizer, device: torch.device, num_epochs: int, patience: int):
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
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch [{epoch}/{num_epochs}]: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%\n")

        stop, best_loss, counter = check_early_stopping(patience, best_loss, test_loss, counter)
        if stop:
            print(f"Early stopping at epoch {epoch}")
            break

    

    


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device):
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
# 5. Main Fine-Tuning Script
# -----------------------------------------------------------------------------
def main():
    num_classes = 2  # Benign and Malignant
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    freeze_features = False  # Set True to freeze feature extractor
    train_with_early_stopping = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(num_classes=num_classes, freeze_features=freeze_features)
    model = model.to(device)
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, learning_rate, freeze_features)
    if train_with_early_stopping:
        print("Training with early stopping...")
        train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, patience=3)
    else:
        for epoch in range(1, num_epochs + 1):
            _, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"Epoch [{epoch}/{num_epochs}]: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%\n")
        

    os.makedirs("model/finetuned_weights/", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved!")

if __name__ == "__main__":
    main()
