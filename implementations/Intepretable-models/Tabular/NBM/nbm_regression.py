"""
This script demonstrates the use of the ConceptNBMNary model for regression tasks using the California Housing dataset.
It includes data preprocessing, model training, evaluation, and visualization of shape functions.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# We use the ConceptNBMNary model with nary=None so that it defaults to using unary interactions.
from model.nbm_model import ConceptNBMNary
from utils.plot_shapefunc import plot_nbm_shape_functions_with_feature_density


# fix random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------------
# Load and preprocess the dataset
# ------------------------------
# Load California Housing dataset
data = fetch_california_housing()
X = data.data  # features
y = data.target  # target (continuous)

# Apply MinMax scaling to the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert data to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
    1
)  # shape: [N, 1]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoader objects for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ------------------------------
# Set up the model, loss, and optimizer
# ------------------------------
num_concepts = X_train_tensor.shape[1]  # number of features (8 for California Housing)
num_classes = 1  # regression output is a single continuous value
num_bases = 100
hidden_dims = (256, 128, 128)
num_subnets = 1
dropout = 0.0
bases_dropout = 0.2
batchnorm = True

# Instantiate the model (nary is left as None so that it uses all unary interactions)
model = ConceptNBMNary(
    num_concepts=num_concepts,
    num_classes=num_classes,
    nary=None,
    num_bases=num_bases,
    hidden_dims=hidden_dims,
    num_subnets=num_subnets,
    dropout=dropout,
    bases_dropout=bases_dropout,
    batchnorm=batchnorm,
)

# Use Mean Squared Error for regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------
# Training loop
# ------------------------------
num_epochs = 50
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # In training mode, the model returns a tuple: (output, features)
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

# ------------------------------
# Evaluation on the test set
# ------------------------------
model.eval()
with torch.no_grad():
    total_loss = 0.0
    for inputs, targets in test_loader:
        # In eval mode, the model returns only the output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
    test_loss = total_loss / len(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
# RMSE = sqrt(test_loss)
print(f"RMSE: {test_loss**0.5:.4f}")


# Plot the shape functions of the model along with feature density

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model.eval()
model.to(device)

feature_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

plot_nbm_shape_functions_with_feature_density(
    model,
    X_test,
    feature_names=feature_names,
    n_points=50,  # more points for a smoother curve
    bins=50,  # more histogram bins
    device=device,
    plot_cols=4,
    red_alpha=0.4,
)
