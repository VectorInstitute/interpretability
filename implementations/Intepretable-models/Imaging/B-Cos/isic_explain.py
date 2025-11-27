import os
import yaml
import random

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Union
from torchvision import transforms

# Import the fine-tuned B-cos ResNet-50 model and explanation utilities
from bcos.minimal_bcos_resnet import BcosResNet, Bottleneck
from bcos.common import BcosUtilMixin

# Import ISIC dataset class
from bcos.isic_data import ISICDataset

torch.manual_seed(42)  # For reproducibility
random.seed(42)  # For reproducibility


# Define a new class that extends BcosResNet with BcosUtilMixin
class BcosResNetExplainer(BcosUtilMixin, BcosResNet):
    """Extends BcosResNet with the explanation capabilities from BcosUtilMixin."""

    pass


# -----------------------------------------------------------------------------
# 1. Load Fine-Tuned Model
# -----------------------------------------------------------------------------


def load_fine_tuned_model(cfg: dict) -> torch.nn.Module:
    """
    Loads the fine-tuned B-cos ResNet-50 model **with explanation support**.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(
        os.path.dirname(__file__), f'{cfg["output_dir"]}/{cfg["output_file"]}'
    )
    model = BcosResNetExplainer(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=cfg["num_classes"],
        in_chans=6,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# -----------------------------------------------------------------------------
# 2. Load Test Dataset
# -----------------------------------------------------------------------------
def get_test_dataset(cfg: dict) -> ISICDataset:
    """
    Loads and returns the test dataset for the ISIC 2016 challenge.

    Parameters:
        cfg (dict): The configuration dictionary containing the dataset paths.

    Returns:
        ISICDataset: A dataset object containing the test images and labels.

    The dataset is loaded from a specified directory and CSV file, and
    transformations are applied to the images including resizing,
    normalization, and conversion to tensor format.
    """

    test_root = cfg["test_dir"]
    test_csv = os.path.join(test_root, cfg["test_csv"])

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return ISICDataset(root_dir=test_root, csv_file=test_csv, transform=transform)


# -----------------------------------------------------------------------------
# 3. Visualize Model Explanations
# -----------------------------------------------------------------------------
# Unnormalize function (inverse of normalization)
def denormalize(
    tensor: torch.Tensor, mean: Union[list, tuple], std: Union[list, tuple]
) -> torch.Tensor:
    """
    Reverse the normalization of a tensor.

    Args:
        tensor (torch.Tensor): The normalized tensor to be denormalized.
        mean (list or tuple): The mean values used for normalization.
        std (list or tuple): The standard deviation values used for normalization.

    Returns:
        torch.Tensor: The denormalized tensor.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean  # Reverse normalization


def visualize_explanations(
    model: torch.nn.Module, dataset: torch.utils.data.Dataset, num_images: int = 6
):
    """
    Visualizes explanations for a given model and dataset.
    Parameters:
    model (torch.nn.Module): The model to be explained.
    dataset (torch.utils.data.Dataset): The dataset containing images and labels.
    num_images (int, optional): The number of images to visualize. Default is 6.

    Returns:
    None: This function saves the visualization as 'isic_explanations.png'.

    The function performs the following steps:
    1. Moves the model to the appropriate device (GPU if available, otherwise CPU).
    2. Selects random indices from the dataset.
    3. Creates a subplot for each image and its corresponding explanation.
    4. For each selected image:
        a. Retrieves the image and label from the dataset.
        b. Moves the image to the appropriate device and ensures it requires gradients.
        c. Obtains the model's explanation for the image.
        d. Denormalizes the image for visualization.
        e. Converts the image to a NumPy array for plotting.
        f. Plots the original image and its explanation side by side.
    5. Adjusts the layout and saves the figure as 'isic_explanations.png'.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Select random indices

    indices = random.sample(range(len(dataset)), num_images)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i, idx in enumerate(indices):
        image, label = dataset[idx]  # Get image and label
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        # Ensure the input tensor requires gradients
        if not image.requires_grad:
            image.requires_grad = True

        # Get model explanation
        with torch.no_grad():
            explanation_output = model.explain(image)

        # ✅ Denormalize before plotting
        image = image.cpu().squeeze().detach()
        image_rgb = denormalize(
            image[:3], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )  # Only first 3 channels (RGB)

        # Convert to NumPy for plotting
        original_image = (
            image_rgb.permute(1, 2, 0).numpy().clip(0, 1)
        )  # Ensure valid range
        explanation_image = explanation_output["explanation"]

        # Plot original image
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f"Original (Label: {label})")
        axes[i, 0].axis("off")

        # Plot explanation
        axes[i, 1].imshow(explanation_image)
        axes[i, 1].set_title(f"Explanation (Pred: {explanation_output['prediction']})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("isic_explanations.png")


# -----------------------------------------------------------------------------
# 4. Main Script Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Get config
    yaml_file = os.path.join(os.path.dirname(__file__), "bcos.yaml")
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    print("Loading fine-tuned model...")
    model = load_fine_tuned_model(config)
    print("Model loaded.")

    print("Loading test dataset...")
    test_dataset = get_test_dataset(config)
    print("Test dataset loaded.")

    print("Visualizing explanations...")
    visualize_explanations(model, test_dataset)
    print("Explanations visualized.")
