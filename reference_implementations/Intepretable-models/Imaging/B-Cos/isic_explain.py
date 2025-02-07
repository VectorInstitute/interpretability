

import os
import torch
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
torch.manual_seed(42)  # For reproducibility
random.seed(42)  # For reproducibility
# Import the fine-tuned B-cos ResNet-50 model and explanation utilities
from model.minimal_bcos_resnet import BcosResNet,Bottleneck
from model.minimal_bcos_resnet import resnet50
from bcos.common import BcosUtilMixin

# Import ISIC dataset class
from isic_fine_tune import ISICDataset  # Ensure `isic_fine_tune.py` is in the same directory


# Define a new class that extends BcosResNet with BcosUtilMixin
class BcosResNetExplainer(BcosUtilMixin, BcosResNet):
    """Extends BcosResNet with the explanation capabilities from BcosUtilMixin."""
    pass

# -----------------------------------------------------------------------------
# 1. Load Fine-Tuned Model
# -----------------------------------------------------------------------------

def load_fine_tuned_model(model_path="model/finetuned_weights/fine_tuned_isic_bcos_resnet50.pth", num_classes=2):
    """
    Loads the fine-tuned B-cos ResNet-50 model **with explanation support**.
    """
    model = BcosResNetExplainer(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_chans=6
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 2. Load Test Dataset
# -----------------------------------------------------------------------------
def get_test_dataset():
    test_root = "/home/dhaneshr/datasets/ISIC_2016/test_set"
    test_csv = os.path.join(test_root, "test_gt.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return ISICDataset(root_dir=test_root, csv_file=test_csv, transform=transform)

# -----------------------------------------------------------------------------
# 3. Visualize Model Explanations
# -----------------------------------------------------------------------------
# Unnormalize function (inverse of normalization)
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean  # Reverse normalization

def visualize_explanations(model, dataset, num_images=6):
    """
    Selects `num_images` random samples from the dataset and visualizes their explanations.
    """
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Select random indices
    
    indices = random.sample(range(len(dataset)), num_images)

    fig, axes = plt.subplots(num_images,2, figsize=(10,5 * num_images))

    for i, idx in enumerate(indices):
        image, label = dataset[idx]  # Get image and label
        image = image.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Add batch dimension

        # Ensure the input tensor requires gradients
        if not image.requires_grad:
            image.requires_grad = True

        # Get model explanation
        with torch.no_grad():
            explanation_output = model.explain(image)

        # ✅ Denormalize before plotting
        image = image.cpu().squeeze().detach()
        image_rgb = denormalize(image[:3], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Only first 3 channels (RGB)

        # Convert to NumPy for plotting
        original_image = image_rgb.permute(1, 2, 0).numpy().clip(0, 1)  # Ensure valid range
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
    print("Loading fine-tuned model...")
    model = load_fine_tuned_model()
    print("Model loaded.")

    print("Loading test dataset...")
    test_dataset = get_test_dataset()
    print("Test dataset loaded.")

    print("Visualizing explanations...")
    visualize_explanations(model, test_dataset)
    print("Explanations visualized.")
