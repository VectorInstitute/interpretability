import os
import csv

import torch
from PIL import Image
from torch.utils.data import Dataset


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
            raise ValueError(
                f"Dataset at {csv_file} is empty or incorrectly formatted!"
            )

        # Label encoding
        self.label_to_index = {"benign": 0, "malignant": 1}

    def __len__(self):
        return len(self.samples)  # ✅ Ensure this is defined

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]

        # Ensure label exists in dictionary
        if label not in self.label_to_index:
            raise ValueError(
                f"Unknown label '{label}' in dataset! Check CSV formatting."
            )

        img_path = os.path.join(self.root_dir, image_name + self.extension)
        image = Image.open(img_path).convert("RGB")  # Open image as RGB

        if self.transform:
            image = self.transform(image)  # Image shape: (3, H, W)

        # Generate 6-channel input: [RGB, 1-RGB]
        inverted_image = 1.0 - image
        image_6ch = torch.cat([image, inverted_image], dim=0)  # Shape: (6, H, W)

        label_idx = self.label_to_index[label]  # Convert to integer label
        return image_6ch, label_idx
