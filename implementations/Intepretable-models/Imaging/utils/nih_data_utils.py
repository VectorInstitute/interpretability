import os
import cv2
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from collections import OrderedDict
from imgaug import augmenters as iaa
from torchvision import transforms
from torch.utils.data import Dataset

NIH_DATA_PATH = "/ssd003/projects/aieng/public/interp_bootcamp/datasets/NIH"
NIH_DATA_ENTRIES = "Data_Entry_2017.csv"
NIH_TRAIN_VAL_LIST = "train_val_list.txt"
NIH_TEST_LIST = "test_list.txt"


def read_image(image_path: str):
    image = cv2.imread(image_path)
    image = image / np.max(image)
    return image


def load_nih_data(data_path: str = NIH_DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ """
    data = pd.read_csv(f"{data_path}/{NIH_DATA_ENTRIES}")
    train_names = f"{data_path}/{NIH_TRAIN_VAL_LIST}"
    test_names = f"{data_path}/{NIH_TEST_LIST}"

    with open(train_names, "r") as f:
        train_images = f.read().splitlines()
    with open(test_names, "r") as f:
        test_images = f.read().splitlines()

    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    train_df = data.loc[data["Image Index"].isin(train_images)]
    test_df = data.loc[data["Image Index"].isin(test_images)]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, test_df


class XrayDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, data_path: str = NIH_DATA_PATH) -> None:
        """ """
        self.data_path = data_path
        self._data_df = data_df
        self._init()
        self._get_all_filepaths()

    def _init(self):
        def get_unique_labels():
            unique_classes = set()
            for i in range(len(self._data_df)):
                row_labels = str.split(self._data_df.iloc[i, :]["Finding Labels"], "|")
                unique_classes.update(row_labels)
            return list(unique_classes)

        self.unique_labels = get_unique_labels()
        self.labels = self._data_df["Finding Labels"].apply(
            lambda x: self.get_label_vector(x)
        )

    def _get_all_filepaths(self):
        """ """
        self._filepaths = OrderedDict()
        for root, dirs, files in os.walk(self.data_path):
            self._filepaths.update(
                [(fname, os.path.join(root, fname)) for fname in files]
            )
            self._filepaths.update(
                [(dirname, os.path.join(root, dirname)) for dirname in dirs]
            )

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, label = self.get_image(idx)
        return image, label

    def get_label_vector(self, row):
        labels = str.split(row, "|")
        target = torch.zeros(len(self.unique_labels))
        for lab in labels:
            lab_idx = self.unique_labels.index(lab)
            target[lab_idx] = 1
        return target

    def get_image(self, idx: int):
        def transform_image(image):
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            input_size = 224
            seq = iaa.Sequential([iaa.Resize((input_size, input_size))])
            image_transform = transforms.Compose(
                [
                    seq.augment_image,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
            return image_transform(image)

        image_name = self._data_df.loc[idx, "Image Index"]
        image_path = self._filepaths[image_name]
        image = read_image(image_path)
        image = transform_image(image)
        return image.float(), self.labels[idx]
