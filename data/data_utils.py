from typing import Tuple

import cv2
import torch
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from data import Config

def read_image(image_path: str):
    image = cv2.imread(image_path)
    image = image/np.max(image)
    return image

def process_us_130_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function for new data downloaded from UCL
    """
    age_transform = { '[0-10)' : 5,
                      '[10-20)' : 15,
                      '[20-30)' : 25,
                      '[30-40)' : 35,
                      '[40-50)' : 45,
                      '[50-60)' : 55,
                      '[60-70)' : 65,
                      '[70-80)' : 75,
                      '[80-90)' : 85,
                      '[90-100)' : 95
                    }
    
    #Apply column specific transformations
    df['age'] = df['age'].apply(lambda x : age_transform[x])
    df['diag_1'] = df['diag_1'].apply(lambda x: x[:x.find(".")])
    df['diag_2'] = df['diag_2'].apply(lambda x: x[:x.find(".")])
    df['diag_3'] = df['diag_3'].apply(lambda x: x[:x.find(".")])
    df['readmitted_binarized'] = df['readmitted'].apply(lambda x: 1 if x=='<30' else 0)
    df['max_glu_serum'] = df['max_glu_serum'].apply(lambda x: 'Unknown' if type(x) != str else x)
    df['A1Cresult'] = df['A1Cresult'].apply(lambda x: 'Unknown' if type(x) != str else x)

    #Drop columns which are not needed
    df = df.drop(['encounter_id', 'patient_nbr', 'examide',
                  'readmitted','weight','payer_code', 'medical_specialty'], axis=1)

    #Frequency encoding of categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()    
    for cat_column in categorical_columns:
        frequency_encoding = df[cat_column].value_counts(normalize=True).to_dict()
        df[f'encoded_{cat_column}'] = df[cat_column].map(frequency_encoding)
        df = df.drop(cat_column, axis=1)

    return df

def split_dataset(
        data,
        labels,
        n_splits,
        stratified=True,
        test_size=0.125,
        random_state=1337):
    """Yields a generator that randomly splits data into (train, validation) set.

    The train set is used for fitting the DNNs/NAMs while the validation set is
    used for early stopping.

    Args:
      data: Training data, with shape (n_samples, n_features), where n_samples
        is the number of samples and n_features is the number of features.
      labels: The target variable, with shape (n_samples), for supervised learning
        problems.  Stratification is done based on the y labels.
      n_splits: Number of re-shuffling & splitting iterations.
      stratified: Whether to preserve the percentage of samples for each class in
        the (train, validation) splits. (only applicable for classification).
      test_size: The proportion of the dataset to include in the test/validation split.
      random_state: Seed used by the random number generator.

    Yields:
      (x1, y1): The training data split.
      (x2, y2): The validation data split.
    """
    
    if stratified:
        ssplit = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state)
    else:
        ssplit = ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state)

    for train_idx, val_idx in ssplit.split(data, labels):
        x1, x2 = data[train_idx], data[val_idx]
        y1, y2 = labels[train_idx], labels[val_idx]
        yield (x1, y1), (x2, y2)

def load_nih_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        """
        nih_files = cfg.get_datafiles('nih')

        data = pd.read_csv(nih_files["Data_Entry_2017.csv"])
        train_names = nih_files["train_val_list.txt"]
        test_names = nih_files["test_list.txt"]

        with open(train_names, 'r') as f:
            train_images = f.read().splitlines()
        with open(test_names, 'r') as f:
            test_images = f.read().splitlines()

        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        train_df = data.loc[data['Image Index'].isin(train_images)]
        test_df  = data.loc[data['Image Index'].isin(test_images)]
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        return train_df, test_df


class XrayDataset(Dataset):
    def __init__(self,
                 cfg: Config,
                 data_df: pd.DataFrame) -> None:
        """
        """
        self._file_paths = cfg.get_datafiles('nih', exclude_subdirs=False)
        self._data_df = data_df
        self._init()

    def _init(self):
        def get_unique_labels():
            unique_classes = set()
            for i in range(len(self._data_df)):
                row_labels = str.split(self._data_df.iloc[i, :]['Finding Labels'], '|')
                unique_classes.update(row_labels)
            return list(unique_classes)

        self.unique_labels = get_unique_labels()
        self.labels = self._data_df['Finding Labels'].apply(lambda x: self.get_label_vector(x))

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, label = self.get_image(idx)
        return image,label

    def get_label_vector(self, row):
        labels = str.split(row, '|')
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
            image_transform = transforms.Compose([seq.augment_image,
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean, std=std)])
            return image_transform(image)

        image_name = self._data_df.loc[idx, 'Image Index']
        image_path = self._file_paths[image_name]
        image = read_image(image_path)
        image = transform_image(image)
        return image.float(), self.labels[idx]