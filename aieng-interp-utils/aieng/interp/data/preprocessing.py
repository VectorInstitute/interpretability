"""Data preprocessing utilities."""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_adult_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], np.ndarray, np.ndarray]:
    """
    Preprocess the Adult Income dataset.

    Encodes categorical variables using LabelEncoder and scales numerical features
    using StandardScaler. Saves the label encoders for future use.

    Parameters
    ----------
    df : pd.DataFrame
        The Adult dataset to preprocess.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed DataFrame with encoded and scaled features.
    categorical_features : list of str
        List of categorical feature names.
    numerical_features : list of str
        List of numerical feature names.
    scaler_mean : np.ndarray
        Mean values from StandardScaler for numerical features.
    scaler_var : np.ndarray
        Variance values from StandardScaler for numerical features.
    """
    df = df.copy()

    df["income"] = df["income"].apply(lambda x: 1 if x == " >50K" else 0)
    df.drop(columns=["education"], inplace=True)
    df.drop(columns=["fnlwgt"], inplace=True)

    categorical_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    numerical_features = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    joblib.dump(label_encoders, "label_encoders.pkl")

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    scaler_mean = scaler.mean_
    scaler_var = scaler.var_

    return df, categorical_features, numerical_features, scaler_mean, scaler_var


def preprocess_adult_test_data(
    df: pd.DataFrame,
    label_encoders: dict,
    scaler_mean: np.ndarray,
    scaler_var: np.ndarray,
) -> pd.DataFrame:
    """
    Preprocess Adult Income test dataset using pre-fitted encoders and scalers.

    Parameters
    ----------
    df : pd.DataFrame
        The Adult test dataset to preprocess.
    label_encoders : dict
        Dictionary of fitted LabelEncoder objects for categorical features.
    scaler_mean : np.ndarray
        Mean values from training set StandardScaler.
    scaler_var : np.ndarray
        Variance values from training set StandardScaler.

    Returns
    -------
    pd.DataFrame
        Preprocessed test DataFrame with encoded and scaled features.
    """
    df = df.copy()

    df["income"] = df["income"].apply(lambda x: 1 if x == " >50K." else 0)
    df.drop(columns=["education"], inplace=True)
    df.drop(columns=["fnlwgt"], inplace=True)

    categorical_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    numerical_features = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    for col in categorical_features:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    df[numerical_features] = (df[numerical_features] - scaler_mean) / np.sqrt(
        scaler_var
    )

    return df


def process_diabetes_us_130(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the Diabetes 130-US hospitals dataset.

    Applies transformations including age encoding, diagnosis cleaning,
    column dropping, and frequency encoding of categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Diabetes 130-US dataset.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed DataFrame with encoded features.
    numerical_columns : list of str
        List of numerical column names.
    """
    age_transform = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }

    df = df.copy()

    df["age"] = df["age"].apply(lambda x: age_transform[x])
    df["diag_1"] = df["diag_1"].apply(lambda x: x[: x.find(".")])
    df["diag_2"] = df["diag_2"].apply(lambda x: x[: x.find(".")])
    df["diag_3"] = df["diag_3"].apply(lambda x: x[: x.find(".")])
    df["readmitted_binarized"] = df["readmitted"].apply(
        lambda x: 1 if x == "<30" else 0
    )
    df["max_glu_serum"] = df["max_glu_serum"].apply(
        lambda x: "Unknown" if not isinstance(x, str) else x
    )
    df["A1Cresult"] = df["A1Cresult"].apply(
        lambda x: "Unknown" if not isinstance(x, str) else x
    )

    columns_to_drop = [
        "encounter_id",
        "patient_nbr",
        "examide",
        "readmitted",
        "weight",
        "payer_code",
        "medical_specialty",
    ]
    df = df.drop(columns_to_drop, axis=1)

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_columns = [col for col in df.columns if col not in categorical_columns]

    for cat_column in categorical_columns:
        frequency_encoding = df[cat_column].value_counts(normalize=True).to_dict()
        df[f"encoded_{cat_column}"] = df[cat_column].map(frequency_encoding)
        df = df.drop(cat_column, axis=1)

    return df, numerical_columns


def split_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    stratified: bool = True,
    test_size: float = 0.125,
    random_state: int = 1337,
):
    """
    Randomly split data into training and validation sets.

    Yields a generator that creates multiple train/validation splits
    for cross-validation or multiple training runs.

    Parameters
    ----------
    data : np.ndarray
        Training data with shape (n_samples, n_features).
    labels : np.ndarray
        Target variable with shape (n_samples,).
    n_splits : int
        Number of re-shuffling and splitting iterations.
    stratified : bool, default=True
        Whether to preserve the percentage of samples for each class
        in the train/validation splits (only applicable for classification).
    test_size : float, default=0.125
        The proportion of the dataset to include in the validation split.
    random_state : int, default=1337
        Seed used by the random number generator.

    Yields
    ------
    (x1, y1) : tuple of np.ndarray
        The training data split.
    (x2, y2) : tuple of np.ndarray
        The validation data split.
    """
    if stratified:
        ssplit = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
    else:
        ssplit = ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )

    for train_idx, val_idx in ssplit.split(data, labels):
        x1, x2 = data[train_idx], data[val_idx]
        y1, y2 = labels[train_idx], labels[val_idx]
        yield (x1, y1), (x2, y2)
