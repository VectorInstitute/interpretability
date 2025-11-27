"""Dataset loading utilities."""

import os

import pandas as pd


def load_adult_dataset(directory: str, filename: str = "adult.data") -> pd.DataFrame:
    """
    Load the Adult Income dataset from a local directory.

    Parameters
    ----------
    directory : str
        The path to the local directory containing the dataset file.
    filename : str, default="adult.data"
        The name of the dataset file.

    Returns
    -------
    pd.DataFrame
        The cleaned Adult dataset with column names.

    Raises
    ------
    FileNotFoundError
        If the dataset file is not found at the specified path.
    """
    file_path = os.path.join(directory, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at {file_path}. Please check the directory path."
        )

    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        na_values=" ?",
    )

    df.dropna(inplace=True)

    return df


def load_diabetes_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the diabetes dataset.

    Parameters
    ----------
    file_path : str
        Path to the diabetes CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the diabetes data.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Diabetes dataset not found at {file_path}")

    return pd.read_csv(file_path)


def load_gas_turbine_dataset(
    folder_path: str, files_to_load: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate gas turbine datasets.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing gas turbine data files.
    files_to_load : list of str, optional
        List of CSV files to load for training/validation.
        If None, defaults to ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv"].

    Returns
    -------
    train_val_data : pd.DataFrame
        Concatenated training and validation data.
    test_data : pd.DataFrame
        Test data from gt_2015.csv.

    Raises
    ------
    FileNotFoundError
        If any of the specified files are not found.
    """
    if files_to_load is None:
        files_to_load = ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv"]

    dataframes = []
    for file in files_to_load:
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        dataframes.append(pd.read_csv(file_path))

    train_val_data = pd.concat(dataframes, ignore_index=True)

    test_file_path = os.path.join(folder_path, "gt_2015.csv")
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")
    test_data = pd.read_csv(test_file_path)

    return train_val_data, test_data
