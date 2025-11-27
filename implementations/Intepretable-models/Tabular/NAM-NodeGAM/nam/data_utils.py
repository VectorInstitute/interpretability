import pandas as pd
from typing import Tuple
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

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