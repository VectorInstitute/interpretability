import pandas as pd
from scipy.io import arff
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,MinMaxScaler, MaxAbsScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def process_csv(df):
    
    
    df["readmitted_binarized"] = df["readmitted"].apply(lambda x: 1 if x=="b'<30'" else 0)
    df = df.drop(['Unnamed: 0',"encounter_id","patient_nbr","examide", "readmitted","weight","payer_code","medical_specialty"], axis=1) #"citoglipton","readmitted","weight","payer_code","medical_specialty"], axis=1)  #discharge????


    # # age transformation was extracted from: https://medium.com/analytics-vidhya/diabetes-130-us-hospitals-for-years-1999-2008-e18d69beea4d
    age_dic = {"b'[0-10)'" : 5,
    "b'[10-20)'" : 15,
    "b'[20-30)'" : 25, 
    "b'[30-40)'" : 35, 
    "b'[40-50)'" : 45, 
    "b'[50-60)'" : 55,
    "b'[60-70)'" : 65, 
    "b'[70-80)'" : 75,
    "b'[80-90)'" : 85,
    "b'[90-100)'" : 95}
    
    df['age'] = df['age'].apply(lambda x : age_dic[x])
    
    df["diag_1"] = df["diag_1"].apply(lambda x: x[:x.find(".")])
    df["diag_2"] = df["diag_2"].apply(lambda x: x[:x.find(".")])
    df["diag_3"] = df["diag_3"].apply(lambda x: x[:x.find(".")])
      
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # https://www.kaggle.com/code/paulo100/tabtransformer-pytorch-dnn-with-attention-eda
    
    for cat_column in categorical_columns:
      frequency_encoding = df[cat_column].value_counts(normalize=True).to_dict()
      df[f"encoded_{cat_column}"] = df[cat_column].map(frequency_encoding)
      df = df.drop(cat_column, axis=1)

    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = [col for col in df.columns if col not in categorical_columns]  
    preprocessor = Preprocessor(numerical_columns,
                                categorical_columns,
                                encoder_categories,
                                emb_dim=3)
    x_nums, x_cats = preprocessor(sample_data)
    return df

def process_csv_transformer(df):
    
    
    df["readmitted_binarized"] = df["readmitted"].apply(lambda x: 1 if x=="b'<30'" else 0)
    
    df = df.drop(['Unnamed: 0',"encounter_id","patient_nbr","examide", "readmitted","weight","payer_code","medical_specialty"], axis=1) #"citoglipton","readmitted","weight","payer_code","medical_specialty"], axis=1)  #discharge????


    # # age transformation was extracted from: https://medium.com/analytics-vidhya/diabetes-130-us-hospitals-for-years-1999-2008-e18d69beea4d
    age_dic = {"b'[0-10)'" : 5,
    "b'[10-20)'" : 15,
    "b'[20-30)'" : 25, 
    "b'[30-40)'" : 35, 
    "b'[40-50)'" : 45, 
    "b'[50-60)'" : 55,
    "b'[60-70)'" : 65, 
    "b'[70-80)'" : 75,
    "b'[80-90)'" : 85,
    "b'[90-100)'" : 95}
    
    df['age'] = df['age'].apply(lambda x : age_dic[x])
    
    df["diag_1"] = df["diag_1"].apply(lambda x: x[:x.find(".")])
    df["diag_2"] = df["diag_2"].apply(lambda x: x[:x.find(".")])
    df["diag_3"] = df["diag_3"].apply(lambda x: x[:x.find(".")])
    df.columns = [col.replace('.', '') for col in df.columns]
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # https://www.kaggle.com/code/paulo100/tabtransformer-pytorch-dnn-with-attention-eda
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = [col for col in df.columns if col not in categorical_columns]  
    return df, numerical_columns, categorical_columns



class dfDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.feature_columns = self.df.drop("readmitted_binarized",axis=1)
        features = self.df.iloc[idx][self.feature_columns].values.astype('float32')
        label = self.df.iloc[idx]["readmitted_binarized"].astype('float32')
        return torch.tensor(features), torch.tensor(label)

# parts of the following functions and classes were extracted from: https://github.com/kherud/neural-additive-models-pt/tree/master
def get_train_test_fold(
        data_x,
        data_y,
        fold_num,
        num_folds,
        stratified=True,
        random_state=42):
    """Returns a specific fold split for K-Fold cross validation.

    Randomly split dataset into `num_folds` consecutive folds and returns the fold
    with index `fold_index` for testing while the `num_folds` - 1 remaining folds
    form the training set.

    Args:
      data_x: Training data, with shape (n_samples, n_features), where n_samples
        is the number of samples and n_features is the number of features.
      data_y: The target variable, with shape (n_samples), for supervised learning
        problems.  Stratification is done based on the y labels.
      fold_num: Index of fold used for testing.
      num_folds: Number of folds.
      stratified: Whether to preserve the percentage of samples for each class in
        the different folds (only applicable for classification).
      random_state: Seed used by the random number generator.

    Returns:
      (x_train, y_train): Training folds containing 1 - (1/`num_folds`) fraction
        of entire data.
      (x_test, y_test): Test fold containing 1/`num_folds` fraction of data.
    """
    if stratified:
        stratified_k_fold = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state)
    else:
        stratified_k_fold = KFold(
            n_splits=num_folds, shuffle=True, random_state=random_state)
    assert fold_num <= num_folds and fold_num > 0, 'Pass a valid fold number.'
    for train_index, test_index in stratified_k_fold.split(data_x, data_y):
        if fold_num == 1:
            x_train, x_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
            return (x_train, y_train), (x_test, y_test)
        else:
            fold_num -= 1

class CustomPipeline(Pipeline):
    """Custom sklearn Pipeline to transform data."""

    def apply_transformation(self, x):
        """Applies all transforms to the data, without applying last estimator.

        Args:
          x: Iterable data to predict on. Must fulfill input requirements of first
            step of the pipeline.

        Returns:
          xt: Transformed data.
        """
        xt = x
        for _, transform in self.steps[:-1]:
            xt = transform.fit_transform(xt)
        return xt

def split_training_dataset(
        data_x,
        data_y,
        n_splits,
        stratified=True,
        test_size=0.125,
        random_state=1337):
    """Yields a generator that randomly splits data into (train, validation) set.

    The train set is used for fitting the DNNs/NAMs while the validation set is
    used for early stopping.

    Args:
      data_x: Training data, with shape (n_samples, n_features), where n_samples
        is the number of samples and n_features is the number of features.
      data_y: The target variable, with shape (n_samples), for supervised learning
        problems.  Stratification is done based on the y labels.
      n_splits: Number of re-shuffling & splitting iterations.
      stratified: Whether to preserve the percentage of samples for each class in
        the (train, validation) splits. (only applicable for classification).
      test_size: The proportion of the dataset to include in the validation split.
      random_state: Seed used by the random number generator.

    Yields:
      (x_train, y_train): The training data split.
      (x_validation, y_validation): The validation data split.
    """
    if stratified:
        stratified_shuffle_split = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state)
    else:
        stratified_shuffle_split = ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state)
    split_gen = stratified_shuffle_split.split(data_x, data_y)

    for train_index, validation_index in split_gen:
        x_train, x_validation = data_x[train_index], data_x[validation_index]
        y_train, y_validation = data_y[train_index], data_y[validation_index]
        assert x_train.shape[0] == y_train.shape[0]
        yield (x_train, y_train), (x_validation, y_validation)



def transform_data(df):
    """Apply a fixed set of transformations to the pd.Dataframe `df`.

    Args:
      df: Input dataframe containing features.

    Returns:
      Transformed dataframe and corresponding column names. The transformations
      include (1) encoding categorical features as a one-hot numeric array, (2)
      identity `FunctionTransformer` for numerical variables. This is followed by
      scaling all features to the range (-1, 1) using min-max scaling.
    """
    column_names = df.columns
    new_column_names = []
    is_categorical = np.array([dt.kind == 'O' for dt in df.dtypes])
    categorical_cols = df.columns.values[is_categorical]
    numerical_cols = df.columns.values[~is_categorical]
    for index, is_cat in enumerate(is_categorical):
        col_name = column_names[index]
        if is_cat:
            new_column_names += [
                '{}: {}'.format(col_name, val) for val in set(df[col_name])
            ]
        else:
            new_column_names.append(col_name)
    cat_ohe_step = ('ohe', OneHotEncoder(sparse_output=False,handle_unknown='ignore'))

    cat_pipe = Pipeline([cat_ohe_step])
    num_pipe = Pipeline([('identity', FunctionTransformer(validate=True))])
    transformers = [('cat', cat_pipe, categorical_cols),
                    ('num', num_pipe, numerical_cols)]
    column_transform = ColumnTransformer(transformers=transformers)

    pipe = CustomPipeline([('column_transform', column_transform), 
                           ('min_max', MinMaxScaler((-1, 1))), ('dummy', None)])
    df = pipe.apply_transformation(df)
    return df, new_column_names


def create_test_train_fold(
        dataset,
        id_fold: int,
        n_folds: int,
        n_splits: int,
        regression: bool = False,
):
    """Splits the dataset into training and held-out test set."""
    
    data_x, data_y = dataset.loc[:, dataset.columns != "readmitted_binarized"], dataset.loc[:, dataset.columns == "readmitted_binarized"]
    data_x, column_names = transform_data(data_x)
    data_x = data_x.astype('float32')
    
    data_y = pd.get_dummies(data_y).values
    data_y = np.argmax(data_y, axis=-1)
    data_y = data_y.astype('float32')
    # Get the training and test set based on the StratifiedKFold split
    (x_train_all, y_train_all), test_dataset = get_train_test_fold(
        data_x,
        data_y,
        fold_num=id_fold,
        num_folds=n_folds,
        stratified=regression)
    data_gen = split_training_dataset(
        x_train_all,
        y_train_all,
        n_splits,
        stratified=regression)
    return data_gen, test_dataset


def create_test_train_fold_embed(
        data_x,data_y,
        id_fold: int,
        n_folds: int,
        n_splits: int,
        regression: bool = False,
):
    data_x = data_x.values.astype('float32')
    data_y = data_y.values.astype('float32')
    (x_train_all, y_train_all), test_dataset = get_train_test_fold(
        data_x,
        data_y,
        fold_num=id_fold,
        num_folds=n_folds,
        stratified=regression)
    data_gen = split_training_dataset(
        x_train_all,
        y_train_all,
        n_splits,
        stratified=regression)
    return data_gen, test_dataset





class TabDataset(torch.utils.data.Dataset):
    def __init__(self, df, numerical_columns,
                 categorical_columns, target=None):
        self.df = df
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        data = {}
        
        for nc in self.numerical_columns:
            
            x = torch.tensor(self.df[nc][index],
                             dtype=torch.float32)
            x = torch.unsqueeze(x, dim=0)
            data[nc] = x
        for cc in self.categorical_columns:
            x = torch.tensor(self.df[cc][index],
                             dtype=torch.int32)
            x = torch.unsqueeze(x, dim=0)
            data[cc] = x
        label = self.df.loc[index,"readmitted_binarized"]
        
        return data, label
