import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score,PrecisionRecallDisplay, RocCurveDisplay, classification_report
import os
from pygam import LinearGAM, s, f
import joblib

np.random.seed(42)
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the folder path
folder_path = '../../../Post-hoc/datasets/gas+turbine+co+and+nox+emission+data+set'

# Load the CSV files
files_to_load = ['gt_2011.csv', 'gt_2012.csv', 'gt_2013.csv', 'gt_2014.csv']
dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in files_to_load]


# Concatenate the dataframes
train_val_data = pd.concat(dataframes, ignore_index=True)
print(train_val_data.head())

# Load the test data
test_data = pd.read_csv(os.path.join(folder_path, 'gt_2015.csv'))


scaler = StandardScaler()
train_val_data = pd.DataFrame(scaler.fit_transform(train_val_data), columns=train_val_data.columns)
test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

# convert to float16 to save memory
train_val_data = train_val_data.astype(np.float16)
test_data = test_data.astype(np.float16)


# Split the training and validation data
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

#use column TEY as target
target = 'TEY'
X_train = train_data.drop(target, axis=1)
y_train = train_data[target]
X_val = val_data.drop(target, axis=1)
y_val = val_data[target]
X_test = test_data.drop(target, axis=1)
y_test = test_data[target]

print(y_train.shape, y_val.shape, y_test.shape)

# import seaborn as sns

# corr_matrix = X_train.corr()
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
# plt.show()

print(X_train.var().sort_values())


# define gam for regression task

# initialize the first term

term = s(0)

# dynamically add more terms - use smooth function for numerical columns and factor function for categorical columns

for i, col in enumerate(X_train.columns):
    if X_train[col].dtype == 'int64':
        term += s(i)
    else:
        term += f(i)
    
gam = LinearGAM(terms=term,n_splines=5, lam=0.01, max_iter=100).fit(X_train.values, y_train.values)  

X_train = X_train.drop(columns=["CDP", "TAT"])  # Adjust based on your dataset

X_small = X_train.sample(frac=0.02, random_state=42)  # Take 20% of data
y_small = y_train.loc[X_small.index]

try:
    gam.fit(X_small, y_small)
except Exception as e:
    print(e)

print(gam.summary())
