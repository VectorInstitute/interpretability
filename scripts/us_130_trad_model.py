import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from new_data import process_csv
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import os
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
import categorical_embedder as ce
from interpret.glassbox import ExplainableBoostingClassifier,merge_ebms
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
import pickle
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def train_embedding(model, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    optimizer = torch_optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    for x1, x2, y in train_dl:
        batch = y.shape[0]
        output = model(x1, x2)
        loss = F.cross_entropy(output, y)   
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch*(loss.item())
    return sum_loss/total

if __name__ == "__main__":

    data = pd.read_csv("/Users/ananyaraval/workspace/interpretability-bootcamp/data/US_130/diabetic_data.csv")
    df = process_csv(data)
    df.index=range(df.shape[0])
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    print("colll1",len(df.columns),len(categorical_columns))
    # for col in categorical_columns:
    #     df[col] = LabelEncoder().fit_transform(df[col])
    #     df[col] = df[col].astype('category')
    print("dfff",df.head())
    X , y = df.drop("readmitted_binarized",axis=1) , df["readmitted_binarized"]
    
    # X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.10, random_state=0)
    embedded_cols = {n: len(col.cat.categories) for n,col in X[categorical_columns].items() if len(col.cat.categories) > 2}
    print("diccc",len(embedded_cols))
    embedded_col_names = embedded_cols.keys()
    embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]

    print("kkkkk",embedding_sizes)
    print("embed",embedded_cols)
    print("colsss",df.columns)

    k = 5
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    auc_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    fi_ni = []
    for i in range(k):
        test_idx = folds[i]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Use all other folds as the training set
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

        # embedding_info = ce.get_embedding_info(X)
        # # print("embeeeee",embedding_info)
        # X_encoded,encoders = ce.get_label_encoded_data(X)
        # # print("encoders",encoders)
        


        # # print("testttt",X_test.shape)
        # embeddings = ce.get_embeddings(X_train, y_train, categorical_embedding_info=embedding_info, 
        #                         is_classification=True, epochs=10,batch_size=128)
        # dfs = ce.get_embeddings_in_dataframe(embeddings=embeddings, encoders=encoders)
        # print("thereee",dfs.keys())
        

        # X_train = ce.fit_transform(X_train, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)
        # X_test = ce.fit_transform(X_test, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)
        
        
        smote = RandomOverSampler(random_state=42)#SMOTE(random_state=42)#RandomOverSampler(random_state=42)#SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train) 
        print("traiiiinnnnnn",X_train.shape)
        lasso = Lasso(alpha=0.0001) # 0.001
        lasso.fit(X_train, y_train)
        # ridge = Ridge(alpha=1.0)
        # ridge.fit(X_train, y_train)
        # Get the coefficients of the features
        coef = lasso.coef_

        # Select non-zero coefficients 
        selected_features = np.where(coef != 0)[0]
        top_features = [X_train.columns[i] for i in selected_features]

        # Make predictions
        
        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]
        ebm_selected = xgb.XGBClassifier(n_estimators= 300, max_depth= 6, learning_rate= 0.01, gamma= 0.01, alpha=0.1)#,reg_lambda=1)#(eta= 0.01,
        # ebm_selected = ExplainableBoostingClassifier()
    #     learning_rate=0.01,
    #     max_bins=128,
    #     max_interaction_bins=64,
    #     min_samples_leaf=5,
    #     early_stopping_rounds=10
    # )
        
        # scaler = StandardScaler()

        # # Fit and transform the features
        # X_train_selected = scaler.fit_transform(X_train_selected)
        # X_test_selected = scaler.transform(X_test_selected)
        # X_train_selected = pd.DataFrame(X_train_selected)
        # X_test_selected = pd.DataFrame(X_test_selected)

        file = open(f'x_test_{i}', 'wb')

        # # dump information to that file

        pickle.dump(X_test_selected, file)

        # # close the file
        file.close()

        file = open(f'y_train_{i}', 'wb')

        # # dump information to that file
        pickle.dump(y_train, file)

        # # close the file
        file.close()
        file = open(f'x_train_{i}', 'wb')

        # # dump information to that file

        pickle.dump(X_train_selected, file)

        # # close the file
        file.close()
        

        file = open(f'y_test_{i}', 'wb')

        # # dump information to that file

        pickle.dump(y_test, file)

        # # close the file
        file.close()
        ebm_selected.fit(X_train_selected, y_train)
        y_prob = ebm_selected.predict_proba(X_test_selected)[:,1]
        y_pred = ebm_selected.predict(X_test_selected)
        y_pred_train = ebm_selected.predict_proba(X_train_selected)[:,1]
        train_auc = roc_auc_score(y_train, y_pred_train)
        val_auc = roc_auc_score(y_test, y_prob)
        val_f1 = f1_score(y_test,y_pred)
        val_precision = precision_score(y_test,y_pred)
        val_recall = recall_score(y_test,y_pred)
        print("val_auc",val_auc)
        print("train_auc",train_auc)
        print("val_f1",val_f1)
        print("val_precision",val_precision)
        print("val_recall",val_recall)
        f1_scores.append(val_f1)
        auc_scores.append(val_auc)
        recall_scores.append(val_recall)
        precision_scores.append(val_precision)
        importances = ebm_selected.feature_importances_
        # # fi[f"{i}"] = importances
        # # importances = ebm_selected.explain_global()
        # # print("dirrrrrrrrrrrrr",importances.data)#,dir(importances))
        # print("improvee",importances)
        # fi.append(importances)
        feature_names = X_train_selected.columns
        # print("ff",feature_names)
        # ni_index = feature_names.get_loc("number_inpatient")#feature_names[0].index("number_inpatient")
        # fi_ni.append(importances[ni_index])

# # Sort the feature importances in descending order
#         # sorted_indices = np.argsort(importances)[::-1]
        
#         # for idx in sorted_indices:
#         #     print(f"{feature_names[idx]}: {importances[idx]:.4f}")
#         plt.figure(figsize=(20, 15))
#         plt.barh(range(len(feature_names)), importances, align='center')
#         plt.yticks(range(len(feature_names)), feature_names)
#         plt.xlabel('Feature Importance')
#         plt.title('XGBoost Feature Importance')

#         # Save the plot to a file
#         plt.savefig(f'xgboost_feature_importance{i}.png')
    

    print("mean f1",np.mean(f1_scores),np.std(f1_scores))
    print("mean auc",np.mean(auc_scores),np.std(auc_scores))
    print("mean precision",np.mean(precision_scores),np.std(precision_scores))
    print("mean recall",np.mean(recall_scores),np.std(recall_scores))
    print("importance mean", np.mean(fi_ni, axis=0))
    print("importance std", np.std(fi_ni, axis=0))




