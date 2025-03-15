from collections import Counter
from typing import Union, List, Tuple, Dict

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from imblearn.over_sampling import RandomOverSampler
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from data import Config, process_us_130_csv

def plot_features(model: Union[xgb.XGBClassifier, ExplainableBoostingClassifier],
                  fold: int,
                  feature_names: List) -> None:
    """
    """
    if isinstance(model, xgb.XGBClassifier):
        importances = model.feature_importances_
        model_type = 'xgb'
        plt.figure(figsize=(20, 15))
        plt.barh(range(len(feature_names)), importances, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel(f'Feature Importance for Fold {fold}')
        plt.title('Feature Importance plot')
        plt.savefig(f'feature_importance_{model_type}_{fold}.png')
    else:
        importances = model.explain_global()
        model_type = 'ebm'
        ebm_plot = importances.visualize()
        ebm_plot.write_image(f"feature_importance_{model_type}_{fold}.png")

def get_classifier(explainable: bool = False
                    ) -> Union[xgb.XGBClassifier, ExplainableBoostingClassifier]:
    """
    """
    if explainable:
        ebm = ExplainableBoostingClassifier(
                            learning_rate=0.01,
                            max_bins=128,
                            max_interaction_bins=64,
                            min_samples_leaf=5,
                            early_stopping_rounds=10)
    else:
        ebm = xgb.XGBClassifier(n_estimators=300,
                                max_depth=6,
                                learning_rate=0.01,
                                gamma=0.01,
                                alpha=0.1)
    return ebm

def get_predictions(model: Union[xgb.XGBClassifier, ExplainableBoostingClassifier],
                    X_train: pd.DataFrame,
                    X_test: pd.DataFrame) -> Tuple:
    """
    """
    y_te_pred = model.predict(X_test)
    y_te_prob = model.predict_proba(X_test)[:,1]
    y_tr_pred = model.predict(X_train)
    y_tr_prob = model.predict_proba(X_train)[:,1]

    return y_te_prob, y_te_pred, y_tr_pred, y_tr_prob

def get_top_features(X_train: pd.DataFrame,
                     y_train: pd.DataFrame)-> List:
    """
    """
    lasso = Lasso(alpha=0.0001)
    smote = RandomOverSampler(random_state=42)

    X_train, y_train = smote.fit_resample(X_train, y_train)
    lasso.fit(X_train, y_train)

    # Select columns with non-zero lasso coefficients 
    selected_feats = np.where(lasso.coef_ != 0)[0]
    top_feats = [X_train.columns[i] for i in selected_feats]

    return top_feats

def train_and_predict(model: Union[xgb.XGBClassifier, ExplainableBoostingClassifier],
                      X: pd.DataFrame,
                      y: pd.DataFrame,
                      visualize: bool = False) -> Dict:
    
    """
    """
    auc, f1, precision, recall = [], [], [], []

    #Run model for 5 fold split
    kf=KFold(n_splits=5, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n~~~ Fold: {i+1} ~~~~")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        #Get top features
        top_feats = get_top_features(X_train, y_train)

        #Extract data with top feature columns
        X_train_selected, X_test_selected = X_train[top_feats], X_test[top_feats]

        #Fit model
        model.fit(X_train_selected, y_train)

        #Save plots
        if visualize:
            plot_features(model, i+1, X_train_selected.columns)

        #Calculate train and test predictions
        y_te_prob, y_te_pred, y_tr_pred, y_tr_prob = get_predictions(model,
                                                        X_train_selected,
                                                        X_test_selected)
        #Calculate train AUC score
        train_auc = roc_auc_score(y_train, y_tr_pred)

        print(f"\nTrain AUC score for fold {i+1}: ", train_auc)

        #Calculate test data scores
        auc.append(roc_auc_score(y_test, y_te_prob))
        f1.append(f1_score(y_test, y_te_pred))
        precision.append(recall_score(y_test, y_te_pred))
        recall.append(precision_score(y_test, y_te_pred))
       
        print(f"\nTest scores for Fold {i+1}: \nF1: {f1[i]} \nAUC: {auc[i]}, \
              \nPrecision: {precision[i]} \nRecall: {recall[i]}")
               
    return {'f1': (np.mean(f1), np.std(f1)),
            'auc': (np.mean(auc), np.std(auc)),
            'precision': (np.mean(precision),np.std(precision)),
            'recall': (np.mean(recall),np.std(recall))}

if __name__ == '__main__':

    #Get data config
    config = Config()
    data_files = config.get_datafiles('us_130')
    file_path = data_files['diabetic_data.csv']

    #Read data
    data = pd.read_csv(file_path)

    #Pre-process data
    df = process_us_130_csv(data)
    
    #Separate training data and labels
    X, y = df.drop('readmitted_binarized', axis=1) , df['readmitted_binarized']
    
    #Get results from XGB model
    print("\n---------------------")
    print("Fitting XGB model..")
    print("---------------------")
    ebm = get_classifier()
    ebm_scores = train_and_predict(ebm, X, y, visualize=True)

    print(f"\nScores of XGB model:")
    print(f"F1: Mean: {ebm_scores['f1'][0]} \t \
          Std. Deviation: {ebm_scores['f1'][1]}")
    print(f"AUC: Mean: {ebm_scores['auc'][0]} \t \
          Std. Deviation: {ebm_scores['auc'][1]}")
    print(f"Precision: Mean: {ebm_scores['precision'][0]} \t \
          Std. Deviation: {ebm_scores['precision'][1]}")
    print(f"Recall: Mean: {ebm_scores['recall'][0]} \t \
          Std. Deviation: {ebm_scores['recall'][1]}")

    #Get results from EBM model
    print("\n---------------------")
    print("Fitting EBM model..")
    print("---------------------")
    ebm_explain = get_classifier(explainable=True)
    ebm_explain_scores = train_and_predict(ebm_explain, X, y, visualize=True)

    print(f"\nScores of Explaining Boosting Classifier:")
    print(f"F1: Mean: {ebm_explain_scores['f1'][0]} \t  \
          Std. Deviation: {ebm_explain_scores['f1'][1]}")
    print(f"AUC: Mean: {ebm_explain_scores['auc'][0]} \t \
          Std. Deviation: {ebm_explain_scores['auc'][1]}")
    print(f"Precision: Mean: {ebm_explain_scores['precision'][0]} \t \
          Std. Deviation: {ebm_explain_scores['precision'][1]}")
    print(f"Recall: Mean: {ebm_explain_scores['recall'][0]} \t \
          Std. Deviation: {ebm_explain_scores['recall'][1]}")
