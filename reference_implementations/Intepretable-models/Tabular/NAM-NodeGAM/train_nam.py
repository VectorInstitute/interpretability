import argparse
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader, TensorDataset

from data import Config, process_us_130_csv
from interp.utils import *
from interp.models import NeuralAdditiveModel, ExULayer, ReLULayer
from data.data_utils import split_dataset


# Some parts were extracted from: https://github.com/kherud/neural-additive-models-pt/tree/master

def get_data_loaders(X: torch.tensor,
                    y: torch.tensor,
                    batch: int) -> Tuple[DataLoader]:
    """
    """
    train_test_split = split_dataset(X, y,
                                    n_splits=1,
                                    stratified=True)
                                                
    (x_train, y_train), (x_test, y_test) = next(train_test_split)

    #Get train val split
    train_val_split = split_dataset(x_train, y_train,
                                    n_splits=1,
                                    stratified=True)
    
    (x_train, y_train), (x_val, y_val) = next(train_val_split)

    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True)
    test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size=batch, shuffle=True)
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=batch, shuffle=True)

    return train_dl, test_dl, val_dl

def get_predictions(model, X):
    """
    """
    model.eval()
    x_true, x_probs = [], []
    with torch.set_grad_enabled(False):
        for _, (features,labels) in enumerate(X):
            features, labels = features.cuda(), labels.cuda()
            logits, _ = model.forward(features)
            prob = torch.sigmoid(logits)

            x_true.extend(labels.tolist())
            x_probs.extend(prob.tolist())
    
    threshold = 0.5
    x_preds = list(map(lambda x: x >= threshold, x_probs))

    return x_true, x_probs, x_preds

def train_and_predict(model, cfgs, train, test, val):
    """
    """
    y_tr_true, y_tr_probs, y_tr_preds = [], [], []
    y_te_true, y_te_probs, y_tr_preds = [], [], []
    y_val_true, y_val_probs, y_val_preds = [], [], []
    auc, f1, precision, recall = [], [], [], []

    classifier_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1) #Add step size here
    optimizer.zero_grad()

    for epoch in range(cfgs.train_params.epochs):
        model.train()

        for _,(features,labels) in enumerate(train): 
            features, labels= features.cuda(),labels.cuda()
            logits, _ = model.forward(features)
            prob = torch.sigmoid(logits)
            loss = classifier_criterion(logits, labels)
        
            model.zero_grad()
            loss.backward()
            optimizer.step() 
         
            y_tr_true.extend(labels.tolist())
            y_tr_preds.extend(prob.tolist())
        
        scheduler.step()
        y_val_true, y_val_probs, y_val_preds = get_predictions(model, val)
        y_te_true, y_te_probs, y_te_preds = get_predictions(model, test)

        val_auc = roc_auc_score(y_val_true, y_val_probs)
        train_auc = roc_auc_score(y_tr_true, y_tr_preds)

        print(f"\n AUC scores for epoch {epoch}: \
              Training AUC: {train_auc}, Validation AUC: {val_auc}")
  
        auc.append(roc_auc_score(y_te_true, y_te_probs))
        f1.append(f1_score(y_te_true, y_te_preds))
        precision.append(precision_score(y_te_true, y_te_preds))
        recall.append(recall_score(y_te_true, y_te_preds))

    return {'f1': (np.mean(f1), np.std(f1)),
            'auc': (np.mean(auc), np.std(auc)),
            'precision': (np.mean(precision),np.std(precision)),
            'recall': (np.mean(recall),np.std(recall))}

if __name__ == "__main__":

    #Get data config
    config = Config()
    data_file = config.get_datafiles('us_130')['diabetic_data.csv']
    
    #Get NAM model config
    nam_config = Config(yaml_path = 'configs/nam.yaml')

    data = pd.read_csv(data_file)
    df = process_us_130_csv(data)

    #Separate training data and labels
    X, y = df.drop("readmitted_binarized", axis=1), df["readmitted_binarized"]
    X, y = torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32)

    #Get train test val split
    train_dl, test_dl, val_dl = get_data_loaders(X, y, nam_config.train_params.batch_size)


    print("------------------------------")
    print("Training NAM on US_130 dataset")
    print("------------------------------")
 
    #Setup
    device = get_device()
    print(f"Using device: {device}")
    random_seed(42, True)

    shallow_units = calculate_n_units(get_full_data(train_dl),
                                      nam_config.nam.n_basis_functions,
                                      nam_config.nam.units_multiplier)
    
    shallow_layer = ExULayer if nam_config.nam.shallow_layer == "exu" else ReLULayer
    hidden_layer = ExULayer if nam_config.nam.hidden_layer == "exu" else ReLULayer

    model = NeuralAdditiveModel(input_size=X.shape[-1],
                                shallow_units=shallow_units,
                                shallow_layer=shallow_layer,
                                hidden_layer=hidden_layer,
                                hidden_dropout=nam_config.nam.dropout,
                                feature_dropout=nam_config.nam.feature_dropout)
    model = model.cuda()   
    nam_scores = train_and_predict(model, nam_config, train_dl, test_dl, val_dl)

    print(f"\nScores of NAM model:")
    print(f"F1: Mean: {nam_scores['f1'][0]} \t  \
          Std. Deviation: {nam_scores['f1'][1]}")
    print(f"AUC: Mean: {nam_scores['auc'][0]} \t \
          Std. Deviation: {nam_scores['auc'][1]}")
    print(f"Precision: Mean: {nam_scores['precision'][0]} \t \
          Std. Deviation: {nam_scores['precision'][1]}")
    print(f"Recall: Mean: {nam_scores['recall'][0]} \t \
          Std. Deviation: {nam_scores['recall'][1]}")
    
