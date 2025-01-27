# NAM 

import os
import sys
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader,SubsetRandomSampler, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
import pickle
import stat
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import NeuralAdditiveModel, calculate_n_units, ExULayer, ReLULayer
from new_data import dfDataset, process_csv, create_test_train_fold, create_test_train_fold_embed, split_training_dataset
from losses import penalized_cross_entropy

def make_parser():
    parser = argparse.ArgumentParser(description='Tabular Deep Learning')
    parser.add_argument('--output_dir', type=str, default='output_weight', help='Output directory')
    
    parser.add_argument('--batch_size', type=int, default=512, help='batch size') 
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs') 
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate') 
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler') 
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')
    parser.add_argument('--feature_dropout', type=int, default=0, help='scheduler step size')
    
    parser.add_argument('--dropout', type=float, default=0, help='dropout') 
    parser.add_argument('--n_basis_functions', type=float, default=1000) # 1000
    parser.add_argument('--units_multiplier', type=float, default=64) #2
    parser.add_argument('--hidden_units', type=list, default=[]) 
    parser.add_argument('--output_regularization', type=float, default=0.0)
    # Misc
    parser.add_argument('--gpus', type=str, default='0', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--weight_decay', type=float, default=0.0) #0.05
    parser.add_argument('--shallow_layer', type=str, default="relu")
    parser.add_argument('--hidden_layer', type=str , default = "relu")
    return parser

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) 
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(42, True)

splits=KFold(n_splits=5,shuffle=True,random_state=42)

def train_nam(args,data_ds):

    batch_size=args.batch_size
    classifier_criterion = penalized_cross_entropy
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
    outer_kfold = KFold(n_splits=5, shuffle=True,random_state=42)
    for test_fold, (outer_train_indices,outer_test_indices) in enumerate(outer_kfold.split(data_ds)):
        print(f"test_fold:{test_fold}")
        outer_train_set = torch.utils.data.Subset(data_ds, outer_train_indices)
        outer_test_set = torch.utils.data.Subset(data_ds, outer_test_indices)
    
        
        test_dl = DataLoader(outer_test_set, batch_size=batch_size)
        inner_kfold = KFold(n_splits=5, shuffle=True,random_state=42)

        file = open('test_dataset_MRI', 'wb')

        # # dump information to that file
        pickle.dump(outer_test_set, file)

        # # close the file
        file.close()
        
        for valid_fold, (inner_train_indices, inner_val_indices) in enumerate(splits.split(outer_train_set)):
    
            print('Fold {}'.format(valid_fold + 1))
            inner_train_set = torch.utils.data.Subset(outer_train_set, inner_train_indices)
            inner_val_set = torch.utils.data.Subset(outer_train_set, inner_val_indices)
            n_units = calculate_n_units(inner_train_set, args.n_basis_functions, args.units_multiplier)
            train_dl =DataLoader(inner_train_set, batch_size=batch_size, shuffle=True)        
            valid_dl = DataLoader(inner_val_set, batch_size=batch_size, shuffle=True)
            

            model = NeuralAdditiveModel(
                input_size=x_train.shape[-1],
                shallow_units=n_units,
                hidden_units=list(map(int, args.hidden_units)),
                shallow_layer=ExULayer if args.shallow_layer == "exu" else ReLULayer,
                hidden_layer=ExULayer if args.hidden_layer == "exu" else ReLULayer,
                hidden_dropout=args.dropout,
                feature_dropout=args.feature_dropout)

            model = model.cuda()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
            optimizer.zero_grad()
            
            for epoch in range(args.num_epochs):
                model.train()
                training_ture=[]
                training_estimated=[]
    
                for num_batches,(features,labells) in enumerate(train_dl): 
            
                    features ,labels= features.cuda(),labels.cuda()
                

                    
                    model = model.cuda()

                    
                    logits, fnns_out = model.forward(features)
                    prob = torch.sigmoid(logits)
                    loss = classifier_criterion(logits, labels, fnns_out, feature_penalty=args.output_regularization)
                
                    model.zero_grad()
                    loss.backward()
                    
                    
                    optimizer.step() 
                    
                    
                    for i in range(len(labells.tolist())):
                        training_ture.append(labels.tolist()[i])
                        training_estimated.append(prob.tolist()[i])
                

                scheduler.step()
                model.eval()
                with torch.set_grad_enabled(False):

                    validation_true = []
                    validation_estimated = []
                    
                    for val_batches,(features,labels) in enumerate(valid_dl):
                        
                        features, labels = features.cuda(), labels.cuda()
                        
            
                        logits, fnns_out = model.forward(features)
    
                        
                
                        
                        prob = torch.sigmoid(logits)
                        
                        for i in range(len(labels.tolist())):
                            validation_true.append(labels.tolist()[i])
                            validation_estimated.append(prob.tolist()[i])

       
                    
            
                val_auc = roc_auc_score(validation_true, validation_estimated)
                
                train_auc = roc_auc_score(training_ture, training_estimated)

                total_train_auc[epoch] = train_auc
                total_val_auc[epoch] = val_auc
                print("epoch",epoch,":","train_AUC:",train_auc,"val_AUC",val_auc)
            model.eval()
            with torch.set_grad_enabled(False):

                
                test_true = []
                test_estimated = []
                
                
                for features,labels in test_dl:
                    

                    features, labels = features.cuda(),labels.cuda()
                    
            
                    logits, fnns_out = model.forward(features)
                
                    prob = torch.sigmoid(logits)  
                    for i in range(len(labels.tolist())):
                        test_true.append(labels.tolist()[i])
                        test_estimated.append(prob.tolist()[i])
                    
                    torch.save(model.state_dict(), os.path.join(args.output_dir,f"nam_new_data__{test_fold}"))
                test_auc = roc_auc_score(test_true, test_estimated)
                print("test_auc",test_auc)
            


    logging.info('Finished training.')
    return 0, total_train_auc, 0, total_val_auc



# Some parts were extracted from: https://github.com/kherud/neural-additive-models-pt/tree/master

def train_model(args,x_train,y_train,x_val,y_val,test_dl):
    global f1_list,auc_list, precision_list, recall_list
    classifier_criterion = nn.BCEWithLogitsLoss() 
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
    model = NeuralAdditiveModel(
        input_size=x_train.shape[-1],
        shallow_units=calculate_n_units(x_train, args.n_basis_functions, args.units_multiplier),
        hidden_units=list(map(int, args.hidden_units)),
        shallow_layer=ExULayer if args.shallow_layer == "exu" else ReLULayer,
        hidden_layer=ExULayer if args.hidden_layer == "exu" else ReLULayer,
        hidden_dropout=args.dropout,
        feature_dropout=args.feature_dropout).cuda()
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    valid_dl = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True)


    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
    optimizer.zero_grad()
    for epoch in range(args.num_epochs):
        model.train()
        training_ture=[]
        training_estimated=[]

        for num_batches,(features,labels) in enumerate(train_dl): 
    
            features ,labels= features.cuda(),labels.cuda()
        

            
            model = model.cuda()

            logits, fnns_out = model.forward(features)
            prob = torch.sigmoid(logits)
            loss = classifier_criterion(logits, labels)
        
            model.zero_grad()
            loss.backward()
            
            
            optimizer.step() 
         
            
            for i in range(len(labels.tolist())):
                training_ture.append(labels.tolist()[i])
                training_estimated.append(prob.tolist()[i])
        

        scheduler.step()
        model.eval()
        with torch.set_grad_enabled(False):

        
            validation_true = []
            validation_estimated = []
            
            for val_batches,(features,labels) in enumerate(valid_dl):
                
                features, labels = features.cuda(), labels.cuda()
                
    
                logits, fnns_out = model.forward(features)

                prob = torch.sigmoid(logits)
                
                for i in range(len(labels.tolist())):
                    validation_true.append(labels.tolist()[i])
                    validation_estimated.append(prob.tolist()[i])


        val_auc = roc_auc_score(validation_true, validation_estimated)
        
        train_auc = roc_auc_score(training_ture, training_estimated)

        


    

        total_train_auc[epoch] = train_auc
        total_val_auc[epoch] = val_auc
        print("epoch",epoch,":","train_AUC:",train_auc,"val_AUC",val_auc)
        model.eval()
    with torch.set_grad_enabled(False):

        
        test_true = []
        test_estimated = []
        test_pred = []
        
        
        for features,labels in test_dl:
            

            features, labels = features.cuda(),labels.cuda()
            
    
            logits, fnns_out = model.forward(features)
        
            prob = torch.sigmoid(logits)
            threshold = 0.5
            pred =(prob >= threshold).float() 
            for i in range(len(labels.tolist())):
                test_true.append(labels.tolist()[i])
                test_estimated.append(prob.tolist()[i])
                test_pred.append(pred.tolist()[i])
            
        test_auc = roc_auc_score(test_true, test_estimated)
        test_f1 = f1_score(test_true,test_pred)
        test_precision = precision_score(test_true,test_pred)
        test_recall = recall_score(test_true,test_pred)

        print("test_auc",test_auc)
        auc_list.append(test_auc)
        f1_list.append(test_f1)
        precision_list.append(test_precision)
        recall_list.append(test_recall)


        


    logging.info('Finished training.')
    return 0, total_train_auc, 0, total_val_auc






# Some parts were extracted from: https://github.com/kherud/neural-additive-models-pt/tree/master
if __name__ == "__main__":

    args = make_parser().parse_args()
    data = pd.read_csv("df.csv")
    df = process_csv(data)
    df.index=range(df.shape[0])
    
    auc_list , precision_list, recall_list, f1_list = [], [], [], []

    for i in range(5):
        print (f"test_fold{i}")
        file = open(f'x_train_{i}', 'rb')
        x_train_all = pickle.load(file)
        file.close()
        file = open(f'y_train_{i}', 'rb')

        y_train_all = pickle.load(file)
        file.close()
        file = open(f'x_test_{i}', 'rb') 
        x_test = pickle.load(file)

        file.close()
        file = open(f'y_test_{i}', 'rb') 
        


        y_test = pickle.load(file)

        file.close()
       
        train = split_training_dataset(
            x_train_all.values.astype('float32'),
            y_train_all.values.astype('float32'),
            n_splits=1,
            stratified=True)
        test_dataset = TensorDataset(torch.tensor(x_test.values.astype('float32')), torch.tensor(y_test.values.astype('float32')))
        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        logging.info("begin training")
        test_scores = []
        
        while True:
            try:
                (x_train, y_train), (x_val, y_val) = next(train)
                _,_,_,_ = train_model(args,x_train, y_train, x_val, y_val,test_dl)
            except StopIteration:
                break
    print("mean f1",np.mean(f1_list),np.std(f1_list))
    print("mean auc",np.mean(auc_list),np.std(auc_list))
    print("mean precision",np.mean(precision_list),np.std(precision_list))
    print("mean recall",np.mean(recall_list),np.std(recall_list))
    
