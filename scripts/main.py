import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datetime import datetime
from model.model import UnetClassifier, ResNetAttention, ProtoNet, text_generation
from model_intepretability.imaging_copy.dataset.dataset import PrototypicalBatchSampler
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP 
from functools import partial
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader,SubsetRandomSampler, TensorDataset
from sklearn.model_selection import KFold
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import torch.nn.functional as F
import pickle
import torchvision.models as models 
import torch.distributed as dist
from utils.utils import visualize_trainable_attention
from losses.losses import prototypical_loss
from tqdm import tqdm
from torch.autograd import Variable
plt.rcParams['figure.figsize'] = [25, 10]
import gradio as gr
from glob import glob
from PIL import Image
from dataset.class_prompts import class_prompts
from model.zeroshot import InferenceModel
import argparse
import gc
from pathlib import Path
from utils.utils import get_roc_auc_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
from dataset.dataset import XrayDataset
from pathlib import Path
import pickle
from utils.zeroshot_utils import calculate_auroc

def make_parser():
    parser = argparse.ArgumentParser(description='Imaging Explainability')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--resize', type=int, default=224, help='Resizing images')
    parser.add_argument('--experiment', type=str, default="baseline", help='Imaging Experiment')

    
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler')
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')

    
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    

    
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--weight_decay', type=int, default=1e-3, help='Seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=20, help='Seed for reproducibility')
    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)
    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=15)
    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)
    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)
    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)
    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=15)
    return parser

splits=KFold(n_splits=5,shuffle=True,random_state=42)
def get_roc_auc_score(y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    with open('all_classes.pkl', 'rb') as all_classes:
        all_classes = pickle.load(all_classes)

    NoFindingIndex = all_classes.get('No Finding', -1)
    

    class_roc_auc_list = []    
    useful_classes_roc_auc_list = []
    
    for i in range(y_true.shape[1]):
        class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        class_roc_auc_list.append(class_roc_auc)
        if i != NoFindingIndex:
            useful_classes_roc_auc_list.append(class_roc_auc)
    return np.mean(np.array(useful_classes_roc_auc_list))

def gather(tensor, tensor_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
  
    rank = dist.get_rank()
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert(tensor_list is not None)
        dist.gather(tensor, gather_list=tensor_list, group=group)
    else:
        dist.gather(tensor, dst=root, group=group)

    

def compute_test_auc(model, test_loader, device_id):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(device_id, non_blocking=True), labels.cuda(device_id, non_blocking=True)
            outputs,_ = model(images)
            preds = torch.sigmoid(outputs) # Assuming binary classification

            # Gather predictions and labels across all processes
            
            gathered_preds = [torch.zeros_like(preds) for _ in range(dist.get_world_size())]
            gathered_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
            
            # Collect predictions and labels from each process
            
            dist.all_gather(gathered_preds,preds)
            dist.all_gather(gathered_labels,labels)
            

            # Only concatenate and store predictions and labels on rank 0
            if dist.get_rank() == 0:
                all_preds.extend([p.cpu() for p in gathered_preds])
                all_labels.extend([l.cpu() for l in gathered_labels])
            

    # On rank 0, compute AUC
    if dist.get_rank() == 0:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        auc = get_roc_auc_score(all_labels.numpy(), all_preds.numpy())
        print(f"AUC: {auc}")
        return auc
   
def aggregate_all_predictions(predictions, labels,device_id):
    

    # Initialize lists to gather predictions and labels from all devices
    
    predictions = torch.tensor(predictions).cuda(device_id)
    labels = torch.tensor(labels).cuda(device_id)
    
    gathered_preds = [torch.zeros_like(predictions) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]

    # Use all_gather to collect all data across devices
    dist.all_gather(gathered_preds, predictions)
    dist.all_gather(gathered_labels, labels)

    # Concatenate data across devices
    all_preds = torch.cat(gathered_preds)
    all_labels =torch.cat(gathered_labels)

    # Convert to numpy for AUC calculation
    return all_preds.cpu().numpy(), all_labels.cpu().numpy()

def load_checkpoint(model, optimizer,file_path):
    checkpoint = torch.load(file_path)
   
    checkpoint_state_dict = checkpoint['model_state_dict']

   
    # Create a new state_dict with only the matching keys
    checkpoint_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_state_dict.items()}
    
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model.state_dict()}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int) -> None:
    """Initialize worker processes with a random seed.

    Parameters
    ----------
    worker_id : int
        ID of the worker process.
    num_workers : int
        Total number of workers that will be initialized.
    rank : int
        The rank of the current process.
    seed : int
        A random seed used determine the worker seed.
    """
    worker_seed = num_workers * rank + worker_id + seed
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    
def save_checkpoint(model, optimizer, epoch, loss,file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)

def setup() -> None:
    """Initialize the process group."""
    dist.init_process_group("nccl")


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()


def train_attention(args, train_ds,test_ds):

    
    
    batch_size=args.batch_size
    

    classifier_criterion = nn.BCEWithLogitsLoss()

    global f1_list,auc_list, precision_list, recall_list
    setup()
    init_fn = partial(
        worker_init_fn,
        num_workers=args.num_workers,
        rank=dist.get_rank(),
        seed=args.rseed,
    )
    
    test_sampler = DistributedSampler(test_ds, shuffle=True)  

    test_dl = DataLoader(test_ds,sampler=test_sampler, batch_size=args.batch_size,worker_init_fn=init_fn)
    
    # setup()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.empty_cache()
    device_id = torch.cuda.current_device()
    
    
   
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_ds)))):
   
        print("fold:", fold)
        
        print('Fold {}'.format(fold + 1))
        
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(train_ds, val_idx)
        train_sampler = DistributedSampler(train_subset, shuffle=True)  
        valid_sampler = DistributedSampler(val_subset, shuffle=True)
        init_fn = partial(worker_init_fn,num_workers=args.num_workers,rank=dist.get_rank(),seed=args.rseed)
        train_dl = DataLoader(train_ds, sampler=train_sampler,batch_size=args.batch_size,worker_init_fn=init_fn,pin_memory=False,drop_last=True,num_workers=args.num_workers)
        valid_dl = DataLoader(train_ds, sampler=valid_sampler, batch_size=args.batch_size,worker_init_fn=init_fn,pin_memory=False,drop_last=True,num_workers=args.num_workers)
    
    
        
        

        model2 = ResNetAttention(models.resnet18(pretrained=True))
        model2 = model2.cuda(device_id)
        model2 = DDP(model2, device_ids=[device_id])
        optimizer = optim.AdamW(model2.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        
        try:
            model2, optimizer, start_epoch, _ = load_checkpoint(model2, optimizer,"output_weight/resnet.pth")
            print(f"Resuming from epoch {start_epoch + 1}")
        except FileNotFoundError:
            print("No checkpoint found, starting frsom scratch")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        optimizer.zero_grad()
        
        
        
        for epoch in range(args.num_epochs):
            if epoch<=20:
                continue
            model2.train()
            num_batches=0
            
            train_loader_examples_num = len(train_dl.dataset)
            training_estimated = np.zeros((train_loader_examples_num, 15), dtype = np.float32)
            training_ture  = np.zeros((train_loader_examples_num, 15), dtype = np.float32)
            k=0
            for num_batches,(images,labels) in enumerate(train_dl): 
                
                
                optimizer.zero_grad()
                images = images.cuda(device_id, non_blocking=True)
                labels = labels.cuda(device_id, non_blocking=True)
                
                logits,attention_weights= model2(images.float())
                
                prob = torch.sigmoid(logits)
                
                loss = classifier_criterion(logits, labels) 
            
                model2.zero_grad()
                loss.backward()
                

                
                optimizer.step()    
                if (num_batches+1)%8==0:
                    save_checkpoint(model2, optimizer, epoch, loss.item(), file_path=f"output_weight/resnet_backup.pth")
                    
                save_checkpoint(model2, optimizer, epoch, loss.item(), file_path=f"output_weight/resnet.pth")          
                
                training_estimated[k: k + prob.shape[0], :] = prob.detach().cpu().numpy()
                training_ture[k: k + prob.shape[0], :] = labels.detach().cpu().numpy()
                k += prob.shape[0]

    

            
            model2.eval()
            
            with torch.set_grad_enabled(False):

                
                val_loader_examples_num = len(valid_dl.dataset)
                validation_estimated = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
                validation_true  = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
                n = 0
                
                k = 0
                for val_batches,(images,labels) in enumerate(valid_dl):
                    
                    labels =  labels.cuda(device_id, non_blocking=True)
                    
                    
                    images = images.cuda(device_id, non_blocking=True)
                    logits,attention_weights = model2(images)
                    
                    
                    
                    prob = torch.sigmoid(logits)
                    
                    validation_estimated[k: k + prob.shape[0], :] = prob.detach().cpu().numpy()
                    validation_true[k: k + prob.shape[0], :] = labels.detach().cpu().numpy()
                    k += prob.shape[0]
        
            val_auc = get_roc_auc_score(validation_true, validation_estimated)  
            
            train_auc = get_roc_auc_score(training_ture, training_estimated)   

            

            print("epoch",epoch,":","train_AUC:",train_auc,"val_AUC",val_auc)
            model2.eval()
        with torch.set_grad_enabled(False):

            
            test_loader_examples_num = len(test_dl.dataset)
            test_estimated = np.zeros((test_loader_examples_num, 15), dtype = np.float32)
            test_true  = np.zeros((test_loader_examples_num, 15), dtype = np.float32)
            k = 0
            
            all_predictions, all_labels = [], []
            for images,labels in test_dl:
                
                
                images, labels = images.cuda(device_id, non_blocking=True), labels.cuda(device_id, non_blocking=True)
                
           
                
                logits,attention_weights = model2(images)
                
                logits = torch.squeeze(logits)
                
                prob = torch.sigmoid(logits)
                
                threshold = 0.5
                pred =(prob >= threshold).float() 
                visualize_trainable_attention(attention_weights[0],images[0],labels[0],pred[0])
                
                test_estimated[k: k + prob.shape[0], :] = prob.detach().cpu().numpy()
                test_true[k: k + prob.shape[0], :] = labels.detach().cpu().numpy()

                k += prob.shape[0]
                
                all_predictions = test_estimated
                all_labels = test_true
                
            
            final_preds, final_labels = aggregate_all_predictions(all_predictions, all_labels,device_id)

            # Calculate AUC on rank 0
            if dist.get_rank() == 0:
                auc = get_roc_auc_score( final_labels, final_preds)
                print(f"Test AUC: {auc}")
    logging.info('Finished training.')
    
    dist.destroy_process_group()
    return 0

def train_prototype(args, train_ds,test_ds):

   

    global f1_list,auc_list, precision_list, recall_list
    setup()
    init_fn = partial(
        worker_init_fn,
        num_workers=args.num_workers,
        rank=dist.get_rank(),
        seed=args.rseed,
    )
    
    test_sampler = DistributedSampler(test_ds, shuffle=True)  

    test_dl = DataLoader(test_ds,sampler=test_sampler, batch_size=args.batch_size,worker_init_fn=init_fn)
    
    # setup()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.empty_cache()
    device_id = torch.cuda.current_device()
    
    
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_ds)))):
  
        print("fold:", fold)
        
        print('Fold {}'.format(fold + 1))
        
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(train_ds, val_idx)
        train_sampler = DistributedSampler(train_subset, shuffle=True)  
        valid_sampler = DistributedSampler(val_subset, shuffle=True)
        classes_per_it_tr = args.classes_per_it_tr
        num_samples_tr = args.num_support_tr + args.num_query_tr
    
        classes_per_it_val = args.classes_per_it_val
        num_samples_val = args.num_support_val + args.num_query_val
        
        classes = train_ds.csv_file["numeric_targets"]
       
        train_sampler = PrototypicalBatchSampler(labels=classes,
                                    classes_per_it=classes_per_it_tr,
                                    num_samples=num_samples_tr,
                                    iterations=args.iterations)
        valid_sampler = PrototypicalBatchSampler(labels=classes,
                                    classes_per_it=classes_per_it_val,
                                    num_samples=num_samples_val,
                                    iterations=args.iterations)
        
        init_fn = partial(worker_init_fn,num_workers=args.num_workers,rank=dist.get_rank(),seed=args.rseed)
        train_dl = DataLoader(train_ds, sampler=train_sampler,batch_size=args.batch_size,worker_init_fn=init_fn,pin_memory=False,drop_last=True,num_workers=args.num_workers)
        valid_dl = DataLoader(train_ds, sampler=valid_sampler, batch_size=args.batch_size,worker_init_fn=init_fn,pin_memory=False,drop_last=True,num_workers=args.num_workers)
        model2 = ProtoNet()
        model2 = model2.cuda(device_id)
        model2 = DDP(model2, device_ids=[device_id])#,find_unused_parameters=True)
        
        
        optimizer = optim.AdamW(model2.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        try:
            model2, optimizer, start_epoch, _ = load_checkpoint(model2, optimizer,"output_weight/proto.pth")
            print(f"Resuming from epoch {start_epoch + 1}")
        except FileNotFoundError:
            print("No checkpoint found, starting frsom scratch")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        optimizer.zero_grad()
        
        
        train_auc = []
        val_auc = []
        for epoch in range(args.num_epochs):
            
            model2.train()
            num_batches=0
            
            k=0
            tr_iter = iter(train_dl)
            
            for batch in tqdm(tr_iter):  
                images,labels = batch
                optimizer.zero_grad()
                images = images.cuda(device_id, non_blocking=True)
                labels = labels.cuda(device_id, non_blocking=True)
                image_rep= model2(images.float())
                active_classes = torch.nonzero(labels.sum(0)).squeeze(1)
                loss, auc,target,prob = prototypical_loss(images,image_rep, target=labels,
                                n_support=args.num_support_tr)
                
                train_auc.append(auc)
                
                loss = Variable(loss, requires_grad = True)
                loss.backward() 
                
                optimizer.step()    
                if (num_batches+1)%100==0:
                    save_checkpoint(model2, optimizer, epoch, loss.item(), file_path=f"output_weight/proto_backup.pth")
                    
                if (num_batches+1)%200==0:
                    save_checkpoint(model2, optimizer, epoch, loss.item(), file_path=f"output_weight/proto.pth")          
                
                break
            avg_auc = np.mean(train_auc[-args.iterations:])   #??????

            print('Avg Train AUC: {}'.format(avg_auc))
            model2.eval()
            
            with torch.set_grad_enabled(False):
                
                k = 0
                for val_batches,(images,labels) in enumerate(valid_dl):
                    
                    labels =  labels.cuda(device_id, non_blocking=True)
                    
                    
                    images = images.cuda(device_id, non_blocking=True)
                    image_rep = model2(images)
                    
                    _, auc,target,prob = prototypical_loss(images,image_rep, target=labels,
                                n_support=args.num_support_val)
                    
                    val_auc.append(auc)#.item())
                    
                    k += prob.shape[0]
        
                avg_val_auc = np.mean(val_auc[-args.iterations:]) 

      
            print("epoch",epoch,":","train_AUC:",avg_auc,"val_AUC",avg_val_auc)
            model2.eval()
        with torch.set_grad_enabled(False):

            
            k = 0
            
            avg_acc = []
            for epoch in range(10):
                for images,labels in test_dl:
                    
                    
                    images, labels = images.cuda(device_id, non_blocking=True), labels.cuda(device_id, non_blocking=True)
                            
                    image_rep = model2(images)
                    _, auc,target,prob = prototypical_loss(images,image_rep, target=labels,
                                n_support=args.num_support_val)
                    avg_acc.append(auc.item())
                

            avg_acc = np.mean(avg_acc)
            print('Test Acc: {}'.format(avg_acc))

        

            

       


    logging.info('Finished training.')
   
    dist.destroy_process_group()
    return 0

# Some parts were extracted from https://github.com/ChantalMP/Xplainer?tab=readme-ov-file
def train_zeroshot(test_ds):

    dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(class_prompts)

    all_labels = []
    all_probs_neg = []

    for images, labels, keys in tqdm(dataloader):
        
        agg_probs = []
        agg_negative_probs = []
        
        image_paths = [Path(image_path) for image_path in images]
        for image_path in image_paths:
            
            probs, negative_probs = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
            agg_probs.append(probs)
            agg_negative_probs.append(negative_probs)

        probs = {}  # Aggregated
        negative_probs = {}  # Aggregated
        for key in agg_probs[0].keys():
            probs[key] = sum([p[key] for p in agg_probs]) / len(agg_probs)  # Mean Aggregation

        for key in agg_negative_probs[0].keys():
            negative_probs[key] = sum([p[key] for p in agg_negative_probs]) / len(agg_negative_probs)  # Mean Aggregation

        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(class_prompts, pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(class_prompts,
                                                                                                   disease_probs=disease_probs,
                                                                                                   negative_disease_probs=negative_disease_probs,
                                                                                                   keys= agg_probs[0].keys())
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        
    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    # evaluation
    all_labels = all_labels.squeeze(1)
    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    

    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    
    print(f"AUROC: {overall_auroc:.5f}\n")

if __name__ == '__main__':
    args = make_parser().parse_args()
    auc_list , precision_list, recall_list, f1_list = [], [], [], []
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)   
    torch.cuda.manual_seed(args.rseed)
    torch.cuda.manual_seed_all(args.rseed)

    image_path = "/datasets/nih-chest-xrays"
    
    csv_file = pd.read_csv(os.path.join(image_path,"Data_Entry_2017.csv"))
    test_split = os.path.join(image_path,"test_list.txt")
    train_val_split = os.path.join(image_path,"train_val_list.txt")
    with open(train_val_split, 'r') as f:
        train_val_images = f.read().splitlines()
    with open(test_split, 'r') as f:
        test_images = f.read().splitlines()

    train_df = csv_file[csv_file['Image Index'].isin(train_val_images)]
    
    test_df = csv_file[csv_file['Image Index'].isin(test_images)]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_ds = XrayDataset(train_df, image_path)
    test_ds = XrayDataset(test_df, image_path)
    


    ### text generation
    # image_name = csv_file.loc[1,'Image Index']
    #     # image_path = os.path.join(self.path_name,  self.csv_file['path'].iloc[idx])
    # image_path = glob(os.path.join(image_path, '**', image_name), recursive=True)[0]
    
    # image2=Image.open(image_path)
    # model = text_generation()
    # gen_text = model.generate_caption(image2)

    if args.experiment=="attention": 
        train_attention(args, train_ds,test_ds)
    elif args.experiment=="prototype": 
        train_prototype(args, train_ds,test_ds)
    elif args.experiment=="concept": 
        image_path = "/datasets/nih-chest-xrays"    
        csv_file = pd.read_csv(os.path.join(image_path,"Data_Entry_2017.csv"))
        test_split = os.path.join(image_path,"test_list.txt")
        with open(test_split, 'r') as f:
            test_images = f.read().splitlines()
            test_df = csv_file[csv_file['Image Index'].isin(test_images)]
            test_df.reset_index(drop=True, inplace=True)
            test_ds = XrayDataset(test_df, image_path)
            train_zeroshot(test_ds)
            


    




    
   
