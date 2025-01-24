
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
# from model import NeuralAdditiveModel, calculate_n_units, ExULayer, ReLULayer, TabNet, NAM_2, TabTransformer, Preprocessor
# from new_data import dfDataset, process_csv, create_test_train_fold, create_test_train_fold_embed, split_training_dataset, process_csv_transformer, TabDataset
# from losses import penalized_cross_entropy
##
from functools import partial
import torch.distributed as dist
from sklearn.preprocessing import OrdinalEncoder
from model.tabnet import  report_ehr_model_combined
from sklearn.model_selection import train_test_split
from dataset.multimodal_dataset import multimodaldataset,process_csv_transformer, process_text
from utils.losses import CLIPLoss, cosine_similarity, precision_at_k
from pytorch_tabnet.pretraining import TabNetPretrainer
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data.distributed import DistributedSampler
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch SickKids Brain MRI')

    
    parser.add_argument('--data_path', type=str, default="df.csv", help='Data path')
    parser.add_argument('--output_dir', type=str, default='output_weight', help='Output directory')
    
    parser.add_argument('--batch_size', type=int, default=64, help='batch size') 
    parser.add_argument('--num_epochs', type=int, default=15, help='number of epochs') #0.0003
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate') #5e-2  #1e-3 #compare:5e-6 #6e-6 (best till now) 0.0002
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler') ##true?
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')
    parser.add_argument('--feature_dropout', type=int, default=0, help='scheduler step size')
    
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout') 
    parser.add_argument('--n_basis_functions', type=float, default=1000) # 1000
    parser.add_argument('--units_multiplier', type=float, default=32) #2
    parser.add_argument('--hidden_units', type=list, default=[]) 
    parser.add_argument('--output_regularization', type=float, default=0.0)
    # Miscs
    parser.add_argument('--gpus', type=str, default='0', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--weight_decay', type=float, default=0.0001) #0.05
    parser.add_argument('--shallow_layer', type=str, default="relu")
    parser.add_argument('--hidden_layer', type=str , default = "relu")
    parser.add_argument("--distributed_mode",action="store_true",help="Enable distributed training",default=True)
    parser.add_argument("--distributed_launcher", default="slurm")
    parser.add_argument("--distributed_backend", default="nccl")
    parser.add_argument("-j","--num_workers",default=4,type=int,metavar="N",help="number of data loading workers (default: 32)")
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

splits=KFold(n_splits=5,shuffle=True,random_state=42)

def save_checkpoint(model, optimizer, epoch, loss,file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)

def load_checkpoint(model, optimizer,file_path):
    checkpoint = torch.load(file_path)
    # print("checkpointttttttt",checkpoint['model_state_dict'].keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
def setup() -> None:
    """Initialize the process group."""
    dist.init_process_group("nccl")#,rank=dist.get_rank(), init_method='env://'


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()


def train_global_local_model(data_ds,numerical_column, categorical_column, num_embeddings,cat_idxs, cat_dims):
    print("availabke",torch.cuda.is_available()) 
    # dist.init_process_group("nccl", rank=1, world_size=2)
    # os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    # os.environ['MASTER_PORT'] = '12355'  # some free port
    
    # torch.distributed.launch
    # dist.init_process_group(backend="gloo", init_method='env://', world_size = 1, rank =0)#, timeout=datetime.timedelta(minutes=60))
    # dist.init_process_group(backend='nccl', init_method='env://', world_size = 2, rank = 1)#, timeout=datetime.timedelta(minutes=60))

    num_batch_accumulate = 8
    model_config = {
    'cat_embedding_dim': 12,
    'num_transformer_blocks': 4,#4,
    'num_heads': 3,#3,
    'tf_dropout_rates': [0., 0., 0., 0.,],
    'ff_dropout_rates': [0., 0., 0., 0.,],
    'mlp_dropout_rates': [0.2, 0.1], 
    'mlp_hidden_units_factors': [2, 1],
    }
    emb_dim = model_config['cat_embedding_dim']
    num_transformer_blocks = model_config['num_transformer_blocks']
    num_heads = model_config['num_heads']
    attn_dropout_rates = model_config['tf_dropout_rates']
    ff_dropout_rates = model_config['ff_dropout_rates']
    mlp_dropout_rates = model_config['mlp_dropout_rates']
    mlp_hidden_units_factors = model_config['mlp_hidden_units_factors']

    global f1_list,auc_list, precision_list, recall_list
    batch_size=args.batch_size
    total_train_auc={}
    total_val_auc={}
    test_auc=[]

    if args.distributed_mode:
        # dist.init_distributed_mode(
        #     launcher=args.distributed_launcher,
        #     backend=args.distributed_backend,
        # )
        setup()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.cuda.empty_cache()
        device_id = torch.cuda.current_device()
    else:
        device_id = None
    
    init_fn = partial(
        worker_init_fn,
        num_workers=args.num_workers,
        rank=dist.get_rank(),
        seed=args.rseed,
    )
    # test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    #    print(train_idx,val_idx)
        print("fold:", fold)
        if fold>0: 
            continue
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        sampler = DistributedSampler(data_ds)  
        init_fn = partial(worker_init_fn,num_workers=args.num_workers,rank=dist.get_rank(),seed=args.rseed)
        train_dl = DataLoader(data_ds, sampler=sampler,batch_size=args.batch_size,worker_init_fn=init_fn,pin_memory=False,
        drop_last=True,num_workers=args.num_workers)
        valid_dl = DataLoader(data_ds, batch_size=args.batch_size, sampler=test_sampler,drop_last=True)
        num_cat_cols, num_num_cols = len(categorical_column), len(numerical_column)
        
        model2 = report_ehr_model_combined(
                 num_cat_cols, num_num_cols,cat_idxs, cat_dims
                 )
        model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model2)
        # set the single device scope, otherwise DistributedDataParallel will
        # use all available devices
        # torch.cuda.set_device(device_id)
        model2 = model2.cuda(device_id)
        model2 = DDP(model2, device_ids=[device_id],find_unused_parameters=True)
        for name, param in model2.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory (GB): {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}")
            print(f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
        
        
        ita_list=[]
        ita_list_val=[]
        local_list=[]
        cl_list=[]
        # params = list(model.parameters())#list(preprocessor.parameters()) + 
        optimizer = optim.AdamW(model2.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        # print("checkpointttttttt",preprocessor.state_dict().keys())


        # try:
        #     model2, optimizer, start_epoch, _ = load_checkpoint(model2, optimizer,"output_weight/tabnet_model_checkpoint_corrected_backup_notes_wt.pth")
        #     # preprocessor, _, _, _ = load_checkpoint(preprocessor, optimizer,"output_weight/preprocessor_checkpoint.pth")
        #     print(f"Resuming from epoch {start_epoch + 1}")
        # except FileNotFoundError:
        #     print("No checkpoint found, starting frsom scratch")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        optimizer.zero_grad()
        cl_list=[]
        for epoch in range(args.num_epochs):
            # preprocessor.train()
            model2.train()
            train_loss = 0
            counter = 0
            num_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
            train_loss = 0
            counter = 0
            num_batches=0
            num_batches_valid=0
            val_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]

            
            report_embeddings = []
            ehr_embeddings=[]
            
            i2t_corr_tr=0
            batch_epoch_tr=0
            i2t_corr_val=0
            t2i_corr_tr=0
            t2i_corr_val=0
            batch_epoch_val=0
            precision_list=[]
            loc_list = []
           
            print("epoch",epoch)
            for num_batches,(text,ehr,num_embeddings) in enumerate(train_dl): 
        
                # features ,labels= features.cuda(),labels.cuda()
                optimizer.zero_grad()
                
                ehr = ehr.squeeze(2)

                # x_nums, x_cats = preprocessor(ehr, numerical_column, categorical_column)
                # # print("leennnnnnnnnnnnnn",len(x_nums))
                # print("feature shape",ehr.shape)
                
                # # x_nums, x_cats= x_nums.cuda(), x_cats.cuda()
                # # model = model.cuda()
                # x_nums , x_cats= x_nums.to(0) , x_cats.to(0)
                
                # # images=images.float()
                # # logits, fnns_out = model(features)
                
                
                
                # input_ids = text['input_ids'].cuda().squeeze(1)
                attention_mask = text['attention_mask'].cuda(device_id, non_blocking=True)
                input_ids = text['input_ids'].squeeze(1).cuda(device_id, non_blocking=True)
                ehr = ehr.cuda(device_id, non_blocking=True)
                # num_embeddings = torch.tensor(num_embeddings).cuda(device_id, non_blocking=True)
                # ehr = ehr.unsqueeze(1)
                # print("ehrrrrr",ehr.shape)
                
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_inference"):
                #         model(input_ids,attention_mask,ehr)

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                report_emb,ehr_emb = model2(input_ids,attention_mask,ehr)#x_nums.squeeze(), x_cats)
                
                # report_embeddings.append(report_emb)
            
                # ehr_embeddings.append(ehr_emb)
                
                
                # if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    
                    

                # report_embeddings = torch.cat(report_embeddings, dim=0)
                # ehr_embeddings=torch.cat(ehr_embeddings, dim=0) 
                report_embeddings=report_emb
            
                ehr_embeddings=ehr_emb        
                bz = len(report_embeddings)
                
                # labs = torch.arange(bz).type_as(report_emb).long()
                # labels = torch.eye(bz).type_as(report_emb)[labs]
                
            
                
                loss_g = CLIPLoss(temperature = 0.1)#.cuda(device_id) # triplet loss
                # print("new embeddings",ehr_embeddings.shape,report_embeddings.shape)
                loss_global,_,_= loss_g(ehr_embeddings,report_embeddings)#
                
                loss0 = loss_global
                i_t_scores=cosine_similarity(ehr_embeddings,report_embeddings)
                # print("ittt_scores",i_t_scores.shape)
                i2t_acc1_tr,i2t_corr_tr_batch,i2t_batch_tr = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                # t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr  = precision_at_k(t_i_scores)#, labels, top_k=(1,))
                # print("pr",i2t_acc1_tr) 
                i2t_corr_tr+=i2t_corr_tr_batch
                
                
                batch_epoch_tr+=i2t_batch_tr

                loss0.backward()


                optimizer.step() 
                
                if (num_batches+1)%8==0:
                    save_checkpoint(model2, optimizer, epoch, loss0.item(), file_path=f"output_weight/tabnet_model_checkpoint_corrected_backup_notes_wt.pth")
                    # print("saved!")
                save_checkpoint(model2, optimizer, epoch, loss0.item(), file_path=f"output_weight/tabnet_model_checkpoint_corrected_notes_wt.pth")
                # save_checkpoint(model, optimizer, epoch, loss0.item(), file_path="output_weight/tabnet_model_checkpoint.pth")
                # save_checkpoint(preprocessor, optimizer, epoch, loss0.item(), file_path="output_weight/preprocessor_checkpoint.pth")
                optimizer.zero_grad() 
                train_loss+=loss0.detach().item()#*batch_size ### Sajith: added detach 
                train_batches+=1

                
                # ehr_embeddings = []
                # report_embeddings = []
                    
                    
                
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches)#*batch_size)  
            print("loss_ita",train_loss)
            ita_list.append(train_loss)
            # scheduler.step()
            # global only: _if_updated_location_attention_local_global__fold
            if epoch%5==0 and epoch>9:
                torch.save(model2.state_dict(), os.path.join(args.output_dir,f"_ssl_model_{fold}__epoch__{epoch}"))
                # torch.save(preprocessor.state_dict(), os.path.join(args.output_dir,f"_ssl_preprocessor_{fold}__epoch__{epoch}__margin{args.margin}"))
            del loss0 ### Sajith 
            i2t_precision_tr=i2t_corr_tr/batch_epoch_tr
            # t2i_precision_tr=t2i_corr_tr/batch_epoch_tr
            print("training precision", (i2t_precision_tr))#+t2i_precision_tr)/2)
            precision_list.append(i2t_precision_tr)
            model2.eval()
#             with torch.set_grad_enabled(False):
    dist.destroy_process_group()


def train_global_local_model2(data_ds,numerical_column, categorical_column, num_embeddings,cat_idxs, cat_dims):
    print("availabke",torch.cuda.is_available()) 
    # dist.init_process_group("nccl", rank=1, world_size=2)
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = '12355'  # some free port
    
    # torch.distributed.launch
    dist.init_process_group(backend="gloo", init_method='env://', world_size = 1, rank =0)#, timeout=datetime.timedelta(minutes=60))
    # dist.init_process_group(backend='nccl', init_method='env://', world_size = 2, rank = 1)#, timeout=datetime.timedelta(minutes=60))

    num_batch_accumulate = 8
    model_config = {
    'cat_embedding_dim': 12,
    'num_transformer_blocks': 4,#4,
    'num_heads': 3,#3,
    'tf_dropout_rates': [0., 0., 0., 0.,],
    'ff_dropout_rates': [0., 0., 0., 0.,],
    'mlp_dropout_rates': [0.2, 0.1], 
    'mlp_hidden_units_factors': [2, 1],
    }
    emb_dim = model_config['cat_embedding_dim']
    num_transformer_blocks = model_config['num_transformer_blocks']
    num_heads = model_config['num_heads']
    attn_dropout_rates = model_config['tf_dropout_rates']
    ff_dropout_rates = model_config['ff_dropout_rates']
    mlp_dropout_rates = model_config['mlp_dropout_rates']
    mlp_hidden_units_factors = model_config['mlp_hidden_units_factors']

    global f1_list,auc_list, precision_list, recall_list
    batch_size=args.batch_size
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
    
    # test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    #    print(train_idx,val_idx)
        print("fold:", fold)
        if fold>0: 
            continue
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        sampler = DistributedSampler(data_ds,rank=0)
        train_dl = DataLoader(data_ds, batch_size=args.batch_size)##, sampler=train_sampler,drop_last=True)
        valid_dl = DataLoader(data_ds, batch_size=args.batch_size, sampler=test_sampler,drop_last=True)
        num_cat_cols, num_num_cols = len(categorical_column), len(numerical_column)
        
        model2 = report_ehr_model_combined(
                 num_cat_cols, num_num_cols,cat_idxs, cat_dims
                 )
        for name, param in model2.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory (GB): {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}")
            print(f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
        # model = nn.DataParallel(model2, device_ids= [0,1])
        model2 = model2.to(0)
        
        model = nn.parallel.DistributedDataParallel(model2, device_ids= [0],output_device=0,find_unused_parameters=True)
        # model = model.cuda()
        model = model.to(0)
        ita_list=[]
        ita_list_val=[]
        local_list=[]
        cl_list=[]
        # params = list(model.parameters())#list(preprocessor.parameters()) + 
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        # print("checkpointttttttt",preprocessor.state_dict().keys())


        # try:
        #     model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer,"output_weight/tabnet_model_checkpoint_corrected.pth")
        #     # preprocessor, _, _, _ = load_checkpoint(preprocessor, optimizer,"output_weight/preprocessor_checkpoint.pth")
        #     print(f"Resuming from epoch {start_epoch + 1}")
        # except FileNotFoundError:
        #     print("No checkpoint found, starting frsom scratch")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        optimizer.zero_grad()
        cl_list=[]
        for epoch in range(args.num_epochs):
            # preprocessor.train()
            model.train()
            train_loss = 0
            counter = 0
            num_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
            train_loss = 0
            counter = 0
            num_batches=0
            num_batches_valid=0
            val_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]

            
            report_embeddings = []
            ehr_embeddings=[]
            
            i2t_corr_tr=0
            batch_epoch_tr=0
            i2t_corr_val=0
            t2i_corr_tr=0
            t2i_corr_val=0
            batch_epoch_val=0
            precision_list=[]
            loc_list = []
           
            print("epoch",epoch)
            for num_batches,(text,ehr,num_embeddings) in enumerate(train_dl): 
        
                # features ,labels= features.cuda(),labels.cuda()
                optimizer.zero_grad()
                
                ehr = ehr.squeeze(2)

                # x_nums, x_cats = preprocessor(ehr, numerical_column, categorical_column)
                # # print("leennnnnnnnnnnnnn",len(x_nums))
                # print("feature shape",ehr.shape)
                
                # # x_nums, x_cats= x_nums.cuda(), x_cats.cuda()
                # # model = model.cuda()
                # x_nums , x_cats= x_nums.to(0) , x_cats.to(0)
                
                # # images=images.float()
                # # logits, fnns_out = model(features)
                
                
                attention_mask = text['attention_mask'].cuda()
                input_ids = text['input_ids'].cuda().squeeze(1)
                attention_mask = attention_mask.to(0)
                input_ids = input_ids.to(0)
                ehr = ehr.to(0)
                num_embeddings = torch.tensor(num_embeddings).to(0)
                # ehr = ehr.unsqueeze(1)
                # print("ehrrrrr",ehr.shape)
                
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_inference"):
                #         model(input_ids,attention_mask,ehr)

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                report_emb,ehr_emb = model(input_ids,attention_mask,ehr)#x_nums.squeeze(), x_cats)
                
                # report_embeddings.append(report_emb)
            
                # ehr_embeddings.append(ehr_emb)
                
                
                # if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    
                    

                report_embeddings = torch.cat(report_embeddings, dim=0)
                ehr_embeddings=torch.cat(ehr_embeddings, dim=0) 
                            
                bz = len(report_embeddings)
                
                # labs = torch.arange(bz).type_as(report_emb).long()
                # labels = torch.eye(bz).type_as(report_emb)[labs]
                
            
                
                loss_g = CLIPLoss(temperature = 0.1) # triplet loss
                # print("new embeddings",ehr_embeddings.shape,report_embeddings.shape)
                loss_global,_,_= loss_g(ehr_embeddings,report_embeddings)#
                
                loss0 = loss_global
                i_t_scores=cosine_similarity(ehr_embeddings,report_embeddings)
                # print("ittt_scores",i_t_scores.shape)
                i2t_acc1_tr,i2t_corr_tr_batch,i2t_batch_tr = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                # t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr  = precision_at_k(t_i_scores)#, labels, top_k=(1,))
                # print("pr",i2t_acc1_tr) 
                i2t_corr_tr+=i2t_corr_tr_batch
                
                
                batch_epoch_tr+=i2t_batch_tr

                loss0.backward()


                optimizer.step() 
                
                if (num_batches+1)%8==0:
                    save_checkpoint(model, optimizer, epoch, loss0.item(), file_path=f"output_weight/tabnet_model_checkpoint_corrected_backup.pth")
                    # print("saved!")
                save_checkpoint(model, optimizer, epoch, loss0.item(), file_path=f"output_weight/tabnet_model_checkpoint_corrected_notes.pth")
                # save_checkpoint(model, optimizer, epoch, loss0.item(), file_path="output_weight/tabnet_model_checkpoint.pth")
                # save_checkpoint(preprocessor, optimizer, epoch, loss0.item(), file_path="output_weight/preprocessor_checkpoint.pth")
                optimizer.zero_grad() 
                train_loss+=loss0.detach().item()#*batch_size ### Sajith: added detach 
                train_batches+=1

                
                ehr_embeddings = []
                report_embeddings = []
                    
                    
                
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches)#*batch_size)  
            print("loss_ita",train_loss)
            ita_list.append(train_loss)
            # scheduler.step()
            # global only: _if_updated_location_attention_local_global__fold
            if epoch%5==0 and epoch>9:
                torch.save(model.state_dict(), os.path.join(args.output_dir,f"_ssl_model_{fold}__epoch__{epoch}"))
                # torch.save(preprocessor.state_dict(), os.path.join(args.output_dir,f"_ssl_preprocessor_{fold}__epoch__{epoch}__margin{args.margin}"))
            del loss0 ### Sajith 
            i2t_precision_tr=i2t_corr_tr/batch_epoch_tr
            # t2i_precision_tr=t2i_corr_tr/batch_epoch_tr
            print("training precision", (i2t_precision_tr))#+t2i_precision_tr)/2)
            precision_list.append(i2t_precision_tr)
            model.eval()
#             with torch.set_grad_enabled(False):
    dist.destroy_process_group()


def train_global_local_model2(data_ds,numerical_column, categorical_column, num_embeddings,cat_idxs, cat_dims):
    print("availabke",torch.cuda.is_available()) 
    # dist.init_process_group("nccl", rank=1, world_size=2)
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = '12355'  # some free port
    
    # torch.distributed.launch
    dist.init_process_group(backend="gloo", init_method='env://', world_size = 1, rank =0)#, timeout=datetime.timedelta(minutes=60))
    # dist.init_process_group(backend='nccl', init_method='env://', world_size = 2, rank = 1)#, timeout=datetime.timedelta(minutes=60))

    num_batch_accumulate = 8
    model_config = {
    'cat_embedding_dim': 12,
    'num_transformer_blocks': 4,#4,
    'num_heads': 3,#3,
    'tf_dropout_rates': [0., 0., 0., 0.,],
    'ff_dropout_rates': [0., 0., 0., 0.,],
    'mlp_dropout_rates': [0.2, 0.1], 
    'mlp_hidden_units_factors': [2, 1],
    }
    emb_dim = model_config['cat_embedding_dim']
    num_transformer_blocks = model_config['num_transformer_blocks']
    num_heads = model_config['num_heads']
    attn_dropout_rates = model_config['tf_dropout_rates']
    ff_dropout_rates = model_config['ff_dropout_rates']
    mlp_dropout_rates = model_config['mlp_dropout_rates']
    mlp_hidden_units_factors = model_config['mlp_hidden_units_factors']

    global f1_list,auc_list, precision_list, recall_list
    batch_size=args.batch_size
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
    
    # test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    #    print(train_idx,val_idx)
        print("fold:", fold)
        if fold>0: 
            continue
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        sampler = DistributedSampler(data_ds,rank=0)
        train_dl = DataLoader(data_ds, batch_size=args.batch_size)##, sampler=train_sampler,drop_last=True)
        valid_dl = DataLoader(data_ds, batch_size=args.batch_size, sampler=test_sampler,drop_last=True)
        num_cat_cols, num_num_cols = len(categorical_column), len(numerical_column)
        
        model2 = report_ehr_model_combined(
                 num_cat_cols, num_num_cols,cat_idxs, cat_dims
                 )
        for name, param in model2.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory (GB): {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}")
            print(f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
        # model = nn.DataParallel(model2, device_ids= [0,1])
        model2 = model2.to(0)
        
        model = nn.parallel.DistributedDataParallel(model2, device_ids= [0],output_device=0,find_unused_parameters=True)
        # model = model.cuda()
        model = model.to(0)
        ita_list=[]
        ita_list_val=[]
        local_list=[]
        cl_list=[]
        # params = list(model.parameters())#list(preprocessor.parameters()) + 
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        # print("checkpointttttttt",preprocessor.state_dict().keys())


        # try:
        #     model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer,"output_weight/tabnet_model_checkpoint_corrected.pth")
        #     # preprocessor, _, _, _ = load_checkpoint(preprocessor, optimizer,"output_weight/preprocessor_checkpoint.pth")
        #     print(f"Resuming from epoch {start_epoch + 1}")
        # except FileNotFoundError:
        #     print("No checkpoint found, starting frsom scratch")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        optimizer.zero_grad()
        cl_list=[]
        for epoch in range(args.num_epochs):
            # preprocessor.train()
            model.train()
            train_loss = 0
            counter = 0
            num_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
            train_loss = 0
            counter = 0
            num_batches=0
            num_batches_valid=0
            val_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]

            
            report_embeddings = []
            ehr_embeddings=[]
            
            i2t_corr_tr=0
            batch_epoch_tr=0
            i2t_corr_val=0
            t2i_corr_tr=0
            t2i_corr_val=0
            batch_epoch_val=0
            precision_list=[]
            loc_list = []
           
            print("epoch",epoch)
            for num_batches,(text,ehr,num_embeddings) in enumerate(train_dl): 
        
                # features ,labels= features.cuda(),labels.cuda()
                optimizer.zero_grad()
                
                ehr = ehr.squeeze(2)

                # x_nums, x_cats = preprocessor(ehr, numerical_column, categorical_column)
                # # print("leennnnnnnnnnnnnn",len(x_nums))
                # print("feature shape",ehr.shape)
                
                # # x_nums, x_cats= x_nums.cuda(), x_cats.cuda()
                # # model = model.cuda()
                # x_nums , x_cats= x_nums.to(0) , x_cats.to(0)
                
                # # images=images.float()
                # # logits, fnns_out = model(features)
                
                
                attention_mask = text['attention_mask'].cuda()
                input_ids = text['input_ids'].cuda().squeeze(1)
                attention_mask = attention_mask.to(0)
                input_ids = input_ids.to(0)
                ehr = ehr.to(0)
                num_embeddings = torch.tensor(num_embeddings).to(0)
                # ehr = ehr.unsqueeze(1)
                # print("ehrrrrr",ehr.shape)
                
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_inference"):
                #         model(input_ids,attention_mask,ehr)

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                report_emb,ehr_emb = model(input_ids,attention_mask,ehr)#x_nums.squeeze(), x_cats)
                
                report_embeddings.append(report_emb)
            
                ehr_embeddings.append(ehr_emb)
                
                
                if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    
                    

                    report_embeddings = torch.cat(report_embeddings, dim=0)
                    ehr_embeddings=torch.cat(ehr_embeddings, dim=0) 
                               
                    bz = len(report_embeddings)
                    
                    # labs = torch.arange(bz).type_as(report_emb).long()
                    # labels = torch.eye(bz).type_as(report_emb)[labs]
                    
                
                    
                    loss_g = CLIPLoss(temperature = 0.1) # triplet loss
                    # print("new embeddings",ehr_embeddings.shape,report_embeddings.shape)
                    loss_global,_,_= loss_g(ehr_embeddings,report_embeddings)#
                    
                    loss0 = loss_global
                    i_t_scores=cosine_similarity(ehr_embeddings,report_embeddings)
                    # print("ittt_scores",i_t_scores.shape)
                    i2t_acc1_tr,i2t_corr_tr_batch,i2t_batch_tr = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                    # t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr  = precision_at_k(t_i_scores)#, labels, top_k=(1,))
                    print("pr",i2t_acc1_tr) 
                    i2t_corr_tr+=i2t_corr_tr_batch
                    
                    
                    batch_epoch_tr+=i2t_batch_tr

                    loss0.backward()


                    optimizer.step() 
                    
                    if (num_batches+1)%8==0:
                        save_checkpoint(model, optimizer, epoch, loss0.item(), file_path=f"output_weight/tabnet_model_checkpoint_corrected_backup.pth")
                        # print("saved!")
                    save_checkpoint(model, optimizer, epoch, loss0.item(), file_path=f"output_weight/tabnet_model_checkpoint_corrected_notes.pth")
                    # save_checkpoint(model, optimizer, epoch, loss0.item(), file_path="output_weight/tabnet_model_checkpoint.pth")
                    # save_checkpoint(preprocessor, optimizer, epoch, loss0.item(), file_path="output_weight/preprocessor_checkpoint.pth")
                    optimizer.zero_grad() 
                    train_loss+=loss0.detach().item()#*batch_size ### Sajith: added detach 
                    train_batches+=1

                    
                    ehr_embeddings = []
                    report_embeddings = []
                    
                    
                
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches)#*batch_size)  
            print("loss_ita",train_loss)
            ita_list.append(train_loss)
            # scheduler.step()
            # global only: _if_updated_location_attention_local_global__fold
            if epoch%5==0 and epoch>9:
                torch.save(model.state_dict(), os.path.join(args.output_dir,f"_ssl_model_{fold}__epoch__{epoch}"))
                # torch.save(preprocessor.state_dict(), os.path.join(args.output_dir,f"_ssl_preprocessor_{fold}__epoch__{epoch}__margin{args.margin}"))
            del loss0 ### Sajith 
            i2t_precision_tr=i2t_corr_tr/batch_epoch_tr
            # t2i_precision_tr=t2i_corr_tr/batch_epoch_tr
            print("training precision", (i2t_precision_tr))#+t2i_precision_tr)/2)
            precision_list.append(i2t_precision_tr)
            model.eval()
#             with torch.set_grad_enabled(False):
    dist.destroy_process_group()



def process_dataframe(df, df_num_cols, df_cat_cols):
    # Extract numerical and categorical columns as NumPy arrays for efficient access
    numerical_data = df[df_num_cols].to_numpy(dtype='float32')
    categorical_data = df[df_cat_cols].to_numpy(dtype='float32')

    # Convert numerical and categorical data to tensors at once
    numerical_tensor = torch.tensor(numerical_data, dtype=torch.float32)
    categorical_tensor = torch.tensor(categorical_data, dtype=torch.float32)

    # Unsqueeze to add a dimension if needed
    numerical_tensor = numerical_tensor.unsqueeze(2)
    categorical_tensor = categorical_tensor.unsqueeze(2)
    print("tensorssss",numerical_tensor.shape,categorical_tensor.shape)
    return numerical_tensor, categorical_tensor

if __name__ == "__main__":

    args = make_parser().parse_args()
    auc_list , precision_list, recall_list, f1_list = [], [], [], []
    df = pd.read_csv("../ssl_df2.csv")
    print("BBBB",df.shape)
    tech = 0
    print("shappppeeppe",df.shape)
    cap_list = []
    
    df =df.dropna(axis = 1)
    df['ed_los'] = pd.to_timedelta(df['ed_los']).dt.seconds / 60
    df = df.drop(columns="ed_los")
    df,df_num_cols, df_cat_cols  = process_csv_transformer(df)
   
    # df[df_cat_cols_non_bools] = df[df_cat_cols].fillna("unknown")
    # df[df_num_cols] = df[df_num_cols].fillna(-1)
    df_cat_cols.remove("outcome_ed_revisit_3d")
    df_cat_cols.remove("outcome_hospitalization")
   
    oe = OrdinalEncoder(handle_unknown='error',
                dtype=np.int64)
    df_cat_cols.remove("text")
    print("fffff")
    # df["clean_text"] = df["text"].apply(lambda x:process_text(x))
    # print("ssssss")
    
    # df.to_csv("clean_text_df.csv")
    df = pd.read_csv("clean_text_df.csv")
    df = df.drop(columns='Unnamed: 0')
    text = df["clean_text"]
    df = df.drop(columns=["clean_text","text","outcome_ed_revisit_3d","outcome_hospitalization"])
    
    
    encoded = oe.fit_transform(df[df_cat_cols].values)
    with open('ordinal_encoder.pkl', 'wb') as file:
        pickle.dump(oe, file)
    df[df_cat_cols] = encoded
    
    encoder_categories = oe.categories_
    num_embedding_list = []
    for i, categorical in enumerate(df_cat_cols):

            num_embedding_list.append(len(encoder_categories[i])),
    
    # df[df_cat_cols] = oe.fit_transform(df[df_cat_cols].values)
    
    df.index=range(df.shape[0])
    #######
    numerical_data = []
    categorical_data = []
    print("first")
    

    numerical_data, categorical_data = process_dataframe(df, df_num_cols, df_cat_cols)
    #######
    # ds = multimodaldataset(df,df_num_cols,df_cat_cols,num_embedding_list,text)
    print("second")
    
    ds = multimodaldataset(df,numerical_data,categorical_data,num_embedding_list,text)
    
    with open('num_emb.pkl', 'wb') as file:
        pickle.dump(num_embedding_list, file)
    
    features = [ col for col in df.columns] 
    # print("catttttttttttttttttttttt",len(features),len(df_cat_cols),len(encoder_categories),len(df_num_cols))
    # cat_idxs = [ i for i, f in enumerate(features) if f in df_cat_cols]
    cat_idxs = [i for i in range(len(df_num_cols),len(df.columns))]
    # print("herrrreeee",df_cat_cols,df_num_cols)
    # print(sara)
    # print("uuuuuuuu",features)
    cat_dims = [ len(encoder_categories[i]) for i, f in enumerate(df_cat_cols)]
    
        # TabNetPretrainer
    
   
#     unsupervised_model = TabNetPretrainer(
#         cat_idxs=cat_idxs,
#         cat_dims=cat_dims,
#         cat_emb_dim=3,
#         optimizer_fn=torch.optim.Adam,
#         optimizer_params=dict(lr=1e-4),
#         mask_type='entmax', # "sparsemax",
#         n_shared_decoder=1, # nb shared glu for decoding
#         n_indep_decoder=1, # nb independent glu for decoding
#     #     grouped_features=[[0, 1]], # you can group features together here
        
#         verbose=5,
#     )
    

#     # unsupervised_model.input_dim=len(df_cat_cols)+len(df_num_cols)
    
#     # unsupervised_model._set_network()
#     # unsupervised_model.network = unsupervised_model.network.to("cpu")
#     # print("unsupervised_model",dir(unsupervised_model.network))
#     unsupervised_model.n_d = 128
#     unsupervised_model.n_a = 128 #64
#     print("dffff",df.shape)
#     unsupervised_model.fit(
#     X_train=df.values,
#     # eval_set=[X_valid],
#     max_epochs=100 , patience=5,
#     batch_size=2048, virtual_batch_size=128,
#     num_workers=0,
#     drop_last=False,
#     pretraining_ratio=0.5
# ) 
#     unsupervised_model.save_model('./tabnet_pretrained_new_dim_2')
#     print(sara)
#     _, embedded_X = unsupervised_model.predict(df.values)
    # print("firssst shape",embedded_X.shape)
    #firssst shape (376021, 240)
    # print("donnnnnneeeee")
    
    _ = train_global_local_model(ds,df_num_cols, df_cat_cols,num_embedding_list,cat_idxs, cat_dims)


# if __name__ == "__main__":

#     args = make_parser().parse_args()
#     auc_list , precision_list, recall_list, f1_list = [], [], [], []
#     df = pd.read_csv("../ssl_df.csv")
#     tech = 0
#     print("shappppeeppe",df.shape)
#     cap_list = []
#     # for text in df["text"]:
        
#     #     cap_list.extend(find_all_caps_words(text))
#     # print("setttttt",set(cap_list))
#     df =df.dropna(axis = 1)
     
    
#     df,df_num_cols, df_cat_cols  = process_csv_transformer(df)
#     # bool_columns = df.select_dtypes(include='bool')
#     # df[df_cat_cols_non_bools] = df[df_cat_cols].fillna("unknown")
#     # df[df_num_cols] = df[df_num_cols].fillna(-1)

#     # print("null columns",df.iloc[:,[6, 9, 14, 17, 19, 20, 21, 22, 23, 24, 25, 26]].isna().sum())
#     # for col in df.columns:
#     #     print(col,df[col].isna().sum())

#     # df = df.drop(df.columns[19],axis=1)
#     # df = df.drop(df.columns[22],axis=1)
    
    
#     # for col in [19,22]:#[6, 9, 12, 15, 19, 21, 22, 23, 24, 25, 26, 27, 28]:
#     #     print(df.iloc[:,col].isna().sum())
#     # df_cat_cols.remove("discharge_location")
#     # df_cat_cols.remove("ethnicity")
#     # print("null",df.isna().sum())
#     oe = OrdinalEncoder(handle_unknown='error',
#                 dtype=np.int64)
#     df_cat_cols.remove("text")
#     encoded = oe.fit_transform(df[df_cat_cols].values)
#     df[df_cat_cols] = encoded
#     encoder_categories = oe.categories_
    
#     df[df_cat_cols] = oe.fit_transform(df[df_cat_cols].values)
    
#     df.index=range(df.shape[0])
    

#     ds = multimodaldataset(df,df_num_cols,df_cat_cols)


#     _ = train_global_local_model(ds,df_num_cols, df_cat_cols, encoder_categories)
