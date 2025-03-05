import os
import random
from functools import partial

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP 
import torchvision.models as tv_models
from typing import Tuple

from data import Config
from interp.metrics import get_multiclass_roc_auc_score, get_dist_auc
from interp.utils import *
from interp.models import ResNetAttention
from utils import cleanup, worker_init_fn, save_checkpoint, setup_distributed_training
from data import XrayDataset, load_nih_data

def get_data_loaders(cfg: Config,
                     train_ds: XrayDataset,
                     test_ds: XrayDataset) -> Tuple[DataLoader, DataLoader]:
    """
    """
    init_fn = partial(
        worker_init_fn,
        num_workers=cfg.train_params.num_workers,
        rank=dist.get_rank(),
        seed=cfg.rseed
    )

    train_sampler = DistributedSampler(train_ds,
                                       shuffle=True)  
    test_sampler = DistributedSampler(test_ds,
                                      shuffle=True)
    train_dl = DataLoader(train_ds,
                    sampler=train_sampler,
                    batch_size=42,
                    worker_init_fn=init_fn,
                    pin_memory=False,
                    drop_last=True,
                    num_workers=4)
    test_dl = DataLoader(test_ds,
                    sampler=test_sampler,
                    batch_size=42,
                    worker_init_fn=init_fn)
    return train_dl, test_dl

def eval_step(model: ResNetAttention,
              data_dl: DataLoader,
              device_id: int) -> Tuple[np.array, np.array]:
    """
    """
    model.eval()

    num_classes = len(data_dl.dataset.unique_labels)
    y_probs = np.empty((0, num_classes), dtype = np.float32)
    y_true  = np.empty((0, num_classes), dtype = np.float32)

    with torch.set_grad_enabled(False):
        for images,labels in data_dl:
            images = images.cuda(device_id, non_blocking=True)
            labels = labels.cuda(device_id, non_blocking=True)
            logits, attention_weights = model(images)
            
            logits = torch.squeeze(logits)
            prob = torch.sigmoid(logits)
            
            y_probs = np.concatenate((y_probs, prob.detach().cpu().numpy()))
            y_true = np.concatenate((y_true, labels.detach().cpu().numpy()))

    return y_probs, y_true

def train_selfattention(model: ResNetAttention,
                        cfg: Config,
                        train_dl: DataLoader,
                        test_dl: DataLoader,
                        device_id: int) -> float:
    """
    """
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.optimizer.lr,
                            weight_decay=float(cfg.train_params.weight_decay))
    classifier_criterion = nn.BCEWithLogitsLoss()
    no_disease_class_idx = train_dl.dataset.unique_labels.index('No Finding')
    num_unique_classes = len(train_dl.dataset.unique_labels)

    for epoch in range(cfg.train_params.num_epochs):
        print(f'\nTraining Epoch {epoch} on device {device_id}')
        model.train() 

        y_tr_probs = np.empty((0, num_unique_classes),
                                dtype = np.float32)
        y_tr_true  = np.empty((0, num_unique_classes),
                                dtype = np.float32)

        for num_batches,(images, labels) in enumerate(train_dl):

            optimizer.zero_grad()
            images = images.cuda(device_id, non_blocking=True)
            labels = labels.cuda(device_id, non_blocking=True)

            logits, _ = model(images)
            prob = torch.sigmoid(logits)
            loss = classifier_criterion(logits, labels) 
            loss.backward()
            optimizer.step()

            print(f'\n Loss on device_id {device_id} for epoch {epoch}: {loss}')
            
            y_tr_probs = np.concatenate((y_tr_probs, prob.detach().cpu().numpy()))
            y_tr_true = np.concatenate((y_tr_true, labels.detach().cpu().numpy()))
        
        y_tr_probs = np.delete(y_tr_probs, no_disease_class_idx, axis=1)
        y_tr_true = np.delete(y_tr_true, no_disease_class_idx, axis=1)
        
        train_auc = get_multiclass_roc_auc_score(y_tr_probs, y_tr_true)
        if dist.get_rank() == 0:  
            print(f'Train AUC on device_id {device_id} for epoch {epoch}: {train_auc}')
        
        if dist.get_rank() == 0 and (epoch+1)%4 == 0:
            print(f'Saving checkpoint for epoch {epoch}')
            save_checkpoint(model, optimizer, epoch, loss.item(),
                            file_path=f'{epoch}_{cfg.output_file}')
        
        y_te_probs, y_te_true = eval_step(model, test_dl, device_id)
        y_te_probs = np.delete(y_te_probs, no_disease_class_idx, axis=1)
        y_te_true = np.delete(y_te_true, no_disease_class_idx, axis=1)

        test_auc = get_dist_auc(y_te_probs, y_te_true)
        if dist.get_rank() == 0:
            print(f'Test AUC device_id {device_id} for epoch {epoch}: {test_auc}')

        break
    if dist.get_rank() == 0:
        print('Saving final model')
        save_checkpoint(model, optimizer, epoch, loss.item(),
                    file_path=cfg.output_file)

    return test_auc


if __name__ == '__main__':
    #Get data config
    config = Config()

    print(type(config))
    #Get Self attention model config
    sa_config = Config(yaml_path = 'configs/selfattention.yaml')

    train_df, test_df = load_nih_data(config)

    #Load NIH dataset, get dataloaders
    train_ds = XrayDataset(config, train_df)
    test_ds = XrayDataset(config, test_df)

    #Distributed setup
    print(f'Setting up distributed training..')
    device_id = setup_distributed_training()
    print(f'Device id: {device_id}')

    #Get data loaders
    train_dl, test_dl = get_data_loaders(sa_config, train_ds, test_ds)

    print("--------------------------------------------")
    print("Training Self Attention model on NIH dataset")
    print("--------------------------------------------")

    #Create Self attention model
    random_seed(42, True)
    model = ResNetAttention(tv_models.resnet18(pretrained=True))
    model = model.cuda(device_id)
    model = DDP(model, device_ids=[device_id])
    test_auc = train_selfattention(model, sa_config, train_dl, test_dl, device_id)
    print(f"\Test score of Self attention model: {test_auc}")

    #Clean distributed setup
    cleanup()