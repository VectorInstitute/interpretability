import os
import yaml
import random
from functools import partial

import torch
import ultraimport
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

from resnetattention import ResNetAttention

ultraimport('__dir__/../utils/metrics.py', ['get_multiclass_roc_auc_score', 'get_dist_auc'], add_to_ns=True)
ultraimport('__dir__/../utils/model_utils.py', ['load_checkpoint', 'save_checkpoint'], add_to_ns=True)
ultraimport('__dir__/../utils/visualize.py', ['visualize_trainable_attention'], add_to_ns=True)
ultraimport('__dir__/../utils/dist_utils.py', ['cleanup', 'worker_init_fn', 'setup_distributed_training'], add_to_ns=True)
ultraimport('__dir__/../utils/nih_data_utils.py', ['XrayDataset', 'load_nih_data'], add_to_ns=True)

def get_data_loaders(cfg,
                     train_ds: XrayDataset,
                     test_ds: XrayDataset) -> Tuple[DataLoader, DataLoader]:
    """
    """
    init_fn = partial(
        worker_init_fn,
        num_workers=cfg['train_params']['num_workers'],
        rank=dist.get_rank(),
        seed=cfg['rseed']
    )

    train_sampler = DistributedSampler(train_ds,
                                       shuffle=True)  
    test_sampler = DistributedSampler(test_ds,
                                      shuffle=True)
    train_dl = DataLoader(train_ds,
                    sampler=train_sampler,
                    batch_size=cfg['train_params']['batch_size'],
                    worker_init_fn=init_fn,
                    pin_memory=False,
                    drop_last=True,
                    num_workers=4)
    test_dl = DataLoader(test_ds,
                    sampler=test_sampler,
                    batch_size=cfg['train_params']['batch_size'],
                    worker_init_fn=init_fn)
    return train_dl, test_dl

def eval_step(model: ResNetAttention,
              cfg,
              data_dl: DataLoader,
              epoch: int,
              device_id: int,
              visualize: bool = True,
              save_dir: str = 'attention_plots') -> Tuple[np.array, np.array]:
    """
    """
    model.eval()
    print(f'\n Evaluating on device {device_id} for epoch {epoch}')

    num_classes = len(data_dl.dataset.unique_labels)
    y_probs = np.empty((0, num_classes), dtype = np.float32)
    y_true  = np.empty((0, num_classes), dtype = np.float32)

    with torch.set_grad_enabled(False):
        for batch_idx, (images,labels) in enumerate(data_dl):
            if dist.get_rank() == 0:
                print(f'\n Evaluating batch {batch_idx}')

            images = images.cuda(device_id, non_blocking=True)
            labels = labels.cuda(device_id, non_blocking=True)
            logits, attention = model(images)

            logits = torch.squeeze(logits)
            prob = torch.sigmoid(logits)
            
            y_probs = np.concatenate((y_probs, prob.detach().cpu().numpy()))
            y_true = np.concatenate((y_true, labels.detach().cpu().numpy()))

    if visualize:
        pred = (prob >= 0.5).float() 
        visualize_trainable_attention(attention[0], images[0],
                                        pred[0], num_classes,
                                        save_plots=True,
                                        save_dir = f'{cfg["plot_folder"]}/epoch_{epoch}')

    return y_probs, y_true

def train_selfattention(model: ResNetAttention,
                        cfg,
                        train_dl: DataLoader,
                        test_dl: DataLoader,
                        device_id: int,
                        visualize: bool = False,
                        resume_training: bool = False) -> float:
    """ 
    """
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['optimizer']['lr'],
                            weight_decay=float(cfg['train_params']['weight_decay']))
    classifier_criterion = nn.BCEWithLogitsLoss()
    no_disease_class_idx = train_dl.dataset.unique_labels.index('No Finding')
    num_unique_classes = len(train_dl.dataset.unique_labels)
    train_loss, start_epoch = 0.0, -1

    if resume_training:
        checkpoint_file = f'{cfg["checkpoint_dir"]}/{cfg["checkpoint_file"]}'
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer,
                                                            checkpoint_file)

    for epoch in range(start_epoch+1, cfg['train_params']['num_epochs']):
        print(f'\nTraining Epoch {epoch} on device {device_id}..')
        model.train() 

        y_tr_probs = np.empty((0, num_unique_classes),
                                dtype = np.float32)
        y_tr_true  = np.empty((0, num_unique_classes),
                                dtype = np.float32)

        for batch_idx,(images, labels) in enumerate(train_dl):
            if dist.get_rank() == 0:
                print(f'Training batch {batch_idx}')

            optimizer.zero_grad()
            images = images.cuda(device_id, non_blocking=True)
            labels = labels.cuda(device_id, non_blocking=True)

            logits, attention = model(images)
            prob = torch.sigmoid(logits)
            loss = classifier_criterion(logits, labels) 
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            y_tr_probs = np.concatenate((y_tr_probs, prob.detach().cpu().numpy()))
            y_tr_true = np.concatenate((y_tr_true, labels.detach().cpu().numpy()))

        if dist.get_rank() == 0:
            print(f'\n Avg. training Loss for epoch {epoch}: \
                {train_loss/len(train_dl.dataset)}')

        y_tr_probs = np.delete(y_tr_probs, no_disease_class_idx, axis=1)
        y_tr_true = np.delete(y_tr_true, no_disease_class_idx, axis=1)

        train_auc = get_multiclass_roc_auc_score(y_tr_probs, y_tr_true)
        print(f'\n Train AUC on device_id {device_id} for epoch {epoch}: \
                    {train_auc}')

        if dist.get_rank() == 0 and (epoch)%4 == 0:
            print(f'Saving checkpoint for epoch {epoch}')
            checkpoint_file = f'{cfg["checkpoint_dir"]}/{epoch}_{cfg["output_file"]}'
            save_checkpoint(model, optimizer, epoch, loss.item(), checkpoint_file)

        print(f'Evaluating test set on device {device_id}.. ')
        y_te_probs, y_te_true = eval_step(model, cfg, test_dl,
                                          epoch, device_id, visualize=True)
    
        y_te_probs = np.delete(y_te_probs, no_disease_class_idx, axis=1)
        y_te_true = np.delete(y_te_true, no_disease_class_idx, axis=1)
        test_auc = get_dist_auc(y_te_probs, y_te_true)

        if dist.get_rank() == 0:
            print(f'Test AUC device_id {device_id} for epoch {epoch}: {test_auc}')

    if dist.get_rank() == 0:
        print('Saving final model')
        save_checkpoint(model, optimizer, epoch, loss.item(),
                    f'{cfg["checkpoint_dir"]}/{cfg["output_file"]}')  
    return test_auc

if __name__ == '__main__':

    #Get Self attention model config
    yaml_file = os.path.join(os.path.dirname(__file__), 'selfattention.yaml')
    with open(yaml_file, 'r') as f:
        sa_config = yaml.safe_load(f)
    
    #Get NIH data files
    train_df, test_df = load_nih_data(data_path = sa_config['data_dir'])

    print('Loading NIH dataset..')
    #Load NIH dataset, get dataloaders
    train_ds = XrayDataset(train_df, data_path = sa_config['data_dir'])
    test_ds = XrayDataset(test_df, data_path = sa_config['data_dir'])

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
    random.seed(42, True)
    model = ResNetAttention(tv_models.resnet18(pretrained=True))
    model = model.cuda(device_id)
    model = DDP(model, device_ids=[device_id])
    test_auc = train_selfattention(model, sa_config, train_dl, test_dl, device_id, visualize=True, resume_training=True)
    print(f"\n Test score of Self attention model: {test_auc}")

    #Clean distributed setup
    cleanup()
