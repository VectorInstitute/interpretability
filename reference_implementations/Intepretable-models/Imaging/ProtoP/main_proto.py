import os
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, Subset
from proto_model import ProtoNet
from utils.load_mnist_utils import create_episode, load_mnist_data, MNistDataset
from torch.nn.parallel import DistributedDataParallel as DDP 
from functools import partial
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from losses.losses import compute_prototypes, compute_loss_and_accuracy
from utils.dist_utils import setup, cleanup
from utils.model_utils import load_checkpoint, save_checkpoint
from utils.utils import set_random_seed
import yaml
plt.rcParams['figure.figsize'] = [25, 10]
import warnings
warnings.filterwarnings("ignore")





def compute_test_auc(model, test_loader, device_id):
    model.eval()
    all_preds = []
    all_labels = [] 
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(device_id, non_blocking=True), labels.cuda(device_id, non_blocking=True)
            outputs,_ = model(images)
            preds = torch.sigmoid(outputs)#.squeeze()  # Assuming binary classification

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

def setup_ddp(sa_config,worker_init_fn):
    init_fn = partial(
        worker_init_fn,
        num_workers=sa_config['num_workers'],
        rank=dist.get_rank(),
        seed=sa_config['rseed'],
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.empty_cache()
    device_id = torch.cuda.current_device()
    return device_id

# Testing function
def test_protonet(model, test_dataset, n_way=5, k_shot=5, q_query=15, num_episodes=100):
    model.eval()
    total_accuracy = 0.0
    
    with torch.no_grad():
        for _ in range(num_episodes):
            support_set, query_set = create_episode(test_dataset, n_way, k_shot, q_query)
            prototypes,unique_classes = compute_prototypes(support_set, model)
            _, accuracy = compute_loss_and_accuracy(query_set, prototypes, model,unique_classes,visualize_heatmap=True)
            
            total_accuracy += accuracy
    
    avg_accuracy = total_accuracy / num_episodes
    print(f"Test Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy

def train_prototype(sa_config, train_ds,test_ds):
    f1_list,auc_list,precision_list,recall_list= [] , [] , [] , []
    setup()
    device_id = setup_ddp(sa_config,worker_init_fn)
    test_sampler = DistributedSampler(test_ds, shuffle=True)  
    splits=KFold(n_splits=5,shuffle=True,random_state=42)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_ds)))):
        if dist.get_rank() == 0:
            print("fold:", fold)
            print('Fold {}'.format(fold + 1))
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(train_ds, val_idx)
        train_sampler = DistributedSampler(train_subset, shuffle=True)  
        valid_sampler = DistributedSampler(val_subset, shuffle=True)
        classes = train_ds.tupple[1]
        model2 = ProtoNet()
        model2 = model2.cuda(device_id)
        model2 = DDP(model2, device_ids=[device_id])
        optimizer = optim.AdamW(model2.parameters(), lr=float(sa_config['lr']),weight_decay=float(sa_config['weight_decay']))
        if sa_config['initialize_with_checkpoint']:
            print("Loading checkpoint")
            model2, optimizer, start_epoch, _ = load_checkpoint(model2, optimizer,sa_config['checkpoint_path'])
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            print("starting from scratch")
            start_epoch=0
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        optimizer.zero_grad()
        for epoch in range(start_epoch, sa_config['num_epochs']):
            model2.train()
            n_way = sa_config['n_way']
            k_shot = sa_config['k_shot']
            q_query = sa_config['q_query']
            num_episodes = sa_config['num_episodes']    
            accuracy_epoch = 0
            total_loss = 0
            for i in range(num_episodes):
                optimizer.zero_grad()
                support_set, query_set = create_episode(train_ds, n_way, k_shot, q_query)
                prototypes,support_labels = compute_prototypes(support_set, model2)
                unique_classes = torch.unique(support_labels)
                if epoch<19:
                    loss, accuracy = compute_loss_and_accuracy(query_set, prototypes, model2,unique_classes,visualize_heatmap=False) 
                else:
                    loss, accuracy = compute_loss_and_accuracy(query_set, prototypes, model2,unique_classes,visualize_heatmap=True)
                total_loss += loss.item()
                accuracy_epoch+= accuracy
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping episode {i} due to NaN/Inf loss")
                    continue
                loss.backward()
                optimizer.step()
            avg_loss = total_loss/num_episodes
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {accuracy_epoch/num_episodes:.4f}")
            save_checkpoint(model2, optimizer, epoch, loss.detach(), file_path=sa_config['checkpoint_path'])          
        test_protonet(model2, test_ds)    
    return f1_list,auc_list, precision_list, recall_list

if __name__ == '__main__':
    #Get Proto model config
    yaml_file = os.path.join(os.path.dirname(__file__), 'proto.yaml')
    with open(yaml_file, 'r') as f:
        sa_config = yaml.safe_load(f)
    auc_list , precision_list, recall_list, f1_list = [], [], [], []
    set_random_seed(sa_config['rseed'])
    image_path = sa_config['image_path']
    train_ds, test_ds = load_mnist_data(image_path)
    f1_list,auc_list, precision_list, recall_list = train_prototype(sa_config, train_ds,test_ds)   