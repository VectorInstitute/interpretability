import os
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, Subset
from model.model import ProtoNet
from dataset.dataset import MNistDataset, create_episode
from torch.nn.parallel import DistributedDataParallel as DDP 
from functools import partial
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import torch.nn.functional as F
import pickle
import torch.distributed as dist
from losses.losses import compute_prototypes, compute_loss_and_accuracy
from tqdm import tqdm
from torch.autograd import Variable
plt.rcParams['figure.figsize'] = [25, 10]



def make_parser():
    parser = argparse.ArgumentParser(description='Imaging Explainability')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--class_names', type=list, default=['Normal', 'CHF', 'pneumonia'], help='Label names for classification')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--resize', type=int, default=224, help='Resizing images')
    parser.add_argument('--experiment', type=str, default="baseline", help='Imaging Experiment')

    # Training
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler')
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')

    ## UNET Specific arguments.
    parser.add_argument('--gamma', type=float, default=1.0, help='Used to set the weighting value between the classifier and the segmentation in Unet')
    parser.add_argument('--model_teacher', type=str, default='timm-efficientnet-b0', help='model_teacher')
    parser.add_argument('--pretrained_name', type=str, default='noisy-student', help='model pretrained value')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--second_loss', type=str, default='ce', choices=['dice', 'ce'], help='Segmentation loss')

    # Misc
    parser.add_argument('--gpus', type=str, default='7', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--viz', default=False, action='store_true', help='[USE] Vizdom')
    parser.add_argument('--gcam_viz', default=False, action='store_true', help='[USE] Used for displaying the GradCam results')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--weight_decay', type=int, default=1e-5, help='Seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=1, help='Seed for reproducibility')
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

    
splits=KFold(n_splits=5,shuffle=True,random_state=42)
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
        print("lllll",all_labels.shape)
        print("sssss",all_preds.shape)
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
    
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model.state_dict()}# and "mlp.mlp_layers.dense_1" not in k and "mlp.mlp_layers.dense_0" not in k and "num_norm" not in k}
    
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
    dist.init_process_group("nccl")#,rank=dist.get_rank(), init_method='env://')


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()



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
        if fold!=0:
            continue
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
        
        classes = train_ds.tupple[1]
        
        
        
        init_fn = partial(worker_init_fn,num_workers=args.num_workers,rank=dist.get_rank(),seed=args.rseed)
        
        model2 = ProtoNet()
        
        model2 = model2.cuda(device_id)
        model2 = DDP(model2, device_ids=[device_id])
        
        
        optimizer = optim.AdamW(model2.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        try:
            model2, optimizer, start_epoch, _ = load_checkpoint(model2, optimizer,"output_weight/proto_mnist.pth")
            print(f"Resuming from epoch {start_epoch + 1}")
        except FileNotFoundError:
            print("No checkpoint found, starting frsom scratch")
            start_epoch=0
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        optimizer.zero_grad()
        
        
       
        for epoch in range(start_epoch, args.num_epochs):
            
            model2.train()
            


            n_way = 5
            k_shot = 5 
            q_query = 5
            num_episodes = 100
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
            save_checkpoint(model2, optimizer, epoch, loss.detach(), file_path=f"output_weight/proto_mnist.pth")          
        test_protonet(model2, test_ds)
            
            
    return 0

if __name__ == '__main__':
    args = make_parser().parse_args()
    auc_list , precision_list, recall_list, f1_list = [], [], [], []
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)   
    torch.cuda.manual_seed(args.rseed)
    torch.cuda.manual_seed_all(args.rseed)
    image_path = "/datasets/MNIST/processed"
    file_path_train = os.path.join(image_path,"training.pt")
    x_train =torch.load(file_path_train)
    file_path_test = os.path.join(image_path,"test.pt")
    x_test =torch.load(file_path_test)
    train_ds = MNistDataset(x_train, image_path)
    test_ds = MNistDataset(x_test, image_path)
    train_prototype(args, train_ds,test_ds)





    




    
   