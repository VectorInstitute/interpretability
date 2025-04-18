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
plt.rcParams['figure.figsize'] = [25, 10]



def make_parser():
    parser = argparse.ArgumentParser(description='Imaging Explainability')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
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
    return parser

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

def setup_ddp(args,worker_init_fn):
    init_fn = partial(
        worker_init_fn,
        num_workers=args.num_workers,
        rank=dist.get_rank(),
        seed=args.rseed,
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

def train_prototype(args, train_ds,test_ds):
    
    f1_list,auc_list,precision_list,recall_list= [] , [] , [] , []
    setup()
    device_id = setup_ddp(args,worker_init_fn)
    test_sampler = DistributedSampler(test_ds, shuffle=True)  
    splits=KFold(n_splits=5,shuffle=True,random_state=42)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_ds)))):
        if dist.get_rank() == 0
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
    return f1_list,auc_list, precision_list, recall_list

if __name__ == '__main__':
    args = make_parser().parse_args()
    auc_list , precision_list, recall_list, f1_list = [], [], [], []
    set_random_seed(args.rseed)
    image_path = "/datasets/MNIST/processed"
    train_ds, test_ds = load_mnist_data(image_path)
    f1_list,auc_list, precision_list, recall_list = train_prototype(args, train_ds,test_ds)   