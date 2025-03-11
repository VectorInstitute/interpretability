import os
import random
import torch
import torch.distributed as dist

def setup() -> None:
    """Initialize the process group."""
    dist.init_process_group(backend="nccl")

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

def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, epoch, loss, file_path):
    """
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)

def setup_distributed_training():
    """
    """
    setup()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.empty_cache()
    device_id = torch.cuda.current_device()
    return device_id

def load_checkpoint(model, optimizer, file_path):
    """
    """
    checkpoint = torch.load(file_path)
   
    checkpoint_state_dict = checkpoint['model_state_dict']

    # Create a new state_dict with only the matching keys
    checkpoint_state_dict = {
        k.replace('module.', ''): v \
            for k, v in checkpoint_state_dict.items()
        }
    
    filtered_state_dict = { k: v
        for k, v in checkpoint_state_dict.items()
            if k in model.state_dict()
    }
    
    model.load_state_dict(filtered_state_dict, strict=False)
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss