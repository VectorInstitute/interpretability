import torch.distributed as dist

def setup() -> None:
    """Initialize the process group."""
    dist.init_process_group("nccl")


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()

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