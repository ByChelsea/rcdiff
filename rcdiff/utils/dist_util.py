"""
Helpers for distributed training.
"""

import os
import socket
import torch

import torch as th
import torch.distributed as dist
GPUS_PER_NODE = 4

SETUP_RETRY_COUNT = 3


def setup_dist(gpus_per_node=1):
    """
    Setup a distributed process group without MPI.

    Parameters:
    - gpus_per_node (int): Number of GPUs available on a single node.
    """
    if dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % gpus_per_node)
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    torch.cuda.set_device(local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://"
    )


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
