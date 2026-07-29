# This file defines typical data loaders to be constructed in the runner
# Dataloader
#     Dataset
#     Collator
import cv2
import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.sampler import RandomSampler, BatchSampler, SequentialSampler
from easyvolcap.engine import cfg, args
from easyvolcap.engine import DATASETS, DATASAMPLERS, DATALOADERS
from easyvolcap.dataloaders.datasamplers import IterationBasedBatchSampler
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import setup_deterministic
from easyvolcap.utils.data_utils import default_collate, default_convert


# https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def worker_init_fn(worker_id, fix_random, allow_tf32, deterministic, benchmark):
    cv2.setNumThreads(1)
    setup_deterministic(fix_random, allow_tf32, deterministic, benchmark, worker_id)

    worker_info = get_worker_info()
    dataset = worker_info.dataset


def update_fn(batches: List[dotdict]):
    elem = batches[0]
    keys = list(elem.keys())  # all keys of the batch
    for key in keys:
        if isinstance(elem[key], torch.Tensor) or isinstance(elem[key], np.ndarray):  # support tensor image shape filling
            images = [batch[key] for batch in batches]
            if all([isinstance(img, torch.Tensor) for img in images]) or all([isinstance(img, np.ndarray) for img in images]):  # skip some of the shapes
                shapes = [image.shape for image in images]  # B, S,
                shapes = torch.as_tensor(shapes)
                if ((shapes - shapes[0]) != 0).any():  # shape mismatch
                    max_shapes = shapes.max(dim=0)[0].numpy().tolist()  # all max shapes, S,
                    for i, (image, batch) in enumerate(zip(images, batches)):
                        if isinstance(image, torch.Tensor):
                            canvas = image.new_zeros(max_shapes)
                        elif isinstance(image, np.ndarray):
                            canvas = np.zeros(max_shapes, dtype=image.dtype)
                        canvas[[slice(s) for s in image.shape]] = image  # will the fancy indexing work?
                        batch[key] = canvas  # implicitly update batches
        elif isinstance(elem[key], list):
            for b in batches: b[key] = {i: v for i, v in enumerate(b[key])}
            update_fn([b[key] for b in batches])  # inplace update
            for b in batches: b[key] = [v for v in b[key].values()]
        elif isinstance(elem[key], dict):
            update_fn([b[key] for b in batches])  # inplace update
        else:
            pass  # nothing to do


def collate_fn(batches: List[dotdict]):
    update_fn(batches)
    return default_collate(batches)


@DATALOADERS.register_module()
class DefaultDataloader(DataLoader):
    def __init__(self,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 pin_memory: bool = True,
                 max_iter: int = cfg.runner_cfg.ep_iter * cfg.runner_cfg.epochs,

                 fix_random: bool = cfg.fix_random,
                 allow_tf32: bool = cfg.allow_tf32,
                 deterministic: bool = cfg.deterministic,
                 benchmark: bool = cfg.benchmark,

                 dataset_cfg: dotdict = dotdict(),
                 sampler_cfg: dotdict = dotdict(type=RandomSampler.__name__),
                 batch_sampler_cfg: dotdict = dotdict(type=BatchSampler.__name__),
                 ):
        if batch_sampler_cfg.batch_size == -1: batch_sampler_cfg.batch_size = len(dataset)
        dataset = DATASETS.build(dataset_cfg)
        if dataset_cfg.split == 'TRAIN' or dataset_cfg.split == 'train':
            sampler = RandomSampler(data_source=dataset)
        else:
            sampler = SequentialSampler(data_source=dataset)
        batch_sampler = BatchSampler(sampler=sampler, **batch_sampler_cfg)
        if max_iter != -1: batch_sampler = IterationBasedBatchSampler(batch_sampler, max_iter)

        # GUI related special config
        if benchmark == 'train': benchmark = args.type == 'train'  # for static sized input

        # Initialization of dataloader object
        super().__init__(dataset=dataset,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         collate_fn=collate_fn,
                         worker_init_fn=partial(worker_init_fn, fix_random=fix_random, allow_tf32=allow_tf32, deterministic=deterministic, benchmark=benchmark),
                         prefetch_factor=prefetch_factor if num_workers > 0 else None if torch.__version__[0] >= '2' else 2,
                         )
