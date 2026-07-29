import os
import torch
from os.path import join
from datetime import datetime
from collections import deque
from copy import copy, deepcopy
from torch.utils.tensorboard import SummaryWriter

from easyvolcap.engine import cfg
from easyvolcap.engine import RECORDERS
from easyvolcap.engine.config import WithoutKey
from easyvolcap.utils.base_utils import dotdict, default_dotdict
from easyvolcap.utils.dist_utils import get_rank
from easyvolcap.utils.console_utils import *


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.value = 0

    def update(self, value):
        if isinstance(value, torch.Tensor): value = value.mean().item()

        self.deque.append(value)
        self.count += 1
        self.total += value
        self.value = value

    @property
    def latest(self):
        d = torch.as_tensor(list(self.deque))
        return d[-1].item()

    @property
    def median(self):
        d = torch.as_tensor(list(self.deque))
        return d.float().median().item()

    @property
    def avg(self):
        d = torch.as_tensor(list(self.deque))
        return d.float().mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def val(self):
        return self.value


@RECORDERS.register_module()
class EgoMoCapTensorboardRecorder:
    def __init__(self,
                 record_dir: str = f'data/record/{cfg.exp_name}',
                 resume: bool = True,
                 verbose: bool = True,
                 record_config: bool = True,
                 ):
        if not resume:
            if os.path.isdir(record_dir) and len(os.listdir(record_dir)):
                try: run('rm -r {}'.format(record_dir), quite=not verbose)
                except: pass
        self.record_dir = record_dir

        self.epoch = 0
        self.iter = 0
        self.scalar_stats = default_dotdict(SmoothedValue)

        self.image_stats = dotdict()

        if record_config:
            with WithoutKey('eglctx', cfg):
                cfg_file = join(record_dir, f'{cfg.exp_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.yaml')
                cfg.dump(cfg_file, indent=4)
                log(f'Saved config file to {blue(cfg_file)}')

    def update_scalar_stats(self, scalar_stats: dotdict):
        for k, v in scalar_stats.items():
            self.scalar_stats[k].update(v)

        keys = list(self.scalar_stats.keys())
        for k in keys:
            if k not in scalar_stats:
                del self.scalar_stats[k]

    def update_image_stats(self, image_stats: dotdict):
        self.image_stats.clear()
        for k, v in image_stats.items():
            self.image_stats[k] = v

        keys = list(self.image_stats.keys())
        for k in keys:
            if k not in image_stats:
                del self.image_stats[k]

    def state_dict(self):
        state = dotdict()
        state.iter = self.iter
        state.epoch = self.epoch
        return state

    def load_state_dict(self, state_dict: dotdict):
        state_dict = dotdict(state_dict)
        self.iter = state_dict.iter
        self.epoch = state_dict.epoch

    @property
    def log_stats(self):
        log_stats = dotdict()
        log_stats.epoch = str(self.epoch)
        log_stats.iter = str(self.iter)
        for k, v in self.scalar_stats.items():
            log_stats[k] = f'{v.avg:.6f}' if isinstance(v, SmoothedValue) else v
        log_stats.lr = f'{self.scalar_stats.lr.val:.6f}'
        log_stats.data = f'{self.scalar_stats.data.val:.4f}'
        log_stats.batch = f'{self.scalar_stats.batch.val:.4f}'
        log_stats.max_mem = f'{self.scalar_stats.max_mem.val:.0f}'
        return log_stats

    def __str__(self):
        return '  '.join([k + ': ' + v for k, v in self.log_stats.items()])

    def record(self,
               prefix: str,
               step: int = None,
               scalar_stats: dotdict = None,
               image_stats: dotdict = None,
               ):
        pattern = prefix + '/{}'
        step = step or self.iter
        scalar_stats = scalar_stats or self.scalar_stats
        image_stats = image_stats or self.image_stats

        if not hasattr(self, 'writer'):
            self.writer = SummaryWriter(log_dir=self.record_dir, filename_suffix=datetime.now().strftime("%Y%m%d_%H%M"))

        for k, v in scalar_stats.items():
            v = v.median if isinstance(v, SmoothedValue) else v
            self.writer.add_scalar(pattern.format(k), v, step)

        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step, dataformats='HWC')
