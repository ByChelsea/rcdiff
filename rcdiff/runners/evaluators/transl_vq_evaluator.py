from easyvolcap.engine import cfg, args
from easyvolcap.utils.console_utils import *
from easyvolcap.engine import EVALUATORS, cfg
import os
import numpy as np
import torch


def error_xy(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.sqrt(((x - y) ** 2).sum(-1)).mean()
    elif torch.is_tensor(x) and torch.is_tensor(y):
        return ((x - y) ** 2).sum(-1).sqrt().mean()
    else:
        raise TypeError("Inputs must be both NumPy arrays or both PyTorch tensors.")


@EVALUATORS.register_module()
class TranslVQEvaluator:
    def __init__(self,
                 metric_names: List = []
                 ) -> None:
        # metrics
        self.metrics = dotdict()
        self.metric_names = metric_names

    def evaluate(self, output, batch, dataloader):
        metrics = dotdict()
        for metric_name in self.metric_names:
            metric = self.__getattribute__(f'compute_{metric_name}')(output, batch)
            for k, v in metric.items():
                metrics[k] = v
                if k in self.metrics:
                    self.metrics[k].append(v)
                else:
                    self.metrics[k] = [v]

        return metrics

    def summarize(self):
        summary = dotdict()
        if len(self.metrics):
            for key in self.metrics.keys():
                values = self.metrics[key]
                summary[f'{key}_mean'] = np.mean(values)
                # summary[f'{key}_std'] = np.std(values)
        self.metrics.clear()  # clear mean after extracting summary
        log(summary)
        return summary

    def compute_transl_error(self, output, batch):
        error = dotdict()
        gt, pred = output.transl_gt, output.transl_pred
        error.transl = error_xy(gt, pred).item()
        return error

    def save_and_visualize(self, output, batch, exp_name, epoch, iter, dataloader, vis=False):
        # paths
        name = batch.meta.file_name[0]
        save_dir = f'data/trained_model/{exp_name}/eval/{epoch}'
        os.makedirs(save_dir, exist_ok=True)
        leader_path = os.path.join(save_dir, f'{name}_01.npy')
        follower_path = os.path.join(save_dir, f'{name}_00.npy')
        leader, follower = output.leader_motion, output.follower_motion
        np.save(leader_path, leader)
        np.save(follower_path, follower)




