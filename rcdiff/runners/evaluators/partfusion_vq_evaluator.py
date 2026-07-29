from easyvolcap.engine import cfg, args
from easyvolcap.utils.console_utils import *
from easyvolcap.engine import EVALUATORS, cfg
import os
from typing import List
import numpy as np
import torch


def _to_joints(x):
    return x.reshape(*x.shape[:2], -1, 3)


def _mpjpe_mm(pred, gt):
    pred, gt = _to_joints(pred), _to_joints(gt)
    return np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0


def _mpjve_mm(pred, gt):
    pred_vel = _to_joints(pred[:, 1:] - pred[:, :-1])
    gt_vel = _to_joints(gt[:, 1:] - gt[:, :-1])
    return np.linalg.norm(pred_vel - gt_vel, axis=-1).mean() * 1000.0


@EVALUATORS.register_module()
class PartFusionVQEvaluator:
    def __init__(self,
                 metric_names: List = None
                 ) -> None:
        # metrics
        self.metrics = dotdict()
        self.metric_names = metric_names or []

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

    def _get_role(self, batch):
        name = batch.meta.file_name[0]
        if name.endswith('_01.npy'):
            return 'leader'
        if name.endswith('_00.npy'):
            return 'follower'
        return 'motion'

    def compute_mpjpe(self, output, batch):
        error = dotdict()
        gt, pred = output.gt_motion, output.pred_motion
        role = self._get_role(batch)
        error[f'mpjpe_{role}_mm'] = float(_mpjpe_mm(pred, gt))
        return error

    def compute_mpjve(self, output, batch):
        error = dotdict()
        gt, pred = output.gt_motion, output.pred_motion
        role = self._get_role(batch)
        error[f'mpjve_{role}_mm'] = float(_mpjve_mm(pred, gt))
        return error

    def save_and_visualize(self, output, batch, exp_name, epoch, iter, dataloader, vis=False):
        name = batch.meta.file_name[0][:-4]
        save_dir = f'data/trained_model/{exp_name}/eval/{epoch}'
        os.makedirs(save_dir, exist_ok=True)
        pred_path = os.path.join(save_dir, f'{name}.npy')
        pred = output.pred_motion
        np.save(pred_path, pred)
        if output.usage_counts_ups is not None:
            total_usage_up = torch.stack(output.usage_counts_ups, dim=0).sum(dim=0)
            total_usage_down = torch.stack(output.usage_counts_downs, dim=0).sum(dim=0)
            total_usage_lhand = torch.stack(output.usage_counts_lhands, dim=0).sum(dim=0)
            total_usage_rhand = torch.stack(output.usage_counts_rhands, dim=0).sum(dim=0)
            np.save(os.path.join(save_dir, 'up_usage.npy'), total_usage_up.cpu().numpy())
            np.save(os.path.join(save_dir, 'down_usage.npy'), total_usage_down.cpu().numpy())
            np.save(os.path.join(save_dir, 'lhand_usage.npy'), total_usage_lhand.cpu().numpy())
            np.save(os.path.join(save_dir, 'rhand_usage.npy'), total_usage_rhand.cpu().numpy())




