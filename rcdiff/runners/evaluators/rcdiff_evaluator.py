import os
import numpy as np
from scipy.ndimage import gaussian_filter

from easyvolcap.engine import EVALUATORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import log


RC_DIFF_SAVE_SMOOTH_SIGMA = 0.8


def smooth_motion(motion):
    motion = np.asarray(motion)
    sigma = [0.0] * motion.ndim
    time_axis = 1 if motion.ndim >= 4 else 0
    sigma[time_axis] = RC_DIFF_SAVE_SMOOTH_SIGMA
    return gaussian_filter(motion, sigma=tuple(sigma), mode='nearest')


@EVALUATORS.register_module()
class RCDiffEvaluator:
    def __init__(self,
                 metric_names: list = [],
                 ) -> None:
        self.metrics = dotdict()
        self.metric_names = metric_names

    def evaluate(self, output, batch, dataloader):
        metrics = dotdict()
        for metric_name in self.metric_names:
            metric = getattr(self, f"compute_{metric_name}")(output, batch, dataloader)
            for key, value in metric.items():
                metrics[key] = value
                self.metrics.setdefault(key, []).append(value)

        return metrics

    def summarize(self):
        summary = dotdict()
        for key, values in self.metrics.items():
            summary[f"{key}_mean"] = np.mean(values)
        self.metrics.clear()
        log(summary)
        return summary

    def save_and_visualize(self, output, batch, exp_name, epoch, iter, dataloader, vis=False):
        name = batch.meta.file_name[0]
        save_dir = f'data/trained_model/{exp_name}/eval/{epoch}'
        os.makedirs(save_dir, exist_ok=True)
        leader_path = os.path.join(save_dir, f'{name}_01.npy')
        follower_path = os.path.join(save_dir, f'{name}_00.npy')
        contact_path = os.path.join(save_dir, f'{name}_c.npy')
        follower = smooth_motion(output.follower_motion)
        leader = smooth_motion(output.leader_motion)
        contact = output.get("contact_matrix", None)
        np.save(leader_path, leader)
        np.save(follower_path, follower)
        if contact is not None:
            np.save(contact_path, contact)

        save_dir_cf = os.path.join(save_dir, 'cf')
        os.makedirs(save_dir_cf, exist_ok=True)
        leader_path_cf = os.path.join(save_dir_cf, f'{name}_01.npy')
        follower_path_cf = os.path.join(save_dir_cf, f'{name}_00.npy')
        if output.dictl is not None:
            np.save(follower_path_cf, output.dictf)
            np.save(leader_path_cf, output.dictl)
