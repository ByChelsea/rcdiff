from easyvolcap.engine import cfg, args
from easyvolcap.utils.console_utils import *
from easyvolcap.engine import EVALUATORS, cfg
import os
import numpy as np
import torch


def error_xy(x, y):
    return ((x - y) ** 2).sum(-1).sqrt().mean()


@EVALUATORS.register_module()
class ContactVQEvaluator:
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

    def compute_accuracy(self, output, batch):
        pred, target = output.output_contact, batch.contact
        TP = (pred * target).sum()  # True Positive
        FP = (pred * (1 - target)).sum()  # False Positive
        FN = ((1 - pred) * target).sum()  # False Negative
        TN = ((1 - pred) * (1 - target)).sum()  # True Negative
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        error = dotdict()
        error.accuracy = accuracy.item()
        return error

    def compute_precision(self, output, batch):
        pred, target = output.output_contact, batch.contact
        TP = (pred * target).sum()  # True Positive
        FP = (pred * (1 - target)).sum()  # False Positive
        FN = ((1 - pred) * target).sum()  # False Negative
        TN = ((1 - pred) * (1 - target)).sum()  # True Negative
        precision = TP / (TP + FP + 1e-10)

        error = dotdict()
        error.precision = precision.item()
        return error

    def compute_recall(self, output, batch):
        pred, target = output.output_contact, batch.contact
        TP = (pred * target).sum()  # True Positive
        FP = (pred * (1 - target)).sum()  # False Positive
        FN = ((1 - pred) * target).sum()  # False Negative
        TN = ((1 - pred) * (1 - target)).sum()  # True Negative
        recall = TP / (TP + FN + 1e-10)

        error = dotdict()
        error.recall = recall.item()
        return error

    def compute_f1(self, output, batch):
        pred, target = output.output_contact, batch.contact
        TP = (pred * target).sum()  # True Positive
        FP = (pred * (1 - target)).sum()  # False Positive
        FN = ((1 - pred) * target).sum()  # False Negative
        TN = ((1 - pred) * (1 - target)).sum()  # True Negative
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        error = dotdict()
        error.f1 = f1.item()
        return error

    def compute_iou(self, output, batch):
        pred, target = output.output_contact, batch.contact
        TP = (pred * target).sum()  # True Positive
        FP = (pred * (1 - target)).sum()  # False Positive
        FN = ((1 - pred) * target).sum()  # False Negative
        TN = ((1 - pred) * (1 - target)).sum()  # True Negative
        iou = TP / (TP + FP + FN + 1e-10)

        error = dotdict()
        error.iou = iou.item()
        return error

    def compute_transl_error(self, output, batch):
        error = dotdict()
        gt, pred = output.gt_transl, output.output_transl
        error.transl = error_xy(gt, pred).item()
        return error

    def save_and_visualize(self, output, batch, exp_name, epoch, iter, dataloader, vis=False):
        # paths
        name = batch.meta.file_name[0]
        save_dir = f'data/trained_model/{exp_name}/eval/{epoch}'
        os.makedirs(save_dir, exist_ok=True)
        gt_path = os.path.join(save_dir, f'{name}_gt.npy')
        pred_path = os.path.join(save_dir, f'{name}_pred.npy')
        np.save(gt_path, batch.contact.cpu().numpy())
        np.save(pred_path, output.contact_prob.detach().cpu().numpy())




