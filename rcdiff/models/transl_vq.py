import torch
from torch import nn
from typing import Union
import numpy as np
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import to_x
from easyvolcap.engine import MODELS, NETWORKS


@MODELS.register_module()
class TranslVQVAE(nn.Module):
    def __init__(self,
                 network_cfg: dotdict,
                 ):
        super().__init__()
        self.network = NETWORKS.build(network_cfg)

    def forward(self, batch: dotdict, dataloader: bool = None):
        pose_seql, pose_seqf = batch.pos3dl, batch.pos3df

        transl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]) * 20.0
        output_transl, loss, metrics = self.network(transl)
        output = dotdict()
        output.pred_transl = output_transl
        output.loss = loss['loss']
        output.scalar_stats = dotdict(loss=loss['loss'],
                                      recons_loss=loss['recons_loss'],
                                      commit_loss=loss['commit_loss'],
                                      velocity_loss=loss['velocity_loss'],
                                      acceleration_loss=loss['acceleration_loss'])

        return output

    def inference(self, batch: dotdict, dataloader: bool = None):
        pose_seql_eval, pose_seqf_eval  =  batch.pos3dl, batch.pos3df
        transl_eval = (pose_seqf_eval[:, :, :3] - pose_seql_eval[:, :, :3]).clone() * 20.0
        pose_seql_eval[:, :, :3] = pose_seql_eval[:, :, :3] - pose_seql_eval[:, :1, :3]

        transl_out, loss, _ = self.network(transl_eval)

        pose_seqf_eval[:, :, :3] = pose_seql_eval[:, :, :3] + transl_out / 20.0
        pose_seqf_eval = pose_seqf_eval.cpu().data.numpy()

        left_twist = pose_seqf_eval[:, :, 60:63]
        pose_seqf_eval[:, :, 75:120] = pose_seqf_eval[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

        right_twist = pose_seqf_eval[:, :, 63:66]
        pose_seqf_eval[:, :, 120:165] = pose_seqf_eval[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

        root = pose_seqf_eval[:, :, :3]
        pose_seqf_eval = pose_seqf_eval + np.tile(root, (1, 1, 55))
        pose_seqf_eval[:, :, :3] = root

        pose_seql_eval = pose_seql_eval.cpu().data.numpy()
        left_twist = pose_seql_eval[:, :, 60:63]
        pose_seql_eval[:, :, 75:120] = pose_seql_eval[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

        right_twist = pose_seql_eval[:, :, 63:66]
        pose_seql_eval[:, :, 120:165] = pose_seql_eval[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

        root = pose_seql_eval[:, :, :3]
        pose_seql_eval = pose_seql_eval + np.tile(root, (1, 1, 55))
        pose_seql_eval[:, :, :3] = root

        output = dotdict()
        output.follower_motion = pose_seqf_eval
        output.leader_motion = pose_seql_eval
        output.transl_pred = transl_out
        output.transl_gt = transl_eval
        return output

