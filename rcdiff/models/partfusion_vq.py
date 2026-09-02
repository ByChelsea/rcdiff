import torch
from torch import nn
from typing import Union
import numpy as np
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import to_x
from easyvolcap.engine import MODELS, NETWORKS


@MODELS.register_module()
class PartVQVAEs(nn.Module):
    def __init__(self,
                 network_cfg: dotdict,
                 ):
        super().__init__()
        self.network = NETWORKS.build(network_cfg)
        self.usage_counts_ups = []
        self.usage_counts_downs = []
        self.usage_counts_lhands = []
        self.usage_counts_rhands = []
        self.counts = 0

    def forward(self, batch: dotdict, dataloader: bool = None):
        pose_seq, rot_seq = batch.pos3d.float(), batch.rotmat.float()

        pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
        pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
        global_vel = pose_seq[:, :, :3].clone().detach()
        pose_seq[:, :, :3] = 0

        output_pos, output_rot, output_shift, loss, metrics = self.network(pose_seq, rot_seq, global_vel)
        output = dotdict()
        output.pred_pos = output_pos
        output.pred_rot = output_rot
        output.loss = loss['loss']
        output.scalar_stats = dotdict(loss=loss['loss'],
                                      recons_loss=loss['recons_loss'],
                                      commit_loss=loss['commit_loss'],
                                      velocity_loss=loss['velocity_loss'],
                                      acceleration_loss=loss['acceleration_loss'])

        return output

    def inference(self, batch: dotdict, dataloader: bool = None):
        pose_seq_eval = batch.pos3d.float()
        rot_seq_eval = batch.rotmat.float()

        src_pos_eval = pose_seq_eval[:, :].clone()
        src_pos_eval[:, :-1, :3] = src_pos_eval[:, 1:, :3] - src_pos_eval[:, :-1, :3]
        src_pos_eval[:, -1, :3] = src_pos_eval[:, -2, :3]
        global_vel = src_pos_eval[:, :, :3].clone().detach()
        src_pos_eval[:, :, :3] = 0

        codebook_size = 512
        output = self.network.encode(src_pos_eval)
        self.usage_counts_ups.append(torch.bincount(output[0][0].view(-1), minlength=codebook_size))
        self.usage_counts_downs.append(torch.bincount(output[1][0].view(-1), minlength=codebook_size))
        self.usage_counts_lhands.append(torch.bincount(output[2][0].view(-1), minlength=codebook_size))
        self.usage_counts_rhands.append(torch.bincount(output[3][0].view(-1), minlength=codebook_size))
        self.counts += 1

        pose_seq_out, rot_seq_out, shift_out, loss, _ = self.network(src_pos_eval, rot_seq_eval, global_vel)

        pose_seq_out = self.get_final_motion(pose_seq_out, global_vel)
        pose_seq_gt = self.get_final_motion(src_pos_eval, global_vel)

        output = dotdict()
        output.pred_motion = pose_seq_out
        output.gt_motion = pose_seq_gt
        if self.counts == 68:
            output.usage_counts_ups = self.usage_counts_ups
            output.usage_counts_downs = self.usage_counts_downs
            output.usage_counts_lhands = self.usage_counts_lhands
            output.usage_counts_rhands = self.usage_counts_rhands
        else:
            output.usage_counts_ups, output.usage_counts_downs, \
            output.usage_counts_lhands, output.usage_counts_rhands = None, None, None, None
        return output

    def get_final_motion(self, pose_seq_out, global_vel):
        pose_seq_out[:, 0, :3] = 0
        for iii in range(1, pose_seq_out.size(1)):
            pose_seq_out[:, iii, :3] = pose_seq_out[:, iii - 1, :3] + global_vel[:, iii - 1, :]

        pose_seq_out = pose_seq_out.cpu().data.numpy()
        left_twist = pose_seq_out[:, :, 60:63]
        pose_seq_out[:, :, 75:120] = pose_seq_out[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))
        right_twist = pose_seq_out[:, :, 63:66]
        pose_seq_out[:, :, 120:165] = pose_seq_out[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))
        root = pose_seq_out[:, :, :3]
        pose_seq_out = pose_seq_out + np.tile(root, (1, 1, 55))
        pose_seq_out[:, :, :3] = root
        return pose_seq_out


