import torch
from torch import nn
from typing import Union
import numpy as np
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import to_x
from easyvolcap.engine import MODELS, NETWORKS


@MODELS.register_module()
class ContactVQVAE(nn.Module):
    def __init__(self,
                 network_cfg: dotdict,
                 transl: bool = False,
                 ):
        super().__init__()
        self.network = NETWORKS.build(network_cfg)
        self.transl = transl

    def forward(self, batch: dotdict, dataloader: bool = None):
        output = dotdict()
        if self.transl:
            pose_seql, pose_seqf = batch.pos3dl, batch.pos3df
            transl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]) * 20.0
            output_contact, output_transl, loss, metrics = self.network(batch.contact, transl)
            output.output_transl = output_transl
            output.gt_transl = transl
        else:
            output_contact, loss, metrics = self.network(batch.contact)
            output.output_transl = None

        output.output_contact = output_contact
        output.loss = loss['loss']
        output.scalar_stats = dotdict(loss=loss['loss'],
                                      recons_loss=loss['recons_loss'],
                                      commit_loss=loss['commit_loss'])

        return output

    def inference(self, batch: dotdict, dataloader: bool = None):
        output = dotdict()
        if self.transl:
            pose_seql, pose_seqf = batch.pos3dl, batch.pos3df
            transl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]) * 20.0
            output_contact, output_transl, loss, metrics = self.network(batch.contact, transl)
            output.output_transl = output_transl
            output.gt_transl = transl
        else:
            output_contact, loss, metrics = self.network(batch.contact)
            output_contact = torch.sigmoid(output_contact)
            contact_prob = output_contact.clone()
            threshold = 0.5
            output_contact = (output_contact > threshold).float()
            output.output_transl = None

        output.output_contact = output_contact
        output.contact_prob = contact_prob
        return output

