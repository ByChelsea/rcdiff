import torch
import torch.nn as nn

from .modules.part_vq_encoder import PartVQEncoder
from .modules.encdec import Decoder, assert_shape
from development.utils.metrics import average_metrics
from easyvolcap.engine import NETWORKS


smpl_down = [0, 1, 2, 4, 5, 7, 8, 10, 11]
smpl_up = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
smpl_lhand = list(range(25, 40))
smpl_rhand = list(range(40, 55))


def _loss_fn(x_target, x_pred, weighted=False):
    if not weighted:
        return torch.mean(torch.abs(x_pred - x_target))
    else:
        n, tt, c = x_target.size()
        return torch.mean(torch.abs(x_pred - x_target))


@NETWORKS.register_module()
class SepVQVAE(nn.Module):
    def __init__(self, up_half, down_half, lhand, rhand, joint_channel, rot_joint_channel, pos_channel, emb_width,
                 downs_t, strides_t, levels, multipliers, width, depth, m_conv, dilation_growth_rate,
                 vqvae_reverse_decoder_dilation, rot_channel, commit, acc, vel):
        super().__init__()
        self.chanel_num = joint_channel
        self.rot_chanel_num = rot_joint_channel
        self.vqvae_up = PartVQEncoder(up_half)
        self.vqvae_down = PartVQEncoder(down_half)
        self.vqvae_lhand = PartVQEncoder(lhand)
        self.vqvae_rhand = PartVQEncoder(rhand)
        self.levels = levels
        self.commit = commit
        self.acc = acc
        self.vel = vel
        block_kwargs = dict(width=width, depth=depth, m_conv=m_conv, dilation_growth_rate=dilation_growth_rate,
                            dilation_cycle=None, reverse_decoder_dilation=vqvae_reverse_decoder_dilation)

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        decoder = lambda level: Decoder(pos_channel, emb_width, level + 1,
                                        downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        decoder_rot = lambda level: Decoder(rot_channel, emb_width, level + 1,
                                            downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        decoder_root = lambda level: Decoder(joint_channel, emb_width, level + 1,
                                             downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        fuse = lambda level: nn.Linear(emb_width * 4, emb_width)
        self.decoders = nn.ModuleList()
        self.decoders_rot = nn.ModuleList()
        self.decoders_root = nn.ModuleList()
        self.fuses = nn.ModuleList()
        for level in range(levels):
            self.decoders.append(decoder(level))
            self.decoders_rot.append(decoder_rot(level))
            self.decoders_root.append(decoder_root(level))
            self.fuses.append(fuse(level))

    def decode(self, x, start_level=0, end_level=None, bs_chunks=1):
        """
        zs are list with two elements: z for up and z for down
        """
        zup, zdown, zlhand, zrhand = x[0], x[1], x[2], x[3]

        x_outs = []
        x_outs_vel = []
        x_outs_pos = []

        for level in range(self.levels):
            xs_quantised = torch.cat([zup[level], zlhand[level], zrhand[level], zdown[level]], dim=1).permute(0, 2, 1).contiguous()
            xs_quantised = self.fuses[level](xs_quantised).permute(0, 2, 1).contiguous()
            decoder_pos = self.decoders[level]
            decoder = self.decoders_rot[level]
            decoder_root = self.decoders_root[level]
            x_pos_out = decoder_pos([xs_quantised], all_levels=False)  # [128, 165, 240]
            x_out = decoder([xs_quantised], all_levels=False)  # [128, 495, 240]
            x_vel_out = decoder_root([xs_quantised], all_levels=False)  # [128, 3, 240]

            x_outs.append(x_out.permute(0, 2, 1).contiguous())
            x_outs_pos.append(x_pos_out.permute(0, 2, 1).contiguous())
            x_outs_vel.append(x_vel_out.permute(0, 2, 1).contiguous())

        return x_outs_pos[0], x_outs[0], x_outs_vel[0]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        b, t, c = x.size()
        zup = self.vqvae_up.encode(x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_up].view(b, t, -1),
                                   start_level, end_level, bs_chunks)
        zdown = self.vqvae_down.encode(
            x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_down].view(b, t, -1), start_level, end_level,
            bs_chunks)
        zlhand = self.vqvae_lhand.encode(
            x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_lhand].view(b, t, -1), start_level,
            end_level, bs_chunks)
        zrhand = self.vqvae_rhand.encode(
            x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_rhand].view(b, t, -1), start_level,
            end_level, bs_chunks)
        return (zup, zdown, zlhand, zrhand)

    def encode_latent(self, x, start_level=0, end_level=None, bs_chunks=1):
        b, t, c = x.size()
        zup = self.vqvae_up.encode_latent(x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_up].view(b, t, -1),
                                          start_level, end_level, bs_chunks)
        zdown = self.vqvae_down.encode_latent(
            x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_down].view(b, t, -1), start_level, end_level, bs_chunks)
        zlhand = self.vqvae_lhand.encode_latent(
            x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_lhand].view(b, t, -1), start_level, end_level, bs_chunks)
        zrhand = self.vqvae_rhand.encode_latent(
            x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_rhand].view(b, t, -1), start_level, end_level, bs_chunks)
        return (zup, zdown, zlhand, zrhand)

    def from_latent_to_idx(self, x):
        xup, xdown, xlhand, xrhand = x[0].permute(0, 2, 1), x[1].permute(0, 2, 1), \
                                     x[2].permute(0, 2, 1), x[3].permute(0, 2, 1)
        zup = self.vqvae_up.from_latent_to_idx([xup])
        zdown = self.vqvae_down.from_latent_to_idx([xdown])
        zlhand = self.vqvae_lhand.from_latent_to_idx([xlhand])
        zrhand = self.vqvae_rhand.from_latent_to_idx([xrhand])
        return zup, zdown, zlhand, zrhand

    def from_latent_to_fea(self, x, path=False):
        xup, xdown, xlhand, xrhand = x[0].permute(0, 2, 1), x[1].permute(0, 2, 1), \
                                     x[2].permute(0, 2, 1), x[3].permute(0, 2, 1)
        zup = self.vqvae_up.from_latent_to_fea([xup], path)
        zdown = self.vqvae_down.from_latent_to_fea([xdown], path)
        zlhand = self.vqvae_lhand.from_latent_to_fea([xlhand], path)
        zrhand = self.vqvae_rhand.from_latent_to_fea([xrhand], path)
        return zup, zdown, zlhand, zrhand

    def sample(self, n_samples):
        xup = self.vqvae_up.sample(n_samples)[0].permute(0, 2, 1).contiguous()
        xdown = self.vqvae_down.sample(n_samples)[0].permute(0, 2, 1).contiguous()
        xlhand = self.vqvae_lhand.sample(n_samples)[0].permute(0, 2, 1).contiguous()
        xrhand = self.vqvae_rhand.sample(n_samples)[0].permute(0, 2, 1).contiguous()
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        _, _, clh = xlhand.size()
        _, _, crh = xrhand.size()

        x = torch.zeros(b, t, (cup + cdown + clh + crh) // self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup // self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown // self.chanel_num, self.chanel_num)
        x[:, :, smpl_lhand] = xlhand.view(b, t, clh // self.chanel_num, self.chanel_num)
        x[:, :, smpl_rhand] = xrhand.view(b, t, crh // self.chanel_num, self.chanel_num)
        return x

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def add_suffix(self, metrics_data, suffix):
        metrics_data = [
            {f"{key}_{suffix}": value for key, value in metrics.items()}
            for metrics in metrics_data
        ]
        return metrics_data

    def forward(self, x, xrot, xshift):
        b, t, c = x.size()
        _, _, crot = xrot.size()

        x, xrot = x.view(b, t, c // self.chanel_num, self.chanel_num), xrot.view(b, t, crot // self.rot_chanel_num,
                                                                                 self.rot_chanel_num)

        xup = x[:, :, smpl_up, :].view(b, t, -1)
        xdown = x[:, :, smpl_down, :].view(b, t, -1)
        xlhand = x[:, :, smpl_lhand, :].view(b, t, -1)
        xrhand = x[:, :, smpl_rhand, :].view(b, t, -1)

        xrot = xrot.view(b, t, -1)
        x = x.view(b, t, -1)

        _, xs_quantised_up, commit_losses_up, quantiser_metrics_up = self.vqvae_up(xup)
        _, xs_quantised_lhand, commit_losses_lhand, quantiser_metrics_lhand = self.vqvae_lhand(xlhand)
        _, xs_quantised_rhand, commit_losses_rhand, quantiser_metrics_rhand = self.vqvae_rhand(xrhand)
        _, xs_quantised_down, commit_losses_down, quantiser_metrics_down = self.vqvae_down(xdown)
        commit_loss = sum(commit_losses_up) + sum(commit_losses_lhand) + sum(commit_losses_rhand) + sum(commit_losses_down)
        if self.training:
            quantiser_metrics_up = self.add_suffix(quantiser_metrics_up, 'up')
            quantiser_metrics_lhand = self.add_suffix(quantiser_metrics_lhand, 'lhand')
            quantiser_metrics_rhand = self.add_suffix(quantiser_metrics_rhand, 'rhand')
            quantiser_metrics_down = self.add_suffix(quantiser_metrics_down, 'down')
            quantiser_metrics = [
                {**quantiser_metrics_up[0], **quantiser_metrics_lhand[0], **quantiser_metrics_rhand[0], **quantiser_metrics_down[0]}
            ]
        else:
            quantiser_metrics = quantiser_metrics_up

        x_outs = []
        x_outs_vel = []
        x_outs_pos = []

        for level in range(self.levels):
            xs_quantised = torch.cat([xs_quantised_up[level], xs_quantised_lhand[level],
                                      xs_quantised_rhand[level], xs_quantised_down[level]], dim=1).permute(0, 2, 1).contiguous()
            xs_quantised = self.fuses[level](xs_quantised).permute(0, 2, 1).contiguous()
            decoder_pos = self.decoders[level]
            decoder = self.decoders_rot[level]
            decoder_root = self.decoders_root[level]
            x_pos_out = decoder_pos([xs_quantised], all_levels=False)  # [128, 165, 240]
            x_out = decoder([xs_quantised], all_levels=False)  # [128, 495, 240]
            x_vel_out = decoder_root([xs_quantised], all_levels=False)  # [128, 3, 240]
            assert_shape(x_out, self.preprocess(xrot).shape)

            x_outs.append(x_out)
            x_outs_pos.append(x_pos_out)
            x_outs_vel.append(x_vel_out)

        recons_loss = torch.zeros(()).to(x.device)
        regularization = torch.zeros(()).to(x.device)
        velocity_loss = torch.zeros(()).to(x.device)
        acceleration_loss = torch.zeros(()).to(x.device)
        x_target = xrot.float()
        x_target_pos = x.float()
        x_target_shift = xshift.float()

        metrics = {}
        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            x_out_vel = self.postprocess(x_outs_vel[level])
            x_out_pos = self.postprocess(x_outs_pos[level])

            this_recons_loss = _loss_fn(x_target_pos, x_out_pos) + _loss_fn(x_target, x_out) + _loss_fn(x_target_shift,
                                                                                                        x_out_vel)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            recons_loss += this_recons_loss
            regularization += torch.mean((x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1]) ** 2)

            velocity_loss += _loss_fn(x_out_pos[:, 1:] - x_out_pos[:, :-1], x_target_pos[:, 1:] - x_target_pos[:, :-1]) \
                             + _loss_fn(x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])

            acceleration_loss += _loss_fn(x_out_pos[:, 2:] + x_out_pos[:, :-2] - 2 * x_out_pos[:, 1:-1],
                                          x_target_pos[:, 2:] + x_target_pos[:, :-2] - 2 * x_target_pos[:, 1:-1]) \
                                 + _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1],
                                            x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1]) + \
                                 _loss_fn(x_target_shift[:, 1:] - x_target_shift[:, :-1],
                                          x_out_vel[:, 1:] - x_out_vel[:, :-1])

        loss = recons_loss + commit_loss * self.commit + self.vel * velocity_loss + self.acc * acceleration_loss
        with torch.no_grad():
            l1_loss = _loss_fn(x_target_pos, x_out_pos)

        quantiser_metrics = average_metrics(quantiser_metrics)
        metrics.update(dict(
            recons_loss=recons_loss,
            l1_loss=l1_loss,
            commit_loss=commit_loss,
            regularization=regularization,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        loss_list = {'loss': loss, 'recons_loss': recons_loss, 'commit_loss': commit_loss,
                     'velocity_loss': velocity_loss, 'acceleration_loss': acceleration_loss}

        return x_out_pos, x_out, x_out_vel, loss_list, [metrics]
