import numpy as np
import torch as t
import torch.nn as nn

from .modules.encdec import Encoder, Decoder, assert_shape
from .modules.bottleneck import NoBottleneck, Bottleneck
from development.utils.metrics import average_metrics
from easyvolcap.engine import NETWORKS


def dont_update(params):
    for param in params:
        param.requires_grad = False


def update(params):
    for param in params:
        param.requires_grad = True


def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target))


@NETWORKS.register_module()
class VQVAE(nn.Module):
    def __init__(self, input_dim, sample_length, levels, downs_t, strides_t, emb_width, l_bins, l_mu, commit,
                 hvqvae_multipliers, use_bottleneck, width, depth, m_conv, dilation_growth_rate,
                 vqvae_reverse_decoder_dilation, reg=0, acc=0, vel=0):
        super().__init__()
        input_shape = (sample_length, input_dim)
        mu = l_mu
        multipliers = hvqvae_multipliers
        dilation_cycle = None
        block_kwargs = dict(width=width, depth=depth, m_conv=m_conv, dilation_growth_rate=dilation_growth_rate,
                            dilation_cycle=dilation_cycle, reverse_decoder_dilation=vqvae_reverse_decoder_dilation)

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels

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

        encoder = lambda level: Encoder(x_channels, emb_width, level + 1,
                                        downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        decoder = lambda level: Decoder(x_channels, emb_width, level + 1,
                                        downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.reg = reg
        self.acc = acc
        self.vel = vel
        # self.spectral = spectral
        # self.multispectral = multispectral

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def _feat(self, zs, start_level=0, end_level=None, bs_chunks=1):
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)

        return xs_quantised[0]

    def feat(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_feats = []

        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_feat = self._feat(zs_i, start_level=start_level, end_level=end_level)
            x_feats.append(x_feat)

        return t.cat(x_feats, dim=0)

    def _decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = zs
        assert len(xs_quantised) == end_level - start_level

        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]  # [1, 512, 588]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def _encode_latent(self, x, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        return xs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def encode_latent(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode_latent(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def from_latent_to_idx(self, x):
        zs = self.bottleneck.encode([x.permute(0, 2, 1)])
        return zs[0]

    def from_latent_to_fea(self, x, path=False):
        x = x.permute(0, 2, 1)
        zs = self.bottleneck.encode([x])
        zs = self.bottleneck.decode(zs)[0]
        if path:
            zs = x + (zs - x).detach()
        return zs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x):
        metrics = {}

        N = x.shape[0]  # 128, 240, 3

        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level + 1], all_levels=False)
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        recons_loss = t.zeros(()).to(x.device)
        regularization = t.zeros(()).to(x.device)
        velocity_loss = t.zeros(()).to(x.device)
        acceleration_loss = t.zeros(()).to(x.device)
        x_target = x.float()

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            this_recons_loss = _loss_fn(x_target, x_out)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            recons_loss += this_recons_loss
            regularization += t.mean((x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1]) ** 2)

            velocity_loss += _loss_fn(x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
            acceleration_loss += _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1],
                                          x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        commit_loss = sum(commit_losses)
        loss = recons_loss + commit_loss * self.commit + self.reg * regularization + self.vel * velocity_loss + self.acc * acceleration_loss

        loss_list = {'loss': loss, 'recons_loss': recons_loss, 'commit_loss': commit_loss, 'velocity_loss': velocity_loss, 'acceleration_loss': acceleration_loss}

        with t.no_grad():
            l1_loss = _loss_fn(x_target, x_out)

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

        return x_out, loss_list, metrics
