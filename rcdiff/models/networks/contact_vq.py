import numpy as np
import torch as t
import torch.nn as nn

from .modules.encdec import Encoder, Decoder, assert_shape
from .modules.bottleneck import NoBottleneck, Bottleneck
from development.utils.metrics import average_metrics
from easyvolcap.engine import NETWORKS
from torch.nn import functional as F


class PreprocessC(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        B, L = x.shape[:2]
        x = x.view(B * L, 1, 24, 24)
        x = self.cnn(x)
        return x.view(B, L, 512)


class PostprocessC(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Unflatten(-1, (128, 2, 2)),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=3, mode='nearest'),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        B, L = x.shape[:2]
        x = x.reshape(B * L, 512)
        x = self.cnn(x)
        return x.reshape(B, L, 24, 24)


def dont_update(params):
    for param in params:
        param.requires_grad = False


def update(params):
    for param in params:
        param.requires_grad = True


def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


def _loss_fn(x_target, x_pred):
    return nn.functional.binary_cross_entropy(x_pred, x_target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean', eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        if self.alpha == 'auto':
            pos_weight = (targets == 0).sum() / (targets.numel() + self.eps)
            alpha = pos_weight
        else:
            alpha = self.alpha

        p = t.sigmoid(inputs)

        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        p_t = p * targets + (1 - p) * (1 - targets)  # p if y=1 else 1-p
        modulating_factor = (1 - p_t) ** self.gamma

        loss = modulating_factor * ce_loss

        if alpha is not None:
            alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
            loss *= alpha_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



@NETWORKS.register_module()
class VQVAEC(nn.Module):
    def __init__(self, input_dim, sample_length, levels, downs_t, strides_t, emb_width, l_bins, l_mu, commit,
                 hvqvae_multipliers, use_bottleneck, width, depth, m_conv, dilation_growth_rate,
                 vqvae_reverse_decoder_dilation, persistence, sparsity, recon):
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
        self.persistence = persistence
        self.sparsity = sparsity
        self.recon = recon

        self.preprocessc = PreprocessC()
        self.postprocessc = PostprocessC()
        self.loss_focal = FocalLoss(alpha=0.8, gamma=2.0)

    def preprocess(self, x):
        x = self.preprocessc(x)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        x = self.postprocessc(x)
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
        contact_persistence_loss = t.zeros(()).to(x.device)
        contact_sparsity_loss = t.zeros(()).to(x.device)
        x_target = x.float()

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            this_recons_loss = self.loss_focal(x_out, x_target)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            recons_loss += this_recons_loss
            contact_persistence_loss += t.mean((x_out[:, 1:] - x_out[:, :-1]).abs())
            contact_sparsity_loss += t.mean(x_out.pow(2))

        commit_loss = sum(commit_losses)
        loss = (self.recon * recons_loss + self.commit * commit_loss +
                self.persistence * contact_persistence_loss + self.sparsity * contact_sparsity_loss)

        with t.no_grad():
            l1_loss = self.loss_focal(x_out, x_target)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            l1_loss=l1_loss,
            commit_loss=commit_loss,
            contact_persistence_loss=contact_persistence_loss,
            contact_sparsity_loss=contact_sparsity_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        loss_list = {'loss': loss, 'recons_loss': recons_loss, 'commit_loss': commit_loss}

        return x_out, loss_list, metrics
