import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from development.models.networks.modules.diffusion_transformer import TemporalDiffusionTransformerDecoderLayer
from development.utils.normalization import LatentNormalizer


class RCDiff(nn.Module):
    def __init__(self, njoints, nfeats, f_name='npys', latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, activation="gelu",
                 leader_joints=55, follower_joints=36, causal=True, block_size=100,
                 pred_mode='x0', **kargs):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.causal = causal
        self.pred_mode = pred_mode

        self.latent_dim = latent_dim
        self.block_size = block_size

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout, block_size=block_size)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    mot1_latent_dim=latent_dim,
                    time_embed_dim=latent_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout
                )
            )

        self.names_x = ["up", "down", "lhand", "rhand", "transl", "contact"]
        self.names_cmx = ["up", "down", "lhand", "rhand", "music"]
        self.Kx = len(self.names_x)
        self.Kc = len(self.names_cmx)

        all_types = ["up", "down", "lhand", "rhand", "transl", "contact", "music"]
        self.part_adapters = nn.ModuleDict({
            name: PartAdapter(latent_dim, latent_dim, dropout)
            for name in all_types
        })
        self.output_process = OutputProcess(latent_dim, leader_joints, follower_joints, self.names_x)

        self.normalizer = LatentNormalizer(
            stats_dir=f_name,
            parts=['up', 'down', 'lhand', 'rhand', 'transl', 'contact', 'music'],
        )

    def _split_part_sequence(self, x, num_parts):
        B, S, D = x.shape
        assert S % num_parts == 0, "Sequence length must be divisible by the number of parts"
        T = S // num_parts
        parts = [x[:, i * T:(i + 1) * T, :] for i in range(num_parts)]
        return parts

    def _project_part_features(self, parts, names):
        return [self.part_adapters[name](part) for part, name in zip(parts, names)]

    def _pack_part_sequence(self, parts, names):
        x = torch.stack(parts, dim=2)
        B, T, K, D = x.shape
        x = rearrange(x, 'b t k d -> b (k t) d').contiguous()

        base = torch.arange(T, device=x.device)
        pos_chunks = []
        for name in names:
            offset = self.sequence_pos_encoder.offset_id[name] * self.block_size
            pos_chunks.append(base + offset)
        pos_index = torch.cat(pos_chunks, dim=0)

        return x + self.sequence_pos_encoder.pe[pos_index, 0, :]

    def _prepare_transformer_inputs(self, noisy_latents, condition_latents):
        noisy_parts = self._split_part_sequence(noisy_latents, self.Kx)
        condition_parts = self._split_part_sequence(condition_latents, self.Kc)

        noisy_parts = self._project_part_features(noisy_parts, self.names_x)
        condition_parts = self._project_part_features(condition_parts, self.names_cmx)

        noisy_tokens = self._pack_part_sequence(noisy_parts, self.names_x)
        condition_tokens = self._pack_part_sequence(condition_parts, self.names_cmx)
        return noisy_tokens, condition_tokens

    def _decode_prediction_preview(self, output, timesteps, motoken_net, transl_net, contact_net):
        if self.training or timesteps[0] % 5 != 0:
            return None, None, None, None, None

        B, L, D = output.size()
        l = L // 6
        contact = self.normalizer.denormalize(output[:, -l:], 'contact')
        contact = contact_net.from_latent_to_fea(contact, path=True)
        contact = contact_net.decode([contact])

        part_names = ['up', 'down', 'lhand', 'rhand']
        motion_parts = [output[:, i * l:(i + 1) * l].clone() for i in range(4)]
        motion_parts = [
            fea for fea in motoken_net.from_latent_to_fea(
                [self.normalizer.denormalize(part, name) for part, name in zip(motion_parts, part_names)],
                path=True
            )
        ]
        transl = self.normalizer.denormalize(output[:, 4 * l:5 * l], 'transl')
        transl = transl_net.from_latent_to_fea(transl, path=True)
        pose_sample, rotmat_sample, vel_sample = motoken_net.decode([[p] for p in motion_parts])
        lf_transl = transl_net.decode([transl])
        return pose_sample, rotmat_sample, vel_sample, lf_transl, contact

    def forward(self, x, timesteps, cmx, motoken_net=None, transl_net=None, contact_net=None, **kargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        noisy_tokens, condition_tokens = self._prepare_transformer_inputs(x, cmx)
        timestep_embedding = self.embed_timestep(timesteps).permute(1, 0, 2).squeeze(1)
        for module in self.temporal_decoder_blocks:
            noisy_tokens, s_attn, c_attn = module(noisy_tokens, condition_tokens, timestep_embedding)

        pred = self.output_process(noisy_tokens)
        pose_sample, rotmat_sample, vel_sample, lf_transl, contact = self._decode_prediction_preview(
            pred, timesteps, motoken_net, transl_net, contact_net
        )
        return pred, pose_sample, rotmat_sample, vel_sample, lf_transl, contact


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, block_size=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.block_size = block_size

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

        self.offset_id = {
            "up": 0, "down": 1, "lhand": 2, "rhand": 3,
            "transl": 4, "contact": 5, "music": 6,
        }

    def forward(self, x, names):
        K = len(names)
        assert x.shape[0] % K == 0, "Sequence length must be T*K"
        T = x.shape[0] // K

        base = torch.arange(T, device=x.device)  # [0..T-1]
        pos_chunks = []
        for nm in names:
            off = self.offset_id[nm] * self.block_size
            pos_chunks.append(base + off)
        pos_index = torch.cat(pos_chunks, dim=0)  # [S]

        x = x + self.pe[pos_index, :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class PartAdapter(nn.Module):
    def __init__(self, d_in, d_model, p_drop=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.ln   = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        return self.drop(self.ln(self.proj(x)))


class OutputProcess(nn.Module):
    def __init__(self, latent_dim, leader_joints, follower_joints, names_x):
        super().__init__()
        self.leader_joints = leader_joints
        self.follower_joints = follower_joints
        self.names_x = names_x

        def head():
            return nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            )

        self.heads = nn.ModuleDict({
            "up": head(),
            "down": head(),
            "lhand": head(),
            "rhand": head(),
            "transl": head(),
            "contact": head()
        })

    def forward(self, tokens):
        B, S, D = tokens.shape
        K = len(self.names_x)
        assert S % K == 0, f"Seq len {S} must be divisible by K={K}"
        T = S // K

        x_view = tokens.view(B, K, T, D)
        parts_TBD = [x_view[:, ki, :, :] for ki in range(K)]

        outs = []
        for part_TBD, name in zip(parts_TBD, self.names_x):
            h = self.heads[name]
            outs.append(h(part_TBD))  # [B,T,D]

        x = torch.stack(outs, dim=1)                 # [B,K,T,D]
        x = x.view(B, S, D)                          # [B,K*T,D]
        return x



