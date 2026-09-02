import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def pe1d_sincos(seq_length, dim):
    if dim % 2 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={dim})")
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    def __init__(self, seq_length, num, dim, dropout, grad=False):
        super().__init__()
        self.embed = nn.ParameterList([
            nn.Parameter(data=pe1d_sincos(seq_length, dim), requires_grad=grad)
            for _ in range(num)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.num = num

    def forward(self, x):
        B, N, _ = x.shape
        l = int(N / self.num)
        partial_embeds = [embed[:l] for embed in self.embed]
        pos_embeds = torch.cat(partial_embeds, dim=0)
        x = x.permute(1, 0, 2) + pos_embeds.expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x


def get_subsequent_mask(seq_len, sliding_window_size, causal=True):
    if causal:
        mask = torch.ones((seq_len, seq_len)).float()
        mask = torch.tril(mask, diagonal=sliding_window_size)
    else:
        mask = torch.ones((seq_len, seq_len)).float()
    return mask


class MaskAttention(nn.Module):
    def __init__(self, block_size, look_forward, n_embd, downsample_rate=4, n_head=16, attn_pdrop=0.1,
                 resid_pdrop=0.1, causal=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask",
            get_subsequent_mask(
                (block_size + look_forward) * downsample_rate,
                look_forward * downsample_rate,
                causal,
            )[None, None],
        )
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MusicBlock(nn.Module):
    def __init__(self, block_size, look_forward, n_embd, downsample_rate=4, resid_pdrop=0.1, causal=True, n_head=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MaskAttention(block_size, look_forward, n_embd, downsample_rate, resid_pdrop=resid_pdrop,
                                  causal=causal, n_head=n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MusicTrans(nn.Module):
    def __init__(self, block_size, n_embd, n_music, downsample_rate=4, embd_pdrop=0.1, n_layer=3,
                 look_forward=0, causal=True):
        super().__init__()
        self.block_size = block_size
        self.look_forward = look_forward
        self.downsample_rate = downsample_rate
        self.pos_emb = PositionEmbedding(
            (block_size + look_forward) * downsample_rate,
            1,
            n_embd,
            embd_pdrop,
            True,
        )
        self.cond_emb = nn.Linear(n_music, n_embd)
        self.blocks = nn.Sequential(
            *[
                MusicBlock(block_size, look_forward, n_embd, downsample_rate, resid_pdrop=embd_pdrop, causal=causal)
                for _ in range(n_layer)
            ]
        )
        self.downsample = nn.Linear(n_embd * downsample_rate, n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, music):
        b, t, c = music.size()
        max_length = (self.block_size + self.look_forward) * self.downsample_rate
        assert t <= max_length, "Cannot forward, model block size is exhausted."
        assert t % self.downsample_rate == 0, "Music sequence length must be divisible by downsample_rate."
        x = self.cond_emb(music)
        x = self.pos_emb(x)
        x = self.blocks(x)
        b, t, c = x.size()
        x = self.downsample(x.view(b, t // self.downsample_rate, c * self.downsample_rate))
        return x if self.look_forward == 0 else x[:, :-self.look_forward, :].contiguous()
