import os
import numpy as np
import torch
import torch.nn as nn


class LatentNormalizer(nn.Module):
    def __init__(self, stats_dir, parts, eps=1e-6):
        super().__init__()
        self.parts = tuple(parts)
        self.eps = eps

        for part in self.parts:
            mean = self._load_stat(stats_dir, part, "mean")
            std = self._load_stat(stats_dir, part, "std").clamp_min(eps)
            self.register_buffer(f"{part}_mean", mean)
            self.register_buffer(f"{part}_std", std)

    @staticmethod
    def _load_stat(stats_dir, part, suffix):
        path = os.path.join(stats_dir, f"{part}_{suffix}.npy")
        return torch.tensor(np.load(path), dtype=torch.float32).view(1, 1, -1)

    def normalize(self, x, part):
        mean = getattr(self, f"{part}_mean")
        std = getattr(self, f"{part}_std")
        return (x - mean) / std

    def denormalize(self, x, part):
        mean = getattr(self, f"{part}_mean")
        std = getattr(self, f"{part}_std")
        return x * std + mean
