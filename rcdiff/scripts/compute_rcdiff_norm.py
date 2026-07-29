import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm


PARTS = ("up", "down", "lhand", "rhand", "transl", "contact", "music")


def load_config(path):
    from rcdiff.utils.engine_utils import parse_args_list
    return parse_args_list([f"-c={path}"])


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sum_sq = None

    def update(self, x):
        x = x.detach().float()
        x = x.reshape(-1, x.shape[-1])
        batch_sum = x.sum(dim=0, dtype=torch.float64).cpu()
        batch_sum_sq = (x.double() * x.double()).sum(dim=0).cpu()

        if self.sum is None:
            self.sum = batch_sum
            self.sum_sq = batch_sum_sq
        else:
            self.sum += batch_sum
            self.sum_sq += batch_sum_sq
        self.count += x.shape[0]

    def mean_std(self, eps=1e-6):
        mean = self.sum / self.count
        var = self.sum_sq / self.count - mean * mean
        std = var.clamp_min(eps * eps).sqrt()
        return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        for key, value in batch.items():
            batch[key] = to_device(value, device)
    elif isinstance(batch, list):
        for i, value in enumerate(batch):
            batch[i] = to_device(value, device)
    return batch


def encode_batch(batch, motoken_net, transl_net, contact_net):
    music_seq = batch.music
    leader_pose = batch.pos3dl.clone()
    follower_pose = batch.pos3df.clone()
    contact_seq = batch.contact

    leader_to_follower = (follower_pose[:, :, :3] - leader_pose[:, :, :3]).clone() * 20.0

    follower_pose[:, :-1, :3] = follower_pose[:, 1:, :3] - follower_pose[:, :-1, :3]
    follower_pose[:, -1, :3] = follower_pose[:, -2, :3]

    leader_pose[:, :, :3] = 0
    follower_pose[:, :, :3] = 0

    leader_latents = [q[0].permute(0, 2, 1) for q in motoken_net.encode_latent(leader_pose)]
    follower_latents = [q[0].permute(0, 2, 1) for q in motoken_net.encode_latent(follower_pose)]
    translation_latent = transl_net.encode_latent(leader_to_follower)[0].permute(0, 2, 1)
    contact_latent = contact_net.encode_latent(contact_seq)[0].permute(0, 2, 1)

    return OrderedDict(
        up=torch.cat([leader_latents[0], follower_latents[0]], dim=0),
        down=torch.cat([leader_latents[1], follower_latents[1]], dim=0),
        lhand=torch.cat([leader_latents[2], follower_latents[2]], dim=0),
        rhand=torch.cat([leader_latents[3], follower_latents[3]], dim=0),
        transl=translation_latent,
        contact=contact_latent,
        music=music_seq,
    )


def save_stats(stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for part, meter in stats.items():
        mean, std = meter.mean_std()
        np.save(os.path.join(out_dir, f"{part}_mean.npy"), mean)
        np.save(os.path.join(out_dir, f"{part}_std.npy"), std)
        print(f"{part:>7}: count={meter.count} dim={mean.shape[0]} "
              f"mean_abs={np.abs(mean).mean():.6f} std_mean={std.mean():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/exps/rcdiff.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.argv = [sys.argv[0]]

    from easyvolcap.engine import DATALOADERS
    from rcdiff.utils.engine_utils import discover_modules
    from rcdiff.utils.net_utils import load_other_network

    cfg = load_config(args.config)
    discover_modules()

    out_dir = args.out or cfg.model_cfg.get("f_name", "npys")
    dataloader_cfg = cfg.dataloader_cfg
    dataloader_cfg.max_iter = -1
    dataloader_cfg.num_workers = args.num_workers
    dataloader_cfg.dataset_cfg.split = "train"
    if args.batch_size is not None:
        dataloader_cfg.batch_sampler_cfg.batch_size = args.batch_size
        dataloader_cfg.batch_sampler_cfg.drop_last = False

    motoken_cfg = load_config(cfg.motoken_cfg_file)
    transl_cfg = load_config(cfg.transl_cfg_file)
    contact_cfg = load_config(cfg.contact_cfg_file)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    motoken_net = load_other_network(motoken_cfg).to(device).eval()
    transl_net = load_other_network(transl_cfg).to(device).eval()
    contact_net = load_other_network(contact_cfg).to(device).eval()
    dataloader = DATALOADERS.build(dataloader_cfg)

    stats = OrderedDict((part, RunningStats()) for part in PARTS)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing RCDiff norm"):
            batch = to_device(batch, device)
            encoded = encode_batch(batch, motoken_net, transl_net, contact_net)
            for part, value in encoded.items():
                stats[part].update(value)

    save_stats(stats, out_dir)
    print(f"Saved RCDiff norm files to: {out_dir}")


if __name__ == "__main__":
    main()
