import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import functools
import torch.distributed as dist
import time
import math

from easyvolcap.engine import MODELS, cfg, call_from_cfg
from easyvolcap.utils.data_utils import dotdict
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from development.models.networks.rcdiff import RCDiff
from development.models.networks.diffusion.sampler import UniformSampler, LossAwareSampler
from development.models.networks.modules.music_transformer import MusicTrans
from development.models.networks.diffusion import gaussian_diffusion as gd
from development.models.networks.diffusion.respace import SpacedDiffusion, space_timesteps
from development.utils import dist_util
from development.utils.normalization import LatentNormalizer
from scipy.spatial.transform import Rotation as R


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    Args:
        rotmats: A np array of shape (..., 3, 3)
    Returns:
        A np array of shape (..., 3)
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3 and len(rotmats.shape) >= 3, 'invalid input dimension'
    orig_shape = rotmats.shape[:-2]
    rots = np.reshape(rotmats, [-1, 3, 3])
    r = R.from_matrix(rots)
    aas = r.as_rotvec()
    return np.reshape(aas, orig_shape + (3,))


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = 1000
    scale_beta = 1.
    timestep_respacing = args.timestep_respacing
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        num_person=args.num_person
    )


def get_joints24(joint3d):
    joint22 = joint3d[:, :, :22]

    joint_left_hand_center = (joint3d[:, :, 25] + joint3d[:, :, 28] +
                              joint3d[:, :, 31] + joint3d[:, :, 34] +
                              joint3d[:, :, 37]) / 5.0  # [B, T, 3]
    joint_right_hand_center = (joint3d[:, :, 40] + joint3d[:, :, 43] +
                               joint3d[:, :, 46] + joint3d[:, :, 49] +
                               joint3d[:, :, 52]) / 5.0  # [B, T, 3]

    joint24 = torch.cat([joint22,
                         joint_left_hand_center.unsqueeze(2),
                         joint_right_hand_center.unsqueeze(2)], dim=2)
    return joint24


def cond_fn(x, t, p_mean_var, **kwargs):
    device = x.device
    if t[0] % 5 != 0:
        return torch.zeros_like(x)

    follower_pose = p_mean_var["pred_pose"].to(device)
    lf_transl = p_mean_var["pred_transl"].to(device)
    contact_logits = p_mean_var["pred_contact"].to(device)
    leader_pose = kwargs['pose_seql'].to(device)

    transll = leader_pose[:, :, :3]
    follower_motion, leader_motion = get_glb_pos(follower_pose, transll, lf_transl, leader_pose)
    follower_xyz = follower_motion.reshape(x.shape[0], -1, 55, 3)
    leader_xyz = leader_motion.reshape(x.shape[0], -1, 55, 3)

    follower_xyz, leader_xyz = get_joints24(follower_xyz), get_joints24(leader_xyz)

    contact_prob = torch.sigmoid(contact_logits)
    contact_mask = (contact_prob > 0.5).float()

    eps = 1e-8
    dist_sq = (follower_xyz.unsqueeze(3) - leader_xyz.unsqueeze(2)).pow(2).sum(dim=-1)
    weighted_loss = dist_sq * contact_mask
    weighted_loss = weighted_loss.sum() / (contact_mask.sum() + eps)

    lam = 100
    loss = weighted_loss
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    grad_norm = grad.norm(dim=(1, 2), keepdim=True).clamp_min(eps)
    grad_hat = grad / grad_norm
    return - lam * grad_hat


def get_glb_pos(pose_sample, transll, lf_transl, pose_seql):
    pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)].clone() + lf_transl.clone() / 20.0

    left_twist = pose_sample[:, :, 60:63]
    pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + left_twist.repeat(1, 1, 15)

    right_twist = pose_sample[:, :, 63:66]
    pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + right_twist.repeat(1, 1, 15)

    root = pose_sample[:, :, :3]
    pose_sample = pose_sample + root.repeat(1, 1, 55)
    pose_sample[:, :, :3] = root

    pose_seql[:, :, :3] = transll.clone()
    left_twist = pose_seql[:, :, 60:63]
    pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + left_twist.repeat(1, 1, 15)

    right_twist = pose_seql[:, :, 63:66]
    pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + right_twist.repeat(1, 1, 15)

    root = pose_seql[:, :, :3]
    pose_seql = pose_seql + root.repeat(1, 1, 55)
    pose_seql[:, :, :3] = root

    return pose_sample, pose_seql


@MODELS.register_module()
class TransformerDiffusion(nn.Module):
    def __init__(self,
                 network_cfg: dotdict,
                 diffusion_cfg: dotdict,
                 block_size: int = 48,
                 ds_rate: int = 4,
                 use_ddim: bool = True,
                 music_dim: int = 512,
                 f_name: str = 'npys',
                 causal: bool = True,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs)
        self.block_size = block_size
        self.ds_rate = ds_rate

        self.use_ddim = use_ddim
        self.music_trans = MusicTrans(block_size, music_dim, 54, causal=causal)

        self.network = RCDiff(**network_cfg)
        self.diffusion = create_gaussian_diffusion(diffusion_cfg)
        self.schedule_sampler = UniformSampler(self.diffusion)
        self.use_ddp = False
        self.ddp_model = self.network
        self.normalizer = LatentNormalizer(
            stats_dir=f_name,
            parts=['up', 'down', 'lhand', 'rhand', 'transl', 'contact', 'music'],
        )

    def forward(self, batch, dataloader):
        music_seq, _, leader_latents, follower_latents, translation_latent, contact_latent, \
            _ = self.prepare_inputs(batch)
        target_parts, condition_parts, _, _, music_seq, _ = \
            self.normalize_inputs(music_seq, leader_latents, follower_latents, translation_latent, contact_latent)
        target_seq = torch.cat(target_parts, dim=1)

        cond_music = self.music_trans(music_seq)
        condition_seq = torch.cat(condition_parts + [cond_music], dim=1)

        t, weights = self.schedule_sampler.sample(target_seq.shape[0], dist_util.dev())
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            target_seq,
            t,
            cmx=condition_seq,
        )

        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        return dotdict(loss=losses['loss'],
                       scalar_stats=dotdict(diff_mse=losses['diff_mse'],
                                            loss_total=losses['loss']))

    def prepare_inputs(self, batch):
        music_seq = batch.music
        pose_seql = batch.pos3dl.clone()
        pose_seqf = batch.pos3df.clone()
        contact_seq = batch.contact

        lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
        transll = pose_seql[:, :, :3].clone()
        transll = transll - transll[:, :1, :3].clone()

        pose_seqf[:, :-1, :3] = pose_seqf[:, 1:, :3] - pose_seqf[:, :-1, :3]
        pose_seqf[:, -1, :3] = pose_seqf[:, -2, :3]

        pose_seql[:, :, :3], pose_seqf[:, :, :3] = 0, 0

        leader_latents = [q[0].permute(0, 2, 1) for q in cfg.motoken_net.encode_latent(pose_seql)]
        follower_latents = [q[0].permute(0, 2, 1) for q in cfg.motoken_net.encode_latent(pose_seqf)]
        translation_latent = cfg.transl_net.encode_latent(lftransl)[0].permute(0, 2, 1)
        contact_latent = cfg.contact_net.encode_latent(contact_seq)[0].permute(0, 2, 1)

        return music_seq, transll, leader_latents, follower_latents, translation_latent, contact_latent, \
            pose_seql

    def normalize_inputs(self, music_seq, leader_latents, follower_latents, translation_latent, contact_latent):
        target_part_names = ['up', 'down', 'lhand', 'rhand', 'transl', 'contact']

        leader_latents = [q.clone() for q in leader_latents]
        target_parts = [q.clone() for q in follower_latents] + [
            translation_latent.clone(),
            contact_latent.clone().detach(),
        ]

        leader_latents = [self.normalizer.normalize(c, n) for c, n in zip(leader_latents, target_part_names[:4])]
        target_parts = [self.normalizer.normalize(i, n) for i, n in zip(target_parts, target_part_names)]
        music_seq = self.normalizer.normalize(music_seq, 'music')

        if self.training:
            music_seq_iter = None
            condition_parts = leader_latents
        else:
            condition_parts = [c[:, :self.block_size].clone() for c in leader_latents]
            target_parts = [i[:, :0].clone() for i in target_parts]
            music_seq_iter = music_seq[:, :self.ds_rate * self.block_size]

        return target_parts, condition_parts, target_part_names, music_seq_iter, music_seq, leader_latents

    def get_leader_pose_window(self, pose_seql, end, seq_len):
        pose_end = self.ds_rate * end
        pose_len = self.ds_rate * seq_len
        return pose_seql[:, :pose_end][:, -pose_len:].clone()

    def run_sampling_loop(self, generated_parts, condition_parts, music_seq, pose_seql_iter,
                          leader_latents, pose_seql, music_seq_iter):
        sample_fn = self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop
        total_len, seq_len = leader_latents[0].shape[1], self.block_size
        iters = math.ceil(total_len / self.block_size)

        for k in range(iters):
            cond_music = self.music_trans(music_seq_iter)
            condition_seq = torch.cat(condition_parts + [cond_music], dim=1)
            model_kwargs = dotdict(cmx=condition_seq, motoken_net=cfg.motoken_net,
                                   transl_net=cfg.transl_net, contact_net=cfg.contact_net,
                                   pose_seql=pose_seql_iter)

            init_image = None
            shape = (
                condition_parts[0].size(0),
                condition_parts[0].size(1) * len(generated_parts),
                condition_parts[0].size(2),
            )

            pred_latent = sample_fn(self.network, shape, clip_denoised=False,
                                    model_kwargs=model_kwargs, skip_timesteps=0,
                                    init_image=init_image, progress=True, noise=None,
                                    cond_fn=cond_fn, cond_fn_with_grad=True
                                    )

            num_parts = len(generated_parts)
            for i in range(num_parts):
                generated_parts[i] = torch.cat(
                    [generated_parts[i], pred_latent[:, seq_len * i: seq_len * (i + 1)]],
                    dim=1,
                )

            end = min((k + 2) * self.block_size, total_len)
            seq_len = min(self.block_size, total_len - (k + 1) * self.block_size)

            condition_parts = [q[:, :end][:, -seq_len:].clone() for q in leader_latents]
            music_seq_iter = music_seq[:, :self.ds_rate * end][:, -seq_len * self.ds_rate:]
            pose_seql_iter = self.get_leader_pose_window(pose_seql, end, seq_len)

        return generated_parts

    def postprocess_results(self, generated_parts, target_part_names, transll, batch, pose_seql):
        contact_idx = self.normalizer.denormalize(generated_parts[target_part_names.index('contact')], 'contact')
        contact_idx = cfg.contact_net.from_latent_to_fea(contact_idx)
        contact_matrix = cfg.contact_net.decode([contact_idx]).cpu().detach().numpy()

        for i, name in enumerate(target_part_names[:4]):
            generated_parts[i] = self.normalizer.denormalize(generated_parts[i], name)
        generated_parts[:4] = cfg.motoken_net.from_latent_to_fea(generated_parts[:4])
        generated_parts[4] = cfg.transl_net.from_latent_to_fea(self.normalizer.denormalize(generated_parts[4], 'transl'))

        zs = [[o] for o in generated_parts[:4]]
        transl_z = [generated_parts[4]]
        pose_sample, rotmat_sample, vel_sample = cfg.motoken_net.decode(zs)
        lf_transl = cfg.transl_net.decode(transl_z)
        pose_sample, pose_seql = get_glb_pos(pose_sample, transll, lf_transl, pose_seql)
        pose_sample, pose_seql = pose_sample.cpu().detach().numpy(), pose_seql.cpu().detach().numpy()

        rotmatf = get_closest_rotmat(rotmat_sample.cpu().detach().numpy().reshape([-1, 3, 3]))
        smpl_poses_f = rotmat2aa(rotmatf).reshape(-1, 55, 3)
        rotmatl = get_closest_rotmat(batch['rotmatl'].cpu().detach().numpy().reshape([-1, 3, 3]))
        smpl_poses_l = rotmat2aa(rotmatl).reshape(-1, 55, 3)
        translf = (transll[:, :lf_transl.size(1)] + lf_transl / 20.0).reshape([-1, 3]).cpu().detach().numpy()
        transll = transll.reshape([-1, 3]).cpu().detach().numpy()

        dictf = np.array([{'poses': smpl_poses_f, 'global_orient': smpl_poses_f[:, 0],
                           'transl': translf, 'meta': {'gender': 'female'}}])
        dictl = np.array([{'poses': smpl_poses_l, 'global_orient': smpl_poses_l[:, 0],
                           'transl': transll, 'meta': {'gender': 'male'}}])

        return dotdict(follower_motion=pose_sample, leader_motion=pose_seql,
                       contact_matrix=contact_matrix, dictf=dictf, dictl=dictl)

    def inference(self, batch, dataloader, if_categorial=False):
        music_seq, transll, leader_latents, follower_latents, translation_latent, contact_latent, \
            pose_seql = self.prepare_inputs(batch)

        generated_parts, condition_parts, target_part_names, music_seq_iter, music_seq, leader_latents = \
            self.normalize_inputs(music_seq, leader_latents, follower_latents, translation_latent, contact_latent)
        pose_seql_iter = self.get_leader_pose_window(pose_seql, self.block_size, self.block_size)

        generated_parts = self.run_sampling_loop(
            generated_parts, condition_parts, music_seq, pose_seql_iter, leader_latents, pose_seql, music_seq_iter
        )

        return self.postprocess_results(generated_parts, target_part_names, transll, batch, pose_seql)

