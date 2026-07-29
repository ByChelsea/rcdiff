import os

import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

from .features.kinetic import extract_kinetic_features
from .features.manual_new import extract_manual_features


SMPLX_FEAT_POINT = [0, 7, 8, 10, 11, 15, 16, 17, 20, 21]


def normalize(gt_features, pred_features):
    mean = gt_features.mean(axis=0)
    std = gt_features.std(axis=0)
    return (gt_features - mean) / (std + 1e-10), (pred_features - mean) / (std + 1e-10)


def calc_fid(pred_features, gt_features):
    mu_pred = np.mean(pred_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    mu_gt = np.mean(gt_features, axis=0)
    sigma_gt = np.cov(gt_features, rowvar=False)

    diff = mu_pred - mu_gt
    eps = 1e-5
    covmean, _ = linalg.sqrtm(sigma_pred.dot(sigma_gt), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_pred.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_pred + offset).dot(sigma_gt + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma_pred) + np.trace(sigma_gt) - 2 * np.trace(covmean))


def calculate_avg_distance(feature_list):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    return float(dist / ((n * n - n) / 2))


def load_joints(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    return np.asarray(data).reshape(-1, 55, 3)


def follower_files(root):
    return sorted(
        name for name in os.listdir(root)
        if name.endswith("_00.npy") and os.path.isfile(os.path.join(root, name))
    )


def metric_file_sets(pred_root, gt_root):
    pred_files = follower_files(pred_root)
    gt_files = follower_files(gt_root)
    if not pred_files:
        raise RuntimeError(f"No predicted *_00.npy files found in {pred_root}")
    if not gt_files:
        raise RuntimeError(f"No ground-truth *_00.npy files found in {gt_root}")
    return pred_files, gt_files


def to_smpl24_relative(joints55):
    joints24 = np.zeros((joints55.shape[0], 24, 3), dtype=joints55.dtype)
    joints24[:, :22] = joints55[:, :22]
    joints24[:, 22] = (joints55[:, 25] + joints55[:, 28] + joints55[:, 31] + joints55[:, 34] + joints55[:, 37]) / 5.0
    joints24[:, 23] = (joints55[:, 40] + joints55[:, 43] + joints55[:, 46] + joints55[:, 49] + joints55[:, 52]) / 5.0
    joints24 = joints24.reshape(-1, 72)
    root = joints24[:1, :3]
    return (joints24 - np.tile(root, (1, 24))).reshape(-1, 24, 3)


def solo_features(root, files):
    kinetic = []
    geometric = []
    for name in files:
        joints24 = to_smpl24_relative(load_joints(os.path.join(root, name)))
        kinetic.append(extract_kinetic_features(joints24))
        geometric.append(extract_manual_features(joints24))
    return np.stack(kinetic), np.stack(geometric)


def duet_feature(follower, leader):
    length = min(follower.shape[0], leader.shape[0])
    follower = follower[:length]
    leader = leader[:length]
    feature = np.sqrt(np.sum(
        (follower[:, SMPLX_FEAT_POINT][:, :, None, :] - leader[:, SMPLX_FEAT_POINT][:, None, :, :]) ** 2,
        axis=-1,
    )).reshape(length, -1)
    return feature.mean(axis=0)


def duet_features(root, files):
    features = []
    for follower_name in files:
        leader_name = follower_name.replace("_00.npy", "_01.npy")
        leader_path = os.path.join(root, leader_name)
        if not os.path.exists(leader_path):
            raise RuntimeError(f"Missing leader file for duet metrics: {leader_path}")
        follower = load_joints(os.path.join(root, follower_name))
        leader = load_joints(leader_path)
        features.append(duet_feature(follower, leader))
    return np.stack(features)


def calc_db(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 55, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    return argrelextrema(kinetic_vel, np.less), len(kinetic_vel)


def beat_alignment(reference_beats, motion_beats):
    score = 0.0
    for beat in reference_beats:
        score += np.exp(-np.min((motion_beats - beat) ** 2) / 2 / 9)
    return float(score / len(reference_beats))


def music_name_from_motion_name(name):
    parts = name.split("_")
    return f"{parts[0]}_{parts[1]}_{parts[2]}.npy"


def calc_bas(root, music_root, files):
    scores = []
    for name in files:
        music_path = os.path.join(music_root, music_name_from_motion_name(name))
        if not os.path.exists(music_path):
            raise RuntimeError(f"Missing music feature for BAS: {music_path}")
        joints = load_joints(os.path.join(root, name))
        music = np.load(music_path)
        length = min(len(joints), len(music))
        dance_beats, dance_length = calc_db(joints[:length])
        music_beats = np.where(music[:dance_length, 53].astype(bool))[0]
        scores.append(beat_alignment(music_beats, dance_beats[0]))
    return float(np.mean(scores))


def calc_bed(root, files):
    scores = []
    for follower_name in files:
        leader_name = follower_name.replace("_00.npy", "_01.npy")
        leader_path = os.path.join(root, leader_name)
        if not os.path.exists(leader_path):
            raise RuntimeError(f"Missing leader file for BED: {leader_path}")
        follower_beats, _ = calc_db(load_joints(os.path.join(root, follower_name)))
        leader_beats, _ = calc_db(load_joints(leader_path))
        scores.append(beat_alignment(leader_beats[0], follower_beats[0]))
    return float(np.mean(scores))


def motion_quality_metrics(pred_root, gt_root, pred_files, gt_files):
    pred_k, pred_g = solo_features(pred_root, pred_files)
    gt_k, gt_g = solo_features(gt_root, gt_files)
    return motion_quality_metrics_from_features(pred_k, pred_g, gt_k, gt_g)


def motion_quality_metrics_from_features(pred_k, pred_g, gt_k, gt_g):
    gt_k, pred_k = normalize(gt_k, pred_k)
    gt_g, pred_g = normalize(gt_g, pred_g)
    return {
        "FID_k": calc_fid(pred_k, gt_k),
        "FID_g": calc_fid(pred_g, gt_g),
        "Div_k": calculate_avg_distance(pred_k),
        "Div_g": calculate_avg_distance(pred_g),
    }


def interactive_metrics(pred_root, gt_root, pred_files, gt_files):
    pred_cd = duet_features(pred_root, pred_files)
    gt_cd = duet_features(gt_root, gt_files)
    return interactive_metrics_from_features(pred_root, pred_files, pred_cd, gt_cd)


def interactive_metrics_from_features(pred_root, pred_files, pred_cd, gt_cd):
    gt_cd, pred_cd = normalize(gt_cd, pred_cd)
    return {
        "FID_cd": calc_fid(pred_cd, gt_cd),
        "Div_cd": calculate_avg_distance(pred_cd),
        "BED": calc_bed(pred_root, pred_files),
    }


def reference_features(gt_root, gt_files):
    gt_k, gt_g = solo_features(gt_root, gt_files)
    gt_cd = duet_features(gt_root, gt_files)
    return dict(gt_k=gt_k, gt_g=gt_g, gt_cd=gt_cd)
