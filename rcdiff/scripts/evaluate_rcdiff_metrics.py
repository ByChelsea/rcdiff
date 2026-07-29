import argparse
import json
import os
import sys
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def format_value(value):
    return "-" if value is None else f"{value:.4f}"


def stage(message, pred_root=None):
    prefix = f"[metrics:{pred_root}]" if pred_root else "[metrics]"
    print(f"{prefix} Computing {message}...", flush=True)


def print_metrics(metrics, pred_root):
    print(f"RCDiff evaluation metrics: {pred_root}")
    for key in ["FID_k", "FID_g", "Div_k", "Div_g", "FID_cd", "Div_cd", "CF", "BED", "BAS"]:
        if key in metrics:
            print(f"{key:>7}: {format_value(metrics[key])}")
    print(f"samples: {metrics['num_samples']}")


def gt_cache_dir(cache_root, gt_root):
    key = hashlib.sha1(os.path.abspath(gt_root).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_root, key)


def load_or_compute_gt_refs(args):
    from rcdiff.utils.motion_metrics.metrics import follower_files, reference_features
    import numpy as np

    gt_files = follower_files(args.gt_root)
    if not gt_files:
        raise RuntimeError(f"No ground-truth *_00.npy files found in {args.gt_root}")

    cache_dir = gt_cache_dir(args.cache_root, args.gt_root)
    meta_path = os.path.join(cache_dir, "meta.json")
    refs = {
        "gt_k": os.path.join(cache_dir, "gt_k.npy"),
        "gt_g": os.path.join(cache_dir, "gt_g.npy"),
        "gt_cd": os.path.join(cache_dir, "gt_cd.npy"),
    }

    if not args.refresh_gt_cache and os.path.exists(meta_path) and all(os.path.exists(p) for p in refs.values()):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("gt_root") == os.path.abspath(args.gt_root) and meta.get("gt_files") == gt_files:
            stage(f"loading GT reference features from {cache_dir}")
            return {key: np.load(path) for key, path in refs.items()}

    stage("GT reference features")
    gt_refs = reference_features(args.gt_root, gt_files)
    os.makedirs(cache_dir, exist_ok=True)
    for key, value in gt_refs.items():
        np.save(refs[key], value)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dict(gt_root=os.path.abspath(args.gt_root), gt_files=gt_files), f, indent=2)
    stage(f"saved GT reference features to {cache_dir}")
    return gt_refs


def evaluate_one(pred_root, args, gt_refs):
    from rcdiff.utils.motion_metrics.metrics import (
        calc_bas,
        metric_file_sets,
        duet_features,
        interactive_metrics_from_features,
        motion_quality_metrics_from_features,
        solo_features,
    )

    stage("file lists", pred_root)
    pred_files, _ = metric_file_sets(pred_root, args.gt_root)
    metrics = {"num_samples": len(pred_files)}

    stage("solo metrics (FID_k, FID_g, Div_k, Div_g)", pred_root)
    pred_k, pred_g = solo_features(pred_root, pred_files)
    metrics.update(motion_quality_metrics_from_features(pred_k, pred_g, gt_refs["gt_k"], gt_refs["gt_g"]))

    stage("interactive metrics (FID_cd, Div_cd, BED)", pred_root)
    pred_cd = duet_features(pred_root, pred_files)
    metrics.update(interactive_metrics_from_features(pred_root, pred_files, pred_cd, gt_refs["gt_cd"]))

    stage("rhythmic metric (BAS)", pred_root)
    metrics["BAS"] = calc_bas(pred_root, args.music_root, pred_files)

    if args.smplx_model_root and not args.skip_cf:
        from rcdiff.utils.motion_metrics.contact.collision import compute_contact_frequency

        stage("contact frequency (CF)", pred_root)
        cf_root = args.cf_root or os.path.join(pred_root, "cf")
        if not os.path.isdir(cf_root):
            raise RuntimeError(f"CF requires predicted SMPL-X dicts, but this folder does not exist: {cf_root}")
        metrics["CF"] = compute_contact_frequency(
            cf_root,
            args.smplx_model_root,
            max_collisions=args.max_collisions,
            device=args.cf_device,
        )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-root",
        required=True,
        nargs="+",
        help="One or more folders containing predicted *_00.npy and *_01.npy joints",
    )
    parser.add_argument(
        "--gt-root",
        required=True,
        help="Ground-truth motion reference root for FID/Div. The official setup uses data/motion/pos3d/all.",
    )
    parser.add_argument(
        "--music-root",
        required=True,
        help="Music feature root for BAS. The official solo metric setup uses data/music/feature/all.",
    )
    parser.add_argument("--cf-root", default=None, help="Folder containing predicted SMPL-X *_00.npy and *_01.npy dicts")
    parser.add_argument("--smplx-model-root", default=None, help="SMPL-X model root for official mesh-collision CF")
    parser.add_argument("--skip-cf", action="store_true", help="Skip CF even when --smplx-model-root is provided.")
    parser.add_argument("--cf-device", default="cuda")
    parser.add_argument("--max-collisions", type=int, default=20)
    parser.add_argument("--cache-root", default="data/metric_cache", help="Folder for cached GT reference features.")
    parser.add_argument("--refresh-gt-cache", action="store_true", help="Recompute GT reference features even if cache exists.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    # Official setup:
    # - solo metrics.py uses motion/pos3d/all and music/feature/all.
    # - duet metrics_duet.py uses motion/pos3d/all; its music/feature/test variable is not used.
    if args.cf_root and len(args.pred_root) > 1:
        raise RuntimeError("--cf-root is only supported when evaluating one --pred-root")
    gt_refs = load_or_compute_gt_refs(args)

    all_metrics = {}
    for pred_root in args.pred_root:
        metrics = evaluate_one(pred_root, args, gt_refs)
        all_metrics[pred_root] = metrics
        print_metrics(metrics, pred_root)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_metrics if len(all_metrics) > 1 else next(iter(all_metrics.values())), f, indent=2)


if __name__ == "__main__":
    main()
