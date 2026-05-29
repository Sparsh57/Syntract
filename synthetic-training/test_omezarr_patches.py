"""Test scale-aware OME-Zarr patch extraction (optionally with model inference).

This script is for testing/evaluation, not training. It samples patches using
physical-space-aware logic and reports scale/coverage metadata.

Example:
    python test_omezarr_patches.py \
        --zarr_path /path/to/data.ome.zarr \
        --num_patches 8 \
        --patch_size 128 128 128 \
        --target_voxel_size_um 500 500 500 \
        --level_sampling closest \
        --save_dir ./omezarr_test_outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datamodules.omezarr import PhysicalScaleOMEZarrDataset


def _try_import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def _jsonable(obj: Any):
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _best_slice_indices(volume_zyx: np.ndarray) -> tuple[int, int, int]:
    """Pick informative z/y/x slice indices based on max summed intensity."""
    if volume_zyx.ndim != 3:
        return (0, 0, 0)
    z_scores = volume_zyx.sum(axis=(1, 2))
    y_scores = volume_zyx.sum(axis=(0, 2))
    x_scores = volume_zyx.sum(axis=(0, 1))
    z_idx = int(np.argmax(z_scores)) if z_scores.size else 0
    y_idx = int(np.argmax(y_scores)) if y_scores.size else 0
    x_idx = int(np.argmax(x_scores)) if x_scores.size else 0
    return z_idx, y_idx, x_idx


def _load_model(checkpoint_path: str, device: torch.device):
    from unet3d import FlexibleUNet3D

    model = FlexibleUNet3D(learning_rate=1e-4)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser("Test OME-Zarr scale-aware patch loading")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to .ome.zarr root")
    parser.add_argument("--num_patches", type=int, default=8, help="How many patches to sample")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128], metavar=("D", "H", "W"))
    parser.add_argument(
        "--target_voxel_size_um",
        nargs=3,
        type=float,
        default=[500.0, 500.0, 500.0],
        metavar=("Z_UM", "Y_UM", "X_UM"),
        help="Target voxel spacing in um (500 um = 0.5 mm)",
    )
    parser.add_argument(
        "--physical_patch_size_um",
        nargs=3,
        type=float,
        default=None,
        metavar=("Z_UM", "Y_UM", "X_UM"),
        help="Optional explicit physical FOV in um. If omitted, uses patch_size * target_voxel_size_um.",
    )
    parser.add_argument(
        "--level_sampling",
        type=str,
        default="closest",
        choices=["closest", "random", "weighted_random", "cycle"],
        help="Pyramid level selection strategy (ignored if --fixed_level is set)",
    )
    parser.add_argument(
        "--fixed_level",
        type=int,
        default=None,
        help="Force a specific pyramid level index (e.g., 0 for highest resolution)",
    )
    parser.add_argument("--channel_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None, help="Optional folder to save .npz + metadata")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint for patch-level inference (FlexibleUNet3D)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for predictions")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="syntract3d", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="omezarr_patch_test", help="W&B run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team (optional)")
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        default=True,
        help="Run W&B in offline mode (default: true)",
    )
    parser.add_argument(
        "--wandb_online",
        dest="wandb_offline",
        action="store_false",
        help="Run W&B in online mode",
    )

    args = parser.parse_args()

    wandb = None
    wandb_enabled = not bool(args.no_wandb)
    if wandb_enabled:
        wandb = _try_import_wandb()
        if wandb is None:
            print("W&B requested but not installed/importable. Continuing without W&B logging.")
            wandb_enabled = False
        else:
            if bool(args.wandb_offline):
                os.environ["WANDB_MODE"] = "offline"
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config={
                    "zarr_path": args.zarr_path,
                    "num_patches": int(args.num_patches),
                    "patch_size": list(args.patch_size),
                    "target_voxel_size_um": list(args.target_voxel_size_um),
                    "physical_patch_size_um": (
                        list(args.physical_patch_size_um) if args.physical_patch_size_um else None
                    ),
                    "level_sampling": args.level_sampling,
                    "channel_index": int(args.channel_index),
                    "seed": int(args.seed),
                    "model_checkpoint": args.model_checkpoint,
                    "threshold": float(args.threshold),
                },
            )
            if run is not None and getattr(run, "url", None):
                print(f"W&B run URL: {run.url}")

    selected_level_sampling: int | str
    if args.fixed_level is not None:
        selected_level_sampling = int(args.fixed_level)
    else:
        selected_level_sampling = args.level_sampling

    dataset = PhysicalScaleOMEZarrDataset(
        zarr_path=args.zarr_path,
        output_patch_size=tuple(args.patch_size),
        samples_per_epoch=int(args.num_patches),
        target_voxel_size_um=tuple(args.target_voxel_size_um),
        physical_patch_size_um=(tuple(args.physical_patch_size_um) if args.physical_patch_size_um else None),
        level_sampling=selected_level_sampling,
        channel_index=args.channel_index,
        allow_padding=True,
        seed=args.seed,
        return_metadata=True,
    )

    if args.fixed_level is not None:
        if args.fixed_level < 0 or args.fixed_level >= len(dataset.levels):
            raise ValueError(
                f"--fixed_level={args.fixed_level} is out of range. "
                f"Valid levels: 0..{len(dataset.levels)-1}"
            )

    print("Parsed OME-Zarr levels:")
    for row in dataset.describe_levels():
        print(
            f"  level={row['level']} path={row['path']} shape_zyx={row['shape_zyx']} "
            f"voxel_um_zyx={row['voxel_size_um_zyx']}"
        )
    if wandb_enabled and wandb is not None:
        wandb.log({"dataset/levels": _jsonable(dataset.describe_levels())})

    save_dir: Path | None = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_checkpoint:
        model = _load_model(args.model_checkpoint, device=device)
        print(f"Loaded model checkpoint: {args.model_checkpoint}")
        print(f"Inference device: {device}")

    level_counts: dict[int, int] = {}
    warning_count = 0
    pred_ratios = []
    unique_signatures = set()

    for idx in range(len(dataset)):
        image_t, meta = dataset[idx]
        image = image_t.squeeze(0).cpu().numpy()  # (D, H, W)
        level_idx = int(meta["level_index"])
        level_counts[level_idx] = level_counts.get(level_idx, 0) + 1
        signature = (
            int(meta["level_index"]),
            tuple(meta.get("origin_zyx", [])),
            tuple(meta.get("read_window_vox_zyx", [])),
        )
        unique_signatures.add(signature)

        msg = (
            f"[{idx + 1}/{len(dataset)}] level={meta['level_index']} "
            f"voxel_um={meta['source_voxel_size_um_zyx']} "
            f"requested_phys_um={meta['requested_physical_size_um_zyx']} "
            f"coverage={meta.get('source_coverage_fraction_zyx')}"
        )
        print(msg)
        if meta.get("warnings"):
            warning_count += 1
            print(f"  warnings: {meta['warnings']}")

        pred_bin = None
        pred_ratio = None
        if model is not None:
            with torch.no_grad():
                logits = model(image_t.unsqueeze(0).to(device))
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred_bin = (prob >= float(args.threshold)).astype(np.uint8)
            pred_ratio = float(pred_bin.mean())
            pred_ratios.append(pred_ratio)
            print(f"  pred foreground ratio: {100.0 * pred_ratio:.3f}%")

        if save_dir is not None:
            stem = f"patch_{idx:04d}"
            out_npz = save_dir / f"{stem}.npz"
            payload = {"image": image.astype(np.float32)}
            if pred_bin is not None:
                payload["pred"] = pred_bin.astype(np.uint8)
            np.savez_compressed(out_npz, **payload)

            out_meta = save_dir / f"{stem}.json"
            out_meta.write_text(json.dumps(_jsonable(meta), indent=2))

        if wandb_enabled and wandb is not None:
            z_idx, y_idx, x_idx = _best_slice_indices(image)
            nonzero_fraction = float((image > 0).mean())
            log_payload = {
                "patch/index": int(idx),
                "patch/level_index": level_idx,
                "patch/coverage_z": float(meta.get("source_coverage_fraction_zyx", [1.0, 1.0, 1.0])[0]),
                "patch/coverage_y": float(meta.get("source_coverage_fraction_zyx", [1.0, 1.0, 1.0])[1]),
                "patch/coverage_x": float(meta.get("source_coverage_fraction_zyx", [1.0, 1.0, 1.0])[2]),
                "patch/warning_count": int(len(meta.get("warnings", []))),
                "patch/nonzero_fraction": nonzero_fraction,
                "patch/image_axial": wandb.Image(image[z_idx], caption=f"Patch {idx} axial z={z_idx}"),
                "patch/image_coronal": wandb.Image(image[:, y_idx, :], caption=f"Patch {idx} coronal y={y_idx}"),
                "patch/image_sagittal": wandb.Image(image[:, :, x_idx], caption=f"Patch {idx} sagittal x={x_idx}"),
            }
            if pred_bin is not None:
                log_payload["patch/pred_foreground_ratio"] = float(pred_ratio)
                log_payload["patch/pred_axial"] = wandb.Image(
                    pred_bin[z_idx], caption=f"Patch {idx} pred axial z={z_idx}"
                )
                log_payload["patch/pred_coronal"] = wandb.Image(
                    pred_bin[:, y_idx, :], caption=f"Patch {idx} pred coronal y={y_idx}"
                )
                log_payload["patch/pred_sagittal"] = wandb.Image(
                    pred_bin[:, :, x_idx], caption=f"Patch {idx} pred sagittal x={x_idx}"
                )
            wandb.log(log_payload, step=int(idx))

    if wandb_enabled and wandb is not None:
        summary_payload = {
            "summary/num_patches": int(len(dataset)),
            "summary/warning_patches": int(warning_count),
            "summary/level_counts": _jsonable(level_counts),
            "summary/unique_patch_signatures": int(len(unique_signatures)),
        }
        if pred_ratios:
            summary_payload["summary/pred_foreground_ratio_mean"] = float(np.mean(pred_ratios))
            summary_payload["summary/pred_foreground_ratio_std"] = float(np.std(pred_ratios))
        wandb.log(summary_payload)
        wandb.finish()

    if len(unique_signatures) <= 1 and len(dataset) > 1:
        print(
            "NOTE: All sampled patches were identical by (level, origin, window). "
            "This usually means requested physical FOV exceeds source extent, forcing full-volume padded reads."
        )

    print("Done.")


if __name__ == "__main__":
    main()
