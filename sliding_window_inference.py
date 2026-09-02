#!/usr/bin/env python3
"""Sliding-window 3D inference over a large volume.

Tiles the model's 128³ (configurable) patch over an input volume with a
configurable stride, blends overlapping predictions with Gaussian weights,
and saves the assembled probability map and binary mask.

Input modes (pick exactly one):
  --input FILE           Pre-extracted numpy (.npy) or NIfTI (.nii.gz) volume
  --zarr PATH            Extract a sub-region from an OME-Zarr on disk
  --synthetic_size Z Y X Tile Model_prediction/patch_0001_3d.nii.gz for local testing

Examples:
  # Local end-to-end test (no zarr needed)
  python sliding_window_inference.py \\
      --synthetic_size 384 384 384 \\
      --checkpoint Model_prediction/best_3d-epoch=129-val_loss=0.0491.ckpt \\
      --output_prefix results/synthetic_test

  # From a pre-extracted NIfTI
  python sliding_window_inference.py \\
      --input my_region.nii.gz \\
      --checkpoint Model_prediction/best_3d-epoch=129-val_loss=0.0491.ckpt \\
      --output_prefix results/my_region --save_nifti

  # From OME-Zarr on the cluster
  python sliding_window_inference.py \\
      --zarr /orcd/data/.../data.ome.zarr \\
      --region_center_zyx 19 12000 20000 \\
      --region_size_zyx 256 512 512 \\
      --checkpoint Model_prediction/best_3d-epoch=129-val_loss=0.0491.ckpt \\
      --output_prefix results/area1 --save_nifti
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTH_DIR = os.path.join(SCRIPT_DIR, "synthetic-training")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SYNTH_DIR not in sys.path:
    sys.path.insert(0, SYNTH_DIR)

DEFAULT_REF_PATCH = os.path.join(SCRIPT_DIR, "Model_prediction", "patch_0001_3d.nii.gz")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device, pos_weight: float = 1.0):
    try:
        from synthetic_training.unet3d import FlexibleUNet3D
    except ImportError:
        from unet3d import FlexibleUNet3D  # type: ignore

    ckpt = torch.load(checkpoint_path, map_location=device)
    hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    allowed = [
        "batch_size", "learning_rate", "weight_decay", "warmup_epochs",
        "min_features", "max_features", "num_stages", "loss",
        "freeze_encoder", "pos_weight", "in_channels",
    ]
    kwargs = {k: hparams[k] for k in allowed if k in hparams}
    kwargs.setdefault("learning_rate", 1e-4)
    kwargs.setdefault("pos_weight", float(pos_weight))

    model = FlexibleUNet3D(**kwargs)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"Model loaded — hparams={kwargs}  missing={len(missing)}  unexpected={len(unexpected)}")
    return model


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_global(vol: np.ndarray) -> np.ndarray:
    """1-99 percentile → [0, 1] over the full volume. Matches training (datasets.py)."""
    v_lo, v_hi = np.percentile(vol, [1.0, 99.0])
    if v_hi > v_lo:
        return np.clip((vol - v_lo) / (v_hi - v_lo), 0.0, 1.0).astype(np.float32)
    return np.zeros_like(vol, dtype=np.float32)


# ---------------------------------------------------------------------------
# Gaussian weight map
# ---------------------------------------------------------------------------

def _gaussian_weight_map(patch_size: tuple[int, int, int]) -> np.ndarray:
    """3D Gaussian weight map (D, H, W), σ = 0.5 in normalised [-1, 1] space.

    Values taper from 1.0 at the center to ~0.13 at the patch border.
    Down-weights predictions near the edge where receptive-field context is cut.
    """
    sigma = 0.5
    weight = np.ones(patch_size, dtype=np.float32)
    for axis, size in enumerate(patch_size):
        g = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        w = np.exp(-0.5 * (g / sigma) ** 2)
        shape = [1, 1, 1]
        shape[axis] = size
        weight *= w.reshape(shape)
    return weight


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def _window_starts(dim_size: int, patch_size: int, stride: int) -> list[int]:
    """Start positions that cover [0, dim_size) fully at the given stride."""
    starts = list(range(0, max(dim_size - patch_size, 0) + 1, stride))
    last = dim_size - patch_size
    if last >= 0 and (not starts or starts[-1] < last):
        starts.append(last)
    return starts


def sliding_window_inference(
    volume_norm: np.ndarray,
    model: torch.nn.Module,
    patch_size: tuple[int, int, int],
    stride: int,
    device: torch.device,
) -> np.ndarray:
    """Run sliding-window inference and return a probability map (same shape as input).

    Args:
        volume_norm: float32 (D, H, W) already normalised to [0, 1]
        patch_size:  (pD, pH, pW) must match the checkpoint's expected input
        stride:      stride in voxels per axis
    Returns:
        prob: float32 (D, H, W) blended probability map
    """
    D, H, W = volume_norm.shape
    pD, pH, pW = patch_size
    pad_d, pad_h, pad_w = pD // 2, pH // 2, pW // 2

    # Reflect-pad so every window is fully in-bounds (border predictions get
    # natural context from the reflected data rather than zeros).
    vol_pad = np.pad(
        volume_norm,
        [(pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)],
        mode="reflect",
    )
    pD_sz, pH_sz, pW_sz = vol_pad.shape

    weight_map = _gaussian_weight_map(patch_size)  # (pD, pH, pW)

    # float32 (not float64) — plenty of precision for averaging [0,1] probs, and
    # halves memory. For big regions the padded accumulators dominate RAM.
    acc_num = np.zeros(vol_pad.shape, dtype=np.float32)
    acc_den = np.zeros(vol_pad.shape, dtype=np.float32)

    z_starts = _window_starts(pD_sz, pD, stride)
    y_starts = _window_starts(pH_sz, pH, stride)
    x_starts = _window_starts(pW_sz, pW, stride)
    total = len(z_starts) * len(y_starts) * len(x_starts)

    print(f"Padded volume: {vol_pad.shape}")
    print(f"Windows: Z×Y×X = {len(z_starts)}×{len(y_starts)}×{len(x_starts)} = {total} patches")

    done = 0
    log_every = max(1, total // 20)
    t0 = time.time()

    with torch.no_grad():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    patch_np = vol_pad[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW]
                    t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(device)
                    prob_patch = torch.sigmoid(model(t)).squeeze().cpu().numpy()
                    w_prob = (prob_patch * weight_map).astype(np.float32)
                    acc_num[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW] += w_prob
                    acc_den[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW] += weight_map
                    done += 1
                    if done % log_every == 0 or done == total:
                        elapsed = time.time() - t0
                        eta = elapsed / done * (total - done)
                        print(f"  [{done:>5}/{total}]  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    # Free the padded input before the divide; do the divide in place so we never
    # hold a second full-size float array at once.
    del vol_pad
    np.maximum(acc_den, 1e-8, out=acc_den)
    acc_num /= acc_den
    del acc_den
    prob_pad = acc_num  # float32, blended probability (padded)
    # Remove padding to recover original shape
    return np.ascontiguousarray(prob_pad[pad_d:pad_d + D, pad_h:pad_h + H, pad_w:pad_w + W])


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _read_zarr_region(
    zarr_path: str,
    center_zyx: tuple[int, int, int],
    size_zyx: tuple[int, int, int],
    level: int = 0,
    channel_index: int = 0,
) -> np.ndarray:
    """Extract a sub-volume cube from an OME-Zarr at the given level."""
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr not installed: pip install zarr ome-zarr")

    store = zarr.open(zarr_path, mode="r")
    arr = None
    for key in (str(level), level):
        try:
            arr = store[key]
            break
        except (KeyError, TypeError):
            pass
    if arr is None:
        arr = store  # root is the array

    shape = arr.shape
    cz, cy, cx = center_zyx
    dz, dy, dx = size_zyx

    z0 = max(0, cz - dz // 2); z1 = min(shape[-3], z0 + dz)
    y0 = max(0, cy - dy // 2); y1 = min(shape[-2], y0 + dy)
    x0 = max(0, cx - dx // 2); x1 = min(shape[-1], x0 + dx)
    print(f"Zarr shape={shape}  reading [{z0}:{z1}, {y0}:{y1}, {x0}:{x1}]")

    ndim = arr.ndim
    if ndim == 3:
        raw = np.asarray(arr[z0:z1, y0:y1, x0:x1])
    elif ndim == 4 and shape[0] <= 16:  # (C, Z, Y, X)
        raw = np.asarray(arr[channel_index, z0:z1, y0:y1, x0:x1])
    elif ndim == 4:  # (Z, Y, X, C)
        raw = np.asarray(arr[z0:z1, y0:y1, x0:x1, channel_index])
    else:
        raise ValueError(f"Unexpected zarr shape {shape}; adjust _read_zarr_region manually")

    return raw.astype(np.float32)


def _tile_synthetic(ref_path: str, target_size_zyx: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Tile the reference 128³ synthetic patch to fill target_size_zyx (then crop)."""
    import nibabel as nib
    img = nib.load(ref_path)
    ref = img.get_fdata().astype(np.float32)
    rD, rH, rW = ref.shape
    tD, tH, tW = target_size_zyx

    nD = int(np.ceil(tD / rD))
    nH = int(np.ceil(tH / rH))
    nW = int(np.ceil(tW / rW))

    tiled = np.tile(ref, (nD, nH, nW))[:tD, :tH, :tW]
    print(f"Synthetic tiling: ref {ref.shape} → tiled {tiled.shape}")
    return tiled, img.affine


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sliding-window 3D inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", metavar="FILE",
                     help="Pre-extracted volume (.npy or .nii.gz)")
    inp.add_argument("--zarr", metavar="PATH",
                     help="OME-Zarr store (also set --region_center_zyx)")
    inp.add_argument("--synthetic_size", nargs=3, type=int, metavar=("Z", "Y", "X"),
                     help="Tile patch_0001_3d.nii.gz to this size for local testing")

    ap.add_argument("--region_center_zyx", nargs=3, type=int, metavar=("Z", "Y", "X"),
                    help="Center voxel (level-0 zarr space) for --zarr mode")
    ap.add_argument("--region_size_zyx", nargs=3, type=int, default=[256, 512, 512],
                    metavar=("Z", "Y", "X"),
                    help="Sub-volume size to extract from zarr")
    ap.add_argument("--zarr_level", type=int, default=0)
    ap.add_argument("--channel_index", type=int, default=0)

    ap.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    ap.add_argument("--pos_weight", type=float, default=1.0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))

    ap.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128],
                    metavar=("Z", "Y", "X"))
    ap.add_argument("--stride", type=int, default=64,
                    help="Stride in voxels per axis (64 = 50%% overlap)")
    ap.add_argument("--threshold", type=float, default=0.5)

    ap.add_argument("--output_prefix", required=True,
                    help="Outputs: <prefix>_probability.npy, _binary.npy, _summary.json")
    ap.add_argument("--save_nifti", action="store_true",
                    help="Also save _probability.nii.gz and _binary.nii.gz")
    ap.add_argument("--voxel_size_mm", type=float, default=0.001,
                    help="Voxel size for NIfTI output header (mm)")
    ap.add_argument("--ref_patch", default=DEFAULT_REF_PATCH,
                    help="Reference patch for --synthetic_size mode")

    args = ap.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model = _load_model(args.checkpoint, device, pos_weight=args.pos_weight)

    # Read volume
    affine = None
    if args.input:
        p = args.input
        if p.endswith(".npy"):
            volume = np.load(p).astype(np.float32)
        else:
            import nibabel as nib
            img = nib.load(p)
            volume = img.get_fdata().astype(np.float32)
            affine = img.affine
        source = p
    elif args.zarr:
        if args.region_center_zyx is None:
            ap.error("--zarr requires --region_center_zyx")
        volume = _read_zarr_region(
            args.zarr,
            tuple(args.region_center_zyx),
            tuple(args.region_size_zyx),
            level=args.zarr_level,
            channel_index=args.channel_index,
        )
        source = f"{args.zarr}  center={args.region_center_zyx}"
    else:
        volume, affine = _tile_synthetic(args.ref_patch, tuple(args.synthetic_size))
        source = f"synthetic tile ({args.ref_patch})"

    print(f"Source: {source}")
    print(f"Volume: shape={volume.shape}  dtype={volume.dtype}  "
          f"min={volume.min():.2f}  p1={np.percentile(volume,1):.2f}  "
          f"p99={np.percentile(volume,99):.2f}  max={volume.max():.2f}")

    volume_norm = normalize_global(volume)
    del volume            # raw copy no longer needed; free before padding/accumulators
    import gc; gc.collect()
    print(f"Normalised: min={volume_norm.min():.3f}  max={volume_norm.max():.3f}")

    patch_size = tuple(args.patch_size)
    t0 = time.time()
    prob = sliding_window_inference(volume_norm, model, patch_size, args.stride, device)
    binary = (prob >= args.threshold).astype(np.uint8)
    elapsed = time.time() - t0

    pos_frac = float(binary.mean())
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"pos_frac={pos_frac:.4f}  prob_median={np.median(prob):.4f}  "
          f"prob_p99={np.percentile(prob,99):.4f}  prob_max={prob.max():.4f}")

    # Save
    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.save(f"{args.output_prefix}_probability.npy", prob)
    np.save(f"{args.output_prefix}_binary.npy", binary)
    np.save(f"{args.output_prefix}_image.npy", volume_norm)   # normalised input — use as background
    print(f"Saved: {args.output_prefix}_probability.npy")
    print(f"Saved: {args.output_prefix}_binary.npy")
    print(f"Saved: {args.output_prefix}_image.npy")

    if args.save_nifti:
        import nibabel as nib
        vs = args.voxel_size_mm
        out_affine = affine if affine is not None else np.diag([vs, vs, vs, 1.0])
        nib.save(nib.Nifti1Image(volume_norm, out_affine),
                 f"{args.output_prefix}_image.nii.gz")
        nib.save(nib.Nifti1Image(prob, out_affine),
                 f"{args.output_prefix}_probability.nii.gz")
        nib.save(nib.Nifti1Image(binary.astype(np.float32), out_affine),
                 f"{args.output_prefix}_binary.nii.gz")
        print(f"Saved: {args.output_prefix}_image.nii.gz")
        print(f"Saved: {args.output_prefix}_probability.nii.gz")
        print(f"Saved: {args.output_prefix}_binary.nii.gz")

    summary = {
        "source": source,
        "volume_shape": list(volume_norm.shape),
        "patch_size": list(patch_size),
        "stride": args.stride,
        "threshold": args.threshold,
        "pos_frac": pos_frac,
        "prob_median": float(np.median(prob)),
        "prob_p99": float(np.percentile(prob, 99)),
        "elapsed_s": round(elapsed, 1),
        "device": str(device),
        "checkpoint": args.checkpoint,
    }
    with open(f"{args.output_prefix}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {args.output_prefix}_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
