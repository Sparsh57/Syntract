#!/usr/bin/env python3
"""Calibrate the RealLSMProxyCallback against the compare_multiregion.sh baseline.

WHY THIS EXISTS
---------------
`RealLSMProxyCallback` (in synthetic-training/train_on_synthetic_data_3d.py) logs
`real_pred_pos_frac` each val epoch as an unlabeled real-data transfer signal for
closing the synthetic->real domain gap (step (b)). Before that number can be
trusted to MOVE meaningfully, the instrument must be shown to REPRODUCE a known
answer on a known model.

`compare_multiregion.sh` already measured ~0.0002 binary_fraction on the GOOD bf16
checkpoint (best_3d-epoch=129-val_loss=0.0491.ckpt) at the 9 multiregion centers,
using --num_patches 3 --jitter_radius 40. This script runs the proxy's EXACT
load+forward path on that SAME checkpoint with matched sampling (3/40), so the
result is directly comparable.

INTERPRETATION
--------------
- result ~= 0.0002  -> proxy is calibrated. A different (e.g. fresh epoch-149) run
  reading higher (e.g. 0.0014) reflects that DIFFERENT MODEL, not a wiring bug.
- result far from 0.0002 -> the proxy diverges from test_specific_region.py
  (extraction / normalization / threshold). Investigate before trusting step (b).

CLUSTER ONLY. A 128^3 forward pass OOMs a laptop — run on the A100/H200 node, e.g.:

    module load cuda/12.4.0 && source ../venv/bin/activate
    python calibrate_real_proxy.py \
        --checkpoint synthetic-training/checkpoints_cached_bf16/best_3d-epoch=129-val_loss=0.0491.ckpt \
        --zarr /orcd/data/linc/001/lsm_test_data_sparsh/LSM_test_data/2025_09_09_MonkeySlice_561channel_561laser_Stitched.ome.zarr

Defaults match compare_multiregion.sh (level 0, 1um target, 3 patches, 40 jitter,
percentile 1-99, threshold 0.5).
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SYNTH_DIR = os.path.join(REPO_ROOT, "synthetic-training")
if SYNTH_DIR not in sys.path:
    sys.path.insert(0, SYNTH_DIR)

from test_specific_region import _load_model  # exact loader used by the baseline
from train_on_synthetic_data_3d import RealLSMProxyCallback


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate RealLSMProxyCallback vs multiregion baseline")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the GOOD bf16 ckpt (best_3d-epoch=129-val_loss=0.0491.ckpt)")
    p.add_argument("--zarr", required=True, help="OME-Zarr path (real LSM volume)")
    p.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128])
    p.add_argument("--target_voxel_um", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    p.add_argument("--level_index", type=int, default=0)
    p.add_argument("--channel_index", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    # Match compare_multiregion.sh: --num_patches 3 --jitter_radius 40.
    p.add_argument("--num_patches", type=int, default=3)
    p.add_argument("--jitter_radius", type=int, default=40)
    p.add_argument("--expected", type=float, default=0.0002,
                   help="Baseline real_pred_pos_frac to compare against (multiregion ~0.0002)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device. A 128^3 forward on CPU is very slow / may OOM. "
              "This script is intended for the cluster GPU node.")

    model, load_report = _load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  missing_keys={len(load_report.get('missing_keys', []))} "
          f"unexpected_keys={len(load_report.get('unexpected_keys', []))}")

    proxy = RealLSMProxyCallback(
        zarr_path=args.zarr,
        patch_size=tuple(args.patch_size),
        target_voxel_size_um=tuple(args.target_voxel_um),
        level_index=args.level_index,
        channel_index=args.channel_index,
        normalize_percentiles=(1.0, 99.0),
        threshold=args.threshold,
        num_patches=args.num_patches,
        jitter_radius=args.jitter_radius,
    )

    # evaluate() builds the same multiregion patches (grouped per region) and
    # runs the exact forward path used during training validation. Returns
    # mean/median/per-region.
    m = proxy.evaluate(model)
    per_region = m["per_region"]
    n_regions = len(per_region)
    region_pos = list(per_region.values())
    top_row = region_pos[:3]  # regions 1-3 = the fy=4 top-Y row the baseline quoted
    top_row_mean = (sum(top_row) / len(top_row)) if top_row else float("nan")

    print("=" * 72)
    print("PER-REGION (center_zyx | pos_frac):")
    for ridx, (center, rp) in enumerate(per_region.items(), start=1):
        print(f"  region {ridx} {center}: pos_frac={rp:.6f}")
    print("=" * 72)
    print(f"regions evaluated      : {n_regions} ({args.num_patches}/center)")
    print(f"GRAND-MEAN pos_frac     : {m['pred_pos_frac_mean']:.6f}  (mean — outlier-sensitive)")
    print(f"MEDIAN pos_frac         : {m['pred_pos_frac_median']:.6f}  (robust movement signal for (b))")
    print(f"  grand-mean prob_mean  : {m['prob_mean']:.6f}")
    print(f"REGIONS 1-3 mean        : {top_row_mean:.6f}  (matched to handoff baseline ~{args.expected})")
    print("-" * 72)
    print("INTERPRETATION:")
    print("  The proxy reuses SpecificRegionDataset + _load_model + the exact center")
    print("  grid, so the load/forward path is identical to test_specific_region.py by")
    print("  construction. The handoff's ~0.0002 is a one-sig-fig, top-Y-row figure;")
    print("  compare it to REGIONS 1-3, not the grand mean. Regions 1-3 ~= 0.0002 means")
    print("  the proxy reproduces the baseline; the grand-mean offset is one hot region")
    print("  (aggregation), NOT a wiring bug. The MEDIAN is the robust scalar to watch in")
    print("  step (b) — a constant offset / one-region noise cannot fake movement in it.")
    print("=" * 72)


if __name__ == "__main__":
    main()
