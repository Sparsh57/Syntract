# Sliding-window 3D inference — design

**Date:** 2026-06-24
**Branch:** `3_dimension`
**Status:** approved design, pre-implementation

## Problem

The model is trained on 128³ patches. Real LSM volumes (and any larger extracted
sub-volume) are far bigger than 128³. Running inference on a single 128³ crop misses
most of the tissue. The standard solution is **sliding-window inference**: tile 128³
windows over the full volume with a stride, run each window through the model, and
blend overlapping predictions into a single full-resolution output.

The existing `predict_omezarr_thinslab_3d.py` does this but couples it tightly to
OME-Zarr reading and a thin-slab Z-padding workaround (real Z depth < 128). The new
script must work on any pre-extracted 3D array (numpy or NIfTI), which removes the
zarr dependency and makes it runnable locally without the cluster data.

## Goal

A standalone `sliding_window_inference.py` that:
1. Accepts any 3D volume (`.npy` or `.nii.gz`).
2. Tiles it with 128³ windows at a configurable stride.
3. Blends overlapping predictions with Gaussian weights.
4. Saves `_probability.npy` + `_binary.npy` (and optionally NIfTI).
5. Includes a `--synthetic_size` mode that tiles `patch_0001_3d.nii.gz` into a
   bigger volume for local end-to-end testing without real data.

## Design decisions

### Patch size
128³ — fixed, must match the checkpoint. Exposed as `--patch_size` in case a
future checkpoint uses a different size, but defaulted to 128.

### Stride
Default 64 (50 % overlap in every axis). Configurable via `--stride`. Smaller
stride = more overlap = smoother blending but more GPU time. At stride 64 each
interior voxel is covered by 2³ = 8 patches.

### Normalization — global, once
Normalize the full input volume with 1–99 percentile → [0, 1] **before** sliding.
Per-patch normalization (the training path) would cause brightness seams between
tiles because adjacent 128³ crops may have different percentile ranges. Global
normalization avoids seams and is closer to what OME-Zarr inference does (the
zarr reader normalizes a fixed physical window, not an arbitrary crop).

### Overlap blending — Gaussian-weighted average
Each 128³ output patch is multiplied by a 3D Gaussian weight map (σ = patch_size/4,
centered at the patch center) before accumulation. A matching weight map is
accumulated in a denominator volume; the final probability = numerator / denominator.

**Why Gaussian:** border voxels of a patch have incomplete receptive-field context
(the U-Net sees zero-padded or reflected content beyond the crop boundary), so their
predictions are less reliable. Gaussian weighting down-weights borders and up-weights
centers automatically, reducing tile-edge artifacts. This is the standard approach
in nnU-Net and the paper referenced (arXiv 2508.12942).

### Boundary handling — reflect-pad
The input volume is reflect-padded by `patch_size // 2` on every face before
sliding. This ensures every 128³ window is fully inside a valid region (no partial
windows at borders). After inference the output is unpadded back to the original
shape.

### Threshold
Default 0.5 for binary mask. Configurable via `--threshold`.

### Synthetic test mode (`--synthetic_size Z Y X`)
Tiles `Model_prediction/patch_0001_3d.nii.gz` (a 128³ synthetic patch) by repeating
it to fill the requested size (tile then crop to exact size, so non-multiples of 128
work), then runs sliding-window inference on that volume.
Useful for local end-to-end testing without real data.

### Output
- `<output_prefix>_probability.npy` — float32 [0, 1] probability map, original shape
- `<output_prefix>_binary.npy` — uint8 binary mask (probability ≥ threshold)
- `<output_prefix>_probability.nii.gz` — optional NIfTI (if `--save_nifti` and input
  was NIfTI, preserves affine; otherwise identity affine at 0.001mm)
- `<output_prefix>_summary.json` — shape, stride, threshold, pos_frac, runtime

### Device
`--device auto` (tries cuda → mps → cpu in order). Inference batch size 1 per window.

## Interface

```bash
# Run on a real NIfTI volume
python sliding_window_inference.py \
    --input my_volume.nii.gz \
    --checkpoint Model_prediction/best_3d-epoch=129-val_loss=0.0491.ckpt \
    --output_prefix results/my_volume \
    --stride 64 \
    --threshold 0.5 \
    --save_nifti

# Local end-to-end test with synthetic tiling (no zarr needed)
python sliding_window_inference.py \
    --synthetic_size 384 384 384 \
    --checkpoint Model_prediction/best_3d-epoch=129-val_loss=0.0491.ckpt \
    --output_prefix results/synthetic_test \
    --stride 64
```

## Out of scope

- Multi-scale or test-time augmentation.
- OME-Zarr reading (that is `predict_omezarr_thinslab_3d.py`'s job).
- Any change to the model architecture or training.
