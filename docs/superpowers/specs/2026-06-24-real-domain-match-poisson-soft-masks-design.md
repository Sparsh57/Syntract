# Design — Match synthetic training data to real LSM (Poisson shot noise + soft thin masks)

**Date:** 2026-06-24
**Status:** Approved (design); pending spec review
**Continues:** [texture-gap-runlog.md](../../superpowers/plans/texture-gap-runlog.md), [2026-06-04-texture-gap-augmentation.md](../../superpowers/plans/2026-06-04-texture-gap-augmentation.md)

## Problem

The 128³ U-Net trained on synthetic patches fires on **noise blobs**, not fibers, when run on real LSM (dandiset OME-Zarr, sliding-window inference over region 1 & 2). Goal: make the synthetic training set look like the real regions **and** give it better (thinner, more accurate) ground-truth masks, then retrain.

This is a continuation of the texture-gap work, not a greenfield. Prior results already established:

- Pushing realism too hard regressed learnability: grain-2.5 dropped `val_dice` 0.853→0.746 and sanity dice 0.98→0.609. **Learnability is a hard guardrail.**
- `coarse_std` is **not** closeable by render knobs — it comes from the source NIfTI anatomy. Do not re-litigate it.
- The proxy metric is `real_pred_pos_frac_median` (robust to one hot region); gate = ≥2× baseline (≥0.000412) sustained ≥3 val epochs, breadth ≥3 of the floor-7 regions.

## Evidence from the real regions (measured 2026-06-24)

Region 1 & 2 are `(402, 512, 512)` real LSM volumes (`sliding_infer_out/region{1,2}_image.npy`).

| Metric | Real | Synth baseline | grain-2.5 | Note |
|---|---|---|---|---|
| `adj_voxel_corr` | ~0.43 (x≈0.63, y≈0.22) | 0.854 | 0.80 | real is far less smooth; still 2× off even at grain-2.5 |
| `local_std` (8×8) | ~0.15 | — | — | synthetic Gaussian grain tops out ~0.06 |
| horizontal banding | row-mean std **0.074** | present (0.35) | — | **real HAS banding** |
| fiber/bg contrast | low (model-fired spots ≈1.03×) | high | — | model fires on near-zero-contrast noise |

**Two corrections to earlier assumptions:**

1. The model's binary output on real data is **not** a reliable fiber locator. At the slice where it fired most (z=140), the "fiber" was a single isolated 3–4 voxel blob at z=139–141 with 1.03× contrast and zero Z-persistence — i.e. the model fired on noise. This is the documented "fragments into blobs" failure, confirmed visually.
2. Real LSM **does** have horizontal banding (row-mean std 0.074). An earlier edit set `--disable_horizontal_banding`; that moves away from realism and is reverted by this design.

**Root cause of the texture gap:** the pipeline has additive-Gaussian granular noise (signal-independent, σ≈0.025·strength → ~0.05 at strength 2.0), but **no Poisson/shot noise** — the dominant noise source in fluorescence light-sheet. Signal-independent Gaussian cannot reproduce the intensity-dependent speckle of real LSM, which is why even grain-2.0 leaves `adj_voxel_corr` at 0.80.

## Approach: Balanced

Add the **missing physics** and improve the **labels**, while protecting learnability with a stats gate. Three axes:

### Part 1 — Image texture (close the speckle gap)

- **A1. New `apply_poisson_shot_noise()`** in [syntract_viewer/synthetic_image_augmentations.py](../../../syntract_viewer/synthetic_image_augmentations.py).
  - Formula: `noisy = rng.poisson(np.clip(vol, 0, None) * gain) / gain`, on the unit-normalized volume.
  - **`gain`** is the only knob: low gain (~40) = heavy shot noise, high gain (~150) = nearly clean. Variance = mean, so brighter voxels are noisier — matching real LSM.
  - **Image only.** Never applied to the mask. Applied after fiber compositing, like the other image augs.
  - Follows the existing aug shape: `_normalize_unit` → transform → `_restore_range`, `strength`/`random_state`/`verbose` signature, NumPy-only (no hard torch/cupy dependency).
  - Demonstrated: at gain=80 on a toy gradient+fiber, drops `adj_corr` 0.95→0.79 and raises `local_std` 0.02→0.078 with bright-region noise 2× the dark-region noise (signal-dependent), where Gaussian is flat.
- **A2. Re-enable banding.** Revert `--disable_horizontal_banding`; keep `--enable_horizontal_banding --banding_strength 0.35 --banding_axis 1` (real data measurably has it).
- **A3. Keep granular at baseline 1.5**, not 2.0. Poisson becomes the primary texture driver; do not stack two strong noise sources (that was the grain-2.5 over-hardening mistake).

### Part 2 — Mask quality (soft, thin, accurate)

The raw mask is a trilinear splat of the streamline centerline (~1 vox). Current precompute (`mask_smoothing_sigma 1.0`, `mask_binary_threshold 0.2`) blurs it to ~3–4 vox and keeps most of the blur → tubes thicker than real fibers (~2–3 vox), with soft edges that encourage fat-blob predictions.

- **B1. Soft (partial-volume) masks** (`--soft_mask`). Keep fractional trilinear coverage instead of blur-then-threshold. Sub-voxel accurate, no stair-stepping. BCE accepts soft targets; datamodule preserves them (`np.clip(mask,0,1)`); pairs with the clDice topology term (rewards thin continuous centerlines = anti-blob).
- **B2. Match real width.** `mask_smoothing_sigma 1.0 → 0.5` so the soft core stays ~2 vox. No aggressive post-blur.
- **B3. Verify in the probe:** mask width ≈ 2–3 vox, centered on the fiber, fully decoupled from the Poisson noise (noise never leaks into the label).

**Re-baseline note:** soft masks change the training target from binary to continuous. The 0.853 `val_dice` baseline was on binary masks, so the new `val_dice` is **not directly comparable** — re-baseline from the new run's own epoch-0/early epochs.

### Part 3 — Validation & training plan

1. **Stats probe (cheap gate).** Precompute ~30 patches with the new config. Sweep Poisson `gain` (≈40/80/120); measure `adj_voxel_corr` + `local_std` vs real (targets ≈0.43 / 0.15). Pick the gain landing closest **while** fiber/bg separation ≥3–4×. Overlay soft masks → confirm Part 2 B3. **Proceed only if** texture moved toward real *and* learnability floor held.
2. **Full precompute.** 1800 patches, `OUTPUT_DIR=./precomputed_patches_poisson_soft`: Poisson(gain=winner) + granular 1.5 + banding 0.35 ON + `--soft_mask` + `mask_smoothing_sigma 0.5`.
3. **Train** via `train_cached.sh`: new `PATCH_DIR`, fresh checkpoint dir, `--loss bce_cldice`, `--real_proxy_zarr` on the 561nm zarr (same as baseline for continuity).
4. **Judge against three signals (re-baselined):**

   | Signal | Meaning | Pass condition |
   |---|---|---|
   | `real_pred_pos_frac_median` | real-data firing | rises broadly (≥3 of floor-7 regions), sustained ≥3 epochs |
   | sanity dice (soft) | inference path intact | stays high (≥~0.9) |
   | synthetic `val_dice` | learnability guard | no collapse like grain-2.5 |

**Rollback:** if realism re-breaks learnability, raise `gain` (less shot noise). Poisson degrades more gracefully than stacking Gaussian.

## Files touched

- `syntract_viewer/synthetic_image_augmentations.py` — new `apply_poisson_shot_noise()`.
- `syntract_viewer/volume_renderer.py` and/or `synthetic-training/precompute_patches_3d.py` — wire the new aug into the render path with a CLI flag (`--enable_poisson_noise`, `--poisson_gain`), mirroring the existing `--enable_granular_noise` plumbing.
- `synthetic-training/precompute_patches.sh` — new config (Poisson on, banding re-enabled, granular 1.5, `--soft_mask`, `mask_smoothing_sigma 0.5`, new `OUTPUT_DIR`).
- `synthetic-training/train_cached.sh` — point `PATCH_DIR` at the new patches; fresh checkpoint dir.
- `docs/superpowers/plans/texture-gap-runlog.md` — append a new variant section (Poisson + soft masks).

## Non-goals / guardrails

- Do not chase `coarse_std` (anatomical, not closeable).
- Do not literally match real's faint ~1× fiber contrast — that re-breaks learnability. Synthetic masks are ground truth; the fiber must stay separable (≥3–4×).
- Do not stack Poisson on top of strong Gaussian grain (keep grain at 1.5).
- Preserve the fine-resolution fixes (FOV anchoring, streamline-anchored sampling, percentile normalization). Poisson and soft masks are additive, not replacements.
- bf16 precision, gradient clipping — unchanged.
