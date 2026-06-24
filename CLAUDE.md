# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Longer orientation docs: [AI_CONTEXT.md](AI_CONTEXT.md) (read first), [.github/copilot-instructions.md](.github/copilot-instructions.md). Full API + architecture reference: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md). In-depth code review with prioritized bug list: [docs/CODEBASE_REVIEW.md](docs/CODEBASE_REVIEW.md). Keep all three instruction files in sync when changing project-wide rules.

## Commands

```bash
pip install -r requirements.txt          # install deps
pytest                                    # full unit suite
pytest tests/test_nifti_preprocessing.py  # narrow run while editing
python run_comprehensive_tests.py         # custom integration runner
```

`tests/conftest.py` adds `synthesis/` to `sys.path` so tests can import either via the package or as standalone scripts (see "Import patterns" below). Many tests need real NIfTI/TRK/OME-Zarr data and are expensive — prefer the narrowest useful test and state what was skipped.

Single-file pipeline (the orchestrator is `syntract.py`):

```bash
python syntract.py --input brain.nii.gz --trk fibers.trk --output result
python syntract.py ... --use_ants --ants_warp warp.nii.gz --ants_iwarp iwarp.nii.gz --ants_aff affine.mat
python syntract.py ... --3d_output --white_mask wm_mask.nii.gz --total_patches 10 --patch_size 1024 40 1024
python syntract.py ... --disable_patch_processing --new_dim 116 140 96   # traditional full-volume path
```

Batch over a directory of TRK files sharing one NIfTI: `python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/`.

3D synthetic training entry point: `python synthetic-training/train_on_synthetic_data_3d.py --on_the_fly --trk_dir ... --input_nifti ... --checkpoint_dir ... --no_wandb`.

```bash
# Sanity-check a trained checkpoint against a known synthetic patch (dice ≈ 0.98 → inference path OK)
python synthetic-training/sanity_check_synthetic.py --checkpoint best_3d.ckpt --voxel_size 0.001
# Sanity-check the thin-slab sliding-window path
python synthetic-training/sanity_check_thinslab.py --checkpoint best_3d.ckpt
# Test inference on a specific OME-Zarr region
python test_specific_region.py --zarr /path/to/volume.ome.zarr --checkpoint best_3d.ckpt --normalize percentile
```

## Architecture

Two production packages plus a training tree:

- [synthesis/](synthesis/) — core MRI/tractography processing. [main.py](synthesis/main.py) (`process_and_save`) is the traditional full-volume path; [patch_first_processing.py](synthesis/patch_first_processing.py) is the default patch-first path; [nifti_preprocessing.py](synthesis/nifti_preprocessing.py), [streamline_processing.py](synthesis/streamline_processing.py), [densify.py](synthesis/densify.py), [ants_transform_updated.py](synthesis/ants_transform_updated.py), [gpu_utils.py](synthesis/gpu_utils.py).
- [syntract_viewer/](syntract_viewer/) — rendering. [generation.py](syntract_viewer/generation.py) builds 2D synthetic examples; [core.py](syntract_viewer/core.py) saves NIfTI/TRK visualizations and masks; [volume_renderer.py](syntract_viewer/volume_renderer.py) does 3D rendering with streamlines; [improved_cornucopia.py](syntract_viewer/improved_cornucopia.py) drives the weighted preset selection (clean 30% / subtle 30% / moderate 20% / heavy 20%); [synthetic_image_augmentations.py](syntract_viewer/synthetic_image_augmentations.py) is image-only 3D realism for training.
- [synthetic-training/](synthetic-training/) — PyTorch Lightning 2D/3D training, OME-Zarr loading, prediction, preview. Notable: [unet3d.py](synthetic-training/unet3d.py), [loss_functions.py](synthetic-training/loss_functions.py), [datamodules/datasets.py](synthetic-training/datamodules/datasets.py) (`SyntheticDataset3D` / `OnTheFlySyntheticData3D`), [datamodules/dataloaders.py](synthetic-training/datamodules/dataloaders.py) (Lightning DataModules), [datamodules/omezarr.py](synthetic-training/datamodules/omezarr.py) (physical-scale patch extraction → fixed output tensor shapes), [precompute_patches_3d.py](synthetic-training/precompute_patches_3d.py).

The main data flow: load NIfTI+TRK → optional ANTs transform → auto-calculate target dims from physical size and `voxel_size` (`syntract.py::calculate_target_dimensions`, constrained to 32–4000 voxels/dim, aspect ratio preserved) → sample valid patch locations with a streamline-count threshold → resample only those patches → clip streamlines to the patch FOV → render 2D/3D → save masks/summaries.

[cumulative.py](cumulative.py) wraps `process_batch` (CLI + Python API) and `process_patches_inmemory` (no file I/O — returns `images, masks` arrays). It auto-tunes patch counts per file: >100k streamlines → more patches, <10 → minimal patches.

## Project-specific conventions

- **Patch-first is the default and the right answer.** Don't add `--disable_patch_processing` or remove the patch path unless the task explicitly asks for full-volume synthesis. Full-volume is much slower and memory-heavy.
- **CPU fallback is mandatory.** All GPU code goes through [synthesis/gpu_utils.py](synthesis/gpu_utils.py) (`try_gpu_import` → `xp = numpy or cupy`). Don't hard-import `cupy`, CUDA, or MPS.
- **Dual import pattern.** Modules are used both as a package and as standalone scripts, so most imports look like:
  ```python
  try:
      from .module import function
  except ImportError:
      from module import function
  ```
  Preserve this shape when adding new modules.
- **Coordinate systems are easy to break.** Streamlines, RAS, voxel coords, NIfTI affines, and ANTs transforms (warp + inverse warp + affine, converted to RAS+ for TrackVis) all interact in `synthesis/`. Be careful around `ants_transform_updated.py` and `streamline_processing.py`.
- **Mask defaults are intentionally aggressive in current main paths:** `mask_thickness=1` (auto-scales by image size), `density_threshold=0.6`, `min_bundle_size=2000`, `use_high_density_masks=True`. The README lists older lighter defaults (`0.15` / `20`) for documented args — the unified internal defaults are the aggressive ones. Don't mix old and new defaults in the same code path.
- **Memory discipline.** Use `nib.load(..., mmap=True)`, batch + `gc.collect()` between batches, close matplotlib figures (`plt.ioff()`, low `figure.max_open_warning`). Generated NIfTI/TRK/PNG outputs can be huge — don't delete or overwrite user outputs unless explicitly asked.
- **Output naming.** Patches: `{prefix}_{NNNN}.nii.gz` / `.trk`. Visualizations: `{viz_prefix}_{n}_*.png`. Masks: `*_mask_slice{n}.png`.

## Fine-resolution 3D synthesis (sub-micron voxels, e.g. 0.001mm / 64³ patches)

Generating training data to match high-resolution light-sheet (SPIM/OME-Zarr) inference at ~1µm. At these scales the patch FOV (`patch_size * voxel_size`) is far smaller than the source NIfTI voxels and the native streamline spacing, which exposed several issues — all now handled in [synthesis/patch_first_processing.py](synthesis/patch_first_processing.py) and [syntract_viewer/volume_renderer.py](syntract_viewer/volume_renderer.py):

- **Patch FOV must anchor to the requested RAS window, not the voxel-snapped corner.** `calculate_patch_bbox_ras` floors/ceils onto the source voxel grid; at fine target voxels that snap is hundreds of target voxels, which put streamlines on the patch border. `synthesize_patch_region` now sets the target affine origin to `bbox['ras_min']`. Don't revert this.
- **Patch sampling is streamline-anchored.** Uniform random centers almost never hit a streamline when the FOV is sub-millimetre, so `sample_patch_locations_transformed_space` anchors candidate centers on actual streamline points (jittered), falling back to uniform when no streamlines. When the target FOV is smaller than the streamline extent, the target affine is re-centered on the streamline centroid.
- **Densification is lazy and per-patch.** Streamlines are resampled to `voxel_size * 0.5` inside `filter_streamlines_to_patch_ras` (only segments touching the patch), not upstream — densifying a thickened TRK upstream blows up to >1B points. Don't use `densify_streamlines_parallel` for this; it overrides sub-voxel `step_size` with a curvature-adaptive one.
- **Train/inference normalization MUST match.** OME-Zarr inference ([datamodules/omezarr.py](synthetic-training/datamodules/omezarr.py)) uses 1–99 percentile → [0,1]. Both synthetic 3D paths in [datamodules/datasets.py](synthetic-training/datamodules/datasets.py) now do the same. Never reintroduce min-max normalization (outlier-sensitive — one bright fiber sets the scale).
- **Soft (partial-volume) masks.** A binary voxel mask of a 1µm-thin diagonal fiber always stair-steps; `soft_mask=True` in `create_3d_volume_with_streamlines` keeps fractional sub-voxel coverage (smooth, accurate, continuous). The mask rasterizer accumulates trilinear weights (not nearest-voxel). The datamodule preserves soft labels (`np.clip(mask, 0, 1)`, not `mask > 0`); BCE accepts soft targets. For a clean *binary* tube instead, set `soft_mask=False` + `mask_smoothing_sigma≈1.0` + `mask_binary_threshold≈0.2`.
- **Cell-body blob distractors.** `enable_cell_blobs=True` scatters Gaussian blobs into the tissue *image only* (never the mask) so the model learns fiber-vs-cell. See `add_cell_body_blobs`.
- **[thicken_trk.py](thicken_trk.py)** turns a sparse TRK into a denser bundle (parallel offset siblings) and/or adds organic micro-curvature (`--wave_amplitude_um`). Needed because at small FOV a single tractography streamline is straight and sparse; real fiber fields have many curved fibers. Use `--copies 1 --wave_amplitude_um N` to curve without thickening.

## Visualising a 3D training patch

`visualize_one_patch.py` generates one 128³ patch at 0.001mm voxels, renders it, and saves sagittal/coronal/axial slice + matching binary mask to `patch_preview.png`.

```bash
python visualize_one_patch.py                 # seed=42
python visualize_one_patch.py --seed 123      # different patch
python visualize_one_patch.py --out foo.png
```

Key design decisions baked into the script:

- **Uses `registered_trk/aligned_wavy.trk` only.** The plain ANTs-registered TRKs have native step ~0.25mm — 4× the 0.064mm patch FOV — so only 1 streamline survives `filter_streamlines_to_patch_ras` per patch. `aligned_wavy.trk` has step ~0.004mm (7k–16k pts/streamline), giving multiple curved streamlines per patch.
- **`min_streamlines_per_patch=2`** ensures at least 2 streamlines per patch.
- **`tissue_threshold=0.0`** — fibers render in all voxels. At 0.001mm voxels many voxels sit near zero after CLAHE; the default `tissue_threshold=2.0` silently skips most fiber voxels.
- **`fiber_render_mode="additive"`** with `fiber_intensity_min/max` scaled to the tissue range (~6–9 on a 0–40 scale). Using 60–100 (the training defaults) pushes fibers to the 99th percentile of the volume and they get crushed during 1–99 normalisation when cell blobs also contribute.
- **Image slice = mask slice.** The plot picks the slice index with maximum mask signal per axis and shows the same index for both image and mask — not MIP for one and slice for the other.
- **Cornucopia presets** restricted to `["ultra_heavy_speckle", "extreme_noise", "granular_realistic"]`. Presets `random_shapes_background` and `comprehensive_aggressive` produce structured vertical-line patterns that look artificial.

## Cached (precompute) 3D training — the fast path

On-the-fly 3D training at `voxel_size=0.001` is GPU-starved: with `--num_workers 0` each step blocks ~45–60s on single-threaded CPU synthesis (`render~42s` + `extract~17–29s`) while the GPU sits idle ~85%. `--num_workers>0` is gated off for on-the-fly because patch+render run on the GPU ([datasets.py] `supports_multiprocess`). The fix is to precompute patches once, then train from disk:

```bash
sbatch synthetic-training/precompute_patches.sh   # writes synthetic-training/precomputed_patches/<trk_stem>/*_3d.nii.gz
# CONFIRM a healthy count before training:
find synthetic-training/precomputed_patches -name '*_3d.nii.gz' | wc -l
sbatch synthetic-training/train_cached.sh         # --cached_patches --num_workers 12, fresh checkpoint dir
```

Gotchas baked into these scripts (don't re-break):

- **`precompute_patches_3d.py` must match the on-the-fly render config.** It now exposes `--tissue_threshold`, `--enable_cell_blobs`, `--cornucopia_presets`, banding/dash, `--patch_use_gpu`, and derives `new_dim` from physical size like the datamodule. Patches discovered via `rglob` (per-TRK subdirs); discovery matches BOTH `.nii` and `.nii.gz` (extractor writes uncompressed `.nii` when `skip_2d_viz=True` — a `.nii.gz`-only filter silently renders nothing). `_3d` outputs are forced to `.nii.gz` so the renderer's `output_file.replace('.nii.gz','_mask.nii.gz')` names the mask correctly.
- **No double augmentation.** Noise is baked into precomputed patches; `train_cached.sh` leaves the load-time image augs OFF. Thin-slab/empty-patch shape augs ARE applied at load (ported into `SyntheticDataset3D`) to match on-the-fly.
- **`--white_mask` is optional**; the `"None"` string sentinel must become real `None` before `process_patches_inmemory` (else it errors looking for a file named "None").
- **Held-out val split (cached path).** The cached datamodule used to point train AND val at the same `patch_dir`, so `val_loss` only measured training fit, not generalization. `SyntheticDataset3D` now takes `split=("all"|"train"|"val")` + `val_fraction` (default 0.15) + `split_seed`; it shuffles the discovered pairs by `split_seed` BEFORE partitioning so both sides get the same per-TRK mix (a contiguous split would carve along the sorted-by-subdir boundary and could starve one side of the only dense TRK, `aligned_wavy`). It prints per-subdir counts per split as a direct check. The datamodule passes `split="train"`/`split="val"` and disables thinslab/empty-patch augs on val (deterministic, reproducible metric). CLI: `--val_fraction` / `--split_seed`. Set `--val_fraction 0.0` for the legacy val==train behaviour. **A held-out *synthetic* split sits ~0.87 dice and stays flat when you change augmentation in step (b) — synthetic generalization isn't what's broken; it guards against *regressing* it.**
- **Real-LSM transfer proxy (unlabeled).** `RealLSMProxyCallback` in `train_on_synthetic_data_3d.py` runs the model each val epoch on a fixed 3×3 OME-Zarr center grid and logs `real_pred_pos_frac` (mean) and `real_pred_pos_frac_median` (watch this — robust to one hot region). Enable with `--real_proxy_zarr <path>` (default `"None"` → no-op). **Calibrated baseline (2026-06-04):** regions 1–3 median = 0.000200; epoch-149 training run baseline = 0.00140. Anchor all step-(b) movement against the *same run's own* epoch-0 baseline, not the calibration figure. `calibrate_real_proxy.py` re-derives it on cluster GPU (OOMs on laptop).

## 3D training stability and inference

- **Use `bf16-mixed`, not `16-mixed`.** fp16 overflows (~65504) in the forward pass — the AMP GradScaler only guards gradients, so a forward-pass overflow NaN-poisons the weights permanently (seen as: dice climbs to ~0.35 then collapses to flat-NaN around step ~10k). bf16 has fp32 range, needs no scaler, free on H100/H200. `train_on_synthetic_data_3d.py` now auto-selects bf16 when supported and sets `gradient_clip_val=1.0`. A NaN-poisoned `last.ckpt` will reload on resume — start fresh (`--no_resume` / new `--checkpoint_dir`).
- **`sanity_check_synthetic.py`** is the discriminator when real-data inference looks empty: run the trained model through the inference path on a KNOWN synthetic patch+mask. dice≈0.98 ⇒ inference path is correct and the real-data failure is a genuine synthetic→real **domain gap** (not a code bug). `compare_domain_stats.py` then quantifies the gap (synthetic tissue is smoother / has a stronger background gradient+banding than real LSM). `compare_multiregion.sh` sweeps several distant regions to tell a universal gap from a location-specific one.
- **Train/inference normalization is 1–99 percentile over the FULL patch** (training, omezarr.py, and the test scripts use `--normalize percentile`). `nonzero_percentile` excludes zeros and shifts the scale — don't use it for inference matching.

## Known bugs (do not paper over, fix properly)

See [docs/CODEBASE_REVIEW.md](docs/CODEBASE_REVIEW.md) for full details and remediation steps.

| Priority | File | Bug |
|---|---|---|
| P0 | `synthesis/main.py:384` | `process_and_save()` returns `None` (not a dict) when all streamlines are filtered out — breaks every caller |
| P0 | `synthesis/main.py:536,624` | `args.force_compression` / `args.no_compression` are referenced but never added to the argparser → `AttributeError` if that branch is reached |
| P1 | `synthesis/nifti_preprocessing.py:237` | CPU full-volume resampler uses nearest-neighbor (`int()`), not the claimed cubic — GPU and CPU produce different outputs |
| P1 | `synthesis/main.py:568` | Duplicate `__main__` block with different defaults (`--interp rbf` vs `hermite`) — replace with `main()` |
| P2 | `synthesis/streamline_processing.py:251` | `[STREAMLINE DEBUG]` prints fire unconditionally in every transform call, including inside training loops |
| P2 | `synthesis/patch_first_processing.py:555` | `DEBUG VIOLATIONS` prints fire unconditionally regardless of the `debug` flag |

## Things to avoid

- Disabling patch processing by default, hard-coding GPU, mixing mask defaults, leaving matplotlib figures open in batch loops, assuming dimensions instead of using auto-calculation, rewriting large modules for style.
- Reverting the fine-resolution fixes above (FOV anchoring, streamline-anchored sampling, percentile normalization, soft masks) — they fix real bugs exposed at sub-micron voxel sizes, not cosmetic preferences.
- Running a full 128³ 3D forward pass on a laptop CPU — it allocates GB of activations and OOMs the machine. Use a GPU (cluster) for any `*_3d` inference/sanity script.
- `16-mixed` precision for 3D training (use bf16), committing `Model_prediction/` / `*.npy` / `*.ckpt` / `precomputed_patches/` (all gitignored — large binaries).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
