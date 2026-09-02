# SynTract — Full Codebase Documentation

> Generated: 2026-06-11  
> Branch: `3_dimension`  
> Graph: `graphify-out/` (1341 nodes · 2432 edges · 78 communities)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Data Model & Coordinate Systems](#3-data-model--coordinate-systems)
4. [End-to-End Data Flow](#4-end-to-end-data-flow)
5. [Package: `synthesis/`](#5-package-synthesis)
6. [Package: `syntract_viewer/`](#6-package-syntract_viewer)
7. [Package: `synthetic-training/`](#7-package-synthetic-training)
8. [Root-Level Scripts](#8-root-level-scripts)
9. [CLI Reference](#9-cli-reference)
10. [Public Python API](#10-public-python-api)
11. [Configuration & Defaults](#11-configuration--defaults)
12. [Testing](#12-testing)
13. [Dependency Map](#13-dependency-map)
14. [Glossary](#14-glossary)

---

## 1. Project Overview

**SynTract** is a Python neuroimaging toolkit for:

- Resampling and transforming NIfTI MRI volumes (`.nii`, `.nii.gz`) with matching TRK tractography files (`.trk`).
- Extracting spatial patches directly at target resolution, avoiding large full-volume intermediates (*patch-first* default path).
- Rendering synthetic 2D and 3D training images with fiber overlays, cornucopia augmentations, and realistic tissue appearance.
- Training 3D segmentation U-Nets on synthetic fiber data and transferring to real light-sheet microscopy (OME-Zarr) volumes.
- Evaluating the synthetic-to-real domain gap and running inference over thin-slab OME-Zarr data.

**Primary use case**: generate labeled training patches pairing a synthetic MRI-like image with a ground-truth binary (or soft) fiber mask, ready to train a 3D U-Net for axon/fiber segmentation in real microscopy volumes.

---

## 2. Repository Layout

```
syntract-3d/
├── syntract.py                    # Main CLI entry point (single NIfTI+TRK → patches)
├── cumulative.py                  # Batch CLI: one NIfTI + directory of TRKs
├── thicken_trk.py                 # TRK density augmentation (copies + waviness)
├── visualize_one_patch.py         # Quick 3D patch preview script
├── calibrate_real_proxy.py        # Calibrate real-data proxy metric (cluster only)
├── test_specific_region.py        # Targeted OME-Zarr region inference tester
├── validate_dataset.py            # Dataset quality validation helpers
├── compare_predictions.py         # Side-by-side prediction comparison
├── batch_ants_trk_registration.py # Batch ANTs registration helper
│
├── synthesis/                     # Core MRI + tractography processing package
│   ├── __init__.py
│   ├── main.py                    # Full-volume pipeline: process_and_save()
│   ├── patch_first_processing.py  # Patch-first pipeline: process_patch_first_extraction()
│   ├── nifti_preprocessing.py     # NIfTI resampling (GPU cubic + CPU trilinear)
│   ├── streamline_processing.py   # Streamline transform, FOV clip, densify dispatch
│   ├── densify.py                 # Interpolation methods (linear/hermite/RBF)
│   ├── transform.py               # Affine matrix builder: build_new_affine()
│   ├── ants_transform_updated.py  # ANTs warp + inverse warp + affine application
│   ├── gpu_utils.py               # Centralized GPU detection + graceful fallback
│   ├── validation.py              # Parameter validation helpers
│   ├── visualize.py               # Synthesis debugging visualizations
│   ├── slice_simplified.py        # Simplified coronal slice extraction
│   └── compare_interpolation.py   # Interpolation method comparison tool
│
├── syntract_viewer/               # Rendering, augmentation, mask generation package
│   ├── __init__.py
│   ├── core.py                    # NIfTI + TRK visualization; mask saving
│   ├── generation.py              # 2D synthetic example generation (main generator)
│   ├── volume_renderer.py         # 3D volume rendering with GPU line kernels
│   ├── volumetric_3d.py           # 3D volumetric processing pipeline
│   ├── improved_cornucopia.py     # Weighted cornucopia augmentation presets
│   ├── cornucopia_3d.py           # True 3D Cornucopia augmentation (full volume)
│   ├── synthetic_image_augmentations.py  # Image-only 3D realism augmentations
│   ├── contrast.py                # CLAHE + contrast enhancement
│   ├── masking.py                 # Brain + fiber mask creation
│   ├── effects.py                 # Dark-field effects
│   ├── background_enhancement.py  # LPSVD + slice sharpening
│   ├── patch_extraction.py        # Patch visualization helper
│   ├── generate_fiber_examples.py # Spatial-subdivision example generator
│   ├── utils.py                   # Streamline utilities (colormap, densify, etc.)
│   └── orange_blob_generator.py   # Orange injection-site artifact simulator
│
├── synthetic-training/            # PyTorch Lightning training, inference, OME-Zarr
│   ├── train_on_synthetic_data_3d.py   # Primary 3D training script
│   ├── train_on_synthetic_data.py      # Legacy 2D training script
│   ├── unet3d.py                       # FlexibleUNet3D Lightning module
│   ├── unet.py                         # FlexibleUNet (2D) Lightning module
│   ├── loss_functions.py               # Dice, BCE, Focal, clDice, DiceBCE, DiceFocal
│   ├── precompute_patches_3d.py        # Offline 3D patch pre-generation
│   ├── predict_omezarr_thinslab_3d.py  # Thin-slab sliding-window inference
│   ├── predict_synthetic_data_3d.py    # 3D synthetic data prediction
│   ├── predict_synthetic_data.py       # 2D synthetic data prediction
│   ├── sanity_check_synthetic.py       # Sanity check: model on known synthetic patch
│   ├── sanity_check_thinslab.py        # Sanity check: thin-slab path
│   ├── view_prediction_3d.py           # 3D prediction visualization
│   ├── view_prediction_npy_gui.py      # NPY prediction GUI viewer
│   ├── compare_domain_stats.py         # Quantify synthetic→real domain gap
│   ├── gen_3d_patches_offline.py       # Offline 3D patch generator (alternate)
│   ├── preview_patch_gamma.py          # Preview gamma/gain effect on patches
│   ├── preview_realistic_augmentations_3d.py  # QA: clean vs. augmented patches
│   ├── test_omezarr_patches.py         # OME-Zarr patch extraction sanity check
│   └── datamodules/
│       ├── datasets.py            # SyntheticDataset, OnTheFlySyntheticData3D, SyntheticDataset3D
│       ├── dataloaders.py         # Lightning DataModules (OnTheFlyDataModule3D, etc.)
│       └── omezarr.py             # PhysicalScaleOMEZarrDataset + OMEZarrPatchDataModule
│
├── tests/                         # pytest unit and integration tests
│   ├── conftest.py                # sys.path setup
│   ├── test_nifti_preprocessing.py
│   ├── test_streamline_processing.py
│   ├── test_densify.py
│   ├── test_transform.py
│   ├── test_main.py
│   ├── test_comprehensive_integration.py
│   ├── test_synthesis_complete.py
│   └── test_syntract_viewer_complete.py
│
├── graphify-out/                  # Pre-built knowledge graph (AST + LLM extraction)
├── docs/                          # Documentation directory (this file)
├── requirements.txt               # Python dependencies
├── setup.py / pyproject.toml      # Package config
├── CLAUDE.md                      # AI agent instructions (ground truth for agents)
├── AI_CONTEXT.md                  # Orientation doc for AI agents
└── HANDOFF.md                     # Current experiment state / handoff notes
```

---

## 3. Data Model & Coordinate Systems

SynTract operates in four coordinate spaces that must stay consistent throughout a pipeline run:

| Space | Unit | Description |
|---|---|---|
| **Voxel** | integer indices | Array indices `(i, j, k)` into a NIfTI volume |
| **RAS mm** | mm | Right-Anterior-Superior physical space; used by NIfTI affines and TRK streamlines |
| **Patch voxel** | float indices | Local voxel indices within a single patch (0 → patch_size) |
| **ANTs warp** | mm displacement | Deformation field produced by ANTs registration |

**Key invariants:**
- TRK streamlines are stored and passed in RAS mm coordinates unless explicitly noted as voxel coordinates.
- All affine matrices follow the NIfTI convention: `RAS_mm = affine @ [i, j, k, 1]`.
- `nib.as_closest_canonical()` is called on load to normalize axis order (important before patch extraction).
- After ANTs, streamlines are converted to the *fixed image* voxel space via `affine_vox2fix`, then back to RAS for subsequent operations.

**Pitfalls to avoid:**
- Do not apply voxel-space operations to RAS-space streamlines or vice versa.
- Do not use `synthesis/densify.py::densify_streamlines_parallel()` for per-patch lazy densification — it overrides the sub-voxel step size with a curvature-adaptive one. Use `_densify_segment_for_patch()` instead.
- At sub-millimeter voxel sizes the voxel-snapped bbox origin can be hundreds of target voxels off; `synthesize_patch_region()` sets the target affine origin to `bbox['ras_min']` to fix this.

---

## 4. End-to-End Data Flow

### 4.1 Patch-First Path (default)

```
Input: brain.nii.gz + fibers.trk
          │
          ▼
[syntract.py::process_syntract()]
          │
          ├──(use_ants?)──► process_with_ants() → streamlines_ras (RAS mm)
          │                  synthesis/ants_transform_updated.py
          │
          ▼
[synthesis/patch_first_processing.py::process_patch_first_extraction()]
          │
          ├── Load NIfTI (mmap) + TRK → streamlines in RAS mm
          ├── Build streamline bounding boxes (_build_streamline_bounds)
          ├── Auto-center target FOV on streamline centroid (if FOV < streamline extent)
          ├── Build target_affine via synthesis/transform.py::build_new_affine()
          │
          ├─ FOR EACH PATCH:
          │   ├── sample_patch_locations_transformed_space()
          │   │     streamline-anchored sampling (jittered streamline points)
          │   │
          │   ├── calculate_patch_bbox_ras()
          │   │     → {ras_min, ras_max, vox_min, vox_max, center_ras, size_mm}
          │   │
          │   ├── synthesize_patch_region()
          │   │     extract NIfTI sub-volume → resample_nifti_patch()
          │   │     (GPU: torch.grid_sample bilinear; CPU: trilinear)
          │   │     target affine origin = bbox['ras_min']  ← critical fix
          │   │
          │   ├── filter_streamlines_to_patch_ras()
          │   │     bounds check + lazy densification (_densify_segment_for_patch)
          │   │     convert to patch voxel coords + strict bound clipping
          │   │
          │   └── Save patch_{NNNN}.nii.gz + patch_{NNNN}.trk
          │
          └── RETURN results dict: {patches_extracted, patch_details, processing_time}
                    │
                    ▼
          [syntract.py] dispatch to visualization:
          ├──(3d_output)──► syntract_viewer/volume_renderer.py::create_3d_volume_with_streamlines()
          │                  GPU line kernels (CuPy RawKernel) or CPU fallback
          │                  Cornucopia 3D augmentation (cornucopia_3d.py)
          │                  Soft mask (trilinear fiber accumulation)
          │
          └──(2d_output)──► syntract_viewer/patch_extraction.py::_generate_patch_visualization()
                             syntract_viewer/improved_cornucopia.py preset selection
                             syntract_viewer/contrast.py CLAHE
                             syntract_viewer/masking.py fiber + brain masks
                             Save PNG image + PNG mask
```

### 4.2 Full-Volume Path (legacy, `--disable_patch_processing`)

```
Input: brain.nii.gz + fibers.trk
          │
          ▼
synthesis/main.py::process_and_save()
    ├── resample_nifti() → full resampled NIfTI [can be multi-GB]
    ├── transform_and_densify_streamlines() → all streamlines in voxel space
    ├── clip_streamline_to_fov() per streamline
    └── Save resampled.nii + resampled.trk
```

### 4.3 3D Training Pipeline

```
Precompute (offline):
  precompute_patches_3d.py → patch_dir/<trk_stem>/*_3d.nii.gz + *_3d_mask.nii.gz

OR On-the-fly:
  cumulative.py::process_patches_inmemory() → (images, masks) np arrays in RAM

          │
          ▼
synthetic-training/train_on_synthetic_data_3d.py
    ├── OnTheFlyDataModule3D / SyntheticDataset3D (cached)
    ├── FlexibleUNet3D (PyTorch Lightning)
    ├── Loss: DiceBCELoss (default)
    ├── Precision: bf16-mixed (mandatory on H100/H200)
    ├── Val split: synthetic held-out (SyntheticDataset3D split='val')
    ├── Real proxy: RealLSMProxyCallback → real_pred_pos_frac_median (transfer signal)
    └── Checkpoint: best_{val_loss}.ckpt

          │
          ▼
Inference (OME-Zarr):
  predict_omezarr_thinslab_3d.py
    ├── PhysicalScaleOMEZarrDataset (omezarr.py) → physical-scale patches
    ├── Sliding window over thin-slab ZYX region
    └── Save prediction NPY/NIfTI
```

---

## 5. Package: `synthesis/`

### 5.1 `main.py` — Full-volume pipeline

**Entry point:** `process_and_save()`

```python
def process_and_save(
    original_nifti_path: str,
    original_trk_path: str,
    target_voxel_size: float = 0.5,
    target_dimensions: tuple = (116, 140, 96),
    output_prefix: str = "resampled",
    num_jobs: int = 8,
    patch_center: tuple | None = None,
    reduction_method: str | None = None,    # 'mip' | 'mean'
    use_gpu: bool = True,
    interpolation_method: str = 'rbf',      # 'hermite' | 'linear' | 'rbf'
    step_size: float = 0.5,
    max_output_gb: float = 64.0,
    use_ants: bool = False,
    ants_warp_path: str | None = None,
    ants_iwarp_path: str | None = None,
    ants_aff_path: str | None = None,
    force_dimensions: bool = False,
    transform_mri_with_ants: bool = False,
    slice_count: int | None = None,
    enable_slice_extraction: bool = False,
    slice_output_dir: str | None = None,
) -> dict
```

**What it does:**
1. Optionally applies ANTs transforms to both MRI and streamlines.
2. Resamples the full NIfTI volume to `target_dimensions` using cubic (GPU) or trilinear (CPU) interpolation.
3. Transforms all streamlines to the new voxel space, densifies them, clips to FOV.
4. Saves `{output_prefix}.nii` + `{output_prefix}.trk`.
5. Optionally extracts coronal slices.

**Returns** `{'synthesis_outputs': {'nifti': ..., 'trk': ...}, 'slice_extraction': ...}`

---

### 5.2 `patch_first_processing.py` — Patch-first pipeline

**Entry point:** `process_patch_first_extraction()`

```python
def process_patch_first_extraction(
    original_nifti_path: str,
    original_trk_path: str,
    target_voxel_size: float = 0.05,
    target_patch_size: tuple = (700, 1, 700),
    target_dimensions: tuple = (1400, 1000, 1400),
    num_patches: int = 50,
    output_prefix: str = "patch_optimized",
    min_streamlines_per_patch: int = 30,
    use_ants: bool = False,
    ants_warp_path: str | None = None,
    ants_iwarp_path: str | None = None,
    ants_aff_path: str | None = None,
    random_state: int | None = None,
    use_gpu: bool = True,
    white_mask_path: str | None = None,
    use_compressed_nifti: bool = True,
    streamline_margin_fraction: float = 0.0,
    debug: bool = False,
) -> dict
```

**What it does (step by step):**

| Step | Function | Detail |
|---|---|---|
| 1 | `process_with_ants()` | Optional ANTs transform; streamlines → RAS mm |
| 2 | `_build_streamline_bounds()` | Pre-compute per-streamline AABB for fast lookup |
| 3 | `build_new_affine()` | Compute target coordinate system; auto-center if FOV < streamline extent |
| 4 | `sample_patch_locations_transformed_space()` | Anchor on streamline points + jitter |
| 5 | `calculate_patch_bbox_ras()` | Compute RAS + voxel bbox for each center |
| 6 | `synthesize_patch_region()` | Extract + resample NIfTI sub-volume to target resolution |
| 7 | `filter_streamlines_to_patch_ras()` | Lazy densify → clip → convert to patch voxels |
| 8 | `validate_patch_spatial_alignment()` | Verify no streamline points out of bounds |
| 9 | Save `.nii.gz` + `.trk` | Patch files with correct headers |

**Returns** dict with `{success, patches_extracted, patches_failed, patch_details, processing_time}`.

**Key helpers:**

| Function | Signature | Purpose |
|---|---|---|
| `calculate_patch_bbox_ras` | `(center_ras, size_mm, affine) → dict` | RAS + voxel bbox |
| `count_streamlines_in_bbox` | `(streamlines, bbox, bounds?) → int` | Fast AABB-prefiltered count |
| `synthesize_patch_region` | `(nifti_path, bbox, voxel_size, patch_size, ...) → Nifti1Image` | Patch resampling |
| `filter_streamlines_to_patch_ras` | `(streamlines, bbox, affine, size, ...) → list[ndarray]` | Streamline clip + convert |
| `_densify_segment_for_patch` | `(streamline, ras_min, ras_max, step_mm) → ndarray` | Lazy per-segment densification |
| `sample_patch_locations_transformed_space` | `(affine, shape, size_mm, num, ...) → list` | Streamline-anchored sampling |

---

### 5.3 `nifti_preprocessing.py` — NIfTI resampling

| Function | Description |
|---|---|
| `resample_nifti(old_img, new_affine, new_shape, ...)` | Full-volume resampling with GPU CUDA cubic kernel or CPU trilinear via joblib |
| `resample_nifti_patch(patch_img, target_affine, target_shape, use_gpu)` | Lightweight patch resampling: GPU via `torch.nn.functional.grid_sample` (bilinear), CPU via vectorized trilinear |
| `cubic_kernel(x)` | Mitchell-Netravali cubic kernel used in GPU resampling |
| `estimate_memory_usage(shape, dtype)` | Returns estimated GB for a given array shape |

**Note:** `resample_nifti` for the full-volume path uses a custom Numba CUDA kernel for maximum throughput. `resample_nifti_patch` uses PyTorch `grid_sample` for the patch path — no Numba dependency required, just CUDA-capable PyTorch.

---

### 5.4 `streamline_processing.py` — Transform, clip, densify dispatch

| Function | Signature | Description |
|---|---|---|
| `clip_streamline_to_fov` | `(stream, new_shape, use_gpu, epsilon) → list[ndarray]` | Split streamline at FOV boundary; interpolate boundary points |
| `interpolate_to_fov` | `(p1, p2, new_shape, use_gpu) → ndarray` | Find exact boundary intersection |
| `transform_streamline` | `(s_mm, A_new_inv, use_gpu) → ndarray` | Apply affine: mm → voxel |
| `transform_and_densify_streamlines` | `(streamlines_mm, new_affine, new_shape, ...) → list[ndarray]` | Bulk transform + clip + densify dispatch (full-volume path) |

---

### 5.5 `densify.py` — Streamline interpolation

| Function | Description |
|---|---|
| `densify_streamline_subvoxel(streamline, step_size, interp_method, use_gpu)` | Main densification entry point. Dispatch to linear / hermite / RBF |
| `densify_streamlines_parallel(streamlines, step_size, n_jobs, ...)` | Parallel wrapper via joblib |
| `calculate_streamline_curvature(streamline)` | Mean curvature per point |
| `calculate_optimal_step_size(streamline, base_step, ...)` | Adaptive step from curvature |

**Interpolation methods:**
- `linear` — linear resampling at fixed arc-length step
- `hermite` — Hermite cubic spline (default for most paths)
- `rbf` — Radial basis function interpolation (more accurate, slower)

---

### 5.6 `transform.py` — Affine matrix construction

```python
def build_new_affine(
    old_affine: np.ndarray,
    old_shape: tuple,
    new_voxel_size: float | tuple,
    new_shape: tuple,
    patch_center_mm: tuple | None = None,
    use_gpu: bool = False,
) -> np.ndarray
```

Builds a new NIfTI affine that:
- Preserves the RAS orientation of the input volume.
- Uses `new_voxel_size` as the diagonal scaling.
- Centers the new volume on `patch_center_mm` if provided, else on the geometric center of the old volume.

---

### 5.7 `ants_transform_updated.py` — ANTs transform application

**Entry point:** `process_with_ants(warp_path, iwarp_path, aff_path, nifti_path, trk_path, ...)`

**Returns** `(moved_mri, affine_vox2fix, transformed_tractogram, streamlines_voxel)`

Key functions:
- `load_ants_warp(path)` — Load warp displacement field
- `load_ants_aff(path)` — Load ANTs affine `.mat`
- `apply_ants_transform_to_streamlines(streamlines, warp, affine, ...)` — Transform streamlines through warp + affine
- `apply_ants_transform_to_mri(nifti_path, warp, affine, ...)` — MRI warp application
- `check_affine_orientation(affine)` — Validate RAS+ convention

---

### 5.8 `gpu_utils.py` — GPU detection and fallback

**Singleton:** `get_gpu_support()` → `GPUSupport`

```python
class GPUSupport:
    cupy_available: bool
    numba_cuda_available: bool
    
    def get_array_module(prefer_gpu=True) -> np | cp
    def has_full_gpu_support() -> bool      # CuPy + Numba CUDA
    def has_partial_gpu_support() -> bool   # Either
    def try_import_cupy() -> (module, bool)
    def convert_to_numpy(array) -> ndarray
```

**Convenience functions at module level:**
```python
get_array_module(prefer_gpu=True) → np | cp
has_gpu_support() → bool
has_full_gpu_support() → bool
try_gpu_import() → dict   # {'xp', 'cuda', 'cupy_available', 'numba_available', 'gpu_support'}
```

**Usage pattern throughout codebase:**
```python
from synthesis.gpu_utils import try_gpu_import
result = try_gpu_import()
xp = result['xp']           # cupy or numpy
use_gpu = result['cupy_available']
```

---

## 6. Package: `syntract_viewer/`

### 6.1 `volume_renderer.py` — 3D rendering with streamlines

**Entry point:** `create_3d_volume_with_streamlines()`

```python
def create_3d_volume_with_streamlines(
    nifti_file: str,
    trk_file: str,
    output_file: str,
    orientation: str = 'coronal',
    white_mask_path: str | None = None,
    contrast_method: str = 'clahe',
    gamma: float = 1.0,
    fiber_intensity_min: float = 15.0,
    fiber_intensity_max: float = 25.0,
    use_cornucopia_3d: bool = True,
    cornucopia_allowed_presets: list | None = None,
    cornucopia_prob: float = 0.9,
    save_mask: bool = True,
    min_bundle_size: int = 2000,
    use_bilateral_smoothing: bool = False,
    texture_intensity: float = 0.45,
    texture_sigma: float = 0.8,
    clahe_clip_limit: float = 0.003,
    enable_cell_blobs: bool = True,
    cell_blob_count: int = 140,
    cell_blob_intensity: float = 0.35,
    cell_blob_radius_range: tuple = (1.2, 4.0),
    fiber_antialias: bool = True,
    fiber_smoothing_sigma: float = 1.0,
    fiber_brightness_variation: float = 0.5,
    fiber_segment_brightness_variation: float = 0.4,
    soft_mask: bool = True,
    mask_smoothing_sigma: float = 1.0,
)
```

**Rendering pipeline:**
1. Load NIfTI patch + TRK streamlines.
2. Apply CLAHE + texture + optional bilateral smoothing.
3. Render fiber segments via GPU Bresenham line kernels (`CuPy RawKernel`) or CPU fallback.
4. Optionally scatter Gaussian cell-body blobs into the image (distractors; never in the mask).
5. Apply 3D Cornucopia augmentation to image (not mask).
6. Save `output_file` (3D NIfTI image) + `output_file_mask.nii.gz` (soft or binary fiber mask).

**Key detail — soft mask:** At sub-micron voxel sizes a binary diagonal fiber stair-steps badly. `soft_mask=True` uses trilinear sub-voxel accumulation so the mask is fractional coverage in `[0,1]`. The training BCE loss accepts soft targets directly.

**Cell-body blobs:** `add_cell_body_blobs(volume, ...)` injects random Gaussian spheres *only into the image*, not the mask. This prevents the model from treating any bright structure as a fiber.

---

### 6.2 `generation.py` — 2D synthetic example generation

**Entry point:** `generate_examples_original_mode()` / `generate_varied_examples()`

Generates coronal/axial slice images with fiber overlays at various augmentation presets. Calls into:
- `core.py` — base NIfTI + TRK slice visualization
- `contrast.py` — CLAHE and augmentation pipeline
- `masking.py` — high-density fiber mask generation
- `improved_cornucopia.py` — weighted preset selection
- `orange_blob_generator.py` — optional injection artifact

---

### 6.3 `core.py` — NIfTI + TRK base visualization

Main functions:

| Function | Description |
|---|---|
| `visualize_nifti_with_trk(nifti, trk, output_dir, ...)` | Multi-view (axial/coronal/sagittal) visualization |
| `visualize_nifti_with_trk_coronal(nifti, trk, output_dir, ...)` | Coronal-only with high-density masks |
| `visualize_multiple_views(...)` | 3-panel multi-orientation view |

---

### 6.4 `improved_cornucopia.py` — Augmentation preset selection

```python
class ImprovedCornucopiaAugmenter:
    # 16+ named presets across 4 weight categories
    # clean 30% / subtle 30% / moderate 20% / heavy 20%
    
    def augment_fiber_slice(image: ndarray, preset: str | None = None) -> ndarray
    def create_optical_presets() -> dict
```

**Preset categories:**
- `clean` (30%): `clean_optical`, `minimal_noise`, `high_contrast`, `smooth_gradients`
- `subtle` (30%): `subtle_optical`, `slight_speckle`, `mixed_effects`
- `moderate` (20%): `moderate_optical`, `textured_background`, `speckle_heavy`
- `heavy` (20%): `heavy_optical`, `ultra_heavy_speckle`, `extreme_noise`, `granular_realistic`, `debris_field`

**For training data:** restrict to `['ultra_heavy_speckle', 'extreme_noise', 'granular_realistic']` — the others produce structured artifacts (vertical lines) not present in real LSM data.

---

### 6.5 `synthetic_image_augmentations.py` — Image-only 3D realism

```python
def apply_image_only_augmentations(
    image: ndarray,           # 3D float32 volume in [0, 1]
    enable_banding: bool = True,
    enable_speckle: bool = True,
    enable_granular: bool = True,
    enable_dash: bool = True,
    enable_tissue_artifacts: bool = True,
) -> ndarray
```

Functions: `apply_horizontal_banding`, `apply_speckle_dot_noise`, `apply_granular_noise`, `apply_dash_noise`, `apply_tissue_artifacts`, `_normalize_unit`.

**Critical:** These augmentations apply to the **image only**, never to the mask.

---

### 6.6 `masking.py` — Brain and fiber mask creation

| Function | Description |
|---|---|
| `create_aggressive_brain_mask(image, ...)` | Connected-component + morphological brain mask |
| `create_fiber_mask(image, streamlines, ...)` | Rasterize streamlines into binary mask; applies Gaussian smoothing |
| `filter_streamlines_by_density(streamlines, ...)` | Remove isolated low-density streamlines |
| `_generate_high_density_masks(...)` | Multi-view high-density mask pipeline |

**Current defaults (aggressive):**
- `mask_thickness=1` (auto-scaled by image size)
- `density_threshold=0.6`
- `min_bundle_size=2000`
- `use_high_density_masks=True`

---

### 6.7 `contrast.py` — CLAHE enhancement

```python
def apply_contrast_enhancement(image, method='clahe', clip_limit=0.01, ...) -> ndarray
def apply_enhanced_contrast_and_augmentation(image, trk_data, output_dir, ...) -> ndarray
def apply_comprehensive_slice_processing(image, ...) -> ndarray
```

---

### 6.8 `cornucopia_3d.py` — True 3D Cornucopia augmentation

```python
def apply_cornucopia_true_3d(volume: ndarray, preset: str = 'granular_realistic') -> ndarray
```

Applies the full Cornucopia augmentation pipeline volumetrically. Used inside `volume_renderer.py` for 3D training patches.

Individual transforms:
- `apply_3d_aggressive_gamma(volume)` — per-volume gamma
- `apply_3d_bias_field(volume)` — multiplicative field simulation
- `apply_3d_gaussian_mixture_noise(volume)` — 3D noise field
- `apply_3d_noncentral_chi_noise(volume)` — MRI-realistic noise

---

## 7. Package: `synthetic-training/`

### 7.1 `unet3d.py` — FlexibleUNet3D

```python
class FlexibleUNet3D(pl.LightningModule):
    # Input:  (B, 1, D, H, W)  single-channel 3D volume
    # Output: (B, 1, D, H, W)  binary segmentation logits
    
    def __init__(
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 1,
        min_features: int = 32,
        max_features: int = 320,
        num_stages: int = 5,
        loss: str = 'BCE',   # 'BCE' | 'focal' | 'cldice'
        freeze_encoder: bool = False,
        pos_weight: float = 1.0,
        in_channels: int = 1,
    )
```

**Architecture:** Standard U-Net encoder-decoder with:
- `DoubleConv3D` blocks (Conv3d → InstanceNorm3d → LeakyReLU) × 2 per stage
- `num_stages=5` encoder stages with 2× max pooling between stages
- `ConvTranspose3d` upsampling in decoder
- Skip connections (concatenation)
- `Conv3d(features[0], 1, kernel_size=1)` final layer

**Training details:**
- Optimizer: AdamW with cosine annealing + linear warmup
- Precision: **bf16-mixed** (mandatory; fp16 overflows in 3D forward pass)
- Gradient clipping: `gradient_clip_val=1.0` (set in training script)
- Checkpoint: saves best `val_loss`

**Inference:** `predict_volume()` in `predict_synthetic_data_3d.py` — Gaussian importance map weighted sliding window, optional test-time mirroring.

---

### 7.2 `loss_functions.py`

| Class / Function | Description |
|---|---|
| `DiceBCELoss(dice_weight, bce_weight, pos_weight)` | `BinaryDiceLoss + BCEWithLogitsLoss`. Default training loss. |
| `DiceFocalLoss(dice_weight, focal_weight, pos_weight)` | `BinaryDiceLoss + FocalLoss`. Good for sparse fiber labels. |
| `BinaryDiceLoss(smooth)` | Soft Dice over batch+spatial dims; fp32 cast inside to avoid bf16 overflow. |
| `DiceLoss(smooth)` | Per-sample Dice. |
| `FocalLoss(alpha, gamma)` | Class-balanced focal loss; `alpha<0.5` penalizes FP, `alpha>0.5` penalizes FN. |
| `soft_dice_cldice(iter_, alpha)` | Combined Dice + centerline-Dice loss (topology-aware). |
| `soft_cldice(iter_, smooth)` | Pure centerline-Dice. |
| `soft_dice(y_true, y_pred)` | Functional Dice loss. |

**Key fix:** `BinaryDiceLoss.forward()` casts to `float32` before sigmoid/sum to prevent fp16 overflow on large 3D volumes. Always use `DiceBCELoss` (which wraps `BinaryDiceLoss`) rather than any float16-naive loss.

---

### 7.3 `datamodules/datasets.py`

**For 2D training:**
```python
class SyntheticDataset(Dataset):
    # Static list of {'image_path', 'label_path'} pairs
    # Per-image Z-score normalization
    
class OnTheFlySyntheticData(IterableDataset):
    # Generates 2D slice patches on-the-fly via cumulative.process_patches_inmemory()
```

**For 3D training (on-the-fly):**
```python
class OnTheFlySyntheticData3D(IterableDataset):
    # Each iteration: calls process_patches_inmemory() → 3D NIfTI volume + mask
    # Applies: 1-99 percentile normalization, image-only augmentations,
    #          thin-slab / empty-patch shape augs
    # Returns: (image_tensor, mask_tensor)
```

**For 3D training (cached patches):**
```python
class SyntheticDataset3D(Dataset):
    # Loads precomputed *_3d.nii.gz + *_3d_mask.nii.gz pairs from disk
    # split: 'all' | 'train' | 'val'  (deterministic split by split_seed)
    # val_fraction: default 0.15
    # Applies: 1-99 percentile norm, thin-slab/empty-patch shape augs (train only)
    # Returns: (image_tensor, mask_tensor)
```

**Normalization (critical):** All three 3D datasets use **1–99 percentile over the full patch**, then clip to `[0,1]`. This matches the inference normalization in `omezarr.py`. Using min-max or `nonzero_percentile` (excludes zeros, shifts scale) causes silent train/inference mismatch.

---

### 7.4 `datamodules/omezarr.py` — OME-Zarr physical-scale loader

```python
@dataclass
class OMEZarrLevelInfo:
    level_index: int
    path: str
    array: zarr.Array
    axis_names: tuple
    shape_zyx: tuple
    voxel_size_um_zyx: tuple
    spatial_axis_indices_zyx: tuple
    spatial_permutation_to_zyx: tuple

class PhysicalScaleOMEZarrDataset(Dataset):
    # Physical field-of-view patch extraction from OME-Zarr
    # patch_size_um: physical extent in micrometers (fixed across pyramid levels)
    # output_shape: fixed tensor output shape (resampled to this from any level)
    
    def __len__() -> int
    def __getitem__(idx) -> dict  # {'image': tensor, 'coords': (z,y,x), 'level': int}

class OMEZarrPatchDataModule(pl.LightningDataModule):
    # Lightning wrapper around PhysicalScaleOMEZarrDataset
```

**Key features:**
- Parses `multiscales/coordinateTransformations` for physical voxel sizes.
- Converts `patch_size_um` to source-level voxel windows via `_align_axis_and_scale()`.
- Slice-based streaming reads (no full-volume load).
- Normalization: 1–99 percentile (matches training; inference uses `--normalize percentile`).

---

### 7.5 `train_on_synthetic_data_3d.py` — Training script

**CLI:**
```bash
python synthetic-training/train_on_synthetic_data_3d.py \
  --on_the_fly \
  --trk_dir /path/to/trk_files \
  --input_nifti /path/to/brain.nii.gz \
  --checkpoint_dir /path/to/checkpoints \
  --no_wandb \
  [--real_proxy_zarr /path/to/volume.zarr]   # optional real-data transfer signal
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--on_the_fly` | — | Use on-the-fly synthesis (vs. `--cached_patches`) |
| `--cached_patches` | — | Load from `--patch_dir` (precomputed) |
| `--batch_size` | 2 | Batch size (3D volumes are large) |
| `--max_epochs` | 200 | Max training epochs |
| `--val_fraction` | 0.15 | Held-out val fraction from cached patches |
| `--split_seed` | 42 | Seed for train/val split shuffle |
| `--real_proxy_zarr` | None | Path to OME-Zarr for real-data transfer metric |
| `--num_workers` | 0 | Workers (on-the-fly is gated off > 0) |
| `--no_wandb` | — | Disable W&B logging |
| `--no_resume` | — | Start fresh (ignore existing checkpoint) |

**`RealLSMProxyCallback`:** Runs the model each val epoch on a fixed set of real OME-Zarr patches. Logs `real_pred_pos_frac` (mean) and `real_pred_pos_frac_median` (robust — watch this in transfer experiments). Calibrated: median = 0.000200 for the good bf16 checkpoint on regions 1–3.

---

### 7.6 `precompute_patches_3d.py` — Offline patch pre-generation

```bash
python synthetic-training/precompute_patches_3d.py \
  --trk_dir /path/to/trks \
  --input_nifti /path/to/brain.nii.gz \
  --output_dir synthetic-training/precomputed_patches \
  --n_patches 500 \
  --voxel_size 0.001 \
  --patch_size 128 128 128 \
  --tissue_threshold 0.0 \
  --enable_cell_blobs \
  --cornucopia_presets ultra_heavy_speckle extreme_noise granular_realistic
```

**Outputs:** `<output_dir>/<trk_stem>/patch_NNNN_3d.nii.gz` + `patch_NNNN_3d_mask.nii.gz`

---

### 7.7 `predict_omezarr_thinslab_3d.py` — Thin-slab inference

Sliding-window inference over an OME-Zarr volume. Loads the volume in `(z_size × tile_y × tile_x)` slabs, runs `FlexibleUNet3D` on each patch with optional Gaussian importance-map weighting, and saves the result as `.npy`.

---

## 8. Root-Level Scripts

### `syntract.py`

Main CLI for single NIfTI+TRK processing. See [CLI Reference](#9-cli-reference) below.

### `cumulative.py`

Batch processor for one NIfTI + many TRKs.

**Python API:**

```python
from cumulative import process_batch, process_patches_inmemory

# Batch: writes patch files to disk
results = process_batch(
    nifti_file="brain.nii.gz",
    trk_directory="./trk_files/",
    output_dir="results",
    patches=30,
    voxel_size=0.05,
    # ... mask parameters use same unified defaults
)

# In-memory: returns arrays for training (no disk I/O)
images, masks = process_patches_inmemory(
    nifti_file="brain.nii.gz",
    trk_file="fibers.trk",
    num_patches=10,
    patch_size=(128, 128, 128),
    voxel_size=0.001,
)
```

**Auto-tuning:** `process_batch` adjusts patch count per TRK file based on streamline count:
- >100k streamlines → more patches
- <10 streamlines → minimal patches

### `thicken_trk.py`

Generates a denser fiber bundle by creating sibling streamlines around each input streamline.

```bash
python thicken_trk.py \
  --input sparse.trk \
  --output dense.trk \
  --copies 5 \
  --radius_um 50 \
  --wave_amplitude_um 20   # add organic micro-curvature
```

At sub-mm patch FOV, a single tractography streamline is nearly straight. `thicken_trk.py` creates realistic curved multi-fiber fields. Use `aligned_wavy.trk` (step ~0.004mm) in patch preview scripts — plain TRKs at step ~0.25mm yield only 1 streamline per 0.064mm patch.

### `visualize_one_patch.py`

Generates one 128³ patch at 0.001mm voxels and saves a 6-panel visualization (sagittal/coronal/axial image + mask slices).

```bash
python visualize_one_patch.py --seed 42 --out patch_preview.png
```

### `calibrate_real_proxy.py`

Runs the real-data proxy metric on a GOOD checkpoint + the OME-Zarr volume. Run on cluster GPU only (128³ forward pass OOMs on laptop CPU). Calibrated result: regions 1–3 mean = 0.000200.

### `test_specific_region.py`

Runs inference on a specific OME-Zarr region and saves the prediction. Used as the baseline for domain-gap experiments. Accepts `--normalize percentile` to match training normalization.

### `batch_ants_trk_registration.py`

Batch ANTs registration for multiple TRK files:
```bash
python batch_ants_trk_registration.py \
  --nifti brain.nii.gz \
  --trk-dir ./trks/ \
  --warp warp.nii.gz \
  --iwarp iwarp.nii.gz \
  --aff affine.mat \
  --output-dir registered_trk/
```

---

## 9. CLI Reference

### `syntract.py`

```
python syntract.py --input NIFTI --trk TRK [options]

Required:
  --input           Input NIfTI file
  --trk             Input TRK file

Synthesis:
  --output          Output base name (default: "output")
  --voxel_size      Target voxel size in mm (default: 0.05)
  --new_dim X Y Z   Target dimensions (auto-calculated if omitted)
  --skip_synthesis  Use input files directly without resampling

ANTs:
  --use_ants        Enable ANTs transformation
  --ants_warp       ANTs warp field (.nii.gz)
  --ants_iwarp      ANTs inverse warp field
  --ants_aff        ANTs affine (.mat)

Patch Processing (default path):
  --total_patches   Number of patches (default: 50)
  --patch_size W H D  Patch dimensions (default: 600 1 600)
  --patch_output_dir  Output directory (default: "patches")
  --min_streamlines_per_patch  Minimum streamlines (default: 20)
  --disable_patch_processing   Use traditional full-volume path

Mask & Bundle:
  --mask_thickness  Base line thickness (default: 1)
  --density_threshold  Fiber density threshold (default: 0.6)
  --min_bundle_size  Minimum bundle size (default: 2000)
  --label_bundles   Color-code individual bundles

White Mask:
  --white_mask      White matter mask NIfTI (optional)

3D Output:
  --3d_output       Generate 3D NIfTI volumes instead of 2D PNGs
```

### `cumulative.py`

```
python cumulative.py --nifti NIFTI --trk-dir DIR [options]

Required:
  --nifti           Input NIfTI file
  --trk-dir         Directory containing .trk files

Optional:
  --output-dir      Output directory (default: "results")
  --patches         Total patches across all TRKs (default: 30)
  --voxel-size      Target voxel size (default: 0.05)
  --3d              Generate 3D output
  --white-mask      White matter mask file
  [same mask/patch parameters as syntract.py]
```

---

## 10. Public Python API

### Core pipeline

```python
# Patch-first extraction (recommended)
from synthesis.patch_first_processing import process_patch_first_extraction

results = process_patch_first_extraction(
    original_nifti_path="brain.nii.gz",
    original_trk_path="fibers.trk",
    target_voxel_size=0.001,
    target_patch_size=(128, 128, 128),
    num_patches=100,
    output_prefix="output/patch",
    min_streamlines_per_patch=2,
)
# results['patches_extracted'] -> int
# results['patch_details']     -> list of {patch_id, bbox, num_streamlines, files}

# Full-volume synthesis (legacy)
from synthesis.main import process_and_save

process_and_save(
    original_nifti_path="brain.nii.gz",
    original_trk_path="fibers.trk",
    target_voxel_size=0.5,
    target_dimensions=(116, 140, 96),
    output_prefix="resampled",
)
```

### In-memory patch generation (for training)

```python
from cumulative import process_patches_inmemory

images, masks = process_patches_inmemory(
    nifti_file="brain.nii.gz",
    trk_file="fibers.trk",
    num_patches=10,
    patch_size=(128, 128, 128),
    voxel_size=0.001,
    tissue_threshold=0.0,
    enable_cell_blobs=True,
    cornucopia_presets=["granular_realistic"],
)
# images: np.ndarray shape (N, 128, 128, 128) float32 in [0,1]
# masks:  np.ndarray shape (N, 128, 128, 128) float32 in [0,1]
```

### 3D volume rendering

```python
from syntract_viewer.volume_renderer import create_3d_volume_with_streamlines

create_3d_volume_with_streamlines(
    nifti_file="patch_0001.nii.gz",
    trk_file="patch_0001.trk",
    output_file="patch_0001_3d.nii.gz",
    soft_mask=True,
    enable_cell_blobs=True,
    use_cornucopia_3d=True,
    cornucopia_allowed_presets=["granular_realistic"],
)
# Saves: patch_0001_3d.nii.gz + patch_0001_3d_mask.nii.gz
```

### Batch processing

```python
from cumulative import process_batch

results = process_batch(
    nifti_file="brain.nii.gz",
    trk_directory="./trk_files/",
    patches=50,
    voxel_size=0.05,
    threed_output=False,
)
# results['processed'] -> int (number of TRK files processed)
# results['total_patches'] -> int
```

---

## 11. Configuration & Defaults

### Mask defaults (unified across all paths)

```python
mask_thickness = 1          # base line width; auto-scaled by output image size
density_threshold = 0.6     # very aggressive: only high-density fiber regions
min_bundle_size = 2000      # only large, prominent bundles
use_high_density_masks = True
```

### Normalization

```python
# Training, OME-Zarr inference, test scripts all use:
p1, p99 = np.percentile(volume, [1, 99])
volume = np.clip((volume - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
```

**Never use:** min-max normalization or `nonzero_percentile` — both produce train/inference mismatch.

### GPU / precision defaults

```python
# Training precision
precision = "bf16-mixed"   # mandatory; fp16 overflows in 3D forward pass
gradient_clip_val = 1.0

# GPU check (synthesis)
gpu_result = try_gpu_import()  # graceful: falls back to CPU
```

### Patch sampling (fine-resolution)

```python
voxel_size = 0.001          # 1 µm — matches light-sheet inference
patch_size = (128, 128, 128)
min_streamlines_per_patch = 2
tissue_threshold = 0.0      # render fibers in ALL voxels at this scale
```

---

## 12. Testing

### Run tests

```bash
# Full suite
pytest

# Targeted (fast)
pytest tests/test_nifti_preprocessing.py
pytest tests/test_transform.py
pytest tests/test_streamline_processing.py
pytest tests/test_densify.py
pytest tests/test_syntract_viewer_complete.py

# Custom integration runner
python run_comprehensive_tests.py
```

### Test organization

| File | Tests |
|---|---|
| `test_nifti_preprocessing.py` | `resample_nifti`, `resample_nifti_patch`, memory estimation |
| `test_streamline_processing.py` | `clip_streamline_to_fov`, `transform_streamline`, `transform_and_densify_streamlines` |
| `test_densify.py` | Linear / hermite interpolation, edge cases, metrics |
| `test_transform.py` | `build_new_affine` — isotropic, anisotropic, patch_center |
| `test_main.py` | `process_and_save` integration (requires real data) |
| `test_comprehensive_integration.py` | End-to-end pipeline with synthetic test files |
| `test_synthesis_complete.py` | Full synthesis module import + basic function tests |
| `test_syntract_viewer_complete.py` | Core, generation, effects, utils, masking |

### Sanity checks (cluster)

```bash
# Confirm model reproduces synthetic mask (dice ≈ 0.98 → inference path OK)
python synthetic-training/sanity_check_synthetic.py \
  --checkpoint best_3d.ckpt --voxel_size 0.001

# Test thin-slab path
python synthetic-training/sanity_check_thinslab.py \
  --checkpoint best_3d.ckpt
```

---

## 13. Dependency Map

### Core (required)

| Package | Use |
|---|---|
| `nibabel` | NIfTI + TRK I/O |
| `numpy` | All array operations |
| `scipy` | ndimage (zoom, gaussian_filter), RBF interpolation |
| `dipy` | `transform_streamlines` |
| `joblib` | Parallel processing |
| `matplotlib` | Visualization (always use `Agg` backend in batch) |
| `tqdm` | Progress bars |

### Training

| Package | Use |
|---|---|
| `torch` | Neural network, `grid_sample` for patch resampling |
| `pytorch_lightning` | Training loop, checkpointing, callbacks |
| `zarr` | OME-Zarr volume reading |
| `albumentations` | 2D training augmentations |
| `transformers` | Cosine schedule with warmup |

### Optional GPU

| Package | Use |
|---|---|
| `cupy` | GPU array operations in synthesis |
| `numba` | CUDA kernel for full-volume resampling |

### Import pattern (dual-mode)

All modules support both package import and standalone script import:

```python
try:
    from .module import function
except ImportError:
    from module import function
```

Do not break this pattern when adding new modules — tests and scripts use both modes.

---

## 14. Glossary

| Term | Definition |
|---|---|
| **NIfTI** | Neuroimaging Informatics Technology Initiative format (`.nii`, `.nii.gz`) |
| **TRK** | TrackVis tractography format (`.trk`) storing streamline paths in RAS mm |
| **Streamline** | Ordered list of 3D points tracing a white-matter fiber tract |
| **Patch-first** | Processing paradigm: extract small spatial patches directly from original data, skipping full-volume resampling |
| **RAS** | Right-Anterior-Superior coordinate convention (NIfTI standard) |
| **Affine** | 4×4 matrix mapping voxel indices to physical (RAS mm) coordinates |
| **ANTs** | Advanced Normalization Tools; used for non-linear brain registration |
| **Cornucopia** | Augmentation library simulating optical/MRI imaging artifacts |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalization |
| **OME-Zarr** | Open Microscopy Environment Zarr format; multiscale microscopy volumes |
| **Thin-slab** | Sparse data format: real LSM data consists of thin Z-slabs with most voxels zero; the model must handle this during inference |
| **Soft mask** | Fiber mask with fractional coverage values in `[0,1]` (vs. binary); avoids stair-stepping of diagonal fibers |
| **Domain gap** | Statistical difference between synthetic training data and real LSM inference data |
| **real_pred_pos_frac_median** | Median positive fraction across real OME-Zarr regions; robust transfer signal (one hot region can't inflate it) |
| **bf16-mixed** | bfloat16 mixed precision; mandatory for 3D training (fp16 overflows on large 3D volumes) |
| **clDice** | Centerline-Dice: topology-aware loss that penalizes breaks in fiber centerlines |
| **val_dice** | Validation Dice score on the synthetic held-out split (should be ≥0.85) |
| **real_proxy** | `RealLSMProxyCallback` — unlabeled real-data evaluation run each epoch |
