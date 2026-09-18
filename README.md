# SynTract

Synthetic training data and a 3D U-Net for segmenting tracer-labelled fibers in
light-sheet microscopy (LSM) volumes. No manual voxel labels are needed: fibers
are synthesised by rendering MRI tractography streamlines into blockface-style
tissue, with realistic microscopy noise, cell-body distractors, and soft
partial-volume masks, then a 3D U-Net is trained on those patches and applied to
real OME-Zarr volumes with sliding-window inference.

```
registered TRK + blockface NIfTI
        │  preprocessing/   patch-first extraction (streamline-anchored, physical-scale FOV)
        ▼
128³ tissue patch + clipped streamlines
        │  rendering/       3D fiber rendering, soft masks, noise / artifact augmentation
        ▼
(volume, mask) training pairs  ── precompute to disk or generate on the fly
        │  training/        PyTorch Lightning 3D U-Net (bf16, Dice+BCE / clDice)
        ▼
checkpoint  ──►  sliding_window_inference.py / predict_real_region.py on OME-Zarr
```

## Repository layout

| Path | Purpose |
|---|---|
| `preprocessing/` | NIfTI + TRK resampling, ANTs transforms, patch-first patch extraction (`patch_extraction.py`), full-volume path (`full_volume.py`) |
| `rendering/` | 3D volume rendering with streamlines (`volume_renderer.py`), noise and artifact augmentations, 2D slice rendering and masks |
| `training/` | 3D U-Net (`unet3d.py`), losses, datasets and datamodules, `train_3d.py`, `precompute_patches_3d.py`, inference and sanity-check scripts, SLURM examples |
| `syntract.py` | Single NIfTI + TRK pipeline CLI (2D/3D patches and visualisations) |
| `batch_processing.py` | Batch over many TRK files; also the in-memory `process_patches_inmemory` API used by training |
| `sliding_window_inference.py` | Tile a checkpoint over a large volume or OME-Zarr region |
| `predict_real_region.py` | Run a checkpoint on patches around chosen coordinates of a real OME-Zarr volume |
| `thicken_trk.py` | Turn a sparse TRK into a dense, gently curved bundle for sub-micron synthesis |
| `batch_ants_trk_registration.py` | Apply ANTs warps to a directory of TRK files |
| `fiber_extract_3d.py` | Classical (no learning) fiber extraction baseline |
| `tests/` | pytest suite |
| `docs/` | API reference (`DOCUMENTATION.md`), design notes and ADRs |

## Installation

Python 3.10 or newer. A CUDA GPU is required for training and for any 128³
inference; the preprocessing and rendering code falls back to CPU automatically.

```bash
git clone https://github.com/Sparsh57/Syntract-3D.git
cd Syntract-3D
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # exact pins used for the paper experiments
# optional GPU acceleration of preprocessing/rendering (CUDA 12):
pip install cupy-cuda12x numba
```

`pip install -e .` installs the packages plus the `syntract`, `syntract-batch`
and `syntract-full-volume` console scripts.

## Data

The repository does not ship data. You need:

- a reference **blockface NIfTI** volume (`.nii.gz`),
- **TRK streamlines** registered to that volume (see `batch_ants_trk_registration.py`
  for applying ANTs warps, and `thicken_trk.py` for densifying a sparse bundle),
- optionally a **white-matter mask** NIfTI,
- for inference, an **OME-Zarr** light-sheet volume with a multiscale pyramid.

## Quick start: the 3D pipeline

### 1. Densify the tractography

At 1 µm voxels a single tractography streamline is straight and sparse. Build a
dense, wavy bundle once:

```bash
python thicken_trk.py --input registered_trk/aligned.trk --output registered_trk/dense/aligned_dense.trk \
    --copies 50 --radius_mm 0.025 --seed 42
# curvature only, no thickening:
python thicken_trk.py --input in.trk --output out.trk --copies 1 --wave_amplitude_um 5
```

### 2. Preview one training patch

```bash
python visualize_one_patch.py --nifti brain.nii.gz --trk registered_trk/dense/aligned_dense.trk --seed 42
```

writes `patch_preview.png` with sagittal / coronal / axial image slices and the
matching mask.

### 3. Precompute patches (recommended)

On-the-fly synthesis at 1 µm starves the GPU, so render the training set once:

```bash
python training/precompute_patches_3d.py \
    --trk_dir registered_trk/dense --input_nifti brain.nii.gz \
    --output_dir training/precomputed_patches --patches_per_trk 1800 \
    --patch_size 128 128 128 --voxel_size 0.001 --min_streamlines_per_patch 2 \
    --use_cornucopia_3d --cornucopia_presets ultra_heavy_speckle extreme_noise granular_realistic \
    --tissue_threshold 0.0 --enable_cell_blobs --cell_blob_count 200
find training/precomputed_patches -name '*_3d.nii.gz' | wc -l   # confirm a healthy count
```

`training/precompute_patches.sh` is the full SLURM job with every render knob
used for the paper.

### 4. Train

```bash
python training/train_3d.py --cached_patches --patch_dir training/precomputed_patches \
    --trk_dir registered_trk/dense --input_nifti brain.nii.gz \
    --checkpoint_dir checkpoints/ --epochs 150 --batch_size 4 --num_workers 12 \
    --patch_size 128 128 128 --voxel_size 0.001 --loss BCE --pos_weight 5.0 --no_wandb
```

Notes:

- `--on_the_fly` (the default) generates patches during training instead; every
  render flag of `precompute_patches_3d.py` is also accepted here.
- Precision is auto-selected: `bf16-mixed` on Ampere/Hopper GPUs. Do not force
  `16-mixed`; fp16 overflows in the forward pass and NaN-poisons the weights.
- `--val_fraction 0.15` holds out a disjoint synthetic validation split.
- `--real_proxy_zarr /path/to/volume.ome.zarr` logs an unlabelled transfer proxy
  (predicted-positive fraction and fiber continuity on fixed real regions) each
  validation epoch.
- Multi-GPU: `training/train_cached.sh` and `training/train_multigpu.sh` are
  `torchrun` SLURM examples; edit the partition lines for your cluster.

### 5. Sanity-check a checkpoint

```bash
# dice ≈ 0.98 on a known synthetic patch means the inference path is correct
python training/sanity_check_synthetic.py --checkpoint checkpoints/best_3d.ckpt --voxel_size 0.001
# thin-slab (zero-padded Z) sliding-window path
python training/sanity_check_thinslab.py --checkpoint checkpoints/best_3d.ckpt
```

### 6. Inference on real light-sheet data

```bash
# a region of an OME-Zarr, Gaussian-blended sliding window, outputs .npy (+ optional NIfTI)
python sliding_window_inference.py --zarr /path/to/volume.ome.zarr \
    --region_center_zyx 200 13000 16400 --region_size_zyx 256 512 512 \
    --checkpoint checkpoints/best_3d.ckpt --output_prefix results/region1 --stride 64 --save_nifti

# patches around hand-picked coordinates, with debug PNGs
python predict_real_region.py --zarr_path /path/to/volume.ome.zarr \
    --model_checkpoint checkpoints/best_3d.ckpt --center_coords 19 12000 20000 \
    --patch_size 128 128 128 --normalize percentile --output_dir results/region_a
```

`explore_zarr_pick_region.py` shows a Z-MIP of a coarse pyramid level so you can
click a region centre; `view_sliding_results.py` and `view_neuroglancer.py`
browse the (memory-mapped) outputs. `compare_multiregion.sh` sweeps a fixed 3×3
grid of regions to tell a universal domain gap from a location-specific one, and
`training/compare_domain_stats.py` quantifies synthetic-vs-real intensity statistics.

## Design decisions that matter for reproducibility

- **Train and inference normalisation are identical**: 1–99 percentile of the
  full patch, mapped to [0, 1]. Never replace this with min-max.
- **Patch FOV is anchored in physical space** (`patch_size × voxel_size`), and
  patch centres are sampled on streamline points, so sub-millimetre patches
  still contain fibers.
- **Masks are soft** (trilinear partial-volume weights) by default; a binary
  tube is available with `--mask_smoothing_sigma 1.0 --mask_binary_threshold 0.2`.
- **Cell-body blobs** are added to the image only, never the mask, so the model
  learns fiber-versus-cell.
- **Inference-shape augmentation** (random thin Z slabs and empty patches)
  matches the zero-padded thin-slab inputs seen at inference.

The exact configuration behind the reported model is
`training/precompute_patches.sh` followed by `training/train_cached.sh`.

## 2D synthetic visualisations

The original 2D pipeline is still available for generating dark-field style
slice images with masks:

```bash
python syntract.py --input brain.nii.gz --trk fibers.trk --output result           # patch-first, default
python syntract.py ... --use_ants --ants_warp warp.nii.gz --ants_iwarp iwarp.nii.gz --ants_aff affine.mat
python syntract.py ... --3d_output --white_mask wm_mask.nii.gz --total_patches 10 --patch_size 1024 40 1024
python syntract.py ... --disable_patch_processing --new_dim 116 140 96             # full-volume path (slow)
python batch_processing.py --nifti brain.nii.gz --trk-dir ./trk_files/ --total-patches 50
```

Key defaults (`syntract.py`; `batch_processing.py` uses the same values with
dashes): `--voxel_size 0.05`, `--total_patches 50`, `--patch_size 600 1 600`,
`--min_streamlines_per_patch 20`, `--mask_thickness 1`,
`--density_threshold 0.6`, `--min_bundle_size 2000`, high-density masks on.
Outputs are `patches/{prefix}_{NNNN}.nii.gz` / `.trk`, `{viz_prefix}_{n}_*.png`
and `*_mask_slice{n}.png`. Full parameter reference: `docs/DOCUMENTATION.md`.

Python API:

```python
from syntract import process_syntract
from batch_processing import process_batch, process_patches_inmemory

result = process_syntract(input_nifti="brain.nii.gz", input_trk="fibers.trk",
                          output_base="out", new_dim=None, voxel_size=0.05)
images, masks = process_patches_inmemory(input_nifti="brain.nii.gz", trk_file="fibers.trk",
                                         num_patches=8, patch_size=[512, 1, 512], random_state=42)
```

## Classical baseline

`fiber_extract_3d.py` extracts fibers without learning (denoise, structure-tensor
lineness, threshold, component filtering) for comparison:

```bash
python fiber_extract_3d.py --input region.nii.gz --out_dir fiber3d_out --voxel_size 1.16 1.16 1.0
```

## Tests

```bash
pip install pytest
pytest                          # unit suite (tests needing real data skip when it is absent)
python run_comprehensive_tests.py
```

## Citation

See `CITATION.cff`. A paper reference will be added on publication.

## License

MIT, see `LICENSE`.
