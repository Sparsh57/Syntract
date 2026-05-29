# synthetic-training
ML Pipeline for segmentation using synthetic data. 

## Scale-aware OME-Zarr testing loader

Use `datamodules/omezarr.py` when the source data is OME-Zarr (multi-scale) and
you want patch extraction to respect physical voxel size.

### Why this matters

A fixed tensor shape (for example `128 x 128 x 128`) does not guarantee a fixed
physical field-of-view. OME-Zarr levels can have very different voxel spacings,
so the dataloader should convert:

- physical patch size (um) -> source voxel window (per level)
- source voxel window -> fixed output tensor shape

### Quick usage

```python
from datamodules.omezarr import PhysicalScaleOMEZarrDataset
from torch.utils.data import DataLoader

dataset = PhysicalScaleOMEZarrDataset(
    zarr_path="/path/to/data.ome.zarr",
    output_patch_size=(128, 128, 128),      # model input shape
    target_voxel_size_um=(500.0, 500.0, 500.0),  # 0.5 mm
    level_sampling="closest",               # nearest pyramid level to target spacing
    samples_per_epoch=256,
    allow_padding=True,                     # important if requested physical window exceeds extent
    return_metadata=True,
)

print(dataset.describe_levels())            # inspect parsed levels and voxel sizes

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
images, meta = next(iter(loader))
print(images.shape)  # (B, 1, 128, 128, 128)
```

### Patch testing script (no training)

```bash
python test_omezarr_patches.py \
  --zarr_path /path/to/data.ome.zarr \
  --num_patches 8 \
  --patch_size 128 128 128 \
  --target_voxel_size_um 500 500 500 \
  --fixed_level 0 \
  --save_dir ./omezarr_test_outputs
```

Optional: add `--model_checkpoint /path/to/last.ckpt` for patch-level inference sanity checks.
Use `--fixed_level` to force a specific pyramid level (0 = highest resolution).

W&B is supported:

```bash
python test_omezarr_patches.py \
  --zarr_path /path/to/data.ome.zarr \
  --model_checkpoint /path/to/last.ckpt \
  --wandb_project syntract3d \
  --wandb_run_name omezarr_test_run \
  --wandb_online
```

Use `--no_wandb` to disable logging. Default mode is offline (`--wandb_offline`).

## Synthetic realism augmentations

The 3D synthetic training path now applies image-only realism augmentations to the
training images by default:

- `--enable_tissue_artifacts` / `--disable_tissue_artifacts`
- `--enable_granular_noise` / `--disable_granular_noise`
- `--enable_speckle_noise` / `--disable_speckle_noise`
- `--artifact_strength`
- `--granular_noise_strength`
- `--speckle_noise_strength`
- `--speckle_noise_density`
- `--speckle_noise_sigma`
- `--fiber_intensity_min` / `--fiber_intensity_max`
- `--fiber_brightness_variation`
- `--fiber_segment_brightness_variation`
- `--fiber_density_gamma`
- `--fiber_min_visibility`
- `--fiber_target_intensity`
- `--fiber_max_boost` / `--fiber_opacity`
- `--fiber_smoothing_sigma`
- `--fiber_antialias`
- `--max_streamlines_rendered`

These perturb only the rendered image volume. Sparse speckle dots are deliberately
not added to the mask, so they behave as bright non-fiber examples. Fiber
brightness variation is also image-only: the mask continues to come from the
same streamline geometry.

Training defaults in `train_on_synthetic_data_3d.py` are set for the current
realistic 128^3 on-the-fly setup:

- `--on_the_fly`
- `--batch_size 4`
- `--patch_size 128 128 128`
- `--voxel_size 0.05`
- `--streamline_margin_fraction 0.10`
- `--batch_group_factor 10`
- `--num_workers 0`
- `--accumulate_grad_batches 1`

Mask generation keeps the original defaults: `--mask_smoothing_sigma 2.0` and
`--mask_binary_threshold 0.01`.

`precompute_patches_3d.py` exposes the same flags, but keeps baked-in artefacts
off by default so cached patches can remain a clean base for training-time
augmentation.

Preview clean vs. augmented examples before training:

```bash
python preview_realistic_augmentations_3d.py \
  --trk_dir /path/to/trks \
  --input_nifti /path/to/brain.nii.gz \
  --output_dir ./augmentation_preview \
  --num_patches 3 \
  --patch_size 128 128 128
```

### Notes

- `level_sampling="closest"` is the safest default for inference/training at a target spacing.
- Use `level_sampling="random"` or `"weighted_random"` for multiscale augmentation.
- Reads are streamed from Zarr slices; the full volume is never loaded into memory.
- Returned metadata includes `source_coverage_fraction_zyx` and warning messages when
  requested physical FOV is larger than source extent (your 1 um vs 0.5 mm mismatch case).
