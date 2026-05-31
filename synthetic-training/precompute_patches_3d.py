"""
Pre-generate 3D synthetic patches to avoid on-the-fly rendering during training.

Usage example:
    python precompute_patches_3d.py \
        --trk_dir ../registered_trk \
        --input_nifti ../sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz \
        --output_dir ./precomputed_patches \
        --patch_size 128 128 128 \
        --voxel_size 0.05 \
        --patches_per_trk 50

Then train with:
    python train_on_synthetic_data_3d.py \
        --patch_dir ./precomputed_patches \
        --checkpoint_dir checkpoints/ \
        --epochs 150 \
        --batch_size 2 \
        --num_workers 4
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cumulative import process_patches_inmemory  # type: ignore


def get_create_3d_fn():
    try:
        from syntract_viewer.volume_renderer import create_3d_volume_with_streamlines
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from syntract_viewer.volume_renderer import create_3d_volume_with_streamlines
    return create_3d_volume_with_streamlines


def render_trk(
    trk_path: Path,
    input_nifti: str,
    white_mask: str,
    output_dir: Path,
    patch_size,
    voxel_size: float,
    min_streamlines_per_patch: int,
    min_bundle_size: int,
    streamline_margin_fraction: float,
    patches_per_trk: int,
    use_cornucopia_3d: bool,
    cornucopia_allowed_presets,
    tissue_threshold: float,
    enable_cell_blobs: bool,
    cell_blob_count: int,
    cell_blob_intensity: float,
    cell_blob_radius_range,
    enable_tissue_artifacts: bool,
    enable_granular_noise: bool,
    enable_speckle_noise: bool,
    enable_dash_noise: bool,
    enable_horizontal_banding: bool,
    granular_noise_strength: float,
    artifact_strength: float,
    speckle_noise_strength: float,
    speckle_noise_density: float,
    speckle_noise_sigma: float,
    speckle_square_size: int,
    dash_noise_strength: float,
    dash_noise_density: float,
    dash_length_sigma: float,
    dash_cross_sigma: float,
    banding_strength: float,
    banding_axis: int,
    fiber_intensity_min: float,
    fiber_intensity_max: float,
    fiber_max_boost: float,
    fiber_opacity: float,
    fiber_smoothing_sigma: float,
    fiber_antialias: bool,
    min_streamlines_rendered: int,
    fiber_brightness_variation: float,
    fiber_segment_brightness_variation: float,
    fiber_render_mode: str,
    fiber_density_gamma: float,
    fiber_min_visibility: float,
    fiber_target_intensity: float,
    background_max_intensity: float,
    mask_smoothing_sigma: float,
    mask_binary_threshold: float,
    soft_mask: bool,
    use_gpu: bool,
    patch_use_gpu: bool,
    new_dim,
    temp_dir_base: str,
):
    target_dir = output_dir / trk_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    # The "None" string sentinel (argparse default) must become a real None,
    # otherwise process_patches_inmemory treats it as a (missing) file path.
    white_mask_arg = None if (white_mask is None or str(white_mask).lower() == "none") else white_mask

    process_patches_inmemory(
        input_nifti=input_nifti,
        trk_file=str(trk_path),
        num_patches=patches_per_trk,
        patch_size=patch_size,
        min_streamlines_per_patch=min_streamlines_per_patch,
        min_bundle_size=min_bundle_size,
        voxel_size=voxel_size,
        new_dim=new_dim,
        white_mask_file=white_mask_arg,
        patches_output_dir=str(target_dir),
        skip_2d_viz=True,
        temp_dir_base=temp_dir_base,
        patch_use_gpu=patch_use_gpu,
        streamline_margin_fraction=streamline_margin_fraction,
    )

    # Match BOTH .nii and .nii.gz: with skip_2d_viz=True the extractor writes
    # uncompressed .nii (use_compressed_nifti=not skip_2d_viz in cumulative.py),
    # so a .nii.gz-only filter silently finds nothing and never renders. Pair
    # each volume with its sibling .trk by stripping the matching extension
    # (mirrors OnTheFlySyntheticData3D._iter_patch_pairs).
    nii_files = sorted(
        [
            f
            for f in os.listdir(target_dir)
            if (f.endswith(".nii.gz") or f.endswith(".nii"))
            and "_3d" not in f and "_mask" not in f and "_white" not in f
        ]
    )
    create_3d = get_create_3d_fn()

    for nii_f in nii_files:
        nii_ext = ".nii.gz" if nii_f.endswith(".nii.gz") else ".nii"
        patch_base = nii_f[: -len(nii_ext)]
        trk_f = patch_base + ".trk"
        nii_path = target_dir / nii_f
        trk_path_out = target_dir / trk_f
        if not trk_path_out.exists():
            continue
        # Always emit compressed _3d outputs so the cached training loader
        # (SyntheticDataset3D globs *_3d.nii.gz) finds them regardless of whether
        # the raw extracted patch was .nii or .nii.gz.
        out_path = target_dir / f"{patch_base}_3d.nii.gz"
        mask_path = target_dir / f"{patch_base}_3d_mask.nii.gz"
        if out_path.exists() and mask_path.exists():
            continue

        create_3d(
            nifti_file=str(nii_path),
            trk_file=str(trk_path_out),
            output_file=str(out_path),
            white_mask_path=white_mask_arg,
            save_mask=True,
            use_cornucopia_3d=use_cornucopia_3d,
            cornucopia_allowed_presets=cornucopia_allowed_presets,
            tissue_threshold=tissue_threshold,
            enable_cell_blobs=enable_cell_blobs,
            cell_blob_count=cell_blob_count,
            cell_blob_intensity=cell_blob_intensity,
            cell_blob_radius_range=cell_blob_radius_range,
            fiber_intensity_min=fiber_intensity_min,
            fiber_intensity_max=fiber_intensity_max,
            fiber_max_boost=fiber_max_boost,
            fiber_opacity=fiber_opacity,
            fiber_smoothing_sigma=fiber_smoothing_sigma,
            fiber_antialias=fiber_antialias,
            min_streamlines_rendered=min_streamlines_rendered,
            fiber_brightness_variation=fiber_brightness_variation,
            fiber_segment_brightness_variation=fiber_segment_brightness_variation,
            fiber_render_mode=fiber_render_mode,
            fiber_density_gamma=fiber_density_gamma,
            fiber_min_visibility=fiber_min_visibility,
            fiber_target_intensity=fiber_target_intensity,
            background_max_intensity=background_max_intensity,
            enable_tissue_artifacts=enable_tissue_artifacts,
            enable_granular_noise=enable_granular_noise,
            enable_speckle_noise=enable_speckle_noise,
            enable_dash_noise=enable_dash_noise,
            enable_horizontal_banding=enable_horizontal_banding,
            artifact_strength=artifact_strength,
            granular_noise_strength=granular_noise_strength,
            speckle_noise_strength=speckle_noise_strength,
            speckle_noise_density=speckle_noise_density,
            speckle_noise_sigma=speckle_noise_sigma,
            speckle_square_size=speckle_square_size,
            dash_noise_strength=dash_noise_strength,
            dash_noise_density=dash_noise_density,
            dash_length_sigma=dash_length_sigma,
            dash_cross_sigma=dash_cross_sigma,
            banding_strength=banding_strength,
            banding_axis=banding_axis,
            mask_smoothing_sigma=mask_smoothing_sigma,
            mask_binary_threshold=mask_binary_threshold,
            soft_mask=soft_mask,
            use_gpu=use_gpu,
        )


def main():
    parser = argparse.ArgumentParser("Precompute 3D patches for cached training")
    parser.add_argument("--trk_dir", required=True, type=str, help="Directory with .trk files")
    parser.add_argument("--input_nifti", required=True, type=str, help="Input NIfTI volume")
    parser.add_argument("--output_dir", required=True, type=str, help="Where to store precomputed patches")
    parser.add_argument("--white_mask", default="None", type=str, help='White matter mask path or "None"')
    parser.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128], help="D H W (voxels)")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Target voxel size in mm")
    parser.add_argument("--min_streamlines_per_patch", type=int, default=5)
    parser.add_argument("--min_bundle_size", type=int, default=5)
    parser.add_argument("--streamline_margin_fraction", type=float, default=0.15,
                        help="Require sampled patches to contain streamlines away from patch borders (default: 0.15)")
    parser.add_argument("--patches_per_trk", type=int, default=50, help="How many patches to render per .trk file")
    parser.add_argument("--use_cornucopia_3d", action="store_true", help="Enable cornucopia 3D rendering")
    parser.add_argument("--cornucopia_presets", nargs="+", default=None,
                        help="Restrict cornucopia to these presets (e.g. ultra_heavy_speckle extreme_noise granular_realistic)")
    parser.add_argument("--tissue_threshold", type=float, default=2.0,
                        help="Voxels below this are treated as background and skip fiber rendering. "
                             "At sub-micron voxels use 0.0 (see CLAUDE.md fine-resolution notes)")
    parser.add_argument("--enable_cell_blobs", dest="enable_cell_blobs", action="store_true", default=False,
                        help="Scatter Gaussian cell-body blobs into the image only (never the mask)")
    parser.add_argument("--cell_blob_count", type=int, default=60)
    parser.add_argument("--cell_blob_intensity", type=float, default=0.3)
    parser.add_argument("--cell_blob_radius_min", type=float, default=1.5)
    parser.add_argument("--cell_blob_radius_max", type=float, default=4.0)
    parser.add_argument("--patch_use_gpu", dest="patch_use_gpu", action="store_true", default=True,
                        help="Use GPU for patch resampling/extraction (default on)")
    parser.add_argument("--patch_on_cpu", dest="patch_use_gpu", action="store_false",
                        help="Force CPU patch extraction")
    parser.add_argument("--soft_mask", dest="soft_mask", action="store_true", default=False,
                        help="Keep fractional sub-voxel mask coverage (recommended at fine voxel sizes)")
    parser.add_argument("--enable_dash_noise", dest="enable_dash_noise", action="store_true", default=False)
    parser.add_argument("--disable_dash_noise", dest="enable_dash_noise", action="store_false")
    parser.add_argument("--enable_horizontal_banding", dest="enable_horizontal_banding", action="store_true", default=False)
    parser.add_argument("--disable_horizontal_banding", dest="enable_horizontal_banding", action="store_false")
    parser.add_argument("--speckle_square_size", type=int, default=2)
    parser.add_argument("--dash_noise_strength", type=float, default=0.55)
    parser.add_argument("--dash_noise_density", type=float, default=0.0005)
    parser.add_argument("--dash_length_sigma", type=float, default=4.0)
    parser.add_argument("--dash_cross_sigma", type=float, default=0.3)
    parser.add_argument("--banding_strength", type=float, default=0.18)
    parser.add_argument("--banding_axis", type=int, default=1)
    parser.add_argument("--enable_tissue_artifacts", dest="enable_tissue_artifacts", action="store_true", default=False,
                        help="Bake image-only tissue-like artefacts into precomputed images")
    parser.add_argument("--disable_tissue_artifacts", dest="enable_tissue_artifacts", action="store_false",
                        help="Disable tissue-like artefacts")
    parser.add_argument("--enable_granular_noise", dest="enable_granular_noise", action="store_true", default=False,
                        help="Bake image-only fine Cornucopia granular noise into precomputed images")
    parser.add_argument("--disable_granular_noise", dest="enable_granular_noise", action="store_false",
                        help="Disable fine granular noise")
    parser.add_argument("--enable_speckle_noise", dest="enable_speckle_noise", action="store_true", default=False,
                        help="Bake sparse image-only whitish-grey dot artefacts into precomputed images")
    parser.add_argument("--disable_speckle_noise", dest="enable_speckle_noise", action="store_false",
                        help="Disable sparse dot artefacts")
    parser.add_argument("--granular_noise_strength", type=float, default=0.35,
                        help="Strength of fine granular noise (default: 0.35)")
    parser.add_argument("--artifact_strength", type=float, default=0.45,
                        help="Strength of tissue-like artefacts (default: 0.45)")
    parser.add_argument("--speckle_noise_strength", type=float, default=0.35,
                        help="Brightness strength of sparse dot artefacts (default: 0.35)")
    parser.add_argument("--speckle_noise_density", type=float, default=0.0012,
                        help="Probability of a dot seed per tissue voxel (default: 0.0012)")
    parser.add_argument("--speckle_noise_sigma", type=float, default=0.35,
                        help="Gaussian size of dot artefacts; below 1 keeps dots small (default: 0.35)")
    parser.add_argument("--fiber_intensity_min", type=float, default=5.0,
                        help="Minimum added fiber intensity. Lower blends fibers into tissue (default: 5.0)")
    parser.add_argument("--fiber_intensity_max", type=float, default=10.0,
                        help="Maximum added fiber intensity. Lower blends fibers into tissue (default: 10.0)")
    parser.add_argument("--fiber_max_boost", type=float, default=5.0,
                        help="Cap added fiber brightness above local tissue; use negative to disable (default: 5.0)")
    parser.add_argument("--fiber_opacity", type=float, default=0.72,
                        help="Multiplier on added fiber brightness after capping (default: 0.72)")
    parser.add_argument("--fiber_smoothing_sigma", type=float, default=0.35,
                        help="Smooth only the rendered fiber boost to reduce pixel stair-steps (default: 0.35)")
    parser.add_argument("--fiber_antialias", action="store_true",
                        help="Use CPU subvoxel antialias rendering for less blocky streamlines")
    parser.add_argument("--min_streamlines_rendered", type=int, default=20,
                        help="Skip patch if fewer than this many streamlines are present; 0 disables check")
    parser.add_argument("--fiber_brightness_variation", type=float, default=0.35,
                        help="Per-streamline brightness variation; image only, mask unchanged (default: 0.35)")
    parser.add_argument("--fiber_segment_brightness_variation", type=float, default=0.15,
                        help="Per-segment brightness jitter along each streamline (default: 0.15)")
    parser.add_argument("--fiber_render_mode", default="embedded", choices=["additive", "density", "embedded"],
                        help="Fiber rendering mode. embedded blends thin fiber texture into tissue")
    parser.add_argument("--fiber_density_gamma", type=float, default=2.2,
                        help="Density-mode gamma. Higher values make bundles thinner (default: 2.2)")
    parser.add_argument("--fiber_min_visibility", type=float, default=0.15,
                        help="Minimum normalized visibility for true fiber voxels (default: 0.15)")
    parser.add_argument("--fiber_target_intensity", type=float, default=10.0,
                        help="Embedded mode fiber brightness boost, or density-mode target level (default: 10.0)")
    parser.add_argument("--background_max_intensity", type=float, default=30.0,
                        help="Cap tissue/background before fiber composition; negative disables (default: 30.0)")
    parser.add_argument("--mask_smoothing_sigma", type=float, default=2.0,
                        help="3D mask smoothing sigma. Lower makes masks thinner (default: 2.0)")
    parser.add_argument("--mask_binary_threshold", type=float, default=0.01,
                        help="Normalized mask threshold. Higher makes masks thinner (default: 0.01)")
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU rendering even if CUDA is available")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trk_paths = sorted(Path(args.trk_dir).glob("*.trk"))
    if len(trk_paths) == 0:
        raise ValueError(f"No .trk files found in {args.trk_dir}")

    use_gpu = not args.no_gpu
    fast_tmp = "/dev/shm" if os.path.exists("/dev/shm") else None

    # Match the on-the-fly path: derive target dimensions from physical size and
    # voxel_size so patch FOV anchoring is identical (see datasets.py
    # OnTheFlySyntheticData3D._compute_target_dimensions).
    new_dim = None
    try:
        import nibabel as nib
        import numpy as _np
        nii = nib.load(args.input_nifti, mmap=True)
        shape = _np.array(nii.shape[:3], dtype=_np.float64)
        zooms = _np.array(nii.header.get_zooms()[:3], dtype=_np.float64)
        td = _np.round((shape * zooms) / float(args.voxel_size)).astype(int)
        td = _np.maximum(td, 32)
        new_dim = tuple(int(v) for v in td)
        print(f"Target dimensions for voxel_size={args.voxel_size}: {new_dim}")
    except Exception as exc:
        print(f"Could not precompute target dimensions ({exc}); falling back to pipeline default.")

    cell_blob_radius_range = (args.cell_blob_radius_min, args.cell_blob_radius_max)

    for idx, trk_path in enumerate(trk_paths, start=1):
        print(f"[{idx}/{len(trk_paths)}] Processing {trk_path.name}", flush=True)
        render_trk(
            trk_path=trk_path,
            input_nifti=args.input_nifti,
            white_mask=args.white_mask,
            output_dir=output_dir,
            patch_size=args.patch_size,
            voxel_size=args.voxel_size,
            min_streamlines_per_patch=args.min_streamlines_per_patch,
            min_bundle_size=args.min_bundle_size,
            streamline_margin_fraction=args.streamline_margin_fraction,
            patches_per_trk=args.patches_per_trk,
            use_cornucopia_3d=args.use_cornucopia_3d,
            cornucopia_allowed_presets=args.cornucopia_presets,
            tissue_threshold=args.tissue_threshold,
            enable_cell_blobs=args.enable_cell_blobs,
            cell_blob_count=args.cell_blob_count,
            cell_blob_intensity=args.cell_blob_intensity,
            cell_blob_radius_range=cell_blob_radius_range,
            enable_tissue_artifacts=args.enable_tissue_artifacts,
            enable_granular_noise=args.enable_granular_noise,
            enable_speckle_noise=args.enable_speckle_noise,
            enable_dash_noise=args.enable_dash_noise,
            enable_horizontal_banding=args.enable_horizontal_banding,
            granular_noise_strength=args.granular_noise_strength,
            artifact_strength=args.artifact_strength,
            speckle_noise_strength=args.speckle_noise_strength,
            speckle_noise_density=args.speckle_noise_density,
            speckle_noise_sigma=args.speckle_noise_sigma,
            speckle_square_size=args.speckle_square_size,
            dash_noise_strength=args.dash_noise_strength,
            dash_noise_density=args.dash_noise_density,
            dash_length_sigma=args.dash_length_sigma,
            dash_cross_sigma=args.dash_cross_sigma,
            banding_strength=args.banding_strength,
            banding_axis=args.banding_axis,
            fiber_intensity_min=args.fiber_intensity_min,
            fiber_intensity_max=args.fiber_intensity_max,
            fiber_max_boost=None if args.fiber_max_boost < 0 else args.fiber_max_boost,
            fiber_opacity=args.fiber_opacity,
            fiber_smoothing_sigma=args.fiber_smoothing_sigma,
            fiber_antialias=args.fiber_antialias,
            min_streamlines_rendered=None if args.min_streamlines_rendered <= 0 else args.min_streamlines_rendered,
            fiber_brightness_variation=args.fiber_brightness_variation,
            fiber_segment_brightness_variation=args.fiber_segment_brightness_variation,
            fiber_render_mode=args.fiber_render_mode,
            fiber_density_gamma=args.fiber_density_gamma,
            fiber_min_visibility=args.fiber_min_visibility,
            fiber_target_intensity=args.fiber_target_intensity,
            background_max_intensity=None if args.background_max_intensity < 0 else args.background_max_intensity,
            mask_smoothing_sigma=args.mask_smoothing_sigma,
            mask_binary_threshold=args.mask_binary_threshold,
            soft_mask=args.soft_mask,
            use_gpu=use_gpu,
            patch_use_gpu=args.patch_use_gpu,
            new_dim=new_dim,
            temp_dir_base=fast_tmp,
        )
        rendered = len(list((output_dir / trk_path.stem).glob("*_3d.nii.gz")))
        total = len(list(output_dir.rglob("*_3d.nii.gz")))
        print(f"[{idx}/{len(trk_paths)}] Done {trk_path.name}: "
              f"{rendered} patches (running total {total})", flush=True)
    print(f"Done. Cached patches in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
