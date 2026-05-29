"""
Generate clean vs. artefact-augmented synthetic 3D patches for visual QA.

Example:
    python preview_realistic_augmentations_3d.py \
        --trk_dir /path/to/trks \
        --input_nifti /path/to/brain.nii.gz \
        --output_dir ./augmentation_preview \
        --num_patches 3 \
        --patch_size 128 128 128
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cumulative import process_patches_inmemory  # type: ignore
from syntract_viewer.volume_renderer import create_3d_volume_with_streamlines


def _find_trk(args) -> Path:
    if args.trk_file:
        trk = Path(args.trk_file)
        if not trk.exists():
            raise FileNotFoundError(f"TRK file not found: {trk}")
        return trk
    trk_paths = sorted(Path(args.trk_dir).glob("*.trk"))
    if not trk_paths:
        raise ValueError(f"No .trk files found in {args.trk_dir}")
    return trk_paths[0]


def _iter_patch_pairs(patches_dir: Path):
    def split_nifti_ext(path: Path):
        name = path.name
        if name.endswith(".nii.gz"):
            return name[:-7], ".nii.gz"
        if name.endswith(".nii"):
            return name[:-4], ".nii"
        return None, None

    nii_files = sorted(
        f
        for f in patches_dir.iterdir()
        if (f.name.endswith(".nii.gz") or f.name.endswith(".nii"))
        and "_3d" not in f.name
        and "_mask" not in f.name
        and "_white" not in f.name
    )
    for nii_path in nii_files:
        stem, _ = split_nifti_ext(nii_path)
        if stem is None:
            continue
        trk_path = nii_path.with_name(f"{stem}.trk")
        if trk_path.exists():
            yield nii_path, trk_path


def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = np.asarray(slice_2d, dtype=np.float32)
    lo, hi = np.percentile(slice_2d, [1, 99])
    if hi <= lo:
        lo, hi = float(slice_2d.min()), float(slice_2d.max())
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.float32)
    return np.clip((slice_2d - lo) / (hi - lo), 0.0, 1.0)


def _load_volume(path: Path, fallback_shape=None) -> np.ndarray:
    if path is not None and path.exists():
        return nib.load(str(path)).get_fdata().astype(np.float32)
    if fallback_shape is None:
        raise FileNotFoundError(f"Volume not found: {path}")
    return np.zeros(fallback_shape, dtype=np.float32)


def _save_comparison_png(clean: np.ndarray, augmented: np.ndarray, mask: np.ndarray, output_png: Path):
    mask_scores = mask.sum(axis=(0, 1))
    z_idx = int(np.argmax(mask_scores)) if np.any(mask_scores > 0) else clean.shape[2] // 2

    positive_residual = np.maximum(augmented - clean, 0.0).astype(np.float32, copy=False)
    if mask is not None and mask.shape == positive_residual.shape:
        positive_residual = positive_residual.copy()
        positive_residual[mask > 0] = 0.0
    positive_values = positive_residual[positive_residual > 0]
    dot_z_idx = z_idx
    if positive_values.size:
        thresh = float(np.percentile(positive_values, 99.5))
        high_residual = np.where(positive_residual >= thresh, positive_residual, 0.0)
        dot_scores = high_residual.sum(axis=(0, 1))
        if np.any(dot_scores > 0):
            dot_z_idx = int(np.argmax(dot_scores))

    clean_sl = _normalize_slice(clean[:, :, z_idx]).T
    aug_sl = _normalize_slice(augmented[:, :, z_idx]).T
    diff_sl = _normalize_slice(np.abs(augmented[:, :, z_idx] - clean[:, :, z_idx])).T
    mask_sl = (mask[:, :, z_idx] > 0).T
    dot_sl = _normalize_slice(augmented[:, :, dot_z_idx]).T

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    axes[0].imshow(clean_sl, cmap="gray", origin="lower")
    axes[0].set_title("clean")
    axes[1].imshow(aug_sl, cmap="gray", origin="lower")
    axes[1].set_title("augmented")
    axes[2].imshow(diff_sl, cmap="magma", origin="lower")
    axes[2].set_title("difference")
    axes[3].imshow(aug_sl, cmap="gray", origin="lower")
    axes[3].imshow(np.ma.masked_where(~mask_sl, mask_sl), cmap="autumn", alpha=0.45, origin="lower")
    axes[3].set_title("augmented + mask")
    axes[4].imshow(dot_sl, cmap="gray", origin="lower")
    axes[4].set_title("dot-focused slice")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"fiber slice z={z_idx}, dot slice z={dot_z_idx}")
    fig.savefig(output_png, dpi=140)
    plt.close(fig)


def _save_training_inputs_png(
    raw_patch: np.ndarray,
    white_mask: np.ndarray,
    training_image: np.ndarray,
    fiber_mask: np.ndarray,
    output_png: Path,
):
    mask_scores = fiber_mask.sum(axis=(0, 1))
    if np.any(mask_scores > 0):
        z_idx = int(np.argmax(mask_scores))
    else:
        wm_scores = white_mask.sum(axis=(0, 1))
        z_idx = int(np.argmax(wm_scores)) if np.any(wm_scores > 0) else raw_patch.shape[2] // 2

    raw_sl = _normalize_slice(raw_patch[:, :, z_idx]).T
    wm_sl = (white_mask[:, :, z_idx] > 0).T
    image_sl = _normalize_slice(training_image[:, :, z_idx]).T
    fiber_sl = (fiber_mask[:, :, z_idx] > 0).T

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    axes[0].imshow(raw_sl, cmap="gray", origin="lower")
    axes[0].set_title("raw patch")
    axes[1].imshow(wm_sl, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("white mask")
    axes[2].imshow(image_sl, cmap="gray", origin="lower")
    axes[2].set_title("training image")
    axes[3].imshow(fiber_sl, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[3].set_title("fiber target")
    axes[4].imshow(image_sl, cmap="gray", origin="lower")
    axes[4].imshow(np.ma.masked_where(~fiber_sl, fiber_sl), cmap="autumn", alpha=0.45, origin="lower")
    axes[4].set_title("image + target")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"training inputs, slice z={z_idx}")
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Preview clean vs realistic synthetic 3D augmentations")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trk_dir", type=str, help="Directory with .trk files")
    source.add_argument("--trk_file", type=str, help="Single .trk file")
    parser.add_argument("--input_nifti", required=True, type=str)
    parser.add_argument("--white_mask", default="None", type=str)
    parser.add_argument("--output_dir", default="./augmentation_preview", type=str)
    parser.add_argument("--num_patches", default=3, type=int)
    parser.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128], metavar=("D", "H", "W"))
    parser.add_argument("--voxel_size", default=0.05, type=float,
                        help="Target voxel size in mm. Default 0.05 matches native blockface resolution.")
    parser.add_argument("--min_streamlines_per_patch", default=5, type=int)
    parser.add_argument("--min_bundle_size", default=5, type=int)
    parser.add_argument("--streamline_margin_fraction", default=0.10, type=float,
                        help="Require sampled patches to contain streamlines away from patch borders")
    parser.add_argument("--artifact_strength", default=0.45, type=float)
    parser.add_argument("--granular_noise_strength", default=0.35, type=float)
    parser.add_argument("--speckle_noise_strength", default=0.60, type=float,
                        help="Brightness of bright square speck artefacts (blockface dust/particles)")
    parser.add_argument("--speckle_noise_density", default=0.005, type=float,
                        help="Probability of a speck seed per tissue voxel")
    parser.add_argument("--speckle_noise_sigma", default=0.0, type=float,
                        help="Gaussian sigma for specks; 0=sharp edges (use square mode instead)")
    parser.add_argument("--speckle_square_size", default=2, type=int,
                        help="Uniform filter size for square speck spreading; 1=single pixel")
    parser.add_argument("--dash_noise_strength", default=0.55, type=float,
                        help="Brightness of short diagonal dash artefacts")
    parser.add_argument("--dash_noise_density", default=0.0005, type=float,
                        help="Probability of a dash seed per tissue voxel")
    parser.add_argument("--dash_length_sigma", default=4.0, type=float,
                        help="1-D Gaussian sigma along primary axis controlling dash length")
    parser.add_argument("--banding_strength", default=0.12, type=float,
                        help="Amplitude of horizontal blockface banding (0=off)")
    parser.add_argument("--banding_axis", default=1, type=int,
                        help="Volume axis along which banding varies (1=coronal/Y)")
    parser.add_argument("--fiber_intensity_min", default=60.0, type=float,
                        help="Minimum added fiber brightness")
    parser.add_argument("--fiber_intensity_max", default=100.0, type=float,
                        help="Maximum added fiber brightness")
    parser.add_argument("--fiber_max_boost", default=90.0, type=float,
                        help="Cap added fiber brightness above local tissue; use negative to disable")
    parser.add_argument("--fiber_opacity", default=1.0, type=float,
                        help="Multiplier on added fiber brightness after capping")
    parser.add_argument("--fiber_smoothing_sigma", default=0.0, type=float,
                        help="Gaussian spread of fiber boost; 0=single-pixel strands")
    parser.add_argument("--fiber_antialias", dest="fiber_antialias", action="store_true", default=True,
                        help="Use subvoxel antialias rendering for less blocky streamlines")
    parser.add_argument("--no_fiber_antialias", dest="fiber_antialias", action="store_false",
                        help="Disable subvoxel antialias rendering")
    parser.add_argument("--min_streamlines_rendered", default=20, type=int,
                        help="Skip patch if fewer than this many streamlines are present; 0 disables check")
    parser.add_argument("--fiber_brightness_variation", default=0.60, type=float,
                        help="Per-streamline brightness variation; image only, mask unchanged")
    parser.add_argument("--fiber_segment_brightness_variation", default=0.35, type=float,
                        help="Per-segment brightness jitter along each streamline")
    parser.add_argument("--fiber_render_mode", default="additive", choices=["additive", "density", "embedded"],
                        help="additive renders each streamline as a direct brightness boost")
    parser.add_argument("--fiber_density_gamma", default=5.0, type=float,
                        help="Density-mode gamma; higher = thinner bundles")
    parser.add_argument("--fiber_min_visibility", default=0.0, type=float,
                        help="Minimum normalized visibility for fiber voxels")
    parser.add_argument("--fiber_target_intensity", default=25.0, type=float,
                        help="Embedded/density mode fiber brightness target")
    parser.add_argument("--background_max_intensity", default=10.0, type=float,
                        help="Cap tissue/background intensity to increase fiber contrast; negative disables")
    parser.add_argument("--mask_smoothing_sigma", default=2.0, type=float,
                        help="3D mask smoothing sigma used for the fiber target")
    parser.add_argument("--mask_binary_threshold", default=0.01, type=float,
                        help="Normalized mask threshold used for the fiber target")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--use_cornucopia_3d", action="store_true", default=True,
                        help="Apply TRUE 3D Cornucopia augmentation on the augmented render")
    parser.add_argument("--cornucopia_preset", default=None, type=str,
                        help="Force a specific preset (e.g. granular_realistic). None=random.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_patches"
    clean_dir = output_dir / "clean"
    aug_dir = output_dir / "augmented"
    png_dir = output_dir / "comparisons"
    training_png_dir = output_dir / "training_inputs"
    for path in (raw_dir, clean_dir, aug_dir, png_dir, training_png_dir):
        path.mkdir(parents=True, exist_ok=True)

    white_mask = None if args.white_mask is None or args.white_mask.lower() == "none" else args.white_mask
    trk_path = _find_trk(args)

    print(f"Extracting {args.num_patches} raw patches from {trk_path.name}")
    process_patches_inmemory(
        input_nifti=args.input_nifti,
        trk_file=str(trk_path),
        num_patches=int(args.num_patches),
        patch_size=list(args.patch_size),
        min_streamlines_per_patch=int(args.min_streamlines_per_patch),
        min_bundle_size=int(args.min_bundle_size),
        voxel_size=float(args.voxel_size),
        white_mask_file=white_mask,
        patches_output_dir=str(raw_dir),
        skip_2d_viz=True,
        patch_use_gpu=not args.no_gpu,
        streamline_margin_fraction=float(args.streamline_margin_fraction),
        random_state=int(args.seed),
    )

    pairs = list(_iter_patch_pairs(raw_dir))[: int(args.num_patches)]
    if not pairs:
        raise RuntimeError(f"No raw NIfTI/TRK patch pairs were generated in {raw_dir}")

    for idx, (nii_path, patch_trk) in enumerate(pairs):
        clean_out = clean_dir / f"patch_{idx:03d}_clean_3d.nii.gz"
        aug_out = aug_dir / f"patch_{idx:03d}_augmented_3d.nii.gz"
        if nii_path.name.endswith(".nii.gz"):
            patch_stem = nii_path.name[:-7]
            patch_ext = ".nii.gz"
        else:
            patch_stem = nii_path.name[:-4]
            patch_ext = ".nii"
        patch_white = nii_path.with_name(f"{patch_stem}_white_mask{patch_ext}")
        white_mask_path = str(patch_white) if patch_white.exists() else white_mask

        clean_vol, clean_mask = create_3d_volume_with_streamlines(
            nifti_file=str(nii_path),
            trk_file=str(patch_trk),
            output_file=str(clean_out),
            white_mask_path=white_mask_path,
            save_mask=True,
            use_cornucopia_3d=False,
            fiber_intensity_min=float(args.fiber_intensity_min),
            fiber_intensity_max=float(args.fiber_intensity_max),
            fiber_max_boost=None if args.fiber_max_boost < 0 else float(args.fiber_max_boost),
            fiber_opacity=float(args.fiber_opacity),
            fiber_smoothing_sigma=float(args.fiber_smoothing_sigma),
            fiber_antialias=bool(args.fiber_antialias),
            min_streamlines_rendered=None if args.min_streamlines_rendered <= 0 else int(args.min_streamlines_rendered),
            fiber_brightness_variation=float(args.fiber_brightness_variation),
            fiber_segment_brightness_variation=float(args.fiber_segment_brightness_variation),
            fiber_render_mode=str(args.fiber_render_mode),
            fiber_density_gamma=float(args.fiber_density_gamma),
            fiber_min_visibility=float(args.fiber_min_visibility),
            fiber_target_intensity=float(args.fiber_target_intensity),
            background_max_intensity=None if args.background_max_intensity < 0 else float(args.background_max_intensity),
            enable_tissue_artifacts=False,
            enable_granular_noise=False,
            random_state=int(args.seed) + idx,
            mask_smoothing_sigma=float(args.mask_smoothing_sigma),
            mask_binary_threshold=float(args.mask_binary_threshold),
            use_gpu=not args.no_gpu,
            verbose=False,
            save_outputs=True,
            return_arrays=True,
        )
        cornucopia_presets = ([args.cornucopia_preset] if args.cornucopia_preset else None)
        aug_vol, aug_mask = create_3d_volume_with_streamlines(
            nifti_file=str(nii_path),
            trk_file=str(patch_trk),
            output_file=str(aug_out),
            white_mask_path=white_mask_path,
            save_mask=True,
            use_cornucopia_3d=bool(args.use_cornucopia_3d),
            cornucopia_allowed_presets=cornucopia_presets,
            fiber_intensity_min=float(args.fiber_intensity_min),
            fiber_intensity_max=float(args.fiber_intensity_max),
            fiber_max_boost=None if args.fiber_max_boost < 0 else float(args.fiber_max_boost),
            fiber_opacity=float(args.fiber_opacity),
            fiber_smoothing_sigma=float(args.fiber_smoothing_sigma),
            fiber_antialias=bool(args.fiber_antialias),
            min_streamlines_rendered=None if args.min_streamlines_rendered <= 0 else int(args.min_streamlines_rendered),
            fiber_brightness_variation=float(args.fiber_brightness_variation),
            fiber_segment_brightness_variation=float(args.fiber_segment_brightness_variation),
            fiber_render_mode=str(args.fiber_render_mode),
            fiber_density_gamma=float(args.fiber_density_gamma),
            fiber_min_visibility=float(args.fiber_min_visibility),
            fiber_target_intensity=float(args.fiber_target_intensity),
            background_max_intensity=None if args.background_max_intensity < 0 else float(args.background_max_intensity),
            enable_tissue_artifacts=True,
            enable_granular_noise=True,
            enable_speckle_noise=True,
            enable_dash_noise=True,
            enable_horizontal_banding=True,
            artifact_strength=float(args.artifact_strength),
            granular_noise_strength=float(args.granular_noise_strength),
            speckle_noise_strength=float(args.speckle_noise_strength),
            speckle_noise_density=float(args.speckle_noise_density),
            speckle_noise_sigma=float(args.speckle_noise_sigma),
            speckle_square_size=int(args.speckle_square_size),
            dash_noise_strength=float(args.dash_noise_strength),
            dash_noise_density=float(args.dash_noise_density),
            dash_length_sigma=float(args.dash_length_sigma),
            banding_strength=float(args.banding_strength),
            banding_axis=int(args.banding_axis),
            random_state=int(args.seed) + idx,
            mask_smoothing_sigma=float(args.mask_smoothing_sigma),
            mask_binary_threshold=float(args.mask_binary_threshold),
            use_gpu=not args.no_gpu,
            verbose=False,
            save_outputs=True,
            return_arrays=True,
        )

        if clean_mask is None:
            clean_mask = np.zeros_like(clean_vol, dtype=np.float32)
        if aug_mask is not None and clean_mask.shape == aug_mask.shape:
            mask_delta = int(np.count_nonzero((clean_mask > 0) != (aug_mask > 0)))
            if mask_delta:
                print(f"Warning: mask mismatch in patch {idx}: {mask_delta} voxels")
        if aug_mask is None:
            aug_mask = clean_mask

        raw_patch = _load_volume(nii_path)
        white_mask_vol = _load_volume(patch_white, fallback_shape=raw_patch.shape)

        png_path = png_dir / f"patch_{idx:03d}_comparison.png"
        _save_comparison_png(clean_vol, aug_vol, clean_mask, png_path)
        print(f"Wrote {png_path}")

        training_png = training_png_dir / f"patch_{idx:03d}_training_inputs.png"
        _save_training_inputs_png(
            raw_patch=raw_patch,
            white_mask=white_mask_vol,
            training_image=aug_vol,
            fiber_mask=aug_mask,
            output_png=training_png,
        )
        print(f"Wrote {training_png}")

    print(f"Done. Preview outputs are in {output_dir}")


if __name__ == "__main__":
    main()
