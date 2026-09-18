"""
Get one 3D training patch (with at least one streamline) and visualise
sagittal / coronal / axial z-MIP slices for both the image and mask,
matching the style of real_vs_synth_final.png.

Usage:
    python visualize_one_patch.py
    python visualize_one_patch.py --out my_patch.png
    python visualize_one_patch.py --seed 123
"""

import argparse
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (after backend set)
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in [ROOT, os.path.join(ROOT, "training")]:
    if sub not in sys.path:
        sys.path.insert(0, sub)

from batch_processing import process_patches_inmemory
from rendering.volume_renderer import create_3d_volume_with_streamlines

# ---------------------------------------------------------------------------
# Paths – mirror the training command
# ---------------------------------------------------------------------------
INPUT_NIFTI = os.path.join(
    ROOT,
    "examples/example_data/"
    "sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz",
)
# aligned_wavy.trk has 7k–16k pts/streamline (native step ~0.004mm),
# well below the 0.064mm FOV — multiple streamlines land inside each patch.
# The plain ANTs-registered TRKs have ~0.25mm step (4× the FOV), so only
# 1 streamline survives per patch.
WAVY_TRK = os.path.join(ROOT, "registered_trk", "aligned_wavy.trk")

# Training hyperparams (from the torchrun command)
PATCH_SIZE  = [128, 128, 128]
VOXEL_SIZE  = 0.001
MIN_SL      = 2


def get_one_patch(random_state: int = 42, input_nifti: str = INPUT_NIFTI, trk_path: str = WAVY_TRK):
    """
    Return (vol, mask) float32 arrays shaped (D, H, W), normalised to [0,1].
    Uses the same augmentation settings as the reference image:
      - 140 cell-body blobs
      - all noise augmentations enabled
      - mask_smoothing_sigma=1.0, mask_binary_threshold=0.2
    """
    if not os.path.exists(trk_path):
        raise FileNotFoundError(f"aligned_wavy.trk not found: {trk_path}")

    fast_tmp = "/dev/shm" if os.path.exists("/dev/shm") else None

    for attempt in range(5):
        seed = random_state + attempt * 997

        with tempfile.TemporaryDirectory(prefix="syntract_viz_", dir=fast_tmp) as tmp:
            patches_dir = os.path.join(tmp, "patches")
            os.makedirs(patches_dir)

            process_patches_inmemory(
                input_nifti=input_nifti,
                trk_file=trk_path,
                num_patches=4,
                patch_size=PATCH_SIZE,
                min_streamlines_per_patch=MIN_SL,
                voxel_size=VOXEL_SIZE,
                patches_output_dir=patches_dir,
                skip_2d_viz=True,
                temp_dir_base=fast_tmp,
                patch_use_gpu=True,
                streamline_margin_fraction=0.10,
                random_state=seed,
            )

            nii_files = sorted(
                f for f in os.listdir(patches_dir)
                if (f.endswith(".nii.gz") or f.endswith(".nii"))
                and "_3d" not in f and "_mask" not in f and "_white" not in f
            )
            if not nii_files:
                print(f"  attempt {attempt+1}: no patches extracted, retrying…")
                continue

            nii_f = nii_files[0]
            nii_path = os.path.join(patches_dir, nii_f)
            ext = ".nii.gz" if nii_f.endswith(".nii.gz") else ".nii"
            trk_path = os.path.join(patches_dir, nii_f[:-len(ext)] + ".trk")
            out_path  = os.path.join(patches_dir, nii_f[:-len(ext)] + "_3d" + ext)

            if not os.path.exists(trk_path):
                print(f"  attempt {attempt+1}: no matching .trk, retrying…")
                continue

            rendered = create_3d_volume_with_streamlines(
                nifti_file=nii_path,
                trk_file=trk_path,
                output_file=out_path,
                save_mask=True,
                use_cornucopia_3d=True,
                cornucopia_allowed_presets=["ultra_heavy_speckle", "extreme_noise", "granular_realistic"],
                fiber_intensity_min=6.0,
                fiber_intensity_max=9.0,
                fiber_max_boost=None,
                fiber_opacity=1.0,
                fiber_smoothing_sigma=0.3,
                fiber_antialias=True,
                fiber_brightness_variation=0.40,
                fiber_segment_brightness_variation=0.20,
                fiber_render_mode="additive",
                fiber_density_gamma=3.0,
                fiber_min_visibility=0.0,
                fiber_target_intensity=25.0,
                tissue_threshold=0.0,
                mask_smoothing_sigma=1.0,
                mask_binary_threshold=0.2,
                enable_cell_blobs=True,
                cell_blob_count=200,
                cell_blob_intensity=0.5,
                cell_blob_radius_range=(3.0, 8.0),
                enable_tissue_artifacts=False,
                enable_granular_noise=True,
                enable_speckle_noise=True,
                enable_dash_noise=False,
                enable_horizontal_banding=True,
                granular_noise_strength=1.5,
                artifact_strength=0.0,
                speckle_noise_strength=1.5,
                speckle_noise_density=0.04,
                banding_strength=0.35,
                banding_axis=1,
                use_gpu=True,
                save_outputs=False,
                return_arrays=True,
                random_state=seed,
            )

            if rendered is None:
                print(f"  attempt {attempt+1}: renderer returned None, retrying…")
                continue

            vol, mask = rendered
            if vol is None or mask is None:
                print(f"  attempt {attempt+1}: None vol/mask, retrying…")
                continue

            # 1-99 percentile normalisation (matches training + OME-Zarr inference)
            lo, hi = np.percentile(vol, [1.0, 99.0])
            if hi > lo:
                vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            else:
                vol = np.zeros_like(vol, dtype=np.float32)
            mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

            coverage = 100 * mask.mean()
            print(
                f"  Got patch on attempt {attempt+1}  |  "
                f"trk={os.path.basename(trk_path)}  |  "
                f"vol={vol.shape}  mask coverage={coverage:.3f}%"
            )
            return vol, mask

    raise RuntimeError("Could not generate a patch with streamlines after 5 attempts.")


def mip(vol, axis):
    """Maximum intensity projection along the given axis."""
    return np.max(vol, axis=axis)



def plot_patch(vol, mask, out_path: str):
    """
    2×3 grid (vol is D×H×W):
      Row 0: mid-slice of image  — sagittal (mid W), coronal (mid H), axial (mid D)
      Row 1: binary mask MIP     — max projection same axes
    Mid-slice avoids MIP accumulation of noise/blobs across 128 slices.
    """
    D, H, W = vol.shape
    # Pick the slice with the most mask signal — image and mask use SAME slice index
    best_x = int(mask.sum(axis=(0, 1)).argmax())  # sagittal: along W axis
    best_y = int(mask.sum(axis=(0, 2)).argmax())  # coronal:  along H axis
    best_z = int(mask.sum(axis=(1, 2)).argmax())  # axial:    along D axis

    axes_info = [
        (f"Sagittal (X={best_x})", vol[:, :, best_x], mask[:, :, best_x]),
        (f"Coronal  (Y={best_y})", vol[:, best_y, :], mask[:, best_y, :]),
        (f"Axial    (Z={best_z})", vol[best_z, :, :], mask[best_z, :, :]),
    ]

    fig, grid = plt.subplots(2, 3, figsize=(12, 8),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04},
                             facecolor="black")
    fig.suptitle(
        "SYNTH 3D patch  –  image slice (top)  |  mask same slice (bottom)",
        color="white", fontsize=11,
    )

    for col, (title, img_sl, msk_sl) in enumerate(axes_info):
        p99 = np.percentile(img_sl, 99)
        img_display = np.clip(img_sl / max(p99, 1e-6), 0.0, 1.0)
        msk_binary = (msk_sl > 0.2).astype(np.float32)

        a_img = grid[0, col]
        a_img.imshow(img_display, cmap="gray", vmin=0, vmax=1,
                     interpolation="nearest", aspect="auto")
        a_img.set_title(title, color="white", fontsize=8, pad=3)
        a_img.axis("off")

        a_msk = grid[1, col]
        a_msk.imshow(msk_binary, cmap="gray", vmin=0, vmax=1,
                     interpolation="nearest", aspect="auto")
        a_msk.set_title(f"mask cov {100*msk_binary.mean():.3f}%",
                        color="white", fontsize=7, pad=3)
        a_msk.axis("off")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="patch_preview.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nifti", default=INPUT_NIFTI, help="Reference blockface NIfTI volume")
    parser.add_argument("--trk", default=WAVY_TRK, help="Registered (densified) TRK file")
    args = parser.parse_args()

    print("Generating one 3D patch…")
    vol, mask = get_one_patch(random_state=args.seed, input_nifti=args.nifti, trk_path=args.trk)
    plot_patch(vol, mask, out_path=args.out)


if __name__ == "__main__":
    main()
