#!/usr/bin/env python
"""Classical 3D fiber-location extractor (no deep learning, no labels).

Pipeline: robust-normalize -> 3D denoise -> 3D structure-tensor "lineness"
(oriented tube response, suppresses round cells + isotropic noise) -> threshold
-> drop SMALL connected components (remove tiny fibers, keep long connected ones)
-> save 3D NIfTI for visualization (napari / ITK-SNAP / FSLeyes).

Usage:
    python fiber_extract_3d.py --input mf283_slice036/ch488_*.npy --out_dir fiber3d_out
    python fiber_extract_3d.py --input patch.nii.gz --min_voxels 80 --threshold_pct 97
"""
import argparse, os, glob
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, label
from skimage.feature import structure_tensor, structure_tensor_eigenvalues


def robust_norm(a, lo=1.0, hi=99.0):
    a = np.asarray(a, np.float32)
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-8), 0.0, 1.0)


def lineness_3d(vol, sigma=2.0):
    """Structure-tensor tube response. Eigenvalues l0>=l1>=l2 per voxel:
    a fiber (constant along axis, varies across) has gradients in the plane
    perpendicular to it -> l0~l1 large, l2 small. lineness = (l1 - l2): high for
    tubes, ~0 for round blobs (l0~l1~l2) and flat noise."""
    S = structure_tensor(vol, sigma=sigma, order="rc")
    l0, l1, l2 = structure_tensor_eigenvalues(S)  # descending
    denom = l0 + l1 + l2 + 1e-8
    return ((l1 - l2) / denom).astype(np.float32)


def remove_small(binary, min_voxels=60, min_len=0.0):
    """Drop connected components smaller than min_voxels (and shorter than
    min_len bbox-diagonal if set). 26-connectivity. Returns cleaned mask + stats."""
    lab, n = label(binary, structure=np.ones((3, 3, 3), int))
    out = np.zeros_like(binary, dtype=np.uint8)
    kept = removed = 0
    for cid in range(1, n + 1):
        coords = np.argwhere(lab == cid)
        if len(coords) < min_voxels:
            removed += 1
            continue
        if min_len > 0:
            ext = coords.max(0) - coords.min(0) + 1
            if np.sqrt((ext.astype(float) ** 2).sum()) < min_len:
                removed += 1
                continue
        out[lab == cid] = 1
        kept += 1
    return out, kept, removed, n


def extract(vol, voxel_size=(1.16, 1.16, 1.0), denoise_sigma=1.2, st_sigma=2.0,
            threshold_pct=97.0, min_voxels=60, min_len=0.0):
    norm = robust_norm(vol)
    den = gaussian_filter(norm, denoise_sigma)               # kill speckle
    line = lineness_3d(den, sigma=st_sigma)                  # oriented-tube response
    likelihood = gaussian_filter(line * den, 1.0)           # x intensity = fiber-likelihood
    likelihood = likelihood / (likelihood.max() + 1e-8)
    thr = np.percentile(likelihood, threshold_pct)
    binary = likelihood > thr
    mask, kept, removed, total = remove_small(binary, min_voxels, min_len)
    return {
        "denoised": den.astype(np.float32),
        "likelihood": likelihood.astype(np.float32),
        "mask": mask,
        "kept": kept, "removed": removed, "total": total,
    }


def save_nii(arr, path, voxel_size):
    aff = np.diag([float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]), 1.0])
    nib.save(nib.Nifti1Image(np.asarray(arr), aff), path)


def save_tif(arr, path):
    """TIFF for napari (its builtin reader can't open NIfTI). 3D (z,y,x) stack."""
    arr = np.asarray(arr)
    arr = arr.astype("uint8") if "mask" in os.path.basename(path) else arr.astype("float32")
    try:
        import tifffile
        tifffile.imwrite(path, arr)
    except ImportError:
        from skimage.io import imsave
        imsave(path, arr, check_contrast=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".npy or .nii.gz 3D volume (glob ok)")
    ap.add_argument("--out_dir", default="fiber3d_out")
    ap.add_argument("--voxel_size", type=float, nargs=3, default=[1.16, 1.16, 1.0])
    ap.add_argument("--denoise_sigma", type=float, default=1.2)
    ap.add_argument("--st_sigma", type=float, default=2.0)
    ap.add_argument("--threshold_pct", type=float, default=97.0)
    ap.add_argument("--min_voxels", type=int, default=60, help="drop components smaller than this (remove small fibers)")
    ap.add_argument("--min_len", type=float, default=0.0, help="also drop components shorter than this bbox-diagonal")
    ap.add_argument("--crop", type=int, nargs=3, default=None, help="optional center crop z y x")
    args = ap.parse_args()

    path = sorted(glob.glob(args.input))[0]
    vol = (np.load(path) if path.endswith(".npy")
           else nib.load(path).get_fdata()).astype(np.float32)
    if args.crop:
        cz, cy, cx = (s // 2 for s in vol.shape)
        dz, dy, dx = (c // 2 for c in args.crop)
        vol = vol[max(0, cz-dz):cz+dz, max(0, cy-dy):cy+dy, max(0, cx-dx):cx+dx]
    print(f"input {path} shape={vol.shape}")

    r = extract(vol, args.voxel_size, args.denoise_sigma, args.st_sigma,
                args.threshold_pct, args.min_voxels, args.min_len)
    print(f"components: {r['total']} total -> kept {r['kept']}, removed {r['removed']} "
          f"(small). mask coverage {100*r['mask'].mean():.3f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, arr in [("image", robust_norm(vol)), ("fiber_likelihood", r["likelihood"]), ("fiber_mask", r["mask"])]:
        save_nii(arr, f"{args.out_dir}/{name}.nii.gz", args.voxel_size)  # for itksnap / fsleyes
        save_tif(arr, f"{args.out_dir}/{name}.tif")                       # for napari
    print(f"saved 3D -> {args.out_dir}/ (image, fiber_likelihood, fiber_mask) as .nii.gz AND .tif")
    print(f"view: napari {args.out_dir}/image.tif {args.out_dir}/fiber_mask.tif   |   itksnap/fsleyes use the .nii.gz")

    # quick 2D preview so you can sanity-check without a 3D viewer
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        mip_img = robust_norm(vol).max(0); mip_mask = r["mask"].max(0)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(robust_norm(vol).mean(0), cmap="gray"); ax[0].set_title("image (mean-z)")
        ax[1].imshow(r["likelihood"].max(0), cmap="inferno"); ax[1].set_title("fiber likelihood (MIP)")
        ax[2].imshow(mip_img, cmap="gray"); ax[2].imshow(np.ma.masked_less(mip_mask, 0.5), cmap="autumn", alpha=0.8)
        ax[2].set_title(f"fiber mask (MIP), {r['kept']} fibers")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(f"{args.out_dir}/preview.png", dpi=80, bbox_inches="tight")
        print(f"preview -> {args.out_dir}/preview.png")
    except Exception as e:
        print(f"(preview skipped: {e})")


if __name__ == "__main__":
    main()
