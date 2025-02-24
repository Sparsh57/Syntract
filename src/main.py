import argparse
import nibabel as nib
import cupy as cp  # GPU-accelerated NumPy
from numba import cuda
import numpy as np
import os
from nifti_preprocessing import resample_nifti_gpu
from transform import build_new_affine
from streamline_processing import transform_and_densify_streamlines_gpu
from nibabel.streamlines import Tractogram, save as save_trk

def process_and_save(
        old_nifti_path,
        old_trk_path,
        new_voxel_size=0.5,
        new_dim=(116, 140, 96),
        output_prefix="resampled",
        n_jobs=8,
        patch_center=None,
        reduction_method=None
):
    print("\n=== Loading NIfTI ===")
    old_img = nib.load(old_nifti_path, mmap=True)
    old_affine = old_img.affine
    old_shape = old_img.shape[:3]

    print(f"Old shape: {old_shape}")
    print(f"Old affine:\n{old_affine}")

    print("\n=== Building new affine ===")
    A_new = build_new_affine(old_affine, old_shape, new_voxel_size, new_dim, patch_center_mm=patch_center)

    print(f"New affine:\n{A_new}")
    print(f"New dimensions: {new_dim}")

    print("\n=== Resampling NIfTI using GPU ===")
    new_data, tmp_mmap = resample_nifti_gpu(old_img, A_new, new_dim, chunk_size=(64, 64, 64), n_jobs=n_jobs)

    if reduction_method:
        print(f"\n=== Applying Reduction: {reduction_method} ===")
        if reduction_method == 'mip':
            reduced_data = cp.max(new_data, axis=1)
        elif reduction_method == 'mean':
            reduced_data = cp.mean(new_data, axis=1)
        else:
            raise ValueError(f"Unsupported reduction method: {reduction_method}")

        reduced_data = reduced_data[..., cp.newaxis]  # Keep z-axis size 1
        new_data = reduced_data
        new_dim = (new_dim[0], 1, new_dim[2])

    new_img = nib.Nifti1Image(cp.asnumpy(new_data), A_new)
    out_nifti_path = output_prefix + ".nii.gz"
    nib.save(new_img, out_nifti_path)

    if os.path.exists(tmp_mmap):
        os.remove(tmp_mmap)

    print(f"Saved new NIfTI => {out_nifti_path}")

    print("\n=== Loading Tractography Data ===")
    trk_obj = nib.streamlines.load(old_trk_path)
    old_streams_mm = trk_obj.tractogram.streamlines
    print(f"Loaded {len(old_streams_mm)} streamlines.")

    print("\n=== Transforming, Densifying, and Clipping Streamlines Using GPU ===")
    densified_vox = transform_and_densify_streamlines_gpu(
        old_streams_mm, A_new, new_dim, step_size=0.5, n_jobs=n_jobs
    )

    new_trk_header = trk_obj.header.copy()
    new_trk_header["dimensions"] = np.array(new_dim, dtype=np.int16)
    new_voxsize = np.sqrt(np.sum(A_new[:3, :3] ** 2, axis=0))
    new_trk_header["voxel_sizes"] = new_voxsize.astype(np.float32)
    new_trk_header["voxel_to_rasmm"] = A_new.astype(np.float32)

    print("\n=== Saving New .trk File ===")
    new_tractogram = Tractogram(densified_vox, affine_to_rasmm=A_new)
    out_trk_path = output_prefix + ".trk"
    save_trk(new_tractogram, out_trk_path, header=new_trk_header)

    print(f"Saved new .trk => {out_trk_path}")
    print("\n==== Process Completed Successfully! ====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and resample NIfTI and streamline tractography data.")

    parser.add_argument("old_nifti_path", type=str, help="Path to input NIfTI (.nii or .nii.gz) file.")
    parser.add_argument("old_trk_path", type=str, help="Path to input TRK (.trk) file.")
    parser.add_argument("--voxel_size", type=float, default=0.5, help="New voxel size (default: 0.5 mm).")
    parser.add_argument("--new_dim", type=int, nargs=3, default=[116, 140, 96], help="New image dimensions (x, y, z).")
    parser.add_argument("--output_prefix", type=str, default="resampled", help="Prefix for output files.")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel jobs (-1 for all CPUs).")
    parser.add_argument("--patch_center", type=float, nargs=3, default=None, help="Optional patch center in mm.")
    parser.add_argument("--reduction", type=str, choices=["mip", "mean"], default=None,
                        help="Optional reduction along z-axis.")

    args = parser.parse_args()

    process_and_save(
        old_nifti_path=args.old_nifti_path,
        old_trk_path=args.old_trk_path,
        new_voxel_size=args.voxel_size,
        new_dim=tuple(args.new_dim),
        output_prefix=args.output_prefix,
        n_jobs=args.n_jobs,
        patch_center=tuple(args.patch_center) if args.patch_center else None,
        reduction_method=args.reduction
    )
