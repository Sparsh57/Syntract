import argparse
import nibabel as nib
import numpy as np
import os
from nifti_preprocessing import resample_nifti
from transform import build_new_affine
from streamline_processing import transform_and_densify_streamlines
from nibabel.streamlines import Tractogram, save as save_trk

def process_and_save(
        old_nifti_path,
        old_trk_path,
        new_voxel_size=0.5,
        new_dim=(116, 140, 96),
        output_prefix="resampled",
        n_jobs=8,
        patch_center=None,
        reduction_method=None,
        use_gpu=True,
        interp_method='hermite',
        step_size=0.5,
        max_output_gb=64.0
):
    """
    Processing and saving NIfTI and streamline data with new parameters.

    Parameters
    ----------
    old_nifti_path : str
        Path to the original NIfTI file.
    old_trk_path : str
        Path to the original streamline file.
    new_voxel_size : float, optional
        Desired voxel size, default is 0.5.
    new_dim : tuple, optional
        New dimensions, default is (116, 140, 96).
    output_prefix : str, optional
        Prefix for output files, default is "resampled".
    n_jobs : int, optional
        Number of parallel jobs, default is 8.
    patch_center : tuple, optional
        Center for patching, default is None.
    reduction_method : str, optional
        Method for reduction, default is None.
    use_gpu : bool, optional
        Whether to use GPU, default is True.
    interp_method : str, optional
        Interpolation method, default is 'hermite'.
    step_size : float, optional
        Step size for streamline densification, by default 0.5.
    max_output_gb : float, optional
        Maximum output size in GB, by default 64 GB.
    """
    # Import appropriate array library based on use_gpu setting
    if use_gpu:
        try:
            import cupy as xp
            from numba import cuda
            print("Using GPU acceleration")
        except ImportError:
            print("Warning: Could not import GPU libraries. Falling back to CPU.")
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp
        print("Using CPU processing")

    print("\n=== Loading NIfTI ===")
    old_img = nib.load(old_nifti_path, mmap=True)
    old_affine = old_img.affine
    old_shape = old_img.shape[:3]
    old_voxel_sizes = np.array(old_img.header.get_zooms()[:3])

    print(f"Old shape: {old_shape}")
    print(f"Old voxel sizes: {old_voxel_sizes}")
    print(f"Old affine:\n{old_affine}")
    
    print("\n=== Building new affine ===")
    print(f"Using dimensions for affine: {new_dim}")
    A_new = build_new_affine(old_affine, old_shape, new_voxel_size, new_dim, patch_center_mm=patch_center, use_gpu=use_gpu)

    print(f"New affine:\n{A_new}")
    print(f"New dimensions: {new_dim}")
    print(f"Using these dimensions for ALL subsequent processing")
    
    print(f"\n=== Resampling NIfTI using {'GPU' if use_gpu else 'CPU'} ===")
    print(f"Resampling to dimensions: {new_dim}")
    print(f"Memory limit: {max_output_gb} GB")
    new_data, tmp_mmap = resample_nifti(old_img, A_new, new_dim, chunk_size=(64, 64, 64), n_jobs=n_jobs, use_gpu=use_gpu, max_output_gb=max_output_gb)
    print(f"Resampled data shape: {new_data.shape}")
    
    # Verify shape matches expected dimensions
    if new_data.shape[:3] != new_dim:
        print(f"WARNING: Resampled shape {new_data.shape[:3]} does not match expected dimensions {new_dim}")
        print("This could lead to streamline clipping issues!")

    if reduction_method:
        print(f"\n=== Applying Reduction: {reduction_method} ===")
        if use_gpu:
            if reduction_method == 'mip':
                reduced_data = xp.max(new_data, axis=1)
            elif reduction_method == 'mean':
                reduced_data = xp.mean(new_data, axis=1)
            else:
                raise ValueError(f"Unsupported reduction method: {reduction_method}")
            
            reduced_data = reduced_data[..., xp.newaxis]  # Keep z-axis size 1
            new_data = reduced_data
        else:
            if reduction_method == 'mip':
                reduced_data = np.max(new_data, axis=1)
            elif reduction_method == 'mean':
                reduced_data = np.mean(new_data, axis=1)
            else:
                raise ValueError(f"Unsupported reduction method: {reduction_method}")
            
            reduced_data = reduced_data[..., np.newaxis]  # Keep z-axis size 1
            new_data = reduced_data
            
        new_dim = (new_dim[0], 1, new_dim[2])

    # Convert to numpy array for saving if using GPU
    if use_gpu:
        new_data_np = xp.asnumpy(new_data)
    else:
        new_data_np = new_data
    
    print(f"Final data shape before saving: {new_data_np.shape}")
        
    new_img = nib.Nifti1Image(new_data_np, A_new)
    out_nifti_path = output_prefix + ".nii.gz"
    nib.save(new_img, out_nifti_path)

    if os.path.exists(tmp_mmap):
        os.remove(tmp_mmap)

    print(f"Saved new NIfTI => {out_nifti_path}")

    print("\n=== Loading Tractography Data ===")
    trk_obj = nib.streamlines.load(old_trk_path)
    old_streams_mm = trk_obj.tractogram.streamlines
    print(f"Loaded {len(old_streams_mm)} streamlines.")
    
    # Get statistics on the original streamlines
    total_points = sum(len(s) for s in old_streams_mm)
    avg_points = total_points / len(old_streams_mm) if len(old_streams_mm) > 0 else 0
    print(f"Total points in original streamlines: {total_points}")
    print(f"Average points per streamline: {avg_points:.2f}")
    
    print(f"\n=== Transforming, Densifying, and Clipping Streamlines Using {'GPU' if use_gpu else 'CPU'} with {interp_method} interpolation ===")
    print(f"Step size: {step_size}, Voxel size: {new_voxel_size}")
    print(f"FOV clipping: Enabled")
    print(f"Using dimensions: {new_dim}")
    
    # Process streamlines with clipping always enabled
    densified_vox = transform_and_densify_streamlines(
        old_streams_mm, A_new, new_dim, step_size=step_size, n_jobs=n_jobs, 
        use_gpu=use_gpu, interp_method=interp_method
    )
    
    # Report statistics on the processed streamlines
    print(f"\nProcessed {len(densified_vox)} streamlines.")
    if len(densified_vox) == 0:
        print("WARNING: No streamlines were processed! Check your parameters.")
        print("Try a larger voxel size or adjust the step size.")
        return
        
    total_points_new = sum(len(s) for s in densified_vox)
    avg_points_new = total_points_new / len(densified_vox) if len(densified_vox) > 0 else 0
    print(f"Total points in processed streamlines: {total_points_new}")
    print(f"Average points per streamline: {avg_points_new:.2f}")
    print(f"Change in streamline count: {len(densified_vox) - len(old_streams_mm)} ({(len(densified_vox) - len(old_streams_mm))/len(old_streams_mm)*100:.1f}%)")
    print(f"Change in point count: {total_points_new - total_points} ({(total_points_new - total_points)/total_points*100:.1f}%)")

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

    parser.add_argument("--input", type=str, required=True, help="Path to input NIfTI (.nii or .nii.gz) file.")
    parser.add_argument("--trk", type=str, required=True, help="Path to input TRK (.trk) file.")
    parser.add_argument("--output", type=str, default="resampled", help="Prefix for output files.")
    parser.add_argument("--voxel_size", type=float, default=0.5, help="New voxel size (default: 0.5 mm).")
    parser.add_argument("--new_dim", type=int, nargs=3, default=[116, 140, 96], help="New image dimensions (x, y, z).")
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel jobs (-1 for all CPUs).")
    parser.add_argument("--patch_center", type=float, nargs=3, default=None, help="Optional patch center in mm.")
    parser.add_argument("--reduction", type=str, choices=["mip", "mean"], default=None,
                        help="Optional reduction along z-axis.")
    parser.add_argument("--use_gpu", type=lambda x: str(x).lower() != 'false', nargs='?', const=True, default=True,
                        help="Use GPU acceleration (default: True). Set to False with --use_gpu=False")
    parser.add_argument("--cpu", action="store_true", help="Force CPU processing (disables GPU).")
    parser.add_argument("--interp", type=str, choices=["hermite", "linear", "rbf"], default="hermite",
                        help="Interpolation method for streamlines (default: hermite).")
    parser.add_argument("--step_size", type=float, default=0.5, 
                        help="Step size for streamline densification (default: 0.5).")
    parser.add_argument("--max_gb", type=float, default=64.0,
                        help="Maximum output size in GB (default: 64.0). Dimensions will be automatically reduced if exceeded.")

    args = parser.parse_args()

    # Check for very large dimensions that might cause memory issues
    requested_dim = tuple(args.new_dim)
    if np.prod(requested_dim) > 100_000_000:  # More than 100M voxels
        print(f"WARNING: Requested dimensions {requested_dim} are very large!")
        print(f"Consider using lower-resolution dimensions or smaller voxel size.")
    
    # Determine GPU usage based on command line arguments
    use_gpu = not args.cpu and args.use_gpu  # If --cpu is specified or --use_gpu=False, use_gpu will be False
    
    print(f"Processing mode: {'CPU' if not use_gpu else 'GPU'}")
    
    # For compatibility with original argument names
    old_nifti_path = args.input
    old_trk_path = args.trk
    output_prefix = args.output
    
    process_and_save(
        old_nifti_path=old_nifti_path,
        old_trk_path=old_trk_path,
        new_voxel_size=args.voxel_size,
        new_dim=tuple(args.new_dim),
        output_prefix=output_prefix,
        n_jobs=args.jobs,
        patch_center=tuple(args.patch_center) if args.patch_center else None,
        reduction_method=args.reduction,
        use_gpu=use_gpu,
        interp_method=args.interp,
        step_size=args.step_size,
        max_output_gb=args.max_gb
    )
