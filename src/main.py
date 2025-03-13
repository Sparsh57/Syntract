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
        disable_clipping=False,
        high_res_mode=False
):
    """
    Process and resample NIfTI and tractography data.
    
    Parameters
    ----------
    old_nifti_path : str
        Path to input NIfTI file.
    old_trk_path : str
        Path to input TRK file.
    new_voxel_size : float, optional
        New voxel size, by default 0.5.
    new_dim : tuple, optional
        New dimensions, by default (116, 140, 96).
    output_prefix : str, optional
        Output file prefix, by default "resampled".
    n_jobs : int, optional
        Number of parallel jobs, by default 8.
    patch_center : tuple, optional
        Center point in mm, by default None.
    reduction_method : str, optional
        Reduction method (mip or mean), by default None.
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.
    interp_method : str, optional
        Interpolation method for streamlines ('hermite' or 'linear'), by default 'hermite'.
    step_size : float, optional
        Step size for streamline densification, by default 0.5.
    disable_clipping : bool, optional
        Whether to disable FOV clipping, by default False. Useful for high-resolution data.
    high_res_mode : bool, optional
        Whether to use special high-resolution processing mode, by default False.
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
    
    # Calculate ratio between old and new voxel sizes
    # Significant reduction in voxel size should trigger high-res mode
    min_old_voxel = float(min(old_voxel_sizes))
    voxel_ratio = min_old_voxel / new_voxel_size
    
    # Print detailed diagnostic information
    print(f"\n=== Voxel Size Analysis ===")
    print(f"Original min voxel size: {min_old_voxel:.4f}mm")
    print(f"Target voxel size: {new_voxel_size:.4f}mm")
    print(f"Voxel ratio: {voxel_ratio:.1f}x")
    
    # Always check if we're in high-res territory first
    is_high_res = new_voxel_size < 0.1  # Consider 100 microns or less as high-res
    
    # For significant voxel size changes, automatically use high_res_mode
    # Lower the threshold to 5x from 10x
    if voxel_ratio > 5 and not high_res_mode:
        high_res_mode = True
        print(f"\nAUTOMATICALLY ACTIVATING HIGH-RESOLUTION MODE: Voxel ratio {voxel_ratio:.1f}x exceeds threshold")
        print("Using special processing pathway for extreme resolution changes")
    elif new_voxel_size <= 0.05 and not high_res_mode:  # 50 microns or less
        high_res_mode = True
        print(f"\nAUTOMATICALLY ACTIVATING HIGH-RESOLUTION MODE: Target voxel size {new_voxel_size:.4f}mm is very small")
    
    # Get dimensions explicitly for tracking
    original_dim = new_dim
    
    # High-resolution mode - directly calculate dimensions based on ratio
    if high_res_mode:
        # Calculate new dimensions directly from voxel ratio
        print("\n====== HIGH RESOLUTION MODE ACTIVATED ======")
        print(f"Input voxel size: {min_old_voxel:.4f}mm")
        print(f"Target voxel size: {new_voxel_size:.4f}mm")
        print(f"Voxel ratio: {voxel_ratio:.1f}x")
        
        # Scale dimensions directly based on ratio, ensuring integer scaling
        # Round up the scale factor to ensure we don't clip any data
        scale_factor = max(1, int(np.ceil(voxel_ratio)))
        
        # Scale from original dimensions
        scaled_dim = tuple(int(d * scale_factor) for d in old_shape)
        
        print(f"Original dimensions: {old_shape}")
        print(f"Scale factor: {scale_factor}x (derived from voxel ratio {voxel_ratio:.1f}x)")
        print(f"New dimensions (scaled by {scale_factor}x): {scaled_dim}")
        
        # Override the new dimensions with scaled dimensions
        new_dim = scaled_dim
        
        # Force disable clipping in high resolution mode
        if not disable_clipping:
            disable_clipping = True
            print("FOV clipping automatically disabled for high resolution mode")
        print("=============================================\n")
    elif is_high_res:
        # Standard high-res handling (not extreme)
        auto_scale = min(50, max(2, int(np.ceil(voxel_ratio))))
        
        # Scale either from user-provided dimensions or from original dimensions
        if any(d <= 0 for d in new_dim):
            # If any dimension is invalid, use old_shape as base
            scaled_dim = tuple(int(d * auto_scale) for d in old_shape)
            print(f"\nAUTO-SCALING: Using original dimensions {old_shape} as base")
        else:
            scaled_dim = tuple(int(d * auto_scale) for d in new_dim)
            print(f"\nAUTO-SCALING: Using provided dimensions {new_dim} as base")
            
        new_dim = scaled_dim
        print(f"Auto-scale factor: {auto_scale}x (based on voxel ratio: {voxel_ratio:.1f})")
        print(f"Scaled dimensions: {new_dim}")
        
        # Enable disable_clipping automatically for high-resolution cases
        if not disable_clipping:
            disable_clipping = True
            print(f"\nAUTO-DISABLING FOV clipping for high-resolution data ({new_voxel_size}mm)")
    
    print("\n=== Building new affine ===")
    print(f"Using dimensions for affine: {new_dim}")
    A_new = build_new_affine(old_affine, old_shape, new_voxel_size, new_dim, patch_center_mm=patch_center, use_gpu=use_gpu)

    print(f"New affine:\n{A_new}")
    print(f"New dimensions: {new_dim}")
    print(f"Using these dimensions for ALL subsequent processing")
    
    print(f"\n=== Resampling NIfTI using {'GPU' if use_gpu else 'CPU'} ===")
    print(f"Resampling to dimensions: {new_dim}")
    new_data, tmp_mmap = resample_nifti(old_img, A_new, new_dim, chunk_size=(64, 64, 64), n_jobs=n_jobs, use_gpu=use_gpu)
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

    # For high resolution mode, use a step size that's appropriate
    if high_res_mode:
        # Adjust step size based on voxel size for high-res mode
        recommended_step = new_voxel_size * 2.0
        if step_size < recommended_step * 0.5 or step_size > recommended_step * 5.0:
            old_step_size = step_size
            step_size = recommended_step
            print(f"\nAutomatically adjusting step size for high resolution mode:")
            print(f"Original step size: {old_step_size}mm")
            print(f"New step size: {step_size}mm (2x voxel size)")
    
    print(f"\n=== Transforming, Densifying, and Clipping Streamlines Using {'GPU' if use_gpu else 'CPU'} with {interp_method} interpolation ===")
    print(f"Step size: {step_size}, Voxel size: {new_voxel_size}")
    print(f"FOV clipping: {'Disabled' if disable_clipping else 'Enabled'}")
    print(f"Using dimensions: {new_dim}")
    
    # Special high-resolution processing mode - adds extra debug and forces direct processing
    if high_res_mode:
        print("\n====== PROCESSING STREAMLINES IN HIGH-RESOLUTION MODE ======")
        print("Using direct processing method to preserve all streamlines")
        print(f"Dimensions: {new_dim}, Voxel size: {new_voxel_size}mm, Step size: {step_size}mm")
        
        if not disable_clipping:
            print("WARNING: FOV clipping should be disabled in high-res mode")
            disable_clipping = True
            print("Automatically disabling FOV clipping")
        
        print("FOV clipping forcibly disabled")
        
        # Force printing of bypass debug messages
        os.environ["DEBUG_STREAMLINE_BYPASS"] = "1"
    
    # High-res or normal processing
    densified_vox = transform_and_densify_streamlines(
        old_streams_mm, A_new, new_dim, step_size=step_size, n_jobs=n_jobs, 
        use_gpu=use_gpu, interp_method=interp_method, disable_clipping=disable_clipping,
        high_res_mode=high_res_mode
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

    parser.add_argument("old_nifti_path", type=str, help="Path to input NIfTI (.nii or .nii.gz) file.")
    parser.add_argument("old_trk_path", type=str, help="Path to input TRK (.trk) file.")
    parser.add_argument("--voxel_size", type=float, default=0.5, help="New voxel size (default: 0.5 mm).")
    parser.add_argument("--new_dim", type=int, nargs=3, default=[116, 140, 96], help="New image dimensions (x, y, z).")
    parser.add_argument("--output_prefix", type=str, default="resampled", help="Prefix for output files.")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel jobs (-1 for all CPUs).")
    parser.add_argument("--patch_center", type=float, nargs=3, default=None, help="Optional patch center in mm.")
    parser.add_argument("--reduction", type=str, choices=["mip", "mean"], default=None,
                        help="Optional reduction along z-axis.")
    parser.add_argument("--use_gpu", action="store_true", default=True, 
                        help="Use GPU acceleration (default: True). Set to false with --use_gpu=False.")
    parser.add_argument("--interp_method", type=str, choices=["hermite", "linear"], default="hermite",
                        help="Interpolation method for streamlines (default: hermite).")
    parser.add_argument("--step_size", type=float, default=0.5, 
                        help="Step size for streamline densification (default: 0.5).")
    parser.add_argument("--disable_clipping", action="store_true", default=False,
                        help="Disable FOV clipping to retain all streamlines (useful for high-resolution data).")
    parser.add_argument("--high_res_mode", action="store_true", default=False,
                        help="Use special high-resolution processing mode for extreme resolution changes.")

    args = parser.parse_args()

    process_and_save(
        old_nifti_path=args.old_nifti_path,
        old_trk_path=args.old_trk_path,
        new_voxel_size=args.voxel_size,
        new_dim=tuple(args.new_dim),
        output_prefix=args.output_prefix,
        n_jobs=args.n_jobs,
        patch_center=tuple(args.patch_center) if args.patch_center else None,
        reduction_method=args.reduction,
        use_gpu=args.use_gpu,
        interp_method=args.interp_method,
        step_size=args.step_size,
        disable_clipping=args.disable_clipping,
        high_res_mode=args.high_res_mode
    )
