import argparse
import nibabel as nib
import numpy as np
import os
from nifti_preprocessing import resample_nifti
from transform import build_new_affine
from streamline_processing import transform_and_densify_streamlines, clip_streamline_to_fov
from densify import densify_streamline_subvoxel
from nibabel.streamlines import Tractogram, save as save_trk

# Import the ANTs processing module
from ants_transform import process_with_ants

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
        max_output_gb=64.0,
        # ANTs transform parameters
        use_ants=False,
        ants_warp_path=None,
        ants_iwarp_path=None,
        ants_aff_path=None,
        force_dimensions=False,
        transform_mri_with_ants=False
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
    use_ants : bool, optional
        Whether to use ANTs transforms, default is False.
    ants_warp_path : str, optional
        Path to ANTs warp file. Required if use_ants is True.
    ants_iwarp_path : str, optional
        Path to ANTs inverse warp file. Required if use_ants is True.
    ants_aff_path : str, optional
        Path to ANTs affine file. Required if use_ants is True.
    force_dimensions : bool, optional
        Force using the specified new_dim even when using ANTs (may cause misalignment)
    transform_mri_with_ants : bool, optional
        Whether to transform the MRI with ANTs (default: False). If False, only transform streamlines.
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

    # Check if ANTs transform is requested
    if use_ants:
        # Validate required parameters for ANTs transform
        if not all([ants_warp_path, ants_iwarp_path, ants_aff_path]):
            raise ValueError("When use_ants=True, all ANTs transform paths must be provided.")
        
        # Get the dimensions from the warp file to inform the user
        warp_img = nib.load(ants_warp_path)
        warp_shape = warp_img.shape[:3]
        print(f"ANTs warp field dimensions: {warp_shape}")
        
        # Intermediate outputs (not final)
        ants_mri_output = f"{output_prefix}_ants_intermediate_mri.nii.gz" if transform_mri_with_ants else None
        ants_trk_output = f"{output_prefix}_ants_intermediate_trk.trk"
        
        print("\n=== Step 1: Applying ANTs Transforms ===")
        if not transform_mri_with_ants:
            print("Note: MRI transformation will be skipped as requested")
        
        # Apply ANTs transforms
        moved_mri, affine_vox2fix, transformed_streamlines, streamlines_voxel = process_with_ants(
            ants_warp_path, 
            ants_iwarp_path, 
            ants_aff_path, 
            old_nifti_path, 
            old_trk_path, 
            ants_mri_output, 
            ants_trk_output,
            transform_mri=transform_mri_with_ants
        )
        
        print("\n==== ANTs Transform Process Completed! ====")
        
        if transform_mri_with_ants:
            print(f"ANTs-transformed MRI dimensions: {moved_mri.shape[:3]}")
            print(f"Requested output dimensions: {new_dim}")
            
            # Use the ANTs-transformed data as starting point
            print("\n=== Step 2: Resampling ANTs-transformed data to requested dimensions ===")
            
            # Create a NIfTI image from the transformed data
            old_img = nib.Nifti1Image(moved_mri, affine_vox2fix)
            old_affine = affine_vox2fix
            old_shape = moved_mri.shape[:3]
            
            if old_shape != new_dim:
                print(f"Note: Resampling from ANTs dimensions {old_shape} to requested dimensions {new_dim}")
                print("This second transformation will maintain alignment between MRI and streamlines.")
            else:
                print(f"Requested dimensions {new_dim} already match ANTs transform dimensions.")
                print("No additional resampling needed.")
        else:
            # Use the original MRI file directly
            print("\n=== Step 2: Using original MRI directly ===")
            old_img = nib.load(old_nifti_path, mmap=True)
            old_affine = old_img.affine
            old_shape = old_img.shape[:3]
            print(f"Original MRI dimensions: {old_shape}")
            print(f"Requested output dimensions: {new_dim}")
        
        # Transform the streamlines to RAS coordinates for consistent resampling
        print("\n=== Converting streamlines to RAS coordinates for resampling ===")
        streamlines_ras = []
        
        for streamline in streamlines_voxel:
            # Convert from voxel to RAS coordinates
            ras_streamline = np.dot(streamline, affine_vox2fix[:3, :3].T) + affine_vox2fix[:3, 3]
            streamlines_ras.append(ras_streamline)
        
        print(f"Converted {len(streamlines_ras)} streamlines to RAS coordinates")
        
        # Create a tractogram object for further processing
        old_streams_mm = streamlines_ras
        
        """
        IMPORTANT: The key to maintaining alignment when using custom dimensions with ANTs:
        
        1. We first apply the ANTs transformation to both the MRI and streamlines
           - This puts both in the ANTs "fixed" space with its own dimensions
        
        2. Then we convert the streamlines from ANTs voxel space to world (RAS) coordinates
           - This step is crucial because world coordinates are independent of voxel grid
        
        3. Now both the MRI and streamlines can be consistently resampled to any new dimension
           - The MRI is resampled with its new affine matrix (A_new)
           - The streamlines are transformed using the same A_new matrix
           - This maintains their spatial relationship regardless of dimensions
        
        This approach ensures alignment between MRI and streamlines even when using
        dimensions that differ from the ANTs warp field dimensions.
        """
    else:
        print("\n=== Loading NIfTI ===")
        old_img = nib.load(old_nifti_path, mmap=True)
        old_affine = old_img.affine
        old_shape = old_img.shape[:3]
        
        print("\n=== Loading Tractography Data ===")
        trk_obj = nib.streamlines.load(old_trk_path)
        old_streams_mm = trk_obj.tractogram.streamlines

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
    if use_ants:
        # In the ANTs case, we need to handle the transform_and_densify process differently
        # because we've already converted our streamlines to RAS coordinates
        print(f"\n=== Transforming, Densifying, and Clipping ANTs-Processed Streamlines ===")
        print(f"Using {interp_method} interpolation with step size: {step_size}")
        print(f"Using dimensions: {new_dim}")
        
        # Apply the new affine transformation to convert world coordinates to the new voxel space
        A_new_inv = np.linalg.inv(A_new)
        
        # Convert streamlines from RAS (world) coordinates to the new voxel space
        voxel_streamlines = []
        for streamline in old_streams_mm:
            # Convert from RAS to voxel coordinates in new space
            voxel_streamline = np.dot(streamline, A_new_inv[:3, :3].T) + A_new_inv[:3, 3]
            voxel_streamlines.append(voxel_streamline)
        
        print(f"Transformed {len(voxel_streamlines)} streamlines to new voxel space")
        
        # Now clip the streamlines to the new dimensions and densify
        densified_vox = []
        
        # Process streamlines in parallel for performance
        from joblib import Parallel, delayed
        
        def process_streamline(streamline):
            try:
                # Clip to new FOV
                clipped_segments = clip_streamline_to_fov(streamline, new_dim, use_gpu=use_gpu)
                
                # Densify each segment
                densified_segments = []
                for segment in clipped_segments:
                    if len(segment) >= 2:  # Need at least 2 points for densification
                        try:
                            densified = densify_streamline_subvoxel(
                                segment, step_size=step_size, 
                                interp_method=interp_method, use_gpu=use_gpu
                            )
                            if len(densified) >= 2:
                                densified_segments.append(densified)
                        except Exception as e:
                            print(f"Error densifying segment: {e}")
                            # For troubleshooting, still include the non-densified segment
                            densified_segments.append(segment)
                            
                return densified_segments
            except Exception as e:
                print(f"Error processing streamline: {e}")
                return []
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_streamline)(streamline) for streamline in voxel_streamlines
        )
        
        # Flatten the results
        for result in results:
            densified_vox.extend(result)
            
        print(f"Processed {len(voxel_streamlines)} streamlines into {len(densified_vox)} segments")
    else:
        # Standard processing for non-ANTs case
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

    # Prepare header for saving the final tractogram
    if use_ants:
        # Create a new header for ANTs-transformed data
        new_trk_header = {
            "dimensions": np.array(new_dim, dtype=np.int16),
            "voxel_sizes": np.sqrt(np.sum(A_new[:3, :3] ** 2, axis=0)).astype(np.float32),
            "voxel_to_rasmm": A_new.astype(np.float32)
        }
    else:
        # Use the header from the original tract object
        trk_obj = nib.streamlines.load(old_trk_path)
        new_trk_header = trk_obj.header.copy()
        new_trk_header["dimensions"] = np.array(new_dim, dtype=np.int16)
        new_voxsize = np.sqrt(np.sum(A_new[:3, :3] ** 2, axis=0))
        new_trk_header["voxel_sizes"] = new_voxsize.astype(np.float32)
        new_trk_header["voxel_to_rasmm"] = A_new.astype(np.float32)

    print("\n=== Saving Final Synthesized .trk File ===")
    
    # The streamlines in densified_vox are in voxel coordinates
    # We need to convert them to RAS coordinates for the tractogram
    if len(densified_vox) > 0:
        # Convert streamlines from voxel coordinates to RAS coordinates
        print(f"Converting {len(densified_vox)} streamlines from voxel to RAS coordinates...")
        ras_streamlines = []
        for streamline in densified_vox:
            # Apply affine transform to convert from voxel to RAS coordinates
            ras_streamline = np.dot(streamline, A_new[:3, :3].T) + A_new[:3, 3]
            ras_streamlines.append(ras_streamline)
            
        new_tractogram = Tractogram(ras_streamlines, affine_to_rasmm=np.eye(4))
    else:
        print("WARNING: No valid streamlines to save after processing!")
        new_tractogram = Tractogram([], affine_to_rasmm=np.eye(4))
    
    out_trk_path = output_prefix + ".trk"
    save_trk(new_tractogram, out_trk_path, header=new_trk_header)

    print(f"Saved new .trk => {out_trk_path}")
    print(f"Number of streamlines in final .trk file: {len(new_tractogram)}")
    print("\n==== Synthesis Process Completed Successfully! ====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and resample NIfTI and streamline tractography data.")

    parser.add_argument("--input", type=str, required=True, help="Path to input NIfTI (.nii or .nii.gz) file.")
    parser.add_argument("--trk", type=str, required=True, help="Path to input TRK (.trk) file.")
    parser.add_argument("--output", type=str, default="resampled", help="Prefix for output files.")
    parser.add_argument("--voxel_size", type=float, nargs='+', default=[0.5],
                        help="New voxel size: either a single value for isotropic or three values for anisotropic (e.g., --voxel_size 0.5 0.5 1.0)")
    parser.add_argument("--new_dim", type=int, nargs=3, default=[116, 140, 96], 
                        help="New image dimensions (x, y, z). When using --use_ants, these dimensions will be used for the final output (alignment is maintained).")
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
    
    # Add ANTs transform related arguments
    parser.add_argument("--use_ants", action="store_true", help="Use ANTs transforms for processing.")
    parser.add_argument("--ants_warp", type=str, default=None, help="Path to ANTs warp file (required if use_ants is True).")
    parser.add_argument("--ants_iwarp", type=str, default=None, help="Path to ANTs inverse warp file (required if use_ants is True).")
    parser.add_argument("--ants_aff", type=str, default=None, help="Path to ANTs affine file (required if use_ants is True).")
    parser.add_argument("--force_dimensions", action="store_true", help="Force using the specified new_dim even when using ANTs (may cause misalignment)")
    parser.add_argument("--transform_mri_with_ants", action="store_true", help="Also transform MRI with ANTs (default: only transforms streamlines)")

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
    
    # Handle voxel size: allow either a single float or three floats
    if len(args.voxel_size) == 1:
        voxel_size = args.voxel_size[0]
    elif len(args.voxel_size) == 3:
        voxel_size = tuple(args.voxel_size)
    else:
        raise ValueError("--voxel_size must be either one value (isotropic) or three values (anisotropic)")
    
    # Validate ANTs transform arguments
    if args.use_ants and not all([args.ants_warp, args.ants_iwarp, args.ants_aff]):
        parser.error("When --use_ants is specified, --ants_warp, --ants_iwarp, and --ants_aff must be provided.")
    
    process_and_save(
        old_nifti_path=old_nifti_path,
        old_trk_path=old_trk_path,
        new_voxel_size=voxel_size,
        new_dim=tuple(args.new_dim),
        output_prefix=output_prefix,
        n_jobs=args.jobs,
        patch_center=tuple(args.patch_center) if args.patch_center else None,
        reduction_method=args.reduction,
        use_gpu=use_gpu,
        interp_method=args.interp,
        step_size=args.step_size,
        max_output_gb=args.max_gb,
        use_ants=args.use_ants,
        ants_warp_path=args.ants_warp,
        ants_iwarp_path=args.ants_iwarp,
        ants_aff_path=args.ants_aff,
        force_dimensions=args.force_dimensions,
        transform_mri_with_ants=args.transform_mri_with_ants
    )
