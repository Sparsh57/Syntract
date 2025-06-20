#!/usr/bin/env python
"""
Main Syntract Processing Pipeline

This script serves as the entry point for the Syntract tractography synthesis pipeline.
It orchestrates the complete workflow from raw NIfTI and TRK files to processed,
resampled, and transformed outputs suitable for analysis and visualization.

The pipeline supports:
- Advanced streamline interpolation (Linear, Hermite, RBF)
- GPU-accelerated processing with automatic CPU fallback
- ANTs-based spatial transformations
- Flexible output formatting and memory management

Usage:
    python main.py --input brain.nii.gz --trk fibers.trk [options]

Authors: Sparsh Makharia, LINC Team
License: MIT
"""

import argparse
import os
import sys
import time
import numpy as np
import nibabel as nib
from .nifti_preprocessing import resample_nifti
from .transform import build_new_affine
from .streamline_processing import transform_and_densify_streamlines, clip_streamline_to_fov
from .densify import densify_streamline_subvoxel
from nibabel.streamlines import Tractogram, save as save_trk

from .ants_transform import process_with_ants

def process_and_save(
        original_nifti_path,
        original_trk_path,
        target_voxel_size=0.5,
        target_dimensions=(116, 140, 96),
        output_prefix="resampled",
        num_jobs=8,
        patch_center=None,
        reduction_method=None,
        use_gpu=True,
        interpolation_method='hermite',
        step_size=0.5,
        max_output_gb=64.0,
        use_ants=False,
        ants_warp_path=None,
        ants_iwarp_path=None,
        ants_aff_path=None,
        force_dimensions=False,
        transform_mri_with_ants=False
):
    """Process and save NIfTI and streamline data with new parameters."""
    
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

    if use_ants:
        if not all([ants_warp_path, ants_iwarp_path, ants_aff_path]):
            raise ValueError("When use_ants=True, all ANTs transform paths must be provided.")
        
        warp_img = nib.load(ants_warp_path)
        warp_shape = warp_img.shape[:3]
        print(f"ANTs warp field dimensions: {warp_shape}")
        
        ants_mri_output = f"{output_prefix}_ants_intermediate_mri.nii.gz" if transform_mri_with_ants else None
        ants_trk_output = f"{output_prefix}_ants_intermediate_trk.trk"
        
        print("\n=== Step 1: Applying ANTs Transforms ===")
        if not transform_mri_with_ants:
            print("Note: MRI transformation will be skipped as requested")
        
        moved_mri, affine_vox2fix, transformed_streamlines, streamlines_voxel = process_with_ants(
            ants_warp_path, 
            ants_iwarp_path, 
            ants_aff_path, 
            original_nifti_path, 
            original_trk_path, 
            ants_mri_output, 
            ants_trk_output,
            transform_mri=transform_mri_with_ants
        )
        
        print("\n==== ANTs Transform Process Completed! ====")
        
        if transform_mri_with_ants:
            print(f"ANTs-transformed MRI dimensions: {moved_mri.shape[:3]}")
            print(f"Requested output dimensions: {target_dimensions}")
            
            print("\n=== Step 2: Resampling ANTs-transformed data to requested dimensions ===")
            
            old_img = nib.Nifti1Image(moved_mri, affine_vox2fix)
            old_affine = affine_vox2fix
            old_shape = moved_mri.shape[:3]
            
            if old_shape != target_dimensions:
                print(f"Note: Resampling from ANTs dimensions {old_shape} to requested dimensions {target_dimensions}")
                print("This second transformation will maintain alignment between MRI and streamlines.")
            else:
                print(f"Requested dimensions {target_dimensions} already match ANTs transform dimensions.")
                print("No additional resampling needed.")
        else:
            print("\n=== Step 2: Using original MRI directly ===")
            old_img = nib.load(original_nifti_path, mmap=True)
            old_affine = old_img.affine
            old_shape = old_img.shape[:3]
            print(f"Original MRI dimensions: {old_shape}")
            print(f"Requested output dimensions: {target_dimensions}")
        
        print("\n=== Converting streamlines to RAS coordinates for resampling ===")
        streamlines_ras = []
        
        for streamline in streamlines_voxel:
            ras_streamline = np.dot(streamline, affine_vox2fix[:3, :3].T) + affine_vox2fix[:3, 3]
            streamlines_ras.append(ras_streamline)
        
        print(f"Converted {len(streamlines_ras)} streamlines to RAS coordinates")
        
        old_streams_mm = streamlines_ras
        
    else:
        print("\n=== Loading NIfTI ===")
        old_img = nib.load(original_nifti_path, mmap=True)
        old_affine = old_img.affine
        old_shape = old_img.shape[:3]
        
        print("\n=== Loading Tractography Data ===")
        trk_obj = nib.streamlines.load(original_trk_path)
        old_streams_mm = trk_obj.tractogram.streamlines

    old_voxel_sizes = np.array(old_img.header.get_zooms()[:3])

    print(f"Old shape: {old_shape}")
    print(f"Old voxel sizes: {old_voxel_sizes}")
    print(f"Old affine:\n{old_affine}")
    
    print("\n=== Building new affine ===")
    print(f"Using dimensions for affine: {target_dimensions}")
    A_new = build_new_affine(old_affine, old_shape, target_voxel_size, target_dimensions, patch_center_mm=patch_center, use_gpu=use_gpu)

    print(f"New affine:\n{A_new}")
    print(f"New dimensions: {target_dimensions}")
    print(f"Using these dimensions for ALL subsequent processing")
    
    print(f"\n=== Resampling NIfTI using {'GPU' if use_gpu else 'CPU'} ===")
    print(f"Resampling to dimensions: {target_dimensions}")
    print(f"Memory limit: {max_output_gb} GB")
    new_data, tmp_mmap = resample_nifti(old_img, A_new, target_dimensions, chunk_size=(64, 64, 64), n_jobs=num_jobs, use_gpu=use_gpu, max_output_gb=max_output_gb)
    print(f"Resampled data shape: {new_data.shape}")
    
    if new_data.shape[:3] != target_dimensions:
        print(f"WARNING: Resampled shape {new_data.shape[:3]} does not match expected dimensions {target_dimensions}")
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
            
            reduced_data = reduced_data[..., xp.newaxis]
            new_data = reduced_data
        else:
            if reduction_method == 'mip':
                reduced_data = np.max(new_data, axis=1)
            elif reduction_method == 'mean':
                reduced_data = np.mean(new_data, axis=1)
            else:
                raise ValueError(f"Unsupported reduction method: {reduction_method}")
            
            reduced_data = reduced_data[..., np.newaxis]
            new_data = reduced_data
            
        target_dimensions = (target_dimensions[0], 1, target_dimensions[2])

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
    
    total_points = sum(len(s) for s in old_streams_mm)
    avg_points = total_points / len(old_streams_mm) if len(old_streams_mm) > 0 else 0
    print(f"Total points in original streamlines: {total_points}")
    print(f"Average points per streamline: {avg_points:.2f}")
    
    print(f"\n=== Transforming, Densifying, and Clipping Streamlines Using {'GPU' if use_gpu else 'CPU'} with {interpolation_method} interpolation ===")
    print(f"Step size: {step_size}, Voxel size: {target_voxel_size}")
    print(f"FOV clipping: Enabled")
    print(f"Using dimensions: {target_dimensions}")
    
    if use_ants:
        print(f"\n=== Transforming, Densifying, and Clipping ANTs-Processed Streamlines ===")
        print(f"Using {interpolation_method} interpolation with step size: {step_size}")
        print(f"Using dimensions: {target_dimensions}")
        
        A_new_inv = np.linalg.inv(A_new)
        
        voxel_streamlines = []
        for streamline in old_streams_mm:
            voxel_streamline = np.dot(streamline, A_new_inv[:3, :3].T) + A_new_inv[:3, 3]
            voxel_streamlines.append(voxel_streamline)
        
        print(f"Transformed {len(voxel_streamlines)} streamlines to new voxel space")
        
        densified_vox = []
        
        from joblib import Parallel, delayed
        
        def process_streamline(streamline):
            try:
                clipped_segments = clip_streamline_to_fov(streamline, target_dimensions, use_gpu=use_gpu)
                
                densified_segments = []
                for segment in clipped_segments:
                    if len(segment) >= 2:
                        try:
                            densified = densify_streamline_subvoxel(
                                segment, step_size=step_size, 
                                interp_method=interpolation_method, use_gpu=use_gpu
                            )
                            if len(densified) >= 2:
                                densified_segments.append(densified)
                        except Exception as e:
                            print(f"Error densifying segment: {e}")
                            densified_segments.append(segment)
                            
                return densified_segments
            except Exception as e:
                print(f"Error processing streamline: {e}")
                return []
        
        results = Parallel(num_jobs)(
            delayed(process_streamline)(streamline) for streamline in voxel_streamlines
        )
        
        for result in results:
            densified_vox.extend(result)
            
        print(f"Processed {len(voxel_streamlines)} streamlines into {len(densified_vox)} segments")
    else:
        densified_vox = transform_and_densify_streamlines(
            old_streams_mm, A_new, target_dimensions, step_size=step_size, n_jobs=num_jobs, 
            use_gpu=use_gpu, interp_method=interpolation_method
        )
    
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

    if use_ants:
        new_trk_header = {
            "dimensions": np.array(target_dimensions, dtype=np.int16),
            "voxel_sizes": np.sqrt(np.sum(A_new[:3, :3] ** 2, axis=0)).astype(np.float32),
            "voxel_to_rasmm": A_new.astype(np.float32)
        }
    else:
        trk_obj = nib.streamlines.load(original_trk_path)
        new_trk_header = trk_obj.header.copy()
        new_trk_header["dimensions"] = np.array(target_dimensions, dtype=np.int16)
        new_voxsize = np.sqrt(np.sum(A_new[:3, :3] ** 2, axis=0))
        new_trk_header["voxel_sizes"] = new_voxsize.astype(np.float32)
        new_trk_header["voxel_to_rasmm"] = A_new.astype(np.float32)

    print("\n=== Saving Final Synthesized .trk File ===")
    
    if len(densified_vox) > 0:
        print(f"Converting {len(densified_vox)} streamlines from voxel to RAS coordinates...")
        ras_streamlines = []
        for streamline in densified_vox:
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

def main():
    """Main entry point for the mri-synthesis console script."""
    import sys
    _run_main_with_args(sys.argv[1:])


def _run_main_with_args(args=None):
    """Internal function to run main with specific arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and resample NIfTI and streamline tractography data.")

    parser.add_argument("--input", type=str, required=True, help="Path to input NIfTI (.nii or .nii.gz) file.")
    parser.add_argument("--trk", type=str, required=True, help="Path to input TRK (.trk) file.")
    parser.add_argument("--output", type=str, default="resampled", help="Prefix for output files.")
    parser.add_argument("--voxel_size", type=float, nargs='+', default=[0.5],
                        help="New voxel size: either a single value for isotropic or three values for anisotropic")
    parser.add_argument("--new_dim", type=int, nargs=3, default=[116, 140, 96], 
                        help="New image dimensions (x, y, z). When using --use_ants, these dimensions will be used for the final output.")
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
                        help="Maximum output size in GB (default: 64.0).")
    
    parser.add_argument("--use_ants", action="store_true", help="Use ANTs transforms for processing.")
    parser.add_argument("--ants_warp", type=str, default=None, help="Path to ANTs warp file (required if use_ants is True).")
    parser.add_argument("--ants_iwarp", type=str, default=None, help="Path to ANTs inverse warp file (required if use_ants is True).")
    parser.add_argument("--ants_aff", type=str, default=None, help="Path to ANTs affine file (required if use_ants is True).")
    parser.add_argument("--force_dimensions", action="store_true", help="Force using the specified new_dim even when using ANTs")
    parser.add_argument("--transform_mri_with_ants", action="store_true", help="Also transform MRI with ANTs (default: only transforms streamlines)")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    requested_dim = tuple(args.new_dim)
    if np.prod(requested_dim) > 100_000_000:
        print(f"WARNING: Requested dimensions {requested_dim} are very large!")
        print(f"Consider using lower-resolution dimensions or smaller voxel size.")
    
    use_gpu = not args.cpu and args.use_gpu
    
    print(f"Processing mode: {'CPU' if not use_gpu else 'GPU'}")
    
    original_nifti_path = args.input
    original_trk_path = args.trk
    output_prefix = args.output
    
    if len(args.voxel_size) == 1:
        voxel_size = args.voxel_size[0]
    elif len(args.voxel_size) == 3:
        voxel_size = tuple(args.voxel_size)
    else:
        raise ValueError("--voxel_size must be either one value (isotropic) or three values (anisotropic)")
    
    if args.use_ants and not all([args.ants_warp, args.ants_iwarp, args.ants_aff]):
        parser.error("When --use_ants is specified, --ants_warp, --ants_iwarp, and --ants_aff must be provided.")
    
    process_and_save(
        original_nifti_path=original_nifti_path,
        original_trk_path=original_trk_path,
        target_voxel_size=voxel_size,
        target_dimensions=tuple(args.new_dim),
        output_prefix=output_prefix,
        num_jobs=args.jobs,
        patch_center=tuple(args.patch_center) if args.patch_center else None,
        reduction_method=args.reduction,
        use_gpu=use_gpu,
        interpolation_method=args.interp,
        step_size=args.step_size,
        max_output_gb=args.max_gb,
        use_ants=args.use_ants,
        ants_warp_path=args.ants_warp,
        ants_iwarp_path=args.ants_iwarp,
        ants_aff_path=args.ants_aff,
        force_dimensions=args.force_dimensions,
        transform_mri_with_ants=args.transform_mri_with_ants
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and resample NIfTI and streamline tractography data.")

    parser.add_argument("--input", type=str, required=True, help="Path to input NIfTI (.nii or .nii.gz) file.")
    parser.add_argument("--trk", type=str, required=True, help="Path to input TRK (.trk) file.")
    parser.add_argument("--output", type=str, default="resampled", help="Prefix for output files.")
    parser.add_argument("--voxel_size", type=float, nargs='+', default=[0.5],
                        help="New voxel size: either a single value for isotropic or three values for anisotropic")
    parser.add_argument("--new_dim", type=int, nargs=3, default=[116, 140, 96], 
                        help="New image dimensions (x, y, z). When using --use_ants, these dimensions will be used for the final output.")
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
                        help="Maximum output size in GB (default: 64.0).")
    
    parser.add_argument("--use_ants", action="store_true", help="Use ANTs transforms for processing.")
    parser.add_argument("--ants_warp", type=str, default=None, help="Path to ANTs warp file (required if use_ants is True).")
    parser.add_argument("--ants_iwarp", type=str, default=None, help="Path to ANTs inverse warp file (required if use_ants is True).")
    parser.add_argument("--ants_aff", type=str, default=None, help="Path to ANTs affine file (required if use_ants is True).")
    parser.add_argument("--force_dimensions", action="store_true", help="Force using the specified new_dim even when using ANTs")
    parser.add_argument("--transform_mri_with_ants", action="store_true", help="Also transform MRI with ANTs (default: only transforms streamlines)")

    args = parser.parse_args()

    requested_dim = tuple(args.new_dim)
    if np.prod(requested_dim) > 100_000_000:
        print(f"WARNING: Requested dimensions {requested_dim} are very large!")
        print(f"Consider using lower-resolution dimensions or smaller voxel size.")
    
    use_gpu = not args.cpu and args.use_gpu
    
    print(f"Processing mode: {'CPU' if not use_gpu else 'GPU'}")
    
    original_nifti_path = args.input
    original_trk_path = args.trk
    output_prefix = args.output
    
    if len(args.voxel_size) == 1:
        voxel_size = args.voxel_size[0]
    elif len(args.voxel_size) == 3:
        voxel_size = tuple(args.voxel_size)
    else:
        raise ValueError("--voxel_size must be either one value (isotropic) or three values (anisotropic)")
    
    if args.use_ants and not all([args.ants_warp, args.ants_iwarp, args.ants_aff]):
        parser.error("When --use_ants is specified, --ants_warp, --ants_iwarp, and --ants_aff must be provided.")
    
    process_and_save(
        original_nifti_path=original_nifti_path,
        original_trk_path=original_trk_path,
        target_voxel_size=voxel_size,
        target_dimensions=tuple(args.new_dim),
        output_prefix=output_prefix,
        num_jobs=args.jobs,
        patch_center=tuple(args.patch_center) if args.patch_center else None,
        reduction_method=args.reduction,
        use_gpu=use_gpu,
        interpolation_method=args.interp,
        step_size=args.step_size,
        max_output_gb=args.max_gb,
        use_ants=args.use_ants,
        ants_warp_path=args.ants_warp,
        ants_iwarp_path=args.ants_iwarp,
        ants_aff_path=args.ants_aff,
        force_dimensions=args.force_dimensions,
        transform_mri_with_ants=args.transform_mri_with_ants
    )
