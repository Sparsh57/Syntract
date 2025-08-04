#!/usr/bin/env python
"""
Combined MRI Synthesis and Visualization Pipeline

This script provides a unified interface for both processing NIfTI/TRK data
and generating visualizations. It combines the synthesis pipeline for data
processing with the syntract viewer for visualization generation.

Usage:
    python syntract.py --input brain.nii.gz --trk fibers.trk [options]
    
    Or programmatically:
    from syntract import process_mri_data
    result = process_mri_data('brain.nii.gz', 'fibers.trk', output='results')
"""

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union, Tuple, List

# Import synthesis functions
try:
    from synthesis.main import process_and_save
    SYNTHESIS_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'synthesis'))
        from main import process_and_save
        SYNTHESIS_AVAILABLE = True
    except ImportError:
        SYNTHESIS_AVAILABLE = False
        print("Warning: Synthesis module not available")

# Import syntract viewer functions  
try:
    from syntract_viewer.generate_fiber_examples import generate_examples_original_mode, generate_examples_with_spatial_subdivisions
    SYNTRACT_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'syntract_viewer'))
        from generate_fiber_examples import generate_examples_original_mode, generate_examples_with_spatial_subdivisions
        SYNTRACT_AVAILABLE = True
    except ImportError:
        SYNTRACT_AVAILABLE = False
        print("Warning: Syntract viewer module not available")


def run_synthesis_stage(args, temp_dir):
    """Run the synthesis processing stage."""
    if not SYNTHESIS_AVAILABLE:
        raise RuntimeError("Synthesis module not available. Cannot run processing stage.")
    
    print("\n" + "="*60)
    print("STAGE 1: SYNTHESIS PROCESSING")
    print("="*60)
    
    # Set up synthesis output paths
    synthesis_output = os.path.join(temp_dir, "processed")
    
    # Prepare synthesis parameters
    synthesis_args = {
        'original_nifti_path': args.input,
        'original_trk_path': args.trk,
        'target_voxel_size': args.voxel_size[0] if len(args.voxel_size) == 1 else tuple(args.voxel_size),
        'target_dimensions': tuple(args.new_dim),
        'output_prefix': synthesis_output,
        'num_jobs': args.jobs,
        'patch_center': tuple(args.patch_center) if args.patch_center else None,
        'reduction_method': args.reduction,
        'use_gpu': not args.cpu and args.use_gpu,
        'interpolation_method': args.interp,
        'step_size': args.step_size,
        'max_output_gb': args.max_gb,
        'use_ants': args.use_ants,
        'ants_warp_path': args.ants_warp,
        'ants_iwarp_path': args.ants_iwarp,
        'ants_aff_path': args.ants_aff,
        'force_dimensions': args.force_dimensions,
        'transform_mri_with_ants': args.transform_mri_with_ants
    }
    
    # Run synthesis
    process_and_save(**synthesis_args)
    
    # Return paths to processed files
    processed_nifti = synthesis_output + ".nii.gz"
    processed_trk = synthesis_output + ".trk"
    
    if not os.path.exists(processed_nifti) or not os.path.exists(processed_trk):
        raise RuntimeError("Synthesis stage failed to produce output files")
    
    print(f"Synthesis completed successfully!")
    print(f"Processed NIfTI: {processed_nifti}")
    print(f"Processed TRK: {processed_trk}")
    
    return processed_nifti, processed_trk


def run_visualization_stage(nifti_file, trk_file, args):
    """Run the visualization generation stage."""
    if not SYNTRACT_AVAILABLE:
        raise RuntimeError("Syntract viewer module not available. Cannot run visualization stage.")
    
    print("\n" + "="*60)
    print("STAGE 2: VISUALIZATION GENERATION")
    print("="*60)
    
    # Create output directory for visualizations
    viz_output_dir = args.viz_output_dir
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Prepare visualization arguments
    viz_args = argparse.Namespace()
    
    # Set all the visualization parameters  
    viz_args.nifti = nifti_file
    viz_args.trk = trk_file
    viz_args.output_dir = viz_output_dir
    viz_args.examples = args.n_examples  # Note: visualization code expects 'examples' not 'n_examples'
    viz_args.prefix = args.viz_prefix
    viz_args.view = args.slice_mode
    viz_args.specific_slice = args.specific_slice
    viz_args.streamline_percentage = args.streamline_percentage
    viz_args.roi_sphere = None  # Set to None as in original code
    viz_args.tract_linewidth = args.tract_linewidth
    viz_args.save_masks = args.save_masks
    viz_args.use_high_density_masks = args.use_high_density_masks
    viz_args.label_bundles = args.label_bundles
    viz_args.mask_thickness = args.mask_thickness
    viz_args.min_fiber_pct = args.min_fiber_percentage
    viz_args.max_fiber_pct = args.max_fiber_percentage
    viz_args.min_bundle_size = args.min_bundle_size
    viz_args.density_threshold = args.density_threshold
    viz_args.contrast_method = args.contrast_method
    viz_args.background_preset = args.background_preset
    viz_args.cornucopia_preset = args.cornucopia_preset
    viz_args.enable_sharpening = args.enable_sharpening
    viz_args.sharpening_strength = args.sharpening_strength
    viz_args.close_gaps = args.close_gaps
    viz_args.closing_footprint_size = args.closing_footprint_size
    viz_args.randomize = args.randomize_viz
    viz_args.random_state = args.random_state
    viz_args.spatial_subdivisions = args.use_spatial_subdivisions
    viz_args.n_subdivisions = args.n_subdivisions
    viz_args.max_streamlines_per_region = args.max_streamlines_per_subdivision
    
    # Import background enhancement availability
    try:
        from syntract_viewer.background_enhancement import enhance_slice_background
        viz_args.background_enhancement_available = True
    except ImportError:
        viz_args.background_enhancement_available = False
    
    # Run visualization generation
    if args.use_spatial_subdivisions:
        print("Using spatial subdivisions mode")
        generate_examples_with_spatial_subdivisions(viz_args, viz_args.background_enhancement_available)
    else:
        print("Using original mode")
        generate_examples_original_mode(viz_args, viz_args.background_enhancement_available)
    
    print(f"Visualization generation completed!")
    print(f"Output directory: {viz_output_dir}")


def copy_final_outputs(temp_dir, args):
    """Copy final outputs to the specified locations."""
    if args.keep_processed:
        # Copy processed files to final location
        synthesis_output = os.path.join(temp_dir, "processed")
        processed_nifti = synthesis_output + ".nii.gz"
        processed_trk = synthesis_output + ".trk"
        
        final_nifti = args.output + "_processed.nii.gz"
        final_trk = args.output + "_processed.trk"
        
        if os.path.exists(processed_nifti):
            shutil.copy2(processed_nifti, final_nifti)
            print(f"Saved processed NIfTI: {final_nifti}")
        
        if os.path.exists(processed_trk):
            shutil.copy2(processed_trk, final_trk)
            print(f"Saved processed TRK: {final_trk}")


def process_mri_data(
    input_nifti: str,
    input_trk: str,
    output: str = "combined_output",
    # Pipeline control
    skip_synthesis: bool = False,
    skip_visualization: bool = False,
    keep_processed: bool = False,
    # Synthesis parameters
    voxel_size: Union[float, List[float]] = 0.5,
    new_dim: Tuple[int, int, int] = (116, 140, 96),
    jobs: int = 8,
    patch_center: Optional[Tuple[float, float, float]] = None,
    reduction: Optional[str] = None,
    use_gpu: bool = True,
    interp: str = "hermite",
    step_size: float = 0.5,
    max_gb: float = 64.0,
    # ANTs parameters
    use_ants: bool = False,
    ants_warp: Optional[str] = None,
    ants_iwarp: Optional[str] = None,
    ants_aff: Optional[str] = None,
    force_dimensions: bool = False,
    transform_mri_with_ants: bool = False,
    # Visualization parameters
    viz_output_dir: Optional[str] = None,
    n_examples: int = 5,
    viz_prefix: str = "synthetic_",
    slice_mode: str = "coronal",
    specific_slice: Optional[int] = None,
    streamline_percentage: float = 100.0,
    tract_linewidth: float = 1.0,
    save_masks: bool = False,
    use_high_density_masks: bool = False,
    label_bundles: bool = False,
    mask_thickness: int = 1,
    min_fiber_percentage: float = 10.0,
    max_fiber_percentage: float = 100.0,
    min_bundle_size: int = 20,
    density_threshold: float = 0.15,
    # Enhancement parameters
    contrast_method: str = "clahe",
    background_preset: str = "preserve_edges",
    cornucopia_preset: Optional[str] = None,
    enable_sharpening: bool = True,
    sharpening_strength: float = 0.5,
    close_gaps: bool = False,
    closing_footprint_size: int = 5,
    # Spatial subdivisions parameters
    use_spatial_subdivisions: bool = False,
    n_subdivisions: int = 8,
    max_streamlines_per_subdivision: int = 50000,
    # Miscellaneous parameters
    randomize_viz: bool = False,
    random_state: Optional[int] = None,
    temp_dir: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Process MRI data through synthesis and visualization pipeline.
    
    Args:
        input_nifti: Path to input NIfTI (.nii or .nii.gz) file
        input_trk: Path to input TRK (.trk) file
        output: Base name for output files
        skip_synthesis: Skip synthesis stage and use input files directly
        skip_visualization: Skip visualization stage, only run synthesis
        keep_processed: Keep the processed NIfTI and TRK files from synthesis
        voxel_size: New voxel size (single value for isotropic or list for anisotropic)
        new_dim: New image dimensions (x, y, z)
        jobs: Number of parallel jobs (-1 for all CPUs)
        patch_center: Optional patch center in mm
        reduction: Optional reduction along z-axis ("mip" or "mean")
        use_gpu: Use GPU acceleration
        interp: Interpolation method ("hermite", "linear", "rbf")
        step_size: Step size for streamline densification
        max_gb: Maximum output size in GB
        use_ants: Use ANTs transforms for processing
        ants_warp: Path to ANTs warp file
        ants_iwarp: Path to ANTs inverse warp file
        ants_aff: Path to ANTs affine file
        force_dimensions: Force using specified new_dim even when using ANTs
        transform_mri_with_ants: Also transform MRI with ANTs
        viz_output_dir: Output directory for visualizations
        n_examples: Number of visualization examples to generate
        viz_prefix: Prefix for visualization files
        slice_mode: Slice orientation ("coronal", "axial", "sagittal")
        specific_slice: Specific slice number to visualize
        streamline_percentage: Percentage of streamlines to include
        tract_linewidth: Linewidth for tract visualization
        save_masks: Save fiber masks along with visualizations
        use_high_density_masks: Use masks from high-density fibers
        label_bundles: Label distinct fiber bundles in the masks
        mask_thickness: Thickness of the mask lines in pixels
        min_fiber_percentage: Minimum fiber percentage for visualization
        max_fiber_percentage: Maximum fiber percentage for visualization
        min_bundle_size: Minimum bundle size for fiber labeling
        density_threshold: Density threshold for fiber visualization
        contrast_method: Contrast enhancement method
        background_preset: Background enhancement preset
        cornucopia_preset: Cornucopia augmentation preset
        enable_sharpening: Enable sharpening in background enhancement
        sharpening_strength: Strength of sharpening effect
        close_gaps: Close gaps in fiber visualizations
        closing_footprint_size: Size of morphological closing footprint
        use_spatial_subdivisions: Use spatial subdivisions for visualization
        n_subdivisions: Number of spatial subdivisions
        max_streamlines_per_subdivision: Maximum streamlines per subdivision
        randomize_viz: Randomize visualization parameters
        random_state: Random seed for reproducible results
        temp_dir: Temporary directory for intermediate files
        verbose: Print progress messages
    
    Returns:
        dict: Dictionary containing output paths and status information
        
    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If required modules are unavailable or processing fails
    """
    
    # Validate arguments
    if skip_synthesis and skip_visualization:
        raise ValueError("Cannot skip both synthesis and visualization stages")
    
    if use_ants and not all([ants_warp, ants_iwarp, ants_aff]):
        raise ValueError("When use_ants=True, ants_warp, ants_iwarp, and ants_aff must be provided")
    
    if reduction and reduction not in ["mip", "mean"]:
        raise ValueError("reduction must be 'mip' or 'mean'")
        
    if slice_mode not in ["coronal", "axial", "sagittal"]:
        raise ValueError("slice_mode must be 'coronal', 'axial', or 'sagittal'")
    
    # Normalize voxel_size to list format
    if isinstance(voxel_size, (int, float)):
        voxel_size = [float(voxel_size)]
    else:
        voxel_size = [float(v) for v in voxel_size]
    
    # Set up output directory for visualizations
    if viz_output_dir is None:
        viz_output_dir = f"{output}_visualizations"
    
    # Set up temporary directory
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_temp = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="mri_pipeline_")
        cleanup_temp = True
    
    # Create a mock args object to reuse existing functions
    class Args:
        def __init__(self):
            # Input/Output
            self.input = input_nifti
            self.trk = input_trk
            self.output = output
            self.viz_output_dir = viz_output_dir
            self.keep_processed = keep_processed
            
            # Synthesis parameters
            self.voxel_size = voxel_size
            self.new_dim = new_dim
            self.jobs = jobs
            self.patch_center = patch_center
            self.reduction = reduction
            self.use_gpu = use_gpu
            self.cpu = not use_gpu
            self.interp = interp
            self.step_size = step_size
            self.max_gb = max_gb
            
            # ANTs parameters
            self.use_ants = use_ants
            self.ants_warp = ants_warp
            self.ants_iwarp = ants_iwarp
            self.ants_aff = ants_aff
            self.force_dimensions = force_dimensions
            self.transform_mri_with_ants = transform_mri_with_ants
            
            # Visualization parameters
            self.n_examples = n_examples
            self.viz_prefix = viz_prefix
            self.slice_mode = slice_mode
            self.specific_slice = specific_slice
            self.streamline_percentage = streamline_percentage
            self.tract_linewidth = tract_linewidth
            self.save_masks = save_masks
            self.use_high_density_masks = use_high_density_masks
            self.label_bundles = label_bundles
            self.mask_thickness = mask_thickness
            self.min_fiber_percentage = min_fiber_percentage
            self.max_fiber_percentage = max_fiber_percentage
            self.min_bundle_size = min_bundle_size
            self.density_threshold = density_threshold
            
            # Enhancement parameters
            self.contrast_method = contrast_method
            self.background_preset = background_preset
            self.cornucopia_preset = cornucopia_preset
            self.enable_sharpening = enable_sharpening
            self.sharpening_strength = sharpening_strength
            self.close_gaps = close_gaps
            self.closing_footprint_size = closing_footprint_size
            
            # Spatial subdivisions parameters
            self.use_spatial_subdivisions = use_spatial_subdivisions
            self.n_subdivisions = n_subdivisions
            self.max_streamlines_per_subdivision = max_streamlines_per_subdivision
            
            # Miscellaneous parameters
            self.randomize_viz = randomize_viz
            self.random_state = random_state
    
    args = Args()
    
    result = {
        'success': False,
        'input_nifti': input_nifti,
        'input_trk': input_trk,
        'output_base': output,
        'viz_output_dir': viz_output_dir,
        'processed_nifti': None,
        'processed_trk': None,
        'temp_dir': temp_dir,
        'error': None
    }
    
    try:
        if verbose:
            print(f"Combined MRI Processing and Visualization Pipeline")
            print(f"Input NIfTI: {input_nifti}")
            print(f"Input TRK: {input_trk}")
            print(f"Temporary directory: {temp_dir}")
            print(f"Final output base: {output}")
            print(f"Visualization output: {viz_output_dir}")
        
        # Determine which files to use for visualization
        if skip_synthesis:
            if verbose:
                print("\nSkipping synthesis stage - using original files")
            viz_nifti = input_nifti
            viz_trk = input_trk
        else:
            # Run synthesis stage
            viz_nifti, viz_trk = run_synthesis_stage(args, temp_dir)
            result['processed_nifti'] = viz_nifti
            result['processed_trk'] = viz_trk
        
        # Run visualization stage
        if not skip_visualization:
            run_visualization_stage(viz_nifti, viz_trk, args)
        
        # Copy final outputs if requested
        if not skip_synthesis and keep_processed:
            copy_final_outputs(temp_dir, args)
        
        if verbose:
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            if not skip_visualization:
                print(f"Visualizations saved to: {viz_output_dir}")
            if not skip_synthesis and keep_processed:
                print(f"Processed files saved with prefix: {output}_processed")
        
        result['success'] = True
        return result
    
    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"\nError in pipeline: {e}")
        raise
    
    finally:
        # Clean up temporary directory if we created it
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"Cleaned up temporary directory: {temp_dir}")


def create_ml_dataset(
    input_pairs: List[Tuple[str, str]],
    output_dir: str,
    n_examples_per_pair: int = 5,
    use_spatial_subdivisions: bool = True,
    n_subdivisions: int = 8,
    voxel_size: float = 0.5,
    new_dim: Tuple[int, int, int] = (116, 140, 96),
    background_preset: str = "preserve_edges",
    save_masks: bool = True,
    verbose: bool = True,
    **kwargs
) -> List[dict]:
    """
    Create an ML dataset from multiple NIfTI/TRK pairs.
    
    Args:
        input_pairs: List of (nifti_path, trk_path) tuples
        output_dir: Base output directory for the dataset
        n_examples_per_pair: Number of examples to generate per input pair
        use_spatial_subdivisions: Use spatial subdivisions for more varied examples
        n_subdivisions: Number of spatial subdivisions
        voxel_size: Target voxel size for all outputs
        new_dim: Target dimensions for all outputs
        background_preset: Background enhancement preset
        save_masks: Save segmentation masks along with images
        verbose: Print progress messages
        **kwargs: Additional arguments passed to process_mri_data
    
    Returns:
        List of result dictionaries from each processing run
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    if verbose:
        print(f"Creating ML dataset from {len(input_pairs)} input pairs")
        print(f"Output directory: {output_dir}")
        print(f"Examples per pair: {n_examples_per_pair}")
    
    for i, (nifti_path, trk_path) in enumerate(input_pairs):
        if verbose:
            print(f"\nProcessing pair {i+1}/{len(input_pairs)}: {nifti_path}, {trk_path}")
        
        try:
            # Create subject-specific output directory
            subject_id = f"subject_{i+1:03d}"
            subject_output = os.path.join(output_dir, subject_id)
            viz_output = os.path.join(subject_output, "visualizations")
            
            result = process_mri_data(
                input_nifti=nifti_path,
                input_trk=trk_path,
                output=os.path.join(subject_output, "processed"),
                viz_output_dir=viz_output,
                n_examples=n_examples_per_pair,
                use_spatial_subdivisions=use_spatial_subdivisions,
                n_subdivisions=n_subdivisions,
                voxel_size=voxel_size,
                new_dim=new_dim,
                background_preset=background_preset,
                save_masks=save_masks,
                keep_processed=True,
                verbose=verbose,
                **kwargs
            )
            
            result['subject_id'] = subject_id
            result['subject_output_dir'] = subject_output
            results.append(result)
            
        except Exception as e:
            if verbose:
                print(f"Error processing pair {i+1}: {e}")
            results.append({
                'success': False,
                'error': str(e),
                'input_nifti': nifti_path,
                'input_trk': trk_path,
                'subject_id': f"subject_{i+1:03d}"
            })
    
    if verbose:
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\nDataset creation completed: {successful}/{len(input_pairs)} successful")
    
    return results


def process_single_subject(
    nifti_path: str,
    trk_path: str,
    output_dir: str,
    quick_mode: bool = False,
    high_quality: bool = False,
    **kwargs
) -> dict:
    """
    Process a single subject with preset configurations.
    
    Args:
        nifti_path: Path to NIfTI file
        trk_path: Path to TRK file  
        output_dir: Output directory
        quick_mode: Use faster settings for quick testing
        high_quality: Use high-quality settings for final results
        **kwargs: Additional arguments passed to process_mri_data
    
    Returns:
        Result dictionary from processing
    """
    
    if quick_mode and high_quality:
        raise ValueError("Cannot use both quick_mode and high_quality")
    
    # Set default parameters based on mode
    if quick_mode:
        defaults = {
            'n_examples': 2,
            'use_spatial_subdivisions': False,
            'jobs': 4,
            'use_gpu': True,
            'background_preset': 'preserve_edges',
            'save_masks': False
        }
    elif high_quality:
        defaults = {
            'n_examples': 10,
            'use_spatial_subdivisions': True,
            'n_subdivisions': 12,
            'jobs': -1,
            'use_gpu': True,
            'background_preset': 'high_quality',
            'cornucopia_preset': 'realistic_optical',
            'save_masks': True,
            'use_high_density_masks': True,
            'label_bundles': True
        }
    else:
        defaults = {
            'n_examples': 5,
            'use_spatial_subdivisions': True,
            'n_subdivisions': 8,
            'background_preset': 'preserve_edges',
            'save_masks': True
        }
    
    # Merge with user-provided kwargs
    params = {**defaults, **kwargs}
    
    return process_mri_data(
        input_nifti=nifti_path,
        input_trk=trk_path,
        output=os.path.join(output_dir, "processed"),
        viz_output_dir=os.path.join(output_dir, "visualizations"),
        keep_processed=True,
        **params
    )


def batch_process_directory(
    input_dir: str,
    output_dir: str,
    nifti_pattern: str = "*.nii.gz",
    trk_pattern: str = "*.trk",
    **kwargs
) -> List[dict]:
    """
    Batch process all NIfTI/TRK pairs in a directory.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Output directory for results
        nifti_pattern: Pattern to match NIfTI files
        trk_pattern: Pattern to match TRK files
        **kwargs: Additional arguments passed to process_mri_data
    
    Returns:
        List of result dictionaries
    """
    from pathlib import Path
    import glob
    
    input_path = Path(input_dir)
    nifti_files = list(input_path.glob(nifti_pattern))
    trk_files = list(input_path.glob(trk_pattern))
    
    # Try to match files by base name
    pairs = []
    for nifti_file in nifti_files:
        nifti_base = nifti_file.stem.replace('.nii', '')
        
        # Look for matching TRK file
        matching_trk = None
        for trk_file in trk_files:
            trk_base = trk_file.stem
            if trk_base == nifti_base or nifti_base.startswith(trk_base) or trk_base.startswith(nifti_base):
                matching_trk = trk_file
                break
        
        if matching_trk:
            pairs.append((str(nifti_file), str(matching_trk)))
        else:
            print(f"Warning: No matching TRK file found for {nifti_file}")
    
    if not pairs:
        print(f"No matching NIfTI/TRK pairs found in {input_dir}")
        return []
    
    print(f"Found {len(pairs)} matching pairs")
    return create_ml_dataset(pairs, output_dir, **kwargs)


# Example usage functions
def example_basic_usage():
    """Example of basic usage for single subject processing."""
    print("Example: Basic single subject processing")
    
    # Basic processing with default settings
    result = process_mri_data(
        input_nifti="brain.nii.gz",
        input_trk="fibers.trk",
        output="results"
    )
    
    if result['success']:
        print("Processing completed successfully!")
        print(f"Visualizations: {result['viz_output_dir']}")
    else:
        print(f"Processing failed: {result['error']}")


def example_ml_dataset_creation():
    """Example of creating an ML dataset from multiple subjects."""
    print("Example: ML dataset creation")
    
    # List of input pairs
    input_pairs = [
        ("subject1_brain.nii.gz", "subject1_fibers.trk"),
        ("subject2_brain.nii.gz", "subject2_fibers.trk"),
        ("subject3_brain.nii.gz", "subject3_fibers.trk"),
    ]
    
    # Create dataset with spatial subdivisions
    results = create_ml_dataset(
        input_pairs=input_pairs,
        output_dir="ml_dataset",
        n_examples_per_pair=8,
        use_spatial_subdivisions=True,
        n_subdivisions=10,
        save_masks=True,
        background_preset="high_quality"
    )
    
    successful = sum(1 for r in results if r.get('success', False))
    print(f"Dataset creation: {successful}/{len(input_pairs)} subjects processed successfully")


def example_advanced_usage():
    """Example of advanced usage with custom parameters."""
    print("Example: Advanced processing with custom parameters")
    
    result = process_mri_data(
        input_nifti="brain.nii.gz",
        input_trk="fibers.trk",
        output="advanced_results",
        # Synthesis parameters
        voxel_size=0.3,
        new_dim=(200, 200, 150),
        use_ants=True,
        ants_warp="warp.nii.gz",
        ants_iwarp="iwarp.nii.gz", 
        ants_aff="affine.mat",
        # Visualization parameters
        n_examples=10,
        use_spatial_subdivisions=True,
        n_subdivisions=12,
        save_masks=True,
        label_bundles=True,
        background_preset="high_quality",
        cornucopia_preset="realistic_optical",
        # Enhancement parameters
        close_gaps=True,
        enable_sharpening=True,
        sharpening_strength=0.7
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Combined MRI synthesis and visualization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - process and visualize
  python syntract.py --input brain.nii.gz --trk fibers.trk --output results

  # Skip synthesis, only visualize existing files
  python syntract.py --input brain.nii.gz --trk fibers.trk --output results --skip_synthesis

  # Use ANTs transforms with high-quality visualizations
  python syntract.py --input brain.nii.gz --trk fibers.trk --output results \\
    --use_ants --ants_warp warp.nii.gz --ants_iwarp iwarp.nii.gz --ants_aff affine.mat \\
    --background_preset high_quality --cornucopia_preset realistic_optical

  # Generate spatial subdivisions with custom parameters
  python syntract.py --input brain.nii.gz --trk fibers.trk --output results \\
    --use_spatial_subdivisions --n_subdivisions 10 --n_examples 3
        """
    )

    # Input/Output arguments
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to input NIfTI (.nii or .nii.gz) file")
    parser.add_argument("--trk", type=str, required=True, 
                       help="Path to input TRK (.trk) file")
    parser.add_argument("--output", type=str, default="combined_output", 
                       help="Base name for output files")
    
    # Pipeline control
    parser.add_argument("--skip_synthesis", action="store_true", 
                       help="Skip synthesis stage and use input files directly for visualization")
    parser.add_argument("--skip_visualization", action="store_true", 
                       help="Skip visualization stage, only run synthesis")
    parser.add_argument("--keep_processed", action="store_true",
                       help="Keep the processed NIfTI and TRK files from synthesis stage")
    
    # Synthesis parameters
    synthesis_group = parser.add_argument_group('Synthesis Parameters')
    synthesis_group.add_argument("--voxel_size", type=float, nargs='+', default=[0.5],
                                help="New voxel size: single value for isotropic or three values for anisotropic")
    synthesis_group.add_argument("--new_dim", type=int, nargs=3, default=[116, 140, 96], 
                                help="New image dimensions (x, y, z)")
    synthesis_group.add_argument("--jobs", type=int, default=8, 
                                help="Number of parallel jobs (-1 for all CPUs)")
    synthesis_group.add_argument("--patch_center", type=float, nargs=3, default=None, 
                                help="Optional patch center in mm")
    synthesis_group.add_argument("--reduction", type=str, choices=["mip", "mean"], default=None,
                                help="Optional reduction along z-axis")
    synthesis_group.add_argument("--use_gpu", type=lambda x: str(x).lower() != 'false', 
                                nargs='?', const=True, default=True,
                                help="Use GPU acceleration (default: True)")
    synthesis_group.add_argument("--cpu", action="store_true", 
                                help="Force CPU processing (disables GPU)")
    synthesis_group.add_argument("--interp", type=str, choices=["hermite", "linear", "rbf"], default="hermite",
                                help="Interpolation method for streamlines")
    synthesis_group.add_argument("--step_size", type=float, default=0.5, 
                                help="Step size for streamline densification")
    synthesis_group.add_argument("--max_gb", type=float, default=64.0,
                                help="Maximum output size in GB")
    
    # ANTs parameters
    ants_group = parser.add_argument_group('ANTs Transform Parameters')
    ants_group.add_argument("--use_ants", action="store_true", 
                           help="Use ANTs transforms for processing")
    ants_group.add_argument("--ants_warp", type=str, default=None, 
                           help="Path to ANTs warp file")
    ants_group.add_argument("--ants_iwarp", type=str, default=None, 
                           help="Path to ANTs inverse warp file")
    ants_group.add_argument("--ants_aff", type=str, default=None, 
                           help="Path to ANTs affine file")
    ants_group.add_argument("--force_dimensions", action="store_true", 
                           help="Force using specified new_dim even when using ANTs")
    ants_group.add_argument("--transform_mri_with_ants", action="store_true", 
                           help="Also transform MRI with ANTs (default: only transforms streamlines)")
    
    # Visualization parameters
    viz_group = parser.add_argument_group('Visualization Parameters')
    viz_group.add_argument("--viz_output_dir", type=str, default=None,
                          help="Output directory for visualizations (default: {output}_visualizations)")
    viz_group.add_argument("--n_examples", type=int, default=5, 
                          help="Number of visualization examples to generate")
    viz_group.add_argument("--viz_prefix", type=str, default="synthetic_", 
                          help="Prefix for visualization files")
    viz_group.add_argument("--slice_mode", type=str, choices=["coronal", "axial", "sagittal"], default="coronal",
                          help="Slice orientation for visualization")
    viz_group.add_argument("--specific_slice", type=int, default=None, 
                          help="Specific slice number to visualize")
    viz_group.add_argument("--streamline_percentage", type=float, default=100.0, 
                          help="Percentage of streamlines to include")
    viz_group.add_argument("--tract_linewidth", type=float, default=1.0, 
                          help="Linewidth for tract visualization")
    viz_group.add_argument("--save_masks", action="store_true", 
                          help="Save fiber masks along with visualizations")
    viz_group.add_argument("--use_high_density_masks", action="store_true",
                          help="Use masks from high-density fibers for all density variations")
    viz_group.add_argument("--label_bundles", action="store_true",
                          help="Label distinct fiber bundles in the masks")
    viz_group.add_argument("--mask_thickness", type=int, default=1,
                          help="Thickness of the mask lines in pixels")
    viz_group.add_argument("--min_fiber_percentage", type=float, default=10.0, 
                          help="Minimum fiber percentage for visualization")
    viz_group.add_argument("--max_fiber_percentage", type=float, default=100.0, 
                          help="Maximum fiber percentage for visualization")
    viz_group.add_argument("--min_bundle_size", type=int, default=20,
                          help="Minimum bundle size for fiber labeling")
    viz_group.add_argument("--density_threshold", type=float, default=0.15,
                          help="Density threshold for fiber visualization")
    
    # Enhancement parameters
    enhancement_group = parser.add_argument_group('Enhancement Parameters')
    enhancement_group.add_argument("--contrast_method", type=str, default="clahe", 
                                  help="Contrast enhancement method")
    enhancement_group.add_argument("--background_preset", type=str, 
                                  choices=["smooth_realistic", "high_quality", "clinical_appearance", 
                                          "preserve_edges", "subtle_enhancement", "lpsvd_denoising", 
                                          "lpsvd_aggressive", "lpsvd_conservative", "hybrid_lpsvd"],
                                  default="preserve_edges",
                                  help="Background enhancement preset")
    enhancement_group.add_argument("--cornucopia_preset", type=str,
                                  choices=["clean_optical", "realistic_optical", "noisy_optical", 
                                          "artifact_simulation", "clinical_simulation"],
                                  default=None,
                                  help="Cornucopia augmentation preset")
    enhancement_group.add_argument("--enable_sharpening", action="store_true", default=True,
                                  help="Enable sharpening in background enhancement")
    enhancement_group.add_argument("--sharpening_strength", type=float, default=0.5,
                                  help="Strength of sharpening effect")
    enhancement_group.add_argument("--close_gaps", action="store_true", 
                                  help="Close gaps in fiber visualizations")
    enhancement_group.add_argument("--closing_footprint_size", type=int, default=5, 
                                  help="Size of morphological closing footprint")
    
    # Spatial subdivisions parameters
    subdivision_group = parser.add_argument_group('Spatial Subdivision Parameters')
    subdivision_group.add_argument("--use_spatial_subdivisions", action="store_true",
                                  help="Use spatial subdivisions for visualization generation")
    subdivision_group.add_argument("--n_subdivisions", type=int, default=8,
                                  help="Number of spatial subdivisions")
    subdivision_group.add_argument("--max_streamlines_per_subdivision", type=int, default=50000,
                                  help="Maximum streamlines per subdivision")
    
    # Miscellaneous parameters
    misc_group = parser.add_argument_group('Miscellaneous Parameters')
    misc_group.add_argument("--randomize_viz", action="store_true", 
                           help="Randomize visualization parameters")
    misc_group.add_argument("--random_state", type=int, default=None, 
                           help="Random seed for reproducible results")
    misc_group.add_argument("--temp_dir", type=str, default=None,
                           help="Temporary directory for intermediate files")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_synthesis and args.skip_visualization:
        parser.error("Cannot skip both synthesis and visualization stages")
    
    if args.use_ants and not all([args.ants_warp, args.ants_iwarp, args.ants_aff]):
        parser.error("When --use_ants is specified, --ants_warp, --ants_iwarp, and --ants_aff must be provided")
    
    # Set up output directory for visualizations
    if args.viz_output_dir is None:
        args.viz_output_dir = f"{args.output}_visualizations"
    
    # Set up temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_temp = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="mri_pipeline_")
        cleanup_temp = True
    
    try:
        print(f"Combined MRI Processing and Visualization Pipeline")
        print(f"Input NIfTI: {args.input}")
        print(f"Input TRK: {args.trk}")
        print(f"Temporary directory: {temp_dir}")
        print(f"Final output base: {args.output}")
        print(f"Visualization output: {args.viz_output_dir}")
        
        # Determine which files to use for visualization
        if args.skip_synthesis:
            print("\nSkipping synthesis stage - using original files")
            viz_nifti = args.input
            viz_trk = args.trk
        else:
            # Run synthesis stage
            viz_nifti, viz_trk = run_synthesis_stage(args, temp_dir)
        
        # Run visualization stage
        if not args.skip_visualization:
            run_visualization_stage(viz_nifti, viz_trk, args)
        
        # Copy final outputs if requested
        if not args.skip_synthesis and args.keep_processed:
            copy_final_outputs(temp_dir, args)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if not args.skip_visualization:
            print(f"Visualizations saved to: {args.viz_output_dir}")
        if not args.skip_synthesis and args.keep_processed:
            print(f"Processed files saved with prefix: {args.output}_processed")
    
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        sys.exit(1)
    
    finally:
        # Clean up temporary directory if we created it
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main() 