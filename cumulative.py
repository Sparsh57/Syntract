#!/usr/bin/env python
"""
Cumulative/Batch Processing Script for MRI Synthesis and Visualization

This script allows you to process multiple TRK files with a common NIfTI file
using the syntract pipeline. It provides a programmatic interface to batch
process fiber tract data without needing to use the command line interface
for each file individually.

Usage:
    python cumulative.py

Configuration:
    - Modify the nifti_path variable to point to your NIfTI file
    - Modify the trk_dir variable to point to the directory containing your TRK files
    - Adjust processing_params dictionary to customize the processing parameters

The script will:
1. Find all .trk files in the specified directory
2. Process each one with the common NIfTI file using syntract
3. Generate visualizations and save processed files
4. Provide a summary of successful and failed processing attempts
"""

import os
import sys

# Fix matplotlib backend issue before importing any other modules
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import the syntract function
from syntract import process_syntract


def batch_process_trk_files(nifti_path, trk_dir, **processing_kwargs):
    """
    Batch process multiple TRK files with a common NIfTI file.
    
    Args:
        nifti_path (str): Path to the common NIfTI file
        trk_dir (str): Directory containing TRK files to process
        **processing_kwargs: Additional parameters to pass to process_syntract
    
    Returns:
        dict: Summary of processing results
    """
    results = {
        'successful': [],
        'failed': [],
        'total_processed': 0
    }
    
    # Check if directories exist
    if not os.path.exists(nifti_path):
        print(f"Error: NIfTI file not found: {nifti_path}")
        return results
        
    if not os.path.exists(trk_dir):
        print(f"Error: TRK directory not found: {trk_dir}")
        return results
    
    # Get list of TRK files
    files = [f for f in os.listdir(trk_dir) if f.endswith(".trk")]
    
    if not files:
        print(f"No TRK files found in {trk_dir}")
        return results
    
    print(f"Found {len(files)} TRK files to process")
    print(f"Using NIfTI file: {nifti_path}")
    print("="*60)
    
    # Check if patch extraction is enabled
    enable_patch_extraction = processing_kwargs.get('enable_patch_extraction', False)
    total_patches = processing_kwargs.get('total_patches', 100)
    
    if enable_patch_extraction:
        print(f"Patch extraction enabled - distributing {total_patches} patches across {len(files)} TRK files")
        patches_per_file = max(1, total_patches // len(files))
        remaining_patches = total_patches - (patches_per_file * len(files))
        print(f" Base patches per file: {patches_per_file}")
        if remaining_patches > 0:
            print(f"Extra patches distributed to first {remaining_patches} files")
    
    for i, trk_filename in enumerate(files, 1):
        trk_path = os.path.join(trk_dir, trk_filename)
        # Use naming pattern similar to your CLI command
        base_name = os.path.splitext(trk_filename)[0]
        
        # Create organized folder structure
        main_output_dir = "syntract_submission"
        processed_files_dir = os.path.join(main_output_dir, "processed_files")
        visualizations_dir = os.path.join(main_output_dir, "visualizations", base_name)
        
        # Create directories if they don't exist
        os.makedirs(processed_files_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Set output paths
        output_prefix = os.path.join(processed_files_dir, f"processed_{base_name}")
        viz_output = visualizations_dir
        
        # Handle patch extraction distribution
        current_kwargs = processing_kwargs.copy()
        if enable_patch_extraction:
            # Calculate patches for this specific file
            extra_patch = 1 if (i - 1) < remaining_patches else 0
            patches_for_file = patches_per_file + extra_patch
            
            # Update patch extraction parameters for this file
            current_kwargs['total_patches'] = patches_for_file
            current_kwargs['patch_output_dir'] = os.path.join(main_output_dir, "patches", base_name)
            current_kwargs['patch_prefix'] = f"{base_name}_patch"
            
            print(f"\n📦 Processing {i}/{len(files)}: {trk_filename} ({patches_for_file} patches)")
        else:
            print(f"\nProcessing {i}/{len(files)}: {trk_filename}")
        
        print("-" * 40)
        
        try:
            # Call the syntract function with appropriate parameters
            result = process_syntract(
                input_nifti=nifti_path,
                input_trk=trk_path,
                output_base=output_prefix,
                viz_output_dir=viz_output,
                **current_kwargs  # Use the updated kwargs with distributed patch counts
            )
            
            if result['success']:
                print(f"✓ Successfully processed {trk_filename}")
                results['successful'].append({
                    'filename': trk_filename,
                    'output_base': result['output_base'],
                    'visualization_dir': result['visualization_dir'],
                    'processed_nifti': result['processed_nifti'],
                    'processed_trk': result['processed_trk']
                })
            else:
                print(f"✗ Failed to process {trk_filename}: {result['error']}")
                results['failed'].append({
                    'filename': trk_filename,
                    'error': result['error']
                })
                
        except Exception as e:
            print(f"✗ Error processing {trk_filename}: {str(e)}")
            results['failed'].append({
                'filename': trk_filename,
                'error': str(e)
            })
            # Continue with the next file instead of stopping the entire loop
            continue
        
        results['total_processed'] += 1
    
    return results


def process_single_file_with_your_params():
    """
    Example function showing how to process a single file with your exact CLI parameters.
    This processes the specific file you mentioned in your CLI command.
    """
    # Create organized folder structure for single file processing
    main_output_dir = "syntract_submission"
    processed_files_dir = os.path.join(main_output_dir, "processed_files")
    single_file_viz_dir = os.path.join(main_output_dir, "visualizations", "csdprob_dhollander_seed_x27.5_y23.5_z51.5")
    
    # Create directories if they don't exist
    os.makedirs(processed_files_dir, exist_ok=True)
    os.makedirs(single_file_viz_dir, exist_ok=True)
    
    result = process_syntract(
        input_nifti="examples/example_data/sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz",
        input_trk="examples/example_data/csdprob_dhollander_seed_x27.5_y23.5_z51.5.trk",
        output_base=os.path.join(processed_files_dir, "processed_subject"),
        viz_output_dir=single_file_viz_dir,
        new_dim=[800, 20, 800],
        voxel_size=[0.05, 0.05, 0.05],
        patch_center=[83, 24, 37],
        use_ants=True,
        ants_warp="examples/example_data/sub-MF278_sample-brain_desc-dwi_to_blockface_1Warp.nii.gz",
        ants_iwarp="examples/example_data/sub-MF278_sample-brain_desc-dwi-to-blockface-1InverseWarp.nii.gz",
        ants_aff="examples/example_data/sub-MF278_sample-brain_desc-dwi_to_blockface_0GenericAffine.mat",
        n_examples=5,
        close_gaps=True,
        min_bundle_size=800,
        density_threshold=0.1,
        save_masks=True,
        # Spatial subdivisions (uncomment to enable)
        # use_spatial_subdivisions=True,
        # n_subdivisions=8,
        # max_streamlines_per_subdivision=50000,
        # min_streamlines_per_region=10
    )
    
    print(f"Single file processing result: {result}")
    return result


def get_processing_configurations():
    """
    Returns different processing configurations you can choose from.
    """
    # Base configuration matching your CLI command
    base_config = {
        # Basic processing parameters
        'new_dim': [800, 20, 800],
        'voxel_size': [0.05, 0.05, 0.05],
        'patch_center': [83, 24, 37],
        
        # ANTs transformation parameters
        'use_ants': True,
        'ants_warp': 'examples/example_data/sub-MF278_sample-brain_desc-dwi_to_blockface_1Warp.nii.gz',
        'ants_iwarp': 'examples/example_data/sub-MF278_sample-brain_desc-dwi-to-blockface-1InverseWarp.nii.gz',
        'ants_aff': 'examples/example_data/sub-MF278_sample-brain_desc-dwi_to_blockface_0GenericAffine.mat',
        
        # Visualization parameters
        'n_examples': 5,
        'save_masks': True,
        'close_gaps': True,
        'min_bundle_size': 800,
        'density_threshold': 0.1,
        
        # Other parameters
        'viz_prefix': 'synthetic_',
        'background_preset': 'preserve_edges',
        'slice_mode': 'coronal',
        'streamline_percentage': 100.0,
        'keep_processed': True,
        'jobs': 4,
    }
    
    # Configuration with spatial subdivisions enabled
    subdivisions_config = base_config.copy()
    subdivisions_config.update({
        'use_spatial_subdivisions': True,
        'n_subdivisions': 8,
        'max_streamlines_per_subdivision': 50000,
        'min_streamlines_per_region': 1,  # Lower minimum - some regions might have very few streamlines
        'n_examples': 3,  # Fewer examples since subdivisions create more output
        'skip_empty_regions': True,  # Skip regions with no streamlines
    })
    
    # Alternative configuration with fewer subdivisions for sparse data
    sparse_subdivisions_config = base_config.copy()
    sparse_subdivisions_config.update({
        'use_spatial_subdivisions': True,
        'n_subdivisions': 4,  # Fewer subdivisions for sparse streamline data
        'max_streamlines_per_subdivision': 50000,
        'min_streamlines_per_region': 1,
        'n_examples': 2,  # Even fewer examples
        'skip_empty_regions': True,
    })
    
    # Configuration optimized for very thin dimensions (like 800x20x800)
    thin_dimension_config = base_config.copy()
    thin_dimension_config.update({
        'use_spatial_subdivisions': True,
        'n_subdivisions': 8,  # Back to 8 but with very permissive settings
        'max_streamlines_per_subdivision': 50000,
        'min_streamlines_per_region': 0,  # Allow regions with 0 streamlines
        'n_examples': 1,  # Just 1 example to reduce complexity
        'skip_empty_regions': False,  # Don't skip empty regions for debugging
    })
    
    # Minimal subdivision config for debugging
    debug_subdivisions_config = base_config.copy()
    debug_subdivisions_config.update({
        'use_spatial_subdivisions': True,
        'n_subdivisions': 8,
        'max_streamlines_per_subdivision': 50000,
        'min_streamlines_per_region': 1,  # Require at least 1 streamline
        'n_examples': 1,
        'skip_empty_regions': True,  # Skip empty regions
        'save_masks': False,  # Disable masks to simplify debugging
        'background_preset': None,  # Disable background enhancement
        'cornucopia_preset': None,  # Disable cornucopia augmentation
        'enable_sharpening': False,  # Disable sharpening
        'close_gaps': False,  # Disable gap closing
    })
    
    # High-detail subdivision config for crisp, detailed images
    crisp_subdivisions_config = base_config.copy()
    crisp_subdivisions_config.update({
        'use_spatial_subdivisions': True,
        'n_subdivisions': 8,
        'max_streamlines_per_subdivision': 50000,
        'min_streamlines_per_region': 1,
        'n_examples': 2,
        'skip_empty_regions': True,
        'save_masks': True,
        'background_preset': 'preserve_edges',  # Use edge-preserving enhancement
        'cornucopia_preset': None,  # No augmentation for clarity
        'enable_sharpening': True,  # Enable sharpening for detail
        'sharpening_strength': 1.0,  # Strong sharpening
        'close_gaps': False,  # Keep original fiber detail
        'contrast_method': 'clahe',  # Use CLAHE for local contrast
    })
    
    # Ultra-crisp standard configuration for maximum detail
    ultra_crisp_config = base_config.copy()
    ultra_crisp_config.update({
        'background_preset': None,  # Disable smoothing background enhancement
        'cornucopia_preset': None,  # No augmentation for maximum clarity
        'enable_sharpening': True,  # Enable strong sharpening
        'sharpening_strength': 1.5,  # Very strong sharpening
        'close_gaps': False,  # Don't close gaps to preserve fine detail
        'contrast_method': 'none',  # Minimal contrast adjustment
    })
    
    # Patch extraction configuration - distributes patches across multiple TRK files
    patch_extraction_config = base_config.copy()
    patch_extraction_config.update({
        'enable_patch_extraction': True,
        'total_patches': 100,  # Total patches to distribute across all TRK files
        'patch_size': [128, 128, 128],  # 3D patch size
        'min_streamlines_per_patch': 30,
        'patch_prefix': 'patch',
        'max_patch_trials': 100,
        'random_state': 42,  # For reproducible results
        'n_examples': 5,  # Visualization examples per patch
        'viz_prefix': 'patch_viz_',
        # Lower the primary visualization count since we'll have many patches
        'n_examples': 2,  # Reduce primary visualizations
    })
    
    # High-throughput patch extraction (more patches, smaller size)
    high_throughput_patches_config = base_config.copy()
    high_throughput_patches_config.update({
        'enable_patch_extraction': True,
        'total_patches': 200,  # More patches
        'patch_size': [64, 64, 64],  # Smaller patches for faster processing
        'min_streamlines_per_patch': 15,  # Lower threshold
        'patch_prefix': 'small_patch',
        'max_patch_trials': 50,
        'random_state': None,  # Random each time
        'n_examples': 1,  # Minimal primary visualizations
        'viz_prefix': 'ht_patch_viz_',
    })
    
    # Quality patch extraction (fewer patches, larger size, high quality)
    quality_patches_config = base_config.copy()
    quality_patches_config.update({
        'enable_patch_extraction': True,
        'total_patches': 50,  # Fewer but higher quality patches
        'patch_size': [256, 256, 256],  # Large patches
        'min_streamlines_per_patch': 50,  # Higher threshold for better quality
        'patch_prefix': 'quality_patch',
        'max_patch_trials': 200,  # More trials to find good patches
        'random_state': 123,
        'n_examples': 3,
        'viz_prefix': 'quality_patch_viz_',
        # Enhanced visualization quality
        'background_preset': 'preserve_edges',
        'contrast_method': 'clahe',
        'enable_sharpening': True,
        'sharpening_strength': 1.0,
    })
    
    return {
        'standard': base_config,
        'ultra_crisp': ultra_crisp_config,
        'with_subdivisions': subdivisions_config,
        'sparse_subdivisions': sparse_subdivisions_config,
        'thin_dimension': thin_dimension_config,
        'debug_subdivisions': debug_subdivisions_config,
        'crisp_subdivisions': crisp_subdivisions_config,
        'patch_extraction': patch_extraction_config,
        'high_throughput_patches': high_throughput_patches_config,
        'quality_patches': quality_patches_config
    }


def main():
    """
    Main function for batch processing.
    
    SUBDIVISION SUPPORT:
    - ✅ Subdivisions are now FIXED and working!
    - Fixed the 'NoneType' object has no attribute 'shape' error in subdivision processing
    - Subdivisions create additional output folders organized by spatial regions
    - When enabled, it generates fewer examples per TRK file but with spatial breakdown
    
    CURRENT STATUS:
    - ✅ Standard processing: Works perfectly, generates 5 examples + masks per TRK file
    - ✅ Batch processing: Processes multiple TRK files automatically  
    - ✅ Organized output: Creates clean folder structure in syntract_submission/
    - ✅ Subdivisions: FIXED and working - try 'debug_subdivisions' or 'thin_dimension'
    """
    # Path to your NIfTI file (common to all TRKs)
    nifti_path = "examples/example_data/sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz"

    # Directory containing TRK files - change this to wherever your TRK files are located
    # Options: "dwi", "examples/example_data", or current directory "."
    trk_dir = "dwi"  # Changed from "dwi" to use existing example data
    
    # Choose processing configuration
    configs = get_processing_configurations()
    
    # CHANGE THIS to switch between configurations:
    # 'standard' - your original CLI parameters (no subdivisions)
    # 'ultra_crisp' - standard processing with maximum detail/sharpness (no smoothing)
    # 'with_subdivisions' - same parameters + 8 spatial subdivisions
    # 'sparse_subdivisions' - same parameters + 4 spatial subdivisions (better for sparse data)
    # 'thin_dimension' - same parameters + 8 subdivisions optimized for thin dimensions
    # 'debug_subdivisions' - minimal subdivision config for debugging subdivision issues
    # 'crisp_subdivisions' - high-detail subdivisions with edge preservation and sharpening ⭐ RECOMMENDED
    # 'patch_extraction' - extract 100 patches distributed across all TRK files 🧩 NEW!
    # 'high_throughput_patches' - extract 200 smaller patches for high throughput 🚀 NEW!
    # 'quality_patches' - extract 50 high-quality large patches 💎 NEW!
    config_choice = 'patch_extraction'  # Try the new patch extraction!
    
    processing_params = configs[config_choice]
    
    print("Starting batch processing of TRK files...")
    print(f"NIfTI file: {nifti_path}")
    print(f"TRK directory: {trk_dir}")
    print(f"Configuration: {config_choice}")
    
    # Handle different configuration types
    if processing_params.get('enable_patch_extraction', False):
        print("PATCH EXTRACTION MODE ENABLED")
        print(f"   - Total patches to distribute: {processing_params.get('total_patches', 100)}")
        print(f"   - Patch size: {processing_params.get('patch_size', [128, 128, 128])}")
        print(f"   - Min streamlines per patch: {processing_params.get('min_streamlines_per_patch', 30)}")
        print(f"   - Random state: {processing_params.get('random_state', 'None (random)')}")
        
    elif processing_params.get('use_spatial_subdivisions', False):
        print("🔀 Spatial subdivisions ENABLED")
        print(f"   - Number of subdivisions: {processing_params.get('n_subdivisions', 8)}")
        print(f"   - Max streamlines per subdivision: {processing_params.get('max_streamlines_per_subdivision', 50000)}")
        print(f"   - Min streamlines per region: {processing_params.get('min_streamlines_per_region', 1)}")
        print(f"   - Skip empty regions: {processing_params.get('skip_empty_regions', True)}")
    else:
        print("📊 Using standard processing (no subdivisions/patches)")
        
    print(f"Processing parameters: {processing_params}")
    
    # Run batch processing
    results = batch_process_trk_files(nifti_path, trk_dir, **processing_params)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files processed: {results['total_processed']}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    
    # Show organized folder structure
    print(f"\n📁 Output folder structure:")
    print(f"syntract_submission/")
    print(f"├── processed_files/          # All processed .nii.gz and .trk files")
    if processing_params.get('enable_patch_extraction', False):
        print(f"├── patches/                  # Extracted patches organized by TRK file")
        print(f"│   ├── [trk_file_1]/         # Patches from first TRK file")
        print(f"│   ├── [trk_file_2]/         # Patches from second TRK file")
        print(f"│   └── ...                   # etc.")
    print(f"└── visualizations/           # Visualization folders organized by TRK file")
    
    if results['successful']:
        print("\n✓ Successfully processed files:")
        for item in results['successful']:
            print(f"  - {item['filename']}")
            if item['processed_nifti']:
                processed_file = os.path.basename(item['processed_nifti'])
                print(f"     Processed files: syntract_submission/processed_files/{processed_file}")
            viz_folder = os.path.basename(item['visualization_dir'])
            print(f"   Visualizations: syntract_submission/visualizations/{viz_folder}/")
    
    if results['failed']:
        print("\n✗ Failed to process files:")
        for item in results['failed']:
            print(f"  - {item['filename']}: {item['error']}")
    
    print(f"\n All outputs are organized in the 'syntract_submission' folder!")
    print("Cumulative processing completed!")


if __name__ == "__main__":
    # Choose between single file processing or batch processing
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        print("Running single file processing with your exact CLI parameters...")
        process_single_file_with_your_params()
    else:
        print("Running batch processing...")
        print("(Use 'python cumulative.py single' to process just the specific file from your CLI command)")
        main()