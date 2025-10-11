#!/usr/bin/env python
"""
Batch ANTs Registration for TRK Files

This script takes a folder of TRK files and applies ANTs registration transforms
to each of them, saving the results to an output folder.

Usage:
    python batch_ants_trk_registration.py --input_folder /path/to/trk/files \
                                          --output_folder /path/to/output \
                                          --ants_warp /path/to/warp.nii.gz \
                                          --ants_iwarp /path/to/iwarp.nii.gz \
                                          --ants_aff /path/to/affine.mat \
                                          --reference_mri /path/to/reference.nii.gz
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import nibabel as nib
import numpy as np

# Import ANTs transform functions
try:
    from synthesis.ants_transform_updated import process_with_ants
except ImportError:
    # Fallback to path manipulation for development
    sys.path.append(os.path.join(os.path.dirname(__file__), 'synthesis'))
    from ants_transform_updated import process_with_ants


def validate_ants_files(ants_warp_path, ants_iwarp_path, ants_aff_path):
    """
    Validate that all required ANTs transform files exist.
    
    Parameters
    ----------
    ants_warp_path : str
        Path to ANTs warp file
    ants_iwarp_path : str
        Path to ANTs inverse warp file  
    ants_aff_path : str
        Path to ANTs affine file
        
    Returns
    -------
    bool
        True if all files exist, raises ValueError otherwise
    """
    if not all([ants_warp_path, ants_iwarp_path, ants_aff_path]):
        raise ValueError("All ANTs transform paths must be provided.")
    
    for path, name in [(ants_warp_path, "warp"), 
                       (ants_iwarp_path, "inverse warp"), 
                       (ants_aff_path, "affine")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ANTs {name} file not found: {path}")
    
    return True


def get_trk_files(input_folder):
    """
    Get all TRK files from the input folder.
    
    Parameters
    ----------
    input_folder : str
        Path to folder containing TRK files
        
    Returns
    -------
    list
        List of TRK file paths
    """
    trk_pattern = os.path.join(input_folder, "*.trk")
    trk_files = glob.glob(trk_pattern)
    
    if not trk_files:
        raise FileNotFoundError(f"No TRK files found in {input_folder}")
    
    print(f"Found {len(trk_files)} TRK files to process")
    return sorted(trk_files)


def process_single_trk(trk_path, reference_mri_path, ants_warp_path, 
                      ants_iwarp_path, ants_aff_path, output_folder):
    """
    Process a single TRK file with ANTs registration.
    
    Parameters
    ----------
    trk_path : str
        Path to input TRK file
    reference_mri_path : str
        Path to reference MRI file
    ants_warp_path : str
        Path to ANTs warp file
    ants_iwarp_path : str
        Path to ANTs inverse warp file
    ants_aff_path : str
        Path to ANTs affine file
    output_folder : str
        Path to output folder
        
    Returns
    -------
    tuple
        (success, output_path, error_message)
    """
    try:
        # Generate output filename
        trk_basename = os.path.basename(trk_path)
        trk_name = os.path.splitext(trk_basename)[0]
        output_trk_path = os.path.join(output_folder, f"{trk_name}_ants_registered.trk")
        
        print(f"\nProcessing: {trk_basename}")
        print(f"Output: {os.path.basename(output_trk_path)}")
        
        # Load original TRK file to check streamline count
        from nibabel.streamlines.trk import TrkFile
        original_trk = TrkFile.load(trk_path)
        original_count = len(original_trk.streamlines)
        print(f"  Original streamlines count: {original_count}")
        
        # Apply ANTs transforms (don't save TRK yet)
        moved_mri, affine_vox2fix, transformed_streamlines, streamlines_voxel = process_with_ants(
            path_warp=ants_warp_path,
            path_iwarp=ants_iwarp_path, 
            path_aff=ants_aff_path,
            path_mri=reference_mri_path,
            path_trk=trk_path,
            output_mri=None,  # Don't save MRI for each TRK
            output_trk=None,  # Don't save yet - check streamline count first
            transform_mri=False  # Only transform streamlines
        )
        
        # Get warp field dimensions for validation
        warp_img = nib.load(ants_warp_path)
        warp_shape = warp_img.shape[:3]
        
        print(f"  ✓ ANTs registration completed")
        print(f"  ✓ Warp field dimensions: {warp_shape}")
        print(f"  ✓ Processed {len(streamlines_voxel)} streamlines")
        
        # Only save if we have valid streamlines
        if len(streamlines_voxel) == 0:
            print(f"  ⚠ No valid streamlines found after registration - skipping save")
            return False, None, "No valid streamlines after ANTs registration"
        
        # Save the TRK file now that we know there are valid streamlines
        from nibabel.streamlines.trk import TrkFile, Tractogram
        tract = Tractogram(transformed_streamlines, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tract)
        trk.save(output_trk_path)
        
        print(f"  ✓ Saved to: {output_trk_path}")
        
        return True, output_trk_path, None
        
    except Exception as e:
        error_msg = f"Error processing {trk_path}: {str(e)}"
        print(f"  ✗ {error_msg}")
        return False, None, error_msg


def batch_ants_registration(input_folder, output_folder, ants_warp_path, 
                           ants_iwarp_path, ants_aff_path, reference_mri_path):
    """
    Apply ANTs registration to all TRK files in a folder.
    
    Parameters
    ----------
    input_folder : str
        Path to folder containing TRK files
    output_folder : str
        Path to output folder for registered TRK files
    ants_warp_path : str
        Path to ANTs warp file
    ants_iwarp_path : str
        Path to ANTs inverse warp file
    ants_aff_path : str
        Path to ANTs affine file
    reference_mri_path : str
        Path to reference MRI file
        
    Returns
    -------
    dict
        Summary of processing results
    """
    # Validate inputs
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    if not os.path.exists(reference_mri_path):
        raise FileNotFoundError(f"Reference MRI not found: {reference_mri_path}")
    
    validate_ants_files(ants_warp_path, ants_iwarp_path, ants_aff_path)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Get TRK files
    trk_files = get_trk_files(input_folder)
    
    # Process each TRK file
    results = {
        'successful': [],
        'failed': [],
        'errors': []
    }
    
    print(f"\n=== Starting batch ANTs registration ===")
    print(f"Input folder: {input_folder}")
    print(f"Reference MRI: {reference_mri_path}")
    print(f"ANTs warp: {ants_warp_path}")
    print(f"ANTs inverse warp: {ants_iwarp_path}")
    print(f"ANTs affine: {ants_aff_path}")
    
    for i, trk_path in enumerate(trk_files, 1):
        print(f"\n--- Processing {i}/{len(trk_files)} ---")
        
        success, output_path, error_msg = process_single_trk(
            trk_path, reference_mri_path, ants_warp_path, 
            ants_iwarp_path, ants_aff_path, output_folder
        )
        
        if success:
            results['successful'].append({
                'input': trk_path,
                'output': output_path
            })
        else:
            results['failed'].append(trk_path)
            results['errors'].append(error_msg)
    
    # Print summary
    print(f"\n=== Batch processing completed ===")
    print(f"Successfully processed: {len(results['successful'])}/{len(trk_files)}")
    print(f"Failed: {len(results['failed'])}/{len(trk_files)}")
    
    if results['failed']:
        print(f"\nFailed files:")
        for failed_file in results['failed']:
            print(f"  - {os.path.basename(failed_file)}")
    
    if results['successful']:
        print(f"\nSuccessful outputs saved to: {output_folder}")
        for result in results['successful']:
            print(f"  - {os.path.basename(result['output'])}")
    
    return results


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Apply ANTs registration to a folder of TRK files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python batch_ants_trk_registration.py \\
        --input_folder /path/to/trk/files \\
        --output_folder /path/to/output \\
        --ants_warp /path/to/warp.nii.gz \\
        --ants_iwarp /path/to/iwarp.nii.gz \\
        --ants_aff /path/to/affine.mat \\
        --reference_mri /path/to/reference.nii.gz
        
    # Process TRK files in current directory
    python batch_ants_trk_registration.py \\
        --input_folder ./trk_files \\
        --output_folder ./registered_trk \\
        --ants_warp transforms/warp.nii.gz \\
        --ants_iwarp transforms/iwarp.nii.gz \\
        --ants_aff transforms/affine.mat \\
        --reference_mri reference_brain.nii.gz
        """
    )
    
    parser.add_argument(
        '--input_folder', 
        required=True,
        help='Path to folder containing TRK files to process'
    )
    
    parser.add_argument(
        '--output_folder', 
        required=True,
        help='Path to output folder for registered TRK files'
    )
    
    parser.add_argument(
        '--ants_warp', 
        required=True,
        help='Path to ANTs warp field file (.nii.gz)'
    )
    
    parser.add_argument(
        '--ants_iwarp', 
        required=True,
        help='Path to ANTs inverse warp field file (.nii.gz)'
    )
    
    parser.add_argument(
        '--ants_aff', 
        required=True,
        help='Path to ANTs affine transform file (.mat)'
    )
    
    parser.add_argument(
        '--reference_mri', 
        required=True,
        help='Path to reference MRI file (.nii.gz)'
    )
    
    args = parser.parse_args()
    
    try:
        results = batch_ants_registration(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            ants_warp_path=args.ants_warp,
            ants_iwarp_path=args.ants_iwarp,
            ants_aff_path=args.ants_aff,
            reference_mri_path=args.reference_mri
        )
        
        # Exit with error code if any files failed
        if results['failed']:
            print(f"\nWARNING: {len(results['failed'])} files failed to process")
            sys.exit(1)
        else:
            print(f"\nSUCCESS: All {len(results['successful'])} files processed successfully")
            sys.exit(0)
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()