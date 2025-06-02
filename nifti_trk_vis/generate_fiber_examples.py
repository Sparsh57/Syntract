#!/usr/bin/env python
"""
Generate a dataset of synthetic fiber tract visualizations with ground truth masks for segmentation.

This script creates multiple examples with varying fiber densities and contrast settings,
along with corresponding ground truth masks that can be used to train a segmentation network.

NEW FEATURE: Spatial subdivision mode divides the NIfTI volume into actual spatial grid regions.

Example usage:

# Original mode (existing functionality)
python generate_fiber_examples.py --nifti data/brain.nii.gz --trk data/fibers.trk --output_dir dataset --examples 20

# Spatial subdivision mode (new functionality)  
python generate_fiber_examples.py --nifti data/brain.nii.gz --trk data/fibers.trk --output_dir dataset --spatial_subdivisions --n_subdivisions 8
"""

import os
import sys
import argparse
from pathlib import Path
import tempfile

from nifti_trk_slice_viewer import generate_varied_examples


def generate_examples_original_mode(args):
    """
    Generate examples using the original mode (existing functionality).
    """
    print("🎨 Running in Original Mode")
    
    # Set up contrast enhancement parameters
    contrast_params = {}
    if args.contrast_method == 'clahe':
        contrast_params = {
            'clip_limit': 0.01,  # Default CLAHE clip limit
            'tile_grid_size': (8, 8)  # Default tile grid size
        }
    elif args.contrast_method == 'gamma':
        contrast_params = {'gamma_value': args.gamma_value}
    elif args.contrast_method == 'adaptive_eq':
        contrast_params = {'clip_limit': 0.03}  # Default for adaptive equalization
    # histogram_eq and none don't need parameters
    
    print(f"Generating {args.examples} examples with varying fiber densities ({args.min_fiber_pct}% - {args.max_fiber_pct}%)")
    print(f"Output directory: {args.output_dir}")
    print(f"View: {args.view}")
    print(f"Contrast method: {args.contrast_method}")
    if args.use_high_density_masks:
        print(f"Using high-density masks ({args.max_fiber_pct}%) for all fiber density variations")
    
    # Generate examples with masks
    generate_varied_examples(
        nifti_file=args.nifti,
        trk_file=args.trk,
        output_dir=args.output_dir,
        n_examples=args.examples,
        prefix=args.prefix,
        slice_mode=args.view,
        specific_slice=args.slice_idx,
        min_fiber_percentage=args.min_fiber_pct,
        max_fiber_percentage=args.max_fiber_pct,
        roi_sphere=args.roi_sphere,
        tract_linewidth=args.tract_linewidth,
        save_masks=True,
        mask_thickness=args.mask_thickness,
        density_threshold=args.density_threshold,
        gaussian_sigma=args.gaussian_sigma,
        random_state=args.random_state,
        close_gaps=args.close_gaps,
        closing_footprint_size=args.closing_footprint_size,
        label_bundles=args.label_bundles,
        min_bundle_size=args.min_bundle_size,
        use_high_density_masks=args.use_high_density_masks,
        contrast_method=args.contrast_method,
        contrast_params=contrast_params
    )


def generate_examples_with_spatial_subdivisions(args):
    """
    Generate examples using actual spatial subdivisions of the NIfTI volume into grid regions.
    """
    print("📐 Running in Spatial Subdivision Mode")
    print("🔄 Creating actual spatial grid subdivisions of the NIfTI volume")
    
    # Import here to avoid dependency issues in original mode
    try:
        import nibabel as nib
        import numpy as np
        from nibabel.streamlines import load, save, Tractogram
    except ImportError:
        print("❌ Error: nibabel and numpy are required for spatial subdivision mode")
        print("Please install: pip install nibabel numpy")
        return
    
    # Load data
    print(f"📂 Loading data...")
    nifti_img = nib.load(args.nifti)
    nifti_data = nifti_img.get_fdata()
    trk_file_obj = load(args.trk)
    streamlines = trk_file_obj.streamlines
    
    print(f"   NIfTI shape: {nifti_data.shape}")
    print(f"   Streamlines: {len(streamlines)}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate grid subdivision parameters
    grid_subdivisions = _calculate_grid_subdivisions(nifti_data.shape, args.n_subdivisions)
    
    print(f"\n Creating {len(grid_subdivisions)} spatial grid regions...")
    
    # Process each subdivision
    valid_subdivisions = []
    total_examples = 0
    
    for i, (bounds, region_data) in enumerate(grid_subdivisions):
        x_start, x_end, y_start, y_end, z_start, z_end = bounds
        
        print(f"   Processing region {i+1}/{len(grid_subdivisions)}: ({x_start}:{x_end}, {y_start}:{y_end}, {z_start}:{z_end})")
        
        # Filter streamlines for this region
        region_streamlines = _filter_streamlines_by_region(
            streamlines, trk_file_obj, x_start, x_end, y_start, y_end, z_start, z_end
        )
        
        print(f"      Found {len(region_streamlines)} streamlines")
        
        # Skip if not enough streamlines
        if len(region_streamlines) < args.min_streamlines_per_region:
            print(f"      ⏭️  Skipping (too few streamlines)")
            continue
        
        # Create subdivision directory
        sub_id = len(valid_subdivisions)
        sub_dir = output_path / f"subdivision_{sub_id:03d}"
        sub_dir.mkdir(exist_ok=True)
        
        print(f"   ✅ Processing subdivision {sub_id}: {region_data.shape}, {len(region_streamlines)} streamlines")
        
        # Create temporary files for this region
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_nifti:
            with tempfile.NamedTemporaryFile(suffix='.trk', delete=False) as temp_trk:
                try:
                    # Save region NIfTI
                    region_affine = nifti_img.affine.copy()
                    # Adjust affine for region offset
                    offset = np.array([x_start, y_start, z_start])
                    region_affine[:3, 3] += region_affine[:3, :3] @ offset
                    
                    region_nifti_img = nib.Nifti1Image(region_data, region_affine)
                    nib.save(region_nifti_img, temp_nifti.name)
                    
                    # Save region tractography
                    tractogram = Tractogram(region_streamlines, affine_to_rasmm=region_affine)
                    save(tractogram, temp_trk.name)
                    
                    # Generate examples for this subdivision using generate_varied_examples to get color variety
                    prefix = f"sub_{sub_id:03d}_"
                    
                    # Calculate random state for this subdivision
                    region_random_state = None
                    if args.random_state is not None:
                        region_random_state = args.random_state + sub_id * 1000
                    
                    try:
                        # Use generate_varied_examples to get proper color variation
                        generate_varied_examples(
                            nifti_file=temp_nifti.name,
                            trk_file=temp_trk.name,
                            output_dir=str(sub_dir),
                            n_examples=args.examples_per_subdivision,
                            prefix=prefix,
                            slice_mode=args.view,
                            specific_slice=args.slice_idx,
                            min_fiber_percentage=args.min_fiber_pct,
                            max_fiber_percentage=args.max_fiber_pct,
                            roi_sphere=args.roi_sphere,
                            tract_linewidth=args.tract_linewidth,
                            save_masks=True,
                            mask_thickness=args.mask_thickness,
                            density_threshold=args.density_threshold,
                            gaussian_sigma=args.gaussian_sigma,
                            random_state=region_random_state,
                            close_gaps=args.close_gaps,
                            closing_footprint_size=args.closing_footprint_size,
                            label_bundles=args.label_bundles,
                            min_bundle_size=args.min_bundle_size,
                            use_high_density_masks=args.use_high_density_masks,
                            contrast_method=args.contrast_method,
                            contrast_params={}  # Let function handle defaults
                        )
                        
                        total_examples += args.examples_per_subdivision
                        print(f"      ✅ Generated {args.examples_per_subdivision} examples with color variation")
                        
                    except Exception as e:
                        print(f"      ❌ Failed to generate examples: {str(e)}")
                
                finally:
                    # Clean up temporary files
                    try:
                        os.unlink(temp_nifti.name)
                        os.unlink(temp_trk.name)
                    except:
                        pass
        
        valid_subdivisions.append({
            'id': sub_id,
            'bounds': bounds,
            'shape': region_data.shape,
            'n_streamlines': len(region_streamlines),
            'examples_generated': args.examples_per_subdivision
        })
    
    print(f"\n🎉 Completed! Generated {total_examples} examples from {len(valid_subdivisions)} spatial subdivisions")
    
    # Save summary
    _save_subdivision_summary(output_path, valid_subdivisions, total_examples, args)
    
    return valid_subdivisions


def _calculate_grid_subdivisions(nifti_shape, n_subdivisions):
    """Calculate spatial grid subdivisions of the NIfTI volume."""
    import numpy as np
    
    x_max, y_max, z_max = nifti_shape
    
    # Calculate grid dimensions to create approximately n_subdivisions
    # Try to create a roughly cubic grid
    grid_size = max(1, int(np.ceil(n_subdivisions ** (1/3))))
    
    # Adjust grid size to not exceed available subdivisions
    while grid_size ** 3 > n_subdivisions * 2:  # Allow some flexibility
        grid_size -= 1
    
    grid_size = max(1, grid_size)
    
    print(f"   Creating {grid_size}×{grid_size}×{grid_size} grid (target: {n_subdivisions} subdivisions)")
    
    # Calculate subdivision sizes
    sub_x = max(1, x_max // grid_size)
    sub_y = max(1, y_max // grid_size)  
    sub_z = max(1, z_max // grid_size)
    
    print(f"   Each subdivision size: {sub_x}×{sub_y}×{sub_z}")
    
    subdivisions = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if len(subdivisions) >= n_subdivisions:
                    break
                    
                x_start = i * sub_x
                x_end = min((i + 1) * sub_x, x_max)
                y_start = j * sub_y  
                y_end = min((j + 1) * sub_y, y_max)
                z_start = k * sub_z
                z_end = min((k + 1) * sub_z, z_max)
                
                # Extract region data
                try:
                    import nibabel as nib
                    nifti_img = nib.load(sys.argv[sys.argv.index('--nifti') + 1])  # Get from command line
                    nifti_data = nifti_img.get_fdata()
                    region_data = nifti_data[x_start:x_end, y_start:y_end, z_start:z_end]
                    
                    bounds = (x_start, x_end, y_start, y_end, z_start, z_end)
                    subdivisions.append((bounds, region_data))
                except:
                    # Fallback: just store bounds, extract data later
                    bounds = (x_start, x_end, y_start, y_end, z_start, z_end)
                    subdivisions.append((bounds, None))
            
            if len(subdivisions) >= n_subdivisions:
                break
        if len(subdivisions) >= n_subdivisions:
            break
    
    return subdivisions


def _filter_streamlines_by_region(streamlines, trk_file_obj, x_start, x_end, y_start, y_end, z_start, z_end):
    """Filter streamlines that pass through the specified region."""
    import numpy as np
    
    filtered_streamlines = []
    voxel_to_rasmm = trk_file_obj.header.get('voxel_to_rasmm', None)
    
    for streamline in streamlines:
        # Convert coordinates if needed
        if voxel_to_rasmm is not None:
            # Convert from world space to voxel space
            world_coords = np.hstack([streamline, np.ones((streamline.shape[0], 1))])
            rasmm_to_voxel = np.linalg.inv(voxel_to_rasmm)
            voxel_coords = world_coords @ rasmm_to_voxel.T
            voxel_coords = voxel_coords[:, :3]
        else:
            voxel_coords = streamline
        
        # Check if any point falls within the region
        points_in_region = (
            (voxel_coords[:, 0] >= x_start) & (voxel_coords[:, 0] < x_end) &
            (voxel_coords[:, 1] >= y_start) & (voxel_coords[:, 1] < y_end) &
            (voxel_coords[:, 2] >= z_start) & (voxel_coords[:, 2] < z_end)
        )
        
        if np.any(points_in_region):
            # Adjust coordinates to new origin
            adjusted_streamline = voxel_coords.copy()
            adjusted_streamline[:, 0] -= x_start
            adjusted_streamline[:, 1] -= y_start  
            adjusted_streamline[:, 2] -= z_start
            filtered_streamlines.append(adjusted_streamline)
    
    return filtered_streamlines


def _save_subdivision_summary(output_path, subdivisions, total_examples, args):
    """Save generation summary."""
    summary = {
        'total_examples': total_examples,
        'subdivisions_processed': len(subdivisions),
        'examples_per_subdivision': args.examples_per_subdivision,
        'fiber_percentage_range': [args.min_fiber_pct, args.max_fiber_pct],
        'subdivision_info': subdivisions
    }
    
    import json
    summary_path = output_path / "generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic fiber tract dataset with ground truth masks')
    
    parser.add_argument('--nifti', required=True, help='Path to the NIfTI file')
    parser.add_argument('--trk', required=True, help='Path to the tractography (.trk) file')
    parser.add_argument('--output_dir', default='./fiber_dataset', help='Directory to save the generated dataset')
    parser.add_argument('--prefix', default='fiber_', help='Prefix for output filenames')
    
    # Mode selection
    parser.add_argument('--spatial_subdivisions', action='store_true',
                       help='Use spatial subdivision mode (divide NIfTI volume into actual grid regions)')
    
    # Original parameters
    parser.add_argument('--examples', type=int, default=10, help='Number of examples to generate (original mode)')
    parser.add_argument('--view', choices=['axial', 'coronal', 'all'], default='coronal', 
                        help='Visualization mode (default: coronal)')
    parser.add_argument('--slice_idx', type=int, default=None, 
                        help='Specific slice to visualize (default: automatically chosen)')
    parser.add_argument('--min_fiber_pct', type=float, default=10.0, 
                        help='Minimum percentage of fibers to display (default: 10%%)')
    parser.add_argument('--max_fiber_pct', type=float, default=100.0, 
                        help='Maximum percentage of fibers to display (default: 100%%)')
    parser.add_argument('--roi_center', type=float, nargs=3, default=None,
                        help='Center coordinates (x,y,z) for ROI sphere in voxel space (optional)')
    parser.add_argument('--roi_radius', type=float, default=10.0,
                        help='Radius for ROI sphere in voxel units (default: 10.0)')
    parser.add_argument('--mask_thickness', type=int, default=1,
                        help='Thickness of the mask lines in pixels (default: 1)')
    parser.add_argument('--tract_linewidth', type=float, default=1.0,
                        help='Width of the tract lines (default: 1.0)')
    parser.add_argument('--density_threshold', type=float, default=0.15,
                        help='Threshold for fiber density map (0.0-1.0) (default: 0.15)')
    parser.add_argument('--gaussian_sigma', type=float, default=2.0,
                        help='Sigma for Gaussian smoothing of density map (default: 2.0)')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Random seed for reproducible results')
    parser.add_argument('--close_gaps', action='store_true',
                        help='Apply morphological closing to create contiguous regions for fiber bundles')
    parser.add_argument('--closing_footprint_size', type=int, default=5,
                        help='Size of the footprint for morphological closing operations (default: 5)')
    parser.add_argument('--label_bundles', action='store_true',
                        help='Label distinct fiber bundles in the masks')
    parser.add_argument('--min_bundle_size', type=int, default=20,
                        help='Minimum size (in pixels) for a region to be considered a bundle (default: 20)')
    parser.add_argument('--use_high_density_masks', action='store_true',
                        help='Use masks from high-density fibers for all density variations')
    
    # Contrast enhancement parameters
    parser.add_argument('--contrast_method', choices=['clahe', 'gamma', 'none', 'random'], 
                        default='random', help='Contrast enhancement method (default: random)')
    parser.add_argument('--gamma_value', type=float, default=1.2, 
                        help='Gamma value for gamma correction (default: 1.2)')
    
    # Spatial subdivision parameters
    parser.add_argument('--n_subdivisions', type=int, default=8,
                       help='Number of spatial grid subdivisions to create (default: 8)')
    parser.add_argument('--examples_per_subdivision', type=int, default=3,
                       help='Examples per subdivision (default: 3)')
    parser.add_argument('--min_streamlines_per_region', type=int, default=10,
                       help='Minimum streamlines required for a subdivision region (default: 10)')
    
    args = parser.parse_args()
    
    # Handle random contrast method selection
    if args.contrast_method == 'random':
        import random
        import time
        
        # Use current time for randomization if no specific random_state provided
        # This ensures we get different contrast methods each run
        if args.random_state is not None:
            # Use random_state but add some variation for contrast method selection
            contrast_seed = args.random_state + int(time.time()) % 1000
        else:
            # Use time-based seed for true randomization
            contrast_seed = int(time.time() * 1000) % 100000
            
        random.seed(contrast_seed)
        
        # Select a random contrast method (removed histogram_eq, adaptive_eq, rescale_intensity)
        contrast_options = ['clahe', 'gamma', 'none']
        args.contrast_method = random.choice(contrast_options)
        print(f"🎲 Randomly selected contrast method: {args.contrast_method} (seed: {contrast_seed})")
    else:
        print(f"🎨 Using specified contrast method: {args.contrast_method}")
    
    # Check that input files exist
    if not os.path.exists(args.nifti):
        raise FileNotFoundError(f"NIfTI file not found: {args.nifti}")
    if not os.path.exists(args.trk):
        raise FileNotFoundError(f"Tractography file not found: {args.trk}")
    
    # Create ROI sphere parameter if center is provided
    args.roi_sphere = None
    if args.roi_center is not None:
        args.roi_sphere = (args.roi_center[0], args.roi_center[1], args.roi_center[2], args.roi_radius)
        print(f"Using ROI sphere at ({args.roi_center[0]}, {args.roi_center[1]}, {args.roi_center[2]}) with radius {args.roi_radius}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📁 Input NIfTI: {args.nifti}")
    print(f"📁 Input TRK: {args.trk}")
    print(f"📁 Output Directory: {args.output_dir}")
    
    # Determine which mode to run
    if args.spatial_subdivisions:
        generate_examples_with_spatial_subdivisions(args)
    else:
        generate_examples_original_mode(args)
    
    print("\n🎉 Dataset generation complete!")
    
    # Show what was generated
    if args.spatial_subdivisions:
        print("The dataset includes:")
        print(f"- {args.n_subdivisions} actual spatial grid subdivisions of the NIfTI volume")
        print(f"- {args.examples_per_subdivision} examples per subdivision")
        print(f"- Each subdivision contains a different spatial region of the brain")
        print(f"- Color variations preserved from original mode (BW and blue tints)")
    else:
        print("The dataset includes:")
        print(f"- {args.examples} synthetic images with varying fiber densities")
        print(f"- Corresponding ground truth masks for segmentation")
        if args.label_bundles:
            print(f"- Labeled bundle visualizations with distinct colors for each bundle")
        if args.use_high_density_masks:
            print(f"- All images use the same masks derived from high-density ({args.max_fiber_pct}%) fibers")
    
    # List the files in the output directory
    print("\nGenerated files:")
    for root, _, files in os.walk(args.output_dir):
        for file in sorted(files):
            if file.startswith(args.prefix) or file.endswith(('.png', '.json')):
                rel_path = os.path.relpath(os.path.join(root, file), args.output_dir)
                print(f"  {rel_path}")


if __name__ == "__main__":
    main() 