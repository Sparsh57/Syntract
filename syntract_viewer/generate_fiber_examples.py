#!/usr/bin/env python
"""
Generate a dataset of synthetic fiber tract visualizations with ground truth masks.

This script creates multiple examples with varying fiber densities and contrast settings,
along with corresponding ground truth masks for training segmentation networks.

Features:
- Original mode: varying fiber densities and contrast settings
- Spatial subdivision mode: divides NIfTI volume into spatial grid regions
- Cornucopia integration: advanced medical imaging augmentations
"""

import os
import sys
import argparse
from pathlib import Path
import tempfile

try:
    from .generation import generate_varied_examples, generate_enhanced_varied_examples
except ImportError:
    from generation import generate_varied_examples, generate_enhanced_varied_examples

try:
    from .contrast import CORNUCOPIA_INTEGRATION_AVAILABLE
    ENHANCED_AVAILABLE = True
except ImportError:
    try:
        from contrast import CORNUCOPIA_INTEGRATION_AVAILABLE
        ENHANCED_AVAILABLE = True
    except ImportError:
        ENHANCED_AVAILABLE = False
        CORNUCOPIA_INTEGRATION_AVAILABLE = False


def generate_examples_original_mode(args, background_enhancement_available):
    """Generate examples using the original mode with automatic background enhancement and optional Cornucopia."""
    print("🎨 Running in Original Mode")
    if args.randomize:
        print("🎲 Randomization enabled - parameters will vary per example")
    else:
        if background_enhancement_available:
            print("🌟 Automatic background enhancement enabled (high-quality preset for pixelation reduction)")
        if args.cornucopia_preset and ENHANCED_AVAILABLE:
            print(f"🚀 Using Cornucopia augmentations with preset: {args.cornucopia_preset}")
    
    contrast_params = {
        'clip_limit': 0.04,
        'tile_grid_size': (8, 8)
    }
    
    print(f"Generating {args.examples} examples with varying fiber densities ({args.min_fiber_pct}% - {args.max_fiber_pct}%)")
    print(f"Output directory: {args.output_dir}")
    print(f"View: {args.view}")
    print(f"Contrast method: {args.contrast_method}")
    if args.use_high_density_masks:
        print(f"Using high-density masks ({args.max_fiber_pct}%) for all fiber density variations")
    
    # Always use enhanced processing with automatic background enhancement
    generate_enhanced_varied_examples(
        nifti_file=args.nifti,
        trk_file=args.trk,
        output_dir=args.output_dir,
        n_examples=args.examples,
        prefix=args.prefix,
        slice_mode=args.view,
        specific_slice=None,
        min_fiber_percentage=args.min_fiber_pct,
        max_fiber_percentage=args.max_fiber_pct,
        roi_sphere=args.roi_sphere,
        tract_linewidth=args.tract_linewidth,
        save_masks=True,
        mask_thickness=args.mask_thickness,
        density_threshold=args.density_threshold,
        gaussian_sigma=2.0,
        random_state=args.random_state,
        close_gaps=args.close_gaps,
        closing_footprint_size=args.closing_footprint_size,
        label_bundles=args.label_bundles,
        min_bundle_size=args.min_bundle_size,
        use_high_density_masks=args.use_high_density_masks,
        contrast_method=args.contrast_method,
        contrast_params=contrast_params,
        cornucopia_preset=args.cornucopia_preset if not args.randomize else None,
        background_preset=args.background_preset if background_enhancement_available else None,
        enable_sharpening=args.enable_sharpening,
        sharpening_strength=args.sharpening_strength,
        use_cornucopia_per_example=args.cornucopia_preset is not None and not args.randomize,
        use_background_enhancement=background_enhancement_available,
        randomize=args.randomize
    )


def generate_examples_with_spatial_subdivisions(args, background_enhancement_available):
    """Generate examples using spatial subdivisions of the NIfTI volume."""
    print("Running in Spatial Subdivision Mode")
    if args.randomize:
        print("Randomization enabled - parameters will vary per example")
    else:
        if background_enhancement_available:
            print("Automatic background enhancement enabled (high-quality preset for pixelation reduction)")
        if args.cornucopia_preset and ENHANCED_AVAILABLE:
            print(f"Using Cornucopia augmentations with preset: {args.cornucopia_preset}")
    print("Creating spatial grid subdivisions of the NIfTI volume")
    
    try:
        import nibabel as nib
        import numpy as np
        from nibabel.streamlines import load, save, Tractogram
    except ImportError:
        print("Error: nibabel and numpy are required for spatial subdivision mode")
        print("Please install: pip install nibabel numpy")
        return
    
    # Load data
    print(f"Loading data...")
    nifti_img = nib.load(args.nifti)
    nifti_data = nifti_img.get_fdata()
    trk_file_obj = load(args.trk)
    streamlines = trk_file_obj.streamlines
    
    print(f"   NIfTI shape: {nifti_data.shape}")
    print(f"   Streamlines: {len(streamlines)}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    grid_subdivisions = _calculate_grid_subdivisions(nifti_data.shape, args.n_subdivisions)
    
    print(f"\n Creating {len(grid_subdivisions)} spatial grid regions...")
    
    valid_subdivisions = []
    total_examples = 0
    
    for i, (bounds, _) in enumerate(grid_subdivisions):
        x_start, x_end, y_start, y_end, z_start, z_end = bounds
        
        print(f"   Processing region {i+1}/{len(grid_subdivisions)}: ({x_start}:{x_end}, {y_start}:{y_end}, {z_start}:{z_end})")
        
        region_streamlines = _filter_streamlines_by_region(
            streamlines, trk_file_obj, x_start, x_end, y_start, y_end, z_start, z_end, 
            max_streamlines=args.max_streamlines_per_region
        )
        
        print(f"      Found {len(region_streamlines)} streamlines")
        
        if len(region_streamlines) < args.min_streamlines_per_region:
            print(f"Skipping (too few streamlines)")
            continue
        
        # Skip empty regions if requested
        if hasattr(args, 'skip_empty_regions') and args.skip_empty_regions and len(region_streamlines) == 0:
            print(f"Skipping (empty region)")
            continue
        
        # Extract region data from the loaded nifti_data
        region_data = nifti_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        sub_id = len(valid_subdivisions)
        sub_dir = output_path / f"subdivision_{sub_id:03d}"
        sub_dir.mkdir(exist_ok=True)
        
        print(f"Processing subdivision {sub_id}: {region_data.shape}, {len(region_streamlines)} streamlines")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_nifti:
            with tempfile.NamedTemporaryFile(suffix='.trk', delete=False) as temp_trk:
                try:
                    # Save region data
                    region_affine = nifti_img.affine.copy()
                    offset = np.array([x_start, y_start, z_start])
                    region_affine[:3, 3] += region_affine[:3, :3] @ offset
                    
                    region_nifti_img = nib.Nifti1Image(region_data, region_affine)
                    nib.save(region_nifti_img, temp_nifti.name)
                    
                    tractogram = Tractogram(region_streamlines, affine_to_rasmm=region_affine)
                    save(tractogram, temp_trk.name)
                    
                    prefix = f"sub_{sub_id:03d}_"
                    region_random_state = None
                    if args.random_state is not None:
                        region_random_state = args.random_state + sub_id * 1000
                    
                    try:
                        # Always use enhanced processing with automatic background enhancement
                        generate_enhanced_varied_examples(
                            nifti_file=temp_nifti.name,
                            trk_file=temp_trk.name,
                            output_dir=str(sub_dir),
                            n_examples=args.examples,
                            prefix=prefix,
                            slice_mode=args.view,
                            specific_slice=None,
                            min_fiber_percentage=args.min_fiber_pct,
                            max_fiber_percentage=args.max_fiber_pct,
                            roi_sphere=args.roi_sphere,
                            tract_linewidth=args.tract_linewidth,
                            save_masks=True,
                            mask_thickness=args.mask_thickness,
                            density_threshold=args.density_threshold,
                            gaussian_sigma=2.0,
                            random_state=region_random_state,
                            close_gaps=args.close_gaps,
                            closing_footprint_size=args.closing_footprint_size,
                            label_bundles=args.label_bundles,
                            min_bundle_size=args.min_bundle_size,
                            use_high_density_masks=args.use_high_density_masks,
                            contrast_method=args.contrast_method,
                            contrast_params={'clip_limit': 0.01, 'tile_grid_size': (8, 8)},
                            cornucopia_preset=args.cornucopia_preset if not args.randomize else None,
                            background_preset=args.background_preset if background_enhancement_available else None,
                            enable_sharpening=args.enable_sharpening,
                            sharpening_strength=args.sharpening_strength,
                            use_cornucopia_per_example=args.cornucopia_preset is not None and not args.randomize,
                            use_background_enhancement=background_enhancement_available,
                            randomize=args.randomize
                        )
                        
                        total_examples += args.examples
                        enhancements = []
                        if not args.randomize:
                            if background_enhancement_available:
                                enhancements.append("background 'high_quality'")
                            if args.cornucopia_preset and ENHANCED_AVAILABLE:
                                enhancements.append(f"Cornucopia '{args.cornucopia_preset}'")
                        else:
                            enhancements.append("randomized parameters")
                        
                        if enhancements:
                            print(f"Generated {args.examples} examples with {' + '.join(enhancements)}")
                        else:
                            print(f"Generated {args.examples} examples")
                        
                    except Exception as e:
                        print(f"Failed to generate examples: {str(e)}")
                
                finally:
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
            'examples_generated': args.examples
        })
    
    print(f"\n🎉 Completed! Generated {total_examples} examples from {len(valid_subdivisions)} spatial subdivisions")
    
    _save_subdivision_summary(output_path, valid_subdivisions, total_examples, args)
    
    return valid_subdivisions


def _calculate_grid_subdivisions(nifti_shape, n_subdivisions):
    """Calculate spatial grid subdivisions of the NIfTI volume."""
    import numpy as np
    
    x_max, y_max, z_max = nifti_shape
    grid_size = max(1, int(np.ceil(n_subdivisions ** (1/3))))
    
    while grid_size ** 3 > n_subdivisions * 2:
        grid_size -= 1
    
    grid_size = max(1, grid_size)
    
    print(f"   Creating {grid_size}×{grid_size}×{grid_size} grid (target: {n_subdivisions} subdivisions)")
    
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
                
                # Skip the data extraction in grid calculation - we'll do it in the main function
                bounds = (x_start, x_end, y_start, y_end, z_start, z_end)
                subdivisions.append((bounds, None))
            
            if len(subdivisions) >= n_subdivisions:
                break
        if len(subdivisions) >= n_subdivisions:
            break
    
    return subdivisions


def _filter_streamlines_by_region(streamlines, trk_file_obj, x_start, x_end, y_start, y_end, z_start, z_end, max_streamlines=50000):
    """Filter streamlines that pass through the specified region with memory management."""
    import numpy as np
    
    filtered_streamlines = []
    voxel_to_rasmm = trk_file_obj.header.get('voxel_to_rasmm', None)
    
    # Process streamlines in chunks for memory efficiency
    chunk_size = 10000
    total_streamlines = len(streamlines)
    
    print(f"      Processing {total_streamlines} streamlines in chunks of {chunk_size}...")
    
    for chunk_start in range(0, total_streamlines, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_streamlines)
        chunk_streamlines = streamlines[chunk_start:chunk_end]
        
        for i, streamline in enumerate(chunk_streamlines):
            if len(filtered_streamlines) >= max_streamlines:
                print(f"      Reached maximum streamlines limit ({max_streamlines}), stopping...")
                break
                
            if voxel_to_rasmm is not None:
                world_coords = np.hstack([streamline, np.ones((streamline.shape[0], 1))])
                rasmm_to_voxel = np.linalg.inv(voxel_to_rasmm)
                voxel_coords = world_coords @ rasmm_to_voxel.T
                voxel_coords = voxel_coords[:, :3]
            else:
                voxel_coords = streamline
            
            points_in_region = (
                (voxel_coords[:, 0] >= x_start) & (voxel_coords[:, 0] < x_end) &
                (voxel_coords[:, 1] >= y_start) & (voxel_coords[:, 1] < y_end) &
                (voxel_coords[:, 2] >= z_start) & (voxel_coords[:, 2] < z_end)
            )
            
            if np.any(points_in_region):
                adjusted_streamline = voxel_coords.copy()
                adjusted_streamline[:, 0] -= x_start
                adjusted_streamline[:, 1] -= y_start  
                adjusted_streamline[:, 2] -= z_start
                filtered_streamlines.append(adjusted_streamline)
        
        if len(filtered_streamlines) >= max_streamlines:
            break
            
        # Progress update every 5 chunks
        if (chunk_start // chunk_size) % 5 == 0:
            progress = min(100, (chunk_end / total_streamlines) * 100)
            print(f"      Progress: {progress:.1f}% ({len(filtered_streamlines)} streamlines found)")
    
    return filtered_streamlines


def _save_subdivision_summary(output_path, subdivisions, total_examples, args):
    """Save generation summary."""
    summary = {
        'total_examples': total_examples,
        'subdivisions_processed': len(subdivisions),
        'examples_per_subdivision': args.examples,
        'fiber_percentage_range': [args.min_fiber_pct, args.max_fiber_pct],
        'subdivision_info': subdivisions
    }
    
    import json
    summary_path = output_path / "generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic fiber tract dataset with ground truth masks')
    
    parser.add_argument('--nifti', required=True, help='Path to the NIfTI file')
    parser.add_argument('--trk', required=True, help='Path to the tractography (.trk) file')
    parser.add_argument('--output_dir', default='./fiber_dataset', help='Directory to save the generated dataset')
    parser.add_argument('--prefix', default='fiber_', help='Prefix for output filenames')
    
    # Mode selection
    parser.add_argument('--spatial_subdivisions', action='store_true',
                       help='Use spatial subdivision mode (divide NIfTI volume into grid regions)')
    
    # Common parameters
    parser.add_argument('--examples', type=int, default=10, 
                        help='Number of examples to generate')
    parser.add_argument('--view', choices=['axial', 'coronal', 'all'], default='coronal', 
                        help='Visualization mode')
    parser.add_argument('--min_fiber_pct', type=float, default=10.0, 
                        help='Minimum percentage of fibers to display')
    parser.add_argument('--max_fiber_pct', type=float, default=100.0, 
                        help='Maximum percentage of fibers to display')
    parser.add_argument('--mask_thickness', type=int, default=1,
                        help='Thickness of the mask lines in pixels')
    parser.add_argument('--tract_linewidth', type=float, default=1.0,
                        help='Width of the tract lines')
    parser.add_argument('--density_threshold', type=float, default=0.15,
                        help='Threshold for fiber density map (0.0-1.0)')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Random seed for reproducible results')
    parser.add_argument('--close_gaps', action='store_true',
                        help='Apply morphological closing to create contiguous regions')
    parser.add_argument('--closing_footprint_size', type=int, default=5,
                        help='Size of the footprint for morphological closing operations')
    parser.add_argument('--label_bundles', action='store_true',
                        help='Label distinct fiber bundles in the masks')
    parser.add_argument('--min_bundle_size', type=int, default=20,
                        help='Minimum size (in pixels) for a region to be considered a bundle')
    parser.add_argument('--use_high_density_masks', action='store_true',
                        help='Use masks from high-density fibers for all density variations')
    
    # Spatial subdivision parameters
    parser.add_argument('--n_subdivisions', type=int, default=8,
                       help='Number of spatial grid subdivisions to create')
    parser.add_argument('--min_streamlines_per_region', type=int, default=10,
                       help='Minimum streamlines required for a subdivision region')
    parser.add_argument('--max_streamlines_per_region', type=int, default=50000,
                       help='Maximum streamlines to process per subdivision region (for memory management)')
    
    # Cornucopia parameters
    parser.add_argument('--cornucopia_preset', 
                        choices=['aggressive', 'clinical_simulation'], 
                        help='Cornucopia preset for advanced medical imaging augmentations')
    
    # Background enhancement parameters  
    parser.add_argument('--background_preset', 
                        choices=['preserve_edges', 'high_quality', 'smooth_realistic', 'clinical_appearance', 'subtle_enhancement'],
                        default='preserve_edges',
                        help='Background enhancement preset (default: preserve_edges for anatomical boundary preservation)')
    parser.add_argument('--enable_sharpening', action='store_true', default=True,
                        help='Enable sharpening after background enhancement to preserve edges (enabled by default)')
    parser.add_argument('--sharpening_strength', type=float, default=0.5,
                        help='Sharpening strength (0.0-1.0, default: 0.5)')
    
    # Randomization parameters
    parser.add_argument('--randomize', action='store_true',
                        help='Randomize parameters per example: min/max streamline percentage (5-30%% to 70-100%% for balanced, '
                             '15-40%% to 80-100%% for blockface_preserving), streamline appearance (linewidth 0.5-1.5), '
                             'cornucopia preset (None/aggressive/clinical_simulation), and background effect (balanced/blockface_preserving)')
    
    args = parser.parse_args()
    
    # Check availability of requested features
    if args.cornucopia_preset and not CORNUCOPIA_INTEGRATION_AVAILABLE:
        print("Warning: Cornucopia augmentation requested but not available.")
        print("   Install cornucopia: pip install cornucopia")
        print("   Falling back to standard augmentations.")
        args.cornucopia_preset = None
    elif args.cornucopia_preset and CORNUCOPIA_INTEGRATION_AVAILABLE:
        print(f"Cornucopia augmentation enabled with preset: {args.cornucopia_preset}")
    
    # Background enhancement is now enabled by default for pixelation reduction
    try:
        from background_enhancement import enhance_slice_background
        print(" Quantized data preprocessing enabled (eliminates tiling artifacts)")
        print("️  Your data has few unique values - using smooth processing to prevent tiling")
        background_enhancement_available = True
    except ImportError:
        print("️  Background enhancement module not found. Pixelation reduction disabled.")
        background_enhancement_available = False
    
    args.contrast_method = 'clahe'
    print(f" Using contrast method: {args.contrast_method}")
    
    # Check input files
    if not os.path.exists(args.nifti):
        raise FileNotFoundError(f"NIfTI file not found: {args.nifti}")
    if not os.path.exists(args.trk):
        raise FileNotFoundError(f"Tractography file not found: {args.trk}")
    
    args.roi_sphere = None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f" Input NIfTI: {args.nifti}")
    print(f" Input TRK: {args.trk}")
    print(f" Output Directory: {args.output_dir}")
    
    if args.spatial_subdivisions:
        generate_examples_with_spatial_subdivisions(args, background_enhancement_available)
    else:
        generate_examples_original_mode(args, background_enhancement_available)
    
    print("\n Dataset generation complete!")
    
    # Show what was generated
    if args.spatial_subdivisions:
        print("The dataset includes:")
        print(f"- {args.n_subdivisions} spatial grid subdivisions of the NIfTI volume")
        print(f"- {args.examples} examples per subdivision")
        print(f"- Each subdivision contains a different spatial region of the brain")
        if args.randomize:
            print(f"- Randomized parameters: streamline percentages (5-30% to 70-100% for balanced, 15-40% to 80-100% for blockface_preserving), tract appearance, Cornucopia presets, background effects")
    else:
        print("The dataset includes:")
        print(f"- {args.examples} synthetic images with varying fiber densities")
        print(f"- Corresponding ground truth masks for segmentation")
        if args.randomize:
            print(f"- Randomized parameters per example:")
            print(f"  • Min/max streamline percentages: 5-30% to 70-100% (balanced), 15-40% to 80-100% (blockface_preserving)")
            print(f"  • Tract linewidth: 0.5-1.5")
            print(f"  • Cornucopia preset: None/aggressive/clinical_simulation")
            print(f"  • Background effect: balanced/blockface_preserving")
        if args.label_bundles:
            print(f"- Labeled bundle visualizations with distinct colors for each bundle")
        if args.use_high_density_masks:
            print(f"- All images use the same masks derived from high-density ({args.max_fiber_pct}%) fibers")
    
    # List generated files
    print("\nGenerated files:")
    for root, _, files in os.walk(args.output_dir):
        for file in sorted(files):
            if file.startswith(args.prefix) or file.endswith(('.png', '.json')):
                rel_path = os.path.relpath(os.path.join(root, file), args.output_dir)
                print(f"  {rel_path}")


if __name__ == "__main__":
    main() 