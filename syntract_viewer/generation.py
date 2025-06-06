"""
Example generation functions for creating synthetic fiber tract datasets.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.streamlines import load
from dipy.tracking.streamline import transform_streamlines
import random
from pathlib import Path

from core import visualize_nifti_with_trk, visualize_nifti_with_trk_coronal, visualize_multiple_views
from contrast import apply_enhanced_contrast_and_augmentation, CORNUCOPIA_INTEGRATION_AVAILABLE
from masking import create_aggressive_brain_mask, create_fiber_mask
from effects import apply_smart_dark_field_effect
from utils import select_random_streamlines, densify_streamline, generate_tract_color_variation, get_colormap


def generate_varied_examples(nifti_file, trk_file, output_dir, n_examples=5, prefix="synthetic_", 
                             slice_mode="coronal", intensity_variation=True, tract_color_variation=True,
                             specific_slice=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1, 
                             min_fiber_percentage=10.0, max_fiber_percentage=100.0,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, use_high_density_masks=False,
                             contrast_method='clahe', contrast_params=None):
    """
    Generate multiple varied examples with different contrast settings.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    if contrast_params is None:
        contrast_params = {
            'clip_limit': 0.01,
            'tile_grid_size': (8, 8)
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine slice index if specific_slice is provided
    if specific_slice is not None:
        nii_img = nib.load(nifti_file)
        dims = nii_img.shape
        
        if slice_mode == "axial" and (specific_slice < 0 or specific_slice >= dims[2]):
            print(f"Warning: Slice index {specific_slice} out of bounds for axial dimension {dims[2]}")
            specific_slice = dims[2] // 2
        elif slice_mode == "coronal" and (specific_slice < 0 or specific_slice >= dims[1]):
            print(f"Warning: Slice index {specific_slice} out of bounds for coronal dimension {dims[1]}")
            specific_slice = dims[1] // 2
    
    # Use fiber density range if provided
    use_fiber_range = (min_fiber_percentage > 0 and max_fiber_percentage > min_fiber_percentage)
    
    # Generate high-density masks if requested
    high_density_masks = {}
    high_density_labeled_masks = {}
    
    if use_high_density_masks and save_masks:
        print(f"Generating high-density masks using {max_fiber_percentage}% of fibers...")
        _generate_high_density_masks(
            nifti_file, trk_file, output_dir, prefix, slice_mode, specific_slice,
            max_fiber_percentage, roi_sphere, tract_linewidth, mask_thickness,
            density_threshold, gaussian_sigma, close_gaps, closing_footprint_size,
            label_bundles, min_bundle_size, contrast_method, contrast_params,
            high_density_masks, high_density_labeled_masks, random_state
        )
    
    # Generate examples
    for i in range(n_examples):
        example_random_state = None
        if random_state is not None:
            example_random_state = random_state + i
        
        # Generate random parameters
        if intensity_variation:
            if i % 3 == 0:
                color_scheme = 'bw'
                blue_tint = 0
            elif i % 3 == 1:
                color_scheme = 'blue'
                blue_tint = random.uniform(0.2, 0.4)
            else:
                color_scheme = random.choices(['bw', 'blue'], weights=[0.3, 0.7])[0]
                blue_tint = random.uniform(0.1, 0.4) if color_scheme == 'blue' else 0
            
            intensity_params = {
                'gamma': random.uniform(0.8, 1.2),
                'threshold': random.uniform(0.02, 0.08),
                'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
                'background_boost': random.uniform(0.9, 1.1),
                'color_scheme': color_scheme,
                'blue_tint': blue_tint
            }
        else:
            intensity_params = None
            
        # Generate random tract color
        if tract_color_variation:
            base_r = random.uniform(0.9, 1.0)
            base_g = random.uniform(0.9, 1.0)
            base_b = random.uniform(0.0, 0.2)
            tract_color_base = (base_r, base_g, base_b)
            color_var = random.uniform(0.05, 0.2)
        else:
            tract_color_base = (1.0, 1.0, 0.0)
            color_var = 0.0
            
        output_file = os.path.join(output_dir, f"{prefix}{i+1:03d}.png")
        
        # Vary fiber density if range is specified
        if use_fiber_range:
            if n_examples > 1:
                t = i / (n_examples - 1)
                fiber_pct = min_fiber_percentage + t * (max_fiber_percentage - min_fiber_percentage)
            else:
                fiber_pct = (min_fiber_percentage + max_fiber_percentage) / 2
            print(f"Example {i+1}: Using {fiber_pct:.1f}% of fibers")
        else:
            fiber_pct = streamline_percentage
        
        # Determine whether to save masks
        should_save_masks = save_masks and (not use_high_density_masks or fiber_pct >= max_fiber_percentage)
        
        # Generate visualization
        if slice_mode == "axial":
            visualize_nifti_with_trk(
                nifti_file, trk_file, output_file, n_slices=1,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                slice_idx=specific_slice,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                _apply_high_density_masks_axial(
                    output_file, high_density_masks, high_density_labeled_masks,
                    specific_slice, label_bundles, nifti_file
                )
            
        elif slice_mode == "coronal":
            visualize_nifti_with_trk_coronal(
                nifti_file, trk_file, output_file, n_slices=1,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                slice_idx=specific_slice,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                _apply_high_density_masks_coronal(
                    output_file, high_density_masks, high_density_labeled_masks,
                    specific_slice, label_bundles, nifti_file
                )
            
        else:  # "all"
            visualize_multiple_views(
                nifti_file, trk_file, output_file,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                streamline_percentage=fiber_pct,
                roi_sphere=roi_sphere,
                tract_linewidth=tract_linewidth,
                save_masks=should_save_masks,
                mask_thickness=mask_thickness,
                density_threshold=density_threshold,
                gaussian_sigma=gaussian_sigma,
                random_state=example_random_state,
                close_gaps=close_gaps,
                closing_footprint_size=closing_footprint_size,
                label_bundles=label_bundles,
                min_bundle_size=min_bundle_size,
                contrast_method=contrast_method,
                contrast_params=contrast_params
            )
            
            # Apply high-density masks if requested
            if use_high_density_masks and save_masks and fiber_pct < max_fiber_percentage:
                _apply_high_density_masks_multiview(
                    output_file, high_density_masks, high_density_labeled_masks, label_bundles
                )
        
        print(f"Generated example {i+1}/{n_examples}: {output_file}")


def generate_enhanced_varied_examples(nifti_file, trk_file, output_dir, 
                                    n_examples=5, prefix="enhanced_",
                                    slice_mode="coronal", specific_slice=None,
                                    streamline_percentage=100.0, roi_sphere=None,
                                    tract_linewidth=1.0, save_masks=False,
                                    min_fiber_percentage=10.0, max_fiber_percentage=100.0,
                                    contrast_method='clahe', contrast_params=None,
                                    cornucopia_preset='clinical_simulation',
                                    use_cornucopia_per_example=True,
                                    random_state=None, **kwargs):
    """
    Generate varied examples with enhanced augmentations using Cornucopia.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up augmentation configuration
    augmentation_config = None
    if use_cornucopia_per_example and CORNUCOPIA_INTEGRATION_AVAILABLE:
        try:
            from .cornucopia_augmentation import create_augmentation_presets
            presets = create_augmentation_presets()
            if cornucopia_preset in presets:
                augmentation_config = presets[cornucopia_preset]
            else:
                print(f"Warning: Unknown Cornucopia preset '{cornucopia_preset}', using 'standard'")
                augmentation_config = presets.get('standard', None)
        except Exception as e:
            print(f"Warning: Failed to create augmentation config: {e}")
            augmentation_config = None
    
    if random_state is not None:
        np.random.seed(random_state)
    
    if contrast_params is None:
        contrast_params = {
            'clip_limit': 0.01,
            'tile_grid_size': (8, 8)
        }
    
    print(f"🚀 Generating {n_examples} enhanced examples with Cornucopia augmentations")
    print(f"   Contrast method: {contrast_method}")
    print(f"   Cornucopia preset: {cornucopia_preset}")
    print(f"   Cornucopia available: {CORNUCOPIA_INTEGRATION_AVAILABLE}")
    
    # Generate enhanced examples
    if use_cornucopia_per_example and CORNUCOPIA_INTEGRATION_AVAILABLE and augmentation_config is not None:
        print("🎨 Generating examples with Cornucopia applied to NIfTI slice data...")
        
        base_results = _generate_examples_with_slice_augmentation(
            nifti_file=nifti_file,
            trk_file=trk_file,
            output_dir=output_dir,
            n_examples=n_examples,
            prefix=prefix,
            slice_mode=slice_mode,
            specific_slice=specific_slice,
            streamline_percentage=streamline_percentage,
            roi_sphere=roi_sphere,
            tract_linewidth=tract_linewidth,
            save_masks=save_masks,
            min_fiber_percentage=min_fiber_percentage,
            max_fiber_percentage=max_fiber_percentage,
            contrast_method=contrast_method,
            contrast_params=contrast_params,
            cornucopia_config=augmentation_config,
            random_state=random_state,
            **kwargs
        )
    else:
        # Generate standard examples
        base_results = generate_varied_examples(
            nifti_file=nifti_file,
            trk_file=trk_file,
            output_dir=output_dir,
            n_examples=n_examples,
            prefix=prefix,
            slice_mode=slice_mode,
            specific_slice=specific_slice,
            streamline_percentage=streamline_percentage,
            roi_sphere=roi_sphere,
            tract_linewidth=tract_linewidth,
            save_masks=save_masks,
            min_fiber_percentage=min_fiber_percentage,
            max_fiber_percentage=max_fiber_percentage,
            contrast_method=contrast_method,
            contrast_params=contrast_params,
            random_state=random_state,
            **kwargs
        )
    
    # Create summary report
    summary = {
        'n_examples_generated': n_examples,
        'cornucopia_available': CORNUCOPIA_INTEGRATION_AVAILABLE,
        'cornucopia_preset_used': cornucopia_preset if use_cornucopia_per_example else None,
        'contrast_method': contrast_method,
        'output_directory': str(output_dir),
        'base_results': base_results
    }
    
    return summary


def _generate_examples_with_slice_augmentation(nifti_file, trk_file, output_dir, 
                                             n_examples=5, prefix="enhanced_",
                                             slice_mode="coronal", specific_slice=None,
                                             streamline_percentage=100.0, roi_sphere=None,
                                             tract_linewidth=1.0, save_masks=False,
                                             min_fiber_percentage=10.0, max_fiber_percentage=100.0,
                                             contrast_method='clahe', contrast_params=None,
                                             cornucopia_config=None, random_state=None, **kwargs):
    """Generate examples with Cornucopia augmentation applied to NIfTI slice data."""
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()
    dims = nii_data.shape
    
    tractogram = load(trk_file)
    streamlines = tractogram.streamlines
    print(f"Loaded {len(streamlines)} streamlines from {trk_file}")
    
    affine_inv = np.linalg.inv(nii_img.affine)
    streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
    
    # Determine slice index
    if specific_slice is not None:
        slice_idx = specific_slice
    else:
        if slice_mode == "coronal":
            slice_idx = dims[1] // 2
        elif slice_mode == "axial":
            slice_idx = dims[2] // 2
        else:  # sagittal
            slice_idx = dims[0] // 2
    
    if contrast_params is None:
        contrast_params = {'clip_limit': 0.01, 'tile_grid_size': (8, 8)}
    
    # Generate examples
    for i in range(n_examples):
        example_random_state = None
        if random_state is not None:
            example_random_state = random_state + i
            random.seed(example_random_state)
            np.random.seed(example_random_state)
        
        # Calculate fiber percentage
        if n_examples > 1:
            t = i / (n_examples - 1)
            fiber_pct = min_fiber_percentage + t * (max_fiber_percentage - min_fiber_percentage)
        else:
            fiber_pct = (min_fiber_percentage + max_fiber_percentage) / 2
        
        # Select streamlines
        selected_streamlines = select_random_streamlines(
            streamlines_voxel, fiber_pct, random_state=example_random_state
        )
        
        print(f"Example {i+1}: Using {fiber_pct:.1f}% of fibers ({len(selected_streamlines)} streamlines)")
        
        # Extract slice data
        if slice_mode == "coronal":
            slice_data = nii_data[:, slice_idx, :]
        elif slice_mode == "axial":
            slice_data = nii_data[:, :, slice_idx]
        else:  # sagittal
            slice_data = nii_data[slice_idx, :, :]
        
        # Apply Cornucopia augmentation and contrast enhancement
        enhanced_slice = apply_enhanced_contrast_and_augmentation(
            slice_data,
            contrast_method=contrast_method,
            contrast_params=contrast_params,
            cornucopia_augmentation=cornucopia_config,
            random_state=example_random_state
        )
        
        # Safety check
        if np.all(enhanced_slice == 0) or np.std(enhanced_slice) < 1e-6:
            print(f"   ⚠️  Enhanced slice is empty, using original slice data")
            enhanced_slice = slice_data.copy()
            from .contrast import apply_contrast_enhancement
            enhanced_slice = apply_contrast_enhancement(
                enhanced_slice, 
                clip_limit=contrast_params.get('clip_limit', 0.01),
                tile_grid_size=contrast_params.get('tile_grid_size', (8, 8))
            )
        
        # Generate visualization
        _create_enhanced_visualization(
            enhanced_slice, selected_streamlines, slice_mode, slice_idx, dims,
            output_dir, prefix, i, save_masks, cornucopia_config, slice_data,
            tract_linewidth, example_random_state, **kwargs
        )
    
    return {'examples_generated': n_examples}


def _create_enhanced_visualization(enhanced_slice, selected_streamlines, slice_mode, slice_idx, dims,
                                 output_dir, prefix, example_idx, save_masks, cornucopia_config, slice_data,
                                 tract_linewidth, example_random_state, **kwargs):
    """Create enhanced visualization with streamlines."""
    # Generate intensity parameters
    intensity_params = {
        'gamma': random.uniform(0.8, 1.2),
        'threshold': random.uniform(0.02, 0.08),
        'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
        'background_boost': random.uniform(0.9, 1.1),
        'color_scheme': random.choice(['bw', 'blue']),
        'blue_tint': random.uniform(0.1, 0.4)
    }
    
    # Apply dark field effect
    dark_field_slice = apply_smart_dark_field_effect(
        enhanced_slice, intensity_params, random_state=example_random_state
    )
    
    # Additional background cleanup for Cornucopia artifacts
    if cornucopia_config is not None:
        brain_mask = create_aggressive_brain_mask(slice_data, enhanced_slice)
        background_areas = ~brain_mask.astype(bool)
        dark_field_slice[background_areas] = 0
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    
    # Get colormap
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)
    
    # Display brain slice
    brain_min = np.min(dark_field_slice[dark_field_slice > 0]) if np.any(dark_field_slice > 0) else 0
    brain_max = np.max(dark_field_slice)
    
    ax.imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', 
             interpolation='bicubic', vmin=brain_min, vmax=brain_max)
    ax.set_facecolor('black')
    
    # Create mask if requested
    if save_masks:
        mask = create_fiber_mask(
            selected_streamlines, slice_idx, orientation=slice_mode,
            dims=dims, thickness=kwargs.get('mask_thickness', 1),
            density_threshold=kwargs.get('density_threshold', 0.15),
            gaussian_sigma=kwargs.get('gaussian_sigma', 2.0),
            min_bundle_size=kwargs.get('min_bundle_size', 20)
        )
        mask = np.rot90(mask)
    
    # Overlay streamlines
    _add_streamlines_to_plot(ax, selected_streamlines, slice_mode, slice_idx, dims, tract_linewidth, example_random_state)
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save result
    output_file = os.path.join(output_dir, f"{prefix}{example_idx+1:03d}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
    print(f"Generated example {example_idx+1}: {output_file}")
    
    # Save mask if requested
    if save_masks:
        mask_dir = os.path.dirname(output_file)
        mask_basename = os.path.splitext(os.path.basename(output_file))[0]
        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_idx}.png"
        plt.imsave(mask_filename, mask, cmap='gray')
        print(f"Saved mask for slice {slice_idx} to {mask_filename}")
    
    plt.close()


def _add_streamlines_to_plot(ax, streamlines, slice_mode, slice_idx, dims, tract_linewidth, random_state):
    """Add streamlines to the plot."""
    from matplotlib.collections import LineCollection
    
    segments = []
    colors = []
    
    for sl in streamlines:
        sl_dense = densify_streamline(sl)
        
        # Project streamline based on slice mode
        if slice_mode == "coronal":
            x = sl_dense[:, 0]
            y = sl_dense[:, 1]
            z = sl_dense[:, 2]
            distance_to_slice = np.abs(y - slice_idx)
            min_distance = np.min(distance_to_slice)
            if min_distance > 2.0:
                continue
            z_plot = dims[2] - z - 1
            points = np.array([x, z_plot]).T.reshape(-1, 1, 2)
        elif slice_mode == "axial":
            x = sl_dense[:, 0]
            y = sl_dense[:, 1]
            z = sl_dense[:, 2]
            distance_to_slice = np.abs(z - slice_idx)
            min_distance = np.min(distance_to_slice)
            if min_distance > 2.0:
                continue
            y_plot = dims[1] - y - 1
            points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
        else:  # sagittal
            x = sl_dense[:, 0]
            y = sl_dense[:, 1]
            z = sl_dense[:, 2]
            distance_to_slice = np.abs(x - slice_idx)
            min_distance = np.min(distance_to_slice)
            if min_distance > 2.0:
                continue
            z_plot = dims[2] - z - 1
            points = np.array([y, z_plot]).T.reshape(-1, 1, 2)
        
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Generate varied color
        tract_color = generate_tract_color_variation(
            (1.0, 1.0, 0.0), 0.2, random_state=random_state
        )
        
        # Adjust opacity
        base_opacity = max(0.0, 1.0 - min_distance / 2.0)
        
        for seg in segs:
            segments.append(seg)
            colors.append(tract_color + (base_opacity,))
    
    # Add streamlines to plot
    if segments:
        linewidth = tract_linewidth * random.uniform(0.9, 1.1)
        lc = LineCollection(segments, colors=colors, linewidths=linewidth)
        ax.add_collection(lc)


# Helper functions for high-density mask handling
def _generate_high_density_masks(nifti_file, trk_file, output_dir, prefix, slice_mode, specific_slice,
                                max_fiber_percentage, roi_sphere, tract_linewidth, mask_thickness,
                                density_threshold, gaussian_sigma, close_gaps, closing_footprint_size,
                                label_bundles, min_bundle_size, contrast_method, contrast_params,
                                high_density_masks, high_density_labeled_masks, random_state):
    """Generate high-density masks for use across all examples."""
    # Load data to determine slice position
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    
    if specific_slice is None:
        if slice_mode == "axial":
            slice_idx = dims[2] // 2
        elif slice_mode == "coronal":
            slice_idx = dims[1] // 2
        else:  # sagittal
            slice_idx = dims[0] // 2
    else:
        slice_idx = specific_slice
    
    # Generate mask using high fiber density
    tractogram = load(trk_file)
    streamlines = tractogram.streamlines
    affine_inv = np.linalg.inv(nii_img.affine)
    streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
    
    # Select high percentage of streamlines
    selected_streamlines = select_random_streamlines(
        streamlines_voxel, max_fiber_percentage, random_state=random_state
    )
    
    # Create mask
    if label_bundles:
        mask, labeled_mask = create_fiber_mask(
            selected_streamlines, slice_idx, orientation=slice_mode,
            dims=dims, thickness=mask_thickness, dilate=True,
            density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
            close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
            label_bundles=True, min_bundle_size=min_bundle_size
        )
        high_density_masks[slice_idx] = np.rot90(mask)
        high_density_labeled_masks[slice_idx] = np.rot90(labeled_mask)
    else:
        mask = create_fiber_mask(
            selected_streamlines, slice_idx, orientation=slice_mode,
            dims=dims, thickness=mask_thickness, dilate=True,
            density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
            close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
            min_bundle_size=min_bundle_size
        )
        high_density_masks[slice_idx] = np.rot90(mask)


def _apply_high_density_masks_axial(output_file, high_density_masks, high_density_labeled_masks,
                                   specific_slice, label_bundles, nifti_file):
    """Apply high-density masks for axial view."""
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    
    slice_idx = specific_slice if specific_slice is not None else dims[2] // 2
    
    if slice_idx in high_density_masks:
        mask_dir = os.path.dirname(output_file)
        mask_basename = os.path.splitext(os.path.basename(output_file))[0]
        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_idx}.png"
        
        plt.imsave(mask_filename, high_density_masks[slice_idx], cmap='gray')
        print(f"Applied high-density mask for axial slice {slice_idx}: {mask_filename}")
        
        if label_bundles and slice_idx in high_density_labeled_masks:
            from .utils import visualize_labeled_bundles
            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_idx}.png"
            visualize_labeled_bundles(high_density_labeled_masks[slice_idx], labeled_filename)
            print(f"Applied high-density labeled bundles for axial slice {slice_idx}: {labeled_filename}")


def _apply_high_density_masks_coronal(output_file, high_density_masks, high_density_labeled_masks,
                                     specific_slice, label_bundles, nifti_file):
    """Apply high-density masks for coronal view."""
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    
    slice_idx = specific_slice if specific_slice is not None else dims[1] // 2
    
    if slice_idx in high_density_masks:
        mask_dir = os.path.dirname(output_file)
        mask_basename = os.path.splitext(os.path.basename(output_file))[0]
        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_idx}.png"
        
        plt.imsave(mask_filename, high_density_masks[slice_idx], cmap='gray')
        print(f"Applied high-density mask for coronal slice {slice_idx}: {mask_filename}")
        
        if label_bundles and slice_idx in high_density_labeled_masks:
            from .utils import visualize_labeled_bundles
            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_idx}.png"
            visualize_labeled_bundles(high_density_labeled_masks[slice_idx], labeled_filename)
            print(f"Applied high-density labeled bundles for coronal slice {slice_idx}: {labeled_filename}")


def _apply_high_density_masks_multiview(output_file, high_density_masks, high_density_labeled_masks, label_bundles):
    """Apply high-density masks for multiple views."""
    mask_dir = os.path.dirname(output_file)
    if not mask_dir:
        mask_dir = "../synthesis"
    mask_basename = os.path.splitext(os.path.basename(output_file))[0]
    
    for view, mask in high_density_masks.items():
        if mask is not None:
            mask_filename = f"{mask_dir}/{mask_basename}_mask_{view}.png"
            plt.imsave(mask_filename, mask, cmap='gray')
            print(f"Applied high-density mask for {view} view: {mask_filename}")
            
            if label_bundles and view in high_density_labeled_masks:
                from .utils import visualize_labeled_bundles
                labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_{view}.png"
                visualize_labeled_bundles(high_density_labeled_masks[view], labeled_filename)
                print(f"Applied high-density labeled bundles for {view} view: {labeled_filename}") 