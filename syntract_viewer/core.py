"""
Core visualization functions for NIfTI tractography data.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.streamlines import load
from matplotlib.collections import LineCollection
from dipy.tracking.streamline import transform_streamlines
import random

try:
    from .contrast import apply_contrast_enhancement, apply_comprehensive_slice_processing
    from .effects import apply_smart_dark_field_effect
    from .masking import create_fiber_mask
    from .utils import (
        densify_streamline, 
        generate_tract_color_variation,
        get_colormap,
        select_random_streamlines
    )
except ImportError:
    from contrast import apply_contrast_enhancement, apply_comprehensive_slice_processing
    from effects import apply_smart_dark_field_effect
    from masking import create_fiber_mask
    from utils import (
        densify_streamline, 
        generate_tract_color_variation,
        get_colormap,
        select_random_streamlines
    )


def visualize_nifti_with_trk(nifti_file, trk_file, output_file=None, n_slices=1, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=32, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2,
                             slice_idx=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None,
                             background_enhancement=None, cornucopia_augmentation=None):
    """
    Visualize multiple axial slices of a nifti file with tractography overlaid.
    
    Parameters
    ----------
    background_enhancement : str or dict, optional
        Background enhancement configuration to reduce pixelation
    cornucopia_augmentation : str or dict, optional
        Cornucopia augmentation configuration
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    if contrast_params is None:
        contrast_params = {
            'clip_limit': clahe_clip_limit,
            'tile_grid_size': (max(32, clahe_tile_grid_size), max(32, clahe_tile_grid_size))
        }
    
    # Load data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()
    dims = nii_data.shape

    # Load tractography
    try:
        tractogram = load(trk_file)
        streamlines = tractogram.streamlines
        has_streamlines = True
        print(f"Loaded {len(streamlines)} streamlines from {trk_file}")
    except Exception as e:
        print(f"Error loading tractography: {e}")
        has_streamlines = False

    # Convert streamlines to voxel coordinates
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
        
        if streamline_percentage < 100.0:
            print(f"Randomly selecting {streamline_percentage}% of streamlines")
            streamlines_voxel = select_random_streamlines(streamlines_voxel, streamline_percentage, random_state=random_state)
            print(f"Selected {len(streamlines_voxel)} streamlines")

    # Calculate slice positions
    if slice_idx is not None:
        slice_positions = [slice_idx]
        n_slices = 1
    else:
        slice_positions = np.linspace(dims[2] // 4, 3 * dims[2] // 4, n_slices).astype(int)

    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    # Default intensity parameters
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    fig.patch.set_facecolor('black')
    fiber_masks = []
    labeled_masks = []

    # Process each slice
    for i, slice_idx in enumerate(slice_positions):
        slice_intensity_params = intensity_params.copy() if intensity_params else None
        
        slice_data = nii_data[:, :, slice_idx]
        
        # Apply processing based on user preferences (don't force processing)
        slice_enhanced = apply_comprehensive_slice_processing(
            slice_data,
            background_preset=background_enhancement if isinstance(background_enhancement, str) else None,
            cornucopia_preset=cornucopia_augmentation if isinstance(cornucopia_augmentation, str) else None,
            contrast_method=contrast_method,
            background_params=background_enhancement if isinstance(background_enhancement, dict) else None,
            cornucopia_params=cornucopia_augmentation if isinstance(cornucopia_augmentation, dict) else None,
            contrast_params=contrast_params,
            random_state=random_state
        )
        
        dark_field_slice = apply_smart_dark_field_effect(
            slice_enhanced, 
            slice_intensity_params, 
            preserve_ventricles=False,  # Explicitly disable to prevent fake ventricles
            random_state=random_state
        )

        axes[i].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='bilinear')
        axes[i].set_facecolor('black')
        
        # Create mask if requested
        if save_masks and has_streamlines:
            if label_bundles:
                mask, labeled_mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='axial', 
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    label_bundles=True, min_bundle_size=min_bundle_size
                )
                mask = np.rot90(mask)
                labeled_mask = np.rot90(labeled_mask)
                fiber_masks.append(mask)
                labeled_masks.append(labeled_mask)
            else:
                mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='axial', 
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    min_bundle_size=min_bundle_size
                )
                mask = np.rot90(mask)
                fiber_masks.append(mask)
        
        # Add streamlines
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                x = sl_dense[:, 0]
                y = sl_dense[:, 1]
                z = sl_dense[:, 2]
                
                distance_to_slice = np.abs(z - slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > 2.0:
                    continue
                
                y_plot = dims[1] - y - 1
                points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state)
                base_opacity = max(0.0, 1.0 - min_distance / 2.0)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))
                    
            if segments:
                linewidth = tract_linewidth * random.uniform(0.9, 1.1)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[i].add_collection(lc)
        
        axes[i].axis('off')

    plt.tight_layout()

    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        print(f"Figure saved to {output_file}")
        
        # Save masks if requested
        if save_masks and fiber_masks:
            _save_masks(output_file, fiber_masks, labeled_masks, slice_positions, label_bundles)
    else:
        plt.show()

    if label_bundles and labeled_masks:
        return fig, axes, fiber_masks, labeled_masks
    else:
        return fig, axes, fiber_masks if save_masks else None


def visualize_nifti_with_trk_coronal(nifti_file, trk_file, output_file=None, n_slices=1, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=32, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2,
                             slice_idx=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None,
                             background_enhancement=None, cornucopia_augmentation=None):
    """
    Visualize multiple coronal slices of a nifti file with tractography overlaid.
    
    Parameters
    ----------
    background_enhancement : str or dict, optional
        Background enhancement configuration to reduce pixelation
    cornucopia_augmentation : str or dict, optional
        Cornucopia augmentation configuration
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    if contrast_params is None:
        contrast_params = {
            'clip_limit': clahe_clip_limit,
            'tile_grid_size': (max(32, clahe_tile_grid_size), max(32, clahe_tile_grid_size))
        }
    
    # Load data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()
    dims = nii_data.shape

    # Load tractography
    try:
        tractogram = load(trk_file)
        streamlines = tractogram.streamlines
        has_streamlines = True
        print(f"Loaded {len(streamlines)} streamlines from {trk_file}")
    except Exception as e:
        print(f"Error loading tractography: {e}")
        has_streamlines = False

    # Convert streamlines to voxel coordinates
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
        
        if streamline_percentage < 100.0:
            print(f"Randomly selecting {streamline_percentage}% of streamlines")
            streamlines_voxel = select_random_streamlines(streamlines_voxel, streamline_percentage, random_state=random_state)
            print(f"Selected {len(streamlines_voxel)} streamlines")

    # Calculate slice positions
    if slice_idx is not None:
        slice_positions = [slice_idx]
        n_slices = 1
    else:
        slice_positions = np.linspace(dims[1] // 4, 3 * dims[1] // 4, n_slices).astype(int)

    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    # Default intensity parameters
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    fig.patch.set_facecolor('black')
    fiber_masks = []
    labeled_masks = []

    # Process each slice
    for i, slice_idx in enumerate(slice_positions):
        slice_intensity_params = intensity_params.copy() if intensity_params else None
        
        slice_data = nii_data[:, slice_idx, :]
        
        # Apply processing based on user preferences (don't force processing)
        slice_enhanced = apply_comprehensive_slice_processing(
            slice_data,
            background_preset=background_enhancement if isinstance(background_enhancement, str) else None,
            cornucopia_preset=cornucopia_augmentation if isinstance(cornucopia_augmentation, str) else None,
            contrast_method=contrast_method,
            background_params=background_enhancement if isinstance(background_enhancement, dict) else None,
            cornucopia_params=cornucopia_augmentation if isinstance(cornucopia_augmentation, dict) else None,
            contrast_params=contrast_params,
            random_state=random_state
        )
        
        dark_field_slice = apply_smart_dark_field_effect(
            slice_enhanced, 
            slice_intensity_params, 
            preserve_ventricles=False,  # Explicitly disable to prevent fake ventricles
            random_state=random_state
        )

        axes[i].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='bilinear')
        axes[i].set_facecolor('black')
        
        # Create mask if requested
        if save_masks and has_streamlines:
            if label_bundles:
                mask, labeled_mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='coronal', 
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    label_bundles=True, min_bundle_size=min_bundle_size
                )
                mask = np.rot90(mask)
                labeled_mask = np.rot90(labeled_mask)
                fiber_masks.append(mask)
                labeled_masks.append(labeled_mask)
            else:
                mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='coronal', 
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    min_bundle_size=min_bundle_size
                )
                mask = np.rot90(mask)
                fiber_masks.append(mask)
        
        # Add streamlines
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                x = sl_dense[:, 0]
                y = sl_dense[:, 1]
                z = sl_dense[:, 2]
                
                distance_to_slice = np.abs(y - slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > 2.0:
                    continue
                
                z_plot = dims[2] - z - 1
                points = np.array([x, z_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state)
                base_opacity = max(0.0, 1.0 - min_distance / 2.0)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))
                    
            if segments:
                linewidth = tract_linewidth * random.uniform(0.9, 1.1)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[i].add_collection(lc)
        
        axes[i].axis('off')

    plt.tight_layout()

    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        print(f"Figure saved to {output_file}")
        
        # Save masks if requested
        if save_masks and fiber_masks:
            _save_masks(output_file, fiber_masks, labeled_masks, slice_positions, label_bundles)
    else:
        plt.show()

    if label_bundles and labeled_masks:
        return fig, axes, fiber_masks, labeled_masks
    else:
        return fig, axes, fiber_masks if save_masks else None


def visualize_multiple_views(nifti_file, trk_file, output_file=None, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=32, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2,
                             streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None):
    """
    Visualize axial, coronal, and sagittal views of a nifti file with tractography overlaid.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Load data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()
    dims = nii_data.shape

    # Load tractography
    try:
        tractogram = load(trk_file)
        streamlines = tractogram.streamlines
        has_streamlines = True
        print(f"Loaded {len(streamlines)} streamlines from {trk_file}")
    except Exception as e:
        print(f"Error loading tractography: {e}")
        has_streamlines = False

    # Convert streamlines to voxel coordinates
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
        
        if streamline_percentage < 100.0:
            print(f"Randomly selecting {streamline_percentage}% of streamlines")
            streamlines_voxel = select_random_streamlines(streamlines_voxel, streamline_percentage, random_state=random_state)
            print(f"Selected {len(streamlines_voxel)} streamlines")

    # Create figure with three orientations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Default intensity parameters
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    fig.patch.set_facecolor('black')
    distance_threshold = 1.0

    # Process each view
    views = ['axial', 'coronal', 'sagittal']
    slice_indices = [dims[2] // 2, dims[1] // 2, dims[0] // 2]
    fiber_masks = {}

    for view_idx, (view, slice_idx) in enumerate(zip(views, slice_indices)):
        if view == 'axial':
            slice_data = nii_data[:, :, slice_idx]
        elif view == 'coronal':
            slice_data = nii_data[:, slice_idx, :]
        else:  # sagittal
            slice_data = nii_data[slice_idx, :, :]
        
        # Apply processing based on user preferences (don't force processing)
        slice_enhanced = apply_comprehensive_slice_processing(
            slice_data,
            background_preset='high_quality',
            contrast_method='clahe',
            contrast_params={'clip_limit': clahe_clip_limit, 'tile_grid_size': (clahe_tile_grid_size, clahe_tile_grid_size)}
        )
        
        dark_field_slice = apply_smart_dark_field_effect(
            slice_enhanced, 
            intensity_params, 
            preserve_ventricles=False,  # Explicitly disable to prevent fake ventricles
            random_state=random_state
        )

        axes[view_idx].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='bilinear')
        axes[view_idx].set_facecolor('black')
        
        # Create mask if requested
        if save_masks and has_streamlines:
            mask = create_fiber_mask(
                streamlines_voxel, slice_idx, orientation=view, 
                dims=dims, thickness=mask_thickness, dilate=True,
                density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                min_bundle_size=min_bundle_size
            )
            mask = np.rot90(mask)
            fiber_masks[view] = mask

        # Add streamlines
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                
                if view == 'axial':
                    z = sl_dense[:, 2]
                    distance_to_slice = np.abs(z - slice_idx)
                    min_distance = np.min(distance_to_slice)
                    if min_distance > distance_threshold:
                        continue
                    x = sl_dense[:, 0]
                    y = sl_dense[:, 1]
                    y_plot = dims[1] - y - 1
                    points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
                elif view == 'coronal':
                    y = sl_dense[:, 1]
                    distance_to_slice = np.abs(y - slice_idx)
                    min_distance = np.min(distance_to_slice)
                    if min_distance > distance_threshold:
                        continue
                    x = sl_dense[:, 0]
                    z = sl_dense[:, 2]
                    z_plot = dims[2] - z - 1
                    points = np.array([x, z_plot]).T.reshape(-1, 1, 2)
                else:  # sagittal
                    x = sl_dense[:, 0]
                    distance_to_slice = np.abs(x - slice_idx)
                    min_distance = np.min(distance_to_slice)
                    if min_distance > distance_threshold:
                        continue
                    y = sl_dense[:, 1]
                    z = sl_dense[:, 2]
                    z_plot = dims[2] - z - 1
                    points = np.array([y, z_plot]).T.reshape(-1, 1, 2)
                
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state)
                base_opacity = max(0.0, 1.0 - min_distance / distance_threshold)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))
                    
            if segments:
                linewidth = tract_linewidth * random.uniform(0.9, 1.1)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[view_idx].add_collection(lc)

        axes[view_idx].axis('off')

    plt.tight_layout()

    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black', pad_inches=0)
        print(f"Figure saved to {output_file}")
        
        # Save masks if requested
        if save_masks and has_streamlines:
            mask_dir = os.path.dirname(output_file)
            if not mask_dir:
                mask_dir = "../synthesis"
            mask_basename = os.path.splitext(os.path.basename(output_file))[0]
            
            for view, mask in fiber_masks.items():
                if mask is not None:
                    mask_filename = f"{mask_dir}/{mask_basename}_mask_{view}.png"
                    plt.imsave(mask_filename, mask, cmap='gray')
                    print(f"Saved {view} mask to {mask_filename}")
    else:
        plt.show()

    return fig, axes, fiber_masks if save_masks else None


def _save_masks(output_file, fiber_masks, labeled_masks, slice_positions, label_bundles):
    """Save masks to files."""
    mask_dir = os.path.dirname(output_file)
    if not mask_dir:
        mask_dir = "../synthesis"
    mask_basename = os.path.splitext(os.path.basename(output_file))[0]
    
    for i, mask in enumerate(fiber_masks):
        slice_id = slice_positions[i]
        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_id}.png"
        plt.imsave(mask_filename, mask, cmap='gray')
        print(f"Saved mask for slice {slice_id} to {mask_filename}")
        
        if label_bundles and labeled_masks:
            from .utils import visualize_labeled_bundles
            labeled_mask = labeled_masks[i]
            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_id}.png"
            visualize_labeled_bundles(labeled_mask, labeled_filename)
            print(f"Saved labeled bundles for slice {slice_id} to {labeled_filename}") 