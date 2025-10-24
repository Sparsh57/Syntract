"""
Core visualization functions for NIfTI tractography data.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from nibabel.streamlines import load
from matplotlib.collections import LineCollection
from dipy.tracking.streamline import transform_streamlines
import random

# CRITICAL: Configure matplotlib to be aggressive about memory management
# This prevents OOM issues when generating hundreds of figures
matplotlib.rcParams['figure.max_open_warning'] = 5  # Warn earlier
plt.ioff()  # Turn off interactive mode to prevent figure retention

try:
    from .contrast import apply_contrast_enhancement, apply_comprehensive_slice_processing
    from .effects import apply_balanced_dark_field_effect, apply_blockface_preserving_dark_field_effect
    from .masking import create_fiber_mask
    from .utils import (
        densify_streamline,
        generate_tract_color_variation,
        get_colormap,
        select_random_streamlines
    )
except ImportError:
    from contrast import apply_contrast_enhancement, apply_comprehensive_slice_processing
    from effects import apply_balanced_dark_field_effect, apply_blockface_preserving_dark_field_effect
    from masking import create_fiber_mask
    from utils import (
        densify_streamline,
        generate_tract_color_variation,
        get_colormap,
        select_random_streamlines
    )


def visualize_nifti_with_trk(nifti_file, trk_file, output_file=None, n_slices=1, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=32, intensity_params=None,
                             tract_color_base=(1.0, 0.8, 0.1), tract_color_variation=0.2,
                             slice_idx=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None,
                             background_enhancement=None, cornucopia_augmentation=None,
                             truly_random=False):
    """
    Visualize multiple axial slices of a nifti file with tractography overlaid.

    Parameters
    ----------
    background_enhancement : str or dict, optional
        Background enhancement configuration to reduce pixelation
    cornucopia_augmentation : str or dict, optional
        Cornucopia augmentation configuration
    truly_random : bool, optional
        If True, uses truly random parameters for visualization effects
    """
    if truly_random:
        # Use current time for truly random parameters
        import time
        true_random_seed = int(time.time() * 1000000) % (2**32)
        random.seed(true_random_seed)
        np.random.seed(true_random_seed)
    elif random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if contrast_params is None:
        contrast_params = {
            'clip_limit': clahe_clip_limit,
            'tile_grid_size': (max(32, clahe_tile_grid_size), max(32, clahe_tile_grid_size))
        }

    # Load data
    nii_img = nib.load(nifti_file)
    nii_img = nib.as_closest_canonical(nii_img)
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

        # Use balanced dark field effect for good artifact removal with smooth transitions
        # Preserve bright blockface areas by not forcing background to black
        dark_field_slice = apply_balanced_dark_field_effect(
            slice_enhanced,
            slice_intensity_params,
            random_state=random_state,
            force_background_black=True
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
                    label_bundles=True, min_bundle_size=min_bundle_size,
                    static_streamline_threshold=0.1  # Require at least 0.1 streamline per pixel (any streamline)
                )
                mask = np.rot90(mask)
                labeled_mask = np.rot90(labeled_mask)
                # Apply vertical flip to match the image orientation (bottom to top)
                mask = np.flipud(mask)
                labeled_mask = np.flipud(labeled_mask)
                fiber_masks.append(mask)
                labeled_masks.append(labeled_mask)
            else:
                mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='axial',
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    min_bundle_size=min_bundle_size,
                    static_streamline_threshold=0.1  # Require at least 0.1 streamline per pixel (any streamline)
                )
                mask = np.rot90(mask)
                # Apply vertical flip to match the image orientation (bottom to top)
                mask = np.flipud(mask)
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

                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state, truly_random=truly_random)
                # Reduce fiber contrast to make them less obvious and blend better with dark background
                base_opacity = max(0.0, (1.0 - min_distance / 2.0) * 0.4)

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
        # Import resize utility
        from .utils import save_image_1024
        save_image_1024(output_file, fig, is_mask=False)
        print(f"Figure saved to {output_file} (1024x1024)")

        # Save masks if requested
        if save_masks and fiber_masks:
            _save_masks(output_file, fiber_masks, labeled_masks, slice_positions, label_bundles)
        
        # CRITICAL: Close figure immediately after saving to prevent memory accumulation
        plt.close(fig)
        
        # Return None for fig since it's been closed
        if label_bundles and labeled_masks:
            return None, None, fiber_masks, labeled_masks
        else:
            return None, None, fiber_masks if save_masks else None
    else:
        plt.show()
        # Don't close if showing interactively
        if label_bundles and labeled_masks:
            return fig, axes, fiber_masks, labeled_masks
        else:
            return fig, axes, fiber_masks if save_masks else None


def visualize_nifti_with_trk_coronal(nifti_file, trk_file, output_file=None, n_slices=1, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=32, intensity_params=None,
                             tract_color_base=(1.0, 0.8, 0.1), tract_color_variation=0.2,
                             slice_idx=None, streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None,
                             background_enhancement=None, cornucopia_augmentation=None,
                             truly_random=False, output_image_size=(1024, 1024),
                             use_high_density_masks=False, max_fiber_percentage=100.0,
                             min_fiber_percentage=10.0):
    """
    Visualize multiple coronal slices of a nifti file with tractography overlaid.

    Parameters
    ----------
    background_enhancement : str or dict, optional
        Background enhancement configuration to reduce pixelation
    cornucopia_augmentation : str or dict, optional
        Cornucopia augmentation configuration
    truly_random : bool, optional
        If True, uses truly random parameters for visualization effects
    """
    if truly_random:
        # Use current time for truly random parameters
        import time
        true_random_seed = int(time.time() * 1000000) % (2**32)
        random.seed(true_random_seed)
        np.random.seed(true_random_seed)
    elif random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if contrast_params is None:
        contrast_params = {
            'clip_limit': clahe_clip_limit,
            'tile_grid_size': (max(32, clahe_tile_grid_size), max(32, clahe_tile_grid_size))
        }

    # Load data
    nii_img = nib.load(nifti_file)
    nii_img = nib.as_closest_canonical(nii_img)
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
    fig, axes = plt.subplots(1, n_slices, figsize=(10 * n_slices, 10))  # Increased for better 1024x1024 output
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

        # Use balanced dark field effect for good artifact removal with smooth transitions
        # Preserve bright blockface areas by not forcing background to black
        dark_field_slice = apply_balanced_dark_field_effect(
            slice_enhanced,
            slice_intensity_params,
            random_state=random_state,
            force_background_black=True
        )

        axes[i].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='bilinear')
        axes[i].set_facecolor('black')

        # Create mask if requested - skip regular masks if high-density masks are enabled
        if save_masks and has_streamlines and not use_high_density_masks:
            # Adaptive parameters based on output image size
            output_size = max(output_image_size) if output_image_size else 256
            size_scale = output_size / 256.0  # Scale factor relative to base 256x256
            
            # Adaptive parameters for better mask quality at different sizes
            adaptive_thickness = max(1, int(mask_thickness * size_scale))
            adaptive_density_threshold = max(0.02, density_threshold * (1.0 / max(1.0, size_scale * 1.5)))  # More aggressive threshold reduction for larger images
            adaptive_gaussian_sigma = gaussian_sigma * size_scale
            adaptive_min_bundle_size = max(1, int(min_bundle_size * (size_scale * 0.001)))  # Tiny bundles allowed for larger images
            
            if label_bundles:
                mask, labeled_mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='coronal',
                    dims=dims, thickness=adaptive_thickness, dilate=True,
                    density_threshold=adaptive_density_threshold, gaussian_sigma=adaptive_gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    label_bundles=True, min_bundle_size=adaptive_min_bundle_size,
                    static_streamline_threshold=0.1  # Require at least 0.1 streamline per pixel (any streamline)
                )
                mask = np.rot90(mask)
                labeled_mask = np.rot90(labeled_mask)
                fiber_masks.append(mask)
                labeled_masks.append(labeled_mask)
            else:
                mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='coronal',
                    dims=dims, thickness=adaptive_thickness, dilate=True,
                    density_threshold=adaptive_density_threshold, gaussian_sigma=adaptive_gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    min_bundle_size=adaptive_min_bundle_size,
                    static_streamline_threshold=0.1  # Require at least 0.1 streamline per pixel (any streamline)
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

                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state, truly_random=truly_random)
                # Reduce fiber contrast to make them less obvious and blend better with dark background
                base_opacity = max(0.0, (1.0 - min_distance / 2.0) * 0.4)

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
        # Import resize utility
        from .utils import save_image_1024
        save_image_1024(output_file, fig, is_mask=False, target_size=output_image_size)
        print(f"Figure saved to {output_file} ({output_image_size[0]}x{output_image_size[1]})")

        # Save masks if requested
        if save_masks:
            if use_high_density_masks:
                # Only generate and save high-density masks (skip regular masks)
                _generate_and_apply_high_density_mask_coronal(
                    nifti_file, trk_file, output_file, slice_idx, 
                    max_fiber_percentage, tract_linewidth, mask_thickness,
                    density_threshold, gaussian_sigma, close_gaps, closing_footprint_size,
                    label_bundles, min_bundle_size, output_image_size
                )
            elif fiber_masks:
                # Save regular masks only when high-density masks are not enabled
                _save_masks(output_file, fiber_masks, labeled_masks, slice_positions, label_bundles, output_image_size)
        
        plt.close(fig)
        
        # Return None for fig since it's been closed
        if label_bundles and labeled_masks:
            return None, None, fiber_masks, labeled_masks
        else:
            return None, None, fiber_masks if save_masks else None
    else:
        plt.show()
        # Don't close if showing interactively
        if label_bundles and labeled_masks:
            return fig, axes, fiber_masks, labeled_masks
        else:
            return fig, axes, fiber_masks if save_masks else None


def visualize_multiple_views(nifti_file, trk_file, output_file=None, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=32, intensity_params=None,
                             tract_color_base=(1.0, 0.8, 0.1), tract_color_variation=0.2,
                             streamline_percentage=100.0, roi_sphere=None,
                             tract_linewidth=1.0, save_masks=False, mask_thickness=1,
                             density_threshold=0.15, gaussian_sigma=2.0, random_state=None,
                             close_gaps=False, closing_footprint_size=5, label_bundles=False,
                             min_bundle_size=20, contrast_method='clahe', contrast_params=None,
                             truly_random=False):
    """
    Visualize axial, coronal, and sagittal views of a nifti file with tractography overlaid.
    """
    if truly_random:
        # Use current time for truly random parameters
        import time
        true_random_seed = int(time.time() * 1000000) % (2**32)
        random.seed(true_random_seed)
        np.random.seed(true_random_seed)
    elif random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # Load data
    nii_img = nib.load(nifti_file)
    nii_img = nib.as_closest_canonical(nii_img)
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
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # Increased for better 1024x1024 output

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

        # Use balanced dark field effect for good artifact removal with smooth transitions
        # Preserve bright blockface areas by not forcing background to black
        dark_field_slice = apply_balanced_dark_field_effect(
            slice_enhanced,
            intensity_params,
            random_state=random_state,
            force_background_black=True
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
                min_bundle_size=min_bundle_size,
                static_streamline_threshold=15  # Require at least 15 streamlines per pixel
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
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation, random_state=random_state, truly_random=truly_random)
                # Reduce fiber contrast to make them less obvious and blend better with dark background
                base_opacity = max(0.0, (1.0 - min_distance / 2.0) * 0.4)
                
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
        # Import resize utility
        from .utils import save_image_1024
        save_image_1024(output_file, fig, is_mask=False)
        print(f"Figure saved to {output_file} (1024x1024)")
        
        # Save masks if requested
        if save_masks and has_streamlines:
            mask_dir = os.path.dirname(output_file)
            if not mask_dir:
                mask_dir = "../synthesis"
            mask_basename = os.path.splitext(os.path.basename(output_file))[0]
            
            for view, mask in fiber_masks.items():
                if mask is not None:
                    mask_filename = f"{mask_dir}/{mask_basename}_mask_{view}.png"
                    # Import resize utility
                    from .utils import save_image_1024
                    save_image_1024(mask_filename, mask, is_mask=True)
                    print(f"Saved {view} mask to {mask_filename} (1024x1024)")
        
        plt.close(fig)
        
        # Return None for fig since it's been closed
        return None, None, fiber_masks if save_masks else None
    else:
        plt.show()
        # Don't close if showing interactively
        return fig, axes, fiber_masks if save_masks else None


def _save_masks(output_file, fiber_masks, labeled_masks, slice_positions, label_bundles, output_image_size=(1024, 1024)):
    """Save masks to files."""
    mask_dir = os.path.dirname(output_file)
    if not mask_dir:
        mask_dir = "../synthesis"
    mask_basename = os.path.splitext(os.path.basename(output_file))[0]
    
    for i, mask in enumerate(fiber_masks):
        slice_id = slice_positions[i]
        mask_filename = f"{mask_dir}/{mask_basename}_mask_slice{slice_id}.png"
        # Import resize utility
        from .utils import save_image_1024
        save_image_1024(mask_filename, mask, is_mask=True, target_size=output_image_size)
        print(f"Saved mask for slice {slice_id} to {mask_filename} ({output_image_size[0]}x{output_image_size[1]})")
        
        if label_bundles and labeled_masks:
            from .utils import visualize_labeled_bundles
            labeled_mask = labeled_masks[i]
            labeled_filename = f"{mask_dir}/{mask_basename}_labeled_bundles_slice{slice_id}.png"
            visualize_labeled_bundles(labeled_mask, labeled_filename)
            print(f"Saved labeled bundles for slice {slice_id} to {labeled_filename}") 


def _generate_and_apply_high_density_mask_coronal(nifti_file, trk_file, output_file, slice_idx, 
                                                  max_fiber_percentage, tract_linewidth, mask_thickness,
                                                  density_threshold, gaussian_sigma, close_gaps, 
                                                  closing_footprint_size, label_bundles, min_bundle_size, 
                                                  output_image_size, static_streamline_threshold=25):
    """Generate and apply high-density mask for coronal view."""
    import nibabel as nib
    import numpy as np
    from dipy.io.streamline import load_tractogram
    
    # Load data
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    affine_inv = np.linalg.inv(nii_img.affine)
    
    # Load tractogram with bbox_valid_check disabled to handle coordinate issues
    tractogram = load_tractogram(trk_file, nii_img, bbox_valid_check=False)
    streamlines = tractogram.streamlines
    
    # Convert streamlines to voxel coordinates
    streamlines_voxel = []
    for sl in streamlines:
        if len(sl) > 0:
            # Apply inverse affine to convert from world to voxel coordinates
            sl_voxel = np.dot(sl, affine_inv[:3, :3].T) + affine_inv[:3, 3]
            streamlines_voxel.append(sl_voxel)
    
    # Use high fiber percentage for dense mask
    n_select = max(1, int(len(streamlines_voxel) * max_fiber_percentage / 100.0))
    selected_streamlines = streamlines_voxel[:n_select]
    
    # Adaptive parameters based on output image size
    output_size = max(output_image_size) if output_image_size else 256
    size_scale = output_size / 256.0  # Scale factor relative to base 256x256
    
    # HIGH-DENSITY specific parameters - extremely aggressive filtering for only largest bundles
    adaptive_thickness = max(10, int(mask_thickness * size_scale * 6))  # Much thicker lines for prominent bundles
    # Extremely aggressive density threshold - only largest, most prominent bundles
    high_density_threshold = max(0.5, 0.8 * (1.0 / max(1.0, size_scale * 0.3)))  # Very aggressive threshold for dense areas
    adaptive_gaussian_sigma = gaussian_sigma * size_scale * 1.3  # Moderate smoothing for connectivity
    # Extremely permissive minimum bundle size - keeps tiny bundles
    high_density_min_bundle_size = max(1, int(5 * size_scale))  # Extremely permissive bundle size filtering
    
    print(f"High-density mask parameters for {output_size}px: thickness={adaptive_thickness}, density_threshold={high_density_threshold:.3f}, gaussian_sigma={adaptive_gaussian_sigma:.1f}, min_bundle_size={high_density_min_bundle_size}")
    
    # Generate high-density mask with static absolute threshold
    from .masking import create_fiber_mask
    high_density_mask = create_fiber_mask(
        selected_streamlines, slice_idx, orientation='coronal', dims=dims,
        thickness=adaptive_thickness, dilate=True, density_threshold=high_density_threshold,
        gaussian_sigma=adaptive_gaussian_sigma, close_gaps=True,  # Enable gap closing for connectivity
        closing_footprint_size=max(3, int(closing_footprint_size * size_scale * 1.2)),  # Moderate footprint for bundle joining
        label_bundles=False, min_bundle_size=high_density_min_bundle_size,
        static_streamline_threshold=0.1  # Require at least 0.1 streamline per pixel for high-density masks
    )
    
    # Additional processing to limit to at most 4 largest bundles
    if np.any(high_density_mask):
        from skimage import measure
        labeled_mask = measure.label(high_density_mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        if len(regions) > 4:
            # Sort regions by area (largest first) and keep only top 4
            regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
            top_4_regions = regions_sorted[:4]
            
            # Create new mask with only top 4 largest bundles
            filtered_mask = np.zeros_like(high_density_mask)
            for region in top_4_regions:
                filtered_mask[labeled_mask == region.label] = 1
            high_density_mask = filtered_mask
            
            print(f"Limited high-density mask to top 4 bundles (from {len(regions)} total)")
        else:
            print(f"High-density mask contains {len(regions)} bundles (≤4, keeping all)")
    
    # Apply the same rotation as the regular masks
    high_density_mask = np.rot90(high_density_mask)
    # Apply vertical flip to match the image orientation (bottom to top)
    high_density_mask = np.flipud(high_density_mask)
    
    # Save high-density mask
    mask_dir = os.path.dirname(output_file)
    mask_basename = os.path.splitext(os.path.basename(output_file))[0]
    mask_filename = f"{mask_dir}/{mask_basename}_high_density_mask_slice{slice_idx}.png"
    
    from .utils import save_image_1024
    save_image_1024(mask_filename, high_density_mask, is_mask=True, target_size=output_image_size)
    print(f"Applied high-density mask for coronal slice {slice_idx}: {mask_filename} ({output_image_size[0]}x{output_image_size[1]})")
