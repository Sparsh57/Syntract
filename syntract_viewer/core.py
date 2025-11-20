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
            # Pass the non-rotated background - mask generation happens in the same coordinate space
            
            if label_bundles:
                mask, labeled_mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='axial',
                    dims=dims, thickness=mask_thickness, dilate=True,
                    density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    label_bundles=True, min_bundle_size=min_bundle_size,
                    static_streamline_threshold=0.1,  # Require at least 0.1 streamline per pixel (any streamline)
                    background_image=dark_field_slice
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
                    static_streamline_threshold=0.1,  # Require at least 0.1 streamline per pixel (any streamline)
                    background_image=dark_field_slice
                )
                mask = np.rot90(mask)
                # Apply vertical flip to match the image orientation (bottom to top)
                mask = np.flipud(mask)
                fiber_masks.append(mask)

        # Add streamlines
        if has_streamlines:
            segments = []
            colors = []
            
            # Tissue presence threshold - only render streamlines where there's actual white matter/tissue
            # White matter appears bright in MRI. Using 0.76 threshold for bright tissue regions
            # This ensures streamlines only appear in regions with dense tissue information
            tissue_threshold = 0.76
            background_rotated = np.rot90(dark_field_slice)
            # Use ORIGINAL tissue data for anatomical checking, not augmented display
            tissue_reference = np.rot90(slice_data)
            
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
                    # DUAL THRESHOLD: Check both original tissue AND augmented display
                    # Get midpoint of segment for checking
                    midpoint = seg.mean(axis=0)
                    x_mid, y_mid = int(round(midpoint[0])), int(round(midpoint[1]))
                    
                    # Check bounds
                    if 0 <= y_mid < tissue_reference.shape[0] and 0 <= x_mid < tissue_reference.shape[1]:
                        # THRESHOLD 1: Check ORIGINAL tissue intensity (anatomical validity)
                        tissue_value = tissue_reference[y_mid, x_mid]
                        tissue_max = np.max(tissue_reference)
                        if tissue_max > 0:
                            tissue_value_normalized = tissue_value / tissue_max
                        else:
                            tissue_value_normalized = 0.0
                        
                        # Skip if insufficient original tissue
                        if tissue_value_normalized < tissue_threshold:
                            continue
                        
                        # THRESHOLD 2: Check AUGMENTED display brightness (visual validity)
                        augmented_value = background_rotated[y_mid, x_mid]
                        augmented_max = np.max(background_rotated)
                        if augmented_max > 0:
                            augmented_normalized = augmented_value / augmented_max
                        else:
                            augmented_normalized = 0.0
                        
                        # Skip if augmented area is too dark (prevents black area rendering)
                        # Threshold 0.65 = only render over bright areas (aggressively prevents dark rendering)
                        if augmented_normalized < 0.65:
                            continue
                    else:
                        # Out of bounds, skip
                        continue
                    
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
                             min_fiber_percentage=10.0, white_matter_only=False):
    """
    Visualize multiple coronal slices of a nifti file with tractography overlaid.

    Parameters
    ----------
    white_matter_only : bool, optional
        If True, filter streamlines by tissue (0.76) and display (0.65) thresholds.
        If False (default), render all streamlines regardless of background.
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
            
            # Pass the non-rotated background - mask generation happens in the same coordinate space
            
            if label_bundles:
                mask, labeled_mask = create_fiber_mask(
                    streamlines_voxel, slice_idx, orientation='coronal',
                    dims=dims, thickness=adaptive_thickness, dilate=True,
                    density_threshold=adaptive_density_threshold, gaussian_sigma=adaptive_gaussian_sigma,
                    close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                    label_bundles=True, min_bundle_size=adaptive_min_bundle_size,
                    static_streamline_threshold=0.1,  # Require at least 0.1 streamline per pixel (any streamline)
                    background_image=dark_field_slice
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
                    static_streamline_threshold=0.1,  # Require at least 0.1 streamline per pixel (any streamline)
                    background_image=dark_field_slice
                )
                mask = np.rot90(mask)
                fiber_masks.append(mask)

        # Add streamlines
        if has_streamlines:
            segments = []
            colors = []
            # Prepare rotated background for augmented display checking
            background_rotated = np.rot90(dark_field_slice)
            # Use ORIGINAL tissue data for anatomical checking
            tissue_reference = np.rot90(slice_data)
            tissue_threshold = 0.76
            
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
                    # Only apply filtering if white_matter_only is enabled
                    if white_matter_only:
                        # DUAL THRESHOLD: Check both original tissue AND augmented display
                        midpoint = seg.mean(axis=0)
                        x_mid, y_mid = int(round(midpoint[0])), int(round(midpoint[1]))
                        
                        # Check bounds
                        if 0 <= y_mid < tissue_reference.shape[0] and 0 <= x_mid < tissue_reference.shape[1]:
                            # THRESHOLD 1: Original tissue (anatomical validity)
                            tissue_value = tissue_reference[y_mid, x_mid]
                            tissue_max = np.max(tissue_reference)
                            tissue_value_normalized = tissue_value / tissue_max if tissue_max > 0 else 0.0
                            
                            if tissue_value_normalized < tissue_threshold:
                                continue
                            
                            # THRESHOLD 2: Augmented display (visual validity)
                            augmented_value = background_rotated[y_mid, x_mid]
                            augmented_max = np.max(background_rotated)
                            augmented_normalized = augmented_value / augmented_max if augmented_max > 0 else 0.0
                            
                            if augmented_normalized < 0.65:  # Skip black/dark areas (aggressive threshold)
                                continue
                        else:
                            continue
                    
                    # Render segment (either passed filtering or filtering disabled)
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
                # Pass the augmented dark_field_slice to ensure mask uses same data as visualization
                _generate_and_apply_high_density_mask_coronal(
                    nifti_file, trk_file, output_file, slice_idx, 
                    max_fiber_percentage, tract_linewidth, mask_thickness,
                    density_threshold, gaussian_sigma, close_gaps, closing_footprint_size,
                    label_bundles, min_bundle_size, output_image_size,
                    augmented_display_slice=dark_field_slice, white_matter_only=white_matter_only
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
            # Pass the non-rotated background - mask generation happens in the same coordinate space
            
            mask = create_fiber_mask(
                streamlines_voxel, slice_idx, orientation=view, 
                dims=dims, thickness=mask_thickness, dilate=True,
                density_threshold=density_threshold, gaussian_sigma=gaussian_sigma,
                close_gaps=close_gaps, closing_footprint_size=closing_footprint_size,
                min_bundle_size=min_bundle_size,
                static_streamline_threshold=15,  # Require at least 15 streamlines per pixel
                background_image=dark_field_slice
            )
            mask = np.rot90(mask)
            fiber_masks[view] = mask

        # Add streamlines
        if has_streamlines:
            segments = []
            colors = []
            # Prepare rotated background for augmented display checking
            background_rotated = np.rot90(dark_field_slice)
            # Use ORIGINAL tissue data for anatomical checking
            tissue_reference = np.rot90(slice_data)
            tissue_threshold = 0.76
            
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
                    # DUAL THRESHOLD: Check both original tissue AND augmented display
                    midpoint = seg.mean(axis=0)
                    x_mid, y_mid = int(round(midpoint[0])), int(round(midpoint[1]))
                    
                    # Check bounds
                    if 0 <= y_mid < tissue_reference.shape[0] and 0 <= x_mid < tissue_reference.shape[1]:
                        # THRESHOLD 1: Original tissue (anatomical validity)
                        tissue_value = tissue_reference[y_mid, x_mid]
                        tissue_max = np.max(tissue_reference)
                        tissue_value_normalized = tissue_value / tissue_max if tissue_max > 0 else 0.0
                        
                        if tissue_value_normalized < tissue_threshold:
                            continue
                        
                        # THRESHOLD 2: Augmented display (visual validity)
                        augmented_value = background_rotated[y_mid, x_mid]
                        augmented_max = np.max(background_rotated)
                        augmented_normalized = augmented_value / augmented_max if augmented_max > 0 else 0.0
                        
                        if augmented_normalized < 0.65:  # Skip black/dark areas (aggressive threshold)
                            continue
                    else:
                        continue
                    
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
                                                  output_image_size, static_streamline_threshold=25,
                                                  augmented_display_slice=None, white_matter_only=False):
    """Generate and apply high-density mask for coronal view.
    
    Parameters
    ----------
    augmented_display_slice : ndarray, optional
        Pre-computed augmented display slice from visualization (BEFORE rotation).
        If provided, this will be used instead of recomputing augmentation.
    white_matter_only : bool, optional
        If True, filter streamlines by tissue/display thresholds before mask generation.
        If False (default), generate mask from all streamlines.
    """
    import nibabel as nib
    import numpy as np
    from dipy.io.streamline import load_tractogram
    
    # Load data
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    affine_inv = np.linalg.inv(nii_img.affine)
    
    # CRITICAL FIX: Use output_image_size as the target dimensions for mask generation
    # This ensures masks are generated at the correct size regardless of patch NIfTI dimensions
    # For coronal orientation, we need (X, Z) dimensions from output_image_size
    if output_image_size:
        # output_image_size is (height, width) = (X, Z) for coronal view
        target_mask_dims = (output_image_size[0], dims[1], output_image_size[1])
        print(f"Using target mask dimensions from output_image_size: {target_mask_dims} (instead of patch NIfTI dims: {dims})")
    else:
        target_mask_dims = dims
        print(f"Using patch NIfTI dimensions for mask: {dims}")
    
    # Load tractogram using simple load (no validation) to handle registered TRK files
    from nibabel.streamlines import load as load_trk
    tractogram = load_trk(trk_file)
    streamlines = tractogram.streamlines
    
    # Convert streamlines to voxel coordinates
    streamlines_voxel = []
    for sl in streamlines:
        if len(sl) > 0:
            # Apply inverse affine to convert from world to voxel coordinates
            sl_voxel = np.dot(sl, affine_inv[:3, :3].T) + affine_inv[:3, 3]
            streamlines_voxel.append(sl_voxel)
    
    # Use high fiber percentage for dense mask (select BEFORE filtering and scaling)
    n_select = max(1, int(len(streamlines_voxel) * max_fiber_percentage / 100.0))
    selected_streamlines = streamlines_voxel[:n_select]
    
    # Filter streamlines by tissue and display thresholds BEFORE scaling
    # Create a validity mask based on tissue and display thresholds
    # This will be used to filter the final mask, not the streamlines themselves
    nii_data = nii_img.get_fdata()
    slice_data = nii_data[:, slice_idx, :]
    
    # Normalize slice_data to [0, 1] for consistent threshold comparison
    if np.max(slice_data) > 1.0:
        slice_data_normalized = slice_data / np.max(slice_data) if np.max(slice_data) > 0 else slice_data
    else:
        slice_data_normalized = slice_data
    
    # Use pre-computed augmented display if provided, otherwise generate it
    if augmented_display_slice is not None:
        # Use the exact same augmented display as the visualization
        display_slice = augmented_display_slice
        print("   High-density mask: Using pre-computed augmented display from visualization")
    else:
        # Fallback: Generate augmentation (may not match visualization)
        import time
        augmentation_seed = int(time.time() * 1000000 + slice_idx) % (2**32)
        
        from .contrast import apply_comprehensive_slice_processing
        slice_enhanced = apply_comprehensive_slice_processing(
            slice_data,
            background_preset='preserve_edges',
            cornucopia_preset='clean_optical',
            contrast_method='clahe',
            background_params=None,
            cornucopia_params=None,
            contrast_params={'clip_limit': 0.01, 'tile_grid_size': (32, 32)},
            random_state=augmentation_seed
        )
        
        from .effects import apply_balanced_dark_field_effect
        intensity_params = {
            'gamma': 1.0,
            'brightness': 0.0,
            'contrast': 1.0,
            'threshold': 0.02,
            'contrast_stretch': (0.5, 99.5),
            'background_boost': 1.0,
            'color_scheme': 'bw',
            'blue_tint': 0.0
        }
        display_slice = apply_balanced_dark_field_effect(
            slice_enhanced,
            intensity_params,
            random_state=augmentation_seed,
            force_background_black=True
        )
    
    # CRITICAL: Use SAME coordinate system as visualization rendering
    # Visualization uses ROTATED data: tissue_reference = np.rot90(slice_data)
    tissue_reference = np.rot90(slice_data_normalized)
    background_rotated = np.rot90(display_slice)
    
    # Only apply tissue/display filtering if white_matter_only is enabled
    if white_matter_only:
        # Use EXACT same thresholds as visualization (lines 212, 267)
        tissue_threshold = 0.76  # Match visualization line 212
        display_threshold = 0.65  # Match visualization line 267
        
        # Normalize by max (per-slice, matching visualization lines 247, 257)
        tissue_max = np.max(tissue_reference)
        if tissue_max > 0:
            tissue_normalized_for_check = tissue_reference / tissue_max
        else:
            tissue_normalized_for_check = tissue_reference
        
        display_max = np.max(background_rotated)
        if display_max > 0:
            display_normalized_for_check = background_rotated / display_max
        else:
            display_normalized_for_check = background_rotated
        
        print(f"   High-density mask: White matter filtering ENABLED: tissue={tissue_threshold}, display={display_threshold}")
        print(f"   High-density mask: Tissue range [{np.min(tissue_normalized_for_check):.3f}, {np.max(tissue_normalized_for_check):.3f}], Display range [{np.min(display_normalized_for_check):.3f}, {np.max(display_normalized_for_check):.3f}]")
    else:
        print("   High-density mask: White matter filtering DISABLED - using all streamlines")
    
    # Point-by-point filtering based on white_matter_only setting
    if white_matter_only:
        # CORRECT APPROACH: Filter streamlines POINT-BY-POINT using ROTATED coordinates
        # Match visualization's coordinate system exactly
        filtered_streamlines = []
        total_points_before = 0
        total_points_after = 0
        points_failed_tissue = 0
        points_failed_display = 0
        points_out_of_bounds = 0
        
        for streamline in selected_streamlines:
            total_points_before += len(streamline)
            
            # Check each point against validity thresholds
            valid_points = []
            for point in streamline:
                x_coord = int(np.round(point[0]))
                y_coord = int(np.round(point[1]))
                z_coord = int(np.round(point[2]))
                
                # CORONAL ORIENTATION: Check if point is near this slice (slice along Y axis)
                # For coronal: slice_idx is the Y position, check Y coordinate
                distance_to_slice = abs(y_coord - slice_idx)
                if distance_to_slice > 2.0:
                    # Skip points far from slice (won't affect this slice's visualization)
                    continue
                
                # Convert to plot coordinates for CORONAL view (match visualization line 503)
                # After np.rot90, (X, Z) becomes (Z, X), so we plot (x, z_plot)
                z_plot = dims[2] - z_coord - 1
                
                # Check bounds in ROTATED coordinate system (row=z_plot, col=x_coord)
                if 0 <= z_plot < tissue_normalized_for_check.shape[0] and 0 <= x_coord < tissue_normalized_for_check.shape[1]:
                    # Check tissue threshold (match visualization line 509)
                    tissue_value = tissue_normalized_for_check[z_plot, x_coord]
                    if tissue_value < tissue_threshold:
                        # Skip this point - insufficient tissue
                        points_failed_tissue += 1
                        continue
                    
                    # Check display threshold (match visualization line 517)
                    display_value = display_normalized_for_check[z_plot, x_coord]
                    if display_value < display_threshold:
                        # Skip this point - too dark
                        points_failed_display += 1
                        continue
                    
                    # Point passes both thresholds
                    valid_points.append(point)
                else:
                    # Out of bounds, skip this point
                    points_out_of_bounds += 1
            
            # Only keep streamlines that have at least some valid points
            if len(valid_points) >= 2:  # Need at least 2 points to draw a line
                filtered_streamlines.append(np.array(valid_points))
                total_points_after += len(valid_points)
    
        print(f"   High-density mask: Filtered streamlines point-by-point: {len(selected_streamlines)} streamlines → {len(filtered_streamlines)} streamlines")
        print(f"   High-density mask: Total points: {total_points_before} → {total_points_after} ({100*total_points_after/max(1,total_points_before):.1f}% retained)")
        print(f"   High-density mask: Points rejected - tissue: {points_failed_tissue}, display: {points_failed_display}, out_of_bounds: {points_out_of_bounds}")
        
        selected_streamlines = filtered_streamlines
    else:
        # No filtering - use all streamlines
        print(f"   High-density mask: Using all {len(selected_streamlines)} streamlines without filtering")
    
    # Scale streamline coordinates to target mask dimensions if dimensions differ
    if target_mask_dims != dims:
        scale_factors = np.array([
            target_mask_dims[0] / dims[0],
            target_mask_dims[1] / dims[1],
            target_mask_dims[2] / dims[2]
        ])
        print(f"Scaling streamlines by factors: {scale_factors}")
        
        scaled_streamlines = []
        for sl in selected_streamlines:
            scaled_sl = sl * scale_factors
            scaled_streamlines.append(scaled_sl)
        selected_streamlines = scaled_streamlines
    output_size = max(output_image_size) if output_image_size else 256
    size_scale = output_size / 256.0  # Scale factor relative to base 256x256
    
    # BALANCED mask parameters - capture filtered streamlines with moderate sensitivity
    adaptive_thickness = max(6, int(mask_thickness * size_scale * 3))  # Moderate line thickness
    # Low but not minimal density threshold - capture small bundles but filter noise
    high_density_threshold = 0.15  # Moderate threshold to accept small streamlines while filtering noise
    adaptive_gaussian_sigma = gaussian_sigma * size_scale * 1.0  # Standard smoothing
    # Permissive minimum bundle size - accept small bundles
    high_density_min_bundle_size = max(1, int(2 * size_scale))  # Small bundles accepted
    
    print(f"High-density mask parameters for {output_size}px: thickness={adaptive_thickness}, density_threshold={high_density_threshold:.3f}, gaussian_sigma={adaptive_gaussian_sigma:.1f}, min_bundle_size={high_density_min_bundle_size}")
    
    # Generate high-density mask with static absolute threshold
    # CRITICAL: Use target_mask_dims instead of original dims
    from .masking import create_fiber_mask
    high_density_mask = create_fiber_mask(
        selected_streamlines, slice_idx, orientation='coronal', dims=target_mask_dims,
        thickness=adaptive_thickness, dilate=True, density_threshold=high_density_threshold,
        gaussian_sigma=adaptive_gaussian_sigma, close_gaps=True,  # Enable gap closing for connectivity
        closing_footprint_size=max(5, int(closing_footprint_size * size_scale * 1.5)),  # Larger footprint for better connectivity
        label_bundles=False, min_bundle_size=high_density_min_bundle_size,
        static_streamline_threshold=0.05  # Low threshold - accept sparse streamlines
    )
    
    # No restrictive bundle filtering - keep all filtered streamlines regardless of size/count
    # This ensures even sparse streamlines are captured in the mask
    if np.any(high_density_mask):
        from skimage import measure
        labeled_mask = measure.label(high_density_mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        print(f"High-density mask contains {len(regions)} bundle regions, keeping all")
    
    # No need for post-mask filtering since we filtered streamlines point-by-point
    # The mask is already generated only from valid streamline points
    print(f"   High-density mask: Generated from filtered streamlines, {np.sum(high_density_mask)} pixels")
    
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
