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

from scipy.spatial import cKDTree


def filter_streamlines_by_density(streamlines, slice_idx, min_neighbors=3, search_radius=5.0):
    """
    Filter streamlines based on local density - remove isolated streamlines.
    
    Parameters:
    -----------
    streamlines : list of arrays
        List of streamlines in voxel coordinates
    slice_idx : int
        The coronal slice index to focus on
    min_neighbors : int
        Minimum number of nearby streamlines required (default: 3)
    search_radius : float
        Search radius in voxels for finding neighbors (default: 5.0)
    
    Returns:
    --------
    list of arrays
        Filtered streamlines that have sufficient neighbors
    """
    if len(streamlines) == 0:
        return streamlines
    
    # Extract representative points for each streamline on the slice
    representative_points = []
    valid_indices = []
    
    for idx, streamline in enumerate(streamlines):
        # Get points near the slice
        slice_points = streamline[np.abs(streamline[:, 1] - slice_idx) < 0.5]
        if len(slice_points) > 0:
            # Use median point as representative
            rep_point = np.median(slice_points, axis=0)
            representative_points.append(rep_point)
            valid_indices.append(idx)
    
    if len(representative_points) == 0:
        return streamlines
    
    representative_points = np.array(representative_points)
    
    # Build KD-tree for efficient neighbor search
    tree = cKDTree(representative_points)
    
    # Count neighbors for each streamline (including itself)
    neighbor_counts = tree.query_ball_point(representative_points, search_radius, return_length=True)
    
    # Filter streamlines that have enough neighbors
    filtered_streamlines = []
    removed_count = 0
    
    for i, idx in enumerate(valid_indices):
        # Subtract 1 because query includes the point itself
        num_neighbors = neighbor_counts[i] - 1
        if num_neighbors >= min_neighbors:
            filtered_streamlines.append(streamlines[idx])
        else:
            removed_count += 1
    
    print(f"  Density filtering: removed {removed_count} isolated streamlines (kept {len(filtered_streamlines)}/{len(streamlines)})")
    
    return filtered_streamlines


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
        # CRITICAL FIX: Use TRK's affine for pre-registered files
        try:
            trk_affine = tractogram.affine
            affine_diff = np.abs(trk_affine - nii_img.affine).max()
            if affine_diff > 0.1:
                print(f"  Pre-registered TRK detected, using TRK affine")
                affine_inv = np.linalg.inv(trk_affine)
            else:
                affine_inv = np.linalg.inv(nii_img.affine)
        except:
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

        # Use blockface-preserving dark field (no inversion) to keep original intensities
        # This preserves the normal NIfTI appearance while darkening the background
        dark_field_slice = apply_blockface_preserving_dark_field_effect(
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
                             min_fiber_percentage=10.0, white_mask_file=None):
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
    
    # Load white mask if provided
    white_mask_data = None
    if white_mask_file and os.path.exists(white_mask_file):
        try:
            white_mask_img = nib.load(white_mask_file)
            white_mask_img = nib.as_closest_canonical(white_mask_img)
            white_mask_data = white_mask_img.get_fdata()
            # Convert to binary mask
            white_mask_data = (white_mask_data > 0.5).astype(np.uint8)
            print(f"Loaded white mask from {white_mask_file}")
            print(f"  White mask shape: {white_mask_data.shape}")
            print(f"  NIfTI shape: {dims}")
        except Exception as e:
            print(f"Warning: Could not load white mask: {e}")
            white_mask_data = None

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
        # CRITICAL FIX: Use TRK's affine for pre-registered files
        try:
            trk_affine = tractogram.affine
            affine_diff = np.abs(trk_affine - nii_img.affine).max()
            if affine_diff > 0.1:
                print(f"  Pre-registered TRK detected, using TRK affine")
                affine_inv = np.linalg.inv(trk_affine)
            else:
                affine_inv = np.linalg.inv(nii_img.affine)
        except:
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

        # Create streamline mask BEFORE dark field effect if streamlines present
        streamline_mask_for_effect = None
        if has_streamlines:
            streamline_mask_for_effect = create_fiber_mask(
                streamlines_voxel, slice_idx, orientation='coronal',
                dims=dims, thickness=10, dilate=False,  # Increased from mask_thickness to 10 to capture nearby streamlines
                density_threshold=0, gaussian_sigma=0,
                close_gaps=False, label_bundles=False,
                static_streamline_threshold=0.01  # Lower threshold to catch more streamlines
            )

        # Use blockface-preserving dark field (no inversion) to keep original intensities
        # This preserves the normal NIfTI appearance while darkening the background
        dark_field_slice = apply_blockface_preserving_dark_field_effect(
            slice_enhanced,
            slice_intensity_params,
            random_state=random_state,
            force_background_black=True
        )

        axes[i].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='bilinear')
        axes[i].set_facecolor('black')

        # Extract white mask slice if available (needed for both filtering and drawing)
        white_mask_slice = None
        if white_mask_data is not None:
            try:
                # Handle thin slices (middle dimension is 1)
                if white_mask_data.shape[1] == 1:
                    # For thin slices, squeeze the middle dimension
                    white_mask_slice = white_mask_data[:, 0, :]
                else:
                    # Normal case: extract slice at slice_idx
                    white_mask_slice = white_mask_data[:, slice_idx, :]
                
                # Validate shape
                if white_mask_slice.ndim == 1:
                    print(f"  Warning: White mask slice is 1D (shape: {white_mask_slice.shape}), skipping mask filtering")
                    white_mask_slice = None
                else:
                    print(f"  Using white mask for slice {slice_idx}, shape: {white_mask_slice.shape}")
            except Exception as e:
                print(f"  Warning: Could not extract white mask slice: {e}")
                white_mask_slice = None

        # Filter streamlines by white mask before creating masks
        streamlines_for_mask = streamlines_voxel
        if white_mask_slice is not None and has_streamlines:
            filtered_streamlines = []
            for streamline in streamlines_voxel:
                # Check if streamline passes through white mask
                # Find points in this coronal slice
                slice_points = streamline[np.abs(streamline[:, 1] - slice_idx) < 0.5]
                if len(slice_points) == 0:
                    continue
                
                # Check if any point is in the white mask
                # NOTE: Use RAW coordinates - no transformation needed!
                # The z_plot transformation (dims[2] - z - 1) is ONLY for visualization plotting,
                # NOT for mask coordinate lookup
                valid = False
                for point in slice_points:
                    x_idx = int(np.clip(point[0], 0, white_mask_slice.shape[0] - 1))
                    z_idx = int(np.clip(point[2], 0, white_mask_slice.shape[1] - 1))
                    
                    if white_mask_slice[x_idx, z_idx] > 0:
                        valid = True
                        break
                
                if valid:
                    filtered_streamlines.append(streamline)
            
            streamlines_for_mask = filtered_streamlines

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
                    streamlines_for_mask, slice_idx, orientation='coronal',
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
                    streamlines_for_mask, slice_idx, orientation='coronal',
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
            
            # Use streamlines_for_mask (filtered) instead of streamlines_voxel (unfiltered)
            streamlines_to_draw = streamlines_for_mask if white_mask_slice is not None else streamlines_voxel
            
            for sl in streamlines_to_draw:
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

                for seg_idx, seg in enumerate(segs):
                    # SEGMENT-LEVEL white mask filtering
                    # Even though streamlines are filtered, we need to ensure segments don't extend into non-WM areas
                    if white_mask_slice is not None:
                        # Check both endpoints of the segment using ORIGINAL z coordinates (not z_plot)
                        # seg shape is (2, 2) where seg[0] is start point [x, z_plot], seg[1] is end point [x, z_plot]
                        start_x = int(np.clip(seg[0, 0], 0, white_mask_slice.shape[0] - 1))
                        # Use original z array (not z_plot) for mask lookup
                        start_z = int(np.clip(z[seg_idx], 0, white_mask_slice.shape[1] - 1))
                        
                        end_x = int(np.clip(seg[1, 0], 0, white_mask_slice.shape[0] - 1))
                        end_z = int(np.clip(z[seg_idx + 1], 0, white_mask_slice.shape[1] - 1))
                        
                        # Check if BOTH endpoints are in white matter
                        start_in_wm = white_mask_slice[start_x, start_z] > 0
                        end_in_wm = white_mask_slice[end_x, end_z] > 0
                        
                        # Only draw segment if at least one endpoint is in white matter
                        # This allows segments that partially cross WM boundaries
                        if not (start_in_wm or end_in_wm):
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
                    label_bundles, min_bundle_size, output_image_size, white_mask_file=white_mask_file
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
        # CRITICAL FIX: Use TRK's affine for pre-registered files
        try:
            trk_affine = tractogram.affine
            affine_diff = np.abs(trk_affine - nii_img.affine).max()
            if affine_diff > 0.1:
                print(f"  Pre-registered TRK detected, using TRK affine")
                affine_inv = np.linalg.inv(trk_affine)
            else:
                affine_inv = np.linalg.inv(nii_img.affine)
        except:
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

        # Create streamline mask BEFORE dark field effect if streamlines present
        streamline_mask_for_effect = None
        if has_streamlines:
            streamline_mask_for_effect = create_fiber_mask(
                streamlines_voxel, slice_idx, orientation=view,
                dims=dims, thickness=10, dilate=False,  # Increased from mask_thickness to 10 to capture nearby streamlines
                density_threshold=0, gaussian_sigma=0,
                close_gaps=False, label_bundles=False,
                static_streamline_threshold=0.01  # Lower threshold to catch more streamlines
            )

        # Use blockface-preserving dark field (no inversion) to keep original intensities
        # This preserves the normal NIfTI appearance while darkening the background
        dark_field_slice = apply_blockface_preserving_dark_field_effect(
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
                                                  output_image_size, static_streamline_threshold=0.05, white_mask_file=None):
    """Generate and apply high-density mask for coronal view."""
    import nibabel as nib
    import numpy as np
    from dipy.io.streamline import load_tractogram
    import os
    
    # Load white mask if provided
    white_mask_data = None
    if white_mask_file and os.path.exists(white_mask_file):
        try:
            white_mask_img = nib.load(white_mask_file)
            white_mask_img = nib.as_closest_canonical(white_mask_img)
            white_mask_data = white_mask_img.get_fdata()
            white_mask_data = (white_mask_data > 0.5).astype(np.uint8)
            print(f"Loaded white mask for filtering: {white_mask_file}")
        except Exception as e:
            print(f"Warning: Could not load white mask for filtering: {e}")
            white_mask_data = None
    
    # Load data
    nii_img = nib.load(nifti_file)
    dims = nii_img.shape
    
    # CRITICAL FIX: Use TRK's affine for pre-registered files  
    try:
        tractogram = load(trk_file)
        trk_affine = tractogram.affine
        affine_diff = np.abs(trk_affine - nii_img.affine).max()
        if affine_diff > 0.1:
            print(f"  Pre-registered TRK detected, using TRK affine for mask generation")
            affine_inv = np.linalg.inv(trk_affine)
        else:
            affine_inv = np.linalg.inv(nii_img.affine)
    except:
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
    
    # Store white mask info for filtering streamlines AND post-mask filtering
    white_mask_slice = None
    if white_mask_data is not None:
        print(f"Loading white mask for streamline filtering...")
        try:
            if white_mask_data.shape[1] == 1:
                white_mask_slice = white_mask_data[:, 0, :]
            else:
                white_mask_slice = white_mask_data[:, slice_idx, :]
            print(f"  White mask slice shape: {white_mask_slice.shape}, slice_idx: {slice_idx}")
        except Exception as e:
            print(f"  Warning: Could not extract white mask slice: {e}")
    
    # Filter streamlines by white mask BEFORE mask generation (like visualization does)
    if white_mask_slice is not None and len(streamlines_voxel) > 0:
        print(f"Filtering {len(streamlines_voxel)} streamlines by white mask...")
        filtered_streamlines = []
        for streamline in streamlines_voxel:
            # Check if streamline passes through white mask
            slice_points = streamline[np.abs(streamline[:, 1] - slice_idx) < 0.5]
            if len(slice_points) == 0:
                continue
            
            # Check if any point is in the white mask
            valid = False
            for point in slice_points:
                x_idx = int(np.clip(point[0], 0, white_mask_slice.shape[0] - 1))
                z_idx = int(np.clip(point[2], 0, white_mask_slice.shape[1] - 1))
                if white_mask_slice[x_idx, z_idx] > 0:
                    valid = True
                    break
            
            if valid:
                filtered_streamlines.append(streamline)
        
        print(f"  Filtered to {len(filtered_streamlines)} streamlines (from {len(streamlines_voxel)})")
        streamlines_voxel = filtered_streamlines
    
    # Apply density-based filtering to remove isolated streamlines (after white matter filtering)
    if len(streamlines_voxel) > 0:
        streamlines_voxel = filter_streamlines_by_density(
            streamlines_voxel, 
            slice_idx=slice_idx,
            min_neighbors=1,  # Require at least 1 nearby streamline
            search_radius=12.0  # Search within 12 voxels
        )
    
    # Scale streamline coordinates to target mask dimensions if dimensions differ
    if target_mask_dims != dims:
        scale_factors = np.array([
            target_mask_dims[0] / dims[0],
            target_mask_dims[1] / dims[1],
            target_mask_dims[2] / dims[2]
        ])
        print(f"Scaling streamlines by factors: {scale_factors}")
        
        scaled_streamlines = []
        for sl in streamlines_voxel:
            scaled_sl = sl * scale_factors
            scaled_streamlines.append(scaled_sl)
        streamlines_voxel = scaled_streamlines
    
    # Use high fiber percentage for dense mask
    n_select = max(1, int(len(streamlines_voxel) * max_fiber_percentage / 100.0))
    selected_streamlines = streamlines_voxel[:n_select]
    
    print(f"  Mask generation: using {len(selected_streamlines)} streamlines (from {len(streamlines_voxel)} after filtering)")
    
    # Adaptive parameters based on output image size
    output_size = max(output_image_size) if output_image_size else 256
    size_scale = output_size / 256.0  # Scale factor relative to base 256x256
    
    # HIGH-DENSITY specific parameters - wider masks that merge nearby bundles
    adaptive_thickness = max(2, int(mask_thickness * size_scale))  # Wider for better visibility
    # Lower density threshold to capture sparser white matter tracts
    high_density_threshold = max(0.01, density_threshold * (1.0 / max(1.0, size_scale)))  # Permissive threshold
    adaptive_gaussian_sigma = max(1.0, gaussian_sigma * size_scale * 0.5)  # More smoothing for connectivity
    # Permissive minimum bundle size - keeps smaller bundles
    high_density_min_bundle_size = max(1, int(min_bundle_size * size_scale * 0.01))  # Very permissive bundle size filtering
    
    print(f"High-density mask parameters for {output_size}px: thickness={adaptive_thickness}, density_threshold={high_density_threshold:.3f}, gaussian_sigma={adaptive_gaussian_sigma:.1f}, min_bundle_size={high_density_min_bundle_size}")
    
    # Generate high-density mask with static absolute threshold
    # CRITICAL: Use target_mask_dims instead of original dims
    from .masking import create_fiber_mask
    high_density_mask = create_fiber_mask(
        selected_streamlines, slice_idx, orientation='coronal', dims=target_mask_dims,
        thickness=adaptive_thickness, dilate=True, density_threshold=high_density_threshold,
        gaussian_sigma=adaptive_gaussian_sigma, close_gaps=close_gaps,  # Use parameter from caller
        closing_footprint_size=closing_footprint_size,  # Use parameter from caller
        label_bundles=False, min_bundle_size=high_density_min_bundle_size,
        static_streamline_threshold=static_streamline_threshold,  # Use parameter from caller
        white_mask_slice=white_mask_slice  # Pass white mask for segment-level filtering
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
    
    # SKIP post-mask white matter filtering - we already filtered streamlines
    # This avoids double-filtering that makes masks too strict
    
    # Smoothen the mask VERY LIGHTLY to avoid merging bundles
    if np.any(high_density_mask):
        from scipy.ndimage import gaussian_filter
        from skimage import morphology, measure
        
        print(f"  Light smoothing to preserve separation...")
        
        # Label bundles first to smooth each independently
        labeled_mask = measure.label(high_density_mask, connectivity=2)
        num_bundles = labeled_mask.max()
        
        if num_bundles > 0:
            smoothed_mask = np.zeros_like(high_density_mask)
            
            for bundle_id in range(1, num_bundles + 1):
                bundle = (labeled_mask == bundle_id).astype(float)
                
                # Very light Gaussian smoothing per bundle
                bundle_smoothed = gaussian_filter(bundle, sigma=0.8)
                
                # Threshold back to binary
                bundle_binary = (bundle_smoothed > 0.3).astype(np.uint8)
                
                # Add to result
                smoothed_mask = np.maximum(smoothed_mask, bundle_binary)
            
            high_density_mask = smoothed_mask
        
        print(f"  Smoothing complete")
    
    # Apply the same rotation as the regular masks
    high_density_mask = np.rot90(high_density_mask)
    # Apply vertical flip to match the image orientation (bottom to top)
    high_density_mask = np.flipud(high_density_mask)
    
    # DEBUG: Check mask content
    mask_nonzero = np.sum(high_density_mask > 0)
    mask_total = high_density_mask.size
    mask_min = np.min(high_density_mask)
    mask_max = np.max(high_density_mask)
    print(f"  DEBUG: Mask has {mask_nonzero}/{mask_total} non-zero pixels ({100*mask_nonzero/mask_total:.2f}%)")
    print(f"  DEBUG: Mask value range: [{mask_min}, {mask_max}]")
    print(f"  DEBUG: Mask dtype: {high_density_mask.dtype}")
    print(f"  DEBUG: Used {len(selected_streamlines)} streamlines after filtering")
    
    if mask_nonzero == 0:
        print(f"  ERROR: Mask is completely black! This means no streamlines passed the filtering criteria.")
        print(f"  Suggestions: 1) Lower static_streamline_threshold (currently {static_streamline_threshold})")
        print(f"               2) Check if streamlines actually intersect this slice")
        print(f"               3) If using white_matter_only, verify white mask is correct")
    
    # Save high-density mask
    mask_dir = os.path.dirname(output_file)
    mask_basename = os.path.splitext(os.path.basename(output_file))[0]
    mask_filename = f"{mask_dir}/{mask_basename}_high_density_mask_slice{slice_idx}.png"
    
    from .utils import save_image_1024
    save_image_1024(mask_filename, high_density_mask, is_mask=True, target_size=output_image_size)
    print(f"Applied high-density mask for coronal slice {slice_idx}: {mask_filename} ({output_image_size[0]}x{output_image_size[1]})")
