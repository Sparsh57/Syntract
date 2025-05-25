import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.streamlines import load
from matplotlib.colors import LinearSegmentedColormap
from skimage import exposure, filters, util
from dipy.tracking.streamline import transform_streamlines
from matplotlib.collections import LineCollection
import random


def apply_clahe_to_slice(slice_data, clip_limit=0.01, tile_grid_size=(8, 8)):
    # Normalize to [0, 1] for CLAHE
    slice_norm = (slice_data - np.min(slice_data)) / (np.ptp(slice_data) + 1e-8)
    slice_clahe = exposure.equalize_adapthist(slice_norm, clip_limit=clip_limit, kernel_size=tile_grid_size)
    return slice_clahe


def apply_dark_field_effect(slice_clahe, intensity_params=None):
    """
    Apply a dark field microscopy effect with controllable parameters.
    
    Parameters
    ----------
    slice_clahe : ndarray
        CLAHE-processed slice
    intensity_params : dict, optional
        Parameters to control the dark field effect:
        - gamma: float, gamma correction value (default: random between 0.8-1.2)
        - threshold: float, threshold for deep black (default: random between 0.02-0.08)
        - contrast_stretch: tuple, percentiles for contrast stretching (default: (0.5, 99.5))
        - background_boost: float, factor to enhance background (default: random between 0.9-1.1)
        - color_scheme: str, 'bw' for black and white, 'blue' for bluish tint (default: random)
        - blue_tint: float, amount of blue tint to apply (default: random between 0.1-0.4)
    
    Returns
    -------
    ndarray
        Dark field processed image
    """
    # Default parameters with randomization if not specified
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Invert colors - dark field has dark background with bright structures
    inverted = 1 - slice_clahe
    
    # Apply gamma correction with parameter
    dark_field = np.power(inverted, intensity_params['gamma'])
    
    # Apply bilateral filter to enhance edges while preserving smoothness
    dark_field = filters.gaussian(dark_field, sigma=0.5)
    
    # Enhance contrast using percentile-based stretching
    p_low, p_high = intensity_params['contrast_stretch']
    p1, p99 = np.percentile(dark_field, (p_low, p_high))
    dark_field_stretched = exposure.rescale_intensity(dark_field, in_range=(p1, p99))
    
    # Apply background boost to control intensity of tissue
    dark_field_stretched = dark_field_stretched * intensity_params['background_boost']
    
    # Ensure very deep black background by aggressively clipping low values
    threshold = intensity_params['threshold']
    dark_field_stretched[dark_field_stretched < threshold] = 0
    
    # Add subtle noise to simulate microscopy grain
    noise_level = random.uniform(0.005, 0.02)
    noise = noise_level * np.random.normal(0, 1, dark_field_stretched.shape)
    dark_field_stretched = np.clip(dark_field_stretched + noise, 0, 1)
    
    return dark_field_stretched


def densify_streamline(streamline, step=0.2):
    # Linear interpolation to densify streamline for smoothness
    if len(streamline) < 2:
        return streamline
    diffs = np.cumsum(np.r_[0, np.linalg.norm(np.diff(streamline, axis=0), axis=1)])
    n_points = max(int(diffs[-1] / step), 2)
    new_dists = np.linspace(0, diffs[-1], n_points)
    new_points = np.empty((n_points, 3))
    for i in range(3):
        new_points[:, i] = np.interp(new_dists, diffs, streamline[:, i])
    return new_points


def get_intensity_along_streamline(slice_data, x, y):
    # x, y are in voxel coordinates; slice_data is 2D
    x = np.clip(np.round(x).astype(int), 0, slice_data.shape[1] - 1)
    y = np.clip(np.round(y).astype(int), 0, slice_data.shape[0] - 1)
    return slice_data[y, x]


def generate_tract_color_variation(base_color=(1.0, 1.0, 0.0), variation=0.2):
    """
    Generate a variation of the base tract color.
    
    Parameters
    ----------
    base_color : tuple
        Base RGB color (default: yellow)
    variation : float
        Amount of variation to apply (default: 0.2)
        
    Returns
    -------
    tuple
        Varied RGB color with alpha=1.0
    """
    r, g, b = base_color
    # Apply random variation to each channel
    r_var = np.clip(r + random.uniform(-variation, variation), 0.7, 1.0)
    g_var = np.clip(g + random.uniform(-variation, variation), 0.7, 1.0)
    b_var = np.clip(b + random.uniform(-variation, variation), 0.0, 0.3)  # Keep blue low for yellowish colors
    
    return (r_var, g_var, b_var)


def get_colormap(color_scheme='bw', blue_tint=0.3):
    """
    Get appropriate colormap for dark field visualization
    
    Parameters
    ----------
    color_scheme : str
        'bw' for black and white, 'blue' for bluish-gray tint
    blue_tint : float
        Amount of blue tint to apply (0.0-1.0), higher values increase blue intensity
        
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Colormap for visualization
    """
    if color_scheme == 'blue':
        # Bluish-gray colormap for dark field effect
        dark_field_cmap = LinearSegmentedColormap.from_list('dark_field_blue', [
            (0, 0, 0),                                      # Pure black
            (0.35, 0.35, 0.35 + blue_tint),                 # Dark gray with blue tint
            (0.55, 0.55, 0.55 + min(0.45, blue_tint * 1.5)) # Light gray with blue tint
        ], N=256)
    else:  # 'bw' or any other value defaults to black and white
        # Pure black and white colormap
        dark_field_cmap = LinearSegmentedColormap.from_list('dark_field_bw', [
            (0, 0, 0),       # Pure black
            (0.4, 0.4, 0.4), # Dark gray
            (0.7, 0.7, 0.7)  # Light gray
        ], N=256)
    
    return dark_field_cmap


def visualize_nifti_with_trk(nifti_file, trk_file, output_file=None, n_slices=1, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=8, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2,
                             slice_idx=None):
    """
    Visualize multiple axial slices of a nifti file with tractography overlaid
    
    Parameters
    ----------
    nifti_file : str
        Path to the nifti file
    trk_file : str
        Path to the trk file
    output_file : str, optional
        Path to save the output image. If None, the image will be displayed.
    n_slices : int, optional
        Number of slices to display (default: 1)
    cmap : str, optional
        Colormap for the nifti data
    clahe_clip_limit : float, optional
        CLAHE clip limit (default: 0.01)
    clahe_tile_grid_size : int, optional
        CLAHE tile grid size (default: 8)
    intensity_params : dict, optional
        Parameters for dark field effect (see apply_dark_field_effect)
    tract_color_base : tuple, optional
        Base RGB color for tracts (default: yellow)
    tract_color_variation : float, optional
        Variation in tract color (default: 0.2)
    slice_idx : int, optional
        Specific slice index to visualize. If None, slices will be evenly spaced.
    """
    # Load nifti data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()

    # Get dimensions
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

    # Convert streamlines to voxel coordinates if available
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))

    # Calculate slice positions or use specific slice
    if slice_idx is not None:
        slice_positions = [slice_idx]
        n_slices = 1
    else:
        # Calculate slice positions for axial view (z-axis)
        slice_positions = np.linspace(dims[2] // 4, 3 * dims[2] // 4, n_slices).astype(int)

    # Create figure for axial view
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    # Use default intensity parameters if none provided
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap based on parameters
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    # Set figure background to black for better dark field appearance
    fig.patch.set_facecolor('black')

    # Plot each slice with tractography
    for i, slice_idx in enumerate(slice_positions):
        # Use a copy of the intensity parameters for this slice
        slice_intensity_params = intensity_params.copy() if intensity_params else None
        
        slice_data = nii_data[:, :, slice_idx]
        slice_clahe = apply_clahe_to_slice(slice_data, clip_limit=clahe_clip_limit,
                                           tile_grid_size=(clahe_tile_grid_size, clahe_tile_grid_size))
        
        # Apply dark field effect to mimic dark field microscopy
        dark_field_slice = apply_dark_field_effect(slice_clahe, slice_intensity_params)
        
        # Display with colormap for dark field effect
        axes[i].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='nearest')
        axes[i].set_facecolor('black')  # Set axes background to black
        
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                # Project all points onto the axial plane
                x = sl_dense[:, 0]
                y = sl_dense[:, 1]
                z = sl_dense[:, 2]
                
                # Only include streamlines near this slice
                distance_to_slice = np.abs(z - slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > 2.0:  # Only show streamlines within 2 voxels of the slice
                    continue
                
                # For np.rot90, swap x/y and flip y
                y_plot = dims[1] - y - 1
                points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Generate varied color for this streamline
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation)
                
                # Adjust opacity based on distance to slice
                base_opacity = max(0.0, 1.0 - min_distance / 2.0)
                
                # Make streamlines bright with the generated color
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))  # Add variable alpha
                    
            if segments:
                # Use linewidth between 0.8-1.2 for more natural appearance
                linewidth = random.uniform(0.8, 1.2)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[i].add_collection(lc)
        axes[i].set_title(f"Axial - Slice {slice_idx}", color='white')  # White text for black background
        axes[i].axis('off')

    plt.tight_layout()

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {output_file}")
    else:
        plt.show()

    return fig, axes


def visualize_nifti_with_trk_coronal(nifti_file, trk_file, output_file=None, n_slices=1, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=8, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2,
                             slice_idx=None):
    """
    Visualize multiple coronal slices of a nifti file with tractography overlaid
    
    Parameters
    ----------
    nifti_file : str
        Path to the nifti file
    trk_file : str
        Path to the trk file
    output_file : str, optional
        Path to save the output image. If None, the image will be displayed.
    n_slices : int, optional
        Number of slices to display (default: 1)
    cmap : str, optional
        Colormap for the nifti data
    clahe_clip_limit : float, optional
        CLAHE clip limit (default: 0.01)
    clahe_tile_grid_size : int, optional
        CLAHE tile grid size (default: 8)
    intensity_params : dict, optional
        Parameters for dark field effect (see apply_dark_field_effect)
    tract_color_base : tuple, optional
        Base RGB color for tracts (default: yellow)
    tract_color_variation : float, optional
        Variation in tract color (default: 0.2)
    slice_idx : int, optional
        Specific slice index to visualize. If None, slices will be evenly spaced.
    """
    # Load nifti data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()

    # Get dimensions
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

    # Convert streamlines to voxel coordinates if available
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))

    # Calculate slice positions or use specific slice
    if slice_idx is not None:
        slice_positions = [slice_idx]
        n_slices = 1
    else:
        # Calculate slice positions for coronal view (y-axis)
        slice_positions = np.linspace(dims[1] // 4, 3 * dims[1] // 4, n_slices).astype(int)

    # Create figure for coronal view
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    # Use default intensity parameters if none provided
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap based on parameters
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    # Set figure background to black for better dark field appearance
    fig.patch.set_facecolor('black')

    # Plot each slice with tractography
    for i, slice_idx in enumerate(slice_positions):
        # Use a copy of the intensity parameters for this slice
        slice_intensity_params = intensity_params.copy() if intensity_params else None
        
        # Get coronal slice (x-z plane)
        slice_data = nii_data[:, slice_idx, :]
        slice_clahe = apply_clahe_to_slice(slice_data, clip_limit=clahe_clip_limit,
                                           tile_grid_size=(clahe_tile_grid_size, clahe_tile_grid_size))
        
        # Apply dark field effect to mimic dark field microscopy
        dark_field_slice = apply_dark_field_effect(slice_clahe, slice_intensity_params)
        
        # Display with colormap for dark field effect
        axes[i].imshow(np.rot90(dark_field_slice), cmap=dark_field_cmap, aspect='equal', interpolation='nearest')
        axes[i].set_facecolor('black')  # Set axes background to black
        
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                # Extract coordinates
                x = sl_dense[:, 0]
                y = sl_dense[:, 1]
                z = sl_dense[:, 2]
                
                # Only include streamlines near this coronal slice
                distance_to_slice = np.abs(y - slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > 2.0:  # Only show streamlines within 2 voxels of the slice
                    continue
                
                # For coronal view, we need x and z coordinates, with z flipped for display
                z_plot = dims[2] - z - 1
                points = np.array([x, z_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Generate varied color for this streamline
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation)
                
                # Adjust opacity based on distance to slice
                base_opacity = max(0.0, 1.0 - min_distance / 2.0)
                
                # Make streamlines bright with the generated color
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))  # Add variable alpha
                    
            if segments:
                # Use linewidth between 0.8-1.2 for more natural appearance
                linewidth = random.uniform(0.8, 1.2)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[i].add_collection(lc)
        
        axes[i].set_title(f"Coronal - Slice {slice_idx}", color='white')
        axes[i].axis('off')

    plt.tight_layout()

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {output_file}")
    else:
        plt.show()

    return fig, axes


def visualize_multiple_views(nifti_file, trk_file, output_file=None, cmap='gray',
                             clahe_clip_limit=0.01, clahe_tile_grid_size=8, intensity_params=None,
                             tract_color_base=(1.0, 1.0, 0.0), tract_color_variation=0.2):
    """
    Visualize axial, coronal, and sagittal views of a nifti file with tractography overlaid
    
    Parameters
    ----------
    nifti_file : str
        Path to the nifti file
    trk_file : str
        Path to the trk file
    output_file : str, optional
        Path to save the output image. If None, the image will be displayed.
    cmap : str, optional
        Colormap for the nifti data
    clahe_clip_limit : float, optional
        CLAHE clip limit (default: 0.01)
    clahe_tile_grid_size : int, optional
        CLAHE tile grid size (default: 8)
    intensity_params : dict, optional
        Parameters for dark field effect (see apply_dark_field_effect)
    tract_color_base : tuple, optional
        Base RGB color for tracts (default: yellow)
    tract_color_variation : float, optional
        Variation in tract color (default: 0.2)
    """
    # Load nifti data
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()

    # Get dimensions
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

    # Convert streamlines to voxel coordinates if available
    streamlines_voxel = []
    if has_streamlines:
        affine_inv = np.linalg.inv(nii_img.affine)
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))

    # Create figure with three orientations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Use default intensity parameters if none provided
    if intensity_params is None:
        intensity_params = {
            'gamma': random.uniform(0.8, 1.2),
            'threshold': random.uniform(0.02, 0.08),
            'contrast_stretch': (random.uniform(0.1, 1.0), random.uniform(99.0, 99.9)),
            'background_boost': random.uniform(0.9, 1.1),
            'color_scheme': random.choice(['bw', 'blue']),
            'blue_tint': random.uniform(0.1, 0.4)
        }
    
    # Get colormap based on parameters
    color_scheme = intensity_params.get('color_scheme', 'bw')
    blue_tint = intensity_params.get('blue_tint', 0.3)
    dark_field_cmap = get_colormap(color_scheme, blue_tint)

    # Set figure background to black for better dark field appearance
    fig.patch.set_facecolor('black')

    # Distance threshold for tractography visualization
    distance_threshold = 1.0

    # Use the same parameters for all views for consistency
    axial_params = intensity_params.copy()
    coronal_params = intensity_params.copy()
    sagittal_params = intensity_params.copy()

    # 1. Axial view (middle slice)
    axial_slice_idx = dims[2] // 2
    axial_slice = nii_data[:, :, axial_slice_idx]
    axial_clahe = apply_clahe_to_slice(axial_slice, clip_limit=clahe_clip_limit,
                                       tile_grid_size=(clahe_tile_grid_size, clahe_tile_grid_size))
    
    # Apply dark field effect
    axial_dark_field = apply_dark_field_effect(axial_clahe, axial_params)
    
    axes[0].imshow(np.rot90(axial_dark_field), cmap=dark_field_cmap, aspect='equal', interpolation='nearest')
    axes[0].set_title(f"Axial - Slice {axial_slice_idx}", color='white')
    axes[0].set_facecolor('black')
    
    if has_streamlines:
        segments = []
        colors = []
        for sl in streamlines_voxel:
            sl_dense = densify_streamline(sl)
            # Filter points near the slice
            z = sl_dense[:, 2]
            distance_to_slice = np.abs(z - axial_slice_idx)
            min_distance = np.min(distance_to_slice)
            if min_distance > distance_threshold:
                continue
                
            # Project onto axial plane
            x = sl_dense[:, 0]
            y = sl_dense[:, 1]
            y_plot = dims[1] - y - 1
            points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Generate color variation for this streamline
            tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation)
            
            # Adjust opacity based on distance to slice
            base_opacity = max(0.0, 1.0 - min_distance / distance_threshold)
            
            for seg in segs:
                segments.append(seg)
                colors.append(tract_color + (base_opacity,))  # Add variable alpha
                
        if segments:
            # Use variable linewidth for more natural appearance
            linewidth = random.uniform(0.8, 1.2)
            lc = LineCollection(segments, colors=colors, linewidths=linewidth)
            axes[0].add_collection(lc)
            
        # 2. Coronal view (middle slice)
        coronal_slice_idx = dims[1] // 2
        coronal_slice = nii_data[:, coronal_slice_idx, :]
        coronal_clahe = apply_clahe_to_slice(coronal_slice, clip_limit=clahe_clip_limit,
                                             tile_grid_size=(clahe_tile_grid_size, clahe_tile_grid_size))
        
        # Apply dark field effect
        coronal_dark_field = apply_dark_field_effect(coronal_clahe, coronal_params)
        
        axes[1].imshow(np.rot90(coronal_dark_field), cmap=dark_field_cmap, aspect='equal', interpolation='nearest')
        axes[1].set_title(f"Coronal - Slice {coronal_slice_idx}", color='white')
        axes[1].set_facecolor('black')

        # Add streamlines to coronal view
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                # Filter points near the slice
                y = sl_dense[:, 1]
                distance_to_slice = np.abs(y - coronal_slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > distance_threshold:
                    continue
                    
                # Project points onto coronal plane (x,z)
                x = sl_dense[:, 0]
                z = sl_dense[:, 2]
                # For np.rot90, swap x/z and flip z
                z_plot = dims[2] - z - 1
                points = np.array([x, z_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Generate color variation
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation)
                
                # Adjust opacity based on distance to slice
                base_opacity = max(0.0, 1.0 - min_distance / distance_threshold)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))  # Add variable alpha
                    
            if segments:
                # Use variable linewidth
                linewidth = random.uniform(0.8, 1.2)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[1].add_collection(lc)

        # 3. Sagittal view (middle slice)
        sagittal_slice_idx = dims[0] // 2
        sagittal_slice = nii_data[sagittal_slice_idx, :, :]
        sagittal_clahe = apply_clahe_to_slice(sagittal_slice, clip_limit=clahe_clip_limit,
                                              tile_grid_size=(clahe_tile_grid_size, clahe_tile_grid_size))
        
        # Apply dark field effect
        sagittal_dark_field = apply_dark_field_effect(sagittal_clahe, sagittal_params)
        
        axes[2].imshow(np.rot90(sagittal_dark_field), cmap=dark_field_cmap, aspect='equal', interpolation='nearest')
        axes[2].set_title(f"Sagittal - Slice {sagittal_slice_idx}", color='white')
        axes[2].set_facecolor('black')

        # Add streamlines to sagittal view
        if has_streamlines:
            segments = []
            colors = []
            for sl in streamlines_voxel:
                sl_dense = densify_streamline(sl)
                # Filter points near the slice
                x = sl_dense[:, 0]
                distance_to_slice = np.abs(x - sagittal_slice_idx)
                min_distance = np.min(distance_to_slice)
                if min_distance > distance_threshold:
                    continue
                    
                # Project points onto sagittal plane (y,z)
                y = sl_dense[:, 1]
                z = sl_dense[:, 2]
                # For np.rot90, swap y/z and flip z
                z_plot = dims[2] - z - 1
                points = np.array([y, z_plot]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Generate color variation
                tract_color = generate_tract_color_variation(tract_color_base, tract_color_variation)
                
                # Adjust opacity based on distance to slice
                base_opacity = max(0.0, 1.0 - min_distance / distance_threshold)
                
                for seg in segs:
                    segments.append(seg)
                    colors.append(tract_color + (base_opacity,))  # Add variable alpha
                    
            if segments:
                # Use variable linewidth
                linewidth = random.uniform(0.8, 1.2)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth)
                axes[2].add_collection(lc)

    # Remove axes
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {output_file}")
    else:
        plt.show()

    return fig, axes


def generate_varied_examples(nifti_file, trk_file, output_dir, n_examples=5, prefix="synthetic_", 
                             slice_mode="coronal", intensity_variation=True, tract_color_variation=True,
                             specific_slice=None):
    """
    Generate multiple varied examples with different contrast settings
    
    Parameters
    ----------
    nifti_file : str
        Path to the nifti file
    trk_file : str
        Path to the trk file
    output_dir : str
        Directory to save output images
    n_examples : int, optional
        Number of examples to generate (default: 5)
    prefix : str, optional
        Prefix for output filenames (default: "synthetic_")
    slice_mode : str, optional
        "axial", "coronal", or "all" for multiple views (default: "coronal")
    intensity_variation : bool, optional
        Whether to vary intensity parameters (default: True)
    tract_color_variation : bool, optional
        Whether to vary tract colors (default: True)
    specific_slice : int, optional
        If provided, use this specific slice index for all examples
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the slice index if specific_slice is provided
    if specific_slice is not None:
        # Load the nifti to get dimensions
        nii_img = nib.load(nifti_file)
        dims = nii_img.shape
        
        # Verify the slice index is within bounds
        if slice_mode == "axial" and (specific_slice < 0 or specific_slice >= dims[2]):
            print(f"Warning: Slice index {specific_slice} out of bounds for axial dimension {dims[2]}")
            specific_slice = dims[2] // 2
        elif slice_mode == "coronal" and (specific_slice < 0 or specific_slice >= dims[1]):
            print(f"Warning: Slice index {specific_slice} out of bounds for coronal dimension {dims[1]}")
            specific_slice = dims[1] // 2
    
    for i in range(n_examples):
        # Generate random parameters for this example
        if intensity_variation:
            # Vary color scheme every other example to ensure we get both types
            if i % 3 == 0:
                color_scheme = 'bw'  # Black and white
                blue_tint = 0
            elif i % 3 == 1:
                color_scheme = 'blue'  # Blue tint
                blue_tint = random.uniform(0.2, 0.4)
            else:
                # Random choice with weighted preference for blue
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
            
        # Generate random tract color if enabled
        if tract_color_variation:
            # Randomly select from a range of yellowish colors
            base_r = random.uniform(0.9, 1.0)
            base_g = random.uniform(0.9, 1.0)
            base_b = random.uniform(0.0, 0.2)
            tract_color_base = (base_r, base_g, base_b)
            color_var = random.uniform(0.05, 0.2)
        else:
            tract_color_base = (1.0, 1.0, 0.0)  # Default yellow
            color_var = 0.0
            
        # Set output filename
        output_file = os.path.join(output_dir, f"{prefix}{i+1:03d}.png")
        
        # Generate visualization
        if slice_mode == "axial":
            # Use single slice for variation
            n_slices = 1
            visualize_nifti_with_trk(
                nifti_file, trk_file, output_file, n_slices=n_slices,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                slice_idx=specific_slice
            )
        elif slice_mode == "coronal":
            # Use single slice for variation
            visualize_nifti_with_trk_coronal(
                nifti_file, trk_file, output_file, n_slices=1,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var,
                slice_idx=specific_slice
            )
        else:
            visualize_multiple_views(
                nifti_file, trk_file, output_file,
                intensity_params=intensity_params,
                tract_color_base=tract_color_base, 
                tract_color_variation=color_var
            )
        
        print(f"Generated example {i+1}/{n_examples}: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize nifti slices with overlaid tractography')
    parser.add_argument('nifti_file', help='Path to the nifti file')
    parser.add_argument('trk_file', help='Path to the tractography file')
    parser.add_argument('--output', '-o', help='Output image file (optional)', default=None)
    parser.add_argument('--slices', '-s', type=int, help='Number of slices to display (for axial/coronal view)', default=1)
    parser.add_argument('--slice_idx', '-i', type=int, help='Specific slice index to visualize', default=None)
    parser.add_argument('--mode', '-m', choices=['axial', 'coronal', 'all'], help='Visualization mode', default='coronal')
    parser.add_argument('--clahe_clip_limit', type=float, default=0.01, help='CLAHE clip limit (default: 0.01)')
    parser.add_argument('--clahe_tile_grid_size', type=int, default=8, help='CLAHE tile grid size (default: 8)')
    parser.add_argument('--generate_examples', '-g', type=int, default=0, 
                        help='Generate N varied examples (default: 0 = disabled)')
    parser.add_argument('--output_dir', '-d', default='./synthetic_examples',
                        help='Output directory for generated examples (default: ./synthetic_examples)')
    parser.add_argument('--color_scheme', '-c', choices=['bw', 'blue', 'random'], default='random',
                        help='Color scheme to use (black-white, bluish, or random)')
    parser.add_argument('--blue_tint', '-b', type=float, default=0.3,
                        help='Amount of blue tint (0.0-1.0) when using blue color scheme')

    args = parser.parse_args()

    # Validate file existence
    if not os.path.exists(args.nifti_file):
        print(f"Error: Nifti file {args.nifti_file} does not exist")
        exit(1)

    if not os.path.exists(args.trk_file):
        print(f"Error: Tractography file {args.trk_file} does not exist")
        exit(1)

    # Set up color scheme parameters
    if args.color_scheme != 'random':
        intensity_params = {
            'color_scheme': args.color_scheme,
            'blue_tint': args.blue_tint if args.color_scheme == 'blue' else 0.0
        }
    else:
        intensity_params = None  # Use random default

    # Generate examples if requested
    if args.generate_examples > 0:
        generate_varied_examples(
            args.nifti_file, args.trk_file, args.output_dir, 
            n_examples=args.generate_examples, 
            slice_mode=args.mode,
            specific_slice=args.slice_idx
        )
    else:
        # Run visualization based on mode
        if args.mode == 'axial':
            visualize_nifti_with_trk(
                args.nifti_file, args.trk_file, args.output, args.slices,
                clahe_clip_limit=args.clahe_clip_limit, 
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                intensity_params=intensity_params,
                slice_idx=args.slice_idx
            )
        elif args.mode == 'coronal':
            visualize_nifti_with_trk_coronal(
                args.nifti_file, args.trk_file, args.output, args.slices,
                clahe_clip_limit=args.clahe_clip_limit, 
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                intensity_params=intensity_params,
                slice_idx=args.slice_idx
            )
        else:
            visualize_multiple_views(
                args.nifti_file, args.trk_file, args.output,
                clahe_clip_limit=args.clahe_clip_limit, 
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                intensity_params=intensity_params
            )
