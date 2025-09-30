"""
Orange Streamline Generator for Synthetic Brain Data

This module generates realistic orange-colored streamlines to simulate injection site artifacts
commonly seen in real brain imaging data. These orange streamlines help train models to ignore
orange fiber contamination from injection sites during fiber tract analysis.
"""

import numpy as np
import cv2
from scipy import ndimage
import random
from scipy import ndimage
from skimage.morphology import disk, ball
from skimage.filters import gaussian
import random


def generate_orange_streamlines_for_injection_site(patch_center, patch_size, num_streamlines=None):
    """
    Generate dense orange streamlines to simulate injection site fiber contamination.
    
    Args:
        patch_center (tuple): (x, y, z) center coordinates of the injection area
        patch_size (tuple): (width, height, depth) size of the patch
        num_streamlines (int, optional): Number of orange streamlines to generate
    
    Returns:
        list: List of orange streamline coordinates
    """
    if num_streamlines is None:
        num_streamlines = np.random.randint(50, 200)  # Dense streamlines for injection site
    
    orange_streamlines = []
    width, height, depth = patch_size
    center_x, center_y, center_z = patch_center
    
    # Create injection site area (irregular shape)
    injection_radius = min(width, height, depth) * np.random.uniform(0.15, 0.35)
    
    for i in range(num_streamlines):
        # Generate streamlines radiating from injection site with some randomness
        streamline_points = []
        
        # Start point near injection center with some spread
        start_offset_x = np.random.normal(0, injection_radius * 0.3)
        start_offset_y = np.random.normal(0, injection_radius * 0.3)
        start_offset_z = np.random.normal(0, injection_radius * 0.2)
        
        start_x = center_x + start_offset_x
        start_y = center_y + start_offset_y
        start_z = center_z + start_offset_z
        
        # Ensure start point is within bounds
        start_x = np.clip(start_x, 5, width - 5)
        start_y = np.clip(start_y, 5, height - 5)
        start_z = np.clip(start_z, 0, depth - 1)
        
        # Generate streamline path (short, dense fibers typical of injection sites)
        streamline_length = np.random.randint(20, 80)  # Short fibers
        current_x, current_y, current_z = start_x, start_y, start_z
        
        # Random direction with bias towards radiating outward
        direction_x = np.random.normal(0, 1)
        direction_y = np.random.normal(0, 1)
        direction_z = np.random.normal(0, 0.3)  # Less Z variation
        
        # Normalize direction
        direction_length = np.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
        if direction_length > 0:
            direction_x /= direction_length
            direction_y /= direction_length
            direction_z /= direction_length
        
        for step in range(streamline_length):
            streamline_points.append([current_x, current_y, current_z])
            
            # Small step size for dense appearance
            step_size = np.random.uniform(0.5, 1.5)
            
            # Add some curvature/noise to make it look natural
            noise_x = np.random.normal(0, 0.1)
            noise_y = np.random.normal(0, 0.1)
            noise_z = np.random.normal(0, 0.05)
            
            current_x += (direction_x + noise_x) * step_size
            current_y += (direction_y + noise_y) * step_size
            current_z += (direction_z + noise_z) * step_size
            
            # Bounds checking
            if (current_x < 0 or current_x >= width or 
                current_y < 0 or current_y >= height or 
                current_z < 0 or current_z >= depth):
                break
        
        if len(streamline_points) > 5:  # Only keep substantial streamlines
            orange_streamlines.append(np.array(streamline_points))
    
    return orange_streamlines


def add_orange_streamlines_to_visualization(fig, ax, orange_streamlines, color='orange', linewidth=2, alpha=0.8):
    """
    Add orange streamlines to a matplotlib visualization.
    
    Args:
        fig: matplotlib figure
        ax: matplotlib axis
        orange_streamlines: list of streamline coordinate arrays
        color: color for the streamlines (default: bright orange)
        linewidth: width of streamline rendering
        alpha: transparency of streamlines
    """
    orange_color = '#FF8C00'  # Bright orange color
    
    for streamline in orange_streamlines:
        if len(streamline) > 1:
            # Draw streamline
            ax.plot(streamline[:, 0], streamline[:, 1], 
                   color=orange_color, linewidth=linewidth, alpha=alpha, 
                   solid_capstyle='round', solid_joinstyle='round')
    
    return len(orange_streamlines)
    """
    Generate a mask for orange blob placement with adaptive sizing.
    
    Args:
        width (int): Image width
        height (int): Image height
        num_blobs (int, optional): Number of blobs to generate. If None, randomly chooses 1-3.
    
    Returns:
        numpy.ndarray: Binary mask where True indicates blob locations
    """
    if num_blobs is None:
        num_blobs = np.random.randint(1, 4)  # 1-3 blobs
    
    mask = np.zeros((height, width), dtype=bool)
    
    # Calculate adaptive blob size based on image dimensions
    # Use 2-6% of the image area for blob coverage (realistic injection site size)
    min_coverage = 0.02  # 2%
    max_coverage = 0.06  # 6%
    
    for _ in range(num_blobs):
        # Random coverage percentage for this blob
        coverage = np.random.uniform(min_coverage, max_coverage)
        blob_area = int(width * height * coverage / num_blobs)
        
        # Convert area to approximate radius (assuming circular blob)
        radius = int(np.sqrt(blob_area / np.pi))
        radius = max(15, min(radius, min(width, height) // 3))  # Reasonable bounds with larger minimum
        
        # Generate random center for the injection site
        center_x = np.random.randint(radius, width - radius)
        center_y = np.random.randint(radius, height - radius)
        
        # Create irregular injection site shape instead of perfect circle
        # Start with multiple seed points to create organic shape
        num_seeds = np.random.randint(3, 8)  # Multiple origin points
        temp_mask = np.zeros((height, width), dtype=bool)
        
        for seed in range(num_seeds):
            # Offset seed points around main center
            offset_x = center_x + np.random.randint(-radius//3, radius//3)
            offset_y = center_y + np.random.randint(-radius//3, radius//3)
            
            # Ensure seeds stay within bounds
            offset_x = np.clip(offset_x, radius//2, width - radius//2)
            offset_y = np.clip(offset_y, radius//2, height - radius//2)
            
            # Create irregular blob around each seed
            y, x = np.ogrid[:height, :width]
            # Vary the radius for each seed to create irregular shape
            seed_radius = radius * np.random.uniform(0.6, 1.2)
            # Add some elliptical distortion
            x_stretch = np.random.uniform(0.7, 1.3)
            y_stretch = np.random.uniform(0.7, 1.3)
            
            ellipse_mask = (((x - offset_x) * x_stretch) ** 2 + ((y - offset_y) * y_stretch) ** 2) <= seed_radius ** 2
            temp_mask = temp_mask | ellipse_mask
        
        # Apply morphological operations to create more realistic injection site texture
        from scipy import ndimage
        
        # Dilate and erode to create irregular boundaries
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        
        # Random dilation/erosion for organic shape
        if np.random.random() > 0.5:
            temp_mask = ndimage.binary_dilation(temp_mask, structure, iterations=np.random.randint(1, 3))
        if np.random.random() > 0.5:
            temp_mask = ndimage.binary_erosion(temp_mask, structure, iterations=np.random.randint(1, 2))
        
        # Add some noise to edges for realistic texture
        noise = np.random.random((height, width)) > 0.85  # Sparse noise
        edge_noise = ndimage.binary_dilation(temp_mask, iterations=2) & ~temp_mask
        temp_mask = temp_mask | (edge_noise & noise)
        
        mask = mask | temp_mask
    
    return mask.astype(np.float32)


def create_orange_coloration(image, mask, orange_params=None):
    """
    Apply orange coloration to specific regions of an image.
    
    Parameters:
    -----------
    image : ndarray
        Input image (grayscale or RGB)
    mask : ndarray
        Binary or float mask indicating where to apply orange coloration
    orange_params : dict, optional
        Parameters controlling orange appearance:
        - intensity: how strong the orange effect is (0-1, default: 0.6)
        - hue_shift: hue adjustment for orange color (default: 30 degrees)
        - saturation: saturation level (0-1, default: 0.7)
        - blend_mode: 'overlay', 'multiply', 'screen' (default: 'overlay')
        - feather: edge softening amount (default: 3)
    
    Returns:
    --------
    orange_image : ndarray
        Image with orange coloration applied
    """
    if orange_params is None:
        orange_params = {}
    
    # Default parameters - increased for brighter, more visible orange
    intensity = orange_params.get('intensity', 0.8)  # Increased from 0.6
    hue_shift = orange_params.get('hue_shift', 30)  # degrees
    saturation = orange_params.get('saturation', 0.9)  # Increased from 0.7
    blend_mode = orange_params.get('blend_mode', 'overlay')
    feather = orange_params.get('feather', 2)  # Reduced for sharper edges
    
    # Ensure image is in float format
    if image.dtype != np.float32:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.copy()
    
    # Ensure mask is same shape as image (handle 2D mask with 3D image)
    if len(image_float.shape) == 3 and len(mask.shape) == 2:
        # Expand mask to match image channels
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, image_float.shape[-1], axis=-1)
    elif len(image_float.shape) == 2 and len(mask.shape) == 2:
        # Both are 2D, keep as is
        pass
    elif len(image_float.shape) == 3 and len(mask.shape) == 3:
        # Both are 3D, ensure compatible
        if mask.shape[-1] == 1 and image_float.shape[-1] > 1:
            mask = np.repeat(mask, image_float.shape[-1], axis=-1)
    
    # Convert to RGB if grayscale
    if len(image_float.shape) == 2:
        image_rgb = np.stack([image_float] * 3, axis=-1)
    else:
        image_rgb = image_float.copy()
    
    # Feather the mask edges
    if feather > 0:
        if len(mask.shape) == 3:
            # Apply gaussian to each channel separately
            mask_feathered = np.zeros_like(mask)
            for i in range(mask.shape[-1]):
                mask_feathered[..., i] = gaussian(mask[..., i].astype(np.float32), sigma=feather)
        else:
            mask_feathered = gaussian(mask.astype(np.float32), sigma=feather)
    else:
        mask_feathered = mask.astype(np.float32)
    
    # Create orange color - brighter and more saturated for better visibility
    # Make it a vibrant orange that stands out against brain tissue
    orange_color = np.array([1.0, 0.6, 0.0])  # Bright orange
    
    # Create orange overlay
    orange_overlay = np.zeros_like(image_rgb)
    if len(orange_overlay.shape) == 3:
        for i in range(3):
            orange_overlay[..., i] = orange_color[i] * mask_feathered[..., min(i, mask_feathered.shape[-1]-1)]
    else:
        # Grayscale case
        orange_overlay = orange_color[0] * mask_feathered  # Use red component for grayscale
    
    # Apply blending
    if blend_mode == 'overlay':
        # Overlay blend mode
        result = np.where(image_rgb < 0.5,
                         2 * image_rgb * orange_overlay,
                         1 - 2 * (1 - image_rgb) * (1 - orange_overlay))
    elif blend_mode == 'multiply':
        result = image_rgb * (1 + orange_overlay * saturation)
    elif blend_mode == 'screen':
        result = 1 - (1 - image_rgb) * (1 - orange_overlay)
    elif blend_mode == 'normal':
        # Normal blend mode - direct replacement for fully opaque injection sites
        result = orange_overlay * saturation
    else:
        # Default to simple addition
        result = image_rgb + orange_overlay * saturation
    
    # Blend with original based on intensity and mask
    if len(mask_feathered.shape) == 3:
        final_mask = mask_feathered[..., 0] * intensity  # Use first channel for intensity
        final_mask = np.expand_dims(final_mask, axis=-1)
        final_mask = np.repeat(final_mask, image_rgb.shape[-1], axis=-1)
    else:
        final_mask = mask_feathered * intensity
        if len(result.shape) == 3:
            final_mask = np.expand_dims(final_mask, axis=-1)
            final_mask = np.repeat(final_mask, result.shape[-1], axis=-1)
    
    result = image_rgb * (1 - final_mask) + result * final_mask
    
    # Ensure values are in valid range
    result = np.clip(result, 0, 1)
    
    # Convert back to original format
    if image.dtype != np.float32:
        result = (result * 255).astype(image.dtype)
    
    return result


def add_orange_injection_artifacts(image, num_sites=None, size_range=(20, 100), 
                                  intensity_range=(0.8, 1.0), opacity_range=(1.0, 1.0),
                                  random_state=42, site_params=None):
    """
    Add orange injection site artifacts to a brain image.
    
    This is the main function that combines blob generation and orange coloration
    to simulate realistic injection site artifacts.
    
    Parameters:
    -----------
    image : ndarray
        Input brain image (2D or 3D, grayscale or RGB)
    num_sites : int, optional
        Number of injection sites to simulate (default: 1-2)
    site_params : dict, optional
        Parameters for injection site appearance
    random_state : int, optional
        Random seed for reproducible results
    
    Returns:
    --------
    orange_image : ndarray
        Image with orange injection artifacts
    artifact_mask : ndarray
        Mask showing where artifacts were added
    """
    if random_state is not None and isinstance(random_state, int):
        np.random.seed(random_state)
        random.seed(random_state)
    
    if num_sites is None:
        num_sites = random.randint(1, 2)
    
    if site_params is None:
        site_params = {}
    
    # Get image shape
    image_shape = image.shape
    if len(image_shape) == 3 and image_shape[-1] in [1, 3, 4]:
        # RGB/RGBA image
        spatial_shape = image_shape[:-1]
    else:
        # Grayscale or 3D volume
        spatial_shape = image_shape
    
    # Generate blob mask for each injection site
    total_mask = np.zeros(spatial_shape, dtype=np.float32)
    
    for site_idx in range(num_sites):
        # Vary parameters for each site
        blob_params = {
            'num_blobs': random.randint(1, 2),
            'min_size': random.uniform(8, 15),
            'max_size': random.uniform(15, 25),
            'irregularity': random.uniform(0.2, 0.5),
            'connectivity': random.uniform(0.1, 0.4)
        }
        blob_params.update(site_params.get('blob_params', {}))
        
        site_mask = generate_orange_blob_mask(spatial_shape[1], spatial_shape[0], blob_params.get('num_blobs'))
        total_mask = np.maximum(total_mask, site_mask)
    
    # Apply orange coloration with fully opaque, realistic injection site appearance
    orange_params = {
        'intensity': random.uniform(0.8, 1.0),  # High intensity for realistic injection sites
        'saturation': random.uniform(0.9, 1.0),  # High saturation for vivid orange
        'opacity': 1.0,  # Fully opaque injection sites
        'blend_mode': 'normal',  # Direct replacement for opaque appearance
        'feather': random.uniform(0.5, 1.5)  # Minimal feathering for defined edges
    }
    orange_params.update(site_params.get('orange_params', {}))
    
    orange_image = create_orange_coloration(image, total_mask, orange_params)
    
    return orange_image, total_mask


def add_orange_blobs_to_streamlines_visualization(image, streamline_mask=None, 
                                                 proximity_factor=0.3, **kwargs):
    """
    Add orange blobs specifically near streamline regions to simulate
    injection artifacts that commonly occur near fiber tracts.
    
    Parameters:
    -----------
    image : ndarray
        Input visualization image
    streamline_mask : ndarray, optional
        Binary mask of streamline locations (if available)
    proximity_factor : float
        How close to place blobs near streamlines (0-1)
    **kwargs : dict
        Additional parameters passed to add_orange_injection_artifacts
    
    Returns:
    --------
    orange_image : ndarray
        Image with orange blobs near streamlines
    artifact_mask : ndarray
        Mask showing where artifacts were added
    """
    if streamline_mask is not None:
        # Modify site placement to be near streamlines
        if 'site_params' not in kwargs:
            kwargs['site_params'] = {}
        
        # Create a probability map favoring areas near streamlines
        if proximity_factor > 0:
            # Dilate streamline mask to create preferred regions
            from scipy.ndimage import binary_dilation
            preferred_regions = binary_dilation(streamline_mask, iterations=10)
            
            # This could be used to bias blob placement (implementation depends on needs)
            # For now, we'll just use the standard approach
            pass
    
    return add_orange_injection_artifacts(image, **kwargs)


# Convenience function for integration with existing pipeline
def apply_orange_artifacts(image, enable=True, **kwargs):
    """
    Convenience function to optionally apply orange artifacts.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    enable : bool
        Whether to apply orange artifacts
    **kwargs : dict
        Parameters passed to add_orange_injection_artifacts
    
    Returns:
    --------
    result_image : ndarray
        Image with or without orange artifacts
    artifact_mask : ndarray or None
        Artifact mask (None if enable=False)
    """
    if not enable:
        return image, None
    
    return add_orange_injection_artifacts(image, **kwargs)


def apply_orange_blobs_to_saved_image(image_path, output_path=None, random_state=None):
    """
    Apply orange blobs to an already saved image file.
    
    Parameters:
    -----------
    image_path : str
        Path to the saved image file
    random_state : int, optional
        Random seed for reproducible blob generation
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Load the saved image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Apply orange artifacts
        orange_image, artifact_mask = apply_orange_artifacts(
            image_array, 
            enable=True,
            num_sites=np.random.randint(1, 2) if random_state is None else None,
            random_state=random_state
        )
        
        # Save the modified image
        if orange_image.dtype != np.uint8:
            orange_image = (orange_image * 255).astype(np.uint8)
        
        orange_pil = Image.fromarray(orange_image)
        save_path = output_path if output_path else image_path
        orange_pil.save(save_path)
        
        return save_path
        
    except Exception as e:
        print(f"Warning: Failed to apply orange blobs to {image_path}: {e}")
        return None