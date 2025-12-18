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



def generate_orange_blob_mask(width, height, num_blobs=None):
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
