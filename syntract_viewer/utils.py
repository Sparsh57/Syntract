"""
Utility functions for NIfTI tractography visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
from PIL import Image
import io


def select_random_streamlines(streamlines, percentage=10.0, random_state=None):
    """Randomly sample a subset of streamlines for visualization or analysis."""
    if len(streamlines) == 0:
        return streamlines
        
    if percentage >= 100.0:
        return streamlines
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_select = max(1, int(len(streamlines) * percentage / 100.0))
    indices = np.random.choice(len(streamlines), n_select, replace=False)
    return [streamlines[i] for i in indices]


def densify_streamline(streamline, step=0.2):
    """Linear interpolation to densify streamline for smoothness."""
    if len(streamline) < 2:
        return streamline
    diffs = np.cumsum(np.r_[0, np.linalg.norm(np.diff(streamline, axis=0), axis=1)])
    n_points = max(int(diffs[-1] / step), 2)
    new_dists = np.linspace(0, diffs[-1], n_points)
    new_points = np.empty((n_points, 3))
    for i in range(3):
        new_points[:, i] = np.interp(new_dists, diffs, streamline[:, i])
    return new_points


def generate_tract_color_variation(base_color=(1.0, 0.8, 0.1), variation=0.2, random_state=None, truly_random=False):
    """Generate color variations for tract visualization with option for true randomization."""
    if truly_random:
        # Use current time for truly random colors
        import time
        true_random_seed = int(time.time() * 1000000) % (2**32)
        random.seed(true_random_seed)
        np.random.seed(true_random_seed)
    elif random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    r, g, b = base_color
    r_var = np.clip(r + random.uniform(-variation, variation), 0.7, 1.0)
    g_var = np.clip(g + random.uniform(-variation, variation), 0.7, 1.0)
    b_var = np.clip(b + random.uniform(-variation, variation), 0.0, 0.3)
    
    return (r_var, g_var, b_var)


def get_colormap(color_scheme='bw', blue_tint=0.1):
    """
    Returns a dark field-style colormap.
    'bw' gives pure black to light grey.
    'blue' gives dark greyscale with minimal blue tone.
    """
    if color_scheme == 'blue':
        dark_field_cmap = LinearSegmentedColormap.from_list('dark_field_blue', [
            (0.0, (0.0, 0.0, 0.0)),
            (0.3, (0.08, 0.08, 0.08 + blue_tint * 0.08)),
            (0.6, (0.18, 0.18, 0.18 + blue_tint * 0.12)),
            (0.85, (0.35, 0.35, 0.35 + blue_tint * 0.15)),
            (1.0, (0.55, 0.55, 0.55 + blue_tint * 0.18))
        ], N=256)
    else:
        dark_field_cmap = LinearSegmentedColormap.from_list('dark_field_bw', [
            (0.0, (0.0, 0.0, 0.0)),
            (0.3, (0.08, 0.08, 0.08)),
            (0.6, (0.18, 0.18, 0.18)),
            (0.85, (0.35, 0.35, 0.35)),
            (1.0, (0.55, 0.55, 0.55))
        ], N=256)

    dark_field_cmap.set_under('black')
    dark_field_cmap.set_bad('black')

    return dark_field_cmap


def visualize_labeled_bundles(labeled_mask, output_file=None, colormap='viridis', background_color='black'):
    """Visualize labeled fiber bundles with different colors."""
    fig, ax = plt.subplots(figsize=(10, 10))  # Increased for better 1024x1024 output
    
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    cmap = plt.get_cmap(colormap)
    num_bundles = np.max(labeled_mask)
    
    if num_bundles == 0:
        ax.text(0.5, 0.5, "No bundles found", color='white', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        background = (labeled_mask == 0)
        colored_mask = np.zeros((*labeled_mask.shape, 4), dtype=np.float32)
        
        for i in range(1, num_bundles + 1):
            bundle_mask = (labeled_mask == i)
            
            if not np.any(bundle_mask):
                continue
                
            color_val = (i - 0.5) / num_bundles
            rgba = cmap(color_val)
            colored_mask[bundle_mask] = rgba
            
        colored_mask[background, 3] = 0
        ax.imshow(colored_mask)
        ax.set_title(f"Fiber Bundles ({num_bundles} total)", color='white', fontsize=14)
        
        for i in range(1, num_bundles + 1):
            bundle_mask = (labeled_mask == i)
            if np.any(bundle_mask):
                y_indices, x_indices = np.where(bundle_mask)
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                
                ax.text(center_x, center_y, str(i), color='white', 
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    
    ax.axis('off')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Labeled bundles visualization saved to {output_file}")
    else:
        plt.show()
        
    return fig, ax


def resize_image_to_1024(image_data, target_size=(1024, 1024), is_mask=False):
    """
    Resize image data to 1024x1024 pixels using PIL for high-quality resampling.
    
    Args:
        image_data: numpy array of image data (2D for grayscale, 3D for RGB)
        target_size: tuple of (width, height) for target size, default (1024, 1024)
        is_mask: boolean indicating if this is a binary mask (affects value scaling)
    
    Returns:
        numpy array of resized image
    """
    # Handle different input types
    if isinstance(image_data, np.ndarray):
        # Handle mask vs image scaling differently
        if is_mask:
            # For masks, ensure binary values are properly scaled to 0-255
            if image_data.dtype == np.bool_ or (image_data.dtype in [np.uint8, np.int32, np.int64] and image_data.max() <= 1):
                # Binary mask: scale 0->0, 1->255
                image_data = (image_data * 255).astype(np.uint8)
            elif image_data.dtype in [np.float32, np.float64]:
                if image_data.max() <= 1.0:
                    # Float mask in 0-1 range: scale to 0-255
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
            # If already uint8 with values > 1, assume it's already properly scaled
        else:
            # For regular images, normalize to 0-255 range if needed
            if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
        
        # Convert to PIL Image
        if len(image_data.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image_data, mode='L')
        elif len(image_data.shape) == 3:  # RGB/RGBA
            if image_data.shape[2] == 3:
                pil_image = Image.fromarray(image_data, mode='RGB')
            elif image_data.shape[2] == 4:
                pil_image = Image.fromarray(image_data, mode='RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {image_data.shape[2]}")
        else:
            raise ValueError(f"Unsupported image shape: {image_data.shape}")
    else:
        raise ValueError("Input must be a numpy array")
    
    # Resize using high-quality resampling
    if is_mask:
        # For masks, use nearest neighbor to preserve binary values
        resized_pil = pil_image.resize(target_size, Image.NEAREST)
    else:
        # For images, use high-quality Lanczos resampling
        resized_pil = pil_image.resize(target_size, Image.LANCZOS)
    
    # Convert back to numpy array
    resized_array = np.array(resized_pil)
    
    # For binary masks, ensure values remain binary after resizing
    if is_mask and resized_array.max() > 1:
        # Threshold to maintain binary nature: anything > 127 becomes 255, else 0
        resized_array = (resized_array > 127).astype(np.uint8) * 255
    
    return resized_array


def save_image_1024(output_path, fig_or_data, is_mask=False, target_size=(1024, 1024)):
    """
    Save image or mask to 1024x1024 pixels.
    
    Args:
        output_path: Path to save the image
        fig_or_data: Either a matplotlib figure or numpy array
        is_mask: Boolean indicating if this is a mask (affects processing)
        target_size: Target size tuple, default (1024, 1024)
    """
    if isinstance(fig_or_data, plt.Figure):
        # Handle matplotlib figure
        # Save to memory buffer first
        buf = io.BytesIO()
        fig_or_data.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                           facecolor='black', pad_inches=0)
        buf.seek(0)
        
        # Open with PIL and resize
        pil_image = Image.open(buf)
        resized_image = pil_image.resize(target_size, Image.LANCZOS)
        resized_image.save(output_path)
        buf.close()
        
    elif isinstance(fig_or_data, np.ndarray):
        # Handle numpy array (mask or image)
        resized_data = resize_image_to_1024(fig_or_data, target_size, is_mask=is_mask)
        
        if is_mask:
            # For masks, convert to RGB (white mask on black background)
            if len(resized_data.shape) > 2:
                resized_data = resized_data[:, :, 0]  # Take first channel if multi-channel
            
            # Create RGB mask: black background (0,0,0), white foreground (255,255,255)
            rgb_mask = np.zeros((resized_data.shape[0], resized_data.shape[1], 3), dtype=np.uint8)
            mask_pixels = resized_data > 0  # Find mask pixels
            rgb_mask[mask_pixels] = [255, 255, 255]  # Set mask pixels to white
            
            pil_image = Image.fromarray(rgb_mask, mode='RGB')
        else:
            # For regular images
            if len(resized_data.shape) == 2:
                pil_image = Image.fromarray(resized_data, mode='L')
            else:
                pil_image = Image.fromarray(resized_data, mode='RGB')
        
        pil_image.save(output_path)
    else:
        raise ValueError("fig_or_data must be either a matplotlib Figure or numpy array") 