"""
Utility functions for NIfTI tractography visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random


def select_random_streamlines(streamlines, percentage=10.0, random_state=None):
    """
    Randomly sample a subset of streamlines for visualization or analysis.
    """
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


def generate_tract_color_variation(base_color=(1.0, 0.8, 0.1), variation=0.2, random_state=None):
    """
    Generate a variation of the base tract color.
    """
    if random_state is not None:
        random.seed(random_state)
        
    r, g, b = base_color
    r_var = np.clip(r + random.uniform(-variation, variation), 0.7, 1.0)
    g_var = np.clip(g + random.uniform(-variation, variation), 0.7, 1.0)
    b_var = np.clip(b + random.uniform(-variation, variation), 0.0, 0.3)
    
    return (r_var, g_var, b_var)


def get_colormap(color_scheme='bw', blue_tint=0.1):
    """
    Returns a dark field-style colormap with added white and minimal blue tint.
    'bw' gives pure black to lighter grey with some white.
    'blue' gives a dark greyscale with very minimal blue tone (no red).
    """
    if color_scheme == 'blue':
        # Very minimal blue tint with more brightness
        dark_field_cmap = LinearSegmentedColormap.from_list('dark_field_blue', [
            (0.0, (0.0, 0.0, 0.0)),                                    # pure black
            (0.3, (0.08, 0.08, 0.08 + blue_tint * 0.08)),              # dark with very minimal blue
            (0.6, (0.18, 0.18, 0.18 + blue_tint * 0.12)),              # medium-dark grey with slight blue
            (0.85, (0.35, 0.35, 0.35 + blue_tint * 0.15)),             # lighter grey with minimal blue
            (1.0, (0.55, 0.55, 0.55 + blue_tint * 0.18))               # light grey approaching white
        ], N=256)
    else:
        # Black to light grey with more white (no color)
        dark_field_cmap = LinearSegmentedColormap.from_list('dark_field_bw', [
            (0.0, (0.0, 0.0, 0.0)),      # pure black
            (0.3, (0.08, 0.08, 0.08)),   # dark grey
            (0.6, (0.18, 0.18, 0.18)),   # medium-dark grey
            (0.85, (0.35, 0.35, 0.35)),  # lighter grey
            (1.0, (0.55, 0.55, 0.55))    # light grey approaching white
        ], N=256)

    dark_field_cmap.set_under('black')
    dark_field_cmap.set_bad('black')

    return dark_field_cmap


def visualize_labeled_bundles(labeled_mask, output_file=None, colormap='viridis', background_color='black'):
    """
    Visualize labeled fiber bundles with different colors.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
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