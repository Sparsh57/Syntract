#!/usr/bin/env python
"""
TRUE 3D VOLUMETRIC PROCESSING - PROPER 3D CLAHE
Applies CLAHE to entire 3D volume as a single unit.
NO tiling artifacts - kernel size adapts to volume dimensions.
"""

import numpy as np
from skimage import exposure
from scipy.ndimage import gaussian_filter, uniform_filter
import warnings
import random

# Import TRUE 3D cornucopia augmentation
try:
    from .cornucopia_3d import apply_cornucopia_true_3d
    CORNUCOPIA_AVAILABLE = True
except ImportError:
    try:
        from cornucopia_3d import apply_cornucopia_true_3d
        CORNUCOPIA_AVAILABLE = True
    except ImportError:
        CORNUCOPIA_AVAILABLE = False
        print("Warning: TRUE 3D Cornucopia augmentation not available")


def normalize_blockface_slices(volume_3d, axis=1):
    """
    EXPERT: Normalize blockface slice intensities BEFORE any processing.
    
    Blockface imaging creates stacked 2D photos with varying lighting/contrast.
    This function normalizes each slice to eliminate inter-slice discontinuities
    at the SOURCE, not just smoothing artifacts after the fact.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Raw input volume (blockface stacked data)
    axis : int
        Stacking axis (default: 1 for coronal blockface stacks)
    
    Returns
    -------
    np.ndarray
        Slice-normalized volume with eliminated intensity discontinuities
    """
    print(f"    Pre-processing blockface slices (axis={axis})...")
    
    volume_normalized = volume_3d.copy()
    num_slices = volume_3d.shape[axis]
    
    # Compute target histogram from middle slices (usually best quality)
    mid_start = num_slices // 3
    mid_end = 2 * num_slices // 3
    
    if axis == 0:
        reference_volume = volume_3d[mid_start:mid_end, :, :]
    elif axis == 1:
        reference_volume = volume_3d[:, mid_start:mid_end, :]
    else:
        reference_volume = volume_3d[:, :, mid_start:mid_end]
    
    # Compute reference statistics
    ref_mean = np.mean(reference_volume)
    ref_std = np.std(reference_volume)
    ref_min = np.percentile(reference_volume, 1)
    ref_max = np.percentile(reference_volume, 99)
    
    print(f"      Normalizing {num_slices} slices to reference (mean={ref_mean:.1f}, std={ref_std:.1f})...")
    
    # Normalize each slice individually
    for i in range(num_slices):
        # Extract slice
        if axis == 0:
            slice_data = volume_3d[i, :, :]
        elif axis == 1:
            slice_data = volume_3d[:, i, :]
        else:
            slice_data = volume_3d[:, :, i]
        
        # Compute slice statistics
        slice_mean = np.mean(slice_data)
        slice_std = np.std(slice_data)
        
        if slice_std > 1e-6:  # Avoid division by zero
            # Z-score normalization then rescale to reference
            slice_normalized = (slice_data - slice_mean) / slice_std
            slice_normalized = slice_normalized * ref_std + ref_mean
            
            # Clip to reference range
            slice_normalized = np.clip(slice_normalized, ref_min, ref_max)
            
            # Put back
            if axis == 0:
                volume_normalized[i, :, :] = slice_normalized
            elif axis == 1:
                volume_normalized[:, i, :] = slice_normalized
            else:
                volume_normalized[:, :, i] = slice_normalized
    
    print(f"      ✓ Slice normalization complete - eliminated source intensity variations")
    
    return volume_normalized


def apply_interslice_smoothing(volume_3d, axis=1, sigma=1.5):
    """
    Apply smoothing ONLY between adjacent slices, not within slices.
    
    This targets blockface stacking boundaries without blurring tissue details.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input volume
    axis : int
        Stacking axis (default: 1 for coronal)
    sigma : float
        Smoothing strength (default: 1.5)
    
    Returns
    -------
    np.ndarray
        Volume with smoothed inter-slice transitions
    """
    from scipy.ndimage import gaussian_filter1d
    
    print(f"      Smoothing inter-slice boundaries (axis={axis}, sigma={sigma})...")
    
    # Apply 1D gaussian ONLY along the stacking axis
    volume_smooth = gaussian_filter1d(volume_3d.astype(np.float32), sigma=sigma, axis=axis)
    
    return volume_smooth


def apply_true_3d_clahe(volume_3d, clip_limit=0.01, adaptive=False):
    """
    Apply CLAHE to ENTIRE 3D volume with adaptive kernel sizing.
    
    NOW INCLUDES: Pre-normalization for blockface data to eliminate source discontinuities.
    
    The kernel size is automatically set to cover the entire volume (or most of it)
    to avoid tiling artifacts. This processes the volume as a single 3D entity.
    
    Parameters
    ----------
    volume_3d : np.ndarray, shape (X, Y, Z)
        Input 3D volume (raw blockface data)
    clip_limit : float
        Clip limit for CLAHE (default: 0.01)
    adaptive : bool
        If True, kernel size adapts to volume size
        If False, uses global histogram (single tile) - RECOMMENDED for blockface
    
    Returns
    -------
    np.ndarray
        Enhanced 3D volume in range [0, 1] with eliminated slice boundaries
    """
    # CRITICAL: Normalize blockface slices FIRST - this is the root cause fix
    # Blockface = stacked photos with different lighting per slice
    volume_normalized = normalize_blockface_slices(volume_3d, axis=1)  # axis=1 for coronal stacks
    
    # Apply inter-slice smoothing to any remaining boundaries
    volume_normalized = apply_interslice_smoothing(volume_normalized, axis=1, sigma=1.2)
    
    # Now normalize to [0, 1] for CLAHE
    vol_min = np.percentile(volume_normalized, 1)
    vol_max = np.percentile(volume_normalized, 99)
    volume_norm = np.clip((volume_normalized - vol_min) / (vol_max - vol_min + 1e-8), 0, 1)
    
    if adaptive:
        # Adaptive kernel: use LARGE kernel that covers most of the volume
        # This minimizes tiling while still allowing some local adaptation
        shape = volume_norm.shape
        kernel_size = (
            max(shape[0] // 2, 64),  # At least 64, or half the volume
            max(shape[1] // 2, 64),
            max(shape[2] // 2, 64)
        )
        print(f"    Applying adaptive 3D CLAHE (volume={shape}, kernel={kernel_size}, clip={clip_limit})...")
    else:
        # Global: kernel = entire volume (single tile = no boundaries)
        kernel_size = volume_norm.shape
        print(f"    Applying GLOBAL 3D CLAHE (single histogram for entire volume, clip={clip_limit})...")
    
    # Apply 3D CLAHE with computed kernel size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume_clahe = exposure.equalize_adapthist(
            volume_norm,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=256
        )
    
    # Post-CLAHE: Final targeted smoothing at any residual boundaries
    # Much gentler now that source normalization is done
    print("    Final inter-slice boundary smoothing...")
    
    # Detect and smooth only remaining high-frequency discontinuities
    from scipy.ndimage import gaussian_filter1d
    
    # Check for discontinuities along stacking axis
    grad_1d = np.gradient(volume_clahe, axis=1)  # axis=1 for coronal
    grad_abs = np.abs(grad_1d)
    grad_smooth = gaussian_filter1d(grad_abs, sigma=2.0, axis=1)
    
    # Only smooth where there are still visible discontinuities
    threshold = np.percentile(grad_smooth, 90)
    high_grad_mask = (grad_smooth > threshold).astype(np.float32)
    high_grad_mask_smooth = gaussian_filter(high_grad_mask, sigma=1.0)
    
    # Very gentle final smoothing
    volume_final_smooth = gaussian_filter1d(volume_clahe, sigma=0.5, axis=1)
    
    # Blend: smooth only at residual boundaries
    volume_smooth = (1 - high_grad_mask_smooth * 0.4) * volume_clahe + \
                    (high_grad_mask_smooth * 0.4) * volume_final_smooth
    
    return volume_smooth


def enhance_multiscale_3d(volume_3d, sigmas=[5.0, 10.0, 20.0], weights=[0.5, 0.3, 0.2]):
    """
    Multi-scale smooth contrast enhancement - captures details at different scales.
    
    Combines multiple Gaussian scales to enhance features from fine to coarse.
    Completely smooth - no tile boundaries at any scale.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume in range [0, 1]
    sigmas : list of float
        Gaussian smoothing scales (smaller = finer details)
    weights : list of float
        Weights for each scale (should sum to ~1.0)
    
    Returns
    -------
    np.ndarray
        Multi-scale enhanced volume
    """
    print(f"    Multi-scale enhancement (scales: {sigmas})...")
    
    enhanced = np.zeros_like(volume_3d)
    
    for sigma, weight in zip(sigmas, weights):
        # Compute difference from smoothed version at this scale
        smoothed = gaussian_filter(volume_3d, sigma=sigma)
        detail = volume_3d - smoothed
        
        # Add weighted detail back
        enhanced += weight * detail
    
    # Combine with original
    enhanced = volume_3d + 0.5 * enhanced
    enhanced = np.clip(enhanced, 0, 1)
    
    # Final smoothing to ensure continuity
    enhanced = gaussian_filter(enhanced, sigma=2.0)
    
    return enhanced


def enhance_unsharp_mask_3d(volume_3d, sigma=10.0, amount=1.5):
    """
    3D Unsharp masking - classic sharpening technique without artifacts.
    
    Enhances edges and fine details while maintaining smooth appearance.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume in range [0, 1]
    sigma : float
        Blur radius (larger = affects larger features)
    amount : float
        Sharpening strength (1.0-3.0 typical range)
    
    Returns
    -------
    np.ndarray
        Sharpened volume
    """
    print(f"    Unsharp mask enhancement (sigma={sigma}, amount={amount})...")
    
    # Create blurred version
    blurred = gaussian_filter(volume_3d, sigma=sigma)
    
    # Extract high-frequency details
    detail = volume_3d - blurred
    
    # Add amplified details back
    enhanced = volume_3d + amount * detail
    enhanced = np.clip(enhanced, 0, 1)
    
    # Subtle smoothing to prevent any sharp discontinuities
    enhanced = gaussian_filter(enhanced, sigma=1.5)
    
    return enhanced


def enhance_adaptive_local_3d(volume_3d, sigma_small=8.0, sigma_large=20.0, strength=0.8):
    """
    Adaptive local enhancement - enhances based on local content.
    
    Uses dual-scale approach to adapt enhancement to tissue structure.
    Preserves smooth transitions everywhere.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume in range [0, 1]
    sigma_small : float
        Small-scale features
    sigma_large : float
        Large-scale features
    strength : float
        Enhancement strength (0.5-1.5)
    
    Returns
    -------
    np.ndarray
        Adaptively enhanced volume
    """
    print(f"    Adaptive local enhancement (dual-scale: {sigma_small}/{sigma_large})...")
    
    # Compute local statistics at two scales
    local_mean_small = gaussian_filter(volume_3d, sigma=sigma_small)
    local_mean_large = gaussian_filter(volume_3d, sigma=sigma_large)
    
    # Local contrast measure
    local_contrast = np.abs(local_mean_small - local_mean_large)
    local_contrast = gaussian_filter(local_contrast, sigma=3.0)  # Smooth the contrast map
    
    # Enhance more where there's more local structure
    enhancement_map = 1.0 + strength * local_contrast
    
    # Apply adaptive enhancement
    enhanced = volume_3d * enhancement_map
    enhanced = np.clip(enhanced, 0, 1)
    
    # Final smoothing for seamless appearance
    enhanced = gaussian_filter(enhanced, sigma=2.0)
    
    return enhanced


def enhance_bilateral_inspired_3d(volume_3d, sigma_spatial=12.0, intensity_range=0.15):
    """
    Bilateral filter-inspired enhancement - edge-preserving smoothing.
    
    Enhances while preserving edges using intensity-aware filtering.
    Completely smooth implementation without edge artifacts.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume in range [0, 1]
    sigma_spatial : float
        Spatial smoothing scale
    intensity_range : float
        Intensity similarity threshold
    
    Returns
    -------
    np.ndarray
        Edge-preserved enhanced volume
    """
    print(f"    Bilateral-inspired enhancement (sigma={sigma_spatial})...")
    
    # Compute intensity-weighted local mean
    smoothed = gaussian_filter(volume_3d, sigma=sigma_spatial)
    
    # Compute local variance as proxy for edges
    local_var = gaussian_filter(volume_3d**2, sigma=sigma_spatial) - smoothed**2
    local_var = np.clip(local_var, 0, None)
    
    # Adaptive enhancement: more at edges, less in smooth regions
    edge_weight = 1.0 - np.exp(-local_var / (intensity_range**2))
    edge_weight = gaussian_filter(edge_weight, sigma=3.0)
    
    # Blend original with smoothed based on edge content
    enhanced = edge_weight * volume_3d + (1 - edge_weight) * smoothed
    
    # Subtle contrast boost
    enhanced = 0.3 * smoothed + 0.7 * enhanced
    enhanced = np.clip(enhanced, 0, 1)
    
    return enhanced


def add_3d_texture_field(volume_3d, intensity=0.02, sigma=8.0):
    """
    Add ultra-smooth 3D texture/noise field to volume.
    Creates realistic tissue variation WITHOUT any visible boundaries.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume in range [0, 1]
    intensity : float
        Strength of texture (default: 0.02 for very subtle effect)
    sigma : float
        Gaussian smoothing factor for 3D noise field
        Larger = smoother (default: 8.0 for ultra-smooth)
    
    Returns
    -------
    np.ndarray
        Volume with added 3D texture
    """
    print(f"    Adding ultra-smooth 3D texture field (sigma={sigma}, intensity={intensity})...")
    
    # Generate 3D noise field
    noise_3d = np.random.randn(*volume_3d.shape) * intensity
    
    # Apply HEAVY smoothing in 3D space - ensures no visible patterns
    noise_3d_smooth = gaussian_filter(noise_3d, sigma=sigma)
    
    # Add to volume and clip
    volume_textured = np.clip(volume_3d + noise_3d_smooth, 0, 1)
    
    return volume_textured


def apply_3d_gamma_darkfield(volume_3d, gamma=2.2):
    """
    Apply gamma correction to entire 3D volume for dark field microscopy appearance.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume in range [0, 1]
    gamma : float
        Gamma value (>1 darkens, <1 brightens)
        Default: 2.2 for dark field appearance
    
    Returns
    -------
    np.ndarray
        Gamma-corrected volume
    """
    print(f"    Applying 3D gamma correction (gamma={gamma})...")
    return np.power(volume_3d, gamma)


def apply_bilateral_smoothing_3d(volume_3d, sigma_spatial=0.8, sigma_intensity=0.25, iterations=1):
    """
    Apply gentle edge-preserving bilateral-style filtering to reduce discontinuities.
    
    This approximates bilateral filtering using iterative Gaussian smoothing
    with intensity-based weighting. Parameters tuned to eliminate artifacts
    while preserving fine details and sharpness.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume in range [0, 1]
    sigma_spatial : float
        Spatial gaussian sigma (default: 1.2 - gentle smoothing)
    sigma_intensity : float
        Intensity gaussian sigma for edge preservation (default: 0.20 - strong edge preservation)
    iterations : int
        Number of filtering passes (default: 1 - minimal processing)
    
    Returns
    -------
    np.ndarray
        Smoothed volume with preserved edges
    """
    print(f"    Applying gentle bilateral smoothing (spatial={sigma_spatial}, intensity={sigma_intensity})...")
    
    filtered = volume_3d.copy()
    
    for _ in range(iterations):
        # Spatial smoothing
        spatial_smooth = gaussian_filter(filtered, sigma=sigma_spatial)
        
        # Intensity-based weighting (preserve edges)
        diff = np.abs(filtered - spatial_smooth)
        weights = np.exp(-diff**2 / (2 * sigma_intensity**2))
        
        # Blend based on edge strength
        filtered = weights * spatial_smooth + (1 - weights) * filtered
    
    return filtered


def process_volume_full_3d(volume_3d,
                           use_clahe=True,
                           clahe_adaptive=False,
                           clahe_clip_limit=0.01,
                           add_texture=True,
                           texture_intensity=0.02,
                           texture_sigma=10.0,
                           gamma=2.2,
                           scaling_factor=150.0,
                           use_bilateral=True,
                           use_cornucopia=False,
                           cornucopia_preset=None,
                           cornucopia_allowed_presets=None,
                           cornucopia_prob=0.9,
                           random_state=None):
    """
    Complete 3D volumetric processing with PROPER 3D CLAHE.
    CLAHE is applied to the ENTIRE volume as a single unit - no tiling artifacts.
    
    Parameters
    ----------
    volume_3d : np.ndarray, shape (X, Y, Z)
        Input 3D volume (any value range)
    use_clahe : bool
        Apply TRUE 3D CLAHE (recommended)
    clahe_adaptive : bool
        If True, uses adaptive kernel (large, but not entire volume)
        If False, uses global histogram (entire volume as single tile)
    clahe_clip_limit : float
        CLAHE clip limit (0.01 typical)
    add_texture : bool
        Add ultra-smooth 3D texture field
    texture_intensity : float
        Texture strength (very subtle by default)
    texture_sigma : float
        Texture smoothing (larger = smoother, no visible patterns)
    gamma : float
        Gamma correction for dark field effect
    scaling_factor : float
        Output scaling (e.g., 150.0 or 255.0)
    use_bilateral : bool
        Apply bilateral filtering for edge-preserving smoothness (recommended)
    use_cornucopia : bool
        Apply TRUE 3D cornucopia augmentation (NO slice iteration, NO discontinuities)
    cornucopia_preset : str or None
        Specific preset or None to randomly select from allowed list
        Options: 'extreme_noise', 'random_shapes_background',
                 'comprehensive_aggressive', 'ultra_heavy_speckle'
    cornucopia_allowed_presets : list or None
        List of allowed presets for random selection
    cornucopia_prob : float
        Probability of applying cornucopia (0.0-1.0)
    random_state : int or None
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Processed 3D volume, ready for streamline rendering
    """
    print(f"  TRUE 3D CLAHE PROCESSING (entire volume, no tiling):")
    
    # Step 0: Initial normalization to [0, 1] range
    print("    Initial normalization...")
    vol_min = np.percentile(volume_3d, 1)
    vol_max = np.percentile(volume_3d, 99)
    volume_normalized = np.clip((volume_3d - vol_min) / (vol_max - vol_min + 1e-8), 0, 1)
    
    # Step 1: Apply TRUE 3D cornucopia augmentation EARLY (before CLAHE)
    # KEY: Uses 3D noise fields and 3D transforms - NO slice-by-slice iteration
    # This ELIMINATES inter-slice discontinuities at the source!
    if use_cornucopia:
        # Scale to 0-255 for cornucopia (it expects this range)
        volume_for_corn = (volume_normalized * 255.0).astype(np.float32)
        
        volume_augmented = apply_cornucopia_true_3d(
            volume_for_corn,
            preset=cornucopia_preset,
            allowed_presets=cornucopia_allowed_presets,
            apply_prob=cornucopia_prob,
            random_state=random_state
        )
        
        # Normalize back to [0, 1]
        aug_min = np.percentile(volume_augmented, 1)
        aug_max = np.percentile(volume_augmented, 99)
        volume_normalized = np.clip((volume_augmented - aug_min) / (aug_max - aug_min + 1e-8), 0, 1)
        
        # Note: NO inter-slice smoothing needed! TRUE 3D operations have no discontinuities
    
    # Step 2: Apply TRUE 3D CLAHE to entire volume
    if use_clahe:
        volume_processed = apply_true_3d_clahe(
            volume_normalized,
            clip_limit=clahe_clip_limit,
            adaptive=clahe_adaptive
        )
        
        # Apply gentle bilateral smoothing for edge-preserving discontinuity reduction
        if use_bilateral:
            volume_processed = apply_bilateral_smoothing_3d(
                volume_processed,
                sigma_spatial=0.8,
                sigma_intensity=0.25,
                iterations=1
            )
    else:
        # Use normalized volume directly
        volume_processed = volume_normalized
    
    # Step 3: Add ultra-smooth 3D texture field (optional)
    if add_texture:
        volume_processed = add_3d_texture_field(
            volume_processed,
            intensity=texture_intensity,
            sigma=texture_sigma
        )
    
    # Step 4: Gamma correction for dark field appearance
    volume_processed = apply_3d_gamma_darkfield(volume_processed, gamma=gamma)
    
    # Step 5: Scale to output range
    print(f"    Scaling to output range (factor={scaling_factor})...")
    volume_output = (volume_processed * scaling_factor).astype(np.float32)
    
    print(f"    ✓ 3D processing complete - shape: {volume_output.shape}, " + 
          f"range: [{volume_output.min():.2f}, {volume_output.max():.2f}]")
    
    return volume_output
