"""
TRUE 3D CORNUCOPIA AUGMENTATION

Applies cornucopia effects to ENTIRE 3D volume at once - NO slice-by-slice iteration.
This eliminates inter-slice discontinuities by using 3D noise fields and 3D transforms.
"""

import numpy as np
from scipy import ndimage
import random
from typing import Optional, List


def apply_3d_gaussian_mixture_noise(volume_3d, sigma_range=(0.3, 1.0), prob=0.98, random_state=None, verbose=True):
    """
    Apply Gaussian mixture noise to ENTIRE 3D volume.
    
    Generates 3D noise field - NOT slice-by-slice.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume
    sigma_range : tuple
        Range for Gaussian noise standard deviation
    prob : float
        Probability of applying
    random_state : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Volume with 3D Gaussian mixture noise
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if random.random() > prob:
        return volume_3d
    
    if verbose:
        print(f"      Applying 3D Gaussian mixture noise (sigma={sigma_range})...")
    
    sigma = random.uniform(*sigma_range)
    
    # Generate 3D noise field (multiple Gaussian components)
    shape = volume_3d.shape
    noise1 = np.random.normal(0, sigma, shape)
    noise2 = np.random.normal(0, sigma * 0.5, shape)
    noise3 = np.random.normal(0, sigma * 1.5, shape)
    
    # Mix the components
    mixture_noise = 0.4 * noise1 + 0.3 * noise2 + 0.3 * noise3
    
    # Apply as additive noise
    result = volume_3d + mixture_noise
    
    return result


def apply_3d_noncentral_chi_noise(volume_3d, df_range=(1, 6), nc_range=(1.0, 4.0),
                                  prob=0.98, random_state=None, verbose=True):
    """
    Apply noncentral chi noise to ENTIRE 3D volume (speckle-like pattern).
    
    Generates 3D noise field - NOT slice-by-slice.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume
    df_range : tuple
        Degrees of freedom range
    nc_range : tuple
        Noncentrality parameter range
    prob : float
        Probability of applying
    random_state : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Volume with 3D noncentral chi noise
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if random.random() > prob:
        return volume_3d
    
    if verbose:
        print(f"      Applying 3D noncentral chi noise (df={df_range}, nc={nc_range})...")
    
    df = random.uniform(*df_range)
    nc = random.uniform(*nc_range)
    
    # Generate 3D noncentral chi noise field
    shape = volume_3d.shape
    chi_noise = np.random.noncentral_chisquare(df, nc, shape)
    
    # Normalize to reasonable range
    chi_noise = chi_noise / np.max(chi_noise) * 0.5
    
    # Apply as multiplicative noise
    result = volume_3d * (1.0 + chi_noise)
    
    return result


def apply_3d_aggressive_gamma(volume_3d, gamma_range=(0.1, 4.0), prob=0.98, random_state=None, verbose=True):
    """
    Apply aggressive gamma correction to ENTIRE 3D volume.
    
    Single gamma value for entire volume - NO per-slice variation.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume in range [0, 1] or similar
    gamma_range : tuple
        Range for gamma values
    prob : float
        Probability of applying
    random_state : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Gamma-corrected volume
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    if random.random() > prob:
        return volume_3d
    
    gamma = random.uniform(*gamma_range)
    if verbose:
        print(f"      Applying 3D aggressive gamma (gamma={gamma:.3f})...")
    
    # Normalize to [0, 1] if needed
    v_min, v_max = volume_3d.min(), volume_3d.max()
    if v_max > 1.5:
        volume_norm = (volume_3d - v_min) / (v_max - v_min + 1e-8)
    else:
        volume_norm = volume_3d
    
    # Apply gamma to ENTIRE volume at once
    result = np.power(np.clip(volume_norm, 0, 1), gamma)
    
    # Scale back to original range
    if v_max > 1.5:
        result = result * (v_max - v_min) + v_min
    
    return result


def apply_3d_bias_field(volume_3d, strength_range=(0.5, 1.5), prob=0.98,
                        smoothing_sigma=None, random_state=None, verbose=True):
    """
    Apply smooth 3D bias field (multiplicative intensity variation).
    
    Creates SINGLE 3D polynomial field - NOT per-slice.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume
    strength_range : tuple
        Range for bias field strength
    prob : float
        Probability of applying
    smoothing_sigma : float, optional
        Gaussian smoothing for field (auto-calculated if None)
    random_state : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Volume with 3D bias field applied
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    if random.random() > prob:
        return volume_3d
    
    strength = random.uniform(*strength_range)
    if verbose:
        print(f"      Applying 3D bias field (strength={strength:.3f})...")
    
    # Get volume dimensions
    d, h, w = volume_3d.shape
    
    # Create 3D coordinate grids
    z_coords, y_coords, x_coords = np.mgrid[0:d, 0:h, 0:w]
    
    # Normalize coordinates to [-1, 1]
    z_norm = (z_coords - d/2) / (d/2)
    y_norm = (y_coords - h/2) / (h/2)
    x_norm = (x_coords - w/2) / (w/2)
    
    # Create smooth 3D polynomial bias field
    bias_field = 1.0 + strength * (
        0.3 * x_norm + 
        0.3 * y_norm + 
        0.2 * z_norm +
        0.1 * (x_norm**2 - y_norm**2) +
        0.1 * (y_norm**2 - z_norm**2)
    )
    
    # Apply heavy Gaussian smoothing to make it very smooth
    if smoothing_sigma is None:
        smoothing_sigma = min(d, h, w) / 20
    
    bias_field = ndimage.gaussian_filter(bias_field, sigma=smoothing_sigma)
    
    # Apply multiplicative bias field
    result = volume_3d * bias_field
    
    return result


def apply_cornucopia_true_3d(volume_3d, 
                             preset='comprehensive_aggressive',
                             allowed_presets=None,
                             apply_prob=0.9,
                             random_state=None,
                             verbose=True):
    """
    Apply cornucopia augmentation using TRUE 3D operations.
    
    KEY DIFFERENCE: All noise and intensity transforms are applied to the
    ENTIRE 3D volume at once, NOT slice-by-slice. This eliminates inter-slice
    discontinuities completely.
    
    Parameters
    ----------
    volume_3d : np.ndarray
        Input 3D volume (should be in 0-255 range for best results)
    preset : str or None
        Cornucopia preset to use. Options:
        - 'extreme_noise': 3D Gaussian mixture + aggressive gamma
        - 'random_shapes_background': 3D Gaussian mixture + aggressive gamma
        - 'comprehensive_aggressive': 3D noncentral chi + aggressive bias field
        - 'ultra_heavy_speckle': 3D noncentral chi + aggressive bias field
        If None, randomly selects from allowed_presets
    allowed_presets : list or None
        List of allowed presets to randomly choose from
    apply_prob : float
        Probability of applying augmentation (0.0-1.0)
    random_state : int or None
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Augmented 3D volume with NO inter-slice discontinuities
    """
    # Set random seed
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Define allowed presets
    if allowed_presets is None:
        allowed_presets = [
            'granular_realistic',
            'extreme_noise',
            'random_shapes_background',
            'comprehensive_aggressive',
            'ultra_heavy_speckle'
        ]

    # Select preset
    if preset is None:
        preset = random.choice(allowed_presets)
    elif preset not in allowed_presets:
        if verbose:
            print(f"    Warning: Preset '{preset}' not in allowed list, using 'comprehensive_aggressive'")
        preset = 'comprehensive_aggressive'
    
    # Check if we should apply augmentation
    if random.random() > apply_prob:
        if verbose:
            print(f"    Skipping cornucopia augmentation (apply_prob={apply_prob})")
        return volume_3d
    
    if verbose:
        print(f"    Applying TRUE 3D cornucopia augmentation:")
        print(f"      Preset: '{preset}'")
        print(f"      Volume shape: {volume_3d.shape}")
        print(f"      Method: ENTIRE volume at once (NO slice iteration)")
    
    # Apply augmentation based on preset
    volume_augmented = volume_3d.copy()
    
    if preset == 'extreme_noise':
        # 3D Gaussian mixture + aggressive gamma
        volume_augmented = apply_3d_gaussian_mixture_noise(
            volume_augmented,
            sigma_range=(0.6, 2.2),
            prob=0.98,
            random_state=random_state,
            verbose=verbose,
        )
        volume_augmented = apply_3d_aggressive_gamma(
            volume_augmented,
            gamma_range=(0.05, 8.0),
            prob=0.98,
            random_state=random_state + 1 if random_state else None,
            verbose=verbose,
        )
    
    elif preset == 'random_shapes_background':
        # 3D Gaussian mixture + aggressive gamma (no shapes)
        volume_augmented = apply_3d_gaussian_mixture_noise(
            volume_augmented,
            sigma_range=(0.5, 2.0),
            prob=0.98,
            random_state=random_state,
            verbose=verbose,
        )
        volume_augmented = apply_3d_aggressive_gamma(
            volume_augmented,
            gamma_range=(0.1, 4.0),
            prob=0.95,
            random_state=random_state + 1 if random_state else None,
            verbose=verbose,
        )
    
    elif preset == 'comprehensive_aggressive':
        # 3D noncentral chi + aggressive bias field
        volume_augmented = apply_3d_noncentral_chi_noise(
            volume_augmented,
            df_range=(0.5, 4),
            nc_range=(2.0, 7.0),
            prob=0.98,
            random_state=random_state,
            verbose=verbose,
        )
        volume_augmented = apply_3d_bias_field(
            volume_augmented,
            strength_range=(0.8, 2.2),
            prob=0.98,
            random_state=random_state + 1 if random_state else None,
            verbose=verbose,
        )
    
    elif preset == 'ultra_heavy_speckle':
        # 3D noncentral chi + aggressive bias field
        volume_augmented = apply_3d_noncentral_chi_noise(
            volume_augmented,
            df_range=(0.3, 4),
            nc_range=(1.5, 6.0),
            prob=0.98,
            random_state=random_state,
            verbose=verbose,
        )
        volume_augmented = apply_3d_bias_field(
            volume_augmented,
            strength_range=(0.6, 1.8),
            prob=0.98,
            random_state=random_state + 1 if random_state else None,
            verbose=verbose,
        )

    elif preset == 'granular_realistic':
        # Dense granular texture (Gaussian mixture) + speckle-like chi noise + mild bias field.
        # Keeps fiber signal intact while burying it in realistic tissue clutter so the model
        # must learn to identify structure rather than memorise clean signal.
        volume_augmented = apply_3d_gaussian_mixture_noise(
            volume_augmented,
            sigma_range=(0.2, 0.8),
            prob=0.98,
            random_state=random_state,
            verbose=verbose,
        )
        volume_augmented = apply_3d_noncentral_chi_noise(
            volume_augmented,
            df_range=(1, 4),
            nc_range=(0.5, 2.5),
            prob=0.95,
            random_state=random_state + 1 if random_state else None,
            verbose=verbose,
        )
        volume_augmented = apply_3d_bias_field(
            volume_augmented,
            strength_range=(0.92, 1.12),
            prob=0.90,
            random_state=random_state + 2 if random_state else None,
            verbose=verbose,
        )

    if verbose:
        print(f"      ✓ TRUE 3D cornucopia augmentation complete")
        print(f"        Output range: [{volume_augmented.min():.2f}, {volume_augmented.max():.2f}]")
        print(f"        NO inter-slice discontinuities (all operations were 3D)")
    
    return volume_augmented
