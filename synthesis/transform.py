import numpy as np


def build_new_affine(old_affine, old_shape, new_voxel_size, new_shape, patch_center_mm=None, use_gpu=True):
    """
    Build a new affine matrix for resampling while preserving spatial alignment.

    This function creates a new affine transformation that:
    1. Scales the voxel size according to new_voxel_size
    2. Preserves the orientation (rotation/shear) from the original affine
    3. Centers the new volume at the same physical location as the original

    Parameters
    ----------
    old_affine : np.ndarray
        Original 4x4 affine matrix (voxel→world RAS coordinates)
    old_shape : tuple
        Shape of the original volume (x, y, z)
    new_voxel_size : float or tuple
        Desired voxel size in mm. If float, assumes isotropic.
    new_shape : tuple
        New volume shape (x, y, z)
    patch_center_mm : tuple, optional
        Custom center point in world mm coordinates. If None, uses volume center.
    use_gpu : bool, optional
        Whether to use GPU acceleration

    Returns
    -------
    np.ndarray
        New 4x4 affine matrix that maps new voxel coordinates to world RAS

    Notes
    -----
    The transformation preserves:
    - Physical center location (unless patch_center_mm is specified)
    - Orientation axes from original affine
    - RAS coordinate system compliance
    """
    # Use centralized GPU utilities
    try:
        from synthesis.gpu_utils import get_array_module
    except ImportError:
        try:
            from .gpu_utils import get_array_module
        except ImportError:
            from gpu_utils import get_array_module
    
    xp = get_array_module(prefer_gpu=use_gpu)
    use_gpu = (xp.__name__ == 'cupy') if hasattr(xp, '__name__') else False

    # Input validation
    if not isinstance(old_affine, np.ndarray) or old_affine.shape != (4, 4):
        raise ValueError(f"old_affine must be a 4x4 numpy array, got shape {old_affine.shape}")
    
    if len(old_shape) != 3 or len(new_shape) != 3:
        raise ValueError("old_shape and new_shape must be 3-element tuples (x, y, z)")
    
    if isinstance(new_voxel_size, (int, float)):
        new_voxel_size = (new_voxel_size,) * 3
    elif len(new_voxel_size) != 3:
        raise ValueError("new_voxel_size must be a scalar or 3-element tuple")

    # Ensure we work with float64 for precision in affine calculations
    old_affine_device = xp.asarray(old_affine, dtype=xp.float64)
    
    # Extract rotation/orientation matrix and compute current voxel sizes
    R_in = xp.asarray(old_affine[:3, :3], dtype=xp.float64)
    old_scales = xp.sqrt(xp.sum(R_in ** 2, axis=0))
    
    # Check for degenerate affine (zero or very small voxel sizes)
    if xp.any(old_scales < 1e-10):
        raise ValueError(f"Degenerate affine matrix: voxel sizes {old_scales} too small")
    
    # Compute scaling factors and new rotation matrix
    sf = xp.array(new_voxel_size, dtype=xp.float64) / old_scales
    S = xp.diag(sf)
    R_new = R_in @ S

    # Compute center locations in world coordinates
    if patch_center_mm is None:
        # Use geometric center of original volume
        old_center_vox = (xp.array(old_shape, dtype=xp.float64) - 1) / 2.0
        # Transform to world coordinates: [x, y, z, 1] → world_coords
        old_center_homogeneous = xp.hstack([old_center_vox, 1.0])
        old_center_mm = (old_affine_device @ old_center_homogeneous)[:3]
    else:
        # Use user-specified center
        if len(patch_center_mm) != 3:
            raise ValueError("patch_center_mm must be a 3-element coordinate")
        old_center_mm = xp.array(patch_center_mm, dtype=xp.float64)

    # Compute new volume center and translation
    new_center_vox = (xp.array(new_shape, dtype=xp.float64) - 1) / 2.0
    t_new = old_center_mm - R_new @ new_center_vox

    A_new = xp.eye(4, dtype=xp.float64)
    A_new[:3, :3] = R_new
    A_new[:3, 3] = t_new

    if hasattr(A_new, 'get'):
        return A_new.get()
    else:
        return A_new