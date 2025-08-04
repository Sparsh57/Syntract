import numpy as np


def build_new_affine(old_affine, old_shape, new_voxel_size, new_shape, patch_center_mm=None, use_gpu=True):
    """
    Build a new affine matrix for resampling while keeping the center.

    Parameters
    ----------
    old_affine : np.ndarray
        Original affine matrix.
    old_shape : tuple
        Shape of the original volume.
    new_voxel_size : float or tuple
        Desired voxel size.
    new_shape : tuple
        New volume shape.
    patch_center_mm : tuple, optional
        Center point in mm for resampling.
    use_gpu : bool, optional
        Whether to use GPU.

    Returns
    -------
    np.ndarray
        New affine matrix.
    """
    if use_gpu:
        try:
            from .gpu_utils import get_array_module, has_gpu_support
            xp = get_array_module(prefer_gpu=True)
            if not has_gpu_support():
                import numpy as xp
                use_gpu = False
            xp.array([1, 2, 3])
        except (ImportError, RuntimeError):
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp

    if isinstance(new_voxel_size, (int, float)):
        new_voxel_size = (new_voxel_size,) * 3

    old_affine_device = xp.asarray(old_affine)
    
    R_in = xp.asarray(old_affine[:3, :3])
    old_scales = xp.sqrt(xp.sum(R_in ** 2, axis=0))
    sf = xp.array(new_voxel_size) / old_scales
    S = xp.diag(sf)
    R_new = R_in @ S

    if patch_center_mm is None:
        old_center_vox = (xp.array(old_shape) - 1) / 2.0
        old_center_mm = old_affine_device @ xp.hstack([old_center_vox, 1])
        old_center_mm = old_center_mm[:3]
    else:
        old_center_mm = xp.array(patch_center_mm)

    new_center_vox = (xp.array(new_shape) - 1) / 2.0
    t_new = old_center_mm - R_new @ new_center_vox

    A_new = xp.eye(4, dtype=xp.float64)
    A_new[:3, :3] = R_new
    A_new[:3, 3] = t_new

    if use_gpu:
        return xp.asnumpy(A_new)
    else:
        return A_new