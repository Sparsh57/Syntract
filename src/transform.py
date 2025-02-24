import cupy as cp


def build_new_affine(old_affine, old_shape, new_voxel_size, new_shape, patch_center_mm=None):
    """
    Builds a new affine matrix for resampling while maintaining the center using GPU acceleration.

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
        Center point in mm for resampling, by default None.

    Returns
    -------
    np.ndarray
        New affine matrix.
    """
    if isinstance(new_voxel_size, (int, float)):
        new_voxel_size = (new_voxel_size,) * 3

    R_in = cp.asarray(old_affine[:3, :3])
    old_scales = cp.sqrt(cp.sum(R_in ** 2, axis=0))
    sf = cp.array(new_voxel_size) / old_scales
    S = cp.diag(sf)
    R_new = R_in @ S

    if patch_center_mm is None:
        old_center_vox = (cp.array(old_shape) - 1) / 2.0
        old_center_mm = old_affine @ cp.hstack([old_center_vox, 1])
        old_center_mm = old_center_mm[:3]
    else:
        old_center_mm = cp.array(patch_center_mm)

    new_center_vox = (cp.array(new_shape) - 1) / 2.0
    t_new = old_center_mm - R_new @ new_center_vox

    A_new = cp.eye(4, dtype=cp.float64)
    A_new[:3, :3] = R_new
    A_new[:3, 3] = t_new

    return cp.asnumpy(A_new)