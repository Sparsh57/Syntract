import cupy as cp
def build_new_affine(old_affine, old_shape, new_voxel_size, new_shape, patch_center_mm=None):
    if isinstance(new_voxel_size, (int, float)):
        new_voxel_size = (new_voxel_size,) * 3

    # Convert old_affine to a CuPy array
    old_affine_cp = cp.asarray(old_affine)
    R_in = old_affine_cp[:3, :3]
    old_scales = cp.sqrt(cp.sum(R_in ** 2, axis=0))
    sf = cp.array(new_voxel_size) / old_scales
    S = cp.diag(sf)
    R_new = R_in @ S

    if patch_center_mm is None:
        old_center_vox = (cp.array(old_shape) - 1) / 2.0
        # Ensure the homogeneous coordinate is a CuPy array
        old_center_hom = cp.hstack([old_center_vox, cp.array([1], dtype=old_center_vox.dtype)])
        old_center_mm = old_affine_cp @ old_center_hom
        old_center_mm = old_center_mm[:3]
    else:
        old_center_mm = cp.array(patch_center_mm)

    new_center_vox = (cp.array(new_shape) - 1) / 2.0
    t_new = old_center_mm - R_new @ new_center_vox

    A_new = cp.eye(4, dtype=cp.float64)
    A_new[:3, :3] = R_new
    A_new[:3, 3] = t_new

    return cp.asnumpy(A_new)

