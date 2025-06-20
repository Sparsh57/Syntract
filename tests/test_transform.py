import numpy as np
from synthesis.transform import build_new_affine

def test_build_new_affine():
    old_affine = np.eye(4)
    old_shape = (100, 100, 100)
    new_voxel_size = (2.0, 2.0, 2.0)
    new_shape = (50, 50, 50)

    new_affine = build_new_affine(old_affine, old_shape, new_voxel_size, new_shape)

    assert new_affine.shape == (4, 4), "Affine matrix should be 4x4"
    assert np.allclose(new_affine[:3, :3], np.diag([2.0, 2.0, 2.0])), "Scaling should match new voxel size"