import numpy as np
import nibabel as nib
from synthesis.nifti_preprocessing import resample_nifti


def test_resample_nifti():
    # Create a dummy 3D array
    data = np.random.rand(10, 10, 10)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    new_affine = np.eye(4) * 2  # Scale the affine
    new_affine[3, 3] = 1  # Preserve homogeneous transformation
    new_shape = (20, 20, 20)

    resampled_data, _ = resample_nifti(img, new_affine, new_shape, n_jobs=1)

    assert resampled_data.shape == new_shape, "Resampled NIfTI should have correct dimensions"