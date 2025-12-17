import os
import numpy as np
import nibabel as nib
import pytest
from nibabel.streamlines import Tractogram, save as save_trk
from synthesis.main import process_and_save


def _create_test_inputs(tmpdir):
    """Create lightweight NIfTI and TRK inputs for the pipeline test."""
    nifti_path = tmpdir.join("test_input.nii.gz")
    trk_path = tmpdir.join("test_input.trk")

    # Small synthetic volume to keep resampling fast
    data = np.zeros((8, 8, 8), dtype=np.float32)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), str(nifti_path))

    # Minimal streamline
    streamlines = [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
    tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    save_trk(tractogram, str(trk_path))

    return str(nifti_path), str(trk_path)


def test_process_pipeline(tmpdir):
    test_nifti_path, test_trk_path = _create_test_inputs(tmpdir)
    output_prefix = str(tmpdir.join("test_output"))

    process_and_save(
        original_nifti_path=test_nifti_path,
        original_trk_path=test_trk_path,
        target_voxel_size=0.5,
        target_dimensions=(20, 20, 20),
        output_prefix=output_prefix,
        use_gpu=False,
        interpolation_method="linear"
    )

    nifti_exists = os.path.exists(output_prefix + ".nii") or os.path.exists(output_prefix + ".nii.gz")
    assert nifti_exists, "Output NIfTI file was not created"
    assert os.path.exists(output_prefix + ".trk"), "Output TRK file was not created"
