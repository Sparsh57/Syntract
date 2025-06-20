import os
import pytest
from synthesis.main import process_and_save

def test_process_pipeline(tmpdir):
    # Setup test inputs
    test_nifti_path = "examples/example_data/dtifit_S0.nii.gz"
    test_trk_path = "examples/example_data/motor.trk"
    output_prefix = str(tmpdir.join("test_output"))

    process_and_save(
        original_nifti_path=test_nifti_path,
        original_trk_path=test_trk_path,
        target_voxel_size=0.5,
        target_dimensions=(50, 50, 50),
        output_prefix=output_prefix
    )

    assert os.path.exists(output_prefix + ".nii.gz"), "Output NIfTI file was not created"
    assert os.path.exists(output_prefix + ".trk"), "Output TRK file was not created"