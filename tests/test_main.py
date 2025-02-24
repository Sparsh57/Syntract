import os
import pytest
from main import process_and_save

@pytest.mark.slow
def test_process_pipeline(tmpdir):
    # Setup test inputs
    test_nifti_path = "examples/example_data/dtifit_S0.nii.gz"
    test_trk_path = "examples/example_data/motor.trk"
    output_prefix = str(tmpdir.join("test_output"))

    process_and_save(
        old_nifti_path=test_nifti_path,
        old_trk_path=test_trk_path,
        new_voxel_size=0.5,
        new_dim=(50, 50, 50),
        output_prefix=output_prefix,
        n_jobs=1
    )

    assert os.path.exists(output_prefix + ".nii.gz"), "Output NIfTI file was not created"
    assert os.path.exists(output_prefix + ".trk"), "Output TRK file was not created"