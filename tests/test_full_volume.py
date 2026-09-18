import os

import pytest

from preprocessing.full_volume import process_and_save

EXAMPLE_NIFTI = "examples/example_data/dtifit_S0.nii.gz"
EXAMPLE_TRK = "examples/example_data/motor.trk"


@pytest.mark.skipif(
    not (os.path.exists(EXAMPLE_NIFTI) and os.path.exists(EXAMPLE_TRK)),
    reason="example NIfTI/TRK pair not present (examples/ is not distributed with the repo)",
)
def test_process_pipeline(tmpdir):
    output_prefix = str(tmpdir.join("test_output"))

    result = process_and_save(
        original_nifti_path=EXAMPLE_NIFTI,
        original_trk_path=EXAMPLE_TRK,
        target_voxel_size=0.5,
        target_dimensions=(50, 50, 50),
        output_prefix=output_prefix,
        use_gpu=False,
    )

    # process_and_save always writes an uncompressed .nii next to the .trk
    assert result["error"] is None
    assert os.path.exists(output_prefix + ".nii"), "Output NIfTI file was not created"
    assert os.path.exists(output_prefix + ".trk"), "Output TRK file was not created"
