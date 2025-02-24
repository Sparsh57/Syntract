import numpy as np
from streamline_processing import clip_streamline_to_fov, transform_and_densify_streamlines
from densify import densify_streamline_subvoxel


def test_clip_streamline_fully_inside():
    """
    Ensure that a streamline entirely inside the FOV is unchanged.
    """
    streamline = np.array([[5, 5, 5], [10, 10, 10], [15, 15, 15]])
    new_shape = (20, 20, 20)

    clipped_segments = clip_streamline_to_fov(streamline, new_shape)
    assert len(clipped_segments) == 1, "Streamline should remain inside FOV without changes"
    assert np.array_equal(clipped_segments[0], streamline), "Clipped output should match input"


def test_clip_streamline_partially_outside():
    """
    Ensure that streamlines are clipped at the boundary of the FOV.
    """
    streamline = np.array([[-5, -5, -5], [10, 10, 10], [25, 25, 25]])
    new_shape = (20, 20, 20)

    clipped_segments = clip_streamline_to_fov(streamline, new_shape)

    print(f"Original streamline:\n{streamline}")
    print(f"Clipped segments:\n{clipped_segments}")

    assert clipped_segments is not None, "Function should return a list, not None"
    assert isinstance(clipped_segments, list), "Output should be a list of segments"
    assert len(clipped_segments) > 0, "Streamline should be clipped into at least one segment"
    assert all(np.all(segment >= 0) and np.all(segment < new_shape) for segment in clipped_segments), "All points should be inside FOV"


def test_clip_streamline_completely_outside():
    """
    Ensure that a streamline entirely outside the FOV is removed.
    """
    streamline = np.array([[50, 50, 50], [60, 60, 60]])
    new_shape = (20, 20, 20)

    clipped_segments = clip_streamline_to_fov(streamline, new_shape)
    assert len(clipped_segments) == 0, "Streamline outside FOV should be removed"


def test_transform_and_densify_basic():
    """
    Test that streamlines transform correctly.
    """
    streamline = np.array([[0, 0, 0], [5, 5, 5], [10, 10, 10]])
    old_streams_mm = [streamline]

    # Identity affine should not change anything
    A_new = np.eye(4)
    new_shape = (20, 20, 20)

    densified_vox = transform_and_densify_streamlines(old_streams_mm, A_new, new_shape, step_size=0.5, n_jobs=1)

    assert len(densified_vox) > 0, "Should return at least one densified streamline"
    for segment in densified_vox:
        assert segment.shape[1] == 3, "Each segment should have (N,3) shape"
        assert np.all(segment >= 0) and np.all(segment < new_shape), "All points should be inside FOV"


def test_transform_and_densify_with_affine():
    """
    Test that affine transformation is correctly applied.
    """
    streamline = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    old_streams_mm = [streamline]

    # Scale transformation (double the size)
    A_new = np.eye(4)
    A_new[0, 0] = 2  # Scale x-axis
    A_new[1, 1] = 2  # Scale y-axis
    A_new[2, 2] = 2  # Scale z-axis
    new_shape = (50, 50, 50)

    densified_vox = transform_and_densify_streamlines(old_streams_mm, A_new, new_shape, step_size=0.5, n_jobs=1)

    assert len(densified_vox) > 0, "Transformed streamline should still exist"
    for segment in densified_vox:
        assert np.all(segment < new_shape), "All points should be inside the new shape"