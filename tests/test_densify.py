import numpy as np
from densify import densify_streamline_subvoxel

def test_densify_single_point():
    streamline = np.array([[1.0, 2.0, 3.0]])
    result = densify_streamline_subvoxel(streamline, step_size=0.5)
    assert np.array_equal(result, streamline), "Densification should not change a single-point streamline"

def test_densify_two_points():
    streamline = np.array([[0, 0, 0], [1, 1, 1]])
    result = densify_streamline_subvoxel(streamline, step_size=0.5)
    assert len(result) > 2, "Densification should add points between two endpoints"


def test_densify_spacing():
    streamline = np.array([[0, 0, 0], [1, 0, 0]])
    result = densify_streamline_subvoxel(streamline, step_size=0.2)
    distances = np.linalg.norm(np.diff(result, axis=0), axis=1)

    # Allow a small numerical tolerance to avoid floating-point issues
    assert np.all(distances <= 0.2001), "Interpolated points should be within step size (with small tolerance)"