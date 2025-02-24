import numpy as np
import os
import shutil
import tempfile
from joblib import Parallel, delayed
from densify import densify_streamline_subvoxel


def clip_streamline_to_fov(stream, new_shape):
    """
    Clips a streamline to ensure it remains within the new field of view (FOV).

    Parameters
    ----------
    stream : np.ndarray
        (N, 3) Streamline coordinates.
    new_shape : tuple
        Dimensions of the new field of view.

    Returns
    -------
    list of np.ndarray
        List of clipped streamline segments.
    """
    new_shape = np.array(new_shape)
    inside = np.all((stream >= 0) & (stream < new_shape), axis=1)

    segments = []
    current_segment = []

    for i in range(len(stream)):
        if inside[i]:  # If point is inside FOV
            current_segment.append(stream[i])
        else:  # If point is outside
            if len(current_segment) >= 2:
                segments.append(np.array(current_segment, dtype=np.float32))
            current_segment = []

            # Check if interpolation is needed
            if i > 0 and inside[i - 1]:  # Previous point was inside, current is outside
                p1, p2 = stream[i - 1], stream[i]
                clipped_point = interpolate_to_fov(p1, p2, new_shape)
                if clipped_point is not None:
                    segments.append(np.array([p1, clipped_point], dtype=np.float32))

    if len(current_segment) >= 2:
        segments.append(np.array(current_segment, dtype=np.float32))

    return segments


def interpolate_to_fov(p1, p2, new_shape):
    """
    Interpolates a point on the streamline to the FOV boundary.

    Parameters
    ----------
    p1 : np.ndarray
        Inside-FOV point.
    p2 : np.ndarray
        Outside-FOV point.
    new_shape : tuple
        Dimensions of the FOV.

    Returns
    -------
    np.ndarray or None
        Interpolated boundary point or None if interpolation fails.
    """
    direction = p2 - p1
    t_min = np.inf

    for dim in range(3):
        if direction[dim] != 0:
            if p2[dim] < 0:
                t = (0 - p1[dim]) / direction[dim]
            elif p2[dim] >= new_shape[dim]:
                t = (new_shape[dim] - 1 - p1[dim]) / direction[dim]
            else:
                continue  # No crossing in this dimension

            if 0 <= t < t_min:
                t_min = t

    if t_min == np.inf:
        return None  # No valid intersection

    return p1 + t_min * direction

def transform_and_densify_streamlines(old_streams_mm, A_new, new_shape, step_size=0.5, n_jobs=-1):
    """
    Transforms, densifies, and clips streamlines in parallel.

    Parameters
    ----------
    old_streams_mm : list of np.ndarray
        List of streamlines in mm coordinates.
    A_new : np.ndarray
        New affine transformation matrix.
    new_shape : tuple
        Dimensions of the new field of view.
    step_size : float, optional
        Step size for densification, by default 0.5.
    n_jobs : int, optional
        Number of CPU cores for parallel processing, by default -1 (all CPUs).

    Returns
    -------
    list of np.ndarray
        Transformed, densified, and clipped streamlines in voxel coordinates.
    """
    A_new_inv = np.linalg.inv(A_new)
    temp_dir = tempfile.mkdtemp()

    def _process_one_and_save(s_mm, idx):
        try:
            # Ensure streamline is a numpy array with float32 type.
            s_mm = np.asarray(s_mm, dtype=np.float32)

            # Handle cases where s_mm is 1D (e.g., a single point).
            if s_mm.ndim == 1:
                s_mm = s_mm.reshape(1, -1)

            # Ensure shape (N, 3) format.
            if s_mm.shape[1] != 3:
                s_mm = s_mm.T

            # Apply transformation to voxel space.
            hom_mm = np.hstack([s_mm, np.ones((len(s_mm), 1), dtype=np.float32)])
            s_new_vox = (A_new_inv @ hom_mm.T).T[:, :3]

            # Densify streamline
            densified = densify_streamline_subvoxel(s_new_vox, step_size)

            # Clip to new FOV
            segments = clip_streamline_to_fov(densified, new_shape)

            # Save valid segments to temporary file
            if segments:
                temp_file = os.path.join(temp_dir, f"seg_{idx}.npy")
                with open(temp_file, 'wb') as f:
                    np.save(f, segments)
                return temp_file
        except Exception as e:
            print(f"Error processing streamline {idx}: {str(e)}")
        return None

    # Process all streamlines in parallel
    temp_files = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_and_save)(s_mm, idx)
        for idx, s_mm in enumerate(old_streams_mm)
    )

    # Load processed segments from temporary files
    densified_streams_vox = []
    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            with open(temp_file, 'rb') as f:
                segments = np.load(f, allow_pickle=True)
                densified_streams_vox.extend(segments)
            os.remove(temp_file)

    shutil.rmtree(temp_dir, ignore_errors=True)

    return densified_streams_vox