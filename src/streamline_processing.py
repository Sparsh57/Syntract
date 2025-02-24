import cupy as cp
import os
import shutil
import tempfile
from joblib import Parallel, delayed
from numba import cuda
from densify import densify_streamline_subvoxel


def clip_streamline_to_fov(stream, new_shape):
    new_shape = cp.array(new_shape)
    inside = cp.all((stream >= 0) & (stream < new_shape), axis=1)
    segments = []
    current_segment = []

    for i in range(len(stream)):
        if inside[i]:  # If point is inside FOV
            current_segment.append(stream[i])
        else:
            if len(current_segment) >= 2:
                segments.append(cp.array(current_segment, dtype=cp.float32))
            current_segment = []

            if i > 0 and inside[i - 1]:
                p1, p2 = stream[i - 1], stream[i]
                clipped_point = interpolate_to_fov(p1, p2, new_shape)
                if clipped_point is not None:
                    segments.append(cp.array([p1, clipped_point], dtype=cp.float32))

    if len(current_segment) >= 2:
        segments.append(cp.array(current_segment, dtype=cp.float32))

    return segments


def interpolate_to_fov(p1, p2, new_shape):
    direction = p2 - p1
    t_min = cp.inf

    for dim in range(3):
        if direction[dim] != 0:
            if p2[dim] < 0:
                t = (0 - p1[dim]) / direction[dim]
            elif p2[dim] >= new_shape[dim]:
                t = (new_shape[dim] - 1 - p1[dim]) / direction[dim]
            else:
                continue

            if 0 <= t < t_min:
                t_min = t

    if t_min == cp.inf:
        return None

    return p1 + t_min * direction


def transform_and_densify_streamlines_gpu(old_streams_mm, A_new, new_shape, step_size=0.5, n_jobs=-1):
    A_new_inv = cp.linalg.inv(cp.asarray(A_new))
    temp_dir = tempfile.mkdtemp()

    @cuda.jit
    def transform_kernel(s_mm, A_new_inv, output):
        idx = cuda.grid(1)
        if idx < s_mm.shape[0]:
            hom_mm = cp.append(s_mm[idx], 1)
            output[idx] = (A_new_inv @ hom_mm)[:3]

    def _process_one_and_save(s_mm, idx):
        try:
            s_mm = cp.asarray(s_mm, dtype=cp.float32)
            output = cp.zeros_like(s_mm)
            threads_per_block = 256
            blocks_per_grid = (s_mm.shape[0] + threads_per_block - 1) // threads_per_block
            transform_kernel[blocks_per_grid, threads_per_block](s_mm, A_new_inv, output)
            densified = densify_streamline_subvoxel(output, step_size)
            segments = clip_streamline_to_fov(densified, new_shape)

            if segments:
                temp_file = os.path.join(temp_dir, f"seg_{idx}.npy")
                with open(temp_file, 'wb') as f:
                    cp.save(f, segments)
                return temp_file
        except Exception as e:
            print(f"Error processing streamline {idx}: {str(e)}")
        return None

    temp_files = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_and_save)(s_mm, idx)
        for idx, s_mm in enumerate(old_streams_mm)
    )

    densified_streams_vox = []
    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            with open(temp_file, 'rb') as f:
                segments = cp.load(f, allow_pickle=True)
                densified_streams_vox.extend(segments)
            os.remove(temp_file)

    shutil.rmtree(temp_dir, ignore_errors=True)
    return densified_streams_vox
