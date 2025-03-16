import numpy as np
import nibabel as nib
import os
import tempfile
import gc
import math
from joblib import Parallel, delayed


def estimate_memory_usage(shape, dtype=np.float32):
    """
    Estimating memory usage for an array with the given shape and dtype.

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    dtype : numpy.dtype, optional
        Data type, by default np.float32.

    Returns
    -------
    float
        Estimated memory usage in GB.
    """
    bytes_per_element = np.dtype(dtype).itemsize
    total_elements = np.prod(shape)
    return (total_elements * bytes_per_element) / (1024 ** 3)


def resample_nifti(old_img, new_affine, new_shape, chunk_size=(64, 64, 64), n_jobs=-1, use_gpu=True, max_output_gb=64):
    """
    Resampling a NIfTI image to a new resolution and shape using either GPU or CPU.

    Parameters
    ----------
    old_img : nibabel.Nifti1Image
        Original NIfTI image.
    new_affine : np.ndarray
        New affine transformation matrix.
    new_shape : tuple
        Desired output shape.
    chunk_size : tuple, optional
        Processing chunk size, by default (64, 64, 64).
    n_jobs : int, optional
        Number of CPU cores for parallel processing, by default -1 (all cores).
    use_gpu : bool, optional
        Whether using GPU acceleration, by default True.
    max_output_gb : float, optional
        Maximum allowed output size in GB, by default 64.

    Returns
    -------
    np.ndarray
        Resampled image data.
    str
        Path to temporary memory-mapped file.
    """
    est_memory_gb = estimate_memory_usage(new_shape)
    print(f"Estimated memory: {est_memory_gb:.2f} GB for shape {new_shape}")

    if est_memory_gb > max_output_gb:
        scale_factor = math.pow(max_output_gb / max(0.1, est_memory_gb), 1 / 3)
        safe_shape = tuple(int(dim * scale_factor) for dim in new_shape)
        print(
            f"WARNING: {est_memory_gb:.2f} GB required. Reducing dimensions to {safe_shape} (~{estimate_memory_usage(safe_shape):.2f} GB).")
        new_shape = safe_shape

    if est_memory_gb > 10:
        print("Using memory mapping for large output")
        mmap_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy').name
        output_mmap = np.lib.format.open_memmap(mmap_file, mode='w+', dtype=np.float32, shape=new_shape)
    else:
        mmap_file = None
        output_mmap = None

    if use_gpu:
        try:
            import cupy as xp
            from numba import cuda
            print("Using GPU for resampling")

            @cuda.jit
            def resample_kernel(new_data, data_in, new_affine, old_affine_inv, new_shape):
                x, y, z = cuda.grid(3)
                if x < new_shape[0] and y < new_shape[1] and z < new_shape[2]:
                    x_mm = new_affine[0, 0] * x + new_affine[0, 1] * y + new_affine[0, 2] * z + new_affine[0, 3]
                    y_mm = new_affine[1, 0] * x + new_affine[1, 1] * y + new_affine[1, 2] * z + new_affine[1, 3]
                    z_mm = new_affine[2, 0] * x + new_affine[2, 1] * y + new_affine[2, 2] * z + new_affine[2, 3]
                    i = old_affine_inv[0, 0] * x_mm + old_affine_inv[0, 1] * y_mm + old_affine_inv[0, 2] * z_mm + \
                        old_affine_inv[0, 3]
                    j = old_affine_inv[1, 0] * x_mm + old_affine_inv[1, 1] * y_mm + old_affine_inv[1, 2] * z_mm + \
                        old_affine_inv[1, 3]
                    k = old_affine_inv[2, 0] * x_mm + old_affine_inv[2, 1] * y_mm + old_affine_inv[2, 2] * z_mm + \
                        old_affine_inv[2, 3]
                    i_int = int(i)
                    j_int = int(j)
                    k_int = int(k)
                    if 0 <= i_int < data_in.shape[0] and 0 <= j_int < data_in.shape[1] and 0 <= k_int < data_in.shape[
                        2]:
                        new_data[x, y, z] = data_in[i_int, j_int, k_int]

            data_in = xp.asarray(old_img.get_fdata(), dtype=xp.float32)
            old_affine_inv = xp.linalg.inv(xp.asarray(old_img.affine))

            if output_mmap is not None:
                max_chunk = (min(chunk_size[0], new_shape[0]),
                             min(chunk_size[1], new_shape[1]),
                             min(chunk_size[2], new_shape[2]))
                for x_start in range(0, new_shape[0], max_chunk[0]):
                    x_end = min(x_start + max_chunk[0], new_shape[0])
                    for y_start in range(0, new_shape[1], max_chunk[1]):
                        y_end = min(y_start + max_chunk[1], new_shape[1])
                        for z_start in range(0, new_shape[2], max_chunk[2]):
                            z_end = min(z_start + max_chunk[2], new_shape[2])
                            chunk_shape = (x_end - x_start, y_end - y_start, z_end - z_start)
                            chunk_data = xp.zeros(chunk_shape, dtype=xp.float32)

                            @cuda.jit
                            def resample_chunk_kernel(chunk_data, data_in, new_affine, old_affine_inv, x_off, y_off,
                                                      z_off):
                                x, y, z = cuda.grid(3)
                                if x < chunk_data.shape[0] and y < chunk_data.shape[1] and z < chunk_data.shape[2]:
                                    global_x = x + x_off
                                    global_y = y + y_off
                                    global_z = z + z_off
                                    x_mm = new_affine[0, 0] * global_x + new_affine[0, 1] * global_y + new_affine[
                                        0, 2] * global_z + new_affine[0, 3]
                                    y_mm = new_affine[1, 0] * global_x + new_affine[1, 1] * global_y + new_affine[
                                        1, 2] * global_z + new_affine[1, 3]
                                    z_mm = new_affine[2, 0] * global_x + new_affine[2, 1] * global_y + new_affine[
                                        2, 2] * global_z + new_affine[2, 3]
                                    i = old_affine_inv[0, 0] * x_mm + old_affine_inv[0, 1] * y_mm + old_affine_inv[
                                        0, 2] * z_mm + old_affine_inv[0, 3]
                                    j = old_affine_inv[1, 0] * x_mm + old_affine_inv[1, 1] * y_mm + old_affine_inv[
                                        1, 2] * z_mm + old_affine_inv[1, 3]
                                    k = old_affine_inv[2, 0] * x_mm + old_affine_inv[2, 1] * y_mm + old_affine_inv[
                                        2, 2] * z_mm + old_affine_inv[2, 3]
                                    i_int = int(i)
                                    j_int = int(j)
                                    k_int = int(k)
                                    if 0 <= i_int < data_in.shape[0] and 0 <= j_int < data_in.shape[1] and 0 <= k_int < \
                                            data_in.shape[2]:
                                        chunk_data[x, y, z] = data_in[i_int, j_int, k_int]

                            threads = (8, 8, 8)
                            blocks = tuple((dim + threads[i] - 1) // threads[i] for i, dim in enumerate(chunk_shape))
                            resample_chunk_kernel[blocks, threads](chunk_data, data_in, xp.asarray(new_affine),
                                                                   old_affine_inv, x_start, y_start, z_start)
                            output_mmap[x_start:x_end, y_start:y_end, z_start:z_end] = xp.asnumpy(chunk_data)
                            del chunk_data
                            gc.collect()
                new_data = output_mmap
            else:
                new_data = xp.zeros(new_shape, dtype=xp.float32)
                threads = (8, 8, 8)
                blocks = tuple((dim + threads[i] - 1) // threads[i] for i, dim in enumerate(new_shape))
                resample_kernel[blocks, threads](new_data, data_in, xp.asarray(new_affine), old_affine_inv, new_shape)
                new_data = xp.asnumpy(new_data)
        except Exception as e:
            print(f"GPU processing failed: {e}")
            print("Falling back to CPU")
            import numpy as xp
            use_gpu = False

    if not use_gpu:
        print("Using CPU for resampling")
        data_in = old_img.get_fdata().astype(np.float32)
        old_affine_inv = np.linalg.inv(old_img.affine)
        new_data = output_mmap if output_mmap is not None else np.zeros(new_shape, dtype=np.float32)

        def resample_chunk(sx, ex, sy, ey, sz, ez):
            chunk = np.zeros((ex - sx, ey - sy, ez - sz), dtype=np.float32)
            for x in range(sx, ex):
                for y in range(sy, ey):
                    for z in range(sz, ez):
                        out_vox = np.array([x, y, z, 1], dtype=np.float32)
                        xyz_mm = new_affine @ out_vox
                        xyz_old = old_affine_inv @ xyz_mm
                        i, j, k = int(xyz_old[0]), int(xyz_old[1]), int(xyz_old[2])
                        if 0 <= i < data_in.shape[0] and 0 <= j < data_in.shape[1] and 0 <= k < data_in.shape[2]:
                            chunk[x - sx, y - sy, z - sz] = data_in[i, j, k]
            return chunk, sx, ex, sy, ey, sz, ez

        chunks = []
        for x_start in range(0, new_shape[0], chunk_size[0]):
            x_end = min(x_start + chunk_size[0], new_shape[0])
            for y_start in range(0, new_shape[1], chunk_size[1]):
                y_end = min(y_start + chunk_size[1], new_shape[1])
                for z_start in range(0, new_shape[2], chunk_size[2]):
                    z_end = min(z_start + chunk_size[2], new_shape[2])
                    chunks.append((x_start, x_end, y_start, y_end, z_start, z_end))

        results = Parallel(n_jobs=n_jobs)(
            delayed(resample_chunk)(sx, ex, sy, ey, sz, ez) for sx, ex, sy, ey, sz, ez in chunks
        )

        for chunk_data, sx, ex, sy, ey, sz, ez in results:
            new_data[sx:ex, sy:ey, sz:ez] = chunk_data

    if output_mmap is not None:
        output_mmap.flush()

    if use_gpu and isinstance(new_data, type(xp.zeros(1))):
        new_data = xp.asnumpy(new_data)

    return new_data, mmap_file if output_mmap is not None else tempfile.NamedTemporaryFile(delete=False).name