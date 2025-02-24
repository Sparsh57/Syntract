import cupy as cp
import nibabel as nib
import os
import tempfile
from numba import cuda
import numpy as np
from joblib import Parallel, delayed

def resample_nifti_gpu(old_img, new_affine, new_shape, chunk_size=(64, 64, 64), n_jobs=-1):
    """
    Resamples a NIfTI image to a new resolution and shape using GPU acceleration.

    Parameters
    ----------
    old_img : nibabel.Nifti1Image
        Original NIfTI image.
    new_affine : np.ndarray
        New affine transformation matrix.
    new_shape : tuple
        Desired shape of the resampled volume.
    chunk_size : tuple, optional
        Processing chunk size, by default (64, 64, 64).
    n_jobs : int, optional
        Number of CPU cores for parallel processing, by default -1 (all cores).

    Returns
    -------
    np.memmap
        Resampled image data.
    """
    data_in = cp.asarray(old_img.get_fdata(), dtype=cp.float32)
    old_affine_inv = cp.linalg.inv(cp.asarray(old_img.affine))
    temp_fd, temp_path = tempfile.mkstemp(suffix=".dat")
    os.close(temp_fd)
    new_data = cp.zeros(new_shape, dtype=cp.float32)

    @cuda.jit
    def resample_kernel(new_data, data_in, new_affine, old_affine_inv, new_shape):
        x, y, z = cuda.grid(3)
        if x < new_shape[0] and y < new_shape[1] and z < new_shape[2]:
            out_vox = cp.array([x, y, z, 1], dtype=cp.float32)
            xyz_mm = new_affine @ out_vox
            xyz_old_vox = old_affine_inv @ xyz_mm
            i, j, k = int(xyz_old_vox[0]), int(xyz_old_vox[1]), int(xyz_old_vox[2])
            if 0 <= i < data_in.shape[0] and 0 <= j < data_in.shape[1] and 0 <= k < data_in.shape[2]:
                new_data[x, y, z] = data_in[i, j, k]

    threads_per_block = (8, 8, 8)
    blocks_per_grid = tuple((dim + threads_per_block[i] - 1) // threads_per_block[i] for i, dim in enumerate(new_shape))
    resample_kernel[blocks_per_grid, threads_per_block](new_data, data_in, cp.asarray(new_affine), old_affine_inv, new_shape)

    new_data = cp.asnumpy(new_data)
    new_data_memmap = np.memmap(temp_path, dtype=np.float32, mode='w+', shape=new_shape)
    new_data_memmap[:] = new_data[:]
    new_data_memmap.flush()
    return new_data_memmap, temp_path