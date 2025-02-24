import numpy as np
import nibabel as nib
import os
import tempfile
from scipy.ndimage import map_coordinates
from joblib import Parallel, delayed

def resample_nifti(old_img, new_affine, new_shape, chunk_size=(64, 64, 64), n_jobs=-1):
    """
    Resamples a NIfTI image to a new resolution and shape in parallel.

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
    data_in = np.asarray(old_img.dataobj, dtype=np.float32)
    old_affine_inv = np.linalg.inv(old_img.affine)
    temp_fd, temp_path = tempfile.mkstemp(suffix=".dat")
    os.close(temp_fd)
    new_data = np.memmap(temp_path, dtype=np.float32, mode='w+', shape=new_shape)

    def _resample_chunk(i0, i1, j0, j1, k0, k1):
        I, J, K = np.mgrid[i0:i1, j0:j1, k0:k1]
        out_vox = np.vstack([I.ravel(), J.ravel(), K.ravel(), np.ones(I.size)])
        xyz_mm = new_affine @ out_vox
        xyz_old_vox = old_affine_inv @ xyz_mm
        chunk_values = map_coordinates(data_in, xyz_old_vox[:3], order=1, mode='nearest')
        return (i0, i1, j0, j1, k0, k1, chunk_values)

    chunk_list = [
        (i0, min(i0 + chunk_size[0], new_shape[0]),
         j0, min(j0 + chunk_size[1], new_shape[1]),
         k0, min(k0 + chunk_size[2], new_shape[2]))
        for i0 in range(0, new_shape[0], chunk_size[0])
        for j0 in range(0, new_shape[1], chunk_size[1])
        for k0 in range(0, new_shape[2], chunk_size[2])
    ]

    results = Parallel(n_jobs=n_jobs)(delayed(_resample_chunk)(*chunk) for chunk in chunk_list)

    for (i0, i1, j0, j1, k0, k1, chunk_values) in results:
        new_data[i0:i1, j0:j1, k0:k1] = chunk_values.reshape((i1 - i0, j1 - j0, k1 - k0))

    new_data.flush()
    return new_data, temp_path