import numpy as np
import nibabel as nib
import os
import tempfile
from joblib import Parallel, delayed

def resample_nifti(old_img, new_affine, new_shape, chunk_size=(64, 64, 64), n_jobs=-1, use_gpu=True):
    """
    Resamples a NIfTI image to a new resolution and shape using either GPU or CPU.

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
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.

    Returns
    -------
    np.ndarray
        Resampled image data.
    str
        Path to temporary memory-mapped file.
    """
    # Choose the appropriate array library
    if use_gpu:
        try:
            import cupy as xp
            from numba import cuda
            print("Using GPU for nifti resampling")
            
            # GPU-accelerated resampling function
            @cuda.jit
            def resample_kernel(new_data, data_in, new_affine, old_affine_inv, new_shape):
                x, y, z = cuda.grid(3)
                if x < new_shape[0] and y < new_shape[1] and z < new_shape[2]:
                    out_vox = xp.array([x, y, z, 1], dtype=xp.float32)
                    xyz_mm = new_affine @ out_vox
                    xyz_old_vox = old_affine_inv @ xyz_mm
                    i, j, k = int(xyz_old_vox[0]), int(xyz_old_vox[1]), int(xyz_old_vox[2])
                    if 0 <= i < data_in.shape[0] and 0 <= j < data_in.shape[1] and 0 <= k < data_in.shape[2]:
                        new_data[x, y, z] = data_in[i, j, k]
                        
            # Convert data to GPU
            data_in = xp.asarray(old_img.get_fdata(), dtype=xp.float32)
            old_affine_inv = xp.linalg.inv(xp.asarray(old_img.affine))
            
            # Create output array on GPU
            new_data = xp.zeros(new_shape, dtype=xp.float32)
            
            # Set up CUDA grid
            threads_per_block = (8, 8, 8)
            blocks_per_grid = tuple((dim + threads_per_block[i] - 1) // threads_per_block[i] 
                                  for i, dim in enumerate(new_shape))
            
            # Execute kernel
            resample_kernel[blocks_per_grid, threads_per_block](
                new_data, data_in, xp.asarray(new_affine), old_affine_inv, new_shape
            )
            
        except ImportError:
            print("Warning: Could not import GPU libraries. Falling back to CPU for nifti resampling.")
            import numpy as xp
            use_gpu = False
    
    if not use_gpu:
        # CPU implementation
        print("Using CPU for nifti resampling")
        import numpy as xp
        
        # Get input data
        data_in = old_img.get_fdata().astype(np.float32)
        old_affine_inv = np.linalg.inv(old_img.affine)
        
        # Create output array
        new_data = np.zeros(new_shape, dtype=np.float32)
        
        # Define CPU resampling function for parallel processing
        def resample_chunk(start_x, end_x, start_y, end_y, start_z, end_z):
            chunk_data = np.zeros((end_x - start_x, end_y - start_y, end_z - start_z), dtype=np.float32)
            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    for z in range(start_z, end_z):
                        out_vox = np.array([x, y, z, 1], dtype=np.float32)
                        xyz_mm = new_affine @ out_vox
                        xyz_old_vox = old_affine_inv @ xyz_mm
                        i, j, k = int(xyz_old_vox[0]), int(xyz_old_vox[1]), int(xyz_old_vox[2])
                        if 0 <= i < data_in.shape[0] and 0 <= j < data_in.shape[1] and 0 <= k < data_in.shape[2]:
                            chunk_data[x - start_x, y - start_y, z - start_z] = data_in[i, j, k]
            return start_x, start_y, start_z, chunk_data
        
        # Process in chunks for better memory usage
        chunks = []
        for x_start in range(0, new_shape[0], chunk_size[0]):
            for y_start in range(0, new_shape[1], chunk_size[1]):
                for z_start in range(0, new_shape[2], chunk_size[2]):
                    x_end = min(x_start + chunk_size[0], new_shape[0])
                    y_end = min(y_start + chunk_size[1], new_shape[1])
                    z_end = min(z_start + chunk_size[2], new_shape[2])
                    chunks.append((x_start, x_end, y_start, y_end, z_start, z_end))
        
        # Process chunks in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(resample_chunk)(x_s, x_e, y_s, y_e, z_s, z_e) for x_s, x_e, y_s, y_e, z_s, z_e in chunks
        )
        
        # Combine results
        for x_s, y_s, z_s, chunk_data in results:
            x_e, y_e, z_e = x_s + chunk_data.shape[0], y_s + chunk_data.shape[1], z_s + chunk_data.shape[2]
            new_data[x_s:x_e, y_s:y_e, z_s:z_e] = chunk_data

    # Create memory-mapped file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".dat")
    os.close(temp_fd)
    
    # Convert back to numpy if using GPU
    if use_gpu:
        new_data_np = xp.asnumpy(new_data)
    else:
        new_data_np = new_data
    
    # Save to memmap
    new_data_memmap = np.memmap(temp_path, dtype=np.float32, mode='w+', shape=new_shape)
    new_data_memmap[:] = new_data_np[:]
    new_data_memmap.flush()
    
    return new_data, temp_path