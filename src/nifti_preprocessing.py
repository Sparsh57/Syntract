import numpy as np
import nibabel as nib
import os
import tempfile
import gc
import math
from joblib import Parallel, delayed

def estimate_memory_usage(shape, dtype=np.float32):
    """
    Estimate memory usage for an array with given shape and dtype.
    
    Parameters
    ----------
    shape : tuple
        Shape of the array.
    dtype : numpy.dtype, optional
        Data type of the array, by default np.float32.
        
    Returns
    -------
    float
        Estimated memory usage in GB.
    """
    # Calculate bytes per element based on dtype
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Calculate total elements and convert to GB
    total_elements = np.prod(shape)
    memory_gb = (total_elements * bytes_per_element) / (1024**3)
    
    return memory_gb

def resample_nifti(old_img, new_affine, new_shape, chunk_size=(64, 64, 64), n_jobs=-1, use_gpu=True, max_output_gb=64):
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
    max_output_gb : float, optional
        Maximum allowed output size in GB, by default 64 GB.

    Returns
    -------
    np.ndarray
        Resampled image data.
    str
        Path to temporary memory-mapped file.
    """
    # Check if dimensions are reasonable - prevent extreme memory usage
    est_memory_gb = estimate_memory_usage(new_shape)
    
    print(f"Estimated memory requirement: {est_memory_gb:.2f} GB for output shape {new_shape}")
    
    if est_memory_gb > max_output_gb:
        # Calculate maximum safe dimensions
        scale_factor = math.pow(max_output_gb / max(0.1, est_memory_gb), 1/3)
        safe_shape = tuple(int(dim * scale_factor) for dim in new_shape)
        
        print(f"WARNING: Requested dimensions would require {est_memory_gb:.2f} GB of memory!")
        print(f"Automatically reducing dimensions to {safe_shape} (requires ~{estimate_memory_usage(safe_shape):.2f} GB)")
        print("Consider using a smaller voxel size or manually specifying smaller dimensions")
        
        # Update the new shape to the safe shape
        new_shape = safe_shape
    
    # For extremely large dimensions, use memory mapping even with reduced dimensions
    if est_memory_gb > 10:
        print("Using memory-mapped array for large output dimensions")
        mmap_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy').name
        output_mmap = np.lib.format.open_memmap(mmap_file, mode='w+', 
                                               dtype=np.float32, 
                                               shape=new_shape)
    else:
        mmap_file = None
        output_mmap = None
    
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
                    # In Numba CUDA kernels, we can't use CuPy functions like xp.array
                    # Instead, we need to perform the operations manually
                    # Create the output voxel coordinates
                    x_mm = new_affine[0, 0] * x + new_affine[0, 1] * y + new_affine[0, 2] * z + new_affine[0, 3]
                    y_mm = new_affine[1, 0] * x + new_affine[1, 1] * y + new_affine[1, 2] * z + new_affine[1, 3]
                    z_mm = new_affine[2, 0] * x + new_affine[2, 1] * y + new_affine[2, 2] * z + new_affine[2, 3]
                    
                    # Transform to input voxel space
                    i = old_affine_inv[0, 0] * x_mm + old_affine_inv[0, 1] * y_mm + old_affine_inv[0, 2] * z_mm + old_affine_inv[0, 3]
                    j = old_affine_inv[1, 0] * x_mm + old_affine_inv[1, 1] * y_mm + old_affine_inv[1, 2] * z_mm + old_affine_inv[1, 3]
                    k = old_affine_inv[2, 0] * x_mm + old_affine_inv[2, 1] * y_mm + old_affine_inv[2, 2] * z_mm + old_affine_inv[2, 3]
                    
                    # Round to nearest integer
                    i_int = int(i)
                    j_int = int(j)
                    k_int = int(k)
                    
                    # Check if the indices are within the input volume bounds
                    if 0 <= i_int < data_in.shape[0] and 0 <= j_int < data_in.shape[1] and 0 <= k_int < data_in.shape[2]:
                        new_data[x, y, z] = data_in[i_int, j_int, k_int]
            
            # Convert data to GPU
            data_in = xp.asarray(old_img.get_fdata(), dtype=xp.float32)
            old_affine_inv = xp.linalg.inv(xp.asarray(old_img.affine))
            
            # Create output array on GPU - use mmap if available
            if output_mmap is not None:
                # Process in chunks to avoid GPU memory issues
                new_data = None
                max_chunk_size = min(chunk_size[0], new_shape[0]), min(chunk_size[1], new_shape[1]), min(chunk_size[2], new_shape[2])
                
                for x_start in range(0, new_shape[0], max_chunk_size[0]):
                    x_end = min(x_start + max_chunk_size[0], new_shape[0])
                    for y_start in range(0, new_shape[1], max_chunk_size[1]):
                        y_end = min(y_start + max_chunk_size[1], new_shape[1])
                        for z_start in range(0, new_shape[2], max_chunk_size[2]):
                            z_end = min(z_start + max_chunk_size[2], new_shape[2])
                            
                            # Process this chunk
                            chunk_shape = (x_end - x_start, y_end - y_start, z_end - z_start)
                            chunk_data = xp.zeros(chunk_shape, dtype=xp.float32)
                            
                            # Set up CUDA grid for this chunk
                            threads_per_block = (8, 8, 8)
                            blocks_per_grid = tuple((dim + threads_per_block[i] - 1) // threads_per_block[i] 
                                                  for i, dim in enumerate(chunk_shape))
                            
                            # Define offset resampling kernel for chunks
                            @cuda.jit
                            def resample_chunk_kernel(chunk_data, data_in, new_affine, old_affine_inv, x_offset, y_offset, z_offset):
                                x, y, z = cuda.grid(3)
                                if x < chunk_data.shape[0] and y < chunk_data.shape[1] and z < chunk_data.shape[2]:
                                    # Global coordinates with offset
                                    global_x = x + x_offset
                                    global_y = y + y_offset
                                    global_z = z + z_offset
                                    
                                    # Compute world coordinates
                                    x_mm = new_affine[0, 0] * global_x + new_affine[0, 1] * global_y + new_affine[0, 2] * global_z + new_affine[0, 3]
                                    y_mm = new_affine[1, 0] * global_x + new_affine[1, 1] * global_y + new_affine[1, 2] * global_z + new_affine[1, 3]
                                    z_mm = new_affine[2, 0] * global_x + new_affine[2, 1] * global_y + new_affine[2, 2] * global_z + new_affine[2, 3]
                                    
                                    # Transform to input voxel space
                                    i = old_affine_inv[0, 0] * x_mm + old_affine_inv[0, 1] * y_mm + old_affine_inv[0, 2] * z_mm + old_affine_inv[0, 3]
                                    j = old_affine_inv[1, 0] * x_mm + old_affine_inv[1, 1] * y_mm + old_affine_inv[1, 2] * z_mm + old_affine_inv[1, 3]
                                    k = old_affine_inv[2, 0] * x_mm + old_affine_inv[2, 1] * y_mm + old_affine_inv[2, 2] * z_mm + old_affine_inv[2, 3]
                                    
                                    # Round to nearest integer
                                    i_int = int(i)
                                    j_int = int(j)
                                    k_int = int(k)
                                    
                                    # Check if the indices are within the input volume bounds
                                    if 0 <= i_int < data_in.shape[0] and 0 <= j_int < data_in.shape[1] and 0 <= k_int < data_in.shape[2]:
                                        chunk_data[x, y, z] = data_in[i_int, j_int, k_int]
                            
                            # Execute kernel for this chunk
                            resample_chunk_kernel[blocks_per_grid, threads_per_block](
                                chunk_data, data_in, xp.asarray(new_affine), old_affine_inv, x_start, y_start, z_start
                            )
                            
                            # Copy chunk data to mmap
                            output_mmap[x_start:x_end, y_start:y_end, z_start:z_end] = xp.asnumpy(chunk_data)
                            
                            # Force garbage collection to free memory
                            del chunk_data
                            gc.collect()
                
                # Return the memory-mapped array
                new_data = output_mmap
            else:
                # Standard GPU processing for smaller arrays
                new_data = xp.zeros(new_shape, dtype=xp.float32)
                
                # Set up CUDA grid
                threads_per_block = (8, 8, 8)
                blocks_per_grid = tuple((dim + threads_per_block[i] - 1) // threads_per_block[i] 
                                      for i, dim in enumerate(new_shape))
                
                # Execute kernel
                resample_kernel[blocks_per_grid, threads_per_block](
                    new_data, data_in, xp.asarray(new_affine), old_affine_inv, new_shape
                )
                
                # If output is on GPU, convert back to CPU
                if isinstance(new_data, xp.ndarray):
                    new_data = xp.asnumpy(new_data)
            
        except Exception as e:
            print(f"Warning: GPU processing failed with error: {e}")
            print("Falling back to CPU for nifti resampling")
            import numpy as xp
            use_gpu = False
    
    if not use_gpu:
        # CPU implementation
        print("Using CPU for nifti resampling")
        import numpy as xp
        
        # Get input data
        data_in = old_img.get_fdata().astype(np.float32)
        old_affine_inv = np.linalg.inv(old_img.affine)
        
        # Use memory-mapped array if one was created, otherwise create output array
        if output_mmap is not None:
            new_data = output_mmap
        else:
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
            return chunk_data, start_x, end_x, start_y, end_y, start_z, end_z
        
        # Process in chunks for better memory usage
        chunks = []
        for x_start in range(0, new_shape[0], chunk_size[0]):
            x_end = min(x_start + chunk_size[0], new_shape[0])
            for y_start in range(0, new_shape[1], chunk_size[1]):
                y_end = min(y_start + chunk_size[1], new_shape[1])
                for z_start in range(0, new_shape[2], chunk_size[2]):
                    z_end = min(z_start + chunk_size[2], new_shape[2])
                    chunks.append((x_start, x_end, y_start, y_end, z_start, z_end))
        
        # Process chunks in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(resample_chunk)(x_s, x_e, y_s, y_e, z_s, z_e) 
            for x_s, x_e, y_s, y_e, z_s, z_e in chunks
        )
        
        # Combine results
        for chunk_data, x_s, x_e, y_s, y_e, z_s, z_e in results:
            new_data[x_s:x_e, y_s:y_e, z_s:z_e] = chunk_data
    
    # Ensure mmap is flushed if we used memory mapping
    if output_mmap is not None:
        output_mmap.flush()
    
    # If using GPU, convert back to CPU
    if use_gpu and isinstance(new_data, type(xp.zeros(1))):
        new_data = xp.asnumpy(new_data)
    
    # Return the processed data and the temporary file path
    return new_data, mmap_file if output_mmap is not None else tempfile.NamedTemporaryFile(delete=False).name