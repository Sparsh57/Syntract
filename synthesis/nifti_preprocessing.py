import numpy as np
import nibabel as nib
import os
import tempfile
import gc
import math
from joblib import Parallel, delayed

def cubic_kernel(x):
    """Cubic interpolation kernel (Mitchell-Netravali)"""
    x = np.abs(x)
    if isinstance(x, np.ndarray):
        result = np.zeros_like(x)
        # First piece: 0 <= x < 1
        mask1 = x < 1
        result[mask1] = (21*x[mask1]**3 - 36*x[mask1]**2 + 16) / 18
        # Second piece: 1 <= x < 2  
        mask2 = (x >= 1) & (x < 2)
        result[mask2] = (-7*x[mask2]**3 + 36*x[mask2]**2 - 60*x[mask2] + 32) / 18
        return result
    else:
        if x < 1:
            return (21*x**3 - 36*x**2 + 16) / 18
        elif x < 2:
            return (-7*x**3 + 36*x**2 - 60*x + 32) / 18
        else:
            return 0

def interpolate_point(data_in, i, j, k):
    """High-quality cubic interpolation for a single point"""
    # Tricubic interpolation
    i_center, j_center, k_center = int(round(i)), int(round(j)), int(round(k))
    
    result = 0.0
    total_weight = 0.0
    
    for di in range(-1, 3):  # 4x4x4 neighborhood
        for dj in range(-1, 3):
            for dk in range(-1, 3):
                ii, jj, kk = i_center + di, j_center + dj, k_center + dk
                
                if (0 <= ii < data_in.shape[0] and 0 <= jj < data_in.shape[1] and 0 <= kk < data_in.shape[2]):
                    weight_i = cubic_kernel(i - ii)
                    weight_j = cubic_kernel(j - jj) 
                    weight_k = cubic_kernel(k - kk)
                    weight = weight_i * weight_j * weight_k
                    
                    result += data_in[ii, jj, kk] * weight
                    total_weight += weight
    
    return result / max(total_weight, 1e-10)

def estimate_memory_usage(shape, dtype=np.float32):
    """
    Estimate memory usage for an array with given shape and dtype.

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    dtype : numpy.dtype, optional
        Data type of the array.

    Returns
    -------
    float
        Estimated memory usage in GB.
    """
    bytes_per_element = np.dtype(dtype).itemsize
    total_elements = np.prod(shape)
    memory_gb = (total_elements * bytes_per_element) / (1024**3)
    return memory_gb

def resample_nifti(old_img, new_affine, new_shape, chunk_size=(64, 64, 64), n_jobs=-1, use_gpu=True, max_output_gb=64):
    """
    Resample a NIfTI image to a new resolution and shape using high-quality cubic interpolation.

    Parameters
    ----------
    old_img : nibabel.Nifti1Image
        Original NIfTI image.
    new_affine : np.ndarray
        New affine transformation matrix.
    new_shape : tuple
        Desired shape of the resampled volume.
    chunk_size : tuple, optional
        Processing chunk size.
    n_jobs : int, optional
        Number of CPU cores for parallel processing.
    use_gpu : bool, optional
        Whether to use GPU acceleration.
    max_output_gb : float, optional
        Maximum allowed output size in GB.

    Returns
    -------
    np.ndarray
        Resampled image data.
    str
        Path to temporary memory-mapped file.
    """
    est_memory_gb = estimate_memory_usage(new_shape)
    
    if est_memory_gb > max_output_gb:
        scale_factor = math.pow(max_output_gb / max(0.1, est_memory_gb), 1/3)
        safe_shape = tuple(int(dim * scale_factor) for dim in new_shape)
        print(f"WARNING: Reducing dimensions from {new_shape} to {safe_shape} for memory safety")
        new_shape = safe_shape
    
    if est_memory_gb > 10:
        mmap_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy').name
        output_mmap = np.lib.format.open_memmap(mmap_file, mode='w+', 
                                               dtype=np.float32, 
                                               shape=new_shape)
    else:
        mmap_file = None
        output_mmap = None
    
    if use_gpu:
        try:
            import cupy as xp
            from numba import cuda
            import math
            
            @cuda.jit(device=True)
            def cubic_kernel_device(x):
                """Cubic interpolation kernel for CUDA device"""
                x = abs(x)
                if x < 1:
                    return (21*x*x*x - 36*x*x + 16) / 18
                elif x < 2:
                    return (-7*x*x*x + 36*x*x - 60*x + 32) / 18
                else:
                    return 0.0
            
            @cuda.jit(device=True)
            def interpolate_point_device(data_in, i, j, k):
                """High-quality cubic interpolation for CUDA device"""
                i_center = int(round(i))
                j_center = int(round(j))
                k_center = int(round(k))
                
                result = 0.0
                total_weight = 0.0
                
                for di in range(-1, 3):  # 4x4x4 neighborhood
                    for dj in range(-1, 3):
                        for dk in range(-1, 3):
                            ii = i_center + di
                            jj = j_center + dj
                            kk = k_center + dk
                            
                            if (0 <= ii < data_in.shape[0] and 0 <= jj < data_in.shape[1] and 0 <= kk < data_in.shape[2]):
                                weight_i = cubic_kernel_device(i - ii)
                                weight_j = cubic_kernel_device(j - jj)
                                weight_k = cubic_kernel_device(k - kk)
                                weight = weight_i * weight_j * weight_k
                                
                                result += data_in[ii, jj, kk] * weight
                                total_weight += weight
                
                return result / max(total_weight, 1e-10)
            
            @cuda.jit
            def resample_kernel(new_data, data_in, new_affine, old_affine_inv, new_shape):
                x, y, z = cuda.grid(3)
                if x < new_shape[0] and y < new_shape[1] and z < new_shape[2]:
                    x_mm = new_affine[0, 0] * x + new_affine[0, 1] * y + new_affine[0, 2] * z + new_affine[0, 3]
                    y_mm = new_affine[1, 0] * x + new_affine[1, 1] * y + new_affine[1, 2] * z + new_affine[1, 3]
                    z_mm = new_affine[2, 0] * x + new_affine[2, 1] * y + new_affine[2, 2] * z + new_affine[2, 3]
                    
                    i = old_affine_inv[0, 0] * x_mm + old_affine_inv[0, 1] * y_mm + old_affine_inv[0, 2] * z_mm + old_affine_inv[0, 3]
                    j = old_affine_inv[1, 0] * x_mm + old_affine_inv[1, 1] * y_mm + old_affine_inv[1, 2] * z_mm + old_affine_inv[1, 3]
                    k = old_affine_inv[2, 0] * x_mm + old_affine_inv[2, 1] * y_mm + old_affine_inv[2, 2] * z_mm + old_affine_inv[2, 3]
                    
                    # Use cubic interpolation for high quality results
                    new_data[x, y, z] = interpolate_point_device(data_in, i, j, k)
            
            data_in = xp.asarray(old_img.get_fdata(), dtype=xp.float32)
            old_affine_inv = xp.linalg.inv(xp.asarray(old_img.affine))
            
            if output_mmap is not None:
                new_data = None
                max_chunk_size = min(chunk_size[0], new_shape[0]), min(chunk_size[1], new_shape[1]), min(chunk_size[2], new_shape[2])
                
                @cuda.jit
                def resample_chunk_kernel(chunk_data, data_in, new_affine, old_affine_inv, x_offset, y_offset, z_offset):
                    x, y, z = cuda.grid(3)
                    if x < chunk_data.shape[0] and y < chunk_data.shape[1] and z < chunk_data.shape[2]:
                        global_x = x + x_offset
                        global_y = y + y_offset
                        global_z = z + z_offset
                        
                        x_mm = new_affine[0, 0] * global_x + new_affine[0, 1] * global_y + new_affine[0, 2] * global_z + new_affine[0, 3]
                        y_mm = new_affine[1, 0] * global_x + new_affine[1, 1] * global_y + new_affine[1, 2] * global_z + new_affine[1, 3]
                        z_mm = new_affine[2, 0] * global_x + new_affine[2, 1] * global_y + new_affine[2, 2] * global_z + new_affine[2, 3]
                        
                        i = old_affine_inv[0, 0] * x_mm + old_affine_inv[0, 1] * y_mm + old_affine_inv[0, 2] * z_mm + old_affine_inv[0, 3]
                        j = old_affine_inv[1, 0] * x_mm + old_affine_inv[1, 1] * y_mm + old_affine_inv[1, 2] * z_mm + old_affine_inv[1, 3]
                        k = old_affine_inv[2, 0] * x_mm + old_affine_inv[2, 1] * y_mm + old_affine_inv[2, 2] * z_mm + old_affine_inv[2, 3]
                        
                        # Use cubic interpolation for high quality results
                        val = interpolate_point_device(data_in, i, j, k)
                        chunk_data[x, y, z] = val
                
                for x_start in range(0, new_shape[0], max_chunk_size[0]):
                    x_end = min(x_start + max_chunk_size[0], new_shape[0])
                    for y_start in range(0, new_shape[1], max_chunk_size[1]):
                        y_end = min(y_start + max_chunk_size[1], new_shape[1])
                        for z_start in range(0, new_shape[2], max_chunk_size[2]):
                            z_end = min(z_start + max_chunk_size[2], new_shape[2])
                            
                            chunk_shape = (x_end - x_start, y_end - y_start, z_end - z_start)
                            chunk_data = xp.zeros(chunk_shape, dtype=xp.float32)
                            
                            threads_per_block = (8, 8, 8)
                            blocks_per_grid = tuple((dim + threads_per_block[i] - 1) // threads_per_block[i] 
                                                  for i, dim in enumerate(chunk_shape))
                            
                            resample_chunk_kernel[blocks_per_grid, threads_per_block](
                                chunk_data, data_in, xp.asarray(new_affine), old_affine_inv, x_start, y_start, z_start
                            )
                            
                            output_mmap[x_start:x_end, y_start:y_end, z_start:z_end] = chunk_data.get()
                            del chunk_data
                            gc.collect()
                
                new_data = output_mmap
            else:
                new_data = xp.zeros(new_shape, dtype=xp.float32)
                
                threads_per_block = (8, 8, 8)
                blocks_per_grid = tuple((dim + threads_per_block[i] - 1) // threads_per_block[i] 
                                      for i, dim in enumerate(new_shape))
                
                resample_kernel[blocks_per_grid, threads_per_block](
                    new_data, data_in, xp.asarray(new_affine), old_affine_inv, new_shape
                )
                
                if isinstance(new_data, xp.ndarray):
                    new_data = new_data.get()
            
        except Exception as e:
            print(f"GPU processing failed: {e}. Falling back to CPU.")
            import numpy as xp
            use_gpu = False
    
    if not use_gpu:
        import numpy as xp
        
        data_in = old_img.get_fdata().astype(np.float32)
        old_affine_inv = np.linalg.inv(old_img.affine)
        
        if output_mmap is not None:
            new_data = output_mmap
        else:
            new_data = np.zeros(new_shape, dtype=np.float32)
        
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
        
        chunks = []
        for x_start in range(0, new_shape[0], chunk_size[0]):
            x_end = min(x_start + chunk_size[0], new_shape[0])
            for y_start in range(0, new_shape[1], chunk_size[1]):
                y_end = min(y_start + chunk_size[1], new_shape[1])
                for z_start in range(0, new_shape[2], chunk_size[2]):
                    z_end = min(z_start + chunk_size[2], new_shape[2])
                    chunks.append((x_start, x_end, y_start, y_end, z_start, z_end))
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(resample_chunk)(x_s, x_e, y_s, y_e, z_s, z_e) 
            for x_s, x_e, y_s, y_e, z_s, z_e in chunks
        )
        
        for chunk_data, x_s, x_e, y_s, y_e, z_s, z_e in results:
            new_data[x_s:x_e, y_s:y_e, z_s:z_e] = chunk_data
    
    if output_mmap is not None:
        output_mmap.flush()
    
    if use_gpu and hasattr(new_data, 'get'):
        new_data = new_data.get()
    
    return new_data, mmap_file


def resample_nifti_patch(patch_img, target_affine, target_shape, use_gpu=False):
    """
    Resample a small NIfTI patch to target resolution.
    
    This is an optimized version of resample_nifti specifically for small patches
    that doesn't require chunking or memory mapping.
    
    Parameters
    ----------
    patch_img : nibabel.Nifti1Image
        Input patch image
    target_affine : np.ndarray
        Target affine transformation matrix
    target_shape : tuple
        Target shape (x, y, z)
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    np.ndarray
        Resampled patch data
    """
    try:
        # Import GPU utilities if requested
        if use_gpu:
            try:
                from .gpu_utils import try_gpu_import
                gpu_result = try_gpu_import()
                xp = gpu_result['xp']
                use_gpu = gpu_result['cupy_available']
            except ImportError:
                import numpy as xp
                use_gpu = False
        else:
            import numpy as xp
    except ImportError:
        import numpy as xp
        use_gpu = False
    
    # Get input data and affine
    data_in = patch_img.get_fdata().astype(np.float32)
    old_affine_inv = np.linalg.inv(patch_img.affine)
    
    # Initialize output array
    if use_gpu:
        new_data = xp.zeros(target_shape, dtype=xp.float32)
    else:
        new_data = np.zeros(target_shape, dtype=np.float32)
    
    # Perform resampling - vectorized for better performance
    x_coords, y_coords, z_coords = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]
    coords_flat = np.vstack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel(), np.ones(x_coords.size)])
    
    # Transform to world coordinates then to source voxel coordinates
    world_coords = target_affine @ coords_flat
    old_vox_coords = old_affine_inv @ world_coords
    
    # Extract integer coordinates
    i_coords = old_vox_coords[0, :].astype(int)
    j_coords = old_vox_coords[1, :].astype(int) 
    k_coords = old_vox_coords[2, :].astype(int)
    
    # Create validity mask
    valid_mask = ((i_coords >= 0) & (i_coords < data_in.shape[0]) &
                  (j_coords >= 0) & (j_coords < data_in.shape[1]) &
                  (k_coords >= 0) & (k_coords < data_in.shape[2]))
    
    # Sample valid points
    if use_gpu:
        # Transfer data to GPU if using GPU
        data_in_gpu = xp.array(data_in)
        valid_indices = xp.array(np.where(valid_mask)[0])
        i_valid = xp.array(i_coords[valid_mask])
        j_valid = xp.array(j_coords[valid_mask])
        k_valid = xp.array(k_coords[valid_mask])
        
        # Sample using advanced indexing
        sampled_values = data_in_gpu[i_valid, j_valid, k_valid]
        
        # Place values in output array
        output_flat = xp.zeros(x_coords.size, dtype=xp.float32)
        output_flat[valid_indices] = sampled_values
        new_data = output_flat.reshape(target_shape)
    else:
        # CPU version
        output_flat = np.zeros(x_coords.size, dtype=np.float32)
        valid_indices = np.where(valid_mask)[0]
        output_flat[valid_indices] = data_in[i_coords[valid_mask], j_coords[valid_mask], k_coords[valid_mask]]
        new_data = output_flat.reshape(target_shape)
    
    # Convert back to numpy if on GPU
    if use_gpu and hasattr(new_data, 'get'):
        new_data = new_data.get()
    
    return new_data