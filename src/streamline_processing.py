import numpy as np
import os
import shutil
import tempfile
from joblib import Parallel, delayed
from densify import densify_streamline_subvoxel, densify_streamlines_parallel


def clip_streamline_to_fov(stream, new_shape, use_gpu=True, epsilon=1e-6):
    """
    Clipping a streamline to the field of view.
    
    Parameters
    ----------
    stream : array-like
        The streamline to clip.
    new_shape : tuple
        Shape of the field of view.
    use_gpu : bool, optional
        Whether to use GPU, default is True.
    epsilon : float, optional
        Small tolerance value for boundary checks, by default 1e-6.
        
    Returns
    -------
    list
        List of clipped streamline segments.
    """
    # Choose appropriate array library
    if use_gpu:
        try:
            import cupy as xp
            print("Using GPU for streamline clipping")
        except ImportError:
            print("Warning: Could not import cupy. Falling back to CPU for streamline clipping.")
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp
        
    if len(stream) == 0:
        return []
        
    new_shape = xp.array(new_shape)
    
    # Add small epsilon to avoid floating point precision issues at boundaries
    # Use a more generous boundary for clipping (extend by 20% on all sides)
    boundary_expansion = 0.2  # 20% expansion
    expanded_min = xp.array([-boundary_expansion * dim for dim in new_shape])
    expanded_max = xp.array([dim + boundary_expansion * dim for dim in new_shape])
    
    # Check if points are within the expanded boundaries
    inside = xp.all((stream >= expanded_min) & (stream < expanded_max), axis=1)
    
    # Check if any points are inside to avoid unnecessary processing
    if not xp.any(inside):
        return []
        
    segments = []
    current_segment = []

    for i in range(len(stream)):
        if inside[i]:  # If point is inside FOV (with expanded boundaries)
            current_segment.append(stream[i])
        else:
            if len(current_segment) >= 2:
                # Convert to avoid reference issues and ensure float32 precision
                segments.append(xp.array(current_segment, dtype=xp.float32))
            current_segment = []

            # Check if we have a transition from inside to outside
            if i > 0 and inside[i - 1]:
                p1, p2 = stream[i - 1], stream[i]
                clipped_point = interpolate_to_fov_expanded(p1, p2, new_shape, expanded_min, expanded_max, use_gpu)
                if clipped_point is not None:
                    segments.append(xp.array([p1, clipped_point], dtype=xp.float32))
            # Check if we have a transition from outside to inside
            elif i < len(stream) - 1 and inside[i + 1]:
                p1, p2 = stream[i], stream[i + 1]
                clipped_point = interpolate_to_fov_expanded(p2, p1, new_shape, expanded_min, expanded_max, use_gpu)
                if clipped_point is not None:
                    segments.append(xp.array([clipped_point, p2], dtype=xp.float32))

    if len(current_segment) >= 2:
        segments.append(xp.array(current_segment, dtype=xp.float32))

    # Convert segments back to numpy arrays if they were created on GPU
    # This ensures consistency in return type across all code paths
    if use_gpu:
        try:
            import numpy as np
            numpy_segments = []
            for segment in segments:
                # Convert from GPU to CPU if needed
                if hasattr(xp, 'asnumpy'):  # cupy has asnumpy, numpy doesn't
                    numpy_segments.append(np.array(xp.asnumpy(segment), dtype=np.float32))
                else:
                    numpy_segments.append(np.array(segment, dtype=np.float32))
            return numpy_segments
        except Exception as e:
            print(f"Warning: Error converting segments to numpy: {e}")
            # Fall back to returning as is
            return segments
    else:
        # If we used numpy, ensure all segments are numpy arrays (not lists)
        import numpy as np
        return [np.array(segment, dtype=np.float32) if not isinstance(segment, np.ndarray) else segment 
                for segment in segments]


def interpolate_to_fov_expanded(p1, p2, new_shape, expanded_min, expanded_max, use_gpu=True):
    """
    Interpolates a point on the boundary of the expanded field of view.
    
    Parameters
    ----------
    p1 : array-like
        First point (inside FOV).
    p2 : array-like
        Second point (outside FOV).
    new_shape : array-like
        Shape of the field of view.
    expanded_min : array-like
        Minimum coordinates of expanded FOV.
    expanded_max : array-like
        Maximum coordinates of expanded FOV.
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.
        
    Returns
    -------
    array-like
        Interpolated point, or None if no valid intersection.
    """
    # Choose appropriate array library
    if use_gpu:
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp
        
    direction = p2 - p1
    t_min = xp.inf

    for dim in range(3):
        if direction[dim] != 0:
            if p2[dim] < expanded_min[dim]:
                t = (expanded_min[dim] - p1[dim]) / direction[dim]
            elif p2[dim] >= expanded_max[dim]:
                t = (expanded_max[dim] - 1 - p1[dim]) / direction[dim]
            else:
                continue

            if 0 <= t < t_min:
                t_min = t

    if t_min == xp.inf:
        return None

    return p1 + t_min * direction


def transform_streamline(s_mm, A_new_inv, use_gpu=True):
    """
    Transform a streamline from mm space to voxel space.
    
    Parameters
    ----------
    s_mm : array-like
        Streamline in mm space.
    A_new_inv : array-like
        Inverse of the affine transformation matrix.
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.
        
    Returns
    -------
    array-like
        Transformed streamline in voxel space.
    """
    # Choose appropriate array library
    if use_gpu:
        try:
            import cupy as xp
            from numba import cuda
            
            # Convert to GPU arrays
            s_mm_gpu = xp.asarray(s_mm, dtype=xp.float32)
            output = xp.zeros_like(s_mm_gpu)
            
            # GPU kernel for transformation
            @cuda.jit
            def transform_kernel(s_mm, A_new_inv, output):
                idx = cuda.grid(1)
                if idx < s_mm.shape[0]:
                    # Create homogeneous coordinates
                    hom_mm = xp.zeros(4, dtype=xp.float32)
                    hom_mm[0] = s_mm[idx, 0]
                    hom_mm[1] = s_mm[idx, 1]
                    hom_mm[2] = s_mm[idx, 2]
                    hom_mm[3] = 1.0
                    
                    # Apply transformation
                    result = xp.zeros(3, dtype=xp.float32)
                    for i in range(3):
                        for j in range(4):
                            result[i] += A_new_inv[i, j] * hom_mm[j]
                    
                    # Store result
                    output[idx, 0] = result[0]
                    output[idx, 1] = result[1]
                    output[idx, 2] = result[2]
            
            # Run kernel
            threads_per_block = 256
            blocks_per_grid = (s_mm_gpu.shape[0] + threads_per_block - 1) // threads_per_block
            transform_kernel[blocks_per_grid, threads_per_block](s_mm_gpu, A_new_inv, output)
            
            return output
            
        except ImportError:
            print("Warning: Could not import GPU libraries. Falling back to CPU for streamline transformation.")
            import numpy as xp
            use_gpu = False
    
    # CPU implementation (fallback)
    if not use_gpu:
        import numpy as xp
        
        # Convert to homogeneous coordinates
        homogeneous = xp.hstack([s_mm, xp.ones((len(s_mm), 1))])
        
        # Apply transformation
        output = xp.zeros((len(s_mm), 3), dtype=xp.float32)
        for i in range(len(s_mm)):
            output[i] = (A_new_inv @ homogeneous[i])[:3]
            
        return output


def transform_and_densify_streamlines(
    streamlines_mm, new_affine, new_shape, step_size=0.5,
    n_jobs=8, use_gpu=True, interp_method='hermite',
    disable_fov_clipping=False
):
    """
    Transform streamlines from mm space to voxel space and apply densification.

    Parameters
    ----------
    streamlines_mm : list of arrays
        List of streamlines in mm space.
    new_affine : numpy.ndarray
        Target affine transformation matrix.
    new_shape : tuple
        Target shape dimensions.
    step_size : float, optional
        Step size for densification, by default 0.5.
    n_jobs : int, optional
        Number of parallel jobs, by default 8.
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.
    interp_method : str, optional
        Interpolation method, by default 'hermite'.
    disable_fov_clipping : bool, optional
        Whether to disable FOV clipping, by default False.

    Returns
    -------
    list
        List of transformed and densified streamlines in voxel space.
    """
    if len(streamlines_mm) == 0:
        return []
    
    # Always import numpy as np for consistent reference
    import numpy as np
    
    # Import appropriate array library for computation
    if use_gpu:
        try:
            import cupy as xp
            print("Using GPU for streamline transformation")
        except ImportError:
            print("Warning: Could not import cupy. Falling back to CPU for streamline transformation.")
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp
    
    # Extract voxel size from affine
    voxel_size = np.sqrt(np.sum(new_affine[:3, :3] ** 2, axis=0))
    print(f"Voxel size from affine: {voxel_size}")
    print(f"New shape: {new_shape}")
    print(f"Step size: {step_size}")
    
    # Calculate inverse affine for transforming from mm to voxel space
    try:
        A_new_inv = np.linalg.inv(new_affine)
    except np.linalg.LinAlgError:
        print("Error: Affine matrix is not invertible. Check your affine matrix.")
        return []
    
    # Transform streamlines to voxel space
    voxel_streamlines = []
    
    # Get the min and max coordinates of all streamlines to analyze clipping
    all_points_mm = np.vstack([np.asarray(s) for s in streamlines_mm])
    min_mm = np.min(all_points_mm, axis=0)
    max_mm = np.max(all_points_mm, axis=0)
    print(f"Streamline mm-space bounds: Min={min_mm}, Max={max_mm}")
    
    # Transform these bounds to voxel space
    min_mm_homog = np.append(min_mm, 1)
    max_mm_homog = np.append(max_mm, 1)
    min_vox = A_new_inv @ min_mm_homog
    max_vox = A_new_inv @ max_mm_homog
    print(f"Transformed voxel-space bounds: Min={min_vox[:3]}, Max={max_vox[:3]}")
    print(f"Volume dimensions: {new_shape}")
    
    # Check if bounds are outside the volume
    min_outside = np.any(min_vox[:3] < -0.5) or np.any(min_vox[:3] >= np.array(new_shape) + 0.5)
    max_outside = np.any(max_vox[:3] < -0.5) or np.any(max_vox[:3] >= np.array(new_shape) + 0.5)
    if min_outside or max_outside:
        print("WARNING: Streamlines extend beyond the volume boundaries!")
        problematic_dims = []
        for i in range(3):
            if min_vox[i] < -0.5 or max_vox[i] >= new_shape[i] + 0.5:
                problematic_dims.append(f"dim {i}: [{min_vox[i]:.2f}, {max_vox[i]:.2f}] vs [0, {new_shape[i]}]")
        if problematic_dims:
            print(f"Problem dimensions: {', '.join(problematic_dims)}")
    
    # Determine whether to use FOV clipping
    if disable_fov_clipping:
        print("FOV clipping: DISABLED")
    else:
        print("FOV clipping: ENABLED")
    
    # For GPU processing, batch transform the streamlines
    if use_gpu and len(streamlines_mm) > 0:
        try:
            # Process as many streamlines as possible in parallel depending on GPU memory
            voxel_streamlines = parallel_transform_streamlines(streamlines_mm, A_new_inv, use_gpu=True)
        except Exception as e:
            print(f"GPU transformation failed with error: {e}")
            print("Falling back to CPU processing for transformation")
            use_gpu = False
    
    # For CPU processing or if GPU failed
    if not use_gpu or not voxel_streamlines:
        # Transform each streamline
        voxel_streamlines = [transform_streamline(s, A_new_inv, use_gpu=False) for s in streamlines_mm]
    
    # Clip streamlines to FOV if enabled
    if not disable_fov_clipping:
        original_count = len(voxel_streamlines)
        clipped_streamlines = []
        
        for s in voxel_streamlines:
            # Clip the streamline to the FOV
            segments = clip_streamline_to_fov(s, new_shape, use_gpu=use_gpu, epsilon=0.01)
            clipped_streamlines.extend(segments)
        
        retained_count = len(clipped_streamlines)
        print(f"Clipping Stats: {retained_count}/{original_count} streamlines retained after FOV clipping ({retained_count/original_count*100:.1f}%)")
        
        if retained_count < original_count * 0.5:
            print("WARNING: Over 50% of streamlines were clipped!")
            
        if retained_count == 0:
            print("ERROR: All streamlines were clipped!")
            # Print detailed bounds info to help debug
            if len(voxel_streamlines) > 0:
                flat_vox = np.vstack([np.asarray(s) for s in voxel_streamlines])
                min_vox_flat = np.min(flat_vox, axis=0)
                max_vox_flat = np.max(flat_vox, axis=0)
                print(f"Detailed streamline bounds: Min={min_vox_flat}, Max={max_vox_flat}")
                print(f"Expected valid bounds: Min=[0, 0, 0], Max={[d-1 for d in new_shape]}")
                
                # Check each dimension
                for i in range(3):
                    if min_vox_flat[i] < 0 or max_vox_flat[i] >= new_shape[i]:
                        print(f"Dimension {i}: Min={min_vox_flat[i]:.2f}, Max={max_vox_flat[i]:.2f}, Shape={new_shape[i]}")
                        if min_vox_flat[i] < 0:
                            print(f"  Streamlines are {-min_vox_flat[i]:.2f} voxels below boundary")
                        if max_vox_flat[i] >= new_shape[i]:
                            print(f"  Streamlines are {max_vox_flat[i] - new_shape[i] + 1:.2f} voxels above boundary")
            
        voxel_streamlines = clipped_streamlines
    
    # Densify the streamlines to ensure even spacing
    print(f"Processing streamline 0/{len(voxel_streamlines)}...")
    
    # Parallel version for densification
    try:
        densified_streamlines = densify_streamlines_parallel(
            voxel_streamlines, step_size, n_jobs=n_jobs, 
            interp_method=interp_method, use_gpu=use_gpu
        )
        print(f"Densified {len(densified_streamlines)}/{len(voxel_streamlines)} streamlines successfully")
    except Exception as e:
        print(f"Error during densification: {e}")
        print("Falling back to simple streamline conversion")
        densified_streamlines = voxel_streamlines

    print(f"Final streamline count after processing: {len(densified_streamlines)}")
    return densified_streamlines
