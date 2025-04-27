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
    inside = xp.all((stream >= -epsilon) & (stream < new_shape + epsilon), axis=1)
    
    # Check if any points are inside to avoid unnecessary processing
    if not xp.any(inside):
        return []
        
    segments = []
    current_segment = []

    for i in range(len(stream)):
        if inside[i]:  # If point is inside FOV
            current_segment.append(stream[i])
        else:
            if len(current_segment) >= 2:
                # Convert to avoid reference issues and ensure float32 precision
                segments.append(xp.array(current_segment, dtype=xp.float32))
            current_segment = []

            # Check if we have a transition from inside to outside
            if i > 0 and inside[i - 1]:
                p1, p2 = stream[i - 1], stream[i]
                clipped_point = interpolate_to_fov(p1, p2, new_shape, use_gpu)
                if clipped_point is not None:
                    segments.append(xp.array([p1, clipped_point], dtype=xp.float32))
            # Check if we have a transition from outside to inside
            elif i < len(stream) - 1 and inside[i + 1]:
                p1, p2 = stream[i], stream[i + 1]
                clipped_point = interpolate_to_fov(p2, p1, new_shape, use_gpu)
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


def interpolate_to_fov(p1, p2, new_shape, use_gpu=True):
    """
    Interpolates a point on the boundary of the field of view.
    
    Parameters
    ----------
    p1 : array-like
        First point (inside FOV).
    p2 : array-like
        Second point (outside FOV).
    new_shape : array-like
        Shape of the field of view.
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
            if p2[dim] < 0:
                t = (0 - p1[dim]) / direction[dim]
            elif p2[dim] >= new_shape[dim]:
                t = (new_shape[dim] - 1 - p1[dim]) / direction[dim]
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
    n_jobs=8, use_gpu=True, interp_method='hermite'
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
        Interpolation method ('hermite' or 'linear'), by default 'hermite'.

    Returns
    -------
    list of arrays
        Densified streamlines in voxel space.
    """
    if not isinstance(streamlines_mm, list):
        streamlines_mm = list(streamlines_mm)
    
    # Handle empty input
    if not streamlines_mm:
        return []

    inv_A = np.linalg.inv(new_affine)
    
    # Get the scaling factor from the affine
    voxel_size = np.sqrt(np.sum(new_affine[:3, :3] ** 2, axis=0))
    
    # Diagnostic info about voxel scaling
    print(f"\n[STREAMLINE DEBUG] Transform diagnostics:")
    print(f"Voxel size from affine: {voxel_size}")
    print(f"New shape: {new_shape}")
    print(f"Step size: {step_size}")
    print(f"FOV clipping: ENABLED")
        
    # Transform from mm space to voxel space
    transformed_streams = []
    for s in streamlines_mm:
        # Create homogeneous coordinates
        h = np.hstack((s, np.ones((len(s), 1))))
        # Transform to voxel space using the inverse of the affine matrix
        s_vox = h @ inv_A.T
        # Take only the first 3 columns (x, y, z coordinates)
        transformed_streams.append(s_vox[:, :3])
    
    # Counts for clipping diagnostics
    total_streamlines = len(transformed_streams)
    
    # Apply clipping (always enabled)
    # Clip streamlines to the field of view (new_shape)
    clipped_streams = []
    for s in transformed_streams:
        # For 3D volumes, check if streamline points are within the volume
        mask = np.all((s >= 0) & (s < np.array(new_shape)), axis=1)
        if np.any(mask):
            # Create a new streamline with just the points inside the volume
            # WARNING: This approach creates list-type streamlines that cause problems downstream
            # Instead, we'll directly use numpy indexing to keep array-type streamlines
            if np.sum(mask) >= 2:  # Only keep if 2+ points
                # This keeps s as a numpy array and avoids creating a list
                filtered_array = s[mask]
                clipped_streams.append(filtered_array)
                
    # Counts after clipping
    clipped_count = len(clipped_streams)
    
    # Show clipping stats
    print(f"Clipping Stats: {clipped_count}/{total_streamlines} streamlines retained after FOV clipping ({clipped_count/total_streamlines*100:.1f}%)")
    if clipped_count < total_streamlines * 0.5:
        print(f"WARNING: Over 50% of streamlines were clipped!")
    if clipped_count == 0:
        print("ERROR: All streamlines were clipped!")
        return []
    
    # Ensure all streamlines are proper numpy arrays, not Python lists or other types
    # This is CRITICAL for GPU processing with CuPy
    clean_streams = []
    for i, s in enumerate(clipped_streams):
        try:
            # First handle different types
            if hasattr(s, 'get'):  # Check if it's a CuPy array
                s = s.get()
            
            if isinstance(s, list):
                # Deep validation for list-type streamlines
                if not all(isinstance(p, (list, tuple, np.ndarray)) for p in s):
                    print(f"Warning: Streamline {i} contains invalid point types - filtering")
                    # Filter to only valid points
                    valid_points = []
                    for p in s:
                        if isinstance(p, (list, tuple)) and len(p) == 3:
                            valid_points.append(np.array(p, dtype=np.float32))
                        elif isinstance(p, np.ndarray) and p.shape[-1] == 3:
                            valid_points.append(p.astype(np.float32))
                    
                    if len(valid_points) < 2:
                        print(f"Warning: Removing streamline {i} - insufficient valid points")
                        continue  # Skip this streamline
                    
                    # Create clean numpy array from valid points
                    s = np.array(valid_points, dtype=np.float32)
                else:
                    # All points are valid, simple conversion
                    s = np.array(s, dtype=np.float32)
            elif not isinstance(s, np.ndarray):
                # Try to convert unknown types to numpy
                try:
                    s = np.array(s, dtype=np.float32)
                except Exception as e:
                    print(f"Error: Could not convert streamline {i} to numpy array: {e}")
                    continue  # Skip this streamline
            
            # Final validation
            if not isinstance(s, np.ndarray):
                print(f"Error: Streamline {i} is not a numpy array after conversion")
                continue
            
            if len(s) < 2:
                print(f"Warning: Streamline {i} has too few points ({len(s)}) - skipping")
                continue
                
            if s.dtype != np.float32:
                s = s.astype(np.float32)
                
            # Add to clean list
            clean_streams.append(s)
        except Exception as e:
            print(f"Error processing streamline {i}: {e}")
            continue
    
    # Update the streamline list with only the clean streamlines
    if len(clean_streams) == 0:
        print("ERROR: No valid streamlines after cleaning! Check your input data.")
        return []
        
    print(f"Cleaned streamlines: {len(clean_streams)}/{len(clipped_streams)} retained after validation")
    clipped_streams = clean_streams
            
    # Perform a final check before densification
    print(f"Preparing for densification: {len(clipped_streams)} streamlines")
    for i, s in enumerate(clipped_streams[:5]):  # Show sample of first 5 streamlines
        print(f"Sample streamline {i}: type={type(s)}, shape={s.shape if hasattr(s, 'shape') else 'unknown'}, dtype={s.dtype if hasattr(s, 'dtype') else 'unknown'}")
        
    # Apply densification in voxel space
    min_voxel_size = min(voxel_size)
    densified_streams = densify_streamlines_parallel(
        clipped_streams, step_size, n_jobs=n_jobs, use_gpu=use_gpu,
        interp_method=interp_method, voxel_size=min_voxel_size
    )
    
    # Report final streamline count
    print(f"Final streamline count after processing: {len(densified_streams)}")
    if len(densified_streams) == 0:
        print("WARNING: No streamlines were processed! Check your parameters.")
        print("Try a larger voxel size or adjust the step size.")
    
    return densified_streams
