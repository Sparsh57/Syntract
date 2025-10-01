import numpy as np
import os
import shutil
import tempfile
from joblib import Parallel, delayed
try:
    from .densify import densify_streamline_subvoxel, densify_streamlines_parallel
except ImportError:
    from densify import densify_streamline_subvoxel, densify_streamlines_parallel

def clip_streamline_to_fov(stream, new_shape, use_gpu=True, epsilon=1e-6):
    """Clip a streamline to the field of view."""
    import numpy as np
    
    # Ensure stream is a numpy array first
    if not isinstance(stream, np.ndarray):
        try:
            stream = np.array(stream, dtype=np.float32)
        except Exception as e:
            raise TypeError(f"Cannot convert stream of type {type(stream)} to numpy array: {e}")
    
    # Ensure stream is float32
    if stream.dtype != np.float32:
        stream = stream.astype(np.float32)
    
    # Auto-detect and setup array module
    xp = np
    use_cupy = False
    
    if use_gpu:
        try:
            import cupy as cp
            # Test CuPy availability and functionality
            try:
                test_array = cp.array([1.0], dtype=cp.float32)
                _ = cp.sum(test_array)  # Simple operation test
                xp = cp
                use_cupy = True
            except Exception as e:
                xp = np
                use_cupy = False
        except ImportError:
            xp = np
            use_cupy = False
    else:
        xp = np
        use_cupy = False
    
    # Convert data to appropriate array type
    if use_cupy:
        try:
            stream_device = xp.asarray(stream)
            new_shape_device = xp.array(new_shape)
        except Exception as e:
            xp = np
            use_cupy = False
            stream_device = stream
            new_shape_device = np.array(new_shape)
    else:
        stream_device = stream
        new_shape_device = np.array(new_shape)
        
    if len(stream_device) == 0:
        return []
        
    # GPU-accelerated inside/outside computation
    inside = xp.all((stream_device >= -epsilon) & (stream_device < new_shape_device + epsilon), axis=1)
    
    if not xp.any(inside):
        return []
        
    segments = []
    current_segment = []

    for i in range(len(stream_device)):
        if inside[i]:
            # Convert to CPU for list operations to avoid CuPy/NumPy mixing issues
            if use_cupy:
                point = stream_device[i].get()  # Convert CuPy to NumPy
            else:
                point = np.array(stream_device[i])  # Ensure NumPy array
            current_segment.append(point)
        else:
            if len(current_segment) >= 2:
                # Create numpy array from the accumulated points
                segment_array = np.array(current_segment, dtype=np.float32)
                segments.append(segment_array)
            current_segment = []

            if i > 0 and inside[i - 1]:
                p1 = stream_device[i - 1]
                p2 = stream_device[i]
                
                # Convert to numpy for interpolation to maintain consistency
                if use_cupy:
                    p1_np = p1.get()
                    p2_np = p2.get()
                    new_shape_np = new_shape_device.get()
                else:
                    p1_np = np.array(p1)
                    p2_np = np.array(p2)
                    new_shape_np = new_shape_device
                    
                clipped_point = interpolate_to_fov(p1_np, p2_np, new_shape_np, use_gpu=use_cupy)
                if clipped_point is not None:
                    segment_array = np.array([p1_np, clipped_point], dtype=np.float32)
                    segments.append(segment_array)
                    
            elif i < len(stream_device) - 1 and inside[i + 1]:
                p1 = stream_device[i]
                p2 = stream_device[i + 1]
                
                # Convert to numpy for interpolation to maintain consistency
                if use_cupy:
                    p1_np = p1.get()
                    p2_np = p2.get()
                    new_shape_np = new_shape_device.get()
                else:
                    p1_np = np.array(p1)
                    p2_np = np.array(p2)
                    new_shape_np = new_shape_device
                    
                clipped_point = interpolate_to_fov(p2_np, p1_np, new_shape_np, use_gpu=use_cupy)
                if clipped_point is not None:
                    segment_array = np.array([clipped_point, p2_np], dtype=np.float32)
                    segments.append(segment_array)

    if len(current_segment) >= 2:
        # Create numpy array from the accumulated points
        segment_array = np.array(current_segment, dtype=np.float32)
        segments.append(segment_array)

    # Ensure all segments are numpy arrays with float32 dtype
    numpy_segments = []
    for segment in segments:
        if isinstance(segment, np.ndarray):
            numpy_segments.append(segment.astype(np.float32))
        else:
            numpy_segments.append(np.array(segment, dtype=np.float32))
    
    return numpy_segments


def interpolate_to_fov(p1, p2, new_shape, use_gpu=True):
    """Interpolate a point on the boundary of the field of view."""
    import numpy as np
    
    # Auto-detect array module
    xp = np
    if use_gpu:
        try:
            import cupy as cp
            # Test if CuPy is working
            try:
                test_array = cp.array([1.0])
                _ = cp.sum(test_array)
                xp = cp
            except Exception:
                xp = np
        except ImportError:
            xp = np
    
    # Convert inputs to appropriate array type
    p1 = xp.asarray(p1, dtype=xp.float32)
    p2 = xp.asarray(p2, dtype=xp.float32)
    new_shape = xp.asarray(new_shape)
        
    direction = p2 - p1
    t_min = float('inf')

    for dim in range(3):
        if abs(direction[dim]) > 1e-10:
            if p2[dim] < 0:
                t = (0 - p1[dim]) / direction[dim]
                if 0 <= t <= 1 and t < t_min:
                    t_min = t
            elif p2[dim] >= new_shape[dim]:
                t = (new_shape[dim] - 1 - p1[dim]) / direction[dim]
                if 0 <= t <= 1 and t < t_min:
                    t_min = t

    if t_min == float('inf') or t_min < 0:
        clipped_point = xp.clip(p1, 0, new_shape - 1)
        # Convert to NumPy for return consistency
        if hasattr(clipped_point, 'get'):
            return clipped_point.get().astype(np.float32)
        return clipped_point.astype(np.float32)

    intersection = p1 + t_min * direction
    intersection = xp.clip(intersection, 0, new_shape - 1)
    
    # Convert to NumPy for return consistency
    if hasattr(intersection, 'get'):
        return intersection.get().astype(np.float32)
    return intersection.astype(np.float32)


def transform_streamline(s_mm, A_new_inv, use_gpu=True):
    """Transform a streamline from mm space to voxel space."""
    if use_gpu:
        try:
            import cupy as xp
            
            # Convert inputs to CuPy arrays
            s_mm_gpu = xp.asarray(s_mm, dtype=xp.float32)
            A_new_inv_gpu = xp.asarray(A_new_inv, dtype=xp.float32)
            
            # Create homogeneous coordinates
            homogeneous = xp.hstack([s_mm_gpu, xp.ones((len(s_mm_gpu), 1), dtype=xp.float32)])
            
            # Matrix multiplication using CuPy
            transformed = homogeneous @ A_new_inv_gpu.T
            
            # Return only the first 3 columns (x, y, z)
            result = transformed[:, :3]
            
            # Convert back to NumPy for consistency
            if hasattr(result, 'get'):
                return result.get().astype(np.float32)
            return result.astype(np.float32)
            
        except (ImportError, Exception) as e:
            use_gpu = False
    
    if not use_gpu:
        import numpy as xp
        
        homogeneous = xp.hstack([s_mm, xp.ones((len(s_mm), 1))])
        
        output = xp.zeros((len(s_mm), 3), dtype=xp.float32)
        for i in range(len(s_mm)):
            output[i] = (A_new_inv @ homogeneous[i])[:3]
            
        return output


def transform_and_densify_streamlines(
    streamlines_mm, new_affine, new_shape, step_size=0.5,
    n_jobs=8, use_gpu=True, interp_method='hermite'
):
    """Transform streamlines from mm space to voxel space and apply densification."""
    if not isinstance(streamlines_mm, list):
        streamlines_mm = list(streamlines_mm)
    
    if not streamlines_mm:
        return []

    inv_A = np.linalg.inv(new_affine)
    voxel_size = np.sqrt(np.sum(new_affine[:3, :3] ** 2, axis=0))
    
    print(f"\n[STREAMLINE DEBUG] Transform diagnostics:")
    print(f"Voxel size from affine: {voxel_size}")
    print(f"New shape: {new_shape}")
    print(f"Step size: {step_size}")
    print(f"FOV clipping: ENABLED")
        
    # Transform from mm space to voxel space
    transformed_streams = []
    for s in streamlines_mm:
        h = np.hstack((s, np.ones((len(s), 1))))
        s_vox = h @ inv_A.T
        transformed_streams.append(s_vox[:, :3])
    
    total_streamlines = len(transformed_streams)
    
    # Apply clipping
    clipped_streams = []
    for s in transformed_streams:
        mask = np.all((s >= 0) & (s < np.array(new_shape)), axis=1)
        if np.any(mask):
            if np.sum(mask) >= 2:
                filtered_array = s[mask]
                clipped_streams.append(filtered_array)
    
    clipped_count = len(clipped_streams)
    
    print(f"Clipping Stats: {clipped_count}/{total_streamlines} streamlines retained after FOV clipping ({clipped_count/total_streamlines*100:.1f}%)")
    if clipped_count < total_streamlines * 0.5:
        print(f"WARNING: Over 50% of streamlines were clipped!")
    if clipped_count == 0:
        print("ERROR: All streamlines were clipped!")
        return []
    
    # Clean streamlines
    clean_streams = []
    for i, s in enumerate(clipped_streams):
        try:
            if hasattr(s, 'get'):
                s = s.get()
            
            if isinstance(s, list):
                if not all(isinstance(p, (list, tuple, np.ndarray)) for p in s):
                    continue
                    
                valid_points = []
                for p in s:
                    if isinstance(p, (list, tuple)) and len(p) == 3:
                        valid_points.append(np.array(p, dtype=np.float32))
                    elif isinstance(p, np.ndarray) and p.shape[-1] == 3:
                        valid_points.append(p.astype(np.float32))
                
                if len(valid_points) < 2:
                    continue
                
                s = np.array(valid_points, dtype=np.float32)
            elif not isinstance(s, np.ndarray):
                try:
                    s = np.array(s, dtype=np.float32)
                except Exception:
                    continue
            
            if not isinstance(s, np.ndarray):
                continue
            
            if len(s) < 2:
                continue
                
            if s.dtype != np.float32:
                s = s.astype(np.float32)
                
            clean_streams.append(s)
        except Exception:
            continue
    
    if len(clean_streams) == 0:
        print("ERROR: No valid streamlines after cleaning! Check your input data.")
        return []
        
    print(f"Cleaned streamlines: {len(clean_streams)}/{len(clipped_streams)} retained after validation")
    clipped_streams = clean_streams
    
    print(f"Preparing for densification: {len(clipped_streams)} streamlines")
        
    # Apply densification in voxel space
    min_voxel_size = min(voxel_size)
    densified_streams = densify_streamlines_parallel(
        clipped_streams, step_size, n_jobs=n_jobs, use_gpu=use_gpu,
        interp_method=interp_method, voxel_size=min_voxel_size
    )
    
    print(f"Final streamline count after processing: {len(densified_streams)}")
    if len(densified_streams) == 0:
        print("WARNING: No streamlines were processed! Check your parameters.")
        print("Try a larger voxel size or adjust the step size.")
    
    return densified_streams
