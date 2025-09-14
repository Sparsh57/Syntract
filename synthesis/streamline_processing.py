import numpy as np
import os
import shutil
import tempfile
from joblib import Parallel, delayed
try:
    from .densify import densify_streamline_subvoxel, densify_streamlines_parallel
except ImportError:
    from densify import densify_streamline_subvoxel, densify_streamlines_parallel

def to_backend(x, xp):
    if xp.__name__ == "cupy":
        import cupy
        return cupy.asarray(x)
    else:
        import numpy as np
        return np.asarray(x)

def clip_streamline_to_fov(stream, new_shape, use_gpu=True, epsilon=1e-6):
    """Clip a streamline to the field of view."""
    if use_gpu:
        try:
            import cupy as xp
            # Convert numpy array to cupy array if needed
            if hasattr(stream, 'get'):  # Already a cupy array
                stream_gpu = stream
            else:
                stream_gpu = xp.asarray(stream, dtype=xp.float32)
        except ImportError:
            import numpy as xp
            use_gpu = False
            stream_gpu = xp.asarray(stream, dtype=xp.float32)
    else:
        import numpy as xp
        stream_gpu = xp.asarray(stream, dtype=xp.float32)
        
    if len(stream_gpu) == 0:
        return []
        
    new_shape = xp.array(new_shape)
    inside = xp.all((stream_gpu >= -epsilon) & (stream_gpu < new_shape + epsilon), axis=1)
    
    if not xp.any(inside):
        return []
        
    segments = []
    current_segment = []

    for i in range(len(stream_gpu)):
        if inside[i]:
            current_segment.append(stream_gpu[i])
        else:
            if len(current_segment) >= 2:
                segments.append(xp.array(current_segment, dtype=xp.float32))
            current_segment = []

            if i > 0 and inside[i - 1]:
                p1, p2 = stream_gpu[i - 1], stream_gpu[i]
                clipped_point = interpolate_to_fov(p1, p2, new_shape, use_gpu)
                if clipped_point is not None:
                    segments.append(xp.array([to_backend(p1, xp), to_backend(clipped_point, xp)], dtype=xp.float32))
            elif i < len(stream_gpu) - 1 and inside[i + 1]:
                p1, p2 = stream_gpu[i], stream_gpu[i + 1]
                clipped_point = interpolate_to_fov(p2, p1, new_shape, use_gpu)
                if clipped_point is not None:
                    segments.append(xp.array([to_backend(p1, xp), to_backend(clipped_point, xp)], dtype=xp.float32))


    if len(current_segment) >= 2:
        segments.append(xp.array(current_segment, dtype=xp.float32))

    if use_gpu:
        try:
            import numpy as np
            numpy_segments = []
            for segment in segments:
                if hasattr(segment, 'get'):  # CuPy array
                    numpy_segments.append(np.array(segment.get(), dtype=np.float32))
                else:
                    numpy_segments.append(np.array(segment, dtype=np.float32))
            return numpy_segments
        except Exception:
            # Fallback: convert any CuPy arrays to NumPy
            import numpy as np
            fallback_segments = []
            for segment in segments:
                if hasattr(segment, 'get'):  # CuPy array
                    fallback_segments.append(np.array(segment.get(), dtype=np.float32))
                else:
                    fallback_segments.append(np.array(segment, dtype=np.float32))
            return fallback_segments
    else:
        import numpy as np
        result_segments = []
        for segment in segments:
            if isinstance(segment, np.ndarray):
                result_segments.append(segment)
            elif hasattr(segment, 'get'):  # CuPy array
                result_segments.append(segment.get())
            else:
                result_segments.append(np.array(segment, dtype=np.float32))
        return result_segments


def interpolate_to_fov(p1, p2, new_shape, use_gpu=True):
    """Interpolate a point on the boundary of the field of view."""
    if use_gpu:
        try:
            import cupy as xp
            # Convert to cupy arrays if needed
            if hasattr(p1, 'get'):  # Already cupy arrays
                p1_gpu = p1
                p2_gpu = p2
            else:
                p1_gpu = xp.asarray(p1, dtype=xp.float32)
                p2_gpu = xp.asarray(p2, dtype=xp.float32)
        except ImportError:
            import numpy as xp
            use_gpu = False
            p1_gpu = xp.asarray(p1, dtype=xp.float32)
            p2_gpu = xp.asarray(p2, dtype=xp.float32)
    else:
        import numpy as xp
        p1_gpu = xp.asarray(p1, dtype=xp.float32)
        p2_gpu = xp.asarray(p2, dtype=xp.float32)
        
    new_shape = xp.asarray(new_shape)
        
    direction = p2_gpu - p1_gpu
    t_min = float('inf')

    for dim in range(3):
        if abs(direction[dim]) > 1e-10:
            if p2_gpu[dim] < 0:
                t = (0 - p1_gpu[dim]) / direction[dim]
                if 0 <= t <= 1 and t < t_min:
                    t_min = t
            elif p2_gpu[dim] >= new_shape[dim]:
                t = (new_shape[dim] - 1 - p1_gpu[dim]) / direction[dim]
                if 0 <= t <= 1 and t < t_min:
                    t_min = t

    if t_min == float('inf') or t_min < 0:
        clipped_point = xp.clip(p1_gpu, 0, xp.asarray(new_shape) - 1)
        # Convert CuPy array to NumPy if needed
        if use_gpu and hasattr(clipped_point, 'get'):
            clipped_point = clipped_point.get()
        return clipped_point

    intersection = p1_gpu + t_min * direction
    intersection = xp.clip(intersection, 0, xp.asarray(new_shape) - 1)
    
    # Convert CuPy array to NumPy if needed
    if use_gpu and hasattr(intersection, 'get'):
        intersection = intersection.get()
    
    return intersection


def transform_streamline(s_mm, A_new_inv, use_gpu=True):
    """Transform a streamline from mm space to voxel space."""
    if use_gpu:
        try:
            import cupy as xp
            from numba import cuda
            
            s_mm_gpu = xp.asarray(s_mm, dtype=xp.float32)
            A_new_inv_gpu = xp.asarray(A_new_inv, dtype=xp.float32)
            output = xp.zeros_like(s_mm_gpu)
            
            @cuda.jit
            def transform_kernel(s_mm, A_new_inv, output):
                idx = cuda.grid(1)
                if idx < s_mm.shape[0]:
                    # Create homogeneous coordinates manually (can't use xp.zeros in CUDA kernel)
                    hom_mm_0 = s_mm[idx, 0]
                    hom_mm_1 = s_mm[idx, 1]
                    hom_mm_2 = s_mm[idx, 2]
                    hom_mm_3 = 1.0
                    
                    # Calculate transformed coordinates manually (can't use xp.zeros in CUDA kernel)
                    result_0 = A_new_inv[0, 0] * hom_mm_0 + A_new_inv[0, 1] * hom_mm_1 + A_new_inv[0, 2] * hom_mm_2 + A_new_inv[0, 3] * hom_mm_3
                    result_1 = A_new_inv[1, 0] * hom_mm_0 + A_new_inv[1, 1] * hom_mm_1 + A_new_inv[1, 2] * hom_mm_2 + A_new_inv[1, 3] * hom_mm_3
                    result_2 = A_new_inv[2, 0] * hom_mm_0 + A_new_inv[2, 1] * hom_mm_1 + A_new_inv[2, 2] * hom_mm_2 + A_new_inv[2, 3] * hom_mm_3
                    
                    output[idx, 0] = result_0
                    output[idx, 1] = result_1
                    output[idx, 2] = result_2
            
            threads_per_block = 256
            blocks_per_grid = (s_mm_gpu.shape[0] + threads_per_block - 1) // threads_per_block
            transform_kernel[blocks_per_grid, threads_per_block](s_mm_gpu, A_new_inv_gpu, output)
            
            # Convert back to numpy if needed for compatibility
            if hasattr(output, 'get'):  # CuPy array
                return output.get()
            else:
                return output
            
        except ImportError:
            import numpy as xp
            use_gpu = False
        except Exception as e:
            # Handle CUDA runtime errors or other GPU-related issues
            print(f"GPU processing failed: {e}. Falling back to CPU.")
            import numpy as xp
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
                    if hasattr(s, 'get'):  # CuPy array
                        s = s.get()
                    else:
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

