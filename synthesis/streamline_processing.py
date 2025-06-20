import numpy as np
import os
import shutil
import tempfile
from joblib import Parallel, delayed
from .densify import densify_streamline_subvoxel, densify_streamlines_parallel


def clip_streamline_to_fov(stream, new_shape, use_gpu=True, epsilon=1e-6):
    """Clip a streamline to the field of view."""
    if use_gpu:
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp
        
    if len(stream) == 0:
        return []
        
    new_shape = xp.array(new_shape)
    inside = xp.all((stream >= -epsilon) & (stream < new_shape + epsilon), axis=1)
    
    if not xp.any(inside):
        return []
        
    segments = []
    current_segment = []

    for i in range(len(stream)):
        if inside[i]:
            current_segment.append(stream[i])
        else:
            if len(current_segment) >= 2:
                segments.append(xp.array(current_segment, dtype=xp.float32))
            current_segment = []

            if i > 0 and inside[i - 1]:
                p1, p2 = stream[i - 1], stream[i]
                clipped_point = interpolate_to_fov(p1, p2, new_shape, use_gpu)
                if clipped_point is not None:
                    segments.append(xp.array([p1, clipped_point], dtype=xp.float32))
            elif i < len(stream) - 1 and inside[i + 1]:
                p1, p2 = stream[i], stream[i + 1]
                clipped_point = interpolate_to_fov(p2, p1, new_shape, use_gpu)
                if clipped_point is not None:
                    segments.append(xp.array([clipped_point, p2], dtype=xp.float32))

    if len(current_segment) >= 2:
        segments.append(xp.array(current_segment, dtype=xp.float32))

    if use_gpu:
        try:
            import numpy as np
            numpy_segments = []
            for segment in segments:
                if hasattr(xp, 'asnumpy'):
                    numpy_segments.append(np.array(xp.asnumpy(segment), dtype=np.float32))
                else:
                    numpy_segments.append(np.array(segment, dtype=np.float32))
            return numpy_segments
        except Exception:
            return segments
    else:
        import numpy as np
        return [np.array(segment, dtype=np.float32) if not isinstance(segment, np.ndarray) else segment 
                for segment in segments]


def interpolate_to_fov(p1, p2, new_shape, use_gpu=True):
    """Interpolate a point on the boundary of the field of view."""
    if use_gpu:
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp
        
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
        clipped_point = xp.clip(p1, 0, xp.asarray(new_shape) - 1)
        return clipped_point

    intersection = p1 + t_min * direction
    intersection = xp.clip(intersection, 0, xp.asarray(new_shape) - 1)
    
    if use_gpu and hasattr(xp, 'asnumpy'):
        intersection = xp.asnumpy(intersection)
    
    return intersection


def transform_streamline(s_mm, A_new_inv, use_gpu=True):
    """Transform a streamline from mm space to voxel space."""
    if use_gpu:
        try:
            import cupy as xp
            from numba import cuda
            
            s_mm_gpu = xp.asarray(s_mm, dtype=xp.float32)
            output = xp.zeros_like(s_mm_gpu)
            
            @cuda.jit
            def transform_kernel(s_mm, A_new_inv, output):
                idx = cuda.grid(1)
                if idx < s_mm.shape[0]:
                    hom_mm = xp.zeros(4, dtype=xp.float32)
                    hom_mm[0] = s_mm[idx, 0]
                    hom_mm[1] = s_mm[idx, 1]
                    hom_mm[2] = s_mm[idx, 2]
                    hom_mm[3] = 1.0
                    
                    result = xp.zeros(3, dtype=xp.float32)
                    for i in range(3):
                        for j in range(4):
                            result[i] += A_new_inv[i, j] * hom_mm[j]
                    
                    output[idx, 0] = result[0]
                    output[idx, 1] = result[1]
                    output[idx, 2] = result[2]
            
            threads_per_block = 256
            blocks_per_grid = (s_mm_gpu.shape[0] + threads_per_block - 1) // threads_per_block
            transform_kernel[blocks_per_grid, threads_per_block](s_mm_gpu, A_new_inv, output)
            
            return output
            
        except ImportError:
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
