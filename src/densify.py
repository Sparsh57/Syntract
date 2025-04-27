import numpy as np
from joblib import Parallel, delayed
import os

def linear_interpolate(p0, p1, t, xp=np):
    """
    Performing linear interpolation between two points.
    
    Parameters
    ----------
    p0, p1 : array-like
        Start and end points.
    t : float
        Interpolation parameter between 0 and 1.
    xp : module, optional
        Array library to use (numpy or cupy), default is numpy.
        
    Returns
    -------
    array-like
        Interpolated point.
    """
    # Linear interpolation: p = p0 + t * (p1 - p0)
    return p0 + t * (p1 - p0)

def hermite_interpolate(p0, p1, m0, m1, t, xp=np):
    """
    Perform cubic Hermite interpolation between two points.

    Parameters
    ----------
    p0, p1 : array-like
        The start and end points.
    m0, m1 : array-like
        The tangent vectors at start and end points.
    t : float
        Interpolation parameter between 0 and 1.
    xp : module, optional
        Array library to use (numpy or cupy), by default numpy.

    Returns
    -------
    array-like
        The interpolated point.
    """
    # Cubic Hermite interpolation formula
    t2 = t * t
    t3 = t2 * t
    
    h00 = 2*t3 - 3*t2 + 1    # Hermite basis function for position at p0
    h10 = t3 - 2*t2 + t      # Hermite basis function for tangent at p0
    h01 = -2*t3 + 3*t2       # Hermite basis function for position at p1
    h11 = t3 - t2            # Hermite basis function for tangent at p1
    
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

def calculate_streamline_metrics(streamlines, metrics=None):
    """
    Calculate various metrics for a set of streamlines.

    Parameters
    ----------
    streamlines : list
        List of streamlines, where each streamline is an array of shape (N, 3).
    metrics : list, optional
        List of metrics to calculate, by default ['curvature', 'length', 'torsion'].

    Returns
    -------
    dict
        Dictionary of calculated metrics.
    """
    if metrics is None:
        metrics = ['curvature', 'length', 'torsion']
    
    results = {}
    
    # Initialize result containers
    if 'curvature' in metrics:
        results['curvature'] = []
        results['mean_curvature'] = 0
        results['max_curvature'] = 0
    
    if 'length' in metrics:
        results['length'] = []
        results['total_length'] = 0
        results['mean_length'] = 0
    
    if 'torsion' in metrics:
        results['torsion'] = []
        results['mean_torsion'] = 0
    
    total_points = 0
    
    # Process each streamline
    for stream in streamlines:
        if len(stream) < 3:
            continue
            
        # Convert to numpy if not already
        if not isinstance(stream, np.ndarray):
            stream = np.array(stream)
            
        # Calculate tangent vectors (first derivatives)
        tangents = np.zeros_like(stream)
        tangents[1:-1] = (stream[2:] - stream[:-2]) / 2.0
        tangents[0] = stream[1] - stream[0]
        tangents[-1] = stream[-1] - stream[-2]
        
        # Normalize tangents
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_norms = np.where(tangent_norms > 1e-12, tangent_norms, 1.0)
        tangents = tangents / tangent_norms
        
        # Calculate curvature
        if 'curvature' in metrics:
            # Second derivatives
            second_derivs = np.zeros_like(stream)
            second_derivs[1:-1] = (tangents[2:] - tangents[:-2]) / 2.0
            second_derivs[0] = tangents[1] - tangents[0]
            second_derivs[-1] = tangents[-1] - tangents[-2]
            
            # Curvature = |T'|
            curvature = np.linalg.norm(second_derivs, axis=1)
            results['curvature'].append(curvature)
            results['mean_curvature'] += np.sum(curvature)
            results['max_curvature'] = max(results['max_curvature'], np.max(curvature))
            
        # Calculate length
        if 'length' in metrics:
            segment_lengths = np.linalg.norm(stream[1:] - stream[:-1], axis=1)
            length = np.sum(segment_lengths)
            results['length'].append(length)
            results['total_length'] += length
            
        # Calculate torsion
        if 'torsion' in metrics and len(stream) > 3:
            # Third derivatives
            third_derivs = np.zeros_like(stream)
            third_derivs[1:-1] = (second_derivs[2:] - second_derivs[:-2]) / 2.0
            third_derivs[0] = second_derivs[1] - second_derivs[0]
            third_derivs[-1] = second_derivs[-1] - second_derivs[-2]
            
            # Torsion calculation (simplified)
            cross_products = np.zeros_like(stream)
            for i in range(1, len(stream)-1):
                cross_products[i] = np.cross(tangents[i], second_derivs[i])
                
            torsion = np.zeros(len(stream))
            for i in range(1, len(stream)-1):
                if np.linalg.norm(cross_products[i]) > 1e-12:
                    torsion[i] = np.dot(cross_products[i], third_derivs[i]) / np.linalg.norm(cross_products[i])**2
                    
            results['torsion'].append(torsion)
            results['mean_torsion'] += np.sum(np.abs(torsion))
            
        total_points += len(stream)
    
    # Calculate averages
    if total_points > 0:
        if 'curvature' in metrics:
            results['mean_curvature'] /= total_points
        if 'torsion' in metrics:
            results['mean_torsion'] /= total_points
    
    if len(results.get('length', [])) > 0:
        results['mean_length'] = results['total_length'] / len(results['length'])
    
    return results

def densify_streamlines_parallel(streamlines, step_size, n_jobs=8, use_gpu=True, interp_method='hermite', voxel_size=1.0):
    """
    Densify multiple streamlines in parallel.
    
    Parameters
    ----------
    streamlines : list of arrays
        List of streamlines to densify.
    step_size : float
        Step size for densification.
    n_jobs : int, optional
        Number of parallel jobs, by default 8.
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.
    interp_method : str, optional
        Interpolation method ('hermite' or 'linear'), by default 'hermite'.
    voxel_size : float, optional
        Size of the voxels, by default 1.0.
        
    Returns
    -------
    list of arrays
        Densified streamlines.
    """
    if len(streamlines) == 0:
        return []

    # Process streamlines in parallel
    if n_jobs == 1:
        # More efficient with a progress counter
        densified = []
        success_count = 0
        total_count = len(streamlines)
        total = len(streamlines)
        for i, streamline in enumerate(streamlines):
            if i % 1000 == 0:
                print(f"Processing streamline {i}/{total}...")
            
            # Skip invalid streamlines
            if len(streamlines[i]) < 2:
                continue
                
            try:
                # First ensure streamline is proper numpy array before densification
                streamline = streamlines[i]
                if isinstance(streamline, list):
                    try:
                        # Deep validation of list-type streamlines
                        # Ensure all points are properly formatted as 3D points
                        if not all(isinstance(p, (list, tuple, np.ndarray)) for p in streamline):
                            print(f"Error densifying streamline {i}: Contains invalid point types")
                            continue
                        
                        # Validate each point is a 3D coordinate
                        valid_points = []
                        for p in streamline:
                            if isinstance(p, (list, tuple)) and len(p) == 3:
                                valid_points.append(np.array(p, dtype=np.float32))
                            elif isinstance(p, np.ndarray) and p.shape[-1] == 3:
                                valid_points.append(p.astype(np.float32))
                            else:
                                # Skip invalid points
                                continue
                        
                        if len(valid_points) < 2:
                            print(f"Error densifying streamline {i}: Insufficient valid points after filtering")
                            continue
                            
                        # Create a clean numpy array from validated points
                        streamline = np.array(valid_points, dtype=np.float32)
                    except Exception as e:
                        print(f"Error densifying streamline {i}: Failed to convert list to array: {e}")
                        continue
                elif not isinstance(streamline, np.ndarray):
                    print(f"Error densifying streamline {i}: Unsupported type {type(streamline)}")
                    continue
                    
                # Apply densification
                result = densify_streamline_subvoxel(
                    streamline, step_size, 
                    use_gpu=use_gpu, interp_method=interp_method,
                    voxel_size=voxel_size
                )
                
                # Ensure result is a numpy array before appending
                if not isinstance(result, np.ndarray):
                    try:
                        print(f"Converting result for streamline {i} from {type(result)} to numpy array")
                        result = np.array(result, dtype=np.float32)
                    except Exception as e:
                        print(f"Error converting result to numpy array: {e}")
                        continue
                        
                densified.append(result)
                success_count += 1
            except Exception as e:
                print(f"Error densifying streamline {i}: {e}")
                
        print(f"Densified {success_count}/{total_count} streamlines successfully")
        # Final unconditional enforcement for all results
        densified = [np.array(s, dtype=np.float32) if not isinstance(s, np.ndarray) else s for s in densified]
        return densified
    else:
        # Import numpy here to ensure it's available in both code paths
        import numpy as np
        
        # Parallel processing
        from joblib import Parallel, delayed
        
        def _process_one(streamline, idx, total):
            if idx % 1000 == 0:
                print(f"Processing streamline {idx}/{total}...")
            
            try:
                # Convert to numpy array if needed
                if isinstance(streamline, list):
                    try:
                        # Ensure all points are properly formatted as 3D points
                        if not all(isinstance(p, (list, tuple, np.ndarray)) for p in streamline):
                            print(f"Warning: Streamline {idx} contains invalid point types. Skipping.")
                            return None
                        
                        # Validate each point is a 3D coordinate
                        valid_points = []
                        for p in streamline:
                            if isinstance(p, (list, tuple)) and len(p) == 3:
                                valid_points.append(np.array(p, dtype=np.float32))
                            elif isinstance(p, np.ndarray) and p.shape[-1] == 3:
                                valid_points.append(p.astype(np.float32))
                            else:
                                # Skip invalid points
                                continue
                        
                        if len(valid_points) < 2:
                            print(f"Warning: Streamline {idx} has insufficient valid points after filtering. Skipping.")
                            return None
                            
                        # Create a clean numpy array from validated points
                        streamline = np.array(valid_points, dtype=np.float32)
                    except Exception as e:
                        print(f"Error processing list-type streamline {idx}: {e}")
                        return None
                        
                elif hasattr(streamline, 'get'):  # Handle CuPy arrays
                    streamline = streamline.get()
                    streamline = np.asarray(streamline, dtype=np.float32)
                elif not isinstance(streamline, np.ndarray):
                    try:
                        streamline = np.asarray(streamline, dtype=np.float32)
                    except Exception as e:
                        print(f"Error converting streamline {idx} to numpy array: {e}")
                        return None
                
                if len(streamline) < 2:
                    print(f"Warning: Streamline {idx} has {len(streamline)} points, skipping.")
                    return None
                
                # Directly pass the numpy array to the densification function
                d = densify_streamline_subvoxel(
                    streamline, step_size, use_gpu=use_gpu,
                    interp_method=interp_method, voxel_size=voxel_size
                )
                
                # Ensure output is numpy array
                if hasattr(d, 'get'):  # Convert CuPy to numpy
                    d = d.get()
                    
                # Make sure we return a numpy array, not a list
                if not isinstance(d, np.ndarray):
                    try:
                        d = np.array(d, dtype=np.float32)
                    except Exception as e:
                        print(f"Error converting densified streamline to numpy array: {e}")
                        return None
                return d
            except Exception as e:
                print(f"Error densifying streamline {idx}: {e}")
                return None
        
        total = len(streamlines)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_one)(streamline, idx, total) 
            for idx, streamline in enumerate(streamlines)
        )
        
        # Filter out None results and ensure each streamline has at least 2 points
        densified = []
        for r in results:
            if r is None or len(r) < 2:
                continue
                
            # Additional check to ensure numpy array type
            if not isinstance(r, np.ndarray):
                try:
                    r = np.array(r, dtype=np.float32)
                except Exception as e:
                    print(f"Error: Skipping invalid result - {e}")
                    continue
                    
            densified.append(r)
        
        # Report on densification results
        print(f"Densified {len(densified)}/{total} streamlines successfully")
        # Final unconditional enforcement for all results
        densified = [np.array(s, dtype=np.float32) if not isinstance(s, np.ndarray) else s for s in densified]
        return densified

def rbf_interpolate_streamline(streamline, step_size, function='thin_plate', epsilon=None, xp=np):
    """
    Perform RBF interpolation for a streamline.
    """
    # Skip if too few points
    if len(streamline) < 4:
        return streamline

    # Calculate the total length of the streamline
    diffs = xp.diff(streamline, axis=0)
    segment_lengths = xp.sqrt(xp.sum(diffs**2, axis=1))
    total_length = xp.sum(segment_lengths)

    # Skip densification if streamline is shorter than step_size
    if total_length < step_size:
        return streamline

    # Create parameterization (cumulative distance)
    t = xp.concatenate(([0], xp.cumsum(segment_lengths)))

    # Calculate new sampling points
    n_steps = max(int(xp.ceil(total_length / step_size)), 2)
    t_new = xp.linspace(0, t[-1], n_steps)

    # RBF interpolation requires scipy - convert to CPU if necessary
    if hasattr(xp, 'asnumpy'):  # Check if using CuPy
        import numpy as np
        t_cpu = xp.asnumpy(t)
        streamline_cpu = xp.asnumpy(streamline)
        t_new_cpu = xp.asnumpy(t_new)
        from scipy.interpolate import Rbf
        result = np.zeros((len(t_new_cpu), streamline_cpu.shape[1]), dtype=np.float32)
        for dim in range(streamline_cpu.shape[1]):
            rbf = Rbf(t_cpu, streamline_cpu[:, dim], function=function, epsilon=epsilon)
            result[:, dim] = rbf(t_new_cpu)
        return xp.asarray(result)
    else:
        from scipy.interpolate import Rbf
        result = xp.zeros((len(t_new), streamline.shape[1]), dtype=xp.float32)
        for dim in range(streamline.shape[1]):
            rbf = Rbf(t, streamline[:, dim], function=function, epsilon=epsilon)
            result[:, dim] = rbf(t_new)
        return result

def densify_streamline_subvoxel(streamline, step_size, use_gpu=True, interp_method='hermite', voxel_size=1.0):
    """
    Densify a streamline with sub-voxel precision.

    Parameters
    ----------
    streamline : array-like
        The streamline to densify.
    step_size : float
        Step size for densification.
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.
    interp_method : str, optional
        Interpolation method ('hermite' or 'linear'), by default 'hermite'.
    voxel_size : float, optional
        Size of the voxels, affects tangent scaling in Hermite interpolation, by default 1.0.

    Returns
    -------
    array-like
        Densified streamline.
    """
    # First import numpy to ensure it's available in all code paths
    import numpy as np
    
    # Ensure we have a numpy array, not a list
    if isinstance(streamline, list):
        try:
            # More robust conversion with error reporting
            if not all(isinstance(point, (list, tuple, np.ndarray)) for point in streamline):
                raise TypeError(f"Streamline contains non-array elements: {[type(x) for x in streamline][:5]}...")
            # Ensure all points are properly formatted as arrays
            clean_points = []
            for point in streamline:
                if isinstance(point, (list, tuple)):
                    if len(point) != 3:
                        raise ValueError(f"Expected 3D point but got {len(point)}D: {point}")
                    clean_points.append(np.array(point, dtype=np.float32))
                elif isinstance(point, np.ndarray):
                    clean_points.append(point.astype(np.float32))
                else:
                    raise TypeError(f"Unexpected point type: {type(point)}")
            streamline = np.array(clean_points, dtype=np.float32)
        except Exception as e:
            raise TypeError(f"Failed to convert list to numpy array: {e}")

    # Check if streamline is valid
    if len(streamline) < 2:
        print(f"Warning: Cannot densify streamline with less than 2 points (has {len(streamline)})")
        return streamline

    # Check for debug flags
    debug_tangents = os.environ.get("DEBUG_TANGENTS") == "1"

    # Debug: Print interpolation method being used
    if debug_tangents:
        print(f"[DENSIFY] Using {interp_method} interpolation with step size {step_size}, voxel size {voxel_size}mm")
        print(f"[DENSIFY] Streamline points: {len(streamline)}")
    
    # Choose appropriate array library based on use_gpu flag
    if use_gpu:
        try:
            import cupy as xp
            print(f"[DEBUG] GPU processing with CuPy - streamline type: {type(streamline)}, shape: {streamline.shape if hasattr(streamline, 'shape') else 'unknown'}")
            
            # First ensure we have a proper numpy array before sending to cupy
            if not isinstance(streamline, np.ndarray):
                print(f"[DEBUG] Converting streamline from {type(streamline)} to numpy array")
                streamline = np.array(streamline, dtype=np.float32)
            
            # Then convert to cupy - this is where type errors can occur
            try:
                streamline_device = xp.asarray(streamline)
                print(f"[DEBUG] CuPy conversion successful - device array shape: {streamline_device.shape}")
            except Exception as e:
                print(f"CuPy conversion error: {e}")
                print(f"Streamline type: {type(streamline)}, shape: {np.shape(streamline)}")
                if isinstance(streamline, list):
                    print(f"List contents: {[type(x) for x in streamline]}")
                    raise TypeError("Cannot convert list directly to CuPy array, must be numpy array first")
                # Fall back to CPU if cupy conversion fails
                print("[DEBUG] Falling back to CPU due to CuPy conversion error")
                import numpy as xp
                streamline_device = xp.asarray(streamline)
                use_gpu = False
        except ImportError:
            print("Warning: Could not import cupy. Falling back to CPU.")
            import numpy as xp
            streamline_device = xp.asarray(streamline)
            use_gpu = False
    else:
        # Always use numpy for CPU mode
        xp = np
        # Still ensure numpy array format even in CPU mode
        if not isinstance(streamline, np.ndarray):
            streamline = np.array(streamline, dtype=np.float32)
        streamline_device = xp.asarray(streamline)

    # Calculate the total length of the streamline
    diffs = xp.diff(streamline_device, axis=0)
    segment_lengths = xp.sqrt(xp.sum(diffs**2, axis=1))
    total_length = xp.sum(segment_lengths)
    
    if debug_tangents:
        print(f"[DENSIFY] Total streamline length: {total_length:.4f}mm")
        print(f"[DENSIFY] Mean segment length: {total_length/len(segment_lengths):.4f}mm")
    
    # Skip densification if streamline is shorter than step_size
    if total_length < step_size:
        if debug_tangents:
            print(f"[DENSIFY] Streamline too short for densification: {total_length:.4f} < {step_size:.4f}")
        return streamline

    # Calculate the number of steps needed
    # We use ceil to ensure points are at most step_size apart
    # Adding 1 ensures the segments are shorter than step_size, not just equal to step_size
    n_steps = int(xp.ceil(total_length / step_size)) + 1

    # Calculate cumulative distance along the streamline
    cumulative_lengths = xp.concatenate(([0], xp.cumsum(segment_lengths)))
    normalized_distances = cumulative_lengths / total_length
    
    # Create evenly spaced points along the streamline
    xi = xp.linspace(0, 1, n_steps)
    
    # Check for NaN values that could cause issues
    if xp.any(xp.isnan(normalized_distances)) or xp.any(xp.isnan(xi)):
        print("Warning: NaN values detected in distance calculation. Using original streamline.")
        return streamline
        
    # Safety check - ensure all arrays are correct type
    if use_gpu and hasattr(xp, 'ndarray'):
        # Check if arrays are proper CuPy arrays when in GPU mode
        if not isinstance(normalized_distances, xp.ndarray):
            normalized_distances = xp.asarray(normalized_distances)
        if not isinstance(xi, xp.ndarray):
            xi = xp.asarray(xi)
    else:
        # Check if arrays are proper NumPy arrays when in CPU mode
        if not isinstance(normalized_distances, np.ndarray):
            normalized_distances = np.asarray(normalized_distances)
        if not isinstance(xi, np.ndarray):
            xi = np.asarray(xi)
    
    # Calculate tangents for Hermite interpolation
    if interp_method == 'hermite':
        # Initialize tangents
        tangents = xp.zeros_like(streamline_device)
        
        # Determine tangent scaling based on voxel size
        base_scale = 1.0
        
        # Calculate voxel-based scaling factor - higher for smaller voxels
        voxel_scale = (1.0 / max(0.01, voxel_size))
        # Cap the scaling to avoid extreme values
        voxel_scale = min(5.0, voxel_scale)
        
        # Final scaling factor is a combination of the base scale and voxel scale
        final_scale = base_scale * voxel_scale
        
        if debug_tangents:
            print(f"[TANGENT] Voxel size: {voxel_size:.4f}mm")
            print(f"[TANGENT] Base scale: {base_scale:.4f}")
            print(f"[TANGENT] Voxel scale: {voxel_scale:.4f}")
            print(f"[TANGENT] Final scale factor: {final_scale:.4f}")
        
        # Calculate tangents for interior points using finite differences
        for i in range(1, len(streamline_device) - 1):
            # Use finite differences with scaling for better curvature
            pt_prev = streamline_device[i-1]
            pt_next = streamline_device[i+1]
            tangent = (pt_next - pt_prev) * 0.5
            
            # Apply scaling
            tangents[i] = tangent * final_scale
            
            # Debug: print first tangent
            if i == 1 and debug_tangents:
                tangent_magnitude = xp.linalg.norm(tangent)
                scaled_magnitude = xp.linalg.norm(tangents[i])
                print(f"[TANGENT] Example - Point {i}:")
                print(f"  Original tangent: {tangent}, magnitude: {tangent_magnitude:.4f}")
                print(f"  Scaled tangent: {tangents[i]}, magnitude: {scaled_magnitude:.4f}")
        
        # Calculate tangents for endpoints using same scaling
        tangents[0] = (streamline_device[1] - streamline_device[0]) * final_scale
        tangents[-1] = (streamline_device[-1] - streamline_device[-2]) * final_scale
        
        # Save original magnitudes for reference
        if debug_tangents:
            orig_magnitudes = xp.sqrt(xp.sum(tangents**2, axis=1))
            print(f"[TANGENT] Original magnitudes: mean={xp.mean(orig_magnitudes):.4f}, max={xp.max(orig_magnitudes):.4f}")
        
        # Normalize tangents for standard mode, but with less aggressive normalization
        # This allows for more distinctive curvature differences between methods
        tangent_norms = xp.sqrt(xp.sum(tangents**2, axis=1, keepdims=True))
        # Avoid division by zero
        tangent_norms = xp.where(tangent_norms > 1e-10, tangent_norms, 1e-10)
        
        # Scale normalization based on voxel size to maintain some of the curvature effect
        normalization_factor = max(0.5, min(1.0, voxel_size))
        
        if debug_tangents:
            print(f"[TANGENT] Normalization factor: {normalization_factor:.4f}")
        
        tangents = tangents / (tangent_norms * normalization_factor)
        
        # Debug: print normalized magnitudes
        if debug_tangents:
            normalized_magnitudes = xp.sqrt(xp.sum(tangents**2, axis=1))
            print(f"[TANGENT] Normalized magnitudes: mean={xp.mean(normalized_magnitudes):.4f}, max={xp.max(normalized_magnitudes):.4f}")

    if interp_method == 'rbf':
        try:
            densified_streamline = rbf_interpolate_streamline(
                streamline_device, 
                step_size, 
                function='thin_plate', 
                epsilon=None, 
                xp=xp
            )
            return xp.asnumpy(densified_streamline) if use_gpu and hasattr(xp, 'asnumpy') else densified_streamline
        except Exception as e:
            print(f"RBF interpolation failed: {e}. Falling back to linear interpolation.")
            interp_method = 'linear'

    # Interpolate each coordinate
    # Pre-allocate result array
    result_shape = (len(xi), streamline_device.shape[1])
    if use_gpu and hasattr(xp, 'zeros'):
        result_array = xp.zeros(result_shape, dtype=xp.float32)
    else:
        result_array = np.zeros(result_shape, dtype=np.float32)
    
    # Interpolate each coordinate
    for dim in range(streamline_device.shape[1]):
        y = streamline_device[:, dim]
        
        if interp_method == 'hermite':
            # Use Hermite cubic interpolation
            if use_gpu:
                # Use numpy for tangent import
                import numpy as np
                # We need to convert the data back to CPU for scipy
                if hasattr(xp, 'asnumpy'):  # If we're using CuPy
                    t_values = xp.asnumpy(normalized_distances)
                    interp_points = xp.asnumpy(xi)
                    y_values = xp.asnumpy(y)
                    tangent_values = xp.asnumpy(tangents[:, dim])
                else:  # Using NumPy already
                    t_values = np.array(normalized_distances)
                    interp_points = np.array(xi)
                    y_values = np.array(y)
                    tangent_values = np.array(tangents[:, dim])
                
                # Use scipy on CPU for Hermite interpolation
                try:
                    from scipy.interpolate import CubicHermiteSpline
                    interpolator = CubicHermiteSpline(t_values, y_values, tangent_values)
                    interpolated_cpu = interpolator(interp_points)
                    
                    # Convert back to device 
                    interpolated = xp.asarray(interpolated_cpu)
                except Exception as e:
                    print(f"Hermite interpolation failed in GPU path: {e}. Falling back to linear interpolation.")
                    # Use linear interpolation as fallback
                    interpolated = xp.interp(xi, normalized_distances, y)
            else:
                # CPU path
                try:
                    from scipy.interpolate import CubicHermiteSpline
                    interpolator = CubicHermiteSpline(normalized_distances, y, tangents[:, dim])
                    interpolated = interpolator(xi)
                except Exception as e:
                    print(f"Hermite interpolation failed: {e}. Falling back to linear interpolation.")
                    interpolated = xp.interp(xi, normalized_distances, y)
        else:
            # Use linear interpolation
            interpolated = xp.interp(xi, normalized_distances, y)
            
        # Store the interpolated values in the result array
        result_array[:, dim] = interpolated
    
    # No need for column_stack since we've built the array directly
    densified_streamline = result_array
    
    # Convert back to numpy if on GPU
    if use_gpu and hasattr(xp, 'asnumpy'):
        densified_streamline = xp.asnumpy(densified_streamline)
    
    # Ensure output is numpy array, not a list
    if not isinstance(densified_streamline, np.ndarray):
        densified_streamline = np.array(densified_streamline, dtype=np.float32)
    
    # Debug: info about the densified streamline
    if debug_tangents:
        print(f"[DENSIFY] Original points: {len(streamline)}, Densified points: {len(densified_streamline)}")
        
        if interp_method == 'hermite':
            # Compare curvature between original and densified
            if len(streamline) > 2 and len(densified_streamline) > 2:
                # Simple curvature calculation for comparison
                def calc_curvature(points):
                    import numpy as np  # Import numpy here for use in this function
                    if len(points) < 3:
                        return 0
                    # Calculate first derivatives (tangents)
                    tangents = np.zeros_like(points)
                    tangents[1:-1] = (points[2:] - points[:-2]) / 2.0
                    tangents[0] = points[1] - points[0]
                    tangents[-1] = points[-1] - points[-2]
                    # Normalize tangents
                    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
                    norms = np.where(norms > 1e-10, norms, 1e-10)
                    tangents = tangents / norms
                    # Calculate second derivatives
                    second_derivs = np.zeros_like(points)
                    second_derivs[1:-1] = (tangents[2:] - tangents[:-2]) / 2.0
                    # Curvature is the magnitude of the second derivative
                    curvatures = np.linalg.norm(second_derivs, axis=1)
                    return np.mean(curvatures)
                orig_curvature = calc_curvature(streamline)
                new_curvature = calc_curvature(densified_streamline)
                print(f"[CURVATURE] Original streamline: {orig_curvature:.6f}")
                print(f"[CURVATURE] Densified streamline: {new_curvature:.6f}")
                # Avoid division by zero for percentage change calculation
                if orig_curvature > 0:
                    print(f"[CURVATURE] Change: {(new_curvature-orig_curvature)/orig_curvature*100:.2f}%")
                else:
                    print(f"[CURVATURE] Change: N/A (original curvature was zero)")
    # Final unconditional enforcement
    if not isinstance(densified_streamline, np.ndarray):
        densified_streamline = np.array(densified_streamline, dtype=np.float32)
    return densified_streamline


# Test the interpolation functions with a simple case when module is run directly
if __name__ == "__main__":
    print("Testing interpolation functions...")
    
    # Create a simple test streamline
    test_stream = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [2, 0, 0]
    ], dtype=np.float32)
    
    print(f"Original streamline shape: {test_stream.shape}")
    
    # Test linear interpolation
    linear_result = densify_streamline_subvoxel(
        test_stream.copy(), step_size=0.5, use_gpu=False, interp_method='linear'
    )
    print(f"Linear interpolation result shape: {linear_result.shape}")
    print(f"Linear first few points:\n{linear_result[:5]}")
    
    # Test Hermite interpolation
    hermite_result = densify_streamline_subvoxel(
        test_stream.copy(), step_size=0.5, use_gpu=False, interp_method='hermite'
    )
    print(f"Hermite interpolation result shape: {hermite_result.shape}")
    print(f"Hermite first few points:\n{hermite_result[:5]}")
    
    # Compare results
    if np.array_equal(linear_result, hermite_result):
        print("ERROR: Both methods produced identical results!")
    else:
        difference = np.mean(np.abs(linear_result[:min(len(linear_result), len(hermite_result))] - 
                                   hermite_result[:min(len(linear_result), len(hermite_result))]))
        print(f"Mean difference between methods: {difference:.6f}")
        print("Test passed: interpolation methods produce different results.")
