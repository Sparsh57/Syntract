import numpy as np
from joblib import Parallel, delayed
import os

def linear_interpolate(p0, p1, t, xp=np):
    """
    Perform linear interpolation between two points.
    
    Parameters
    ----------
    p0, p1 : array-like
        The start and end points.
    t : float
        Interpolation parameter between 0 and 1.
    xp : module, optional
        Array library to use (numpy or cupy), by default numpy.
        
    Returns
    -------
    array-like
        The interpolated point.
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

def densify_streamlines_parallel(streamlines, step_size, n_jobs=8, use_gpu=True, interp_method='hermite', high_res_mode=False, voxel_size=1.0):
    """
    Densify a list of streamlines in parallel.

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
    high_res_mode : bool, optional
        Special high-resolution processing mode, by default False.
    voxel_size : float, optional
        Size of the voxels, affects tangent scaling in Hermite interpolation, by default 1.0.

    Returns
    -------
    list of arrays
        Densified streamlines.
    """
    # Import numpy here to ensure it's available in both code paths
    import numpy as np
    
    if not streamlines:
        return []
    
    # Process streamlines in parallel
    if n_jobs == 1:
        # Sequential processing
        densified = []
        total = len(streamlines)
        for i, streamline in enumerate(streamlines):
            if i % 1000 == 0:
                print(f"Processing streamline {i}/{total}...")
            try:
                # Ensure we're passing a numpy array
                if isinstance(streamline, list):
                    streamline = np.array(streamline, dtype=np.float32)
                
                d = densify_streamline_subvoxel(streamline, step_size, use_gpu, interp_method, 
                                               high_res_mode=high_res_mode, voxel_size=voxel_size)
                if len(d) >= 2:  # Only keep if at least 2 points
                    densified.append(d)
            except Exception as e:
                print(f"Error densifying streamline {i}: {e}")
        return densified
    else:
        # Parallel processing with shared memory
        def _process_one(streamline, idx, total):
            if idx % 1000 == 0:
                print(f"Processing streamline {idx}/{total}...")
            try:
                # Import numpy locally within the processing function
                import numpy as np
                
                # First check the type of streamline
                streamline_type = type(streamline)
                
                # Ensure we're passing a numpy array
                if not isinstance(streamline, np.ndarray):
                    try:
                        # Convert the streamline to a numpy array
                        streamline = np.array(streamline, dtype=np.float32)
                        
                        # Additional check to make sure conversion worked 
                        if not isinstance(streamline, np.ndarray):
                            print(f"Error densifying streamline {idx}: Failed to convert to numpy array, type is still {type(streamline)}")
                            return None
                    except Exception as conversion_error:
                        print(f"Error densifying streamline {idx}: Could not convert to numpy array - {conversion_error}")
                        return None
                
                # Now pass numpy array to the densify function
                return densify_streamline_subvoxel(streamline, step_size, use_gpu, interp_method, 
                                                 high_res_mode=high_res_mode, voxel_size=voxel_size)
            except Exception as e:
                print(f"Error densifying streamline {idx}: {e}")
                print(f"Streamline type was: {streamline_type if 'streamline_type' in locals() else 'unknown'}")
                return None
        
        total = len(streamlines)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_one)(streamline, idx, total) 
            for idx, streamline in enumerate(streamlines)
        )
        
        # Filter out None results and ensure each streamline has at least 2 points
        densified = [r for r in results if r is not None and len(r) >= 2]
        
        # Report on densification results
        if high_res_mode:
            print(f"\n[HIGH-RES DENSIFY] Processed {len(densified)}/{total} streamlines successfully")
            if len(densified) < total:
                print(f"[HIGH-RES DENSIFY] {total - len(densified)} streamlines were filtered out")
        else:
            print(f"Densified {len(densified)}/{total} streamlines successfully")
        
        return densified

def densify_streamline_subvoxel(streamline, step_size, use_gpu=True, interp_method='hermite', high_res_mode=False, voxel_size=1.0):
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
    high_res_mode : bool, optional
        Special high-resolution processing mode, by default False.
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
    if not isinstance(streamline, np.ndarray):
        try:
            streamline = np.array(streamline, dtype=np.float32)
        except Exception as e:
            raise TypeError(f"Failed to convert input to numpy array: {e}")

    # Check if streamline is valid
    if len(streamline) < 2:
        print(f"Warning: Cannot densify streamline with less than 2 points (has {len(streamline)})")
        return streamline

    # Check for debug flags
    debug_tangents = os.environ.get("DEBUG_TANGENTS") == "1" or high_res_mode

    # Debug: Print interpolation method being used
    if debug_tangents:
        print(f"[DENSIFY] Using {interp_method} interpolation with step size {step_size}, voxel size {voxel_size}mm")
        print(f"[DENSIFY] Streamline points: {len(streamline)}")
    
    # Choose appropriate array library based on use_gpu flag
    if use_gpu:
        try:
            import cupy as xp
            # Convert to cupy array
            try:
                streamline_device = xp.asarray(streamline)
            except Exception as e:
                print(f"CuPy conversion error: {e}")
                print("Falling back to CPU")
                xp = np
                streamline_device = xp.asarray(streamline)
                use_gpu = False
        except ImportError:
            print("Warning: Could not import cupy. Falling back to CPU.")
            xp = np
            streamline_device = xp.asarray(streamline)
            use_gpu = False
    else:
        xp = np
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
    n_steps = int(xp.ceil(total_length / step_size)) + 1
    
    # For high-res mode, limit maximum steps per streamline
    if high_res_mode and n_steps > 10000:
        old_n_steps = n_steps
        n_steps = min(10000, n_steps)
        print(f"[HIGH-RES DENSIFY] Limiting steps from {old_n_steps} to {n_steps}")
    
    # Ensure at least one interpolation step
    n_steps = max(len(streamline), n_steps)
    
    # Calculate cumulative distance along the streamline
    cumulative_lengths = xp.concatenate(([0], xp.cumsum(segment_lengths)))
    normalized_distances = cumulative_lengths / total_length
    
    # Create evenly spaced points along the streamline
    xi = xp.linspace(0, 1, n_steps)
    
    # Check for NaN values
    if xp.any(xp.isnan(normalized_distances)) or xp.any(xp.isnan(xi)):
        print("Warning: NaN values detected. Using original streamline.")
        return streamline
    
    # Calculate tangents for Hermite interpolation
    if interp_method == 'hermite':
        tangents = xp.zeros_like(streamline_device)
        
        # Calculate tangent scaling
        base_scale = 1.2 if high_res_mode else 1.0
        voxel_scale = min(5.0, 1.0 / max(0.01, voxel_size))
        final_scale = base_scale * voxel_scale
        
        # Calculate tangents for interior points
        tangents[1:-1] = (streamline_device[2:] - streamline_device[:-2]) * 0.5 * final_scale
        
        # Calculate endpoint tangents
        tangents[0] = (streamline_device[1] - streamline_device[0]) * final_scale
        tangents[-1] = (streamline_device[-1] - streamline_device[-2]) * final_scale
        
        if not high_res_mode:
            # Normalize tangents but preserve some curvature
            tangent_norms = xp.sqrt(xp.sum(tangents**2, axis=1, keepdims=True))
            tangent_norms = xp.where(tangent_norms > 1e-10, tangent_norms, 1e-10)
            normalization_factor = max(0.5, min(1.0, voxel_size))
            tangents = tangents / (tangent_norms * normalization_factor)

    # Interpolate each coordinate
    result = []
    for dim in range(streamline_device.shape[1]):
        y = streamline_device[:, dim]
        
        if interp_method == 'hermite':
            try:
                # Always use numpy/scipy for Hermite interpolation
                from scipy.interpolate import CubicHermiteSpline
                t_values = xp.asnumpy(normalized_distances) if use_gpu else normalized_distances
                y_values = xp.asnumpy(y) if use_gpu else y
                tangent_values = xp.asnumpy(tangents[:, dim]) if use_gpu else tangents[:, dim]
                interp_points = xp.asnumpy(xi) if use_gpu else xi
                
                interpolator = CubicHermiteSpline(t_values, y_values, tangent_values)
                interpolated = interpolator(interp_points)
                
                # Convert back to device array if needed
                interpolated = xp.asarray(interpolated) if use_gpu else interpolated
            except Exception as e:
                print(f"Hermite interpolation failed: {e}. Falling back to linear.")
                interpolated = xp.interp(xi, normalized_distances, y)
        else:
            interpolated = xp.interp(xi, normalized_distances, y)
            
        result.append(interpolated)
    
    # Combine the interpolated coordinates
    densified_streamline = xp.stack(result, axis=1)
    
    # Convert back to numpy if on GPU
    if use_gpu:
        try:
            densified_streamline = xp.asnumpy(densified_streamline)
        except Exception as e:
            print(f"Error converting back to numpy: {e}")
            # If conversion fails, ensure we're returning a numpy array
            densified_streamline = np.array(densified_streamline, dtype=np.float32)
    
    # Final check to ensure we're returning a numpy array
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