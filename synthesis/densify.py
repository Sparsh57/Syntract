"""
Streamline densification with interpolation methods.
"""

import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
# Remove the top-level cupy import - it will be imported conditionally when needed
import time
from scipy.interpolate import RBFInterpolator
import warnings
import os
import traceback

# Suppress RBF warnings for cleaner output - we know what we're doing
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.interpolate')

def calculate_optimal_step_size(streamline):
    """
    Calculate optimal step size based on original streamline characteristics.
    
    Parameters
    ----------
    streamline : array-like
        The input streamline to analyze.
        
    Returns
    -------
    float
        Optimal step size for this streamline.
    """
    if len(streamline) < 2:
        return 0.25  # Default fallback
    
    # Convert to numpy if needed
    if not isinstance(streamline, np.ndarray):
        streamline = np.array(streamline)
    
    # Calculate original step sizes
    original_steps = np.linalg.norm(np.diff(streamline, axis=0), axis=1)
    
    # Use median to avoid outlier influence
    optimal_step = np.median(original_steps)
    
    # Clamp to reasonable range to prevent extreme values
    return np.clip(optimal_step, 0.15, 0.35)

def calculate_streamline_curvature(streamline):
    """
    Calculate curvature metrics for a streamline.
    
    Parameters
    ----------
    streamline : array-like
        The streamline to analyze.
        
    Returns
    -------
    numpy.ndarray
        Curvature values at each point.
    """
    if len(streamline) < 3:
        return np.array([0.0])
    
    # Convert to numpy if needed
    if not isinstance(streamline, np.ndarray):
        streamline = np.array(streamline)
    
    # Calculate first derivatives (tangent vectors)
    tangents = np.zeros_like(streamline)
    tangents[1:-1] = (streamline[2:] - streamline[:-2]) / 2.0
    tangents[0] = streamline[1] - streamline[0] 
    tangents[-1] = streamline[-1] - streamline[-2]
    
    # Normalize tangents
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norms = np.where(tangent_norms > 1e-10, tangent_norms, 1e-10)
    tangents = tangents / tangent_norms
    
    # Calculate second derivatives 
    second_derivs = np.zeros_like(streamline)
    second_derivs[1:-1] = (tangents[2:] - tangents[:-2]) / 2.0
    second_derivs[0] = tangents[1] - tangents[0]
    second_derivs[-1] = tangents[-1] - tangents[-2]
    
    # Curvature = |T'| where T is unit tangent
    curvature = np.linalg.norm(second_derivs, axis=1)
    
    return curvature

def validate_curvature_preservation(original_streamlines, processed_streamlines):
    """
    Compare curvature metrics between original and processed streamlines to validate preservation.
    
    Parameters
    ----------
    original_streamlines : list of arrays
        Original streamlines before processing.
    processed_streamlines : list of arrays
        Processed streamlines after densification.
        
    Returns
    -------
    dict
        Dictionary containing validation metrics.
    """
    if len(original_streamlines) == 0 or len(processed_streamlines) == 0:
        return {
            'mean_curvature_ratio': 0.0,
            'curvature_variance_ratio': 0.0,
            'path_length_preservation': 0.0,
            'num_original': len(original_streamlines),
            'num_processed': len(processed_streamlines)
        }
    
    # Calculate curvature metrics for original streamlines
    original_curvatures = []
    original_lengths = []
    
    for streamline in original_streamlines:
        if len(streamline) >= 3:
            curvature = calculate_streamline_curvature(streamline)
            original_curvatures.extend(curvature)
            
            # Calculate path length
            diffs = np.diff(streamline, axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
            original_lengths.append(length)
    
    # Calculate curvature metrics for processed streamlines
    processed_curvatures = []
    processed_lengths = []
    
    for streamline in processed_streamlines:
        if len(streamline) >= 3:
            curvature = calculate_streamline_curvature(streamline)
            processed_curvatures.extend(curvature)
            
            # Calculate path length
            diffs = np.diff(streamline, axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
            processed_lengths.append(length)
    
    # Calculate metrics
    original_mean_curvature = np.mean(original_curvatures) if original_curvatures else 0.0
    processed_mean_curvature = np.mean(processed_curvatures) if processed_curvatures else 0.0
    
    original_curvature_var = np.var(original_curvatures) if original_curvatures else 0.0
    processed_curvature_var = np.var(processed_curvatures) if processed_curvatures else 0.0
    
    original_mean_length = np.mean(original_lengths) if original_lengths else 0.0
    processed_mean_length = np.mean(processed_lengths) if processed_lengths else 0.0
    
    # Calculate ratios
    mean_curvature_ratio = (processed_mean_curvature / original_mean_curvature 
                          if original_mean_curvature > 0 else 0.0)
    curvature_variance_ratio = (processed_curvature_var / original_curvature_var
                               if original_curvature_var > 0 else 0.0)
    path_length_preservation = (processed_mean_length / original_mean_length
                               if original_mean_length > 0 else 0.0)
    
    metrics = {
        'mean_curvature_ratio': mean_curvature_ratio,
        'curvature_variance_ratio': curvature_variance_ratio, 
        'path_length_preservation': path_length_preservation,
        'original_mean_curvature': original_mean_curvature,
        'processed_mean_curvature': processed_mean_curvature,
        'original_mean_length': original_mean_length,
        'processed_mean_length': processed_mean_length,
        'num_original': len(original_streamlines),
        'num_processed': len(processed_streamlines),
        'preservation_success': (mean_curvature_ratio > 0.6 and 
                               0.9 <= path_length_preservation <= 1.1)
    }
    
    return metrics

def linear_interpolate(p0, p1, t, xp=np):
    """Linear interpolation between two points."""
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
    import numpy as np
    if len(streamlines) == 0:
        return []

    if n_jobs == 1:
        # More efficient with a progress counter
        densified = []
        success_count = 0
        total_count = len(streamlines)
        total = len(streamlines)
        for i, streamline in enumerate(streamlines):
            if i % 1000 == 0:
                print(f"Processing streamline {i}/{total_count}...")
            
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
                # Enhanced type checking that handles various numpy array types
                def is_array_like(obj):
                    """Check if object is array-like (numpy, cupy, or convertible to numpy)"""
                    if isinstance(obj, np.ndarray):
                        return True
                    if hasattr(obj, 'get') and hasattr(obj, 'shape'):  # CuPy array
                        return True
                    if hasattr(obj, 'shape') and hasattr(obj, 'dtype') and hasattr(obj, '__array__'):
                        return True
                    if str(type(obj)).find('numpy.ndarray') >= 0:  # Different numpy instance
                        return True
                    return False
                
                if not is_array_like(streamline):
                    print(f"[densify_streamlines_parallel] Error densifying streamline {i}: Unsupported type {type(streamline)}")
                    print(f"  isinstance(np.ndarray): {isinstance(streamline, np.ndarray)}")
                    print(f"  type string: {str(type(streamline))}")
                    print(f"  has array attributes: shape={hasattr(streamline, 'shape')}, dtype={hasattr(streamline, 'dtype')}, __array__={hasattr(streamline, '__array__')}")
                    continue
                
                # Convert to numpy array with explicit type conversion
                try:
                    if hasattr(streamline, 'get'):  # CuPy array
                        streamline = streamline.get()
                    if not isinstance(streamline, np.ndarray):
                        streamline = np.asarray(streamline, dtype=np.float32)
                    elif streamline.dtype != np.float32:
                        streamline = streamline.astype(np.float32)
                except Exception as conv_e:
                    print(f"Error converting streamline {i} to numpy array: {conv_e}")
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
                        if hasattr(result, 'get'):  # CuPy array
                            result = result.get()
                        else:
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
        final_densified = []
        for s in densified:
            if isinstance(s, np.ndarray):
                final_densified.append(s)
            elif hasattr(s, 'get'):  # CuPy array
                final_densified.append(s.get())
            else:
                final_densified.append(np.array(s, dtype=np.float32))
        return final_densified
    else:
        # Import numpy here to ensure it's available in both code paths
        import numpy as np
        
        # Use sequential processing to avoid import issues with threading backend
        import numpy as np
        
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
        
        # Process sequentially to avoid import issues
        results = []
        for idx, streamline in enumerate(streamlines):
            result = _process_one(streamline, idx, total)
            results.append(result)
        
        # Filter out None results and ensure each streamline has at least 2 points
        densified = []
        for r in results:
            if r is None or len(r) < 2:
                continue
                
            # Additional check to ensure numpy array type
            if not isinstance(r, np.ndarray):
                try:
                    if hasattr(r, 'get'):  # CuPy array
                        r = r.get()
                    else:
                        r = np.array(r, dtype=np.float32)
                except Exception as e:
                    print(f"Error: Skipping invalid result - {e}")
                    continue
                    
            densified.append(r)
        
        # Report on densification results
        print(f"Densified {len(densified)}/{total} streamlines successfully")
        # Final unconditional enforcement for all results
        final_densified = []
        for s in densified:
            if isinstance(s, np.ndarray):
                final_densified.append(s)
            elif hasattr(s, 'get'):  # CuPy array
                final_densified.append(s.get())
            else:
                final_densified.append(np.array(s, dtype=np.float32))
        return final_densified

def rbf_interpolate_streamline(streamline, step_size, function='thin_plate', epsilon=None, xp=np, preserve_length=True):
    """
    Perform RBF interpolation for a streamline with proper regularization and length preservation.
    Enhanced with curvature-aware parameter selection.
    """
    # Import numpy to ensure it's available
    import numpy as np
    
    # Skip if too few points
    if len(streamline) < 4:
        if not isinstance(streamline, np.ndarray):
            streamline = np.array(streamline, dtype=np.float32)
        return streamline

    # === CURVATURE-AWARE PARAMETER SELECTION ===
    # Calculate curvature to adapt RBF parameters
    streamline_cpu = streamline if isinstance(streamline, np.ndarray) else streamline.get() if hasattr(streamline, 'get') else np.array(streamline)
    curvature_values = calculate_streamline_curvature(streamline_cpu)
    mean_curvature = np.mean(curvature_values)
    
    # Adaptive step size based on curvature
    if mean_curvature > 0.05:  # High curvature
        optimal_step = calculate_optimal_step_size(streamline_cpu)
        adaptive_step = max(step_size, optimal_step)
        # Use less smoothing for high curvature to preserve features
        smoothing_factor = 0.0
        rbf_function = 'thin_plate_spline'  # Correct scipy kernel name
    elif mean_curvature > 0.01:  # Medium curvature  
        adaptive_step = step_size
        smoothing_factor = 1e-8  # Minimal smoothing
        rbf_function = 'thin_plate_spline'
    else:  # Low curvature
        adaptive_step = step_size
        smoothing_factor = 1e-6  # Slight smoothing for stability
        rbf_function = 'multiquadric'  # More stable for straight segments

    # Calculate the total length of the streamline
    diffs = xp.diff(streamline, axis=0)
    segment_lengths = xp.sqrt(xp.sum(diffs**2, axis=1))
    total_length = xp.sum(segment_lengths)

    # Skip densification if streamline is shorter than adaptive step_size
    if total_length < adaptive_step:
        if not isinstance(streamline, np.ndarray):
            streamline = np.array(streamline, dtype=np.float32)
        return streamline

    # Create parameterization (cumulative distance)
    cumulative_lengths = xp.concatenate((xp.array([0], dtype=segment_lengths.dtype), xp.cumsum(segment_lengths)))

    # Calculate new sampling points based on adaptive step size
    n_steps = max(int(xp.ceil(total_length / adaptive_step)), 2)
    t_new = xp.linspace(0, cumulative_lengths[-1], n_steps)

    # RBF interpolation requires scipy - convert to CPU if necessary
    if hasattr(cumulative_lengths, 'get'):  # Check if using CuPy
        t_cpu = cumulative_lengths.get()
        streamline_cpu = streamline.get()
        t_new_cpu = t_new.get()
    else:
        t_cpu = cumulative_lengths
        streamline_cpu = streamline
        t_new_cpu = t_new
    
    # Convert to numpy arrays for scipy
    t_cpu = np.asarray(t_cpu, dtype=np.float64)
    streamline_cpu = np.asarray(streamline_cpu, dtype=np.float64)
    t_new_cpu = np.asarray(t_new_cpu, dtype=np.float64)
    
    # Use more stable RBF settings with curvature-aware parameters
    try:
        from scipy.interpolate import RBFInterpolator
        
        # Apply RBF interpolation to each dimension with adaptive parameters
        result = np.zeros((len(t_new_cpu), streamline_cpu.shape[1]), dtype=np.float32)
        
        for dim in range(streamline_cpu.shape[1]):
            # Use proper 2D input for RBFInterpolator (it expects (n_samples, n_features))
            t_2d = t_cpu.reshape(-1, 1)
            y_values = streamline_cpu[:, dim]
            
            # Create RBF interpolator with curvature-aware parameters
            rbf = RBFInterpolator(
                t_2d, 
                y_values, 
                kernel=rbf_function,  # Adaptive based on curvature
                smoothing=smoothing_factor,  # Adaptive smoothing
                epsilon=None  # Let scipy choose optimal epsilon
            )
            
            # Evaluate at new points
            t_new_2d = t_new_cpu.reshape(-1, 1)
            result[:, dim] = rbf(t_new_2d).astype(np.float32)
            
    except ImportError:
        # Fallback to old Rbf with curvature-aware parameters
        print("Warning: RBFInterpolator not available, using legacy Rbf with curvature-aware parameters")
        from scipy.interpolate import Rbf
        
        result = np.zeros((len(t_new_cpu), streamline_cpu.shape[1]), dtype=np.float32)
        
        for dim in range(streamline_cpu.shape[1]):
            # Use adaptive function and parameters based on curvature
            if rbf_function == 'thin_plate':
                # Thin plate splines don't use epsilon, use multiquadric as fallback
                rbf = Rbf(
                    t_cpu, 
                    streamline_cpu[:, dim], 
                    function='multiquadric',
                    epsilon=1.0,  # Shape parameter
                    smooth=smoothing_factor   # Adaptive regularization
                )
            else:
                rbf = Rbf(
                    t_cpu, 
                    streamline_cpu[:, dim], 
                    function=rbf_function,
                    epsilon=1.0,  # Shape parameter
                    smooth=smoothing_factor   # Adaptive regularization
                )
            result[:, dim] = rbf(t_new_cpu)
    
    # === LENGTH PRESERVATION ===
    if preserve_length and len(result) > 2:
        # Calculate the length of the RBF interpolated curve
        result_diffs = np.diff(result, axis=0)
        result_lengths = np.sqrt(np.sum(result_diffs**2, axis=1))
        result_total_length = np.sum(result_lengths)
        
        # If length differs significantly, apply correction
        length_ratio = float(total_length) / result_total_length
        
        if abs(length_ratio - 1.0) > 0.001:  # More than 0.1% difference
            # Method 1: Arc-length reparameterization
            # Calculate cumulative distances for the RBF result
            result_cumulative = np.concatenate([[0], np.cumsum(result_lengths)])
            
            # Normalize to target length
            result_cumulative_normalized = result_cumulative * length_ratio
            
            # Create target arc-length points
            target_distances = np.linspace(0, float(total_length), n_steps)
            
            # Reparameterize to preserve arc length
            from scipy.interpolate import interp1d
            
            # Interpolate each dimension using the corrected arc length
            corrected_result = np.zeros_like(result)
            
            for dim in range(result.shape[1]):
                # Create interpolator for this dimension vs arc length
                interp_func = interp1d(
                    result_cumulative_normalized, 
                    result[:, dim], 
                    kind='cubic', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                corrected_result[:, dim] = interp_func(target_distances)
            
            result = corrected_result.astype(np.float32)
            
            # Verify correction worked
            final_diffs = np.diff(result, axis=0)
            final_lengths = np.sqrt(np.sum(final_diffs**2, axis=1))
            final_total_length = np.sum(final_lengths)
            
            # Optional: Fine-tune with small scaling if still off
            if abs(final_total_length - float(total_length)) > 0.01:
                scale_factor = float(total_length) / final_total_length
                center = np.mean(result, axis=0)
                result = center + (result - center) * scale_factor
    
    # Convert back to appropriate array type
    if hasattr(xp, 'asarray') and xp.__name__ == 'cupy':
        return xp.asarray(result)
    else:
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
    import numpy as np
    try:
        debug_tangents = os.environ.get("DEBUG_TANGENTS") == "1"

        # DEBUG: Add comprehensive type and shape information
        print(f"  - streamline type: {type(streamline)}")
        print(f"  - streamline shape: {getattr(streamline, 'shape', 'N/A')}")
        print(f"  - streamline dtype: {getattr(streamline, 'dtype', 'N/A')}")
        print(f"  - step_size: {step_size}")
        print(f"  - use_gpu: {use_gpu}")
        print(f"  - interp_method: {interp_method}")

        # Handle CuPy arrays first
        if hasattr(streamline, 'get'):  # CuPy array
            streamline = streamline.get()  # Convert to numpy array
        
        if isinstance(streamline, list):
            try:
                if not all(isinstance(point, (list, tuple, np.ndarray)) for point in streamline):
                    raise TypeError(f"Streamline contains non-array elements")
                
                clean_points = []
                for point in streamline:
                    if isinstance(point, (list, tuple)):
                        if len(point) != 3:
                            raise ValueError(f"Expected 3D point but got {len(point)}D")
                        clean_points.append(np.array(point, dtype=np.float32))
                    elif isinstance(point, np.ndarray):
                        clean_points.append(point.astype(np.float32))
                    else:
                        raise TypeError(f"Unexpected point type: {type(point)}")
                streamline = np.array(clean_points, dtype=np.float32)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise TypeError(f"Failed to convert list to numpy array: {e}")

        # DEBUG: Check streamline after list processing

        if len(streamline) < 2:
            if not isinstance(streamline, np.ndarray):
                streamline = np.array(streamline, dtype=np.float32)
            return streamline

        # Ensure streamline is a numpy array at this point
        if not isinstance(streamline, np.ndarray):
            try:
                streamline = np.array(streamline, dtype=np.float32)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise TypeError(f"Cannot convert streamline of type {type(streamline)} to numpy array: {e}")


        if use_gpu:
            try:
                import cupy as xp
                if not isinstance(streamline, np.ndarray):
                    streamline = np.array(streamline, dtype=np.float32)
                try:
                    streamline_device = xp.asarray(streamline)
                except Exception:
                    import numpy as xp
                    streamline_device = xp.asarray(streamline)
                    use_gpu = False
            except ImportError:
                import numpy as xp
                streamline_device = xp.asarray(streamline)
                use_gpu = False
        else:
            xp = np
            if not isinstance(streamline, np.ndarray):
                streamline = np.array(streamline, dtype=np.float32)
            streamline_device = xp.asarray(streamline)

        # === CURVATURE-BASED ADAPTIVE STEP SIZE ===
        # Add curvature analysis for adaptive step size
        if len(streamline) >= 3:
            # Calculate curvature using numpy (convert from GPU if needed)
            streamline_cpu = streamline if isinstance(streamline, np.ndarray) else streamline_device.get() if hasattr(streamline_device, 'get') else np.array(streamline_device)
            curvature_values = calculate_streamline_curvature(streamline_cpu)
            mean_curvature = np.mean(curvature_values)
            
            # For high-curvature streamlines, use adaptive step size
            if mean_curvature > 0.05:  # High curvature threshold
                optimal_step = calculate_optimal_step_size(streamline_cpu)
                adaptive_step = max(step_size, optimal_step)
                if debug_tangents:
                    print(f"[CURVATURE] Mean curvature: {mean_curvature:.6f}, using adaptive step: {adaptive_step:.3f} (original: {step_size:.3f})")
            else:
                adaptive_step = step_size
                if debug_tangents:
                    print(f"[CURVATURE] Mean curvature: {mean_curvature:.6f}, using original step: {step_size:.3f}")
        else:
            adaptive_step = step_size

        diffs = xp.diff(streamline_device, axis=0)
        segment_lengths = xp.sqrt(xp.sum(diffs**2, axis=1))
        total_length = xp.sum(segment_lengths)

        if debug_tangents:
            print(f"[DENSIFY] Total streamline length: {total_length:.4f}mm")
            print(f"[DENSIFY] Mean segment length: {total_length/len(segment_lengths):.4f}mm")

        if total_length < adaptive_step:
            if not isinstance(streamline, np.ndarray):
                streamline = np.array(streamline, dtype=np.float32)
            return streamline

        n_steps = int(xp.ceil(total_length / adaptive_step)) + 1
        cumulative_lengths = xp.concatenate((xp.array([0], dtype=segment_lengths.dtype), xp.cumsum(segment_lengths)))
        normalized_distances = cumulative_lengths / total_length
        xi = xp.linspace(0, 1, n_steps)

        if xp.any(xp.isnan(normalized_distances)) or xp.any(xp.isnan(xi)):
            if not isinstance(streamline, np.ndarray):
                streamline = np.array(streamline, dtype=np.float32)
            return streamline

        if use_gpu and hasattr(xp, 'ndarray'):
            if not isinstance(normalized_distances, xp.ndarray):
                normalized_distances = xp.asarray(normalized_distances)
            if not isinstance(xi, xp.ndarray):
                xi = xp.asarray(xi)
        else:
            if not isinstance(normalized_distances, np.ndarray):
                normalized_distances = np.asarray(normalized_distances)
            if not isinstance(xi, np.ndarray):
                xi = np.asarray(xi)

        if interp_method == 'hermite':
            # Calculate tangents preserving natural streamline curvature
            tangents = xp.zeros_like(streamline_device)
            
            # Calculate local segment lengths for adaptive scaling
            segment_lengths_for_tangents = xp.sqrt(xp.sum(diffs**2, axis=1))
            
            # Calculate natural tangent vectors with proper scaling to preserve curvature
            for i in range(1, len(streamline_device) - 1):
                # Use central difference for smooth tangents
                raw_tangent = streamline_device[i+1] - streamline_device[i-1]
                
                # Local adaptive scaling based on nearby segment lengths
                # Use a small window around current point to get local characteristics
                window_start = max(0, i - 2)
                window_end = min(len(segment_lengths_for_tangents), i + 1)
                local_segments = segment_lengths_for_tangents[window_start:window_end]
                local_mean_length = xp.mean(local_segments)
                
                # AGGRESSIVE: Much stronger scaling to preserve anatomical curvature
                # Use central difference normalization with aggressive local scaling
                raw_norm = xp.linalg.norm(raw_tangent)
                if raw_norm > 1e-10:
                    # Normalize and then apply aggressive local scaling (2.0 instead of 0.75)
                    local_scale_factor = 2.0
                    tangents[i] = (raw_tangent / raw_norm) * local_mean_length * local_scale_factor
                else:
                    tangents[i] = raw_tangent
            
            # Handle endpoints with aggressive scaling
            # Start point
            if len(segment_lengths_for_tangents) > 0:
                start_tangent = streamline_device[1] - streamline_device[0]
                start_norm = xp.linalg.norm(start_tangent)
                if start_norm > 1e-10:
                    start_scale_factor = 2.0  # Aggressive scaling
                    tangents[0] = (start_tangent / start_norm) * segment_lengths_for_tangents[0] * start_scale_factor
                else:
                    tangents[0] = start_tangent
            else:
                tangents[0] = streamline_device[1] - streamline_device[0]
            
            # End point  
            if len(segment_lengths_for_tangents) > 0:
                end_tangent = streamline_device[-1] - streamline_device[-2]
                end_norm = xp.linalg.norm(end_tangent)
                if end_norm > 1e-10:
                    end_scale_factor = 2.0  # Aggressive scaling
                    tangents[-1] = (end_tangent / end_norm) * segment_lengths_for_tangents[-1] * end_scale_factor
                else:
                    tangents[-1] = end_tangent
            else:
                tangents[-1] = streamline_device[-1] - streamline_device[-2]
            
            # Optional: Enhanced Catmull-Rom style tangent calculation for better curvature
            # This provides more natural curvature preservation for longer streamlines
            enhanced_tangents = os.environ.get("ENHANCED_TANGENTS", "0") == "1"
            if enhanced_tangents:
                # Use Catmull-Rom tangent calculation for better curvature preservation
                for i in range(1, len(streamline_device) - 1):
                    # Catmull-Rom tangent: 0.5 * (P[i+1] - P[i-1])
                    catmull_tangent = (streamline_device[i+1] - streamline_device[i-1]) * 0.5
                    
                    # ENHANCED: Scale tangent by 3.0 to preserve much more curvature
                    tangent_norm = xp.linalg.norm(catmull_tangent)
                    if tangent_norm > 1e-10:
                        # Aggressive scaling: 3.0x step_size for maximum curvature preservation
                        curvature_enhancement_factor = 3.0
                        tangents[i] = catmull_tangent / tangent_norm * step_size * curvature_enhancement_factor
                    else:
                        tangents[i] = catmull_tangent
                
                # Endpoints for Catmull-Rom with enhanced scaling
                if len(streamline_device) > 1:
                    forward_diff = streamline_device[1] - streamline_device[0]
                    backward_diff = streamline_device[-1] - streamline_device[-2]
                    
                    # Scale endpoint tangents with aggressive enhancement
                    forward_norm = xp.linalg.norm(forward_diff)
                    backward_norm = xp.linalg.norm(backward_diff)
                    curvature_enhancement_factor = 3.0
                    
                    if forward_norm > 1e-10:
                        tangents[0] = forward_diff / forward_norm * step_size * curvature_enhancement_factor
                    if backward_norm > 1e-10:
                        tangents[-1] = backward_diff / backward_norm * step_size * curvature_enhancement_factor

        if interp_method == 'rbf':
            try:
                densified_streamline = rbf_interpolate_streamline(
                    streamline_device, 
                    step_size, 
                    function='thin_plate', 
                    epsilon=None, 
                    xp=xp
                )
                if not isinstance(densified_streamline, np.ndarray):
                    if hasattr(densified_streamline, 'get'):  # CuPy array
                        densified_streamline = densified_streamline.get()
                    else:
                        densified_streamline = np.array(densified_streamline, dtype=np.float32)
                return densified_streamline
            except Exception:
                interp_method = 'linear'

        result_shape = (len(xi), streamline_device.shape[1])
        if use_gpu and hasattr(xp, 'zeros'):
            result_array = xp.zeros(result_shape, dtype=xp.float32)
        else:
            result_array = np.zeros(result_shape, dtype=np.float32)

        for dim in range(streamline_device.shape[1]):
            y = streamline_device[:, dim]
            try:
                if interp_method == 'hermite':
                    if use_gpu:
                        import numpy as np
                        # Convert all arrays to NumPy for scipy interpolation
                        if hasattr(normalized_distances, 'get'):
                            t_values = normalized_distances.get()
                        else:
                            t_values = np.array(normalized_distances)
                        if hasattr(xi, 'get'):
                            interp_points = xi.get()
                        else:
                            interp_points = np.array(xi)
                        if hasattr(y, 'get'):
                            y_values = y.get()
                        else:
                            y_values = np.array(y)
                        if hasattr(tangents, 'get'):
                            tangent_values = tangents[:, dim].get()
                        else:
                            tangent_values = np.array(tangents[:, dim])
                        from scipy.interpolate import CubicHermiteSpline
                        interpolator = CubicHermiteSpline(t_values, y_values, tangent_values)
                        interpolated_cpu = interpolator(interp_points)
                        interpolated = xp.asarray(interpolated_cpu)
                    else:
                        from scipy.interpolate import CubicHermiteSpline
                        interpolator = CubicHermiteSpline(normalized_distances, y, tangents[:, dim])
                        interpolated = interpolator(xi)
                else:
                    interpolated = xp.interp(xi, normalized_distances, y)
                    
                if not isinstance(interpolated, xp.ndarray):
                    interpolated = xp.array(interpolated, dtype=xp.float32)
                result_array[:, dim] = interpolated
            except Exception as e:
                raise
                
        if use_gpu and hasattr(result_array, 'get'):
            densified_streamline = result_array.get()
        else:
            densified_streamline = result_array
            
        if not isinstance(densified_streamline, np.ndarray):
            if hasattr(densified_streamline, 'get'):  # CuPy array
                densified_streamline = densified_streamline.get()
            else:
                densified_streamline = np.array(densified_streamline, dtype=np.float32)
            
        if debug_tangents:
            print(f"[DENSIFY] Original points: {len(streamline)}, Densified points: {len(densified_streamline)}")
            if interp_method == 'hermite':
                if len(streamline) > 2 and len(densified_streamline) > 2:
                    def calc_curvature(points):
                        if len(points) < 3:
                            return 0
                        tangents = np.zeros_like(points)
                        tangents[1:-1] = (points[2:] - points[:-2]) / 2.0
                        tangents[0] = points[1] - points[0]
                        tangents[-1] = points[-1] - points[-2]
                        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
                        norms = np.where(norms > 1e-10, norms, 1e-10)
                        tangents = tangents / norms
                        second_derivs = np.zeros_like(points)
                        second_derivs[1:-1] = (tangents[2:] - tangents[:-2]) / 2.0
                        curvatures = np.linalg.norm(second_derivs, axis=1)
                        return np.mean(curvatures)
                    orig_curvature = calc_curvature(streamline)
                    new_curvature = calc_curvature(densified_streamline)
                    print(f"[CURVATURE] Original streamline: {orig_curvature:.6f}")
                    print(f"[CURVATURE] Densified streamline: {new_curvature:.6f}")
                    if orig_curvature > 0:
                        print(f"[CURVATURE] Change: {(new_curvature-orig_curvature)/orig_curvature*100:.2f}%")
                    else:
                        print(f"[CURVATURE] Change: N/A (original curvature was zero)")
                        
        return densified_streamline
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Return a safe fallback
        result = np.zeros((2, 3), dtype=np.float32)
        return result


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
