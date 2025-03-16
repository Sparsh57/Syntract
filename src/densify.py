import numpy as np
from joblib import Parallel, delayed
import os


def linear_interpolate(p0, p1, t, xp=np):
    """
    Performing linear interpolation between two points.

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
    return p0 + t * (p1 - p0)


def hermite_interpolate(p0, p1, m0, m1, t, xp=np):
    """
    Performing cubic Hermite interpolation between two points.

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
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def calculate_streamline_metrics(streamlines, metrics=None):
    """
    Calculating metrics for a set of streamlines.

    Parameters
    ----------
    streamlines : list
        List of streamlines, each an array of shape (N, 3).
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
    for stream in streamlines:
        if len(stream) < 3:
            continue
        if not isinstance(stream, np.ndarray):
            stream = np.array(stream)
        tangents = np.zeros_like(stream)
        tangents[1:-1] = (stream[2:] - stream[:-2]) / 2.0
        tangents[0] = stream[1] - stream[0]
        tangents[-1] = stream[-1] - stream[-2]
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_norms = np.where(tangent_norms > 1e-12, tangent_norms, 1.0)
        tangents = tangents / tangent_norms

        if 'curvature' in metrics:
            second_derivs = np.zeros_like(stream)
            second_derivs[1:-1] = (tangents[2:] - tangents[:-2]) / 2.0
            second_derivs[0] = tangents[1] - tangents[0]
            second_derivs[-1] = tangents[-1] - tangents[-2]
            curvature = np.linalg.norm(second_derivs, axis=1)
            results['curvature'].append(curvature)
            results['mean_curvature'] += np.sum(curvature)
            results['max_curvature'] = max(results['max_curvature'], np.max(curvature))

        if 'length' in metrics:
            segment_lengths = np.linalg.norm(stream[1:] - stream[:-1], axis=1)
            length = np.sum(segment_lengths)
            results['length'].append(length)
            results['total_length'] += length

        if 'torsion' in metrics and len(stream) > 3:
            third_derivs = np.zeros_like(stream)
            third_derivs[1:-1] = (second_derivs[2:] - second_derivs[:-2]) / 2.0
            third_derivs[0] = second_derivs[1] - second_derivs[0]
            third_derivs[-1] = second_derivs[-1] - second_derivs[-2]
            cross_products = np.zeros_like(stream)
            for i in range(1, len(stream) - 1):
                cross_products[i] = np.cross(tangents[i], second_derivs[i])
            torsion = np.zeros(len(stream))
            for i in range(1, len(stream) - 1):
                if np.linalg.norm(cross_products[i]) > 1e-12:
                    torsion[i] = np.dot(cross_products[i], third_derivs[i]) / np.linalg.norm(cross_products[i]) ** 2
            results['torsion'].append(torsion)
            results['mean_torsion'] += np.sum(np.abs(torsion))

        total_points += len(stream)

    if total_points > 0:
        if 'curvature' in metrics:
            results['mean_curvature'] /= total_points
        if 'torsion' in metrics:
            results['mean_torsion'] /= total_points
    if len(results.get('length', [])) > 0:
        results['mean_length'] = results['total_length'] / len(results['length'])

    return results


def densify_streamlines_parallel(streamlines, step_size, n_jobs=8, use_gpu=True, interp_method='hermite',
                                 voxel_size=1.0):
    """
    Densifying multiple streamlines in parallel.

    Parameters
    ----------
    streamlines : list of arrays
        List of streamlines to densify.
    step_size : float
        Step size for densification.
    n_jobs : int, optional
        Number of parallel jobs, by default 8.
    use_gpu : bool, optional
        Whether using GPU acceleration, by default True.
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

    if n_jobs == 1:
        densified = []
        total = len(streamlines)
        for i, streamline in enumerate(streamlines):
            if i % 1000 == 0:
                print(f"Processing streamline {i}/{total}...")
            try:
                if isinstance(streamline, list):
                    streamline = np.array(streamline, dtype=np.float32)
                d = densify_streamline_subvoxel(streamline, step_size, use_gpu, interp_method, voxel_size=voxel_size)
                if len(d) >= 2:
                    densified.append(d)
            except Exception as e:
                print(f"Error densifying streamline {i}: {e}")
        return densified
    else:
        from joblib import Parallel, delayed

        def _process_one(streamline, idx, total):
            if idx % 1000 == 0:
                print(f"Processing streamline {idx}/{total}...")
            streamline_type = type(streamline).__name__
            try:
                if isinstance(streamline, list):
                    streamline = np.array(streamline, dtype=np.float32)
                if len(streamline) < 2:
                    print(f"Warning: Streamline {idx} had {len(streamline)} points, skipping.")
                    return None
                d = densify_streamline_subvoxel(
                    streamline, step_size, use_gpu=use_gpu,
                    interp_method=interp_method, voxel_size=voxel_size
                )
                if isinstance(d, list):
                    d = np.array(d, dtype=np.float32)
                return d
            except Exception as e:
                print(f"Error densifying streamline {idx}: {e}")
                print(f"Streamline type was: {streamline_type}")
                return None

        total = len(streamlines)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_one)(streamline, idx, total)
            for idx, streamline in enumerate(streamlines)
        )
        densified = [r for r in results if r is not None and len(r) >= 2]
        print(f"Densified {len(densified)}/{total} streamlines successfully")
        return densified


def densify_streamline_subvoxel(streamline, step_size, use_gpu=True, interp_method='hermite', voxel_size=1.0):
    """
    Densifying a streamline with sub-voxel precision.

    Parameters
    ----------
    streamline : array-like
        The streamline to densify.
    step_size : float
        Step size for densification.
    use_gpu : bool, optional
        Whether using GPU acceleration, by default True.
    interp_method : str, optional
        Interpolation method ('hermite' or 'linear'), by default 'hermite'.
    voxel_size : float, optional
        Voxel size affecting tangent scaling in Hermite interpolation, by default 1.0.

    Returns
    -------
    array-like
        Densified streamline.
    """
    import numpy as np
    if isinstance(streamline, list):
        try:
            streamline = np.array(streamline, dtype=np.float32)
        except Exception as e:
            raise TypeError(f"Failed to convert list to numpy array: {e}")
    if len(streamline) < 2:
        print(f"Warning: Cannot densify streamline with less than 2 points (has {len(streamline)})")
        return streamline

    debug_tangents = os.environ.get("DEBUG_TANGENTS") == "1"
    if debug_tangents:
        print(f"[DENSIFY] Using {interp_method} interpolation with step size {step_size}, voxel size {voxel_size}mm")
        print(f"[DENSIFY] Streamline points: {len(streamline)}")

    if use_gpu:
        try:
            import cupy as xp
            print(f"[DEBUG] GPU processing with CuPy - streamline shape: {streamline.shape}")
            if not isinstance(streamline, np.ndarray):
                print(f"[DEBUG] Converting streamline to numpy array")
                streamline = np.array(streamline, dtype=np.float32)
            try:
                streamline_device = xp.asarray(streamline)
                print(f"[DEBUG] CuPy conversion successful - device shape: {streamline_device.shape}")
            except Exception as e:
                print(f"CuPy conversion error: {e}")
                print("[DEBUG] Falling back to CPU")
                import numpy as xp
                streamline_device = xp.asarray(streamline)
                use_gpu = False
        except ImportError:
            print("Warning: Could not import cupy. Falling back to CPU.")
            import numpy as xp
            streamline_device = xp.asarray(streamline)
            use_gpu = False
    else:
        xp = np
        if not isinstance(streamline, np.ndarray):
            streamline = np.array(streamline, dtype=np.float32)
        streamline_device = xp.asarray(streamline)

    diffs = xp.diff(streamline_device, axis=0)
    segment_lengths = xp.sqrt(xp.sum(diffs ** 2, axis=1))
    total_length = xp.sum(segment_lengths)

    if debug_tangents:
        print(f"[DENSIFY] Total streamline length: {total_length:.4f}mm")
        print(f"[DENSIFY] Mean segment length: {total_length / len(segment_lengths):.4f}mm")

    if total_length < step_size:
        if debug_tangents:
            print(f"[DENSIFY] Streamline too short: {total_length:.4f} < {step_size:.4f}")
        return streamline

    n_steps = int(xp.ceil(total_length / step_size)) + 1
    cumulative_lengths = xp.concatenate(([0], xp.cumsum(segment_lengths)))
    normalized_distances = cumulative_lengths / total_length
    xi = xp.linspace(0, 1, n_steps)

    if xp.any(xp.isnan(normalized_distances)) or xp.any(xp.isnan(xi)):
        print("Warning: NaN values detected. Using original streamline.")
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
        tangents = xp.zeros_like(streamline_device)
        base_scale = 1.0
        voxel_scale = (1.0 / max(0.01, voxel_size))
        voxel_scale = min(5.0, voxel_scale)
        final_scale = base_scale * voxel_scale
        if debug_tangents:
            print(f"[TANGENT] Voxel size: {voxel_size:.4f}mm")
            print(f"[TANGENT] Final scale factor: {final_scale:.4f}")
        for i in range(1, len(streamline_device) - 1):
            pt_prev = streamline_device[i - 1]
            pt_next = streamline_device[i + 1]
            tangent = (pt_next - pt_prev) * 0.5
            tangents[i] = tangent * final_scale
            if i == 1 and debug_tangents:
                tangent_magnitude = xp.linalg.norm(tangent)
                scaled_magnitude = xp.linalg.norm(tangents[i])
                print(f"[TANGENT] Example - Point {i}: Original tangent: {tangent}, magnitude: {tangent_magnitude:.4f}")
                print(f"         Scaled tangent: {tangents[i]}, magnitude: {scaled_magnitude:.4f}")
        tangents[0] = (streamline_device[1] - streamline_device[0]) * final_scale
        tangents[-1] = (streamline_device[-1] - streamline_device[-2]) * final_scale
        if debug_tangents:
            orig_magnitudes = xp.sqrt(xp.sum(tangents ** 2, axis=1))
            print(f"[TANGENT] Mean magnitude: {xp.mean(orig_magnitudes):.4f}, max: {xp.max(orig_magnitudes):.4f}")
        tangent_norms = xp.sqrt(xp.sum(tangents ** 2, axis=1, keepdims=True))
        tangent_norms = xp.where(tangent_norms > 1e-10, tangent_norms, 1e-10)
        normalization_factor = max(0.5, min(1.0, voxel_size))
        if debug_tangents:
            print(f"[TANGENT] Normalization factor: {normalization_factor:.4f}")
        tangents = tangents / (tangent_norms * normalization_factor)
        if debug_tangents:
            normalized_magnitudes = xp.sqrt(xp.sum(tangents ** 2, axis=1))
            print(
                f"[TANGENT] Normalized magnitudes: mean={xp.mean(normalized_magnitudes):.4f}, max={xp.max(normalized_magnitudes):.4f}")

    result_shape = (len(xi), streamline_device.shape[1])
    if use_gpu and hasattr(xp, 'zeros'):
        result_array = xp.zeros(result_shape, dtype=xp.float32)
    else:
        result_array = np.zeros(result_shape, dtype=np.float32)

    for dim in range(streamline_device.shape[1]):
        y = streamline_device[:, dim]
        if interp_method == 'hermite':
            if use_gpu:
                import numpy as np
                if hasattr(xp, 'asnumpy'):
                    t_values = xp.asnumpy(normalized_distances)
                    interp_points = xp.asnumpy(xi)
                    y_values = xp.asnumpy(y)
                    tangent_values = xp.asnumpy(tangents[:, dim])
                else:
                    t_values = np.array(normalized_distances)
                    interp_points = np.array(xi)
                    y_values = np.array(y)
                    tangent_values = np.array(tangents[:, dim])
                try:
                    from scipy.interpolate import CubicHermiteSpline
                    interpolator = CubicHermiteSpline(t_values, y_values, tangent_values)
                    interpolated_cpu = interpolator(interp_points)
                    interpolated = xp.asarray(interpolated_cpu)
                except Exception as e:
                    print(f"Hermite interpolation failed: {e}. Using linear interpolation.")
                    interpolated = xp.interp(xi, normalized_distances, y)
            else:
                try:
                    from scipy.interpolate import CubicHermiteSpline
                    interpolator = CubicHermiteSpline(normalized_distances, y, tangents[:, dim])
                    interpolated = interpolator(xi)
                except Exception as e:
                    print(f"Hermite interpolation failed: {e}. Using linear interpolation.")
                    interpolated = xp.interp(xi, normalized_distances, y)
        else:
            interpolated = xp.interp(xi, normalized_distances, y)
        result_array[:, dim] = interpolated

    densified_streamline = result_array
    if use_gpu and hasattr(xp, 'asnumpy'):
        densified_streamline = xp.asnumpy(densified_streamline)

    if debug_tangents:
        print(f"[DENSIFY] Original points: {len(streamline)}, Densified points: {len(densified_streamline)}")
        if interp_method == 'hermite' and len(streamline) > 2 and len(densified_streamline) > 2:
            def calc_curvature(points):
                import numpy as np
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
            print(f"[CURVATURE] Original: {orig_curvature:.6f}, Densified: {new_curvature:.6f}")
            if orig_curvature > 0:
                print(f"[CURVATURE] Change: {(new_curvature - orig_curvature) / orig_curvature * 100:.2f}%")
            else:
                print("[CURVATURE] Change: N/A (original curvature was zero)")

    return densified_streamline


if __name__ == "__main__":
    print("Testing interpolation functions...")
    test_stream = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [2, 0, 0]
    ], dtype=np.float32)
    print(f"Original streamline shape: {test_stream.shape}")

    linear_result = densify_streamline_subvoxel(
        test_stream.copy(), step_size=0.5, use_gpu=False, interp_method='linear'
    )
    print(f"Linear interpolation result shape: {linear_result.shape}")
    print(f"Linear first few points:\n{linear_result[:5]}")

    hermite_result = densify_streamline_subvoxel(
        test_stream.copy(), step_size=0.5, use_gpu=False, interp_method='hermite'
    )
    print(f"Hermite interpolation result shape: {hermite_result.shape}")
    print(f"Hermite first few points:\n{hermite_result[:5]}")

    if np.array_equal(linear_result, hermite_result):
        print("ERROR: Both methods produced identical results!")
    else:
        difference = np.mean(np.abs(linear_result[:min(len(linear_result), len(hermite_result))] -
                                    hermite_result[:min(len(linear_result), len(hermite_result))]))
        print(f"Mean difference between methods: {difference:.6f}")
        print("Test passed: interpolation methods produce different results.")