import cupy as cp


def densify_streamline_subvoxel(stream, step_size=0.5):
    """
    Densifies a given streamline by interpolating additional points using GPU acceleration.

    Parameters
    ----------
    stream : np.ndarray
        (N, 3) array representing streamline coordinates.
    step_size : float, optional
        Step size for interpolation, by default 0.5.

    Returns
    -------
    np.ndarray
        Densified streamline.
    """
    if len(stream) < 2:
        return cp.asnumpy(stream.astype(cp.float32))

    new_points = []
    for i in range(len(stream) - 1):
        p0 = cp.asarray(stream[i])
        p1 = cp.asarray(stream[i + 1])
        seg_vec = p1 - p0
        seg_len = cp.linalg.norm(seg_vec)
        if seg_len < 1e-12:
            new_points.append(p0)
            continue
        n_steps = int(cp.ceil(seg_len / step_size))
        for s in range(n_steps):
            t = s / n_steps
            interp_pt = p0 + t * seg_vec
            new_points.append(interp_pt)
        if i == len(stream) - 2:
            new_points.append(p1)

    return cp.asnumpy(cp.array(new_points, dtype=cp.float32))