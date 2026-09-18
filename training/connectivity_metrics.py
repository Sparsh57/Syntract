"""Label-free connectivity metrics for the Continuity fiber prior.

A Fiber is a long, connected, continuous curve — never a scattered blob. That
property is computable with NO ground truth, so it serves as both a real-data
quality signal (replacing the gameable `pred_pos_frac`) and a diagnostic.

Measured baseline (2026-06-11): the synthetic training target scores
``tiny_blob_frac=0.0`` / ``len_thickness~56`` (a real connected fiber), while the
current model on real LSM scores ``tiny_blob_frac~0.57-0.67`` / ``len_thickness~5-10``
(scattered blobs). A good fiber segmenter must move the real numbers toward the
synthetic ones.

CPU-only (numpy + scipy; optional skimage for skeleton length). No torch.
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.ndimage import label as _cc_label
except ImportError:  # pragma: no cover - scipy is a hard dep of the project
    _cc_label = None

# 26-connectivity in 3D: two voxels touching at a corner are one fiber, not two.
_STRUCT3D = np.ones((3, 3, 3), dtype=int)


def connectivity_stats(volume, threshold: float = 0.5, tiny_max: int = 3,
                       compute_skeleton: bool = False) -> dict:
    """Connectivity statistics for a 3D probability/binary volume.

    Parameters
    ----------
    volume : array-like, 3D (singleton dims are squeezed)
    threshold : binarization threshold for probabilities
    tiny_max : a connected component with <= this many voxels counts as a blob
    compute_skeleton : also return total skeleton length (slower; needs skimage)

    Returns
    -------
    dict with (all label-free):
      pos_frac        - fraction of positive voxels
      n_components    - number of 26-connected components
      largest_frac    - mass fraction in the largest component (higher = less fragmented)
      tiny_blob_frac  - fraction of components that are tiny blobs (LOWER = better)
      max_comp_len    - bbox diagonal of the largest component, in voxels
      len_thickness   - elongation proxy of the largest component (HIGHER = more fiber-like)
      skeleton_len    - total skeleton voxels (-1 if not computed)
      continuity      - single 0..1 summary, higher = more fiber-like
    """
    v = np.asarray(volume)
    if v.ndim != 3:
        v = np.squeeze(v)
    if v.ndim != 3:
        raise ValueError(f"connectivity_stats expects a 3D volume, got shape {np.shape(volume)}")

    b = v >= threshold
    total = int(b.sum())
    out = {
        "pos_frac": float(b.mean()) if b.size else 0.0,
        "n_components": 0,
        "largest_frac": 0.0,
        "tiny_blob_frac": 1.0,
        "max_comp_len": 0.0,
        "len_thickness": 0.0,
        "skeleton_len": -1.0,
        "continuity": 0.0,
    }
    if total == 0 or _cc_label is None:
        return out

    lab, n = _cc_label(b, structure=_STRUCT3D)
    sizes = np.bincount(lab.ravel())[1:]  # drop background (label 0)
    out["n_components"] = int(n)

    big_label = int(np.argmax(sizes)) + 1
    big = int(sizes.max())
    out["largest_frac"] = float(big / total)
    out["tiny_blob_frac"] = float((sizes <= tiny_max).mean())

    coords = np.argwhere(lab == big_label)
    ext = (coords.max(0) - coords.min(0) + 1).astype(np.float64)
    diag = float(np.sqrt((ext ** 2).sum()))
    out["max_comp_len"] = diag
    # Elongation: a long thin tube has diag >> effective thickness sqrt(vol/diag).
    eff_thickness = max(np.sqrt(big / max(diag, 1.0)), 1e-6)
    out["len_thickness"] = float(diag / eff_thickness)

    if compute_skeleton:
        try:
            from skimage.morphology import skeletonize
            out["skeleton_len"] = float(np.asarray(skeletonize(b)).sum())
        except Exception:
            pass

    # Concentrate mass in few long pieces, punish tiny-blob fragmentation.
    out["continuity"] = float(np.clip(out["largest_frac"] * (1.0 - out["tiny_blob_frac"]), 0.0, 1.0))
    return out


def _self_test():
    """A straight line is one connected fiber; scattered dots are blobs."""
    rng = np.random.default_rng(0)

    line = np.zeros((64, 64, 64), np.float32)
    line[10:54, 32, 32] = 1.0  # a 44-voxel straight fiber along z
    ls = connectivity_stats(line)
    assert ls["n_components"] == 1, ls
    assert ls["tiny_blob_frac"] == 0.0, ls
    assert ls["len_thickness"] > 20, ls
    assert ls["continuity"] > 0.9, ls

    blobs = np.zeros((64, 64, 64), np.float32)
    for _ in range(40):  # 40 isolated single-voxel dots
        z, y, x = rng.integers(0, 64, size=3)
        blobs[z, y, x] = 1.0
    bs = connectivity_stats(blobs)
    assert bs["tiny_blob_frac"] > 0.8, bs
    assert bs["continuity"] < 0.1, bs
    assert bs["len_thickness"] < 5, bs

    empty = connectivity_stats(np.zeros((8, 8, 8), np.float32))
    assert empty["n_components"] == 0 and empty["continuity"] == 0.0, empty

    print("connectivity_metrics self-test passed")
    print("  line :", {k: round(v, 3) for k, v in ls.items()})
    print("  blobs:", {k: round(v, 3) for k, v in bs.items()})


if __name__ == "__main__":
    _self_test()
