"""
Thicken a TRK into a denser, organic-looking fiber bundle.

For each input streamline, generate N sibling streamlines that follow the same
overall path but are offset perpendicular to the LOCAL tangent at every point.
The lateral offset varies smoothly along the arc length (low-pass filtered
random walk) so siblings don't read as rigid translations of the original --
they trace gently meandering parallel paths the way real axonal fibers do
inside a bundle.

Why: at very fine voxel sizes (e.g. 0.001 mm / 64^3 patches = 0.064 mm FOV),
a patch is smaller than the native gap between streamlines (~0.5-1 mm), so
each patch only sees one streamline. Thickening produces bundles dense enough
that patches contain many parallel fibers like real white-matter tracts.

Usage:
    python thicken_trk.py \
        --input registered_trk/aligned_streamlines_standard_ants_registered.trk \
        --output registered_trk/aligned_dense.trk \
        --copies 60 \
        --radius_mm 0.04
"""

import argparse
import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, TrkFile
from scipy.ndimage import gaussian_filter1d


def local_perpendicular_frame(points: np.ndarray) -> tuple:
    """Per-point orthonormal basis (n1, n2) perpendicular to the local tangent.

    Frames are propagated along the streamline using parallel transport so the
    basis varies smoothly without arbitrary twists between neighbouring points.
    """
    n = len(points)
    # Forward differences with reflection for the endpoint
    tangents = np.empty_like(points)
    tangents[:-1] = points[1:] - points[:-1]
    tangents[-1] = tangents[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

    n1 = np.empty_like(points)
    n2 = np.empty_like(points)

    # Seed the first frame: pick any vector not aligned with tangent[0]
    t0 = tangents[0]
    helper = np.array([1.0, 0.0, 0.0]) if abs(t0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = helper - t0 * np.dot(helper, t0)
    n1[0] = v / (np.linalg.norm(v) + 1e-12)
    n2[0] = np.cross(t0, n1[0])

    # Parallel transport: project previous n1 onto plane perpendicular to current tangent
    for i in range(1, n):
        v = n1[i - 1] - tangents[i] * np.dot(n1[i - 1], tangents[i])
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            # Tangent flipped direction; reuse a stable seed
            helper = np.array([1.0, 0.0, 0.0]) if abs(tangents[i][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            v = helper - tangents[i] * np.dot(helper, tangents[i])
            norm = np.linalg.norm(v) + 1e-12
        n1[i] = v / norm
        n2[i] = np.cross(tangents[i], n1[i])
    return n1, n2


def smooth_random_offsets(n_pts: int, radius_mm: float, smoothing_sigma: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Length-`n_pts` 2D vector of slowly-varying offsets in the perpendicular plane.

    Drawn as white noise then heavily Gaussian-smoothed along the arc-length
    axis. Rescaled so the per-streamline 95th percentile of magnitudes equals
    `radius_mm` -- gives every sibling a consistent thickness budget regardless
    of streamline length.
    """
    raw = rng.standard_normal((n_pts, 2))
    smooth = gaussian_filter1d(raw, sigma=smoothing_sigma, axis=0, mode="reflect")
    # Normalise so the typical (95th percentile) magnitude == radius_mm
    mag = np.linalg.norm(smooth, axis=1)
    p95 = np.percentile(mag, 95) if mag.size else 1.0
    if p95 > 1e-9:
        smooth *= radius_mm / p95
    return smooth.astype(np.float32)


def add_waviness(streamline: np.ndarray, densify_um: float, amplitude_um: float,
                 wavelength_um: float, rng: np.random.Generator) -> np.ndarray:
    """Resample a streamline to fine spacing and add smooth lateral waviness.

    At fine voxel sizes a streamline is straight across a small patch (its
    natural curvature is mm-scale), so it renders as a stair-stepped straight
    line. This adds gentle, organic curvature at a chosen wavelength so fibers
    look like real meandering axons rather than ruler-straight lines.

    densify_um/amplitude_um/wavelength_um are in micrometres; the streamline
    coordinates are in mm.
    """
    if amplitude_um <= 0 or len(streamline) < 2:
        return streamline.astype(np.float32)
    step_mm = densify_um / 1000.0
    seg = np.linalg.norm(np.diff(streamline, axis=0), axis=1)
    total = float(seg.sum())
    if total < step_mm or step_mm <= 0:
        return streamline.astype(np.float32)
    # Resample to uniform fine spacing
    n = int(np.ceil(total / step_mm)) + 1
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    xs = np.linspace(0.0, total, n)
    dense = np.empty((n, 3), dtype=np.float32)
    for d in range(3):
        dense[:, d] = np.interp(xs, cum, streamline[:, d])
    # Smooth random lateral offset: white noise low-pass filtered so its
    # correlation length ~= wavelength. sigma in points = wavelength / step.
    n1, n2 = local_perpendicular_frame(dense)
    sigma_pts = max(1.0, (wavelength_um / max(densify_um, 1e-6)))
    raw = rng.standard_normal((n, 2))
    sm = gaussian_filter1d(raw, sigma=sigma_pts, axis=0, mode="reflect")
    mag = np.linalg.norm(sm, axis=1)
    p95 = np.percentile(mag, 95) if mag.size else 1.0
    if p95 > 1e-9:
        sm *= (amplitude_um / 1000.0) / p95  # scale to amplitude in mm
    off = sm[:, 0:1] * n1 + sm[:, 1:2] * n2
    return (dense + off.astype(np.float32)).astype(np.float32)


def thicken_streamline(streamline: np.ndarray, copies: int, radius_mm: float,
                       smoothing_sigma: float, rng: np.random.Generator) -> list:
    """Generate `copies` sibling streamlines around `streamline`.

    Each sibling = streamline + offset(i, sibling) where offset is in the
    LOCAL perpendicular plane at point i. The original is included as copy 0.
    """
    if len(streamline) < 2:
        return [streamline.copy()]

    n1, n2 = local_perpendicular_frame(streamline)
    siblings = [streamline.astype(np.float32)]
    n_pts = len(streamline)

    for _ in range(max(0, copies - 1)):
        offsets_2d = smooth_random_offsets(n_pts, radius_mm, smoothing_sigma, rng)
        # Lift the 2D offsets into 3D using the per-point perpendicular basis
        offsets_3d = offsets_2d[:, 0:1] * n1 + offsets_2d[:, 1:2] * n2
        sibling = streamline + offsets_3d.astype(np.float32)
        siblings.append(sibling.astype(np.float32))
    return siblings


def main():
    parser = argparse.ArgumentParser(description="Thicken a TRK into a dense parallel-fiber bundle.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copies", type=int, default=60,
                        help="Number of streamlines produced per input streamline "
                             "(includes the original). Default: 60.")
    parser.add_argument("--radius_mm", type=float, default=0.04,
                        help="Typical perpendicular offset for sibling fibers (mm). "
                             "Default: 0.04 mm = 40 um.")
    parser.add_argument("--smoothing_sigma", type=float, default=8.0,
                        help="Gaussian sigma (in points along the streamline) used to "
                             "smooth the random offset path. Larger = smoother sibling "
                             "fibers that meander gently. Default: 8.")
    parser.add_argument("--wave_amplitude_um", type=float, default=0.0,
                        help="Add organic micro-curvature: lateral waviness amplitude in "
                             "micrometres (0 = off). Use with --copies 1 to curve the "
                             "original fibers without thickening. Try 4-8 um.")
    parser.add_argument("--wave_wavelength_um", type=float, default=40.0,
                        help="Approximate wavelength of the waviness in micrometres. "
                             "Smaller = tighter curves. Default: 40.")
    parser.add_argument("--wave_densify_um", type=float, default=1.0,
                        help="Resample spacing (um) used when adding waviness. Should be "
                             "<= target voxel size. Default: 1.0.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.input} ...")
    trk = nib.streamlines.load(args.input)
    streamlines = list(trk.tractogram.streamlines)
    print(f"  {len(streamlines)} input streamlines")

    out = []
    for sl in streamlines:
        sibs = thicken_streamline(np.asarray(sl, dtype=np.float32),
                                  args.copies, args.radius_mm,
                                  args.smoothing_sigma, rng)
        if args.wave_amplitude_um > 0:
            sibs = [add_waviness(s, args.wave_densify_um, args.wave_amplitude_um,
                                 args.wave_wavelength_um, rng) for s in sibs]
        out.extend(sibs)
    print(f"  Produced {len(out)} thickened streamlines "
          f"({args.copies} copies x {len(streamlines)} input)")

    tractogram = Tractogram(out, affine_to_rasmm=np.eye(4))
    trk_out = TrkFile(tractogram)
    for key in ("dimensions", "voxel_sizes", "voxel_to_rasmm", "voxel_order"):
        if key in trk.header:
            trk_out.header[key] = trk.header[key]
    trk_out.save(args.output)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
