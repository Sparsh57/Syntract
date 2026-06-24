"""Image-only synthetic artefacts for 3D streamline training volumes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter


def _normalize_unit(volume: np.ndarray):
    volume = np.asarray(volume, dtype=np.float32)
    lo = float(volume.min())
    hi = float(volume.max())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32), lo, hi
    normalized = np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    return normalized, lo, hi


def _restore_range(volume_unit: np.ndarray, lo: float, hi: float, original: np.ndarray) -> np.ndarray:
    if hi <= lo:
        return original.astype(np.float32, copy=False)
    restored = np.clip(volume_unit, 0.0, 1.0) * (hi - lo) + lo
    orig_min = float(np.min(original))
    orig_max = float(np.max(original))
    return np.clip(restored, orig_min, max(orig_max, hi)).astype(np.float32, copy=False)


def apply_tissue_artifacts(
    volume: np.ndarray,
    strength: float = 0.45,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Add smooth tissue-like clutter without changing geometry."""
    strength = max(0.0, float(strength))
    if strength == 0.0:
        return np.asarray(volume, dtype=np.float32)

    rng = np.random.default_rng(random_state)
    original = np.asarray(volume, dtype=np.float32)
    vol, lo, hi = _normalize_unit(original)
    shape = np.array(vol.shape, dtype=np.float32)

    if verbose:
        print(f"    Applying tissue-like artefacts (strength={strength:.3f})")

    coarse_sigma = tuple(np.maximum(shape / 9.0, 3.0))
    medium_sigma = tuple(np.maximum(shape / 28.0, 1.5))
    fine_sigma = tuple(np.maximum(shape / 80.0, 0.75))

    coarse = gaussian_filter(rng.normal(size=vol.shape).astype(np.float32), sigma=coarse_sigma)
    medium = gaussian_filter(rng.normal(size=vol.shape).astype(np.float32), sigma=medium_sigma)
    fine = gaussian_filter(rng.normal(size=vol.shape).astype(np.float32), sigma=fine_sigma)

    coords = np.meshgrid(
        *[np.linspace(-1.0, 1.0, int(s), dtype=np.float32) for s in vol.shape],
        indexing="ij",
    )
    direction = rng.normal(size=vol.ndim).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-6
    projection = sum(float(d) * c for d, c in zip(direction, coords))
    bands = np.sin(
        2.0 * np.pi * (rng.uniform(1.5, 4.0) * projection + rng.uniform(0.0, 1.0))
    ).astype(np.float32)
    bands = gaussian_filter(bands, sigma=tuple(np.maximum(shape / 60.0, 0.8)))

    artifact = 0.42 * coarse + 0.30 * medium + 0.18 * fine + 0.10 * bands
    artifact -= float(np.mean(artifact))
    artifact /= float(np.std(artifact)) + 1e-6

    tissue = (vol > max(0.015, float(np.percentile(vol, 12)))).astype(np.float32)
    tissue = gaussian_filter(tissue, sigma=1.25)
    tissue = np.clip(tissue, 0.0, 1.0)

    multiplicative = 1.0 + (0.11 * strength * artifact * tissue)
    additive = 0.035 * strength * artifact * tissue
    augmented = vol * multiplicative + additive

    bright_specks = rng.random(vol.shape, dtype=np.float32)
    bright_specks = gaussian_filter((bright_specks > 0.9975).astype(np.float32), sigma=1.0)
    if float(bright_specks.max()) > 0:
        bright_specks /= float(bright_specks.max())
    augmented += 0.035 * strength * bright_specks * tissue

    return _restore_range(augmented, lo, hi, original)


def apply_granular_noise(
    volume: np.ndarray,
    strength: float = 0.35,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Add fine grain with Cornucopia noise transforms, falling back to NumPy."""
    strength = max(0.0, float(strength))
    if strength == 0.0:
        return np.asarray(volume, dtype=np.float32)

    original = np.asarray(volume, dtype=np.float32)
    vol, lo, hi = _normalize_unit(original)

    if verbose:
        print(f"    Applying Cornucopia granular noise (strength={strength:.3f})")

    sigma = max(1e-5, 0.025 * strength)
    try:
        import torch
        from cornucopia import GaussianNoiseTransform

        if random_state is not None:
            torch.manual_seed(int(random_state) % (2**31 - 1))
        tensor = torch.from_numpy(vol[np.newaxis].astype(np.float32, copy=False))
        transform = GaussianNoiseTransform(sigma=sigma)
        with torch.no_grad():
            noisy = transform(tensor).squeeze(0).cpu().numpy()
        augmented = noisy.astype(np.float32, copy=False)
    except Exception as exc:
        if verbose:
            print(f"    Cornucopia granular noise unavailable, using NumPy fallback: {exc}")
        rng = np.random.default_rng(random_state)
        augmented = vol + rng.normal(0.0, sigma, size=vol.shape).astype(np.float32)

    return _restore_range(augmented, lo, hi, original)


def apply_poisson_shot_noise(
    volume: np.ndarray,
    gain: float = 80.0,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Add Poisson (shot) noise — the dominant noise in fluorescence light-sheet.

    Photon counting makes per-voxel variance equal the mean, so brighter voxels
    are noisier (signal-dependent), unlike additive Gaussian grain. ``gain`` is
    photons-per-unit-intensity: lower gain = heavier shot noise, higher gain =
    nearly clean. Image-only; never touches the mask. NumPy-only (no torch/cupy).
    """
    gain = float(gain)
    if gain <= 0.0:
        return np.asarray(volume, dtype=np.float32)

    rng = np.random.default_rng(random_state)
    original = np.asarray(volume, dtype=np.float32)
    vol, lo, hi = _normalize_unit(original)

    if verbose:
        print(f"    Applying Poisson shot noise (gain={gain:.1f})")

    lam = np.clip(vol, 0.0, None) * gain
    noisy = (rng.poisson(lam).astype(np.float32)) / gain

    return _restore_range(noisy, lo, hi, original)


def apply_horizontal_banding(
    volume: np.ndarray,
    strength: float = 0.18,
    band_axis: int = 1,
    random_state: Optional[int] = None,
    verbose: bool = False,
    fiber_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulate blockface section-to-section illumination variation (horizontal banding).

    Generates a correlated random signal along band_axis with three components:
    coarse (broad gradients), medium (section groups), and fine (per-slice jitter).
    Applied as multiplicative modulation so tissue structure is preserved.
    band_axis=1 corresponds to the coronal/Y axis where blockface sections stack.
    """
    strength = max(0.0, float(strength))
    if strength == 0.0:
        return np.asarray(volume, dtype=np.float32)

    rng = np.random.default_rng(random_state)
    original = np.asarray(volume, dtype=np.float32)
    vol, lo, hi = _normalize_unit(original)

    if verbose:
        print(f"    Applying horizontal banding (axis={band_axis}, strength={strength:.3f})")

    n = vol.shape[band_axis]

    # Three-scale correlated signal along the slice axis
    coarse = gaussian_filter1d(
        rng.normal(size=n).astype(np.float32),
        sigma=max(1.0, n / 6.0),   # Very broad undulations
    )
    medium = gaussian_filter1d(
        rng.normal(size=n).astype(np.float32),
        sigma=max(0.5, n / 20.0),  # Section-group variation
    )
    fine = rng.normal(size=n).astype(np.float32) * 0.4  # Per-slice jitter

    band_signal = 0.55 * coarse + 0.35 * medium + 0.10 * fine
    max_abs = float(np.max(np.abs(band_signal))) + 1e-6
    band_signal = (band_signal / max_abs).astype(np.float32)

    # Reshape for broadcasting along band_axis
    bcast_shape = [1] * vol.ndim
    bcast_shape[band_axis] = n
    band_signal = band_signal.reshape(bcast_shape)

    modulation = 1.0 + strength * band_signal  # shape broadcastable over vol
    augmented = vol * modulation

    augmented = np.clip(augmented, 0.0, 1.5)
    return _restore_range(augmented, lo, hi, original)


def apply_speckle_dot_noise(
    volume: np.ndarray,
    strength: float = 0.70,
    density: float = 0.008,
    dot_sigma: float = 0.0,
    dot_square_size: int = 2,
    random_state: Optional[int] = None,
    verbose: bool = False,
    fiber_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add bright square speck artefacts mimicking blockface dust/calibration particles.

    Real blockface data shows bright white square-edged specks at moderate density.
    dot_sigma=0 + dot_square_size=2 produces sharp 2-voxel square patches,
    matching the appearance of 1-3 pixel square dust artefacts in real data.

    fiber_mask voxels are excluded so specks never land on fibers.
    """
    strength = max(0.0, float(strength))
    density = max(0.0, float(density))
    if strength == 0.0 or density == 0.0:
        return np.asarray(volume, dtype=np.float32)

    rng = np.random.default_rng(random_state)
    original = np.asarray(volume, dtype=np.float32)
    vol, lo, hi = _normalize_unit(original)

    if verbose:
        print(
            f"    Applying square speck noise "
            f"(strength={strength:.3f}, density={density:.5f}, square={dot_square_size})"
        )

    tissue = vol > max(0.015, float(np.percentile(vol, 10)))
    if fiber_mask is not None:
        tissue = tissue & ~np.asarray(fiber_mask, dtype=bool)
    dots = (rng.random(vol.shape, dtype=np.float32) < density) & tissue
    if not np.any(dots):
        return original.astype(np.float32, copy=False)

    dot_field = np.zeros_like(vol, dtype=np.float32)
    # Bright white specks (0.7–1.0) to match the near-saturated appearance in real data
    dot_field[dots] = rng.uniform(0.70, 1.0, size=int(np.count_nonzero(dots))).astype(np.float32)

    if dot_square_size > 1:
        # Uniform filter creates a square spreading — matches real square particle appearance
        dot_field = uniform_filter(dot_field, size=int(dot_square_size))
        if fiber_mask is not None:
            dot_field[fiber_mask] = 0.0
        peak = float(dot_field.max())
        if peak > 0.0:
            dot_field /= peak
    elif dot_sigma > 0.0:
        dot_field = gaussian_filter(dot_field, sigma=float(dot_sigma))
        if fiber_mask is not None:
            dot_field[fiber_mask] = 0.0
        peak = float(dot_field.max())
        if peak > 0.0:
            dot_field /= peak

    augmented = vol + (0.40 * strength * dot_field)
    if hi <= lo:
        return original.astype(np.float32, copy=False)

    restored = np.clip(augmented, 0.0, 1.0 + 0.50 * strength) * (hi - lo) + lo
    max_allowed = max(float(np.max(original)), hi + (hi - lo) * 0.45 * strength)
    return np.clip(restored, float(np.min(original)), max_allowed).astype(np.float32, copy=False)


def apply_dash_noise(
    volume: np.ndarray,
    strength: float = 0.55,
    density: float = 0.0005,
    length_sigma: float = 4.0,
    cross_sigma: float = 0.3,
    random_state: Optional[int] = None,
    verbose: bool = False,
    fiber_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add short bright dash artefacts mimicking fiber strand appearance in blockface data.

    Seeds are placed in tissue, then stretched via 1-D Gaussian blur along a
    randomly chosen primary axis (cross-section kept tight) to produce
    short bright line segments. In real blockface data these appear as thin
    diagonal dashes 3-10 pixels long.

    fiber_mask voxels are excluded both before seeding and after blur bleed.
    """
    strength = max(0.0, float(strength))
    density = max(0.0, float(density))
    if strength == 0.0 or density == 0.0:
        return np.asarray(volume, dtype=np.float32)

    rng = np.random.default_rng(random_state)
    original = np.asarray(volume, dtype=np.float32)
    vol, lo, hi = _normalize_unit(original)

    if verbose:
        print(f"    Applying dash noise (strength={strength:.3f}, density={density:.5f})")

    tissue = vol > max(0.015, float(np.percentile(vol, 10)))
    if fiber_mask is not None:
        tissue = tissue & ~np.asarray(fiber_mask, dtype=bool)

    seeds = (rng.random(vol.shape, dtype=np.float32) < density) & tissue
    if not np.any(seeds):
        return original.astype(np.float32, copy=False)

    dash_field = np.zeros_like(vol, dtype=np.float32)
    dash_field[seeds] = rng.uniform(0.60, 1.0, size=int(np.count_nonzero(seeds))).astype(np.float32)

    primary_axis = int(rng.integers(0, vol.ndim))
    for ax in range(vol.ndim):
        sigma = max(1.5, float(length_sigma)) if ax == primary_axis else max(0.1, float(cross_sigma))
        dash_field = gaussian_filter1d(dash_field, sigma=sigma, axis=ax)

    if fiber_mask is not None:
        dash_field[fiber_mask] = 0.0
    peak = float(dash_field.max())
    if peak > 0.0:
        dash_field /= peak

    augmented = vol + (0.35 * strength * dash_field)
    return _restore_range(augmented, lo, hi, original)


def apply_image_only_augmentations(
    volume: np.ndarray,
    enable_tissue_artifacts: bool = False,
    enable_granular_noise: bool = False,
    enable_speckle_noise: bool = False,
    enable_dash_noise: bool = False,
    enable_horizontal_banding: bool = False,
    artifact_strength: float = 0.45,
    granular_noise_strength: float = 0.35,
    speckle_noise_strength: float = 0.70,
    speckle_noise_density: float = 0.008,
    speckle_noise_sigma: float = 0.0,
    speckle_square_size: int = 2,
    dash_noise_strength: float = 0.55,
    dash_noise_density: float = 0.0005,
    dash_length_sigma: float = 4.0,
    dash_cross_sigma: float = 0.3,
    banding_strength: float = 0.18,
    banding_axis: int = 1,
    random_state: Optional[int] = None,
    verbose: bool = False,
    fiber_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply synthetic realism augmentations to image intensities only.

    fiber_mask is a boolean array (same shape as volume) marking fiber voxels.
    When provided, speckle/dash artefacts are excluded from those voxels to keep
    the ground-truth mask consistent with the image.
    """
    augmented = np.asarray(volume, dtype=np.float32)
    tissue_seed = random_state
    granular_seed = None if random_state is None else int(random_state) + 104729
    speckle_seed = None if random_state is None else int(random_state) + 208351
    dash_seed = None if random_state is None else int(random_state) + 312973
    banding_seed = None if random_state is None else int(random_state) + 417619

    if enable_horizontal_banding:
        augmented = apply_horizontal_banding(
            augmented,
            strength=banding_strength,
            band_axis=banding_axis,
            random_state=banding_seed,
            verbose=verbose,
            fiber_mask=fiber_mask,
        )
    if enable_tissue_artifacts:
        augmented = apply_tissue_artifacts(
            augmented,
            strength=artifact_strength,
            random_state=tissue_seed,
            verbose=verbose,
        )
    if enable_granular_noise:
        augmented = apply_granular_noise(
            augmented,
            strength=granular_noise_strength,
            random_state=granular_seed,
            verbose=verbose,
        )
    if enable_speckle_noise:
        augmented = apply_speckle_dot_noise(
            augmented,
            strength=speckle_noise_strength,
            density=speckle_noise_density,
            dot_sigma=speckle_noise_sigma,
            dot_square_size=speckle_square_size,
            random_state=speckle_seed,
            verbose=verbose,
            fiber_mask=fiber_mask,
        )
    if enable_dash_noise:
        augmented = apply_dash_noise(
            augmented,
            strength=dash_noise_strength,
            density=dash_noise_density,
            length_sigma=dash_length_sigma,
            cross_sigma=dash_cross_sigma,
            random_state=dash_seed,
            verbose=verbose,
            fiber_mask=fiber_mask,
        )
    return augmented.astype(np.float32, copy=False)
