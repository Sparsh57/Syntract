import numpy as np
import pytest

from syntract_viewer.synthetic_image_augmentations import apply_poisson_shot_noise


def _half_dark_half_bright(n=64):
    """Left half dark (0.15), right half bright (0.85)."""
    vol = np.empty((n, n, n), dtype=np.float32)
    vol[:, :, : n // 2] = 0.15
    vol[:, :, n // 2 :] = 0.85
    return vol


def test_poisson_is_signal_dependent():
    """Brighter voxels must be noisier than darker voxels (variance = mean)."""
    base = _half_dark_half_bright()
    out = apply_poisson_shot_noise(base, gain=80.0, random_state=0)
    n = base.shape[-1]
    dark_noise = (out - base)[:, :, : n // 2].std()
    bright_noise = (out - base)[:, :, n // 2 :].std()
    assert bright_noise > 1.5 * dark_noise


def test_poisson_gain_is_monotonic():
    """Lower gain => more total shot noise.

    Uses a non-constant volume: _normalize_unit collapses a constant array to
    zeros (the shared hi<=lo guard), which would suppress all noise.
    """
    base = _half_dark_half_bright(48)
    low = apply_poisson_shot_noise(base, gain=40.0, random_state=1)
    high = apply_poisson_shot_noise(base, gain=150.0, random_state=1)
    assert (low - base).std() > (high - base).std()


def test_poisson_preserves_shape_and_is_finite():
    base = _half_dark_half_bright(32)
    out = apply_poisson_shot_noise(base, gain=80.0, random_state=2)
    assert out.shape == base.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_poisson_nonpositive_gain_is_noop():
    base = _half_dark_half_bright(16)
    out = apply_poisson_shot_noise(base, gain=0.0, random_state=3)
    np.testing.assert_array_equal(out, base.astype(np.float32))


def test_poisson_reproducible_with_seed():
    base = np.full((24, 24, 24), 0.4, dtype=np.float32)
    a = apply_poisson_shot_noise(base, gain=80.0, random_state=7)
    b = apply_poisson_shot_noise(base, gain=80.0, random_state=7)
    np.testing.assert_array_equal(a, b)


from syntract_viewer.synthetic_image_augmentations import apply_image_only_augmentations


def test_dispatcher_poisson_disabled_is_identity():
    base = _half_dark_half_bright(24)
    out = apply_image_only_augmentations(base, enable_poisson_noise=False, random_state=0)
    np.testing.assert_array_equal(out, base.astype(np.float32))


def test_dispatcher_poisson_enabled_changes_volume():
    base = _half_dark_half_bright(24)
    out = apply_image_only_augmentations(
        base, enable_poisson_noise=True, poisson_gain=60.0, random_state=0
    )
    assert not np.array_equal(out, base.astype(np.float32))
    assert out.shape == base.shape
