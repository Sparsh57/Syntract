# Real-Domain-Match (Poisson Shot Noise + Soft Thin Masks) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add signal-dependent Poisson shot noise to the synthetic image pipeline and switch the training masks to soft, thin (~2 vox) partial-volume labels, so the 128³ U-Net trains on data that matches real LSM and stops firing on noise blobs.

**Architecture:** Add one image-only augmentation function (`apply_poisson_shot_noise`) following the existing `_normalize_unit → transform → _restore_range` pattern, wire it through the same plumbing chain as `granular_noise` (dispatcher → `volume_renderer` → `precompute_patches_3d` → SLURM scripts), then flip the precompute config to enable Poisson + re-enable banding + soft masks. A cheap stats-probe gate validates texture/learnability before the full 1800-patch precompute.

**Tech Stack:** Python, NumPy (no hard torch/cupy dependency in the new aug), scipy.ndimage, pytest, PyTorch Lightning training (unchanged), SLURM.

## Global Constraints

- **CPU fallback mandatory** — new aug must be NumPy-only (no hard `cupy`/`torch` import); mirror `apply_granular_noise`'s NumPy fallback shape.
- **Image-only** — Poisson noise applies to the image volume ONLY, never the mask.
- **Dual import pattern** — `try: from .module import X / except ImportError: from module import X` where the codebase already uses it.
- **Preserve fine-resolution fixes** — FOV anchoring, streamline-anchored sampling, 1–99 percentile normalization, soft-mask rasterizer. Poisson + soft masks are additive, not replacements.
- **Learnability guardrail** — fiber/background separation must stay ≥3–4×; do NOT match real's degenerate ~1× contrast. Keep granular noise at 1.5 (do not stack strong Gaussian on Poisson).
- **`gain` is the only Poisson knob** — low gain (~40) = heavy noise, high gain (~150) = nearly clean. Default 80.0.
- **No double augmentation** — `train_cached.sh` leaves load-time image augs OFF (noise baked at precompute).

---

### Task 1: `apply_poisson_shot_noise()` augmentation function

**Files:**
- Modify: `syntract_viewer/synthetic_image_augmentations.py` (add function after `apply_granular_noise`, ~line 126)
- Test: `tests/test_poisson_noise.py` (create)

**Interfaces:**
- Consumes: existing module-private `_normalize_unit(volume) -> (unit, lo, hi)` and `_restore_range(unit, lo, hi, original) -> np.ndarray`.
- Produces: `apply_poisson_shot_noise(volume: np.ndarray, gain: float = 80.0, random_state: Optional[int] = None, verbose: bool = False) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_poisson_noise.py`:

```python
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
    """Lower gain => more total shot noise."""
    base = np.full((48, 48, 48), 0.5, dtype=np.float32)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_poisson_noise.py -v`
Expected: FAIL with `ImportError`/`cannot import name 'apply_poisson_shot_noise'`.

- [ ] **Step 3: Implement the function**

In `syntract_viewer/synthetic_image_augmentations.py`, insert directly after `apply_granular_noise` (after its `return _restore_range(...)`, ~line 126):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_poisson_noise.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add syntract_viewer/synthetic_image_augmentations.py tests/test_poisson_noise.py
git commit -m "feat: add apply_poisson_shot_noise (signal-dependent LSM noise)"
```

---

### Task 2: Wire Poisson into the `apply_image_only_augmentations` dispatcher

**Files:**
- Modify: `syntract_viewer/synthetic_image_augmentations.py:309-390` (`apply_image_only_augmentations`)
- Test: `tests/test_poisson_noise.py` (append)

**Interfaces:**
- Consumes: `apply_poisson_shot_noise` (Task 1).
- Produces: `apply_image_only_augmentations(..., enable_poisson_noise: bool = False, poisson_gain: float = 80.0, ...)` — two new keyword params; everything else unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_poisson_noise.py`:

```python
from syntract_viewer.synthetic_image_augmentations import apply_image_only_augmentations


def test_dispatcher_poisson_disabled_is_identity():
    base = np.full((24, 24, 24), 0.5, dtype=np.float32)
    out = apply_image_only_augmentations(base, enable_poisson_noise=False, random_state=0)
    np.testing.assert_array_equal(out, base.astype(np.float32))


def test_dispatcher_poisson_enabled_changes_volume():
    base = np.full((24, 24, 24), 0.5, dtype=np.float32)
    out = apply_image_only_augmentations(
        base, enable_poisson_noise=True, poisson_gain=60.0, random_state=0
    )
    assert not np.array_equal(out, base.astype(np.float32))
    assert out.shape == base.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_poisson_noise.py -k dispatcher -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'enable_poisson_noise'`.

- [ ] **Step 3: Add the params and the call block**

In the signature (after `enable_horizontal_banding: bool = False,` at line 315), add:

```python
    enable_poisson_noise: bool = False,
```

In the signature (after `granular_noise_strength: float = 0.35,` at line 317), add:

```python
    poisson_gain: float = 80.0,
```

Add a seed alongside the others (after `banding_seed = ...` at line 343):

```python
    poisson_seed = None if random_state is None else int(random_state) + 521093
```

Insert the call block immediately after the `enable_granular_noise` block (after its closing `)` at line 367, before `if enable_speckle_noise:`):

```python
    if enable_poisson_noise:
        augmented = apply_poisson_shot_noise(
            augmented,
            gain=poisson_gain,
            random_state=poisson_seed,
            verbose=verbose,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_poisson_noise.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add syntract_viewer/synthetic_image_augmentations.py tests/test_poisson_noise.py
git commit -m "feat: wire Poisson shot noise into apply_image_only_augmentations"
```

---

### Task 3: Thread Poisson through `volume_renderer.create_3d_volume_with_streamlines`

**Files:**
- Modify: `syntract_viewer/volume_renderer.py` (signature ~283-288, trigger ~1027, call ~1044-1049, CLI ~1153-1158, args pass-through ~1202-1205)

**Interfaces:**
- Consumes: `apply_image_only_augmentations(..., enable_poisson_noise, poisson_gain, ...)` (Task 2).
- Produces: `create_3d_volume_with_streamlines(..., enable_poisson_noise=False, poisson_gain=80.0, ...)` and `--enable_poisson_noise` / `--poisson_gain` CLI flags.

- [ ] **Step 1: Add the function params**

In the signature, after `enable_horizontal_banding=False,` (line 286), add:

```python
                                       enable_poisson_noise=False,
```

After `granular_noise_strength=0.35,` (line 288), add:

```python
                                       poisson_gain=80.0,
```

- [ ] **Step 2: Add to the augmentation trigger condition**

At line 1027, change:

```python
    if enable_tissue_artifacts or enable_granular_noise or enable_speckle_noise or enable_dash_noise or enable_horizontal_banding:
```

to:

```python
    if enable_tissue_artifacts or enable_granular_noise or enable_speckle_noise or enable_dash_noise or enable_horizontal_banding or enable_poisson_noise:
```

- [ ] **Step 3: Pass into the dispatcher call**

In the `apply_image_only_augmentations(...)` call, after `enable_granular_noise=enable_granular_noise,` (line 1044), add:

```python
            enable_poisson_noise=enable_poisson_noise,
```

After `granular_noise_strength=granular_noise_strength,` (line 1049), add:

```python
            poisson_gain=poisson_gain,
```

- [ ] **Step 4: Add CLI args**

After the `--granular_noise_strength` arg (line 1158), add:

```python
    parser.add_argument('--enable_poisson_noise', action='store_true',
                        help='Add signal-dependent Poisson shot noise (image only)')
    parser.add_argument('--poisson_gain', type=float, default=80.0,
                        help='Photons-per-unit; lower=more shot noise, higher=cleaner')
```

- [ ] **Step 5: Pass args into the function call**

In the `__main__` call to `create_3d_volume_with_streamlines(...)`, after `granular_noise_strength=args.granular_noise_strength,` (line 1205), add:

```python
        enable_poisson_noise=args.enable_poisson_noise,
        poisson_gain=args.poisson_gain,
```

- [ ] **Step 6: Smoke-test the wiring**

Run:

```bash
python -c "from syntract_viewer.volume_renderer import create_3d_volume_with_streamlines as f; import inspect; p=inspect.signature(f).parameters; assert 'enable_poisson_noise' in p and 'poisson_gain' in p; print('OK')"
python syntract_viewer/volume_renderer.py --help 2>&1 | grep -E "poisson"
```

Expected: `OK`, and both `--enable_poisson_noise` and `--poisson_gain` listed.

- [ ] **Step 7: Commit**

```bash
git add syntract_viewer/volume_renderer.py
git commit -m "feat: thread Poisson shot noise through volume_renderer"
```

---

### Task 4: Thread Poisson through `precompute_patches_3d.py`

**Files:**
- Modify: `synthetic-training/precompute_patches_3d.py` (inner fn signature ~62-66, pass-through ~183-188, CLI ~252-262, args pass-through ~355-361)

**Interfaces:**
- Consumes: `create_3d_volume_with_streamlines(..., enable_poisson_noise, poisson_gain, ...)` (Task 3).
- Produces: `--enable_poisson_noise` / `--poisson_gain` CLI flags on the precompute script.

- [ ] **Step 1: Add to the inner worker function signature**

In the inner function that takes `enable_granular_noise: bool,` (line 62), add after `enable_horizontal_banding: bool,` (line 65):

```python
    enable_poisson_noise: bool,
```

After `granular_noise_strength: float,` (line 66), add:

```python
    poisson_gain: float,
```

- [ ] **Step 2: Pass into the `create_3d_volume_with_streamlines` call**

After `enable_horizontal_banding=enable_horizontal_banding,` (line 186), add:

```python
            enable_poisson_noise=enable_poisson_noise,
```

After `granular_noise_strength=granular_noise_strength,` (line 188), add:

```python
            poisson_gain=poisson_gain,
```

- [ ] **Step 3: Add CLI args**

After the `--granular_noise_strength` arg block (line 260-262), add:

```python
    parser.add_argument("--enable_poisson_noise", dest="enable_poisson_noise",
                        action="store_true", default=False,
                        help="Add signal-dependent Poisson shot noise (image only)")
    parser.add_argument("--poisson_gain", type=float, default=80.0,
                        help="Photons-per-unit; lower=more shot noise, higher=cleaner")
```

- [ ] **Step 4: Pass args into the worker call**

After `enable_horizontal_banding=args.enable_horizontal_banding,` (line 358), add:

```python
            enable_poisson_noise=args.enable_poisson_noise,
```

After `granular_noise_strength=args.granular_noise_strength,` (line 359), add:

```python
            poisson_gain=args.poisson_gain,
```

- [ ] **Step 5: Smoke-test the CLI**

Run:

```bash
python synthetic-training/precompute_patches_3d.py --help 2>&1 | grep -E "poisson"
```

Expected: `--enable_poisson_noise` and `--poisson_gain` both listed.

- [ ] **Step 6: Commit**

```bash
git add synthetic-training/precompute_patches_3d.py
git commit -m "feat: thread Poisson shot noise through precompute_patches_3d"
```

---

### Task 5: Update the SLURM configs (precompute + train)

**Files:**
- Modify: `synthetic-training/precompute_patches.sh`
- Modify: `synthetic-training/train_cached.sh`

**Interfaces:**
- Consumes: `--enable_poisson_noise` / `--poisson_gain` (Task 4); `--soft_mask`, `--mask_smoothing_sigma` (existing).
- Produces: patches in `./precomputed_patches_poisson_soft`; training reads that dir.

- [ ] **Step 1: Set the new precompute config**

In `synthetic-training/precompute_patches.sh`:

1. Change `OUTPUT_DIR` default to:

```bash
OUTPUT_DIR=${OUTPUT_DIR:-./precomputed_patches_poisson_soft}
```

2. In the `python -u precompute_patches_3d.py` invocation:
   - Change `--granular_noise_strength 2.0` back to `--granular_noise_strength 1.5`.
   - Change `--disable_horizontal_banding` back to `--enable_horizontal_banding --banding_strength 0.35 --banding_axis 1`.
   - Change `--mask_smoothing_sigma 1.0` to `--mask_smoothing_sigma 0.5`.
   - Add `--soft_mask`.
   - Add `--enable_poisson_noise --poisson_gain 80` (gain set to probe-winner in Task 6; 80 is the starting default).

The mask/noise block should read:

```bash
    --mask_smoothing_sigma 0.5 --mask_binary_threshold 0.2 \
    --soft_mask \
    --disable_tissue_artifacts \
    --enable_poisson_noise --poisson_gain 80 \
    --enable_granular_noise --granular_noise_strength 1.5 \
    --enable_speckle_noise --speckle_noise_strength 1.5 --speckle_noise_density 0.04 \
    --disable_dash_noise \
    --enable_horizontal_banding --banding_strength 0.35 --banding_axis 1
```

- [ ] **Step 2: Point training at the new patches with a fresh checkpoint dir**

In `synthetic-training/train_cached.sh`:

1. Change `PATCH_DIR` default to:

```bash
PATCH_DIR=${PATCH_DIR:-./precomputed_patches_poisson_soft}
```

2. Change `--checkpoint_dir checkpoints_cached_bf16/` to:

```bash
    --checkpoint_dir checkpoints_poisson_soft_bf16/ \
```

3. Leave load-time image augs OFF (already the case — do not add `--enable_*` image-aug flags). `--loss bce_cldice` and `--real_proxy_zarr` (561nm zarr) stay as-is for baseline continuity.

- [ ] **Step 3: Verify the scripts parse (syntax only, no submit)**

Run:

```bash
bash -n synthetic-training/precompute_patches.sh && echo "precompute OK"
bash -n synthetic-training/train_cached.sh && echo "train OK"
grep -E "poisson|soft_mask|smoothing_sigma 0.5|banding_strength 0.35|granular_noise_strength 1.5|precomputed_patches_poisson_soft|checkpoints_poisson_soft" synthetic-training/precompute_patches.sh synthetic-training/train_cached.sh
```

Expected: both `OK`, and every expected flag present.

- [ ] **Step 4: Commit**

```bash
git add synthetic-training/precompute_patches.sh synthetic-training/train_cached.sh
git commit -m "chore: precompute/train config for Poisson + soft thin masks"
```

---

### Task 6: Stats-probe gate + runlog section (validation before full run)

**Files:**
- Modify: `docs/superpowers/plans/texture-gap-runlog.md` (append a new variant section)

**Interfaces:**
- Consumes: the new precompute config (Task 5); existing `compare_domain_stats.py`.
- Produces: a recorded gain decision + GO/NO-GO for the full precompute.

This task is a runbook (runs on the cluster GPU; do NOT run a 128³ pass on a laptop). Each gain produces a tiny patch set; `compare_domain_stats.py` scores it against real LSM.

- [ ] **Step 1: Append the runlog section**

Add to the end of `docs/superpowers/plans/texture-gap-runlog.md`:

```markdown
## Poisson + soft-mask variant (banding ON, grain 1.5, soft masks)  ← CURRENT

### Knob changes
- NEW: `--enable_poisson_noise --poisson_gain <G>` (signal-dependent shot noise)
- banding: ON (`--enable_horizontal_banding --banding_strength 0.35`) — real data has it
- grain: 1.5 (baseline; Poisson is the primary texture driver, do not stack)
- masks: `--soft_mask` + `--mask_smoothing_sigma 0.5` (partial-volume, ~2 vox)
- OUTPUT_DIR: `./precomputed_patches_poisson_soft`
- checkpoints: `checkpoints_poisson_soft_bf16/`

### Phase 1 — Stats probe (gain sweep)
Target: BACKGROUND-ONLY adj_voxel_corr ~0.43, local_std ~0.15. Use background-only
corr (mask out fiber voxels) — whole-patch corr is inflated by fiber density (thick6
raised it 0.67->0.72; that's a fiber artefact, not texture). Contrast is NOT a gate:
validated locally that it's fiber-render-determined (~2.3x, matches real) and Poisson
does NOT dilute it (ON==OFF), so the old "3-4x floor" is dropped — learnability is
decided by sanity dice in Phase 3.

| gain | bg-only adj_corr | local_std | contrast (info only) | verdict |
|------|------------------|-----------|----------------------|---------|
| 80   | 0.78 (laptop)    | ~0.12     | 2.8x                 | too gentle |
| 40   | PENDING | PENDING | PENDING | PENDING |
| 20   | PENDING | PENDING | PENDING | PENDING |

Chosen gain: PENDING. Mask width check (~2-3 vox, centered, noise-decoupled): PENDING.
Gate (bg-only corr moved toward real 0.43): PENDING.

### Phase 2 — Full precompute + retrain
SLURM precompute job: PENDING | train job: PENDING | W&B run: PENDING

### Phase 3 — Result (re-baselined for soft masks)
| Signal | baseline | result | pass? |
|--------|----------|--------|-------|
| real_pred_pos_frac_median | 0.000206 (binary) | PENDING | PENDING |
| sanity dice (soft) | ~0.98 | PENDING | PENDING |
| synthetic val_dice | re-baseline run epoch-0 | PENDING | PENDING |

Verdict: PENDING.
```

- [ ] **Step 2: Run the gain sweep (cluster)**

For each gain in 40, 20 (80 already measured ≈0.78, too gentle), precompute a tiny
probe set. `precompute_patches.sh` already takes `POISSON_GAIN` and builds the thick6
TRK automatically:

```bash
for G in 40 20; do
  OUTPUT_DIR=./precomputed_patches_probe_g${G} PATCHES_PER_TRK=3 \
    POISSON_GAIN=$G sbatch synthetic-training/precompute_patches.sh
done
```

- [ ] **Step 3: Score each probe against real LSM**

```bash
for G in 40 20; do
  echo "=== gain $G ==="
  python compare_domain_stats.py \
    --synthetic_dir ./precomputed_patches_probe_g${G} \
    --real_dir debug_patches   # the real-LSM reference used for the 0.854 baseline
done
```

Fill the Phase-1 table. Pick the gain whose **background-only** `adj_voxel_corr` and
`local_std` are closest to real (0.43 / 0.15). Contrast is informational only — do not
gate on it. (If `compare_domain_stats.py` reports whole-patch corr, subtract the fiber
contribution or measure bg-only as in the local validation script.)

- [ ] **Step 4: Verify mask width on a probe patch**

```bash
python - <<'PY'
import numpy as np, nibabel as nib, glob
m = sorted(glob.glob('precomputed_patches_probe_g80/**/*_3d_mask.nii.gz', recursive=True))[0]
mask = nib.load(m).get_fdata()
from scipy.ndimage import distance_transform_edt
core = mask > 0.5
print('soft mask: max=%.3f  pos_frac=%.5f  max_halfwidth=%.1f vox' %
      (mask.max(), (core).mean(), distance_transform_edt(core).max()))
PY
```

Expected: full width ≈ 2–3 vox (max_halfwidth ≈ 1.0–1.5), `max ≤ 1.0` (soft).

- [ ] **Step 5: Set the winning gain and record GO/NO-GO**

Edit `synthetic-training/precompute_patches.sh` so `--poisson_gain` is the chosen value. Fill the runlog Phase-1 verdict. Commit:

```bash
git add synthetic-training/precompute_patches.sh docs/superpowers/plans/texture-gap-runlog.md
git commit -m "docs: Poisson+soft-mask probe results, lock winning gain"
```

- [ ] **Step 6: Full precompute + train (only if gate PASS)**

```bash
sbatch synthetic-training/precompute_patches.sh
# confirm a healthy count BEFORE training:
find synthetic-training/precomputed_patches_poisson_soft -name '*_3d.nii.gz' | wc -l
sbatch synthetic-training/train_cached.sh
```

Then track the three Phase-3 signals each val epoch and fill the runlog. If learnability regresses, raise `--poisson_gain` (less noise) and re-probe.

---

## Self-Review

**Spec coverage:**
- Part 1 A1 (Poisson fn) → Task 1. A1 wiring → Tasks 2–4. A2 (re-enable banding) → Task 5 Step 1. A3 (grain 1.5) → Task 5 Step 1. ✓
- Part 2 B1 (soft masks) + B2 (sigma 0.5) → Task 5 Step 1. B3 (verify width/decoupling) → Task 6 Step 4. ✓
- Part 3 (probe gate, full precompute, train, three-signal judging, rollback) → Task 6. ✓
- Files-touched list in spec → all covered (aug fn, volume_renderer, precompute_3d, the two .sh, runlog). ✓

**Placeholder scan:** Probe-result tables in Task 6 contain `PENDING` cells — these are intentional run-output slots in a runbook, not code placeholders. All code steps contain complete code. ✓

**Type consistency:** `enable_poisson_noise: bool` and `poisson_gain: float = 80.0` used identically across Tasks 1–4 (function param, dispatcher kwarg, renderer kwarg, precompute kwarg, CLI dest). `apply_poisson_shot_noise(volume, gain, random_state, verbose)` signature consistent between Task 1 (def) and Task 2 (call). ✓
