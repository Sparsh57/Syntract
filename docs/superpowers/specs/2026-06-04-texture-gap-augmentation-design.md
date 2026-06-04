# Texture-gap augmentation — design (step (b))

**Date:** 2026-06-04
**Branch:** `3_dimension`
**Status:** approved design, pre-implementation
**Depends on:** committed `985e983` (held-out cached val split + calibrated real-LSM proxy)

## Problem

A 3D U-Net trained entirely on synthetic MRI-tractography volumes segments fibers
at dice ~0.98 on synthetic patches and ~0.87 on held-out synthetic patches, but
produces ~0 coverage on real light-sheet microscopy (LSM). Every confound was
eliminated in prior sessions (broken path, thin-slab geometry, normalization
order, overfit, X/Y-locality) — the residual cause is a genuine **synthetic→real
texture domain gap**.

`compare_domain_stats.py` quantifies it (synthetic vs real, normalized identically
with 1–99 percentile):

- `coarse_std` (large-scale background gradient/banding): synth **0.12** vs real **0.08** — synthetic has a fake background gradient/banding real LSM lacks.
- `adj_voxel_corr` (texture grain scale): synth **0.44** vs real **0.29** — synthetic tissue is smoother/blurrier than real fine grain.

## Goal

Make synthetic tissue texture statistically resemble real LSM enough to move the
calibrated real-LSM proxy (`real_pred_pos_frac_median`) off its ~0.0002 floor,
without regressing held-out synthetic generalization (`val_dice` ~0.85).

This spec covers the **texture-axis intervention and its bisection only**. If the
texture axes fail (see §"Branches"), the pivot to fiber appearance/contrast is a
SEPARATE design, explicitly out of scope here.

## Strategy (Approach A: combined-change-then-bisect, stats-gated)

Change all three texture axes at once in the first retrain to get fastest to a
signal off the near-zero floor; bisect to recover attribution only if it moves.
A cheap stats pre-filter guards every retrain so no 4×H200 run is spent on patches
that did not even close the measured stats gap.

### The three axes (all baked at PRECOMPUTE time — render knobs, not train-time)

| # | Axis | Knob change | Targets |
|---|------|-------------|---------|
| 1 | Drop banding | `--enable_horizontal_banding` → `--disable_horizontal_banding` (fully off) | `coarse_std` ↓ |
| 2 | Flatten background | lower `--background_max_intensity` from its current default **30.0** (cap the bright background ceiling/gradient) | `coarse_std` ↓ |
| 3 | Finer granular noise | raise `--granular_noise_strength` and/or restrict `--cornucopia_presets` to finer-grain presets | `adj_voxel_corr` ↓ |

Decisions (engineering calls, baked in):
- **Banding fully OFF**, not merely reduced. It is the prime structured-cue suspect
  (a periodic signal absent in real LSM that the model may key on); a clean removal
  gives the clearest bisection signal.
- **`background_max_intensity` derived from data, not guessed.** Tune it during the
  stats pre-filter until synthetic `coarse_std`/`p99` lands near real's, then lock.

## Workflow

### Phase 1 — Stats pre-filter (cheap gate, NO retrain)

1. Generate a small batch (a few dozen) of combined-3-axis `_3d.nii.gz` patches via
   the precompute path with the three knobs changed.
2. Run `compare_domain_stats.py` on those vs the real LSM patches.
3. Render a preview PNG of one variant patch (`visualize_one_patch.py`) as a human
   sanity check.
4. **Gate:** proceed only if BOTH `coarse_std` → ~0.08 and `adj_voxel_corr` → ~0.29
   moved meaningfully toward real, AND the preview looks plausible. If a knob
   saturated and a stat barely moved, adjust the knob value and regenerate — do NOT
   retrain a dud. This is where `background_max_intensity` is tuned empirically.

### Phase 2 — Full precompute + retrain

1. Full precompute into a FRESH patch dir (e.g. `precomputed_patches_flatbg/`) so the
   baseline patches survive for bisection. Confirm count:
   `find <dir> -name '*_3d.nii.gz' | wc -l` (≥100, the train_cached guard).
2. Retrain via the cached pipeline into a FRESH checkpoint dir with `--no_resume`
   (bf16; committed val-split + proxy active). Proxy logs each val epoch.

### Phase 3 — Judge the result (layered success bar)

Compare against the FRESH 1/0 run's OWN proxy baseline (median + per-region), NOT
the 3/40 calibration figure.

- **Primary:** `real_pred_pos_frac_median` lifts ≥2× over its baseline, SUSTAINED
  across ≥3 consecutive val epochs (not a single spike). Median is robust to the
  region-9 outlier found in calibration.
- **Confirmation:** the lift is BROAD — ≥2–3 distinct regions' per-region fracs rise,
  not just one.
- **Guardrail:** held-out synthetic `val_dice` stays healthy (~0.85+) — we did not
  wreck synthetic generalization to chase real transfer.

### Branches

- **Median lifts (broad + sustained) →** success → bisect (§Phase 4).
- **No lift but stats closed →** decisive: the gap is NOT these texture axes. PIVOT
  to fiber appearance/contrast in tissue (separate design; out of scope here).
- **No lift and `val_dice` dropped →** the change hurt synthetic learning; revert and
  reconsider knob magnitudes.

### Phase 4 — Bisection (only if combined moved the proxy)

Binary-style split to minimize retrains (each variant = precompute + retrain, both
stats-gated as in Phase 1):

1. Split axes into two groups: Group X = {drop banding} (prime suspect, isolated),
   Group Y = {flatten bg + finer grain}.
2. Read proxy median for each:
   - X alone reproduces most of the combined lift → banding was the culprit. Done.
   - Y carries it → one more split between flatten-bg and finer-grain to isolate.
   - Neither alone matches the combined → INTERACTION (axes only help together);
     keep the combined config, stop bisecting.
3. Lock the winning config into `precompute_patches.sh` / `train_cached.sh` defaults
   and record it in CLAUDE.md.

Worst case ~3 extra retrains (= isolating all three); best case 1–2. Banding-first
ordering exploits the prior that it is the most likely single cause.

## Out of scope

- Fiber appearance/contrast intervention (the pivot path) — its own design if reached.
- Any change to the model architecture, loss, or the (now-fixed) val split / proxy.
- Real labeled-data fine-tuning.

## Files touched

- Render knobs live in `precompute_patches_3d.py` CLI and the cluster scripts
  `precompute_patches.sh` / `train_cached.sh` (untracked, synced manually). The
  Python defaults flow `precompute_patches_3d.py` → `create_3d_volume_with_streamlines`
  (`syntract_viewer/volume_renderer.py`).
- No committed-code changes are required to RUN the experiment (knobs are CLI flags);
  the only committed change is locking the winning config + a CLAUDE.md note once a
  winner is found.
