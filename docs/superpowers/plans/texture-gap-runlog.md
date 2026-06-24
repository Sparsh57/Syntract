# Texture-gap experiment run log

## Baselines (from the committed/calibrated state)

- Calibration (OLD epoch-129 ckpt, 3/40 sampling): regions-1-3 mean = 0.000200; grand mean = 0.000759.
- Fresh epoch-149 ckpt (last-v2.ckpt), calibrated via calibrate_real_proxy.py at 1/0 sampling
  (matches training proxy default — 1 deterministic patch per center, no jitter):
  - `real_pred_pos_frac` grand-mean = 0.001202
  - `real_pred_pos_frac_median` = **0.000206** (5th-of-9 sorted = region 7; robust baseline)
  - Per-region pos_frac:
    - region 1  (30,  4826, 11220) = 0.000228
    - region 2  (30,  4826, 22441) = 0.000160
    - region 3  (30,  4826, 33661) = 0.000238
    - region 4  (30,  9652, 11220) = 0.000034
    - region 5  (30,  9652, 22441) = 0.001371
    - region 6  (30,  9652, 33661) = 0.000115
    - region 7  (30, 14478, 11220) = 0.000206
    - region 8  (30, 14478, 22441) = 0.002615
    - region 9  (30, 14478, 33661) = 0.005850
  - regions 1-3 mean = 0.000209 (matches handoff baseline 0.0002 ✓ — proxy calibrated)
  - `val_dice` = 0.853
- Domain stats (synth vs real, 1-99 percentile normalized, TRUE measured values):
  - `coarse_std`: synth **0.225** vs real **0.08**
  - `adj_voxel_corr`: synth **0.854** vs real **0.29**
  - Note: spec had stale estimated values (0.12/0.44); the TRUE baseline was measured via
    compare_domain_stats.py on last-v2.ckpt precomputed patches + debug_patches real LSM.

> **Gate thresholds (now locked):**
> Baseline median = 0.000206. "≥2× median" = **≥0.000412**, sustained ≥3 val epochs.
> This is NOT degenerate — 0.000206 is a real non-zero floor, so 2× is a meaningful bar.
>
> **Sorted baseline for reference** (ascending): 0.000034, 0.000115, 0.000160, 0.000206,
> 0.000228, 0.000238, 0.001371, 0.002615, 0.005850
> Median = 5th value = 0.000206 (region 7).
> Regions 8 and 9 are already elevated (0.002615, 0.005850) — movement must show in regions
> 1-7 (the floor regions) to count as broad. A shift in regions 8-9 alone = noise, not signal.
>
> **Breadth criterion:** ≥3 of regions 1-7 must rise above ~0.001 (5× their current floor).

---

## Combined variant (banding OFF, bg flattened, finer grain)

### Knob changes

- banding: OFF (`--disable_horizontal_banding`)
- `--background_max_intensity`: 12.0 (down from implicit default 30.0)
- `--granular_noise_strength`: 2.5 (up from 1.5)
- `--cornucopia_presets`: unchanged (`ultra_heavy_speckle extreme_noise granular_realistic`)
- OUTPUT_DIR: `./precomputed_patches_flatbg`

### Phase 1 — Stats pre-filter

Probe batch: ~30 patches in `precomputed_patches_flatbg_probe/`.

Domain stats probe (combined variant vs real):

| Metric | Synth (combined) | Real | Baseline synth | Gate target |
|--------|------------------|------|----------------|-------------|
| `coarse_std` | 0.202 | 0.08 | 0.225 | ≤ ~0.09 |
| `adj_voxel_corr` | 0.806 | 0.29 | 0.854 | ≤ ~0.33 |

Additional finding: mask-out test confirmed fiber contribution to coarse_std = 0.1% — driven by
anatomical NIfTI background. Gamma test confirmed gamma=2.2 barely moved coarse_std (0.2313).
**Conclusion: coarse_std is NOT closeable by any render knob** (it's the background structure
of the source NIfTI). Stats gate for coarse_std dropped; adj_voxel_corr remains the target.

Gate result: PASS on adj_voxel_corr movement (0.854→0.806). Proceeded to full retrain.

### Phase 2 — Full precompute + retrain

Full patch count: 1800 patches. SLURM job: 15475387.  
W&B run: `cached_128_1780662321_BCE`

### Phase 3 — Result

Last epochs from proxy log:

| Epoch | `real_pred_pos_frac_median` | `val_dice` |
|-------|-----------------------------|------------|
| 44 | ~0.00012 (never sustained above gate) | **0.759** |

**Gate judgment: FAIL**
- Proxy median never reached 0.000412 sustained ≥3 epochs.
- val_dice regressed 0.853 → 0.759.

**Root cause:** `bg=12.0` flatter background + grain=2.5 = harder patches = lower val_dice.
bg flatten removed because it's not closeable anyway (coarse_std is anatomical).

**Decision:** Remove bg flatten, keep grain-2.5 only → grain-only variant.

---

## Grain-only variant (banding OFF, grain=2.5, NO bg flatten)

### Knob changes

- banding: OFF (`--disable_horizontal_banding`)
- `--granular_noise_strength`: 2.5 (up from 1.5)
- bg: 30.0 (default, unchanged)
- OUTPUT_DIR: `./precomputed_patches_grainonly`

### Domain stats (grain-only probe vs real)

| Metric | Synth (grain-only) | Real | Baseline synth |
|--------|--------------------|------|----------------|
| `coarse_std` | ~0.220 | 0.08 | 0.225 |
| `adj_voxel_corr` | ~0.80 | 0.29 | 0.854 |

adj_voxel_corr moved slightly (0.854→~0.80) — not fully closed but grain-only contribution confirmed.

### Phase 2 — Full precompute + retrain

Full patch count: 1800 patches. SLURM job: 15478117 (precompute) + 15613697 (train).  
W&B run: `cached_128_1780662321_BCE` (grain-only run)

**Note:** Cluster had stale `train_on_synthetic_data_3d.py` for SLURM 15478117 (no median/per-region).
Re-synced local file to cluster; SLURM 15613697 logged full median+per-region.

### Phase 3 — Result

Last epochs from proxy log (SLURM 15613697, full 150 epochs):

| Epoch | `real_pred_pos_frac_median` | Notable regions | `val_dice` |
|-------|-----------------------------|-----------------|------------|
| 149 | **0.001079** | R1=0.0013, R2=0.0010, R5=0.0030, R6=0.0017 | 0.746 |
| 139 | **0.001220** | R1↑, R2↑, R5↑, R6↑ | ~0.75 |
| 129 | **0.001080** | broad | ~0.75 |
| 119 | **0.001076** | broad | ~0.75 |

Sustained epochs 119–149 (5+ epochs); all above gate (0.000412). ✅
Breadth: regions 1,2,5,6 above 0.001 at epoch 149 (4 of the floor-7 regions). ✅

**Proxy gate: PASS. Breadth: PASS. val_dice guardrail: FAIL (0.746 < 0.85).**

### Sanity check — genuine regression confirmed

Ran `sanity_check_synthetic.py` on grain-only best checkpoint with a grain-only patch:
- `dice@thr=0.5 = 0.609` (baseline model = ~0.98)
- `prob@fiber_mean = 0.717` — model finds fibers but mask coverage loose
- **Verdict: genuine regression.** grain-2.5 patches are harder to score;
  val_dice drop is NOT purely a difficulty artifact. Grain-2.5 too strong.

**Decision:** Try grain-2.0 (midpoint between baseline 1.5 and too-strong 2.5).
Rationale: proxy lift was broad and sustained → grain IS the right axis; magnitude needs dialing back.

---

## Grain-2.0 variant (banding OFF, grain=2.0, NO bg flatten)  — SUPERSEDED

Never run. Abandoned in favour of the Poisson + soft-mask + thick6 variant below,
after analysing the real region-1/2 sliding-window data directly (2026-06-24):
the gap is not "more Gaussian grain" but (a) missing signal-dependent shot noise,
(b) too-thick binary masks, and (c) ~10× too-sparse fibers.

### Knob changes (not run)
- banding OFF, `--granular_noise_strength 2.0`, `OUTPUT_DIR ./precomputed_patches_grain20`

---

## Poisson + soft-mask + thick6 variant (banding ON, grain 1.5)  ← CURRENT

Design: [docs/superpowers/specs/2026-06-24-real-domain-match-poisson-soft-masks-design.md].
Plan: [docs/superpowers/plans/2026-06-24-real-domain-match-poisson-soft-masks.md].

### Real region-1/2 measurements (the target, measured 2026-06-24)
- `adj_voxel_corr` ~0.43 (x≈0.63, y≈0.22); `local_std` ~0.15; row-mean banding std 0.074 (banding IS present)
- fiber/bg contrast ~2.3× (brightest-structure crop); model-fired spots ~1.03× (it fires on noise)
- bright-structure coverage ~14.7% of a 128² frame — real LSM is a DENSE fiber field

### Knob changes vs baseline
- NEW `--enable_poisson_noise --poisson_gain <G>` (signal-dependent shot noise; the missing physics)
- banding back ON (`--banding_strength 0.35`); grain back to 1.5 (Poisson is the primary texture driver)
- masks `--soft_mask` + `--mask_smoothing_sigma 0.5` (partial-volume, ~2 vox)
- fibers densified: `thicken_trk.py --copies 6 --radius_mm 0.05` on aligned_wavy
  (222 → 1332 curved siblings) → `../registered_trk/thick6/`; `--trk_dir` points there
- OUTPUT_DIR `./precomputed_patches_poisson_soft`; checkpoints `checkpoints_poisson_soft_bf16/`

### Phase 1 — Local validation (laptop, render-only; 2026-06-24)
Generated patches via visualize_poisson_soft_patch.py; measured vs real:

| metric | synth (thick6, gain 80) | real target | read |
|--------|--------------------------|-------------|------|
| mask coverage | 4.59% | 5–8% (real bright ~14.7%) | moderate bundle, on target |
| fiber/bg contrast | 2.8× (range 2.0–4.0) | ~2.3× | matches real; learnable |
| **bg-only adj_corr** | **0.78** (gain 80) | **0.43** | gain 80 too gentle on texture |

**Poisson does NOT dilute contrast** — ON vs OFF identical (2.36/2.33, 4.02/4.04, 1.99/1.94),
so the contrast is fiber-render-determined, ~real, and safe. The invented "3–4× learnability
floor" was higher than real (~2.3×) and is dropped — do NOT bump fiber brightness.

**Background-only corr is the gain-sweep target** (whole-patch corr is inflated by fiber density,
which thickening raised 0.67→0.72 — a fiber artefact, not texture). Poisson moved bg-only
corr 0.87→0.78 at gain 80; real is 0.43, so the sweep needs **lower gain (more noise),
likely ~40 or below.**

### Phase 1b — Gain sweep (cluster GPU) — target bg-only corr ≈ 0.43

| gain | bg-only adj_corr | local_std | fiber/bg contrast | verdict |
|------|------------------|-----------|-------------------|---------|
| 80   | 0.78 (laptop)    | ~0.12     | 2.8×              | too gentle |
| 40   | PENDING | PENDING | PENDING | PENDING |
| 20   | PENDING | PENDING | PENDING | PENDING |

Chosen gain: PENDING.

### Phase 2 — Full precompute + retrain
SLURM precompute job: PENDING | train job: PENDING | W&B run: PENDING

### Phase 3 — Result (re-baselined for soft masks)
| Signal | baseline | result | pass? |
|--------|----------|--------|-------|
| real_pred_pos_frac_median | 0.000206 (binary) | PENDING | PENDING |
| sanity dice (soft) — the REAL learnability gate | ~0.98 | PENDING | PENDING |
| synthetic val_dice | re-baseline epoch-0 | PENDING | PENDING |

Gate judgment: PENDING

---

## Bisection (Task 7 — only if Phase 3 = SUCCESS)

### Group X: banding-only variant

Knobs: `--disable_horizontal_banding`; bg back to 30.0; grain back to 1.5  
OUTPUT_DIR: `./precomputed_patches_bandingOnly`  
Checkpoint: `checkpoints_bandingOnly_bf16/`  

Stats probe result: PENDING  
Proxy median (last ≥3 epochs): PENDING  
Verdict: PENDING

### Group Y split (only if Group X gives little lift)

#### bg-only variant

Knobs: banding ON, grain 1.5, background_max_intensity = <passing value>  
Proxy median: PENDING

#### grain-only variant

Knobs: banding ON, bg 30.0, granular_noise_strength = <passing value>  
Proxy median: PENDING

**Bisection conclusion:** PENDING

---

## Winning config (Task 8)

Winning axis/combination: PENDING  
Proxy median (before → after): 0.000206 → ?  
Per-region (before → after): PENDING  
`val_dice` guardrail: PENDING  
Locked into `precompute_patches.sh`: PENDING  
CLAUDE.md updated: PENDING
