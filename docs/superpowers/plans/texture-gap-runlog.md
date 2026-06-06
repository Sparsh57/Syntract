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
- Domain stats (synth vs real, 1-99 percentile normalized):
  - `coarse_std`: synth 0.12 vs real 0.08
  - `adj_voxel_corr`: synth 0.44 vs real 0.29

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

## Combined variant (banding OFF, bg flattened, finer grain) — FAILED

### Knob changes

- banding: OFF (`--disable_horizontal_banding`)
- `--background_max_intensity`: 12.0 (down from default 30.0)
- `--granular_noise_strength`: 2.5 (up from 1.5)
- OUTPUT_DIR: `./precomputed_patches_flatbg`

### Phase 1 — Stats probe results (true matched baseline discovered here)

> NOTE: Documented baseline stats (0.12/0.44) were wrong — true matched baseline measured here.

| Metric | Real | True baseline synth | Combined variant | Grain-2.5 only | Grain-2.0 only |
|--------|------|--------------------|--------------------|----------------|----------------|
| `coarse_std` | 0.079 | 0.225 | 0.213 | 0.218 | 0.201 |
| `adj_voxel_corr` | 0.293 | 0.854 | 0.690 | 0.691 | 0.717 |
| `fine_std` | 0.159 | 0.097 | 0.121 | 0.114 | 0.113 |
| `grain/ratio` | 2.024 | 0.435 | 0.630 | 0.528 | 0.573 |

Key findings from single-axis probes:
- `coarse_std` is NOT driven by fibers (mask-out test: 0.1% contribution) — it is anatomical background structure in the source NIfTI, unmovable by any CLI knob
- `adj_voxel_corr` responds strongly to grain strength — grain is the lever
- `bg=12.0` did NOT raise coarse_std (hypothesis was wrong); it hurt val_dice by making patches harder to learn from
- Grain-2.5 alone moved `adj_voxel_corr` 0.854→0.691 without bg flatten

### Phase 2 — Full precompute + retrain results

Full patch count: 1800  
SLURM job id: 15478117  
W&B run: `cached_128_1780662321_BCE`

Proxy trajectory (mean only — median not logged, cluster had older callback):

| Epoch | mean | note |
|-------|------|------|
| 4 | 0.01718 | early spike |
| 14 | 0.00270 | decaying |
| 29 | 0.00434 | brief bump |
| 79 | 0.00109 | near floor |
| 149 | 0.00146 | ~1.2× baseline — no meaningful lift |

### Phase 3 — Gate judgment: **FAIL**

- Proxy (mean): 0.00146 at epoch 149 vs baseline 0.001202 — ~1.2×, not ≥2× — **FAIL**
- val_dice: 0.759 at epoch 44 (significant regression from 0.853) — **FAIL**
- Breadth: no per-region data; mean trajectory shows no sustained broad lift — **FAIL**

Decision branch: **No lift AND val_dice dropped → revert bg flatten, retry with grain-only at 2.5**

---

## Grain-only variant (banding OFF, grain 2.5, bg default) — IN PROGRESS

### Rationale

Combined variant failed because `bg=12.0` hurt val_dice. Grain-2.5 alone moved
`adj_voxel_corr` 0.854→0.691 without the harmful bg change. Banding OFF kept for
coarse_std improvement (0.225→0.218, small but correct direction).

### Knob changes

- banding: OFF (`--disable_horizontal_banding`)
- `--granular_noise_strength`: 2.5 (up from 1.5)
- `--background_max_intensity`: NOT SET (default 30.0 — no bg flatten)
- OUTPUT_DIR: `./precomputed_patches_grainonly`

### Stats gate

Real LSM patch dir: `./real_lsm_stats_patches/debug_patches/`

| Metric | Real | Baseline | Grain-2.5 only | Gate target |
|--------|------|----------|----------------|-------------|
| `coarse_std` | 0.079 | 0.225 | 0.218 | moved toward real ✓ |
| `adj_voxel_corr` | 0.293 | 0.854 | 0.691 | moved toward real ✓ |

Stats gate: **PASS** (grain-2.5-only probe already ran as single-axis probe B)

### Phase 2 — Full precompute + retrain

Full patch count: PENDING
SLURM job id: PENDING
W&B run: PENDING

### Phase 3 — Result

| Epoch | `real_pred_pos_frac_median` | per-region | `val_dice` |
|-------|----------------------------|------------|------------|
| ? | ? | ? | ? |

**Gate judgment:** PENDING

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
Proxy median (before → after): ? → ?  
Per-region (before → after): PENDING  
`val_dice` guardrail: PENDING  
Locked into `precompute_patches.sh`: PENDING  
CLAUDE.md updated: PENDING
