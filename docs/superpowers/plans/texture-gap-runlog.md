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

## Combined variant (banding OFF, bg flattened, finer grain)

### Knob changes (from Task 2)

- banding: OFF (`--disable_horizontal_banding`)
- `--background_max_intensity`: 12.0 (down from implicit default 30.0; may be tuned in Task 3)
- `--granular_noise_strength`: 2.5 (up from 1.5)
- `--cornucopia_presets`: unchanged (`ultra_heavy_speckle extreme_noise granular_realistic`)
- OUTPUT_DIR: `./precomputed_patches_flatbg`

### Phase 1 — Stats pre-filter (Task 3)

Real LSM patch dir used: `./real_lsm_stats_patches/debug_patches/` (extracted via extract_real_stats_patches.sh)

Probe batch count (`find ./precomputed_patches_flatbg_probe -name '*_3d.nii.gz' | wc -l`): PENDING

| Metric | Synth (combined variant) | Real | Baseline synth | Gate target |
|--------|--------------------------|------|----------------|-------------|
| `coarse_std` | ? | 0.08 | 0.12 | ≤ ~0.09 |
| `adj_voxel_corr` | ? | 0.29 | 0.44 | ≤ ~0.33 |
| `std` | ? | — | — | — |
| `grain/gradient_ratio` | ? | — | — | — |

Gate result: PENDING (PASS = both coarse_std and adj_voxel_corr moved meaningfully toward real)

Final knobs that passed the stats gate: PENDING

### Phase 2 — Full precompute + retrain (Tasks 4-5)

Full patch count (`find ./precomputed_patches_flatbg -name '*_3d.nii.gz' | wc -l`): PENDING

SLURM job id: PENDING  
W&B run name/URL: PENDING

First val epoch sanity: PENDING

### Phase 3 — Result (Task 6)

Last ≥3 val epochs (final epoch reported first):

| Epoch | `real_pred_pos_frac_median` | `real_region1..9` (csv) | `val_dice` |
|-------|-----------------------------|-------------------------|------------|
| ? | ? | ? | ? |
| ? | ? | ? | ? |
| ? | ? | ? | ? |

**Gate judgment:** PENDING

Decision branch: PENDING

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
