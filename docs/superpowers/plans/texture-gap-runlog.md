# Texture-gap experiment run log

## Baselines (from the committed/calibrated state)

- Calibration (OLD epoch-129 ckpt, 3/40 sampling): regions-1-3 mean = 0.000200; grand mean = 0.000759.
- Fresh epoch-149 run (1/0 sampling, baked baseline aug):
  - `real_pred_pos_frac` (mean) = 0.00140
  - `real_pred_pos_frac_median` = **PENDING — pull from epoch-149 W&B run**
  - Per-region: **PENDING — pull from epoch-149 W&B run**
    - `real_region1_pos_frac` = ?
    - `real_region2_pos_frac` = ?
    - `real_region3_pos_frac` = ?
    - `real_region4_pos_frac` = ?
    - `real_region5_pos_frac` = ?
    - `real_region6_pos_frac` = ?
    - `real_region7_pos_frac` = ?
    - `real_region8_pos_frac` = ?
    - `real_region9_pos_frac` = ?
  - `val_dice` = 0.853
- Domain stats (synth vs real, 1-99 percentile normalized):
  - `coarse_std`: synth 0.12 vs real 0.08
  - `adj_voxel_corr`: synth 0.44 vs real 0.29

> **Note on median baseline:** The PRIMARY success criterion is `real_pred_pos_frac_MEDIAN` lifting ≥2×
> over the fresh-run baseline median, SUSTAINED ≥3 val epochs. The mean (0.00140) is NOT the threshold —
> the median must be read from W&B before any gate judgment is possible.
>
> **Degeneracy caveat:** If the baseline median comes back ≈0 (plausible, since calibration showed only
> ~1 of 9 regions fires and the 5th-of-9 sorted value may land at floor), then "≥2×" is degenerate.
> In that case the load-bearing criterion shifts to BREADTH (≥2–3 distinct regions rise, absolute) plus
> a sustained non-zero median — not a ratio.

---

## Combined variant (banding OFF, bg flattened, finer grain)

### Knob changes (from Task 2)

- banding: OFF (`--disable_horizontal_banding`)
- `--background_max_intensity`: 12.0 (down from implicit default 30.0; may be tuned in Task 3)
- `--granular_noise_strength`: 2.5 (up from 1.5)
- `--cornucopia_presets`: unchanged (`ultra_heavy_speckle extreme_noise granular_realistic`)
- OUTPUT_DIR: `./precomputed_patches_flatbg`

### Phase 1 — Stats pre-filter (Task 3)

Real LSM patch dir used: **PENDING — identify from cluster shell history or region_* debug patches**

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
