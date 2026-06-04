# Texture-gap Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the synthetic→real texture domain gap by changing three precompute-time augmentation axes at once (drop banding, flatten background, finer granular noise), gated by a cheap stats check, then bisect to find the cause if the calibrated real-LSM proxy moves.

**Architecture:** This is a CLUSTER EXPERIMENT RUNBOOK, not a code feature. The augmentation knobs are existing CLI flags on `precompute_patches_3d.py` driven by the untracked cluster scripts `precompute_patches.sh` / `train_cached.sh`. No committed-code change is required to RUN the experiment; the only code change is locking the winning config at the end. "Tests" here are empirical gates (domain-stats deltas, proxy median lift) read from GPU runs — not pytest. Every retrain is a 4×H200 SLURM job, so each is guarded by a cheap pre-filter.

**Tech Stack:** MIT ORCD SLURM (H200 train / A100 test), PyTorch Lightning, W&B project `syntract3d`, `precompute_patches_3d.py`, `compare_domain_stats.py`, `visualize_one_patch.py`, `calibrate_real_proxy.py`.

**Spec:** `docs/superpowers/specs/2026-06-04-texture-gap-augmentation-design.md`

---

## Execution environment & conventions

- All commands run on the CLUSTER (`/orcd/home/002/sparsh/syntract-3d/`), venv `../venv`. NOT locally (a 128³ forward OOMs a laptop).
- The `.sh` scripts are untracked (gitignored) and synced to the cluster manually. Editing them is editing a LOCAL cluster file, then `sbatch`-ing it. Do not `git add` them.
- "Baseline proxy" = the FRESH 1/0 run's own `real_pred_pos_frac_median` + per-region readings from W&B (the epoch-149 run logged mean 0.00140). NOT the 3/40 calibration figure (0.000759 grand / 0.0002 regions-1-3).
- Current knobs in `precompute_patches.sh` (the values to CHANGE): `--enable_horizontal_banding --banding_strength 0.35`, `--enable_granular_noise --granular_noise_strength 1.5`, `--cornucopia_presets ultra_heavy_speckle extreme_noise granular_realistic`, and `--background_max_intensity` is NOT passed (so defaults to 30.0 in `precompute_patches_3d.py`).

---

## Task 1: Capture the real-data baseline numbers (no compute)

**Files:** none (record into a scratch note `docs/superpowers/plans/texture-gap-runlog.md`).

- [ ] **Step 1: Create the run log file**

Create `docs/superpowers/plans/texture-gap-runlog.md`:

```markdown
# Texture-gap experiment run log

## Baselines (from the committed/calibrated state)
- Calibration (OLD epoch-129 ckpt, 3/40): regions-1-3 mean = 0.000200; grand mean = 0.000759.
- Fresh epoch-149 run (1/0, baked baseline aug): real_pred_pos_frac (mean) = 0.00140; val_dice = 0.853.
- Domain stats (synth vs real): coarse_std 0.12 vs 0.08; adj_voxel_corr 0.44 vs 0.29.

## Combined variant (banding OFF, bg flattened, finer grain)
- (filled in below)
```

- [ ] **Step 2: Read the fresh-run proxy MEDIAN + per-region from W&B**

On W&B project `syntract3d`, open the epoch-149 cached run. Record the FINAL-epoch values of
`real_pred_pos_frac`, `real_pred_pos_frac_median`, and each `real_regionN_pos_frac` into the run log
under "Fresh epoch-149 run". These are the numbers the combined variant must BEAT.
Expected: median at or below the 0.00140 mean (median is robust to region 9); per-region mostly near 1e-4..1e-3 with one region (region 9) higher.

- [ ] **Step 3: Commit the run log**

```bash
git add docs/superpowers/plans/texture-gap-runlog.md
git commit -m "Add texture-gap experiment run log with baselines"
```

---

## Task 2: Make the combined-variant precompute script (3 knobs changed)

**Files:**
- Modify (LOCAL cluster file, do NOT git add): `synthetic-training/precompute_patches.sh`

- [ ] **Step 1: Copy the precompute script to a variant**

On the cluster:

```bash
cd /orcd/home/002/sparsh/syntract-3d/synthetic-training
cp precompute_patches.sh precompute_patches_flatbg.sh
```

- [ ] **Step 2: Change OUTPUT_DIR to a fresh dir (preserve baseline patches for bisection)**

In `precompute_patches_flatbg.sh`, change the OUTPUT_DIR default line:

```bash
OUTPUT_DIR=${OUTPUT_DIR:-./precomputed_patches_flatbg}
```

- [ ] **Step 3: Apply the three axis changes to the python invocation**

In `precompute_patches_flatbg.sh`, in the `python -u precompute_patches_3d.py \` block:

1. Axis 1 (banding OFF): replace
   `--enable_horizontal_banding --banding_strength 0.35 --banding_axis 1`
   with
   `--disable_horizontal_banding`

2. Axis 2 (flatten background): add this flag to the block (start point; tuned in Task 3):
   `--background_max_intensity 12.0 \`
   (down from the implicit default 30.0)

3. Axis 3 (finer granular noise): raise the strength — change
   `--enable_granular_noise --granular_noise_strength 1.5`
   to
   `--enable_granular_noise --granular_noise_strength 2.5`

Leave `--cornucopia_presets ultra_heavy_speckle extreme_noise granular_realistic` AS-IS for now
(grain is being pushed via granular_noise_strength; preset changes are a separate lever held in reserve).

- [ ] **Step 4: Confirm the small-batch override already works (no edit needed)**

The script ALREADY supports the override: `precompute_patches.sh` line ~34 has
`PATCHES_PER_TRK=${PATCHES_PER_TRK:-300}` and passes `--patches_per_trk "${PATCHES_PER_TRK}"` to the
python call. So Task 3 can generate a probe batch with `PATCHES_PER_TRK=20 bash precompute_patches_flatbg.sh`.
No change required — just confirm the variant copy still has these two lines after your edits.

---

## Task 3: Phase 1 — Stats pre-filter (cheap gate, NO retrain)

**Files:** none committed. Produces patches in `./precomputed_patches_flatbg_probe/` + a stats printout + a preview PNG.

- [ ] **Step 1: Generate a SMALL probe batch of combined-variant patches**

On the cluster (A100 or H200; small job):

```bash
cd /orcd/home/002/sparsh/syntract-3d/synthetic-training
OUTPUT_DIR=./precomputed_patches_flatbg_probe PATCHES_PER_TRK=20 bash precompute_patches_flatbg.sh
```

Expected: a few dozen `*_3d.nii.gz` in `./precomputed_patches_flatbg_probe/<trk_stem>/`.
Confirm: `find ./precomputed_patches_flatbg_probe -name '*_3d.nii.gz' | wc -l` returns ≥20.
(If 0: the `.nii` vs `.nii.gz` discovery bug — see CLAUDE.md; check both extensions are matched.)

- [ ] **Step 2: Run the domain-stats comparison vs real**

```bash
cd /orcd/home/002/sparsh/syntract-3d/synthetic-training
python compare_domain_stats.py \
  --synth_dir ./precomputed_patches_flatbg_probe \
  --real_dir <DIR OF REAL LSM PATCHES used by prior compare_domain_stats runs>
```

(The real patch dir is whatever the prior domain-stats run used — find it in shell history or the
multiregion/region_* debug patches. If none saved, extract a handful via test_specific_region.py with
--save_debug_patches first.)
Record the printed `coarse_std`, `adj_voxel_corr`, `std`, `grain/gradient_ratio` (SYNTH column) into the run log.

- [ ] **Step 3: GATE — did the stats move toward real?**

PASS condition (both must hold, vs the baseline synth column coarse_std 0.12 / adj_voxel_corr 0.44):
- `coarse_std` dropped clearly toward real 0.08 (e.g. ≤ ~0.09).
- `adj_voxel_corr` dropped clearly toward real 0.29 (e.g. ≤ ~0.33).

If FAIL on coarse_std: lower `--background_max_intensity` further (e.g. 12.0 → 8.0), regenerate (Step 1), recheck.
If FAIL on adj_voxel_corr: raise `--granular_noise_strength` (e.g. 2.5 → 3.5) OR swap a cornucopia preset to a finer one, regenerate, recheck.
Iterate Steps 1-3 until BOTH pass. Each iteration is cheap (no training). Record the final knob values that passed.

- [ ] **Step 4: Render a preview PNG sanity check**

```bash
cd /orcd/home/002/sparsh/syntract-3d
python visualize_one_patch.py --out flatbg_preview.png
```

(If `visualize_one_patch.py` uses fixed render knobs rather than reading the variant, pass the same
3 changed knobs to it, OR just `scp` one `precomputed_patches_flatbg_probe/.../*_3d.nii.gz` down and
view a slice. The goal is a human "this looks like plausible flat-bg fine-grain tissue, not garbage".)
Eyeball it. If it looks like structured artifact / blown-out / empty, fix knobs and return to Step 1.

- [ ] **Step 5: Lock the passing knob values into the variant script + log**

Edit `precompute_patches_flatbg.sh` so `--background_max_intensity` and `--granular_noise_strength`
hold the values that passed the gate. Append to the run log:

```markdown
### Combined variant — knobs that passed the stats gate
- banding: OFF
- background_max_intensity: <value>
- granular_noise_strength: <value>
- cornucopia_presets: <unchanged or list>
- stats achieved: coarse_std=<v> (real 0.08), adj_voxel_corr=<v> (real 0.29)
```

---

## Task 4: Phase 2 — Full precompute of the combined variant

**Files:** none committed. Produces `./precomputed_patches_flatbg/` (full count).

- [ ] **Step 1: Run the full precompute (NOT the probe override)**

```bash
cd /orcd/home/002/sparsh/syntract-3d/synthetic-training
sbatch precompute_patches_flatbg.sh
```

(This uses the FULL default PATCHES_PER_TRK and writes to `./precomputed_patches_flatbg`.)

- [ ] **Step 2: CONFIRM a healthy patch count before training**

```bash
find /orcd/home/002/sparsh/syntract-3d/synthetic-training/precomputed_patches_flatbg -name '*_3d.nii.gz' | wc -l
```

Expected: comparable to the baseline `precomputed_patches/` count (the train_cached guard refuses < 100).
If far lower than baseline: a TRK starved or the discovery bug bit — investigate before training.
Record the count in the run log.

---

## Task 5: Phase 2 — Retrain on the combined-variant patches

**Files:**
- Modify (LOCAL cluster file, do NOT git add): `synthetic-training/train_cached.sh`

- [ ] **Step 1: Copy train_cached.sh to a variant pointing at the new patches + fresh checkpoint dir**

```bash
cd /orcd/home/002/sparsh/syntract-3d/synthetic-training
cp train_cached.sh train_cached_flatbg.sh
```

In `train_cached_flatbg.sh`:
- Change `PATCH_DIR=${PATCH_DIR:-./precomputed_patches}` → `PATCH_DIR=${PATCH_DIR:-./precomputed_patches_flatbg}`
- Change `--checkpoint_dir checkpoints_cached_bf16/` → `--checkpoint_dir checkpoints_flatbg_bf16/`
- Keep `--no_resume` (fresh init; the new checkpoint dir is empty anyway).
- Keep `--real_proxy_zarr <zarr>` so the proxy logs each val epoch.
- Keep `--val_fraction 0.15` (committed default) so the held-out synthetic split + val_dice guardrail are active.

- [ ] **Step 2: Launch the retrain**

```bash
cd /orcd/home/002/sparsh/syntract-3d/synthetic-training
sbatch train_cached_flatbg.sh
```

Record the SLURM job id and W&B run name/URL in the run log.

- [ ] **Step 3: Watch the FIRST val epoch (NCCL + proxy sanity)**

Tail the SLURM output. On the first val epoch, confirm the `[real-proxy]` line prints (mean/median/per_region)
and the job does not hang (rank-0 does 9 zarr reads while other ranks wait; patches cache after epoch 1).
Confirm loss is FINITE past step ~10k (bf16 fix). If NaN/flat → stop, it's not a texture issue.

---

## Task 6: Phase 3 — Judge the result against the success bar

**Files:** run log.

- [ ] **Step 1: Read the proxy trajectory from W&B**

From the `train_cached_flatbg` run, record across the LAST ≥3 val epochs:
`real_pred_pos_frac_median` (primary), each `real_regionN_pos_frac` (breadth), and `val_dice` (guardrail).

- [ ] **Step 2: Apply the layered success bar**

- PRIMARY: did `real_pred_pos_frac_median` lift ≥2× over the fresh-run baseline median (Task 1 Step 2),
  SUSTAINED across the last ≥3 val epochs (not a single spike)?
- CONFIRMATION: did ≥2-3 DISTINCT regions' `real_regionN_pos_frac` rise (broad, not one region)?
- GUARDRAIL: is `val_dice` still ~0.85+ (synthetic generalization not wrecked)?

- [ ] **Step 3: Branch on the verdict (record decision + reasoning in the run log)**

- MEDIAN lifts (broad + sustained) AND val_dice healthy → SUCCESS → go to Task 7 (bisect).
- NO lift but stats closed (Task 3 passed) → decisive: gap is NOT these texture axes →
  STOP this plan; open a SEPARATE brainstorm for the fiber-appearance/contrast pivot (out of scope here).
- NO lift AND val_dice dropped → the change hurt synthetic learning → revert; reconsider knob magnitudes
  (likely background flattened too far or grain too strong). Loop back to Task 3 with gentler values.

---

## Task 7: Phase 4 — Bisection (ONLY if Task 6 = SUCCESS)

**Files:** LOCAL cluster `.sh` variants (do NOT git add). Each sub-task = 1 precompute + 1 retrain, stats-gated.

- [ ] **Step 1: Build the "banding-only" variant (Group X = prime suspect, isolated)**

Copy `precompute_patches_flatbg.sh` → `precompute_patches_bandingOnly.sh`. From the COMBINED config,
REVERT axes 2 and 3 back to baseline (background_max_intensity back to 30.0 / drop the flag;
granular_noise_strength back to 1.5), keeping ONLY banding OFF. OUTPUT_DIR=`./precomputed_patches_bandingOnly`.
Run the Task 3 stats probe on it (quick), then full precompute (Task 4 pattern), then retrain
(Task 5 pattern) into `checkpoints_bandingOnly_bf16/`.

- [ ] **Step 2: Read the banding-only proxy median; compare to the combined lift**

- If banding-only reproduces MOST of the combined median lift → BANDING was the culprit. Done; go to Step 4.
- If banding-only gives LITTLE lift → the cause is in Group Y (bg + grain) → Step 3.

- [ ] **Step 3: Split Group Y (bg-flatten vs finer-grain) only if needed**

Build `precompute_patches_bgOnly.sh` (banding back ON, grain back to 1.5, ONLY background_max_intensity
lowered) → precompute → retrain → read proxy. Whatever carries the lift is the contributor. If NEITHER
bg-only nor grain-only matches the combined lift → it's an INTERACTION; keep the COMBINED config.

- [ ] **Step 4: Record the winning config**

Append the winning axis (or "interaction → keep combined") + its proxy median + per-region to the run log.

---

## Task 8: Lock the winning config (the only committed-code change)

**Files:**
- Modify: `CLAUDE.md` (document the winning augmentation config + result)
- Modify (LOCAL cluster, manual sync, NOT git add): `precompute_patches.sh` defaults → the winner

- [ ] **Step 1: Update the cluster precompute_patches.sh defaults to the winning knobs**

Edit (local cluster) `precompute_patches.sh` so its default knobs ARE the winning config, so future
precomputes use it without overrides. (Untracked file — sync manually; do not git add.)

- [ ] **Step 2: Document the winner in CLAUDE.md**

Add to the "Cached (precompute) 3D training" or a new "Texture-gap fix" subsection: which axis closed the
gap, the before/after proxy median + per-region, the before/after domain stats, and the val_dice guardrail
value. Reference the spec and run log.

- [ ] **Step 3: Commit the CLAUDE.md + run log**

```bash
cd /Users/sparshmakharia/Documents/syntract-3d
git add CLAUDE.md docs/superpowers/plans/texture-gap-runlog.md
git commit -m "Document texture-gap augmentation fix and winning config"
```

- [ ] **Step 4: Update the knowledge graph + memory**

```bash
cd /Users/sparshmakharia/Documents/syntract-3d
graphify update .
```

Update the memory file `project_val_split_proxy.md` (or a new memory) with the step-(b) outcome.

---

## Self-review notes

- Spec coverage: Phase 1 (Task 3), Phase 2 (Tasks 4-5), Phase 3 + branches (Task 6), Phase 4 bisection (Task 7), lock-winner (Task 8). Three axes + derived background_max_intensity + banding-fully-off all in Task 2-3. ✓
- The "no committed-code change to run" reality is stated up front; the only git commits are the run log (Task 1), CLAUDE.md + run log (Task 8) — honest.
- Out-of-scope fiber-appearance pivot is a STOP+separate-brainstorm in Task 6, not a task here. ✓
- Real-patch dir for compare_domain_stats is the one genuine unknown (Task 3 Step 2) — flagged explicitly rather than hidden behind a placeholder.
