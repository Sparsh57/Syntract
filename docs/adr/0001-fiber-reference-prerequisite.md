# ADR 0001 — Pause model iteration until an external fiber reference exists

- Status: Accepted
- Date: 2026-06-11
- Deciders: project owner (Sparsh), via goal-clarification (grill-with-docs) session

## Context

The project's deliverable is a voxel-wise **Fiber Segmenter** for real **LSM
data** (the 561nm monkey-slice OME-Zarr, LINC). See `CONTEXT.md` for canonical
terms.

Symptoms that triggered this session:
- The trained 3D U-Net predicts ~nothing on real LSM (`pred_pos_frac ≈ 0.0002`),
  while scoring well on synthetic (held-out synthetic dice ≈ 0.85).
- Months of run-log effort (augmentation, cornucopia presets, `pos_weight`
  sweeps) produced no trustworthy movement on real data.

What we established this session by inspecting the actual model-input patches
(`Model_prediction/patch_0001_model_input.npy`):
1. The real data is **not empty noise.** Single 2D slices look like grain +
   bright cell puncta (why the data was perceived as "noise"); a depth
   max-projection reveals coherent diagonal/parallel elongated structure.
2. That structure is **real 3D content, not a light-sheet stripe artifact** — a
   depth-decorrelation test (through-depth correlation collapses to ~0 within
   ~10 slices) and orthogonal cross-sections (no full-span curtains) rule out the
   artifact hypothesis.
3. The model is not merely empty — it fires on the **brightest region** and
   misses the elongated structure. It learned "bright tube" from synthetic data;
   real fibers are thin and **low-contrast**. This is a concrete, visible
   synthetic↔real domain gap.

The binding constraint, however, is upstream of all modeling:
- The **domain owner cannot distinguish a fiber from a non-fiber** in this data,
  and there is **no ground truth**.
- There is **no co-registration** linking the LSM to the `MF278`
  blockface/tractography frame (the omezarr loader samples by physical scale at
  arbitrary grid locations), and it is **unconfirmed** whether the LSM is even
  the same brain as `MF278`.
- Consequently the segmentation **target is undefined**: we cannot validate,
  annotate, judge synthetic realism, or grade any detector (classical or
  learned). `pred_pos_frac` is a *sparsity* check, never a correctness metric.

## Decision

**Pause model and augmentation iteration.** Make **"obtain one external source of
fiber ground-truth / reference"** the prerequisite milestone (Milestone 0).

Next action (owner): a **provenance lookup with the data producer / LINC**:
- (a) **What does the 561nm channel label?** (injected tracer? myelin/other
  stain? autofluorescence?) — likely *defines* what a fiber is and at what scale.
- (b) **Is the LSM the same brain as `MF278`, and can it be registered to the
  blockface/tractography frame?** — if yes, the existing TRKs *locate* fibers in
  the LSM with no eye-recognition needed.

If neither is reachable, do **not** resume training. Fall back to imaging-physics
characterization plus an external expert consult to define the target first.

## Consequences

- **Positive:** stops optimizing an undefined target; once a reference lands it
  simultaneously unblocks validation (a real metric replacing `pred_pos_frac`),
  annotation, and a correct spec for the synthetic generator. It also reframes
  the ML scope — the learned model's real job is probably distractor rejection
  (fiber vs. cell/vessel) and/or being a fast generalizing stand-in for a
  reference detector, not pure synthetic→real transfer.
- **Negative:** blocks the ML track until provenance resolves (collaborator-paced,
  possibly days–weeks). Some synthetic-only effort is sunk.
- **Preserved, not discarded:** the synthesis + 3D training infrastructure stays;
  it becomes useful again — and correctly targeted — once the fiber is defined.

## Alternatives considered

- **Keep tuning synthetic augmentation (status quo).** Rejected: optimizes an
  undefined target; the run log already shows no trustworthy real-data signal.
- **Use Frangi/Sato tubularity as pseudo-labels.** Deferred: the domain owner
  reports it does not work well on this data, and — lacking a reference — its
  output cannot be verified anyway. May return as a *teacher* once a reference
  confirms what real fibers look like.
- **Hand annotation.** Currently impossible: the expert cannot recognize fibers
  by eye, so there is nothing to annotate from.

## Update 2026-06-11 — reference dataset located; channel key is institutional

Investigated the real data directly (extractors: `extract_lsm_patches.py`,
`extract_mf283_slice036.py`):

- The original LINC asset the model ran on (`...561laser_xyMIP_Stitched.ome.zarr`)
  is a **2D depth max-projection** (z=1), where through-plane fibers appear as
  puncta indistinguishable from cell bodies.
- A richer reference exists: **DANDI 001412 / MF283 slice036** — a true 3D volume
  (402 × ~29k × 45k @ ~1µm) with **four fluorescence channels (488/561/594/660)**
  and an **OCT** label-free image. Channels visibly differ (488/594 streaks;
  561/660 puncta), so the fiber channel is identifiable in principle.
- A fiber-vs-banding decorrelation test on 488/594 was **inconclusive**: the
  streaks are z-aligned (high through-depth correlation), consistent with BOTH
  through-plane fibers AND illumination banding — the test cannot separate them.
- The **channel→label key is not in any machine-readable metadata** (001412 asset
  meta, 001372 `dataset_description.json`, OME-Zarr `omero` attrs all lack it).

**Confirmed next action (owner / human):** obtain the channel-label key from LINC
(protocol / gallery.lincbrain.org / producer) — which wavelength labels fibers.
Then: (a) that channel defines the synthetic-fiber appearance, and (b) register
the OCT to it for a label-free validation reference. Model iteration stays paused
until the fiber channel is identified.

## Update 2026-06-11 (later) — Continuity path: modeling UNBLOCKED without a reference

The channel key turned out to be unreachable (owner cannot obtain it; not in any
metadata). Rather than stay blocked, we adopted a **structural prior** that needs
no labels, no OCT, and no provenance: a Fiber is a **long connected continuous
curve, not a blob** (see CONTEXT.md "Continuity"). This supersedes the "pause
everything" stance — modeling can proceed, judged by connectivity.

Evidence: connected-component analysis shows the synthetic targets ARE connected
fibers (tiny_blob_frac=0.0, length/thickness ~56) but the model on real LSM emits
scattered blobs (tiny_blob_frac 0.57-0.67, length/thickness ~5-10) — a
continuity-transfer failure, not a target or reference problem.

Implemented this session:
- `synthetic-training/connectivity_metrics.py` — label-free connectivity stats
  (continuity, tiny_blob_frac, max_comp_len, ...) with a self-test.
- `RealLSMProxyCallback` now logs `real_continuity_median` / `real_tiny_blob_frac`
  / `real_max_comp_len_median` each val epoch — the real-data QUALITY signal that
  (unlike `pred_pos_frac`) a blob-predictor cannot game.
- `BCEClDiceLoss` (`--loss bce_cldice`) — DiceBCE + clDice topology term; wired
  into `unet3d.py` and set in `train_cached.sh`. Targets the blob failure directly.

Next experiment (no external dependency): run cached training with
`--loss bce_cldice`; watch `real_continuity_median` rise from its ~0.09 baseline
toward the synthetic ~0.5 and `real_tiny_blob_frac` fall from ~0.6. The remaining
root-cause lever is low-contrast synthesis so faint real fibers stay connected.
