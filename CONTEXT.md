# SynTract — Context Glossary

Canonical domain language for this project. This file is a glossary only — no
implementation details, no decisions, no roadmap. When code or conversation
uses a term differently from a definition below, that conflict must be resolved.

Status: drafted 2026-06-11 during a goal-clarification (grill-with-docs) session.
Terms marked **(open)** are not yet fully resolved.

---

## Mission

The deliverable is a **Fiber Segmenter**: a model that takes real **LSM data**
and produces a **Fiber Segmentation** of it. Everything else in the repository
(synthesis, viewer, training) exists to serve that deliverable.

## Terms

### Fiber
The structure being segmented: a per-voxel label indicating "this voxel belongs
to a fiber." A Fiber is the segmentation target, not a yes/no patch property.
**Real referent (observed 2026-06-11):** in real LSM, a Fiber appears as a
**thin, low-contrast, locally-parallel elongated structure running through
depth**, surrounded by bright **cell-body puncta** (distractors). It is detected
by a tubularity filter, not by brightness. **Open:** confirming fiber vs. vessel
vs. light-sheet stripe artifact, and whether the target is individual fibers or
bundles, still needs a domain expert. The **synthetic Fiber** (a bright,
high-contrast rendered tube) does NOT match this referent — the trained model
fires on the brightest region and misses the actual low-contrast tubes. Closing
this synthesis↔reality mismatch is the core modeling problem.

### Fiber Segmentation
The product. A per-voxel mask over a volume (`D×H×W`), binary or soft
(partial-volume), marking Fiber voxels. This — not region-level detection — is
the primary output. Success is measured by correctness of *which voxels*, not
merely *whether* fiber is present.

### Detection
A *derived*, secondary view: "does this region contain Fiber (yes/no)," obtained
by thresholding the positive fraction of a Fiber Segmentation. Detection is a
consequence of the product, never the product itself.

### LSM data
The real target domain: light-sheet microscopy volumes (SPIM), stored as
OME-Zarr, at roughly micron-scale voxels. The model must work on *this*, not on
the synthetic training data. Synthetic data is a means; LSM data is the end.

### Synthetic data
Generated training patches (MRI/tractography → rendered fiber-on-tissue
volumes). A *proxy* for LSM data, never the evaluation target. Performance on
synthetic data only guards against *regressing* segmentation skill; it does not
establish real-data correctness.

### Real-data correctness  **(approach found 2026-06-11: Continuity)**
Whether a Fiber Segmentation is right *on LSM data*. Dense labels are
unreachable AND the team cannot recognize a Fiber by eye — so correctness is
judged by a STRUCTURAL prior instead (see Continuity). The `pred_pos_frac`
signal is a *sparsity* check only and a blob-predictor games it; the connectivity
metric does not. This is a label-free, OCT-free, provenance-free path to a real
quality signal.

### Continuity (fiber prior)
The defining property of a Fiber: it is a **long, connected, continuous curve —
never an isolated blob.** This is computable with no labels (connected-component
count, largest-component length, % tiny blobs, skeleton length,
length/thickness ratio) and is therefore BOTH the real-data quality metric and a
training signal. Measured 2026-06-11: synthetic targets are connected fibers
(0% tiny blobs, length/thickness ~56), but the model on real data produces
scattered blobs (57–67% of components ≤3 voxels, length/thickness ~5–10) — a
continuity-transfer failure, not a target problem. Levers: clDice topology loss
(`soft_dice_cldice`, already in the codebase), low-contrast synthesis so faint
fibers stay connected, and connectivity post-processing. Caveat: continuity is
necessary not sufficient (a continuously-traced vessel would also score well).

### Legibility (real-fiber)
The property of fiber structure being identifiable in raw LSM. **State as of
2026-06-11:**
- *Settled — not noise:* single 2D slices look like grain + bright cell puncta
  (why the data was perceived as "noise"), but a depth max-projection reveals
  coherent diagonal/parallel elongated structure. Evidence:
  `Model_prediction/patch_0001_model_input.npy` (561nm monkey-slice OME-Zarr).
- *Settled — not an artifact:* a depth-decorrelation test (through-depth
  correlation collapses to ~0 within ~10 slices) plus orthogonal cross-sections
  (no full-span curtains) rule out the light-sheet stripe-artifact hypothesis.
  The structure is real 3D biological content.
- *STILL UNRESOLVED — identity:* whether that real structure is **fibers** vs.
  vasculature vs. cell-dense laminae is unknown. The domain owner cannot tell by
  eye ("unclear to me"). Ruling out the artifact only establishes there IS
  something worth identifying — it does NOT confirm it is a Fiber. This identity
  question is the project blocker; resolving it requires a Fiber reference
  (external).

### Tubularity reference  **(disputed)**
A classical tube-detection filter (Sato/Frangi). Proposed as a label-free handle
on real fibers, but the domain expert reports it does NOT work well on this data
(over-fires / traces the wrong structures). Its viability as a pseudo-label or
validation anchor is therefore unconfirmed. May return as a *teacher* once a
Fiber reference confirms what real fibers look like.

### Fiber reference (external)
The missing keystone: any source of "this is a Fiber, here" that does NOT depend
on the team recognizing fibers by eye. **Best candidate located 2026-06-11:**
DANDI **001412** `derivatives/stitched`, subject **MF283 slice036**, which has
FOUR co-located fluorescence channels (acq-488/561/594/660) plus an **OCT**
label-free structural image. The four channels visibly label different things
(488/594 = elongated streaks; 561/660 = discrete puncta), so identifying the
fiber channel is now concrete — BUT the channel→label key (which wavelength
stains what) is **not in any reachable metadata** (DANDI asset meta, 001372
`dataset_description.json`, and the OME-Zarr `omero` attrs were all checked and
lack it). The key is **institutional**: LINC imaging protocol /
gallery.lincbrain.org / the data producer. Obtaining it is **Milestone 0**.
Secondary path: register the **OCT** to a fluorescence channel for a label-free
cross-modal fiber confirmation. See
`docs/adr/0001-fiber-reference-prerequisite.md`.
