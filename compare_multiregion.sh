#!/bin/bash
#SBATCH --job-name=syntract_multiregion
#SBATCH --partition=gpu            # edit for your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -o multiregion_%j.txt
#SBATCH -e multiregion_err_%j.txt

# Is the synthetic->real gap UNIVERSAL or just the one spot we sampled?
# The original test sampled 10 near-identical patches around ONE center
# (jitter ±20 voxels ~ 20um). This sweeps several DISTANT regions across the
# slice so we can tell "model fails everywhere on real LSM" from "model fails
# at this particular location."
#
# Level-0 is 1um. Y/X extent is read from the zarr at runtime, then centers are
# placed across the slice (no hardcoded coords that might be out of bounds).

set -euo pipefail
echo "Running on node: $(hostname)"
module purge
module load deprecated-modules
module load cuda/12.4.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source venv/bin/activate
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline   # sweep many regions; skip per-run online logging noise

ZARR="${ZARR:?Set ZARR to the OME-Zarr volume path}"
CKPT="${CKPT:-training/checkpoints_cached_bf16/best_3d-epoch=129-val_loss=0.0491.ckpt}"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; ls -lat training/checkpoints_cached_bf16/*.ckpt >&2 || true; exit 1; }

# Probe level-0 shape (z y x) so centers stay in-bounds.
read -r SZ SY SX < <(python3 - "$ZARR" <<'PY'
import sys, zarr, numpy as np
g = zarr.open(sys.argv[1], mode="r")
# level 0 is the first multiscale array; find largest 3D-ish array
def shape_of(a):
    s=[d for d in a.shape if d>1]
    return s[-3:] if len(s)>=3 else a.shape
best=None
for k in g.array_keys() if hasattr(g,'array_keys') else []:
    pass
# OME-Zarr: arrays are typically named "0","1",... ; take "0"
a = g["0"] if "0" in g else g
s = shape_of(a)
print(int(s[-3]), int(s[-2]), int(s[-1]))
PY
)
echo "Level-0 spatial shape (z y x): $SZ $SY $SX"

# Build a 3x3 grid of YX centers at 1/4, 1/2, 3/4 of each axis; Z near the slab.
CZ=$(( SZ / 2 ))
[ "$CZ" -lt 30 ] && CZ=30
COORDS=()
for fy in 4 2 1; do for fx in 4 2 1; do
  CY=$(( SY * 1 / fy )); [ "$fy" -eq 1 ] && CY=$(( SY * 3 / 4 ))
  CX=$(( SX * 1 / fx )); [ "$fx" -eq 1 ] && CX=$(( SX * 3 / 4 ))
  COORDS+=("$CZ $CY $CX")
done; done

nvidia-smi
i=0
for c in "${COORDS[@]}"; do
  i=$((i+1))
  echo "=================================================================="
  echo "REGION $i/${#COORDS[@]}: center_coords $c"
  srun python3 predict_real_region.py \
    --zarr_path "$ZARR" \
    --model_checkpoint "$CKPT" \
    --center_coords $c \
    --patch_size 128 128 128 \
    --target_voxel_size_um 1 1 1 \
    --num_patches 3 \
    --normalize percentile \
    --jitter_radius 40 \
    --save_debug_patches 1 \
    --output_dir "./multiregion/region_${i}" \
    --level_index 0 || echo "REGION $i failed (likely out-of-bounds) — continuing"
done

echo "=================================================================="
echo "SUMMARY: prediction coverage per region (nonzero binary fraction)"
python3 - <<'PY'
import glob, numpy as np, os
for d in sorted(glob.glob("./multiregion/region_*")):
    bins = glob.glob(os.path.join(d, "**", "*_binary.npy"), recursive=True)
    if not bins:
        print(f"{os.path.basename(d)}: no binary output"); continue
    covs=[100*(np.load(b)>0).mean() for b in bins]
    print(f"{os.path.basename(d)}: mean pred_cover={np.mean(covs):.4f}%  (n={len(covs)})")
print("\nIf ALL regions ~0.03% -> universal domain gap. If some regions fire "
      "(>0.3%) -> location-specific; the model DOES work on parts of real data.")
PY
