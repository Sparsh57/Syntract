#!/bin/bash
#SBATCH --job-name=syntract_calib
#SBATCH --partition=gpu            # edit for your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -o calib_%j.txt
#SBATCH -e calib_err_%j.txt

# Calibrate RealLSMProxyCallback against the compare_multiregion.sh baseline.
# Runs the proxy's EXACT load+forward (SpecificRegionDataset + _load_model + the
# same center grid -> identical instrument by construction) on the GOOD bf16
# ckpt at matched 3-patch/40-jitter sampling, and prints a PER-REGION breakdown.
#
# Compare REGIONS 1-3 (top-Y row) against the handoff's ~0.0002 (= 0.02%
# pred_cover) — that figure was quoted for regions 1-3, NOT the 9-center grand
# mean. The grand mean runs higher because lower-Y rows catch more structure;
# that's aggregation, not a wiring bug. The proxy only needs to detect MOVEMENT
# in step (b), which a constant offset can't fake.
#
# GOLD STANDARD if the §5b outputs survive: pool multiregion/region_*/ the same
# way (compare_multiregion.sh prints mean pred_cover% per region) and compare.
#
# GPU only (128^3 forward OOMs a laptop).

set -euo pipefail
echo "Running on node: $(hostname)"
module purge
module load deprecated-modules
module load cuda/12.4.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source venv/bin/activate
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

ZARR="${ZARR:?Set ZARR to the OME-Zarr volume path}"
CKPT="${CKPT:-training/checkpoints_cached_bf16/best_3d-epoch=129-val_loss=0.0491.ckpt}"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; ls -lat training/checkpoints_cached_bf16/*.ckpt >&2 || true; exit 1; }

nvidia-smi

srun python calibrate_real_proxy.py \
    --checkpoint "$CKPT" \
    --zarr "$ZARR" \
    --patch_size 128 128 128 \
    --target_voxel_um 1 1 1 \
    --level_index 0 \
    --num_patches 3 \
    --jitter_radius 40 \
    --threshold 0.5 \
    --expected 0.0002
