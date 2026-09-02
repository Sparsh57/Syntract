#!/bin/bash
#SBATCH --job-name=syntract_sliding_infer
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o sliding_infer_%j.txt
#SBATCH -e sliding_infer_err_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sparshmakharia@gmail.com

# Run sliding-window 3D inference over two selected regions of the real LSM zarr.
# Edit REGION_1_CENTER and REGION_2_CENTER to your chosen (Z Y X) coords.
# Submit from anywhere: sbatch synthetic-training/sliding_window_infer.sh

set -euo pipefail

# Resolve repo root from this script's location — works regardless of where sbatch is run from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

module purge
module load cuda/12.9.1
if [ -n "${CUDA_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
# NOTE: this resolves to the venv ONE LEVEL ABOVE the checkout, unlike the
# root sliding_window_infer.sh which uses ${REPO_ROOT}/venv. Unverified which
# is correct on the cluster — a completed run in sliding_infer_out/ suggests
# this path works there, so don't "fix" it without checking on the cluster.
source "${REPO_ROOT}/../venv/bin/activate"
export PYTHONUNBUFFERED=1

ZARR=/orcd/data/linc/001/lsm_test_data_sparsh/LSM_test_data/2025_09_09_MonkeySlice_561channel_561laser_Stitched.ome.zarr
CKPT="${REPO_ROOT}/synthetic-training/checkpoints_cached_bf16/best_3d-epoch=129-val_loss=0.0491.ckpt"
OUT="${REPO_ROOT}/sliding_infer_out"

# ---- Edit these two lines with your chosen region centers (Z Y X) ----
REGION_1_CENTER="19 12000 20000"
REGION_2_CENTER="19 14478 18000"
# ----------------------------------------------------------------------

REGION_SIZE="256 512 512"   # sub-volume to extract per region (Z Y X voxels)
STRIDE=64                   # 50% overlap; lower = smoother but slower

echo "REPO_ROOT:  ${REPO_ROOT}"
echo "CKPT:       ${CKPT}"
echo "OUT:        ${OUT}"
nvidia-smi

echo "=== Region 1 ==="
python "${REPO_ROOT}/sliding_window_inference.py" \
    --zarr "${ZARR}" \
    --region_center_zyx ${REGION_1_CENTER} \
    --region_size_zyx ${REGION_SIZE} \
    --checkpoint "${CKPT}" \
    --output_prefix "${OUT}/region1" \
    --stride "${STRIDE}" \
    --threshold 0.5 \
    --save_nifti

echo "=== Region 2 ==="
python "${REPO_ROOT}/sliding_window_inference.py" \
    --zarr "${ZARR}" \
    --region_center_zyx ${REGION_2_CENTER} \
    --region_size_zyx ${REGION_SIZE} \
    --checkpoint "${CKPT}" \
    --output_prefix "${OUT}/region2" \
    --stride "${STRIDE}" \
    --threshold 0.5 \
    --save_nifti

echo "=== Done. Results in ${OUT}/ ==="
ls -lh "${OUT}/"
