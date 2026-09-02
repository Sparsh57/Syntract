#!/bin/bash
#SBATCH --job-name=syntract_sliding_infer
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH -o sliding_infer_%j.txt
#SBATCH -e sliding_infer_err_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sparshmakharia@gmail.com

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"

module purge
module load cuda/12.9.1
if [ -n "${CUDA_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
source "${REPO_ROOT}/venv/bin/activate"
export PYTHONUNBUFFERED=1

ZARR=/orcd/data/linc/001/lsm_test_data_sparsh/LSM_test_data/sub-MF283-sample-slice037_acq-594.ome.zarr
# Use the newest PERIODIC snapshot (epoch-NNN.ckpt) = the fully-trained model.
# (best_3d/last freeze at epoch 14 when val_loss goes non-finite; the periodic
# checkpoints save at train-epoch end regardless, so epoch-150 is the latest.)
CKPT_DIR="/orcd/data/linc/001/lsm_test_data_sparsh/syntract_cache/checkpoints_fat_tracer"
CKPT=$(ls -t "${CKPT_DIR}"/epoch-*.ckpt 2>/dev/null | head -1)
CKPT="${CKPT:-${CKPT_DIR}/last.ckpt}"
# Outputs are ~120 GB at full depth — MUST live on the data filesystem, not home
# (home is quota-capped at 200 GB).
OUT="/orcd/data/linc/001/lsm_test_data_sparsh/syntract_cache/sliding_infer_out_slice037"

# ---- Region center (Z Y X), level-0 voxels. Big fibrous-core box on slice037. ----
# Z range: 0-402, Y range: 0-28988, X range: 0-45339
REGION_1_CENTER="200 13000 16400"
# ----------------------------------------------------------------------------------

# FULL depth (402) x ~7.1 x 6.1 mm. Peaks ~311 GB RAM (reflect-pad + 2 float32
# accumulators), so needs --mem=400G. mit_preemptable has 2 TB nodes, so it fits.
REGION_SIZE="402 6144 6144"
STRIDE=64

echo "REPO_ROOT: ${REPO_ROOT}"
echo "CKPT:      ${CKPT}"
echo "OUT:       ${OUT}"
nvidia-smi

echo "=== Region 1 (slice037 fibrous core, ~7x6mm) ==="
python "${REPO_ROOT}/sliding_window_inference.py" \
    --zarr "${ZARR}" \
    --region_center_zyx ${REGION_1_CENTER} \
    --region_size_zyx ${REGION_SIZE} \
    --checkpoint "${CKPT}" \
    --output_prefix "${OUT}/region1" \
    --stride "${STRIDE}" \
    --threshold 0.5 \
    --save_nifti

echo "=== Done. Results in ${OUT}/ ==="
ls -lh "${OUT}/"
