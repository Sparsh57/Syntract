#!/bin/bash
set -euo pipefail

#SBATCH --job-name=syntract
#SBATCH --partition=mit_preemptable,ou_bcs_low,ou_bcs_normal,ou_bcs_high,mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sparshmakharia@gmail.com

# Load CUDA and fix environment
module --ignore_cache load "cuda/12.4"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate venv
source ../venv/bin/activate

# Robust scratch/tmp handling (avoid permission errors on /scratch)
TMP_BASE="${TMPDIR:-/tmp}"
export TMPDIR="${TMP_BASE%/}/syntract_${USER}_${SLURM_JOB_ID:-manual}"
if ! mkdir -p "$TMPDIR"; then
  export TMPDIR="/tmp/syntract_${USER}_${SLURM_JOB_ID:-manual}"
  mkdir -p "$TMPDIR"
fi
echo "TMPDIR: $TMPDIR"

# Print debug info
echo "Running Syntract 3D on H100"
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
PY
python3 - <<'PY'
from numba import cuda
print(cuda.is_available())
PY
python3 -c "import importlib.util; exit(0) if importlib.util.find_spec('cupy') else exit(1)" \
  || pip install cupy-cuda12x
nvidia-smi

python train_on_synthetic_data_3d.py \
    --on_the_fly \
    --trk_dir ../registered_trk \
    --input_nifti ../sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz \
    --checkpoint_dir checkpoints/ \
    --no_wandb \
    --epochs 150 \
    --batches_per_epoch 50 \
    --batch_size 2 \
    --accumulate_grad_batches 4 \
    --num_workers 0 \
    --batch_group_factor 4 \
    --patch_size 128 128 128 \
    --voxel_size 0.05 \
    --min_streamlines_per_patch 5 \
    --num_stages 5 \
    --min_features 32 \
    --max_features 320 \
    --lr 1e-4 \
    --warmup_epochs 10 \
    --check_val_every_n_epoch 5
