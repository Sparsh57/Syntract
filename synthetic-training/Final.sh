#!/bin/bash
#SBATCH --job-name=syntract
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sparshmakharia@gmail.com

set -euo pipefail

# Load CUDA
module --ignore_cache load cuda/12.4

# Safer CUDA path setup
if [ -n "${CUDA_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# Activate venv
source ../venv/bin/activate

# Multi-GPU defaults (NUM_GPUS should not exceed allocated GPUs in --gres)
NUM_GPUS=${NUM_GPUS:-4}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-25}
EPOCHS=${EPOCHS:-150}

# W&B
export WANDB_MODE=online
export WANDB_PROJECT=syntract3d
export WANDB_RUN_NAME="test_128_patch_fixed_metrics_$(date +%s)"
export WANDB_NOTES="Testing with 128^3 patch size, fixed metrics, increased mask threshold"

# Temp directory (robust fallback if /scratch is unavailable)
TMP_BASE="${TMPDIR:-/tmp}"
export TMPDIR="${TMP_BASE%/}/syntract_${USER}_${SLURM_JOB_ID:-manual}"
if ! mkdir -p "$TMPDIR"; then
  export TMPDIR="/tmp/syntract_${USER}_${SLURM_JOB_ID:-manual}"
  mkdir -p "$TMPDIR"
fi

# Debug info
echo "Running Syntract on H200 (multi-GPU)"
echo "W&B Project: ${WANDB_PROJECT}"
echo "W&B Run: ${WANDB_RUN_NAME}"
echo "TMPDIR: ${TMPDIR}"
echo "BATCHES_PER_EPOCH: ${BATCHES_PER_EPOCH}"

python3 -c "import importlib.util; exit(0) if importlib.util.find_spec('cupy') else exit(1)" \
  || pip install cupy-cuda12x

nvidia-smi

CUDA_RUNTIME_GPUS=$(python3 - <<'PY'
import torch

try:
    print(int(torch._C._cuda_getDeviceCount()))
except Exception as exc:
    print(f"ERROR: {exc}")
PY
)
if [[ "${CUDA_RUNTIME_GPUS}" == ERROR:* ]] || ! [[ "${CUDA_RUNTIME_GPUS}" =~ ^[0-9]+$ ]]; then
  echo "Unable to query CUDA runtime GPU count: ${CUDA_RUNTIME_GPUS}" >&2
  exit 1
fi
if [ "${CUDA_RUNTIME_GPUS}" -lt 1 ]; then
  echo "CUDA runtime reports no usable GPUs." >&2
  exit 1
fi
if [ "${NUM_GPUS}" -gt "${CUDA_RUNTIME_GPUS}" ]; then
  echo "Requested NUM_GPUS=${NUM_GPUS}, but CUDA runtime exposes ${CUDA_RUNTIME_GPUS}. Using ${CUDA_RUNTIME_GPUS}."
  NUM_GPUS="${CUDA_RUNTIME_GPUS}"
fi

limit_cuda_visible_devices() {
  local keep="$1"
  local current="${CUDA_VISIBLE_DEVICES:-}"
  local new_visible=""

  if [ "${keep}" -lt 1 ]; then
    return
  fi

  if [ -n "${current}" ] && [ "${current}" != "NoDevFiles" ]; then
    IFS=',' read -r -a visible_parts <<< "${current}"
    if [ "${#visible_parts[@]}" -gt "${keep}" ]; then
      new_visible=$(IFS=,; echo "${visible_parts[*]:0:${keep}}")
      export CUDA_VISIBLE_DEVICES="${new_visible}"
      echo "Trimmed CUDA_VISIBLE_DEVICES from '${current}' to '${CUDA_VISIBLE_DEVICES}'."
    else
      echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    fi
  else
    local ids=()
    local i
    for ((i = 0; i < keep; i++)); do
      ids+=("${i}")
    done
    new_visible=$(IFS=,; echo "${ids[*]}")
    export CUDA_VISIBLE_DEVICES="${new_visible}"
    echo "Set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
  fi
}

limit_cuda_visible_devices "${NUM_GPUS}"

if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "${NUM_GPUS}" -gt 0 ]; then
  export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NUM_GPUS))
  if [ "${OMP_NUM_THREADS}" -lt 1 ]; then
    export OMP_NUM_THREADS=1
  fi
fi
echo "NUM_GPUS: ${NUM_GPUS}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-1}"

python3 - <<'PY'
import os
import torch

if hasattr(torch.cuda.device_count, "cache_clear"):
    torch.cuda.device_count.cache_clear()

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime device count:", int(torch._C._cuda_getDeviceCount()))
print("PyTorch CUDA device count:", torch.cuda.device_count())
for idx in range(int(torch._C._cuda_getDeviceCount())):
    print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
PY
python3 - <<'PY'
from numba import cuda
print("Numba CUDA available:", cuda.is_available())
PY

srun --ntasks=1 --nodes=1 torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" train_on_synthetic_data_3d.py \
    --on_the_fly \
    --trk_dir ../registered_trk \
    --input_nifti ../sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz \
    --checkpoint_dir checkpoints/ \
    --epochs "${EPOCHS}" \
    --wandb_online \
    --batches_per_epoch "${BATCHES_PER_EPOCH}" \
    --val_batches 4 \
    --batch_size 4 \
    --check_val_every_n_epoch 5 \
    --accumulate_grad_batches 1 \
    --batch_group_factor 10 \
    --num_workers 0 \
    --devices "${NUM_GPUS}" \
    --strategy ddp_find_unused_parameters_false \
    --patch_size 128 128 128 \
    --voxel_size 0.05 \
    --min_streamlines_per_patch 5 \
    --num_stages 5 \
    --min_features 32 \
    --max_features 320 \
    --lr 1e-4 \
    --warmup_epochs 10
