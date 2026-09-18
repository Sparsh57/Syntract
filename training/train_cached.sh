#!/bin/bash
#SBATCH --job-name=syntract_cached
#SBATCH --partition=gpu            # edit for your cluster
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt

# Train on PRE-COMPUTED patches (run precompute_patches.sh first).
# The GPU no longer blocks on synthesis: data loading is plain disk I/O across
# --num_workers processes, so all 4 H200s stay busy.
#
# IMPORTANT — no double augmentation: granular/speckle/dash/banding/cell-blobs
# are already BAKED into the cached patches. The cached dataset's load-time
# image augs are therefore DISABLED here (they default off; we do not pass the
# --enable_* flags). Thin-slab / empty-patch shape augs are applied at load time
# by SyntheticDataset3D (defaults thinslab_prob=0.3, empty_patch_prob=0.05),
# matching the on-the-fly path.

set -euo pipefail

module purge
module load cuda/12.9.1
if [ -n "${CUDA_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
source ../venv/bin/activate

NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-150}
PATCH_DIR=${PATCH_DIR:-./precomputed_patches_fat_tracer}

export WANDB_MODE=online
export WANDB_PROJECT=syntract3d
export WANDB_RUN_NAME="cached_128_$(date +%s)"

TMP_BASE="${TMPDIR:-/tmp}"
export TMPDIR="${TMP_BASE%/}/syntract_${USER}_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" || { export TMPDIR="/tmp/syntract_${USER}_${SLURM_JOB_ID:-manual}"; mkdir -p "$TMPDIR"; }
echo "PATCH_DIR: ${PATCH_DIR} | NUM_GPUS: ${NUM_GPUS}"

# Guard: refuse to burn 4xH200 if the patch cache is empty/missing.
N_PATCHES=$(find "${PATCH_DIR}" -name '*_3d.nii.gz' 2>/dev/null | wc -l | tr -d ' ')
echo "Cached *_3d.nii.gz patches found: ${N_PATCHES}"
if [ "${N_PATCHES}" -lt 100 ]; then
  echo "ERROR: only ${N_PATCHES} cached patches in ${PATCH_DIR}. Run precompute_patches.sh"
  echo "and confirm a healthy count before training (starved TRKs => few patches)." >&2
  exit 1
fi

nvidia-smi

srun --ntasks=1 --nodes=1 torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" train_3d.py \
    --cached_patches \
    --patch_dir "${PATCH_DIR}" \
    --trk_dir ../registered_trk/fine_res \
    --input_nifti ../sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz \
    --checkpoint_dir "${CKPT_DIR:-checkpoints/}" \
    --no_resume \
    --epochs "${EPOCHS}" --wandb_online --wandb_name "${WANDB_RUN_NAME}" \
    --val_batches 4 --batch_size 4 \
    --check_val_every_n_epoch 5 --accumulate_grad_batches 1 \
    --num_workers 12 --devices "${NUM_GPUS}" \
    --strategy ddp_find_unused_parameters_false \
    --patch_size 128 128 128 --voxel_size 0.001 \
    --num_stages 5 --min_features 32 --max_features 320 \
    --lr 1e-4 --warmup_epochs 3 \
    --real_proxy_zarr None \
    --loss BCE --pos_weight 5.0

# Notes:
#  --num_workers 12: cached loading is disk I/O; raise toward 24 if CPU-bound.
#  THICK-BINARY config (2026-06-26): the first poisson_soft run COLLAPSED — best
#    ckpt epoch=64 val_loss=1.0032, real pred_pos_frac ~1e-5, predictions single
#    voxels. Root cause found in the masks: SOFT masks had mean positive target
#    0.026 (93% of positive voxels < 0.1) -> BCE correctly learned ~0. Fixed by
#    re-precomputing with BINARY masks (solid 1.0 targets), thicker fibers, and
#    per-patch THICKNESS DOMAIN-RANDOMIZATION (fiber/mask sigma vary per patch:
#    width ~3.5-8 vox) so the model learns a continuous tubular STRUCTURE, not a
#    fixed width. --loss BCE + --pos_weight 5 match the known-good baseline
#    (val_loss 0.049). Watch: train_pred_pos_frac climbs off zero toward ~0.006,
#    val_loss drops well below 1.0. Re-add clDice only AFTER it predicts something.
#  --warmup_epochs 3: 10 wasted ~250 steps before full LR.
#  Render knobs are NOT passed here — they were baked at precompute time.
