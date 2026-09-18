#!/bin/bash
#SBATCH --job-name=syntract_sanity
#SBATCH --partition=gpu            # edit for your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH -o sanity_%j.txt
#SBATCH -e sanity_err_%j.txt

# Tier-1 discriminating test: does the GOOD bf16 model reproduce a KNOWN
# synthetic mask through the inference forward path?
#   dice > 0.5  -> inference path OK; real-data failure is a domain gap.
#   dice ~ 0    -> inference path is broken (normalize/axis/scale), NOT domain.
# Runs a full 128^3 forward pass on a GPU (seconds). Do NOT run on a laptop CPU.

set -euo pipefail

echo "Running on node: $(hostname)"
module purge
module load deprecated-modules
module load cuda/12.4.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source venv/bin/activate
export PYTHONUNBUFFERED=1

# Use the bf16 run that actually learned (val_loss ~0.049), NOT the diverged
# fp16 / pos_weight=0.3 checkpoints (val_loss ~1.0+).
CKPT="${CKPT:-training/checkpoints_cached_bf16/best_3d-epoch=129-val_loss=0.0491.ckpt}"
VOLUME="${VOLUME:-Model_prediction/patch_0001_3d.nii.gz}"
MASK="${MASK:-Model_prediction/patch_0001_3d_mask.nii.gz}"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "Available bf16 checkpoints:" >&2
  ls -lat training/checkpoints_cached_bf16/*.ckpt 2>/dev/null >&2 \
    || echo "  (none — did the bf16 run finish?)" >&2
  exit 1
fi
for f in "$VOLUME" "$MASK"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: not found: $f" >&2
    echo "Pick a synthetic patch + mask the bf16 run trained on, e.g. from" >&2
    echo "  training/precomputed_patches/aligned_wavy/*_3d.nii.gz" >&2
    exit 1
  fi
done

echo "CKPT=$CKPT"
echo "VOLUME=$VOLUME"
echo "MASK=$MASK"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

srun python3 training/sanity_check_synthetic.py \
  --checkpoint "$CKPT" \
  --volume "$VOLUME" \
  --mask "$MASK"
