#!/bin/bash
#SBATCH --job-name=syntract_precompute
#SBATCH --partition=gpu            # edit for your cluster
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH -o precompute_%j.txt
#SBATCH -e precompute_err_%j.txt

# Pre-generate 3D patches ONCE to disk, so training never blocks the GPU on
# synthesis. Render knobs mirror train_multigpu.sh exactly so cached patches match the
# on-the-fly config. Run this first; then submit train_cached.sh.

set -euo pipefail

module purge
module load cuda/12.9.1
if [ -n "${CUDA_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
source ../venv/bin/activate

# Stream stdout live to the SLURM log instead of block-buffering it.
export PYTHONUNBUFFERED=1

# Where the cached patches land. Point train_cached.sh --patch_dir here.
OUTPUT_DIR=${OUTPUT_DIR:-./precomputed_patches_fat_tracer}
# How many patches per .trk file. Total = patches_per_trk * (#trk files).
# thick6 dir holds ONE dense TRK (1332 siblings across the volume), so this is
# the total patch count. Aim ~1800 so each epoch sees variety from a fixed pool.
PATCHES_PER_TRK=${PATCHES_PER_TRK:-1800}

TMP_BASE="${TMPDIR:-/tmp}"
export TMPDIR="${TMP_BASE%/}/syntract_${USER}_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" || { export TMPDIR="/tmp/syntract_${USER}_${SLURM_JOB_ID:-manual}"; mkdir -p "$TMPDIR"; }
echo "TMPDIR: ${TMPDIR} | OUTPUT_DIR: ${OUTPUT_DIR} | PATCHES_PER_TRK: ${PATCHES_PER_TRK}"

python3 -c "import importlib.util; exit(0) if importlib.util.find_spec('cupy') else exit(1)" \
  || pip install cupy-cuda12x
nvidia-smi

# DENSE bundle: pack many parallel siblings into a ~25um radius so each rendered
# tracer is FAT and ELONGATED (like real LSM tracers), not a thin line. Smoothing
# a thin line into thickness just makes round blobs; density keeps the fiber shape.
# 50 copies x 222 = 11100 streamlines in a ~25um radius -> tracers ~8-30 vox wide.
THICK_DIR=../registered_trk/dense
THICK_TRK="${THICK_DIR}/aligned_wavy_dense.trk"
if [ ! -f "${THICK_TRK}" ]; then
    echo "Generating dense bundle: ${THICK_TRK}"
    mkdir -p "${THICK_DIR}"
    python -u ../thicken_trk.py \
        --input ../registered_trk/aligned_wavy.trk \
        --output "${THICK_TRK}" \
        --copies 50 --radius_mm 0.025 --seed 42
fi

python -u precompute_patches_3d.py \
    --trk_dir "${THICK_DIR}" \
    --input_nifti ../sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz \
    --output_dir "${OUTPUT_DIR}" \
    --patches_per_trk "${PATCHES_PER_TRK}" \
    --patch_size 128 128 128 \
    --voxel_size 0.001 \
    --min_streamlines_per_patch 2 \
    --min_streamlines_rendered 0 \
    --use_cornucopia_3d \
    --cornucopia_presets ultra_heavy_speckle extreme_noise granular_realistic \
    --tissue_threshold 0.0 \
    --enable_cell_blobs --cell_blob_count 200 --cell_blob_intensity 0.5 \
    --cell_blob_radius_min 3.0 --cell_blob_radius_max 8.0 \
    --fiber_intensity_min 6 --fiber_intensity_max 9 --fiber_max_boost -1 \
    --fiber_smoothing_sigma 0.8 --fiber_smoothing_sigma_range 0.3 1.5 \
    --fiber_brightness_variation 0.40 \
    --fiber_segment_brightness_variation 0.20 --fiber_render_mode additive \
    --fiber_density_gamma 3.0 --fiber_target_intensity 25 \
    --mask_smoothing_sigma 1.0 --mask_smoothing_sigma_range 0.6 3.5 \
    --mask_binary_threshold 0.1 \
    --disable_tissue_artifacts \
    --enable_poisson_noise --poisson_gain ${POISSON_GAIN:-80} \
    --enable_granular_noise --granular_noise_strength 1.5 \
    --enable_speckle_noise --speckle_noise_strength 1.5 --speckle_noise_density 0.04 \
    --disable_dash_noise \
    --enable_horizontal_banding --banding_strength 0.35 --banding_axis 1

echo "=== Precompute complete. Cached patches in ${OUTPUT_DIR} ==="
echo "Sanity check — number of rendered volume/mask pairs:"
find "${OUTPUT_DIR}" -name '*_3d.nii.gz' | wc -l
echo "If this is ~0 or far below expected, the TRKs are streamline-starved at "
echo "voxel_size=0.001 (0.128mm FOV) — fix TRK density before training."
