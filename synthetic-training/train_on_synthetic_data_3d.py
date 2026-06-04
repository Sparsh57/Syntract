"""
3D volumetric training script for synthetic fiber segmentation.

Usage (on-the-fly, offline wandb):
    python train_on_synthetic_data_3d.py \
        --on_the_fly \
        --trk_dir /path/to/trk_files/ \
        --input_nifti /path/to/brain.nii.gz \
        --wandb_name "3d_pretraining" \
        --checkpoint_dir "/path/to/checkpoints/"

Usage (no wandb):
    python train_on_synthetic_data_3d.py \
        --on_the_fly \
        --trk_dir /path/to/trk_files/ \
        --input_nifti /path/to/brain.nii.gz \
        --no_wandb \
        --checkpoint_dir "/path/to/checkpoints/"
"""

import argparse
import os
import sys
import time
from collections import deque

# MPS fallback for ops not yet implemented on Apple Silicon (e.g. max_pool3d)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger, CSVLogger

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch.set_float32_matmul_precision('high')

from datamodules.dataloaders import OnTheFlyDataModule3D
from unet3d import FlexibleUNet3D


def _path_or_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return os.path.abspath(os.path.expanduser(value))


def _require_existing_file(path, label, original=None, hint=None):
    if path is not None and os.path.isfile(path):
        return
    message = f"{label} does not exist: {path}"
    if original is not None and original != path:
        message += f" (from {original})"
    if hint:
        message += f". {hint}"
    raise FileNotFoundError(message)


def _require_existing_dir(path, label, original=None):
    if path is not None and os.path.isdir(path):
        return
    message = f"{label} does not exist or is not a directory: {path}"
    if original is not None and original != path:
        message += f" (from {original})"
    raise FileNotFoundError(message)


def _cuda_runtime_device_count():
    if not hasattr(torch._C, "_cuda_getDeviceCount"):
        return 0
    try:
        return int(torch._C._cuda_getDeviceCount())
    except Exception as exc:
        print(f"Warning: Unable to query CUDA runtime device count: {exc}")
        return 0


def _trim_cuda_visible_devices(max_devices):
    if max_devices < 1:
        return

    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current and current != "NoDevFiles":
        parts = [part.strip() for part in current.split(",") if part.strip()]
        if len(parts) > max_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(parts[:max_devices])
            print(
                "Trimmed CUDA_VISIBLE_DEVICES "
                f"from '{current}' to '{os.environ['CUDA_VISIBLE_DEVICES']}'."
            )
    elif current != "NoDevFiles":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(max_devices))
        print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}.")

    if hasattr(torch.cuda.device_count, "cache_clear"):
        torch.cuda.device_count.cache_clear()


class ThroughputProfilerCallback(Callback):
    """Lightweight profiler to expose where training time is spent."""

    def __init__(self, print_every_batches: int = 20, window: int = 50):
        super().__init__()
        self.print_every_batches = max(1, int(print_every_batches))
        self.window = max(10, int(window))
        self._last_batch_end_t = None
        self._batch_start_t = None
        self._seen = 0
        self.wait_s = deque(maxlen=self.window)
        self.step_total_s = deque(maxlen=self.window)
        self.step_compute_s = deque(maxlen=self.window)
        self.extract_s = deque(maxlen=self.window)
        self.render_s = deque(maxlen=self.window)

    @staticmethod
    def _mean(values):
        return (sum(values) / len(values)) if values else 0.0

    def on_train_start(self, trainer, pl_module):
        self._last_batch_end_t = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._last_batch_end_t is not None:
            self.wait_s.append(now - self._last_batch_end_t)
        self._batch_start_t = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        end_t = time.perf_counter()
        if self._batch_start_t is not None:
            self.step_total_s.append(end_t - self._batch_start_t)
        self.step_compute_s.append(float(getattr(pl_module, "_last_train_step_compute_s", 0.0)))

        dataset = getattr(getattr(trainer, "datamodule", None), "train_dataset", None)
        generated = max(1, int(getattr(dataset, "last_generated_batches", 1)))
        self.extract_s.append(float(getattr(dataset, "last_extract_time_s", 0.0)) / generated)
        self.render_s.append(float(getattr(dataset, "last_render_time_s", 0.0)) / generated)

        self._last_batch_end_t = end_t
        self._seen += 1
        if trainer.is_global_zero and self._seen % self.print_every_batches == 0:
            total = self._mean(self.step_total_s)
            compute = self._mean(self.step_compute_s)
            wait = self._mean(self.wait_s)
            extract = self._mean(self.extract_s)
            render = self._mean(self.render_s)
            post = max(0.0, total - compute)
            print(
                f"[timing] batch={self._seen} | wait={wait:.3f}s | total={total:.3f}s | "
                f"compute(step)={compute:.3f}s | backward+opt+log~={post:.3f}s | "
                f"extract/batch~={extract:.3f}s | render/batch~={render:.3f}s"
            )


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _multiregion_centers(level_shape_zyx):
    """Reproduce the 3x3 YX center grid that compare_multiregion.sh sweeps.

    Z near the slab centre (>=30); Y and X at {1/4, 1/2, 3/4} of each axis.
    Keeping this identical means the proxy tracks predictions at the SAME
    locations used to establish the universal-X/Y domain gap, so a change in
    the proxy is directly comparable to that elimination experiment.
    """
    sz, sy, sx = (int(v) for v in level_shape_zyx)
    cz = max(sz // 2, 30)
    centers = []
    for fy in (4, 2, 1):
        for fx in (4, 2, 1):
            cy = (sy * 3 // 4) if fy == 1 else (sy // fy)
            cx = (sx * 3 // 4) if fx == 1 else (sx // fx)
            centers.append((cz, cy, cx))
    return centers


class RealLSMProxyCallback(Callback):
    """Unlabeled real-data transfer proxy.

    Runs the model on a FIXED set of real OME-Zarr LSM patches each validation
    epoch and logs the predicted-positive fraction + mean probability. There
    are no real labels, so this is NOT a quality metric — it is a *transfer*
    signal: a held-out synthetic val split sits ~0.87 dice regardless of
    augmentation changes (synthetic generalization is not what is broken), so
    it cannot show whether a domain-gap fix moves real-data behaviour. This
    proxy can: if augmentation changes start closing the gap, real_pred_pos_frac
    moves off its near-zero floor.

    Patches are loaded through the SAME path as test_specific_region.py /
    compare_multiregion.sh (SpecificRegionDataset, percentile normalize, the 3x3
    multiregion center grid), so the numbers are comparable to that experiment.

    Disabled (no-op) when no zarr path is provided — which is also why this is
    safe locally: without a zarr it never runs a 128^3 forward pass.
    """

    def __init__(
        self,
        zarr_path,
        patch_size=(128, 128, 128),
        target_voxel_size_um=(1.0, 1.0, 1.0),
        level_index=0,
        channel_index=0,
        normalize_percentiles=(1.0, 99.0),
        threshold=0.5,
        num_patches=1,
        jitter_radius=0,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.patch_size = tuple(int(p) for p in patch_size)
        self.target_voxel_size_um = tuple(float(v) for v in target_voxel_size_um)
        self.level_index = int(level_index)
        self.channel_index = int(channel_index)
        self.normalize_percentiles = (float(normalize_percentiles[0]), float(normalize_percentiles[1]))
        self.threshold = float(threshold)
        # Per-center sampling. Training default is 1 deterministic patch (no
        # jitter) so the metric is reproducible epoch-to-epoch. The standalone
        # calibration script (calibrate_real_proxy.py) sets these to 3/40 to
        # match compare_multiregion.sh's baseline EXACTLY for a like-for-like
        # comparison against its ~0.0002 number.
        self.num_patches = max(1, int(num_patches))
        self.jitter_radius = max(0, int(jitter_radius))
        self._patches = None  # lazily built list of (1,1,D,H,W) tensors
        self._build_failed = False

    def _make_region_dataset(self, center):
        from test_specific_region import SpecificRegionDataset
        return SpecificRegionDataset(
            zarr_path=self.zarr_path,
            center_coords_zyx=center,
            patch_size_zyx=self.patch_size,
            num_patches=self.num_patches,
            level_index=self.level_index,
            channel_index=self.channel_index,
            normalize=True,
            normalize_mode="percentile",
            normalize_percentiles=self.normalize_percentiles,
            target_voxel_size_um=self.target_voxel_size_um,
            jitter_radius=self.jitter_radius,
            seed=42,
        )

    def _build_patches(self, pl_module):
        # Probe the level shape from a throwaway dataset at a safe centre, then
        # build the multiregion 3x3 grid. SpecificRegionDataset exposes .levels, so
        # we reuse its own level-info extraction rather than re-probing zarr.
        probe = self._make_region_dataset((0, 0, 0))
        level_shape = probe.levels[self.level_index].shape_zyx
        centers = _multiregion_centers(level_shape)
        print(
            f"[real-proxy] level={self.level_index} shape_zyx={tuple(int(s) for s in level_shape)} "
            f"-> {len(centers)} fixed multiregion centers"
        )

        # Group patches BY region (center) so per-region metrics are available.
        # The grand mean alone is outlier-dominated — at the 1-patch default one
        # hot region (calibration showed region 9 fires ~23x the floor) can swing
        # the mean more than a broad change across the other eight, making the
        # scalar a weak movement detector for step (b). Per-region + median fix it.
        regions = []  # list of (center, [patch tensors])
        for c in centers:
            try:
                ds = self._make_region_dataset(c)
                region_patches = [ds[i][0].unsqueeze(0) for i in range(len(ds))]  # (1,1,D,H,W)
                if region_patches:
                    regions.append((tuple(int(v) for v in c), region_patches))
            except Exception as exc:  # out-of-bounds region etc. — skip
                print(f"[real-proxy] center {c} skipped: {exc}")
        return regions

    def evaluate(self, module):
        """Run the fixed real patches through ``module``. Returns a metrics dict:
        mean/median pred_pos_frac + prob_mean over regions, and per-region
        pred_pos_frac. Builds patches lazily. Used by both the val-epoch hook and
        the standalone calibration script (IDENTICAL forward path).
        """
        if self._patches is None:
            self._patches = self._build_patches(module)
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        was_training = module.training
        module.eval()
        region_pos, region_prob, per_region = [], [], {}
        with torch.no_grad():
            for center, region_patches in self._patches:
                pf, pm = [], []
                for patch in region_patches:
                    x = patch.to(device, dtype=dtype)
                    prob = torch.sigmoid(module(x).float())
                    pm.append(float(prob.mean().item()))
                    pf.append(float((prob >= self.threshold).float().mean().item()))
                rp = sum(pf) / len(pf) if pf else 0.0
                region_pos.append(rp)
                region_prob.append(sum(pm) / len(pm) if pm else 0.0)
                per_region[center] = rp
        if was_training:
            module.train()
        if not region_pos:
            return {"pred_pos_frac_mean": 0.0, "pred_pos_frac_median": 0.0,
                    "prob_mean": 0.0, "per_region": {}}
        return {
            "pred_pos_frac_mean": float(sum(region_pos) / len(region_pos)),
            "pred_pos_frac_median": float(_median(region_pos)),
            "prob_mean": float(sum(region_prob) / len(region_prob)),
            "per_region": per_region,
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        # on_validation_epoch_end already fires only on validation epochs, so
        # the validation cadence (check_val_every_n_epoch) controls how often
        # this runs — no separate modulo gate (which could misalign with it).
        if not trainer.is_global_zero or self.zarr_path is None or self._build_failed:
            return
        if self._patches is None:
            try:
                self._patches = self._build_patches(pl_module)
            except Exception as exc:
                print(f"[real-proxy] disabled (failed to build patches): {exc}")
                self._build_failed = True
                return
        if not self._patches:
            print("[real-proxy] no in-bounds patches; disabling")
            self._build_failed = True
            return

        m = self.evaluate(pl_module)
        # Mean is outlier-dominated (one hot region can swing it); the MEDIAN is
        # the robust movement signal for step (b). Log both, plus per-region.
        pl_module.log("real_pred_pos_frac", m["pred_pos_frac_mean"], prog_bar=True, rank_zero_only=True)
        pl_module.log("real_pred_pos_frac_median", m["pred_pos_frac_median"], prog_bar=True, rank_zero_only=True)
        pl_module.log("real_prob_mean", m["prob_mean"], rank_zero_only=True)
        for ridx, (center, rp) in enumerate(m["per_region"].items(), start=1):
            pl_module.log(f"real_region{ridx}_pos_frac", rp, rank_zero_only=True)
        print(
            f"[real-proxy] epoch={trainer.current_epoch} regions={len(self._patches)} "
            f"mean={m['pred_pos_frac_mean']:.5f} median={m['pred_pos_frac_median']:.5f} "
            f"prob_mean={m['prob_mean']:.5f} per_region="
            + ", ".join(f"{c}:{rp:.5f}" for c, rp in m["per_region"].items())
        )


def get_args_parser():
    parser = argparse.ArgumentParser('3D Synthetic data training', add_help=False)

    # Data
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size (default: 4 for 128^3 synthetic training)')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batches_per_epoch', default=50, type=int)
    parser.add_argument('--val_batches', default=16, type=int,
                        help='Validation batches per validation epoch. Bumped from 4 '
                             '(8 patches) to 16 so the val metric is less noisy.')
    parser.add_argument('--patch_size', nargs=3, type=int, default=[128, 128, 128],
                        help='3D patch size as D H W (default: 128 128 128)')

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not resume from checkpoint_dir/last.ckpt')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Explicit checkpoint path to resume full trainer state from')
    parser.add_argument('--pretraining_checkpoint', type=str, default="None",
                        help="Path to pretrained 3D model checkpoint")
    parser.add_argument('--wandb_name', type=str, default="unet3d_synthetic_pretraining")

    # Data source
    parser.add_argument('--on_the_fly', dest='on_the_fly', action='store_true', default=True,
                        help='Generate 3D patches on-the-fly during training (default: enabled)')
    parser.add_argument('--cached_patches', dest='on_the_fly', action='store_false',
                        help='Use pre-generated patches from --patch_dir instead of on-the-fly generation')
    parser.add_argument('--patch_dir', type=str, default=None,
                        help='Directory with pre-generated *_3d.nii.gz patches (used when --on_the_fly is not set)')
    parser.add_argument('--val_fraction', type=float, default=0.15,
                        help='Cached path only: fraction of patches held out as a disjoint '
                             'val split (deterministic, shuffled by --split_seed). '
                             'Set 0.0 to disable (val == train, legacy behaviour).')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Seed for the deterministic cached train/val split shuffle.')
    parser.add_argument('--real_proxy_zarr', type=str, default="None",
                        help='Optional OME-Zarr path. When set, an unlabeled real-data transfer '
                             'proxy runs each val epoch on the fixed multiregion centers and logs '
                             'real_pred_pos_frac / real_prob_mean. "None" disables (and is required '
                             'locally — it never runs a 128^3 forward pass without a zarr).')
    parser.add_argument('--real_proxy_level', type=int, default=0,
                        help='Pyramid level for the real-data proxy (0 = highest res, 1um).')
    parser.add_argument('--real_proxy_channel', type=int, default=0,
                        help='Channel index for the real-data proxy.')
    parser.add_argument('--real_proxy_target_voxel_um', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                        help='Target voxel size (um, z y x) for the real-data proxy patches.')
    parser.add_argument('--real_proxy_threshold', type=float, default=0.5,
                        help='Probability threshold for real-data proxy binary coverage.')
    parser.add_argument('--trk_dir', type=str, required=True,
                        help='Directory containing .trk files')
    parser.add_argument('--input_nifti', type=str, required=True,
                        help='Input NIfTI volume')
    parser.add_argument('--white_mask', type=str, default="None",
                        help='Path to white matter mask (use "None" to disable)')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='Target voxel size in mm for patch extraction (default: 0.05mm)')
    parser.add_argument('--min_streamlines_per_patch', type=int, default=20,
                        help='Minimum streamlines per extracted patch (default: 20)')
    parser.add_argument('--streamline_margin_fraction', type=float, default=0.10,
                        help='Require sampled patches to contain streamlines away from patch borders (default: 0.10)')
    parser.add_argument('--enable_tissue_artifacts', dest='enable_tissue_artifacts',
                        action='store_true', default=True,
                        help='Enable image-only tissue-like artefacts for training patches (default: enabled)')
    parser.add_argument('--disable_tissue_artifacts', dest='enable_tissue_artifacts',
                        action='store_false',
                        help='Disable tissue-like artefacts')
    parser.add_argument('--enable_granular_noise', dest='enable_granular_noise',
                        action='store_true', default=True,
                        help='Enable image-only fine Cornucopia granular noise for training patches (default: enabled)')
    parser.add_argument('--disable_granular_noise', dest='enable_granular_noise',
                        action='store_false',
                        help='Disable fine granular noise')
    parser.add_argument('--enable_speckle_noise', dest='enable_speckle_noise',
                        action='store_true', default=True,
                        help='Enable bright square speck artefacts (default: enabled)')
    parser.add_argument('--disable_speckle_noise', dest='enable_speckle_noise',
                        action='store_false')
    parser.add_argument('--enable_dash_noise', dest='enable_dash_noise',
                        action='store_true', default=True,
                        help='Enable short diagonal dash artefacts (default: enabled)')
    parser.add_argument('--disable_dash_noise', dest='enable_dash_noise',
                        action='store_false')
    parser.add_argument('--enable_horizontal_banding', dest='enable_horizontal_banding',
                        action='store_true', default=True,
                        help='Enable blockface horizontal slice banding (default: enabled)')
    parser.add_argument('--disable_horizontal_banding', dest='enable_horizontal_banding',
                        action='store_false')
    parser.add_argument('--granular_noise_strength', type=float, default=0.35)
    parser.add_argument('--artifact_strength', type=float, default=0.45)
    parser.add_argument('--speckle_noise_strength', type=float, default=4.0)
    parser.add_argument('--speckle_noise_density', type=float, default=0.0004)
    parser.add_argument('--speckle_noise_sigma', type=float, default=1.5)
    parser.add_argument('--speckle_square_size', type=int, default=1)
    parser.add_argument('--dash_noise_strength', type=float, default=2.0)
    parser.add_argument('--dash_noise_density', type=float, default=0.0003)
    parser.add_argument('--dash_length_sigma', type=float, default=6.0)
    parser.add_argument('--dash_cross_sigma', type=float, default=0.8)
    parser.add_argument('--banding_strength', type=float, default=0.25)
    parser.add_argument('--banding_axis', type=int, default=1)
    parser.add_argument('--fiber_intensity_min', type=float, default=60.0)
    parser.add_argument('--fiber_intensity_max', type=float, default=100.0)
    parser.add_argument('--fiber_max_boost', type=float, default=10.0)
    parser.add_argument('--fiber_opacity', type=float, default=1.0)
    parser.add_argument('--fiber_smoothing_sigma', type=float, default=0.0)
    parser.add_argument('--fiber_antialias', dest='fiber_antialias', action='store_true', default=True)
    parser.add_argument('--no_fiber_antialias', dest='fiber_antialias', action='store_false')
    parser.add_argument('--min_streamlines_rendered', type=int, default=20)
    parser.add_argument('--fiber_brightness_variation', type=float, default=0.60)
    parser.add_argument('--fiber_segment_brightness_variation', type=float, default=0.35)
    parser.add_argument('--fiber_render_mode', default='additive', choices=['additive', 'density', 'embedded'])
    parser.add_argument('--fiber_density_gamma', type=float, default=5.0)
    parser.add_argument('--fiber_min_visibility', type=float, default=0.0)
    parser.add_argument('--fiber_target_intensity', type=float, default=25.0)
    parser.add_argument('--background_max_intensity', type=float, default=-1.0,
                        help='Cap background intensity (negative = no cap)')
    parser.add_argument('--mask_smoothing_sigma', type=float, default=0.0)
    parser.add_argument('--mask_binary_threshold', type=float, default=0.01)

    # Fine-resolution rendering knobs (match visualize_one_patch.py at sub-micron voxels).
    # At voxel_size ~0.001mm the renderer default tissue_threshold=2.0 silently skips most
    # fiber voxels, so it must be lowered (~0.0) for fibers to render at all.
    parser.add_argument('--tissue_threshold', type=float, default=2.0,
                        help='Min volume intensity for a voxel to receive fiber paint. '
                             'Set ~0.0 at sub-micron voxel_size or fibers barely render.')
    parser.add_argument('--enable_cell_blobs', dest='enable_cell_blobs', action='store_true', default=False,
                        help='Scatter Gaussian cell-body blob distractors into the image only '
                             '(never the mask) so the model learns fiber-vs-cell.')
    parser.add_argument('--cell_blob_count', type=int, default=60)
    parser.add_argument('--cell_blob_intensity', type=float, default=0.3)
    parser.add_argument('--cell_blob_radius_min', type=float, default=1.5)
    parser.add_argument('--cell_blob_radius_max', type=float, default=4.0)
    parser.add_argument('--cornucopia_presets', nargs='+', default=None,
                        help='Restrict cornucopia 3D presets (e.g. ultra_heavy_speckle '
                             'extreme_noise granular_realistic). Default: renderer chooses, '
                             'which includes structured presets that look artificial at fine res.')

    # Inference-shape augmentations (close the train/inference shape gap)
    parser.add_argument('--thinslab_prob', type=float, default=0.3,
                        help='Probability of zero-padding a contiguous Z slab to simulate '
                             'the thin-slab inference setting (e.g. 60 slices padded into 128).')
    parser.add_argument('--thinslab_min_z', type=int, default=30,
                        help='Minimum number of Z slices kept when thin-slab augmentation fires.')
    parser.add_argument('--thinslab_max_z', type=int, default=120,
                        help='Maximum number of Z slices kept when thin-slab augmentation fires.')
    parser.add_argument('--empty_patch_prob', type=float, default=0.05,
                        help='Probability of replacing the patch with all-zero vol+mask so the '
                             'model learns the "empty input -> empty output" prior.')

    # Model
    parser.add_argument('--loss', default='BCE', choices=['BCE', 'focal', 'cldice'],
                        help='Loss function. BCE+pos_weight<1.0 is the reliable knob; '
                             'focal needs the class-balanced alpha (see DiceFocalLoss).')
    parser.add_argument('--pos_weight', type=float, default=0.3,
                        help='For BCE: torch BCEWithLogitsLoss pos_weight (<1 penalises FP). '
                             'For focal: alpha for the positive class in the class-balanced focal loss.')
    parser.add_argument('--min_features', type=int, default=32,
                        help='Base number of features in first encoder stage')
    parser.add_argument('--max_features', type=int, default=320,
                        help='Maximum features per encoder stage (capped for 3D memory)')
    parser.add_argument('--num_stages', type=int, default=5,
                        help='Number of encoder stages (fewer = less memory)')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # Logging
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb; use CSV logger instead')
    parser.add_argument('--wandb_offline', action='store_true', default=True,
                        help='Run wandb in offline mode (default: True)')
    parser.add_argument('--wandb_online', dest='wandb_offline', action='store_false',
                        help='Run wandb in online mode (requires login)')

    # System
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * this)')
    parser.add_argument('--devices', type=int, default=None,
                        help='Number of CUDA devices to use (default: all available)')
    parser.add_argument('--strategy', type=str, default="auto",
                        choices=["auto", "ddp", "ddp_find_unused_parameters_false"],
                        help='Distributed strategy (default: auto; auto-selects DDP for multi-GPU)')
    parser.add_argument('--prefetch_batches', type=int, default=2,
                        help='Prefetch batches for on-the-fly 3D data generation (threaded, default=2)')
    parser.add_argument('--batch_group_factor', type=int, default=10,
                        help='Generate this many train batches per extraction/render cycle to reduce pipeline overhead')
    parser.add_argument('--move_to_gpu', action='store_true',
                        help='Move generated 3D patches to CUDA before returning from the dataloader')
    parser.add_argument('--render_on_cpu', action='store_true',
                        help='Render synthetic 3D patches on CPU instead of CUDA')
    parser.add_argument('--verbose_generation', action='store_true',
                        help='Enable verbose logging for on-the-fly 3D generation')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5,
                        help='Run validation every N epochs (default: 5)')
    parser.add_argument('--log_every_n_steps', type=int, default=20,
                        help='Trainer logging frequency (default: 20)')

    return parser


def main(args):
    cuda_runtime_devices = _cuda_runtime_device_count()
    if cuda_runtime_devices > 0:
        requested_devices = args.devices or cuda_runtime_devices
        _trim_cuda_visible_devices(min(int(requested_devices), cuda_runtime_devices))

    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA runtime device count: {cuda_runtime_devices}')
    print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}')
    print(f'MPS available:  {torch.backends.mps.is_available()}')
    print(f'3D Training - Checkpoints: {args.checkpoint_dir}')
    print(f'Patch size: {args.patch_size}')
    print(
        "Image-only realism augmentations: "
        f"tissue_artifacts={args.enable_tissue_artifacts} "
        f"(strength={args.artifact_strength}), "
        f"granular_noise={args.enable_granular_noise} "
        f"(strength={args.granular_noise_strength}), "
        f"speckle_noise={args.enable_speckle_noise} "
        f"(strength={args.speckle_noise_strength}, density={args.speckle_noise_density})"
    )
    print(
        "Fiber brightness variation: "
        f"streamline={args.fiber_brightness_variation}, "
        f"segment={args.fiber_segment_brightness_variation}"
    )
    print(
        "Fiber rendering: "
        f"mode={args.fiber_render_mode}, target={args.fiber_target_intensity}, "
        f"opacity={args.fiber_opacity}, density_gamma={args.fiber_density_gamma}, "
        f"min_visibility={args.fiber_min_visibility}"
    )
    print(
        "Mask rendering: "
        f"smoothing_sigma={args.mask_smoothing_sigma}, "
        f"binary_threshold={args.mask_binary_threshold}"
    )
    total_micro_batches = int(args.epochs) * int(args.batches_per_epoch)
    if args.on_the_fly and total_micro_batches >= 5000:
        print(
            f"On-the-fly mode requested for {total_micro_batches} micro-batches. "
            "If throughput is still low, precomputing 3D patches is strongly recommended."
        )
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True

    seed_everything(args.seed, workers=True)

    raw_trk_dir = args.trk_dir
    raw_input_nifti = args.input_nifti
    args.trk_dir = _path_or_none(args.trk_dir)
    args.input_nifti = _path_or_none(args.input_nifti)
    _require_existing_dir(args.trk_dir, "--trk_dir", raw_trk_dir)
    _require_existing_file(args.input_nifti, "--input_nifti", raw_input_nifti)

    # Handle white_mask
    white_mask = _path_or_none(args.white_mask)
    if white_mask is not None:
        _require_existing_file(
            white_mask,
            "--white_mask",
            args.white_mask,
            hint='Pass --white_mask None to disable white-matter filtering.',
        )

    if not args.on_the_fly and args.patch_dir:
        raw_patch_dir = args.patch_dir
        args.patch_dir = _path_or_none(args.patch_dir)
        _require_existing_dir(args.patch_dir, "--patch_dir", raw_patch_dir)

    # Data module
    datamodule = OnTheFlyDataModule3D(
        trk_dir=args.trk_dir,
        input_nifti=args.input_nifti,
        white_mask_file=white_mask,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        train_batches_per_epoch=args.batches_per_epoch,
        val_batches=args.val_batches,
        seed=args.seed,
        on_the_fly=args.on_the_fly,
        patch_dir=args.patch_dir,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        voxel_size=args.voxel_size,
        min_streamlines_per_patch=args.min_streamlines_per_patch,
        streamline_margin_fraction=args.streamline_margin_fraction,
        enable_tissue_artifacts=args.enable_tissue_artifacts,
        enable_granular_noise=args.enable_granular_noise,
        enable_speckle_noise=args.enable_speckle_noise,
        enable_dash_noise=args.enable_dash_noise,
        enable_horizontal_banding=args.enable_horizontal_banding,
        granular_noise_strength=args.granular_noise_strength,
        artifact_strength=args.artifact_strength,
        speckle_noise_strength=args.speckle_noise_strength,
        speckle_noise_density=args.speckle_noise_density,
        speckle_noise_sigma=args.speckle_noise_sigma,
        speckle_square_size=args.speckle_square_size,
        dash_noise_strength=args.dash_noise_strength,
        dash_noise_density=args.dash_noise_density,
        dash_length_sigma=args.dash_length_sigma,
        dash_cross_sigma=args.dash_cross_sigma,
        banding_strength=args.banding_strength,
        banding_axis=args.banding_axis,
        fiber_intensity_min=args.fiber_intensity_min,
        fiber_intensity_max=args.fiber_intensity_max,
        fiber_max_boost=None if args.fiber_max_boost < 0 else args.fiber_max_boost,
        fiber_opacity=args.fiber_opacity,
        fiber_smoothing_sigma=args.fiber_smoothing_sigma,
        fiber_antialias=args.fiber_antialias,
        min_streamlines_rendered=None if args.min_streamlines_rendered <= 0 else args.min_streamlines_rendered,
        fiber_brightness_variation=args.fiber_brightness_variation,
        fiber_segment_brightness_variation=args.fiber_segment_brightness_variation,
        fiber_render_mode=args.fiber_render_mode,
        fiber_density_gamma=args.fiber_density_gamma,
        fiber_min_visibility=args.fiber_min_visibility,
        fiber_target_intensity=args.fiber_target_intensity,
        background_max_intensity=None if args.background_max_intensity < 0 else args.background_max_intensity,
        tissue_threshold=args.tissue_threshold,
        enable_cell_blobs=args.enable_cell_blobs,
        cell_blob_count=args.cell_blob_count,
        cell_blob_intensity=args.cell_blob_intensity,
        cell_blob_radius_range=(args.cell_blob_radius_min, args.cell_blob_radius_max),
        cornucopia_allowed_presets=args.cornucopia_presets,
        mask_smoothing_sigma=args.mask_smoothing_sigma,
        mask_binary_threshold=args.mask_binary_threshold,
        prefetch_batches=args.prefetch_batches,
        move_to_gpu=args.move_to_gpu,
        batch_group_factor=args.batch_group_factor,
        render_use_gpu=not args.render_on_cpu,
        verbose_generation=args.verbose_generation,
        thinslab_prob=args.thinslab_prob,
        thinslab_min_z=args.thinslab_min_z,
        thinslab_max_z=args.thinslab_max_z,
        empty_patch_prob=args.empty_patch_prob,
    )

    # Ensure checkpoint dir exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Checkpoint resume
    checkpoint_path = None
    if args.resume_checkpoint:
        checkpoint_path = args.resume_checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"--resume_checkpoint does not exist: {checkpoint_path}")
        print(f"Resuming trainer state from explicit checkpoint: {checkpoint_path}")
    elif not args.no_resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, "last.ckpt")
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found, starting from scratch")
            checkpoint_path = None
    else:
        existing_last = os.path.join(args.checkpoint_dir, "last.ckpt")
        if os.path.exists(existing_last):
            print(f"Starting fresh; existing checkpoint ignored because --no_resume was set: {existing_last}")
        else:
            print("No checkpoint found, starting from scratch")

    # Logger
    if args.no_wandb:
        logger = CSVLogger(save_dir=args.checkpoint_dir, name="logs")
        print("Using CSV logger (wandb disabled)")
    else:
        import wandb
        wandb_mode = "offline" if args.wandb_offline else "online"
        os.environ.setdefault("WANDB_MODE", wandb_mode)
        # Honor WANDB_PROJECT / WANDB_NAME from the environment (e.g. set in the
        # SLURM script) so runs land in the intended project; fall back to the
        # historical defaults otherwise.
        wandb_project = os.environ.get("WANDB_PROJECT") or "unet3d-training-on-synthetic"
        wandb_run_name = os.environ.get("WANDB_NAME") or f"{args.wandb_name}_{args.loss}"
        # Lightning's WandbLogger only calls wandb.init() on global rank 0, so a
        # multi-GPU (DDP) launch produces a single run rather than one per rank.
        logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            save_code=False,
            log_model=False,
            offline=args.wandb_offline,
        )
        print(f"Using WandB logger (mode: {wandb_mode}, project: {wandb_project}, run: {wandb_run_name})")

    # Model
    model = FlexibleUNet3D(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        loss=args.loss,
        pos_weight=args.pos_weight,
        min_features=args.min_features,
        max_features=args.max_features,
        num_stages=args.num_stages,
        in_channels=1,
    )

    # Load pretrained weights if provided
    if args.pretraining_checkpoint and args.pretraining_checkpoint.lower() != "none":
        print(f"Loading pretrained weights from {args.pretraining_checkpoint}")
        ckpt = torch.load(args.pretraining_checkpoint, map_location="cpu")
        if 'state_dict' in ckpt:
            state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
            model.load_state_dict(state_dict, strict=False)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best_3d-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    timing_callback = ThroughputProfilerCallback(
        print_every_batches=max(10, args.log_every_n_steps),
        window=max(20, args.log_every_n_steps * 2),
    )

    callbacks = [checkpoint_callback, timing_callback]
    real_proxy_zarr = _path_or_none(args.real_proxy_zarr)
    if real_proxy_zarr is not None:
        callbacks.append(RealLSMProxyCallback(
            zarr_path=real_proxy_zarr,
            patch_size=tuple(args.patch_size),
            target_voxel_size_um=tuple(args.real_proxy_target_voxel_um),
            level_index=args.real_proxy_level,
            channel_index=args.real_proxy_channel,
            normalize_percentiles=(1.0, 99.0),
            threshold=args.real_proxy_threshold,
        ))
        print(f"Real-data transfer proxy ENABLED on zarr: {real_proxy_zarr}")
    else:
        print("Real-data transfer proxy disabled (--real_proxy_zarr not set)")

    # Accelerator: ConvTranspose3d and max_pool3d are not supported on MPS,
    # so fall back to CPU on Apple Silicon.
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        accelerator = "gpu"
        cuda_device_count = _cuda_runtime_device_count()
        devices = args.devices or cuda_device_count or 1
        if world_size_env > 1:
            if world_size_env > cuda_device_count:
                raise RuntimeError(
                    f"Distributed launch requested WORLD_SIZE={world_size_env}, "
                    f"but CUDA runtime exposes only {cuda_device_count} usable device(s). "
                    "Relaunch torchrun with --nproc_per_node no larger than the CUDA runtime count."
                )
            # When launched via torchrun/srun in distributed mode, Lightning expects
            # `devices * num_nodes == WORLD_SIZE`.
            if args.devices is not None and int(args.devices) != world_size_env:
                print(
                    f"Distributed launch detected (WORLD_SIZE={world_size_env}, LOCAL_RANK={local_rank_env}). "
                    f"Overriding devices={args.devices} -> {world_size_env}."
                )
            devices = world_size_env
        elif args.devices is not None and args.devices > cuda_device_count:
            print(f"Requested {args.devices} CUDA devices but only {cuda_device_count} available; using {cuda_device_count}.")
            devices = cuda_device_count
        # bf16 has fp32's dynamic range (no ~65504 overflow that NaN-poisons the
        # forward pass under fp16) and needs no GradScaler. Free on H100/H200
        # (Hopper). Falls back to fp16 only on pre-Ampere cards without bf16.
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        gpu_names = [torch.cuda.get_device_name(i) for i in range(min(devices, cuda_device_count))]
        print(f"Using CUDA devices (count={devices}): {gpu_names}")
    elif torch.backends.mps.is_available():
        accelerator = "cpu"
        precision = "32-true"
        devices = 1
        print("Note: Using CPU — ConvTranspose3d/max_pool3d not supported on MPS. "
              "On a CUDA machine these will run on GPU.")
    else:
        accelerator = "cpu"
        precision = "32-true"
        devices = 1

    if accelerator == "gpu" and int(devices) > 1 and args.strategy == "auto":
        strategy = "ddp_find_unused_parameters_false"
    else:
        strategy = args.strategy
    print(f"Trainer strategy: {strategy}")

    # Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        precision=precision,
        gradient_clip_val=1.0,  # guard against the occasional exploding-grad step
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        benchmark=torch.cuda.is_available(),
        use_distributed_sampler=False,
        check_val_every_n_epoch=max(1, args.check_val_every_n_epoch),
        log_every_n_steps=max(1, args.log_every_n_steps),
        num_sanity_val_steps=0,
    )

    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)

    if not args.no_wandb:
        import wandb
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
        # Only rank 0 ran wandb.init() (via Lightning's WandbLogger), so only
        # rank 0 should finish the run.
        if rank_zero_only.rank == 0:
            wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
