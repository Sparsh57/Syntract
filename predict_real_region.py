"""Run the trained 3D model on patches around chosen coordinates of a real OME-Zarr volume.

This script extracts patches centered around manually selected voxel coordinates
that are known to contain real fibers. Useful for validation and debugging
patch extraction from OME-Zarr data.

Example:
    python predict_real_region.py \
        --zarr_path /path/to/data.ome.zarr \
        --center_coords 19 12000 20000 \
        --patch_size 128 128 128 \
        --num_patches 5 \
        --save_dir ./test_patches
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_TRAINING_DIR = os.path.join(SCRIPT_DIR, "training")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SYNTHETIC_TRAINING_DIR not in sys.path:
    sys.path.insert(0, SYNTHETIC_TRAINING_DIR)


def _try_import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def _jsonable(obj: Any):
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _stats(name: str, arr: np.ndarray) -> dict[str, Any]:
    """Compute statistics on an array matching predict_omezarr_thinslab_3d.py style."""
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {
        "name": name,
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
    }
    if finite.size == 0:
        out.update({
            "min": None,
            "p1": None,
            "p50": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
            "nonzero_fraction": None,
        })
        return out
    finite = finite.astype(np.float64, copy=False)
    p = np.percentile(finite, [0, 1, 50, 99, 100])
    out.update({
        "min": float(p[0]),
        "p1": float(p[1]),
        "p50": float(p[2]),
        "p99": float(p[3]),
        "max": float(p[4]),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "nonzero_fraction": float(np.mean(finite != 0)),
    })
    return out


def _best_slice_indices(vol: np.ndarray) -> tuple[int, int, int]:
    if vol.ndim != 3 or min(vol.shape) <= 0:
        return 0, 0, 0
    nz_coords = np.argwhere(vol > 0)
    if nz_coords.shape[0] == 0:
        # Fully empty — fall back to volume center
        return vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
    # Use the center of the nonzero bounding box so the slice cuts through
    # the middle of actual tissue rather than the edge of the data region.
    z = int(round((nz_coords[:, 0].min() + nz_coords[:, 0].max()) / 2))
    y = int(round((nz_coords[:, 1].min() + nz_coords[:, 1].max()) / 2))
    x = int(round((nz_coords[:, 2].min() + nz_coords[:, 2].max()) / 2))
    return z, y, x


def _normalize_for_png(arr2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr2d, dtype=np.float32)
    # Use nonzero voxels only so zero-padded borders don't affect the window.
    nz = arr[arr > 0]
    if nz.size < 10:
        nz = arr[np.isfinite(arr)]
    if nz.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    # p2 as lo preserves tissue context (not too dark), p99.5 as hi clips only
    # the very brightest specular spots — matches Neuroglancer's auto-contrast.
    lo = float(np.percentile(nz, 2))
    hi = float(np.percentile(nz, 99.5))
    if hi <= lo:
        lo, hi = float(nz.min()), float(nz.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def _save_debug_png(path: Path, volumes: list) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False

    tile = 256
    label_h = 32
    left_w = 170
    rows = []
    for name, vol in volumes:
        z, y, x = _best_slice_indices(vol)
        rows.append((name, [
            ("axial z=%d" % z, vol[z]),
            ("coronal y=%d" % y, vol[:, y, :]),
            ("sagittal x=%d" % x, vol[:, :, x]),
        ]))

    sheet = Image.new("RGB", (left_w + 3 * tile, len(rows) * (tile + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    y0 = 0
    for row_name, cells in rows:
        draw.text((8, y0 + label_h + 8), row_name, fill=(0, 0, 0))
        x0 = left_w
        for title, arr2d in cells:
            draw.text((x0 + 4, y0 + 8), title, fill=(0, 0, 0))
            img = Image.fromarray(_normalize_for_png(arr2d), mode="L").convert("RGB")
            h, w = np.asarray(arr2d).shape[:2]
            scale = min(float(tile) / max(1, w), float(tile) / max(1, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            sheet.paste(img, (x0 + (tile - img.width) // 2, y0 + label_h + (tile - img.height) // 2))
            x0 += tile
        y0 += tile + label_h
    sheet.save(path)
    return True


def _resample_xy_clean(vol_zyx: np.ndarray, zoom_y: float, zoom_x: float) -> np.ndarray:
    """Resample only Y and X with clean anti-aliasing.

    For downsamples (zoom < 1): block-mean reduce by the largest integer factor
    that fits, then bilinear zoom the remainder. Block averaging is the correct
    low-pass for downsampling — no Gaussian halo, no aliasing.

    For upsamples (zoom > 1): plain bilinear zoom.

    Z axis is left untouched.
    """
    from scipy.ndimage import zoom as _zoom

    vol = vol_zyx
    # --- Y axis ---
    if zoom_y < 1.0:
        # Block-mean reduce. e.g. zoom_y = 1/50 = 0.02 → factor = 50.
        factor_y = max(1, int(round(1.0 / zoom_y)))
        if factor_y > 1:
            # Trim so length is a multiple of factor_y
            new_len = (vol.shape[1] // factor_y) * factor_y
            if new_len > 0:
                vol = vol[:, :new_len, :]
                vol = vol.reshape(
                    vol.shape[0], new_len // factor_y, factor_y, vol.shape[2]
                ).mean(axis=2)
            # Remaining zoom factor after block reduce
            residual_y = zoom_y * factor_y
        else:
            residual_y = zoom_y
    else:
        residual_y = zoom_y

    # --- X axis ---
    if zoom_x < 1.0:
        factor_x = max(1, int(round(1.0 / zoom_x)))
        if factor_x > 1:
            new_len = (vol.shape[2] // factor_x) * factor_x
            if new_len > 0:
                vol = vol[:, :, :new_len]
                vol = vol.reshape(
                    vol.shape[0], vol.shape[1], new_len // factor_x, factor_x
                ).mean(axis=3)
            residual_x = zoom_x * factor_x
        else:
            residual_x = zoom_x
    else:
        residual_x = zoom_x

    # Apply residual non-integer zoom (close to 1.0) with bilinear
    if abs(residual_y - 1.0) > 0.01 or abs(residual_x - 1.0) > 0.01:
        vol = _zoom(
            vol, (1.0, residual_y, residual_x),
            order=1, mode="nearest", prefilter=False,
        )
    return vol.astype(np.float32, copy=False)


def _choose_level_index(levels, fixed_level, target_voxel_um, min_z_slices=4):
    """Pick the pyramid level index closest to target_voxel_um (XY-plane match).

    Levels with fewer than min_z_slices Z slices are skipped — they can't fill
    a meaningful 3D patch and are only useful for 2D overview rendering.
    """
    if fixed_level is not None:
        return int(fixed_level)
    if target_voxel_um is None:
        return 0
    target = np.asarray(target_voxel_um, dtype=np.float64)
    best_idx, best_err = 0, float("inf")
    for lv in levels:
        if lv.shape_zyx[0] < min_z_slices:
            continue
        voxel = np.asarray(lv.voxel_size_um_zyx, dtype=np.float64)
        # Match on XY only — same strategy as thinslab's closest_xy
        err = float(np.mean(np.abs(np.log2(np.maximum(voxel[1:], 1e-12) / np.maximum(target[1:], 1e-12)))))
        if err < best_err:
            best_err = err
            best_idx = lv.level_index
    return best_idx


class SpecificRegionDataset:
    """Sample patches around specific coordinates at a chosen level."""

    def __init__(
        self,
        zarr_path: str,
        center_coords_zyx: tuple[int, int, int],
        patch_size_zyx: tuple[int, int, int],
        num_patches: int = 1,
        level_index: int = 0,
        channel_index: int = 0,
        allow_padding: bool = True,
        normalize: bool = True,
        normalize_mode: str = "percentile",
        normalize_percentiles: tuple[float, float] = (1.0, 99.0),
        input_gamma: float = 1.0,
        input_gain: float = 1.0,
        target_voxel_size_um: tuple[float, float, float] | None = None,
        jitter_radius: int = 0,
        seed: int = 42,
    ):
        """
        Args:
            zarr_path: Path to .ome.zarr root
            center_coords_zyx: Target voxel coords (z, y, x) at the chosen level
            patch_size_zyx: Output patch size (D, H, W)
            num_patches: How many patches to sample around center (with optional jitter)
            level_index: Pyramid level index to use (0 = highest resolution)
            channel_index: Which channel to extract
            allow_padding: Allow padding if patch extends beyond array bounds
            normalize: Normalize to [0, 1] using percentiles or min-max
            normalize_mode: 'percentile', 'nonzero_percentile', or 'minmax'
            normalize_percentiles: (lo, hi) percentiles for normalization (if mode='percentile')
            input_gamma: gamma applied after normalization (>1 darkens, <1 brightens)
            input_gain: multiplicative gain applied after gamma
            target_voxel_size_um: if set, resample patch to this voxel size (µm) before
                outputting. Z is padded with zeros if the tissue is thinner than the output.
                Normalization is applied to real voxels before padding.
            jitter_radius: Random offset (±) from center in voxels, per patch
            seed: Random seed for jitter
        """
        import zarr

        try:
            from training.datamodules.omezarr import _extract_level_infos
        except (ImportError, ModuleNotFoundError):
            try:
                from datamodules.omezarr import _extract_level_infos
            except (ImportError, ModuleNotFoundError):
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "omezarr",
                    os.path.join(SYNTHETIC_TRAINING_DIR, "datamodules", "omezarr.py")
                )
                if spec and spec.loader:
                    omezarr = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(omezarr)
                    _extract_level_infos = omezarr._extract_level_infos
                else:
                    raise ImportError("Could not load omezarr module")

        self.zarr_path = zarr_path
        self.zarr_group = zarr.open_group(zarr_path, mode="r")
        self.center_coords_zyx = center_coords_zyx
        self.patch_size_zyx = patch_size_zyx
        self.num_patches = int(num_patches)
        self.level_index = int(level_index)
        self.channel_index = int(channel_index)
        self.allow_padding = bool(allow_padding)
        self.normalize = bool(normalize)
        self.normalize_mode = normalize_mode.lower()
        self.normalize_percentiles = normalize_percentiles
        self.input_gamma = float(input_gamma)
        self.input_gain = float(input_gain)
        self.target_voxel_size_um = (
            tuple(float(v) for v in target_voxel_size_um)
            if target_voxel_size_um is not None else None
        )
        self.jitter_radius = int(jitter_radius)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.levels = _extract_level_infos(self.zarr_group, channel_index=self.channel_index)
        if not self.levels:
            raise ValueError("No levels found in OME-Zarr metadata")
        if self.level_index >= len(self.levels):
            raise ValueError(
                f"level_index={self.level_index} out of range. "
                f"Available levels: 0..{len(self.levels) - 1}"
            )

        self.level = self.levels[self.level_index]
        self._validate_center_coords()

    def _validate_center_coords(self):
        """Check that center coords are within bounds."""
        z, y, x = self.center_coords_zyx
        max_z, max_y, max_x = self.level.shape_zyx
        if not (0 <= z < max_z and 0 <= y < max_y and 0 <= x < max_x):
            raise ValueError(
                f"Center coords {(z, y, x)} out of level bounds {self.level.shape_zyx} "
                f"at level {self.level_index}"
            )

    def _extract_patch(self, origin_zyx: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, Any]]:
        """Extract a single patch at specified origin.

        If target_voxel_size_um is set, the physical window corresponding to
        patch_size_zyx output voxels at target resolution is read from the zarr,
        resampled to target voxel size, normalized on the real voxels, then
        zero-padded in Z to reach patch_size_zyx[0] if tissue is thinner.
        """

        patch_d, patch_h, patch_w = self.patch_size_zyx
        origin_z, origin_y, origin_x = origin_zyx
        source_vox = self.level.voxel_size_um_zyx  # (sz, sy, sx) µm

        # --- Determine how many source voxels to read per axis ---
        if self.target_voxel_size_um is not None:
            tv = self.target_voxel_size_um  # target µm per output voxel
            # Z: always read ALL available Z slices at native resolution — never downsample Z.
            # The tissue is a thin slab; we keep every Z slice and pad to patch_d.
            read_z = self.level.shape_zyx[0]
            # XY: read the physical extent corresponding to the output patch at target voxel size
            read_y = max(1, int(round(patch_h * tv[1] / max(source_vox[1], 1e-12))))
            read_x = max(1, int(round(patch_w * tv[2] / max(source_vox[2], 1e-12))))
        else:
            read_z, read_y, read_x = patch_d, patch_h, patch_w

        axis_to_zyx = {
            int(self.level.spatial_axis_indices_zyx[0]): 0,
            int(self.level.spatial_axis_indices_zyx[1]): 1,
            int(self.level.spatial_axis_indices_zyx[2]): 2,
        }
        read_window_zyx = [read_z, read_y, read_x]

        selection: list[int | slice] = []
        actual_read_zyx = []
        for axis_idx, axis_name in enumerate(self.level.axis_names):
            if axis_idx in axis_to_zyx:
                zyx_pos = axis_to_zyx[axis_idx]
                if zyx_pos == 0:
                    start, end = origin_z, origin_z + read_z
                elif zyx_pos == 1:
                    start, end = origin_y, origin_y + read_y
                else:
                    start, end = origin_x, origin_x + read_x
                start = max(0, start)
                end = min(self.level.shape_zyx[zyx_pos], end)
                actual_read_zyx.append(end - start)
                selection.append(slice(start, end))
            else:
                actual_read_zyx.append(None)
                if axis_name == "c":
                    selection.append(self.channel_index)
                else:
                    selection.append(0)

        # Read raw voxels from zarr
        patch = np.asarray(self.level.array[tuple(selection)], dtype=np.float32)
        while patch.ndim > 3 and patch.shape[0] == 1:
            patch = patch[0]
        if patch.ndim != 3:
            raise ValueError(
                f"Expected 3D patch, got shape {patch.shape} after indexing at level {self.level_index}"
            )
        patch_zyx = np.transpose(patch, self.level.spatial_permutation_to_zyx).astype(np.float32, copy=False)

        actual_z, actual_y, actual_x = patch_zyx.shape

        if self.target_voxel_size_um is not None:
            tv = self.target_voxel_size_um
            # Z is never resampled — kept at native resolution, padded later.
            # Only resample XY to the target voxel size.
            zoom_y = source_vox[1] / max(tv[1], 1e-12)
            zoom_x = source_vox[2] / max(tv[2], 1e-12)
            if abs(zoom_y - 1.0) > 0.01 or abs(zoom_x - 1.0) > 0.01:
                # For large downsamples (zoom < 1), bilinear sampling alone aliases —
                # producing the "granular" look. For big upsamples it blurs.
                # Solution: integer block-mean reduce first (anti-aliased, no halo),
                # then bilinear zoom the remainder.
                patch_zyx = _resample_xy_clean(patch_zyx, zoom_y, zoom_x)

            # Crop to output YX size if resampling overshot
            patch_zyx = patch_zyx[:, :patch_h, :patch_w]

        # Normalize on real (non-padded) voxels BEFORE padding
        if self.normalize:
            finite = patch_zyx[np.isfinite(patch_zyx)]
            if self.normalize_mode == "minmax":
                v_lo = float(finite.min()) if finite.size > 0 else 0.0
                v_hi = float(finite.max()) if finite.size > 0 else 0.0
            elif self.normalize_mode == "nonzero_percentile":
                nz = finite[finite != 0]
                if nz.size > 0:
                    lo_pct, hi_pct = self.normalize_percentiles
                    v_lo = float(np.percentile(nz, lo_pct))
                    v_hi = float(np.percentile(nz, hi_pct))
                else:
                    v_lo, v_hi = 0.0, 0.0
            else:  # percentile
                lo_pct, hi_pct = self.normalize_percentiles
                if finite.size > 0:
                    v_lo = float(np.percentile(finite, lo_pct))
                    v_hi = float(np.percentile(finite, hi_pct))
                else:
                    v_lo, v_hi = 0.0, 0.0

            if v_hi > v_lo:
                patch_zyx = np.clip((patch_zyx - v_lo) / (v_hi - v_lo), 0.0, 1.0)
            else:
                patch_zyx = np.zeros_like(patch_zyx, dtype=np.float32)

            if self.input_gamma != 1.0:
                patch_zyx = np.power(patch_zyx, self.input_gamma).astype(np.float32, copy=False)
            if self.input_gain != 1.0:
                patch_zyx = np.clip(patch_zyx * self.input_gain, 0.0, 1.0).astype(np.float32, copy=False)

        # Pad to output size AFTER normalization — zeros stay zero
        real_z = patch_zyx.shape[0]
        pad_z = max(0, patch_d - real_z)
        pad_y = max(0, patch_h - patch_zyx.shape[1])
        pad_x = max(0, patch_w - patch_zyx.shape[2])
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            # Centre the real data in Z so padding is symmetric; pad Y/X at end
            pad_z_before = pad_z // 2
            pad_z_after = pad_z - pad_z_before
            patch_zyx = np.pad(
                patch_zyx,
                ((pad_z_before, pad_z_after), (0, pad_y), (0, pad_x)),
                mode="constant",
                constant_values=0,
            )

        # Final crop to exact output size if resampling was slightly off
        patch_zyx = patch_zyx[:patch_d, :patch_h, :patch_w]

        meta = {
            "level_index": int(self.level_index),
            "origin_zyx": origin_zyx,
            "center_coords_zyx": self.center_coords_zyx,
            "read_window_vox_zyx": tuple(read_window_zyx),
            "actual_read_vox_zyx": (actual_z, actual_y, actual_x),
            "level_shape_zyx": self.level.shape_zyx,
            "voxel_size_um_zyx": self.level.voxel_size_um_zyx,
            "target_voxel_size_um_zyx": self.target_voxel_size_um,
            "output_patch_size_zyx": self.patch_size_zyx,
            "real_z_slices_after_resample": real_z,
            "z_padding": pad_z,
        }

        return patch_zyx, meta

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        # Add optional jitter to center coords (in source-level voxel space).
        center_z, center_y, center_x = self.center_coords_zyx
        if self.jitter_radius > 0:
            jitter_z = self.rng.randint(-self.jitter_radius, self.jitter_radius + 1)
            jitter_y = self.rng.randint(-self.jitter_radius, self.jitter_radius + 1)
            jitter_x = self.rng.randint(-self.jitter_radius, self.jitter_radius + 1)
            center_z += jitter_z
            center_y += jitter_y
            center_x += jitter_x

        # Compute the read window half-size in source-level voxels.
        # When target_voxel_size_um is set, the read window is larger than patch_size
        # (e.g. 128 output voxels at 50 µm = ~2133 source voxels at 3 µm).
        # We must center the READ window on center_coords, not the output patch.
        patch_d, patch_h, patch_w = self.patch_size_zyx
        if self.target_voxel_size_um is not None:
            sv = self.level.voxel_size_um_zyx
            tv = self.target_voxel_size_um
            # Z: always start at 0 — we read the full Z extent of the level
            origin_z = 0
            half_y = int(round(patch_h * tv[1] / max(sv[1], 1e-12))) // 2
            half_x = int(round(patch_w * tv[2] / max(sv[2], 1e-12))) // 2
        else:
            origin_z = center_z - patch_d // 2
            half_y = patch_h // 2
            half_x = patch_w // 2

        origin_y = center_y - half_y
        origin_x = center_x - half_x

        patch, meta = self._extract_patch((origin_z, origin_y, origin_x))
        image_t = torch.from_numpy(patch).float().unsqueeze(0)  # (1, D, H, W)
        return image_t, meta


def _load_model(checkpoint_path: str, device: torch.device, pos_weight: float = 1.0):
    try:
        from training.unet3d import FlexibleUNet3D
    except ImportError:
        from unet3d import FlexibleUNet3D

    ckpt = torch.load(checkpoint_path, map_location=device)
    hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    allowed = [
        "batch_size", "learning_rate", "weight_decay", "warmup_epochs",
        "min_features", "max_features", "num_stages", "loss",
        "freeze_encoder", "pos_weight", "in_channels",
    ]
    kwargs = {k: hparams[k] for k in allowed if k in hparams}
    if "learning_rate" not in kwargs:
        kwargs["learning_rate"] = 1e-4
    if "pos_weight" not in kwargs:
        kwargs["pos_weight"] = float(pos_weight)

    model = FlexibleUNet3D(**kwargs)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model, {"checkpoint": str(checkpoint_path), "missing_keys": missing, "unexpected_keys": unexpected}


def _try_start_wandb(args, config):
    mode = str(os.environ.get("WANDB_MODE", "")).strip().lower()
    requested = bool(args.wandb) or mode in ("online", "offline", "dryrun")
    if bool(args.no_wandb) or not requested:
        return None
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    try:
        import wandb
    except Exception as exc:
        print("W&B requested but import failed; continuing without W&B logging:", exc)
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=_jsonable(config),
    )
    if run is not None:
        print("W&B run:", getattr(run, "url", None) or getattr(run, "name", None))
    return wandb


def main():
    parser = argparse.ArgumentParser("Specific-region OME-Zarr 3D inference")
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--output_dir", default="./specific_region_inference")
    parser.add_argument(
        "--center_coords", nargs=3, type=int, required=True, metavar=("Z", "Y", "X"),
        help="Center voxel coordinates (z, y, x) at the chosen level",
    )
    parser.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128], metavar=("D", "H", "W"))
    parser.add_argument("--num_patches", type=int, default=5)
    parser.add_argument("--level_index", type=int, default=None,
                        help="Pyramid level to use (0=highest res). Mutually exclusive with --target_voxel_size_um.")
    parser.add_argument(
        "--target_voxel_size_um", nargs=3, type=float, default=[50.0, 50.0, 50.0], metavar=("Z", "Y", "X"),
        help="Target voxel size in µm for the output patch (default: 50 50 50). XY is resampled "
             "to this size. Z is never downsampled — all source Z slices are kept at native "
             "resolution and zero-padded to patch_size[0] if thinner than the output.",
    )
    parser.add_argument("--channel_index", type=int, default=0)
    parser.add_argument("--jitter_radius", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize",
        choices=("minmax", "percentile", "nonzero_percentile", "none"),
        default="nonzero_percentile",
        help="Patch normalization. 'nonzero_percentile' (default) clips to percentiles of tissue "
             "voxels only, avoiding zero-background bias. 'percentile' includes background zeros.",
    )
    parser.add_argument("--percentile_low", type=float, default=1.0)
    parser.add_argument("--percentile_high", type=float, default=99.0)
    parser.add_argument("--input_gamma", type=float, default=1.0)
    parser.add_argument("--input_gain", type=float, default=1.0)
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_debug_patches", type=int, default=3,
                        help="Save debug PNGs for the first N patches")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "syntract3d"))
    parser.add_argument("--wandb_run_name", default=os.environ.get("WANDB_RUN_NAME", "specific_region_inference"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "dryrun"), default=None,
        help="Override WANDB_MODE",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug_patches"
    debug_dir.mkdir(parents=True, exist_ok=True)

    patch_size = tuple(int(v) for v in args.patch_size)

    # Always probe all levels upfront — needed to auto-scale center coords.
    import zarr as _zarr
    try:
        from training.datamodules.omezarr import _extract_level_infos as _eli
    except ImportError:
        try:
            from datamodules.omezarr import _extract_level_infos as _eli
        except ImportError:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(
                "omezarr",
                os.path.join(SYNTHETIC_TRAINING_DIR, "datamodules", "omezarr.py")
            )
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _eli = _mod._extract_level_infos
    _probe_group = _zarr.open_group(args.zarr_path, mode="r")
    _all_levels = _eli(_probe_group, channel_index=args.channel_index)

    print("Available pyramid levels:")
    for lv in _all_levels:
        print(f"  level {lv.level_index}: shape_zyx={lv.shape_zyx}, voxel_um={lv.voxel_size_um_zyx}")

    if args.level_index is not None and args.target_voxel_size_um is not None:
        print("Warning: --level_index and --target_voxel_size_um both given; using --level_index.")

    # When target_voxel_size_um is set, always read from level 0 (highest resolution).
    # Resampling to the target voxel size happens inside _extract_patch.
    # This way: (a) max fidelity, (b) center coords are used as-is (no scaling).
    if args.level_index is not None:
        resolved_level = int(args.level_index)
    else:
        resolved_level = 0
        if args.target_voxel_size_um is not None:
            print(f"Forcing level 0 for target voxel size {args.target_voxel_size_um} µm "
                  f"(reading at native res, then resampling).")

    chosen_lv = _all_levels[resolved_level]
    raw_coords = tuple(int(v) for v in args.center_coords)
    # Coords are interpreted in the chosen level's voxel space directly.
    # With resolved_level=0 this is the highest-resolution voxel grid.
    center_coords = (
        max(0, min(raw_coords[0], chosen_lv.shape_zyx[0] - 1)),
        max(0, min(raw_coords[1], chosen_lv.shape_zyx[1] - 1)),
        max(0, min(raw_coords[2], chosen_lv.shape_zyx[2] - 1)),
    )
    if center_coords != raw_coords:
        print(f"Clamped center coords {raw_coords} -> {center_coords} to level {resolved_level} bounds")

    dataset = SpecificRegionDataset(
        zarr_path=args.zarr_path,
        center_coords_zyx=center_coords,
        patch_size_zyx=patch_size,
        num_patches=args.num_patches,
        level_index=resolved_level,
        channel_index=args.channel_index,
        jitter_radius=args.jitter_radius,
        seed=args.seed,
        normalize=args.normalize != "none",
        normalize_mode=args.normalize if args.normalize != "none" else "percentile",
        normalize_percentiles=(args.percentile_low, args.percentile_high),
        input_gamma=args.input_gamma,
        input_gain=args.input_gain,
        target_voxel_size_um=(
            tuple(float(v) for v in args.target_voxel_size_um)
            if args.target_voxel_size_um is not None else None
        ),
    )

    level = dataset.level
    shape_zyx = level.shape_zyx
    source_voxel_um = level.voxel_size_um_zyx

    print("\nSelected level:")
    print("  level:", resolved_level)
    print("  source shape zyx:", shape_zyx)
    print("  source voxel um zyx:", source_voxel_um)
    print("  center coords zyx:", center_coords)
    print("  patch size zyx:", patch_size)
    print("  target voxel um zyx:", args.target_voxel_size_um)
    print("  num patches:", args.num_patches)
    print("  jitter radius:", args.jitter_radius)

    wandb_config = {
        "zarr_path": args.zarr_path,
        "output_dir": str(out_dir),
        "level_index": resolved_level,
        "source_shape_zyx": shape_zyx,
        "source_voxel_um_zyx": source_voxel_um,
        "center_coords_zyx": center_coords,
        "patch_size_zyx": patch_size,
        "num_patches": args.num_patches,
        "jitter_radius": args.jitter_radius,
        "seed": args.seed,
        "normalization": args.normalize,
        "percentile_low": float(args.percentile_low),
        "percentile_high": float(args.percentile_high),
        "input_gamma": float(args.input_gamma),
        "input_gain": float(args.input_gain),
        "model_checkpoint": args.model_checkpoint,
        "threshold": float(args.threshold),
    }
    wandb = _try_start_wandb(args, wandb_config)
    if wandb is not None:
        wandb.log({
            "data/source_z": shape_zyx[0],
            "data/source_y": shape_zyx[1],
            "data/source_x": shape_zyx[2],
            "run/num_patches": args.num_patches,
        })

    if args.device == "auto":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch
        device = torch.device(args.device)

    model = None
    model_load_report = None
    if args.model_checkpoint:
        print("Loading model:", args.model_checkpoint)
        model, model_load_report = _load_model(
            args.model_checkpoint, device=device, pos_weight=args.pos_weight
        )
        print("Loaded model on", str(device))
    else:
        print("No model checkpoint provided. Data extraction and debug output only.")

    patch_reports = []

    for patch_idx in range(len(dataset)):
        image_t, meta = dataset[patch_idx]
        raw_patch = image_t.squeeze(0).cpu().numpy()  # (D, H, W) — already normalised

        nonzero_frac = float(np.mean(raw_patch > 0))
        nz_coords = np.argwhere(raw_patch > 0)
        bbox = (
            tuple(int(v) for v in nz_coords.min(axis=0))
            if nz_coords.shape[0] > 0 else None,
            tuple(int(v) for v in nz_coords.max(axis=0))
            if nz_coords.shape[0] > 0 else None,
        )
        print(f"Patch {patch_idx}: origin={meta['origin_zyx']} | "
              f"nonzero={nonzero_frac:.1%} | "
              f"bbox_min={bbox[0]} bbox_max={bbox[1]} | "
              f"val range=[{raw_patch.min():.3f}, {raw_patch.max():.3f}]")

        patch_meta = {
            "patch_index": int(patch_idx),
            "origin_zyx": meta["origin_zyx"],
            "center_coords_zyx": meta["center_coords_zyx"],
            "voxel_size_um_zyx": meta["voxel_size_um_zyx"],
            "raw_patch_shape_zyx": tuple(int(v) for v in raw_patch.shape),
            "model_input_stats": _stats("model_input", raw_patch),
            "normalization": {
                "method": args.normalize,
                "percentile_low": float(args.percentile_low),
                "percentile_high": float(args.percentile_high),
                "input_gamma": float(args.input_gamma),
                "input_gain": float(args.input_gain),
            },
        }

        prob = None
        pred_bin = None
        if model is not None:
            with torch.no_grad():
                patch_tensor = image_t.unsqueeze(0).to(device)
                logits = model(patch_tensor)
                prob = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
            pred_bin = (prob >= float(args.threshold)).astype(np.uint8)
            patch_meta["probability_stats"] = _stats("probability_zyx", prob)
            patch_meta["binary_stats"] = _stats("binary_zyx", pred_bin)

        # Debug PNG for first N patches (same as thinslab)
        if patch_idx < int(args.save_debug_patches):
            stem = "patch_%04d" % patch_idx
            np.save(debug_dir / (stem + "_model_input.npy"), raw_patch.astype(np.float32))
            if prob is not None:
                np.save(debug_dir / (stem + "_probability.npy"), prob.astype(np.float32))
                np.save(debug_dir / (stem + "_binary.npy"), pred_bin)
            debug_volumes = [("model_input", raw_patch)]
            if prob is not None:
                debug_volumes.append(("probability", prob))
            if pred_bin is not None:
                debug_volumes.append(("binary", pred_bin))
            debug_png_path = debug_dir / (stem + "_slices.png")
            _save_debug_png(debug_png_path, debug_volumes)
            (debug_dir / (stem + "_meta.json")).write_text(
                json.dumps(_jsonable(patch_meta), indent=2)
            )
            if wandb is not None and debug_png_path.exists():
                log_patch = {
                    "debug/%s_slices" % stem: wandb.Image(str(debug_png_path)),
                    "patch/model_input_mean": patch_meta["model_input_stats"]["mean"],
                    "patch/model_input_nonzero_fraction": patch_meta["model_input_stats"]["nonzero_fraction"],
                }
                if prob is not None:
                    log_patch["patch/probability_mean"] = patch_meta["probability_stats"]["mean"]
                    log_patch["patch/binary_fraction"] = patch_meta["binary_stats"]["mean"]
                wandb.log(log_patch, step=patch_idx)

        patch_reports.append(patch_meta)
        if (patch_idx + 1) % 10 == 0 or patch_idx + 1 == len(dataset):
            print("Processed %d/%d patches" % (patch_idx + 1, len(dataset)))

    # Aggregate prediction if model ran
    prediction_summary = None
    if model is not None and patch_reports:
        prob_means = [
            p["probability_stats"]["mean"]
            for p in patch_reports
            if "probability_stats" in p and p["probability_stats"]["mean"] is not None
        ]
        binary_means = [
            p["binary_stats"]["mean"]
            for p in patch_reports
            if "binary_stats" in p and p["binary_stats"]["mean"] is not None
        ]
        prediction_summary = {
            "num_patches_with_predictions": len(prob_means),
            "probability_mean_across_patches": float(np.mean(prob_means)) if prob_means else None,
            "probability_std_across_patches": float(np.std(prob_means)) if prob_means else None,
            "binary_fraction_mean_across_patches": float(np.mean(binary_means)) if binary_means else None,
            "threshold": float(args.threshold),
        }

    summary = {
        "zarr_path": args.zarr_path,
        "selected_level": {
            "level_index": resolved_level,
            "shape_zyx": shape_zyx,
            "voxel_um_zyx": source_voxel_um,
        },
        "center_coords_zyx": center_coords,
        "patch_size_zyx": patch_size,
        "num_patches": args.num_patches,
        "jitter_radius": args.jitter_radius,
        "seed": args.seed,
        "normalization": {
            "method": args.normalize,
            "percentile_low": float(args.percentile_low),
            "percentile_high": float(args.percentile_high),
            "input_gamma": float(args.input_gamma),
            "input_gain": float(args.input_gain),
        },
        "model": model_load_report,
        "prediction_summary": prediction_summary,
        "patch_reports": patch_reports,
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))

    if wandb is not None:
        log_final = {"run/patches_processed": len(patch_reports)}
        if prediction_summary is not None:
            if prediction_summary["probability_mean_across_patches"] is not None:
                log_final["prediction/probability_mean"] = prediction_summary["probability_mean_across_patches"]
            if prediction_summary["binary_fraction_mean_across_patches"] is not None:
                log_final["prediction/binary_fraction"] = prediction_summary["binary_fraction_mean_across_patches"]
        wandb.log(log_final)
        try:
            artifact = wandb.Artifact(args.wandb_run_name + "_outputs", type="prediction")
            artifact.add_file(str(out_dir / "summary.json"))
            for path in sorted(debug_dir.glob("*")):
                if path.is_file():
                    artifact.add_file(str(path))
            wandb.log_artifact(artifact)
        except Exception as exc:
            print("W&B artifact logging failed:", exc)
        wandb.finish()

    print("\nSaved:")
    print(" ", out_dir / "summary.json")
    print(" ", debug_dir)
    return 0


if __name__ == "__main__":
    main()
