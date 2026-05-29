"""Scale-aware OME-Zarr patch sampling for 3D models.

This module focuses on one core issue: model input tensors often have a fixed shape,
but OME-Zarr source data can have different physical voxel sizes across pyramid levels.

It provides:
- Metadata parsing from ``multiscales`` / ``coordinateTransformations``
- Physical-space-aware patch sampling (fixed field-of-view in micrometers)
- Optional multiscale level sampling
- Streaming/chunked reads from Zarr arrays (slice-based; no full-volume load)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
import zarr

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - optional dependency at runtime
    pl = None


NumberLike = Union[int, float]
Tuple3 = Tuple[float, float, float]
IntTuple3 = Tuple[int, int, int]


_UNIT_TO_UM = {
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometers": 1e-3,
    "um": 1.0,
    "µm": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "millimeter": 1_000.0,
    "millimeters": 1_000.0,
    "mm": 1_000.0,
    "meter": 1_000_000.0,
    "meters": 1_000_000.0,
    "m": 1_000_000.0,
}


@dataclass
class OMEZarrLevelInfo:
    level_index: int
    path: str
    array: Any
    axis_names: Tuple[str, ...]
    shape_zyx: IntTuple3
    voxel_size_um_zyx: Tuple3
    spatial_axis_indices_zyx: IntTuple3
    spatial_permutation_to_zyx: IntTuple3



def _to_tuple3(value: Union[NumberLike, Sequence[NumberLike]], name: str) -> Tuple3:
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return (v, v, v)
    if len(value) != 3:
        raise ValueError(f"{name} must be a scalar or length-3 sequence, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))



def _to_int_tuple3(value: Sequence[int], name: str) -> IntTuple3:
    if len(value) != 3:
        raise ValueError(f"{name} must be length-3, got {value!r}")
    return (int(value[0]), int(value[1]), int(value[2]))



def _unit_to_um(unit: Optional[str]) -> float:
    if unit is None:
        return 1.0
    key = str(unit).strip().lower()
    return _UNIT_TO_UM.get(key, 1.0)



def _normalize_axes(axes: Sequence[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for i, axis in enumerate(axes):
        if isinstance(axis, str):
            normalized.append({"name": axis.lower(), "unit": None})
            continue
        if isinstance(axis, Mapping):
            name = str(axis.get("name", f"axis{i}")).lower()
            unit = axis.get("unit")
            normalized.append({"name": name, "unit": unit})
            continue
        normalized.append({"name": f"axis{i}", "unit": None})
    return normalized



def _best_effort_scale(dataset_meta: Mapping[str, Any], ndim_hint: int) -> List[float]:
    # OME-NGFF stores scale in dataset["coordinateTransformations"].
    transforms = dataset_meta.get("coordinateTransformations", [])
    for tr in transforms:
        if isinstance(tr, Mapping) and tr.get("type") == "scale":
            scale = tr.get("scale")
            if isinstance(scale, Sequence):
                return [float(s) for s in scale]
    return [1.0] * int(ndim_hint)



def _align_axis_and_scale(
    axes_meta: Sequence[Dict[str, Any]],
    scale: Sequence[float],
    ndim: int,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    axes = list(axes_meta)
    scale_list = list(scale)

    if len(axes) == len(scale_list) == ndim:
        return axes, scale_list

    # Common fallback: dataset array has dropped leading non-spatial axes.
    if len(axes) >= ndim:
        axes = axes[-ndim:]
    else:
        missing = ndim - len(axes)
        axes = [{"name": f"axis{i}", "unit": None} for i in range(missing)] + axes

    if len(scale_list) >= ndim:
        scale_list = scale_list[-ndim:]
    else:
        scale_list = [1.0] * (ndim - len(scale_list)) + scale_list

    return axes, scale_list



def _resolve_array(node: Any, channel_index: int = 0) -> Any:
    # Direct array
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return node

    # Common OME layout helper: group contains "c/<channel>"
    if hasattr(node, "__contains__") and "c" in node:
        c_node = node["c"]
        c_key = str(channel_index)
        if hasattr(c_node, "__contains__") and c_key in c_node:
            return _resolve_array(c_node[c_key], channel_index=channel_index)

    # First array in group
    if hasattr(node, "arrays"):
        arrays = list(node.arrays())
        if arrays:
            return arrays[0][1]

    # Recursive walk for nested groups
    if hasattr(node, "groups"):
        for _, sub_group in node.groups():
            arr = _resolve_array(sub_group, channel_index=channel_index)
            if arr is not None:
                return arr

    raise ValueError("Could not resolve an array node from dataset path")



def _extract_level_infos(root: Any, channel_index: int = 0) -> List[OMEZarrLevelInfo]:
    attrs = getattr(root, "attrs", None)
    if attrs is None or "multiscales" not in attrs:
        raise ValueError(
            "OME-Zarr multiscales metadata not found at root.attrs['multiscales']. "
            "Pass the OME root group, not an individual array path."
        )

    multiscales = attrs["multiscales"]
    if not multiscales:
        raise ValueError("root.attrs['multiscales'] is empty")

    ms0 = multiscales[0]
    axes_meta = _normalize_axes(ms0.get("axes", []))
    datasets = ms0.get("datasets", [])
    if not datasets:
        raise ValueError("multiscales[0]['datasets'] is empty")

    level_infos: List[OMEZarrLevelInfo] = []
    for level_index, ds in enumerate(datasets):
        if not isinstance(ds, Mapping):
            continue
        path = str(ds.get("path", ""))
        node = root[path] if path else root
        arr = _resolve_array(node, channel_index=channel_index)

        scale = _best_effort_scale(ds, ndim_hint=getattr(arr, "ndim", len(axes_meta)))
        aligned_axes, aligned_scale = _align_axis_and_scale(axes_meta, scale, arr.ndim)

        axis_names = tuple(str(a.get("name", f"axis{i}")).lower() for i, a in enumerate(aligned_axes))
        name_to_idx = {name: i for i, name in enumerate(axis_names)}

        if all(name in name_to_idx for name in ("z", "y", "x")):
            zyx = (name_to_idx["z"], name_to_idx["y"], name_to_idx["x"])
        else:
            if arr.ndim < 3:
                raise ValueError(
                    f"Array at path '{path}' is {arr.ndim}D; expected at least 3D for volumetric sampling."
                )
            # Fallback to trailing dimensions as (z, y, x)
            zyx = (arr.ndim - 3, arr.ndim - 2, arr.ndim - 1)

        # Spatial axes in array order, then permutation to zyx.
        spatial_in_array_order = [idx for idx in range(arr.ndim) if idx in zyx]
        order_labels = [zyx.index(idx) for idx in spatial_in_array_order]
        perm_to_zyx = tuple(order_labels.index(i) for i in range(3))

        scale_per_axis_um = []
        for i, axis in enumerate(aligned_axes):
            unit_um = _unit_to_um(axis.get("unit"))
            scale_per_axis_um.append(float(aligned_scale[i]) * unit_um)

        shape_zyx = (int(arr.shape[zyx[0]]), int(arr.shape[zyx[1]]), int(arr.shape[zyx[2]]))
        voxel_zyx = (
            float(scale_per_axis_um[zyx[0]]),
            float(scale_per_axis_um[zyx[1]]),
            float(scale_per_axis_um[zyx[2]]),
        )

        level_infos.append(
            OMEZarrLevelInfo(
                level_index=level_index,
                path=path,
                array=arr,
                axis_names=axis_names,
                shape_zyx=shape_zyx,
                voxel_size_um_zyx=voxel_zyx,
                spatial_axis_indices_zyx=zyx,
                spatial_permutation_to_zyx=_to_int_tuple3(perm_to_zyx, "perm_to_zyx"),
            )
        )

    if not level_infos:
        raise ValueError("Could not parse any levels from multiscales metadata")

    return level_infos



def _resize_zyx(volume: np.ndarray, out_shape_zyx: IntTuple3, order: int = 1) -> np.ndarray:
    if tuple(volume.shape) == tuple(out_shape_zyx):
        return volume.astype(np.float32, copy=False)

    zoom_factors = [
        float(out_shape_zyx[0]) / max(1, volume.shape[0]),
        float(out_shape_zyx[1]) / max(1, volume.shape[1]),
        float(out_shape_zyx[2]) / max(1, volume.shape[2]),
    ]
    resized = zoom(
        volume,
        zoom=zoom_factors,
        order=int(order),
        mode="nearest",
        prefilter=(int(order) > 1),
    ).astype(np.float32, copy=False)

    # Guard against 1-off rounding differences from interpolation backend.
    dz = out_shape_zyx[0] - resized.shape[0]
    dy = out_shape_zyx[1] - resized.shape[1]
    dx = out_shape_zyx[2] - resized.shape[2]

    if dz == dy == dx == 0:
        return resized

    # Center crop or pad per axis to force exact output shape.
    def _fit_axis(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
        cur = arr.shape[axis]
        if cur == target:
            return arr
        if cur > target:
            start = (cur - target) // 2
            end = start + target
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(start, end)
            return arr[tuple(sl)]
        pad_before = (target - cur) // 2
        pad_after = target - cur - pad_before
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (pad_before, pad_after)
        return np.pad(arr, pad, mode="constant", constant_values=0)

    out = resized
    out = _fit_axis(out, 0, out_shape_zyx[0])
    out = _fit_axis(out, 1, out_shape_zyx[1])
    out = _fit_axis(out, 2, out_shape_zyx[2])
    return out.astype(np.float32, copy=False)


class PhysicalScaleOMEZarrDataset(Dataset):
    """Patch sampler for OME-Zarr with physical-space-aware sampling.

    Sampling flow:
    1. Choose a pyramid level (`level_sampling`)
    2. Convert desired physical patch size (um) -> source voxel window at that level
    3. Read source window by Zarr slicing (streaming/chunked)
    4. Pad if needed (optional), then resample to fixed output tensor shape

    Output:
    - image tensor: (1, D, H, W)
    - metadata dict (optional)
    """

    def __init__(
        self,
        zarr_path: Optional[str] = None,
        zarr_group: Optional[Any] = None,
        output_patch_size: Sequence[int] = (128, 128, 128),
        samples_per_epoch: int = 1024,
        target_voxel_size_um: Optional[Union[NumberLike, Sequence[NumberLike]]] = None,
        physical_patch_size_um: Optional[Union[NumberLike, Sequence[NumberLike]]] = None,
        level_sampling: Union[str, int, Sequence[int]] = "closest",
        allow_padding: bool = True,
        non_spatial_indices: Optional[Mapping[str, int]] = None,
        channel_index: int = 0,
        interpolation_order: int = 1,
        normalize: bool = True,
        normalize_percentiles: Tuple[float, float] = (1.0, 99.0),
        seed: Optional[int] = None,
        return_metadata: bool = True,
    ):
        super().__init__()

        if zarr_group is None and zarr_path is None:
            raise ValueError("Provide either zarr_path or zarr_group")

        if zarr_group is None:
            zarr_group = zarr.open_group(zarr_path, mode="r")

        self.zarr_path = zarr_path
        self.zarr_group = zarr_group
        self.output_patch_size = _to_int_tuple3(output_patch_size, "output_patch_size")
        self.samples_per_epoch = int(samples_per_epoch)
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be > 0")

        self.target_voxel_size_um = (
            _to_tuple3(target_voxel_size_um, "target_voxel_size_um")
            if target_voxel_size_um is not None
            else None
        )
        self.physical_patch_size_um = (
            _to_tuple3(physical_patch_size_um, "physical_patch_size_um")
            if physical_patch_size_um is not None
            else None
        )

        self.level_sampling = level_sampling
        self.allow_padding = bool(allow_padding)
        self.non_spatial_indices = {
            str(k).lower(): int(v) for k, v in (non_spatial_indices or {}).items()
        }
        self.channel_index = int(channel_index)
        self.interpolation_order = int(interpolation_order)
        self.normalize = bool(normalize)
        self.normalize_percentiles = (float(normalize_percentiles[0]), float(normalize_percentiles[1]))
        self.seed = seed
        self.return_metadata = bool(return_metadata)

        self.levels = _extract_level_infos(self.zarr_group, channel_index=self.channel_index)
        self._closest_level_idx = self._compute_closest_level_index()

    def _compute_closest_level_index(self) -> int:
        if self.target_voxel_size_um is None:
            return 0

        target = np.asarray(self.target_voxel_size_um, dtype=np.float64)
        errors = []
        for lv in self.levels:
            v = np.asarray(lv.voxel_size_um_zyx, dtype=np.float64)
            # Symmetric relative mismatch in log-space.
            err = np.mean(np.abs(np.log2(np.maximum(v, 1e-12) / np.maximum(target, 1e-12))))
            errors.append(float(err))
        return int(np.argmin(errors))

    def describe_levels(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for lv in self.levels:
            rows.append(
                {
                    "level": lv.level_index,
                    "path": lv.path,
                    "shape_zyx": tuple(int(s) for s in lv.shape_zyx),
                    "voxel_size_um_zyx": tuple(float(v) for v in lv.voxel_size_um_zyx),
                }
            )
        return rows

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng_for_index(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(self.seed) + int(index) * 1_000_003)

    def _sample_level(self, rng: np.random.Generator, index: int) -> OMEZarrLevelInfo:
        mode = self.level_sampling

        if isinstance(mode, (int, np.integer)):
            idx = int(mode)
            return self.levels[idx]

        if isinstance(mode, Sequence) and not isinstance(mode, str):
            choices = [int(x) for x in mode]
            if not choices:
                raise ValueError("level_sampling sequence is empty")
            idx = int(rng.choice(choices))
            return self.levels[idx]

        mode_str = str(mode).lower()
        if mode_str == "closest":
            return self.levels[self._closest_level_idx]
        if mode_str == "cycle":
            return self.levels[int(index) % len(self.levels)]
        if mode_str == "random":
            idx = int(rng.integers(0, len(self.levels)))
            return self.levels[idx]
        if mode_str == "weighted_random":
            if self.target_voxel_size_um is None:
                idx = int(rng.integers(0, len(self.levels)))
                return self.levels[idx]
            target = np.asarray(self.target_voxel_size_um, dtype=np.float64)
            errors = []
            for lv in self.levels:
                v = np.asarray(lv.voxel_size_um_zyx, dtype=np.float64)
                err = np.mean(np.abs(np.log2(np.maximum(v, 1e-12) / np.maximum(target, 1e-12))))
                errors.append(float(err))
            weights = np.exp(-np.asarray(errors, dtype=np.float64))
            weights = weights / np.sum(weights)
            idx = int(rng.choice(np.arange(len(self.levels)), p=weights))
            return self.levels[idx]

        raise ValueError(
            "Unsupported level_sampling. Use one of: int, sequence[int], "
            "'closest', 'cycle', 'random', 'weighted_random'."
        )

    def _requested_physical_size_um(self, level: OMEZarrLevelInfo) -> Tuple3:
        if self.physical_patch_size_um is not None:
            return self.physical_patch_size_um
        if self.target_voxel_size_um is not None:
            return (
                self.output_patch_size[0] * self.target_voxel_size_um[0],
                self.output_patch_size[1] * self.target_voxel_size_um[1],
                self.output_patch_size[2] * self.target_voxel_size_um[2],
            )
        # No physical target provided: preserve shape semantics at current level.
        return (
            self.output_patch_size[0] * level.voxel_size_um_zyx[0],
            self.output_patch_size[1] * level.voxel_size_um_zyx[1],
            self.output_patch_size[2] * level.voxel_size_um_zyx[2],
        )

    def _sample_patch(self, level: OMEZarrLevelInfo, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, Any]]:
        physical_um = np.asarray(self._requested_physical_size_um(level), dtype=np.float64)
        voxel_um = np.asarray(level.voxel_size_um_zyx, dtype=np.float64)
        level_shape = np.asarray(level.shape_zyx, dtype=np.int64)
        source_extent_um = level_shape.astype(np.float64) * voxel_um

        requested_window = np.maximum(1, np.round(physical_um / np.maximum(voxel_um, 1e-12)).astype(np.int64))

        if not self.allow_padding and np.any(requested_window > level_shape):
            raise ValueError(
                "Requested physical patch does not fit selected level shape and allow_padding=False. "
                f"requested_window={tuple(requested_window.tolist())}, level_shape={tuple(level_shape.tolist())}"
            )

        read_window = np.minimum(requested_window, level_shape)
        origin = []
        for dim_size, win in zip(level_shape.tolist(), read_window.tolist()):
            max_start = dim_size - win
            if max_start <= 0:
                origin.append(0)
            else:
                origin.append(int(rng.integers(0, max_start + 1)))
        origin = np.asarray(origin, dtype=np.int64)

        axis_to_zyx = {
            int(level.spatial_axis_indices_zyx[0]): 0,
            int(level.spatial_axis_indices_zyx[1]): 1,
            int(level.spatial_axis_indices_zyx[2]): 2,
        }

        selection: List[Union[int, slice]] = []
        for axis_idx, axis_name in enumerate(level.axis_names):
            if axis_idx in axis_to_zyx:
                zyx_pos = axis_to_zyx[axis_idx]
                start = int(origin[zyx_pos])
                size = int(read_window[zyx_pos])
                selection.append(slice(start, start + size))
            else:
                if axis_name == "c":
                    axis_sel = self.non_spatial_indices.get("c", self.channel_index)
                else:
                    axis_sel = self.non_spatial_indices.get(axis_name, 0)
                selection.append(int(axis_sel))

        patch = np.asarray(level.array[tuple(selection)], dtype=np.float32)
        # Keep singleton spatial axes (for example z=1 at coarse levels).
        # Only collapse extra leading singleton axes if non-spatial dimensions
        # unexpectedly survive indexing.
        while patch.ndim > 3 and patch.shape[0] == 1:
            patch = patch[0]
        if patch.ndim != 3:
            raise ValueError(
                f"Expected 3D patch after non-spatial indexing, got shape {patch.shape} at level {level.level_index}"
            )

        patch_zyx = np.transpose(patch, level.spatial_permutation_to_zyx).astype(np.float32, copy=False)

        pad_needed = np.maximum(0, requested_window - read_window)
        if np.any(pad_needed > 0):
            patch_zyx = np.pad(
                patch_zyx,
                (
                    (0, int(pad_needed[0])),
                    (0, int(pad_needed[1])),
                    (0, int(pad_needed[2])),
                ),
                mode="constant",
                constant_values=0,
            )

        coverage_fraction = np.clip(
            read_window.astype(np.float64) / np.maximum(requested_window.astype(np.float64), 1.0),
            0.0,
            1.0,
        )
        warnings: List[str] = []
        if np.any(source_extent_um + 1e-6 < physical_um):
            warnings.append(
                "Requested physical patch exceeds available extent at selected level; "
                "padding was applied where needed."
            )

        patch_resized = _resize_zyx(patch_zyx, self.output_patch_size, order=self.interpolation_order)

        if self.normalize:
            lo, hi = self.normalize_percentiles
            v_lo, v_hi = np.percentile(patch_resized, [lo, hi])
            if v_hi > v_lo:
                patch_resized = (patch_resized - v_lo) / (v_hi - v_lo)
                patch_resized = np.clip(patch_resized, 0.0, 1.0)
            else:
                patch_resized = np.zeros_like(patch_resized, dtype=np.float32)

        meta: Dict[str, Any] = {
            "level_index": int(level.level_index),
            "level_path": str(level.path),
            "origin_zyx": tuple(int(v) for v in origin.tolist()),
            "read_window_vox_zyx": tuple(int(v) for v in read_window.tolist()),
            "requested_window_vox_zyx": tuple(int(v) for v in requested_window.tolist()),
            "padded_vox_zyx": tuple(int(v) for v in pad_needed.tolist()),
            "shape_zyx": tuple(int(v) for v in level.shape_zyx),
            "source_extent_um_zyx": tuple(float(v) for v in source_extent_um.tolist()),
            "source_voxel_size_um_zyx": tuple(float(v) for v in level.voxel_size_um_zyx),
            "requested_physical_size_um_zyx": tuple(float(v) for v in physical_um.tolist()),
            "source_coverage_fraction_zyx": tuple(float(v) for v in coverage_fraction.tolist()),
            "target_voxel_size_um_zyx": (
                tuple(float(v) for v in self.target_voxel_size_um)
                if self.target_voxel_size_um is not None
                else None
            ),
            "output_patch_size_zyx": tuple(int(v) for v in self.output_patch_size),
            "warnings": warnings,
        }

        return patch_resized.astype(np.float32, copy=False), meta

    def __getitem__(self, index: int):
        rng = self._rng_for_index(index)
        level = self._sample_level(rng, index=index)
        patch, meta = self._sample_patch(level, rng)

        image_t = torch.from_numpy(patch).float().unsqueeze(0)  # (1, D, H, W)
        if self.return_metadata:
            return image_t, meta
        return image_t


class OMEZarrPatchDataModule(pl.LightningDataModule if pl is not None else object):
    """Lightning DataModule wrapper for `PhysicalScaleOMEZarrDataset`."""

    def __init__(
        self,
        zarr_path: str,
        batch_size: int = 2,
        num_workers: int = 0,
        output_patch_size: Sequence[int] = (128, 128, 128),
        train_samples_per_epoch: int = 1024,
        val_samples_per_epoch: int = 128,
        target_voxel_size_um: Optional[Union[NumberLike, Sequence[NumberLike]]] = None,
        physical_patch_size_um: Optional[Union[NumberLike, Sequence[NumberLike]]] = None,
        level_sampling: Union[str, int, Sequence[int]] = "closest",
        allow_padding: bool = True,
        non_spatial_indices: Optional[Mapping[str, int]] = None,
        channel_index: int = 0,
        interpolation_order: int = 1,
        normalize: bool = True,
        normalize_percentiles: Tuple[float, float] = (1.0, 99.0),
        seed: Optional[int] = None,
        return_metadata: bool = True,
    ):
        if pl is None:
            raise ImportError("pytorch_lightning is required to use OMEZarrPatchDataModule")
        super().__init__()

        self.zarr_path = zarr_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.output_patch_size = tuple(int(x) for x in output_patch_size)
        self.train_samples_per_epoch = int(train_samples_per_epoch)
        self.val_samples_per_epoch = int(val_samples_per_epoch)

        self.dataset_kwargs = dict(
            target_voxel_size_um=target_voxel_size_um,
            physical_patch_size_um=physical_patch_size_um,
            level_sampling=level_sampling,
            allow_padding=allow_padding,
            non_spatial_indices=non_spatial_indices,
            channel_index=channel_index,
            interpolation_order=interpolation_order,
            normalize=normalize,
            normalize_percentiles=normalize_percentiles,
            seed=seed,
            return_metadata=return_metadata,
        )

        self.train_dataset: Optional[PhysicalScaleOMEZarrDataset] = None
        self.val_dataset: Optional[PhysicalScaleOMEZarrDataset] = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = PhysicalScaleOMEZarrDataset(
            zarr_path=self.zarr_path,
            output_patch_size=self.output_patch_size,
            samples_per_epoch=self.train_samples_per_epoch,
            **self.dataset_kwargs,
        )
        self.val_dataset = PhysicalScaleOMEZarrDataset(
            zarr_path=self.zarr_path,
            output_patch_size=self.output_patch_size,
            samples_per_epoch=self.val_samples_per_epoch,
            **self.dataset_kwargs,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=bool(self.num_workers),
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(0, self.num_workers // 2),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=bool(self.num_workers),
        )
