"""Run pragmatic 3D inference on a thin-slab OME-Zarr volume.

This script is meant for the current real-data situation:

- The model expects 128 x 128 x 128 inputs.
- The OME-Zarr may have only a small number of real Z slices.
- We do not stretch the real Z axis to 128. We insert the real Z slab into a
  zero-padded 128-deep tensor, run the model, then crop predictions back to
  the real Z slab.
- XY is sampled with a stride chosen from the requested physical voxel size
  (default 50 um = 0.05 mm).

Example:
    python3 synthetic-training/predict_omezarr_thinslab_3d.py \
        --zarr_path /path/to/data.ome.zarr \
        --model_checkpoint /path/to/best_3d.ckpt \
        --output_dir real_inference_test \
        --fixed_level 0 \
        --target_voxel_size_um 50 50 50 \
        --patch_size 128 128 128 \
        --stride_xy 64
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _unit_to_um(unit):
    if unit is None:
        return 1.0
    unit = str(unit).strip().lower()
    if unit in ("um", "micrometer", "micrometers", "µm"):
        return 1.0
    if unit in ("mm", "millimeter", "millimeters"):
        return 1000.0
    if unit in ("nm", "nanometer", "nanometers"):
        return 0.001
    if unit in ("m", "meter", "meters"):
        return 1000000.0
    return 1.0


def _stats(name, arr):
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    out = {"name": name, "shape": tuple(int(v) for v in arr.shape), "dtype": str(arr.dtype)}
    if finite.size == 0:
        out.update({"min": None, "p1": None, "p50": None, "p99": None, "max": None})
        out.update({"mean": None, "std": None, "nonzero_fraction": None})
        return out
    finite = finite.astype(np.float64, copy=False)
    p = np.percentile(finite, [0, 1, 50, 99, 100])
    out.update(
        {
            "min": float(p[0]),
            "p1": float(p[1]),
            "p50": float(p[2]),
            "p99": float(p[3]),
            "max": float(p[4]),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "nonzero_fraction": float(np.mean(finite != 0)),
        }
    )
    return out


def _get_scale(dataset_meta, ndim):
    for transform in dataset_meta.get("coordinateTransformations", []):
        if hasattr(transform, "get") and transform.get("type") == "scale":
            scale = transform.get("scale")
            if scale is not None:
                return [float(v) for v in scale]
    return [1.0] * int(ndim)


def _resolve_array(node, channel_index):
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return node

    if hasattr(node, "__contains__") and "c" in node:
        cnode = node["c"]
        ckey = str(channel_index)
        if hasattr(cnode, "__contains__") and ckey in cnode:
            return _resolve_array(cnode[ckey], channel_index)

    if hasattr(node, "arrays"):
        arrays = list(node.arrays())
        if arrays:
            return arrays[0][1]

    if hasattr(node, "groups"):
        for _, group in node.groups():
            try:
                return _resolve_array(group, channel_index)
            except Exception:
                pass

    raise ValueError("Could not find an array under this Zarr node")


def _parse_ome_levels(root, channel_index):
    attrs = getattr(root, "attrs", {})
    if "multiscales" not in attrs:
        return []

    ms0 = attrs["multiscales"][0]
    axes = ms0.get("axes", [])
    datasets = ms0.get("datasets", [])
    levels = []

    for level_index, ds in enumerate(datasets):
        path = str(ds.get("path", ""))
        node = root[path] if path else root
        arr = _resolve_array(node, channel_index)

        ndim = int(getattr(arr, "ndim", len(arr.shape)))
        axes_here = list(axes)
        if len(axes_here) >= ndim:
            axes_here = axes_here[-ndim:]
        else:
            missing = ndim - len(axes_here)
            axes_here = [{"name": "axis%d" % i} for i in range(missing)] + axes_here

        scale = _get_scale(ds, ndim)
        if len(scale) >= ndim:
            scale = scale[-ndim:]
        else:
            scale = [1.0] * (ndim - len(scale)) + scale

        axis_names = []
        units = []
        for i, axis in enumerate(axes_here):
            if isinstance(axis, str):
                axis_names.append(axis.lower())
                units.append(None)
            else:
                axis_names.append(str(axis.get("name", "axis%d" % i)).lower())
                units.append(axis.get("unit"))

        name_to_idx = {name: i for i, name in enumerate(axis_names)}
        if all(name in name_to_idx for name in ("z", "y", "x")):
            zyx_axes = (name_to_idx["z"], name_to_idx["y"], name_to_idx["x"])
        elif ndim >= 3:
            zyx_axes = (ndim - 3, ndim - 2, ndim - 1)
        else:
            continue

        voxel_um_per_axis = [scale[i] * _unit_to_um(units[i]) for i in range(ndim)]
        spatial_in_array_order = [idx for idx in range(ndim) if idx in zyx_axes]
        labels = [zyx_axes.index(idx) for idx in spatial_in_array_order]
        perm_to_zyx = tuple(labels.index(i) for i in range(3))

        levels.append(
            {
                "level_index": int(level_index),
                "path": path,
                "array": arr,
                "axis_names": tuple(axis_names),
                "zyx_axes": tuple(int(v) for v in zyx_axes),
                "spatial_permutation_to_zyx": tuple(int(v) for v in perm_to_zyx),
                "shape_zyx": tuple(int(arr.shape[i]) for i in zyx_axes),
                "voxel_um_zyx": tuple(float(voxel_um_per_axis[i]) for i in zyx_axes),
            }
        )
    return levels


def _level_spacing_error(level, target_voxel_um, strategy):
    voxel = np.asarray(level["voxel_um_zyx"], dtype=np.float64)
    target = np.asarray(target_voxel_um, dtype=np.float64)
    if strategy == "closest_xy":
        voxel = voxel[1:]
        target = target[1:]
    elif strategy == "first":
        return float(level["level_index"])
    return float(np.mean(np.abs(np.log2(np.maximum(voxel, 1e-12) / np.maximum(target, 1e-12)))))


def _choose_level(levels, fixed_level, target_voxel_um, level_strategy, min_level_z_slices):
    if fixed_level is not None:
        for level in levels:
            if int(level["level_index"]) == int(fixed_level):
                return level
        raise ValueError("--fixed_level %d not found" % int(fixed_level))

    candidates = [
        level
        for level in levels
        if int(level["shape_zyx"][0]) >= int(min_level_z_slices)
    ]
    if not candidates:
        raise ValueError(
            "No OME-Zarr levels have at least %d Z slices" % int(min_level_z_slices)
        )
    if str(level_strategy) == "first":
        return candidates[0]
    return min(
        candidates,
        key=lambda level: _level_spacing_error(level, target_voxel_um, level_strategy),
    )


def _covering_starts(length, window, stride):
    length = int(length)
    window = int(window)
    stride = max(1, int(stride))
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _select_windows(windows, patch_size, model_grid_shape, max_patches, mode, seed):
    indexed = list(enumerate(windows))
    if not indexed:
        return [], []

    count = int(max_patches)
    if count <= 0 or count >= len(indexed):
        count = len(indexed)

    mode = str(mode)
    if mode == "scan":
        selected = indexed[:count]
    elif mode == "center":
        patch = np.asarray(patch_size, dtype=np.float64)
        grid_center = np.asarray(model_grid_shape, dtype=np.float64) / 2.0

        def distance(row):
            _, start = row
            patch_center = np.asarray(start, dtype=np.float64) + patch / 2.0
            return float(np.sum((patch_center - grid_center) ** 2))

        selected = sorted(indexed, key=distance)[:count]
    elif mode == "even":
        indices = np.linspace(0, len(indexed) - 1, count, dtype=np.int64)
        selected = [indexed[int(i)] for i in indices]
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(len(indexed), size=count, replace=False)
        selected = [indexed[int(i)] for i in indices]
    else:
        raise ValueError("Unknown window subset mode: %s" % mode)

    return [window for _, window in selected], [int(idx) for idx, _ in selected]


def _slice_len(start, stop, step):
    return max(0, (int(stop) - int(start) + int(step) - 1) // int(step))


def _axis_slice_from_model_window(model_start, model_window, source_dim, source_step):
    source_start = int(model_start) * int(source_step)
    source_stop = min(int(source_dim), source_start + int(model_window) * int(source_step))
    return slice(source_start, source_stop, int(source_step))


def _make_selection(level, z_slice, y_slice, x_slice, channel_index):
    axis_to_zyx = {
        int(level["zyx_axes"][0]): 0,
        int(level["zyx_axes"][1]): 1,
        int(level["zyx_axes"][2]): 2,
    }
    spatial_slices = [z_slice, y_slice, x_slice]
    selection = []
    for axis_idx, axis_name in enumerate(level["axis_names"]):
        if axis_idx in axis_to_zyx:
            selection.append(spatial_slices[axis_to_zyx[axis_idx]])
        elif axis_name == "c":
            selection.append(int(channel_index))
        else:
            selection.append(0)
    return tuple(selection)


def _offsets_for_step(step, samples):
    step = int(step)
    samples = max(1, int(samples))
    if step <= 1 or samples <= 1:
        return [0]
    samples = min(samples, step)
    return sorted({int(round(v)) for v in np.linspace(0, step - 1, samples)})


def _read_one_patch_zyx(level, z_slice, y_slice, x_slice, channel_index):
    selection = _make_selection(level, z_slice, y_slice, x_slice, channel_index)
    raw = np.asarray(level["array"][selection])
    patch = raw.astype(np.float32, copy=False)

    while patch.ndim > 3 and patch.shape[0] == 1:
        patch = patch[0]
    if patch.ndim != 3:
        raise ValueError("Expected 3D patch after indexing, got shape %r" % (patch.shape,))

    return np.transpose(patch, level["spatial_permutation_to_zyx"]).astype(np.float32, copy=False)


def _offset_slice(base_slice, offset, source_dim, target_count):
    start = int(base_slice.start) + int(offset)
    if start >= int(source_dim):
        return slice(int(source_dim), int(source_dim), int(base_slice.step))
    stop = min(int(source_dim), start + int(target_count) * int(base_slice.step))
    return slice(start, stop, int(base_slice.step))


def _read_patch_zyx(
    level,
    starts_zyx,
    patch_size,
    source_steps_zyx,
    channel_index,
    xy_downsample_mode,
    offset_mean_samples,
):
    shape_zyx = level["shape_zyx"]
    z_slice = _axis_slice_from_model_window(starts_zyx[0], patch_size[0], shape_zyx[0], source_steps_zyx[0])
    y_slice = _axis_slice_from_model_window(starts_zyx[1], patch_size[1], shape_zyx[1], source_steps_zyx[1])
    x_slice = _axis_slice_from_model_window(starts_zyx[2], patch_size[2], shape_zyx[2], source_steps_zyx[2])

    valid_shape = (
        _slice_len(z_slice.start, z_slice.stop, z_slice.step),
        _slice_len(y_slice.start, y_slice.stop, y_slice.step),
        _slice_len(x_slice.start, x_slice.stop, x_slice.step),
    )
    if str(xy_downsample_mode) == "stride":
        patch_zyx = _read_one_patch_zyx(level, z_slice, y_slice, x_slice, channel_index)
        return patch_zyx, valid_shape, (z_slice, y_slice, x_slice)

    y_offsets = _offsets_for_step(source_steps_zyx[1], offset_mean_samples)
    x_offsets = _offsets_for_step(source_steps_zyx[2], offset_mean_samples)
    acc = np.zeros(valid_shape, dtype=np.float32)
    counts = np.zeros(valid_shape, dtype=np.float32)

    for y_off in y_offsets:
        y_off_slice = _offset_slice(y_slice, y_off, shape_zyx[1], valid_shape[1])
        for x_off in x_offsets:
            x_off_slice = _offset_slice(x_slice, x_off, shape_zyx[2], valid_shape[2])
            part = _read_one_patch_zyx(level, z_slice, y_off_slice, x_off_slice, channel_index)
            z_len = min(part.shape[0], valid_shape[0])
            y_len = min(part.shape[1], valid_shape[1])
            x_len = min(part.shape[2], valid_shape[2])
            if z_len <= 0 or y_len <= 0 or x_len <= 0:
                continue
            acc[:z_len, :y_len, :x_len] += part[:z_len, :y_len, :x_len]
            counts[:z_len, :y_len, :x_len] += 1.0

    patch_zyx = acc / np.maximum(counts, 1.0)
    return patch_zyx, valid_shape, (z_slice, y_slice, x_slice)


def _insert_with_padding(patch_zyx, patch_size, z_pad_position):
    out = np.zeros(tuple(patch_size), dtype=np.float32)
    z_len = min(int(patch_zyx.shape[0]), int(patch_size[0]))
    y_len = min(int(patch_zyx.shape[1]), int(patch_size[1]))
    x_len = min(int(patch_zyx.shape[2]), int(patch_size[2]))

    if z_pad_position == "front":
        z_offset = 0
    elif z_pad_position == "end":
        z_offset = int(patch_size[0]) - z_len
    else:
        z_offset = (int(patch_size[0]) - z_len) // 2

    out[z_offset : z_offset + z_len, :y_len, :x_len] = patch_zyx[:z_len, :y_len, :x_len]
    return out, (z_offset, 0, 0), (z_len, y_len, x_len)


def _normalize_patch(patch, method, percentile_low, percentile_high, input_gamma, input_gain):
    patch = patch.astype(np.float32, copy=False)
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return np.zeros_like(patch, dtype=np.float32), {"lo": None, "hi": None, "method": method}

    if method == "none":
        return patch, {"lo": None, "hi": None, "method": method}

    if method == "minmax":
        lo = float(finite.min())
        hi = float(finite.max())
    elif method == "nonzero_percentile":
        nz = finite[finite != 0]
        if nz.size == 0:
            return np.zeros_like(patch, dtype=np.float32), {"lo": 0.0, "hi": 0.0, "method": method}
        lo, hi = np.percentile(nz, [float(percentile_low), float(percentile_high)])
        lo = float(lo)
        hi = float(hi)
    else:
        lo, hi = np.percentile(finite, [float(percentile_low), float(percentile_high)])
        lo = float(lo)
        hi = float(hi)

    meta = {
        "lo": lo,
        "hi": hi,
        "method": method,
        "input_gamma": float(input_gamma),
        "input_gain": float(input_gain),
    }

    if hi <= lo:
        return np.zeros_like(patch, dtype=np.float32), meta

    normalized = np.clip((patch - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    if float(input_gamma) != 1.0:
        normalized = np.power(normalized, float(input_gamma)).astype(np.float32, copy=False)
    if float(input_gain) != 1.0:
        normalized = np.clip(normalized * float(input_gain), 0.0, 1.0).astype(np.float32, copy=False)
    return normalized, meta


def _normalize_for_png(arr2d):
    arr = np.asarray(arr2d, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def _best_indices(vol):
    if vol.ndim != 3 or min(vol.shape) <= 0:
        return 0, 0, 0
    z = int(np.argmax(vol.sum(axis=(1, 2)))) if vol.shape[0] else 0
    y = int(np.argmax(vol.sum(axis=(0, 2)))) if vol.shape[1] else 0
    x = int(np.argmax(vol.sum(axis=(0, 1)))) if vol.shape[2] else 0
    return z, y, x


def _save_debug_png(path, volumes):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False

    tile = 256
    label_h = 32
    left_w = 170
    rows = []
    for name, vol in volumes:
        z, y, x = _best_indices(vol)
        rows.append(
            (
                name,
                [
                    ("axial z=%d" % z, vol[z]),
                    ("coronal y=%d" % y, vol[:, y, :]),
                    ("sagittal x=%d" % x, vol[:, :, x]),
                ],
            )
        )

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


def _find_checkpoints(model_folder, saved_model):
    if saved_model == "last":
        pattern = "last.ckpt"
    else:
        pattern = "best_3d*.ckpt"
    return sorted(Path(model_folder).glob(pattern))


def _load_model(checkpoint_path, device, pos_weight):
    import torch
    from unet3d import FlexibleUNet3D

    ckpt = torch.load(checkpoint_path, map_location=device)
    hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    allowed = [
        "batch_size",
        "learning_rate",
        "weight_decay",
        "warmup_epochs",
        "min_features",
        "max_features",
        "num_stages",
        "loss",
        "freeze_encoder",
        "pos_weight",
        "in_channels",
    ]
    kwargs = {}
    for key in allowed:
        if key in hparams:
            kwargs[key] = hparams[key]
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


def _save_optional_nifti(output_path, arr_zyx, effective_voxel_um_zyx):
    try:
        import nibabel as nib
    except Exception:
        return False

    voxel_mm = [float(v) / 1000.0 for v in effective_voxel_um_zyx]
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = voxel_mm[2]
    affine[1, 1] = voxel_mm[1]
    affine[2, 2] = voxel_mm[0]
    # NIfTI viewers conventionally display xyz, so store x/y/z order.
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    nib.save(nib.Nifti1Image(arr_xyz, affine), str(output_path))
    return True


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
    parser = argparse.ArgumentParser("Thin-slab OME-Zarr 3D inference with Z padding")
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--output_dir", default="./omezarr_thinslab_inference")
    parser.add_argument("--channel_index", type=int, default=0)
    parser.add_argument(
        "--fixed_level",
        type=int,
        default=None,
        help="Force one OME-Zarr pyramid level. If omitted, --level_strategy chooses a level.",
    )
    parser.add_argument(
        "--level_strategy",
        choices=("closest_xy", "closest_zyx", "first"),
        default="closest_xy",
        help="Level selection when --fixed_level is omitted.",
    )
    parser.add_argument(
        "--min_level_z_slices",
        type=int,
        default=1,
        help="Only auto-select levels with at least this many Z slices.",
    )
    parser.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128], metavar=("Z", "Y", "X"))
    parser.add_argument(
        "--target_voxel_size_um",
        nargs=3,
        type=float,
        default=[50.0, 50.0, 50.0],
        metavar=("Z_UM", "Y_UM", "X_UM"),
    )
    parser.add_argument(
        "--z_sampling",
        choices=("native", "target"),
        default="native",
        help="native keeps all real Z slices and pads; target strides Z toward target_voxel_size_um[0]",
    )
    parser.add_argument(
        "--z_pad_position",
        choices=("center", "front", "end"),
        default="center",
        help="Where the real slab is inserted inside the 128-deep model tensor",
    )
    parser.add_argument("--stride_z", type=int, default=64, help="Sliding-window stride in model Z voxels")
    parser.add_argument("--stride_xy", type=int, default=64, help="Sliding-window stride in model Y/X voxels")
    parser.add_argument("--max_patches", type=int, default=0, help="0 means process every planned patch")
    parser.add_argument(
        "--window_subset",
        choices=("scan", "center", "even", "random"),
        default="scan",
        help="Which sliding-window locations to process when --max_patches limits the run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for --window_subset random")
    parser.add_argument(
        "--xy_downsample_mode",
        choices=("stride", "offset_mean"),
        default="offset_mean",
        help="stride takes one source pixel per target voxel; offset_mean averages several shifted samples.",
    )
    parser.add_argument(
        "--offset_mean_samples",
        type=int,
        default=5,
        help="Samples per XY axis for --xy_downsample_mode offset_mean.",
    )
    parser.add_argument(
        "--normalize",
        choices=("minmax", "percentile", "nonzero_percentile", "none"),
        default="percentile",
        help="Patch normalization before model input. 'percentile' clips to [p_lo, p_hi] "
             "to avoid bright artifacts crushing the real signal under minmax.",
    )
    parser.add_argument("--percentile_low", type=float, default=1.0)
    parser.add_argument("--percentile_high", type=float, default=99.0)
    parser.add_argument(
        "--input_gamma",
        type=float,
        default=1.0,
        help="Post-normalization gamma for model input. >1 darkens; <1 brightens.",
    )
    parser.add_argument(
        "--input_gain",
        type=float,
        default=1.0,
        help="Post-normalization multiplier for model input after gamma.",
    )
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument("--model_folder", default=None)
    parser.add_argument("--saved_model", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_debug_patches", type=int, default=3)
    parser.add_argument("--save_nifti", action="store_true")
    parser.add_argument(
        "--no_save_raw_downsampled",
        action="store_true",
        help="Do not save the sampled real-image volume as raw_downsampled_zyx.npy",
    )
    parser.add_argument("--wandb", action="store_true", help="Log run metadata, patch stats, and artifacts to W&B")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B even if WANDB_MODE is set")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "syntract3d"))
    parser.add_argument("--wandb_run_name", default=os.environ.get("WANDB_RUN_NAME", "omezarr_thinslab_inference"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "dryrun"),
        default=None,
        help="Optional override for WANDB_MODE",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug_patches"
    debug_dir.mkdir(parents=True, exist_ok=True)

    patch_size = tuple(int(v) for v in args.patch_size)
    target_voxel_um = tuple(float(v) for v in args.target_voxel_size_um)

    import zarr

    root = zarr.open(args.zarr_path, mode="r")
    levels = []
    if not (hasattr(root, "shape") and hasattr(root, "dtype")):
        levels = _parse_ome_levels(root, args.channel_index)

    if levels:
        print("OME-Zarr multiscale levels:")
        for lv in levels:
            print(
                "  level=%s path=%s shape_zyx=%s voxel_um_zyx=%s"
                % (lv["level_index"], lv["path"], lv["shape_zyx"], lv["voxel_um_zyx"])
            )
        level = _choose_level(
            levels,
            args.fixed_level,
            target_voxel_um,
            args.level_strategy,
            args.min_level_z_slices,
        )
    else:
        if not (hasattr(root, "shape") and hasattr(root, "dtype")):
            raise ValueError("Path is neither an OME-Zarr root nor a direct Zarr array.")
        if len(root.shape) < 3:
            raise ValueError("Direct array must have at least 3 dimensions.")
        ndim = len(root.shape)
        level = {
            "level_index": 0,
            "path": "direct-array",
            "array": root,
            "axis_names": tuple(["axis%d" % i for i in range(ndim - 3)] + ["z", "y", "x"]),
            "zyx_axes": (ndim - 3, ndim - 2, ndim - 1),
            "spatial_permutation_to_zyx": (0, 1, 2),
            "shape_zyx": tuple(int(v) for v in root.shape[-3:]),
            "voxel_um_zyx": target_voxel_um,
        }

    shape_zyx = tuple(int(v) for v in level["shape_zyx"])
    source_voxel_um = tuple(float(v) for v in level["voxel_um_zyx"])

    source_steps = []
    for axis in range(3):
        if axis == 0 and args.z_sampling == "native":
            step = 1
        else:
            step = max(1, int(round(target_voxel_um[axis] / max(source_voxel_um[axis], 1e-12))))
        source_steps.append(step)
    source_steps = tuple(int(v) for v in source_steps)

    model_grid_shape = tuple(
        int((shape_zyx[i] + source_steps[i] - 1) // source_steps[i])
        for i in range(3)
    )
    effective_voxel_um = tuple(float(source_voxel_um[i] * source_steps[i]) for i in range(3))
    stride_zyx = (int(args.stride_z), int(args.stride_xy), int(args.stride_xy))

    z_starts = _covering_starts(model_grid_shape[0], patch_size[0], stride_zyx[0])
    y_starts = _covering_starts(model_grid_shape[1], patch_size[1], stride_zyx[1])
    x_starts = _covering_starts(model_grid_shape[2], patch_size[2], stride_zyx[2])
    all_windows = [(z, y, x) for z in z_starts for y in y_starts for x in x_starts]
    total_windows = len(all_windows)
    windows, selected_window_indices = _select_windows(
        all_windows,
        patch_size,
        model_grid_shape,
        args.max_patches,
        args.window_subset,
        args.seed,
    )

    print("\nSelected level:")
    print("  level:", level["level_index"], "path:", level["path"])
    print("  source shape zyx:", shape_zyx)
    print("  source voxel um zyx:", source_voxel_um)
    print("  source sampling step zyx:", source_steps)
    print("  effective model-grid voxel um zyx:", effective_voxel_um)
    print("  model grid shape zyx:", model_grid_shape)
    print("  model patch size zyx:", patch_size)
    print("  planned windows:", total_windows, "processing:", len(windows))
    print("  window subset:", args.window_subset)
    if selected_window_indices:
        print("  selected window indices:", selected_window_indices[:20])

    known_scale_notes = []
    if args.z_sampling == "native" and abs(effective_voxel_um[0] - target_voxel_um[0]) > 1e-6:
        known_scale_notes.append(
            "Z is kept at native spacing and padded to the model depth; this is an intentional thin-slab test, not a physical-scale match in Z."
        )
    if model_grid_shape[0] < patch_size[0]:
        known_scale_notes.append(
            "The available Z model-grid depth is smaller than the model patch depth, so Z padding will be used."
        )
    for note in known_scale_notes:
        print("  note:", note)

    wandb_config = {
        "zarr_path": args.zarr_path,
        "output_dir": str(out_dir),
        "selected_level_index": int(level["level_index"]),
        "selected_level_path": level["path"],
        "source_shape_zyx": shape_zyx,
        "source_voxel_um_zyx": source_voxel_um,
        "patch_size_zyx": patch_size,
        "target_voxel_size_um_zyx": target_voxel_um,
        "source_steps_zyx": source_steps,
        "effective_model_grid_voxel_um_zyx": effective_voxel_um,
        "model_grid_shape_zyx": model_grid_shape,
        "stride_zyx": stride_zyx,
        "total_planned_windows": int(total_windows),
        "processed_windows": int(len(windows)),
        "window_subset": args.window_subset,
        "selected_window_indices": selected_window_indices,
        "seed": int(args.seed),
        "z_sampling": args.z_sampling,
        "z_pad_position": args.z_pad_position,
        "level_strategy": args.level_strategy,
        "min_level_z_slices": int(args.min_level_z_slices),
        "fixed_level": args.fixed_level,
        "xy_downsample_mode": args.xy_downsample_mode,
        "offset_mean_samples": int(args.offset_mean_samples),
        "normalization": args.normalize,
        "input_gamma": float(args.input_gamma),
        "input_gain": float(args.input_gain),
        "model_checkpoint": args.model_checkpoint,
        "model_folder": args.model_folder,
        "threshold": float(args.threshold),
    }
    wandb = _try_start_wandb(args, wandb_config)
    if wandb is not None:
        wandb.log(
            {
                "data/source_z": shape_zyx[0],
                "data/source_y": shape_zyx[1],
                "data/source_x": shape_zyx[2],
                "data/model_grid_z": model_grid_shape[0],
                "data/model_grid_y": model_grid_shape[1],
                "data/model_grid_x": model_grid_shape[2],
                "run/total_planned_windows": int(total_windows),
                "run/processed_windows": int(len(windows)),
            }
        )

    import torch

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    checkpoint_paths = []
    if args.model_checkpoint:
        checkpoint_paths = [Path(args.model_checkpoint)]
    elif args.model_folder:
        checkpoint_paths = _find_checkpoints(args.model_folder, args.saved_model)
        if not checkpoint_paths:
            raise FileNotFoundError("No checkpoint found in %s" % args.model_folder)

    models = []
    model_load_reports = []
    if checkpoint_paths:
        for cp in checkpoint_paths:
            print("Loading model:", cp)
            model, load_report = _load_model(cp, device=device, pos_weight=args.pos_weight)
            models.append(model)
            model_load_reports.append(load_report)
        print("Loaded %d model(s) on %s" % (len(models), str(device)))
    else:
        print("No model checkpoint provided. This run will test loading, padding, normalization, and debug output only.")

    prediction_sum = None
    weight_sum = None
    if models:
        prediction_sum = np.zeros(model_grid_shape, dtype=np.float32)
        weight_sum = np.zeros(model_grid_shape, dtype=np.float32)
    raw_sum = None
    raw_weight_sum = None
    if not bool(args.no_save_raw_downsampled):
        raw_sum = np.zeros(model_grid_shape, dtype=np.float32)
        raw_weight_sum = np.zeros(model_grid_shape, dtype=np.float32)

    patch_reports = []

    for patch_idx, start_zyx in enumerate(windows):
        raw_patch, valid_shape, source_slices = _read_patch_zyx(
            level,
            start_zyx,
            patch_size,
            source_steps,
            args.channel_index,
            args.xy_downsample_mode,
            args.offset_mean_samples,
        )
        if raw_sum is not None:
            z0, y0, x0 = start_zyx
            z_len = min(int(raw_patch.shape[0]), int(model_grid_shape[0]) - int(z0))
            y_len = min(int(raw_patch.shape[1]), int(model_grid_shape[1]) - int(y0))
            x_len = min(int(raw_patch.shape[2]), int(model_grid_shape[2]) - int(x0))
            raw_sum[z0 : z0 + z_len, y0 : y0 + y_len, x0 : x0 + x_len] += raw_patch[
                :z_len, :y_len, :x_len
            ]
            raw_weight_sum[z0 : z0 + z_len, y0 : y0 + y_len, x0 : x0 + x_len] += 1.0

        # Normalise the *raw* patch first, then zero-pad. If we normalise after
        # padding, the ~half-volume of padded zeros contaminates the percentile
        # / min-max range and crushes the real signal (see thin-slab predict
        # debug where p99 of the model_input lands at ~0.05).
        raw_patch_norm, norm_meta = _normalize_patch(
            raw_patch,
            args.normalize,
            args.percentile_low,
            args.percentile_high,
            args.input_gamma,
            args.input_gain,
        )
        model_patch, insert_offsets, inserted_shape = _insert_with_padding(
            raw_patch_norm,
            patch_size,
            args.z_pad_position,
        )

        patch_meta = {
            "patch_index": int(patch_idx),
            "model_grid_start_zyx": tuple(int(v) for v in start_zyx),
            "source_slices_zyx": tuple((int(s.start), int(s.stop), int(s.step)) for s in source_slices),
            "raw_patch_shape_zyx": tuple(int(v) for v in raw_patch.shape),
            "valid_shape_zyx": tuple(int(v) for v in valid_shape),
            "model_insert_offsets_zyx": tuple(int(v) for v in insert_offsets),
            "model_inserted_shape_zyx": tuple(int(v) for v in inserted_shape),
            "raw_stats": _stats("raw_patch_zyx", raw_patch),
            "model_input_stats": _stats("model_input", model_patch),
            "normalization": norm_meta,
        }

        prob = None
        if models:
            patch_tensor = torch.from_numpy(model_patch).float().unsqueeze(0).unsqueeze(0).to(device)
            probs = []
            with torch.no_grad():
                for model in models:
                    logits = model(patch_tensor)
                    probs.append(torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy())
            prob = np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32, copy=False)

            z_off, y_off, x_off = insert_offsets
            z_len, y_len, x_len = inserted_shape
            prob_crop = prob[z_off : z_off + z_len, y_off : y_off + y_len, x_off : x_off + x_len]
            z0, y0, x0 = start_zyx
            prediction_sum[z0 : z0 + z_len, y0 : y0 + y_len, x0 : x0 + x_len] += prob_crop
            weight_sum[z0 : z0 + z_len, y0 : y0 + y_len, x0 : x0 + x_len] += 1.0
            patch_meta["probability_stats"] = _stats("probability_model_patch", prob)
            patch_meta["probability_crop_stats"] = _stats("probability_crop_real_slab", prob_crop)

        if patch_idx < int(args.save_debug_patches):
            stem = "patch_%04d" % patch_idx
            np.save(debug_dir / (stem + "_model_input.npy"), model_patch.astype(np.float32))
            pred_bin = None
            if prob is not None:
                np.save(debug_dir / (stem + "_probability.npy"), prob.astype(np.float32))
                pred_bin = (prob >= float(args.threshold)).astype(np.uint8)
                np.save(debug_dir / (stem + "_binary.npy"), pred_bin)
            debug_volumes = [("raw_sampled", raw_patch), ("model_input", model_patch)]
            if prob is not None:
                debug_volumes.append(("probability", prob))
            if pred_bin is not None:
                debug_volumes.append(("binary", pred_bin))
            debug_png_path = debug_dir / (stem + "_slices.png")
            _save_debug_png(debug_png_path, debug_volumes)
            debug_meta_path = debug_dir / (stem + "_meta.json")
            debug_meta_path.write_text(json.dumps(_jsonable(patch_meta), indent=2))
            if wandb is not None and debug_png_path.exists():
                wandb.log(
                    {
                        "debug/%s_slices" % stem: wandb.Image(str(debug_png_path)),
                        "patch/raw_mean": patch_meta["raw_stats"]["mean"],
                        "patch/raw_nonzero_fraction": patch_meta["raw_stats"]["nonzero_fraction"],
                        "patch/model_input_mean": patch_meta["model_input_stats"]["mean"],
                        "patch/model_input_nonzero_fraction": patch_meta["model_input_stats"]["nonzero_fraction"],
                    },
                    step=int(patch_idx),
                )

        patch_reports.append(patch_meta)
        if (patch_idx + 1) % 10 == 0 or patch_idx + 1 == len(windows):
            print("Processed %d/%d patches" % (patch_idx + 1, len(windows)))

    raw_summary = None
    if raw_sum is not None:
        raw_downsampled = raw_sum / np.maximum(raw_weight_sum, 1e-6)
        raw_downsampled = raw_downsampled.astype(np.float32, copy=False)
        np.save(out_dir / "raw_downsampled_zyx.npy", raw_downsampled)
        np.save(out_dir / "raw_weight_map_downsampled_zyx.npy", raw_weight_sum.astype(np.float32))
        raw_summary = {
            "raw_stats": _stats("raw_downsampled_zyx", raw_downsampled),
            "raw_processed_fraction": float(np.mean(raw_weight_sum > 0)),
        }

    prediction_summary = None
    if models:
        probability = prediction_sum / np.maximum(weight_sum, 1e-6)
        probability = np.clip(probability, 0.0, 1.0).astype(np.float32, copy=False)
        binary = (probability >= float(args.threshold)).astype(np.uint8)
        np.save(out_dir / "probability_downsampled_zyx.npy", probability)
        np.save(out_dir / "binary_downsampled_zyx.npy", binary)
        np.save(out_dir / "weight_map_downsampled_zyx.npy", weight_sum.astype(np.float32))
        prediction_summary = {
            "probability_stats": _stats("probability_downsampled_zyx", probability),
            "binary_stats": _stats("binary_downsampled_zyx", binary),
            "processed_fraction": float(np.mean(weight_sum > 0)),
            "threshold": float(args.threshold),
        }
        if bool(args.save_nifti):
            _save_optional_nifti(out_dir / "probability_downsampled_zyx.nii.gz", probability, effective_voxel_um)
            _save_optional_nifti(out_dir / "binary_downsampled_zyx.nii.gz", binary, effective_voxel_um)

    summary = {
        "zarr_path": args.zarr_path,
        "selected_level": {
            "level_index": int(level["level_index"]),
            "path": level["path"],
            "shape_zyx": shape_zyx,
            "voxel_um_zyx": source_voxel_um,
        },
        "patch_size_zyx": patch_size,
        "target_voxel_size_um_zyx": target_voxel_um,
        "target_patch_physical_size_um_zyx": tuple(float(patch_size[i] * target_voxel_um[i]) for i in range(3)),
        "z_sampling": args.z_sampling,
        "z_pad_position": args.z_pad_position,
        "level_strategy": args.level_strategy,
        "min_level_z_slices": int(args.min_level_z_slices),
        "fixed_level": args.fixed_level,
        "xy_downsample_mode": args.xy_downsample_mode,
        "offset_mean_samples": int(args.offset_mean_samples),
        "source_steps_zyx": source_steps,
        "effective_model_grid_voxel_um_zyx": effective_voxel_um,
        "model_grid_shape_zyx": model_grid_shape,
        "stride_zyx": stride_zyx,
        "total_planned_windows": int(total_windows),
        "processed_windows": int(len(windows)),
        "window_subset": args.window_subset,
        "selected_window_indices": selected_window_indices,
        "seed": int(args.seed),
        "normalization": {
            "method": args.normalize,
            "percentile_low": float(args.percentile_low),
            "percentile_high": float(args.percentile_high),
            "input_gamma": float(args.input_gamma),
            "input_gain": float(args.input_gain),
        },
        "known_scale_notes": known_scale_notes,
        "models": model_load_reports,
        "raw_summary": raw_summary,
        "prediction_summary": prediction_summary,
        "patch_reports": patch_reports,
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    if wandb is not None:
        log_payload = {
            "run/processed_windows_final": int(len(windows)),
            "run/summary_written": 1,
        }
        if prediction_summary is not None:
            log_payload.update(
                {
                    "prediction/probability_mean": prediction_summary["probability_stats"]["mean"],
                    "prediction/probability_p99": prediction_summary["probability_stats"]["p99"],
                    "prediction/binary_fraction": prediction_summary["binary_stats"]["mean"],
                    "prediction/processed_fraction": prediction_summary["processed_fraction"],
                }
            )
        wandb.log(log_payload)
        try:
            artifact = wandb.Artifact(args.wandb_run_name + "_outputs", type="prediction")
            artifact.add_file(str(out_dir / "summary.json"))
            for path in sorted(debug_dir.glob("*")):
                if path.is_file():
                    artifact.add_file(str(path))
            for name in (
                "raw_downsampled_zyx.npy",
                "raw_weight_map_downsampled_zyx.npy",
                "probability_downsampled_zyx.npy",
                "binary_downsampled_zyx.npy",
                "weight_map_downsampled_zyx.npy",
                "probability_downsampled_zyx.nii.gz",
                "binary_downsampled_zyx.nii.gz",
            ):
                path = out_dir / name
                if path.exists():
                    artifact.add_file(str(path))
            wandb.log_artifact(artifact)
        except Exception as exc:
            print("W&B artifact logging failed:", exc)
        wandb.finish()

    print("\nSaved:")
    print(" ", out_dir / "summary.json")
    print(" ", debug_dir)
    if raw_sum is not None:
        print(" ", out_dir / "raw_downsampled_zyx.npy")
    if models:
        print(" ", out_dir / "probability_downsampled_zyx.npy")
        print(" ", out_dir / "binary_downsampled_zyx.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
