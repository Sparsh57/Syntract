#!/usr/bin/env python3
"""Interactive zarr region picker.

Loads a coarse pyramid level of an OME-Zarr, shows a Z-MIP, and lets you
click to choose a center. Prints level-0 (full-res) coordinates ready to
paste into sliding_window_infer.sh.

Run on the cluster (needs display or X11):
    python explore_zarr_pick_region.py \
        --zarr /orcd/data/linc/001/lsm_test_data_sparsh/LSM_test_data/2025_09_09_MonkeySlice_561channel_561laser_Stitched.ome.zarr

Or save a PNG overview without a display:
    python explore_zarr_pick_region.py --zarr <path> --save_png zarr_overview.png --no_gui
"""
import argparse
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--zarr", required=True)
ap.add_argument("--coarse_level", type=int, default=3,
                help="Pyramid level to load for overview (3 or 4 is usually small enough)")
ap.add_argument("--channel_index", type=int, default=0)
ap.add_argument("--save_png", default="zarr_overview.png")
ap.add_argument("--no_gui", action="store_true", help="Save PNG only, no interactive window")
ap.add_argument("--region_size_zyx", nargs=3, type=int, default=[256, 512, 512],
                metavar=("Z","Y","X"), help="Extraction size shown in the readout")
args = ap.parse_args()

import zarr

store = zarr.open(args.zarr, mode="r")

# Find coarse level array
arr = None
for key in (str(args.coarse_level), args.coarse_level,
            str(args.coarse_level - 1), args.coarse_level - 1):
    try:
        arr = store[key]; break
    except (KeyError, TypeError):
        pass
if arr is None:
    arr = store
    print("WARNING: could not find pyramid level, using root array")

shape = arr.shape
print(f"Coarse array shape: {shape}  (level {args.coarse_level})")

# Read full coarse volume (should be small at level 3-4)
ndim = arr.ndim
if ndim == 3:
    vol = np.asarray(arr[:]).astype(np.float32)
elif ndim == 4 and shape[0] <= 16:   # (C, Z, Y, X)
    vol = np.asarray(arr[args.channel_index]).astype(np.float32)
elif ndim == 4:                       # (Z, Y, X, C)
    vol = np.asarray(arr[..., args.channel_index]).astype(np.float32)
else:
    print(f"Unexpected shape {shape}, trying first 3 dims")
    vol = np.asarray(arr[0]).astype(np.float32)

print(f"Loaded coarse volume: {vol.shape}  min={vol.min():.1f}  max={vol.max():.1f}")

# Work out scale factors from coarse → level 0
level0_arr = None
for key in ("0", 0):
    try: level0_arr = store[key]; break
    except (KeyError, TypeError): pass
if level0_arr is not None:
    l0_shape = level0_arr.shape
    scale_z = l0_shape[-3] / vol.shape[-3]
    scale_y = l0_shape[-2] / vol.shape[-2]
    scale_x = l0_shape[-1] / vol.shape[-1]
    print(f"Level-0 shape: {l0_shape}  scale factors z={scale_z:.1f} y={scale_y:.1f} x={scale_x:.1f}")
else:
    scale_z = scale_y = scale_x = 2 ** args.coarse_level
    print(f"Could not read level-0 shape; assuming 2^{args.coarse_level}={2**args.coarse_level}x scale")

# Z-MIP of coarse volume, percentile-normalised
mip = vol.max(axis=0)
lo, hi = np.percentile(mip, [1, 99])
mip_norm = np.clip((mip - lo) / max(hi - lo, 1e-6), 0, 1)

import matplotlib
if args.no_gui:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(mip_norm, cmap="gray", vmin=0, vmax=1, aspect="auto")
ax.set_title(
    f"Z-MIP  coarse level {args.coarse_level}  |  CLICK to pick region center\n"
    f"Coarse shape (Y,X)=({vol.shape[-2]},{vol.shape[-1]})  "
    f"→ level-0 scale x{scale_y:.0f}y / x{scale_x:.0f}x",
    fontsize=9,
)

# Overlay a grid every 64 coarse pixels to help estimate positions
for gx in range(0, vol.shape[-1], 64):
    ax.axvline(gx, color='cyan', lw=0.3, alpha=0.4)
for gy in range(0, vol.shape[-2], 64):
    ax.axhline(gy, color='cyan', lw=0.3, alpha=0.4)

clicks = []

def on_click(event):
    if event.inaxes != ax or event.button != 1:
        return
    cy_coarse = event.ydata
    cx_coarse = event.xdata
    cz_coarse = vol.shape[0] // 2   # mid-Z of coarse

    # Map to level-0 voxels
    cz0 = int(round(cz_coarse * scale_z))
    cy0 = int(round(cy_coarse * scale_y))
    cx0 = int(round(cx_coarse * scale_x))

    n = len(clicks) + 1
    clicks.append((cz0, cy0, cx0))

    ax.plot(cx_coarse, cy_coarse, 'r+', markersize=12, markeredgewidth=2)
    ax.text(cx_coarse + 2, cy_coarse - 2, str(n), color='red', fontsize=8)
    fig.canvas.draw_idle()

    rz, ry, rx = args.region_size_zyx
    print(f"\n[Click {n}]  coarse ({cy_coarse:.0f}, {cx_coarse:.0f})")
    print(f"  Level-0 center (Z Y X): {cz0} {cy0} {cx0}")
    print(f"  → paste into SLURM script:")
    print(f'    REGION_{n}_CENTER="{cz0} {cy0} {cx0}"')
    print(f'    REGION_SIZE="{rz} {ry} {rx}"')

if not args.no_gui:
    fig.canvas.mpl_connect("button_press_event", on_click)
    print("\nClick anywhere on the image to pick a region center.")
    print("Close the window when done.\n")

plt.tight_layout()
fig.savefig(args.save_png, dpi=120, bbox_inches="tight")
print(f"Overview saved: {args.save_png}")

if not args.no_gui:
    plt.show()
