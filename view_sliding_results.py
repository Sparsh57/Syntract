#!/usr/bin/env python3
"""Two-panel slice viewer: tissue image (left) | binary mask (right).

Memory-mapped: the .npy files stay on disk and only the CURRENT slice is read,
so it handles the huge full-region outputs (100+ GB) without loading them into
RAM and WITHOUT downloading them. Run it on the cluster with the display
forwarded to your laptop (ssh -X). Scroll the slider or use arrow keys.

    ssh -X sparsh@<login-host>
    cd ~/syntract-3d && source venv/bin/activate
    python view_sliding_results.py --prefix /orcd/.../sliding_infer_out_slice037/region1
    python view_sliding_results.py --prefix .../region1 --axis y
    python view_sliding_results.py --prefix .../region1 --downsample 1   # full-res panel
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True, help="e.g. .../region1 (expects _image.npy, _binary.npy)")
ap.add_argument("--axis", choices=("z", "y", "x"), default="z")
ap.add_argument("--slice", type=int, default=None, help="Starting slice index")
ap.add_argument("--downsample", type=int, default=0,
                help="Display decimation (0 = auto so the panel is ~1500 px; use 1 for full res)")
args = ap.parse_args()

# mmap_mode='r' => arrays stay on disk; slicing reads only what's shown. No download, low RAM.
image = np.load(f"{args.prefix}_image.npy", mmap_mode="r")     # (Z, Y, X)
binary = np.load(f"{args.prefix}_binary.npy", mmap_mode="r")   # (Z, Y, X)
print(f"image={image.shape} {image.dtype}   binary={binary.shape} {binary.dtype}  (memory-mapped)")

AXIS = {"z": 0, "y": 1, "x": 2}[args.axis]
n_slices = image.shape[AXIS]
start = args.slice if args.slice is not None else n_slices // 2

# Auto display-downsample so a 6144-px slice doesn't choke matplotlib over X11.
plane = [s for i, s in enumerate(image.shape) if i != AXIS]
ds = args.downsample if args.downsample > 0 else max(1, max(plane) // 1500)
print(f"display downsample = {ds}x  (in-plane {plane} -> ~{[p // ds for p in plane]})")


def get_slice(vol, idx):
    if AXIS == 0:
        sl = vol[idx]
    elif AXIS == 1:
        sl = vol[:, idx, :]
    else:
        sl = vol[:, :, idx]
    sl = np.asarray(sl)                      # materialise just this one slice
    return sl[::ds, ::ds] if ds > 1 else sl


fig, (ax_img, ax_bin) = plt.subplots(1, 2, figsize=(14, 7))
fig.subplots_adjust(bottom=0.12)
fig.suptitle(f"{args.prefix}  |  axis={args.axis}  |  slider / arrow keys", fontsize=10)

im_left = ax_img.imshow(get_slice(image, start), cmap="gray", vmin=0, vmax=1,
                        aspect="auto", interpolation="nearest")
im_right = ax_bin.imshow(get_slice(binary, start), cmap="gray", vmin=0, vmax=1,
                         aspect="auto", interpolation="nearest")
ax_img.set_title("Tissue (image)", fontsize=9); ax_img.axis("off")
ax_bin.set_title("Binary mask", fontsize=9); ax_bin.axis("off")

ax_slider = fig.add_axes([0.15, 0.04, 0.7, 0.03])
slider = Slider(ax_slider, f"{args.axis.upper()} slice", 0, n_slices - 1,
                valinit=start, valstep=1)


def update(val):
    idx = int(slider.val)
    im_left.set_data(get_slice(image, idx))
    im_right.set_data(get_slice(binary, idx))
    ax_img.set_title(f"Tissue (image)  [{args.axis}={idx}]", fontsize=9)
    ax_bin.set_title(f"Binary mask  [{args.axis}={idx}]", fontsize=9)
    fig.canvas.draw_idle()


slider.on_changed(update)


def on_key(event):
    if event.key in ("right", "up"):
        slider.set_val(min(slider.val + 1, n_slices - 1))
    elif event.key in ("left", "down"):
        slider.set_val(max(slider.val - 1, 0))


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
