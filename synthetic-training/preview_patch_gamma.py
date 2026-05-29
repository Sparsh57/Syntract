"""Create a contact sheet comparing input_gamma/input_gain settings.

This is for choosing real-data model-input brightness from an already saved
debug patch, usually:

    new_test_thinslab_e2e/debug_patches/patch_0000_model_input.npy

The input patch should already be normalized to [0, 1]. The script applies
candidate gamma/gain settings and saves a PNG with axial/coronal/sagittal views.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _parse_setting(text):
    parts = str(text).split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("setting must be gamma,gain")
    return float(parts[0]), float(parts[1])


def _stats(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "no finite"
    p = np.percentile(finite, [50, 95, 99, 99.9])
    return "mean %.4f p50 %.4f p95 %.4f p99 %.4f p99.9 %.4f max %.4f" % (
        float(finite.mean()),
        float(p[0]),
        float(p[1]),
        float(p[2]),
        float(p[3]),
        float(finite.max()),
    )


def _best_indices(vol):
    z = int(np.argmax(vol.sum(axis=(1, 2)))) if vol.shape[0] else 0
    y = int(np.argmax(vol.sum(axis=(0, 2)))) if vol.shape[1] else 0
    x = int(np.argmax(vol.sum(axis=(0, 1)))) if vol.shape[2] else 0
    return z, y, x


def _to_u8(arr, vmax):
    arr = np.asarray(arr, dtype=np.float32)
    out = np.clip(arr / max(float(vmax), 1e-8), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _resize_tile(arr2d, tile, vmax):
    img = Image.fromarray(_to_u8(arr2d, vmax), mode="L").convert("RGB")
    h, w = arr2d.shape[:2]
    scale = min(float(tile) / max(1, w), float(tile) / max(1, h))
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    canvas = Image.new("RGB", (tile, tile), "black")
    canvas.paste(img, ((tile - img.width) // 2, (tile - img.height) // 2))
    return canvas


def main():
    parser = argparse.ArgumentParser("Preview gamma/gain settings for a saved model-input patch")
    parser.add_argument("--patch", required=True, help="Path to patch_XXXX_model_input.npy")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--settings",
        nargs="+",
        type=_parse_setting,
        default=[
            (1.0, 1.0),
            (1.3, 0.9),
            (1.6, 0.85),
            (1.8, 0.8),
            (2.2, 0.7),
            (2.6, 0.65),
        ],
        help="Gamma/gain pairs, for example: 1.8,0.8 2.2,0.7",
    )
    parser.add_argument(
        "--display_vmax",
        type=float,
        default=0.08,
        help="Fixed display max for the PNG. This does not change model input.",
    )
    args = parser.parse_args()

    patch_path = Path(args.patch)
    vol = np.load(str(patch_path)).astype(np.float32, copy=False)
    if vol.ndim != 3:
        raise ValueError("Expected 3D patch, got shape %s" % (vol.shape,))
    vol = np.clip(vol, 0.0, 1.0)

    out_path = Path(args.output) if args.output else patch_path.with_name(
        patch_path.stem.replace("_model_input", "") + "_gamma_preview.png"
    )

    tile = 192
    label_w = 300
    title_h = 42
    row_h = title_h + tile
    sheet = Image.new("RGB", (label_w + 3 * tile, row_h * len(args.settings)), "white")
    draw = ImageDraw.Draw(sheet)

    for row, (gamma, gain) in enumerate(args.settings):
        transformed = np.clip(np.power(vol, float(gamma)) * float(gain), 0.0, 1.0)
        z, y, x = _best_indices(transformed)
        y0 = row * row_h
        title = "gamma %.2f  gain %.2f\n%s" % (gamma, gain, _stats(transformed))
        draw.multiline_text((8, y0 + 8), title, fill=(0, 0, 0), spacing=3)

        cells = [
            ("axial z=%d" % z, transformed[z]),
            ("coronal y=%d" % y, transformed[:, y, :]),
            ("sagittal x=%d" % x, transformed[:, :, x]),
        ]
        for col, (name, arr2d) in enumerate(cells):
            x0 = label_w + col * tile
            draw.text((x0 + 5, y0 + 8), name, fill=(0, 0, 0))
            sheet.paste(_resize_tile(arr2d, tile, args.display_vmax), (x0, y0 + title_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
