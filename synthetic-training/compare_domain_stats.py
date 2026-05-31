"""Quantify the synthetic->real domain gap so we tune the right augmentation knob.

The sanity check proved the model + inference path are correct (dice 0.98 on
synthetic) but produce nothing on real LSM -> a domain gap. This script compares
intensity + texture statistics of synthetic training patches vs real LSM input
patches, normalized IDENTICALLY (training's 1-99 percentile), and prints a
side-by-side so the specific mismatches are explicit (not inferred from renders).

Runs locally on the .npy / .nii.gz you already have (pure numpy, no model):
    python3 compare_domain_stats.py \
        --real_dir ../Model_prediction \
        --synth_dir ../Model_prediction
(real = *_model_input.npy, synth = *_3d.nii.gz in those dirs)
"""
import argparse
import glob
import os
import numpy as np


def norm_train(vol):
    lo, hi = np.percentile(vol, [1.0, 99.0])
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32) if hi > lo else vol * 0


def tissue_band(vol):
    """Restrict to the non-zero Z band (real patches are zero-padded slabs)."""
    znz = np.where((vol > 0).any(axis=(1, 2)))[0]
    return vol[znz.min():znz.max() + 1] if len(znz) else vol


def stats(vol):
    v = vol.ravel()
    nz = v[v > 0]
    # large-scale gradient: std of a heavily blurred version vs fine detail
    from scipy.ndimage import uniform_filter
    coarse = uniform_filter(vol, size=16)
    fine = vol - coarse
    # adjacent-voxel decorrelation (texture grain scale) along X
    a = vol[vol.shape[0] // 2]
    a = a - a.mean()
    base = (a * a).sum()
    corr1 = float((a[:, :-1] * a[:, 1:]).sum() / base) if base > 0 else 0.0
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "nonzero_%": 100 * float((v > 0).mean()),
        "p50": float(np.percentile(nz, 50)) if nz.size else 0.0,
        "p99": float(np.percentile(nz, 99)) if nz.size else 0.0,
        "coarse_std(gradient)": float(coarse.std()),
        "fine_std(grain)": float(fine.std()),
        "grain/gradient_ratio": float(fine.std() / (coarse.std() + 1e-9)),
        "adj_voxel_corr": corr1,
    }


def load_any(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    import nibabel as nib
    return nib.load(path).get_fdata().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--synth_dir", required=True)
    args = ap.parse_args()

    real = sorted(glob.glob(os.path.join(args.real_dir, "*_model_input.npy")))
    synth = sorted(glob.glob(os.path.join(args.synth_dir, "*_3d.nii.gz")))
    synth = [s for s in synth if "_mask" not in s]
    if not real or not synth:
        raise SystemExit(f"need real (*_model_input.npy) and synth (*_3d.nii.gz); "
                         f"found {len(real)} real, {len(synth)} synth")

    def agg(paths, is_real):
        rows = []
        for p in paths:
            v = load_any(p)
            v = tissue_band(v)
            v = norm_train(v)  # identical normalization both sides
            rows.append(stats(v))
        keys = rows[0].keys()
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    rs = agg(real, True)
    ss = agg(synth, False)

    print(f"{'metric':24s} {'REAL(LSM)':>14s} {'SYNTH(train)':>14s}  note")
    print("-" * 72)
    notes = {
        "coarse_std(gradient)": "synth high => fake background gradient/banding",
        "grain/gradient_ratio": "real high => uniform granular; synth low => smooth+gradient",
        "adj_voxel_corr": "higher => smoother/blurrier texture",
        "std": "overall contrast",
    }
    for k in rs:
        note = notes.get(k, "")
        print(f"{k:24s} {rs[k]:14.4f} {ss[k]:14.4f}  {note}")
    print("\nInterpretation:")
    print("  - If SYNTH coarse_std >> REAL: your synthetic background has a large-scale")
    print("    gradient/banding the real data lacks -> reduce/disable banding, flatten bg.")
    print("  - If REAL grain/gradient_ratio >> SYNTH: real is uniform fine grain; make")
    print("    synthetic tissue flatter with finer granular noise (cornucopia presets).")
    print("  - Match std (contrast) and the grain scale (adj_voxel_corr) between columns.")


if __name__ == "__main__":
    raise SystemExit(main())
