"""Break the texture-gap vs thin-slab-geometry confound.

sanity_check_synthetic.py scored dice ~0.98 on a FULL synthetic patch. Every
real-LSM failure is a THIN-SLAB patch (real tissue ~60 Z slices at offset ~34,
padded to 128). So "synthetic texture works / real texture fails" is confounded
with "full volume works / thin-slab fails" — they were never separated.

This takes the SAME synthetic patch that scored 0.98 and applies the real-LSM
thin-slab geometry (zero outside Z=[z0, z0+zext)), then runs the identical
forward pass. It also mirrors test_specific_region.py's normalization ORDER:
normalize on the real (slab) voxels, THEN pad — NOT normalize the padded full
volume. (Training's _apply_inference_shape_augs normalizes the full volume then
zeros the slab — a different order over a different voxel population; if dice
collapses here, that order mismatch is the prime suspect.)

  dice stays high  -> geometry is fine -> genuine synthetic->real TEXTURE gap.
  dice collapses   -> thin-slab handling is the bug -> cheap fix, leave texture.

Run on the CLUSTER (GPU; full 128^3 forward pass OOMs a laptop):
    python3 sanity_check_thinslab.py \
        --checkpoint synthetic-training/checkpoints_cached_bf16/best_3d-epoch=129-val_loss=0.0491.ckpt \
        --volume synthetic-training/precomputed_patches/aligned_wavy/patch_0001_3d.nii.gz \
        --mask   synthetic-training/precomputed_patches/aligned_wavy/patch_0001_3d_mask.nii.gz \
        --z_offset 34 --z_extent 60
"""
import argparse
import numpy as np
import nibabel as nib
import torch


def norm_1_99(vol):
    lo, hi = np.percentile(vol, [1.0, 99.0])
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32) if hi > lo else vol * 0


def run(model, vol_n, device):
    x = torch.from_numpy(vol_n).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy().squeeze()


def dice_at(prob, m, t):
    pred = prob > t
    tp = float((pred & m).sum()); fp = float((pred & ~m).sum()); fn = float((~pred & m).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-6), 100 * float(pred.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--z_offset", type=int, default=34, help="first real-tissue Z slice (match real LSM)")
    ap.add_argument("--z_extent", type=int, default=60, help="number of real-tissue Z slices")
    ap.add_argument("--pos_weight", type=float, default=1.0)
    args = ap.parse_args()

    try:
        from unet3d import FlexibleUNet3D
    except ImportError:
        from synthetic_training.unet3d import FlexibleUNet3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    hp = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    allowed = ["min_features", "max_features", "num_stages", "loss",
               "freeze_encoder", "pos_weight", "in_channels"]
    kwargs = {k: hp[k] for k in allowed if k in hp}
    kwargs.setdefault("pos_weight", args.pos_weight)
    model = FlexibleUNet3D(**kwargs)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    vol = nib.load(args.volume).get_fdata().astype(np.float32)
    mask = (nib.load(args.mask).get_fdata().astype(np.float32) > 0.5)
    D = vol.shape[0]
    z0, zext = args.z_offset, args.z_extent
    z1 = min(D, z0 + zext)

    # --- Baseline: FULL volume, normalize whole (matches sanity_check_synthetic) ---
    prob_full = run(model, norm_1_99(vol), device)
    d_full, c_full = dice_at(prob_full, mask, 0.5)
    print(f"[FULL]            dice@0.5={d_full:.4f} pred_cover={c_full:.4f}%")

    # --- Thin-slab, real-inference ORDER: normalize the slab, THEN pad ---
    slab = vol[z0:z1]
    slab_n = norm_1_99(slab)
    vol_ts = np.zeros_like(vol)
    vol_ts[z0:z1] = slab_n
    prob_ts = run(model, vol_ts, device)
    # restrict dice to the real slab (where a mask can exist)
    m_slab = mask.copy(); m_slab[:z0] = False; m_slab[z1:] = False
    d_ts, c_ts = dice_at(prob_ts, m_slab, 0.5)
    print(f"[THINSLAB norm->pad] dice@0.5={d_ts:.4f} pred_cover={c_ts:.4f}%  "
          f"(real tissue Z=[{z0},{z1}), mask voxels in slab={int(m_slab.sum())})")

    # --- Thin-slab, WRONG order (normalize full then zero) for contrast ---
    vol_full_n = norm_1_99(vol).copy()
    vol_full_n[:z0] = 0; vol_full_n[z1:] = 0
    prob_ts2 = run(model, vol_full_n, device)
    d_ts2, _ = dice_at(prob_ts2, m_slab, 0.5)
    print(f"[THINSLAB norm-full-then-zero] dice@0.5={d_ts2:.4f}")

    print("\nVERDICT:")
    print("  FULL high + THINSLAB high  -> geometry fine -> texture domain gap (augmentation work).")
    print("  FULL high + THINSLAB ~0    -> thin-slab handling is the bug (cheap fix; leave texture).")
    print("  norm->pad vs norm-full differ a lot -> normalization-order is the train/infer discrepancy.")


if __name__ == "__main__":
    raise SystemExit(main())
