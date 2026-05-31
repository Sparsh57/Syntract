"""Tier-1 sanity check: does the trained model reproduce a KNOWN synthetic mask?

This is the discriminating test for "domain gap vs broken inference path vs
overfit". It loads a synthetic patch the model was trained on (vol + mask),
normalizes EXACTLY as training does (1-99 percentile over the full volume,
matching datasets.py), runs a single forward pass, and reports dice/coverage
against the known mask.

  Model trained to ~0.95 dice on this exact data, so:
    - high dice here  -> inference forward path is correct; flat-zero on real
                         data is a genuine domain/generalization gap.
    - low dice here   -> the bug is in the load/normalize/forward path, NOT the
                         data domain. Fix the path before any retraining.

Run on the cluster (needs the checkpoint + torch):
    python3 sanity_check_synthetic.py \
        --checkpoint synthetic-training/checkpoints_cached_bf16/best_3d-epoch=129-val_loss=0.0491.ckpt \
        --volume Model_prediction/patch_0001_3d.nii.gz \
        --mask   Model_prediction/patch_0001_3d_mask.nii.gz
"""
import argparse
import numpy as np
import nibabel as nib
import torch


def normalize_like_training(vol: np.ndarray) -> np.ndarray:
    """1-99 percentile over the FULL volume, then clip to [0,1] (datasets.py:806)."""
    v_lo, v_hi = np.percentile(vol, [1.0, 99.0])
    if v_hi > v_lo:
        return np.clip((vol - v_lo) / (v_hi - v_lo), 0.0, 1.0).astype(np.float32)
    return np.zeros_like(vol, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
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
    print(f"Loaded model with hparams: {kwargs}")

    vol = nib.load(args.volume).get_fdata().astype(np.float32)
    mask = nib.load(args.mask).get_fdata().astype(np.float32)
    vol_n = normalize_like_training(vol)
    print(f"vol raw[min={vol.min():.2f} max={vol.max():.2f}]  "
          f"norm[min={vol_n.min():.3f} max={vol_n.max():.3f}]  "
          f"mask coverage={100*(mask>0.5).mean():.4f}%")

    x = torch.from_numpy(vol_n).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).cpu().numpy().squeeze()

    m = (mask > 0.5)
    for t in (0.1, 0.3, 0.5):
        pred = prob > t
        tp = float((pred & m).sum()); fp = float((pred & ~m).sum()); fn = float((~pred & m).sum())
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        print(f"  thr={t}: pred_cover={100*pred.mean():.4f}%  dice={dice:.4f}")
    print(f"prob: median={np.median(prob):.4f} p99={np.percentile(prob,99):.4f} "
          f"max={prob.max():.4f}  prob@fiber_mean={prob[m].mean() if m.any() else 0:.4f} "
          f"prob@tissue_mean={prob[~m].mean():.4f}")
    print("\nVERDICT: dice>0.5 => inference path OK, real failure is domain gap. "
          "dice~0 => inference path is broken (check normalize/axis/scale), NOT domain.")


if __name__ == "__main__":
    raise SystemExit(main())
