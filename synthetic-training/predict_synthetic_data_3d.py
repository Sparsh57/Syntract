"""
3D volumetric prediction script for fiber segmentation.

Loads a trained FlexibleUNet3D model and performs sliding-window inference
on a 3D NIfTI volume with Gaussian importance weighting and test-time mirroring.

Usage:
    python predict_synthetic_data_3d.py \
        --input_nifti /path/to/volume.nii.gz \
        --output_nifti /path/to/prediction.nii.gz \
        --model_folder /path/to/checkpoints/
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import gaussian_filter, label

from unet3d import FlexibleUNet3D


def get_gaussian_importance_map_3d(patch_size: Tuple[int, ...], sigma_scale: float = 1/8) -> np.ndarray:
    """Creates a 3D Gaussian importance map for patch overlap blending."""
    tmp = np.zeros(patch_size)
    center = tuple(s // 2 for s in patch_size)
    tmp[center] = 1
    sigmas = [s * sigma_scale for s in patch_size]
    gaussian_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_map = gaussian_map / np.max(gaussian_map)
    return gaussian_map


def predict_with_mirroring_3d(model, patch_tensor, device):
    """
    Predict with test-time mirroring augmentation (nnUNet style) for 3D.
    Mirrors along all combinations of D, H, W axes.
    """
    pred = torch.sigmoid(model(patch_tensor))

    # All axis combinations for 3D: dims are (batch, channel, D, H, W) → spatial dims 2,3,4
    mirror_axes = [
        (2,), (3,), (4,),
        (2, 3), (2, 4), (3, 4),
        (2, 3, 4),
    ]
    n_total = len(mirror_axes) + 1  # +1 for original

    for axes in mirror_axes:
        mirrored = torch.flip(patch_tensor, dims=axes)
        pred_mirrored = torch.sigmoid(model(mirrored))
        pred_mirrored = torch.flip(pred_mirrored, dims=axes)
        pred += pred_mirrored

    return pred / n_total


def predict_volume(
    nifti_path: str,
    models: List[torch.nn.Module],
    device: torch.device,
    patch_size: Tuple[int, int, int] = (256, 256, 256),
    stride_ratio: float = 0.5,
    confidence_threshold: float = 0.5,
    min_size: int = 100,
    use_mirroring: bool = True,
) -> np.ndarray:
    """
    Predict a 3D volume using ensemble of models with sliding window.
    """
    # Load NIfTI
    nii_img = nib.load(nifti_path)
    nii_img = nib.as_closest_canonical(nii_img)
    volume = nii_img.get_fdata().astype(np.float32)

    # Normalize to [0, 1]
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)

    D, H, W = volume.shape
    print(f"Volume shape: {D} x {H} x {W}")

    # Initialize prediction and weight maps
    prediction_map = np.zeros((D, H, W), dtype=np.float32)
    weight_map = np.zeros((D, H, W), dtype=np.float32)

    # Gaussian importance map
    gaussian_map = get_gaussian_importance_map_3d(patch_size)

    # Stride
    stride = tuple(int(ps * stride_ratio) for ps in patch_size)
    pd, ph, pw = patch_size

    print(f"Patch size: {patch_size}, stride: {stride}")

    for model_idx, model in enumerate(models):
        print(f"Running prediction with model {model_idx + 1}/{len(models)}")
        model.eval()
        with torch.no_grad():
            for d_start in range(0, max(1, D - pd + stride[0]), stride[0]):
                for h_start in range(0, max(1, H - ph + stride[1]), stride[1]):
                    for w_start in range(0, max(1, W - pw + stride[2]), stride[2]):
                        # Handle border cases
                        d_end = min(d_start + pd, D)
                        h_end = min(h_start + ph, H)
                        w_end = min(w_start + pw, W)
                        d_s = max(0, d_end - pd)
                        h_s = max(0, h_end - ph)
                        w_s = max(0, w_end - pw)

                        patch = volume[d_s:d_end, h_s:h_end, w_s:w_end]

                        # Pad if necessary
                        if patch.shape != patch_size:
                            temp = np.zeros(patch_size, dtype=np.float32)
                            temp[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                            patch = temp

                        # (1, 1, D, H, W)
                        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)

                        if use_mirroring:
                            pred = predict_with_mirroring_3d(model, patch_tensor, device)
                            pred = pred.cpu().numpy().squeeze()
                        else:
                            pred = model(patch_tensor)
                            pred = torch.sigmoid(pred).cpu().numpy().squeeze()

                        # Crop Gaussian map to actual patch region
                        actual_d = d_end - d_s
                        actual_h = h_end - h_s
                        actual_w = w_end - w_s
                        g = gaussian_map[:actual_d, :actual_h, :actual_w]
                        p = pred[:actual_d, :actual_h, :actual_w]

                        prediction_map[d_s:d_end, h_s:h_end, w_s:w_end] += p * g
                        weight_map[d_s:d_end, h_s:h_end, w_s:w_end] += g

    # Average
    prediction_map = prediction_map / np.maximum(weight_map, 1e-7)
    prediction_map = np.clip(prediction_map, 0, 1)

    # Threshold
    binary_prediction = prediction_map > confidence_threshold

    # Remove small components
    if min_size is not None and min_size > 0:
        labels, num_features = label(binary_prediction)
        if num_features > 0:
            component_sizes = np.bincount(labels.ravel())[1:]
            too_small = component_sizes < min_size
            small_labels = np.where(too_small)[0] + 1
            mask = np.isin(labels, small_labels)
            labels[mask] = 0
            binary_prediction = labels > 0

    return binary_prediction.astype(np.uint8), nii_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_nifti', type=str, required=True,
                        help='Input 3D NIfTI volume to segment')
    parser.add_argument('--output_nifti', type=str, required=True,
                        help='Output NIfTI segmentation mask')
    parser.add_argument('--model_folder', type=str, required=True,
                        help='Folder containing model checkpoints')
    parser.add_argument('--saved_model', type=str, default='best',
                        choices=['best', 'last'])
    parser.add_argument('--patch_size', nargs=3, type=int, default=[256, 256, 256],
                        help='Patch size D H W')
    parser.add_argument('--stride_ratio', type=float, default=0.5,
                        help='Stride as fraction of patch size (lower = more overlap)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--min_size', type=int, default=100,
                        help='Minimum connected component size in voxels')
    parser.add_argument('--use_mirroring', action='store_true', default=True,
                        help='Enable test-time mirroring augmentation')
    parser.add_argument('--no_mirroring', dest='use_mirroring', action='store_false')
    parser.add_argument('--pos_weight', type=float, default=1.0)

    args = parser.parse_args()

    print(f"Start time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model(s)
    if args.saved_model == "last":
        pattern = "last.ckpt"
    else:
        pattern = "best_3d*.ckpt"

    checkpoint_paths = sorted(Path(args.model_folder).glob(pattern))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint found matching '{pattern}' in {args.model_folder}")

    models = []
    for cp in checkpoint_paths:
        print(f"Loading model from {cp}")
        model = FlexibleUNet3D(learning_rate=1e-4, pos_weight=args.pos_weight)
        ckpt = torch.load(cp, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model = model.to(device)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} model(s)")

    # Predict
    prediction, nii_ref = predict_volume(
        nifti_path=args.input_nifti,
        models=models,
        device=device,
        patch_size=tuple(args.patch_size),
        stride_ratio=args.stride_ratio,
        confidence_threshold=args.confidence_threshold,
        min_size=args.min_size,
        use_mirroring=args.use_mirroring,
    )

    # Save as NIfTI with same affine as input
    os.makedirs(os.path.dirname(args.output_nifti) or ".", exist_ok=True)
    out_nii = nib.Nifti1Image(prediction, affine=nii_ref.affine)
    nib.save(out_nii, args.output_nifti)

    mask_voxels = np.count_nonzero(prediction)
    print(f"\nSaved prediction: {args.output_nifti}")
    print(f"  Shape: {prediction.shape}")
    print(f"  Mask voxels: {mask_voxels} ({100 * mask_voxels / prediction.size:.2f}%)")


if __name__ == "__main__":
    main()
