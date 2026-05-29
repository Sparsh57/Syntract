import os
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Tuple
from scipy.ndimage import gaussian_filter
from unet import *
from scipy import ndimage

def get_gaussian_importance_map(patch_size: Tuple[int, ...], sigma_scale: float = 1/8) -> np.ndarray:
    """Creates a Gaussian importance map for patch overlap handling."""
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    tmp[tuple(center_coords)] = 1
    
    sigmas = [i * sigma_scale for i in patch_size]
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map)
    return gaussian_importance_map

def predict_with_mirroring(model, patch_tensor, device):
    """
    Predict with test-time mirroring augmentation (nnUNet style)
    """
    pred = torch.sigmoid(model(patch_tensor))  # Original prediction
    
    # Mirror along both axes
    mirror_axes = [(2,), (3,), (2,3)]  # For 2D images (batch, channel, H, W)
    n_mirrors = len(mirror_axes) + 1  # +1 for original prediction
    
    for axis in mirror_axes:
        # Mirror input
        mirrored = torch.flip(patch_tensor, dims=axis)
        # Predict on mirrored input
        pred_mirrored = torch.sigmoid(model(mirrored))
        # Mirror prediction back
        pred_mirrored = torch.flip(pred_mirrored, dims=axis)
        # Add to original prediction
        pred += pred_mirrored
    
    # Average all predictions
    pred = pred / n_mirrors
    return pred

def predict_case(
    image_path: str,
    models: List[torch.nn.Module],
    device: torch.device,
    patch_size: Tuple[int, int] = (1024, 1024),
    stride_ratio: float = 0.5,
    confidence_threshold: float = 0.5,  # Increased threshold
    min_size: int = 20,  # Minimum component size
    use_mirroring: bool = True,
    means: List[float] = [26.370181406062926, 28.796625947529606, 25.990918044351993],
    stds: List[float] = [31.776848516250062, 28.548426642727726, 16.703267170423146],
    greyscale: bool = False
) -> np.ndarray:
    """
    Predict a single case using ensemble of models.
    """
    # Load and preprocess image using PIL
    with Image.open(image_path) as img:
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to numpy array [H, W, C]
        image_array = np.array(img)

        if greyscale==True:
            # convert to grayscale and repeat back to 3 channels
            gray = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])  # (H, W) grayscale weights
            gray = np.expand_dims(gray, axis=-1)                         # (H, W, 1)
            image_array = np.repeat(gray, 3, axis=-1).astype(np.float32)   # (H, W, 3)
        
        # Transpose to [C, H, W] format
        image_array = np.transpose(image_array, (2, 0, 1))
    
    # Convert to float32
    image_array = image_array.astype(np.float32)
    
    # Z-Score normalization
    normalized = np.zeros_like(image_array, dtype=np.float32)
    for c in range(image_array.shape[0]):
        normalized[c] = (image_array[c] - means[c]) / stds[c]
    
    # Get image dimensions
    _, H, W = normalized.shape
    
    # Initialize prediction arrays
    prediction_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    
    # Get Gaussian importance map
    gaussian_map = get_gaussian_importance_map(patch_size)
    
    # Calculate steps for sliding window
    stride = (int(patch_size[0] * stride_ratio), int(patch_size[1] * stride_ratio))
    
    print(f"Processing image of size {H}x{W} with patch size {patch_size}")
    
    # Ensemble prediction
    for model_idx, model in enumerate(models):
        print(f"Running prediction with model {model_idx + 1}/{len(models)}")
        model.eval()
        with torch.no_grad():
            # Sliding window prediction
            for h in range(0, H - patch_size[0] + stride[0], stride[0]):
                for w in range(0, W - patch_size[1] + stride[1], stride[1]):
                    # Handle border cases
                    h_end = min(h + patch_size[0], H)
                    w_end = min(w + patch_size[1], W)
                    h_start = max(0, h_end - patch_size[0])
                    w_start = max(0, w_end - patch_size[1])
                    
                    # Extract patch
                    patch = normalized[:, h_start:h_end, w_start:w_end]
                    
                    # Pad if necessary
                    if patch.shape[1:] != patch_size:
                        temp_patch = np.zeros((patch.shape[0],) + patch_size, dtype=np.float32)
                        temp_patch[:, :patch.shape[1], :patch.shape[2]] = patch
                        patch = temp_patch
                    
                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                    
                     # Predict with mirroring if enabled
                    if use_mirroring:
                        pred = predict_with_mirroring(model, patch_tensor, device)
                        pred = pred.cpu().numpy().squeeze()
                    else:
                        # Original prediction without mirroring
                        pred = model(patch_tensor)
                        pred = torch.sigmoid(pred).cpu().numpy().squeeze()
                    
                    # Apply Gaussian weighting
                    pred = pred * gaussian_map[:h_end-h_start, :w_end-w_start]
                    
                    # Accumulate predictions
                    prediction_map[h_start:h_end, w_start:w_end] += pred
                    weight_map[h_start:h_end, w_start:w_end] += gaussian_map[:h_end-h_start, :w_end-w_start]
    
    # Average predictions
    prediction_map = prediction_map / np.maximum(weight_map, 1e-7)

    # First ensure predictions are in valid range [0,1]
    prediction_map = np.clip(prediction_map, 0, 1)

    # Apply confidence threshold
    binary_prediction = prediction_map > confidence_threshold
    
    if min_size is not None and min_size > 0:
        # Remove small components using scipy.ndimage
        # Label connected components
        labels, num_features = ndimage.label(binary_prediction)
    
        if num_features > 0:  # Only process if we found any components
            # Count size of each component
            component_sizes = np.bincount(labels.ravel())[1:]
            # Find components that are too small
            too_small = component_sizes < min_size
            # Get the actual label numbers of small components
            small_labels = np.where(too_small)[0] + 1  # labels start at 1
            # Set those pixels to 0 in the original label map
            mask = np.isin(labels, small_labels)
            labels[mask] = 0
            # Convert back to binary
            binary_prediction = labels > 0

    # Convert to uint8 (using values 0 and 1)
    return binary_prediction.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default= '/autofs/space/atropos_003/users/kb1049/nnunet_segmentation/nnUNet_raw/Dataset301_FiberBundle/imagesTs_MR243fixed/', 
                        help='Inp1ut folder with images')
    parser.add_argument('--output_folder', type=str, default="/autofs/space/atropos_003/users/kb1049/data_synthesis/results/patch_size_1024/evaluate/MR243fixed/NO_pretrained_few_shot_minsize_800/", #3fold_v2_few_shot_best_minsize800/", 
                        help='Output folder for predictions')
    parser.add_argument('--model_folder', type=str, default= "/autofs/space/atropos_003/users/kb1049/data_synthesis/results/patch_size_1024/finetuning/NO_pretrain_few_shot/",  
                        help='Folder containing model checkpoints')
    parser.add_argument('--saved_model', type=str, default="best", help='Best model or last model')
    parser.add_argument('--min_size', type=int, default=800, help='Remove small components')
    parser.add_argument('--pos_weight', type=float, default=1, help='Weight for positive class (bundle) - must match training')
    parser.add_argument("--greyscale",action="store_true",help="Convert images to grayscale and repeat to 3 channels")
    parser.add_argument("--kfold",action="store_true",help="True if we have cross validation.")

    args = parser.parse_args()

    # Print start time
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.kfold: 
        # Load all models
        models = []
        for fold in range(0, 5):
            if args.saved_model == "last":
                if fold == 0:
                    checkpoint_pattern = "last.ckpt"
                else:
                    checkpoint_pattern = f"last-v{fold}.ckpt"
                checkpoint_path = list(Path(args.model_folder).glob(checkpoint_pattern))[0]
            elif args.saved_model == "best":
                checkpoint_pattern = f"best_fold_{fold+1}-epoch=*-val_loss=*.ckpt"
                checkpoint_path = list(Path(args.model_folder).glob(checkpoint_pattern))[0]
            else: 
                raise ValueError("Choose best or last model.")
            
            print(f"Loading model from {checkpoint_path}")
            model = FlexibleUNet(learning_rate=1e-4, pos_weight=args.pos_weight)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            model = model.to(device)
            model.eval()
            models.append(model)
    else:
        # Load single model
        if args.saved_model == "last":
            checkpoint_pattern = "last.ckpt"
        elif args.saved_model == "best":
            checkpoint_pattern = "best*.ckpt"
        else:
            raise ValueError("Choose 'best' or 'last' for --saved_model")

        checkpoint_paths = list(Path(args.model_folder).glob(checkpoint_pattern))
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(f"No checkpoint found matching pattern: {checkpoint_pattern}")
        
        checkpoint_path = checkpoint_paths[0]
        print(f"Loading model from {checkpoint_path}")

        model = FlexibleUNet(learning_rate=1e-4, pos_weight=args.pos_weight)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model = model.to(device)
        model.eval()
        models = [model]  # Keep as list for predict_case() compatibility

    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Process all images
    input_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith('.png')])
    print(input_files)

    for input_file in input_files:
        print(f"Processing {input_file}")
        
        # Predict
        input_path = os.path.join(args.input_folder, input_file)
        prediction = predict_case(
            image_path=input_path,
            models=models,
            device=device,
            patch_size=(1024, 1024),
            stride_ratio=0.25,  # Increased overlap Default: 0.5
            confidence_threshold=0.5,  #Confidence threshold, Default: 0.5
            min_size=args.min_size,  # Minimum component size
            use_mirroring=True,  # Enable test-time mirroring
            greyscale = args.greyscale
        )

        # Convert binary prediction to proper image format
        prediction = (prediction * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]
        
        # Save prediction
        output_path = os.path.join(args.output_folder, input_file.replace('.png', '_pred.png'))
        Image.fromarray(prediction).save(output_path)
        
        print(f"Saved prediction to {output_path}")

if __name__ == "__main__":
    main()