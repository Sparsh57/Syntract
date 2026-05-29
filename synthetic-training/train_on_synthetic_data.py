import argparse
import numpy as np
import os
from pathlib import Path
import wandb
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

torch.set_float32_matmul_precision('high')  # or 'medium'
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from datamodules.dataloaders import OnTheFlyDataModule
from unet import *
import albumentations as A

print(torch.cuda.is_available())

"""
How to run on the fly version:
python train_on_synthetic_data.py --on_the_fly --white_mask "None" --wandb_name "pretraining_1024_no_WM_MASK_sparse_feb10" --checkpoint_dir "/autofs/space/atropos_003/users/kb1049/data_synthesis/results_3/patch_size_1024/"
"""

def get_args_parser():
    parser = argparse.ArgumentParser('Synthetic data training', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batches_per_epoch', default=80, type=int)
    
    parser.add_argument('--checkpoint_dir', type=str, default="/autofs/space/atropos_003/users/kb1049/data_synthesis/testing1/",
                        help="Directory to save checkpoints")
    parser.add_argument('--pretraining_checkpoint', type=str, default="None",
                        help="Path to pretrained model if we have pretraining")
    parser.add_argument('--wandb_name', type=str, default= "unet_1024_synthetic_pretraining_on_the_fly_without_WM_updated_version_loss", #'/autofs/space/atropos_003/users/kb1049/MAEs/finetuning_segmentation/synthetic_data_testing/best_fold_1-epoch=45-val_loss=0.8604.ckpt',
                    help='Name to save on the wandb run')
    
    parser.add_argument('--model', default='unet', help='Model used for the training')
    parser.add_argument('--loss', default='BCE', help='Loss function used for the training')
    parser.add_argument('--pos_weight', type=float, default=1.0, 
                        help='Weight for positive class (bundle). <1.0 = penalize over-prediction, >1.0 = penalize under-prediction')
    parser.add_argument('--white_mask', type=str, default="/space/aspasia/2/users/linc/000003/derivatives/sub-MF278/micr/sub-MF278_sample-brain_desc-WM_mask_in_blockface_space.nii.gz",
                        help='Path to white matter mask file (use "None" to disable)')

    parser.add_argument("--greyscale",action="store_true",help="Convert images to grayscale and repeat to 3 channels")

    parser.add_argument("--on_the_fly",action="store_true",help="Synthetic patches will be generated on the fly")

    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Number of warmup epochs')

    # other parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')   
    parser.set_defaults(pin_mem=True)

    return parser 

def main(args):

    print('Experiment saved here', args.checkpoint_dir)

    # Fixed random seeds
    seed_everything(args.seed, workers=True)

    # Defining data augmentation (moderate increase for synthetic pretraining)
    train_transform = A.Compose([
    A.Resize(height=1024, width=1024),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),  # Added: simulate different slide orientations
    A.ElasticTransform(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),  # Added: simulate microscope settings
    
    A.OneOf([
    A.GaussNoise(var_limit=(10.0, 50.0)),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
    A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
], p=0.5)  # Increased from 0.3 to 0.5
    ])
    
    # Validation transform: NO augmentation, only resizing
    val_transform = A.Compose([
    A.Resize(height=1024, width=1024),
    
    ])
    trk_dir = sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    input_nifti = "/space/aspasia/2/users/linc/000003/derivatives/sub-MF278/micr/sub-MF278_sample-brain_desc-blockface-stacked-masked-grayscale-level4.nii.gz"
    # Handle white_mask: convert "None" string to None, or use provided path
    white_mask = None if (args.white_mask is None or args.white_mask.lower() == "none") else args.white_mask
    patch_size = [512,1,512]
    
    datamodule = OnTheFlyDataModule(
        trk_dir=trk_dir,
        input_nifti=input_nifti,
        white_mask_file= white_mask,
        batch_size=args.batch_size,
        train_batches_per_epoch = args.batches_per_epoch,
        val_batches = 30,
        num_workers=args.num_workers,
        patch_size=patch_size,
        seed=args.seed,
        train_transform=train_transform,
        val_transform=val_transform)


    # checkpoint name
    checkpoint_name = "last.ckpt"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, starting from scratch")

    # # ------- wandb logging -------
    wandb_logger = WandbLogger(
        project="unet-training-on-synthetic",
        name=f"{args.wandb_name}_{args.loss}",
        save_code = False,
        log_model=False
    )

    model = FlexibleUNet(
        batch_size=args.batch_size, 
        learning_rate=args.lr, 
        warmup_epochs=args.warmup_epochs, 
        weight_decay=args.weight_decay, 
        loss=args.loss,
        pos_weight=args.pos_weight)  # Add class weighting
    if args.pretraining_checkpoint and args.pretraining_checkpoint.lower() != "none":  # Reuse pretrained checkpoint arg for UNet pre-trained weights
        print('Am I in here? Should I be?')
        checkpoint = torch.load(args.pretraining_checkpoint)
        if 'state_dict' in checkpoint:
            # Extract only the UNet model weights and remove the 'model.' prefix
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    # Remove 'model.' prefix to match current architecture
                    new_key = k.replace('model.', '')
                    state_dict[new_key] = v
            # Load the weights
            model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore channel_adapter


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir, # folder where ckpt will be saved
        filename="best_-{epoch:02d}-{val_loss:.4f}",  
        monitor = "val_loss",
        mode = "min",
        save_last=True,           
        save_top_k=1       
        )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices = 1,
        logger=wandb_logger,
        precision=16,
        callbacks=[checkpoint_callback],
    )

    # Train
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        print("Starting training from scratch")
        trainer.fit(model, datamodule=datamodule)
    
    # Finish the current WandB run
    wandb.finish()
        

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()  
    main(args) 