import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from datamodules.datasets import SyntheticDataset, OnTheFlySyntheticData
from datamodules.datasets import SyntheticDataset3D, OnTheFlySyntheticData3D
import os

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import random
import queue
import threading
from pathlib import Path
from typing import Optional, Sequence, Tuple, Iterable

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import albumentations as A


class _ThreadedPrefetcher(IterableDataset):
    def __init__(self, dataset: IterableDataset, prefetch: int = 2):
        super().__init__()
        self.dataset = dataset
        self.prefetch = max(0, prefetch)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.prefetch == 0:
            yield from self.dataset
            return

        q: "queue.Queue" = queue.Queue(self.prefetch)
        stop_token = object()

        def _worker():
            try:
                for item in self.dataset:
                    q.put(item)
                q.put(stop_token)
            except Exception as e:  # propagate errors to main thread
                q.put(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is stop_token:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        t.join()


class OnTheFlyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        trk_dir: str,
        input_nifti: str,
        white_mask_file: str,
        train_transform: str,
        val_transform: str,
        batch_size: int = 8,
        num_workers: int = 4,
        patch_size: Sequence[int] = (512, 1, 512),
        greyscale: bool = False,
        train_batches_per_epoch: int = 200,
        val_batches: int = 50,
        seed: int = 42,
    ):
        super().__init__()
        self.trk_dir = trk_dir
        self.input_nifti = input_nifti
        self.white_mask_file = white_mask_file
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(1, min(8, os.cpu_count() or 1))
        self.patch_size = patch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.greyscale = greyscale
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches = val_batches
        self.seed = seed

    def setup(self, stage=None):
        self.train_dataset = OnTheFlySyntheticData(
            trk_dir=self.trk_dir,
            input_nifti=self.input_nifti,
            white_mask_file=self.white_mask_file,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            batches_per_epoch=self.train_batches_per_epoch,
            transform=self.train_transform,
            seed=self.seed,
            greyscale=self.greyscale,
        )
        self.val_dataset = OnTheFlySyntheticData(
            trk_dir=self.trk_dir,
            input_nifti=self.input_nifti,
            white_mask_file=self.white_mask_file,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            batches_per_epoch=self.val_batches,
            transform=self.val_transform, 
            seed=self.seed + 9999,
            greyscale=self.greyscale,
        )

    def train_dataloader(self):
        pf = 4 if self.num_workers and self.num_workers > 0 else None
        return DataLoader(
            self.train_dataset,
            batch_size=None, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.num_workers),
            prefetch_factor=pf,
        )

    def val_dataloader(self):
        val_workers = max(0, self.num_workers // 2)
        pf = 4 if val_workers and val_workers > 0 else None
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=val_workers,
            pin_memory=True,
            persistent_workers=bool(val_workers),
            prefetch_factor=pf,
        )


class OnTheFlyDataModule3D(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for 3D volumetric training.

    Supports two modes:
      - on_the_fly=True: generates 3D patches during training (OnTheFlySyntheticData3D)
      - on_the_fly=False: loads pre-generated NIfTI patches from disk (SyntheticDataset3D)
    """

    def __init__(
        self,
        trk_dir: str,
        input_nifti: str,
        white_mask_file: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        patch_size: Sequence[int] = (128, 128, 128),
        train_batches_per_epoch: int = 50,
        val_batches: int = 4,
        seed: int = 42,
        on_the_fly: bool = True,
        patch_dir: Optional[str] = None,
        use_cornucopia_3d: bool = True,
        enable_tissue_artifacts: bool = True,
        enable_granular_noise: bool = True,
        enable_speckle_noise: bool = True,
        enable_dash_noise: bool = True,
        enable_horizontal_banding: bool = True,
        granular_noise_strength: float = 0.35,
        artifact_strength: float = 0.45,
        speckle_noise_strength: float = 0.70,
        speckle_noise_density: float = 0.008,
        speckle_noise_sigma: float = 0.0,
        speckle_square_size: int = 2,
        dash_noise_strength: float = 0.55,
        dash_noise_density: float = 0.0005,
        dash_length_sigma: float = 4.0,
        dash_cross_sigma: float = 0.8,
        banding_strength: float = 0.18,
        banding_axis: int = 1,
        fiber_intensity_min: float = 60.0,
        fiber_intensity_max: float = 100.0,
        fiber_max_boost: float = 10.0,
        fiber_opacity: float = 1.0,
        fiber_smoothing_sigma: float = 0.0,
        fiber_antialias: bool = True,
        min_streamlines_rendered: Optional[int] = None,
        fiber_brightness_variation: float = 0.60,
        fiber_segment_brightness_variation: float = 0.35,
        fiber_render_mode: str = "additive",
        fiber_density_gamma: float = 5.0,
        fiber_min_visibility: float = 0.0,
        fiber_target_intensity: float = 25.0,
        background_max_intensity: Optional[float] = None,
        tissue_threshold: float = 2.0,
        enable_cell_blobs: bool = False,
        cell_blob_count: int = 60,
        cell_blob_intensity: float = 0.3,
        cell_blob_radius_range: Sequence[float] = (1.5, 4.0),
        cornucopia_allowed_presets: Optional[Sequence[str]] = None,
        mask_smoothing_sigma: float = 0.0,
        mask_binary_threshold: float = 0.01,
        voxel_size: float = 0.05,
        min_streamlines_per_patch: int = 5,
        streamline_margin_fraction: float = 0.10,
        prefetch_batches: int = 2,
        move_to_gpu: bool = False,
        batch_group_factor: int = 10,
        render_use_gpu: bool = True,
        verbose_generation: bool = False,
        thinslab_prob: float = 0.3,
        thinslab_min_z: int = 30,
        thinslab_max_z: int = 120,
        empty_patch_prob: float = 0.05,
    ):
        super().__init__()
        self.trk_dir = trk_dir
        self.input_nifti = input_nifti
        self.white_mask_file = white_mask_file
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(1, min(8, os.cpu_count() or 1))
        self.patch_size = patch_size
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches = val_batches
        self.seed = seed
        self.on_the_fly = on_the_fly
        self.patch_dir = patch_dir
        self.use_cornucopia_3d = use_cornucopia_3d
        self.enable_tissue_artifacts = bool(enable_tissue_artifacts)
        self.enable_granular_noise = bool(enable_granular_noise)
        self.enable_speckle_noise = bool(enable_speckle_noise)
        self.enable_dash_noise = bool(enable_dash_noise)
        self.enable_horizontal_banding = bool(enable_horizontal_banding)
        self.granular_noise_strength = float(granular_noise_strength)
        self.artifact_strength = float(artifact_strength)
        self.speckle_noise_strength = float(speckle_noise_strength)
        self.speckle_noise_density = float(speckle_noise_density)
        self.speckle_noise_sigma = float(speckle_noise_sigma)
        self.speckle_square_size = int(speckle_square_size)
        self.dash_noise_strength = float(dash_noise_strength)
        self.dash_noise_density = float(dash_noise_density)
        self.dash_length_sigma = float(dash_length_sigma)
        self.dash_cross_sigma = float(dash_cross_sigma)
        self.banding_strength = float(banding_strength)
        self.banding_axis = int(banding_axis)
        self.fiber_intensity_min = float(fiber_intensity_min)
        self.fiber_intensity_max = float(fiber_intensity_max)
        self.fiber_max_boost = None if fiber_max_boost is None else float(fiber_max_boost)
        self.fiber_opacity = float(fiber_opacity)
        self.fiber_smoothing_sigma = float(fiber_smoothing_sigma)
        self.fiber_antialias = bool(fiber_antialias)
        self.min_streamlines_rendered = (
            None if min_streamlines_rendered is None else int(min_streamlines_rendered)
        )
        self.fiber_brightness_variation = float(fiber_brightness_variation)
        self.fiber_segment_brightness_variation = float(fiber_segment_brightness_variation)
        self.fiber_render_mode = str(fiber_render_mode)
        self.fiber_density_gamma = float(fiber_density_gamma)
        self.fiber_min_visibility = float(fiber_min_visibility)
        self.fiber_target_intensity = float(fiber_target_intensity)
        self.background_max_intensity = (
            None if background_max_intensity is None else float(background_max_intensity)
        )
        self.tissue_threshold = float(tissue_threshold)
        self.enable_cell_blobs = bool(enable_cell_blobs)
        self.cell_blob_count = int(cell_blob_count)
        self.cell_blob_intensity = float(cell_blob_intensity)
        self.cell_blob_radius_range = tuple(float(r) for r in cell_blob_radius_range)
        self.cornucopia_allowed_presets = (
            None if cornucopia_allowed_presets is None else list(cornucopia_allowed_presets)
        )
        self.mask_smoothing_sigma = float(mask_smoothing_sigma)
        self.mask_binary_threshold = float(mask_binary_threshold)
        self.voxel_size = voxel_size
        self.min_streamlines_per_patch = min_streamlines_per_patch
        self.streamline_margin_fraction = float(streamline_margin_fraction)
        self.prefetch_batches = prefetch_batches
        self.move_to_gpu = move_to_gpu
        self.batch_group_factor = batch_group_factor
        self.render_use_gpu = render_use_gpu
        self.verbose_generation = verbose_generation
        self.thinslab_prob = float(thinslab_prob)
        self.thinslab_min_z = int(thinslab_min_z)
        self.thinslab_max_z = int(thinslab_max_z)
        self.empty_patch_prob = float(empty_patch_prob)

    def setup(self, stage=None):
        patch_use_gpu = bool(self.render_use_gpu)
        if self.num_workers and self.num_workers > 0 and self.render_use_gpu:
            print("On-the-fly generation keeps num_workers=0 when GPU rendering is enabled (CUDA-fork safety).")
        elif self.num_workers and self.num_workers > 0:
            patch_use_gpu = False

        if self.on_the_fly:
            self.train_dataset = OnTheFlySyntheticData3D(
                trk_dir=self.trk_dir,
                input_nifti=self.input_nifti,
                white_mask_file=self.white_mask_file,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                batches_per_epoch=self.train_batches_per_epoch,
                seed=self.seed,
                use_cornucopia_3d=self.use_cornucopia_3d,
                enable_tissue_artifacts=self.enable_tissue_artifacts,
                enable_granular_noise=self.enable_granular_noise,
                enable_speckle_noise=self.enable_speckle_noise,
                enable_dash_noise=self.enable_dash_noise,
                enable_horizontal_banding=self.enable_horizontal_banding,
                granular_noise_strength=self.granular_noise_strength,
                artifact_strength=self.artifact_strength,
                speckle_noise_strength=self.speckle_noise_strength,
                speckle_noise_density=self.speckle_noise_density,
                speckle_noise_sigma=self.speckle_noise_sigma,
                speckle_square_size=self.speckle_square_size,
                dash_noise_strength=self.dash_noise_strength,
                dash_noise_density=self.dash_noise_density,
                dash_length_sigma=self.dash_length_sigma,
                dash_cross_sigma=self.dash_cross_sigma,
                banding_strength=self.banding_strength,
                banding_axis=self.banding_axis,
                fiber_intensity_min=self.fiber_intensity_min,
                fiber_intensity_max=self.fiber_intensity_max,
                fiber_max_boost=self.fiber_max_boost,
                fiber_opacity=self.fiber_opacity,
                fiber_smoothing_sigma=self.fiber_smoothing_sigma,
                fiber_antialias=self.fiber_antialias,
                min_streamlines_rendered=self.min_streamlines_rendered,
                fiber_brightness_variation=self.fiber_brightness_variation,
                fiber_segment_brightness_variation=self.fiber_segment_brightness_variation,
                fiber_render_mode=self.fiber_render_mode,
                fiber_density_gamma=self.fiber_density_gamma,
                fiber_min_visibility=self.fiber_min_visibility,
                fiber_target_intensity=self.fiber_target_intensity,
                background_max_intensity=self.background_max_intensity,
                tissue_threshold=self.tissue_threshold,
                enable_cell_blobs=self.enable_cell_blobs,
                cell_blob_count=self.cell_blob_count,
                cell_blob_intensity=self.cell_blob_intensity,
                cell_blob_radius_range=self.cell_blob_radius_range,
                cornucopia_allowed_presets=self.cornucopia_allowed_presets,
                mask_smoothing_sigma=self.mask_smoothing_sigma,
                mask_binary_threshold=self.mask_binary_threshold,
                voxel_size=self.voxel_size,
                min_streamlines_per_patch=self.min_streamlines_per_patch,
                streamline_margin_fraction=self.streamline_margin_fraction,
                move_to_gpu=self.move_to_gpu,
                batch_group_factor=self.batch_group_factor,
                render_use_gpu=self.render_use_gpu,
                verbose_generation=self.verbose_generation,
                patch_use_gpu=patch_use_gpu,
                thinslab_prob=self.thinslab_prob,
                thinslab_min_z=self.thinslab_min_z,
                thinslab_max_z=self.thinslab_max_z,
                empty_patch_prob=self.empty_patch_prob,
            )
            # Validation mirrors the training augmentation distribution so the
            # val metric is comparable to training (rather than the trivial
            # clean-synthetic ↔ noisy-synthetic gap that previously pinned
            # val_dice near 0.55 regardless of model quality).  Since we have
            # no real labels for the OME-Zarr inference target, val is purely
            # an in-distribution check.
            self.val_dataset = OnTheFlySyntheticData3D(
                trk_dir=self.trk_dir,
                input_nifti=self.input_nifti,
                white_mask_file=self.white_mask_file,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                batches_per_epoch=self.val_batches,
                seed=self.seed + 9999,
                use_cornucopia_3d=self.use_cornucopia_3d,
                enable_tissue_artifacts=self.enable_tissue_artifacts,
                enable_granular_noise=self.enable_granular_noise,
                enable_speckle_noise=self.enable_speckle_noise,
                enable_dash_noise=self.enable_dash_noise,
                enable_horizontal_banding=self.enable_horizontal_banding,
                granular_noise_strength=self.granular_noise_strength,
                artifact_strength=self.artifact_strength,
                speckle_noise_strength=self.speckle_noise_strength,
                speckle_noise_density=self.speckle_noise_density,
                speckle_noise_sigma=self.speckle_noise_sigma,
                speckle_square_size=self.speckle_square_size,
                dash_noise_strength=self.dash_noise_strength,
                dash_noise_density=self.dash_noise_density,
                dash_length_sigma=self.dash_length_sigma,
                dash_cross_sigma=self.dash_cross_sigma,
                banding_strength=self.banding_strength,
                banding_axis=self.banding_axis,
                fiber_intensity_min=self.fiber_intensity_min,
                fiber_intensity_max=self.fiber_intensity_max,
                fiber_max_boost=self.fiber_max_boost,
                fiber_opacity=self.fiber_opacity,
                fiber_smoothing_sigma=self.fiber_smoothing_sigma,
                fiber_antialias=self.fiber_antialias,
                min_streamlines_rendered=self.min_streamlines_rendered,
                fiber_brightness_variation=self.fiber_brightness_variation,
                fiber_segment_brightness_variation=self.fiber_segment_brightness_variation,
                fiber_render_mode=self.fiber_render_mode,
                fiber_density_gamma=self.fiber_density_gamma,
                fiber_min_visibility=self.fiber_min_visibility,
                fiber_target_intensity=self.fiber_target_intensity,
                background_max_intensity=self.background_max_intensity,
                tissue_threshold=self.tissue_threshold,
                enable_cell_blobs=self.enable_cell_blobs,
                cell_blob_count=self.cell_blob_count,
                cell_blob_intensity=self.cell_blob_intensity,
                cell_blob_radius_range=self.cell_blob_radius_range,
                cornucopia_allowed_presets=self.cornucopia_allowed_presets,
                mask_smoothing_sigma=self.mask_smoothing_sigma,
                mask_binary_threshold=self.mask_binary_threshold,
                voxel_size=self.voxel_size,
                min_streamlines_per_patch=self.min_streamlines_per_patch,
                streamline_margin_fraction=self.streamline_margin_fraction,
                move_to_gpu=self.move_to_gpu,
                batch_group_factor=max(1, self.batch_group_factor // 2),
                render_use_gpu=self.render_use_gpu,
                verbose_generation=self.verbose_generation,
                patch_use_gpu=patch_use_gpu,
                thinslab_prob=self.thinslab_prob,
                thinslab_min_z=self.thinslab_min_z,
                thinslab_max_z=self.thinslab_max_z,
                empty_patch_prob=self.empty_patch_prob,
            )
        else:
            if not self.patch_dir:
                raise ValueError("patch_dir is required when on_the_fly=False")
            self.train_dataset = SyntheticDataset3D(
                patch_dir=self.patch_dir,
                patch_size=self.patch_size,
                enable_tissue_artifacts=self.enable_tissue_artifacts,
                enable_granular_noise=self.enable_granular_noise,
                enable_speckle_noise=self.enable_speckle_noise,
                granular_noise_strength=self.granular_noise_strength,
                artifact_strength=self.artifact_strength,
                speckle_noise_strength=self.speckle_noise_strength,
                speckle_noise_density=self.speckle_noise_density,
                speckle_noise_sigma=self.speckle_noise_sigma,
                seed=self.seed,
            )
            self.val_dataset = SyntheticDataset3D(
                patch_dir=self.patch_dir,
                patch_size=self.patch_size,
                enable_tissue_artifacts=False,
                enable_granular_noise=False,
                enable_speckle_noise=False,
                seed=self.seed + 9999,
            )

    def train_dataloader(self):
        pin_memory = torch.cuda.is_available()
        if self.on_the_fly:
            if self.num_workers > 0 and getattr(self.train_dataset, "supports_multiprocess", False):
                return DataLoader(
                    self.train_dataset,
                    batch_size=None,
                    num_workers=self.num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=True,
                    prefetch_factor=2,
                )
            dataset = _ThreadedPrefetcher(self.train_dataset, prefetch=self.prefetch_batches)
            return DataLoader(
                dataset,
                batch_size=None,  # Dataset yields pre-batched data
                num_workers=0,
                pin_memory=pin_memory,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(self.num_workers),
            prefetch_factor=4 if self.num_workers and self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        pin_memory = torch.cuda.is_available()
        if self.on_the_fly:
            val_workers = max(0, self.num_workers // 2)
            if val_workers > 0 and getattr(self.val_dataset, "supports_multiprocess", False):
                return DataLoader(
                    self.val_dataset,
                    batch_size=None,
                    num_workers=val_workers,
                    pin_memory=pin_memory,
                    persistent_workers=True,
                    prefetch_factor=2,
                )
            dataset = _ThreadedPrefetcher(self.val_dataset, prefetch=self.prefetch_batches)
            return DataLoader(
                dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=pin_memory,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(self.num_workers),
            prefetch_factor=4 if self.num_workers and self.num_workers > 0 else None,
        )
