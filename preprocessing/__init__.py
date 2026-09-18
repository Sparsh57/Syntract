"""
MRI Synthesis and Processing Package

A comprehensive package for processing, transforming, and synthesizing MRI data
with tractography information, including ANTs integration and GPU acceleration.
"""

from .full_volume import process_and_save
from .densify import (
    densify_streamlines_parallel,
    densify_streamline_subvoxel,
    calculate_streamline_metrics
)
from .nifti_resampling import resample_nifti, estimate_memory_usage
from .streamline_processing import (
    transform_and_densify_streamlines,
    transform_streamline,
    clip_streamline_to_fov
)
from .affine import build_new_affine
from .ants_transform import (
    apply_ants_transform_to_mri,
    apply_ants_transform_to_streamlines,
    process_with_ants
)

__version__ = "1.0.0"
__author__ = "LINC Team"
__license__ = "MIT"

__all__ = [
    # Main processing
    'process_and_save',

    # Densification
    'densify_streamlines_parallel',
    'densify_streamline_subvoxel',
    'calculate_streamline_metrics',

    # NIfTI processing
    'resample_nifti',
    'estimate_memory_usage',

    # Streamline processing
    'transform_and_densify_streamlines',
    'transform_streamline',
    'clip_streamline_to_fov',

    # Transforms
    'build_new_affine',

    # ANTs integration
    'apply_ants_transform_to_mri',
    'apply_ants_transform_to_streamlines',
    'process_with_ants',
]
