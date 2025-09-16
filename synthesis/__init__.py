"""
MRI Synthesis and Processing Package

A comprehensive package for processing, transforming, and synthesizing MRI data
with tractography information, including ANTs integration and GPU acceleration.
"""

from .main import process_and_save
from .densify import (
    densify_streamlines_parallel,
    densify_streamline_subvoxel,
    calculate_streamline_metrics
)
from .nifti_preprocessing import resample_nifti, estimate_memory_usage
from .streamline_processing import (
    transform_and_densify_streamlines,
    transform_streamline,
    clip_streamline_to_fov
)
from .transform import build_new_affine
from .visualize import (
    overlay_streamlines_on_blockface_coronal,
    visualize_trk_with_nifti
)
from .ants_transform_updated import (
    apply_ants_transform_to_mri,
    apply_ants_transform_to_streamlines,
    process_with_ants
)
from .compare_interpolation import compare_interpolations
from .slice_simplified import (
    extract_coronal_slices_simple,
    extract_patches_simple,
    geometric_slab_clipping,
    filter_streamlines_by_bounds
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
    
    # Visualization
    'overlay_streamlines_on_blockface_coronal',
    'visualize_trk_with_nifti',
    
    # ANTs integration
    'apply_ants_transform_to_mri',
    'apply_ants_transform_to_streamlines',
    'process_with_ants',
    
    # Comparison tools
    'compare_interpolations',
    
    # Slice selection
    'extract_coronal_slices_simple',
    'extract_patches_simple',
    'geometric_slab_clipping',
    'filter_streamlines_by_bounds'
]