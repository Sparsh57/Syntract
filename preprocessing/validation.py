#!/usr/bin/env python
"""
Comprehensive Validation and Error Handling for MRI Synthesis Pipeline
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import warnings
from typing import Tuple, Optional, List, Dict, Any
import logging


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ValidationWarning(UserWarning):
    """Custom warning for validation issues that don't require stopping."""
    pass


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging for validation and error tracking."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_file_exists(file_path: str, file_type: str = "file") -> str:
    """
    Validate that a file exists and is accessible.
    
    Args:
        file_path: Path to the file
        file_type: Type of file for error messages
        
    Returns:
        Absolute path to the file
        
    Raises:
        ValidationError: If file doesn't exist or isn't accessible
    """
    if not file_path:
        raise ValidationError(f"{file_type} path cannot be empty")
    
    abs_path = os.path.abspath(file_path)
    
    if not os.path.exists(abs_path):
        raise ValidationError(f"{file_type} not found: {abs_path}")
    
    if not os.path.isfile(abs_path):
        raise ValidationError(f"Path is not a file: {abs_path}")
    
    if not os.access(abs_path, os.R_OK):
        raise ValidationError(f"{file_type} is not readable: {abs_path}")
    
    return abs_path


def validate_directory_writable(dir_path: str, create_if_missing: bool = True) -> str:
    """
    Validate that a directory exists and is writable.
    
    Args:
        dir_path: Path to the directory
        create_if_missing: Whether to create directory if it doesn't exist
        
    Returns:
        Absolute path to the directory
        
    Raises:
        ValidationError: If directory issues cannot be resolved
    """
    if not dir_path:
        raise ValidationError("Directory path cannot be empty")
    
    abs_path = os.path.abspath(dir_path)
    
    if not os.path.exists(abs_path):
        if create_if_missing:
            try:
                os.makedirs(abs_path, exist_ok=True)
            except OSError as e:
                raise ValidationError(f"Cannot create directory {abs_path}: {e}")
        else:
            raise ValidationError(f"Directory not found: {abs_path}")
    
    if not os.path.isdir(abs_path):
        raise ValidationError(f"Path is not a directory: {abs_path}")
    
    if not os.access(abs_path, os.W_OK):
        raise ValidationError(f"Directory is not writable: {abs_path}")
    
    return abs_path


def validate_nifti_file(nifti_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Validate NIfTI file and extract metadata.
    
    Args:
        nifti_path: Path to NIfTI file
        
    Returns:
        Tuple of (validated_path, metadata_dict)
        
    Raises:
        ValidationError: If NIfTI file is invalid
    """
    validated_path = validate_file_exists(nifti_path, "NIfTI file")
    
    try:
        img = nib.load(validated_path)
        data = img.get_fdata()
        
        metadata = {
            'shape': img.shape,
            'ndim': img.ndim,
            'voxel_sizes': img.header.get_zooms()[:3],
            'data_type': img.get_data_dtype(),
            'affine': img.affine,
            'orientation': nib.orientations.ornt2axcodes(
                nib.orientations.io_orientation(img.affine)
            ),
            'data_range': (float(data.min()), float(data.max())),
            'has_nan': bool(np.isnan(data).any()),
            'has_inf': bool(np.isinf(data).any()),
            'non_zero_voxels': int(np.count_nonzero(data)),
            'total_voxels': int(data.size)
        }
        
        # Validate basic requirements
        if img.ndim < 3:
            raise ValidationError(f"NIfTI file must be at least 3D, got {img.ndim}D")
        
        if img.ndim > 4:
            warnings.warn(f"NIfTI file has {img.ndim} dimensions, only first 3 will be used", ValidationWarning)
        
        if metadata['has_nan']:
            warnings.warn("NIfTI file contains NaN values", ValidationWarning)
        
        if metadata['has_inf']:
            warnings.warn("NIfTI file contains infinite values", ValidationWarning)
        
        if metadata['non_zero_voxels'] == 0:
            warnings.warn("NIfTI file appears to be empty (all zeros)", ValidationWarning)
        
        # Check for reasonable voxel sizes
        voxel_sizes = metadata['voxel_sizes']
        if any(vs <= 0 for vs in voxel_sizes):
            raise ValidationError(f"Invalid voxel sizes: {voxel_sizes}")
        
        if any(vs > 10.0 for vs in voxel_sizes):  # > 10mm seems unrealistic
            warnings.warn(f"Large voxel sizes detected: {voxel_sizes} mm", ValidationWarning)
        
        return validated_path, metadata
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to load NIfTI file {validated_path}: {e}")


def validate_trk_file(trk_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Validate TRK file and extract metadata.
    
    Args:
        trk_path: Path to TRK file
        
    Returns:
        Tuple of (validated_path, metadata_dict)
        
    Raises:
        ValidationError: If TRK file is invalid
    """
    validated_path = validate_file_exists(trk_path, "TRK file")
    
    try:
        from nibabel.streamlines import load as load_trk
        
        trk_obj = load_trk(validated_path)
        streamlines = trk_obj.streamlines
        
        # Calculate streamline statistics
        if len(streamlines) > 0:
            lengths = [len(s) for s in streamlines]
            coordinates = np.concatenate(streamlines) if streamlines else np.array([])
            
            metadata = {
                'n_streamlines': len(streamlines),
                'total_points': len(coordinates),
                'avg_points_per_streamline': np.mean(lengths) if lengths else 0,
                'min_points_per_streamline': min(lengths) if lengths else 0,
                'max_points_per_streamline': max(lengths) if lengths else 0,
                'coordinate_range': {
                    'x': (float(coordinates[:, 0].min()), float(coordinates[:, 0].max())),
                    'y': (float(coordinates[:, 1].min()), float(coordinates[:, 1].max())),
                    'z': (float(coordinates[:, 2].min()), float(coordinates[:, 2].max()))
                } if len(coordinates) > 0 else None,
                'header': dict(trk_obj.header),
                'coordinate_system': 'world' if np.abs(coordinates).max() > 1000 else 'voxel'
            }
        else:
            metadata = {
                'n_streamlines': 0,
                'total_points': 0,
                'avg_points_per_streamline': 0,
                'min_points_per_streamline': 0,
                'max_points_per_streamline': 0,
                'coordinate_range': None,
                'header': dict(trk_obj.header),
                'coordinate_system': 'unknown'
            }
        
        # Validate basic requirements
        if metadata['n_streamlines'] == 0:
            warnings.warn("TRK file contains no streamlines", ValidationWarning)
        
        if metadata['n_streamlines'] < 100:
            warnings.warn(f"TRK file has only {metadata['n_streamlines']} streamlines - may be sparse", ValidationWarning)
        
        if metadata['avg_points_per_streamline'] < 2:
            warnings.warn("Streamlines have very few points on average", ValidationWarning)
        
        return validated_path, metadata
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to load TRK file {validated_path}: {e}")


def validate_target_dimensions(dimensions: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Validate target dimensions for resampling.
    
    Args:
        dimensions: Target dimensions (x, y, z)
        
    Returns:
        Validated dimensions
        
    Raises:
        ValidationError: If dimensions are invalid
    """
    if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 3:
        raise ValidationError("Target dimensions must be a tuple/list of 3 integers")
    
    try:
        dims = tuple(int(d) for d in dimensions)
    except (ValueError, TypeError):
        raise ValidationError(f"Target dimensions must be integers: {dimensions}")
    
    for i, dim in enumerate(dims):
        if dim <= 0:
            raise ValidationError(f"Dimension {i} must be positive: {dim}")
        if dim > 2000:  # Reasonable upper limit
            warnings.warn(f"Large dimension {i}: {dim} - may require significant memory", ValidationWarning)
    
    # Check total voxel count
    total_voxels = dims[0] * dims[1] * dims[2]
    if total_voxels > 100_000_000:  # 100M voxels
        warnings.warn(f"Large total voxel count: {total_voxels:,} - may require significant memory", ValidationWarning)
    
    return dims


def validate_voxel_size(voxel_size: float) -> float:
    """
    Validate voxel size parameter.
    
    Args:
        voxel_size: Target voxel size in mm
        
    Returns:
        Validated voxel size
        
    Raises:
        ValidationError: If voxel size is invalid
    """
    try:
        vs = float(voxel_size)
    except (ValueError, TypeError):
        raise ValidationError(f"Voxel size must be a number: {voxel_size}")
    
    if vs <= 0:
        raise ValidationError(f"Voxel size must be positive: {vs}")
    
    if vs > 10.0:
        warnings.warn(f"Large voxel size: {vs} mm", ValidationWarning)
    
    if vs < 0.01:
        warnings.warn(f"Very small voxel size: {vs} mm - may result in huge volumes", ValidationWarning)
    
    return vs


def validate_slice_count(slice_count: Optional[int], max_dimension: int) -> Optional[int]:
    """
    Validate slice count parameter.
    
    Args:
        slice_count: Number of slices to extract
        max_dimension: Maximum possible slices (Y dimension)
        
    Returns:
        Validated slice count
        
    Raises:
        ValidationError: If slice count is invalid
    """
    if slice_count is None:
        return None
    
    try:
        sc = int(slice_count)
    except (ValueError, TypeError):
        raise ValidationError(f"Slice count must be an integer: {slice_count}")
    
    if sc <= 0:
        raise ValidationError(f"Slice count must be positive: {sc}")
    
    if sc > max_dimension:
        warnings.warn(f"Slice count ({sc}) exceeds maximum dimension ({max_dimension}) - will be adjusted", ValidationWarning)
        return max_dimension
    
    return sc


def validate_synthesis_parameters(
    nifti_path: str,
    trk_path: str,
    target_dimensions: Tuple[int, int, int],
    target_voxel_size: float,
    output_prefix: str,
    slice_count: Optional[int] = None,
    slice_output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive validation of all synthesis parameters.
    
    Args:
        nifti_path: Path to NIfTI file
        trk_path: Path to TRK file
        target_dimensions: Target dimensions for resampling
        target_voxel_size: Target voxel size
        output_prefix: Output file prefix
        slice_count: Number of slices to extract (optional)
        slice_output_dir: Slice output directory (optional)
        
    Returns:
        Dictionary with validated parameters and metadata
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    logger = setup_logging()
    logger.info("Starting comprehensive parameter validation")
    
    # Validate input files
    logger.info("Validating input files...")
    validated_nifti, nifti_metadata = validate_nifti_file(nifti_path)
    validated_trk, trk_metadata = validate_trk_file(trk_path)
    
    # Validate target parameters
    logger.info("Validating target parameters...")
    validated_dimensions = validate_target_dimensions(target_dimensions)
    validated_voxel_size = validate_voxel_size(target_voxel_size)
    
    # Validate slice extraction parameters
    if slice_count is not None:
        logger.info("Validating slice extraction parameters...")
        validated_slice_count = validate_slice_count(slice_count, validated_dimensions[1])
        
        if slice_output_dir:
            validated_slice_dir = validate_directory_writable(slice_output_dir, create_if_missing=True)
        else:
            validated_slice_dir = None
    else:
        validated_slice_count = None
        validated_slice_dir = None
    
    # Validate output directory
    output_dir = os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else "."
    validate_directory_writable(output_dir, create_if_missing=True)
    
    # Check compatibility between NIfTI and TRK
    logger.info("Checking NIfTI-TRK compatibility...")
    if nifti_metadata['shape'][:3] != tuple(trk_metadata['header'].get('dimensions', [0, 0, 0])):
        warnings.warn("NIfTI and TRK dimensions may not match - this could cause issues", ValidationWarning)
    
    # Memory estimation
    logger.info("Estimating memory requirements...")
    output_voxels = validated_dimensions[0] * validated_dimensions[1] * validated_dimensions[2]
    estimated_memory_gb = (output_voxels * 4) / (1024**3)  # Assume float32
    
    if estimated_memory_gb > 16:
        warnings.warn(f"Estimated memory usage: {estimated_memory_gb:.1f} GB - may require significant RAM", ValidationWarning)
    
    result = {
        'validated_inputs': {
            'nifti_path': validated_nifti,
            'trk_path': validated_trk,
            'target_dimensions': validated_dimensions,
            'target_voxel_size': validated_voxel_size,
            'slice_count': validated_slice_count,
            'slice_output_dir': validated_slice_dir
        },
        'metadata': {
            'nifti': nifti_metadata,
            'trk': trk_metadata,
            'estimated_memory_gb': estimated_memory_gb,
            'output_voxels': output_voxels
        },
        'validation_status': 'success'
    }
    
    logger.info("Parameter validation completed successfully")
    return result


def log_processing_progress(stage: str, progress: float, details: str = "") -> None:
    """
    Log processing progress with consistent formatting.
    
    Args:
        stage: Processing stage name
        progress: Progress as percentage (0-100)
        details: Additional details
    """
    logger = setup_logging()
    progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
    logger.info(f"{stage}: [{progress_bar}] {progress:.1f}% {details}")

