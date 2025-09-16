#!/usr/bin/env python
"""
Updated ANTs Transform Implementation
Adapted to match the exact methodology provided by the user.

This implementation follows the cleaner, more straightforward approach
without complex orientation handling and clipping logic.
"""

import numpy as np
import nibabel as nib
from nibabel.streamlines.trk import TrkFile, Tractogram
from scipy.io import loadmat
from scipy.ndimage import map_coordinates


def load_ants_warp(path):
    """Load ANTs warp and convert it into a RAS displacement.

    NOTE (from ants wiki)
        The forward displacement field is defined by the domain of the
        fixed image. It has the same voxel grid, spacing, and orientation
        as the fixed image, but each voxel is a displacement vector in
        physical space for a point at the center of the voxel in fixed space,
        towards the moving space, before applying the affine transform.

        The inverse displacement field is also in the domain of the fixed
        image. It has the same voxel grid, spacing, and orientation as
        the forward warps - but the displacement vectors in each voxel
        transform points towards the fixed space, after applying the
        inverse affine.
    """
    f = nib.load(path)
    d = f.get_fdata()
    # squeeze dimensions 3
    d = np.squeeze(d, 3)
    assert d.ndim == 4
    # LPS -> RAS
    d[..., :2] *= -1
    return d, f.affine


def load_ants_aff(path):
    """Load ANTs affine and convert it into a RAS2RAS matrix.

    My understanding is that this matrix maps "fixed" coordinates into
    "moving" coordinates.
    """
    f = loadmat(path)
    mat33 = f.get(
        "AffineTransform_double_3_3",
        f.get("AffineTransform_float_3_3")
    )
    
    if mat33 is None:
        # List available keys for debugging
        available_keys = list(f.keys())
        raise KeyError(f"Could not find ANTs affine matrix. "
                      f"Available keys: {available_keys}. "
                      f"Expected 'AffineTransform_double_3_3' or 'AffineTransform_float_3_3'")
    
    if 'fixed' not in f:
        raise KeyError(f"Could not find 'fixed' key in ANTs affine file")
    
    mat = np.eye(4)
    mat[:3, :3] = mat33[:-3].reshape([3, 3])
    mat[:3, -1] = mat33[-3:].flatten()
    # center of rotation
    offset = f['fixed'].flatten()
    mat[:3, -1] += (np.eye(3) - mat[:3, :3]) @ offset
    # LPS2LPS -> RAS2RAS
    mat[:2, :] *= -1
    mat[:, :2] *= -1
    return mat


def load_trk(path):
    """Load a TRK and return the tractogram in RAS"""
    f = TrkFile.load(path)
    return f.streamlines, f.affine


def load_volume(path):
    """Load a nifti volume."""
    f = nib.load(path)
    return f.get_fdata(), f.affine


def apply_ants_transform_to_mri(path_warp, path_aff, path_mri, output_path=None):
    """
    Apply ANTs transforms to MRI data using the exact methodology provided.
    
    Parameters
    ----------
    path_warp : str
        Path to ANTs forward warp file
    path_aff : str
        Path to ANTs affine file
    path_mri : str
        Path to input MRI file
    output_path : str, optional
        Path to save transformed MRI
        
    Returns
    -------
    numpy.ndarray
        Transformed MRI data
    numpy.ndarray
        Affine matrix (vox2fix)
    """
    print("=== Applying ANTs Transform to MRI ===")
    
    # Load transforms
    affine_fix2mov = load_ants_aff(path_aff)
    warp, affine_vox2fix = load_ants_warp(path_warp)
    
    print(f"Loaded warp field shape: {warp.shape}")
    print(f"Fixed space affine (vox2fix):\n{affine_vox2fix}")
    
    # Load MRI
    dwi, affine_vox2mov = load_volume(path_mri)
    affine_mov2vox = np.linalg.inv(affine_vox2mov)
    
    print(f"Input MRI shape: {dwi.shape}")
    print(f"Moving space affine (vox2mov):\n{affine_vox2mov}")
    
    # Create full warp field - this is the exact implementation from your code
    fullwarp = np.stack(np.meshgrid(*[np.arange(x) for x in warp.shape[:3]], indexing='ij'), -1)
    fullwarp = fullwarp @ affine_vox2fix[:3, :3].T + affine_vox2fix[:3, -1]
    fullwarp = fullwarp + warp
    fullwarp = fullwarp @ affine_fix2mov[:3, :3].T + affine_fix2mov[:3, -1]
    fullwarp = fullwarp @ affine_mov2vox[:3, :3].T + affine_mov2vox[:3, -1]
    
    # Apply transformation
    moved = map_coordinates(dwi, np.moveaxis(fullwarp, -1, 0))
    
    print(f"Transformed MRI shape: {moved.shape}")
    
    # Save if requested
    if output_path:
        img = nib.Nifti1Image(moved, affine_vox2fix)
        nib.save(img, output_path)
        print(f"Saved transformed MRI to: {output_path}")
    
    return moved, affine_vox2fix


def apply_ants_transform_to_streamlines(path_iwarp, path_aff, path_trk, output_path=None):
    """
    Apply ANTs transforms to streamlines using the exact methodology provided.
    
    Parameters
    ----------
    path_iwarp : str
        Path to ANTs inverse warp file
    path_aff : str
        Path to ANTs affine file
    path_trk : str
        Path to input TRK file
    output_path : str, optional
        Path to save transformed TRK file
        
    Returns
    -------
    nibabel.streamlines.ArraySequence
        Transformed streamlines (in RAS coordinates)
    numpy.ndarray
        Streamlines in fixed voxel coordinates (for synthesis)
    """
    print("=== Applying ANTs Transform to Streamlines ===")
    
    # Load transforms
    affine_fix2mov = load_ants_aff(path_aff)
    iwarp, affine_vox2fix = load_ants_warp(path_iwarp)
    affine_fix2vox = np.linalg.inv(affine_vox2fix)
    
    print(f"Loaded inverse warp field shape: {iwarp.shape}")
    print(f"Fixed space affine (vox2fix):\n{affine_vox2fix}")
    
    # Load streamlines
    streamlines, trk_affine = load_trk(path_trk)
    streamstack = streamlines._data
    
    print(f"Loaded {len(streamlines)} streamlines")
    print(f"Total points: {len(streamstack)}")
    print(f"TRK affine:\n{trk_affine}")
    
    # Apply transformation - this is the exact implementation from your code
    
    # 1. apply inverse affine to streamlines
    affine_mov2fix = np.linalg.inv(affine_fix2mov)
    streamstack = streamstack @ affine_mov2fix[:3, :3].T + affine_mov2fix[:3, -1]
    
    # 2. convert streamline "fixed RAS" coordinates to "fixed voxels" coordinates
    streamstack = streamstack @ affine_fix2vox[:3, :3].T + affine_fix2vox[:3, -1]
    
    # 3. apply warp to streamlines (output is in fixed RAS)
    streamstack = (
        np.stack([
            map_coordinates(iwarp[..., 0], streamstack.T),
            map_coordinates(iwarp[..., 1], streamstack.T),
            map_coordinates(iwarp[..., 2], streamstack.T),
        ], -1) +
        streamstack @ affine_vox2fix[:3, :3].T +
        affine_vox2fix[:3, -1]
    )
    
    print(f"Applied warp transformation")
    
    # Save warped streamlines (in RAS coordinates)
    streamlines._data = streamstack
    
    if output_path:
        tract = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tract)
        trk.save(output_path)
        print(f"Saved transformed streamlines to: {output_path}")
    
    # Move streamlines to fixed voxel coordinates for synthesis
    streamstack_voxel = streamstack @ affine_fix2vox[:3, :3].T + affine_fix2vox[:3, -1]
    
    print(f"Converted to voxel coordinates for synthesis")
    print(f"Voxel coordinate range:")
    print(f"  X: {streamstack_voxel[:, 0].min():.2f} to {streamstack_voxel[:, 0].max():.2f}")
    print(f"  Y: {streamstack_voxel[:, 1].min():.2f} to {streamstack_voxel[:, 1].max():.2f}")
    print(f"  Z: {streamstack_voxel[:, 2].min():.2f} to {streamstack_voxel[:, 2].max():.2f}")
    
    return streamlines, streamstack_voxel


def process_with_ants_updated(
        path_warp,
        path_iwarp,
        path_aff,
        path_mri,
        path_trk,
        output_mri=None,
        output_trk=None,
        transform_mri=True,
):
    """
    Process MRI and streamlines using ANTs transforms with the updated methodology.
    
    This function exactly matches the workflow provided by the user.
    
    Parameters
    ----------
    path_warp : str
        Path to ANTs forward warp file
    path_iwarp : str
        Path to ANTs inverse warp file  
    path_aff : str
        Path to ANTs affine file
    path_mri : str
        Path to input MRI file
    path_trk : str
        Path to input TRK file
    output_mri : str, optional
        Path to save transformed MRI
    output_trk : str, optional
        Path to save transformed TRK
    transform_mri : bool, optional
        Whether to transform MRI (default: True)
        
    Returns
    -------
    tuple
        (moved_mri, affine_vox2fix, transformed_streamlines, streamlines_voxel)
        - moved_mri: Transformed MRI data (or None if transform_mri=False)
        - affine_vox2fix: Fixed space affine matrix
        - transformed_streamlines: Streamlines in RAS coordinates
        - streamlines_voxel: Streamlines in voxel coordinates (for synthesis)
    """
    print(f"=== Processing with ANTs (Updated Methodology) ===")
    print(f"Forward warp: {path_warp}")
    print(f"Inverse warp: {path_iwarp}")
    print(f"Affine: {path_aff}")
    print(f"MRI: {path_mri}")
    print(f"TRK: {path_trk}")
    print(f"Transform MRI: {transform_mri}")
    
    # Get fixed space affine from warp file
    warp_img = nib.load(path_warp)
    affine_vox2fix = warp_img.affine
    
    # Transform MRI if requested
    if transform_mri:
        moved_mri, affine_vox2fix = apply_ants_transform_to_mri(
            path_warp, path_aff, path_mri, output_mri
        )
    else:
        print("Skipping MRI transformation as requested")
        moved_mri = None
    
    # Transform streamlines
    transformed_streamlines, streamlines_voxel = apply_ants_transform_to_streamlines(
        path_iwarp, path_aff, path_trk, output_trk
    )
    
    # Convert voxel coordinates to list of individual streamlines
    streamlines_list = []
    offsets = transformed_streamlines._offsets
    
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        streamline_segment = streamlines_voxel[start:end]
        
        if len(streamline_segment) >= 2:  # Only keep streamlines with at least 2 points
            streamlines_list.append(streamline_segment.astype(np.float32))
    
    print(f"Final result: {len(streamlines_list)} valid streamlines for synthesis")
    
    return moved_mri, affine_vox2fix, transformed_streamlines, streamlines_list


# Compatibility function to replace the original process_with_ants
def process_with_ants(
        path_warp,
        path_iwarp,
        path_aff,
        path_mri,
        path_trk,
        output_mri=None,
        output_trk=None,
        transform_mri=False,
):
    """
    Compatibility wrapper for the updated ANTs processing.
    
    This maintains the same interface as the original function while using
    the updated methodology.
    """
    return process_with_ants_updated(
        path_warp=path_warp,
        path_iwarp=path_iwarp,
        path_aff=path_aff,
        path_mri=path_mri,
        path_trk=path_trk,
        output_mri=output_mri,
        output_trk=output_trk,
        transform_mri=transform_mri,
    )

