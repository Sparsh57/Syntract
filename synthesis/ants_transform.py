import numpy as np
import nibabel as nib
from nibabel.streamlines.trk import TrkFile, Tractogram
from scipy.io import loadmat
from scipy.ndimage import map_coordinates
from nibabel.affines import apply_affine

def load_ants_warp(path):
    """Load ANTs warp and convert it into a RAS displacement."""
    f = nib.load(path)
    f = nib.as_closest_canonical(f)
    d = f.get_fdata()

    d = np.squeeze(d, 3)
    assert d.ndim == 4
    d[..., :2] *= -1
    return d, f.affine


def load_ants_aff(path):
    """Load ANTs affine and convert it into a RAS2RAS matrix."""
    f = loadmat(path)
    
    # Try common ANTs affine matrix key names
    mat33 = f.get("AffineTransform_double_3_3", 
                  f.get("AffineTransform_float_3_3", None))
    
    if mat33 is None:
        # List available keys for debugging
        available_keys = list(f.keys())
        raise KeyError(f"Could not find ANTs affine matrix in {path}. "
                      f"Available keys: {available_keys}. "
                      f"Expected 'AffineTransform_double_3_3' or 'AffineTransform_float_3_3'")
    
    if 'fixed' not in f:
        raise KeyError(f"Could not find 'fixed' key in ANTs affine file {path}")
    
    mat = np.eye(4)
    mat[:3, :3] = mat33[:-3].reshape([3, 3])
    mat[:3, -1] = mat33[-3:].flatten()
    
    offset = f['fixed'].flatten()
    mat[:3, -1] += (np.eye(3) - mat[:3, :3]) @ offset
    mat[:2, :] *= -1
    mat[:, :2] *= -1
    return mat


def load_trk(path):
    """Load a TRK and return the tractogram in RAS"""
    f = TrkFile.load(path)
    return f.streamlines, f.affine


def load_volume(path):
    """Load a nifti volume."""
    img = nib.load(path)
    img_ras = nib.as_closest_canonical(img)
    return img_ras.get_fdata(), img_ras.affine


def check_affine_orientation(affine):
    """
    Check the orientation of an affine matrix using nibabel orientation codes.

    Parameters
    ----------
    affine : numpy.ndarray
        4x4 affine matrix

    Returns
    -------
    tuple
        Tuple containing (x_flipped, y_flipped, z_flipped) booleans
    """
    # Use nibabel's robust orientation detection
    ornt = nib.orientations.io_orientation(affine)
    
    # Extract flip information from orientation
    # ornt is array of [axis, flip] pairs where flip is 1 or -1
    x_flipped = ornt[0, 1] < 0
    y_flipped = ornt[1, 1] < 0 
    z_flipped = ornt[2, 1] < 0

    return x_flipped, y_flipped, z_flipped


def apply_ants_transform_to_mri(path_warp, path_aff, path_mri, output_path=None):
    """
    Apply ANTs transforms to MRI data.

    Parameters
    ----------
    path_warp : str
        Path to ANTs warp file.
    path_aff : str
        Path to ANTs affine file.
    path_mri : str
        Path to MRI file.
    output_path : str, optional
        Path to save transformed MRI file. If None, doesn't save.

    Returns
    -------
    numpy.ndarray
        Transformed MRI data.
    numpy.ndarray
        Affine matrix of the transformed MRI.
    """
    affine_fix2mov = load_ants_aff(path_aff)
    warp, affine_vox2fix = load_ants_warp(path_warp)

    dwi, affine_vox2mov = load_volume(path_mri)
    affine_mov2vox = np.linalg.inv(affine_vox2mov)

    fullwarp = np.stack(np.meshgrid(*[np.arange(x) for x in warp.shape[:3]], indexing='ij'), -1)
    fullwarp = fullwarp @ affine_vox2fix[:3, :3].T + affine_vox2fix[:3, -1]
    fullwarp = fullwarp + warp
    fullwarp = fullwarp @ affine_fix2mov[:3, :3].T + affine_fix2mov[:3, -1]
    fullwarp = fullwarp @ affine_mov2vox[:3, :3].T + affine_mov2vox[:3, -1]

    moved = map_coordinates(dwi, np.moveaxis(fullwarp, -1, 0))

    if output_path:
        img = nib.Nifti1Image(moved, affine_vox2fix)
        nib.save(img, output_path)

    return moved, affine_vox2fix


def apply_ants_transform_to_streamlines(path_iwarp, path_aff, path_trk, output_path=None):
    """
    Apply ANTs transforms to streamlines.

    Parameters
    ----------
    path_iwarp : str
        Path to ANTs inverse warp file.
    path_aff : str
        Path to ANTs affine file.
    path_trk : str
        Path to TRK file.
    output_path : str, optional
        Path to save transformed TRK file. If None, doesn't save.

    Returns
    -------
    nibabel.streamlines.tractogram.Tractogram
        Transformed tractogram.
    numpy.ndarray
        Streamlines in fixed voxel coordinates (for synthesis).
    """
    affine_fix2mov = load_ants_aff(path_aff)
    iwarp, affine_vox2fix = load_ants_warp(path_iwarp)
    affine_fix2vox = np.linalg.inv(affine_vox2fix)

    fix_x_flip, fix_y_flip, fix_z_flip = check_affine_orientation(affine_vox2fix)

    streamlines, trk_affine = load_trk(path_trk)
    trk_x_flip, trk_y_flip, trk_z_flip = check_affine_orientation(trk_affine)

    streamstack = streamlines._data

    needs_x_flip = (fix_x_flip != trk_x_flip)
    needs_y_flip = (fix_y_flip != trk_y_flip)
    needs_z_flip = (fix_z_flip != trk_z_flip)

    affine_mov2fix = np.linalg.inv(affine_fix2mov)
    streamstack = streamstack @ affine_mov2fix[:3, :3].T + affine_mov2fix[:3, -1]

    streamstack = streamstack @ affine_fix2vox[:3, :3].T + affine_fix2vox[:3, -1]

    warped_coords = np.stack([
        map_coordinates(iwarp[..., 0], streamstack.T),
        map_coordinates(iwarp[..., 1], streamstack.T),
        map_coordinates(iwarp[..., 2], streamstack.T),
    ], -1)
    
    vox2fix_transformed = streamstack @ affine_vox2fix[:3, :3].T + affine_vox2fix[:3, -1]
    streamstack = warped_coords + vox2fix_transformed

    if needs_x_flip:
        fixed_shape = iwarp.shape[:3]
        fixed_center_vox = (np.array(fixed_shape) - 1) / 2
        fixed_center_ras = apply_affine(affine_vox2fix, fixed_center_vox)
        streamstack[:, 0] = 2 * fixed_center_ras[0] - streamstack[:, 0]

    if needs_y_flip:
        fixed_shape = iwarp.shape[:3]
        fixed_center_vox = (np.array(fixed_shape) - 1) / 2
        fixed_center_ras = apply_affine(affine_vox2fix, fixed_center_vox)
        streamstack[:, 1] = 2 * fixed_center_ras[1] - streamstack[:, 1]
    
    if needs_z_flip:
        fixed_shape = iwarp.shape[:3]
        fixed_center_vox = (np.array(fixed_shape) - 1) / 2
        fixed_center_ras = apply_affine(affine_vox2fix, fixed_center_vox)
        streamstack[:, 2] = 2 * fixed_center_ras[2] - streamstack[:, 2]

    streamlines._data = streamstack

    if output_path:
        tract = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tract)
        trk.save(output_path)

    streamstack_for_synthesis = streamstack @ affine_fix2vox[:3, :3].T + affine_fix2vox[:3, -1]
    
    fixed_shape = iwarp.shape[:3]
    streamlines_list = []
    offsets = streamlines._offsets
    
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        streamline_segment = streamstack_for_synthesis[start:end]
        
        if len(streamline_segment) < 2:
            continue
        
        # Use smart clipping that preserves pass-through streamlines
        inside_mask = np.all((streamline_segment >= 0) & (streamline_segment < fixed_shape), axis=1)
        
        if not np.any(inside_mask):
            continue
        
        # Import the smart clipping function from streamline_processing
        try:
            from synthesis.streamline_processing import clip_streamline_to_fov
        except ImportError:
            try:
                from .streamline_processing import clip_streamline_to_fov
            except ImportError:
                from streamline_processing import clip_streamline_to_fov
        
        # Apply smart clipping for ANTs-processed streamlines
        clipped_segments = clip_streamline_to_fov(streamline_segment, fixed_shape, use_gpu=False)
        
        # Add all valid segments
        for segment in clipped_segments:
            if len(segment) >= 2:
                streamlines_list.append(segment.astype(np.float32))
    
    if len(streamlines_list) == 0:
        print("WARNING: No valid streamlines found after ANTs transformation and clipping!")
    
    return streamlines, streamlines_list


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
    Process MRI and streamlines using ANTs transforms.

    Parameters
    ----------
    path_warp : str
        Path to ANTs warp file.
    path_iwarp : str
        Path to ANTs inverse warp file.
    path_aff : str
        Path to ANTs affine file.
    path_mri : str
        Path to MRI file.
    path_trk : str
        Path to TRK file.
    output_mri : str, optional
        Path to save transformed MRI file.
    output_trk : str, optional
        Path to save transformed TRK file.
    transform_mri : bool, optional
        Whether to transform the MRI (default: False).

    Returns
    -------
    tuple
        Tuple containing:
        - Transformed MRI data (or None if transform_mri=False)
        - Affine matrix of the transformed MRI
        - Transformed tractogram
        - Streamlines in fixed voxel coordinates (for synthesis)
    """
    warp_img = nib.load(path_warp)
    warp_dims = warp_img.shape[:3]
    warp_affine = warp_img.affine

    if transform_mri:
        moved_mri, affine_vox2fix = apply_ants_transform_to_mri(
            path_warp, path_aff, path_mri, output_mri
        )
        
        if moved_mri.shape[:3] != warp_dims:
            print(f"WARNING: Transformed MRI dimensions {moved_mri.shape[:3]} don't match warp dimensions {warp_dims}")
    else:
        moved_mri = None
        orig_img = nib.load(path_mri)
        affine_vox2fix = orig_img.affine

    transformed_streamlines, streamlines_list = apply_ants_transform_to_streamlines(
        path_iwarp, path_aff, path_trk, output_trk
    )

    return moved_mri, affine_vox2fix, transformed_streamlines, streamlines_list