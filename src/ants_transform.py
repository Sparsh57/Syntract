import numpy as np
import nibabel as nib
from nibabel.streamlines.trk import TrkFile, Tractogram
from scipy.io import loadmat
from scipy.ndimage import map_coordinates
from nibabel.affines import apply_affine

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
    f = nib.as_closest_canonical(f)
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
    img = nib.load(path)
    img_ras = nib.as_closest_canonical(img)  # reorders/flips so it's RAS
    return img_ras.get_fdata(), img_ras.affine


def check_affine_orientation(affine):
    """
    Check the orientation of an affine matrix to determine if it's RAS or LPS.

    Parameters
    ----------
    affine : numpy.ndarray
        4x4 affine matrix

    Returns
    -------
    tuple
        Tuple containing:
        - bool: True if x-axis is flipped (not RAS)
        - bool: True if y-axis is flipped (not RAS)
        - bool: True if z-axis is flipped (not RAS)
    """
    # Check if the diagonal elements of the rotation part are negative
    # Negative values indicate flipped axes
    x_flipped = affine[0, 0] < 0
    y_flipped = affine[1, 1] < 0
    z_flipped = affine[2, 2] < 0

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
    # Load transforms
    affine_fix2mov = load_ants_aff(path_aff)
    warp, affine_vox2fix = load_ants_warp(path_warp)

    # Load MRI
    dwi, affine_vox2mov = load_volume(path_mri)
    affine_mov2vox = np.linalg.inv(affine_vox2mov)

    # Create full warp field
    fullwarp = np.stack(np.meshgrid(*[np.arange(x) for x in warp.shape[:3]], indexing='ij'), -1)
    fullwarp = fullwarp @ affine_vox2fix[:3, :3].T + affine_vox2fix[:3, -1]
    fullwarp = fullwarp + warp
    fullwarp = fullwarp @ affine_fix2mov[:3, :3].T + affine_fix2mov[:3, -1]
    fullwarp = fullwarp @ affine_mov2vox[:3, :3].T + affine_mov2vox[:3, -1]

    # Apply warp
    moved = map_coordinates(dwi, np.moveaxis(fullwarp, -1, 0))

    # Save if requested
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
    # Load transforms
    affine_fix2mov = load_ants_aff(path_aff)
    iwarp, affine_vox2fix = load_ants_warp(path_iwarp)
    affine_fix2vox = np.linalg.inv(affine_vox2fix)

    # Check orientations to help diagnose potential flipping issues
    fix_x_flip, fix_y_flip, fix_z_flip = check_affine_orientation(affine_vox2fix)
    print(f"Fixed space orientation: x-flipped: {fix_x_flip}, y-flipped: {fix_y_flip}, z-flipped: {fix_z_flip}")

    # Load streamlines
    streamlines, trk_affine = load_trk(path_trk)
    trk_x_flip, trk_y_flip, trk_z_flip = check_affine_orientation(trk_affine)
    print(f"TRK space orientation: x-flipped: {trk_x_flip}, y-flipped: {trk_y_flip}, z-flipped: {trk_z_flip}")

    streamstack = streamlines._data

    # Check if we need to flip x-axis to match orientation
    needs_x_flip = (fix_x_flip != trk_x_flip)
    needs_y_flip = (fix_y_flip != trk_y_flip)
    needs_z_flip = (fix_z_flip != trk_z_flip)

    # 1. Apply inverse affine to streamlines
    affine_mov2fix = np.linalg.inv(affine_fix2mov)
    streamstack = streamstack @ affine_mov2fix[:3, :3].T + affine_mov2fix[:3, -1]

    # 2. Convert streamline "fixed RAS" coordinates to "fixed voxels" coordinates
    streamstack = streamstack @ affine_fix2vox[:3, :3].T + affine_fix2vox[:3, -1]

    # 3. Apply warp to streamlines (output is in fixed RAS)
    warped_coords = np.stack([
        map_coordinates(iwarp[..., 0], streamstack.T),
        map_coordinates(iwarp[..., 1], streamstack.T),
        map_coordinates(iwarp[..., 2], streamstack.T),
    ], -1)
    
    vox2fix_transformed = streamstack @ affine_vox2fix[:3, :3].T + affine_vox2fix[:3, -1]
    streamstack = warped_coords + vox2fix_transformed

    # Apply axis flipping if needed - this corrects for 180-degree rotation
    if needs_x_flip:
        print("Applying x-axis flip to correct 180-degree rotation")
        # Determine the center of the volume in the fixed space
        fixed_shape = iwarp.shape[:3]
        fixed_center_vox = (np.array(fixed_shape) - 1) / 2  # center of the central voxel
        fixed_center_ras = apply_affine(affine_vox2fix, fixed_center_vox)
        
        # Flip around center
        streamstack[:, 0] = 2 * fixed_center_ras[0] - streamstack[:, 0]

    if needs_y_flip:
        print("Applying y-axis flip to correct orientation")
        # Determine the center of the volume in the fixed space
        fixed_shape = iwarp.shape[:3]
        fixed_center_vox = (np.array(fixed_shape) - 1) / 2  # center of the central voxel
        fixed_center_ras = apply_affine(affine_vox2fix, fixed_center_vox)
        
        # Flip around center
        streamstack[:, 1] = 2 * fixed_center_ras[1] - streamstack[:, 1]
    
    if needs_z_flip:
        print("Z-flip needed but not implemented in original code")

    # Save the transformed streamlines
    streamlines._data = streamstack

    # Save if requested
    if output_path:
        tract = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tract)
        trk.save(output_path)

    # Move streamlines back to fixed voxel coordinates for synthesis
    streamstack_for_synthesis = streamstack @ affine_fix2vox[:3, :3].T + affine_fix2vox[:3, -1]
    return streamlines, streamstack_for_synthesis


def process_with_ants(
        path_warp,
        path_iwarp,
        path_aff,
        path_mri,
        path_trk,
        output_mri=None,
        output_trk=None,
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

    Returns
    -------
    tuple
        Tuple containing:
        - Transformed MRI data
        - Affine matrix of the transformed MRI
        - Transformed tractogram
        - Streamlines in fixed voxel coordinates (for synthesis)
    """
    print("\n=== Applying ANTs Transforms ===")

    # Transform MRI
    print("\n=== Transforming MRI with ANTs ===")
    moved_mri, affine_vox2fix = apply_ants_transform_to_mri(
        path_warp, path_aff, path_mri, output_mri
    )
    if output_mri:
        print(f"Saved transformed MRI => {output_mri}")

    # Transform streamlines
    print("\n=== Transforming Streamlines with ANTs ===")
    transformed_streamlines, streamstack_for_synthesis = apply_ants_transform_to_streamlines(
        path_iwarp, path_aff, path_trk, output_trk
    )
    if output_trk:
        print(f"Saved transformed streamlines => {output_trk}")

    return moved_mri, affine_vox2fix, transformed_streamlines, streamstack_for_synthesis