import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.io.streamline import load_trk

def overlay_streamlines_on_blockface_coronal(
    trk_file,
    blockface_nifti_file,
    slice_idx=None,
    save_path=None
):
    nii_img = nib.load(blockface_nifti_file)
    nii_data = nii_img.get_fdata()
    shape = nii_data.shape

    tractogram = load_trk(trk_file, reference=nii_img, bbox_valid_check=False)
    
    # Handle case where load_trk returns False due to header mismatch
    if tractogram is False:
        print("Warning: TRK header doesn't match NIfTI reference. Loading TRK file directly...")
        # Load TRK file directly without reference check
        from nibabel.streamlines import load as nib_load_trk
        tractogram = nib_load_trk(trk_file)
    
    streamlines = tractogram.streamlines

    axis = 1
    if slice_idx is None:
        slice_idx = shape[axis] // 2
    bg_slice = nii_data[:, slice_idx, :]

    plt.figure(figsize=(6, 6))
    plt.imshow(bg_slice.T, cmap='gray', origin='lower')

    affine_inv = np.linalg.inv(nii_img.affine)

    for s in streamlines:
        s_vox = nib.affines.apply_affine(affine_inv, s)
        x = s_vox[:, 0]
        z = s_vox[:, 2]
        y = s_vox[:, 1]
        if np.any(np.abs(y - slice_idx) < 2):
            plt.plot(x, z, color='yellow', linewidth=0.5, alpha=0.7)

    plt.title(f"Streamlines over blockface (coronal slice {slice_idx})")
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_trk_with_nifti(trk_file, nifti_file, save_path=None):
    """
    Visualize streamlines from a .trk file overlaid on a corresponding NIfTI image using MIP views.

    Args:
        trk_file (str): Path to the .trk file containing streamlines.
        nifti_file (str): Path to the corresponding NIfTI file.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()
    
    tractogram = load_trk(trk_file, reference=nii_img, bbox_valid_check=False)
    
    # Handle case where load_trk returns False due to header mismatch
    if tractogram is False:
        print("Warning: TRK header doesn't match NIfTI reference. Loading TRK file directly...")
        # Load TRK file directly without reference check
        from nibabel.streamlines import load as nib_load_trk
        tractogram = nib_load_trk(trk_file)
        
        # When loading directly, streamlines are in the space defined by the TRK header
        # We need to transform them to the NIfTI's coordinate space for visualization
        trk_affine = tractogram.header.get('voxel_to_rasmm', np.eye(4))
        nii_affine = nii_img.affine
        
        # Check if coordinate spaces are different
        if not np.allclose(trk_affine, nii_affine, atol=1e-3):
            print(f"Converting streamlines from TRK space to NIfTI space...")
            
            # Handle the case where TRK uses identity matrix (streamlines in world coords)
            # and NIfTI uses a real affine (slice coordinate system)
            if np.allclose(trk_affine, np.eye(4), atol=1e-3):
                # TRK streamlines are in world coordinates, transform to NIfTI voxel space
                nii_affine_inv = np.linalg.inv(nii_affine)
                
                transformed_streamlines = []
                for stream in tractogram.streamlines:
                    # Transform from world to NIfTI voxel coordinates
                    homogeneous = np.hstack([stream, np.ones((len(stream), 1))])
                    voxel_coords = homogeneous @ nii_affine_inv.T
                    transformed_streamlines.append(voxel_coords[:, :3])
                
                # Create new tractogram with transformed streamlines
                from nibabel.streamlines.tractogram import Tractogram
                tractogram = Tractogram(transformed_streamlines, affine_to_rasmm=nii_affine)
                
            else:
                # General case: transform between two non-identity affines
                transform_matrix = np.linalg.inv(nii_affine) @ trk_affine
                
                transformed_streamlines = []
                for stream in tractogram.streamlines:
                    # Apply transformation to each streamline
                    homogeneous = np.hstack([stream, np.ones((len(stream), 1))])
                    transformed = homogeneous @ transform_matrix.T
                    transformed_streamlines.append(transformed[:, :3])
                
                # Create new tractogram with transformed streamlines
                from nibabel.streamlines.tractogram import Tractogram
                tractogram = Tractogram(transformed_streamlines, affine_to_rasmm=nii_affine)
    
    streamlines = tractogram.streamlines
    
    sagittal_mip = np.max(nii_data, axis=0)
    coronal_mip = np.max(nii_data, axis=1)
    axial_mip = np.max(nii_data, axis=2)

    affine_inv = np.linalg.inv(nii_img.affine)
    streamlines_vox = []
    
    # Check if this is a single slice (Y dimension = 1)
    is_single_slice = nii_data.shape[1] == 1
    
    for s in streamlines:
        s_vox = np.dot(affine_inv[:3, :3], s.T).T + affine_inv[:3, 3]
        
        # For single slice visualization, project all streamlines to the slice plane (Y=0)
        if is_single_slice:
            s_vox[:, 1] = 0.0  # Project all Y coordinates to the slice plane
            
        streamlines_vox.append(s_vox)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.rot90(sagittal_mip), cmap='gray', interpolation='nearest')
    for s in streamlines_vox:
        plt.plot(s[:, 1], nii_data.shape[2] - s[:, 2], color='yellow', linewidth=0.5)
    plt.title('Sagittal MIP')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(coronal_mip), cmap='gray', interpolation='nearest')
    for s in streamlines_vox:
        plt.plot(s[:, 0], nii_data.shape[2] - s[:, 2], color='yellow', linewidth=0.5)
    plt.title('Coronal MIP')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(axial_mip), cmap='gray', interpolation='nearest')
    for s in streamlines_vox:
        plt.plot(s[:, 0], nii_data.shape[1] - s[:, 1], color='yellow', linewidth=0.5)
    plt.title('Axial MIP')
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        visualize_trk_with_nifti(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python visualize.py <trk_file> <nifti_file>")
        print("Example: python visualize.py slice_002_streamlines.trk slice_002.nii.gz")