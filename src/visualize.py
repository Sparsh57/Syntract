import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.io.streamline import load_trk


def visualize_trk_with_nifti(trk_file, nifti_file, save_path=None):
    """
    Visualize streamlines from a .trk file overlaid on a corresponding NIfTI image using MIP views.

    Args:
        trk_file (str): Path to the .trk file containing streamlines.
        nifti_file (str): Path to the corresponding NIfTI file.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    # Load streamlines
    tractogram = load_trk(trk_file, reference=nifti_file, bbox_valid_check=False)
    streamlines = tractogram.streamlines

    # Load NIfTI image
    nii_img = nib.load(nifti_file)
    nii_data = nii_img.get_fdata()

    # Compute MIPs along each axis for background
    sagittal_mip = np.max(nii_data, axis=0)
    coronal_mip = np.max(nii_data, axis=1)
    axial_mip = np.max(nii_data, axis=2)

    # Convert streamlines to voxel coordinates
    affine_inv = np.linalg.inv(nii_img.affine)
    streamlines_vox = [np.dot(affine_inv[:3, :3], s.T).T + affine_inv[:3, 3] for s in streamlines]

    # Plot MIPs with streamlines
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


# Example usage:
visualize_trk_with_nifti("MRISynth/mri_block.trk", "MRISynth/mri_block.nii.gz", save_path="mri_visualization.png")
