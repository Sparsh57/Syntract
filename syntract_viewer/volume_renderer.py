#!/usr/bin/env python
"""
Generate 3D NIfTI volumes with streamlines overlaid.
Applies dark field microscopy-style rendering with subtle fiber visualization.
"""

import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_trk
from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage import gaussian_filter, zoom
from tqdm import tqdm
import os
import sys
import gc

# GPU support with fallback
try:
    from synthesis.gpu_utils import try_gpu_import
    gpu_result = try_gpu_import()
    xp = gpu_result['xp']
    use_gpu = gpu_result['cupy_available']
    if use_gpu:
        import cupy as cp
        import cupyx.scipy.ndimage as gpu_ndimage
        print(f"GPU acceleration enabled for 3D rendering ({gpu_result.get('gpu_name', 'unknown')})")
    else:
        cp = None
        from scipy.ndimage import gaussian_filter
        print("GPU not available, using CPU for 3D rendering")
except ImportError:
    xp = np
    use_gpu = False
    from scipy.ndimage import gaussian_filter
    print("GPU utils not found, using CPU for 3D rendering")
    cp = None

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from syntract_viewer.volumetric_3d import process_volume_full_3d
except ImportError:
    from volumetric_3d import process_volume_full_3d


_GPU_LINE_KERNEL_WITH_MASK = None
_GPU_LINE_KERNEL_NO_MASK = None
_GPU_LINE_KERNEL_DISABLED = False
_GPU_LINE_KERNEL_ERROR = None


def _ensure_gpu_line_kernels():
    global _GPU_LINE_KERNEL_WITH_MASK, _GPU_LINE_KERNEL_NO_MASK
    global _GPU_LINE_KERNEL_DISABLED, _GPU_LINE_KERNEL_ERROR
    if cp is None:
        return False
    if _GPU_LINE_KERNEL_DISABLED:
        return False
    if _GPU_LINE_KERNEL_WITH_MASK is not None and _GPU_LINE_KERNEL_NO_MASK is not None:
        return True

    kernel_with_mask = r'''
    extern "C" __global__
    void render_lines_with_mask(float* volume, float* mask, const int* p0s, const int* p1s,
                                const int nseg, const float intensity, const float tissue_threshold,
                                const int sx, const int sy, const int sz) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= nseg) return;

        int base = idx * 3;
        int x0 = p0s[base + 0], y0 = p0s[base + 1], z0 = p0s[base + 2];
        int x1 = p1s[base + 0], y1 = p1s[base + 1], z1 = p1s[base + 2];

        int dx = abs(x1 - x0), dy = abs(y1 - y0), dz = abs(z1 - z0);
        int xs = (x1 > x0) ? 1 : -1;
        int ys = (y1 > y0) ? 1 : -1;
        int zs = (z1 > z0) ? 1 : -1;

        int x = x0, y = y0, z = z0;
        int d_axis, p1_d, p2_d, p1_err, p2_err;

        if (dx >= dy && dx >= dz) {
            d_axis = dx; p1_d = dy; p2_d = dz;
            p1_err = d_axis / 2; p2_err = d_axis / 2;
            for (int t = 0; t <= d_axis; ++t) {
                if ((unsigned)x < (unsigned)sx && (unsigned)y < (unsigned)sy && (unsigned)z < (unsigned)sz) {
                    int off = (x * sy + y) * sz + z;
                    if (volume[off] >= tissue_threshold) {
                        atomicAdd(&volume[off], intensity);
                        atomicAdd(&mask[off], 1.0f);
                    }
                }
                p1_err += p1_d;
                if (p1_err >= d_axis) { y += ys; p1_err -= d_axis; }
                p2_err += p2_d;
                if (p2_err >= d_axis) { z += zs; p2_err -= d_axis; }
                x += xs;
            }
        } else if (dy >= dx && dy >= dz) {
            d_axis = dy; p1_d = dx; p2_d = dz;
            p1_err = d_axis / 2; p2_err = d_axis / 2;
            for (int t = 0; t <= d_axis; ++t) {
                if ((unsigned)x < (unsigned)sx && (unsigned)y < (unsigned)sy && (unsigned)z < (unsigned)sz) {
                    int off = (x * sy + y) * sz + z;
                    if (volume[off] >= tissue_threshold) {
                        atomicAdd(&volume[off], intensity);
                        atomicAdd(&mask[off], 1.0f);
                    }
                }
                p1_err += p1_d;
                if (p1_err >= d_axis) { x += xs; p1_err -= d_axis; }
                p2_err += p2_d;
                if (p2_err >= d_axis) { z += zs; p2_err -= d_axis; }
                y += ys;
            }
        } else {
            d_axis = dz; p1_d = dx; p2_d = dy;
            p1_err = d_axis / 2; p2_err = d_axis / 2;
            for (int t = 0; t <= d_axis; ++t) {
                if ((unsigned)x < (unsigned)sx && (unsigned)y < (unsigned)sy && (unsigned)z < (unsigned)sz) {
                    int off = (x * sy + y) * sz + z;
                    if (volume[off] >= tissue_threshold) {
                        atomicAdd(&volume[off], intensity);
                        atomicAdd(&mask[off], 1.0f);
                    }
                }
                p1_err += p1_d;
                if (p1_err >= d_axis) { x += xs; p1_err -= d_axis; }
                p2_err += p2_d;
                if (p2_err >= d_axis) { y += ys; p2_err -= d_axis; }
                z += zs;
            }
        }
    }
    '''

    kernel_no_mask = r'''
    extern "C" __global__
    void render_lines_no_mask(float* volume, const int* p0s, const int* p1s,
                              const int nseg, const float intensity, const float tissue_threshold,
                              const int sx, const int sy, const int sz) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= nseg) return;

        int base = idx * 3;
        int x0 = p0s[base + 0], y0 = p0s[base + 1], z0 = p0s[base + 2];
        int x1 = p1s[base + 0], y1 = p1s[base + 1], z1 = p1s[base + 2];

        int dx = abs(x1 - x0), dy = abs(y1 - y0), dz = abs(z1 - z0);
        int xs = (x1 > x0) ? 1 : -1;
        int ys = (y1 > y0) ? 1 : -1;
        int zs = (z1 > z0) ? 1 : -1;

        int x = x0, y = y0, z = z0;
        int d_axis, p1_d, p2_d, p1_err, p2_err;

        if (dx >= dy && dx >= dz) {
            d_axis = dx; p1_d = dy; p2_d = dz;
            p1_err = d_axis / 2; p2_err = d_axis / 2;
            for (int t = 0; t <= d_axis; ++t) {
                if ((unsigned)x < (unsigned)sx && (unsigned)y < (unsigned)sy && (unsigned)z < (unsigned)sz) {
                    int off = (x * sy + y) * sz + z;
                    if (volume[off] >= tissue_threshold) {
                        atomicAdd(&volume[off], intensity);
                    }
                }
                p1_err += p1_d;
                if (p1_err >= d_axis) { y += ys; p1_err -= d_axis; }
                p2_err += p2_d;
                if (p2_err >= d_axis) { z += zs; p2_err -= d_axis; }
                x += xs;
            }
        } else if (dy >= dx && dy >= dz) {
            d_axis = dy; p1_d = dx; p2_d = dz;
            p1_err = d_axis / 2; p2_err = d_axis / 2;
            for (int t = 0; t <= d_axis; ++t) {
                if ((unsigned)x < (unsigned)sx && (unsigned)y < (unsigned)sy && (unsigned)z < (unsigned)sz) {
                    int off = (x * sy + y) * sz + z;
                    if (volume[off] >= tissue_threshold) {
                        atomicAdd(&volume[off], intensity);
                    }
                }
                p1_err += p1_d;
                if (p1_err >= d_axis) { x += xs; p1_err -= d_axis; }
                p2_err += p2_d;
                if (p2_err >= d_axis) { z += zs; p2_err -= d_axis; }
                y += ys;
            }
        } else {
            d_axis = dz; p1_d = dx; p2_d = dy;
            p1_err = d_axis / 2; p2_err = d_axis / 2;
            for (int t = 0; t <= d_axis; ++t) {
                if ((unsigned)x < (unsigned)sx && (unsigned)y < (unsigned)sy && (unsigned)z < (unsigned)sz) {
                    int off = (x * sy + y) * sz + z;
                    if (volume[off] >= tissue_threshold) {
                        atomicAdd(&volume[off], intensity);
                    }
                }
                p1_err += p1_d;
                if (p1_err >= d_axis) { x += xs; p1_err -= d_axis; }
                p2_err += p2_d;
                if (p2_err >= d_axis) { y += ys; p2_err -= d_axis; }
                z += zs;
            }
        }
    }
    '''

    try:
        _GPU_LINE_KERNEL_WITH_MASK = cp.RawKernel(kernel_with_mask, 'render_lines_with_mask')
        _GPU_LINE_KERNEL_NO_MASK = cp.RawKernel(kernel_no_mask, 'render_lines_no_mask')
        return True
    except Exception as e:
        _GPU_LINE_KERNEL_DISABLED = True
        _GPU_LINE_KERNEL_ERROR = str(e)
        _GPU_LINE_KERNEL_WITH_MASK = None
        _GPU_LINE_KERNEL_NO_MASK = None
        return False


def add_cell_body_blobs(volume, n_blobs, intensity_scale, radius_range_vox,
                        rng, bright_fraction=0.85):
    """Scatter Gaussian cell-body-like blobs into the tissue IMAGE (distractors).

    These are NOT added to the fiber mask. They mimic the punctate somatic
    fluorescence that dominates real light-sheet tissue, so the trained model
    must learn to distinguish fibers from cell bodies rather than just firing
    on any bright structure.
    """
    if n_blobs <= 0:
        return volume
    sx, sy, sz = volume.shape
    vmax = float(np.percentile(volume, 99)) if volume.max() > 0 else 1.0
    for _ in range(int(n_blobs)):
        cx = int(rng.integers(0, sx)); cy = int(rng.integers(0, sy)); cz = int(rng.integers(0, sz))
        r = float(rng.uniform(*radius_range_vox))
        rr = int(np.ceil(r * 3))
        x0, x1 = max(0, cx - rr), min(sx, cx + rr + 1)
        y0, y1 = max(0, cy - rr), min(sy, cy + rr + 1)
        z0, z1 = max(0, cz - rr), min(sz, cz + rr + 1)
        xx, yy, zz = np.mgrid[x0:x1, y0:y1, z0:z1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        g = np.exp(-d2 / (2.0 * r * r)).astype(np.float32)
        # Mostly bright cell bodies; a minority dark (voids) for variety.
        sign = 1.0 if rng.random() < bright_fraction else -0.5
        amp = sign * intensity_scale * vmax * float(rng.uniform(0.5, 1.0))
        volume[x0:x1, y0:y1, z0:z1] += amp * g
    return np.clip(volume, 0.0, None)


def create_3d_volume_with_streamlines(nifti_file, trk_file, output_file,
                                       slice_range=None, orientation='coronal',
                                       white_mask_path=None, contrast_method='clahe',
                                       gamma=2.2, scaling_factor=40.0,
                                       fiber_intensity_min=15.0, fiber_intensity_max=25.0,
                                       fiber_max_boost=None,
                                       fiber_opacity=1.0,
                                       fiber_smoothing_sigma=0.0,
                                       fiber_antialias=False,
                                       min_streamlines_rendered=None,
                                       fiber_brightness_variation=0.35,
                                       fiber_segment_brightness_variation=0.15,
                                       fiber_render_mode='additive',
                                       fiber_density_gamma=2.0,
                                       fiber_min_visibility=0.15,
                                       fiber_target_intensity=38.0,
                                       background_max_intensity=None,
                                       tissue_threshold=2.0,
                                       min_bundle_size=None,
                                       use_cornucopia_3d=True,
                                       cornucopia_allowed_presets=None,
                                       cornucopia_prob=0.9,
                                       use_bilateral_smoothing=True,
                                       texture_intensity=0.02,
                                       texture_sigma=8.0,
                                       clahe_clip_limit=0.01,
                                       enable_cell_blobs=False,
                                       cell_blob_count=60,
                                       cell_blob_intensity=0.3,
                                       cell_blob_radius_range=(1.5, 4.0),
                                       enable_tissue_artifacts=False,
                                       enable_granular_noise=False,
                                       enable_speckle_noise=False,
                                       enable_dash_noise=False,
                                       enable_horizontal_banding=False,
                                       enable_poisson_noise=False,
                                       artifact_strength=0.45,
                                       granular_noise_strength=0.35,
                                       poisson_gain=80.0,
                                       speckle_noise_strength=0.70,
                                       speckle_noise_density=0.008,
                                       speckle_noise_sigma=0.0,
                                       speckle_square_size=2,
                                       dash_noise_strength=0.55,
                                       dash_noise_density=0.0005,
                                       dash_length_sigma=4.0,
                                       dash_cross_sigma=0.3,
                                       banding_strength=0.18,
                                       banding_axis=1,
                                       random_state=None,
                                       save_mask=True,
                                       mask_smoothing_sigma=0.0,
                                       mask_binary_threshold=0.01,
                                       soft_mask=False,
                                       use_gpu=True,
                                       verbose=True,
                                       save_outputs=True,
                                       return_arrays=False):
    """
    Create 3D NIfTI volume with streamlines rendered as subtle fibers on dark tissue.
    
    Uses TRUE 3D CLAHE - processes entire volume as single unit, no tiling artifacts.
    Mask generation creates THIN, ACCURATE streamline representations (ground truth),
    not thick segmentation training masks like the 2D pipeline.
    
    Parameters
    ----------
    nifti_file : str
        Path to input NIfTI file
    trk_file : str
        Path to TRK streamline file
    output_file : str
        Path for output 3D NIfTI volume
    slice_range : range, optional
        Slice indices to process (default: all slices)
    orientation : str
        Orientation: 'coronal', 'axial', or 'sagittal'
    white_mask_path : str, optional
        Path to white matter mask for filtering streamlines
    contrast_method : str
        Contrast enhancement: 'clahe' or 'none'
    gamma : float
        Gamma correction for dark field appearance (default: 2.2)
    scaling_factor : float
        Output intensity scaling (default: 40.0 for dark field appearance - dark tissue, bright fibers)
    fiber_intensity_min : float
        Minimum streamline intensity (default: 15.0 for bright white fibers)
    fiber_intensity_max : float
        Maximum streamline intensity (default: 25.0 for bright white fibers)
    fiber_max_boost : float, optional
        Maximum added fiber brightness above local tissue after rendering. This
        prevents overlapping streamlines from saturating into white bands.
    fiber_opacity : float
        Multiplier applied to added fiber brightness after optional capping.
        Lower values make fibers blend into tissue.
    fiber_smoothing_sigma : float
        Small Gaussian smoothing applied only to the rendered fiber brightness
        boost. This reduces voxel stair-stepping without blurring tissue.
    fiber_antialias : bool
        Render CPU streamlines with subvoxel trilinear splats instead of
        integer Bresenham lines. This reduces blocky stair-steps.
    min_streamlines_rendered : int, optional
        If set, randomly render at most this many streamlines. Dense local
        bundles can otherwise merge into a solid block.
    fiber_brightness_variation : float
        Multiplicative per-streamline brightness variation. This changes only
        the image signal, not the generated mask.
    fiber_segment_brightness_variation : float
        Extra per-segment brightness jitter inside a streamline. This creates
        natural bright/dim fragments while keeping the same geometry mask.
    fiber_render_mode : {'additive', 'density', 'embedded'}
        Add fibers directly to the tissue volume, or compose them from a
        separate normalized density map after line rasterization. embedded
        keeps fibers as low-contrast tissue texture rather than painted labels.
    fiber_density_gamma : float
        Gamma applied to normalized fiber density in density mode. Higher
        values keep only centerlines and make bundles thinner.
    fiber_min_visibility : float
        Minimum normalized image visibility for voxels that contain a rendered
        fiber. This prevents sparse streamlines from disappearing after density
        gamma while leaving the mask geometry unchanged.
    fiber_target_intensity : float
        Target white level for fiber centerlines in density mode.
    background_max_intensity : float, optional
        Cap tissue/background before fiber composition. This prevents bright
        white-matter blocks from outshining streamlines.
    tissue_threshold : float
        Minimum tissue intensity for streamline rendering (default: 2.0)
    min_bundle_size : int, optional
        Minimum bundle size for filtering
    use_cornucopia_3d : bool
        Enable cornucopia 3D augmentation with aggressive presets (default: False)
    cornucopia_allowed_presets : list, optional
        List of allowed presets (default: 4 aggressive presets)
    cornucopia_prob : float
        Probability of applying cornucopia (default: 0.9)
    enable_tissue_artifacts : bool
        Add image-only tissue-like artefact clutter after streamline rendering.
    enable_granular_noise : bool
        Add image-only fine Cornucopia granular noise after streamline rendering.
    enable_speckle_noise : bool
        Add sparse image-only whitish-grey dot artefacts. These dots are not
        added to the mask, so they act as hard negatives for fiber learning.
    artifact_strength : float
        Strength multiplier for tissue-like artefacts.
    granular_noise_strength : float
        Strength multiplier for fine granular noise.
    speckle_noise_strength : float
        Strength multiplier for sparse dot artefacts.
    speckle_noise_density : float
        Probability of a dot seed per tissue voxel.
    speckle_noise_sigma : float
        Gaussian sigma for each dot. Values below 1.0 keep dots small.
    random_state : int, optional
        Random seed for reproducibility
    save_mask : bool
        If True, save fiber mask as separate NIfTI file (default: True).
    mask_smoothing_sigma : float
        3D Gaussian smoothing sigma for the generated fiber mask. Lower values
        produce thinner masks.
    mask_binary_threshold : float
        Threshold after mask-density normalization. Higher values produce thinner masks.
    use_gpu : bool
        Whether to use GPU acceleration for processing (default: True)
    save_outputs : bool
        If True, write output NIfTI files to disk (default: True)
    return_arrays : bool
        If True, return in-memory arrays instead of only output path
    
    Returns
    -------
    str or tuple[np.ndarray, np.ndarray | None]
        Path to saved 3D NIfTI volume file when return_arrays=False.
        Otherwise returns (volume, mask_or_none).
    """
    def _vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Load NIfTI with memory-mapped mode for large files
    # This prevents loading entire volume into memory at once
    nii_img = nib.load(nifti_file, mmap=True)
    nii_img = nib.as_closest_canonical(nii_img)
    nii_data = nii_img.get_fdata()
    dims = nii_data.shape
    
    _vprint(f"Input NIfTI dimensions: {dims}")
    
    # Load white matter mask if provided and resample to match data (following 2D methodology)
    white_mask_data = None
    if white_mask_path and os.path.exists(white_mask_path):
        try:
            _vprint(f"Loading white matter mask: {white_mask_path}")
            white_mask_img = nib.load(white_mask_path, mmap=True)
            white_mask_img = nib.as_closest_canonical(white_mask_img)
            white_mask_orig = white_mask_img.get_fdata()
            
            # Remove extra dimensions
            while white_mask_orig.ndim > 3:
                white_mask_orig = np.squeeze(white_mask_orig, axis=-1)
            
            _vprint(f"  Original white mask shape: {white_mask_orig.shape}")
            _vprint(f"  Target NIfTI shape: {dims}")
            
            # Resample to match NIfTI dimensions if needed (following patch_first_processing.py approach)
            if white_mask_orig.shape != dims:
                _vprint(f"  Resampling white mask to match data dimensions...")
                from scipy.ndimage import zoom
                
                # Calculate zoom factors
                zoom_factors = np.array(dims) / np.array(white_mask_orig.shape)
                _vprint(f"  Zoom factors: {zoom_factors}")
                
                # Resample using nearest neighbor to preserve binary mask values (order=0)
                white_mask_resampled = zoom(white_mask_orig, zoom_factors, order=0)
                _vprint(f"  Resampled white mask shape: {white_mask_resampled.shape}")
                
                # Convert to binary mask
                white_mask_data = (white_mask_resampled > 0.5).astype(np.uint8)
            else:
                # Dimensions match - use directly
                white_mask_data = (white_mask_orig > 0.5).astype(np.uint8)
            
            _vprint(
                f"  White matter mask ready: {np.count_nonzero(white_mask_data)} / {white_mask_data.size} voxels ({100*np.count_nonzero(white_mask_data)/white_mask_data.size:.1f}%)"
            )
        except Exception as e:
            _vprint(f"Warning: Could not load/resample white matter mask: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            white_mask_data = None
    
    # Load streamlines
    try:
        tractogram = load_trk(trk_file)
        streamlines = tractogram.streamlines
        _vprint(f"Loaded {len(streamlines)} streamlines")
        
        # Transform to voxel space
        trk_affine = tractogram.affine
        affine_diff = np.abs(trk_affine - nii_img.affine).max()
        if affine_diff > 0.1:
            _vprint("Pre-registered TRK detected, using TRK affine")
            affine_inv = np.linalg.inv(trk_affine)
        else:
            affine_inv = np.linalg.inv(nii_img.affine)
        
        streamlines_voxel = list(transform_streamlines(streamlines, affine_inv))
        
        # Apply bundle size filtering if specified
        if min_bundle_size and len(streamlines_voxel) > min_bundle_size:
            _vprint(f"Note: Total streamlines ({len(streamlines_voxel)}) exceeds min_bundle_size ({min_bundle_size})")
            _vprint(f"      Bundle filtering not applied (would need clustering implementation)")
        
    except Exception as e:
        _vprint(f"Error loading streamlines: {e}")
        return
    
    # Determine slice range
    if slice_range is None:
        if orientation == 'coronal':
            slice_range = range(0, dims[1])
        elif orientation == 'axial':
            slice_range = range(0, dims[2])
        else:  # sagittal
            slice_range = range(0, dims[0])
    
    num_slices = len(slice_range)
    _vprint(f"Processing {num_slices} {orientation} slices from {min(slice_range)} to {max(slice_range)}")
    
    # Create output volume
    if orientation == 'coronal':
        volume_shape = (dims[0], num_slices, dims[2])
    elif orientation == 'axial':
        volume_shape = (dims[0], dims[1], num_slices)
    else:  # sagittal
        volume_shape = (num_slices, dims[1], dims[2])
    
    # ==================================================================
    # PHASE 1: TRUE 3D VOLUMETRIC TISSUE PROCESSING
    # ==================================================================
    # Process the ENTIRE 3D volume at once to eliminate slice boundaries
    _vprint(f"\nPhase 1: Processing full 3D tissue volume (no slice artifacts)...")
    
    # Extract the relevant portion of the 3D volume
    if orientation == 'coronal':
        volume_data = nii_data[:, list(slice_range), :].copy()
    elif orientation == 'axial':
        volume_data = nii_data[:, :, list(slice_range)].copy()
    else:  # sagittal
        volume_data = nii_data[list(slice_range), :, :].copy()
    
    _vprint(f"  Extracted {orientation} volume: {volume_data.shape}")
    
    # Apply TRUE 3D CLAHE - processes ENTIRE volume as single unit
    # Uses GLOBAL mode (single histogram) for maximum smoothness
    use_clahe = (contrast_method == 'clahe' or contrast_method is None)
    
    # Set default cornucopia presets
    if cornucopia_allowed_presets is None and use_cornucopia_3d:
        cornucopia_allowed_presets = [
            'granular_realistic',
            'extreme_noise',
            'random_shapes_background',
            'comprehensive_aggressive',
            'ultra_heavy_speckle'
        ]
    
    output_volume = process_volume_full_3d(
        volume_data,
        use_clahe=use_clahe,                  # TRUE 3D CLAHE
        clahe_adaptive=False,                 # GLOBAL: entire volume = single tile (no boundaries!)
        clahe_clip_limit=clahe_clip_limit,    # Controllable - lower preserves more noise
        add_texture=use_cornucopia_3d,         # 3D texture (intensity/sigma controllable)
        texture_intensity=texture_intensity,
        texture_sigma=texture_sigma,
        gamma=gamma,
        scaling_factor=scaling_factor,
        use_bilateral=use_bilateral_smoothing,
        use_cornucopia=use_cornucopia_3d,
        cornucopia_preset=None,               # Random selection from allowed list
        cornucopia_allowed_presets=cornucopia_allowed_presets,
        cornucopia_prob=cornucopia_prob,
        random_state=random_state,
        verbose=verbose,
    )

    # Inject cell-body blob distractors into the tissue IMAGE (before fibers,
    # not into the mask) so the model must distinguish fibers from cell bodies.
    if enable_cell_blobs and cell_blob_count > 0:
        _vprint(f"  Injecting {cell_blob_count} cell-body blob distractors "
                f"(radius {cell_blob_radius_range} vox, intensity {cell_blob_intensity})...")
        blob_rng = np.random.default_rng(random_state)
        output_volume = add_cell_body_blobs(
            output_volume, cell_blob_count, cell_blob_intensity,
            cell_blob_radius_range, blob_rng,
        )

    # ==================================================================
    # PHASE 2: RENDER STREAMLINES IN TRUE 3D SPACE
    # ==================================================================
    _vprint(f"\nPhase 2: Rendering streamlines in 3D space...")
    
    # Calculate slice offset for coordinate mapping
    slice_start = min(slice_range)
    
    # Helper: 3D line drawing between two 3D points
    def draw_line_3d(p0, p1, intensity, volume, tissue_threshold, mask_volume=None, signal_volume=None):
        """Draw 3D line between two points using 3D Bresenham"""
        x0, y0, z0 = int(np.round(p0[0])), int(np.round(p0[1])), int(np.round(p0[2]))
        x1, y1, z1 = int(np.round(p1[0])), int(np.round(p1[1])), int(np.round(p1[2]))
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        xs = 1 if x1 > x0 else -1
        ys = 1 if y1 > y0 else -1
        zs = 1 if z1 > z0 else -1
        
        # Helper to update voxel
        def update_voxel(vx, vy, vz):
            if 0 <= vx < volume.shape[0] and 0 <= vy < volume.shape[1] and 0 <= vz < volume.shape[2]:
                # Mask always marks the true fiber path regardless of local tissue intensity,
                # so the mask is continuous even through low-signal regions (CSF, boundaries).
                if mask_volume is not None:
                    mask_volume[vx, vy, vz] += 1.0
                if volume[vx, vy, vz] >= tissue_threshold:
                    if fiber_render_mode == 'additive':
                        volume[vx, vy, vz] = min(255.0, volume[vx, vy, vz] + intensity)
                    if signal_volume is not None:
                        signal_volume[vx, vy, vz] += intensity

        # Driving axis
        if dx >= dy and dx >= dz:  # X-axis is driving
            p1_d, p2_d = dy, dz
            p1_inc, p2_inc = ys, zs
            d_axis = dx
            
            p1_err, p2_err = d_axis // 2, d_axis // 2
            x, y, z = x0, y0, z0
            
            for _ in range(d_axis + 1):
                update_voxel(x, y, z)
                
                p1_err += p1_d
                if p1_err >= d_axis:
                    y += p1_inc
                    p1_err -= d_axis
                
                p2_err += p2_d
                if p2_err >= d_axis:
                    z += p2_inc
                    p2_err -= d_axis
                
                x += xs
                
        elif dy >= dx and dy >= dz:  # Y-axis is driving
            p1_d, p2_d = dx, dz
            p1_inc, p2_inc = xs, zs
            d_axis = dy
            
            p1_err, p2_err = d_axis // 2, d_axis // 2
            x, y, z = x0, y0, z0
            
            for _ in range(d_axis + 1):
                update_voxel(x, y, z)
                
                p1_err += p1_d
                if p1_err >= d_axis:
                    x += p1_inc
                    p1_err -= d_axis
                
                p2_err += p2_d
                if p2_err >= d_axis:
                    z += p2_inc
                    p2_err -= d_axis
                
                y += ys
                
        else:  # Z-axis is driving
            p1_d, p2_d = dx, dy
            p1_inc, p2_inc = xs, ys
            d_axis = dz
            
            p1_err, p2_err = d_axis // 2, d_axis // 2
            x, y, z = x0, y0, z0
            
            for _ in range(d_axis + 1):
                update_voxel(x, y, z)
                
                p1_err += p1_d
                if p1_err >= d_axis:
                    x += p1_inc
                    p1_err -= d_axis
                
                p2_err += p2_d
                if p2_err >= d_axis:
                    y += p2_inc
                    p2_err -= d_axis
                
                z += zs

    def draw_line_3d_antialias(p0, p1, intensity, volume, tissue_threshold, mask_volume=None, signal_volume=None):
        """Draw a subvoxel 3D line by trilinear splatting along the segment."""
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            samples = (p0[None, :],)
        else:
            n_steps = max(1, int(np.ceil(length * 2.0)))
            t = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
            samples = (p0[None, :] + t[:, None] * delta[None, :],)

        sx, sy, sz = volume.shape
        for points in samples:
            for point in points:
                base = np.floor(point).astype(np.int32)
                frac = point - base
                for ox in (0, 1):
                    wx = (1.0 - frac[0]) if ox == 0 else frac[0]
                    vx = int(base[0] + ox)
                    if vx < 0 or vx >= sx or wx <= 0:
                        continue
                    for oy in (0, 1):
                        wy = (1.0 - frac[1]) if oy == 0 else frac[1]
                        vy = int(base[1] + oy)
                        if vy < 0 or vy >= sy or wy <= 0:
                            continue
                        for oz in (0, 1):
                            wz = (1.0 - frac[2]) if oz == 0 else frac[2]
                            vz = int(base[2] + oz)
                            weight = float(wx * wy * wz)
                            if vz < 0 or vz >= sz or weight <= 0.015:
                                continue
                            # Mask: accumulate trilinear (partial-volume) weights so the
                            # mask tracks the true sub-voxel fiber centerline instead of
                            # snapping to the nearest grid voxel (which caused stair-steps).
                            if mask_volume is not None:
                                mask_volume[vx, vy, vz] += weight
                            if volume[vx, vy, vz] >= tissue_threshold:
                                if fiber_render_mode == 'additive':
                                    volume[vx, vy, vz] = min(255.0, volume[vx, vy, vz] + intensity * weight)
                                if signal_volume is not None:
                                    signal_volume[vx, vy, vz] += intensity * weight
    
    # Render all streamlines in 3D. The image can use variable brightness,
    # but the mask accumulator always receives geometry weights only.
    fiber_intensity_min = max(0.0, float(fiber_intensity_min))
    fiber_intensity_max = max(0.0, float(fiber_intensity_max))
    if fiber_intensity_max < fiber_intensity_min:
        fiber_intensity_min, fiber_intensity_max = fiber_intensity_max, fiber_intensity_min
    fiber_intensity = max(1e-6, (fiber_intensity_max + fiber_intensity_min) / 2.0)
    fiber_brightness_variation = max(0.0, float(fiber_brightness_variation))
    fiber_segment_brightness_variation = max(0.0, float(fiber_segment_brightness_variation))
    render_rng = np.random.default_rng(random_state)
    variable_fiber_intensity = (
        abs(fiber_intensity_max - fiber_intensity_min) > 1e-6
        or fiber_brightness_variation > 0.0
        or fiber_segment_brightness_variation > 0.0
    )

    def sample_streamline_intensity():
        if fiber_intensity_max > fiber_intensity_min:
            intensity = float(render_rng.uniform(fiber_intensity_min, fiber_intensity_max))
        else:
            intensity = float(fiber_intensity)
        if fiber_brightness_variation > 0.0:
            sigma = min(1.25, fiber_brightness_variation)
            multiplier = float(render_rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))
            intensity *= float(np.clip(multiplier, 0.25, 2.75))
        return max(0.0, intensity)

    def sample_segment_intensity(streamline_intensity):
        intensity = float(streamline_intensity)
        if fiber_segment_brightness_variation > 0.0:
            jitter = float(render_rng.normal(1.0, fiber_segment_brightness_variation))
            intensity *= float(np.clip(jitter, 0.25, 2.25))
        return max(0.0, intensity)

    streamlines_rendered = 0
    fiber_smoothing_sigma = max(0.0, float(fiber_smoothing_sigma))
    fiber_render_mode = str(fiber_render_mode).lower()
    if fiber_render_mode not in ('additive', 'density', 'embedded'):
        raise ValueError(
            f"fiber_render_mode must be 'additive', 'density', or 'embedded', got {fiber_render_mode!r}"
        )
    tone_down_fibers = (
        fiber_render_mode in ('density', 'embedded')
        or fiber_max_boost is not None
        or float(fiber_opacity) < 1.0
        or fiber_smoothing_sigma > 0.0
        or background_max_intensity is not None
    )
    base_volume_for_fibers = output_volume.copy() if tone_down_fibers else None
    
    needs_density_accumulator = fiber_render_mode in ('density', 'embedded')
    needs_signal_accumulator = needs_density_accumulator and variable_fiber_intensity
    force_cpu_variable_intensity = variable_fiber_intensity

    # Initialize mask accumulator if needed
    # Use function parameter use_gpu to decide
    local_xp = xp if (
        use_gpu
        and not force_cpu_variable_intensity
        and 'gpu_result' in globals()
        and gpu_result.get('cupy_available', False)
    ) else np
    mask_accumulator = (
        local_xp.zeros(output_volume.shape, dtype=local_xp.float32)
        if (save_mask or needs_density_accumulator)
        else None
    )
    fiber_signal_accumulator = (
        np.zeros(output_volume.shape, dtype=np.float32)
        if needs_signal_accumulator
        else None
    )
    
    # Transfer to GPU if available
    fiber_antialias = bool(fiber_antialias)
    local_use_gpu = (
        use_gpu
        and not fiber_antialias
        and not force_cpu_variable_intensity
        and 'gpu_result' in globals()
        and gpu_result.get('cupy_available', False)
    )
    if use_gpu and force_cpu_variable_intensity:
        _vprint("Using CPU streamline rasterization for variable fiber brightness.")
    if local_use_gpu:
        _vprint(f"Transferring volume to GPU for streamline rendering")
        output_volume_gpu = local_xp.asarray(output_volume)
    else:
        output_volume_gpu = output_volume
    
    segments_p0 = []
    segments_p1 = []

    if min_streamlines_rendered is not None and int(min_streamlines_rendered) > 0:
        if len(streamlines_voxel) < int(min_streamlines_rendered):
            _vprint(f"Skipping patch: only {len(streamlines_voxel)} streamlines, need at least {min_streamlines_rendered}")
            return None

    for sl in tqdm(streamlines_voxel, desc="Rendering streamlines", disable=not verbose):
        streamline_intensity = sample_streamline_intensity()

        # Map streamlines to output volume coordinates
        sl_mapped = sl.copy()
        
        # Adjust coordinates based on orientation and slice offset
        if orientation == 'coronal':
            # Y dimension is sliced, adjust Y coordinates
            sl_mapped[:, 1] = sl[:, 1] - slice_start
        elif orientation == 'axial':
            # Z dimension is sliced, adjust Z coordinates  
            sl_mapped[:, 2] = sl[:, 2] - slice_start
        else:  # sagittal
            # X dimension is sliced, adjust X coordinates
            sl_mapped[:, 0] = sl[:, 0] - slice_start
        
        # Apply z-flip for correct alignment
        sl_mapped[:, 2] = (dims[2] if orientation != 'axial' else num_slices) - sl_mapped[:, 2] - 1
        
        # Draw lines between consecutive points in 3D
        for i in range(len(sl_mapped) - 1):
            p0 = sl_mapped[i]
            p1 = sl_mapped[i + 1]
            segment_intensity = sample_segment_intensity(streamline_intensity)
            
            # White matter mask check for both endpoints (use original coordinates)
            if white_mask_data is not None:
                x0, y0, z0 = int(np.round(np.clip(sl[i, 0], 0, dims[0]-1))), \
                             int(np.round(np.clip(sl[i, 1], 0, dims[1]-1))), \
                             int(np.round(np.clip(sl[i, 2], 0, dims[2]-1)))
                x1, y1, z1 = int(np.round(np.clip(sl[i+1, 0], 0, dims[0]-1))), \
                             int(np.round(np.clip(sl[i+1, 1], 0, dims[1]-1))), \
                             int(np.round(np.clip(sl[i+1, 2], 0, dims[2]-1)))
                if (white_mask_data[x0, y0, z0] < 0.5 or white_mask_data[x1, y1, z1] < 0.5):
                    continue
            
            # Clip to output volume bounds
            p0_clipped = np.clip(p0, [0, 0, 0], [output_volume.shape[0]-1, output_volume.shape[1]-1, output_volume.shape[2]-1])
            p1_clipped = np.clip(p1, [0, 0, 0], [output_volume.shape[0]-1, output_volume.shape[1]-1, output_volume.shape[2]-1])
            
            if local_use_gpu:
                segments_p0.append(np.rint(p0_clipped).astype(np.int32))
                segments_p1.append(np.rint(p1_clipped).astype(np.int32))
            elif fiber_antialias:
                draw_line_3d_antialias(
                    p0_clipped,
                    p1_clipped,
                    segment_intensity,
                    output_volume,
                    tissue_threshold,
                    mask_accumulator,
                    fiber_signal_accumulator,
                )
            else:
                draw_line_3d(
                    p0_clipped,
                    p1_clipped,
                    segment_intensity,
                    output_volume,
                    tissue_threshold,
                    mask_accumulator,
                    fiber_signal_accumulator,
                )
        
        streamlines_rendered += 1

    if local_use_gpu and len(segments_p0) > 0:
        used_gpu_kernel = False
        try:
            if _ensure_gpu_line_kernels():
                p0_arr = np.stack(segments_p0, axis=0).astype(np.int32, copy=False)
                p1_arr = np.stack(segments_p1, axis=0).astype(np.int32, copy=False)

                p0_gpu = local_xp.asarray(p0_arr)
                p1_gpu = local_xp.asarray(p1_arr)
                nseg = int(p0_arr.shape[0])
                threads = 256
                blocks = (nseg + threads - 1) // threads
                sx, sy, sz = output_volume.shape

                if mask_accumulator is not None:
                    _GPU_LINE_KERNEL_WITH_MASK(
                        (blocks,),
                        (threads,),
                        (
                            output_volume_gpu,
                            mask_accumulator,
                            p0_gpu,
                            p1_gpu,
                            np.int32(nseg),
                            np.float32(fiber_intensity),
                            np.float32(tissue_threshold),
                            np.int32(sx),
                            np.int32(sy),
                            np.int32(sz),
                        ),
                    )
                else:
                    _GPU_LINE_KERNEL_NO_MASK(
                        (blocks,),
                        (threads,),
                        (
                            output_volume_gpu,
                            p0_gpu,
                            p1_gpu,
                            np.int32(nseg),
                            np.float32(fiber_intensity),
                            np.float32(tissue_threshold),
                            np.int32(sx),
                            np.int32(sy),
                            np.int32(sz),
                        ),
                    )
                local_xp.cuda.runtime.deviceSynchronize()
                used_gpu_kernel = True
                _vprint(f"GPU line kernel rendered {nseg} segments")
            elif _GPU_LINE_KERNEL_ERROR:
                print(f"[render] GPU line kernel unavailable, using CPU fallback: {_GPU_LINE_KERNEL_ERROR}")
        except Exception as e:
            print(f"[render] GPU line kernel fallback to CPU line drawing: {e}")

        if not used_gpu_kernel:
            output_volume = local_xp.asnumpy(output_volume_gpu)
            if mask_accumulator is not None:
                mask_accumulator = local_xp.asnumpy(mask_accumulator)
            for p0_clipped, p1_clipped in zip(segments_p0, segments_p1):
                draw_line_3d(p0_clipped, p1_clipped, fiber_intensity, output_volume, tissue_threshold, mask_accumulator)
            local_use_gpu = False
    
    _vprint(f"Rendered {streamlines_rendered} streamlines in 3D volume")
    
    # Transfer back from GPU if used
    if local_use_gpu:
        output_volume = local_xp.asnumpy(output_volume_gpu)

    if fiber_render_mode in ('density', 'embedded') and mask_accumulator is not None and base_volume_for_fibers is not None:
        if fiber_signal_accumulator is not None:
            fiber_density = fiber_signal_accumulator.astype(np.float32, copy=False)
        elif local_use_gpu:
            fiber_density = local_xp.asnumpy(mask_accumulator).astype(np.float32, copy=False)
        else:
            fiber_density = np.asarray(mask_accumulator, dtype=np.float32)

        if fiber_smoothing_sigma > 0.0 and float(fiber_density.max()) > 0.0:
            fiber_density = gaussian_filter(fiber_density, sigma=fiber_smoothing_sigma)

        density_norm = np.zeros_like(fiber_density, dtype=np.float32)
        positive = fiber_density > 0
        if np.any(positive):
            denom = float(np.percentile(fiber_density[positive], 99))
            if denom <= 0.0:
                denom = float(fiber_density[positive].max())
            if denom > 0.0:
                density_norm = np.clip(fiber_density / denom, 0.0, 1.0)
                density_norm = np.power(density_norm, max(0.1, float(fiber_density_gamma)))
                min_visibility = max(0.0, min(1.0, float(fiber_min_visibility)))
                if min_visibility > 0.0:
                    density_norm[positive] = np.maximum(density_norm[positive], min_visibility)

        output_volume = base_volume_for_fibers.astype(np.float32, copy=True)
        if background_max_intensity is not None:
            output_volume = np.minimum(output_volume, float(background_max_intensity))

        opacity = max(0.0, float(fiber_opacity))
        if fiber_render_mode == 'embedded':
            # The reference appearance is a tract-like low-contrast texture:
            # local gray tissue plus thin slightly brighter strands. Avoid
            # pushing fibers to pure white or letting dense bundles fill a block.
            boost = max(0.0, float(fiber_target_intensity)) * density_norm * opacity
            output_volume = output_volume + boost
        else:
            target = max(0.0, float(fiber_target_intensity))
            lift = np.maximum(target - output_volume, 0.0)
            output_volume = output_volume + lift * density_norm * opacity

    elif tone_down_fibers and base_volume_for_fibers is not None:
        fiber_boost = np.maximum(output_volume - base_volume_for_fibers, 0.0)
        if fiber_smoothing_sigma > 0.0 and float(fiber_boost.max()) > 0.0:
            boost_max = float(fiber_boost.max())
            fiber_boost = gaussian_filter(fiber_boost, sigma=fiber_smoothing_sigma)
            smooth_max = float(fiber_boost.max())
            if smooth_max > 0.0:
                fiber_boost *= boost_max / smooth_max
        if fiber_max_boost is not None:
            fiber_boost = np.minimum(fiber_boost, max(0.0, float(fiber_max_boost)))
        output_volume = base_volume_for_fibers + fiber_boost * max(0.0, float(fiber_opacity))
    
    # Image-only realism augmentations. These are intentionally non-geometric and
    # never touch the streamline mask accumulator, so mask/fiber alignment is kept.
    if enable_tissue_artifacts or enable_granular_noise or enable_speckle_noise or enable_dash_noise or enable_horizontal_banding or enable_poisson_noise:
        try:
            from syntract_viewer.synthetic_image_augmentations import apply_image_only_augmentations
        except ImportError:
            from synthetic_image_augmentations import apply_image_only_augmentations

        # Build a boolean fiber exclusion mask from the raw accumulator so that
        # speckle/dash artefacts are never placed on fiber voxels (which would
        # introduce false negatives in the ground-truth mask).
        _fiber_mask_np = None
        if (enable_speckle_noise or enable_dash_noise) and mask_accumulator is not None:
            _ma = mask_accumulator.get() if hasattr(mask_accumulator, 'get') else np.asarray(mask_accumulator)
            _fiber_mask_np = (_ma > 0)

        output_volume = apply_image_only_augmentations(
            output_volume,
            enable_tissue_artifacts=enable_tissue_artifacts,
            enable_granular_noise=enable_granular_noise,
            enable_speckle_noise=enable_speckle_noise,
            enable_dash_noise=enable_dash_noise,
            enable_horizontal_banding=enable_horizontal_banding,
            enable_poisson_noise=enable_poisson_noise,
            artifact_strength=artifact_strength,
            granular_noise_strength=granular_noise_strength,
            poisson_gain=poisson_gain,
            speckle_noise_strength=speckle_noise_strength,
            speckle_noise_density=speckle_noise_density,
            speckle_noise_sigma=speckle_noise_sigma,
            speckle_square_size=speckle_square_size,
            dash_noise_strength=dash_noise_strength,
            dash_noise_density=dash_noise_density,
            dash_length_sigma=dash_length_sigma,
            dash_cross_sigma=dash_cross_sigma,
            banding_strength=banding_strength,
            banding_axis=banding_axis,
            random_state=random_state,
            verbose=verbose,
            fiber_mask=_fiber_mask_np,
        )
    
    # Save output
    if save_outputs:
        output_nii = nib.Nifti1Image(output_volume, affine=nii_img.affine)
        nib.save(output_nii, output_file)
        _vprint(f"\nSaved 3D volume: {output_file}")
        _vprint(f"  Shape: {output_volume.shape}")
        _vprint(f"  Value range: [{output_volume.min():.2f}, {output_volume.max():.2f}]")
    
    # ==================================================================
    # GENERATE AND SAVE FIBER MASK (TRUE 3D GENERATION)
    # ==================================================================
    mask_volume = None
    if save_mask and mask_accumulator is not None:
        _vprint(f"\nPhase 3: Generating TRUE 3D fiber mask (isotropic consistency)...")
        
        mask_smoothing_sigma = max(0.0, float(mask_smoothing_sigma))
        mask_binary_threshold = max(0.0, float(mask_binary_threshold))
        _vprint(
            f"  Applying 3D Gaussian smoothing (sigma={mask_smoothing_sigma}) "
            "for mask connectivity..."
        )
        if local_use_gpu:
            if mask_smoothing_sigma > 0:
                mask_density_smooth = gpu_ndimage.gaussian_filter(mask_accumulator, sigma=mask_smoothing_sigma)
            else:
                mask_density_smooth = mask_accumulator
            mask_density_smooth = local_xp.asnumpy(mask_density_smooth)
        else:
            if mask_smoothing_sigma > 0:
                mask_density_smooth = gaussian_filter(mask_accumulator, sigma=mask_smoothing_sigma)
            else:
                mask_density_smooth = mask_accumulator
        
        # Normalize for consistent thresholding (0.0 to 1.0)
        # This handles varying bundle densities gracefully
        max_val = mask_density_smooth.max()
        if max_val > 0:
            mask_density_smooth /= max_val
            
        if soft_mask:
            # Partial-volume (anti-aliased) mask: keep fractional sub-voxel coverage
            # instead of a hard binary threshold. A binary mask of a thin diagonal
            # always stair-steps; the soft mask smoothly tracks the centerline and is
            # the most accurate ground truth. BCEWithLogitsLoss accepts soft targets.
            _vprint("  Saving SOFT (partial-volume) mask — no binary stair-stepping...")
            mask_volume = mask_density_smooth.astype(np.float32)
        else:
            _vprint(f"  Thresholding normalized density map at {mask_binary_threshold}...")
            mask_volume = (mask_density_smooth > mask_binary_threshold).astype(np.uint8)

        if save_outputs:
            mask_file = output_file.replace('.nii.gz', '_mask.nii.gz')
            mask_nii = nib.Nifti1Image(mask_volume, affine=nii_img.affine)
            nib.save(mask_nii, mask_file)
            mask_voxels = int(np.count_nonzero(mask_volume > (0.0 if soft_mask else 0)))
            mask_percentage = 100 * mask_voxels / mask_volume.size
            _vprint(f"\nSaved fiber mask: {mask_file}")
            _vprint(f"  Shape: {mask_volume.shape}")
            _vprint(f"  Mask coverage: {mask_voxels} voxels ({mask_percentage:.2f}%)")
    
    # Explicit memory cleanup to ensure resources are released
    if local_use_gpu:
        # Free GPU memory
        del mask_accumulator
        if 'output_volume_gpu' in locals():
            del output_volume_gpu
        local_xp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    
    if return_arrays:
        return output_volume.astype(np.float32, copy=False), (
            mask_volume.astype(np.float32, copy=False) if mask_volume is not None else None
        )
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nifti', required=True)
    parser.add_argument('--trk', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--orientation', default='coronal', choices=['coronal', 'axial', 'sagittal'])
    parser.add_argument('--enable_tissue_artifacts', action='store_true',
                        help='Add image-only tissue-like artefacts')
    parser.add_argument('--enable_granular_noise', action='store_true',
                        help='Add image-only fine Cornucopia granular noise')
    parser.add_argument('--enable_speckle_noise', action='store_true',
                        help='Add sparse image-only whitish-grey dot artefacts')
    parser.add_argument('--enable_poisson_noise', action='store_true',
                        help='Add signal-dependent Poisson shot noise (image only)')
    parser.add_argument('--poisson_gain', type=float, default=80.0,
                        help='Photons-per-unit; lower=more shot noise, higher=cleaner')
    parser.add_argument('--artifact_strength', type=float, default=0.45)
    parser.add_argument('--granular_noise_strength', type=float, default=0.35)
    parser.add_argument('--speckle_noise_strength', type=float, default=0.35)
    parser.add_argument('--speckle_noise_density', type=float, default=0.0012)
    parser.add_argument('--speckle_noise_sigma', type=float, default=0.35)
    parser.add_argument('--fiber_intensity_min', type=float, default=15.0)
    parser.add_argument('--fiber_intensity_max', type=float, default=25.0)
    parser.add_argument('--fiber_max_boost', type=float, default=None)
    parser.add_argument('--fiber_opacity', type=float, default=1.0)
    parser.add_argument('--fiber_smoothing_sigma', type=float, default=0.0)
    parser.add_argument('--fiber_antialias', action='store_true')
    parser.add_argument('--min_streamlines_rendered', type=int, default=None)
    parser.add_argument('--fiber_brightness_variation', type=float, default=0.35)
    parser.add_argument('--fiber_segment_brightness_variation', type=float, default=0.15)
    parser.add_argument('--fiber_render_mode', default='additive', choices=['additive', 'density', 'embedded'])
    parser.add_argument('--fiber_density_gamma', type=float, default=2.0)
    parser.add_argument('--fiber_min_visibility', type=float, default=0.15)
    parser.add_argument('--fiber_target_intensity', type=float, default=38.0)
    parser.add_argument('--background_max_intensity', type=float, default=None)
    parser.add_argument('--mask_smoothing_sigma', type=float, default=2.0)
    parser.add_argument('--mask_binary_threshold', type=float, default=0.01)
    
    args = parser.parse_args()
    
    # Determine slice range
    nii = nib.load(args.nifti)
    dims = nii.shape
    
    if args.end is None:
        if args.orientation == 'coronal':
            args.end = dims[1]
        elif args.orientation == 'axial':
            args.end = dims[2]
        else:
            args.end = dims[0]
    
    slice_range = range(args.start, args.end)
    
    create_3d_volume_with_streamlines(
        nifti_file=args.nifti,
        trk_file=args.trk,
        output_file=args.output,
        slice_range=slice_range,
        orientation=args.orientation,
        enable_tissue_artifacts=args.enable_tissue_artifacts,
        enable_granular_noise=args.enable_granular_noise,
        enable_speckle_noise=args.enable_speckle_noise,
        enable_poisson_noise=args.enable_poisson_noise,
        artifact_strength=args.artifact_strength,
        granular_noise_strength=args.granular_noise_strength,
        poisson_gain=args.poisson_gain,
        speckle_noise_strength=args.speckle_noise_strength,
        speckle_noise_density=args.speckle_noise_density,
        speckle_noise_sigma=args.speckle_noise_sigma,
        fiber_intensity_min=args.fiber_intensity_min,
        fiber_intensity_max=args.fiber_intensity_max,
        fiber_max_boost=args.fiber_max_boost,
        fiber_opacity=args.fiber_opacity,
        fiber_smoothing_sigma=args.fiber_smoothing_sigma,
        fiber_antialias=args.fiber_antialias,
        min_streamlines_rendered=args.min_streamlines_rendered,
        fiber_brightness_variation=args.fiber_brightness_variation,
        fiber_segment_brightness_variation=args.fiber_segment_brightness_variation,
        fiber_render_mode=args.fiber_render_mode,
        fiber_density_gamma=args.fiber_density_gamma,
        fiber_min_visibility=args.fiber_min_visibility,
        fiber_target_intensity=args.fiber_target_intensity,
        background_max_intensity=args.background_max_intensity,
        mask_smoothing_sigma=args.mask_smoothing_sigma,
        mask_binary_threshold=args.mask_binary_threshold,
    )
