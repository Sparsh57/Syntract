# SynTract: MRI Synthesis & Tractography Visualization

A streamlined Python pipeline for MRI processing, tractography synthesis, and dark field microscopy-style visualization. Features unified CLI, ANTs integration, and patch-first optimization for dramatic performance improvements in neuroimaging research.

## Features

- Patch Processing: 80-95% performance improvement with  patch extraction (default mode)
- **Unified Pipeline**: Single command processing from raw NIfTI/TRK to visualizations
- **ANTs Integration**: Spatial transformations and registration workflows
- **Auto-Dimension Calculation**: Intelligent target sizing based on input characteristics
- **Zero-Tolerance Spatial Accuracy**: Perfect bounds enforcement with enhanced curvature preservation
- **Memory-Optimized**: Efficient processing for large datasets with minimal memory usage
- **Cornucopia**: Intelligent preset selection for realistic background textures
- **Batch Processing**: Multiple TRK files with shared NIfTI
- **Dark Field Visualization**: Publication-ready medical imaging with enhanced contrast

## Installation

```bash
git clone https://github.com/Sparsh57/Syntract.git
cd Syntract
pip install -r requirements.txt
```

## Quick Start

### Basic Usage (Patch-First Processing - Default)
```bash
# Default: Fast patch-first processing with auto-calculated dimensions
python syntract.py --input brain.nii.gz --trk fibers.trk --output result

# Customized patch processing
python syntract.py --input brain.nii.gz --trk fibers.trk --output result \
  --total_patches 100 --patch_size 800 1 800
```

### Traditional Full-Volume Synthesis (Optional)
```bash
# Disable patch processing for traditional synthesis (slower, more memory)
python syntract.py --input brain.nii.gz --trk fibers.trk --output result \
  --disable_patch_processing --new_dim 116 140 96
```

### With ANTs Transformation
```bash
python syntract.py --input brain.nii.gz --trk fibers.trk --use_ants \
  --ants_warp warp.nii.gz --ants_iwarp iwarp.nii.gz --ants_aff affine.mat
```

### Advanced Patch Processing
```bash
# High-throughput: many small patches
python syntract.py --input brain.nii.gz --trk fibers.trk \
  --total_patches 200 --patch_size 400 1 400 --patch_batch_size 50

# Quality mode: fewer large patches
python syntract.py --input brain.nii.gz --trk fibers.trk \
  --total_patches 50 --patch_size 1024 1 1024
```

### Batch Processing
```bash
# Process multiple TRK files with shared NIfTI
python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/
```

## Parameters

### Essential Arguments
| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` | str | Input NIfTI file path (required) |
| `--trk` | str | Input TRK file path (required) |
| `--output` | str | Output base name (default: "output") |

### Synthesis Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--new_dim` | int×3 | Auto-calculated | Target dimensions (X Y Z) - auto-calculated if not specified |
| `--voxel_size` | float | 0.5 | Target voxel size in mm |
| `--skip_synthesis` | flag | | Skip synthesis and use input files directly |

### ANTs Transformation
| Parameter | Type | Description |
|-----------|------|-------------|
| `--use_ants` | flag | Enable ANTs transformation |
| `--ants_warp` | str | ANTs warp field file |
| `--ants_iwarp` | str | ANTs inverse warp field file |
| `--ants_aff` | str | ANTs affine transformation file |

### Patch Processing (Default Mode)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--disable_patch_processing` | flag | | Disable patch processing and use traditional synthesis |
| `--patch_output_dir` | str | "patches" | Directory for patch outputs |
| `--total_patches` | int | 50 | Total number of patches to extract |
| `--patch_size` | int×3 | [800, 1, 800] | Patch dimensions (width, height, depth) |
| `--min_streamlines_per_patch` | int | 10 | Minimum streamlines required per patch |
| `--patch_batch_size` | int | 50 | Batch size for memory management |
| `--random_state` | int | | Random seed for reproducible extraction |
| `--patch_prefix` | str | "patch" | Prefix for patch files |

### Visualization
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_examples` | int | 3 | Number of visualization examples |
| `--viz_prefix` | str | "synthetic_" | Prefix for visualization files |
| `--enable_orange_blobs` | flag | | Enable orange blob injection site artifacts |
| `--orange_blob_probability` | float | 0.3 | Probability of applying orange blobs (0.0-1.0) |

### Mask & Bundle Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save_masks` | flag | True | Save binary masks alongside visualizations |
| `--use_high_density_masks` | flag | False | Use high-density mask generation |
| `--mask_thickness` | int | 1 | Thickness of generated masks |
| `--density_threshold` | float | 0.15 | Fiber density threshold for masking |
| `--min_bundle_size` | int | 20 | Minimum size for bundle detection |
| `--label_bundles` | flag | False | Label individual fiber bundles |

### Slice Extraction
| Parameter | Type | Description |
|-----------|------|-------------|
| `--slice_count` | int | Number of coronal slices to extract |
| `--slice_output_dir` | str | Directory for slice outputs |
| `--auto_batch_process` | flag | Automatically process all extracted slices |

## Batch Processing (cumulative.py)

Process multiple TRK files with a shared NIfTI file efficiently. The script provides both CLI and Python function interfaces with all syntract options available.

### Command Line Usage
```bash
# Basic batch processing
python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/

# With custom options
python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \
  --total-patches 50 --n-examples 5 --voxel-size 0.05

# With ANTs transformation
python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \
  --use-ants --ants-warp warp.nii.gz --ants-iwarp iwarp.nii.gz --ants-aff affine.mat
```

### Python Function Usage
```python
from cumulative import process_batch

results = process_batch(
    nifti_file='brain.nii.gz',
    trk_directory='./trk_files/',
    patches=50,
    use_ants=True,
    ants_warp='warp.nii.gz',
    ants_iwarp='iwarp.nii.gz', 
    ants_aff='affine.mat'
)
```

### Key Features
- Auto-calculates target dimensions and patch sizes
- Distributes patches evenly across TRK files
- Supports all syntract CLI options
- Provides detailed progress tracking and performance metrics
- Saves processing summary in JSON format

## Output Structure

### Patch Processing (Default)
```
patches/
├── patch_0001.nii.gz     # Patch NIfTI files
├── patch_0001.trk        # Patch TRK files
├── patch_0001_visualization.png  # Visualizations
├── patch_0001_visualization_mask_slice0.png  # Masks
└── patch_extraction_summary.json  # Processing summary
```

### Traditional Processing
```
output_name.nii.gz         # Processed NIfTI
output_name.trk            # Processed TRK
visualizations/            # Generated images (if visualization enabled)
```

### Batch Processing
```
syntract_submission/
├── processed_files/       # All processed .nii.gz and .trk files
├── patches/              # Patches organized by TRK file
└── visualizations/       # Visualizations organized by TRK file
```

## Python API

### Basic Processing
```python
from syntract import process_syntract

result = process_syntract(
    input_nifti="brain.nii.gz",
    input_trk="fibers.trk",
    output_base="processed_data",
    new_dim=[400, 50, 400],
    voxel_size=0.05
)
```

### Batch Processing
```python
from cumulative import process_batch

results = process_batch(
    nifti_file="shared_brain.nii.gz",
    trk_directory="trk_files_directory",
    patches=100,
    voxel_size=0.05
)
```

### Complete Function Signature

The `process_syntract` function accepts all these parameters:

```python
from syntract import process_syntract

result = process_syntract(
    input_nifti,                    # Input NIfTI file path (required)
    input_trk,                      # Input TRK file path (required) 
    output_base,                    # Output base name (required)
    new_dim,                        # Target dimensions [X, Y, Z] (required)
    voxel_size,                     # Target voxel size in mm (required)
    
    # ANTs transformation
    use_ants=False,                 # Enable ANTs transformation
    ants_warp_path=None,           # ANTs warp field file
    ants_iwarp_path=None,          # ANTs inverse warp field file 
    ants_aff_path=None,            # ANTs affine transformation file
    
    # Slice extraction
    slice_count=None,              # Number of coronal slices to extract
    enable_slice_extraction=False, # Enable slice extraction mode
    slice_output_dir=None,         # Directory for slice outputs
    use_simplified_slicing=True,   # Use simplified slicing method
    force_full_slicing=False,      # Force full-resolution slicing
    auto_batch_process=False,      # Auto-process all extracted slices
    
    # Patch extraction
    enable_patch_extraction=False, # Enable 3D patch extraction
    patch_output_dir=None,         # Directory for patch outputs
    total_patches=None,            # Total number of patches to extract
    patch_size=None,               # Patch dimensions [width, height, depth]
    min_streamlines_per_patch=5,   # Minimum streamlines per patch
    patch_prefix="patch_",         # Prefix for patch files
    
    # Visualization
    n_examples=10,                 # Number of visualization examples
    viz_output_dir=None,           # Output directory for visualizations
    viz_prefix="viz_",             # Prefix for visualization files
    enable_orange_blobs=False,     # Enable orange blob artifacts
    orange_blob_probability=0.3,   # Probability of orange blobs (0.0-1.0)
    
    # Mask & Bundle parameters
    save_masks=True,               # Save binary masks alongside visualizations
    use_high_density_masks=False,  # Use high-density mask generation
    mask_thickness=1,              # Thickness of generated masks
    density_threshold=0.15,        # Fiber density threshold for masking
    min_bundle_size=20,            # Minimum size for bundle detection
    label_bundles=False            # Label individual fiber bundles
)
```

## Dependencies

- Core: `numpy`, `nibabel`, `matplotlib`, `scikit-image`, `scipy`, `dipy`
- Optional: `cupy`, `cornucopia-pytorch`, `ants`

## Memory Optimization for Large Datasets

For extracting 500+ patches from large volumes (>5GB), we've implemented several memory optimizations:

### Key Features
- **Memory-Mapped Loading**: Uses `mmap=True` to avoid loading entire volumes into RAM
- **Batch Processing**: Processes patches in batches with garbage collection between batches
- **Volume Caching**: Pre-loads volumes once and reuses for all patches (eliminates redundant I/O)
- **Checkpoint System**: Saves progress every N patches to allow recovery from OOM kills

### Usage
```bash
python3 syntract.py \
  --input large_volume.nii.gz \
  --trk fibers.trk \
  --skip_synthesis \              # Skip if files already processed
  --enable_patch_extraction \
  --total_patches 500 \
  --patch_batch_size 50 \        # Adjust based on available memory
  --patch_size 600 1 600
```


## License

MIT License - see LICENSE file for details.
