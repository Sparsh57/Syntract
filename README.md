# SynTract: MRI Synthesis & Tractography Visualization

A streamlined Python pipeline for MRI processing, tractography synthesis, and dark field microscopy-style visualization. Features unified CLI, ANTs integration, and patch-first optimization for dramatic performance improvements in neuroimaging research.

## Features

- **Patch Processing**: 80-95% performance improvement with patch extraction (default mode)
- **Unified Pipeline**: Single command processing from raw NIfTI/TRK to visualizations
- **ANTs Integration**: Spatial transformations and registration workflows
- **Auto-Dimension Calculation**: Intelligent target sizing based on input characteristics
- **Zero-Tolerance Spatial Accuracy**: Perfect bounds enforcement with enhanced curvature preservation
- **Memory-Optimized**: Efficient processing for large datasets with minimal memory usage
- **Intelligent Batch Processing**: Auto-optimization based on streamline density and file characteristics
- **Cornucopia**: Intelligent preset selection for realistic background textures
- **Batch Processing**: Multiple TRK files with shared NIfTI and memory optimization
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
python cumulative.py  # Edit paths in script
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
| `--voxel_size` | float | 0.05 | Target voxel size in mm |
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
| `--patch_size` | int×3 | [600, 1, 600] | Patch dimensions (width, height, depth) |
| `--min_streamlines_per_patch` | int | 20 | Minimum streamlines required per patch |
| `--patch_batch_size` | int | 50 | Batch size for memory management |
| `--random_state` | int | | Random seed for reproducible extraction |
| `--patch_prefix` | str | "patch" | Prefix for patch files |

### Visualization
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_examples` | int | 10 | Number of visualization examples |
| `--viz_prefix` | str | "synthetic_" | Prefix for visualization files |
| `--enable_orange_blobs` | flag | | Enable orange blob injection site artifacts |
| `--orange_blob_probability` | float | 0.3 | Probability of applying orange blobs (0.0-1.0) |

### Mask & Bundle Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save_masks` | flag | True | Save binary masks alongside visualizations |
| `--use_high_density_masks` | flag | True | Use high-density mask generation |
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

Process multiple TRK files with a shared NIfTI file efficiently. Features intelligent memory optimization and automatic processing strategy adaptation.

### In-Memory Optimization Features

The `process_batch` function includes several intelligent optimizations:

- **Automatic Streamline Analysis**: Analyzes each TRK file to determine optimal processing strategy
- **Dynamic Patch Allocation**: Automatically adjusts patch count based on file size and streamline density
- **Memory-Aware Processing**: Optimizes memory usage for large files (>100,000 streamlines)
- **Auto-Dimension Calculation**: Calculates target dimensions based on input NIfTI characteristics
- **Smart Output Sizing**: Automatically determines output image size from patch dimensions

### Usage

#### Command Line Interface
```bash
# Basic batch processing with auto-optimization
python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/

# With custom parameters
python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \
  --total-patches 50 --n-examples 200 --voxel-size 0.05

# For thin slice data (optimized settings)
python cumulative.py --nifti brain.nii.gz --trk-dir ./trk_files/ \
  --patch-size 256 8 256 --total-patches 30 --n-examples 200
```

#### Python API
```python
from cumulative import process_batch

# Simple batch processing with automatic optimization
results = process_batch(
    nifti_file='brain.nii.gz',
    trk_directory='./trk_files/',
    patches=30,                    # Total patches across all files
    n_examples=200                 # Total visualizations to generate
)

# Advanced configuration
results = process_batch(
    nifti_file='brain.nii.gz',
    trk_directory='./trk_files/',
    output_dir='results',
    patches=50,
    patch_size=[600, 1, 600],      # Auto-determines 600x600 output images
    min_streamlines_per_patch=20,
    voxel_size=0.05,
    new_dim=None,                  # Auto-calculated from input
    n_examples=200,
    enable_orange_blobs=True,
    cleanup_intermediate=True      # Saves disk space
)
```
### Batch ANTs Registration Function
```python
from batch_ants_trk_registration import batch_ants_registration

results = batch_ants_registration(
    input_folder="trk_files/",
    output_folder="registered_trk/", 
    ants_warp_path="warp.nii.gz",
    ants_iwarp_path="iwarp.nii.gz",
    ants_aff_path="affine.mat",
    reference_mri_path="brain.nii.gz"
)
```

#### In-Memory Processing API
```python
from cumulative import process_patches_inmemory

# Generate patches and visualizations in-memory (no file I/O)
images, masks = process_patches_inmemory(
    input_nifti='brain.nii.gz',
    trk_file='fibers.trk',         # Single file or directory
    num_patches=50,
    patch_size=[512, 1, 512],      # 512x512 output images
    enable_orange_blobs=True,      # Enable injection site simulation
    orange_blob_probability=0.3,   # 30% chance per patch
    random_state=42                # For reproducible results
)

# Use results directly with matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
for i in range(min(3, len(images))):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.show()
```
```

### Key Batch Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nifti_file` | str | | Common NIfTI file for all TRK files (required) |
| `trk_directory` | str | | Directory containing TRK files (required) |
| `output_dir` | str | "results" | Output directory |
| `patches` | int | 30 | Total patches to extract across all files |
| `patch_size` | list | [600, 1, 600] | Patch dimensions (auto-calculated if None) |
| `n_examples` | int | 10 | Number of visualization examples to generate |
| `voxel_size` | float | 0.05 | Target voxel size in mm |
| `new_dim` | tuple | None | Target dimensions (auto-calculated if None) |

### Automatic Processing Strategy

The batch processor automatically optimizes based on file characteristics:

- **Large files (>100k streamlines)**: Increases patch count to distribute processing load
- **Sparse files (<10 streamlines)**: Uses minimal patches to avoid empty regions  
- **Standard files**: Uses balanced patch distribution for optimal coverage
- **Memory constraints**: Automatically adjusts cleanup intervals

### Return Structure

The `process_batch` function returns a comprehensive results dictionary:

```python
{
    'successful': [                # List of successfully processed files
        {
            'file': 'fiber1.trk',
            'time': 45.2,           # Processing time in seconds
            'patches': 5,           # Patches allocated to this file
            'patches_extracted': 5   # Actually extracted patches
        }
    ],
    'failed': [                    # List of failed files with error info
        {
            'file': 'fiber2.trk',
            'error': 'Error message',
            'time': 12.1
        }
    ],
    'total_time': 128.5            # Total processing time
}
```

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

### Batch Processing (cumulative.py)
```
results/                       # Main output directory
├── processed/                 # Intermediate processed files
│   ├── processed_fiber1.nii
│   └── processed_fiber1.trk
├── patches/                   # Patches organized by TRK file
│   ├── fiber1/               # Individual TRK file patches
│   │   ├── fiber1_patch_0001_visualization.png
│   │   ├── fiber1_patch_0001_visualization_mask_slice0.png
│   │   └── ...
│   └── fiber2/
│       ├── fiber2_patch_0001_visualization.png
│       └── ...
└── summary.json              # Batch processing results summary
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

# Intelligent batch processing with auto-optimization
results = process_batch(
    nifti_file="shared_brain.nii.gz",
    trk_directory="trk_files_directory",
    output_dir="results",
    patches=50,                    # Total patches across all files
    patch_size=[600, 1, 600],      # Determines 600x600 output images
    n_examples=200,                # Total visualizations to generate
    voxel_size=0.05,
    new_dim=None,                  # Auto-calculated
    enable_orange_blobs=True,
    cleanup_intermediate=True      # Saves disk space
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
    new_dim,                        # Target dimensions [X, Y, Z] - auto-calculated if None
    voxel_size=0.05,                # Target voxel size in mm
    
    # ANTs transformation
    use_ants=False,                 # Enable ANTs transformation
    ants_warp_path=None,           # ANTs warp field file
    ants_iwarp_path=None,          # ANTs inverse warp field file 
    ants_aff_path=None,            # ANTs affine transformation file
    
    # Slice extraction
    slice_count=None,              # Number of coronal slices to extract
    enable_slice_extraction=False, # Enable slice extraction mode
    slice_output_dir=None,         # Directory for slice outputs
    auto_batch_process=False,      # Auto-process all extracted slices
    
    # Patch extraction (Default Mode)
    disable_patch_processing=False, # Disable patch processing (use traditional synthesis)
    total_patches=50,              # Total number of patches to extract
    patch_size=[600, 1, 600],      # Patch dimensions [width, height, depth]
    min_streamlines_per_patch=20,  # Minimum streamlines per patch
    patch_output_dir="patches",    # Directory for patch outputs
    patch_batch_size=50,           # Batch size for memory management
    patch_prefix="patch",          # Prefix for patch files
    cleanup_intermediate=True,     # Remove intermediate files to save space
    
    # Visualization
    n_examples=10,                 # Number of visualization examples
    viz_prefix="synthetic_",       # Prefix for visualization files
    enable_orange_blobs=False,     # Enable orange blob artifacts
    orange_blob_probability=0.3,   # Probability of orange blobs (0.0-1.0)
    
    # Mask & Bundle parameters
    save_masks=True,               # Save binary masks alongside visualizations
    use_high_density_masks=True,   # Use high-density mask generation
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
