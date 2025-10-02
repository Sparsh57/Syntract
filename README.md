# SynTract: MRI Synthesis & Tractography Visualization

A streamlined Python pipeline for MRI processing, tractography synthesis, and dark field microscopy-style visualization. Features unified CLI, ANTs integration, and robust patch extraction for neuroimaging research.

## ✨ Features

- **Unified Pipeline**: Single command processing from raw NIfTI/TRK to visualizations
- **ANTs Integration**: Spatial transformations and registration workflows
- **Patch Extraction**: 3D patches via re-synthesis at random coordinates
- **Memory-Optimized**: Batch processing with checkpoints for large datasets (500+ patches)
- **Skip Synthesis**: Direct patch extraction from preprocessed files
- **Batch Processing**: Multiple TRK files with shared NIfTI
- **Dark Field Visualization**: Publication-ready medical imaging with enhanced contrast

## 🔧 Installation

```bash
git clone https://github.com/Sparsh57/Syntract.git
cd Syntract
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage
```bash
python syntract.py --input brain.nii.gz --trk fibers.trk --output result
```

### With ANTs Transformation
```bash
python syntract.py --input brain.nii.gz --trk fibers.trk --use_ants \
  --ants_warp warp.nii.gz --ants_iwarp iwarp.nii.gz --ants_aff affine.mat
```

### Patch Extraction (Memory-Optimized)
```bash
# With synthesis
python syntract.py --input brain.nii.gz --trk fibers.trk \
  --enable_patch_extraction --total_patches 500 --patch_size 600 1 600 \
  --patch_batch_size 50

# Skip synthesis (use preprocessed files directly)
python syntract.py --input whole_output.nii.gz --trk whole_output.trk \
  --skip_synthesis --enable_patch_extraction --total_patches 500 \
  --patch_batch_size 50
```

### Batch Processing
```bash
python cumulative.py  # Edit paths in script
```

## 📋 Parameters

### Essential Arguments
| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` | str | Input NIfTI file path (required) |
| `--trk` | str | Input TRK file path (required) |
| `--output` | str | Output base name (default: "output") |

### Synthesis Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--new_dim` | int×3 | [116, 140, 96] | Target dimensions (X Y Z) |
| `--voxel_size` | float | 0.5 | Target voxel size in mm |

### ANTs Transformation
| Parameter | Type | Description |
|-----------|------|-------------|
| `--use_ants` | flag | Enable ANTs transformation |
| `--ants_warp` | str | ANTs warp field file |
| `--ants_iwarp` | str | ANTs inverse warp field file |
| `--ants_aff` | str | ANTs affine transformation file |

### Patch Extraction
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable_patch_extraction` | flag | | Enable 3D patch extraction |
| `--patch_output_dir` | str | "patches" | Directory for patch outputs |
| `--total_patches` | int | 10 | Total number of patches to extract |
| `--patch_size` | int×3 | [300, 15, 300] | Patch dimensions (width, height, depth) |
| `--min_streamlines_per_patch` | int | 30 | Minimum streamlines required per patch |
| `--max_patch_trials` | int | 100 | Maximum trials to find adequate streamlines |
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

## 🗂️ Batch Processing (cumulative.py)

Process multiple TRK files with a shared NIfTI file. Supports various configurations:

### Available Configurations
- `standard`: Basic processing with your CLI parameters
- `ultra_crisp`: Maximum detail with edge preservation
- `patch_extraction`: Extract patches distributed across TRK files
- `high_throughput_patches`: 200 smaller patches for fast processing
- `quality_patches`: 50 high-quality large patches

### Key Batch Parameters
| Parameter | Description |
|-----------|-------------|
| `nifti_path` | Common NIfTI file for all TRK files |
| `trk_dir` | Directory containing TRK files |
| `config_choice` | Processing configuration to use |

Change the `config_choice` variable in `cumulative.py` to switch between configurations.

## 📁 Output Structure

### Standard Processing
```
output_name.nii.gz         # Processed NIfTI
output_name.trk            # Processed TRK
visualizations/            # Generated images
```

### Patch Extraction
```
patches/
├── patch_0001.nii.gz     # Patch NIfTI
├── patch_0001.trk        # Patch TRK
├── patch_0001_visualization.png
├── patch_0001_visualization_mask_slice0.png
└── patch_extraction_summary.json
```

### Batch Processing
```
syntract_submission/
├── processed_files/       # All processed .nii.gz and .trk files
├── patches/              # Patches organized by TRK file
└── visualizations/       # Visualizations organized by TRK file
```

## 🐍 Python API

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
from cumulative import batch_process_trk_files

results = batch_process_trk_files(
    nifti_path="shared_brain.nii.gz",
    trk_dir="trk_files_directory",
    new_dim=[800, 20, 800],
    voxel_size=0.05,
    enable_patch_extraction=True,
    total_patches=100
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

## 📦 Dependencies

- Core: `numpy`, `nibabel`, `matplotlib`, `scikit-image`, `scipy`, `dipy`
- Optional: `cupy`, `cornucopia-pytorch`, `ants`

## 🧠 Memory Optimization for Large Datasets

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

### Memory Guidelines
| System Memory | Recommended `--patch_batch_size` |
|---------------|----------------------------------|
| 16 GB | 10-25 |
| 32 GB | 25-50 |
| 64 GB | 50-100 |
| 128+ GB | 100-200 |

For detailed information, see:
- **[MEMORY_OPTIMIZATIONS.md](MEMORY_OPTIMIZATIONS.md)** - Complete memory optimization guide
- **[SKIP_SYNTHESIS_GUIDE.md](SKIP_SYNTHESIS_GUIDE.md)** - Skip synthesis feature documentation

## 📄 License

MIT License - see LICENSE file for details.
