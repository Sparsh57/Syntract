# SynTract: Advanced MRI Processing and Tractography Visualization Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python package for advanced MRI processing, tractography synthesis, and dark field microscopy-style visualization. Features high-performance processing pipelines, sophisticated augmentation capabilities, and a unified command-line interface for neuroimaging research.

## 🚀 New: Combined Pipeline

**SynTract now includes a unified interface that seamlessly combines MRI processing with advanced visualization!**

### Command Line Interface
```bash
# Process MRI data and generate visualizations in one step
python syntract.py \
  --input brain.nii.gz \
  --trk fibers.trk \
  --output processed_data \
  --new_dim 256 256 256 \
  --voxel_size 0.5 \
  --viz_output_dir visualization_dataset \
  --n_examples 50 \
  --save_masks
```

### Python API with process_syntract (Recommended)
```python
from syntract import process_syntract

# Same functionality with programmatic control and better error handling
result = process_syntract(
    input_nifti="brain.nii.gz",
    input_trk="fibers.trk",
    output_base="processed_data",
    new_dim=[256, 256, 256],
    voxel_size=[0.5],
    viz_output_dir="visualization_dataset",
    n_examples=50,
    save_masks=True
)
```

The `process_syntract` function is the main programmatic interface and provides better error handling, return values, and integration into larger workflows compared to the command-line interface.

## Features

### 🔄 Combined Pipeline - One Command, Complete Workflow
- **Unified processing and visualization** in a single command
- **Automatic file handling** between synthesis and visualization stages  
- **Complete parameter mapping** from both synthesis and syntract_viewer
- **End-to-end workflow** from raw data to publication-ready visualizations
- **ANTs integration support** for spatial transformations
- **Flexible output control** with intermediate file retention options

### 🧠 Synthesis - High-Performance Processing Pipeline
- **GPU-accelerated processing** with CUDA/CuPy support and automatic CPU fallback
- **Advanced streamline interpolation** (Linear, Hermite, RBF) with sub-voxel precision
- **Intelligent NIfTI resampling** to arbitrary dimensions and voxel sizes
- **ANTs integration** for spatial transformations and registration workflows
- **Memory-efficient processing** with automatic chunking for large datasets
- **Parallel processing** with configurable job counts and resource optimization

### 🎨 Syntract Viewer - Advanced Visualization Engine
- **Dark field microscopy-style visualization** with publication-ready image generation
- **Multi-orientation support** (axial, coronal, sagittal) with intelligent slice selection
- **Ground truth mask generation** for machine learning training with bundle labeling
- **Advanced contrast enhancement** using CLAHE and adaptive methods
- **Smart brain masking** preserving anatomical structures and ventricles
- **Spatial ROI selection** with fiber density variation for diverse datasets
- **Cornucopia integration** for realistic medical imaging augmentations
- **Batch processing tools** with command-line interface for dataset generation
- **Smart fiber bundle detection** with morphological operations and gap closing
- **Customizable color schemes** including blue-tinted dark field effects

## Installation

### Basic Installation

```bash
git clone https://github.com/yourusername/SynTract.git
cd SynTract
pip install -e .
```

### With Advanced Features

```bash
# With GPU acceleration
pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version

# With Cornucopia medical imaging augmentations
pip install cornucopia-pytorch

# With ANTs registration support
# Install ANTs separately: https://github.com/ANTsX/ANTs
```

## Quick Start

### 🔄 Combined Pipeline (Recommended)

Process MRI data and generate visualizations in one command:

```bash
# Basic combined processing
python syntract.py \
  --input brain.nii.gz \
  --trk streamlines.trk \
  --output resampled_data \
  --viz_output_dir fiber_visualizations \
  --n_examples 20

# Advanced processing with ANTs transforms
python syntract.py \
  --input subject_brain.nii.gz \
  --trk subject_tracts.trk \
  --output processed_subject \
  --new_dim 800 20 800 \
  --voxel_size 0.05 0.05 0.05 \
  --patch_center 83 24 37 \
  --use_ants \
  --ants_warp warp_field.nii.gz \
  --ants_iwarp inverse_warp.nii.gz \
  --ants_aff affine_transform.mat \
  --viz_output_dir subject_visualizations \
  --n_examples 5 \
  --close_gaps \
  --min_bundle_size 800 \
  --density_threshold 0.1 \
  --save_masks
```

### 🧠 Synthesis Only

High-performance MRI and tractography processing:

```bash
python synthesis/main.py \
  --input brain.nii.gz \
  --trk streamlines.trk \
  --output processed \
  --new_dim 256 256 256 \
  --voxel_size 0.5 \
  --interp hermite \
  --use_gpu
```

### 🎨 Visualization Only

Generate dark field microscopy-style visualizations:

```bash
python syntract_viewer/generate_fiber_examples.py \
  --nifti brain.nii.gz \
  --trk streamlines.trk \
  --output_dir visualizations \
  --examples 50 \
  --save_masks \
  --label_bundles
```

### Python API Usage

#### Combined Processing with process_syntract
```python
from syntract import process_syntract

# Process and visualize in one call - the main programmatic interface
result = process_syntract(
    input_nifti="brain.nii.gz",
    input_trk="streamlines.trk",
    output_base="processed_subject",
    viz_output_dir="visualizations",
    new_dim=[256, 256, 256],
    voxel_size=[0.5, 0.5, 0.5],
    n_examples=25,
    save_masks=True,
    label_bundles=True
)

# Check results
if result['success']:
    print(f"Processing completed successfully!")
    print(f"Processed files: {result['processed_nifti']}, {result['processed_trk']}")
    print(f"Visualizations: {result['visualization_dir']}")
else:
    print(f"Processing failed: {result['error']}")
```

#### Advanced Processing with ANTs and Spatial Subdivisions
```python
# Complete workflow with ANTs transformations and spatial subdivisions
result = process_syntract(
    input_nifti="subject_brain.nii.gz",
    input_trk="subject_tracts.trk",
    output_base="processed_subject",
    viz_output_dir="subject_visualizations",
    
    # Processing parameters
    new_dim=[800, 20, 800],
    voxel_size=[0.05, 0.05, 0.05],
    patch_center=[83, 24, 37],
    interp='hermite',
    use_gpu=True,
    
    # ANTs transformation
    use_ants=True,
    ants_warp="warp_field.nii.gz",
    ants_iwarp="inverse_warp.nii.gz",
    ants_aff="affine_transform.mat",
    
    # Visualization options
    n_examples=5,
    save_masks=True,
    label_bundles=True,
    close_gaps=True,
    min_bundle_size=800,
    density_threshold=0.1,
    
    # Spatial subdivisions for region-based analysis
    use_spatial_subdivisions=True,
    n_subdivisions=8,
    max_streamlines_per_subdivision=50000,
    min_streamlines_per_region=10,
    skip_empty_regions=True,
    
    # Enhancement options
    background_preset='preserve_edges',
    enable_sharpening=True,
    sharpening_strength=1.0,
    contrast_method='clahe'
)
```

#### Batch Processing Multiple Files with cumulative.py
```python
from cumulative import batch_process_trk_files

# Process multiple TRK files with shared NIfTI
results = batch_process_trk_files(
    nifti_path="shared_brain.nii.gz",
    trk_dir="trk_files_directory",
    
    # Same parameters as process_syntract
    new_dim=[256, 256, 256],
    voxel_size=[0.5],
    n_examples=10,
    save_masks=True,
    label_bundles=True,
    use_spatial_subdivisions=True,
    n_subdivisions=4
)

print(f"Processed {len(results['successful'])} files successfully")
print(f"Failed: {len(results['failed'])} files")
```

#### Using Predefined Configurations from cumulative.py
```python
from cumulative import get_processing_configurations, batch_process_trk_files

# Get predefined processing configurations
configs = get_processing_configurations()

# Available configurations:
# - 'standard': Basic processing without subdivisions
# - 'ultra_crisp': Maximum detail/sharpness, no smoothing
# - 'with_subdivisions': Standard + 8 spatial subdivisions  
# - 'sparse_subdivisions': Standard + 4 spatial subdivisions (sparse data)
# - 'thin_dimension': Optimized for thin dimensions like 800x20x800
# - 'debug_subdivisions': Minimal config for debugging
# - 'crisp_subdivisions': High-detail subdivisions with edge preservation

# Use crisp subdivisions configuration
processing_params = configs['crisp_subdivisions']

# Batch process with predefined configuration
results = batch_process_trk_files(
    nifti_path="examples/example_data/sub-MF278_sample-brain_desc-blockface_stacked_masked_grayscale_level4.nii.gz",
    trk_dir="dwi",  # Directory containing TRK files
    **processing_params
)
```

#### Command-Line Batch Processing
```bash
# Edit cumulative.py to set your paths and configuration, then run:
python cumulative.py

# Or process a single file with your exact parameters:
python cumulative.py single
```

#### Dark Field Tractography Visualization

```python
from syntract_viewer import visualize_nifti_with_trk

# Create publication-ready dark field visualization
visualize_nifti_with_trk(
    nifti_file="brain.nii.gz",
    trk_file="tracts.trk",
    output_file="dark_field_viz.png",
    slice_mode="coronal",
    save_masks=True,
    label_bundles=True
)
```

#### Machine Learning Dataset Generation

```python
from syntract_viewer import generate_varied_examples

# Generate training dataset with varied fiber densities
generate_varied_examples(
    nifti_file="brain.nii.gz",
    trk_file="tracts.trk",
    output_dir="./training_dataset",
    n_examples=100,
    save_masks=True,
    min_fiber_percentage=5.0,
    max_fiber_percentage=100.0,
    label_bundles=True,
    close_gaps=True
)
```

#### High-Performance Processing

```python
from synthesis import process_and_save

# Process and resample MRI data with streamlines
process_and_save(
    original_nifti_path="brain.nii.gz",
    original_trk_path="streamlines.trk",
    target_voxel_size=0.5,
    target_dimensions=(116, 140, 96),
    use_gpu=True,
    interpolation_method='hermite'
)
```

## Package Architecture

```
SynTract/
├── syntract.py                     # 🔄 Unified CLI and process_syntract function
├── cumulative.py                   # 🔄 Batch processing and predefined configurations
├── synthesis/                      # 🧠 High-performance processing pipeline
│   ├── main.py                    # Main processing workflow
│   ├── nifti_preprocessing.py     # NIfTI resampling with memory management
│   ├── streamline_processing.py   # Streamline transformation and clipping
│   ├── densify.py                 # Advanced interpolation algorithms
│   ├── ants_transform.py          # ANTs integration
│   └── compare_interpolation.py   # Method comparison tools
└── syntract_viewer/               # 🎨 Visualization and dataset generation
    ├── core.py                    # Core visualization functions
    ├── generation.py              # Dataset generation with variations
    ├── contrast.py                # CLAHE and adaptive contrast methods
    ├── masking.py                 # Smart brain and fiber masking
    ├── effects.py                 # Dark field microscopy effects
    ├── utils.py                   # Utilities and color mapping
    ├── cornucopia_augmentation.py # Medical imaging augmentations
    └── generate_fiber_examples.py # Command-line interface
```

## API Reference

### process_syntract Function

The main programmatic interface for the SynTract pipeline, combining both synthesis and visualization in a single function call.

```python
from syntract import process_syntract

result = process_syntract(
    input_nifti,           # Required: Path to input NIfTI file
    input_trk,             # Required: Path to input TRK file
    output_base=None,      # Output base name (auto-generated if None)
    **kwargs               # Additional parameters (see below)
)
```

#### Parameters

**Required:**
- `input_nifti` (str): Path to input NIfTI file
- `input_trk` (str): Path to input TRK file

**Processing Parameters:**
- `output_base` (str): Base name for output files (default: auto-generated)
- `new_dim` (list): Target dimensions [x, y, z] (default: [116, 140, 96])
- `voxel_size` (list): Target voxel size(s) (default: [0.5])
- `patch_center` (list): Patch center in mm [x, y, z] (default: None)
- `interp` (str): Interpolation method: 'hermite', 'linear', 'rbf' (default: 'hermite')
- `step_size` (float): Densification step size (default: 0.5)
- `use_gpu` (bool): Enable GPU acceleration (default: True)
- `jobs` (int): Number of parallel jobs (default: 8)
- `max_gb` (float): Memory limit in GB (default: 64.0)

**ANTs Integration:**
- `use_ants` (bool): Enable ANTs transforms (default: False)
- `ants_warp` (str): ANTs warp field file path
- `ants_iwarp` (str): ANTs inverse warp field file path
- `ants_aff` (str): ANTs affine transform file path
- `force_dimensions` (bool): Force exact dimensions (default: False)
- `transform_mri_with_ants` (bool): Transform MRI with ANTs (default: False)

**Visualization Parameters:**
- `viz_output_dir` (str): Output directory for visualizations
- `n_examples` (int): Number of examples to generate (default: 5)
- `viz_prefix` (str): Prefix for visualization files (default: 'synthetic_')
- `slice_mode` (str): Slice orientation: 'axial', 'coronal', 'sagittal' (default: 'coronal')
- `specific_slice` (int): Specific slice number (default: None)
- `streamline_percentage` (float): Percentage of streamlines to display (default: 100.0)
- `tract_linewidth` (float): Line width for tracts (default: 1.0)

**Mask and Bundle Parameters:**
- `save_masks` (bool): Save fiber masks (default: False)
- `use_high_density_masks` (bool): Use high-density masking (default: False)
- `label_bundles` (bool): Label distinct bundles (default: False)
- `mask_thickness` (int): Mask thickness in pixels (default: 1)
- `min_fiber_percentage` (float): Minimum fiber percentage (default: 10.0)
- `max_fiber_percentage` (float): Maximum fiber percentage (default: 100.0)
- `min_bundle_size` (int): Minimum bundle size (default: 20)
- `density_threshold` (float): Density threshold for masking (default: 0.15)

**Enhancement Parameters:**
- `contrast_method` (str): Contrast method: 'clahe', 'none' (default: 'clahe')
- `background_preset` (str): Background enhancement preset: 'preserve_edges', None (default: 'preserve_edges')
- `cornucopia_preset` (str): Cornucopia augmentation preset (default: None)
- `enable_sharpening` (bool): Enable image sharpening (default: True)
- `sharpening_strength` (float): Sharpening strength 0.0-2.0 (default: 0.5)
- `close_gaps` (bool): Close gaps in fiber masks (default: False)
- `closing_footprint_size` (int): Footprint size for gap closing (default: 5)

**Spatial Subdivisions:**
- `use_spatial_subdivisions` (bool): Enable spatial subdivisions (default: False)
- `n_subdivisions` (int): Number of subdivisions (default: 8)
- `max_streamlines_per_subdivision` (int): Max streamlines per subdivision (default: 50000)
- `min_streamlines_per_region` (int): Min streamlines per region (default: 10)
- `skip_empty_regions` (bool): Skip regions with no streamlines (default: True)

**Pipeline Control:**
- `skip_synthesis` (bool): Skip synthesis stage, use original files (default: False)
- `skip_visualization` (bool): Skip visualization stage (default: False)
- `keep_processed` (bool): Keep processed files (default: True)
- `randomize_viz` (bool): Randomize visualization parameters (default: False)
- `random_state` (int): Random seed for reproducibility (default: None)
- `temp_dir` (str): Custom temporary directory (default: None)

#### Returns

Dictionary with the following keys:
- `success` (bool): Whether the processing completed successfully
- `output_base` (str): Base name used for output files
- `visualization_dir` (str): Directory containing visualizations
- `processed_nifti` (str): Path to processed NIfTI file (if kept)
- `processed_trk` (str): Path to processed TRK file (if kept)
- `error` (str): Error message if processing failed

## Command-Line Reference

### Combined Pipeline Options

```bash
python syntract.py [options]

# Input/Output
--input PATH              Input NIfTI file
--trk PATH               Input TRK file  
--output PREFIX          Output prefix for processed files
--viz_output_dir DIR     Output directory for visualizations

# Processing Parameters
--new_dim X Y Z          Target dimensions (default: 116 140 96)
--voxel_size SIZE        Target voxel size (default: 0.5)
--patch_center X Y Z     Patch center in mm (optional)
--interp METHOD          Interpolation method: hermite|linear|rbf
--step_size FLOAT        Densification step size (default: 0.5)

# ANTs Integration
--use_ants               Enable ANTs transforms
--ants_warp PATH         ANTs warp field file
--ants_iwarp PATH        ANTs inverse warp file
--ants_aff PATH          ANTs affine transform file

# Visualization Options
--n_examples N           Number of examples to generate (default: 5)
--save_masks             Save fiber masks
--label_bundles          Label distinct bundles
--close_gaps             Close gaps in fiber masks
--min_bundle_size N      Minimum bundle size (default: 20)
--density_threshold F    Density threshold (default: 0.15)

# Spatial Subdivisions
--use_spatial_subdivisions    Enable spatial subdivisions
--n_subdivisions N           Number of subdivisions (default: 8)
--max_streamlines_per_subdivision N  Max streamlines per subdivision
--min_streamlines_per_region N       Min streamlines per region

# Performance
--use_gpu               Enable GPU acceleration (default: True)
--jobs N                Number of parallel jobs (default: 8)
--max_gb FLOAT          Memory limit in GB (default: 64)
```

## Core Features

### Dark Field Microscopy Visualization

SynTract's signature feature creates stunning dark field microscopy-style images:

```python
# Custom dark field parameters
intensity_params = {
    'gamma': 1.1,
    'threshold': 0.02,
    'contrast_stretch': (0.5, 99.5),
    'color_scheme': 'blue',
    'blue_tint': 0.3
}

visualize_nifti_with_trk(
    nifti_file="brain.nii.gz",
    trk_file="tracts.trk",
    output_file="publication_figure.png",
    intensity_params=intensity_params,
    tract_linewidth=1.2,
    save_masks=True
)
```

### Smart Fiber Bundle Detection

Advanced masking with morphological operations and bundle labeling:

```python
from syntract_viewer import visualize_nifti_with_trk

# Generate masks with distinct bundle labels
fig, axes, masks, labeled_masks = visualize_nifti_with_trk(
    nifti_file="brain.nii.gz",
    trk_file="tracts.trk",
    save_masks=True,
    label_bundles=True,
    min_bundle_size=50,
    close_gaps=True,
    closing_footprint_size=7,
    density_threshold=0.15
)
```

### GPU-Accelerated Processing

```python
# Automatic GPU detection and utilization
process_and_save(
    original_nifti_path="high_res_brain.nii.gz",
    original_trk_path="dense_streamlines.trk",
    target_dimensions=(512, 512, 512),
    use_gpu=True,
    max_output_gb=16,
    num_jobs=12
)
```

### Advanced Interpolation Methods

```python
# Compare interpolation quality
from synthesis.compare_interpolation import compare_interpolations

metrics = compare_interpolations(
    trk_file="streamlines.trk",
    step_size=0.3,
    methods=['linear', 'hermite', 'rbf'],
    num_streamlines=5000
)
```

## Dependencies

### Core Requirements
- numpy >= 1.18.0
- nibabel >= 3.0.0
- matplotlib >= 3.3.0
- scikit-image >= 0.17.0
- scipy >= 1.5.0
- dipy >= 1.4.0
- joblib >= 1.0.0
- tqdm >= 4.50.0

### Optional Enhancements
- cornucopia-pytorch (advanced medical imaging augmentations)
- cupy >= 9.0.0 (GPU acceleration)
- numba >= 0.50.0 (CUDA kernels)
- ants (spatial transformations and registration)



## Examples and Tutorials

### Example 1: Basic Processing and Visualization

**Command Line:**
```bash
# Download example data (replace with your data)
# Process and visualize in one command
python syntract.py \
  --input example_brain.nii.gz \
  --trk example_tracts.trk \
  --output example_processed \
  --viz_output_dir example_viz \
  --n_examples 10 \
  --save_masks \
  --label_bundles
```

**Python API (Recommended):**
```python
from syntract import process_syntract

# Same processing using the Python API
result = process_syntract(
    input_nifti="example_brain.nii.gz",
    input_trk="example_tracts.trk",
    output_base="example_processed",
    viz_output_dir="example_viz",
    n_examples=10,
    save_masks=True,
    label_bundles=True
)

if result['success']:
    print("✓ Processing completed successfully!")
    print(f"Visualizations: {result['visualization_dir']}")
else:
    print(f"✗ Processing failed: {result['error']}")
```

### Example 2: High-Resolution Processing with ANTs

**Command Line:**
```bash
# For data with ANTs registration
python syntract.py \
  --input high_res_brain.nii.gz \
  --trk high_res_tracts.trk \
  --output hr_processed \
  --new_dim 512 512 512 \
  --voxel_size 0.25 \
  --use_ants \
  --ants_warp warp_field.nii.gz \
  --ants_iwarp inverse_warp.nii.gz \
  --ants_aff affine.mat \
  --viz_output_dir hr_visualizations \
  --n_examples 25
```

**Python API (Recommended):**
```python
from syntract import process_syntract

# High-resolution processing with ANTs
result = process_syntract(
    input_nifti="high_res_brain.nii.gz",
    input_trk="high_res_tracts.trk",
    output_base="hr_processed",
    viz_output_dir="hr_visualizations",
    new_dim=[512, 512, 512],
    voxel_size=[0.25],
    use_ants=True,
    ants_warp="warp_field.nii.gz",
    ants_iwarp="inverse_warp.nii.gz",
    ants_aff="affine.mat",
    n_examples=25
)
```

### Example 3: Machine Learning Dataset Generation

**Command Line:**
```bash
# Generate large training dataset
python syntract.py \
  --input training_brain.nii.gz \
  --trk training_tracts.trk \
  --output training_processed \
  --viz_output_dir ml_dataset \
  --n_examples 500 \
  --save_masks \
  --label_bundles \
  --min_fiber_percentage 5 \
  --max_fiber_percentage 100 \
  --close_gaps
```

**Python API (Recommended):**
```python
from syntract import process_syntract

# Generate machine learning training dataset
result = process_syntract(
    input_nifti="training_brain.nii.gz",
    input_trk="training_tracts.trk",
    output_base="training_processed",
    viz_output_dir="ml_dataset", 
    n_examples=500,
    save_masks=True,
    label_bundles=True,
    min_fiber_percentage=5.0,
    max_fiber_percentage=100.0,
    close_gaps=True
)
```

### Example 4: Batch Processing Multiple Files

**Using cumulative.py for Batch Processing:**
```python
# Create a script using cumulative.py functionality
from cumulative import batch_process_trk_files, get_processing_configurations

# Get configuration optimized for your data type
configs = get_processing_configurations()
selected_config = configs['crisp_subdivisions']  # High-quality with subdivisions

# Process all TRK files in a directory
results = batch_process_trk_files(
    nifti_path="shared_brain.nii.gz",
    trk_dir="./fiber_data",  # Directory with multiple .trk files
    **selected_config
)

# Check results
print(f"Successfully processed: {len(results['successful'])} files")
print(f"Failed: {len(results['failed'])} files")

# Print organized output structure
for success in results['successful']:
    print(f"✓ {success['filename']} → {success['visualization_dir']}")
```

**Command Line Batch Processing:**
```bash
# Edit cumulative.py paths and run batch processing
python cumulative.py

# Process single file with exact parameters from cumulative.py
python cumulative.py single
```

## Contributing

We welcome contributions to SynTract! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/yourusername/SynTract.git
cd SynTract
pip install -e ".[dev]"
pre-commit install
```

### Testing

```bash
pytest tests/
python -m synthesis.compare_interpolation
python syntract.py --help
```

## Related Projects

- [DIPY](https://dipy.org/) - Diffusion imaging in Python
- [NiBabel](https://nipy.org/nibabel/) - Neuroimaging file format access
- [Cornucopia](https://cornucopia.readthedocs.io/) - Medical imaging augmentations
- [ANTs](http://stnava.github.io/ANTs/) - Advanced Normalization Tools
- [CuPy](https://cupy.dev/) - GPU-accelerated computing

## Support
- **Issues**: [GitHub Issues](https://github.com/yourusername/SynTract/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/SynTract/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*SynTract is developed by the LINC Team to advance MRI processing, tractography synthesis, and visualization in neuroimaging research.*


