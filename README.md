# MRISynth: Advanced MRI Processing and Tractography Visualization Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python package for advanced MRI processing, tractography synthesis, and dark field microscopy-style visualization. Features high-performance processing pipelines, sophisticated augmentation capabilities, and a unified command-line interface for neuroimaging research.

## 🚀 New: Combined Pipeline

**MRISynth now includes a unified CLI that seamlessly combines MRI processing with advanced visualization in a single command!**

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
git clone https://github.com/yourusername/MRISynth.git
cd MRISynth
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

#### Combined Processing
```python
from combined_pipeline import process_and_visualize

# Process and visualize in one call
results = process_and_visualize(
    input_nifti="brain.nii.gz",
    input_trk="streamlines.trk",
    output_prefix="processed",
    target_dimensions=(256, 256, 256),
    viz_output_dir="visualizations",
    n_examples=25,
    save_masks=True
)
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
MRISynth/
├── combined_pipeline.py            # 🔄 Unified CLI for complete workflow
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

# Performance
--use_gpu               Enable GPU acceleration (default: True)
--jobs N                Number of parallel jobs (default: 8)
--max_gb FLOAT          Memory limit in GB (default: 64)
```

## Core Features

### Dark Field Microscopy Visualization

MRISynth's signature feature creates stunning dark field microscopy-style images:

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

## Research Applications

### Machine Learning
- **Training Dataset Generation**: Create large, diverse datasets with ground truth masks
- **Data Augmentation**: Realistic medical imaging variations with Cornucopia
- **Quality Control**: Validate model outputs against fiber visualizations
- **Comparative Studies**: Evaluate different tractography algorithms

### Clinical Neuroimaging
- **Surgical Planning**: High-quality visualizations for neurosurgical guidance
- **Patient Education**: Clear, understandable brain connectivity visualizations
- **Research Publications**: Publication-ready figures with dark field aesthetics
- **Multi-Modal Integration**: Combine structural and diffusion MRI data

### High-Performance Computing
- **Large Cohort Studies**: Process thousands of subjects efficiently
- **Cloud Computing**: Scalable processing with GPU acceleration
- **Memory Optimization**: Handle large datasets within memory constraints
- **Batch Processing**: Automated processing pipelines

## Examples and Tutorials

### Example 1: Basic Processing and Visualization
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

### Example 2: High-Resolution Processing with ANTs
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

### Example 3: Machine Learning Dataset Generation
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

## Contributing

We welcome contributions to MRISynth! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/yourusername/MRISynth.git
cd MRISynth
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
- **Issues**: [GitHub Issues](https://github.com/yourusername/MRISynth/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/MRISynth/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*MRISynth is developed by the LINC Team to advance MRI processing, tractography synthesis, and visualization in neuroimaging research.*


