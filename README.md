# Syntract: Advanced NIfTI Tractography Visualization and Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python package for advanced tractography visualization with dark field microscopy effects and machine learning dataset generation. Features high-performance NIfTI processing pipelines and sophisticated augmentation capabilities for neuroimaging research.

## Features

### Syntract Viewer - Core Visualization Engine
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

### Synthesis - High-Performance Processing Pipeline
- **GPU-accelerated processing** with CUDA/CuPy support and automatic CPU fallback
- **Advanced streamline interpolation** (Linear, Hermite, RBF) with sub-voxel precision
- **Intelligent NIfTI resampling** to arbitrary dimensions and voxel sizes
- **ANTs integration** for spatial transformations and registration workflows
- **Memory-efficient processing** with automatic chunking for large datasets
- **Parallel processing** with configurable job counts and resource optimization

## Installation

### Basic Installation

```bash
pip install syntract
```

### Development Installation

```bash
git clone https://github.com/your-repo/syntract.git
cd syntract
pip install -e .
```

### With Advanced Features

```bash
# With GPU acceleration
pip install syntract[gpu]

# With Cornucopia medical imaging augmentations
pip install syntract[cornucopia]

# With ANTs registration support
pip install syntract[ants]

# Complete installation
pip install syntract[all]
```

## Quick Start

### Dark Field Tractography Visualization

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

### Machine Learning Dataset Generation

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

### Advanced Medical Imaging Augmentations

```python
from syntract_viewer import generate_enhanced_varied_examples

# Generate augmented dataset with Cornucopia
generate_enhanced_varied_examples(
    nifti_file="brain.nii.gz",
    trk_file="tracts.trk",
    output_dir="./augmented_dataset",
    cornucopia_preset="clinical_simulation",
    n_examples=200,
    save_masks=True
)
```

### High-Performance Processing

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

### Command Line Usage

```bash
# Generate fiber visualization dataset
generate-fiber-examples \
    --nifti brain.nii.gz \
    --trk tracts.trk \
    --output_dir ./fiber_dataset \
    --examples 500 \
    --save_masks \
    --label_bundles \
    --cornucopia_preset clinical_simulation

# Spatial subdivision dataset
generate-fiber-examples \
    --nifti brain.nii.gz \
    --trk tracts.trk \
    --spatial_subdivisions \
    --n_subdivisions 27 \
    --output_dir ./spatial_dataset

# High-performance processing
python -m synthesis.main \
    --input brain.nii.gz \
    --trk streamlines.trk \
    --output processed \
    --voxel_size 0.5 \
    --use_gpu \
    --interp hermite
```

## Package Architecture

```
Syntract/
├── syntract_viewer/              # Main visualization and dataset generation
│   ├── core.py                  # Core visualization functions
│   ├── generation.py            # Dataset generation with variations
│   ├── contrast.py              # CLAHE and adaptive contrast methods
│   ├── masking.py               # Smart brain and fiber masking
│   ├── effects.py               # Dark field microscopy effects
│   ├── utils.py                 # Utilities and color mapping
│   ├── cornucopia_augmentation.py # Medical imaging augmentations
│   └── generate_fiber_examples.py # Command-line interface
└── synthesis/                   # High-performance processing pipeline
    ├── main.py                  # Main processing workflow
    ├── nifti_preprocessing.py   # NIfTI resampling with memory management
    ├── streamline_processing.py # Streamline transformation and clipping
    ├── densify.py               # Advanced interpolation algorithms
    ├── ants_transform.py        # ANTs integration
    └── compare_interpolation.py # Method comparison tools
```

## Core Syntract Features

### Dark Field Microscopy Visualization

Syntract's signature feature creates stunning dark field microscopy-style images:

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

### Spatial Subdivision Processing

Generate datasets from spatial grid subdivisions:

```python
# Command line spatial subdivision
generate-fiber-examples \
    --nifti brain.nii.gz \
    --trk tracts.trk \
    --spatial_subdivisions \
    --n_subdivisions 64 \
    --examples 25 \
    --min_streamlines_per_region 100 \
    --output_dir ./spatial_grid_dataset
```

### Multi-View Visualization

```python
from syntract_viewer import visualize_multiple_views

# Generate all three orientations
visualize_multiple_views(
    nifti_file="brain.nii.gz",
    trk_file="tracts.trk",
    output_file="multi_view.png",
    save_masks=True,
    streamline_percentage=75.0
)
```

## Cornucopia Medical Imaging Augmentations

When available, Syntract integrates sophisticated medical imaging augmentations:

```python
# Available augmentation presets
presets = {
    'clinical_simulation': {
        'spatial': {'type': 'affine_medical'},
        'intensity': {'type': 'bias_field'},
        'noise': {'type': 'rician_noise', 'intensity': 0.2},
        'contrast': {'type': 'adaptive_contrast'}
    },
    'aggressive': {
        'spatial': {'type': 'elastic_deformation'},
        'intensity': {'type': 'bias_field'},
        'noise': {'type': 'rician_noise', 'intensity': 0.6},
        'contrast': {'type': 'local_contrast'}
    }
}
```

## Processing Pipeline Features

### GPU-Accelerated Resampling

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

### ANTs Registration Integration

```python
# Process with ANTs transformations
process_and_save(
    original_nifti_path="subject.nii.gz",
    original_trk_path="subject_tracts.trk",
    use_ants=True,
    ants_warp_path="subject_to_mni_warp.nii.gz",
    ants_iwarp_path="mni_to_subject_warp.nii.gz",
    ants_aff_path="subject_to_mni_affine.mat",
    target_dimensions=(182, 218, 182),
    transform_mri_with_ants=True
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


## Contributing

We welcome contributions to Syntract! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/Sparsh57/syntract.git
cd syntract
pip install -e ".[dev]"
pre-commit install
```

### Testing

```bash
pytest tests/
python -m synthesis.compare_interpolation
```



## Related Projects

- [DIPY](https://dipy.org/) - Diffusion imaging in Python
- [NiBabel](https://nipy.org/nibabel/) - Neuroimaging file format access
- [Cornucopia](https://cornucopia.readthedocs.io/) - Medical imaging augmentations
- [ANTs](http://stnava.github.io/ANTs/) - Advanced Normalization Tools
- [CuPy](https://cupy.dev/) - GPU-accelerated computing

## Support
- **Issues**: [GitHub Issues](https://github.com/your-repo/syntract/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/syntract/discussions)

---

*Syntract is developed by the LINC Team to advance tractography visualization and machine learning in neuroimaging.*


