# Syntract 🧠

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NIH LINC](https://img.shields.io/badge/Funded%20by-NIH%20LINC-green.svg)](https://linc.cshl.edu/)

**Advanced MRI Tractography Synthesis and Visualization Pipeline**

Syntract is a comprehensive Python toolkit for generating synthetic fiber tractography data and creating high-quality visualizations from MRI scans. Originally developed as part of the NIH-funded LINC (Laboratory for Integrative Neuroscience and Computation) project, it provides state-of-the-art tools for neuroscientific research and medical imaging applications.

## ✨ Key Features

### 🔬 Advanced Processing
- **Multiple Interpolation Methods**: Linear, Hermite spline, and Radial Basis Function (RBF) interpolation
- **GPU Acceleration**: CUDA-optimized processing with automatic CPU fallback
- **ANTs Integration**: Support for Advanced Normalization Tools (ANTs) transformations
- **Spatial Subdivision**: Efficient volume partitioning for large datasets

### 🎨 Visualization & Analysis
- **Synthetic Fiber Generation**: Create realistic fiber tract visualizations with ground truth masks
- **Dark Field Microscopy Effects**: Authentic microscopy-style rendering
- **Multi-Planar Views**: Axial, coronal, and sagittal orientations
- **Interactive Slice Viewer**: Real-time exploration of tractography data

### 🚀 Performance & Scalability
- **Memory Management**: Configurable output size limits and efficient memory usage
- **Parallel Processing**: Multi-threaded operations for faster computation
- **Flexible I/O**: Support for NIfTI (.nii.gz) and TrackVis (.trk) formats

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA toolkit (optional, for GPU acceleration)

### Quick Install
```bash
git clone https://github.com/Sparsh57/Syntract.git
cd Syntract
pip install -r requirements.txt
pip install -e .
```

### GPU Support (Optional)
For CUDA acceleration:
```bash
pip install cupy-cuda11x  # Adjust for your CUDA version
```

### Dependencies
- **Core**: NumPy, SciPy, NiBabel, Joblib
- **Visualization**: Matplotlib
- **Optional**: CuPy (GPU acceleration), ANTs (spatial transforms)

## 🚀 Quick Start

### Basic Fiber Visualization
```bash
python nifti_trk_vis/generate_fiber_examples.py \
    --nifti data/brain.nii.gz \
    --trk data/fibers.trk \
    --output_dir results \
    --examples 10
```

### Advanced Processing with Interpolation
```bash
python synthesis/main.py \
    --input brain.nii.gz \
    --trk fibers.trk \
    --interpolation_method hermite \
    --target_voxel_size 0.5 \
    --use_gpu \
    --output_prefix processed
```

### ANTs-Based Spatial Transformation
```bash
python synthesis/main.py \
    --input brain.nii.gz \
    --trk fibers.trk \
    --use_ants \
    --ants_warp transforms/warp.nii.gz \
    --ants_iwarp transforms/iwarp.nii.gz \
    --ants_aff transforms/affine.mat \
    --target_dimensions 116 140 96
```

## 📊 Usage Examples

### 1. Interactive Slice Viewer
Launch the comprehensive tractography viewer:
```python
from nifti_trk_vis.nifti_trk_slice_viewer import main
main("brain.nii.gz", "fibers.trk")
```

### 2. Custom Interpolation Comparison
Compare different interpolation methods:
```python
from synthesis.compare_interpolation import compare_methods
compare_methods("brain.nii.gz", "fibers.trk", methods=['linear', 'hermite', 'rbf'])
```

## 📁 Project Structure

```
Syntract/
├── synthesis/                 # Core processing pipeline
│   ├── main.py               # Main processing script
│   ├── densify.py            # Streamline densification
│   ├── ants_transform.py     # ANTs integration
│   ├── streamline_processing.py
│   └── nifti_preprocessing.py
├── nifti_trk_vis/            # Visualization tools
│   ├── generate_fiber_examples.py
│   ├── nifti_trk_slice_viewer.py
│   └── smart_brain_masking.py
├── examples/                 # Usage examples
│   ├── spatial_subdivision_example.py
│   └── example_data/
└── tests/                    # Test suite
```

## ⚙️ Configuration Options

### Processing Parameters
| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--target_voxel_size` | Output voxel resolution | 0.5 | float |
| `--interpolation_method` | Streamline interpolation | 'hermite' | 'linear', 'hermite', 'rbf' |
| `--use_gpu` | Enable GPU acceleration | False | True/False |
| `--num_jobs` | Parallel processing threads | 8 | int |
| `--max_output_gb` | Memory limit (GB) | 64.0 | float |

### ANTs Transform Options
| Parameter | Description | Required for ANTs |
|-----------|-------------|-------------------|
| `--ants_warp` | Forward warp field | ✅ |
| `--ants_iwarp` | Inverse warp field | ✅ |
| `--ants_aff` | Affine transformation | ✅ |
| `--force_dimensions` | Override dimension checks | ❌ |

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=synthesis --cov=nifti_trk_vis

# Run specific test category
pytest tests/test_interpolation.py -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/Sparsh57/Syntract.git
cd Syntract
pip install -e ".[dev]"
pre-commit install
```

### Code Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Add comprehensive docstrings for new functions
- Include tests for new features
- Use type hints where appropriate



## 🐛 Known Issues & Troubleshooting

### Common Issues
1. **GPU Memory Errors**: Reduce `max_output_gb` or disable GPU with `--no-use_gpu`
2. **ANTs Transform Failures**: Ensure all transform files are compatible versions

### Performance Tips
- Use GPU acceleration for datasets with >10,000 streamlines
- Enable parallel processing with `--num_jobs` for CPU-bound operations



## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Sparsh57/Syntract/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sparsh57/Syntract/discussions)
---

