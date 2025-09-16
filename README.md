# SynTract: MRI Processing & Tractography Visualization

A Python package for advanced MRI processing, tractography synthesis, and dark field microscopy-style visualization. Features high-performance pipelines, augmentation, and a unified CLI for neuroimaging research.

## 🚀 Key Features

### 🔄 Combined Pipeline
- Unified processing and visualization in a single command
- Automatic file handling between synthesis and visualization
- Complete parameter mapping from both synthesis and syntract_viewer
- End-to-end workflow from raw data to publication-ready visualizations
- ANTs integration support for spatial transformations
- Flexible output control with intermediate file retention options

### 🧠 Synthesis
- GPU-accelerated processing with CUDA/CuPy support and CPU fallback
- Advanced streamline interpolation (Linear, Hermite, RBF) with sub-voxel precision
- Intelligent NIfTI resampling to arbitrary dimensions and voxel sizes
- ANTs integration for spatial transformations and registration workflows
- Memory-efficient processing with automatic chunking for large datasets
- Parallel processing with configurable job counts

### 🎨 Visualization
- Dark field microscopy-style visualization with publication-ready image generation
- Multi-orientation support (axial, coronal, sagittal) with intelligent slice selection
- Ground truth mask generation for machine learning with bundle labeling
- Advanced contrast enhancement using CLAHE and adaptive methods
- Smart brain masking preserving anatomical structures and ventricles
- Spatial ROI selection with fiber density variation for diverse datasets
- Cornucopia integration for realistic medical imaging augmentations
- Batch processing tools for dataset generation
- Smart fiber bundle detection with morphological operations and gap closing
- Customizable color schemes including blue-tinted dark field effects

### 🧩 Patch Extraction (New!)
- Extract 3D patches via re-synthesis at random or specified centers
- Each patch is a full synthesis run centered on a new coordinate
- Output organized in subfolders for each patch

### 🗂️ Batch Processing
- Process multiple TRK files with a shared NIfTI using cumulative.py
- Predefined configurations for different data types and quality levels

### ⚡ Performance & Enhancement
- GPU acceleration (optional, with CuPy)
- Parallel jobs and memory optimization
- Optional Cornucopia augmentations

### 🏷️ Masking & Bundles
- Save fiber masks and label distinct bundles
- High-density masking and bundle size/density thresholds
- Morphological gap closing and smart bundle detection

### 🧩 Spatial Subdivisions
- Enable spatial subdivisions for region-based analysis
- Control number of subdivisions, streamline limits, and region skipping

### 🛠️ Flexible API & CLI
- Python API for programmatic control and integration
- Command-line interface for quick processing and scripting

### 📦 Output
- Publication-ready images, masks, and batch outputs
- Organized output directories for easy dataset management

## Installation
```bash
git clone https://github.com/yourusername/SynTract.git
cd SynTract
pip install -e .
# Optional: pip install cupy-cuda11x cornucopia-pytorch
```

## Quick Start

### Combined Pipeline (CLI)
```bash
python syntract.py \
  --input brain.nii.gz \
  --trk fibers.trk \
  --output processed_data \
  --viz_output_dir visualization_dataset \
  --n_examples 10 \
  --save_masks
```

### Patch Extraction
```bash
python syntract.py \
  --input brain.nii.gz \
  --trk fibers.trk \
  --output patch_data \
  --patch_mode random \
  --n_patches 5 \
  --patch_size 64 64 64 \
  --viz_output_dir patch_visualizations
```

### Python API
```python
from syntract import process_syntract

result = process_syntract(
    input_nifti="brain.nii.gz",
    input_trk="fibers.trk",
    output_base="processed_data",
    viz_output_dir="visualization_dataset",
    n_examples=10,
    save_masks=True
)
```

### Patch Extraction via API
```python
result = process_syntract(
    input_nifti="brain.nii.gz",
    input_trk="fibers.trk",
    patch_mode="random",
    n_patches=5,
    patch_size=[64,64,64],
    viz_output_dir="patch_visualizations"
)
```

### Batch Processing with cumulative.py
You can process multiple TRK files in a directory using the batch API in `cumulative.py`:
```python
from cumulative import batch_process_trk_files

results = batch_process_trk_files(
  nifti_path="shared_brain.nii.gz",
  trk_dir="trk_files_directory",
  new_dim=[256, 256, 256],
  voxel_size=[0.5],
  n_examples=10,
  save_masks=True
)
print(f"Processed {len(results['successful'])} files successfully")
print(f"Failed: {len(results['failed'])} files")
```
## Architecture
```
SynTract/
├── syntract.py           # CLI & main API
├── cumulative.py         # Batch processing
├── synthesis/            # Processing pipeline
└── syntract_viewer/      # Visualization tools
```

## Dependencies
- numpy, nibabel, matplotlib, scikit-image, scipy, dipy, joblib, tqdm
- Optional: cupy, cornucopia-pytorch, ants

## License
MIT License. See LICENSE file.

---
*Developed by the LINC Team for neuroimaging research.*
