# Syntract

## Overview
This repository is part of the **LINC project**, funded by the **NIH**, and focuses on the synthetic generation of tracer data for fiber tracking in MRI scans. The goal is to develop tools that can process, transform, and resample fiber tractography data efficiently for research applications.

## Features
- Densify streamlines for sub-voxel resolution with multiple interpolation methods:
  - Linear interpolation (fast, basic)
  - Hermite interpolation (smooth, accurate)
  - RBF interpolation (very smooth, accurate, slow)
- Transform and clip streamlines to fit a new field of view (FOV)
- Resample NIfTI images in parallel for efficiency
- Process streamline data in a scalable and modular manner
- Supports affine transformations for coordinate adjustments
- Implements parallel processing for high-performance data handling
- GPU acceleration support for faster processing
- Comprehensive interpolation comparison tools
- ANTs transformation support for advanced image and streamline registration
- Modularized code structure for easy extension and customization

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```

### Setup
Clone the repository and install the package:
```bash
git clone https://github.com/Sparsh57/Syntract.git
cd Syntract
pip install -e .
```

## Usage
### Running the Pipeline
To process fiber tracking data, run the following command:
```bash
python src/main.py --input <input_nifti> --trk <input_trk> [options]
```

#### Required Parameters
- `--input`: Path to input NIfTI (.nii or .nii.gz) file
- `--trk`: Path to input TRK (.trk) file

#### Optional Parameters
- `--output`: Prefix for output files (default: "resampled")
- `--voxel_size`: New voxel size in mm, either a single isotropic value or three anisotropic values (default: 0.5)
- `--new_dim`: New image dimensions (x, y, z) (default: 116 140 96)
- `--jobs`: Number of parallel jobs (-1 for all CPUs) (default: 8)
- `--patch_center`: Optional patch center in mm (x y z) (default: None)
- `--reduction`: Optional reduction along z-axis (choices: "mip", "mean") (default: None)
- `--use_gpu`: Use GPU acceleration (default: True)
- `--cpu`: Force CPU processing (disables GPU) (default: False)
- `--interp`: Interpolation method for streamlines (choices: "hermite", "linear", "rbf") (default: "hermite")
- `--step_size`: Step size for streamline densification (default: 0.5)
- `--max_gb`: Maximum output size in GB (default: 64.0)

#### ANTs Transform Parameters
- `--use_ants`: Use ANTs transforms for processing (default: False)
- `--ants_warp`: Path to ANTs warp file (required if use_ants is True)
- `--ants_iwarp`: Path to ANTs inverse warp file (required if use_ants is True)
- `--ants_aff`: Path to ANTs affine file (required if use_ants is True)

#### Example Usage
```bash
# Basic usage with defaults
python src/main.py --input input.nii.gz --trk input.trk

# Using ANTs transforms
python src/main.py \
    --input input.nii.gz \
    --trk input.trk \
    --output ants_transformed \
    --use_ants \
    --ants_warp path/to/warp.nii.gz \
    --ants_iwarp path/to/inverse_warp.nii.gz \
    --ants_aff path/to/affine.mat

# Full example with all parameters including ANTs transforms
python src/main.py \
    --input input.nii.gz \
    --trk input.trk \
    --output processed \
    --voxel_size 0.5 \
    --new_dim 116 140 96 \
    --jobs 8 \
    --patch_center 0 0 0 \
    --reduction mip \
    --use_gpu \
    --interp hermite \
    --step_size 0.5 \
    --max_gb 64.0 \
    --use_ants \
    --ants_warp path/to/warp.nii.gz \
    --ants_iwarp path/to/inverse_warp.nii.gz \
    --ants_aff path/to/affine.mat
```

### Comparing Interpolation Methods
To compare different interpolation methods and visualize the results:
```bash
python src/compare_interpolation.py <input_trk> --step_size 0.5 --voxel_size 1.0 --use_gpu
```

This will:
- Process streamlines using both linear and Hermite interpolation
- Generate visualizations comparing the methods
- Calculate and display metrics for both approaches
- Show curvature and smoothness comparisons

### Example Dataset
Example NIfTI and TRK files are provided in `examples/example_data/`.
To test the pipeline:
```bash
pytest tests/
```

## File Structure
```
Syntract/
│── src/                   # Source code
│   ├── densify.py         # Densification functions
│   ├── transform.py       # Affine transformations
│   ├── streamline_processing.py  # Processing streamlines
│   ├── nifti_preprocessing.py # Resampling NIfTI files
│   ├── ants_transform.py  # ANTs transformation utilities
│   ├── compare_interpolation.py  # Interpolation comparison
│   ├── visualize.py       # Visualization utilities
│   ├── main.py            # Entry point
│── tests/                 # Unit tests
│── examples/              # Example datasets and scripts
│── requirements.txt       # Dependencies
│── setup.py               # Installation script
│── README.md              # Main repository documentation
```

## Performance Considerations
- GPU acceleration is enabled by default for faster processing
- For systems without GPU, the pipeline automatically falls back to CPU processing
- Parallel processing is available for both CPU and GPU modes
- Memory usage can be controlled with the `--max_gb` parameter

## ANTs Transformation Details
The ANTs (Advanced Normalization Tools) integration allows for:
- Advanced non-linear registration between different image spaces
- Transforming both MRI volumes and streamlines data using the same registration
- Proper handling of orientation flips and coordinate system differences
- Direct integration with the rest of the Syntract pipeline
- Apply transformation without additional resampling by using the `--use_ants` flag

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements.

## Project Link
For more information about the **LINC project**, visit the official **LINC GitHub Organization**:
[GitHub - LINC BRAIN](https://github.com/lincbrain)

