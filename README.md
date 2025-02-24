# Fiber Processing

## Overview
This repository is part of the **LINC project**, funded by the **NIH**, and focuses on the synthetic generation of tracer data for fiber tracking in MRI scans. The goal is to develop tools that can process, transform, and resample fiber tractography data efficiently for research applications.

## Features
- Densify streamlines for sub-voxel resolution.
- Transform and clip streamlines to fit a new field of view (FOV).
- Resample NIfTI images in parallel for efficiency.
- Process streamline data in a scalable and modular manner.
- Supports affine transformations for coordinate adjustments.
- Implements parallel processing for high-performance data handling.
- Modularized code structure for easy extension and customization.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```

### Setup
Clone the repository and install the package:
```bash
git clone https://github.com/Sparsh57/MRISynth.git
cd MRISynth
pip install -e .
```

## Usage
### Running the Pipeline
To process fiber tracking data, run the following command:
```bash
python src/main.py <input_nifti> <input_trk> --voxel_size 0.5 --new_dim 116 140 96 --output_prefix output/resampled
```

### Example Dataset
Example NIfTI and TRK files are provided in `examples/example_data/`.
To test the pipeline:
```bash
pytest tests/
```

## File Structure
```
fiber_processing/
│── src/                   # Source code
│   ├── densify.py         # Densification functions
│   ├── transform.py       # Affine transformations
│   ├── streamline_processing.py  # Processing streamlines
│   ├── nifti_processing.py # Resampling NIfTI files
│   ├── main.py            # Entry point
│── tests/                 # Unit tests
│── examples/              # Example datasets and scripts
│── requirements.txt       # Dependencies
│── setup.py               # Installation script
│── README.md              # Main repository documentation
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements.

## Project Link
For more information about the **LINC project**, visit the official **LINC GitHub Organization**:
[GitHub - LINC BRAIN](https://github.com/lincbrain)

