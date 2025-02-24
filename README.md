# Fiber Processing

## Overview
This repository provides tools for processing fiber tracking data from MRI scans, including streamline densification, affine transformations, NIfTI resampling, and streamline clipping.

## Features
- Densify streamlines for sub-voxel resolution.
- Transform and clip streamlines to fit a new field of view (FOV).
- Resample NIfTI images in parallel for efficiency.
- Process streamline data in a scalable and modular manner.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```

### Setup
Clone the repository and install the package:
```bash
git clone https://github.com/yourusername/fiber_processing.git
cd fiber_processing
pip install .
```


## File Structure
```
fiber_processing/
│── src/                   # Source code
│── tests/                 # Unit tests
│── docs/                  # Documentation files
│── requirements.txt       # Dependencies
│── setup.py               # Installation script
│── README.md              # Main repository documentation
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements.
