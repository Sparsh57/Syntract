# Syntract

MRI Tractography Synthesis and Visualization

This project generates synthetic fiber tractography data and creates visualizations from MRI scans. Part of the LINC project funded by NIH.

## Features

- Streamline interpolation (linear, Hermite, RBF)
- Synthetic fiber visualization with ground truth masks
- Dark field microscopy-style effects
- Multiple viewing orientations (axial, coronal, sagittal)
- GPU acceleration with CPU fallback
- ANTs transformation support
- Spatial volume subdivision

## Installation

```bash
git clone https://github.com/Sparsh57/Syntract.git
cd Syntract
pip install -r requirements.txt
pip install -e .
```

## Usage

### Generate fiber visualizations
```bash
python nifti_trk_vis/generate_fiber_examples.py \
    --nifti brain.nii.gz \
    --trk fibers.trk \
    --output_dir results \
    --examples 10
```

### Process with interpolation
```bash
python synthesis/main.py \
    --input brain.nii.gz \
    --trk fibers.trk \
    --interp hermite \
    --use_gpu
```

### ANTs transforms
```bash
python synthesis/main.py \
    --input brain.nii.gz \
    --trk fibers.trk \
    --use_ants \
    --ants_warp warp.nii.gz \
    --ants_iwarp iwarp.nii.gz \
    --ants_aff affine.mat
```

## Structure

- `synthesis/` - Core processing pipeline
- `nifti_trk_vis/` - Visualization tools
- `tests/` - Test suite

## Testing

```bash
pytest tests/
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## License

MIT

