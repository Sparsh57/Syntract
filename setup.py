#!/usr/bin/env python
"""
Setup script for Syntract: Advanced NIfTI Tractography Visualization and Processing.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from __init__.py
def get_version():
    """Get version from __init__.py file."""
    version_file = os.path.join("syntract_viewer", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="syntract",
    version=get_version(),
    author="Sparsh Makharia, LINC Team",
    author_email="sparsh.makharia@example.com",
    description="Advanced NIfTI Tractography Visualization with Dark Field Effects and Machine Learning Dataset Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/syntract",
    project_urls={
        "Bug Tracker": "https://github.com/your-repo/syntract/issues",
        "Documentation": "https://syntract.readthedocs.io/",
        "Source Code": "https://github.com/your-repo/syntract",
        "Changelog": "https://github.com/your-repo/syntract/blob/main/CHANGELOG.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: Console",
    ],
    keywords=[
        "tractography", "visualization", "dark-field", "neuroimaging", "nifti", 
        "machine-learning", "medical-imaging", "augmentation", "brain-mapping", 
        "white-matter", "streamlines", "fiber-tracking", "dipy", "dataset-generation",
        "mri", "resampling", "interpolation", "ants", "registration", 
        "gpu-acceleration", "parallel-processing", "cornucopia"
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "nibabel>=3.0.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.17.0",
        "scipy>=1.5.0",
        "dipy>=1.4.0",
        "joblib>=1.0.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "cornucopia": [
            "cornucopia-pytorch",
            "torch>=1.8.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
            "numba>=0.50.0",
        ],
        "ants": [
            # ANTs is typically installed system-wide
            # Users need to install ANTs separately
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "pre-commit>=2.0",
            "twine>=3.0",
            "memory-profiler>=0.58.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-gallery>=0.10",
            "matplotlib>=3.3.0",
            "numpydoc>=1.1",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-benchmark>=3.4.0",
            "memory-profiler>=0.58.0",
        ],
        "all": [
            "cornucopia-pytorch",
            "torch>=1.8.0",
            "cupy>=9.0.0",
            "numba>=0.50.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Primary visualization and dataset generation tools
            "generate-fiber-examples=syntract_viewer.generate_fiber_examples:main",
            "syntract-visualize=syntract_viewer.core:main",
            "syntract=syntract_viewer.generate_fiber_examples:main",
            # High-performance processing pipeline
            "mri-synthesis=synthesis.main:main",
            "syntract-process=synthesis.main:main",
            "compare-interpolation=synthesis.compare_interpolation:main",
        ],
    },
    package_data={
        "syntract_viewer": [
            "*.py",
            "data/*.json",
            "examples/*.py",
        ],
        "synthesis": [
            "*.py",
            "data/*.json",
            "examples/*.py",
            "tests/*.py",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    license_files=["LICENSE"],
)