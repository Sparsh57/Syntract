from setuptools import setup, find_packages

setup(
    name="Syntract",
    version="0.1",
    packages=find_packages(include=["synthesis*", "nifti_trk_vis*"]),
    install_requires=[
        "numpy",
        "nibabel",
        "joblib",
        "pytest",
        "flake8",
    ],
    extras_require={
        # Optional GPU dependencies
        "cuda": [
            "cupy-cuda11x"  # Adjust based on CUDA version
        ],
        "cpu": [
            "cupy"  # CPU-only fallback
        ]
    }
)