from setuptools import setup, find_packages

setup(
    name="Syntract",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
            "cupy-cuda11x"  # you can change this based on target CUDA version
        ],
        "cpu": [
            "cupy"  # fallback version of cupy that doesn't require a GPU
        ]
    }
)
