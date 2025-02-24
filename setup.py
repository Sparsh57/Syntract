from setuptools import setup, find_packages

setup(
    name="MRISynth",
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
)
