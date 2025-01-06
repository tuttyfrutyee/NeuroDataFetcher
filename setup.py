from setuptools import setup, find_packages

setup(
    name="neurodatafetcher",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "h5py>=3.12.1",
        "numpy>=2.2.1",
        "pandas>=2.2.3",
        "pynwb>=2.8.3",
        "scipy>=1.14.1",
        "setuptools>=59.6.0",
    ],
)
