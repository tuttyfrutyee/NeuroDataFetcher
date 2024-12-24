from setuptools import setup, find_packages

setup(
    name="neuro_data_fetcher",  # Replace with your desired package name
    version="0.1.0",
    author="Atakan(aka Anakin)",
    author_email="atakan@stanford.edu",
    description="A Python package for fetching and processing neuroscience data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tuttyfrutyee/NeuroDataFetcher.git",  # Replace with your repository URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # "h5py>=3.12.1",
        # "numpy>=2.2.1",
        # "pandas>=2.2.3",
        # "pynwb>=2.8.3",
        # "scipy>=1.14.1",
        # "setuptools>=59.6.0",
    ],
)
