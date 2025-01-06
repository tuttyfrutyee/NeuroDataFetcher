from setuptools import setup, find_packages

setup(
    name="neurodatafetcher",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "neurodatafetcher": [
            "DataFetchers/**/*",
            "Utils/**/*",
        ]
    },
    install_requires=[
        "h5py>=3.12.1",
        "numpy>=2.2.1",
        "pandas>=2.2.3",
        "pynwb>=2.8.3",
        "scipy>=1.14.1",
        "setuptools>=59.6.0",
    ],
)
