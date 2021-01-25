from setuptools import find_packages, setup

setup(
    name="climatico",
    author="Maria J. Molina, NCAR",
    author_email="molina@ucar.edu",
    description="Package to explore changes to ENSO and global SSTs",
    url="https://github.com/mariajmolina/climatico",
    version="0.9",
    packages=find_packages(),
    package_dir={'climatico': 'climatico'},
    license="MIT License",
)
