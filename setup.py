from setuptools import setup

setup(
    name = "S3ID",
    version = "0.0.1",

    description = "Python implementation for subspace identification with missing data.",

    # Project site
    url = "https://github.com/mackelab/S3ID",

    # Author details
    author = "Marcel Nonnenmacher",
    author_email = "marcel.nonnenmacher@caesar.de",

    # License info
    license = "BSD-2",

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved",
        "Programming Language :: Python :: 3 :: Only",
    ],

    keywords = ["missing-data", "subspace identification"],
    
    packages = ['S3ID']
)

