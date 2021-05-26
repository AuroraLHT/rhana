import os 

from setuptools import setup
from setuptools import find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="rhana",
    version="0.0.1",
    description="rhana: RHeed ANAlysis",
    long_description=readme,
    author="H Liang, V Stanev, A. G kusne, and I Takeuchi",
    author_email="auroralht@gmail.com, vstanev@umd.edu, aaron.kusne@nist.gov, takeuchi@umd.edu",
    packages=find_packages(),
    install_requires=[
        requirements,
    ],
    include_package_data=True,

    classifiers=['Programming Language :: Python :: 3.8',
                  'Development Status :: 4 - Beta',
                  'Intended Audience :: Science/Research',
                  'Intended Audience :: System Administrators',
                  'Intended Audience :: Information Technology',
                  'Operating System :: OS Independent',
                  'Topic :: Other/Nonlisted Topic',
                  'Topic :: Scientific/Engineering']
)