import setuptools
from setuptools import setup
import os
import sys

long_description = "codes to generate mutated pathway based immunotherapy predictors"


requirements = [
    'numpy>=1.17.3',
    'tensorflow>=2.0.0',
    'pandas>=1.3.5',
    'scikit-learn>=1.0',
    'lifelines>=0.27.0',
    'matplotlib>=3.5.1',
    'scipy>=1.7.3',
]


setup(
    name="mutated-pathways-ML",
    version="1",
    author="Add list",
    description="codes to generate mutated pathway based immunotherapy predictors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    test_suite='nose.collector',
    tests_require=['nose'],

)
