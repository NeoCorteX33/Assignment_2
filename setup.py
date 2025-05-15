# setup.py
from setuptools import setup, find_packages

setup(
    name='assignment_2',
    version='0.1.0',
    packages=find_packages(where='src'), # Find packages in the 'src' directory
    package_dir={'': 'src'},
    install_requires=[
    'setuptools~=80.1.0',
    'ipykernel~=6.29',
    'numpy>=2.0.2',
    'pandas>=2.0',
    'scikit-learn>=1.6',
    'scipy>=1.11',
    'matplotlib>=3.9.4',
    'seaborn>=0.12',
    'lightgbm>=4.0',
    'atom-ml>=5.2.0'
    ],
    python_requires='>=3.11',
)