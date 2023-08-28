from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='FedBEVT',
    version='0.1.0',
    packages=find_packages(),
    license='',
    author='Rui Song',
    author_email='Rui Song',
    description='FedBEVT',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
