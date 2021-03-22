from setuptools import setup, find_packages
from pathlib import Path

install_requires = [
    "allennlp>=0.9.0",
"torch>=1.2.0",
"matplotlib",
"plotly",
"pytest",
"mypy==0.701",
"mypy-extensions==0.4.1"
    ]
setup(
    name='boxes',
    version='0.0.1',
    description='PyTorch Boxes',
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={'boxes': ['py.typed']},
    install_requires=install_requires,
    zip_safe=False)
