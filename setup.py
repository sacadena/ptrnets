#!/usr/bin/env python
from os import path

from setuptools import find_packages
from setuptools import setup

here = path.abspath(path.dirname(__file__))

requirements = [
    "torch>=1.4.0,<2.0.0",
    "torchvision>=0.5.0,<0.15.0",
    "tqdm>=4.42.1",
    "gdown>=4.5.1",
    "tomli>=1.0.0",
]

setup(
    name="ptrnets",
    version="0.1.0",
    description="Easy access to pretrained models for system identification",
    author="Santiago Cadena",
    author_email="santiago.cadena@uni-tuebingen.de",
    packages=find_packages(exclude=[]),
    install_requires=requirements,
)
