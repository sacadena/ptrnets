#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='ptrnets',
    version='0.0.0',
    description='Easy access to pretrained models for system identification',
    author='Santiago Cadena',
    author_email='santiago.cadena@uni-tuebingen.de',
    packages=find_packages(exclude=[]),
    install_requires=requirements,
)
