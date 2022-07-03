#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().split()

setup(
    name='ptrnets',
    version='0.0.0',
    description='Easy access to pretrained models for system identification',
    author='Santiago Cadena',
    author_email='santiago.cadena@uni-tuebingen.de',
    packages=find_packages(exclude=[]),
    install_requires=requirements,
    dependency_links=["git+https://github.com/dicarlolab/CORnet@master#egg=CORnet-0.1.0",
                      "git+https://github.com/sacadena/midlevel-reps.git@visualpriors#egg=visualpriors-0.3.5"]
)
