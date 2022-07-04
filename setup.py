#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

requirements = ["torch>=1.4.0",
		"torchvision>=0.5.0",
		"tqdm>=4.42.1",
		"gdown>=4.5.1",
		"visualpriors==0.3.5"]

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
