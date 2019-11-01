#!/usr/bin/env python

from setuptools import setup
from setuptools.command.install import install
import os
import requests
import subprocess


setup(name='generativeopenset',
    version='0.1.0',
    description='PyTorch open set classification with generative models',
    author='Larry Neal',
    author_email='nealla@lwneal.com',
    packages=[
        'generativeopenset',
    ],
    #scripts=[
    #    'scripts/gnomehat',
    #],
    install_requires=[
        "torch",
        "numpy",
    ],
    python_requires='>3',
)
