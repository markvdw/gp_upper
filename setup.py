#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from setuptools import setup

setup(name='gp_upper',
      version="0.1",
      author="Mark van der Wilk",
      author_email="mv310@cam.ac.uk",
      description="Gaussian process regression upper bounds",
      license="Apache License 2.0",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/markvdw/gp_upper",
      ext_modules=[],
      packages=["gp_upper"],
      test_suite='testing',
      install_requires=['gpflow>=0.4.0'])
