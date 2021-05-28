#!/usr/bin/env python

# Stlen from python-pdal
# Stolen from Shapely's setup.py
# Onvironment variables influence this script.
#
# MDAL_LIBRARY_PATH: a path to a PDAL C++ shared library.
#
#
# NB: within this setup scripts, software versions are evaluated according
# to https://www.python.org/dev/peps/pep-0440/.

import logging
import os
import platform
import sys
import numpy
import glob
import sysconfig
import setuptools
from skbuild import setup
from packaging.version import Version

# Get the version from the pdal module
module_version = None
with open('mdal/__init__.py', 'r') as fp:
    for line in fp:
        if line.startswith("__version__"):
            module_version = Version(line.split("=")[1].strip().strip("\"'"))
            break

if not module_version:
    raise ValueError("Could not determine Python package version")

# Handle UTF-8 encoding of certain text files.
open_kwds = {}
if sys.version_info >= (3,):
    open_kwds['encoding'] = 'utf-8'

with open('README.rst', 'r', **open_kwds) as fp:
    readme = fp.read()

with open('CHANGES.txt', 'r', **open_kwds) as fp:
    changes = fp.read()

long_description = readme + '\n\n' +  changes

# https://github.com/ktbarrett/hello/blob/master/CMakeLists.txt

setup_args = dict(
    name                = 'mdal',
    version             = str(module_version),
    requires            = ['Python (>=3.0)', 'Numpy','meshio'],
    description         = 'Mesh data processing',
    license             = 'MIT',
    keywords            = 'mesh data spatial',
    author              = 'Paul Harwood',
    author_email        = 'runette@gmail.com',
    maintainer          = 'Paul Harwood',
    maintainer_email    = 'runette@gmail.com',
    url                 = 'https://mdal.xyz',
    long_description    = long_description,
    long_description_content_type = 'text/x-rst',
    test_suite          = 'test',
    packages            = [
        'mdal',
    ],
    include_package_data = False,
    classifiers         = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: GIS',
    ],

)
output = setup(**setup_args)

