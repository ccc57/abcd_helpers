#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup file for abcd_helpers."""

import os

from setuptools import find_packages, setup

NAME = "abcd_helpers"
DESCRIPTION = "A collection of functions for ABCD data analysis"
URL = "https://github.com/nih-fmrif/abcd_helpers"
EMAIL = "johnleenimh@gmail.com"
AUTHOR = "abcd_helpers developers"

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with open(os.path.join(here, "README.md"), encoding="utf-8") as fp:
    long_description = fp.read()

with open(os.path.join(here, "requirements.txt")) as fp:
    REQUIRED = fp.readlines()

about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
)
