# Version: 0.29

"""The Versioneer - like a rocketeer, but for versions.

The Versioneer
==============

* like a rocketeer, but for versions!
* https://github.com/python-versioneer/python-versioneer
* Brian Warner
* License: Public Domain (Unlicense)
* Compatible with: Python 3.7, 3.8, 3.9, 3.10, 3.11 and pypy3
* [![Latest Version][pypi-image]][pypi-url]
* [![Build Status][travis-image]][travis-url]

This is a tool for managing a recorded version number in setuptools-based
python projects. The goal is to remove the tedious and error-prone "update
the embedded version string" step from your release process. Making a new
release should be as easy as recording a new tag in your version-control
system, and maybe making new tarballs.


## Quick Install

Versioneer provides two installation modes. The "classic" vendored mode installs
a copy of versioneer into your repository. The experimental build-time dependency mode
is intended to allow you to skip this step and simplify the process of upgrading.

### Vendored mode

* `pip install versioneer` to somewhere in your $PATH
   * A [conda-forge recipe](https://github.com/conda-forge/versioneer-feedstock) is
     available, so you can also use `conda install -c conda-forge versioneer`
* add a `[tool.versioneer]` section to your `pyproject.toml` or a
  `[versioneer]` section to your `setup.cfg` (see [Install](INSTALL.md))
   * Note that you will need to add `tomli; python_version < "3.11"` to your
     build-time dependencies if you use `pyproject.toml`
* run `versioneer install --vendor` in your source tree, commit the results
* verify version information with `python setup.py version`

### Build-time dependency mode

* `pip install versioneer` to somewhere in your $PATH
   * A [conda-forge recipe](https://github.com/conda-forge/versioneer-feedstock) is
     available, so you can also use `conda install -c conda-forge versioneer`
* add a `[tool.versioneer]` section to your `pyproject.toml` or a
  `[versioneer]` section to your `setup.cfg` (see [Install](INSTALL.md))
* add `versioneer` (with `[toml]` extra, if configuring in `pyproject.toml`)
  to the `requires` key of the `build-system` table in `pyproject.toml`:
  