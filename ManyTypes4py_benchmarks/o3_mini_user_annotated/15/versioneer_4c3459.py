#!/usr/bin/env python3
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
  ```toml
  [build-system]
  requires = ["setuptools", "versioneer[toml]"]
  build-backend = "setuptools.build_meta"
  ```
* run `versioneer install --no-vendor` in your source tree, commit the results
* verify version information with `python setup.py version`

## Version Identifiers

Source trees come from a variety of places:

* a version-control system checkout (mostly used by developers)
* a nightly tarball, produced by build automation
* a snapshot tarball, produced by a web-based VCS browser, like github's
  "tarball from tag" feature
* a release tarball, produced by "setup.py sdist", distributed through PyPI

Within each source tree, the version identifier (either a string or a number,
this tool is format-agnostic) can come from a variety of places:

* ask the VCS tool itself, e.g. "git describe" (for checkouts), which knows
  about recent "tags" and an absolute revision-id
* the name of the directory into which the tarball was unpacked
* an expanded VCS keyword ($Id$, etc)
* a `_version.py` created by some earlier build step

For released software, the version identifier is closely related to a VCS
tag. Some projects use tag names that include more than just the version
string (e.g. "myproject-1.2" instead of just "1.2"), in which case the tool
needs to strip the tag prefix to extract the version identifier. For
unreleased software (between tags), the version identifier should provide
enough information to help developers recreate the same tree, while also
giving them an idea of roughly how old the tree is (after version 1.2, before
version 1.3). Many VCS systems can report a description that captures this,
for example `git describe --tags --dirty --always` reports things like
"0.7-1-g574ab98-dirty" to indicate that the checkout is one revision past the
0.7 tag, has a unique revision id of "574ab98", and is "dirty" (it has
uncommitted changes).

The version identifier is used for multiple purposes:

* to allow the module to self-identify its version: `myproject.__version__`
* to choose a name and prefix for a 'setup.py sdist' tarball

## Theory of Operation

Versioneer works by adding a special `_version.py` file into your source
tree, where your `__init__.py` can import it. This `_version.py` knows how to
dynamically ask the VCS tool for version information at import time.

`_version.py` also contains `$Revision$` markers, and the installation
process marks `_version.py` to have this marker rewritten with a tag name
during the `git archive` command. As a result, generated tarballs will
contain enough information to get the proper version.

To allow `setup.py` to compute a version too, a `versioneer.py` is added to
the top level of your source tree, next to `setup.py` and the `setup.cfg`
that configures it. This overrides several distutils/setuptools commands to
compute the version when invoked, and changes `setup.py build` and `setup.py
sdist` to replace `_version.py` with a small static file that contains just
the generated version data.

## Installation

See [INSTALL.md](./INSTALL.md) for detailed installation instructions.

## Version-String Flavors

Code which uses Versioneer can learn about its version string at runtime by
importing `_version` from your main `__init__.py` file and running the
`get_versions()` function. From the "outside" (e.g. in `setup.py`), you can
import the top-level `versioneer.py` and run `get_versions()`.

Both functions return a dictionary with different flavors of version
information:

* `['version']`: A condensed version string, rendered using the selected
  style. This is the most commonly used value for the project's version
  string. The default "pep440" style yields strings like `0.11`,
  `0.11+2.g1076c97`, or `0.11+2.g1076c97.dirty`. See the "Styles" section
  below for alternative styles.

* `['full-revisionid']`: detailed revision identifier. For Git, this is the
  full SHA1 commit id, e.g. "1076c978a8d3cfc70f408fe5974aa6c092c949ac".

* `['date']`: Date and time of the latest `HEAD` commit. For Git, it is the
  commit date in ISO 8601 format. This will be None if the date is not
  available.

* `['dirty']`: a boolean, True if the tree has uncommitted changes. Note that
  this is only accurate if run in a VCS checkout, otherwise it is likely to
  be False or None

* `['error']`: if the version string could not be computed, this will be set
  to a string describing the problem, otherwise it will be None. It may be
  useful to throw an exception in setup.py if this is set, to avoid e.g.
  creating tarballs with a version string of "unknown".

Some variants are more useful than others. Including `full-revisionid` in a
bug report should allow developers to reconstruct the exact code being tested
(or indicate the presence of local changes that should be shared with the
developers). `version` is suitable for display in an "about" box or a CLI
`--version` output: it can be easily compared against release notes and lists
of bugs fixed in various releases.

The installer adds the following text to your `__init__.py` to place a basic
version in `YOURPROJECT.__version__`:

    from ._version import get_versions
    __version__ = get_versions()['version']
    del get_versions

## Styles

The setup.cfg `style=` configuration controls how the VCS information is
rendered into a version string.

The default style, "pep440", produces a PEP440-compliant string, equal to the
un-prefixed tag name for actual releases, and containing an additional "local
version" section with more detail for in-between builds. For Git, this is
TAG[+DISTANCE.gHEX[.dirty]] , using information from `git describe --tags
--dirty --always`. For example "0.11+2.g1076c97.dirty" indicates that the
tree is like the "1076c97" commit but has uncommitted changes (".dirty"), and
that this commit is two revisions ("+2") beyond the "0.11" tag. For released
software (exactly equal to a known tag), the identifier will only contain the
stripped tag, e.g. "0.11".

Other styles are available. See [details.md](details.md) in the Versioneer
source tree for descriptions.

## Debugging

Versioneer tries to avoid fatal errors: if something goes wrong, it will tend
to return a version of "0+unknown". To investigate the problem, run `setup.py
version`, which will run the version-lookup code in a verbose mode, and will
display the full contents of `get_versions()` (including the `error` string,
which may help identify what went wrong).

## Known Limitations

Some situations are known to cause problems for Versioneer. This details the
most significant ones. More can be found on Github
[issues page](https://github.com/python-versioneer/python-versioneer/issues).

### Subprojects

Versioneer has limited support for source trees in which `setup.py` is not in
the root directory (e.g. `setup.py` and `.git/` are *not* siblings). The are
two common reasons why `setup.py` might not be in the root:

* Source trees which contain multiple subprojects, such as
  [Buildbot](https://github.com/buildbot/buildbot), which contains both
  "master" and "slave" subprojects, each with their own `setup.py`,
  `setup.cfg`, and `tox.ini`. Projects like these produce multiple PyPI
  distributions (and upload multiple independently-installable tarballs).
* Source trees whose main purpose is to contain a C library, but which also
  provide bindings to Python (and perhaps other languages) in subdirectories.

Versioneer will look for `.git` in parent directories, and most operations
should get the right version string. However `pip` and `setuptools` have bugs
and implementation details which frequently cause `pip install .` from a
subproject directory to fail to find a correct version string (so it usually
defaults to `0+unknown`).

`pip install --editable .` should work correctly. `setup.py install` might
work too.

Pip-8.1.1 is known to have this problem, but hopefully it will get fixed in
some later version.

[Bug #38](https://github.com/python-versioneer/python-versioneer/issues/38) is tracking
this issue. The discussion in
[PR #61](https://github.com/python-versioneer/python-versioneer/pull/61) describes the
issue from the Versioneer side in more detail.
[pip PR#3176](https://github.com/pypa/pip/pull/3176) and
[pip PR#3615](https://github.com/pypa/pip/pull/3615) contain work to improve
pip to let Versioneer work correctly.

Versioneer-0.16 and earlier only looked for a `.git` directory next to the
`setup.cfg`, so subprojects were completely unsupported with those releases.

### Editable installs with setuptools <= 18.5

`setup.py develop` and `pip install --editable .` allow you to install a
project into a virtualenv once, then continue editing the source code (and
test) without re-installing after every change.

"Entry-point scripts" (`setup(entry_points={"console_scripts": ..})`) are a
convenient way to specify executable scripts that should be installed along
with the python package.

These both work as expected when using modern setuptools. When using
setuptools-18.5 or earlier, however, certain operations will cause
`pkg_resources.DistributionNotFound` errors when running the entrypoint
script, which must be resolved by re-installing the package. This happens
when the install happens with one version, then the egg_info data is
regenerated while a different version is checked out. Many setup.py commands
cause egg_info to be rebuilt (including `sdist`, `wheel`, and installing into
a different virtualenv), so this can be surprising.

[Bug #83](https://github.com/python-versioneer/python-versioneer/issues/83) describes
this one, but upgrading to a newer version of setuptools should probably
resolve it.


## Updating Versioneer

To upgrade your project to a new release of Versioneer, do the following:

* install the new Versioneer (`pip install -U versioneer` or equivalent)
* edit `setup.cfg` and `pyproject.toml`, if necessary,
  to include any new configuration settings indicated by the release notes.
  See [UPGRADING](./UPGRADING.md) for details.
* re-run `versioneer install --[no-]vendor` in your source tree, to replace
  `SRC/_version.py`
* commit any changed files

## Future Directions

This tool is designed to make it easily extended to other version-control
systems: all VCS-specific components are in separate directories like
src/git/ . The top-level `versioneer.py` script is assembled from these
components by running make-versioneer.py . In the future, make-versioneer.py
will take a VCS name as an argument, and will construct a version of
`versioneer.py` that is specific to the given VCS. It might also take the
configuration arguments that are currently provided manually during
installation by editing setup.py . Alternatively, it might go the other
direction and include code from all supported VCS systems, reducing the
number of intermediate scripts.

## Similar projects

* [setuptools_scm](https://github.com/pypa/setuptools_scm/) - a non-vendored build-time
  dependency
* [minver](https://github.com/jbweston/miniver) - a lightweight reimplementation of
  versioneer
* [versioningit](https://github.com/jwodder/versioningit) - a PEP 518-based setuptools
  plugin

## License

To make Versioneer easier to embed, all its code is dedicated to the public
domain. The `_version.py` that it creates is also in the public domain.
Specifically, both are released under the "Unlicense", as described in
https://unlicense.org/.

[pypi-image]: https://img.shields.io/pypi/v/versioneer.svg
[pypi-url]: https://pypi.python.org/pypi/versioneer/
[travis-image]:
https://img.shields.io/travis/com/python-versioneer/python-versioneer.svg
[travis-url]: https://travis-ci.com/github/python-versioneer/python-versioneer

"""
# pylint:disable=invalid-name,import-outside-toplevel,missing-function-docstring
# pylint:disable=missing-class-docstring,too-many-branches,too-many-statements
# pylint:disable=raise-missing-from,too-many-lines,too-many-locals,import-error
# pylint:disable=too-few-public-methods,redefined-outer-name,consider-using-with
# pylint:disable=attribute-defined-outside-init,too-many-arguments

import configparser
import errno
import functools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union, cast

have_tomllib: bool = True
if sys.version_info >= (3, 11):
    import tomllib  # type: ignore
else:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        have_tomllib = False


class VersioneerConfig:
    """Container for Versioneer configuration parameters."""
    VCS: str
    style: str
    tag_prefix: str
    versionfile_source: str
    versionfile_build: Optional[str]
    parentdir_prefix: Optional[str]
    verbose: Optional[bool]


def get_root() -> str:
    """Get the project root directory.

    We require that all commands are run from the project root, i.e. the
    directory that contains setup.py, setup.cfg, and versioneer.py .
    """
    root: str = os.path.realpath(os.path.abspath(os.getcwd()))
    setup_py: str = os.path.join(root, "setup.py")
    pyproject_toml: str = os.path.join(root, "pyproject.toml")
    versioneer_py: str = os.path.join(root, "versioneer.py")
    if not (
        os.path.exists(setup_py)
        or os.path.exists(pyproject_toml)
        or os.path.exists(versioneer_py)
    ):
        # allow 'python path/to/setup.py COMMAND'
        root = os.path.dirname(os.path.realpath(os.path.abspath(sys.argv[0])))
        setup_py = os.path.join(root, "setup.py")
        pyproject_toml = os.path.join(root, "pyproject.toml")
        versioneer_py = os.path.join(root, "versioneer.py")
    if not (
        os.path.exists(setup_py)
        or os.path.exists(pyproject_toml)
        or os.path.exists(versioneer_py)
    ):
        err: str = (
            "Versioneer was unable to run the project root directory. "
            "Versioneer requires setup.py to be executed from "
            "its immediate directory (like 'python setup.py COMMAND'), "
            "or in a way that lets it use sys.argv[0] to find the root "
            "(like 'python path/to/setup.py COMMAND')."
        )
        raise VersioneerBadRootError(err)
    try:
        # Certain runtime workflows (setup.py install/develop in a setuptools
        # tree) execute all dependencies in a single python process, so
        # "versioneer" may be imported multiple times, and python's shared
        # module-import table will cache the first one. So we can't use
        # os.path.dirname(__file__), as that will find whichever
        # versioneer.py was first imported, even in later projects.
        my_path: str = os.path.realpath(os.path.abspath(__file__))
        me_dir: str = os.path.normcase(os.path.splitext(my_path)[0])
        vsr_dir: str = os.path.normcase(os.path.splitext(versioneer_py)[0])
        if me_dir != vsr_dir and "VERSIONEER_PEP518" not in globals():
            print(
                "Warning: build in %s is using versioneer.py from %s"
                % (os.path.dirname(my_path), versioneer_py)
            )
    except NameError:
        pass
    return root


def get_config_from_root(root: str) -> VersioneerConfig:
    """Read the project setup.cfg file to determine Versioneer config."""
    root_pth: Path = Path(root)
    pyproject_toml: Path = root_pth / "pyproject.toml"
    setup_cfg: Path = root_pth / "setup.cfg"
    section: Union[Dict[str, Any], configparser.SectionProxy, None] = None
    if pyproject_toml.exists() and have_tomllib:
        try:
            with open(pyproject_toml, "rb") as fobj:
                pp: Dict[str, Any] = tomllib.load(fobj)
            section = pp["tool"]["versioneer"]
        except (tomllib.TOMLDecodeError, KeyError) as e:
            print(f"Failed to load config from {pyproject_toml}: {e}")
            print("Try to load it from setup.cfg")
    if not section:
        parser: configparser.ConfigParser = configparser.ConfigParser()
        with open(setup_cfg) as cfg_file:
            parser.read_file(cfg_file)
        parser.get("versioneer", "VCS")  # raise error if missing

        section = parser["versioneer"]

    cfg: VersioneerConfig = VersioneerConfig()
    cfg.VCS = section["VCS"]
    cfg.style = section.get("style", "")
    cfg.versionfile_source = cast(str, section.get("versionfile_source"))
    cfg.versionfile_build = section.get("versionfile_build")
    cfg.tag_prefix = cast(str, section.get("tag_prefix"))
    if cfg.tag_prefix in ("''", '""', None):
        cfg.tag_prefix = ""
    cfg.parentdir_prefix = section.get("parentdir_prefix")
    if isinstance(section, configparser.SectionProxy):
        cfg.verbose = section.getboolean("verbose")
    else:
        cfg.verbose = section.get("verbose")
    return cfg


class NotThisMethod(Exception):
    """Exception raised if a method is not valid for the current scenario."""


# these dictionaries contain VCS-specific tools
LONG_VERSION_PY: Dict[str, str] = {}
HANDLERS: Dict[str, Dict[str, Callable]] = {}


def register_vcs_handler(vcs: str, method: str) -> Callable[[Callable], Callable]:
    """Create decorator to mark a method as the handler of a VCS."""
    def decorate(f: Callable) -> Callable:
        HANDLERS.setdefault(vcs, {})[method] = f
        return f
    return decorate


def run_command(
    commands: List[str],
    args: List[str],
    cwd: Optional[str] = None,
    verbose: bool = False,
    hide_stderr: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process: Optional[subprocess.Popen] = None

    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        startupinfo: subprocess.STARTUPINFO = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs["startupinfo"] = startupinfo

    for command in commands:
        try:
            dispcmd: str = str([command] + args)
            process = subprocess.Popen(
                [command] + args,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=(subprocess.PIPE if hide_stderr else None),
                **popen_kwargs,
            )
            break
        except OSError as e:
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % dispcmd)
                print(e)
            return None, None
    else:
        if verbose:
            print("unable to find command, tried %s" % (commands,))
        return None, None
    stdout_bytes: bytes = process.communicate()[0]
    stdout: str = stdout_bytes.strip().decode()
    if process.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % dispcmd)
            print("stdout was %s" % stdout)
        return None, process.returncode
    return stdout, process.returncode


LONG_VERSION_PY["git"] = r'''
# This file helps to compute a version number in source trees obtained from
# git-archive tarball (such as those provided by githubs download-from-tag
# feature). Distribution tarballs (built by setup.py sdist) and build
# directories (produced by setup.py build) will contain a much shorter file
# that just contains the computed version number.

# This file is released into the public domain.
# Generated by versioneer-0.29
# https://github.com/python-versioneer/python-versioneer

"""Git implementation of _version.py."""

import errno
import os
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import functools


def get_keywords() -> Dict[str, str]:
    """Get the keywords needed to look up the version information."""
    git_refnames: str = "%(DOLLAR)sFormat:%%d%(DOLLAR)s"
    git_full: str = "%(DOLLAR)sFormat:%%H%(DOLLAR)s"
    git_date: str = "%(DOLLAR)sFormat:%%ci%(DOLLAR)s"
    keywords: Dict[str, str] = {"refnames": git_refnames, "full": git_full, "date": git_date}
    return keywords


class VersioneerConfig:
    """Container for Versioneer configuration parameters."""
    VCS: str
    style: str
    tag_prefix: str
    parentdir_prefix: str
    versionfile_source: str
    verbose: bool


def get_config() -> VersioneerConfig:
    """Create, populate and return the VersioneerConfig() object."""
    cfg: VersioneerConfig = VersioneerConfig()
    cfg.VCS = "git"
    cfg.style = "%(STYLE)s"
    cfg.tag_prefix = "%(TAG_PREFIX)s"
    cfg.parentdir_prefix = "%(PARENTDIR_PREFIX)s"
    cfg.versionfile_source = "%(VERSIONFILE_SOURCE)s"
    cfg.verbose = False
    return cfg


class NotThisMethod(Exception):
    """Exception raised if a method is not valid for the current scenario."""


LONG_VERSION_PY: Dict[str, str] = {}
HANDLERS: Dict[str, Dict[str, Callable]] = {}


def register_vcs_handler(vcs: str, method: str) -> Callable[[Callable], Callable]:
    """Create decorator to mark a method as the handler of a VCS."""
    def decorate(f: Callable) -> Callable:
        if vcs not in HANDLERS:
            HANDLERS[vcs] = {}
        HANDLERS[vcs][method] = f
        return f
    return decorate


def run_command(
    commands: List[str],
    args: List[str],
    cwd: Optional[str] = None,
    verbose: bool = False,
    hide_stderr: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process: Optional[subprocess.Popen] = None

    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        startupinfo: subprocess.STARTUPINFO = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs["startupinfo"] = startupinfo

    for command in commands:
        try:
            dispcmd: str = str([command] + args)
            process = subprocess.Popen([command] + args, cwd=cwd, env=env,
                                       stdout=subprocess.PIPE,
                                       stderr=(subprocess.PIPE if hide_stderr else None), **popen_kwargs)
            break
        except OSError as e:
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % dispcmd)
                print(e)
            return None, None
    else:
        if verbose:
            print("unable to find command, tried %s" % (commands,))
        return None, None
    stdout: str = process.communicate()[0].strip().decode()
    if process.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % dispcmd)
            print("stdout was %s" % stdout)
        return None, process.returncode
    return stdout, process.returncode


def versions_from_parentdir(
    parentdir_prefix: str,
    root: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Try to determine the version from the parent directory name."""
    rootdirs: List[str] = []

    for _ in range(3):
        dirname: str = os.path.basename(root)
        if dirname.startswith(parentdir_prefix):
            return {"version": dirname[len(parentdir_prefix):],
                    "full-revisionid": None,
                    "dirty": False, "error": None, "date": None}
        rootdirs.append(root)
        root = os.path.dirname(root)
    if verbose:
        print("Tried directories %s but none started with prefix %s" %
              (str(rootdirs), parentdir_prefix))
    raise NotThisMethod("rootdir doesn't start with parentdir_prefix")


@register_vcs_handler("git", "get_keywords")
def git_get_keywords(versionfile_abs: str) -> Dict[str, str]:
    """Extract version information from the given file."""
    keywords: Dict[str, str] = {}
    try:
        with open(versionfile_abs, "r") as fobj:
            for line in fobj:
                if line.strip().startswith("git_refnames ="):
                    mo = re.search(r'=\s*"(.*)"', line)
                    if mo:
                        keywords["refnames"] = mo.group(1)
                if line.strip().startswith("git_full ="):
                    mo = re.search(r'=\s*"(.*)"', line)
                    if mo:
                        keywords["full"] = mo.group(1)
                if line.strip().startswith("git_date ="):
                    mo = re.search(r'=\s*"(.*)"', line)
                    if mo:
                        keywords["date"] = mo.group(1)
    except OSError:
        pass
    return keywords


@register_vcs_handler("git", "keywords")
def git_versions_from_keywords(
    keywords: Dict[str, str],
    tag_prefix: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Get version information from git keywords."""
    if "refnames" not in keywords:
        raise NotThisMethod("Short version file found")
    date: Optional[str] = keywords.get("date")
    if date is not None:
        date = date.splitlines()[-1]
        date = date.strip().replace(" ", "T", 1).replace(" ", "", 1)
    refnames: str = keywords["refnames"].strip()
    if refnames.startswith("$Format"):
        if verbose:
            print("keywords are unexpanded, not using")
        raise NotThisMethod("unexpanded keywords, not a git-archive tarball")
    refs: set = {r.strip() for r in refnames.strip("()").split(",")}
    TAG: str = "tag: "
    tags: set = {r[len(TAG):] for r in refs if r.startswith(TAG)}
    if not tags:
        tags = {r for r in refs if re.search(r'\d', r)}
        if verbose:
            print("discarding '%s', no digits" % ",".join(refs - tags))
    if verbose:
        print("likely tags: %s" % ",".join(sorted(tags)))
    for ref in sorted(tags):
        if ref.startswith(tag_prefix):
            r: str = ref[len(tag_prefix):]
            if not re.match(r'\d', r):
                continue
            if verbose:
                print("picking %s" % r)
            return {"version": r,
                    "full-revisionid": keywords["full"].strip(),
                    "dirty": False, "error": None,
                    "date": date}
    if verbose:
        print("no suitable tags, using unknown + full revision id")
    return {"version": "0+unknown",
            "full-revisionid": keywords["full"].strip(),
            "dirty": False, "error": "no suitable tags", "date": None}


@register_vcs_handler("git", "pieces_from_vcs")
def git_pieces_from_vcs(
    tag_prefix: str, root: str, verbose: bool, runner: Callable = run_command
) -> Dict[str, Any]:
    """Get version from 'git describe' in the root of the source tree."""
    GITS: List[str] = ["git"]
    if sys.platform == "win32":
        GITS = ["git.cmd", "git.exe"]

    env: Dict[str, str] = os.environ.copy()
    env.pop("GIT_DIR", None)
    runner = functools.partial(runner, env=env)

    _, rc = runner(GITS, ["rev-parse", "--git-dir"], cwd=root,
                   hide_stderr=not verbose)
    if rc != 0:
        if verbose:
            print("Directory %s not under git control" % root)
        raise NotThisMethod("'git rev-parse --git-dir' returned error")

    describe_out, rc = runner(GITS, [
        "describe", "--tags", "--dirty", "--always", "--long",
        "--match", f"{tag_prefix}[[:digit:]]*"
    ], cwd=root)
    if describe_out is None:
        raise NotThisMethod("'git describe' failed")
    describe_out = describe_out.strip()
    full_out, rc = runner(GITS, ["rev-parse", "HEAD"], cwd=root)
    if full_out is None:
        raise NotThisMethod("'git rev-parse' failed")
    full_out = full_out.strip()

    pieces: Dict[str, Any] = {}
    pieces["long"] = full_out
    pieces["short"] = full_out[:7]
    pieces["error"] = None

    branch_name, rc = runner(GITS, ["rev-parse", "--abbrev-ref", "HEAD"],
                             cwd=root)
    if rc != 0 or branch_name is None:
        raise NotThisMethod("'git rev-parse --abbrev-ref' returned error")
    branch_name = branch_name.strip()

    if branch_name == "HEAD":
        branches_raw, rc = runner(GITS, ["branch", "--contains"], cwd=root)
        if rc != 0 or branches_raw is None:
            raise NotThisMethod("'git branch --contains' returned error")
        branches: List[str] = branches_raw.split("\n")
        if "(" in branches[0]:
            branches.pop(0)
        branches = [branch[2:] for branch in branches]
        if "master" in branches:
            branch_name = "master"
        elif not branches:
            branch_name = None  # type: ignore
        else:
            branch_name = branches[0]
    pieces["branch"] = branch_name

    git_describe: str = describe_out
    dirty: bool = git_describe.endswith("-dirty")
    pieces["dirty"] = dirty
    if dirty:
        git_describe = git_describe[:git_describe.rindex("-dirty")]

    if "-" in git_describe:
        mo = re.search(r'^(.+)-(\d+)-g([0-9a-f]+)$', git_describe)
        if not mo:
            pieces["error"] = ("unable to parse git-describe output: '%s'" % describe_out)
            return pieces
        full_tag: str = mo.group(1)
        if not full_tag.startswith(tag_prefix):
            if verbose:
                fmt: str = "tag '%s' doesn't start with prefix '%s'"
                print(fmt % (full_tag, tag_prefix))
            pieces["error"] = ("tag '%s' doesn't start with prefix '%s'" % (full_tag, tag_prefix))
            return pieces
        pieces["closest-tag"] = full_tag[len(tag_prefix):]
        pieces["distance"] = int(mo.group(2))
        pieces["short"] = mo.group(3)
    else:
        pieces["closest-tag"] = None
        out, rc = runner(GITS, ["rev-list", "HEAD", "--left-right"], cwd=root)
        pieces["distance"] = len(out.split())
    date: str = runner(GITS, ["show", "-s", "--format=%ci", "HEAD"], cwd=root)[0].strip()
    date = date.splitlines()[-1]
    pieces["date"] = date.strip().replace(" ", "T", 1).replace(" ", "", 1)
    return pieces


def plus_or_dot(pieces: Dict[str, Any]) -> str:
    """Return a + if we don't already have one, else return a ."""
    if "+" in pieces.get("closest-tag", ""):
        return "."
    return "+"


def render_pep440(pieces: Dict[str, Any]) -> str:
    """Build up version string, with post-release 'local version identifier'."""
    if pieces["closest-tag"]:
        rendered: str = pieces["closest-tag"]
        if pieces["distance"] or pieces["dirty"]:
            rendered += plus_or_dot(pieces)
            rendered += "%d.g%s" % (pieces["distance"], pieces["short"])
    else:
        rendered = "0+untagged.%d.g%s" % (pieces["distance"], pieces["short"])
        if pieces["dirty"]:
            rendered += ".dirty"
    return rendered


def render_pep440_branch(pieces: Dict[str, Any]) -> str:
    """TAG[[.dev0]+DISTANCE.gHEX[.dirty]] ."""
    if pieces["closest-tag"]:
        rendered: str = pieces["closest-tag"]
        if pieces["distance"] or pieces["dirty"]:
            if pieces["branch"] != "master":
                rendered += ".dev0"
            rendered += plus_or_dot(pieces)
            rendered += "%d.g%s" % (pieces["distance"], pieces["short"])
            if pieces["dirty"]:
                rendered += ".dirty"
    else:
        rendered = "0"
        if pieces["branch"] != "master":
            rendered += ".dev0"
        rendered += "+untagged.%d.g%s" % (pieces["distance"], pieces["short"])
        if pieces["dirty"]:
            rendered += ".dirty"
    return rendered


def pep440_split_post(ver: str) -> Tuple[str, Optional[int]]:
    """Split pep440 version string at the post-release segment."""
    vc: List[str] = ver.split(".post")
    return vc[0], int(vc[1] or 0) if len(vc) == 2 else None


def render_pep440_pre(pieces: Dict[str, Any]) -> str:
    """TAG[.postN.devDISTANCE] -- No -dirty."""
    if pieces["closest-tag"]:
        if pieces["distance"]:
            tag_version, post_version = pep440_split_post(pieces["closest-tag"])
            rendered: str = tag_version
            if post_version is not None:
                rendered += ".post%d.dev%d" % (post_version + 1, pieces["distance"])
            else:
                rendered += ".post0.dev%d" % (pieces["distance"])
        else:
            rendered = pieces["closest-tag"]
    else:
        rendered = "0.post0.dev%d" % pieces["distance"]
    return rendered


def render_pep440_post(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]+gHEX] ."""
    if pieces["closest-tag"]:
        rendered: str = pieces["closest-tag"]
        if pieces["distance"] or pieces["dirty"]:
            rendered += ".post%d" % pieces["distance"]
            if pieces["dirty"]:
                rendered += ".dev0"
            rendered += plus_or_dot(pieces)
            rendered += "g%s" % pieces["short"]
    else:
        rendered = "0.post%d" % pieces["distance"]
        if pieces["dirty"]:
            rendered += ".dev0"
        rendered += "+g%s" % pieces["short"]
    return rendered


def render_pep440_post_branch(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]+gHEX[.dirty]] ."""
    if pieces["closest-tag"]:
        rendered: str = pieces["closest-tag"]
        if pieces["distance"] or pieces["dirty"]:
            rendered += ".post%d" % pieces["distance"]
            if pieces["branch"] != "master":
                rendered += ".dev0"
            rendered += plus_or_dot(pieces)
            rendered += "g%s" % pieces["short"]
            if pieces["dirty"]:
                rendered += ".dirty"
    else:
        rendered = "0.post%d" % pieces["distance"]
        if pieces["branch"] != "master":
            rendered += ".dev0"
        rendered += "+g%s" % pieces["short"]
        if pieces["dirty"]:
            rendered += ".dirty"
    return rendered


def render_pep440_old(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]] ."""
    if pieces["closest-tag"]:
        rendered: str = pieces["closest-tag"]
        if pieces["distance"] or pieces["dirty"]:
            rendered += ".post%d" % pieces["distance"]
            if pieces["dirty"]:
                rendered += ".dev0"
    else:
        rendered = "0.post%d" % pieces["distance"]
        if pieces["dirty"]:
            rendered += ".dev0"
    return rendered


def render_git_describe(pieces: Dict[str, Any]) -> str:
    """TAG[-DISTANCE-gHEX][-dirty]."""
    if pieces["closest-tag"]:
        rendered: str = pieces["closest-tag"]
        if pieces["distance"]:
            rendered += "-%d-g%s" % (pieces["distance"], pieces["short"])
    else:
        rendered = pieces["short"]
    if pieces["dirty"]:
        rendered += "-dirty"
    return rendered


def render_git_describe_long(pieces: Dict[str, Any]) -> str:
    """TAG-DISTANCE-gHEX[-dirty]."""
    if pieces["closest-tag"]:
        rendered: str = pieces["closest-tag"]
        rendered += "-%d-g%s" % (pieces["distance"], pieces["short"])
    else:
        rendered = pieces["short"]
    if pieces["dirty"]:
        rendered += "-dirty"
    return rendered


def render(pieces: Dict[str, Any], style: str) -> Dict[str, Any]:
    """Render the given version pieces into the requested style."""
    if pieces["error"]:
        return {"version": "unknown",
                "full-revisionid": pieces.get("long"),
                "dirty": None,
                "error": pieces["error"],
                "date": None}

    if not style or style == "default":
        style = "pep440"

    if style == "pep440":
        rendered: str = render_pep440(pieces)
    elif style == "pep440-branch":
        rendered = render_pep440_branch(pieces)
    elif style == "pep440-pre":
        rendered = render_pep440_pre(pieces)
    elif style == "pep440-post":
        rendered = render_pep440_post(pieces)
    elif style == "pep440-post-branch":
        rendered = render_pep440_post_branch(pieces)
    elif style == "pep440-old":
        rendered = render_pep440_old(pieces)
    elif style == "git-describe":
        rendered = render_git_describe(pieces)
    elif style == "git-describe-long":
        rendered = render_git_describe_long(pieces)
    else:
        raise ValueError("unknown style '%s'" % style)

    return {"version": rendered, "full-revisionid": pieces["long"],
            "dirty": pieces["dirty"], "error": None,
            "date": pieces.get("date")}


def get_versions() -> Dict[str, Any]:
    """Get version information or return default if unable to do so."""
    if "versioneer" in sys.modules:
        del sys.modules["versioneer"]

    root: str = get_root()
    cfg: VersioneerConfig = get_config_from_root(root)

    assert cfg.VCS is not None, "please set [versioneer]VCS= in setup.cfg"
    handlers: Optional[Dict[str, Callable]] = HANDLERS.get(cfg.VCS)
    assert handlers, "unrecognized VCS '%s'" % cfg.VCS
    verbose: bool = bool(cfg.verbose)
    assert cfg.versionfile_source is not None, "please set versioneer.versionfile_source"
    assert cfg.tag_prefix is not None, "please set versioneer.tag_prefix"

    versionfile_abs: str = os.path.join(root, cfg.versionfile_source)

    get_keywords_f: Optional[Callable] = handlers.get("get_keywords")
    from_keywords_f: Optional[Callable] = handlers.get("keywords")
    if get_keywords_f and from_keywords_f:
        try:
            keywords: Dict[str, str] = get_keywords_f(versionfile_abs)
            ver: Dict[str, Any] = from_keywords_f(keywords, cfg.tag_prefix, verbose)
            if verbose:
                print("got version from expanded keyword %s" % ver)
            return ver
        except NotThisMethod:
            pass

    try:
        ver = versions_from_file(versionfile_abs)
        if verbose:
            print("got version from file %s %s" % (versionfile_abs, ver))
        return ver
    except NotThisMethod:
        pass

    from_vcs_f: Optional[Callable] = handlers.get("pieces_from_vcs")
    if from_vcs_f:
        try:
            pieces: Dict[str, Any] = from_vcs_f(cfg.tag_prefix, root, verbose)
            ver = render(pieces, cfg.style)
            if verbose:
                print("got version from VCS %s" % ver)
            return ver
        except NotThisMethod:
            pass

    try:
        if cfg.parentdir_prefix:
            ver = versions_from_parentdir(cfg.parentdir_prefix, root, verbose)
            if verbose:
                print("got version from parentdir %s" % ver)
            return ver
    except NotThisMethod:
        pass

    if verbose:
        print("unable to compute version")

    return {"version": "0+unknown",
            "full-revisionid": None,
            "dirty": None,
            "error": "unable to compute version",
            "date": None}


def get_version() -> str:
    """Get the short version string for this project."""
    return get_versions()["version"]


def get_cmdclass(cmdclass: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the custom setuptools subclasses used by Versioneer."""
    if "versioneer" in sys.modules:
        del sys.modules["versioneer"]

    cmds: Dict[str, Any] = {} if cmdclass is None else cmdclass.copy()

    from setuptools import Command

    class cmd_version(Command):
        description: str = "report generated version string"
        user_options: List[Tuple[str, str, str]] = []
        boolean_options: List[str] = []

        def initialize_options(self) -> None:
            pass

        def finalize_options(self) -> None:
            pass

        def run(self) -> None:
            vers: Dict[str, Any] = get_versions(verbose=True)
            print("Version: %s" % vers["version"])
            print(" full-revisionid: %s" % vers.get("full-revisionid"))
            print(" dirty: %s" % vers.get("dirty"))
            print(" date: %s" % vers.get("date"))
            if vers["error"]:
                print(" error: %s" % vers["error"])

    cmds["version"] = cmd_version

    if "build_py" in cmds:
        _build_py: Any = cmds["build_py"]
    else:
        from setuptools.command.build_py import build_py as _build_py
    class cmd_build_py(_build_py):
        def run(self) -> None:
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            versions: Dict[str, Any] = get_versions()
            _build_py.run(self)
            if getattr(self, "editable_mode", False):
                return
            if cfg.versionfile_build:
                target_versionfile: str = os.path.join(self.build_lib, cfg.versionfile_build)
                print("UPDATING %s" % target_versionfile)
                write_to_version_file(target_versionfile, versions)
    cmds["build_py"] = cmd_build_py

    if "build_ext" in cmds:
        _build_ext: Any = cmds["build_ext"]
    else:
        from setuptools.command.build_ext import build_ext as _build_ext
    class cmd_build_ext(_build_ext):
        def run(self) -> None:
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            versions: Dict[str, Any] = get_versions()
            _build_ext.run(self)
            if self.inplace:
                return
            if not cfg.versionfile_build:
                return
            target_versionfile: str = os.path.join(self.build_lib, cfg.versionfile_build)
            if not os.path.exists(target_versionfile):
                print(
                    f"Warning: {target_versionfile} does not exist, skipping "
                    "version update. This can happen if you are running build_ext "
                    "without first running build_py."
                )
                return
            print("UPDATING %s" % target_versionfile)
            write_to_version_file(target_versionfile, versions)
    cmds["build_ext"] = cmd_build_ext

    if "cx_Freeze" in sys.modules:
        from cx_Freeze.dist import build_exe as _build_exe  # type: ignore
        class cmd_build_exe(_build_exe):
            def run(self) -> None:
                root: str = get_root()
                cfg: VersioneerConfig = get_config_from_root(root)
                versions: Dict[str, Any] = get_versions()
                target_versionfile: str = cfg.versionfile_source
                print("UPDATING %s" % target_versionfile)
                write_to_version_file(target_versionfile, versions)
                _build_exe.run(self)
                os.unlink(target_versionfile)
                with open(cfg.versionfile_source, "w") as f:
                    LONG: str = LONG_VERSION_PY[cfg.VCS]
                    f.write(
                        LONG
                        % {
                            "DOLLAR": "$",
                            "STYLE": cfg.style,
                            "TAG_PREFIX": cfg.tag_prefix,
                            "PARENTDIR_PREFIX": cfg.parentdir_prefix,
                            "VERSIONFILE_SOURCE": cfg.versionfile_source,
                        }
                    )
        cmds["build_exe"] = cmd_build_exe
        del cmds["build_py"]

    if "py2exe" in sys.modules:
        try:
            from py2exe.setuptools_buildexe import py2exe as _py2exe  # type: ignore
        except ImportError:
            from py2exe.distutils_buildexe import py2exe as _py2exe  # type: ignore
        class cmd_py2exe(_py2exe):
            def run(self) -> None:
                root: str = get_root()
                cfg: VersioneerConfig = get_config_from_root(root)
                versions: Dict[str, Any] = get_versions()
                target_versionfile: str = cfg.versionfile_source
                print("UPDATING %s" % target_versionfile)
                write_to_version_file(target_versionfile, versions)
                _py2exe.run(self)
                os.unlink(target_versionfile)
                with open(cfg.versionfile_source, "w") as f:
                    LONG: str = LONG_VERSION_PY[cfg.VCS]
                    f.write(
                        LONG
                        % {
                            "DOLLAR": "$",
                            "STYLE": cfg.style,
                            "TAG_PREFIX": cfg.tag_prefix,
                            "PARENTDIR_PREFIX": cfg.parentdir_prefix,
                            "VERSIONFILE_SOURCE": cfg.versionfile_source,
                        }
                    )
        cmds["py2exe"] = cmd_py2exe

    if "egg_info" in cmds:
        _egg_info: Any = cmds["egg_info"]
    else:
        from setuptools.command.egg_info import egg_info as _egg_info
    class cmd_egg_info(_egg_info):
        def find_sources(self) -> None:
            super().find_sources()
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            self.filelist.append("versioneer.py")
            if cfg.versionfile_source:
                self.filelist.append(cfg.versionfile_source)
            self.filelist.sort()
            self.filelist.remove_duplicates()
            from setuptools import unicode_utils
            normalized: List[str] = [
                unicode_utils.filesys_decode(f).replace(os.sep, "/")
                for f in self.filelist.files
            ]
            manifest_filename: str = os.path.join(self.egg_info, "SOURCES.txt")
            with open(manifest_filename, "w") as fobj:
                fobj.write("\n".join(normalized))
    cmds["egg_info"] = cmd_egg_info

    if "sdist" in cmds:
        _sdist: Any = cmds["sdist"]
    else:
        from setuptools.command.sdist import sdist as _sdist
    class cmd_sdist(_sdist):
        def run(self) -> None:
            versions: Dict[str, Any] = get_versions()
            self._versioneer_generated_versions = versions
            self.distribution.metadata.version = versions["version"]
            return _sdist.run(self)

        def make_release_tree(self, base_dir: str, files: List[str]) -> None:
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            _sdist.make_release_tree(self, base_dir, files)
            target_versionfile: str = os.path.join(base_dir, cfg.versionfile_source)
            print("UPDATING %s" % target_versionfile)
            write_to_version_file(
                target_versionfile, self._versioneer_generated_versions
            )
    cmds["sdist"] = cmd_sdist

    return cmds


CONFIG_ERROR: str = """
setup.cfg is missing the necessary Versioneer configuration. You need
a section like:

 [versioneer]
 VCS = git
 style = pep440
 versionfile_source = src/myproject/_version.py
 versionfile_build = myproject/_version.py
 tag_prefix =
 parentdir_prefix =

You will also need to edit your setup.py to use the results:

 import versioneer
 setup(version=versioneer.get_version(),
       cmdclass=versioneer.get_cmdclass(), ...)

Please read the docstring in ./versioneer.py for configuration instructions,
edit setup.cfg, and re-run the installer or 'python versioneer.py setup'.
"""

SAMPLE_CONFIG: str = """
# See the docstring in versioneer.py for instructions. Note that you must
# re-run 'versioneer.py setup' after changing this section, and commit the
# resulting files.

[versioneer]
#VCS = git
#style = pep440
#versionfile_source =
#versionfile_build =
#tag_prefix =
#parentdir_prefix =

"""

OLD_SNIPPET: str = """
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
"""

INIT_PY_SNIPPET: str = """
from . import {0}
__version__ = {0}.get_versions()['version']
"""


def do_vcs_install(versionfile_source: str, ipy: Optional[str]) -> None:
    """Git-specific installation logic for Versioneer."""
    GITS: List[str] = ["git"]
    if sys.platform == "win32":
        GITS = ["git.cmd", "git.exe"]
    files: List[str] = [versionfile_source]
    if ipy:
        files.append(ipy)
    if "VERSIONEER_PEP518" not in globals():
        try:
            my_path: str = __file__
            if my_path.endswith((".pyc", ".pyo")):
                my_path = os.path.splitext(my_path)[0] + ".py"
            versioneer_file: str = os.path.relpath(my_path)
        except NameError:
            versioneer_file = "versioneer.py"
        files.append(versioneer_file)
    present: bool = False
    try:
        with open(".gitattributes", "r") as fobj:
            for line in fobj:
                if line.strip().startswith(versionfile_source):
                    if "export-subst" in line.strip().split()[1:]:
                        present = True
                        break
    except OSError:
        pass
    if not present:
        with open(".gitattributes", "a+") as fobj:
            fobj.write(f"{versionfile_source} export-subst\n")
        files.append(".gitattributes")
    run_command(GITS, ["add", "--"] + files)


def versions_from_file(filename: str) -> Dict[str, Any]:
    """Try to determine the version from _version.py if present."""
    try:
        with open(filename) as f:
            contents: str = f.read()
    except OSError:
        raise NotThisMethod("unable to read _version.py")
    mo = re.search(
        r"version_json = '''\n(.*)'''  # END VERSION_JSON", contents, re.M | re.S
    )
    if not mo:
        mo = re.search(
            r"version_json = '''\r\n(.*)'''  # END VERSION_JSON", contents, re.M | re.S
        )
    if not mo:
        raise NotThisMethod("no version_json in _version.py")
    return json.loads(mo.group(1))


SHORT_VERSION_PY: str = """
# This file was generated by 'versioneer.py' (0.29) from
# revision-control system data, or from the parent directory name of an
# unpacked source archive. Distribution tarballs contain a pre-generated copy
# of this file.

import json

version_json = '''
%s
'''  # END VERSION_JSON


def get_versions():
    return json.loads(version_json)
"""


def write_to_version_file(filename: str, versions: Dict[str, Any]) -> None:
    """Write the given version number to the given _version.py file."""
    contents: str = json.dumps(versions, sort_keys=True, indent=1, separators=(",", ": "))
    with open(filename, "w") as f:
        f.write(SHORT_VERSION_PY % contents)
    print("set %s to '%s'" % (filename, versions["version"]))


def do_setup() -> int:
    """Do main VCS-independent setup function for installing Versioneer."""
    root: str = get_root()
    try:
        cfg: VersioneerConfig = get_config_from_root(root)
    except (OSError, configparser.NoSectionError, configparser.NoOptionError) as e:
        if isinstance(e, (OSError, configparser.NoSectionError)):
            print("Adding sample versioneer config to setup.cfg", file=sys.stderr)
            with open(os.path.join(root, "setup.cfg"), "a") as f:
                f.write(SAMPLE_CONFIG)
        print(CONFIG_ERROR, file=sys.stderr)
        return 1
    print(" creating %s" % cfg.versionfile_source)
    with open(cfg.versionfile_source, "w") as f:
        LONG: str = LONG_VERSION_PY[cfg.VCS]
        f.write(
            LONG
            % {
                "DOLLAR": "$",
                "STYLE": cfg.style,
                "TAG_PREFIX": cfg.tag_prefix,
                "PARENTDIR_PREFIX": cfg.parentdir_prefix,
                "VERSIONFILE_SOURCE": cfg.versionfile_source,
            }
        )
    ipy: str = os.path.join(os.path.dirname(cfg.versionfile_source), "__init__.py")
    maybe_ipy: Optional[str] = ipy
    if os.path.exists(ipy):
        try:
            with open(ipy, "r") as f:
                old: str = f.read()
        except OSError:
            old = ""
        module: str = os.path.splitext(os.path.basename(cfg.versionfile_source))[0]
        snippet: str = INIT_PY_SNIPPET.format(module)
        if OLD_SNIPPET in old:
            print(" replacing boilerplate in %s" % ipy)
            with open(ipy, "w") as f:
                f.write(old.replace(OLD_SNIPPET, snippet))
        elif snippet not in old:
            print(" appending to %s" % ipy)
            with open(ipy, "a") as f:
                f.write(snippet)
        else:
            print(" %s unmodified" % ipy)
    else:
        print(" %s doesn't exist, ok" % ipy)
        maybe_ipy = None
    do_vcs_install(cfg.versionfile_source, maybe_ipy)
    return 0


def scan_setup_py() -> int:
    """Validate the contents of setup.py against Versioneer's expectations."""
    found: set = set()
    setters: bool = False
    errors: int = 0
    with open("setup.py", "r") as f:
        for line in f.readlines():
            if "import versioneer" in line:
                found.add("import")
            if "versioneer.get_cmdclass()" in line:
                found.add("cmdclass")
            if "versioneer.get_version()" in line:
                found.add("get_version")
            if "versioneer.VCS" in line:
                setters = True
            if "versioneer.versionfile_source" in line:
                setters = True
    if len(found) != 3:
        print("")
        print("Your setup.py appears to be missing some important items")
        print("(but I might be wrong). Please make sure it has something")
        print("roughly like the following:")
        print("")
        print(" import versioneer")
        print(" setup( version=versioneer.get_version(),")
        print("        cmdclass=versioneer.get_cmdclass(),  ...)")
        print("")
        errors += 1
    if setters:
        print("You should remove lines like 'versioneer.VCS = ' and")
        print("'versioneer.versionfile_source = ' . This configuration")
        print("now lives in setup.cfg, and should be removed from setup.py")
        print("")
        errors += 1
    return errors


def setup_command() -> NoReturn:
    """Set up Versioneer and exit with appropriate error code."""
    errors: int = do_setup()
    errors += scan_setup_py()
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    cmd: str = sys.argv[1]
    if cmd == "setup":
        setup_command()