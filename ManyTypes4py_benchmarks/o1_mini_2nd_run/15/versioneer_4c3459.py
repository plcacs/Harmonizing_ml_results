"""The Versioneer - like a rocketeer, but for versions.

The Versioneer
==============
...
[pypi-url]: https://pypi.python.org/pypi/versioneer/
[travis-image]:
https://img.shields.io/travis/com/python-versioneer/python-versioneer.svg
[travis-url]: https://travis-ci.com/github/python-versioneer/python-versioneer

"""
import configparser
import errno
import functools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
    cast,
)

have_tomllib: bool = True
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        have_tomllib = False


class VersioneerConfig:
    """Container for Versioneer configuration parameters."""

    VCS: Optional[str] = None
    style: Optional[str] = None
    versionfile_source: Optional[str] = None
    versionfile_build: Optional[str] = None
    tag_prefix: Optional[str] = None
    parentdir_prefix: Optional[str] = None
    verbose: Optional[bool] = None


def get_root() -> str:
    """Get the project root directory.

    We require that all commands are run from the project root, i.e. the
    directory that contains setup.py, setup.cfg, and versioneer.py .
    """
    root: str = os.path.realpath(os.path.abspath(os.getcwd()))
    setup_py: str = os.path.join(root, 'setup.py')
    pyproject_toml: str = os.path.join(root, 'pyproject.toml')
    versioneer_py: str = os.path.join(root, 'versioneer.py')
    if not (os.path.exists(setup_py) or os.path.exists(pyproject_toml) or os.path.exists(versioneer_py)):
        root = os.path.dirname(os.path.realpath(os.path.abspath(sys.argv[0])))
        setup_py = os.path.join(root, 'setup.py')
        pyproject_toml = os.path.join(root, 'pyproject.toml')
        versioneer_py = os.path.join(root, 'versioneer.py')
    if not (os.path.exists(setup_py) or os.path.exists(pyproject_toml) or os.path.exists(versioneer_py)):
        err = (
            "Versioneer was unable to run the project root directory. Versioneer requires setup.py to be executed from its immediate directory (like 'python setup.py COMMAND'), or in a way that lets it use sys.argv[0] to find the root (like 'python path/to/setup.py COMMAND')."
        )
        raise VersioneerBadRootError(err)
    try:
        my_path: str = os.path.realpath(os.path.abspath(__file__))
        me_dir: str = os.path.normcase(os.path.splitext(my_path)[0])
        vsr_dir: str = os.path.normcase(os.path.splitext(versioneer_py)[0])
        if me_dir != vsr_dir and 'VERSIONEER_PEP518' not in globals():
            print('Warning: build in %s is using versioneer.py from %s' % (os.path.dirname(my_path), versioneer_py))
    except NameError:
        pass
    return root


def get_config_from_root(root: str) -> VersioneerConfig:
    """Read the project setup.cfg file to determine Versioneer config."""
    root_pth: Path = Path(root)
    pyproject_toml: Path = root_pth / 'pyproject.toml'
    setup_cfg: Path = root_pth / 'setup.cfg'
    section: Optional[Union[configparser.SectionProxy, Dict[str, str]]] = None
    if pyproject_toml.exists() and have_tomllib:
        try:
            with open(pyproject_toml, 'rb') as fobj:
                pp: Dict[str, Any] = tomllib.load(fobj)
            section = pp['tool']['versioneer']
        except (tomllib.TOMLDecodeError, KeyError) as e:
            print(f'Failed to load config from {pyproject_toml}: {e}')
            print('Try to load it from setup.cfg')
    if not section:
        parser: configparser.ConfigParser = configparser.ConfigParser()
        with open(setup_cfg) as cfg_file:
            parser.read_file(cfg_file)
        parser.get('versioneer', 'VCS')
        section = parser['versioneer']
    cfg: VersioneerConfig = VersioneerConfig()
    cfg.VCS = section['VCS']
    cfg.style = section.get('style', '')
    cfg.versionfile_source = cast(Optional[str], section.get('versionfile_source'))
    cfg.versionfile_build = section.get('versionfile_build')
    cfg.tag_prefix = cast(Optional[str], section.get('tag_prefix'))
    if cfg.tag_prefix in ("''", '""', None):
        cfg.tag_prefix = ''
    cfg.parentdir_prefix = section.get('parentdir_prefix')
    if isinstance(section, configparser.SectionProxy):
        cfg.verbose = section.getboolean('verbose')
    else:
        cfg.verbose = section.get('verbose') is not None
    return cfg


class NotThisMethod(Exception):
    """Exception raised if a method is not valid for the current scenario."""


LONG_VERSION_PY: Dict[str, str] = {}
HANDLERS: Dict[str, Dict[str, Callable[..., Any]]] = {}


def register_vcs_handler(vcs: str, method: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create decorator to mark a method as the handler of a VCS."""

    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        """Store f in HANDLERS[vcs][method]."""
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
    process: Optional[subprocess.Popen[Any]] = None
    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == 'win32':
        startupinfo: subprocess.STARTUPINFO = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs['startupinfo'] = startupinfo
    for command in commands:
        try:
            dispcmd: str = str([command] + args)
            process = subprocess.Popen(
                [command] + args,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if hide_stderr else None,
                **popen_kwargs
            )
            break
        except OSError as e:
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print('unable to run %s' % dispcmd)
                print(e)
            return (None, None)
    else:
        if verbose:
            print('unable to find command, tried %s' % (commands,))
        return (None, None)
    if process.stdout:
        stdout_bytes: bytes = process.stdout.read()
    else:
        stdout_bytes = b''
    stdout: str = stdout_bytes.strip().decode()
    if process.returncode != 0:
        if verbose:
            print('unable to run %s (error)' % dispcmd)
            print('stdout was %s' % stdout)
        return (None, process.returncode)
    return (stdout, process.returncode)


LONG_VERSION_PY['git'] = '\n# This file helps to compute a version number in source trees obtained from\n# git-archive tarball (such as those provided by githubs download-from-tag\n# feature). Distribution tarballs (built by setup.py sdist) and build\n# directories (produced by setup.py build) will contain a much shorter file\n# that just contains the generated version data.\n\n# This file is released into the public domain.\n# Generated by versioneer-0.29\n# https://github.com/python-versioneer/python-versioneer\n\n"""Git implementation of _version.py."""\n\nimport errno\nimport os\nimport re\nimport subprocess\nimport sys\nfrom typing import Any, Callable, Dict, List, Optional, Tuple\nimport functools\n\n\ndef get_keywords() -> Dict[str, str]:\n    """Get the keywords needed to look up the version information."""\n    # these strings will be replaced by git during git-archive.\n    # setup.py/versioneer.py will grep for the variable names, so they must\n    # each be defined on a line of their own. _version.py will just call\n    # get_keywords().\n    git_refnames = "%(DOLLAR)sFormat:%%d%(DOLLAR)s"\n    git_full = "%(DOLLAR)sFormat:%%H%(DOLLAR)s"\n    git_date = "%(DOLLAR)sFormat:%%ci%(DOLLAR)s"\n    keywords = {"refnames": git_refnames, "full": git_full, "date": git_date}\n    return keywords\n\n\nclass VersioneerConfig:\n    """Container for Versioneer configuration parameters."""\n\n    VCS: str\n    style: str\n    tag_prefix: str\n    parentdir_prefix: str\n    versionfile_source: str\n    verbose: bool\n\n\ndef get_config() -> VersioneerConfig:\n    """Create, populate and return the VersioneerConfig() object."""\n    # these strings are filled in when \'setup.py versioneer\' creates\n    # _version.py\n    cfg = VersioneerConfig()\n    cfg.VCS = "git"\n    cfg.style = "%(STYLE)s"\n    cfg.tag_prefix = "%(TAG_PREFIX)s"\n    cfg.parentdir_prefix = "%(PARENTDIR_PREFIX)s"\n    cfg.versionfile_source = "%(VERSIONFILE_SOURCE)s"\n    cfg.verbose = False\n    return cfg\n\n\nclass NotThisMethod(Exception):\n    """Exception raised if a method is not valid for the current scenario."""\n\n\nLONG_VERSION_PY: Dict[str, str] = {}\nHANDLERS: Dict[str, Dict[str, Callable[..., Any]]] = {}\n\n\ndef register_vcs_handler(vcs: str, method: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # decorator\n    """Create decorator to mark a method as the handler of a VCS."""\n    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:\n        """Store f in HANDLERS[vcs][method]."""\n        if vcs not in HANDLERS:\n            HANDLERS[vcs] = {}\n        HANDLERS[vcs][method] = f\n        return f\n    return decorate\n\n\ndef run_command(\n    commands: List[str],\n    args: List[str],\n    cwd: Optional[str] = None,\n    verbose: bool = False,\n    hide_stderr: bool = False,\n    env: Optional[Dict[str, str]] = None,\n) -> Tuple[Optional[str], Optional[int]]:\n    """Call the given command(s)."""\n    assert isinstance(commands, list)\n    process: Optional[subprocess.Popen[Any]] = None\n\n    popen_kwargs: Dict[str, Any] = {}\n    if sys.platform == "win32":\n        # This hides the console window if pythonw.exe is used\n        startupinfo = subprocess.STARTUPINFO()\n        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW\n        popen_kwargs["startupinfo"] = startupinfo\n\n    for command in commands:\n        try:\n            dispcmd = str([command] + args)\n            # remember shell=False, so use git.cmd on windows, not just git\n            process = subprocess.Popen([command] + args, cwd=cwd, env=env,\n                                       stdout=subprocess.PIPE,\n                                       stderr=(subprocess.PIPE if hide_stderr\n                                               else None), **popen_kwargs)\n            break\n        except OSError as e:\n            if e.errno == errno.ENOENT:\n                continue\n            if verbose:\n                print("unable to run %%s" %% dispcmd)\n                print(e)\n            return None, None\n    else:\n        if verbose:\n            print("unable to find command, tried %%s" %% (commands,))\n        return None, None\n    stdout_bytes: bytes = process.communicate()[0].strip()\n    stdout: str = stdout_bytes.decode()\n    if process.returncode != 0:\n        if verbose:\n            print("unable to run %%s (error)" %% dispcmd)\n            print("stdout was %%s" %% stdout)\n        return None, process.returncode\n    return stdout, process.returncode\n'  # noqa


@register_vcs_handler("git", "get_keywords")
def git_get_keywords(versionfile_abs: str) -> Dict[str, str]:
    """Extract version information from the given file."""
    keywords: Dict[str, str] = {}
    try:
        with open(versionfile_abs, "r") as fobj:
            for line in fobj:
                if line.strip().startswith("git_refnames ="):
                    mo = re.search(r'=\\s*"(.*)"', line)
                    if mo:
                        keywords["refnames"] = mo.group(1)
                if line.strip().startswith("git_full ="):
                    mo = re.search(r'=\\s*"(.*)"', line)
                    if mo:
                        keywords["full"] = mo.group(1)
                if line.strip().startswith("git_date ="):
                    mo = re.search(r'=\\s*"(.*)"', line)
                    if mo:
                        keywords["date"] = mo.group(1)
    except OSError:
        pass
    return keywords


@register_vcs_handler("git", "keywords")
def git_versions_from_keywords(
    keywords: Dict[str, str], tag_prefix: str, verbose: bool
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
    tags: set = {r[len(TAG) :] for r in refs if r.startswith(TAG)}
    if not tags:
        tags = {r for r in refs if re.search(r"\\d", r)}
        if verbose:
            print("discarding '%s', no digits" % ",".join(refs - tags))
    if verbose:
        print("likely tags: %s" % ",".join(sorted(tags)))
    for ref in sorted(tags):
        if ref.startswith(tag_prefix):
            r: str = ref[len(tag_prefix) :]
            if not re.match(r"\\d", r):
                continue
            if verbose:
                print("picking %s" % r)
            return {
                "version": r,
                "full-revisionid": keywords["full"].strip(),
                "dirty": False,
                "error": None,
                "date": date,
            }
    if verbose:
        print("no suitable tags, using unknown + full revision id")
    return {
        "version": "0+unknown",
        "full-revisionid": keywords["full"].strip(),
        "dirty": False,
        "error": "no suitable tags",
        "date": None,
    }


@register_vcs_handler("git", "pieces_from_vcs")
def git_pieces_from_vcs(
    tag_prefix: str,
    root: str,
    verbose: bool,
    runner: Callable[..., Tuple[Optional[str], Optional[int]]] = run_command,
) -> Dict[str, Any]:
    """Get version from 'git describe' in the root of the source tree.

    This only gets called if the git-archive 'subst' keywords were *not*
    expanded, and _version.py hasn't already been rewritten with a short
    version string, meaning we're inside a checked out source tree.
    """
    GITS: List[str] = ["git"]
    if sys.platform == "win32":
        GITS = ["git.cmd", "git.exe"]

    env: Dict[str, str] = os.environ.copy()
    env.pop("GIT_DIR", None)
    runner = functools.partial(runner, env=env)

    _, rc = runner(GITS, ["rev-parse", "--git-dir"], cwd=root, verbose=verbose, hide_stderr=not verbose)
    if rc != 0:
        if verbose:
            print("Directory %s not under git control" % root)
        raise NotThisMethod("'git rev-parse --git-dir' returned error")

    describe_out, rc = runner(
        GITS,
        [
            "describe",
            "--tags",
            "--dirty",
            "--always",
            "--long",
            "--match",
            f"{tag_prefix}[[:digit:]]*",
        ],
        cwd=root,
    )
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

    branch_name, rc = runner(GITS, ["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    if rc != 0 or branch_name is None:
        raise NotThisMethod("'git rev-parse --abbrev-ref' returned error")
    branch_name = branch_name.strip()

    if branch_name == "HEAD":
        branches, rc = runner(GITS, ["branch", "--contains"], cwd=root)
        if rc != 0 or branches is None:
            raise NotThisMethod("'git branch --contains' returned error")
        branches = branches.split("\n")

        if "(" in branches[0]:
            branches.pop(0)

        branches = [branch[2:] for branch in branches]
        if "master" in branches:
            branch_name = "master"
        elif not branches:
            branch_name = None
        else:
            branch_name = branches[0]

    pieces["branch"] = branch_name

    git_describe: str = describe_out
    dirty: bool = git_describe.endswith("-dirty")
    pieces["dirty"] = dirty
    if dirty:
        git_describe = git_describe[: git_describe.rindex("-dirty")]

    if "-" in git_describe:
        mo = re.search(r"^(.+)-(\\d+)-g([0-9a-f]+)$", git_describe)
        if not mo:
            pieces["error"] = "unable to parse git-describe output: '%s'" % describe_out
            return pieces

        full_tag: str = mo.group(1)
        if not full_tag.startswith(tag_prefix):
            if verbose:
                fmt = "tag '%s' doesn't start with prefix '%s'"
                print(fmt % (full_tag, tag_prefix))
            pieces["error"] = "tag '%s' doesn't start with prefix '%s'" % (full_tag, tag_prefix)
            return pieces
        pieces["closest-tag"] = full_tag[len(tag_prefix) :]
        pieces["distance"] = int(mo.group(2))
        pieces["short"] = mo.group(3)
    else:
        pieces["closest-tag"] = None
        out, rc = runner(GITS, ["rev-list", "HEAD", "--left-right"], cwd=root)
        pieces["distance"] = len(out.split())

    date_out, _ = runner(GITS, ["show", "-s", "--format=%ci", "HEAD"], cwd=root)
    date: Optional[str] = None
    if date_out:
        date = date_out.splitlines()[-1]
        date = date.strip().replace(" ", "T", 1).replace(" ", "", 1)
    pieces["date"] = date

    return pieces


def do_vcs_install(versionfile_source: str, ipy: Optional[str]) -> None:
    """Git-specific installation logic for Versioneer.

    For Git, this means creating/changing .gitattributes to mark _version.py
    for export-subst keyword substitution.
    """
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


def versions_from_parentdir(
    parentdir_prefix: str, root: str, verbose: bool
) -> Dict[str, Any]:
    """Try to determine the version from the parent directory name.

    Source tarballs conventionally unpack into a directory that includes both
    the project name and a version string. We will also support searching up
    two directory levels for an appropriately named parent directory
    """
    rootdirs: List[str] = []
    for _ in range(3):
        dirname: str = os.path.basename(root)
        if dirname.startswith(parentdir_prefix):
            return {
                "version": dirname[len(parentdir_prefix) :],
                "full-revisionid": None,
                "dirty": False,
                "error": None,
                "date": None,
            }
        rootdirs.append(root)
        root = os.path.dirname(root)
    if verbose:
        print(
            'Tried directories %s but none started with prefix %s'
            % (str(rootdirs), parentdir_prefix)
        )
    raise NotThisMethod("rootdir doesn't start with parentdir_prefix")


SHORT_VERSION_PY: str = "\n# This file was generated by 'versioneer.py' (0.29) from\n# revision-control system data, or from the parent directory name of an\n# unpacked source archive. Distribution tarballs contain a pre-generated copy\n# of this file.\n\nimport json\n\nversion_json = '''\n%s\n'''  # END VERSION_JSON\n\n\ndef get_versions() -> Dict[str, Any]:\n    return json.loads(version_json)\n"


def versions_from_file(filename: str) -> Dict[str, Any]:
    """Try to determine the version from _version.py if present."""
    try:
        with open(filename) as f:
            contents: str = f.read()
    except OSError:
        raise NotThisMethod("unable to read _version.py")
    mo: Optional[re.Match[str]] = re.search(
        r"version_json = '''\\n(.*)'''  # END VERSION_JSON", contents, re.M | re.S
    )
    if not mo:
        mo = re.search(
            r"version_json = '''\\r\\n(.*)'''  # END VERSION_JSON", contents, re.M | re.S
        )
    if not mo:
        raise NotThisMethod("no version_json in _version.py")
    return json.loads(mo.group(1))


def write_to_version_file(filename: str, versions: Dict[str, Any]) -> None:
    """Write the given version number to the given _version.py file."""
    contents: str = json.dumps(versions, sort_keys=True, indent=1, separators=(",", ": "))
    with open(filename, "w") as f:
        f.write(SHORT_VERSION_PY % contents)
    print("set %s to '%s'" % (filename, versions["version"]))


def plus_or_dot(pieces: Dict[str, Any]) -> str:
    """Return a + if we don't already have one, else return a ."""
    if "+" in pieces.get("closest-tag", ""):
        return "."
    return "+"


def render_pep440(pieces: Dict[str, Any]) -> str:
    """Build up version string, with post-release "local version identifier".

    Our goal: TAG[+DISTANCE.gHEX[.dirty]] . Note that if you
    get a tagged build and then dirty it, you'll get TAG+0.gHEX.dirty

    Exceptions:
    1: no tags. git_describe was just HEX. 0+untagged.DISTANCE.gHEX[.dirty]
    """
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
    """TAG[[.dev0]+DISTANCE.gHEX[.dirty]] .

    The ".dev0" means not master branch. Note that .dev0 sorts backwards
    (a feature branch will appear "older" than the master branch).

    Exceptions:
    1: no tags. 0[.dev0]+untagged.DISTANCE.gHEX[.dirty]
    """
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
    """Split pep440 version string at the post-release segment.

    Returns the release segments before the post-release and the
    post-release version number (or -1 if no post-release segment is present).
    """
    vc = str.split(ver, ".post")
    return (vc[0], int(vc[1] or 0) if len(vc) == 2 else None)


def render_pep440_pre(pieces: Dict[str, Any]) -> str:
    """TAG[.postN.devDISTANCE] -- No -dirty.

    Exceptions:
    1: no tags. 0.post0.devDISTANCE
    """
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
    """TAG[.postDISTANCE[.dev0]+gHEX] .

    The ".dev0" means dirty. Note that .dev0 sorts backwards
    (a dirty tree will appear "older" than the corresponding clean one),
    but you shouldn't be releasing software with -dirty anyways.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]
    """
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
    """TAG[.postDISTANCE[.dev0]+gHEX[.dirty]] .

    The ".dev0" means not master branch.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]+gHEX[.dirty]
    """
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
    """TAG[.postDISTANCE[.dev0]] .

    The ".dev0" means dirty.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]
    """
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
    """TAG[-DISTANCE-gHEX][-dirty].

    Like 'git describe --tags --dirty --always'.

    Exceptions:
    1: no tags. HEX[-dirty]  (note: no 'g' prefix)
    """
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
    """TAG-DISTANCE-gHEX[-dirty].

    Like 'git describe --tags --dirty --always -long'.
    The distance/hash is unconditional.

    Exceptions:
    1: no tags. HEX[-dirty]  (note: no 'g' prefix)
    """
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
        return {
            "version": "unknown",
            "full-revisionid": pieces.get("long"),
            "dirty": None,
            "error": pieces["error"],
            "date": None,
        }

    if not style or style == "default":
        style = "pep440"  # the default

    if style == "pep440":
        rendered = render_pep440(pieces)
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

    return {
        "version": rendered,
        "full-revisionid": pieces["long"],
        "dirty": pieces["dirty"],
        "error": None,
        "date": pieces.get("date"),
    }


class VersioneerBadRootError(Exception):
    """The project root directory is unknown or missing key files."""


def get_versions(verbose: bool = False) -> Dict[str, Any]:
    """Get the project version from whatever source is available.

    Returns dict with two keys: 'version' and 'full'.
    """
    if "versioneer" in sys.modules:
        del sys.modules["versioneer"]
    root: str = get_root()
    cfg: VersioneerConfig = get_config_from_root(root)
    assert cfg.VCS is not None, "please set [versioneer]VCS= in setup.cfg"
    handlers = HANDLERS.get(cfg.VCS)
    assert handlers, "unrecognized VCS '%s'" % cfg.VCS
    verbose = verbose or bool(cfg.verbose)
    assert cfg.versionfile_source is not None, "please set versioneer.versionfile_source"
    assert cfg.tag_prefix is not None, "please set versioneer.tag_prefix"
    versionfile_abs: str = os.path.join(root, cfg.versionfile_source)
    get_keywords_f: Optional[Callable[..., Dict[str, str]]] = handlers.get("get_keywords")
    from_keywords_f: Optional[Callable[..., Dict[str, Any]]] = handlers.get("keywords")
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
        ver: Dict[str, Any] = versions_from_file(versionfile_abs)
        if verbose:
            print("got version from file %s %s" % (versionfile_abs, ver))
        return ver
    except NotThisMethod:
        pass
    from_vcs_f: Optional[Callable[..., Dict[str, Any]]] = handlers.get("pieces_from_vcs")
    if from_vcs_f:
        try:
            pieces: Dict[str, Any] = from_vcs_f(cfg.tag_prefix, root, verbose)
            ver: Dict[str, Any] = render(pieces, cfg.style)
            if verbose:
                print("got version from VCS %s" % ver)
            return ver
        except NotThisMethod:
            pass
    try:
        if cfg.parentdir_prefix:
            ver: Dict[str, Any] = versions_from_parentdir(cfg.parentdir_prefix, root, verbose)
            if verbose:
                print("got version from parentdir %s" % ver)
            return ver
    except NotThisMethod:
        pass
    if verbose:
        print("unable to compute version")
    return {
        "version": "0+unknown",
        "full-revisionid": None,
        "dirty": None,
        "error": "unable to compute version",
        "date": None,
    }


def get_version() -> str:
    """Get the short version string for this project."""
    return get_versions()["version"]


def get_cmdclass(cmdclass: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the custom setuptools subclasses used by Versioneer.

    If the package uses a different cmdclass (e.g. one from numpy), it
    should be provide as an argument.
    """
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
        _build_py = cmds["build_py"]
    else:
        from setuptools.command.build_py import build_py as _build_py

    class cmd_build_py(_build_py):

        def run(self) -> None:
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            versions: Dict[str, Any] = get_versions()
            super().run()
            if getattr(self, "editable_mode", False):
                return
            if cfg.versionfile_build:
                target_versionfile: str = os.path.join(self.build_lib, cfg.versionfile_build)
                print("UPDATING %s" % target_versionfile)
                write_to_version_file(target_versionfile, versions)

    cmds["build_py"] = cmd_build_py

    if "build_ext" in cmds:
        _build_ext = cmds["build_ext"]
    else:
        from setuptools.command.build_ext import build_ext as _build_ext

    class cmd_build_ext(_build_ext):

        def run(self) -> None:
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            versions: Dict[str, Any] = get_versions()
            super().run()
            if not cfg.versionfile_build:
                return
            target_versionfile: str = os.path.join(self.build_lib, cfg.versionfile_build)
            if not os.path.exists(target_versionfile):
                print(
                    f"Warning: {target_versionfile} does not exist, skipping version update. "
                    "This can happen if you are running build_ext without first running build_py."
                )
                return
            print("UPDATING %s" % target_versionfile)
            write_to_version_file(target_versionfile, versions)

    cmds["build_ext"] = cmd_build_ext

    if "cx_Freeze" in sys.modules:
        from cx_Freeze.dist import build_exe as _build_exe

        class cmd_build_exe(_build_exe):

            def run(self) -> None:
                root: str = get_root()
                cfg: VersioneerConfig = get_config_from_root(root)
                versions: Dict[str, Any] = get_versions()
                target_versionfile: str = cfg.versionfile_source
                print("UPDATING %s" % target_versionfile)
                write_to_version_file(target_versionfile, versions)
                super().run()
                if target_versionfile:
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
            from py2exe.setuptools_buildexe import py2exe as _py2exe
        except ImportError:
            from py2exe.distutils_buildexe import py2exe as _py2exe

        class cmd_py2exe(_py2exe):

            def run(self) -> None:
                root: str = get_root()
                cfg: VersioneerConfig = get_config_from_root(root)
                versions: Dict[str, Any] = get_versions()
                target_versionfile: str = cfg.versionfile_source
                print("UPDATING %s" % target_versionfile)
                write_to_version_file(target_versionfile, versions)
                super().run()
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
        _egg_info = cmds["egg_info"]
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
                unicode_utils.filesys_decode(f).replace(os.sep, "/") for f in self.filelist.files
            ]
            manifest_filename: str = os.path.join(self.egg_info, "SOURCES.txt")
            with open(manifest_filename, "w") as fobj:
                fobj.write("\n".join(normalized))

    cmds["egg_info"] = cmd_egg_info

    if "sdist" in cmds:
        _sdist = cmds["sdist"]
    else:
        from setuptools.command.sdist import sdist as _sdist

    class cmd_sdist(_sdist):

        def run(self) -> None:
            versions: Dict[str, Any] = get_versions()
            self._versioneer_generated_versions = versions
            self.distribution.metadata.version = versions["version"]
            super().run()

        def make_release_tree(self, base_dir: str, files: List[str]) -> None:
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            super().make_release_tree(base_dir, files)
            target_versionfile: str = os.path.join(base_dir, cfg.versionfile_source)
            print("UPDATING %s" % target_versionfile)
            write_to_version_file(target_versionfile, self._versioneer_generated_versions)

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
 parentdir_prefix = myproject-

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

INIT_PY_SNIPPET: str = "\nfrom . import {0}\n__version__ = {0}.get_versions()['version']\n"


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
