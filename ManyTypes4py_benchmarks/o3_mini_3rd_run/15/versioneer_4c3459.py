#!/usr/bin/env python3
"""
The Versioneer - like a rocketeer, but for versions.

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

... [docstring truncated for brevity] ...
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
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union, cast

have_tomllib: bool = True
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        have_tomllib = False

class VersioneerConfig:
    """Container for Versioneer configuration parameters."""
    VCS: str = ""
    style: str = ""
    versionfile_source: str = ""
    versionfile_build: Optional[str] = None
    tag_prefix: str = ""
    parentdir_prefix: Optional[str] = None
    verbose: bool = False

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
        err: str = ("Versioneer was unable to run the project root directory. "
                    "Versioneer requires setup.py to be executed from its immediate directory (like 'python setup.py COMMAND'), or in a way that lets it use sys.argv[0] to find the root (like 'python path/to/setup.py COMMAND').")
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
    pyproject_toml = root_pth / 'pyproject.toml'
    setup_cfg = root_pth / 'setup.cfg'
    section: Optional[Any] = None
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
    cfg.versionfile_source = cast(str, section.get('versionfile_source'))
    cfg.versionfile_build = section.get('versionfile_build')
    cfg.tag_prefix = cast(str, section.get('tag_prefix'))
    if cfg.tag_prefix in ("''", '""', None):
        cfg.tag_prefix = ''
    cfg.parentdir_prefix = section.get('parentdir_prefix')
    if isinstance(section, configparser.SectionProxy):
        cfg.verbose = section.getboolean('verbose')
    else:
        cfg.verbose = section.get('verbose')
    return cfg

class NotThisMethod(Exception):
    """Exception raised if a method is not valid for the current scenario."""
    pass

LONG_VERSION_PY: Dict[str, str] = {}
HANDLERS: Dict[str, Dict[str, Callable[..., Any]]] = {}

def register_vcs_handler(vcs: str, method: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create decorator to mark a method as the handler of a VCS."""
    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        """Store f in HANDLERS[vcs][method]."""
        HANDLERS.setdefault(vcs, {})[method] = f
        return f
    return decorate

def run_command(commands: List[str], args: List[str], cwd: Optional[str] = None, verbose: bool = False, hide_stderr: bool = False, env: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[int]]:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process: Optional[subprocess.Popen] = None
    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == 'win32':
        startupinfo: subprocess.STARTUPINFO = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs['startupinfo'] = startupinfo
    for command in commands:
        try:
            dispcmd: str = str([command] + args)
            process = subprocess.Popen([command] + args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE if hide_stderr else None, **popen_kwargs)
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
    stdout_bytes: bytes = process.communicate()[0].strip()
    stdout: str = stdout_bytes.decode()
    if process.returncode != 0:
        if verbose:
            print('unable to run %s (error)' % dispcmd)
            print('stdout was %s' % stdout)
        return (None, process.returncode)
    return (stdout, process.returncode)

LONG_VERSION_PY['git'] = (
    "\n# This file helps to compute a version number in source trees obtained from\n"
    "# git-archive tarball (such as those provided by githubs download-from-tag\n"
    "# feature). Distribution tarballs (built by setup.py sdist) and build\n"
    "# directories (produced by setup.py build) will contain a much shorter file\n"
    "# that just contains the computed version number.\n\n"
    "# This file is released into the public domain.\n"
    "# Generated by versioneer-0.29\n"
    "# https://github.com/python-versioneer/python-versioneer\n\n"
    '"""Git implementation of _version.py."""\n\n'
    "import errno\n"
    "import os\n"
    "import re\n"
    "import subprocess\n"
    "import sys\n"
    "from typing import Any, Callable, Dict, List, Optional, Tuple\n"
    "import functools\n\n\n"
    "def get_keywords() -> Dict[str, str]:\n"
    '    """Get the keywords needed to look up the version information."""\n'
    "    git_refnames = \"%(DOLLAR)sFormat:%%d%(DOLLAR)s\"\n"
    "    git_full = \"%(DOLLAR)sFormat:%%H%(DOLLAR)s\"\n"
    "    git_date = \"%(DOLLAR)sFormat:%%ci%(DOLLAR)s\"\n"
    "    keywords = {\"refnames\": git_refnames, \"full\": git_full, \"date\": git_date}\n"
    "    return keywords\n\n\n"
    "class VersioneerConfig:\n"
    "    \"\"\"Container for Versioneer configuration parameters.\"\"\"\n\n"
    "    VCS: str\n"
    "    style: str\n"
    "    tag_prefix: str\n"
    "    parentdir_prefix: str\n"
    "    versionfile_source: str\n"
    "    verbose: bool\n\n\n"
    "def get_config() -> VersioneerConfig:\n"
    "    \"\"\"Create, populate and return the VersioneerConfig() object.\"\"\"\n"
    "    cfg = VersioneerConfig()\n"
    "    cfg.VCS = \"git\"\n"
    "    cfg.style = \"%(STYLE)s\"\n"
    "    cfg.tag_prefix = \"%(TAG_PREFIX)s\"\n"
    "    cfg.parentdir_prefix = \"%(PARENTDIR_PREFIX)s\"\n"
    "    cfg.versionfile_source = \"%(VERSIONFILE_SOURCE)s\"\n"
    "    cfg.verbose = False\n"
    "    return cfg\n\n\n"
    "class NotThisMethod(Exception):\n"
    "    \"\"\"Exception raised if a method is not valid for the current scenario.\"\"\"\n\n\n"
    "LONG_VERSION_PY: Dict[str, str] = {}\n"
    "HANDLERS: Dict[str, Dict[str, Callable]] = {}\n\n\n"
    "def register_vcs_handler(vcs: str, method: str) -> Callable:\n"
    "    def decorate(f: Callable) -> Callable:\n"
    "        if vcs not in HANDLERS:\n"
    "            HANDLERS[vcs] = {}\n"
    "        HANDLERS[vcs][method] = f\n"
    "        return f\n"
    "    return decorate\n\n\n"
    "def run_command(commands: List[str], args: List[str], cwd: Optional[str] = None, verbose: bool = False, hide_stderr: bool = False, env: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[int]]:\n"
    "    assert isinstance(commands, list)\n"
    "    process = None\n\n"
    "    popen_kwargs: Dict[str, Any] = {}\n"
    "    if sys.platform == \"win32\":\n"
    "        startupinfo = subprocess.STARTUPINFO()\n"
    "        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW\n"
    "        popen_kwargs[\"startupinfo\"] = startupinfo\n\n"
    "    for command in commands:\n"
    "        try:\n"
    "            dispcmd = str([command] + args)\n"
    "            process = subprocess.Popen([command] + args, cwd=cwd, env=env,\n"
    "                                       stdout=subprocess.PIPE,\n"
    "                                       stderr=(subprocess.PIPE if hide_stderr\n"
    "                                               else None), **popen_kwargs)\n"
    "            break\n"
    "        except OSError as e:\n"
    "            if e.errno == errno.ENOENT:\n"
    "                continue\n"
    "            if verbose:\n"
    "                print(\"unable to run %%s\" %% dispcmd)\n"
    "                print(e)\n"
    "            return None, None\n"
    "    else:\n"
    "        if verbose:\n"
    "            print(\"unable to find command, tried %%s\" %% (commands,))\n"
    "        return None, None\n"
    "    stdout = process.communicate()[0].strip().decode()\n"
    "    if process.returncode != 0:\n"
    "        if verbose:\n"
    "            print(\"unable to run %%s (error)\" %% dispcmd)\n"
    "            print(\"stdout was %%s\" %% stdout)\n"
    "        return None, process.returncode\n"
    "    return stdout, process.returncode\n\n\n"
    "def versions_from_parentdir(parentdir_prefix: str, root: str, verbose: bool) -> Dict[str, Any]:\n"
    "    rootdirs = []\n\n"
    "    for _ in range(3):\n"
    "        dirname = os.path.basename(root)\n"
    "        if dirname.startswith(parentdir_prefix):\n"
    "            return {\"version\": dirname[len(parentdir_prefix):],\n"
    "                    \"full-revisionid\": None,\n"
    "                    \"dirty\": False, \"error\": None, \"date\": None}\n"
    "        rootdirs.append(root)\n"
    "        root = os.path.dirname(root)\n\n"
    "    if verbose:\n"
    "        print(\"Tried directories %%s but none started with prefix %%s\" %%\n"
    "              (str(rootdirs), parentdir_prefix))\n"
    "    raise NotThisMethod(\"rootdir doesn't start with parentdir_prefix\")\n\n\n"
    "@register_vcs_handler(\"git\", \"get_keywords\")\n"
    "def git_get_keywords(versionfile_abs: str) -> Dict[str, str]:\n"
    "    keywords: Dict[str, str] = {}\n"
    "    try:\n"
    "        with open(versionfile_abs, \"r\") as fobj:\n"
    "            for line in fobj:\n"
    "                if line.strip().startswith(\"git_refnames =\"):\n"
    "                    mo = re.search(r'=\\s*\"(.*)\"', line)\n"
    "                    if mo:\n"
    "                        keywords[\"refnames\"] = mo.group(1)\n"
    "                if line.strip().startswith(\"git_full =\"):\n"
    "                    mo = re.search(r'=\\s*\"(.*)\"', line)\n"
    "                    if mo:\n"
    "                        keywords[\"full\"] = mo.group(1)\n"
    "                if line.strip().startswith(\"git_date =\"):\n"
    "                    mo = re.search(r'=\\s*\"(.*)\"', line)\n"
    "                    if mo:\n"
    "                        keywords[\"date\"] = mo.group(1)\n"
    "    except OSError:\n"
    "        pass\n"
    "    return keywords\n\n\n"
    "@register_vcs_handler(\"git\", \"keywords\")\n"
    "def git_versions_from_keywords(keywords: Dict[str, str], tag_prefix: str, verbose: bool) -> Dict[str, Any]:\n"
    "    if \"refnames\" not in keywords:\n"
    "        raise NotThisMethod(\"Short version file found\")\n"
    "    date = keywords.get(\"date\")\n"
    "    if date is not None:\n"
    "        date = date.splitlines()[-1]\n"
    "        date = date.strip().replace(\" \", \"T\", 1).replace(\" \", \"\", 1)\n"
    "    refnames = keywords[\"refnames\"].strip()\n"
    "    if refnames.startswith(\"$Format\"):\n"
    "        if verbose:\n"
    "            print(\"keywords are unexpanded, not using\")\n"
    "        raise NotThisMethod(\"unexpanded keywords, not a git-archive tarball\")\n"
    "    refs = {r.strip() for r in refnames.strip(\"()\").split(\",\")}\n"
    "    TAG = \"tag: \"\n"
    "    tags = {r[len(TAG):] for r in refs if r.startswith(TAG)}\n"
    "    if not tags:\n"
    "        tags = {r for r in refs if re.search(r'\\d', r)}\n"
    "        if verbose:\n"
    "            print(\"discarding '%s', no digits\" % ','.join(refs - tags))\n"
    "    if verbose:\n"
    "        print(\"likely tags: %s\" % ','.join(sorted(tags)))\n"
    "    for ref in sorted(tags):\n"
    "        if ref.startswith(tag_prefix):\n"
    "            r = ref[len(tag_prefix):]\n"
    "            if not re.match(r'\\d', r):\n"
    "                continue\n"
    "            if verbose:\n"
    "                print(\"picking %s\" % r)\n"
    "            return {\"version\": r,\n"
    "                    \"full-revisionid\": keywords[\"full\"].strip(),\n"
    "                    \"dirty\": False, \"error\": None, \"date\": date}\n"
    "    if verbose:\n"
    "        print(\"no suitable tags, using unknown + full revision id\")\n"
    "    return {\"version\": \"0+unknown\",\n"
    "            \"full-revisionid\": keywords[\"full\"].strip(),\n"
    "            \"dirty\": False, \"error\": \"no suitable tags\", \"date\": None}\n\n\n"
    "@register_vcs_handler(\"git\", \"pieces_from_vcs\")\n"
    "def git_pieces_from_vcs(tag_prefix: str, root: str, verbose: bool, runner: Callable[..., Any] = run_command) -> Dict[str, Any]:\n"
    "    GITS: List[str] = [\"git\"]\n"
    "    if sys.platform == \"win32\":\n"
    "        GITS = [\"git.cmd\", \"git.exe\"]\n\n"
    "    env: Dict[str, str] = os.environ.copy()\n"
    "    env.pop(\"GIT_DIR\", None)\n"
    "    runner = functools.partial(runner, env=env)\n\n"
    "    _, rc = runner(GITS, [\"rev-parse\", \"--git-dir\"], cwd=root, hide_stderr=not verbose)\n"
    "    if rc != 0:\n"
    "        if verbose:\n"
    "            print(\"Directory %s not under git control\" % root)\n"
    "        raise NotThisMethod(\"'git rev-parse --git-dir' returned error\")\n\n"
    "    describe_out, rc = runner(GITS, [\n"
    "        \"describe\", \"--tags\", \"--dirty\", \"--always\", \"--long\",\n"
    "        \"--match\", f\"{tag_prefix}[[:digit:]]*\"\n"
    "    ], cwd=root)\n"
    "    if describe_out is None:\n"
    "        raise NotThisMethod(\"'git describe' failed\")\n"
    "    describe_out = describe_out.strip()\n"
    "    full_out, rc = runner(GITS, [\"rev-parse\", \"HEAD\"], cwd=root)\n"
    "    if full_out is None:\n"
    "        raise NotThisMethod(\"'git rev-parse' failed\")\n"
    "    full_out = full_out.strip()\n\n"
    "    pieces: Dict[str, Any] = {}\n"
    "    pieces[\"long\"] = full_out\n"
    "    pieces[\"short\"] = full_out[:7]\n"
    "    pieces[\"error\"] = None\n\n"
    "    branch_name, rc = runner(GITS, [\"rev-parse\", \"--abbrev-ref\", \"HEAD\"], cwd=root)\n"
    "    if rc != 0 or branch_name is None:\n"
    "        raise NotThisMethod(\"'git rev-parse --abbrev-ref' returned error\")\n"
    "    branch_name = branch_name.strip()\n\n"
    "    if branch_name == \"HEAD\":\n"
    "        branches, rc = runner(GITS, [\"branch\", \"--contains\"], cwd=root)\n"
    "        if rc != 0 or branches is None:\n"
    "            raise NotThisMethod(\"'git branch --contains' returned error\")\n"
    "        branches = branches.split(\"\\n\")\n\n"
    "        if \"(\" in branches[0]:\n"
    "            branches.pop(0)\n\n"
    "        branches = [branch[2:] for branch in branches]\n"
    "        if \"master\" in branches:\n"
    "            branch_name = \"master\"\n"
    "        elif not branches:\n"
    "            branch_name = None\n"
    "        else:\n"
    "            branch_name = branches[0]\n\n"
    "    pieces[\"branch\"] = branch_name\n\n"
    "    git_describe: str = describe_out\n\n"
    "    dirty: bool = git_describe.endswith(\"-dirty\")\n"
    "    pieces[\"dirty\"] = dirty\n"
    "    if dirty:\n"
    "        git_describe = git_describe[:git_describe.rindex(\"-dirty\")]\n\n"
    "    if \"-\" in git_describe:\n"
    "        mo = re.search(r'^(.+)-(\\d+)-g([0-9a-f]+)$', git_describe)\n"
    "        if not mo:\n"
    "            pieces[\"error\"] = (\"unable to parse git-describe output: '%s'\" % describe_out)\n"
    "            return pieces\n\n"
    "        full_tag: str = mo.group(1)\n"
    "        if not full_tag.startswith(tag_prefix):\n"
    "            if verbose:\n"
    "                fmt: str = \"tag '%s' doesn't start with prefix '%s'\"\n"
    "                print(fmt % (full_tag, tag_prefix))\n"
    "            pieces[\"error\"] = (\"tag '%s' doesn't start with prefix '%s'\" % (full_tag, tag_prefix))\n"
    "            return pieces\n"
    "        pieces[\"closest-tag\"] = full_tag[len(tag_prefix):]\n"
    "        pieces[\"distance\"] = int(mo.group(2))\n"
    "        pieces[\"short\"] = mo.group(3)\n"
    "    else:\n"
    "        pieces[\"closest-tag\"] = None\n"
    "        out, rc = runner(GITS, [\"rev-list\", \"HEAD\", \"--left-right\"], cwd=root)\n"
    "        pieces[\"distance\"] = len(out.split())\n\n"
    "    date: str = runner(GITS, [\"show\", \"-s\", \"--format=%ci\", \"HEAD\"], cwd=root)[0].strip()\n"
    "    date = date.splitlines()[-1]\n"
    "    pieces[\"date\"] = date.strip().replace(\" \", \"T\", 1).replace(\" \", \"\", 1)\n\n"
    "    return pieces\n\n\n"
    "def plus_or_dot(pieces: Dict[str, Any]) -> str:\n"
    "    if \"+\" in pieces.get(\"closest-tag\", \"\"):\n"
    "        return \".\"\n"
    "    return \"+\"\n\n\n"
    "def render_pep440(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces[\"closest-tag\"]:\n"
    "        rendered = pieces[\"closest-tag\"]\n"
    "        if pieces[\"distance\"] or pieces[\"dirty\"]:\n"
    "            rendered += plus_or_dot(pieces)\n"
    "            rendered += \"%d.g%s\" % (pieces[\"distance\"], pieces[\"short\"])\n"
    "    else:\n"
    "        rendered = \"0+untagged.%d.g%s\" % (pieces[\"distance\"], pieces[\"short\"])\n"
    "        if pieces[\"dirty\"]:\n"
    "            rendered += \".dirty\"\n"
    "    return rendered\n\n\n"
    "def render_pep440_branch(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces[\"closest-tag\"]:\n"
    "        rendered = pieces[\"closest-tag\"]\n"
    "        if pieces[\"distance\"] or pieces[\"dirty\"]:\n"
    "            if pieces[\"branch\"] != \"master\":\n"
    "                rendered += \".dev0\"\n"
    "            rendered += plus_or_dot(pieces)\n"
    "            rendered += \"%d.g%s\" % (pieces[\"distance\"], pieces[\"short\"])\n"
    "            if pieces[\"dirty\"]:\n"
    "                rendered += \".dirty\"\n"
    "    else:\n"
    "        rendered = \"0\"\n"
    "        if pieces[\"branch\"] != \"master\":\n"
    "            rendered += \".dev0\"\n"
    "        rendered += \"+untagged.%d.g%s\" % (pieces[\"distance\"], pieces[\"short\"])\n"
    "        if pieces[\"dirty\"]:\n"
    "            rendered += \".dirty\"\n"
    "    return rendered\n\n\n"
    "def pep440_split_post(ver: str) -> Tuple[str, Optional[int]]:\n"
    "    vc = str.split(ver, \".post\")\n"
    "    return vc[0], int(vc[1] or 0) if len(vc) == 2 else None\n\n\n"
    "def render_pep440_pre(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces[\"closest-tag\"]:\n"
    "        if pieces[\"distance\"]:\n"
    "            tag_version, post_version = pep440_split_post(pieces[\"closest-tag\"])\n"
    "            rendered = tag_version\n"
    "            if post_version is not None:\n"
    "                rendered += \".post%d.dev%d\" % (post_version + 1, pieces[\"distance\"])\n"
    "            else:\n"
    "                rendered += \".post0.dev%d\" % pieces[\"distance\"]\n"
    "        else:\n"
    "            rendered = pieces[\"closest-tag\"]\n"
    "    else:\n"
    "        rendered = \"0.post0.dev%d\" % pieces[\"distance\"]\n"
    "    return rendered\n\n\n"
    "def render_pep440_post(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces[\"closest-tag\"]:\n"
    "        rendered = pieces[\"closest-tag\"]\n"
    "        if pieces[\"distance\"] or pieces[\"dirty\"]:\n"
    "            rendered += \".post%d\" % pieces[\"distance\"]\n"
    "            if pieces[\"dirty\"]:\n"
    "                rendered += \".dev0\"\n"
    "            rendered += plus_or_dot(pieces)\n"
    "            rendered += \"g%s\" % pieces[\"short\"]\n"
    "    else:\n"
    "        rendered = \"0.post%d\" % pieces[\"distance\"]\n"
    "        if pieces[\"dirty\"]:\n"
    "            rendered += \".dev0\"\n"
    "        rendered += \"+g%s\" % pieces[\"short\"]\n"
    "    return rendered\n\n\n"
    "def render_pep440_post_branch(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces[\"closest-tag\"]:\n"
    "        rendered = pieces[\"closest-tag\"]\n"
    "        if pieces[\"distance\"] or pieces[\"dirty\"]:\n"
    "            rendered += \".post%d\" % pieces[\"distance\"]\n"
    "            if pieces[\"branch\"] != \"master\":\n"
    "                rendered += \".dev0\"\n"
    "            rendered += plus_or_dot(pieces)\n"
    "            rendered += \"g%s\" % pieces[\"short\"]\n"
    "            if pieces[\"dirty\"]:\n"
    "                rendered += \".dirty\"\n"
    "    else:\n"
    "        rendered = \"0.post%d\" % pieces[\"distance\"]\n"
    "        if pieces[\"branch\"] != \"master\":\n"
    "            rendered += \".dev0\"\n"
    "        rendered += \"+g%s\" % pieces[\"short\"]\n"
    "        if pieces[\"dirty\"]:\n"
    "            rendered += \".dirty\"\n"
    "    return rendered\n\n\n"
    "def render_pep440_old(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces[\"closest-tag\"]:\n"
    "        rendered = pieces[\"closest-tag\"]\n"
    "        if pieces[\"distance\"] or pieces[\"dirty\"]:\n"
    "            rendered += \".post%d\" % pieces[\"distance\"]\n"
    "            if pieces[\"dirty\"]:\n"
    "                rendered += \".dev0\"\n"
    "    else:\n"
    "        rendered = \"0.post%d\" % pieces[\"distance\"]\n"
    "        if pieces[\"dirty\"]:\n"
    "            rendered += \".dev0\"\n"
    "    return rendered\n\n\n"
    "def render_git_describe(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces[\"closest-tag\"]:\n"
    "        rendered = pieces[\"closest-tag\"]\n"
    "        if pieces[\"distance\"]:\n"
    "            rendered += \"-%d-g%s\" % (pieces[\"distance\"], pieces[\"short\"])\n"
    "    else:\n"
    "        rendered = pieces[\"short\"]\n"
    "    if pieces[\"dirty\"]:\n"
    "        rendered += \"-dirty\"\n"
    "    return rendered\n\n\n"
    "def render_git_describe_long(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces[\"closest-tag\"]:\n"
    "        rendered = pieces[\"closest-tag\"]\n"
    "        rendered += \"-%d-g%s\" % (pieces[\"distance\"], pieces[\"short\"])\n"
    "    else:\n"
    "        rendered = pieces[\"short\"]\n"
    "    if pieces[\"dirty\"]:\n"
    "        rendered += \"-dirty\"\n"
    "    return rendered\n\n\n"
    "def render(pieces: Dict[str, Any], style: str) -> Dict[str, Any]:\n"
    "    if pieces[\"error\"]:\n"
    "        return {\"version\": \"unknown\",\n"
    "                \"full-revisionid\": pieces.get(\"long\"),\n"
    "                \"dirty\": None,\n"
    "                \"error\": pieces[\"error\"],\n"
    "                \"date\": None}\n\n"
    "    if not style or style == \"default\":\n"
    "        style = \"pep440\"\n\n"
    "    if style == \"pep440\":\n"
    "        rendered = render_pep440(pieces)\n"
    "    elif style == \"pep440-branch\":\n"
    "        rendered = render_pep440_branch(pieces)\n"
    "    elif style == \"pep440-pre\":\n"
    "        rendered = render_pep440_pre(pieces)\n"
    "    elif style == \"pep440-post\":\n"
    "        rendered = render_pep440_post(pieces)\n"
    "    elif style == \"pep440-post-branch\":\n"
    "        rendered = render_pep440_post_branch(pieces)\n"
    "    elif style == \"pep440-old\":\n"
    "        rendered = render_pep440_old(pieces)\n"
    "    elif style == \"git-describe\":\n"
    "        rendered = render_git_describe(pieces)\n"
    "    elif style == \"git-describe-long\":\n"
    "        rendered = render_git_describe_long(pieces)\n"
    "    else:\n"
    "        raise ValueError(\"unknown style '%s'\" % style)\n\n"
    "    return {\"version\": rendered, \"full-revisionid\": pieces[\"long\"],\n"
    "            \"dirty\": pieces[\"dirty\"], \"error\": None,\n"
    "            \"date\": pieces.get(\"date\")}\n\n\n"
    "def get_versions() -> Dict[str, Any]:\n"
    "    cfg = get_config()\n"
    "    verbose = cfg.verbose\n\n"
    "    try:\n"
    "        return git_versions_from_keywords(get_keywords(), cfg.tag_prefix, verbose)\n"
    "    except NotThisMethod:\n"
    "        pass\n\n"
    "    try:\n"
    "        root = os.path.realpath(__file__)\n"
    "        for _ in cfg.versionfile_source.split('/'):\n"
    "            root = os.path.dirname(root)\n"
    "    except NameError:\n"
    "        return {\"version\": \"0+unknown\", \"full-revisionid\": None,\n"
    "                \"dirty\": None,\n"
    "                \"error\": \"unable to find root of source tree\",\n"
    "                \"date\": None}\n\n"
    "    try:\n"
    "        pieces = git_pieces_from_vcs(cfg.tag_prefix, root, verbose)\n"
    "        return render(pieces, cfg.style)\n"
    "    except NotThisMethod:\n"
    "        pass\n\n"
    "    try:\n"
    "        if cfg.parentdir_prefix:\n"
    "            return versions_from_parentdir(cfg.parentdir_prefix, root, verbose)\n"
    "    except NotThisMethod:\n"
    "        pass\n\n"
    "    return {\"version\": \"0+unknown\", \"full-revisionid\": None,\n"
    "            \"dirty\": None,\n"
    "            \"error\": \"unable to compute version\", \"date\": None}\n\n\n"
    "def get_version() -> str:\n"
    "    return get_versions()['version']\n\n\n"
    "def get_cmdclass(cmdclass: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:\n"
    "    if 'versioneer' in sys.modules:\n"
    "        del sys.modules['versioneer']\n"
    "    cmds: Dict[str, Any] = {} if cmdclass is None else cmdclass.copy()\n"
    "    from setuptools import Command\n\n"
    "    class cmd_version(Command):\n"
    "        description: str = 'report generated version string'\n"
    "        user_options: List[Any] = []\n"
    "        boolean_options: List[Any] = []\n\n"
    "        def initialize_options(self) -> None:\n"
    "            pass\n\n"
    "        def finalize_options(self) -> None:\n"
    "            pass\n\n"
    "        def run(self) -> None:\n"
    "            vers: Dict[str, Any] = get_versions(verbose=True)\n"
    "            print('Version: %s' % vers['version'])\n"
    "            print(' full-revisionid: %s' % vers.get('full-revisionid'))\n"
    "            print(' dirty: %s' % vers.get('dirty'))\n"
    "            print(' date: %s' % vers.get('date'))\n"
    "            if vers['error']:\n"
    "                print(' error: %s' % vers['error'])\n\n"
    "    cmds['version'] = cmd_version\n"
    "    if 'build_py' in cmds:\n"
    "        _build_py = cmds['build_py']\n"
    "    else:\n"
    "        from setuptools.command.build_py import build_py as _build_py\n\n"
    "    class cmd_build_py(_build_py):\n\n"
    "        def run(self) -> None:\n"
    "            root = get_root()\n"
    "            cfg = get_config_from_root(root)\n"
    "            versions = get_versions()\n"
    "            _build_py.run(self)\n"
    "            if getattr(self, 'editable_mode', False):\n"
    "                return\n"
    "            if cfg.versionfile_build:\n"
    "                target_versionfile: str = os.path.join(self.build_lib, cfg.versionfile_build)\n"
    "                print('UPDATING %s' % target_versionfile)\n"
    "                write_to_version_file(target_versionfile, versions)\n\n"
    "    cmds['build_py'] = cmd_build_py\n"
    "    if 'build_ext' in cmds:\n"
    "        _build_ext = cmds['build_ext']\n"
    "    else:\n"
    "        from setuptools.command.build_ext import build_ext as _build_ext\n\n"
    "    class cmd_build_ext(_build_ext):\n\n"
    "        def run(self) -> None:\n"
    "            root = get_root()\n"
    "            cfg = get_config_from_root(root)\n"
    "            versions = get_versions()\n"
    "            _build_ext.run(self)\n"
    "            if self.inplace:\n"
    "                return\n"
    "            if not cfg.versionfile_build:\n"
    "                return\n"
    "            target_versionfile: str = os.path.join(self.build_lib, cfg.versionfile_build)\n"
    "            if not os.path.exists(target_versionfile):\n"
    "                print(f'Warning: {target_versionfile} does not exist, skipping version update. This can happen if you are running build_ext without first running build_py.')\n"
    "                return\n"
    "            print('UPDATING %s' % target_versionfile)\n"
    "            write_to_version_file(target_versionfile, versions)\n\n"
    "    cmds['build_ext'] = cmd_build_ext\n"
    "    if 'cx_Freeze' in sys.modules:\n"
    "        from cx_Freeze.dist import build_exe as _build_exe\n\n"
    "        class cmd_build_exe(_build_exe):\n\n"
    "            def run(self) -> None:\n"
    "                root = get_root()\n"
    "                cfg = get_config_from_root(root)\n"
    "                versions = get_versions()\n"
    "                target_versionfile: str = cfg.versionfile_source\n"
    "                print('UPDATING %s' % target_versionfile)\n"
    "                write_to_version_file(target_versionfile, versions)\n"
    "                _build_exe.run(self)\n"
    "                os.unlink(target_versionfile)\n"
    "                with open(cfg.versionfile_source, 'w') as f:\n"
    "                    LONG = LONG_VERSION_PY[cfg.VCS]\n"
    "                    f.write(LONG % {'DOLLAR': '$', 'STYLE': cfg.style, 'TAG_PREFIX': cfg.tag_prefix, 'PARENTDIR_PREFIX': cfg.parentdir_prefix, 'VERSIONFILE_SOURCE': cfg.versionfile_source})\n"
    "        cmds['build_exe'] = cmd_build_exe\n"
    "        del cmds['build_py']\n"
    "    if 'py2exe' in sys.modules:\n"
    "        try:\n"
    "            from py2exe.setuptools_buildexe import py2exe as _py2exe\n"
    "        except ImportError:\n"
    "            from py2exe.distutils_buildexe import py2exe as _py2exe\n\n"
    "        class cmd_py2exe(_py2exe):\n\n"
    "            def run(self) -> None:\n"
    "                root = get_root()\n"
    "                cfg = get_config_from_root(root)\n"
    "                versions = get_versions()\n"
    "                target_versionfile: str = cfg.versionfile_source\n"
    "                print('UPDATING %s' % target_versionfile)\n"
    "                write_to_version_file(target_versionfile, versions)\n"
    "                _py2exe.run(self)\n"
    "                os.unlink(target_versionfile)\n"
    "                with open(cfg.versionfile_source, 'w') as f:\n"
    "                    LONG = LONG_VERSION_PY[cfg.VCS]\n"
    "                    f.write(LONG % {'DOLLAR': '$', 'STYLE': cfg.style, 'TAG_PREFIX': cfg.tag_prefix, 'PARENTDIR_PREFIX': cfg.parentdir_prefix, 'VERSIONFILE_SOURCE': cfg.versionfile_source})\n"
    "        cmds['py2exe'] = cmd_py2exe\n"
    "    if 'egg_info' in cmds:\n"
    "        _egg_info = cmds['egg_info']\n"
    "    else:\n"
    "        from setuptools.command.egg_info import egg_info as _egg_info\n\n"
    "    class cmd_egg_info(_egg_info):\n\n"
    "        def find_sources(self) -> None:\n"
    "            super().find_sources()\n"
    "            root = get_root()\n"
    "            cfg = get_config_from_root(root)\n"
    "            self.filelist.append('versioneer.py')\n"
    "            if cfg.versionfile_source:\n"
    "                self.filelist.append(cfg.versionfile_source)\n"
    "            self.filelist.sort()\n"
    "            self.filelist.remove_duplicates()\n"
    "            from setuptools import unicode_utils\n"
    "            normalized = [unicode_utils.filesys_decode(f).replace(os.sep, '/') for f in self.filelist.files]\n"
    "            manifest_filename: str = os.path.join(self.egg_info, 'SOURCES.txt')\n"
    "            with open(manifest_filename, 'w') as fobj:\n"
    "                fobj.write('\\n'.join(normalized))\n\n"
    "    cmds['egg_info'] = cmd_egg_info\n"
    "    if 'sdist' in cmds:\n"
    "        _sdist = cmds['sdist']\n"
    "    else:\n"
    "        from setuptools.command.sdist import sdist as _sdist\n\n"
    "    class cmd_sdist(_sdist):\n\n"
    "        def run(self) -> Any:\n"
    "            versions = get_versions()\n"
    "            self._versioneer_generated_versions = versions\n"
    "            self.distribution.metadata.version = versions['version']\n"
    "            return _sdist.run(self)\n\n"
    "        def make_release_tree(self, base_dir: str, files: List[str]) -> None:\n"
    "            root = get_root()\n"
    "            cfg = get_config_from_root(root)\n"
    "            _sdist.make_release_tree(self, base_dir, files)\n"
    "            target_versionfile: str = os.path.join(base_dir, cfg.versionfile_source)\n"
    "            print('UPDATING %s' % target_versionfile)\n"
    "            write_to_version_file(target_versionfile, self._versioneer_generated_versions)\n\n"
    "    cmds['sdist'] = cmd_sdist\n"
    "    return cmds\n\n\n"
    "CONFIG_ERROR = \"\\nsetup.cfg is missing the necessary Versioneer configuration. You need\\n"
    "a section like:\\n\\n [versioneer]\\n VCS = git\\n style = pep440\\n versionfile_source = src/myproject/_version.py\\n versionfile_build = myproject/_version.py\\n tag_prefix =\\n parentdir_prefix = myproject-\\n\\n"
    "You will also need to edit your setup.py to use the results:\\n\\n import versioneer\\n setup(version=versioneer.get_version(),\\n       cmdclass=versioneer.get_cmdclass(), ...)\\n\\n"
    "Please read the docstring in ./versioneer.py for configuration instructions,\\nedit setup.cfg, and re-run the installer or 'python versioneer.py setup'.\\n\"\n"
    "SAMPLE_CONFIG = \"\\n# See the docstring in versioneer.py for instructions. Note that you must\\n"
    "# re-run 'versioneer.py setup' after changing this section, and commit the\\n"
    "# resulting files.\\n\\n[versioneer]\\n#VCS = git\\n#style = pep440\\n#versionfile_source =\\n#versionfile_build =\\n#tag_prefix =\\n#parentdir_prefix =\\n\\n\"\n"
    "OLD_SNIPPET = \"\\nfrom ._version import get_versions\\n__version__ = get_versions()['version']\\ndel get_versions\\n\"\n"
    "INIT_PY_SNIPPET = \"\\nfrom . import {0}\\n__version__ = {0}.get_versions()['version']\\n\"\n\n\n"
    "def do_vcs_install(versionfile_source: str, ipy: Optional[str]) -> None:\n"
    "    GITS: List[str] = ['git']\n"
    "    if sys.platform == 'win32':\n"
    "        GITS = ['git.cmd', 'git.exe']\n"
    "    files: List[str] = [versionfile_source]\n"
    "    if ipy:\n"
    "        files.append(ipy)\n"
    "    if 'VERSIONEER_PEP518' not in globals():\n"
    "        try:\n"
    "            my_path: str = __file__\n"
    "            if my_path.endswith(('.pyc', '.pyo')):\n"
    "                my_path = os.path.splitext(my_path)[0] + '.py'\n"
    "            versioneer_file: str = os.path.relpath(my_path)\n"
    "        except NameError:\n"
    "            versioneer_file = 'versioneer.py'\n"
    "        files.append(versioneer_file)\n"
    "    present: bool = False\n"
    "    try:\n"
    "        with open('.gitattributes', 'r') as fobj:\n"
    "            for line in fobj:\n"
    "                if line.strip().startswith(versionfile_source):\n"
    "                    if 'export-subst' in line.strip().split()[1:]:\n"
    "                        present = True\n"
    "                        break\n"
    "    except OSError:\n"
    "        pass\n"
    "    if not present:\n"
    "        with open('.gitattributes', 'a+') as fobj:\n"
    "            fobj.write(f'{versionfile_source} export-subst\\n')\n"
    "        files.append('.gitattributes')\n"
    "    run_command(GITS, ['add', '--'] + files)\n\n\n"
    "def versions_from_parentdir(parentdir_prefix: str, root: str, verbose: bool) -> Dict[str, Any]:\n"
    "    rootdirs = []\n"
    "    for _ in range(3):\n"
    "        dirname = os.path.basename(root)\n"
    "        if dirname.startswith(parentdir_prefix):\n"
    "            return {'version': dirname[len(parentdir_prefix):], 'full-revisionid': None, 'dirty': False, 'error': None, 'date': None}\n"
    "        rootdirs.append(root)\n"
    "        root = os.path.dirname(root)\n"
    "    if verbose:\n"
    "        print('Tried directories %s but none started with prefix %s' % (str(rootdirs), parentdir_prefix))\n"
    "    raise NotThisMethod(\"rootdir doesn't start with parentdir_prefix\")\n\n\n"
    "SHORT_VERSION_PY = \"\\n# This file was generated by 'versioneer.py' (0.29) from\\n"
    "# revision-control system data, or from the parent directory name of an\\n"
    "# unpacked source archive. Distribution tarballs contain a pre-generated copy\\n"
    "# of this file.\\n\\nimport json\\n\\nversion_json = '''\\n%s\\n'''  # END VERSION_JSON\\n\\n\\ndef get_versions():\\n    return json.loads(version_json)\\n\"\n\n\n"
    "def versions_from_file(filename: str) -> Dict[str, Any]:\n"
    "    try:\n"
    "        with open(filename) as f:\n"
    "            contents: str = f.read()\n"
    "    except OSError:\n"
    "        raise NotThisMethod('unable to read _version.py')\n"
    "    mo = re.search(\"version_json = '''\\\\n(.*)'''  # END VERSION_JSON\", contents, re.M | re.S)\n"
    "    if not mo:\n"
    "        mo = re.search(\"version_json = '''\\\\r\\\\n(.*)'''  # END VERSION_JSON\", contents, re.M | re.S)\n"
    "    if not mo:\n"
    "        raise NotThisMethod('no version_json in _version.py')\n"
    "    return json.loads(mo.group(1))\n\n\n"
    "def write_to_version_file(filename: str, versions: Dict[str, Any]) -> None:\n"
    "    contents: str = json.dumps(versions, sort_keys=True, indent=1, separators=(',', ': '))\n"
    "    with open(filename, 'w') as f:\n"
    "        f.write(SHORT_VERSION_PY % contents)\n"
    "    print(\"set %s to '%s'\" % (filename, versions['version']))\n\n\n"
    "def plus_or_dot(pieces: Dict[str, Any]) -> str:\n"
    "    if '+' in pieces.get('closest-tag', ''):\n"
    "        return '.'\n"
    "    return '+'\n\n\n"
    "def render_pep440(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces['closest-tag']:\n"
    "        rendered = pieces['closest-tag']\n"
    "        if pieces['distance'] or pieces['dirty']:\n"
    "            rendered += plus_or_dot(pieces)\n"
    "            rendered += '%d.g%s' % (pieces['distance'], pieces['short'])\n"
    "    else:\n"
    "        rendered = '0+untagged.%d.g%s' % (pieces['distance'], pieces['short'])\n"
    "        if pieces['dirty']:\n"
    "            rendered += '.dirty'\n"
    "    return rendered\n\n\n"
    "def render_pep440_branch(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces['closest-tag']:\n"
    "        rendered = pieces['closest-tag']\n"
    "        if pieces['distance'] or pieces['dirty']:\n"
    "            if pieces['branch'] != 'master':\n"
    "                rendered += '.dev0'\n"
    "            rendered += plus_or_dot(pieces)\n"
    "            rendered += '%d.g%s' % (pieces['distance'], pieces['short'])\n"
    "            if pieces['dirty']:\n"
    "                rendered += '.dirty'\n"
    "    else:\n"
    "        rendered = '0'\n"
    "        if pieces['branch'] != 'master':\n"
    "            rendered += '.dev0'\n"
    "        rendered += '+untagged.%d.g%s' % (pieces['distance'], pieces['short'])\n"
    "        if pieces['dirty']:\n"
    "            rendered += '.dirty'\n"
    "    return rendered\n\n\n"
    "def pep440_split_post(ver: str) -> Tuple[str, Optional[int]]:\n"
    "    vc = str.split(ver, '.post')\n"
    "    return vc[0], int(vc[1] or 0) if len(vc) == 2 else None\n\n\n"
    "def render_pep440_pre(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces['closest-tag']:\n"
    "        if pieces['distance']:\n"
    "            tag_version, post_version = pep440_split_post(pieces['closest-tag'])\n"
    "            rendered = tag_version\n"
    "            if post_version is not None:\n"
    "                rendered += '.post%d.dev%d' % (post_version + 1, pieces['distance'])\n"
    "            else:\n"
    "                rendered += '.post0.dev%d' % pieces['distance']\n"
    "        else:\n"
    "            rendered = pieces['closest-tag']\n"
    "    else:\n"
    "        rendered = '0.post0.dev%d' % pieces['distance']\n"
    "    return rendered\n\n\n"
    "def render_pep440_post(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces['closest-tag']:\n"
    "        rendered = pieces['closest-tag']\n"
    "        if pieces['distance'] or pieces['dirty']:\n"
    "            rendered += '.post%d' % pieces['distance']\n"
    "            if pieces['dirty']:\n"
    "                rendered += '.dev0'\n"
    "            rendered += plus_or_dot(pieces)\n"
    "            rendered += 'g%s' % pieces['short']\n"
    "    else:\n"
    "        rendered = '0.post%d' % pieces['distance']\n"
    "        if pieces['dirty']:\n"
    "            rendered += '.dev0'\n"
    "        rendered += '+g%s' % pieces['short']\n"
    "    return rendered\n\n\n"
    "def render_pep440_post_branch(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces['closest-tag']:\n"
    "        rendered = pieces['closest-tag']\n"
    "        if pieces['distance'] or pieces['dirty']:\n"
    "            rendered += '.post%d' % pieces['distance']\n"
    "            if pieces['branch'] != 'master':\n"
    "                rendered += '.dev0'\n"
    "            rendered += plus_or_dot(pieces)\n"
    "            rendered += 'g%s' % pieces['short']\n"
    "            if pieces['dirty']:\n"
    "                rendered += '.dirty'\n"
    "    else:\n"
    "        rendered = '0.post%d' % pieces['distance']\n"
    "        if pieces['branch'] != 'master':\n"
    "            rendered += '.dev0'\n"
    "        rendered += '+g%s' % pieces['short']\n"
    "        if pieces['dirty']:\n"
    "            rendered += '.dirty'\n"
    "    return rendered\n\n\n"
    "def render_pep440_old(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces['closest-tag']:\n"
    "        rendered = pieces['closest-tag']\n"
    "        if pieces['distance'] or pieces['dirty']:\n"
    "            rendered += '.post%d' % pieces['distance']\n"
    "            if pieces['dirty']:\n"
    "                rendered += '.dev0'\n"
    "    else:\n"
    "        rendered = '0.post%d' % pieces['distance']\n"
    "        if pieces['dirty']:\n"
    "            rendered += '.dev0'\n"
    "    return rendered\n\n\n"
    "def render_git_describe(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces['closest-tag']:\n"
    "        rendered = pieces['closest-tag']\n"
    "        if pieces['distance']:\n"
    "            rendered += '-%d-g%s' % (pieces['distance'], pieces['short'])\n"
    "    else:\n"
    "        rendered = pieces['short']\n"
    "    if pieces['dirty']:\n"
    "        rendered += '-dirty'\n"
    "    return rendered\n\n\n"
    "def render_git_describe_long(pieces: Dict[str, Any]) -> str:\n"
    "    if pieces['closest-tag']:\n"
    "        rendered = pieces['closest-tag']\n"
    "        rendered += '-%d-g%s' % (pieces['distance'], pieces['short'])\n"
    "    else:\n"
    "        rendered = pieces['short']\n"
    "    if pieces['dirty']:\n"
    "        rendered += '-dirty'\n"
    "    return rendered\n\n\n"
    "def render(pieces: Dict[str, Any], style: str) -> Dict[str, Any]:\n"
    "    if pieces['error']:\n"
    "        return {'version': 'unknown',\n"
    "                'full-revisionid': pieces.get('long'),\n"
    "                'dirty': None,\n"
    "                'error': pieces['error'],\n"
    "                'date': None}\n\n"
    "    if not style or style == 'default':\n"
    "        style = 'pep440'\n\n"
    "    if style == 'pep440':\n"
    "        rendered = render_pep440(pieces)\n"
    "    elif style == 'pep440-branch':\n"
    "        rendered = render_pep440_branch(pieces)\n"
    "    elif style == 'pep440-pre':\n"
    "        rendered = render_pep440_pre(pieces)\n"
    "    elif style == 'pep440-post':\n"
    "        rendered = render_pep440_post(pieces)\n"
    "    elif style == 'pep440-post-branch':\n"
    "        rendered = render_pep440_post_branch(pieces)\n"
    "    elif style == 'pep440-old':\n"
    "        rendered = render_pep440_old(pieces)\n"
    "    elif style == 'git-describe':\n"
    "        rendered = render_git_describe(pieces)\n"
    "    elif style == 'git-describe-long':\n"
    "        rendered = render_git_describe_long(pieces)\n"
    "    else:\n"
    "        raise ValueError(\"unknown style '%s'\" % style)\n\n"
    "    return {'version': rendered, 'full-revisionid': pieces['long'],\n"
    "            'dirty': pieces['dirty'], 'error': None,\n"
    "            'date': pieces.get('date')}\n"
)
    
@register_vcs_handler('git', 'get_keywords')
def git_get_keywords(versionfile_abs: str) -> Dict[str, str]:
    """Extract version information from the given file."""
    keywords: Dict[str, str] = {}
    try:
        with open(versionfile_abs, 'r') as fobj:
            for line in fobj:
                if line.strip().startswith('git_refnames ='):
                    mo = re.search('=\\s*"(.*)"', line)
                    if mo:
                        keywords['refnames'] = mo.group(1)
                if line.strip().startswith('git_full ='):
                    mo = re.search('=\\s*"(.*)"', line)
                    if mo:
                        keywords['full'] = mo.group(1)
                if line.strip().startswith('git_date ='):
                    mo = re.search('=\\s*"(.*)"', line)
                    if mo:
                        keywords['date'] = mo.group(1)
    except OSError:
        pass
    return keywords

@register_vcs_handler('git', 'keywords')
def git_versions_from_keywords(keywords: Dict[str, str], tag_prefix: str, verbose: bool) -> Dict[str, Any]:
    """Get version information from git keywords."""
    if 'refnames' not in keywords:
        raise NotThisMethod('Short version file found')
    date: Optional[str] = keywords.get('date')
    if date is not None:
        date = date.splitlines()[-1]
        date = date.strip().replace(' ', 'T', 1).replace(' ', '', 1)
    refnames: str = keywords['refnames'].strip()
    if refnames.startswith('$Format'):
        if verbose:
            print('keywords are unexpanded, not using')
        raise NotThisMethod('unexpanded keywords, not a git-archive tarball')
    refs: set = {r.strip() for r in refnames.strip('()').split(',')}
    TAG: str = 'tag: '
    tags: set = {r[len(TAG):] for r in refs if r.startswith(TAG)}
    if not tags:
        tags = {r for r in refs if re.search('\\d', r)}
        if verbose:
            print("discarding '%s', no digits" % ','.join(refs - tags))
    if verbose:
        print('likely tags: %s' % ','.join(sorted(tags)))
    for ref in sorted(tags):
        if ref.startswith(tag_prefix):
            r: str = ref[len(tag_prefix):]
            if not re.match('\\d', r):
                continue
            if verbose:
                print('picking %s' % r)
            return {'version': r,
                    'full-revisionid': keywords['full'].strip(),
                    'dirty': False, 'error': None, 'date': date}
    if verbose:
        print('no suitable tags, using unknown + full revision id')
    return {'version': '0+unknown',
            'full-revisionid': keywords['full'].strip(),
            'dirty': False, 'error': 'no suitable tags', 'date': None}

@register_vcs_handler('git', 'pieces_from_vcs')
def git_pieces_from_vcs(tag_prefix: str, root: str, verbose: bool, runner: Callable[..., Any] = run_command) -> Dict[str, Any]:
    """Get version from 'git describe' in the root of the source tree.

    This only gets called if the git-archive 'subst' keywords were *not*
    expanded, and _version.py hasn't already been rewritten with a short
    version string, meaning we're inside a checked out source tree.
    """
    GITS: List[str] = ['git']
    if sys.platform == 'win32':
        GITS = ['git.cmd', 'git.exe']
    env: Dict[str, str] = os.environ.copy()
    env.pop('GIT_DIR', None)
    runner = functools.partial(runner, env=env)
    _, rc = runner(GITS, ['rev-parse', '--git-dir'], cwd=root, hide_stderr=not verbose)
    if rc != 0:
        if verbose:
            print('Directory %s not under git control' % root)
        raise NotThisMethod("'git rev-parse --git-dir' returned error")
    describe_out, rc = runner(GITS, [
        "describe", "--tags", "--dirty", "--always", "--long",
        "--match", f"{tag_prefix}[[:digit:]]*"
    ], cwd=root)
    if describe_out is None:
        raise NotThisMethod("'git describe' failed")
    describe_out = describe_out.strip()
    full_out, rc = runner(GITS, ['rev-parse', 'HEAD'], cwd=root)
    if full_out is None:
        raise NotThisMethod("'git rev-parse' failed")
    full_out = full_out.strip()
    pieces: Dict[str, Any] = {}
    pieces["long"] = full_out
    pieces["short"] = full_out[:7]
    pieces["error"] = None
    branch_name, rc = runner(GITS, ['rev-parse', '--abbrev-ref', 'HEAD'], cwd=root)
    if rc != 0 or branch_name is None:
        raise NotThisMethod("'git rev-parse --abbrev-ref' returned error")
    branch_name = branch_name.strip()
    if branch_name == 'HEAD':
        branches, rc = runner(GITS, ['branch', '--contains'], cwd=root)
        if rc != 0 or branches is None:
            raise NotThisMethod("'git branch --contains' returned error")
        branches = branches.split('\n')
        if '(' in branches[0]:
            branches.pop(0)
        branches = [branch[2:] for branch in branches]
        if 'master' in branches:
            branch_name = 'master'
        elif not branches:
            branch_name = None
        else:
            branch_name = branches[0]
    pieces["branch"] = branch_name
    git_describe: str = describe_out
    dirty: bool = git_describe.endswith('-dirty')
    pieces["dirty"] = dirty
    if dirty:
        git_describe = git_describe[:git_describe.rindex('-dirty')]
    if "-" in git_describe:
        mo = re.search('^(.+)-(\\d+)-g([0-9a-f]+)$', git_describe)
        if not mo:
            pieces["error"] = "unable to parse git-describe output: '%s'" % describe_out
            return pieces
        full_tag: str = mo.group(1)
        if not full_tag.startswith(tag_prefix):
            if verbose:
                fmt: str = "tag '%s' doesn't start with prefix '%s'"
                print(fmt % (full_tag, tag_prefix))
            pieces["error"] = "tag '%s' doesn't start with prefix '%s'" % (full_tag, tag_prefix)
            return pieces
        pieces["closest-tag"] = full_tag[len(tag_prefix):]
        pieces["distance"] = int(mo.group(2))
        pieces["short"] = mo.group(3)
    else:
        pieces["closest-tag"] = None
        out, rc = runner(GITS, ['rev-list', 'HEAD', '--left-right'], cwd=root)
        pieces["distance"] = len(out.split())
    date: str = runner(GITS, ['show', '-s', '--format=%ci', 'HEAD'], cwd=root)[0].strip()
    date = date.splitlines()[-1]
    pieces["date"] = date.strip().replace(" ", "T", 1).replace(" ", "", 1)
    return pieces

def do_vcs_install(versionfile_source: str, ipy: Optional[str]) -> None:
    """Git-specific installation logic for Versioneer."""
    GITS: List[str] = ['git']
    if sys.platform == 'win32':
        GITS = ['git.cmd', 'git.exe']
    files: List[str] = [versionfile_source]
    if ipy:
        files.append(ipy)
    if 'VERSIONEER_PEP518' not in globals():
        try:
            my_path: str = __file__
            if my_path.endswith(('.pyc', '.pyo')):
                my_path = os.path.splitext(my_path)[0] + '.py'
            versioneer_file: str = os.path.relpath(my_path)
        except NameError:
            versioneer_file = 'versioneer.py'
        files.append(versioneer_file)
    present: bool = False
    try:
        with open('.gitattributes', 'r') as fobj:
            for line in fobj:
                if line.strip().startswith(versionfile_source):
                    if 'export-subst' in line.strip().split()[1:]:
                        present = True
                        break
    except OSError:
        pass
    if not present:
        with open('.gitattributes', 'a+') as fobj:
            fobj.write(f'{versionfile_source} export-subst\n')
        files.append('.gitattributes')
    run_command(GITS, ['add', '--'] + files)

def versions_from_parentdir(parentdir_prefix: str, root: str, verbose: bool) -> Dict[str, Any]:
    """Try to determine the version from the parent directory name."""
    rootdirs: List[str] = []
    for _ in range(3):
        dirname: str = os.path.basename(root)
        if dirname.startswith(parentdir_prefix):
            return {'version': dirname[len(parentdir_prefix):], 'full-revisionid': None, 'dirty': False, 'error': None, 'date': None}
        rootdirs.append(root)
        root = os.path.dirname(root)
    if verbose:
        print('Tried directories %s but none started with prefix %s' % (str(rootdirs), parentdir_prefix))
    raise NotThisMethod("rootdir doesn't start with parentdir_prefix")

SHORT_VERSION_PY: str = (
    "\n# This file was generated by 'versioneer.py' (0.29) from\n"
    "# revision-control system data, or from the parent directory name of an\n"
    "# unpacked source archive. Distribution tarballs contain a pre-generated copy\n"
    "# of this file.\n\nimport json\n\nversion_json = '''\n%s\n'''  # END VERSION_JSON\n\n\ndef get_versions():\n    return json.loads(version_json)\n"
)

def versions_from_file(filename: str) -> Dict[str, Any]:
    """Try to determine the version from _version.py if present."""
    try:
        with open(filename) as f:
            contents: str = f.read()
    except OSError:
        raise NotThisMethod('unable to read _version.py')
    mo = re.search("version_json = '''\\n(.*)'''  # END VERSION_JSON", contents, re.M | re.S)
    if not mo:
        mo = re.search("version_json = '''\\r\\n(.*)'''  # END VERSION_JSON", contents, re.M | re.S)
    if not mo:
        raise NotThisMethod('no version_json in _version.py')
    return json.loads(mo.group(1))

def write_to_version_file(filename: str, versions: Dict[str, Any]) -> None:
    """Write the given version number to the given _version.py file."""
    contents: str = json.dumps(versions, sort_keys=True, indent=1, separators=(',', ': '))
    with open(filename, 'w') as f:
        f.write(SHORT_VERSION_PY % contents)
    print("set %s to '%s'" % (filename, versions['version']))

def plus_or_dot(pieces: Dict[str, Any]) -> str:
    """Return a + if we don't already have one, else return a ."""
    if '+' in pieces.get('closest-tag', ''):
        return '.'
    return '+'

def render_pep440(pieces: Dict[str, Any]) -> str:
    """Build up version string, with post-release local version identifier."""
    if pieces['closest-tag']:
        rendered: str = pieces['closest-tag']
        if pieces['distance']:
            rendered += plus_or_dot(pieces)
            rendered += '%d.g%s' % (pieces['distance'], pieces['short'])
    else:
        rendered = '0+untagged.%d.g%s' % (pieces['distance'], pieces['short'])
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered

def render_pep440_branch(pieces: Dict[str, Any]) -> str:
    """Build up version string for branch versions."""
    if pieces['closest-tag']:
        rendered: str = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            if pieces['branch'] != 'master':
                rendered += '.dev0'
            rendered += plus_or_dot(pieces)
            rendered += '%d.g%s' % (pieces['distance'], pieces['short'])
            if pieces['dirty']:
                rendered += '.dirty'
    else:
        rendered = '0'
        if pieces['branch'] != 'master':
            rendered += '.dev0'
        rendered += '+untagged.%d.g%s' % (pieces['distance'], pieces['short'])
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered

def pep440_split_post(ver: str) -> Tuple[str, Optional[int]]:
    """Split pep440 version string at the post-release segment."""
    vc = str.split(ver, ".post")
    return vc[0], int(vc[1] or 0) if len(vc) == 2 else None

def render_pep440_pre(pieces: Dict[str, Any]) -> str:
    """Build up pre-release version string."""
    if pieces['closest-tag']:
        if pieces['distance']:
            tag_version, post_version = pep440_split_post(pieces['closest-tag'])
            rendered: str = tag_version
            if post_version is not None:
                rendered += ".post%d.dev%d" % (post_version + 1, pieces['distance'])
            else:
                rendered += ".post0.dev%d" % pieces['distance']
        else:
            rendered = pieces['closest-tag']
    else:
        rendered = "0.post0.dev%d" % pieces['distance']
    return rendered

def render_pep440_post(pieces: Dict[str, Any]) -> str:
    """Build up post-release version string."""
    if pieces['closest-tag']:
        rendered: str = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += ".post%d" % pieces['distance']
            if pieces['dirty']:
                rendered += ".dev0"
            rendered += plus_or_dot(pieces)
            rendered += "g%s" % pieces['short']
    else:
        rendered = "0.post%d" % pieces['distance']
        if pieces['dirty']:
            rendered += ".dev0"
        rendered += "+g%s" % pieces['short']
    return rendered

def render_pep440_post_branch(pieces: Dict[str, Any]) -> str:
    """Build up post-release branch version string."""
    if pieces['closest-tag']:
        rendered: str = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += ".post%d" % pieces['distance']
            if pieces['branch'] != 'master':
                rendered += ".dev0"
            rendered += plus_or_dot(pieces)
            rendered += "g%s" % pieces['short']
            if pieces['dirty']:
                rendered += ".dirty"
    else:
        rendered = "0.post%d" % pieces['distance']
        if pieces['branch'] != 'master':
            rendered += ".dev0"
        rendered += "+g%s" % pieces['short']
        if pieces['dirty']:
            rendered += ".dirty"
    return rendered

def render_pep440_old(pieces: Dict[str, Any]) -> str:
    """Build up old-style pep440 version string."""
    if pieces['closest-tag']:
        rendered: str = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += ".post%d" % pieces['distance']
            if pieces['dirty']:
                rendered += ".dev0"
    else:
        rendered = "0.post%d" % pieces['distance']
        if pieces['dirty']:
            rendered += ".dev0"
    return rendered

def render_git_describe(pieces: Dict[str, Any]) -> str:
    """Build up version string like 'git describe'."""
    if pieces['closest-tag']:
        rendered: str = pieces['closest-tag']
        if pieces['distance']:
            rendered += "-%d-g%s" % (pieces['distance'], pieces['short'])
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += "-dirty"
    return rendered

def render_git_describe_long(pieces: Dict[str, Any]) -> str:
    """Build up long git describe version string."""
    if pieces['closest-tag']:
        rendered: str = pieces['closest-tag']
        rendered += "-%d-g%s" % (pieces['distance'], pieces['short'])
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += "-dirty"
    return rendered

def render(pieces: Dict[str, Any], style: str) -> Dict[str, Any]:
    """Render the given version pieces into the requested style."""
    if pieces['error']:
        return {'version': 'unknown',
                'full-revisionid': pieces.get('long'),
                'dirty': None,
                'error': pieces['error'],
                'date': None}
    if not style or style == 'default':
        style = 'pep440'
    if style == 'pep440':
        rendered: str = render_pep440(pieces)
    elif style == 'pep440-branch':
        rendered = render_pep440_branch(pieces)
    elif style == 'pep440-pre':
        rendered = render_pep440_pre(pieces)
    elif style == 'pep440-post':
        rendered = render_pep440_post(pieces)
    elif style == 'pep440-post-branch':
        rendered = render_pep440_post_branch(pieces)
    elif style == 'pep440-old':
        rendered = render_pep440_old(pieces)
    elif style == 'git-describe':
        rendered = render_git_describe(pieces)
    elif style == 'git-describe-long':
        rendered = render_git_describe_long(pieces)
    else:
        raise ValueError("unknown style '%s'" % style)
    return {'version': rendered, 'full-revisionid': pieces['long'],
            'dirty': pieces['dirty'], 'error': None,
            'date': pieces.get('date')}

class VersioneerBadRootError(Exception):
    """The project root directory is unknown or missing key files."""
    pass

def get_versions(verbose: bool = False) -> Dict[str, Any]:
    """Get the project version from whatever source is available."""
    if 'versioneer' in sys.modules:
        del sys.modules['versioneer']
    root: str = get_root()
    cfg: VersioneerConfig = get_config_from_root(root)
    assert cfg.VCS is not None, 'please set [versioneer]VCS= in setup.cfg'
    handlers: Optional[Dict[str, Callable[..., Any]]] = HANDLERS.get(cfg.VCS)
    assert handlers, "unrecognized VCS '%s'" % cfg.VCS
    verbose = verbose or bool(cfg.verbose)
    assert cfg.versionfile_source is not None, 'please set versioneer.versionfile_source'
    assert cfg.tag_prefix is not None, 'please set versioneer.tag_prefix'
    versionfile_abs: str = os.path.join(root, cfg.versionfile_source)
    get_keywords_f: Optional[Callable[[str], Dict[str, str]]] = handlers.get('get_keywords')
    from_keywords_f: Optional[Callable[[Dict[str, str], str, bool], Dict[str, Any]]] = handlers.get('keywords')
    if get_keywords_f and from_keywords_f:
        try:
            keywords: Dict[str, str] = get_keywords_f(versionfile_abs)
            ver: Dict[str, Any] = from_keywords_f(keywords, cfg.tag_prefix, verbose)
            if verbose:
                print('got version from expanded keyword %s' % ver)
            return ver
        except NotThisMethod:
            pass
    try:
        ver = versions_from_file(versionfile_abs)
        if verbose:
            print('got version from file %s %s' % (versionfile_abs, ver))
        return ver
    except NotThisMethod:
        pass
    from_vcs_f: Optional[Callable[..., Dict[str, Any]]] = handlers.get('pieces_from_vcs')
    if from_vcs_f:
        try:
            pieces = from_vcs_f(cfg.tag_prefix, root, verbose)
            ver = render(pieces, cfg.style)
            if verbose:
                print('got version from VCS %s' % ver)
            return ver
        except NotThisMethod:
            pass
    try:
        if cfg.parentdir_prefix:
            ver = versions_from_parentdir(cfg.parentdir_prefix, root, verbose)
            if verbose:
                print('got version from parentdir %s' % ver)
            return ver
    except NotThisMethod:
        pass
    if verbose:
        print('unable to compute version')
    return {'version': '0+unknown', 'full-revisionid': None, 'dirty': None, 'error': 'unable to compute version', 'date': None}

def get_version() -> str:
    """Get the short version string for this project."""
    return get_versions()['version']

def get_cmdclass(cmdclass: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the custom setuptools subclasses used by Versioneer."""
    if 'versioneer' in sys.modules:
        del sys.modules['versioneer']
    cmds: Dict[str, Any] = {} if cmdclass is None else cmdclass.copy()
    from setuptools import Command

    class cmd_version(Command):
        description: str = 'report generated version string'
        user_options: List[Any] = []
        boolean_options: List[Any] = []

        def initialize_options(self) -> None:
            pass

        def finalize_options(self) -> None:
            pass

        def run(self) -> None:
            vers: Dict[str, Any] = get_versions(verbose=True)
            print('Version: %s' % vers['version'])
            print(' full-revisionid: %s' % vers.get('full-revisionid'))
            print(' dirty: %s' % vers.get('dirty'))
            print(' date: %s' % vers.get('date'))
            if vers['error']:
                print(' error: %s' % vers['error'])
    cmds['version'] = cmd_version
    if 'build_py' in cmds:
        _build_py = cmds['build_py']
    else:
        from setuptools.command.build_py import build_py as _build_py

    class cmd_build_py(_build_py):
        def run(self) -> None:
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            versions: Dict[str, Any] = get_versions()
            _build_py.run(self)
            if getattr(self, 'editable_mode', False):
                return
            if cfg.versionfile_build:
                target_versionfile: str = os.path.join(self.build_lib, cfg.versionfile_build)
                print('UPDATING %s' % target_versionfile)
                write_to_version_file(target_versionfile, versions)
    cmds['build_py'] = cmd_build_py
    if 'build_ext' in cmds:
        _build_ext = cmds['build_ext']
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
                print(f'Warning: {target_versionfile} does not exist, skipping version update. This can happen if you are running build_ext without first running build_py.')
                return
            print('UPDATING %s' % target_versionfile)
            write_to_version_file(target_versionfile, versions)
    cmds['build_ext'] = cmd_build_ext
    if 'cx_Freeze' in sys.modules:
        from cx_Freeze.dist import build_exe as _build_exe

        class cmd_build_exe(_build_exe):
            def run(self) -> None:
                root: str = get_root()
                cfg: VersioneerConfig = get_config_from_root(root)
                versions: Dict[str, Any] = get_versions()
                target_versionfile: str = cfg.versionfile_source
                print('UPDATING %s' % target_versionfile)
                write_to_version_file(target_versionfile, versions)
                _build_exe.run(self)
                os.unlink(target_versionfile)
                with open(cfg.versionfile_source, 'w') as f:
                    LONG: str = LONG_VERSION_PY[cfg.VCS]
                    f.write(LONG % {'DOLLAR': '$', 'STYLE': cfg.style, 'TAG_PREFIX': cfg.tag_prefix, 'PARENTDIR_PREFIX': cfg.parentdir_prefix, 'VERSIONFILE_SOURCE': cfg.versionfile_source})
        cmds['build_exe'] = cmd_build_exe
        del cmds['build_py']
    if 'py2exe' in sys.modules:
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
                print('UPDATING %s' % target_versionfile)
                write_to_version_file(target_versionfile, versions)
                _py2exe.run(self)
                os.unlink(target_versionfile)
                with open(cfg.versionfile_source, 'w') as f:
                    LONG: str = LONG_VERSION_PY[cfg.VCS]
                    f.write(LONG % {'DOLLAR': '$', 'STYLE': cfg.style, 'TAG_PREFIX': cfg.tag_prefix, 'PARENTDIR_PREFIX': cfg.parentdir_prefix, 'VERSIONFILE_SOURCE': cfg.versionfile_source})
        cmds['py2exe'] = cmd_py2exe
    if 'egg_info' in cmds:
        _egg_info = cmds['egg_info']
    else:
        from setuptools.command.egg_info import egg_info as _egg_info

    class cmd_egg_info(_egg_info):
        def find_sources(self) -> None:
            super().find_sources()
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            self.filelist.append('versioneer.py')
            if cfg.versionfile_source:
                self.filelist.append(cfg.versionfile_source)
            self.filelist.sort()
            self.filelist.remove_duplicates()
            from setuptools import unicode_utils
            normalized: List[str] = [unicode_utils.filesys_decode(f).replace(os.sep, '/') for f in self.filelist.files]
            manifest_filename: str = os.path.join(self.egg_info, 'SOURCES.txt')
            with open(manifest_filename, 'w') as fobj:
                fobj.write('\n'.join(normalized))
    cmds['egg_info'] = cmd_egg_info
    if 'sdist' in cmds:
        _sdist = cmds['sdist']
    else:
        from setuptools.command.sdist import sdist as _sdist

    class cmd_sdist(_sdist):
        def run(self) -> Any:
            versions: Dict[str, Any] = get_versions()
            self._versioneer_generated_versions = versions
            self.distribution.metadata.version = versions['version']
            return _sdist.run(self)

        def make_release_tree(self, base_dir: str, files: List[str]) -> None:
            root: str = get_root()
            cfg: VersioneerConfig = get_config_from_root(root)
            _sdist.make_release_tree(self, base_dir, files)
            target_versionfile: str = os.path.join(base_dir, cfg.versionfile_source)
            print('UPDATING %s' % target_versionfile)
            write_to_version_file(target_versionfile, self._versioneer_generated_versions)
    cmds['sdist'] = cmd_sdist
    return cmds

CONFIG_ERROR: str = (
    "\nsetup.cfg is missing the necessary Versioneer configuration. You need\na section like:\n\n [versioneer]\n VCS = git\n style = pep440\n versionfile_source = src/myproject/_version.py\n versionfile_build = myproject/_version.py\n tag_prefix =\n parentdir_prefix = myproject-\n\n"
    "You will also need to edit your setup.py to use the results:\n\n import versioneer\n setup(version=versioneer.get_version(),\n       cmdclass=versioneer.get_cmdclass(), ...)\n\nPlease read the docstring in ./versioneer.py for configuration instructions,\nedit setup.cfg, and re-run the installer or 'python versioneer.py setup'.\n"
)
SAMPLE_CONFIG: str = (
    "\n# See the docstring in versioneer.py for instructions. Note that you must\n# re-run 'versioneer.py setup' after changing this section, and commit the\n# resulting files.\n\n[versioneer]\n#VCS = git\n#style = pep440\n#versionfile_source =\n#versionfile_build =\n#tag_prefix =\n#parentdir_prefix =\n\n"
)
OLD_SNIPPET: str = "\nfrom ._version import get_versions\n__version__ = get_versions()['version']\ndel get_versions\n"
INIT_PY_SNIPPET: str = "\nfrom . import {0}\n__version__ = {0}.get_versions()['version']\n"

def do_setup() -> int:
    """Do main VCS-independent setup function for installing Versioneer."""
    root: str = get_root()
    try:
        cfg: VersioneerConfig = get_config_from_root(root)
    except (OSError, configparser.NoSectionError, configparser.NoOptionError) as e:
        if isinstance(e, (OSError, configparser.NoSectionError)):
            print('Adding sample versioneer config to setup.cfg', file=sys.stderr)
            with open(os.path.join(root, 'setup.cfg'), 'a') as f:
                f.write(SAMPLE_CONFIG)
        print(CONFIG_ERROR, file=sys.stderr)
        return 1
    print(' creating %s' % cfg.versionfile_source)
    with open(cfg.versionfile_source, 'w') as f:
        LONG: str = LONG_VERSION_PY[cfg.VCS]
        f.write(LONG % {'DOLLAR': '$', 'STYLE': cfg.style, 'TAG_PREFIX': cfg.tag_prefix, 'PARENTDIR_PREFIX': cfg.parentdir_prefix, 'VERSIONFILE_SOURCE': cfg.versionfile_source})
    ipy: str = os.path.join(os.path.dirname(cfg.versionfile_source), '__init__.py')
    maybe_ipy: Optional[str] = ipy
    if os.path.exists(ipy):
        try:
            with open(ipy, 'r') as f:
                old: str = f.read()
        except OSError:
            old = ''
        module: str = os.path.splitext(os.path.basename(cfg.versionfile_source))[0]
        snippet: str = INIT_PY_SNIPPET.format(module)
        if OLD_SNIPPET in old:
            print(' replacing boilerplate in %s' % ipy)
            with open(ipy, 'w') as f:
                f.write(old.replace(OLD_SNIPPET, snippet))
        elif snippet not in old:
            print(' appending to %s' % ipy)
            with open(ipy, 'a') as f:
                f.write(snippet)
        else:
            print(' %s unmodified' % ipy)
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
    with open('setup.py', 'r') as f:
        for line in f.readlines():
            if 'import versioneer' in line:
                found.add('import')
            if 'versioneer.get_cmdclass()' in line:
                found.add('cmdclass')
            if 'versioneer.get_version()' in line:
                found.add('get_version')
            if 'versioneer.VCS' in line:
                setters = True
            if 'versioneer.versionfile_source' in line:
                setters = True
    if len(found) != 3:
        print('')
        print('Your setup.py appears to be missing some important items')
        print('(but I might be wrong). Please make sure it has something')
        print('roughly like the following:')
        print('')
        print(' import versioneer')
        print(' setup( version=versioneer.get_version(),')
        print('        cmdclass=versioneer.get_cmdclass(),  ...)')
        print('')
        errors += 1
    if setters:
        print("You should remove lines like 'versioneer.VCS = ' and")
        print("'versioneer.versionfile_source = ' . This configuration")
        print('now lives in setup.cfg, and should be removed from setup.py')
        print('')
        errors += 1
    return errors

def setup_command() -> None:
    """Set up Versioneer and exit with appropriate error code."""
    errors: int = do_setup()
    errors += scan_setup_py()
    sys.exit(1 if errors else 0)

if __name__ == '__main__':
    cmd: str = sys.argv[1]
    if cmd == 'setup':
        setup_command()