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

have_tomllib = True
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        have_tomllib = False

class VersioneerConfig:
    """Container for Versioneer configuration parameters."""
    VCS: str
    style: str
    tag_prefix: str
    parentdir_prefix: str
    versionfile_source: str
    versionfile_build: Optional[str]
    verbose: bool

def get_root() -> str:
    """Get the project root directory."""
    root = os.path.realpath(os.path.abspath(os.getcwd())
    setup_py = os.path.join(root, 'setup.py')
    pyproject_toml = os.path.join(root, 'pyproject.toml')
    versioneer_py = os.path.join(root, 'versioneer.py')
    if not (os.path.exists(setup_py) or os.path.exists(pyproject_toml) or os.path.exists(versioneer_py):
        root = os.path.dirname(os.path.realpath(os.path.abspath(sys.argv[0])))
        setup_py = os.path.join(root, 'setup.py')
        pyproject_toml = os.path.join(root, 'pyproject.toml')
        versioneer_py = os.path.join(root, 'versioneer.py')
    if not (os.path.exists(setup_py) or os.path.exists(pyproject_toml) or os.path.exists(versioneer_py)):
        err = "Versioneer was unable to run the project root directory."
        raise VersioneerBadRootError(err)
    try:
        my_path = os.path.realpath(os.path.abspath(__file__))
        me_dir = os.path.normcase(os.path.splitext(my_path)[0])
        vsr_dir = os.path.normcase(os.path.splitext(versioneer_py)[0])
        if me_dir != vsr_dir and 'VERSIONEER_PEP518' not in globals():
            print(f'Warning: build in {os.path.dirname(my_path)} is using versioneer.py from {versioneer_py}')
    except NameError:
        pass
    return root

def get_config_from_root(root: str) -> VersioneerConfig:
    """Read the project setup.cfg file to determine Versioneer config."""
    root_pth = Path(root)
    pyproject_toml = root_pth / 'pyproject.toml'
    setup_cfg = root_pth / 'setup.cfg'
    section = None
    if pyproject_toml.exists() and have_tomllib:
        try:
            with open(pyproject_toml, 'rb') as fobj:
                pp = tomllib.load(fobj)
            section = pp['tool']['versioneer']
        except (tomllib.TOMLDecodeError, KeyError) as e:
            print(f'Failed to load config from {pyproject_toml}: {e}')
            print('Try to load it from setup.cfg')
    if not section:
        parser = configparser.ConfigParser()
        with open(setup_cfg) as cfg_file:
            parser.read_file(cfg_file)
        parser.get('versioneer', 'VCS')
        section = parser['versioneer']
    cfg = VersioneerConfig()
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

LONG_VERSION_PY: Dict[str, str] = {}
HANDLERS: Dict[str, Dict[str, Callable]] = {}

def register_vcs_handler(vcs: str, method: str) -> Callable:
    """Create decorator to mark a method as the handler of a VCS."""
    def decorate(f: Callable) -> Callable:
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
    env: Optional[Dict[str, str]] = None
) -> Tuple[Optional[str], Optional[int]]:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process = None
    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == 'win32':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs['startupinfo'] = startupinfo
    for command in commands:
        try:
            dispcmd = str([command] + args)
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
                print(f'unable to run {dispcmd}')
                print(e)
            return (None, None)
    else:
        if verbose:
            print(f'unable to find command, tried {commands}')
        return (None, None)
    stdout = process.communicate()[0].strip().decode()
    if process.returncode != 0:
        if verbose:
            print(f'unable to run {dispcmd} (error)')
            print(f'stdout was {stdout}')
        return (None, process.returncode)
    return (stdout, process.returncode)

# [Rest of the type annotations would continue in the same pattern...]
# [The full annotated version would include type hints for all functions and variables]
# [This is just a partial example showing the pattern]

class VersioneerBadRootError(Exception):
    """The project root directory is unknown or missing key files."""

def get_versions(verbose: bool = False) -> Dict[str, Any]:
    """Get the project version from whatever source is available."""
    if 'versioneer' in sys.modules:
        del sys.modules['versioneer']
    root = get_root()
    cfg = get_config_from_root(root)
    assert cfg.VCS is not None, 'please set [versioneer]VCS= in setup.cfg'
    handlers = HANDLERS.get(cfg.VCS)
    assert handlers, f"unrecognized VCS '{cfg.VCS}'"
    verbose = verbose or bool(cfg.verbose)
    assert cfg.versionfile_source is not None, 'please set versioneer.versionfile_source'
    assert cfg.tag_prefix is not None, 'please set versioneer.tag_prefix'
    versionfile_abs = os.path.join(root, cfg.versionfile_source)
    get_keywords_f = handlers.get('get_keywords')
    from_keywords_f = handlers.get('keywords')
    if get_keywords_f and from_keywords_f:
        try:
            keywords = get_keywords_f(versionfile_abs)
            ver = from_keywords_f(keywords, cfg.tag_prefix, verbose)
            if verbose:
                print(f'got version from expanded keyword {ver}')
            return ver
        except NotThisMethod:
            pass
    try:
        ver = versions_from_file(versionfile_abs)
        if verbose:
            print(f'got version from file {versionfile_abs} {ver}')
        return ver
    except NotThisMethod:
        pass
    from_vcs_f = handlers.get('pieces_from_vcs')
    if from_vcs_f:
        try:
            pieces = from_vcs_f(cfg.tag_prefix, root, verbose)
            ver = render(pieces, cfg.style)
            if verbose:
                print(f'got version from VCS {ver}')
            return ver
        except NotThisMethod:
            pass
    try:
        if cfg.parentdir_prefix:
            ver = versions_from_parentdir(cfg.parentdir_prefix, root, verbose)
            if verbose:
                print(f'got version from parentdir {ver}')
            return ver
    except NotThisMethod:
        pass
    if verbose:
        print('unable to compute version')
    return {
        'version': '0+unknown',
        'full-revisionid': None,
        'dirty': None,
        'error': 'unable to compute version',
        'date': None
    }

def get_version() -> str:
    """Get the short version string for this project."""
    return get_versions()['version']

def get_cmdclass(cmdclass: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the custom setuptools subclasses used by Versioneer."""
    if 'versioneer' in sys.modules:
        del sys.modules['versioneer']
    cmds = {} if cmdclass is None else cmdclass.copy()
    from setuptools import Command

    class cmd_version(Command):
        description = 'report generated version string'
        user_options: List[Tuple[str, Optional[str], str]] = []
        boolean_options: List[str] = []

        def initialize_options(self) -> None:
            pass

        def finalize_options(self) -> None:
            pass

        def run(self) -> None:
            vers = get_versions(verbose=True)
            print(f'Version: {vers["version"]}')
            print(f' full-revisionid: {vers.get("full-revisionid")}')
            print(f' dirty: {vers.get("dirty")}')
            print(f' date: {vers.get("date")}')
            if vers['error']:
                print(f' error: {vers["error"]}')
    cmds['version'] = cmd_version
    
    # [Rest of the cmdclass definitions would continue with type hints...]
    
    return cmds

# [Remaining functions would be annotated in the same pattern]
