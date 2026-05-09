from collections.abc import Callable
import errno
import functools
import os
import re
import subprocess
import sys

def get_keywords() -> dict:
    """Get the keywords needed to look up the version information."""
    git_refnames = '$Format:%d$'
    git_full = '$Format:%H$'
    git_date = '$Format:%ci$'
    keywords = {'refnames': git_refnames, 'full': git_full, 'date': git_date}
    return keywords

class VersioneerConfig:
    """Container for Versioneer configuration parameters."""

def get_config() -> VersioneerConfig:
    """Create, populate and return the VersioneerConfig() object."""
    cfg = VersioneerConfig()
    cfg.VCS = 'git'
    cfg.style = 'pep440'
    cfg.tag_prefix = 'v'
    cfg.parentdir_prefix = 'pandas-'
    cfg.versionfile_source = 'pandas/_version.py'
    cfg.verbose = False
    return cfg

class NotThisMethod(Exception):
    """Exception raised if a method is not valid for the current scenario."""

LONG_VERSION_PY = {}
HANDLERS = {}

def register_vcs_handler(vcs: str, method: str) -> Callable:
    """Create decorator to mark a method as the handler of a VCS."""

    def decorate(f) -> Callable:
        """Store f in HANDLERS[vcs][method]."""
        if vcs not in HANDLERS:
            HANDLERS[vcs] = {}
        HANDLERS[vcs][method] = f
        return f
    return decorate

def run_command(commands: list, args: list, cwd: str = None, verbose: bool = False, hide_stderr: bool = False, env: dict = None) -> tuple:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process = None
    popen_kwargs = {}
    if sys.platform == 'win32':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs['startupinfo'] = startupinfo
    for command in commands:
        dispcmd = str([command] + args)
        try:
            process = subprocess.Popen([command] + args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE if hide_stderr else None, **popen_kwargs)
            break
        except OSError:
            e = sys.exc_info()[1]
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

# ... rest of the code ...
