"""Git implementation of _version.py."""
from collections.abc import Callable
import errno
import functools
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

def get_keywords() -> Dict[str, str]:
    """Get the keywords needed to look up the version information."""
    git_refnames = '$Format:%d$'
    git_full = '$Format:%H$'
    git_date = '$Format:%ci$'
    keywords = {'refnames': git_refnames, 'full': git_full, 'date': git_date}
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

LONG_VERSION_PY: Dict[str, str] = {}
HANDLERS: Dict[str, Dict[str, Callable[..., Any]]] = {}

def register_vcs_handler(vcs: str, method: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create decorator to mark a method as the handler of a VCS."""

    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        """Store f in HANDLERS[vcs][method]."""
        if vcs not in HANDLERS:
            HANDLERS[vcs] = {}
        HANDLERS[vcs][method] = f
        return f
    return decorate

def run_command(commands: List[str], args: List[str], cwd: Optional[str] = None, verbose: bool = False, hide_stderr: bool = False, env: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[int]]:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process = None
    popen_kwargs: Dict[str, Any] = {}
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

def versions_from_parentdir(parentdir_prefix: str, root: str, verbose: bool) -> Dict[str, Optional[Union[str, bool]]]:
    """Try to determine the version from the parent directory name."""
    rootdirs = []
    for _ in range(3):
        dirname = os.path.basename(root)
        if dirname.startswith(parentdir_prefix):
            return {'version': dirname[len(parentdir_prefix):], 'full-revisionid': None, 'dirty': False, 'error': None, 'date': None}
        rootdirs.append(root)
        root = os.path.dirname(root)
    if verbose:
        print(f'Tried directories {rootdirs!s}             but none started with prefix {parentdir_prefix}')
    raise NotThisMethod("rootdir doesn't start with parentdir_prefix")

@register_vcs_handler('git', 'get_keywords')
def git_get_keywords(versionfile_abs: str) -> Dict[str, str]:
    """Extract version information from the given file."""
    keywords = {}
    try:
        with open(versionfile_abs, encoding='utf-8') as fobj:
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
def git_versions_from_keywords(keywords: Dict[str, str], tag_prefix: str, verbose: bool) -> Dict[str, Optional[Union[str, bool]]]:
    """Get version information from git keywords."""
    if 'refnames' not in keywords:
        raise NotThisMethod('Short version file found')
    date = keywords.get('date')
    if date is not None:
        date = date.splitlines()[-1]
        date = date.strip().replace(' ', 'T', 1).replace(' ', '', 1)
    refnames = keywords['refnames'].strip()
    if refnames.startswith('$Format'):
        if verbose:
            print('keywords are unexpanded, not using')
        raise NotThisMethod('unexpanded keywords, not a git-archive tarball')
    refs = {r.strip() for r in refnames.strip('()').split(',')}
    TAG = 'tag: '
    tags = {r[len(TAG):] for r in refs if r.startswith(TAG)}
    if not tags:
        tags = {r for r in refs if re.search('\\d', r)}
        if verbose:
            print(f"discarding '{','.join(refs - tags)}', no digits")
    if verbose:
        print(f'likely tags: {','.join(sorted(tags))}')
    for ref in sorted(tags):
        if ref.startswith(tag_prefix):
            r = ref[len(tag_prefix):]
            if not re.match('\\d', r):
                continue
            if verbose:
                print(f'picking {r}')
            return {'version': r, 'full-revisionid': keywords['full'].strip(), 'dirty': False, 'error': None, 'date': date}
    if verbose:
        print('no suitable tags, using unknown + full revision id')
    return {'version': '0+unknown', 'full-revisionid': keywords['full'].strip(), 'dirty': False, 'error': 'no suitable tags', 'date': None}

@register_vcs_handler('git', 'pieces_from_vcs')
def git_pieces_from_vcs(tag_prefix: str, root: str, verbose: bool, runner: Callable[..., Tuple[Optional[str], Optional[int]]] = run_command) -> Dict[str, Any]:
    """Get version from 'git describe' in the root of the source tree."""
    GITS = ['git']
    if sys.platform == 'win32':
        GITS = ['git.cmd', 'git.exe']
    env = os.environ.copy()
    env.pop('GIT_DIR', None)
    runner = functools.partial(runner, env=env)
    _, rc = runner(GITS, ['rev-parse', '--git-dir'], cwd=root, hide_stderr=not verbose)
    if rc != 0:
        if verbose:
            print(f'Directory {root} not under git control')
        raise NotThisMethod("'git rev-parse --git-dir' returned error")
    describe_out, rc = runner(GITS, ['describe', '--tags', '--dirty', '--always', '--long', '--match', f'{tag_prefix}[[:digit:]]*'], cwd=root)
    if describe_out is None:
        raise NotThisMethod("'git describe' failed")
    describe_out = describe_out.strip()
    full_out, rc = runner(GITS, ['rev-parse', 'HEAD'], cwd=root)
    if full_out is None:
        raise NotThisMethod("'git rev-parse' failed")
    full_out = full_out.strip()
    pieces: Dict[str, Any] = {}
    pieces['long'] = full_out
    pieces['short'] = full_out[:7]
    pieces['error'] = None
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
    pieces['branch'] = branch_name
    git_describe = describe_out
    dirty = git_describe.endswith('-dirty')
    pieces['dirty'] = dirty
    if dirty:
        git_describe = git_describe[:git_describe.rindex('-dirty')]
    if '-' in git_describe:
        mo = re.search('^(.+)-(\\d+)-g([0-9a-f]+)$', git_describe)
        if not mo:
            pieces['error'] = f"unable to parse git-describe output: '{describe_out}'"
            return pieces
        full_tag = mo.group(1)
        if not full_tag.startswith(tag_prefix):
            if verbose:
                fmt = "tag '%s' doesn't start with prefix '%s'"
                print(fmt % (full_tag, tag_prefix))
            pieces['error'] = f"tag '{full_tag}' doesn't start with prefix '{tag_prefix}'"
            return pieces
        pieces['closest-tag'] = full_tag[len(tag_prefix):]
        pieces['distance'] = int(mo.group(2))
        pieces['short'] = mo.group(3)
    else:
        pieces['closest-tag'] = None
        out, rc = runner(GITS, ['rev-list', 'HEAD', '--left-right'], cwd=root)
        pieces['distance'] = len(out.split())
    date = runner(GITS, ['show', '-s', '--format=%ci', 'HEAD'], cwd=root)[0].strip()
    date = date.splitlines()[-1]
    pieces['date'] = date.strip().replace(' ', 'T', 1).replace(' ', '', 1)
    return pieces

def plus_or_dot(pieces: Dict[str, Any]) -> str:
    """Return a + if we don't already have one, else return a ."""
    if '+' in pieces.get('closest-tag', ''):
        return '.'
    return '+'

def render_pep440(pieces: Dict[str, Any]) -> str:
    """Build up version string, with post-release "local version identifier"."""
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += plus_or_dot(pieces)
            rendered += f'{pieces['distance']}.g{pieces['short']}'
            if pieces['dirty']:
                rendered += '.dirty'
    else:
        rendered = f'0+untagged.{pieces['distance']}.g{pieces['short']}'
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered

def render_pep440_branch(pieces: Dict[str, Any]) -> str:
    """TAG[[.dev0]+DISTANCE.gHEX[.dirty]] ."""
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            if pieces['branch'] != 'master':
                rendered += '.dev0'
            rendered += plus_or_dot(pieces)
            rendered += f'{pieces['distance']}.g{pieces['short']}'
            if pieces['dirty']:
                rendered += '.dirty'
    else:
        rendered = '0'
        if pieces['branch'] != 'master':
            rendered += '.dev0'
        rendered += f'+untagged.{pieces['distance']}.g{pieces['short']}'
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered

def pep440_split_post(ver: str) -> Tuple[str, Optional[int]]:
    """Split pep440 version string at the post-release segment."""
    vc = str.split(ver, '.post')
    return (vc[0], int(vc[1] or 0) if len(vc) == 2 else None)

def render_pep440_pre(pieces: Dict[str, Any]) -> str:
    """TAG[.postN.devDISTANCE] -- No -dirty."""
    if pieces['closest-tag']:
        if pieces['distance']:
            tag_version, post_version = pep440_split_post(pieces['closest-tag'])
            rendered = tag_version
            if post_version is not None:
                rendered += f'.post{post_version + 1}.dev{pieces['distance']}'
            else:
                rendered += f'.post0.dev{pieces['distance']}'
        else:
            rendered = pieces['closest-tag']
    else:
        rendered = f'0.post0.dev{pieces['distance']}'
    return rendered

def render_pep440_post(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]+gHEX] ."""
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += f'.post{pieces['distance']}'
            if pieces['dirty']:
                rendered += '.dev0'
            rendered += plus_or_dot(pieces)
            rendered += f'g{pieces['short']}'
    else:
        rendered = f'0.post{pieces['distance']}'
        if pieces['dirty']:
            rendered += '.dev0'
        rendered += f'+g{pieces['short']}'
    return rendered

def render_pep440_post_branch(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]+gHEX[.dirty]] ."""
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += f'.post{pieces['distance']}'
            if pieces['branch'] != 'master':
                rendered += '.dev0'
            rendered += plus_or_dot(pieces)
            rendered += f'g{pieces['short']}'
            if pieces['dirty']:
                rendered += '.dirty'
    else:
        rendered = f'0.post{pieces['distance']}'
        if pieces['branch'] != 'master':
            rendered += '.dev0'
        rendered += f'+g{pieces['short']}'
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered

def render_pep440_old(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]] ."""
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += f'0.post{pieces['distance']}'
            if pieces['dirty']:
                rendered += '.dev0'
    else:
        rendered = f'0.post{pieces['distance']}'
        if pieces['dirty']:
            rendered += '.dev0'
    return rendered

def render_git_describe(pieces: Dict[str, Any]) -> str:
    """TAG[-DISTANCE-gHEX][-dirty]."""
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance']:
            rendered += f'-{pieces['distance']}-g{pieces['short']}'
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += '-dirty'
    return rendered

def render_git_describe_long(pieces: Dict[str, Any]) -> str:
    """TAG-DISTANCE-gHEX[-dirty]."""
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        rendered += f'-{pieces['distance']}-g{pieces['short']}'
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += '-dirty'
    return rendered

def render(pieces: Dict[str, Any], style: str) -> Dict[str, Optional[Union[str, bool]]]:
    """Render the given version pieces into the requested style."""
    if pieces['error']:
        return {'version': 'unknown', 'full-revisionid': pieces.get('long'), 'dirty': None, 'error': pieces['error'], 'date': None}
