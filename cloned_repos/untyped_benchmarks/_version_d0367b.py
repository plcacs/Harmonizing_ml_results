"""Git implementation of _version.py."""
import errno
import functools
import os
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

def get_keywords():
    """Get the keywords needed to look up the version information."""
    git_refnames = '$Format:%d$'
    git_full = '$Format:%H$'
    git_date = '$Format:%ci$'
    keywords = {'refnames': git_refnames, 'full': git_full, 'date': git_date}
    return keywords

class VersioneerConfig:
    """Container for Versioneer configuration parameters."""

def get_config():
    """Create, populate and return the VersioneerConfig() object."""
    cfg = VersioneerConfig()
    cfg.VCS = 'git'
    cfg.style = 'pep440'
    cfg.tag_prefix = ''
    cfg.parentdir_prefix = ''
    cfg.versionfile_source = 'src/prefect/_version.py'
    cfg.verbose = False
    return cfg

class NotThisMethod(Exception):
    """Exception raised if a method is not valid for the current scenario."""
LONG_VERSION_PY = {}
HANDLERS = {}

def register_vcs_handler(vcs, method):
    """Create decorator to mark a method as the handler of a VCS."""

    def decorate(f):
        """Store f in HANDLERS[vcs][method]."""
        if vcs not in HANDLERS:
            HANDLERS[vcs] = {}
        HANDLERS[vcs][method] = f
        return f
    return decorate

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process = None
    popen_kwargs = {}
    if sys.platform == 'win32':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs['startupinfo'] = startupinfo
    for command in commands:
        try:
            dispcmd = str([command] + args)
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
    stdout = process.communicate()[0].strip().decode()
    if process.returncode != 0:
        if verbose:
            print('unable to run %s (error)' % dispcmd)
            print('stdout was %s' % stdout)
        return (None, process.returncode)
    return (stdout, process.returncode)

def versions_from_parentdir(parentdir_prefix, root, verbose):
    """Try to determine the version from the parent directory name.

    Source tarballs conventionally unpack into a directory that includes both
    the project name and a version string. We will also support searching up
    two directory levels for an appropriately named parent directory
    """
    rootdirs = []
    for _ in range(3):
        dirname = os.path.basename(root)
        if dirname.startswith(parentdir_prefix):
            return {'version': dirname[len(parentdir_prefix):], 'full-revisionid': None, 'dirty': False, 'error': None, 'date': None}
        rootdirs.append(root)
        root = os.path.dirname(root)
    if verbose:
        print('Tried directories %s but none started with prefix %s' % (str(rootdirs), parentdir_prefix))
    raise NotThisMethod("rootdir doesn't start with parentdir_prefix")

@register_vcs_handler('git', 'get_keywords')
def git_get_keywords(versionfile_abs):
    """Extract version information from the given file."""
    keywords = {}
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
def git_versions_from_keywords(keywords, tag_prefix, verbose):
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
            print("discarding '%s', no digits" % ','.join(refs - tags))
    if verbose:
        print('likely tags: %s' % ','.join(sorted(tags)))
    for ref in sorted(tags):
        if ref.startswith(tag_prefix):
            r = ref[len(tag_prefix):]
            if not re.match('\\d', r):
                continue
            if verbose:
                print('picking %s' % r)
            return {'version': r, 'full-revisionid': keywords['full'].strip(), 'dirty': False, 'error': None, 'date': date}
    if verbose:
        print('no suitable tags, using unknown + full revision id')
    return {'version': '0+unknown', 'full-revisionid': keywords['full'].strip(), 'dirty': False, 'error': 'no suitable tags', 'date': None}

@register_vcs_handler('git', 'pieces_from_vcs')
def git_pieces_from_vcs(tag_prefix, root, verbose, runner=run_command):
    """Get version from 'git describe' in the root of the source tree.

    This only gets called if the git-archive 'subst' keywords were *not*
    expanded, and _version.py hasn't already been rewritten with a short
    version string, meaning we're inside a checked out source tree.
    """
    GITS = ['git']
    if sys.platform == 'win32':
        GITS = ['git.cmd', 'git.exe']
    env = os.environ.copy()
    env.pop('GIT_DIR', None)
    runner = functools.partial(runner, env=env)
    _, rc = runner(GITS, ['rev-parse', '--git-dir'], cwd=root, hide_stderr=not verbose)
    if rc != 0:
        if verbose:
            print('Directory %s not under git control' % root)
        raise NotThisMethod("'git rev-parse --git-dir' returned error")
    describe_out, rc = runner(GITS, ['describe', '--tags', '--dirty', '--always', '--long', '--match', f'{tag_prefix}[[:digit:]]*'], cwd=root)
    if describe_out is None:
        raise NotThisMethod("'git describe' failed")
    describe_out = describe_out.strip()
    full_out, rc = runner(GITS, ['rev-parse', 'HEAD'], cwd=root)
    if full_out is None:
        raise NotThisMethod("'git rev-parse' failed")
    full_out = full_out.strip()
    pieces = {}
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
            pieces['error'] = "unable to parse git-describe output: '%s'" % describe_out
            return pieces
        full_tag = mo.group(1)
        if not full_tag.startswith(tag_prefix):
            if verbose:
                fmt = "tag '%s' doesn't start with prefix '%s'"
                print(fmt % (full_tag, tag_prefix))
            pieces['error'] = "tag '%s' doesn't start with prefix '%s'" % (full_tag, tag_prefix)
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

def plus_or_dot(pieces):
    """Return a + if we don't already have one, else return a ."""
    if '+' in pieces.get('closest-tag', ''):
        return '.'
    return '+'

def render_pep440(pieces):
    """Build up version string, with post-release "local version identifier".

    Our goal: TAG[+DISTANCE.gHEX[.dirty]] . Note that if you
    get a tagged build and then dirty it, you'll get TAG+0.gHEX.dirty

    Exceptions:
    1: no tags. git_describe was just HEX. 0+untagged.DISTANCE.gHEX[.dirty]
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += plus_or_dot(pieces)
            rendered += '%d.g%s' % (pieces['distance'], pieces['short'])
            if pieces['dirty']:
                rendered += '.dirty'
    else:
        rendered = '0+untagged.%d.g%s' % (pieces['distance'], pieces['short'])
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered

def render_pep440_branch(pieces):
    """TAG[[.dev0]+DISTANCE.gHEX[.dirty]] .

    The ".dev0" means not master branch. Note that .dev0 sorts backwards
    (a feature branch will appear "older" than the master branch).

    Exceptions:
    1: no tags. 0[.dev0]+untagged.DISTANCE.gHEX[.dirty]
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
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

def pep440_split_post(ver):
    """Split pep440 version string at the post-release segment.

    Returns the release segments before the post-release and the
    post-release version number (or -1 if no post-release segment is present).
    """
    vc = str.split(ver, '.post')
    return (vc[0], int(vc[1] or 0) if len(vc) == 2 else None)

def render_pep440_pre(pieces):
    """TAG[.postN.devDISTANCE] -- No -dirty.

    Exceptions:
    1: no tags. 0.post0.devDISTANCE
    """
    if pieces['closest-tag']:
        if pieces['distance']:
            tag_version, post_version = pep440_split_post(pieces['closest-tag'])
            rendered = tag_version
            if post_version is not None:
                rendered += '.post%d.dev%d' % (post_version + 1, pieces['distance'])
            else:
                rendered += '.post0.dev%d' % pieces['distance']
        else:
            rendered = pieces['closest-tag']
    else:
        rendered = '0.post0.dev%d' % pieces['distance']
    return rendered

def render_pep440_post(pieces):
    """TAG[.postDISTANCE[.dev0]+gHEX] .

    The ".dev0" means dirty. Note that .dev0 sorts backwards
    (a dirty tree will appear "older" than the corresponding clean one),
    but you shouldn't be releasing software with -dirty anyways.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += '.post%d' % pieces['distance']
            if pieces['dirty']:
                rendered += '.dev0'
            rendered += plus_or_dot(pieces)
            rendered += 'g%s' % pieces['short']
    else:
        rendered = '0.post%d' % pieces['distance']
        if pieces['dirty']:
            rendered += '.dev0'
        rendered += '+g%s' % pieces['short']
    return rendered

def render_pep440_post_branch(pieces):
    """TAG[.postDISTANCE[.dev0]+gHEX[.dirty]] .

    The ".dev0" means not master branch.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]+gHEX[.dirty]
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += '.post%d' % pieces['distance']
            if pieces['branch'] != 'master':
                rendered += '.dev0'
            rendered += plus_or_dot(pieces)
            rendered += 'g%s' % pieces['short']
            if pieces['dirty']:
                rendered += '.dirty'
    else:
        rendered = '0.post%d' % pieces['distance']
        if pieces['branch'] != 'master':
            rendered += '.dev0'
        rendered += '+g%s' % pieces['short']
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered

def render_pep440_old(pieces):
    """TAG[.postDISTANCE[.dev0]] .

    The ".dev0" means dirty.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += '.post%d' % pieces['distance']
            if pieces['dirty']:
                rendered += '.dev0'
    else:
        rendered = '0.post%d' % pieces['distance']
        if pieces['dirty']:
            rendered += '.dev0'
    return rendered

def render_git_describe(pieces):
    """TAG[-DISTANCE-gHEX][-dirty].

    Like 'git describe --tags --dirty --always'.

    Exceptions:
    1: no tags. HEX[-dirty]  (note: no 'g' prefix)
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance']:
            rendered += '-%d-g%s' % (pieces['distance'], pieces['short'])
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += '-dirty'
    return rendered

def render_git_describe_long(pieces):
    """TAG-DISTANCE-gHEX[-dirty].

    Like 'git describe --tags --dirty --always -long'.
    The distance/hash is unconditional.

    Exceptions:
    1: no tags. HEX[-dirty]  (note: no 'g' prefix)
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        rendered += '-%d-g%s' % (pieces['distance'], pieces['short'])
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += '-dirty'
    return rendered

def render(pieces, style):
    """Render the given version pieces into the requested style."""
    if pieces['error']:
        return {'version': 'unknown', 'full-revisionid': pieces.get('long'), 'dirty': None, 'error': pieces['error'], 'date': None}
    if not style or style == 'default':
        style = 'pep440'
    if style == 'pep440':
        rendered = render_pep440(pieces)
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
    return {'version': rendered, 'full-revisionid': pieces['long'], 'dirty': pieces['dirty'], 'error': None, 'date': pieces.get('date')}

def get_versions():
    """Get version information or return default if unable to do so."""
    cfg = get_config()
    verbose = cfg.verbose
    try:
        return git_versions_from_keywords(get_keywords(), cfg.tag_prefix, verbose)
    except NotThisMethod:
        pass
    try:
        root = os.path.realpath(__file__)
        for _ in cfg.versionfile_source.split('/'):
            root = os.path.dirname(root)
    except NameError:
        return {'version': '0+unknown', 'full-revisionid': None, 'dirty': None, 'error': 'unable to find root of source tree', 'date': None}
    try:
        pieces = git_pieces_from_vcs(cfg.tag_prefix, root, verbose)
        return render(pieces, cfg.style)
    except NotThisMethod:
        pass
    try:
        if cfg.parentdir_prefix:
            return versions_from_parentdir(cfg.parentdir_prefix, root, verbose)
    except NotThisMethod:
        pass
    return {'version': '0+unknown', 'full-revisionid': None, 'dirty': None, 'error': 'unable to compute version', 'date': None}