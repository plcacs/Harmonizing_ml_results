import os.path
import re
from typing import Optional, Tuple, List, Match, Pattern, Any
from packaging import version
from dbt.events.types import GitNothingToDo, GitProgressCheckedOutAt, GitProgressCheckoutRevision, GitProgressPullingNewDependency, GitProgressUpdatedCheckoutRange, GitProgressUpdatingExistingDependency, GitSparseCheckoutSubdirectory
from dbt.exceptions import CommandResultError, DbtRuntimeError, GitCheckoutError, GitCloningError, UnknownGitCloningProblemError
from dbt_common.clients.system import rmdir, run_cmd
from dbt_common.events.functions import fire_event

def _is_commit(revision: str) -> bool:
    return bool(re.match('\\b[0-9a-f]{40}\\b', revision))

def clone(repo: str, cwd: str, dirname: Optional[str] = None, remove_git_dir: bool = False, revision: Optional[str] = None, subdirectory: Optional[str] = None) -> Tuple[bytes, bytes]:
    has_revision: bool = revision is not None
    is_commit: bool = _is_commit(revision or '')
    clone_cmd: List[str] = ['git', 'clone', '--depth', '1']
    if subdirectory:
        fire_event(GitSparseCheckoutSubdirectory(subdir=subdirectory))
        out, _ = run_cmd(cwd, ['git', '--version'], env={'LC_ALL': 'C'})
        git_version = version.parse(re.search('\\d+\\.\\d+\\.\\d+', out.decode('utf-8')).group(0))
        if not git_version >= version.parse('2.25.0'):
            raise RuntimeError('Please update your git version to pull a dbt package from a subdirectory: your version is {}, >= 2.25.0 needed'.format(git_version))
        clone_cmd.extend(['--filter=blob:none', '--sparse'])
    if has_revision and (not is_commit):
        clone_cmd.extend(['--branch', revision])
    clone_cmd.append(repo)
    if dirname is not None:
        clone_cmd.append(dirname)
    try:
        result: Tuple[bytes, bytes] = run_cmd(cwd, clone_cmd, env={'LC_ALL': 'C'})
    except CommandResultError as exc:
        raise GitCloningError(repo, revision, exc)
    if subdirectory:
        cwd_subdir: str = os.path.join(cwd, dirname or '')
        clone_cmd_subdir: List[str] = ['git', 'sparse-checkout', 'set', subdirectory]
        try:
            run_cmd(cwd_subdir, clone_cmd_subdir)
        except CommandResultError as exc:
            raise GitCloningError(repo, revision, exc)
    if remove_git_dir:
        rmdir(os.path.join(dirname, '.git'))
    return result

def list_tags(cwd: str) -> List[str]:
    out, err = run_cmd(cwd, ['git', 'tag', '--list'], env={'LC_ALL': 'C'})
    tags: List[str] = out.decode('utf-8').strip().split('\n')
    return tags

def _checkout(cwd: str, repo: str, revision: str) -> Tuple[bytes, bytes]:
    fire_event(GitProgressCheckoutRevision(revision=revision))
    fetch_cmd: List[str] = ['git', 'fetch', 'origin', '--depth', '1']
    if _is_commit(revision):
        run_cmd(cwd, fetch_cmd + [revision])
    else:
        run_cmd(cwd, ['git', 'remote', 'set-branches', 'origin', revision])
        run_cmd(cwd, fetch_cmd + ['--tags', revision])
    if _is_commit(revision):
        spec: str = revision
    elif revision in list_tags(cwd):
        spec = 'tags/{}'.format(revision)
    else:
        spec = 'origin/{}'.format(revision)
    out, err = run_cmd(cwd, ['git', 'reset', '--hard', spec], env={'LC_ALL': 'C'})
    return (out, err)

def checkout(cwd: str, repo: str, revision: Optional[str] = None) -> Tuple[bytes, bytes]:
    if revision is None:
        revision = 'HEAD'
    try:
        return _checkout(cwd, repo, revision)
    except CommandResultError as exc:
        raise GitCheckoutError(repo=repo, revision=revision, error=exc)

def get_current_sha(cwd: str) -> str:
    out, err = run_cmd(cwd, ['git', 'rev-parse', 'HEAD'], env={'LC_ALL': 'C'})
    return out.decode('utf-8').strip()

def remove_remote(cwd: str) -> Tuple[bytes, bytes]:
    return run_cmd(cwd, ['git', 'remote', 'rm', 'origin'], env={'LC_ALL': 'C'})

def clone_and_checkout(repo: str, cwd: str, dirname: Optional[str] = None, remove_git_dir: bool = False, revision: Optional[str] = None, subdirectory: Optional[str] = None) -> str:
    exists: Optional[Match[str]] = None
    try:
        _, err = clone(repo, cwd, dirname=dirname, remove_git_dir=remove_git_dir, subdirectory=subdirectory)
    except CommandResultError as exc:
        err = exc.stderr
        exists = re.match("fatal: destination path '(.+)' already exists", err)
        if not exists:
            raise UnknownGitCloningProblemError(repo)
    directory: Optional[str] = None
    start_sha: Optional[str] = None
    if exists:
        directory = exists.group(1)
        fire_event(GitProgressUpdatingExistingDependency(dir=directory))
    else:
        matches: Optional[Match[str]] = re.match("Cloning into '(.+)'", err.decode('utf-8'))
        if matches is None:
            raise DbtRuntimeError(f'Error cloning {repo} - never saw "Cloning into ..." from git')
        directory = matches.group(1)
        fire_event(GitProgressPullingNewDependency(dir=directory))
    full_path: str = os.path.join(cwd, directory)
    start_sha = get_current_sha(full_path)
    checkout(full_path, repo, revision)
    end_sha: str = get_current_sha(full_path)
    if exists:
        if start_sha == end_sha:
            fire_event(GitNothingToDo(sha=start_sha[:7]))
        else:
            fire_event(GitProgressUpdatedCheckoutRange(start_sha=start_sha[:7], end_sha=end_sha[:7]))
    else:
        fire_event(GitProgressCheckedOutAt(end_sha=end_sha[:7]))
    return os.path.join(directory, subdirectory or '')
