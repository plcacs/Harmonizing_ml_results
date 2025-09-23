"""Helper script for psf/black's diff-shades Github Actions integration.

diff-shades is a tool for analyzing what happens when you run Black on
OSS code capturing it for comparisons or other usage. It's used here to
help measure the impact of a change *before* landing it (in particular
posting a comment on completion for PRs).

This script exists as a more maintainable alternative to using inline
Javascript in the workflow YAML files. The revision configuration and
resolving, caching, and PR comment logic is contained here.

For more information, please see the developer docs:

https://black.readthedocs.io/en/latest/contributing/gauging_changes.html#diff-shades
"""
import json
import os
import platform
import pprint
import subprocess
import sys
import zipfile
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Any, Final, Literal, Dict, List, Optional, Union
import click
import urllib3
from packaging.version import Version
COMMENT_FILE: Final[str] = '.pr-comment.json'
DIFF_STEP_NAME: Final[str] = 'Generate HTML diff report'
DOCS_URL: Final[str] = 'https://black.readthedocs.io/en/latest/contributing/gauging_changes.html#diff-shades'
USER_AGENT: Final[str] = f'psf/black diff-shades workflow via urllib3/{urllib3.__version__}'
SHA_LENGTH: Final[int] = 10
GH_API_TOKEN: Final[Optional[str]] = os.getenv('GITHUB_TOKEN')
REPO: Final[str] = os.getenv('GITHUB_REPOSITORY', default='psf/black')
http: Final[urllib3.PoolManager] = urllib3.PoolManager()

def set_output(name: str, value: str) -> None:
    if len(value) < 200:
        print(f"[INFO]: setting '{name}' to '{value}'")
    else:
        print(f"[INFO]: setting '{name}' to [{len(value)} chars]")
    if 'GITHUB_OUTPUT' in os.environ:
        if '\n' in value:
            delimiter: str = b64encode(os.urandom(16)).decode()
            value = f'{delimiter}\n{value}\n{delimiter}'
            command: str = f'{name}<<{value}'
        else:
            command = f'{name}={value}'
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            print(command, file=f)

def http_get(url: str, *, is_json: bool = True, **kwargs: Any) -> Any:
    headers: Dict[str, str] = kwargs.get('headers') or {}
    headers['User-Agent'] = USER_AGENT
    if 'github' in url:
        if GH_API_TOKEN:
            headers['Authorization'] = f'token {GH_API_TOKEN}'
        headers['Accept'] = 'application/vnd.github.v3+json'
    kwargs['headers'] = headers
    r: urllib3.HTTPResponse = http.request('GET', url, **kwargs)
    data: Any
    if is_json:
        data = json.loads(r.data.decode('utf-8'))
    else:
        data = r.data
    print(f'[INFO]: issued GET request for {r.geturl()}')
    if not 200 <= r.status < 300:
        pprint.pprint(dict(r.info()))
        pprint.pprint(data)
        raise RuntimeError(f'unexpected status code: {r.status}')
    return data

def get_main_revision() -> str:
    data: List[Dict[str, Any]] = http_get(f'https://api.github.com/repos/{REPO}/commits', fields={'per_page': '1', 'sha': 'main'})
    assert isinstance(data[0]['sha'], str)
    return data[0]['sha']

def get_pr_revision(pr: int) -> str:
    data: Dict[str, Any] = http_get(f'https://api.github.com/repos/{REPO}/pulls/{pr}')
    assert isinstance(data['head']['sha'], str)
    return data['head']['sha']

def get_pypi_version() -> Version:
    data: Dict[str, Any] = http_get('https://pypi.org/pypi/black/json')
    versions: List[Version] = [Version(v) for v in data['releases']]
    sorted_versions: List[Version] = sorted(versions, reverse=True)
    return sorted_versions[0]

@click.group()
def main() -> None:
    pass

@main.command('config', help='Acquire run configuration and metadata.')
@click.argument('event', type=click.Choice(['push', 'pull_request']))
def config(event: Literal['push', 'pull_request']) -> None:
    import diff_shades
    jobs: List[Dict[str, str]] = []
    if event == 'push':
        jobs = [{'mode': 'preview-changes', 'force-flag': '--force-preview-style'}]
        baseline_name: str = str(get_pypi_version())
        baseline_cmd: str = f'git checkout {baseline_name}'
        target_rev: Optional[str] = os.getenv('GITHUB_SHA')
        assert target_rev is not None
        target_name: str = 'main-' + target_rev[:SHA_LENGTH]
        target_cmd: str = f'git checkout {target_rev}'
    elif event == 'pull_request':
        jobs = [{'mode': 'preview-changes', 'force-flag': '--force-preview-style'}, {'mode': 'assert-no-changes', 'force-flag': '--force-stable-style'}]
        baseline_rev: str = get_main_revision()
        baseline_name = 'main-' + baseline_rev[:SHA_LENGTH]
        baseline_cmd = f'git checkout {baseline_rev}'
        pr_ref: Optional[str] = os.getenv('GITHUB_REF')
        assert pr_ref is not None
        pr_num: int = int(pr_ref[10:-6])
        pr_rev: str = get_pr_revision(pr_num)
        target_name = f'pr-{pr_num}-{pr_rev[:SHA_LENGTH]}'
        target_cmd = f'gh pr checkout {pr_num} && git merge origin/main'
    env: str = f'{platform.system()}-{platform.python_version()}-{diff_shades.__version__}'
    for entry in jobs:
        entry['baseline-analysis'] = f"{entry['mode']}-{baseline_name}.json"
        entry['baseline-setup-cmd'] = baseline_cmd
        entry['target-analysis'] = f"{entry['mode']}-{target_name}.json"
        entry['target-setup-cmd'] = target_cmd
        entry['baseline-cache-key'] = f"{env}-{baseline_name}-{entry['mode']}"
        if event == 'pull_request':
            entry['baseline-sha'] = baseline_rev
            entry['target-sha'] = pr_rev
    set_output('matrix', json.dumps(jobs, indent=None))
    pprint.pprint(jobs)

@main.command('comment-body', help='Generate the body for a summary PR comment.')
@click.argument('baseline', type=click.Path(exists=True, path_type=Path))
@click.argument('target', type=click.Path(exists=True, path_type=Path))
@click.argument('baseline-sha')
@click.argument('target-sha')
@click.argument('pr-num', type=int)
def comment_body(baseline: Path, target: Path, baseline_sha: str, target_sha: str, pr_num: int) -> None:
    cmd: List[str] = [sys.executable, '-m', 'diff_shades', '--no-color', 'compare', str(baseline), str(target), '--quiet', '--check']
    proc: subprocess.CompletedProcess = subprocess.run(cmd, stdout=subprocess.PIPE, encoding='utf-8')
    body: str
    if not proc.returncode:
        body = f'**diff-shades** reports zero changes comparing this PR ({target_sha}) to main ({baseline_sha}).\n\n---\n\n'
    else:
        body = f'**diff-shades** results comparing this PR ({target_sha}) to main ({baseline_sha}). The full diff is [available in the logs]($job-diff-url) under the "{DIFF_STEP_NAME}" step.'
        body += '\n