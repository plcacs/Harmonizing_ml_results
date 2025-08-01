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
from typing import Any, Final, Literal, List, Dict
import click
import urllib3
from packaging.version import Version

COMMENT_FILE: Final[str] = '.pr-comment.json'
DIFF_STEP_NAME: Final[str] = 'Generate HTML diff report'
DOCS_URL: Final[str] = 'https://black.readthedocs.io/en/latest/contributing/gauging_changes.html#diff-shades'
USER_AGENT: Final[str] = f'psf/black diff-shades workflow via urllib3/{urllib3.__version__}'
SHA_LENGTH: Final[int] = 10
GH_API_TOKEN: Final[str] = os.getenv('GITHUB_TOKEN', '')
REPO: Final[str] = os.getenv('GITHUB_REPOSITORY', default='psf/black')
http = urllib3.PoolManager()

def set_output(name: str, value: str) -> None:
    if len(value) < 200:
        print(f"[INFO]: setting '{name}' to '{value}'")
    else:
        print(f"[INFO]: setting '{name}' to [{len(value)} chars]")
    if 'GITHUB_OUTPUT' in os.environ:
        if '\n' in value:
            delimiter = b64encode(os.urandom(16)).decode()
            value = f'{delimiter}\n{value}\n{delimiter}'
            command = f'{name}<<{value}'
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
    r = http.request('GET', url, **kwargs)
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
    data = http_get(f'https://api.github.com/repos/{REPO}/commits', fields={'per_page': '1', 'sha': 'main'})
    assert isinstance(data[0]['sha'], str)
    return data[0]['sha']

def get_pr_revision(pr: int) -> str:
    data = http_get(f'https://api.github.com/repos/{REPO}/pulls/{pr}')
    assert isinstance(data['head']['sha'], str)
    return data['head']['sha']

def get_pypi_version() -> Version:
    data = http_get('https://pypi.org/pypi/black/json')
    versions = [Version(v) for v in data['releases']]
    sorted_versions = sorted(versions, reverse=True)
    return sorted_versions[0]

@click.group()
def main() -> None:
    pass

@main.command('config', help='Acquire run configuration and metadata.')
@click.argument('event', type=click.Choice(['push', 'pull_request']))
def config(event: str) -> None:
    import diff_shades
    if event == 'push':
        jobs: List[Dict[str, Any]] = [{'mode': 'preview-changes', 'force-flag': '--force-preview-style'}]
        baseline_name = str(get_pypi_version())
        baseline_cmd = f'git checkout {baseline_name}'
        target_rev = os.getenv('GITHUB_SHA')
        assert target_rev is not None
        target_name = 'main-' + target_rev[:SHA_LENGTH]
        target_cmd = f'git checkout {target_rev}'
    elif event == 'pull_request':
        jobs = [{'mode': 'preview-changes', 'force-flag': '--force-preview-style'}, {'mode': 'assert-no-changes', 'force-flag': '--force-stable-style'}]
        baseline_rev = get_main_revision()
        baseline_name = 'main-' + baseline_rev[:SHA_LENGTH]
        baseline_cmd = f'git checkout {baseline_rev}'
        pr_ref = os.getenv('GITHUB_REF')
        assert pr_ref is not None
        pr_num = int(pr_ref[10:-6])
        pr_rev = get_pr_revision(pr_num)
        target_name = f'pr-{pr_num}-{pr_rev[:SHA_LENGTH]}'
        target_cmd = f'gh pr checkout {pr_num} && git merge origin/main'
    env = f'{platform.system()}-{platform.python_version()}-{diff_shades.__version__}'
    for entry in jobs:
        entry['baseline-analysis'] = f'{entry["mode"]}-{baseline_name}.json'
        entry['baseline-setup-cmd'] = baseline_cmd
        entry['target-analysis'] = f'{entry["mode"]}-{target_name}.json'
        entry['target-setup-cmd'] = target_cmd
        entry['baseline-cache-key'] = f'{env}-{baseline_name}-{entry["mode"]}'
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
    cmd = [sys.executable, '-m', 'diff_shades', '--no-color', 'compare', str(baseline), str(target), '--quiet', '--check']
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, encoding='utf-8')
    if not proc.returncode:
        body = f'**diff-shades** reports zero changes comparing this PR ({target_sha}) to main ({baseline_sha}).\n\n---\n\n'
    else:
        body = f'**diff-shades** results comparing this PR ({target_sha}) to main ({baseline_sha}). The full diff is [available in the logs]($job-diff-url) under the "{DIFF_STEP_NAME}" step.'
        body += '\n