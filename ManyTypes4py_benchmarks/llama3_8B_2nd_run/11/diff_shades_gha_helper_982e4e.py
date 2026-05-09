from typing import Any, Final, Literal
import click
import urllib3
from packaging.version import Version
import os
import json
import platform
import subprocess
import sys
import zipfile
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import pprint

COMMENT_FILE: Final[str] = '.pr-comment.json'
DIFF_STEP_NAME: Final[str] = 'Generate HTML diff report'
DOCS_URL: Final[str] = 'https://black.readthedocs.io/en/latest/contributing/gauging_changes.html#diff-shades'
USER_AGENT: Final[str] = f'psf/black diff-shades workflow via urllib3/{urllib3.__version__}'
SHA_LENGTH: Final[int] = 10
GH_API_TOKEN: Final[str] = os.getenv('GITHUB_TOKEN')
REPO: Final[str] = os.getenv('GITHUB_REPOSITORY', default='psf/black')

def set_output(name: str, value: Any) -> None:
    ...

def http_get(url: str, *, is_json: bool = True, **kwargs: Any) -> Any:
    ...

def get_main_revision() -> str:
    ...

def get_pr_revision(pr: int) -> str:
    ...

def get_pypi_version() -> Version:
    ...

@click.group()
def main() -> None:
    ...

@main.command('config', help='Acquire run configuration and metadata.')
@click.argument('event', type=click.Choice(['push', 'pull_request']))
def config(event: str) -> None:
    ...

@main.command('comment-body', help='Generate the body for a summary PR comment.')
@click.argument('baseline', type=click.Path(exists=True, path_type=Path))
@click.argument('target', type=click.Path(exists=True, path_type=Path))
@click.argument('baseline-sha', type=str)
@click.argument('target-sha', type=str)
@click.argument('pr-num', type=int)
def comment_body(baseline: Path, target: Path, baseline_sha: str, target_sha: str, pr_num: int) -> None:
    ...

@main.command('comment-details', help='Get PR comment resources from a workflow run.')
@click.argument('run-id', type=str)
def comment_details(run_id: str) -> None:
    ...
