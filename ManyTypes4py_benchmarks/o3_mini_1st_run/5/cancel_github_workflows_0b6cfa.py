#!/usr/bin/env python3
"""
Manually cancel previous GitHub Action workflow runs in queue.

Example:
  # Set up
  export GITHUB_TOKEN={{ your personal github access token }}
  export GITHUB_REPOSITORY=apache/superset

  # cancel previous jobs for a PR, will even cancel the running ones
  ./cancel_github_workflows.py 1042

  # cancel previous jobs for a branch
  ./cancel_github_workflows.py my-branch

  # cancel all jobs of a PR, including the latest runs
  ./cancel_github_workflows.py 1024 --include-last
"""
import os
from collections.abc import Iterator
from typing import Any, Optional, Union, Iterable, List, Tuple, Dict
import click
import requests
from click.exceptions import ClickException
from dateutil import parser

github_token: Optional[str] = os.environ.get('GITHUB_TOKEN')
github_repo: str = os.environ.get('GITHUB_REPOSITORY', 'apache/superset')

def request(method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
    resp: Dict[str, Any] = requests.request(method, f'https://api.github.com/{endpoint.lstrip("/")}', headers={'Authorization': f'Bearer {github_token}'}, **kwargs).json()
    if 'message' in resp:
        raise ClickException(f"{endpoint} >> {resp['message']} <<")
    return resp

def list_runs(repo: str, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """List all github workflow runs.
    Returns:
      An iterator that will iterate through all pages of matching runs."""
    if params is None:
        params = {}
    page: int = 1
    total_count: int = 10000
    while page * 100 < total_count:
        result: Dict[str, Any] = request('GET', f'/repos/{repo}/actions/runs', params={**params, 'per_page': 100, 'page': page})
        total_count = result['total_count']
        yield from result['workflow_runs']
        page += 1

def cancel_run(repo: str, run_id: Union[int, str]) -> Dict[str, Any]:
    return request('POST', f'/repos/{repo}/actions/runs/{run_id}/cancel')

def get_pull_request(repo: str, pull_number: Union[int, str]) -> Dict[str, Any]:
    return request('GET', f'/repos/{repo}/pulls/{pull_number}')

def get_runs(
    repo: str, 
    branch: Optional[str] = None, 
    user: Optional[str] = None, 
    statuses: Tuple[str, ...] = ('queued', 'in_progress'), 
    events: Tuple[str, ...] = ('pull_request', 'push')
) -> List[Dict[str, Any]]:
    """Get workflow runs associated with the given branch"""
    return [
        item
        for event in events
        for status in statuses
        for item in list_runs(repo, {'event': event, 'status': status})
        if (branch is None or branch == item['head_branch']) and (user is None or user == item['head_repository']['owner']['login'])
    ]

def print_commit(commit: Dict[str, Any], branch: str) -> None:
    """Print out commit message for verification"""
    indented_message: str = '    \n'.join(commit['message'].split('\n'))
    date_str: str = parser.parse(commit['timestamp']).astimezone(tz=None).strftime('%a, %d %b %Y %H:%M:%S')
    print(f"HEAD {commit['id']} ({branch})\nAuthor: {commit['author']['name']} <{commit['author']['email']}>\nDate:   {date_str}\n\n    {indented_message}\n")

@click.command()
@click.option('--repo', default=github_repo, help='The github repository name. For example, apache/superset.')
@click.option('--event', type=click.Choice(['pull_request', 'push', 'issue']), default=['pull_request', 'push'], show_default=True, multiple=True)
@click.option('--include-last/--skip-last', default=False, show_default=True, help='Whether to also cancel the latest run.')
@click.option('--include-running/--skip-running', default=True, show_default=True, help='Whether to also cancel running workflows.')
@click.argument('branch_or_pull', required=False)
def cancel_github_workflows(
    branch_or_pull: Optional[str],
    repo: str,
    event: Tuple[str, ...],
    include_last: bool,
    include_running: bool
) -> None:
    """Cancel running or queued GitHub workflows by branch or pull request ID"""
    if not github_token:
        raise ClickException('Please provide GITHUB_TOKEN as an env variable')
    statuses: Tuple[str, ...] = ('queued', 'in_progress') if include_running else ('queued',)
    events: Tuple[str, ...] = event
    pr: Optional[Dict[str, Any]] = None
    if branch_or_pull is None:
        title: str = 'all jobs' if include_last else 'all duplicate jobs'
    elif branch_or_pull.isdigit():
        pr = get_pull_request(repo, pull_number=branch_or_pull)
        title = f"pull request #{pr['number']} - {pr['title']}"
    else:
        title = f"branch [{branch_or_pull}]"
    print(f'\nCancel {("active" if include_running else "previous")} workflow runs for {title}\n')
    if pr:
        runs: List[Dict[str, Any]] = get_runs(repo, statuses=statuses, events=event, branch=pr['head']['ref'], user=pr['user']['login'])
    else:
        user: Optional[str] = None
        branch: Optional[str] = branch_or_pull
        if branch and ':' in branch:
            parts = branch.split(':', 2)
            if len(parts) == 2:
                user, branch = parts
            else:
                user, branch = parts[0], parts[1]
        runs = get_runs(repo, branch=branch, user=user, statuses=statuses, events=events)
    runs.sort(key=lambda x: x['created_at'])
    if runs:
        print(f'Found {len(runs)} potential runs of\n   status: {statuses}\n   event: {events}\n')
    else:
        print(f'No {" or ".join(statuses)} workflow runs found.\n')
        return
    if not include_last:
        seen: set[str] = set()
        dups: List[Dict[str, Any]] = []
        for item in reversed(runs):
            key: str = f"{item['event']}_{item['head_branch']}_{item['workflow_id']}"
            if key in seen:
                dups.append(item)
            else:
                seen.add(key)
        if not dups:
            print('Only the latest runs are in queue. Use --include-last to force cancelling them.\n')
            return
        runs = dups[::-1]
    last_sha: Optional[str] = None
    print(f'\nCancelling {len(runs)} jobs...\n')
    for entry in runs:
        head_commit: Dict[str, Any] = entry['head_commit']
        if head_commit['id'] != last_sha:
            last_sha = head_commit['id']
            print('')
            print_commit(head_commit, entry['head_branch'])
        try:
            print(f"[{entry['status']}] {entry['name']}", end='\r')
            cancel_run(repo, entry['id'])
            print(f"[Canceled] {entry['name']}     ")
        except ClickException as error:
            print(f"[Error: {error.message}] {entry['name']}    ")
    print('')

if __name__ == '__main__':
    cancel_github_workflows()