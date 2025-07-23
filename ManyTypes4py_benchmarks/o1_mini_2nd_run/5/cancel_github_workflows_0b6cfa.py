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
from typing import Any, Dict, List, Optional, Set, Tuple
import click
import requests
from click.exceptions import ClickException
from dateutil import parser

github_token: Optional[str] = os.environ.get('GITHUB_TOKEN')
github_repo: str = os.environ.get('GITHUB_REPOSITORY', 'apache/superset')


def request(method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
    resp: Dict[str, Any] = requests.request(
        method,
        f'https://api.github.com/{endpoint.lstrip("/")}',
        headers={'Authorization': f'Bearer {github_token}'},
        **kwargs
    ).json()
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
        result: Dict[str, Any] = request(
            'GET',
            f'/repos/{repo}/actions/runs',
            params={**params, 'per_page': 100, 'page': page}
        )
        total_count = result['total_count']
        workflow_runs: List[Dict[str, Any]] = result.get('workflow_runs', [])
        yield from workflow_runs
        page += 1


def cancel_run(repo: str, run_id: int) -> Dict[str, Any]:
    return request('POST', f'/repos/{repo}/actions/runs/{run_id}/cancel')


def get_pull_request(repo: str, pull_number: Union[str, int]) -> Dict[str, Any]:
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
        if (branch is None or branch == item.get('head_branch'))
        and (user is None or user == item.get('head_repository', {}).get('owner', {}).get('login'))
    ]


def print_commit(commit: Dict[str, Any], branch: str) -> None:
    """Print out commit message for verification"""
    indented_message: str = '    \n'.join(commit.get('message', '').split('\n'))
    date_str: str = parser.parse(commit.get('timestamp', '')).astimezone(tz=None).strftime('%a, %d %b %Y %H:%M:%S')
    print(
        f"HEAD {commit.get('id')} ({branch})\n"
        f"Author: {commit.get('author', {}).get('name')} <{commit.get('author', {}).get('email')}>\n"
        f"Date:   {date_str}\n\n"
        f"    {indented_message}\n"
    )


@click.command()
@click.option(
    '--repo',
    default=github_repo,
    help='The github repository name. For example, apache/superset.',
    type=str
)
@click.option(
    '--event',
    type=click.Choice(['pull_request', 'push', 'issue'], case_sensitive=False),
    default=['pull_request', 'push'],
    show_default=True,
    multiple=True
)
@click.option(
    '--include-last/--skip-last',
    default=False,
    show_default=True,
    help='Whether to also cancel the latest run.'
)
@click.option(
    '--include-running/--skip-running',
    default=True,
    show_default=True,
    help='Whether to also cancel running workflows.'
)
@click.argument('branch_or_pull', required=False, type=str)
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
    events_selected: Tuple[str, ...] = event
    pr: Optional[Dict[str, Any]] = None
    if branch_or_pull is None:
        title: str = 'all jobs' if include_last else 'all duplicate jobs'
    elif branch_or_pull.isdigit():
        pr = get_pull_request(repo, pull_number=int(branch_or_pull))
        title = f"pull request #{pr.get('number')} - {pr.get('title')}"
    else:
        title = f"branch [{branch_or_pull}]"
    print(f"\nCancel {'active' if include_running else 'previous'} workflow runs for {title}\n")
    if pr:
        runs: List[Dict[str, Any]] = get_runs(
            repo,
            statuses=statuses,
            events=events_selected,
            branch=pr.get('head', {}).get('ref'),
            user=pr.get('user', {}).get('login')
        )
    else:
        user: Optional[str] = None
        branch: Optional[str] = branch_or_pull
        if branch and ':' in branch:
            user, branch = branch.split(':', 1)
        runs = get_runs(
            repo,
            branch=branch,
            user=user,
            statuses=statuses,
            events=events_selected
        )
    runs_sorted: List[Dict[str, Any]] = sorted(runs, key=lambda x: x.get('created_at', ''))
    if runs_sorted:
        print(
            f"Found {len(runs_sorted)} potential runs of\n"
            f"   status: {statuses}\n"
            f"   event: {events_selected}\n"
        )
    else:
        print(f"No {' or '.join(statuses)} workflow runs found.\n")
        return
    if not include_last:
        seen: Set[str] = set()
        dups: List[Dict[str, Any]] = []
        for item in reversed(runs_sorted):
            key = f"{item.get('event')}_{item.get('head_branch')}_{item.get('workflow_id')}"
            if key in seen:
                dups.append(item)
            else:
                seen.add(key)
        if not dups:
            print('Only the latest runs are in queue. Use --include-last to force cancelling them.\n')
            return
        runs_sorted = dups[::-1]
    last_sha: Optional[str] = None
    print(f"\nCancelling {len(runs_sorted)} jobs...\n")
    for entry in runs_sorted:
        head_commit: Dict[str, Any] = entry.get('head_commit', {})
        if head_commit.get('id') != last_sha:
            last_sha = head_commit.get('id')
            print('')
            print_commit(head_commit, entry.get('head_branch', 'unknown'))
        try:
            print(f"[{entry.get('status')}] {entry.get('name')}", end='\r')
            cancel_run(repo, int(entry.get('id')))
            print(f"[Canceled] {entry.get('name')}     ")
        except ClickException as error:
            print(f"[Error: {error.message}] {entry.get('name')}    ")
    print('')


if __name__ == '__main__':
    cancel_github_workflows()
