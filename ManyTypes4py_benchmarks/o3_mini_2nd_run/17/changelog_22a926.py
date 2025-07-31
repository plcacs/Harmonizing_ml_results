#!/usr/bin/env python3
import csv as lib_csv
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import click
from click.core import Context
try:
    from github import BadCredentialsException, Github, PullRequest, Repository
except ModuleNotFoundError:
    print('PyGitHub is a required package for this script')
    exit(1)

SUPERSET_REPO: str = 'apache/superset'
SUPERSET_PULL_REQUEST_TYPES: str = '^(fix|feat|chore|refactor|docs|build|ci|/gmi)'
SUPERSET_RISKY_LABELS: str = '^(blocking|risk|hold|revert|security vulnerability)'

@dataclass
class GitLog:
    sha: str
    author: str
    author_email: str
    time: str
    message: str
    pr_number: Optional[int] = None

    def __eq__(self, other: object) -> bool:
        """A log entry is considered equal if it has the same PR number"""
        if isinstance(other, GitLog):
            return other.pr_number == self.pr_number
        return False

    def __repr__(self) -> str:
        return f'[{self.pr_number}]: {self.message} {self.time} {self.author}'

class GitChangeLog:
    def __init__(self, version: str, logs: List[GitLog], access_token: Optional[str] = None, risk: bool = False) -> None:
        self._version: str = version
        self._logs: List[GitLog] = logs
        self._pr_logs_with_details: Dict[int, Dict[str, Union[int, bool, str]]] = {}
        self._github_login_cache: Dict[str, str] = {}
        self._github_prs: Dict[int, PullRequest] = {}
        self._wait: int = 10
        github_token: Optional[str] = access_token or os.environ.get('GITHUB_TOKEN')
        self._github: Github = Github(github_token)
        self._show_risk: bool = risk
        self._superset_repo: Optional[Repository] = None

    def _fetch_github_pr(self, pr_number: int) -> PullRequest:
        """
        Fetches a github PR info
        """
        try:
            github_repo: Repository = self._github.get_repo(SUPERSET_REPO)
            self._superset_repo = github_repo
            pull_request: Optional[PullRequest] = self._github_prs.get(pr_number)
            if not pull_request:
                pull_request = github_repo.get_pull(pr_number)
                self._github_prs[pr_number] = pull_request
        except BadCredentialsException:
            print('Bad credentials to github provided use access_token parameter or set GITHUB_TOKEN')
            sys.exit(1)
        return pull_request

    def _get_github_login(self, git_log: GitLog) -> str:
        """
        Tries to fetch a github login (username) from a git author
        """
        author_name: str = git_log.author
        github_login: Optional[str] = self._github_login_cache.get(author_name)
        if github_login:
            return github_login
        if git_log.pr_number:
            pr_info: PullRequest = self._fetch_github_pr(git_log.pr_number)
            if pr_info:
                github_login = pr_info.user.login
            else:
                github_login = author_name
        self._github_login_cache[author_name] = github_login
        return github_login

    def _has_commit_migrations(self, git_sha: str) -> bool:
        assert self._superset_repo is not None
        commit = self._superset_repo.get_commit(sha=git_sha)
        return any('superset/migrations/versions/' in file.filename for file in commit.files)

    def _get_pull_request_details(self, git_log: GitLog) -> Dict[str, Union[int, bool, str]]:
        pr_number: Optional[int] = git_log.pr_number
        if pr_number:
            detail: Optional[Dict[str, Union[int, bool, str]]] = self._pr_logs_with_details.get(pr_number)
            if detail:
                return detail
            pr_info: PullRequest = self._fetch_github_pr(pr_number)
        else:
            pr_info = None
        has_migrations: bool = self._has_commit_migrations(git_log.sha)
        title: str = pr_info.title if pr_info else git_log.message
        pr_type_match = re.match(SUPERSET_PULL_REQUEST_TYPES, title)
        pr_type: str = pr_type_match.group().strip('"') if pr_type_match else ""
        labels: str = ' | '.join([label.name for label in pr_info.labels]) if pr_info else ""
        is_risky: bool = self._is_risk_pull_request(pr_info.labels) if pr_info else False
        detail = {
            'id': pr_number if pr_number else 0, 
            'has_migrations': has_migrations, 
            'labels': labels, 
            'title': title, 
            'type': pr_type, 
            'is_risky': is_risky or has_migrations
        }
        if pr_number:
            self._pr_logs_with_details[pr_number] = detail
        return detail

    def _is_risk_pull_request(self, labels: List[Any]) -> bool:
        for label in labels:
            risk_label = re.match(SUPERSET_RISKY_LABELS, label.name)
            if risk_label is not None:
                return True
        return False

    def _get_changelog_version_head(self) -> str:
        if not len(self._logs):
            print('No changes found between revisions. Make sure your branch is up to date.')
            sys.exit(1)
        return f'### {self._version} ({self._logs[0].time})'

    def _parse_change_log(self, changelog: Dict[str, str], pr_info: Dict[str, Union[int, bool, str]], github_login: str) -> None:
        formatted_pr: str = (
            f"- [#{pr_info.get('id')}](https://github.com/{SUPERSET_REPO}/pull/{pr_info.get('id')}) "
            f"{pr_info.get('title')} (@{github_login})\n"
        )
        if pr_info.get('has_migrations'):
            changelog['Database Migrations'] += formatted_pr
        elif pr_info.get('type') == 'fix':
            changelog['Fixes'] += formatted_pr
        elif pr_info.get('type') == 'feat':
            changelog['Features'] += formatted_pr
        else:
            changelog['Others'] += formatted_pr

    def __repr__(self) -> str:
        result: str = f'\n{self._get_changelog_version_head()}\n'
        changelog: Dict[str, str] = {'Database Migrations': '\n', 'Features': '\n', 'Fixes': '\n', 'Others': '\n'}
        for i, log in enumerate(self._logs):
            github_login: str = self._get_github_login(log)
            pr_info: Dict[str, Union[int, bool, str]] = self._get_pull_request_details(log)
            if not github_login:
                github_login = log.author
            if self._show_risk:
                if pr_info.get('is_risky'):
                    result += f"- [#{log.pr_number}](https://github.com/{SUPERSET_REPO}/pull/{log.pr_number}) {pr_info.get('title')} (@{github_login})  {pr_info.get('labels')} \n"
            else:
                self._parse_change_log(changelog, pr_info, github_login)
            print(f'\r {i}/{len(self._logs)}', end='', flush=True)
        if self._show_risk:
            return result
        for key in changelog:
            result += f'**{key}** {changelog[key]}\n'
        return result

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for log in self._logs:
            yield {
                'pr_number': log.pr_number,
                'pr_link': f'https://github.com/{SUPERSET_REPO}/pull/{log.pr_number}',
                'message': log.message,
                'time': log.time,
                'author': log.author,
                'email': log.author_email,
                'sha': log.sha
            }

class GitLogs:
    def __init__(self, git_ref: str) -> None:
        self._git_ref: str = git_ref
        self._logs: List[GitLog] = []

    @property
    def git_ref(self) -> str:
        return self._git_ref

    @property
    def logs(self) -> List[GitLog]:
        return self._logs

    def fetch(self) -> None:
        self._logs = list(map(self._parse_log, self._git_logs()))[::-1]

    def diff(self, git_logs: 'GitLogs') -> List[GitLog]:
        return [log for log in git_logs.logs if log not in self._logs]

    def __repr__(self) -> str:
        return f'{self._git_ref}, Log count:{len(self._logs)}'

    @staticmethod
    def _git_get_current_head() -> str:
        output: str = os.popen('git status | head -1').read()
        match = re.match(r'(?:HEAD detached at|On branch) (.*)', output)
        if not match:
            return ''
        return match.group(1)

    def _git_checkout(self, git_ref: str) -> None:
        os.popen(f'git checkout {git_ref}').read()
        current_head: str = self._git_get_current_head()
        if current_head != git_ref:
            print(f'Could not checkout {git_ref}')
            sys.exit(1)

    def _git_logs(self) -> List[str]:
        current_git_ref: str = self._git_get_current_head()
        self._git_checkout(self._git_ref)
        output: List[str] = os.popen('git --no-pager log --pretty=format:"%h|%an|%ae|%ad|%s|"').read().split('\n')
        self._git_checkout(current_git_ref)
        return output

    @staticmethod
    def _parse_log(log_item: str) -> GitLog:
        pr_number: Optional[int] = None
        split_log_item: List[str] = log_item.split('|')
        match = re.match(r'.*\(#(\d*)\)', split_log_item[4])
        if match:
            pr_number = int(match.group(1))
        return GitLog(
            sha=split_log_item[0],
            author=split_log_item[1],
            author_email=split_log_item[2],
            time=split_log_item[3],
            message=split_log_item[4],
            pr_number=pr_number
        )

@dataclass
class BaseParameters:
    previous_logs: GitLogs
    current_logs: GitLogs

def print_title(message: str) -> None:
    print(f'{50 * "-"}')
    print(message)
    print(f'{50 * "-"}')

@click.group()
@click.pass_context
@click.option('--previous_version', help='The previous release version', required=True)
@click.option('--current_version', help='The current release version', required=True)
def cli(ctx: Context, previous_version: str, current_version: str) -> None:
    """Welcome to change log generator"""
    previous_logs: GitLogs = GitLogs(previous_version)
    current_logs: GitLogs = GitLogs(current_version)
    previous_logs.fetch()
    current_logs.fetch()
    base_parameters: BaseParameters = BaseParameters(previous_logs, current_logs)
    ctx.obj = base_parameters

@cli.command('compare')
@click.pass_obj
def compare(base_parameters: BaseParameters) -> None:
    """Compares both versions (by PR)"""
    previous_logs: GitLogs = base_parameters.previous_logs
    current_logs: GitLogs = base_parameters.current_logs
    print_title(f'Pull requests from {current_logs.git_ref} not in {previous_logs.git_ref}')
    previous_diff_logs: List[GitLog] = previous_logs.diff(current_logs)
    for diff_log in previous_diff_logs:
        print(f'{diff_log}')
    print_title(f'Pull requests from {previous_logs.git_ref} not in {current_logs.git_ref}')
    current_diff_logs: List[GitLog] = current_logs.diff(previous_logs)
    for diff_log in current_diff_logs:
        print(f'{diff_log}')

@cli.command('changelog')
@click.option('--csv', help='The csv filename to export the changelog to')
@click.option('--access_token', help='The github access token, if not provided will try to fetch from GITHUB_TOKEN env var')
@click.option('--risk', is_flag=True, help='show all pull requests with risky labels')
@click.pass_obj
def change_log(base_parameters: BaseParameters, csv: Optional[str], access_token: Optional[str], risk: bool) -> None:
    """Outputs a changelog (by PR)"""
    previous_logs: GitLogs = base_parameters.previous_logs
    current_logs: GitLogs = base_parameters.current_logs
    previous_diff_logs: List[GitLog] = previous_logs.diff(current_logs)
    logs: GitChangeLog = GitChangeLog(current_logs.git_ref, previous_diff_logs[::-1], access_token=access_token, risk=risk)
    if csv:
        with open(csv, 'w', newline='') as csv_file:
            log_items: List[Dict[str, Any]] = list(logs)
            if log_items:
                field_names = log_items[0].keys()
                writer = lib_csv.DictWriter(csv_file, delimiter=',', quotechar='"', quoting=lib_csv.QUOTE_ALL, fieldnames=field_names)
                writer.writeheader()
                for log in logs:
                    writer.writerow(log)
    else:
        print('Fetching github usernames, this may take a while:')
        print(logs)

cli()