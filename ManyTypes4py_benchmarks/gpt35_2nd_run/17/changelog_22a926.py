from typing import Any, Optional, Union, List, Dict

class GitLog:
    pr_number: Optional[int] = None
    author_email: str = ''

class GitChangeLog:
    def __init__(self, version: str, logs: List[GitLog], access_token: Optional[str] = None, risk: bool = False):
    def _fetch_github_pr(self, pr_number: int) -> PullRequest:
    def _get_github_login(self, git_log: GitLog) -> str:
    def _has_commit_migrations(self, git_sha: str) -> bool:
    def _get_pull_request_details(self, git_log: GitLog) -> Dict[str, Any]:
    def _is_risk_pull_request(self, labels: List[str]) -> bool
    def _get_changelog_version_head(self) -> str:
    def _parse_change_log(self, changelog: Dict[str, str], pr_info: Dict[str, Any], github_login: str) -> None:
    def __repr__(self) -> str:
    def __iter__(self) -> Iterator[Dict[str, Any]]

class GitLogs:
    def __init__(self, git_ref: str):
    def fetch(self) -> None:
    def diff(self, git_logs: 'GitLogs') -> List[GitLog]:
    def __repr__(self) -> str:
    @staticmethod
    def _git_get_current_head() -> str:
    def _git_checkout(self, git_ref: str) -> None:
    def _git_logs(self) -> List[str]:
    @staticmethod
    def _parse_log(log_item: str) -> GitLog

class BaseParameters:
    pass

def print_title(message: str) -> None:
