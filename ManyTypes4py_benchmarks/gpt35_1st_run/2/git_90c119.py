from typing import List, Dict

def get_assignee_string(assignees: List[Dict[str, str]]) -> str:
def get_push_commits_event_message(user_name: str, compare_url: str, branch_name: str, commits_data: List[Dict[str, str]], is_truncated: bool = False, deleted: bool = False, force_push: bool = False) -> str:
def get_force_push_commits_event_message(user_name: str, url: str, branch_name: str, head: str) -> str:
def get_create_branch_event_message(user_name: str, url: str, branch_name: str) -> str:
def get_remove_branch_event_message(user_name: str, branch_name: str) -> str:
def get_pull_request_event_message(*, user_name: str, action: str, url: str, number: int = None, target_branch: str = None, base_branch: str = None, message: str = None, assignee: str = None, assignees: List[Dict[str, str]] = None, assignee_updated: str = None, reviewer: str = None, type: str = 'PR', title: str = None) -> str:
def get_issue_event_message(*, user_name: str, action: str, url: str, number: int = None, message: str = None, assignee: str = None, assignees: List[Dict[str, str]] = None, assignee_updated: str = None, title: str = None) -> str:
def get_issue_labeled_or_unlabeled_event_message(user_name: str, action: str, url: str, number: int, label_name: str, user_url: str, title: str = None) -> str:
def get_issue_milestoned_or_demilestoned_event_message(user_name: str, action: str, url: str, number: int, milestone_name: str, milestone_url: str, user_url: str, title: str = None) -> str:
def get_push_tag_event_message(user_name: str, tag_name: str, tag_url: str = None, action: str = 'pushed') -> str:
def get_commits_comment_action_message(user_name: str, action: str, commit_url: str, sha: str, message: str = None) -> str:
def get_commits_content(commits_data: List[Dict[str, str]], is_truncated: bool = False) -> str:
def get_release_event_message(user_name: str, action: str, tagname: str, release_name: str, url: str) -> str:
def get_short_sha(sha: str) -> str:
def get_all_committers(commits_data: List[Dict[str, str]]) -> List[Tuple[str, int]]:
