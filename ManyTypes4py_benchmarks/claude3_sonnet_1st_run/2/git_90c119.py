import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

TOPIC_WITH_BRANCH_TEMPLATE: str = '{repo} / {branch}'
TOPIC_WITH_PR_OR_ISSUE_INFO_TEMPLATE: str = '{repo} / {type} #{id} {title}'
TOPIC_WITH_RELEASE_TEMPLATE: str = '{repo} / {tag} {title}'
EMPTY_SHA: str = '0000000000000000000000000000000000000000'
COMMITS_LIMIT: int = 20
COMMIT_ROW_TEMPLATE: str = '* {commit_msg} ([{commit_short_sha}]({commit_url}))\n'
COMMITS_MORE_THAN_LIMIT_TEMPLATE: str = '[and {commits_number} more commit(s)]'
COMMIT_OR_COMMITS: str = 'commit{}'
PUSH_PUSHED_TEXT_WITH_URL: str = '[{push_type}]({compare_url}) {number_of_commits} {commit_or_commits}'
PUSH_PUSHED_TEXT_WITHOUT_URL: str = '{push_type} {number_of_commits} {commit_or_commits}'
PUSH_COMMITS_BASE: str = '{user_name} {pushed_text} to branch {branch_name}.'
PUSH_COMMITS_MESSAGE_TEMPLATE_WITH_COMMITTERS: str = PUSH_COMMITS_BASE + ' {committers_details}.\n\n{commits_data}\n'
PUSH_COMMITS_MESSAGE_TEMPLATE_WITHOUT_COMMITTERS: str = PUSH_COMMITS_BASE + '\n\n{commits_data}\n'
PUSH_DELETE_BRANCH_MESSAGE_TEMPLATE: str = '{user_name} [deleted]({compare_url}) the branch {branch_name}.'
PUSH_LOCAL_BRANCH_WITHOUT_COMMITS_MESSAGE_TEMPLATE: str = '{user_name} [{push_type}]({compare_url}) the branch {branch_name}.'
PUSH_LOCAL_BRANCH_WITHOUT_COMMITS_MESSAGE_WITHOUT_URL_TEMPLATE: str = '{user_name} {push_type} the branch {branch_name}.'
PUSH_COMMITS_MESSAGE_EXTENSION: str = 'Commits by {}'
PUSH_COMMITTERS_LIMIT_INFO: int = 3
FORCE_PUSH_COMMITS_MESSAGE_TEMPLATE: str = '{user_name} [force pushed]({url}) to branch {branch_name}. Head is now {head}.'
CREATE_BRANCH_MESSAGE_TEMPLATE: str = '{user_name} created [{branch_name}]({url}) branch.'
CREATE_BRANCH_WITHOUT_URL_MESSAGE_TEMPLATE: str = '{user_name} created {branch_name} branch.'
REMOVE_BRANCH_MESSAGE_TEMPLATE: str = '{user_name} deleted branch {branch_name}.'
ISSUE_LABELED_OR_UNLABELED_MESSAGE_TEMPLATE: str = '[{user_name}]({user_url}) {action} the {label_name} label {preposition} [Issue #{id}]({url}).'
ISSUE_LABELED_OR_UNLABELED_MESSAGE_TEMPLATE_WITH_TITLE: str = '[{user_name}]({user_url}) {action} the {label_name} label {preposition} [Issue #{id} {title}]({url}).'
ISSUE_MILESTONED_OR_DEMILESTONED_MESSAGE_TEMPLATE: str = '[{user_name}]({user_url}) {action} milestone [{milestone_name}]({milestone_url}) {preposition} [issue #{id}]({url}).'
ISSUE_MILESTONED_OR_DEMILESTONED_MESSAGE_TEMPLATE_WITH_TITLE: str = '[{user_name}]({user_url}) {action} milestone [{milestone_name}]({milestone_url}) {preposition} [issue #{id} {title}]({url}).'
PULL_REQUEST_OR_ISSUE_MESSAGE_TEMPLATE: str = '{user_name} {action}{assignee} [{type}{id}{title}]({url})'
PULL_REQUEST_OR_ISSUE_ASSIGNEE_INFO_TEMPLATE: str = '(assigned to {assignee})'
PULL_REQUEST_REVIEWER_INFO_TEMPLATE: str = '(assigned reviewers: {reviewer})'
PULL_REQUEST_BRANCH_INFO_TEMPLATE: str = 'from `{target}` to `{base}`'
CONTENT_MESSAGE_TEMPLATE: str = '\n~~~ quote\n{message}\n~~~'
COMMITS_COMMENT_MESSAGE_TEMPLATE: str = '{user_name} {action} on [{sha}]({url})'
PUSH_TAGS_MESSAGE_TEMPLATE: str = '{user_name} {action} tag {tag}'
TAG_WITH_URL_TEMPLATE: str = '[{tag_name}]({tag_url})'
TAG_WITHOUT_URL_TEMPLATE: str = '{tag_name}'
RELEASE_MESSAGE_TEMPLATE: str = '{user_name} {action} release [{release_name}]({url}) for tag {tagname}.'
RELEASE_MESSAGE_TEMPLATE_WITHOUT_USER_NAME: str = 'Release [{release_name}]({url}) for tag {tagname} was {action}.'
RELEASE_MESSAGE_TEMPLATE_WITHOUT_USER_NAME_WITHOUT_URL: str = 'Release {release_name} for tag {tagname} was {action}.'

def get_assignee_string(assignees: List[Dict[str, str]]) -> str:
    assignees_string: str = ''
    if len(assignees) == 1:
        assignees_string = '{username}'.format(**assignees[0])
    else:
        usernames: List[str] = [a['username'] for a in assignees]
        assignees_string = ', '.join(usernames[:-1]) + ' and ' + usernames[-1]
    return assignees_string

def get_push_commits_event_message(user_name: str, compare_url: Optional[str], branch_name: str, 
                                  commits_data: List[Dict[str, Any]], is_truncated: bool = False, 
                                  deleted: bool = False, force_push: bool = False) -> str:
    if not commits_data and deleted:
        return PUSH_DELETE_BRANCH_MESSAGE_TEMPLATE.format(user_name=user_name, compare_url=compare_url, branch_name=branch_name)
    push_type: str = 'force pushed' if force_push else 'pushed'
    if not commits_data and (not deleted):
        if compare_url:
            return PUSH_LOCAL_BRANCH_WITHOUT_COMMITS_MESSAGE_TEMPLATE.format(push_type=push_type, user_name=user_name, compare_url=compare_url, branch_name=branch_name)
        return PUSH_LOCAL_BRANCH_WITHOUT_COMMITS_MESSAGE_WITHOUT_URL_TEMPLATE.format(push_type=push_type, user_name=user_name, branch_name=branch_name)
    pushed_message_template: str = PUSH_PUSHED_TEXT_WITH_URL if compare_url else PUSH_PUSHED_TEXT_WITHOUT_URL
    pushed_text_message: str = pushed_message_template.format(push_type=push_type, compare_url=compare_url, number_of_commits=len(commits_data), commit_or_commits=COMMIT_OR_COMMITS.format('s' if len(commits_data) > 1 else ''))
    committers_items: List[Tuple[str, int]] = get_all_committers(commits_data)
    if len(committers_items) == 1 and user_name == committers_items[0][0]:
        return PUSH_COMMITS_MESSAGE_TEMPLATE_WITHOUT_COMMITTERS.format(user_name=user_name, pushed_text=pushed_text_message, branch_name=branch_name, commits_data=get_commits_content(commits_data, is_truncated)).rstrip()
    else:
        committers_details: str = '{} ({})'.format(*committers_items[0])
        for name, number_of_commits in committers_items[1:-1]:
            committers_details = f'{committers_details}, {name} ({number_of_commits})'
        if len(committers_items) > 1:
            committers_details = '{} and {} ({})'.format(committers_details, *committers_items[-1])
        return PUSH_COMMITS_MESSAGE_TEMPLATE_WITH_COMMITTERS.format(user_name=user_name, pushed_text=pushed_text_message, branch_name=branch_name, committers_details=PUSH_COMMITS_MESSAGE_EXTENSION.format(committers_details), commits_data=get_commits_content(commits_data, is_truncated)).rstrip()

def get_force_push_commits_event_message(user_name: str, url: str, branch_name: str, head: str) -> str:
    return FORCE_PUSH_COMMITS_MESSAGE_TEMPLATE.format(user_name=user_name, url=url, branch_name=branch_name, head=head)

def get_create_branch_event_message(user_name: str, url: Optional[str], branch_name: str) -> str:
    if url is None:
        return CREATE_BRANCH_WITHOUT_URL_MESSAGE_TEMPLATE.format(user_name=user_name, branch_name=branch_name)
    return CREATE_BRANCH_MESSAGE_TEMPLATE.format(user_name=user_name, url=url, branch_name=branch_name)

def get_remove_branch_event_message(user_name: str, branch_name: str) -> str:
    return REMOVE_BRANCH_MESSAGE_TEMPLATE.format(user_name=user_name, branch_name=branch_name)

def get_pull_request_event_message(*, user_name: str, action: str, url: str, number: Optional[int] = None, 
                                  target_branch: Optional[str] = None, base_branch: Optional[str] = None, 
                                  message: Optional[str] = None, assignee: Optional[str] = None, 
                                  assignees: Optional[List[Dict[str, str]]] = None, assignee_updated: Optional[str] = None, 
                                  reviewer: Optional[str] = None, type: str = 'PR', title: Optional[str] = None) -> str:
    kwargs: Dict[str, str] = {'user_name': user_name, 'action': action, 'type': type, 'url': url, 
                             'id': f' #{number}' if number is not None else '', 
                             'title': f' {title}' if title is not None else '', 
                             'assignee': {'assigned': f' {assignee_updated} to', 'unassigned': f' {assignee_updated} from'}.get(action, '')}
    main_message: str = PULL_REQUEST_OR_ISSUE_MESSAGE_TEMPLATE.format(**kwargs)
    if target_branch and base_branch:
        branch_info: str = PULL_REQUEST_BRANCH_INFO_TEMPLATE.format(target=target_branch, base=base_branch)
        main_message = f'{main_message} {branch_info}'
    if assignees:
        assignee_string: str = get_assignee_string(assignees)
        assignee_info: str = PULL_REQUEST_OR_ISSUE_ASSIGNEE_INFO_TEMPLATE.format(assignee=assignee_string)
    elif assignee:
        assignee_info = PULL_REQUEST_OR_ISSUE_ASSIGNEE_INFO_TEMPLATE.format(assignee=assignee)
    elif reviewer:
        assignee_info = PULL_REQUEST_REVIEWER_INFO_TEMPLATE.format(reviewer=reviewer)
    if assignees or assignee or reviewer:
        main_message = f'{main_message} {assignee_info}'
    punctuation: str = ':' if message else '.'
    if assignees or assignee or (target_branch and base_branch) or (title is None) or (title[-1] not in string.punctuation):
        main_message = f'{main_message}{punctuation}'
    if message:
        main_message += '\n' + CONTENT_MESSAGE_TEMPLATE.format(message=message)
    return main_message.rstrip()

def get_issue_event_message(*, user_name: str, action: str, url: str, number: Optional[int] = None, 
                           message: Optional[str] = None, assignee: Optional[str] = None, 
                           assignees: Optional[List[Dict[str, str]]] = None, assignee_updated: Optional[str] = None, 
                           title: Optional[str] = None) -> str:
    return get_pull_request_event_message(user_name=user_name, action=action, url=url, number=number, 
                                         message=message, assignee=assignee, assignees=assignees, 
                                         assignee_updated=assignee_updated, type='issue', title=title)

def get_issue_labeled_or_unlabeled_event_message(user_name: str, action: str, url: str, 
                                                number: int, label_name: str, user_url: str, 
                                                title: Optional[str] = None) -> str:
    args: Dict[str, Any] = {'user_name': user_name, 'action': action, 'url': url, 'id': number, 
                           'label_name': label_name, 'user_url': user_url, 'title': title, 
                           'preposition': 'to' if action == 'added' else 'from'}
    if title is not None:
        return ISSUE_LABELED_OR_UNLABELED_MESSAGE_TEMPLATE_WITH_TITLE.format(**args)
    return ISSUE_LABELED_OR_UNLABELED_MESSAGE_TEMPLATE.format(**args)

def get_issue_milestoned_or_demilestoned_event_message(user_name: str, action: str, url: str, 
                                                      number: int, milestone_name: str, 
                                                      milestone_url: str, user_url: str, 
                                                      title: Optional[str] = None) -> str:
    args: Dict[str, Any] = {'user_name': user_name, 'action': action, 'url': url, 'id': number, 
                           'milestone_name': milestone_name, 'milestone_url': milestone_url, 
                           'user_url': user_url, 'title': title, 
                           'preposition': 'to' if action == 'added' else 'from'}
    if title is not None:
        return ISSUE_MILESTONED_OR_DEMILESTONED_MESSAGE_TEMPLATE_WITH_TITLE.format(**args)
    return ISSUE_MILESTONED_OR_DEMILESTONED_MESSAGE_TEMPLATE.format(**args)

def get_push_tag_event_message(user_name: str, tag_name: str, tag_url: Optional[str] = None, action: str = 'pushed') -> str:
    if tag_url:
        tag_part: str = TAG_WITH_URL_TEMPLATE.format(tag_name=tag_name, tag_url=tag_url)
    else:
        tag_part = TAG_WITHOUT_URL_TEMPLATE.format(tag_name=tag_name)
    message: str = PUSH_TAGS_MESSAGE_TEMPLATE.format(user_name=user_name, action=action, tag=tag_part)
    if tag_name[-1] not in string.punctuation:
        message = f'{message}.'
    return message

def get_commits_comment_action_message(user_name: str, action: str, commit_url: str, sha: str, message: Optional[str] = None) -> str:
    content: str = COMMITS_COMMENT_MESSAGE_TEMPLATE.format(user_name=user_name, action=action, sha=get_short_sha(sha), url=commit_url)
    punctuation: str = ':' if message else '.'
    content = f'{content}{punctuation}'
    if message:
        content += CONTENT_MESSAGE_TEMPLATE.format(message=message)
    return content

def get_commits_content(commits_data: List[Dict[str, Any]], is_truncated: bool = False) -> str:
    commits_content: str = ''
    for commit in commits_data[:COMMITS_LIMIT]:
        commits_content += COMMIT_ROW_TEMPLATE.format(commit_short_sha=get_short_sha(commit['sha']), commit_url=commit.get('url'), commit_msg=commit['message'].partition('\n')[0])
    if len(commits_data) > COMMITS_LIMIT:
        commits_content += COMMITS_MORE_THAN_LIMIT_TEMPLATE.format(commits_number=len(commits_data) - COMMITS_LIMIT)
    elif is_truncated:
        commits_content += COMMITS_MORE_THAN_LIMIT_TEMPLATE.format(commits_number='').replace('  ', ' ')
    return commits_content.rstrip()

def get_release_event_message(user_name: str, action: str, tagname: str, release_name: str, url: str) -> str:
    content: str = RELEASE_MESSAGE_TEMPLATE.format(user_name=user_name, action=action, tagname=tagname, release_name=release_name, url=url)
    return content

def get_short_sha(sha: str) -> str:
    return sha[:11]

def get_all_committers(commits_data: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    committers: Dict[str, int] = defaultdict(int)
    for commit in commits_data:
        committers[commit['name']] += 1
    committers_items: List[Tuple[str, int]] = sorted(committers.items(), key=lambda item: (-item[1], item[0]))
    committers_values: List[int] = [c_i[1] for c_i in committers_items]
    if len(committers) > PUSH_COMMITTERS_LIMIT_INFO:
        others_number_of_commits: int = sum(committers_values[PUSH_COMMITTERS_LIMIT_INFO:])
        committers_items = committers_items[:PUSH_COMMITTERS_LIMIT_INFO]
        committers_items.append(('others', others_number_of_commits))
    return committers_items
