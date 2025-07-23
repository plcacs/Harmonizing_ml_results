import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
import httpx
from github import Github
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings

awaiting_label = 'awaiting-review'
lang_all_label = 'lang-all'
approved_label = 'approved-1'
github_graphql_url = 'https://api.github.com/graphql'
questions_translations_category_id = 'DIC_kwDOCZduT84CT5P9'
all_discussions_query = '''
query Q($category_id: ID) {
  repository(name: "fastapi", owner: "fastapi") {
    discussions(categoryId: $category_id, first: 100) {
      nodes {
        title
        id
        number
        labels(first: 10) {
          edges {
            node {
              id
              name
            }
          }
        }
      }
    }
  }
}
'''
translation_discussion_query = '''
query Q($after: String, $discussion_number: Int!) {
  repository(name: "fastapi", owner: "fastapi") {
    discussion(number: $discussion_number) {
      comments(first: 100, after: $after) {
        edges {
          cursor
          node {
            id
            url
            body
          }
        }
      }
    }
  }
}
'''
add_comment_mutation = '''
mutation Q($discussion_id: ID!, $body: String!) {
  addDiscussionComment(input: {discussionId: $discussion_id, body: $body}) {
    comment {
      id
      url
      body
    }
  }
}
'''
update_comment_mutation = '''
mutation Q($comment_id: ID!, $body: String!) {
  updateDiscussionComment(input: {commentId: $comment_id, body: $body}) {
    comment {
      id
      url
      body
    }
  }
}
'''

class Comment(BaseModel):
    pass

class UpdateDiscussionComment(BaseModel):
    pass

class UpdateCommentData(BaseModel):
    pass

class UpdateCommentResponse(BaseModel):
    pass

class AddDiscussionComment(BaseModel):
    pass

class AddCommentData(BaseModel):
    pass

class AddCommentResponse(BaseModel):
    pass

class CommentsEdge(BaseModel):
    pass

class Comments(BaseModel):
    pass

class CommentsDiscussion(BaseModel):
    pass

class CommentsRepository(BaseModel):
    pass

class CommentsData(BaseModel):
    pass

class CommentsResponse(BaseModel):
    pass

class AllDiscussionsLabelNode(BaseModel):
    pass

class AllDiscussionsLabelsEdge(BaseModel):
    pass

class AllDiscussionsDiscussionLabels(BaseModel):
    pass

class AllDiscussionsDiscussionNode(BaseModel):
    pass

class AllDiscussionsDiscussions(BaseModel):
    pass

class AllDiscussionsRepository(BaseModel):
    pass

class AllDiscussionsData(BaseModel):
    pass

class AllDiscussionsResponse(BaseModel):
    pass

class Settings(BaseSettings):
    model_config: Dict[str, bool] = {'env_ignore_empty': True}
    github_event_name: Optional[str] = None
    httpx_timeout: int = 30
    debug: bool = False
    number: Optional[int] = None

class PartialGitHubEventIssue(BaseModel):
    number: Optional[int] = None

class PartialGitHubEvent(BaseModel):
    pull_request: Optional[PartialGitHubEventIssue] = None

def get_graphql_response(*, settings: Settings, query: str, after: Optional[str] = None, category_id: Optional[str] = None, discussion_number: Optional[int] = None, discussion_id: Optional[str] = None, comment_id: Optional[str] = None, body: Optional[str] = None) -> Dict[str, Any]:
    headers = {'Authorization': f'token {settings.github_token.get_secret_value()}'}
    variables = {'after': after, 'category_id': category_id, 'discussion_number': discussion_number, 'discussion_id': discussion_id, 'comment_id': comment_id, 'body': body}
    response = httpx.post(github_graphql_url, headers=headers, timeout=settings.httpx_timeout, json={'query': query, 'variables': variables, 'operationName': 'Q'})
    if response.status_code != 200:
        logging.error(f'Response was not 200, after: {after}, category_id: {category_id}')
        logging.error(response.text)
        raise RuntimeError(response.text)
    data = response.json()
    if 'errors' in data:
        logging.error(f'Errors in response, after: {after}, category_id: {category_id}')
        logging.error(data['errors'])
        logging.error(response.text)
        raise RuntimeError(response.text)
    return cast(Dict[str, Any], data)

def get_graphql_translation_discussions(*, settings: Settings) -> List[AllDiscussionsDiscussionNode]:
    data = get_graphql_response(settings=settings, query=all_discussions_query, category_id=questions_translations_category_id)
    graphql_response = AllDiscussionsResponse.model_validate(data)
    return graphql_response.data.repository.discussions.nodes

def get_graphql_translation_discussion_comments_edges(*, settings: Settings, discussion_number: int, after: Optional[str] = None) -> List[CommentsEdge]:
    data = get_graphql_response(settings=settings, query=translation_discussion_query, discussion_number=discussion_number, after=after)
    graphql_response = CommentsResponse.model_validate(data)
    return graphql_response.data.repository.discussion.comments.edges

def get_graphql_translation_discussion_comments(*, settings: Settings, discussion_number: int) -> List[Comment]:
    comment_nodes: List[Comment] = []
    discussion_edges = get_graphql_translation_discussion_comments_edges(settings=settings, discussion_number=discussion_number)
    while discussion_edges:
        for discussion_edge in discussion_edges:
            comment_nodes.append(discussion_edge.node)
        last_edge = discussion_edges[-1]
        discussion_edges = get_graphql_translation_discussion_comments_edges(settings=settings, discussion_number=discussion_number, after=last_edge.cursor)
    return comment_nodes

def create_comment(*, settings: Settings, discussion_id: str, body: str) -> Comment:
    data = get_graphql_response(settings=settings, query=add_comment_mutation, discussion_id=discussion_id, body=body)
    response = AddCommentResponse.model_validate(data)
    return response.data.addDiscussionComment.comment

def update_comment(*, settings: Settings, comment_id: str, body: str) -> Comment:
    data = get_graphql_response(settings=settings, query=update_comment_mutation, comment_id=comment_id, body=body)
    response = UpdateCommentResponse.model_validate(data)
    return response.data.updateDiscussionComment.comment

def main() -> None:
    settings = Settings()
    if settings.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.debug(f'Using config: {settings.model_dump_json()}')
    g = Github(settings.github_token.get_secret_value())
    repo = g.get_repo(settings.github_repository)
    if not settings.github_event_path.is_file():
        raise RuntimeError(f'No github event file available at: {settings.github_event_path}')
    contents = settings.github_event_path.read_text()
    github_event = PartialGitHubEvent.model_validate_json(contents)
    logging.info(f'Using GitHub event: {github_event}')
    number = github_event.pull_request and github_event.pull_request.number or settings.number
    if number is None:
        raise RuntimeError('No PR number available')
    sleep_time = random.random() * 10
    logging.info(f'Sleeping for {sleep_time} seconds to avoid race conditions and multiple comments')
    time.sleep(sleep_time)
    logging.debug(f'Processing PR: #{number}')
    pr = repo.get_pull(number)
    label_strs = {label.name for label in pr.get_labels()}
    langs: List[str] = []
    for label in label_strs:
        if label.startswith('lang-') and (not label == lang_all_label):
            langs.append(label[5:])
    logging.info(f'PR #{pr.number} has labels: {label_strs}')
    if not langs or lang_all_label not in label_strs:
        logging.info(f"PR #{pr.number} doesn't seem to be a translation PR, skipping")
        sys.exit(0)
    discussions = get_graphql_translation_discussions(settings=settings)
    lang_to_discussion_map: Dict[str, AllDiscussionsDiscussionNode] = {}
    for discussion in discussions:
        for edge in discussion.labels.edges:
            label = edge.node.name
            if label.startswith('lang-') and (not label == lang_all_label):
                lang = label[5:]
                lang_to_discussion_map[lang] = discussion
    logging.debug(f'Using translations map: {lang_to_discussion_map}')
    new_translation_message = f"Good news everyone! 😉 There's a new translation PR to be reviewed: #{pr.number} by @{pr.user.login}. 🎉 This requires 2 approvals from native speakers to be merged. 🤓"
    done_translation_message = f"~There's a new translation PR to be reviewed: #{pr.number} by @{pr.user.login}~ Good job! This is done. 🍰☕"
    for lang in langs:
        if lang not in lang_to_discussion_map:
            log_message = f'Could not find discussion for language: {lang}'
            logging.error(log_message)
            raise RuntimeError(log_message)
        discussion = lang_to_discussion_map[lang]
        logging.info(f'Found a translation discussion for language: {lang} in discussion: #{discussion.number}')
        already_notified_comment: Optional[Comment] = None
        already_done_comment: Optional[Comment] = None
        logging.info(f'Checking current comments in discussion: #{discussion.number} to see if already notified about this PR: #{pr.number}')
        comments = get_graphql_translation_discussion_comments(settings=settings, discussion_number=discussion.number)
        for comment in comments:
            if new_translation_message in comment.body:
                already_notified_comment = comment
            elif done_translation_message in comment.body:
                already_done_comment = comment
        logging.info(f'Already notified comment: {already_notified_comment}, already done comment: {already_done_comment}')
        if pr.state == 'open' and awaiting_label in label_strs:
            logging.info(f'This PR seems to be a language translation and awaiting reviews: #{pr.number}')
            if already_notified_comment:
                logging.info(f'This PR #{pr.number} was already notified in comment: {already_notified_comment.url}')
            else:
                logging.info(f'Writing notification comment about PR #{pr.number} in Discussion: #{discussion.number}')
                comment = create_comment(settings=settings, discussion_id=discussion.id, body=new_translation_message)
                logging.info(f'Notified in comment: {comment.url}')
        elif pr.state == 'closed' or approved_label in label_strs:
            logging.info(f'Already approved or closed PR #{pr.number}')
            if already_done_comment:
                logging.info(f'This PR #{pr.number} was already marked as done in comment: {already_done_comment.url}')
            elif already_notified_comment:
                updated_comment = update_comment(settings=settings, comment_id=already_notified_comment.id, body=done_translation_message)
                logging.info(f'Marked as done in comment: {updated_comment.url}')
        else:
            logging.info(f"There doesn't seem to be anything to be done about PR #{pr.number}")
    logging.info('Finished')

if __name__ == '__main__':
    main()
