class Comment(BaseModel):
    id: str
    url: str
    body: str

class UpdateDiscussionComment(BaseModel):
    comment: Comment

class UpdateCommentData(BaseModel):
    updateDiscussionComment: UpdateDiscussionComment

class UpdateCommentResponse(BaseModel):
    data: UpdateCommentData

class AddDiscussionComment(BaseModel):
    comment: Comment

class AddCommentData(BaseModel):
    addDiscussionComment: AddDiscussionComment

class AddCommentResponse(BaseModel):
    data: AddCommentData

class CommentsEdge(BaseModel):
    cursor: str
    node: Comment

class Comments(BaseModel):
    edges: List[CommentsEdge]

class CommentsDiscussion(BaseModel):
    comments: Comments

class CommentsRepository(BaseModel):
    discussion: CommentsDiscussion

class CommentsData(BaseModel):
    repository: CommentsRepository

class CommentsResponse(BaseModel):
    data: CommentsData

class AllDiscussionsLabelNode(BaseModel):
    id: str
    name: str

class AllDiscussionsLabelsEdge(BaseModel):
    node: AllDiscussionsLabelNode

class AllDiscussionsDiscussionLabels(BaseModel):
    edges: List[AllDiscussionsLabelsEdge]

class AllDiscussionsDiscussionNode(BaseModel):
    title: str
    id: str
    number: int
    labels: AllDiscussionsDiscussionLabels

class AllDiscussionsDiscussions(BaseModel):
    nodes: List[AllDiscussionsDiscussionNode]

class AllDiscussionsRepository(BaseModel):
    discussions: AllDiscussionsDiscussions

class AllDiscussionsData(BaseModel):
    repository: AllDiscussionsRepository

class AllDiscussionsResponse(BaseModel):
    data: AllDiscussionsData

class Settings(BaseSettings):
    model_config: Dict[str, bool]
    github_event_name: str
    httpx_timeout: int
    debug: bool
    number: int
    github_token: SecretStr
    github_repository: str
    github_event_path: Path

class PartialGitHubEventIssue(BaseModel):
    number: int

class PartialGitHubEvent(BaseModel):
    pull_request: PartialGitHubEventIssue

def get_graphql_response(*, settings: Settings, query: str, after: str = None, category_id: str = None, discussion_number: int = None, discussion_id: str = None, comment_id: str = None, body: str = None) -> Dict[str, Any]:
    ...

def get_graphql_translation_discussions(*, settings: Settings) -> List[AllDiscussionsDiscussionNode]:
    ...

def get_graphql_translation_discussion_comments_edges(*, settings: Settings, discussion_number: int, after: str = None) -> List[CommentsEdge]:
    ...

def get_graphql_translation_discussion_comments(*, settings: Settings, discussion_number: int) -> List[Comment]:
    ...

def create_comment(*, settings: Settings, discussion_id: str, body: str) -> Comment:
    ...

def update_comment(*, settings: Settings, comment_id: str, body: str) -> Comment:
    ...

def main() -> None:
    ...
