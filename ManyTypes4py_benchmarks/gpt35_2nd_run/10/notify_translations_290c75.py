def get_graphql_response(*, settings: Settings, query: str, after: Union[str, None] = None, category_id: Union[str, None] = None, discussion_number: Union[int, None] = None, discussion_id: Union[str, None] = None, comment_id: Union[str, None] = None, body: Union[str, None] = None) -> Dict[str, Any]:

def get_graphql_translation_discussions(*, settings: Settings) -> List[AllDiscussionsDiscussionNode]:

def get_graphql_translation_discussion_comments_edges(*, settings: Settings, discussion_number: int, after: Union[str, None] = None) -> List[CommentsEdge]:

def get_graphql_translation_discussion_comments(*, settings: Settings, discussion_number: int) -> List[Comments]:

def create_comment(*, settings: Settings, discussion_id: str, body: str) -> Comment:

def update_comment(*, settings: Settings, comment_id: str, body: str) -> UpdateDiscussionComment:
