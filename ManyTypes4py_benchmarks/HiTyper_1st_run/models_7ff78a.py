"""
Sample code for MonkeyType demonstration exercise at PyCon 2018.

"""
from datetime import datetime
import enum
from typing import Collection, Dict, Generic, List, NamedTuple, NewType, Optional, TypeVar
UserId = NewType('UserId', int)
FeedEntryId = NewType('FeedEntryId', int)
InboxEventId = NewType('InboxEventId', int)

class FeedEntry:

    def __init__(self, id: Union[str, bool], user_id, caption, published) -> None:
        self.id = id
        self.user_id = user_id
        self.caption = caption
        self.published = published

class User:

    def __init__(self, id: Union[str, bool], name: Union[str, list[str], list], following: Union[bool, str, typing.Sequence[str], None]) -> None:
        self.id = id
        self.name = name
        self.following = following

class EventType(enum.Enum):
    COMMENTED = 'commented'
    FOLLOWED = 'followed'
    LIKED = 'liked'

class InboxEvent:

    def __init__(self, id: Union[str, bool], user_id, published) -> None:
        self.id = id
        self.user_id = user_id
        self.published = published

class CommentedEvent(InboxEvent):
    type = EventType.COMMENTED

    def __init__(self, id: Union[str, bool], user_id, published, feedentry_id, commenter_id, comment_text) -> None:
        super().__init__(id, user_id, published)
        self.feedentry_id = feedentry_id
        self.commenter_id = commenter_id
        self.comment_text = comment_text

class LikedEvent(InboxEvent):
    type = EventType.LIKED

    def __init__(self, id: Union[str, bool], user_id, published, feedentry_id, liker_id) -> None:
        super().__init__(id, user_id, published)
        self.feedentry_id = feedentry_id
        self.liker_id = liker_id

class FollowedEvent(InboxEvent):
    type = EventType.FOLLOWED

    def __init__(self, id: Union[str, bool], user_id, published, follower_id) -> None:
        super().__init__(id, user_id, published)
        self.follower_id = follower_id

class RepoInterface:

    def get_feed_entries_by_ids(self, ids: Union[str, int]) -> None:
        raise NotImplementedError()

    def get_feed_entries_for_user_id(self, user_id: Union[str, int]) -> None:
        raise NotImplementedError()

    def get_users_by_ids(self, ids: Union[str, int]) -> None:
        raise NotImplementedError()

    def get_inbox_events_for_user_id(self, user_id: Union[str, int]) -> None:
        raise NotImplementedError()
T = TypeVar('T', bound=InboxEvent)

class AggregatedItem(NamedTuple):
    pass

class AggregatorInterface(Generic[T]):

    def __init__(self, repo) -> None:
        self.repo = repo

    def add(self, event: Any) -> None:
        pass

    def aggregate(self) -> list:
        return []