from typing import Any, Protocol, TypeAlias, TypeVar

ZerverFieldsT = dict[str, Any]

class SubscriberHandler:
    stream_info: dict[int, Any]
    direct_message_group_info: dict[int, Any]

    def __init__(self) -> None:
        self.stream_info = {}
        self.direct_message_group_info = {}

    def set_info(self, users: Any, stream_id: int = None, direct_message_group_id: int = None) -> None:
        if stream_id is not None:
            self.stream_info[stream_id] = users
        elif direct_message_group_id is not None:
            self.direct_message_group_info[direct_message_group_id] = users
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

    def get_users(self, stream_id: int = None, direct_message_group_id: int = None) -> Any:
        if stream_id is not None:
            return self.stream_info[stream_id]
        elif direct_message_group_id is not None:
            return self.direct_message_group_info[direct_message_group_id]
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

class GetUsers(Protocol):
    def __call__(self, stream_id: int = ..., direct_message_group_id: int = ...) -> Any

ListJobData = TypeVar('ListJobData')
ExternalId = TypeVar('ExternalId')
