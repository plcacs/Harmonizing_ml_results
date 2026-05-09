import logging
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, AbstractSet
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Protocol, TypeAlias, TypeVar

ZerverFieldsT: TypeAlias = dict[str, Any]

class SubscriberHandler:
    def __init__(self):
        self.stream_info: dict[int, set[int]] = {}
        self.direct_message_group_info: dict[int, set[int]] = {}

    def set_info(self, users: set[int], stream_id: int = None, direct_message_group_id: int = None):
        if stream_id is not None:
            self.stream_info[stream_id] = users
        elif direct_message_group_id is not None:
            self.direct_message_group_info[direct_message_group_id] = users
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

    def get_users(self, stream_id: int = None, direct_message_group_id: int = None) -> set[int]:
        if stream_id is not None:
            return self.stream_info.get(stream_id, set())
        elif direct_message_group_id is not None:
            return self.direct_message_group_info.get(direct_message_group_id, set())
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

# ... rest of the code ...
