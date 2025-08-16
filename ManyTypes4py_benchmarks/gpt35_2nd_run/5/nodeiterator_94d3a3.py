import base64
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from lzma import LZMAError
from typing import Any, Callable, Dict, Iterable, Iterator, NamedTuple, Optional, Tuple, TypeVar
from .exceptions import AbortDownloadException, InvalidArgumentException
from .instaloadercontext import InstaloaderContext

class FrozenNodeIterator(NamedTuple):
    query_hash: Optional[str]  # The GraphQL ``query_hash`` parameter.
    query_variables: Optional[Dict[str, Any]]  # The GraphQL ``query_variables`` parameter.
    query_referer: Optional[str]  # The HTTP referer used for the GraphQL query.
    context_username: Optional[str]  # The username who created the iterator, or ``None``.
    total_index: int  # Number of items that have already been returned.
    best_before: Optional[float]  # Date when parts of the stored nodes might have expired.
    remaining_data: Optional[Dict[str, Any]]  # The already-retrieved, yet-unprocessed ``edges`` and the ``page_info`` at time of freezing.
    first_node: Optional[Any]  # Node data of the first item, if an item has been produced.
    doc_id: Optional[str]  # The GraphQL ``doc_id`` parameter.

T = TypeVar('T')

class NodeIterator(Iterator[T]):
    _graphql_page_length: int = 12
    _shelf_life: timedelta = timedelta(days=29)

    def __init__(self, context: InstaloaderContext, query_hash: Optional[str], edge_extractor: Callable, node_wrapper: Callable, query_variables: Optional[Dict[str, Any]] = None, query_referer: Optional[str] = None, first_data: Optional[Dict[str, Any]] = None, is_first: Optional[Callable] = None, doc_id: Optional[str] = None):
        self._context = context
        self._query_hash = query_hash
        self._doc_id = doc_id
        self._edge_extractor = edge_extractor
        self._node_wrapper = node_wrapper
        self._query_variables = query_variables if query_variables is not None else {}
        self._query_referer = query_referer
        self._page_index = 0
        self._total_index = 0
        if first_data is not None:
            self._data = first_data
            self._best_before = datetime.now() + NodeIterator._shelf_life
        else:
            self._data = self._query()
        self._first_node = None
        self._is_first = is_first

    def _query(self, after: Optional[str] = None) -> Dict[str, Any]:
        ...

    def _query_doc_id(self, doc_id: str, after: Optional[str] = None) -> Dict[str, Any]:
        ...

    def _query_query_hash(self, query_hash: str, after: Optional[str] = None) -> Dict[str, Any]:
        ...

    def __iter__(self) -> 'NodeIterator':
        ...

    def __next__(self) -> T:
        ...

    @property
    def count(self) -> Optional[int]:
        ...

    @property
    def total_index(self) -> int:
        ...

    @property
    def magic(self) -> str:
        ...

    @property
    def first_item(self) -> Any:
        ...

    @staticmethod
    def page_length() -> int:
        ...

    def freeze(self) -> FrozenNodeIterator:
        ...

    def thaw(self, frozen: FrozenNodeIterator) -> None:
        ...

@contextmanager
def resumable_iteration(context: InstaloaderContext, iterator: NodeIterator, load: Callable[[InstaloaderContext, str], FrozenNodeIterator], save: Callable[[FrozenNodeIterator, str], None], format_path: Callable[[str], str], check_bbd: bool = True, enabled: bool = True) -> Iterator[Tuple[bool, int]]:
    ...
