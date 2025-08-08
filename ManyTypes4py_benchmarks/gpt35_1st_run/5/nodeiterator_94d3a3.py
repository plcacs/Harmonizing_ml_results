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
        if self._doc_id is not None:
            return self._query_doc_id(self._doc_id, after)
        else:
            assert self._query_hash is not None
            return self._query_query_hash(self._query_hash, after)

    def _query_doc_id(self, doc_id: str, after: Optional[str] = None) -> Dict[str, Any]:
        pagination_variables = {'__relay_internal__pv__PolarisFeedShareMenurelayprovider': False}
        if after is not None:
            pagination_variables['after'] = after
            pagination_variables['before'] = None
            pagination_variables['first'] = 12
            pagination_variables['last'] = None
        data = self._edge_extractor(self._context.doc_id_graphql_query(doc_id, {**self._query_variables, **pagination_variables}, self._query_referer))
        self._best_before = datetime.now() + NodeIterator._shelf_life
        return data

    def _query_query_hash(self, query_hash: str, after: Optional[str] = None) -> Dict[str, Any]:
        pagination_variables = {'first': NodeIterator._graphql_page_length}
        if after is not None:
            pagination_variables['after'] = after
        data = self._edge_extractor(self._context.graphql_query(query_hash, {**self._query_variables, **pagination_variables}, self._query_referer))
        self._best_before = datetime.now() + NodeIterator._shelf_life
        return data

    def __iter__(self) -> 'NodeIterator':
        return self

    def __next__(self) -> T:
        if self._page_index < len(self._data['edges']):
            node = self._data['edges'][self._page_index]['node']
            page_index, total_index = (self._page_index, self._total_index)
            try:
                self._page_index += 1
                self._total_index += 1
            except KeyboardInterrupt:
                self._page_index, self._total_index = (page_index, total_index)
                raise
            item = self._node_wrapper(node)
            if self._is_first is not None:
                if self._is_first(item, self.first_item):
                    self._first_node = node
            elif self._first_node is None:
                self._first_node = node
            return item
        if self._data.get('page_info', {}).get('has_next_page'):
            query_response = self._query(self._data['page_info']['end_cursor'])
            if self._data['edges'] != query_response['edges'] and len(query_response['edges']) > 0:
                page_index, data = (self._page_index, self._data)
                try:
                    self._page_index = 0
                    self._data = query_response
                except KeyboardInterrupt:
                    self._page_index, self._data = (page_index, data)
                    raise
                return self.__next__()
        raise StopIteration()

    @property
    def count(self) -> Optional[int]:
        return self._data.get('count') if self._data is not None else None

    @property
    def total_index(self) -> int:
        return self._total_index

    @property
    def magic(self) -> str:
        magic_hash = hashlib.blake2b(digest_size=6)
        magic_hash.update(json.dumps([self._query_hash, self._query_variables, self._query_referer, self._context.username]).encode())
        return base64.urlsafe_b64encode(magic_hash.digest()).decode()

    @property
    def first_item(self) -> Optional[Any]:
        return self._node_wrapper(self._first_node) if self._first_node is not None else None

    @staticmethod
    def page_length() -> int:
        return NodeIterator._graphql_page_length

    def freeze(self) -> FrozenNodeIterator:
        remaining_data = None
        if self._data is not None:
            remaining_data = {**self._data, 'edges': self._data['edges'][max(self._page_index - 1, 0):]}
        return FrozenNodeIterator(query_hash=self._query_hash, query_variables=self._query_variables, query_referer=self._query_referer, context_username=self._context.username, total_index=max(self.total_index - 1, 0), best_before=self._best_before.timestamp() if self._best_before else None, remaining_data=remaining_data, first_node=self._first_node, doc_id=self._doc_id)

    def thaw(self, frozen: FrozenNodeIterator) -> None:
        if self._total_index or self._page_index:
            raise InvalidArgumentException('thaw() called on already-used iterator.')
        if self._query_hash != frozen.query_hash or self._query_variables != frozen.query_variables or self._query_referer != frozen.query_referer or (self._context.username != frozen.context_username) or (self._doc_id != frozen.doc_id):
            raise InvalidArgumentException('Mismatching resume information.')
        if not frozen.best_before:
            raise InvalidArgumentException('"best before" date missing.')
        if frozen.remaining_data is None:
            raise InvalidArgumentException('"remaining_data" missing.')
        self._total_index = frozen.total_index
        self._best_before = datetime.fromtimestamp(frozen.best_before)
        self._data = frozen.remaining_data
        if frozen.first_node is not None:
            self._first_node = frozen.first_node

@contextmanager
def resumable_iteration(context: InstaloaderContext, iterator: NodeIterator, load: Callable[[InstaloaderContext, str], FrozenNodeIterator], save: Callable[[FrozenNodeIterator, str], None], format_path: Callable[[str], str], check_bbd: bool = True, enabled: bool = True) -> Iterator[Tuple[bool, int]]:
    if not enabled or not isinstance(iterator, NodeIterator):
        yield (False, 0)
        return
    is_resuming = False
    start_index = 0
    resume_file_path = format_path(iterator.magic)
    resume_file_exists = os.path.isfile(resume_file_path)
    if resume_file_exists:
        try:
            fni = load(context, resume_file_path)
            if not isinstance(fni, FrozenNodeIterator):
                raise InvalidArgumentException('Invalid type.')
            if check_bbd and fni.best_before and (datetime.fromtimestamp(fni.best_before) < datetime.now()):
                raise InvalidArgumentException('"Best before" date exceeded.')
            iterator.thaw(fni)
            is_resuming = True
            start_index = iterator.total_index
            context.log('Resuming from {}.'.format(resume_file_path))
        except (InvalidArgumentException, LZMAError, json.decoder.JSONDecodeError, EOFError) as exc:
            context.error('Warning: Not resuming from {}: {}'.format(resume_file_path, exc))
    try:
        yield (is_resuming, start_index)
    except (KeyboardInterrupt, AbortDownloadException):
        if os.path.dirname(resume_file_path):
            os.makedirs(os.path.dirname(resume_file_path), exist_ok=True)
        save(iterator.freeze(), resume_file_path)
        context.log('\nSaved resume information to {}.'.format(resume_file_path))
        raise
    if resume_file_exists:
        os.unlink(resume_file_path)
        context.log('Iteration complete, deleted resume information file {}.'.format(resume_file_path))
