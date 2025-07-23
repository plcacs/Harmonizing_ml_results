import base64
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from lzma import LZMAError
from typing import Any, Callable, Dict, Iterable, Iterator, NamedTuple, Optional, Tuple, TypeVar, Generic, cast
from .exceptions import AbortDownloadException, InvalidArgumentException
from .instaloadercontext import InstaloaderContext

T = TypeVar('T')
EdgeExtractor = Callable[[Dict[str, Any]], Dict[str, Any]]
NodeWrapper = Callable[[Dict[str, Any]], T]
IsFirstCallback = Callable[[T, Optional[T]], bool]

class FrozenNodeIterator(NamedTuple):
    query_hash: Optional[str]
    query_variables: Dict[str, Any]
    query_referer: Optional[str]
    context_username: Optional[str]
    total_index: int
    best_before: Optional[float]
    remaining_data: Optional[Dict[str, Any]]
    first_node: Optional[Dict[str, Any]]
    doc_id: Optional[str]

FrozenNodeIterator.query_hash.__doc__ = 'The GraphQL ``query_hash`` parameter.'
FrozenNodeIterator.query_variables.__doc__ = 'The GraphQL ``query_variables`` parameter.'
FrozenNodeIterator.query_referer.__doc__ = 'The HTTP referer used for the GraphQL query.'
FrozenNodeIterator.context_username.__doc__ = 'The username who created the iterator, or ``None``.'
FrozenNodeIterator.total_index.__doc__ = 'Number of items that have already been returned.'
FrozenNodeIterator.best_before.__doc__ = 'Date when parts of the stored nodes might have expired.'
FrozenNodeIterator.remaining_data.__doc__ = 'The already-retrieved, yet-unprocessed ``edges`` and the ``page_info`` at time of freezing.'
FrozenNodeIterator.first_node.__doc__ = 'Node data of the first item, if an item has been produced.'
FrozenNodeIterator.doc_id.__doc__ = 'The GraphQL ``doc_id`` parameter.'

class NodeIterator(Iterator[T], Generic[T]):
    _graphql_page_length: int = 12
    _shelf_life: timedelta = timedelta(days=29)

    def __init__(
        self,
        context: InstaloaderContext,
        query_hash: Optional[str],
        edge_extractor: EdgeExtractor,
        node_wrapper: NodeWrapper[T],
        query_variables: Optional[Dict[str, Any]] = None,
        query_referer: Optional[str] = None,
        first_data: Optional[Dict[str, Any]] = None,
        is_first: Optional[IsFirstCallback[T]] = None,
        doc_id: Optional[str] = None
    ) -> None:
        self._context: InstaloaderContext = context
        self._query_hash: Optional[str] = query_hash
        self._doc_id: Optional[str] = doc_id
        self._edge_extractor: EdgeExtractor = edge_extractor
        self._node_wrapper: NodeWrapper[T] = node_wrapper
        self._query_variables: Dict[str, Any] = query_variables if query_variables is not None else {}
        self._query_referer: Optional[str] = query_referer
        self._page_index: int = 0
        self._total_index: int = 0
        if first_data is not None:
            self._data: Dict[str, Any] = first_data
            self._best_before: datetime = datetime.now() + NodeIterator._shelf_life
        else:
            self._data: Dict[str, Any] = self._query()
        self._first_node: Optional[Dict[str, Any]] = None
        self._is_first: Optional[IsFirstCallback[T]] = is_first

    def _query(self, after: Optional[str] = None) -> Dict[str, Any]:
        if self._doc_id is not None:
            return self._query_doc_id(self._doc_id, after)
        else:
            assert self._query_hash is not None
            return self._query_query_hash(self._query_hash, after)

    def _query_doc_id(self, doc_id: str, after: Optional[str] = None) -> Dict[str, Any]:
        pagination_variables: Dict[str, Any] = {'__relay_internal__pv__PolarisFeedShareMenurelayprovider': False}
        if after is not None:
            pagination_variables['after'] = after
            pagination_variables['before'] = None
            pagination_variables['first'] = 12
            pagination_variables['last'] = None
        data: Dict[str, Any] = self._edge_extractor(
            self._context.doc_id_graphql_query(doc_id, {**self._query_variables, **pagination_variables}, self._query_referer)
        )
        self._best_before = datetime.now() + NodeIterator._shelf_life
        return data

    def _query_query_hash(self, query_hash: str, after: Optional[str] = None) -> Dict[str, Any]:
        pagination_variables: Dict[str, Any] = {'first': NodeIterator._graphql_page_length}
        if after is not None:
            pagination_variables['after'] = after
        data: Dict[str, Any] = self._edge_extractor(
            self._context.graphql_query(query_hash, {**self._query_variables, **pagination_variables}, self._query_referer)
        )
        self._best_before = datetime.now() + NodeIterator._shelf_life
        return data

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self._page_index < len(self._data['edges']):
            node: Dict[str, Any] = self._data['edges'][self._page_index]['node']
            page_index: int
            total_index: int
            page_index, total_index = (self._page_index, self._total_index)
            try:
                self._page_index += 1
                self._total_index += 1
            except KeyboardInterrupt:
                self._page_index, self._total_index = (page_index, total_index)
                raise
            item: T = self._node_wrapper(node)
            if self._is_first is not None:
                if self._is_first(item, self.first_item):
                    self._first_node = node
            elif self._first_node is None:
                self._first_node = node
            return item
        if self._data.get('page_info', {}).get('has_next_page'):
            query_response: Dict[str, Any] = self._query(self._data['page_info']['end_cursor'])
            if self._data['edges'] != query_response['edges'] and len(query_response['edges']) > 0:
                page_index: int
                data: Dict[str, Any]
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
        """The ``count`` as returned by Instagram. This is not always the total count this iterator will yield."""
        return self._data.get('count') if self._data is not None else None

    @property
    def total_index(self) -> int:
        """Number of items that have already been returned."""
        return self._total_index

    @property
    def magic(self) -> str:
        """Magic string for easily identifying a matching iterator file for resuming (hash of some parameters)."""
        magic_hash = hashlib.blake2b(digest_size=6)
        magic_hash.update(json.dumps([self._query_hash, self._query_variables, self._query_referer, self._context.username]).encode())
        return base64.urlsafe_b64encode(magic_hash.digest()).decode()

    @property
    def first_item(self) -> Optional[T]:
        """
        If this iterator has produced any items, returns the first item produced.

        It is possible to override what is considered the first item (for example, to consider the
        newest item in case items are not in strict chronological order) by passing a callback
        function as the `is_first` parameter when creating the class.

        .. versionadded:: 4.8
        .. versionchanged:: 4.9.2
           What is considered the first item can be overridden.
        """
        return self._node_wrapper(self._first_node) if self._first_node is not None else None

    @staticmethod
    def page_length() -> int:
        return NodeIterator._graphql_page_length

    def freeze(self) -> FrozenNodeIterator:
        """Freeze the iterator for later resuming."""
        remaining_data: Optional[Dict[str, Any]] = None
        if self._data is not None:
            remaining_data = {**self._data, 'edges': self._data['edges'][max(self._page_index - 1, 0):]}
        return FrozenNodeIterator(
            query_hash=self._query_hash,
            query_variables=self._query_variables,
            query_referer=self._query_referer,
            context_username=self._context.username,
            total_index=max(self.total_index - 1, 0),
            best_before=self._best_before.timestamp() if self._best_before else None,
            remaining_data=remaining_data,
            first_node=self._first_node,
            doc_id=self._doc_id
        )

    def thaw(self, frozen: FrozenNodeIterator) -> None:
        """
        Use this iterator for resuming from earlier iteration.

        :raises InvalidArgumentException:
           If

           - the iterator on which this method is called has already been used, or
           - the given :class:`FrozenNodeIterator` does not match, i.e. belongs to a different iteration.
        """
        if self._total_index or self._page_index:
            raise InvalidArgumentException('thaw() called on already-used iterator.')
        if (self._query_hash != frozen.query_hash or self._query_variables != frozen.query_variables or 
            self._query_referer != frozen.query_referer or (self._context.username != frozen.context_username) or 
            (self._doc_id != frozen.doc_id)):
            raise InvalidArgumentException('Mismatching resume information.')
        if not frozen.best_before:
            raise InvalidArgumentException('"best before" date missing.')
        if frozen.remaining_data is None:
            raise InvalidArgumentException('"remaining_data" missing.')
        self._total_index = frozen.total_index
        self._best_before = datetime.fromtimestamp(cast(float, frozen.best_before))
        self._data = frozen.remaining_data
        if frozen.first_node is not None:
            self._first_node = frozen.first_node

LoadFunction = Callable[[InstaloaderContext, str], Any]
SaveFunction = Callable[[FrozenNodeIterator, str], None]
FormatPathFunction = Callable[[str], str]

@contextmanager
def resumable_iteration(
    context: InstaloaderContext,
    iterator: Iterator[Any],
    load: LoadFunction,
    save: SaveFunction,
    format_path: FormatPathFunction,
    check_bbd: bool = True,
    enabled: bool = True
) -> Iterator[Tuple[bool, int]]:
    """
    High-level context manager to handle a resumable iteration that can be interrupted
    with a :class:`KeyboardInterrupt` or an :class:`AbortDownloadException`.

    It can be used as follows to automatically load a previously-saved state into the iterator, save the iterator's
    state when interrupted, and delete the resume file upon completion::

       post_iterator = profile.get_posts()
       with resumable_iteration(
               context=L.context,
               iterator=post_iterator,
               load=lambda _, path: FrozenNodeIterator(**json.load(open(path))),
               save=lambda fni, path: json.dump(fni._asdict(), open(path, 'w')),
               format_path=lambda magic: "resume_info_{}.json".format(magic)
       ) as (is_resuming, start_index):
           for post in post_iterator:
               do_something_with(post)

    It yields a tuple (is_resuming, start_index).

    When the passed iterator is not a :class:`NodeIterator`, it behaves as if ``resumable_iteration`` was not used,
    just executing the inner body.

    :param context: The :class:`InstaloaderContext`.
    :param iterator: The fresh :class:`NodeIterator`.
    :param load: Loads a FrozenNodeIterator from given path. The object is ignored if it has a different type.
    :param save: Saves the given FrozenNodeIterator to the given path.
    :param format_path: Returns the path to the resume file for the given magic.
    :param check_bbd: Whether to check the best before date and reject an expired FrozenNodeIterator.
    :param enabled: Set to False to disable all functionality and simply execute the inner body.

    .. versionchanged:: 4.7
       Also interrupt on :class:`AbortDownloadException`.
    """
    if not enabled or not isinstance(iterator, NodeIterator):
        yield (False, 0)
        return
    is_resuming: bool = False
    start_index: int = 0
    resume_file_path: str = format_path(iterator.magic)
    resume_file_exists: bool = os.path.isfile(resume_file_path)
    if resume_file_exists:
        try:
            fni: Any = load(context, resume_file_path)
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
