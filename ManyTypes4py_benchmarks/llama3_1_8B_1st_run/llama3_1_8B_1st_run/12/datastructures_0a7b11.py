from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope

class Address(typing.NamedTuple):
    pass

_KeyType = typing.TypeVar('_KeyType')
_CovariantValueType = typing.TypeVar('_CovariantValueType', covariant=True)

class URL(typing.Protocol):
    def __init__(self, url: str = '', scope: typing.Optional[Scope] = None, **components: typing.Any) -> None:
        ...

    @property
    def components(self) -> urlsplit:
        ...

    @property
    def scheme(self) -> str:
        ...

    @property
    def netloc(self) -> str:
        ...

    @property
    def path(self) -> str:
        ...

    @property
    def query(self) -> str:
        ...

    @property
    def fragment(self) -> str:
        ...

    @property
    def username(self) -> typing.Optional[str]:
        ...

    @property
    def password(self) -> typing.Optional[str]:
        ...

    @property
    def hostname(self) -> typing.Optional[str]:
        ...

    @property
    def port(self) -> typing.Optional[int]:
        ...

    @property
    def is_secure(self) -> bool:
        ...

    def replace(self, **kwargs: typing.Any) -> 'URL':
        ...

    def include_query_params(self, **kwargs: typing.Any) -> 'URL':
        ...

    def replace_query_params(self, **kwargs: typing.Any) -> 'URL':
        ...

    def remove_query_params(self, keys: typing.Union[str, typing.List[str]]) -> 'URL':
        ...

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

class URLPath(str):
    """
    A URL path string that may also hold an associated protocol and/or host.
    Used by the routing to return `url_path_for` matches.
    """

    def __new__(cls, path: str, protocol: str = '', host: str = '') -> str:
        assert protocol in ('http', 'websocket', '')
        return str.__new__(cls, path)

    def __init__(self, path: str, protocol: str = '', host: str = '') -> None:
        self.protocol = protocol
        self.host = host

    def make_absolute_url(self, base_url: typing.Union[str, URL]) -> URL:
        ...

class Secret(typing.Protocol):
    def __init__(self, value: typing.Any) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __bool__(self) -> bool:
        ...

class CommaSeparatedStrings(typing.Sequence[str]):
    def __init__(self, value: typing.Union[str, typing.Sequence[str]]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: typing.Any) -> str:
        ...

    def __iter__(self) -> typing.Iterator[str]:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class ImmutableMultiDict(typing.Mapping[_KeyType, _CovariantValueType]):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

    def getlist(self, key: _KeyType) -> typing.List[_CovariantValueType]:
        ...

    def keys(self) -> typing.KeysView[_KeyType]:
        ...

    def values(self) -> typing.ValuesView[_CovariantValueType]:
        ...

    def items(self) -> typing.ItemsView[_KeyType, _CovariantValueType]:
        ...

    def multi_items(self) -> typing.List[tuple[_KeyType, _CovariantValueType]]:
        ...

    def __getitem__(self, key: _KeyType) -> _CovariantValueType:
        ...

    def __contains__(self, key: _KeyType) -> bool:
        ...

    def __iter__(self) -> typing.Iterator[_KeyType]:
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class MultiDict(ImmutableMultiDict[_KeyType, _CovariantValueType]):
    def __setitem__(self, key: _KeyType, value: _CovariantValueType) -> None:
        ...

    def __delitem__(self, key: _KeyType) -> None:
        ...

    def pop(self, key: _KeyType, default: typing.Any = ...) -> _CovariantValueType:
        ...

    def popitem(self) -> typing.Tuple[_KeyType, _CovariantValueType]:
        ...

    def poplist(self, key: _KeyType) -> typing.List[_CovariantValueType]:
        ...

    def clear(self) -> None:
        ...

    def setdefault(self, key: _KeyType, default: typing.Any = ...) -> _CovariantValueType:
        ...

    def setlist(self, key: _KeyType, values: typing.List[_CovariantValueType]) -> None:
        ...

    def append(self, key: _KeyType, value: _CovariantValueType) -> None:
        ...

    def update(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

class QueryParams(ImmutableMultiDict[str, str]):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

class UploadFile(typing.Protocol):
    def __init__(self, file: typing.Any, *, size: typing.Optional[int] = None, filename: typing.Optional[str] = None, headers: typing.Optional[Headers] = None) -> None:
        ...

    @property
    def content_type(self) -> typing.Optional[str]:
        ...

    @property
    def _in_memory(self) -> bool:
        ...

    async def write(self, data: typing.Any) -> None:
        ...

    async def read(self, size: typing.Optional[int] = -1) -> typing.Any:
        ...

    async def seek(self, offset: int) -> None:
        ...

    async def close(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

class FormData(ImmutableMultiDict[str, typing.Union[UploadFile, str]]):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

    async def close(self) -> None:
        ...

class Headers(typing.Mapping[str, str]):
    def __init__(self, headers: typing.Optional[typing.Mapping[str, str]] = None, raw: typing.Optional[typing.List[tuple[str, str]]] = None, scope: typing.Optional[Scope] = None) -> None:
        ...

    @property
    def raw(self) -> typing.List[tuple[str, str]]:
        ...

    def keys(self) -> typing.KeysView[str]:
        ...

    def values(self) -> typing.ValuesView[str]:
        ...

    def items(self) -> typing.ItemsView[str, str]:
        ...

    def getlist(self, key: str) -> typing.List[str]:
        ...

    def mutablecopy(self) -> MutableHeaders:
        ...

    def __getitem__(self, key: str) -> str:
        ...

    def __contains__(self, key: str) -> bool:
        ...

    def __iter__(self) -> typing.Iterator[str]:
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class MutableHeaders(Headers):
    def __setitem__(self, key: str, value: str) -> None:
        ...

    def __delitem__(self, key: str) -> None:
        ...

    def __ior__(self, other: typing.Mapping[str, str]) -> 'MutableHeaders':
        ...

    def __or__(self, other: typing.Mapping[str, str]) -> 'MutableHeaders':
        ...

    @property
    def raw(self) -> typing.List[tuple[str, str]]:
        ...

    def setdefault(self, key: str, value: str) -> str:
        ...

    def update(self, other: typing.Mapping[str, str]) -> None:
        ...

    def append(self, key: str, value: str) -> None:
        ...

    def add_vary_header(self, vary: str) -> None:
        ...

class State(typing.Protocol):
    def __init__(self, state: typing.Any = None) -> None:
        ...

    def __setattr__(self, key: str, value: typing.Any) -> None:
        ...

    def __getattr__(self, key: str) -> typing.Any:
        ...

    def __delattr__(self, key: str) -> None:
        ...
