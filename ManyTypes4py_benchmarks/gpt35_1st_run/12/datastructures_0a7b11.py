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

class URL:

    def __init__(self, url: str = '', scope: typing.Optional[Scope] = None, **components: typing.Any) -> None:
        ...

    @property
    def components(self) -> SplitResult:
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
    def username(self) -> str:
        ...

    @property
    def password(self) -> str:
        ...

    @property
    def hostname(self) -> str:
        ...

    @property
    def port(self) -> typing.Optional[int]:
        ...

    @property
    def is_secure(self) -> bool:
        ...

    def replace(self, **kwargs: typing.Any) -> URL:
        ...

    def include_query_params(self, **kwargs: typing.Any) -> URL:
        ...

    def replace_query_params(self, **kwargs: typing.Any) -> URL:
        ...

    def remove_query_params(self, keys: typing.Union[str, typing.List[str]]) -> URL:
        ...

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

class URLPath(str):
    ...

class Secret:

    def __init__(self, value: str) -> None:
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

    def __getitem__(self, index: int) -> str:
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

    def multi_items(self) -> typing.List[typing.Tuple[_KeyType, _CovariantValueType]]:
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

class MultiDict(ImmutableMultiDict[typing.Any, typing.Any]):

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def __delitem__(self, key: typing.Any) -> None:
        ...

    def pop(self, key: typing.Any, default: typing.Any = None) -> typing.Any:
        ...

    def popitem(self) -> typing.Tuple[typing.Any, typing.Any]:
        ...

    def poplist(self, key: typing.Any) -> typing.List[typing.Any]:
        ...

    def clear(self) -> None:
        ...

    def setdefault(self, key: typing.Any, default: typing.Any = None) -> typing.Any:
        ...

    def setlist(self, key: typing.Any, values: typing.Sequence[typing.Any]) -> None:
        ...

    def append(self, key: typing.Any, value: typing.Any) -> None:
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

class UploadFile:

    def __init__(self, file: typing.Any, *, size: typing.Optional[int] = None, filename: typing.Optional[str] = None, headers: typing.Optional[Headers] = None) -> None:
        ...

    @property
    def content_type(self) -> typing.Optional[str]:
        ...

    @property
    def _in_memory(self) -> bool:
        ...

    async def write(self, data: bytes) -> None:
        ...

    async def read(self, size: int = -1) -> bytes:
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

    def __init__(self, headers: typing.Optional[typing.Mapping[str, str]] = None, raw: typing.Optional[typing.List[typing.Tuple[bytes, bytes]]] = None, scope: typing.Optional[Scope] = None) -> None:
        ...

    @property
    def raw(self) -> typing.List[typing.Tuple[bytes, bytes]]:
        ...

    def keys(self) -> typing.List[str]:
        ...

    def values(self) -> typing.List[str]:
        ...

    def items(self) -> typing.List[typing.Tuple[str, str]]:
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

    def __ior__(self, other: typing.Mapping[str, str]) -> MutableHeaders:
        ...

    def __or__(self, other: typing.Mapping[str, str]) -> MutableHeaders:
        ...

    @property
    def raw(self) -> typing.List[typing.Tuple[bytes, bytes]]:
        ...

    def setdefault(self, key: str, value: str) -> str:
        ...

    def update(self, other: typing.Mapping[str, str]) -> None:
        ...

    def append(self, key: str, value: str) -> None:
        ...

    def add_vary_header(self, vary: str) -> None:
        ...
