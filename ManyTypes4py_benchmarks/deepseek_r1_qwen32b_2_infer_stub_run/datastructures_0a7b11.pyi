from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from urllib.parse import SplitResult
from starlette.types import Scope
from shlex import shlex

_KeyType = TypeVar('_KeyType')
_CovariantValueType = TypeVar('_CovariantValueType', covariant=True)

class Address(typing.NamedTuple):
    ...

class URL:
    _url: str
    _components: SplitResult

    def __init__(self, url: str = '', scope: Optional[Scope] = None, **components: Any) -> None:
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
    def port(self) -> Optional[int]:
        ...

    @property
    def is_secure(self) -> bool:
        ...

    def replace(self, **kwargs: Any) -> URL:
        ...

    def include_query_params(self, **kwargs: Any) -> URL:
        ...

    def replace_query_params(self, **kwargs: Any) -> URL:
        ...

    def remove_query_params(self, keys: Union[str, List[str]]) -> URL:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

class URLPath(str):
    protocol: str
    host: str

    def __new__(cls, path: str, protocol: str = '', host: str = '') -> URLPath:
        ...

    def __init__(self, path: str, protocol: str = '', host: str = '') -> None:
        ...

    def make_absolute_url(self, base_url: Union[str, URL]) -> URL:
        ...

class Secret:
    _value: str

    def __init__(self, value: str) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __bool__(self) -> bool:
        ...

class CommaSeparatedStrings(typing.Sequence[str]):
    def __init__(self, value: Union[str, List[str]]) -> None:
        ...

    def __getitem__(self, index: int) -> str:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> typing.Iterator[str]:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

class ImmutableMultiDict(typing.Mapping[_KeyType, _CovariantValueType]):
    _dict: Dict[_KeyType, _CovariantValueType]
    _list: List[Tuple[_KeyType, _CovariantValueType]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def getlist(self, key: _KeyType) -> List[_CovariantValueType]:
        ...

    def keys(self) -> List[_KeyType]:
        ...

    def values(self) -> List[_CovariantValueType]:
        ...

    def items(self) -> List[Tuple[_KeyType, _CovariantValueType]]:
        ...

    def multi_items(self) -> List[Tuple[_KeyType, _CovariantValueType]]:
        ...

    def __getitem__(self, key: _KeyType) -> _CovariantValueType:
        ...

    def __contains__(self, key: _KeyType) -> bool:
        ...

    def __iter__(self) -> typing.Iterator[_KeyType]:
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class MultiDict(ImmutableMultiDict[typing.Any, typing.Any]):
    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def __delitem__(self, key: typing.Any) -> None:
        ...

    def pop(self, key: typing.Any, default: Optional[typing.Any] = None) -> typing.Any:
        ...

    def popitem(self) -> Tuple[typing.Any, typing.Any]:
        ...

    def poplist(self, key: typing.Any) -> List[typing.Any]:
        ...

    def clear(self) -> None:
        ...

    def setdefault(self, key: typing.Any, default: Optional[typing.Any] = None) -> typing.Any:
        ...

    def setlist(self, key: typing.Any, values: List[typing.Any]) -> None:
        ...

    def append(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def update(self, *args: Any, **kwargs: Any) -> None:
        ...

class QueryParams(ImmutableMultiDict[str, str]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

class UploadFile:
    filename: str
    file: Any
    size: Optional[int]
    headers: Headers

    def __init__(self, file: Any, size: Optional[int] = None, filename: Optional[str] = None, headers: Optional[Headers] = None) -> None:
        ...

    @property
    def content_type(self) -> Optional[str]:
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

class FormData(ImmutableMultiDict[str, Union[UploadFile, str]]):
    async def close(self) -> None:
        ...

class Headers(typing.Mapping[str, str]):
    _list: List[Tuple[bytes, bytes]]

    def __init__(self, headers: Optional[Dict[str, str]] = None, raw: Optional[List[Tuple[bytes, bytes]]] = None, scope: Optional[Scope] = None) -> None:
        ...

    @property
    def raw(self) -> List[Tuple[bytes, bytes]]:
        ...

    def keys(self) -> List[str]:
        ...

    def values(self) -> List[str]:
        ...

    def items(self) -> List[Tuple[str, str]]:
        ...

    def getlist(self, key: str) -> List[str]:
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

    def __eq__(self, other: Any) -> bool:
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

    def setdefault(self, key: str, value: str) -> str:
        ...

    def update(self, other: typing.Mapping[str, str]) -> None:
        ...

    def append(self, key: str, value: str) -> None:
        ...

    def add_vary_header(self, vary: str) -> None:
        ...

class State:
    _state: Dict[str, Any]

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        ...

    def __setattr__(self, key: str, value: Any) -> None:
        ...

    def __getattr__(self, key: str) -> Any:
        ...

    def __delattr__(self, key: str) -> None:
        ...