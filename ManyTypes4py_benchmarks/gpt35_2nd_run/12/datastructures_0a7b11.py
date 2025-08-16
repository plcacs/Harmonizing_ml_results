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
    ...

class CommaSeparatedStrings(typing.Sequence[str]):
    ...

class ImmutableMultiDict(typing.Mapping[_KeyType, _CovariantValueType]):
    ...

class MultiDict(ImmutableMultiDict[typing.Any, typing.Any]):
    ...

class QueryParams(ImmutableMultiDict[str, str]):
    ...

class UploadFile:
    ...

class FormData(ImmutableMultiDict[str, typing.Union[UploadFile, str]]):
    ...

class Headers(typing.Mapping[str, str]):
    ...

class MutableHeaders(Headers):
    ...

class State:
    ...
