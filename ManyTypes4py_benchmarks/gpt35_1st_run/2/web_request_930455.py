from __future__ import annotations
import asyncio
import datetime
import io
import re
import socket
import string
import sys
import tempfile
import types
from http.cookies import SimpleCookie
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Final, Iterator, Mapping, MutableMapping, Optional, Pattern, Tuple, Union, cast
from urllib.parse import parse_qsl
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import _SENTINEL, ETAG_ANY, LIST_QUOTED_ETAG_RE, ChainMapProxy, ETag, HeadersMixin, frozen_dataclass_decorator, is_expected_content_type, parse_http_date, reify, sentinel, set_exception
from .http_parser import RawRequestMessage
from .http_writer import HttpVersion
from .multipart import BodyPartReader, MultipartReader
from .streams import EmptyStreamReader, StreamReader
from .typedefs import DEFAULT_JSON_DECODER, JSONDecoder, LooseHeaders, RawHeaders, StrOrURL
from .web_exceptions import HTTPBadRequest, HTTPRequestEntityTooLarge, HTTPUnsupportedMediaType
from .web_response import StreamResponse
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any
__all__ = ('BaseRequest', 'FileField', 'Request')
if TYPE_CHECKING:
    from .web_app import Application
    from .web_protocol import RequestHandler
    from .web_urldispatcher import UrlMappingMatchInfo

@frozen_dataclass_decorator
class FileField:
    pass

_TCHAR: Final = string.digits + string.ascii_letters + "!#$%&'*+.^_`|~-"
_TOKEN: Final = f'[{_TCHAR}]+'
_QDTEXT: Final = '[{}]'.format(''.join((chr(c) for c in (9, 32, 33) + tuple(range(35, 127))))
_QUOTED_PAIR: Final = '\\\\[\\t !-~]'
_QUOTED_STRING: Final = '"(?:{quoted_pair}|{qdtext})*"'.format(qdtext=_QDTEXT, quoted_pair=_QUOTED_PAIR)
_FORWARDED_PAIR: Final = '({token})=({token}|{quoted_string})(:\\d{{1,4}})?'.format(token=_TOKEN, quoted_string=_QUOTED_STRING)
_QUOTED_PAIR_REPLACE_RE: Final = re.compile('\\\\([\\t !-~])')
_FORWARDED_PAIR_RE: Final = re.compile(_FORWARDED_PAIR)

class BaseRequest(MutableMapping[str, Any], HeadersMixin):
    POST_METHODS: Final = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT, hdrs.METH_TRACE, hdrs.METH_DELETE}
    _post: Optional[MultiDictProxy] = None
    _read_bytes: Optional[bytes] = None

    def __init__(self, message: RawRequestMessage, payload: StreamReader, protocol: Any, payload_writer: AbstractStreamWriter, task: Any, loop: Any, *, client_max_size: int = 1024 ** 2, state: Optional[Dict] = None, scheme: Optional[str] = None, host: Optional[str] = None, remote: Optional[str] = None) -> None:
        self._message = message
        self._protocol = protocol
        self._payload_writer = payload_writer
        self._payload = payload
        self._headers = message.headers
        self._method = message.method
        self._version = message.version
        self._cache: Dict[str, Any] = {}
        url = message.url
        if url.absolute:
            if scheme is not None:
                url = url.with_scheme(scheme)
            if host is not None:
                url = url.with_host(host)
            self._cache['url'] = url
            self._cache['host'] = url.host
            self._cache['scheme'] = url.scheme
            self._rel_url = url.relative()
        else:
            self._rel_url = url
            if scheme is not None:
                self._cache['scheme'] = scheme
            if host is not None:
                self._cache['host'] = host
        self._state = {} if state is None else state
        self._task = task
        self._client_max_size = client_max_size
        self._loop = loop
        transport = protocol.transport
        assert transport is not None
        self._transport_sslcontext = transport.get_extra_info('sslcontext')
        self._transport_peername = transport.get_extra_info('peername')
        if remote is not None:
            self._cache['remote'] = remote

    def clone(self, *, method: Any = sentinel, rel_url: Any = sentinel, headers: Any = sentinel, scheme: Any = sentinel, host: Any = sentinel, remote: Any = sentinel, client_max_size: Any = sentinel) -> BaseRequest:
        ...

    @property
    def task(self) -> Any:
        ...

    @property
    def protocol(self) -> Any:
        ...

    @property
    def transport(self) -> Any:
        ...

    @property
    def writer(self) -> AbstractStreamWriter:
        ...

    @property
    def client_max_size(self) -> int:
        ...

    @reify
    def rel_url(self) -> URL:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        ...

    def __delitem__(self, key: str) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    @reify
    def secure(self) -> bool:
        ...

    @reify
    def forwarded(self) -> Tuple[Mapping[str, str], ...]:
        ...

    @reify
    def scheme(self) -> str:
        ...

    @reify
    def method(self) -> str:
        ...

    @reify
    def version(self) -> HttpVersion:
        ...

    @reify
    def host(self) -> str:
        ...

    @reify
    def remote(self) -> str:
        ...

    @reify
    def url(self) -> URL:
        ...

    @reify
    def path(self) -> str:
        ...

    @reify
    def path_qs(self) -> str:
        ...

    @reify
    def raw_path(self) -> str:
        ...

    @reify
    def query(self) -> MultiDictProxy:
        ...

    @reify
    def query_string(self) -> str:
        ...

    @reify
    def headers(self) -> CIMultiDictProxy:
        ...

    @reify
    def raw_headers(self) -> Tuple[Tuple[bytes, bytes], ...]:
        ...

    @reify
    def if_modified_since(self) -> Optional[datetime.datetime]:
        ...

    @reify
    def if_unmodified_since(self) -> Optional[datetime.datetime]:
        ...

    @reify
    def if_match(self) -> Optional[Tuple[ETag, ...]]:
        ...

    @reify
    def if_none_match(self) -> Optional[Tuple[ETag, ...]]:
        ...

    @reify
    def if_range(self) -> Optional[datetime.datetime]:
        ...

    @reify
    def keep_alive(self) -> bool:
        ...

    @reify
    def cookies(self) -> MappingProxyType[str, str]:
        ...

    @reify
    def http_range(self) -> slice:
        ...

    @reify
    def content(self) -> StreamReader:
        ...

    @property
    def can_read_body(self) -> bool:
        ...

    @reify
    def body_exists(self) -> bool:
        ...

    async def release(self) -> None:
        ...

    async def read(self) -> bytes:
        ...

    async def text(self) -> str:
        ...

    async def json(self, *, loads: JSONDecoder = DEFAULT_JSON_DECODER, content_type: str = 'application/json') -> Any:
        ...

    async def multipart(self) -> MultipartReader:
        ...

    async def post(self) -> MultiDictProxy:
        ...

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __bool__(self) -> bool:
        ...

    async def _prepare_hook(self, response: StreamResponse) -> None:
        ...

    def _cancel(self, exc: Exception) -> None:
        ...

    def _finish(self) -> None:
        ...

class Request(BaseRequest):
    _match_info: Optional[UrlMappingMatchInfo] = None

    def clone(self, *, method: Any = sentinel, rel_url: Any = sentinel, headers: Any = sentinel, scheme: Any = sentinel, host: Any = sentinel, remote: Any = sentinel, client_max_size: Any = sentinel) -> Request:
        ...

    @reify
    def match_info(self) -> UrlMappingMatchInfo:
        ...

    @property
    def app(self) -> Application:
        ...

    @property
    def config_dict(self) -> ChainMapProxy:
        ...

    async def _prepare_hook(self, response: StreamResponse) -> None:
        ...
