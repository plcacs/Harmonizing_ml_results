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
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Final,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Pattern,
    Tuple,
    Union,
    cast,
    List,
    Set,
    Sequence,
    TypeVar,
    Generic,
    Callable,
    Awaitable,
    Coroutine,
    Type,
    overload,
)
from urllib.parse import parse_qsl
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
    _SENTINEL,
    ETAG_ANY,
    LIST_QUOTED_ETAG_RE,
    ChainMapProxy,
    ETag,
    HeadersMixin,
    frozen_dataclass_decorator,
    is_expected_content_type,
    parse_http_date,
    reify,
    sentinel,
    set_exception,
)
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
    from typing_extensions import Self

__all__ = ('BaseRequest', 'FileField', 'Request')

if TYPE_CHECKING:
    from .web_app import Application
    from .web_protocol import RequestHandler
    from .web_urldispatcher import UrlMappingMatchInfo

T = TypeVar('T')
JSONType = TypeVar('JSONType', bound=Any)

@frozen_dataclass_decorator
class FileField:
    name: str
    filename: Optional[str]
    file: io.BufferedReader
    content_type: str
    headers: 'CIMultiDict[str]'

_TCHAR: Final[str] = string.digits + string.ascii_letters + "!#$%&'*+.^_`|~-"
_TOKEN: Final[str] = f'[{_TCHAR}]+'
_QDTEXT: Final[str] = '[{}]'.format(''.join((chr(c) for c in (9, 32, 33) + tuple(range(35, 127)))))
_QUOTED_PAIR: Final[str] = '\\\\[\\t !-~]'
_QUOTED_STRING: Final[str] = '"(?:{quoted_pair}|{qdtext})*"'.format(qdtext=_QDTEXT, quoted_pair=_QUOTED_PAIR)
_FORWARDED_PAIR: Final[str] = '({token})=({token}|{quoted_string})(:\\d{{1,4}})?'.format(token=_TOKEN, quoted_string=_QUOTED_STRING)
_QUOTED_PAIR_REPLACE_RE: Final[Pattern[str]] = re.compile('\\\\([\\t !-~])')
_FORWARDED_PAIR_RE: Final[Pattern[str]] = re.compile(_FORWARDED_PAIR)

class BaseRequest(MutableMapping[str, Any], HeadersMixin):
    POST_METHODS: Final[Set[str]] = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT, hdrs.METH_TRACE, hdrs.METH_DELETE}
    _post: Optional[MultiDictProxy[Union[str, FileField]]]
    _read_bytes: Optional[bytes]

    def __init__(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: 'RequestHandler',
        payload_writer: AbstractStreamWriter,
        task: 'asyncio.Task[Any]',
        loop: asyncio.AbstractEventLoop,
        *,
        client_max_size: int = 1024 ** 2,
        state: Optional[Dict[str, Any]] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        remote: Optional[str] = None,
    ) -> None:
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

    def clone(
        self,
        *,
        method: Union[str, Any] = sentinel,
        rel_url: Union[StrOrURL, Any] = sentinel,
        headers: Optional[LooseHeaders] = sentinel,
        scheme: Union[str, Any] = sentinel,
        host: Union[str, Any] = sentinel,
        remote: Union[str, Any] = sentinel,
        client_max_size: Union[int, Any] = sentinel,
    ) -> Self:
        if self._read_bytes:
            raise RuntimeError('Cannot clone request after reading its content')
        dct: Dict[str, Any] = {}
        if method is not sentinel:
            dct['method'] = method
        if rel_url is not sentinel:
            new_url = URL(rel_url)
            dct['url'] = new_url
            dct['path'] = str(new_url)
        if headers is not sentinel:
            new_headers = CIMultiDictProxy(CIMultiDict(headers))
            dct['headers'] = new_headers
            dct['raw_headers'] = tuple(((k.encode('utf-8'), v.encode('utf-8')) for k, v in new_headers.items()))
        message = self._message._replace(**dct)
        kwargs: Dict[str, Any] = {}
        if scheme is not sentinel:
            kwargs['scheme'] = scheme
        if host is not sentinel:
            kwargs['host'] = host
        if remote is not sentinel:
            kwargs['remote'] = remote
        if client_max_size is sentinel:
            client_max_size = self._client_max_size
        return self.__class__(
            message,
            self._payload,
            self._protocol,
            self._payload_writer,
            self._task,
            self._loop,
            client_max_size=client_max_size,
            state=self._state.copy(),
            **kwargs,
        )

    @property
    def task(self) -> 'asyncio.Task[Any]':
        return self._task

    @property
    def protocol(self) -> 'RequestHandler':
        return self._protocol

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        return self._protocol.transport

    @property
    def writer(self) -> AbstractStreamWriter:
        return self._payload_writer

    @property
    def client_max_size(self) -> int:
        return self._client_max_size

    @reify
    def rel_url(self) -> URL:
        return self._rel_url

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state[key] = value

    def __delitem__(self, key: str) -> None:
        del self._state[key]

    def __len__(self) -> int:
        return len(self._state)

    def __iter__(self) -> Iterator[str]:
        return iter(self._state)

    @reify
    def secure(self) -> bool:
        return self.scheme == 'https'

    @reify
    def forwarded(self) -> Tuple[MappingProxyType[str, str], ...]:
        elems: List[Dict[str, str]] = []
        for field_value in self._message.headers.getall(hdrs.FORWARDED, ()):
            length = len(field_value)
            pos = 0
            need_separator = False
            elem: Dict[str, str] = {}
            elems.append(types.MappingProxyType(elem))
            while 0 <= pos < length:
                match = _FORWARDED_PAIR_RE.match(field_value, pos)
                if match is not None:
                    if need_separator:
                        pos = field_value.find(',', pos)
                    else:
                        name, value, port = match.groups()
                        if value[0] == '"':
                            value = _QUOTED_PAIR_REPLACE_RE.sub('\\1', value[1:-1])
                        if port:
                            value += port
                        elem[name.lower()] = value
                        pos += len(match.group(0))
                        need_separator = True
                elif field_value[pos] == ',':
                    need_separator = False
                    elem = {}
                    elems.append(types.MappingProxyType(elem))
                    pos += 1
                elif field_value[pos] == ';':
                    need_separator = False
                    pos += 1
                elif field_value[pos] in ' \t':
                    pos += 1
                else:
                    pos = field_value.find(',', pos)
        return tuple(elems)

    @reify
    def scheme(self) -> str:
        if self._transport_sslcontext:
            return 'https'
        else:
            return 'http'

    @reify
    def method(self) -> str:
        return self._method

    @reify
    def version(self) -> HttpVersion:
        return self._version

    @reify
    def host(self) -> str:
        host = self._message.headers.get(hdrs.HOST)
        if host is not None:
            return host
        return socket.getfqdn()

    @reify
    def remote(self) -> Optional[str]:
        if self._transport_peername is None:
            return None
        if isinstance(self._transport_peername, (list, tuple)):
            return str(self._transport_peername[0])
        return str(self._transport_peername)

    @reify
    def url(self) -> URL:
        return URL.build(scheme=self.scheme, authority=self.host).join(self._rel_url)

    @reify
    def path(self) -> str:
        return self._rel_url.path

    @reify
    def path_qs(self) -> str:
        return str(self._rel_url)

    @reify
    def raw_path(self) -> str:
        return self._message.path

    @reify
    def query(self) -> 'MultiDictProxy[str]':
        return self._rel_url.query

    @reify
    def query_string(self) -> str:
        return self._rel_url.query_string

    @reify
    def headers(self) -> 'CIMultiDictProxy[str]':
        return self._headers

    @reify
    def raw_headers(self) -> RawHeaders:
        return self._message.raw_headers

    @reify
    def if_modified_since(self) -> Optional[datetime.datetime]:
        return parse_http_date(self.headers.get(hdrs.IF_MODIFIED_SINCE))

    @reify
    def if_unmodified_since(self) -> Optional[datetime.datetime]:
        return parse_http_date(self.headers.get(hdrs.IF_UNMODIFIED_SINCE))

    @staticmethod
    def _etag_values(etag_header: str) -> Iterator[ETag]:
        if etag_header == ETAG_ANY:
            yield ETag(is_weak=False, value=ETAG_ANY)
        else:
            for match in LIST_QUOTED_ETAG_RE.finditer(etag_header):
                is_weak, value, garbage = match.group(2, 3, 4)
                if garbage:
                    break
                yield ETag(is_weak=bool(is_weak), value=value)

    @classmethod
    def _if_match_or_none_impl(cls, header_value: Optional[str]) -> Optional[Tuple[ETag, ...]]:
        if not header_value:
            return None
        return tuple(cls._etag_values(header_value))

    @reify
    def if_match(self) -> Optional[Tuple[ETag, ...]]:
        return self._if_match_or_none_impl(self.headers.get(hdrs.IF_MATCH))

    @reify
    def if_none_match(self) -> Optional[Tuple[ETag, ...]]:
        return self._if_match_or_none_impl(self.headers.get(hdrs.IF_NONE_MATCH))

    @reify
    def if_range(self) -> Optional[datetime.datetime]:
        return parse_http_date(self.headers.get(hdrs.IF_RANGE))

    @reify
    def keep_alive(self) -> bool:
        return not self._message.should_close

    @reify
    def cookies(self) -> Mapping[str, str]:
        raw = self.headers.get(hdrs.COOKIE, '')
        parsed = SimpleCookie(raw)
        return MappingProxyType({key: val.value for key, val in parsed.items()})

    @reify
    def http_range(self) -> slice:
        rng = self._headers.get(hdrs.RANGE)
        start, end = (None, None)
        if rng is not None:
            try:
                pattern = '^bytes=(\\d*)-(\\d*)$'
                start, end = re.findall(pattern, rng)[0]
            except IndexError:
                raise ValueError('range not in acceptable format')
            end = int(end) if end else None
            start = int(start) if start else None
            if start is None and end is not None:
                start = -end
                end = None
            if start is not None and end is not None:
                end += 1
                if start >= end:
                    raise ValueError('start cannot be after end')
            if start is end is None:
                raise ValueError('No start or end of range specified')
        return slice(start, end, 1)

    @reify
    def content(self) -> StreamReader:
        return self._payload

    @property
    def can_read_body(self) -> bool:
        return not self._payload.at_eof()

    @reify
    def body_exists(self) -> bool:
        return type(self._payload) is not EmptyStreamReader

    async def release(self) -> None:
        while not self._payload.at_eof():
            await self._payload.readany()

    async def read(self) -> bytes:
        if self._read_bytes is None:
            body = bytearray()
            while True:
                chunk = await self._payload.readany()
                body.extend(chunk)
                if self._client_max_size:
                    body_size = len(body)
                    if body_size > self._client_max_size:
                        raise HTTPRequestEntityTooLarge(max_size=self._client_max_size, actual_size=body_size)
                if not chunk:
                    break
            self._read_bytes = bytes(body)
        return self._read_bytes

    async def text(self) -> str:
        bytes_body = await self.read()
        encoding = self.charset or 'utf-8'
        try:
            return bytes_body.decode(encoding)
        except LookupError:
            raise HTTPUnsupportedMediaType()

    async def json(self, *, loads: JSONDecoder = DEFAULT_JSON_DECODER, content_type: str = 'application/json') -> Any:
        body = await self.text()
        if content_type:
            if not is_expected_content_type(self.content_type, content_type):
                raise HTTPBadRequest(text='Attempt to decode JSON with unexpected mimetype: %s' % self.content_type)
        return loads(body)

    async def multipart(self) -> MultipartReader:
        return MultipartReader(self._headers, self._payload)

    async def post(self) -> 'MultiDictProxy[Union[str, FileField]]':
        if self._post is not None:
            return self._post
        if self._method not in self.POST_METHODS:
            self._post = MultiDictProxy(MultiDict())
            return self._post
        content_type = self.content_type
        if content_type not in ('', 'application/x-www-form-urlencoded', 'multipart/form-data'):
            self._post = MultiDictProxy(MultiDict())
            return self._post
        out = MultiDict()
        if content_type == 'multipart/form-data':
            multipart = await self.multipart()
            max_size = self._client_max_size
            field = await multipart.next()
            while field is not None:
                size = 0
                field_ct = field.headers.get(hdrs.CONTENT_TYPE)
                if isinstance(field, BodyPartReader):
                    assert field.name is not None
                    if field.filename:
                        tmp = await self._loop.run_in_executor(None, tempfile.TemporaryFile)
                        chunk = await field.read_chunk(size=2 ** 16)
                        while chunk:
                            chunk = field.decode(chunk)
                            await self._loop.run_in_executor(None, tmp.write, chunk)
                            size += len(chunk)
                            if 0 < max_size < size:
                                await self._loop.run_in_executor(None, tmp.close)
                                raise HTTPRequestEntityTooLarge(max_size=max_size, actual_size=size)
                            chunk = await field.read_chunk(size=2 ** 16)
                        await self._loop.run_in_executor(None, tmp.seek, 