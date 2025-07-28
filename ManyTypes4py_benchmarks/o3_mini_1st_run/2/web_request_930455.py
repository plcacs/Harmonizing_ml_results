import asyncio
import datetime
import io
import re
import socket
import string
import sys
import tempfile
import types
from collections.abc import Iterator, Mapping
from http.cookies import SimpleCookie
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    MutableMapping,
    Optional,
    Pattern,
    Tuple,
    Union,
    cast,
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
    Self = Any

if TYPE_CHECKING:
    from .web_app import Application
    from .web_protocol import RequestHandler
    from .web_urldispatcher import UrlMappingMatchInfo

__all__ = ('BaseRequest', 'FileField', 'Request')

@frozen_dataclass_decorator
class FileField:
    # The FileField dataclass; add appropriate attributes if needed.
    # For now, using pass as in the original code.
    pass

_TCHAR: str = string.digits + string.ascii_letters + "!#$%&'*+.^_`|~-"
_TOKEN: str = f'[{_TCHAR}]+'
_QDTEXT: str = '[{}]'.format(''.join((chr(c) for c in (9, 32, 33) + tuple(range(35, 127)))))
_QUOTED_PAIR: str = '\\\\[\\t !-~]'
_QUOTED_STRING: str = '"(?:{quoted_pair}|{qdtext})*"'.format(qdtext=_QDTEXT, quoted_pair=_QUOTED_PAIR)
_FORWARDED_PAIR: str = '({token})=({token}|{quoted_string})(:\\d{{1,4}})?'.format(token=_TOKEN, quoted_string=_QUOTED_STRING)
_QUOTED_PAIR_REPLACE_RE: Pattern[str] = re.compile('\\\\([\\t !-~])')
_FORWARDED_PAIR_RE: Pattern[str] = re.compile(_FORWARDED_PAIR)


class BaseRequest(MutableMapping[str, Any], HeadersMixin):
    POST_METHODS: Final[set[str]] = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT, hdrs.METH_TRACE, hdrs.METH_DELETE}
    _post: Optional[MultiDictProxy[Any]] = None
    _read_bytes: Optional[bytes] = None

    def __init__(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: Any,
        payload_writer: AbstractStreamWriter,
        task: asyncio.Task[Any],
        loop: asyncio.AbstractEventLoop,
        *,
        client_max_size: int = 1024 ** 2,
        state: Optional[Dict[str, Any]] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        remote: Optional[str] = None,
    ) -> None:
        self._message: RawRequestMessage = message
        self._protocol: Any = protocol
        self._payload_writer: AbstractStreamWriter = payload_writer
        self._payload: StreamReader = payload
        self._headers: CIMultiDictProxy[str] = message.headers
        self._method: str = message.method
        self._version: HttpVersion = message.version
        self._cache: Dict[str, Any] = {}
        url: URL = message.url
        if url.absolute:
            if scheme is not None:
                url = url.with_scheme(scheme)
            if host is not None:
                url = url.with_host(host)
            self._cache['url'] = url
            self._cache['host'] = url.host
            self._cache['scheme'] = url.scheme
            self._rel_url: URL = url.relative()
        else:
            self._rel_url = url
            if scheme is not None:
                self._cache['scheme'] = scheme
            if host is not None:
                self._cache['host'] = host
        self._state: Dict[str, Any] = {} if state is None else state
        self._task: asyncio.Task[Any] = task
        self._client_max_size: int = client_max_size
        self._loop: asyncio.AbstractEventLoop = loop
        transport: Any = protocol.transport
        assert transport is not None
        self._transport_sslcontext: Any = transport.get_extra_info('sslcontext')
        self._transport_peername: Any = transport.get_extra_info('peername')
        if remote is not None:
            self._cache['remote'] = remote

    def clone(
        self,
        *,
        method: Any = sentinel,
        rel_url: Any = sentinel,
        headers: Any = sentinel,
        scheme: Any = sentinel,
        host: Any = sentinel,
        remote: Any = sentinel,
        client_max_size: Any = sentinel,
    ) -> Self:
        if self._read_bytes:
            raise RuntimeError('Cannot clone request after reading its content')
        dct: Dict[str, Any] = {}
        if method is not sentinel:
            dct['method'] = method
        if rel_url is not sentinel:
            new_url: URL = URL(rel_url)
            dct['url'] = new_url
            dct['path'] = str(new_url)
        if headers is not sentinel:
            new_headers: CIMultiDictProxy[str] = CIMultiDictProxy(CIMultiDict(headers))
            dct['headers'] = new_headers
            dct['raw_headers'] = tuple(((k.encode('utf-8'), v.encode('utf-8')) for k, v in new_headers.items()))
        message: RawRequestMessage = self._message._replace(**dct)
        kwargs: Dict[str, Any] = {}
        if scheme is not sentinel:
            kwargs['scheme'] = scheme
        if host is not sentinel:
            kwargs['host'] = host
        if remote is not sentinel:
            kwargs['remote'] = remote
        if client_max_size is sentinel:
            client_max_size = self._client_max_size
        return self.__class__(message, self._payload, self._protocol, self._payload_writer, self._task, self._loop, client_max_size=client_max_size, state=self._state.copy(), **kwargs)

    @property
    def task(self) -> asyncio.Task[Any]:
        return self._task

    @property
    def protocol(self) -> Any:
        return self._protocol

    @property
    def transport(self) -> Any:
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
    def forwarded(self) -> Tuple[Mapping[str, str], ...]:
        elems: list[Mapping[str, str]] = []
        for field_value in self._message.headers.getall(hdrs.FORWARDED, ()):
            length: int = len(field_value)
            pos: int = 0
            need_separator: bool = False
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
        host: Optional[str] = self._message.headers.get(hdrs.HOST)
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
    def query(self) -> MultiDictProxy[str]:
        return self._rel_url.query

    @reify
    def query_string(self) -> str:
        return self._rel_url.query_string

    @reify
    def headers(self) -> CIMultiDictProxy[str]:
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
        raw: str = self.headers.get(hdrs.COOKIE, '')
        parsed: SimpleCookie = SimpleCookie(raw)
        return MappingProxyType({key: val.value for key, val in parsed.items()})

    @reify
    def http_range(self) -> slice:
        rng: Optional[str] = self._headers.get(hdrs.RANGE)
        start: Optional[int] = None
        end: Optional[int] = None
        if rng is not None:
            try:
                pattern: str = '^bytes=(\\d*)-(\\d*)$'
                start_str, end_str = re.findall(pattern, rng)[0]
            except IndexError:
                raise ValueError('range not in acceptable format')
            end = int(end_str) if end_str else None
            start = int(start_str) if start_str else None
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
                chunk: bytes = await self._payload.readany()
                body.extend(chunk)
                if self._client_max_size:
                    body_size: int = len(body)
                    if body_size > self._client_max_size:
                        raise HTTPRequestEntityTooLarge(max_size=self._client_max_size, actual_size=body_size)
                if not chunk:
                    break
            self._read_bytes = bytes(body)
        return self._read_bytes

    async def text(self) -> str:
        bytes_body: bytes = await self.read()
        encoding: str = getattr(self, 'charset', 'utf-8')
        try:
            return bytes_body.decode(encoding)
        except LookupError:
            raise HTTPUnsupportedMediaType()

    async def json(self, *, loads: JSONDecoder = DEFAULT_JSON_DECODER, content_type: str = 'application/json') -> Any:
        body: str = await self.text()
        if content_type:
            if not is_expected_content_type(self.content_type, content_type):
                raise HTTPBadRequest(text='Attempt to decode JSON with unexpected mimetype: %s' % self.content_type)
        return loads(body)

    async def multipart(self) -> MultipartReader:
        return MultipartReader(self._headers, self._payload)

    async def post(self) -> MultiDictProxy[Any]:
        if self._post is not None:
            return self._post
        if self._method not in self.POST_METHODS:
            self._post = MultiDictProxy(MultiDict())
            return self._post
        content_type: str = self.content_type
        if content_type not in ('', 'application/x-www-form-urlencoded', 'multipart/form-data'):
            self._post = MultiDictProxy(MultiDict())
            return self._post
        out: MultiDict = MultiDict()
        if content_type == 'multipart/form-data':
            multipart: MultipartReader = await self.multipart()
            max_size: int = self._client_max_size
            field: Optional[Union[BodyPartReader, Any]] = await multipart.next()
            while field is not None:
                size: int = 0
                field_ct: Optional[str] = field.headers.get(hdrs.CONTENT_TYPE)
                if isinstance(field, BodyPartReader):
                    assert field.name is not None
                    if field.filename:
                        tmp: io.BufferedRandom = await self._loop.run_in_executor(None, tempfile.TemporaryFile)
                        chunk: bytes = await field.read_chunk(size=2 ** 16)
                        while chunk:
                            chunk_decoded: bytes = field.decode(chunk)
                            await self._loop.run_in_executor(None, tmp.write, chunk_decoded)
                            size += len(chunk_decoded)
                            if 0 < max_size < size:
                                await self._loop.run_in_executor(None, tmp.close)
                                raise HTTPRequestEntityTooLarge(max_size=max_size, actual_size=size)
                            chunk = await field.read_chunk(size=2 ** 16)
                        await self._loop.run_in_executor(None, tmp.seek, 0)
                        if field_ct is None:
                            field_ct = 'application/octet-stream'
                        ff = FileField(field.name, field.filename, cast(io.BufferedReader, tmp), field_ct, field.headers)
                        out.add(field.name, ff)
                    else:
                        value: bytes = await field.read(decode=True)
                        if field_ct is None or field_ct.startswith('text/'):
                            charset: str = field.get_charset(default='utf-8')
                            out.add(field.name, value.decode(charset))
                        else:
                            out.add(field.name, value)
                        size += len(value)
                        if 0 < max_size < size:
                            raise HTTPRequestEntityTooLarge(max_size=max_size, actual_size=size)
                else:
                    raise ValueError('To decode nested multipart you need to use custom reader')
                field = await multipart.next()
        else:
            data: bytes = await self.read()
            if data:
                charset: str = getattr(self, 'charset', 'utf-8')
                bytes_query: bytes = data.rstrip()
                try:
                    query: str = bytes_query.decode(charset)
                except LookupError:
                    raise HTTPUnsupportedMediaType()
                out.extend(parse_qsl(qs=query, keep_blank_values=True, encoding=charset))
        self._post = MultiDictProxy(out)
        return self._post

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        transport: Any = self._protocol.transport
        if transport is None:
            return default
        return transport.get_extra_info(name, default)

    def __repr__(self) -> str:
        ascii_encodable_path: str = self.path.encode('ascii', 'backslashreplace').decode('ascii')
        return '<{} {} {} >'.format(self.__class__.__name__, self._method, ascii_encodable_path)

    def __eq__(self, other: Any) -> bool:
        return id(self) == id(other)

    def __bool__(self) -> bool:
        return True

    async def _prepare_hook(self, response: StreamResponse) -> None:
        return

    def _cancel(self, exc: BaseException) -> None:
        set_exception(self._payload, exc)

    def _finish(self) -> None:
        if self._post is None or self.content_type != 'multipart/form-data':
            return
        for file_name, file_field_object in self._post.items():
            if isinstance(file_field_object, FileField):
                file_field_object.file.close()


class Request(BaseRequest):
    _match_info: Optional["UrlMappingMatchInfo"] = None

    def clone(
        self,
        *,
        method: Any = sentinel,
        rel_url: Any = sentinel,
        headers: Any = sentinel,
        scheme: Any = sentinel,
        host: Any = sentinel,
        remote: Any = sentinel,
        client_max_size: Any = sentinel,
    ) -> "Request":
        ret: BaseRequest = super().clone(
            method=method,
            rel_url=rel_url,
            headers=headers,
            scheme=scheme,
            host=host,
            remote=remote,
            client_max_size=client_max_size,
        )
        new_ret: Request = cast(Request, ret)
        new_ret._match_info = self._match_info
        return new_ret

    @reify
    def match_info(self) -> "UrlMappingMatchInfo":
        match_info = self._match_info
        assert match_info is not None
        return match_info

    @property
    def app(self) -> "Application":
        match_info = self._match_info
        assert match_info is not None
        return match_info.current_app

    @property
    def config_dict(self) -> ChainMapProxy:
        match_info = self._match_info
        assert match_info is not None
        lst = match_info.apps
        app: "Application" = self.app
        idx: int = lst.index(app)
        sublist = list(reversed(lst[:idx + 1]))
        return ChainMapProxy(sublist)

    async def _prepare_hook(self, response: StreamResponse) -> None:
        match_info = self._match_info
        if match_info is None:
            return
        for app in match_info._apps:
            if (on_response_prepare := app.on_response_prepare):
                await on_response_prepare.send(self, response)