import asyncio
import collections.abc
import datetime
import enum
import json
import math
import time
import warnings
import zlib
from concurrent.futures import Executor
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    MutableMapping,
    Optional,
    Union,
    cast,
)
from multidict import CIMultiDict, CIMultiDictProxy, istr
from . import hdrs, payload
from .abc import AbstractStreamWriter
from .compression_utils import ZLibCompressor
from .helpers import (
    ETAG_ANY,
    QUOTED_ETAG_RE,
    CookieMixin,
    ETag,
    HeadersMixin,
    must_be_empty_body,
    parse_http_date,
    populate_with_cookies,
    rfc822_formatted_time,
    sentinel,
    should_remove_content_length,
    validate_etag_value,
)
from .http import SERVER_SOFTWARE, HttpVersion10, HttpVersion11
from .payload import Payload
from .typedefs import JSONEncoder, LooseHeaders

if TYPE_CHECKING:
    from .web_request import BaseRequest
    BaseClass = MutableMapping[str, Any]
else:
    BaseClass = collections.abc.MutableMapping

REASON_PHRASES: Dict[int, str] = {http_status.value: http_status.phrase for http_status in HTTPStatus}
LARGE_BODY_SIZE: int = 1024 ** 2
__all__ = ('ContentCoding', 'StreamResponse', 'Response', 'json_response')


class ContentCoding(enum.Enum):
    deflate: str = 'deflate'
    gzip: str = 'gzip'
    identity: str = 'identity'


CONTENT_CODINGS: Dict[str, ContentCoding] = {coding.value: coding for coding in ContentCoding}


class StreamResponse(BaseClass, HeadersMixin, CookieMixin):
    _length_check: bool = True
    _body: Optional[bytes] = None
    _keep_alive: Optional[bool] = None
    _chunked: bool = False
    _compression: bool = False
    _compression_strategy: int = zlib.Z_DEFAULT_STRATEGY
    _compression_force: Optional[ContentCoding] = None
    _req: Optional["BaseRequest"] = None
    _payload_writer: Optional[AbstractStreamWriter] = None
    _eof_sent: bool = False
    _must_be_empty_body: Optional[bool] = None
    _body_length: int = 0
    _state: Dict[str, Any]
    _headers: CIMultiDict
    _content_type: Optional[str] = None
    _content_dict: Optional[Dict[str, str]] = None

    def __init__(
        self,
        *,
        status: int = 200,
        reason: Optional[str] = None,
        headers: Optional[LooseHeaders] = None,
        _real_headers: Optional[CIMultiDict] = None,
    ) -> None:
        """Initialize a new stream response object.

        _real_headers is an internal parameter used to pass a pre-populated
        headers object. It is used by the `Response` class to avoid copying
        the headers when creating a new response object. It is not intended
        to be used by external code.
        """
        self._state = {}  # type: Dict[str, Any]
        if _real_headers is not None:
            self._headers = _real_headers
        elif headers is not None:
            self._headers = CIMultiDict(headers)
        else:
            self._headers = CIMultiDict()
        self._set_status(status, reason)

    @property
    def prepared(self) -> bool:
        return self._eof_sent or self._payload_writer is not None

    @property
    def task(self) -> Optional[asyncio.Task]:
        if self._req:
            return self._req.task
        else:
            return None

    @property
    def status(self) -> int:
        return self._status

    @property
    def chunked(self) -> bool:
        return self._chunked

    @property
    def compression(self) -> bool:
        return self._compression

    @property
    def reason(self) -> str:
        return self._reason

    def set_status(self, status: int, reason: Optional[str] = None) -> None:
        assert not self.prepared, 'Cannot change the response status code after the headers have been sent'
        self._set_status(status, reason)

    def _set_status(self, status: int, reason: Optional[str]) -> None:
        self._status = status
        if reason is None:
            reason = REASON_PHRASES.get(self._status, '')
        elif '\n' in reason:
            raise ValueError('Reason cannot contain \\n')
        self._reason = reason

    @property
    def keep_alive(self) -> Optional[bool]:
        return self._keep_alive

    def force_close(self) -> None:
        self._keep_alive = False

    @property
    def body_length(self) -> int:
        return self._body_length

    def enable_chunked_encoding(self) -> None:
        """Enables automatic chunked transfer encoding."""
        if hdrs.CONTENT_LENGTH in self._headers:
            raise RuntimeError("You can't enable chunked encoding when a content length is set")
        self._chunked = True

    def enable_compression(self, force: Optional[ContentCoding] = None, strategy: int = zlib.Z_DEFAULT_STRATEGY) -> None:
        """Enables response compression encoding."""
        self._compression = True
        self._compression_force = force
        self._compression_strategy = strategy

    @property
    def headers(self) -> CIMultiDict:
        return self._headers

    @property
    def content_length(self) -> Optional[int]:
        return super().content_length

    @content_length.setter
    def content_length(self, value: Optional[int]) -> None:
        if value is not None:
            value = int(value)
            if self._chunked:
                raise RuntimeError("You can't set content length when chunked encoding is enable")
            self._headers[hdrs.CONTENT_LENGTH] = str(value)
        else:
            self._headers.pop(hdrs.CONTENT_LENGTH, None)

    @property
    def content_type(self) -> Optional[str]:
        return super().content_type

    @content_type.setter
    def content_type(self, value: str) -> None:
        _ = self.content_type
        self._content_type = str(value)
        self._generate_content_type_header()

    @property
    def charset(self) -> Optional[str]:
        return super().charset

    @charset.setter
    def charset(self, value: Optional[str]) -> None:
        ctype = self.content_type
        if ctype == 'application/octet-stream':
            raise RuntimeError("Setting charset for application/octet-stream doesn't make sense, setup content_type first")
        assert self._content_dict is not None
        if value is None:
            self._content_dict.pop('charset', None)
        else:
            self._content_dict['charset'] = str(value).lower()
        self._generate_content_type_header()

    @property
    def last_modified(self) -> Optional[datetime.datetime]:
        """The value of Last-Modified HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self._headers.get(hdrs.LAST_MODIFIED))

    @last_modified.setter
    def last_modified(self, value: Union[None, int, float, datetime.datetime, str]) -> None:
        if value is None:
            self._headers.pop(hdrs.LAST_MODIFIED, None)
        elif isinstance(value, (int, float)):
            self._headers[hdrs.LAST_MODIFIED] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(math.ceil(value)))
        elif isinstance(value, datetime.datetime):
            self._headers[hdrs.LAST_MODIFIED] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', value.utctimetuple())
        elif isinstance(value, str):
            self._headers[hdrs.LAST_MODIFIED] = value
        else:
            msg = f'Unsupported type for last_modified: {type(value).__name__}'
            raise TypeError(msg)

    @property
    def etag(self) -> Optional[ETag]:
        quoted_value: Optional[str] = self._headers.get(hdrs.ETAG)
        if not quoted_value:
            return None
        elif quoted_value == ETAG_ANY:
            return ETag(value=ETAG_ANY)
        match = QUOTED_ETAG_RE.fullmatch(quoted_value)
        if not match:
            return None
        is_weak, value = match.group(1, 2)
        return ETag(is_weak=bool(is_weak), value=value)

    @etag.setter
    def etag(self, value: Union[None, str, ETag]) -> None:
        if value is None:
            self._headers.pop(hdrs.ETAG, None)
        elif isinstance(value, str) and value == ETAG_ANY or (isinstance(value, ETag) and value.value == ETAG_ANY):
            self._headers[hdrs.ETAG] = ETAG_ANY
        elif isinstance(value, str):
            validate_etag_value(value)
            self._headers[hdrs.ETAG] = f'"{value}"'
        elif isinstance(value, ETag) and isinstance(value.value, str):
            validate_etag_value(value.value)
            hdr_value = f'W/"{value.value}"' if value.is_weak else f'"{value.value}"'
            self._headers[hdrs.ETAG] = hdr_value
        else:
            raise ValueError(f'Unsupported etag type: {type(value)}. etag must be str, ETag or None')

    def _generate_content_type_header(self, CONTENT_TYPE: str = hdrs.CONTENT_TYPE) -> None:
        assert self._content_dict is not None
        assert self._content_type is not None
        params: str = '; '.join((f'{k}={v}' for k, v in self._content_dict.items()))
        if params:
            ctype: str = self._content_type + '; ' + params
        else:
            ctype = self._content_type
        self._headers[CONTENT_TYPE] = ctype

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        if coding is ContentCoding.identity:
            return
        assert self._payload_writer is not None
        self._headers[hdrs.CONTENT_ENCODING] = coding.value
        self._payload_writer.enable_compression(coding.value, self._compression_strategy)
        self._headers.popall(hdrs.CONTENT_LENGTH, None)

    async def _start_compression(self, request: "BaseRequest") -> None:
        if self._compression_force:
            await self._do_start_compression(self._compression_force)
            return
        accept_encoding: str = request.headers.get(hdrs.ACCEPT_ENCODING, '').lower()
        for value, coding in CONTENT_CODINGS.items():
            if value in accept_encoding:
                await self._do_start_compression(coding)
                return

    async def prepare(self, request: "BaseRequest") -> AbstractStreamWriter:
        if self._eof_sent:
            return cast(AbstractStreamWriter, None)  # type: ignore
        if self._payload_writer is not None:
            return self._payload_writer
        self._must_be_empty_body = must_be_empty_body(request.method, self.status)
        return await self._start(request)

    async def _start(self, request: "BaseRequest") -> AbstractStreamWriter:
        self._req = request
        writer: AbstractStreamWriter = self._payload_writer = request._payload_writer  # type: ignore
        await self._prepare_headers()
        await request._prepare_hook(self)  # type: ignore
        await self._write_headers()
        return writer

    async def _prepare_headers(self) -> None:
        request: "BaseRequest" = self._req  # type: ignore
        assert request is not None
        writer: AbstractStreamWriter = self._payload_writer  # type: ignore
        assert writer is not None
        keep_alive: Optional[bool] = self._keep_alive
        if keep_alive is None:
            keep_alive = request.keep_alive  # type: ignore
        self._keep_alive = keep_alive
        version = request.version  # type: ignore
        headers = self._headers
        if self._cookies:
            populate_with_cookies(headers, self._cookies)
        if self._compression:
            await self._start_compression(request)
        if self._chunked:
            if version != HttpVersion11:
                raise RuntimeError('Using chunked encoding is forbidden for HTTP/{0.major}.{0.minor}'.format(request.version))
            if not self._must_be_empty_body:
                writer.enable_chunking()
                headers[hdrs.TRANSFER_ENCODING] = 'chunked'
        elif self._length_check:
            writer.length = self.content_length  # type: ignore
            if writer.length is None:
                if version >= HttpVersion11:
                    if not self._must_be_empty_body:
                        writer.enable_chunking()
                        headers[hdrs.TRANSFER_ENCODING] = 'chunked'
                elif not self._must_be_empty_body:
                    keep_alive = False
        if self._must_be_empty_body:
            if hdrs.CONTENT_LENGTH in headers and should_remove_content_length(request.method, self.status):
                del headers[hdrs.CONTENT_LENGTH]
            if hdrs.TRANSFER_ENCODING in headers:
                del headers[hdrs.TRANSFER_ENCODING]
        elif (writer.length if self._length_check else self.content_length) != 0:
            headers.setdefault(hdrs.CONTENT_TYPE, 'application/octet-stream')
        headers.setdefault(hdrs.DATE, rfc822_formatted_time())
        headers.setdefault(hdrs.SERVER, SERVER_SOFTWARE)
        if hdrs.CONNECTION not in headers:
            if keep_alive:
                if version == HttpVersion10:
                    headers[hdrs.CONNECTION] = 'keep-alive'
            elif version == HttpVersion11:
                headers[hdrs.CONNECTION] = 'close'

    async def _write_headers(self) -> None:
        request: "BaseRequest" = self._req  # type: ignore
        assert request is not None
        writer: AbstractStreamWriter = self._payload_writer  # type: ignore
        assert writer is not None
        version = request.version  # type: ignore
        status_line: str = f'HTTP/{version[0]}.{version[1]} {self._status} {self._reason}'
        await writer.write_headers(status_line, self._headers)

    async def write(self, data: Union[bytes, bytearray, memoryview]) -> None:
        assert isinstance(data, (bytes, bytearray, memoryview)), 'data argument must be byte-ish (%r)' % type(data)
        if self._eof_sent:
            raise RuntimeError('Cannot call write() after write_eof()')
        if self._payload_writer is None:
            raise RuntimeError('Cannot call write() before prepare()')
        await self._payload_writer.write(data)

    async def drain(self) -> None:
        assert not self._eof_sent, 'EOF has already been sent'
        assert self._payload_writer is not None, 'Response has not been started'
        warnings.warn('drain method is deprecated, use await resp.write()', DeprecationWarning, stacklevel=2)
        await self._payload_writer.drain()

    async def write_eof(self, data: Union[bytes, bytearray, memoryview] = b'') -> None:
        assert isinstance(data, (bytes, bytearray, memoryview)), 'data argument must be byte-ish (%r)' % type(data)
        if self._eof_sent:
            return
        assert self._payload_writer is not None, 'Response has not been started'
        await self._payload_writer.write_eof(data)
        self._eof_sent = True
        self._req = None
        self._body_length = self._payload_writer.output_size  # type: ignore
        self._payload_writer = None

    def __repr__(self) -> str:
        if self._eof_sent:
            info: str = 'eof'
        elif self.prepared:
            assert self._req is not None
            info = f'{self._req.method} {self._req.path} '
        else:
            info = 'not prepared'
        return f'<{self.__class__.__name__} {self.reason} {info}>'

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

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __bool__(self) -> bool:
        return True


class Response(StreamResponse):
    _compressed_body: Optional[bytes] = None
    _zlib_executor_size: Optional[int]
    _zlib_executor: Optional[Executor]

    def __init__(
        self,
        *,
        body: Any = None,
        status: int = 200,
        reason: Optional[str] = None,
        text: Optional[str] = None,
        headers: Optional[LooseHeaders] = None,
        content_type: Optional[str] = None,
        charset: Optional[str] = None,
        zlib_executor_size: Optional[int] = None,
        zlib_executor: Optional[Executor] = None,
    ) -> None:
        if body is not None and text is not None:
            raise ValueError('body and text are not allowed together')
        if headers is None:
            real_headers: CIMultiDict = CIMultiDict()
        elif not isinstance(headers, CIMultiDict):
            real_headers = CIMultiDict(headers)
        else:
            real_headers = headers
        if content_type is not None and 'charset' in content_type:
            raise ValueError('charset must not be in content_type argument')
        if text is not None:
            if hdrs.CONTENT_TYPE in real_headers:
                if content_type or charset:
                    raise ValueError('passing both Content-Type header and content_type or charset params is forbidden')
            else:
                if not isinstance(text, str):
                    raise TypeError('text argument must be str (%r)' % type(text))
                if content_type is None:
                    content_type = 'text/plain'
                if charset is None:
                    charset = 'utf-8'
                real_headers[hdrs.CONTENT_TYPE] = content_type + '; charset=' + charset
                body = text.encode(charset)
                text = None
        elif hdrs.CONTENT_TYPE in real_headers:
            if content_type is not None or charset is not None:
                raise ValueError('passing both Content-Type header and content_type or charset params is forbidden')
        elif content_type is not None:
            if charset is not None:
                content_type += '; charset=' + charset
            real_headers[hdrs.CONTENT_TYPE] = content_type
        super().__init__(status=status, reason=reason, _real_headers=real_headers)
        if text is not None:
            self.text = text
        else:
            self.body = body
        self._zlib_executor_size = zlib_executor_size
        self._zlib_executor = zlib_executor

    @property
    def body(self) -> Optional[bytes]:
        return self._body

    @body.setter
    def body(self, body: Any) -> None:
        if body is None:
            self._body = None
        elif isinstance(body, (bytes, bytearray)):
            self._body = body
        else:
            try:
                self._body = body = payload.PAYLOAD_REGISTRY.get(body)
            except payload.LookupError:
                raise ValueError('Unsupported body type %r' % type(body))
            headers = self._headers
            if hdrs.CONTENT_TYPE not in headers:
                headers[hdrs.CONTENT_TYPE] = body.content_type  # type: ignore
            if body.headers:
                for key, value in body.headers.items():
                    if key not in headers:
                        headers[key] = value
        self._compressed_body = None

    @property
    def text(self) -> Optional[str]:
        if self._body is None:
            return None
        return self._body.decode(self.charset or 'utf-8')  # type: ignore

    @text.setter
    def text(self, text: str) -> None:
        assert isinstance(text, str), 'text argument must be str (%r)' % type(text)
        if self.content_type == 'application/octet-stream':
            self.content_type = 'text/plain'
        if self.charset is None:
            self.charset = 'utf-8'
        self._body = text.encode(self.charset or 'utf-8')  # type: ignore
        self._compressed_body = None

    @property
    def content_length(self) -> Optional[int]:
        if self._chunked:
            return None
        if hdrs.CONTENT_LENGTH in self._headers:
            return int(self._headers[hdrs.CONTENT_LENGTH])
        if self._compressed_body is not None:
            return len(self._compressed_body)
        elif isinstance(self._body, Payload):
            return None
        elif self._body is not None:
            return len(self._body)
        else:
            return 0

    @content_length.setter
    def content_length(self, value: Any) -> None:
        raise RuntimeError('Content length is set automatically')

    async def write_eof(self, data: Union[bytes, bytearray, memoryview] = b'') -> None:
        if self._eof_sent:
            return
        if self._compressed_body is None:
            body = self._body
        else:
            body = self._compressed_body
        assert not data, f'data arg is not supported, got {data!r}'
        assert self._req is not None
        assert self._payload_writer is not None
        if body is None or self._must_be_empty_body:
            await super().write_eof()
        elif isinstance(self._body, Payload):
            await self._body.write(self._payload_writer)
            await super().write_eof()
        else:
            await super().write_eof(cast(bytes, body))

    async def _start(self, request: "BaseRequest") -> AbstractStreamWriter:
        if hdrs.CONTENT_LENGTH in self._headers:
            if should_remove_content_length(request.method, self.status):
                del self._headers[hdrs.CONTENT_LENGTH]
        elif not self._chunked:
            if isinstance(self._body, Payload):
                if self._body.size is not None:
                    self._headers[hdrs.CONTENT_LENGTH] = str(self._body.size)
            else:
                body_len: Union[int, str] = len(self._body) if self._body else '0'
                if body_len != '0' or (self.status != 304 and request.method not in hdrs.METH_HEAD_ALL):
                    self._headers[hdrs.CONTENT_LENGTH] = str(body_len)
        return await super()._start(request)

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        if self._chunked or isinstance(self._body, Payload):
            return await super()._do_start_compression(coding)
        if coding is ContentCoding.identity:
            return
        compressor: ZLibCompressor = ZLibCompressor(
            encoding=coding.value,
            max_sync_chunk_size=self._zlib_executor_size,
            executor=self._zlib_executor,
        )
        assert self._body is not None
        if self._zlib_executor_size is None and len(self._body) > LARGE_BODY_SIZE:
            warnings.warn(
                f'Synchronous compression of large response bodies ({len(self._body)} bytes) might block the async event loop. Consider providing a custom value to zlib_executor_size/zlib_executor response properties or disabling compression on it.'
            )
        self._compressed_body = await compressor.compress(self._body) + compressor.flush()
        self._headers[hdrs.CONTENT_ENCODING] = coding.value
        self._headers[hdrs.CONTENT_LENGTH] = str(len(self._compressed_body))


def json_response(
    data: Any = sentinel,
    *,
    text: Optional[str] = None,
    body: Any = None,
    status: int = 200,
    reason: Optional[str] = None,
    headers: Optional[LooseHeaders] = None,
    content_type: str = 'application/json',
    dumps: JSONEncoder = json.dumps,
) -> Response:
    if data is not sentinel:
        if text or body:
            raise ValueError('only one of data, text, or body should be specified')
        else:
            text = dumps(data)
    return Response(text=text, body=body, status=status, reason=reason, headers=headers, content_type=content_type)