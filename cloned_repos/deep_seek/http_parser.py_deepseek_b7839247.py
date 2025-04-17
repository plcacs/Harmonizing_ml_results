```python
import abc
import asyncio
import re
import string
from contextlib import suppress
from enum import IntEnum
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    List,
    Literal,
    NamedTuple,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from multidict import CIMultiDict, CIMultiDictProxy, istr
from yarl import URL

from . import hdrs
from .base_protocol import BaseProtocol
from .compression_utils import HAS_BROTLI, BrotliDecompressor, ZLibDecompressor
from .helpers import (
    _EXC_SENTINEL,
    DEBUG,
    EMPTY_BODY_METHODS,
    EMPTY_BODY_STATUS_CODES,
    NO_EXTENSIONS,
    BaseTimerContext,
    set_exception,
)
from .http_exceptions import (
    BadHttpMessage,
    BadHttpMethod,
    BadStatusLine,
    ContentEncodingError,
    ContentLengthError,
    InvalidHeader,
    InvalidURLError,
    LineTooLong,
    TransferEncodingError,
)
from .http_writer import HttpVersion, HttpVersion10
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import RawHeaders

__all__ = (
    "HeadersParser",
    "HttpParser",
    "HttpRequestParser",
    "HttpResponseParser",
    "RawRequestMessage",
    "RawResponseMessage",
)

_SEP = Literal[b"\r\n", b"\n"]

ASCIISET: Final[Set[str]] = set(string.printable)

_TCHAR_SPECIALS: Final[str] = re.escape("!#$%&'*+-.^_`|~")
TOKENRE: Final[Pattern[str]] = re.compile(f"[0-9A-Za-z{_TCHAR_SPECIALS}]+")
VERSRE: Final[Pattern[str]] = re.compile(r"HTTP/(\d)\.(\d)", re.ASCII)
DIGITS: Final[Pattern[str]] = re.compile(r"\d+", re.ASCII)
HEXDIGITS: Final[Pattern[bytes]] = re.compile(rb"[0-9a-fA-F]+")


class RawRequestMessage(NamedTuple):
    method: str
    path: str
    version: HttpVersion
    headers: CIMultiDictProxy[str]
    raw_headers: RawHeaders
    should_close: bool
    compression: Optional[str]
    upgrade: bool
    chunked: bool
    url: URL


class RawResponseMessage(NamedTuple):
    version: HttpVersion
    code: int
    reason: str
    headers: CIMultiDictProxy[str]
    raw_headers: RawHeaders
    should_close: bool
    compression: Optional[str]
    upgrade: bool
    chunked: bool


_MsgT = TypeVar("_MsgT", RawRequestMessage, RawResponseMessage)


class ParseState(IntEnum):
    PARSE_NONE = 0
    PARSE_LENGTH = 1
    PARSE_CHUNKED = 2
    PARSE_UNTIL_EOF = 3


class ChunkState(IntEnum):
    PARSE_CHUNKED_SIZE = 0
    PARSE_CHUNKED_CHUNK = 1
    PARSE_CHUNKED_CHUNK_EOF = 2
    PARSE_MAYBE_TRAILERS = 3
    PARSE_TRAILERS = 4


class HeadersParser:
    def __init__(
        self,
        max_line_size: int = 8190,
        max_field_size: int = 8190,
        lax: bool = False,
    ) -> None:
        self.max_line_size: int = max_line_size
        self.max_field_size: int = max_field_size
        self._lax: bool = lax

    def parse_headers(
        self, lines: List[bytes]
    ) -> Tuple[CIMultiDictProxy[str], RawHeaders]:
        headers: CIMultiDict[str] = CIMultiDict()
        raw_headers: List[Tuple[bytes, bytes]] = []

        lines_idx = 1
        line = lines[1]
        line_count = len(lines)

        while line:
            try:
                bname, bvalue = line.split(b":", 1)
            except ValueError:
                raise InvalidHeader(line) from None

            if len(bname) == 0:
                raise InvalidHeader(bname)

            if {bname[0], bname[-1]} & {32, 9}:
                raise InvalidHeader(line)

            bvalue = bvalue.lstrip(b" \t")
            if len(bname) > self.max_field_size:
                raise LineTooLong(
                    f"request header name {bname.decode('utf8', 'backslashreplace')}",
                    str(self.max_field_size),
                    str(len(bname)),
                )
            name = bname.decode("utf-8", "surrogateescape")
            if not TOKENRE.fullmatch(name):
                raise InvalidHeader(bname)

            header_length = len(bvalue)

            lines_idx += 1
            line = lines[lines_idx]

            continuation = self._lax and line and line[0] in (32, 9)

            if continuation:
                bvalue_lst = [bvalue]
                while continuation:
                    header_length += len(line)
                    if header_length > self.max_field_size:
                        raise LineTooLong(
                            f"request header field {bname.decode('utf8', 'backslashreplace')}",
                            str(self.max_field_size),
                            str(header_length),
                        )
                    bvalue_lst.append(line)

                    lines_idx += 1
                    if lines_idx < line_count:
                        line = lines[lines_idx]
                        continuation = line and line[0] in (32, 9)
                    else:
                        line = b""
                        break
                bvalue = b"".join(bvalue_lst)
            else:
                if header_length > self.max_field_size:
                    raise LineTooLong(
                        f"request header field {bname.decode('utf8', 'backslashreplace')}",
                        str(self.max_field_size),
                        str(header_length),
                    )

            bvalue = bvalue.strip(b" \t")
            value = bvalue.decode("utf-8", "surrogateescape")

            if "\n" in value or "\r" in value or "\x00" in value:
                raise InvalidHeader(bvalue)

            headers.add(name, value)
            raw_headers.append((bname, bvalue))

        return (CIMultiDictProxy(headers), tuple(raw_headers))


def _is_supported_upgrade(headers: CIMultiDictProxy[str]) -> bool:
    return headers.get(hdrs.UPGRADE, "").lower() in {"tcp", "websocket"}


class HttpParser(abc.ABC, Generic[_MsgT]):
    lax: ClassVar[bool] = False

    def __init__(
        self,
        protocol: BaseProtocol,
        loop: asyncio.AbstractEventLoop,
        limit: int,
        max_line_size: int = 8190,
        max_field_size: int = 8190,
        timer: Optional[BaseTimerContext] = None,
        code: Optional[int] = None,
        method: Optional[str] = None,
        payload_exception: Optional[Type[BaseException]] = None,
        response_with_body: bool = True,
        read_until_eof: bool = False,
        auto_decompress: bool = True,
    ) -> None:
        self.protocol: BaseProtocol = protocol
        self.loop: asyncio.AbstractEventLoop = loop
        self.max_line_size: int = max_line_size
        self.max_field_size: int = max_field_size
        self.timer: Optional[BaseTimerContext] = timer
        self.code: Optional[int] = code
        self.method: Optional[str] = method
        self.payload_exception: Optional[Type[BaseException]] = payload_exception
        self.response_with_body: bool = response_with_body
        self.read_until_eof: bool = read_until_eof
        self._auto_decompress: bool = auto_decompress
        self._limit: int = limit
        self._headers_parser: HeadersParser = HeadersParser(
            max_line_size, max_field_size, self.lax
        )
        self._lines: List[bytes] = []
        self._tail: bytes = b""
        self._upgraded: bool = False
        self._payload: Optional[StreamReader] = None
        self._payload_parser: Optional[HttpPayloadParser] = None

    @abc.abstractmethod
    def parse_message(self, lines: List[bytes]) -> _MsgT: ...

    @abc.abstractmethod
    def _is_chunked_te(self, te: str) -> bool: ...

    def feed_eof(self) -> Optional[_MsgT]:
        if self._payload_parser is not None:
            self._payload_parser.feed_eof()
            self._payload_parser = None
        else:
            if self._tail:
                self._lines.append(self._tail)

            if self._lines:
                if self._lines[-1] != b"\r\n":
                    self._lines.append(b"")
                with suppress(Exception):
                    return self.parse_message(self._lines)
        return None

    def feed_data(
        self,
        data: bytes,
        SEP: _SEP = b"\r\n",
        EMPTY: bytes = b"",
        CONTENT_LENGTH: istr = hdrs.CONTENT_LENGTH,
        METH_CONNECT: str = hdrs.METH_CONNECT,
        SEC_WEBSOCKET_KEY1: istr = hdrs.SEC_WEBSOCKET_KEY1,
    ) -> Tuple[List[Tuple[_MsgT, StreamReader]], bool, bytes]:
        messages: List[Tuple[_MsgT, StreamReader]] = []
        should_close = False

        if self._tail:
            data, self._tail = self._tail + data, b""

        data_len = len(data)
        start_pos = 0
        loop = self.loop

        while start_pos < data_len:
            if self._payload_parser is None and not self._upgraded:
                pos = data.find(SEP, start_pos)
                if pos == start_pos and not self._lines:
                    start_pos = pos + len(SEP)
                    continue

                if pos >= start_pos:
                    if should_close:
                        raise BadHttpMessage("Data after `Connection: close`")

                    line = data[start_pos:pos]
                    if SEP == b"\n":
                        line = line.rstrip(b"\r")
                    self._lines.append(line)
                    start_pos = pos + len(SEP)

                    if self._lines[-1] == EMPTY:
                        try:
                            msg = self.parse_message(self._lines)
                        finally:
                            self._lines.clear()

                        def get_content_length() -> Optional[int]:
                            length_hdr = msg.headers.get(CONTENT_LENGTH)
                            if length_hdr is None:
                                return None
                            if not DIGITS.fullmatch(length_hdr):
                                raise InvalidHeader(CONTENT_LENGTH)
                            return int(length_hdr)

                        length = get_content_length()
                        if SEC_WEBSOCKET_KEY1 in msg.headers:
                            raise InvalidHeader(SEC_WEBSOCKET_KEY1)

                        self._upgraded = msg.upgrade and _is_supported_upgrade(
                            msg.headers
                        )

                        method = getattr(msg, "method", self.method)
                        code = getattr(msg, "code", 0)

                        assert self.protocol is not None
                        empty_body = code in EMPTY_BODY_STATUS_CODES or bool(
                            method and method in EMPTY_BODY_METHODS
                        )
                        if not empty_body and (
                            ((length is not None and length > 0) or msg.chunked)
                            and not self._upgraded
                        ):
                            payload = StreamReader(
                                self.protocol,
                                timer=self.timer,
                                loop=loop,
                                limit=self._limit,
                            )
                            payload_parser = HttpPayloadParser(
                                payload,
                                length=length,
                                chunked=msg.chunked,
                                method=method,
                                compression=msg.compression,
                                code=self.code,
                                response_with_body=self.response_with_body,
                                auto_decompress=self._auto_decompress,
                                lax=self.lax,
                            )
                            if not payload_parser.done:
                                self._payload_parser = payload_parser
                        elif method == METH_CONNECT:
                            assert isinstance(msg, RawRequestMessage)
                            payload = StreamReader(
                                self.protocol,
                                timer=self.timer,
                                loop=loop,
                                limit=self._limit,
                            )
                            self._upgraded = True
                            self._payload_parser = HttpPayloadParser(
                                payload,
                                method=cast(str, msg.method),
                                compression=msg.compression,
                                auto_decompress=self._auto_decompress,
                                lax=self.lax,
                            )
                        elif not empty_body and length is None and self.read_until_eof:
                            payload = StreamReader(
                                self.protocol,
                                timer=self.timer,
                                loop=loop,
                                limit=self._limit,
                            )
                            payload_parser = HttpPayloadParser(
                                payload,
                                length=length,
                                chunked=msg.chunked,
                                method=method,
                                compression=msg.compression,
                                code=self.code,
                                response_with_body=self.response_with_body,
                                auto_decompress=self._auto_decompress,
                                lax=self.lax,
                            )
                            if not payload_parser.done:
                                self._payload_parser = payload_parser
                        else:
                            payload = EMPTY_PAYLOAD

                        messages.append((msg, payload))
                        should_close = msg.should_close
                else:
                    self._tail = data[start_pos:]
                    data = b""
                    break

            elif self._payload_parser is None and self._upgraded:
                assert not self._lines
                break

            elif data and start_pos < data_len:
                assert not self._lines
                assert self._payload_parser is not None
                try:
                    eof, data = self._payload_parser.feed_data(data[start_pos:], SEP)
                except BaseException as underlying_exc:
                    reraised_exc = underlying_exc
                    if self.payload_exception is not None:
                        reraised_exc = self.payload_exception(str(underlying_exc))

                    set_exception(
                        self._payload_parser.payload,
                        reraised_exc,
                        underlying_exc,
                    )

                    eof = True
                    data = b""

                if eof:
                    start_pos = 0
                    data_len = len(data)
                    self._payload_parser = None
                    continue
            else:
                break

        if data and start_pos < data_len:
            data = data[start_pos:]
        else:
            data = b""

        return messages, self._upgraded, data

    def parse_headers(
        self, lines: List[bytes]
    ) -> Tuple[
        CIMultiDictProxy[str], RawHeaders, Optional[bool], Optional[str], bool, bool
    ]:
        headers, raw_headers = self._headers_parser.parse_headers(lines)
        close_conn: Optional[bool] = None
        encoding: Optional[str] = None
        upgrade: bool = False
        chunked: bool = False

        singletons = (
            hdrs.CONTENT_LENGTH,
            hdrs.CONTENT_LOCATION,
            hdrs.CONTENT_RANGE,
            hdrs.CONTENT_TYPE,
            hdrs.ETAG,
            hdrs.HOST,
            hdrs.MAX_FORWARDS,
            hdrs.SERVER,
            hdrs.TRANSFER_ENCODING,
            hdrs.USER_AGENT,
        )
        bad_hdr = next((h for h in singletons if len(headers.getall(h, ())) > 1), None)
        if bad_hdr is not None:
            raise BadHttpMessage(f"Duplicate '{bad_hdr}' header found.")

        conn = headers.get(hdrs.CONNECTION)
        if conn:
            v = conn.lower()
            if v == "close":
                close_conn = True
            elif v == "keep-alive":
                close_conn = False
            elif v == "upgrade" and headers.get(hdrs.UPGRADE):
                upgrade = True

        enc = headers.get(hdrs.CONTENT_ENCODING)
        if enc:
            enc = enc.lower()
            if enc in ("gzip", "deflate", "br"):
                encoding = enc

        te = headers.get(hdrs.TRANSFER_ENCODING)
        if te is not None:
            if self._is_chunked_te(te):
                chunked = True

            if hdrs.CONTENT_LENGTH in headers:
                raise BadHttpMessage(
                    "Transfer-Encoding can't be present with Content-Length",
                )

        return (headers, raw_headers, close_conn, encoding, upgrade, chunked)

    def set_upgraded(self, val: bool) -> None:
        self._upgraded = val


class HttpRequestParser(HttpParser[RawRequestMessage]):
    def parse_message(self, lines: List[bytes]) -> RawRequestMessage:
        line = lines[0].decode("utf-8", "surrogateescape")
        try:
            method, path, version = line.split(" ", maxsplit=2)
        except ValueError:
            raise BadHttpMethod(line) from None

        if len(path) > self.max_line_size:
            raise LineTooLong(
                "Status line is too long", str(self.max_line_size), str(len(path))
            )

        if not TOKENRE.fullmatch(method):
            raise BadHttpMethod(method)

        match = VERSRE.fullmatch(version)
        if match is None:
            raise BadStatusLine(line)
        version_o = HttpVersion(int(match.group(1)), int(match.group(2)))

        if method == "CONNECT":
            url = URL.build(authority=path, encoded=True)
        elif path.startswith("/"):
            path_part, _hash_separator, url_fragment = path.partition("#")
            path_part, _question_mark_separator, qs_part = path_part.partition("?")
            url = URL.build(
                path=path_part,
                query_string=qs_part,
                fragment=url_fragment,
                encoded=True,
            )
        elif path == "*" and method == "OPTIONS":
            url = URL(path, encoded=True)
        else:
            url = URL(path, encoded=True)
            if url.sche