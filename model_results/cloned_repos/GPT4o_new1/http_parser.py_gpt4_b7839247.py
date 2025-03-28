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
        self, max_line_size: int = 8190, max_field_size: int = 8190, lax: bool = False
    ) -> None:
        self.max_line_size = max_line_size
        self.max_field_size = max_field_size
        self._lax = lax

    def parse_headers(
        self, lines: List[bytes]
    ) -> Tuple["CIMultiDictProxy[str]", RawHeaders]:
        headers: CIMultiDict[str] = CIMultiDict()
        raw_headers: List[Tuple[bytes, bytes]] = []

        lines_idx: int = 1
        line: bytes = lines[1]
        line_count: int = len(lines)

        while line:
            try:
                bname, bvalue = line.split(b":", 1)
            except ValueError:
                raise InvalidHeader(line) from None

            if len(bname) == 0:
                raise InvalidHeader(bname)

            if {bname[0], bname[-1]} & {32, 9}:  # {" ", "\t"}
                raise InvalidHeader(line)

            bvalue = bvalue.lstrip(b" \t")
            if len(bname) > self.max_field_size:
                raise LineTooLong(
                    "request header name {}".format(
                        bname.decode("utf8", "backslashreplace")
                    ),
                    str(self.max_field_size),
                    str(len(bname)),
                )
            name: str = bname.decode("utf-8", "surrogateescape")
            if not TOKENRE.fullmatch(name):
                raise InvalidHeader(bname)

            header_length: int = len(bvalue)

            lines_idx += 1
            line = lines[lines_idx]

            continuation: bool = self._lax and line and line[0] in (32, 9)  # (' ', '\t')

            if continuation:
                bvalue_lst: List[bytes] = [bvalue]
                while continuation:
                    header_length += len(line)
                    if header_length > self.max_field_size:
                        raise LineTooLong(
                            "request header field {}".format(
                                bname.decode("utf8", "backslashreplace")
                            ),
                            str(self.max_field_size),
                            str(header_length),
                        )
                    bvalue_lst.append(line)

                    lines_idx += 1
                    if lines_idx < line_count:
                        line = lines[lines_idx]
                        if line:
                            continuation = line[0] in (32, 9)  # (' ', '\t')
                    else:
                        line = b""
                        break
                bvalue = b"".join(bvalue_lst)
            else:
                if header_length > self.max_field_size:
                    raise LineTooLong(
                        "request header field {}".format(
                            bname.decode("utf8", "backslashreplace")
                        ),
                        str(self.max_field_size),
                        str(header_length),
                    )

            bvalue = bvalue.strip(b" \t")
            value: str = bvalue.decode("utf-8", "surrogateescape")

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
        self.protocol = protocol
        self.loop = loop
        self.max_line_size = max_line_size
        self.max_field_size = max_field_size
        self.timer = timer
        self.code = code
        self.method = method
        self.payload_exception = payload_exception
        self.response_with_body = response_with_body
        self.read_until_eof = read_until_eof

        self._lines: List[bytes] = []
        self._tail: bytes = b""
        self._upgraded: bool = False
        self._payload: Optional[StreamReader] = None
        self._payload_parser: Optional[HttpPayloadParser] = None
        self._auto_decompress = auto_decompress
        self._limit = limit
        self._headers_parser = HeadersParser(max_line_size, max_field_size, self.lax)

    @abc.abstractmethod
    def parse_message(self, lines: List[bytes]) -> _MsgT:
        ...

    @abc.abstractmethod
    def _is_chunked_te(self, te: str) -> bool:
        ...

    def feed_eof(self) -> Optional[_MsgT]:
        if self._payload_parser is not None:
            self._payload_parser.feed_eof()
            self._payload_parser = None
        else:
            if self._tail:
                self._lines.append(self._tail)

            if self._lines:
                if self._lines[-1] != "\r\n":
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

        if self._tail:
            data, self._tail = self._tail + data, b""

        data_len: int = len(data)
        start_pos: int = 0
        loop = self.loop

        should_close: bool = False
        while start_pos < data_len:
            if self._payload_parser is None and not self._upgraded:
                pos: int = data.find(SEP, start_pos)
                if pos == start_pos and not self._lines:
                    start_pos = pos + len(SEP)
                    continue

                if pos >= start_pos:
                    if should_close:
                        raise BadHttpMessage("Data after `Connection: close`")

                    line: bytes = data[start_pos:pos]
                    if SEP == b"\n":
                        line = line.rstrip(b"\r")
                    self._lines.append(line)
                    start_pos = pos + len(SEP)

                    if self._lines[-1] == EMPTY:
                        try:
                            msg: _MsgT = self.parse_message(self._lines)
                        finally:
                            self._lines.clear()

                        def get_content_length() -> Optional[int]:
                            length_hdr: Optional[str] = msg.headers.get(CONTENT_LENGTH)
                            if length_hdr is None:
                                return None

                            if not DIGITS.fullmatch(length_hdr):
                                raise InvalidHeader(CONTENT_LENGTH)

                            return int(length_hdr)

                        length: Optional[int] = get_content_length()
                        if SEC_WEBSOCKET_KEY1 in msg.headers:
                            raise InvalidHeader(SEC_WEBSOCKET_KEY1)

                        self._upgraded = msg.upgrade and _is_supported_upgrade(
                            msg.headers
                        )

                        method: Optional[str] = getattr(msg, "method", self.method)
                        code: int = getattr(msg, "code", 0)

                        assert self.protocol is not None
                        empty_body: bool = code in EMPTY_BODY_STATUS_CODES or bool(
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
                                method=msg.method,
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
                    data = EMPTY
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
            data = EMPTY

        return messages, self._upgraded, data

    def parse_headers(
        self, lines: List[bytes]
    ) -> Tuple[
        "CIMultiDictProxy[str]", RawHeaders, Optional[bool], Optional[str], bool, bool
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
        bad_hdr: Optional[str] = next(
            (h for h in singletons if len(headers.getall(h, ())) > 1), None
        )
        if bad_hdr is not None:
            raise BadHttpMessage(f"Duplicate '{bad_hdr}' header found.")

        conn: Optional[str] = headers.get(hdrs.CONNECTION)
        if conn:
            v: str = conn.lower()
            if v == "close":
                close_conn = True
            elif v == "keep-alive":
                close_conn = False
            elif v == "upgrade" and headers.get(hdrs.UPGRADE):
                upgrade = True

        enc: Optional[str] = headers.get(hdrs.CONTENT_ENCODING)
        if enc:
            enc = enc.lower()
            if enc in ("gzip", "deflate", "br"):
                encoding = enc

        te: Optional[str] = headers.get(hdrs.TRANSFER_ENCODING)
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
        line: str = lines[0].decode("utf-8", "surrogateescape")
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

        match: Optional[re.Match[str]] = VERSRE.fullmatch(version)
        if match is None:
            raise BadStatusLine(line)
        version_o: HttpVersion = HttpVersion(int(match.group(1)), int(match.group(2)))

        if method == "CONNECT":
            url: URL = URL.build(authority=path, encoded=True)
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
            if url.scheme == "":
                raise InvalidURLError(
                    path.encode(errors="surrogateescape").decode("latin1")
                )

        (
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        ) = self.parse_headers(lines)

        if close is None:
            if version_o <= HttpVersion10:
                close = True
            else:
                close = False

        return RawRequestMessage(
            method,
            path,
            version_o,
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
            url,
        )

    def _is_chunked_te(self, te: str) -> bool:
        if te.rsplit(",", maxsplit=1)[-1].strip(" \t").lower() == "chunked":
            return True
        raise BadHttpMessage("Request has invalid `Transfer-Encoding`")


class HttpResponseParser(HttpParser[RawResponseMessage]):
    lax = not DEBUG

    def feed_data(
        self,
        data: bytes,
        SEP: Optional[_SEP] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Tuple[RawResponseMessage, StreamReader]], bool, bytes]:
        if SEP is None:
            SEP = b"\r\n" if DEBUG else b"\n"
        return super().feed_data(data, SEP, *args, **kwargs)

    def parse_message(self, lines: List[bytes]) -> RawResponseMessage:
        line: str = lines[0].decode("utf-8", "surrogateescape")
        try:
            version, status = line.split(maxsplit=1)
        except ValueError:
            raise BadStatusLine(line) from None

        try:
            status, reason = status.split(maxsplit=1)
        except ValueError:
            status = status.strip()
            reason = ""

        if len(reason) > self.max_line_size:
            raise LineTooLong(
                "Status line is too long", str(self.max_line_size), str(len(reason))
            )

        match: Optional[re.Match[str]] = VERSRE.fullmatch(version)
        if match is None:
            raise BadStatusLine(line)
        version_o: HttpVersion = HttpVersion(int(match.group(1)), int(match.group(2)))

        if len(status) != 3 or not DIGITS.fullmatch(status):
            raise BadStatusLine(line)
        status_i: int = int(status)

        (
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        ) = self.parse_headers(lines)

        if close is None:
            if version_o <= HttpVersion10:
                close = True
            elif 100 <= status_i < 200 or status_i in {204, 304}:
                close = False
            elif hdrs.CONTENT_LENGTH in headers or hdrs.TRANSFER_ENCODING in headers:
                close = False
            else:
                close = True

        return RawResponseMessage(
            version_o,
            status_i,
            reason.strip(),
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        )

    def _is_chunked_te(self, te: str) -> bool:
        return te.rsplit(",", maxsplit=1)[-1].strip(" \t").lower() == "chunked"


class HttpPayloadParser:
    def __init__(
        self,
        payload: StreamReader,
        length: Optional[int] = None,
        chunked: bool = False,
        compression: Optional[str] = None,
        code: Optional[int] = None,
        method: Optional[str] = None,
        response_with_body: bool = True,
        auto_decompress: bool = True,
        lax: bool = False,
    ) -> None:
        self._length: int = 0
        self._type: ParseState = ParseState.PARSE_UNTIL_EOF
        self._chunk: ChunkState = ChunkState.PARSE_CHUNKED_SIZE
        self._chunk_size: int = 0
        self._chunk_tail: bytes = b""
        self._auto_decompress = auto_decompress
        self._lax = lax
        self.done: bool = False

        if response_with_body and compression and self._auto_decompress:
            real_payload: Union[StreamReader, DeflateBuffer] = DeflateBuffer(
                payload, compression
            )
        else:
            real_payload = payload

        if not response_with_body:
            self._type = ParseState.PARSE_NONE
            real_payload.feed_eof()
            self.done = True
        elif chunked:
            self._type = ParseState.PARSE_CHUNKED
        elif length is not None:
            self._type = ParseState.PARSE_LENGTH
            self._length = length
            if self._length == 0:
                real_payload.feed_eof()
                self.done = True

        self.payload = real_payload

    def feed_eof(self) -> None:
        if self._type == ParseState.PARSE_UNTIL_EOF:
            self.payload.feed_eof()
        elif self._type == ParseState.PARSE_LENGTH:
            raise ContentLengthError(
                "Not enough data for satisfy content length header."
            )
        elif self._type == ParseState.PARSE_CHUNKED:
            raise TransferEncodingError(
                "Not enough data for satisfy transfer length header."
            )

    def feed_data(
        self, chunk: bytes, SEP: _SEP = b"\r\n", CHUNK_EXT: bytes = b";"
    ) -> Tuple[bool, bytes]:
        if self._type == ParseState.PARSE_LENGTH:
            required: int = self._length
            self._length = max(required - len(chunk), 0)
            self.payload.feed_data(chunk[:required])
            if self._length == 0:
                self.payload.feed_eof()
                return True, chunk[required:]

        elif self._type == ParseState.PARSE_CHUNKED:
            if self._chunk_tail:
                chunk = self._chunk_tail + chunk
                self._chunk_tail = b""

            while chunk:
                if self._chunk == ChunkState.PARSE_CHUNKED_SIZE:
                    pos: int = chunk.find(SEP)
                    if pos >= 0:
                        i: int = chunk.find(CHUNK_EXT, 0, pos)
                        if i >= 0:
                            size_b: bytes = chunk[:i]
                            if b"\n" in (ext := chunk[i:pos]):
                                exc = BadHttpMessage(
                                    f"Unexpected LF in chunk-extension: {ext!r}"
                                )
                                set_exception(self.payload, exc)
                                raise exc
                        else:
                            size_b = chunk[:pos]

                        if self._lax:
                            size_b = size_b.strip()

                        if not re.fullmatch(HEXDIGITS, size_b):
                            exc = TransferEncodingError(
                                chunk[:pos].decode("ascii", "surrogateescape")
                            )
                            set_exception(self.payload, exc)
                            raise exc
                        size: int = int(bytes(size_b), 16)

                        chunk = chunk[pos + len(SEP) :]
                        if size == 0:
                            self._chunk = ChunkState.PARSE_MAYBE_TRAILERS
                            if self._lax and chunk.startswith(b"\r"):
                                chunk = chunk[1:]
                        else:
                            self._chunk = ChunkState.PARSE_CHUNKED_CHUNK
                            self._chunk_size = size
                            self.payload.begin_http_chunk_receiving()
                    else:
                        self._chunk_tail = chunk
                        return False, b""

                if self._chunk == ChunkState.PARSE_CHUNKED_CHUNK:
                    required: int = self._chunk_size
                    self._chunk_size = max(required - len(chunk), 0)
                    self.payload.feed_data(chunk[:required])

                    if self._chunk_size:
                        return False, b""
                    chunk = chunk[required:]
                    self._chunk = ChunkState.PARSE_CHUNKED_CHUNK_EOF
                    self.payload.end_http_chunk_receiving()

                if self._chunk == ChunkState.PARSE_CHUNKED_CHUNK_EOF:
                    if self._lax and chunk.startswith(b"\r"):
                        chunk = chunk[1:]
                    if chunk[: len(SEP)] == SEP:
                        chunk = chunk[len(SEP) :]
                        self._chunk = ChunkState.PARSE_CHUNKED_SIZE
                    else:
                        self._chunk_tail = chunk
                        return False, b""

                if self._chunk == ChunkState.PARSE_MAYBE_TRAILERS:
                    head: bytes = chunk[: len(SEP)]
                    if head == SEP:
                        self.payload.feed_eof()
                        return True, chunk[len(SEP) :]
                    if not head:
                        return False, b""
                    if head == SEP[:1]:
                        self._chunk_tail = head
                        return False, b""
                    self._chunk = ChunkState.PARSE_TRAILERS

                if self._chunk == ChunkState.PARSE_TRAILERS:
                    pos: int = chunk.find(SEP)
                    if pos >= 0:
                        chunk = chunk[pos + len(SEP) :]
                        self._chunk = ChunkState.PARSE_MAYBE_TRAILERS
                    else:
                        self._chunk_tail = chunk
                        return False, b""

        elif self._type == ParseState.PARSE_UNTIL_EOF:
            self.payload.feed_data(chunk)

        return False, b""


class DeflateBuffer:
    def __init__(self, out: StreamReader, encoding: Optional[str]) -> None:
        self.out = out
        self.size: int = 0
        self.encoding = encoding
        self._started_decoding: bool = False

        self.decompressor: Union[BrotliDecompressor, ZLibDecompressor]
        if encoding == "br":
            if not HAS_BROTLI:
                raise ContentEncodingError(
                    "Can not decode content-encoding: brotli (br). "
                    "Please install `Brotli`"
                )
            self.decompressor = BrotliDecompressor()
        else:
            self.decompressor = ZLibDecompressor(encoding=encoding)

    def set_exception(
        self,
        exc: Union[Type[BaseException], BaseException],
        exc_cause: BaseException = _EXC_SENTINEL,
    ) -> None:
        set_exception(self.out, exc, exc_cause)

    def feed_data(self, chunk: bytes) -> None:
        if not chunk:
            return

        self.size += len(chunk)

        if (
            not self._started_decoding
            and self.encoding == "deflate"
            and chunk[0] & 0xF != 8
        ):
            self.decompressor = ZLibDecompressor(
                encoding=self.encoding, suppress_deflate_header=True
            )

        try:
            chunk = self.decompressor.decompress_sync(chunk)
        except Exception:
            raise ContentEncodingError(
                "Can not decode content-encoding: %s" % self.encoding
            )

        self._started_decoding = True

        if chunk:
            self.out.feed_data(chunk)

    def feed_eof(self) -> None:
        chunk: bytes = self.decompressor.flush()

        if chunk or self.size > 0:
            self.out.feed_data(chunk)
            if self.encoding == "deflate" and not self.decompressor.eof:  # type: ignore[union-attr]
                raise ContentEncodingError("deflate")

        self.out.feed_eof()

    def begin_http_chunk_receiving(self) -> None:
        self.out.begin_http_chunk_receiving()

    def end_http_chunk_receiving(self) -> None:
        self.out.end_http_chunk_receiving()


HttpRequestParserPy = HttpRequestParser
HttpResponseParserPy = HttpResponseParser
RawRequestMessagePy = RawRequestMessage
RawResponseMessagePy = RawResponseMessage

with suppress(ImportError):
    if not NO_EXTENSIONS:
        from ._http_parser import (  # type: ignore[import-not-found,no-redef]
            HttpRequestParser,
            HttpResponseParser,
            RawRequestMessage,
            RawResponseMessage,
        )

        HttpRequestParserC = HttpRequestParser
        HttpResponseParserC = HttpResponseParser
        RawRequestMessageC = RawRequestMessage
        RawResponseMessageC = RawResponseMessage
