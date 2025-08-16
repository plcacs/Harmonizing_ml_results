import base64
import binascii
import json
import re
import sys
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    Callable,
    Awaitable,
    AsyncIterator,
)
from urllib.parse import parse_qsl, unquote, urlencode

from multidict import CIMultiDict, CIMultiDictProxy

from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import (
    CONTENT_DISPOSITION,
    CONTENT_ENCODING,
    CONTENT_LENGTH,
    CONTENT_TRANSFER_ENCODING,
    CONTENT_TYPE,
)
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import (
    JsonPayload,
    LookupError,
    Order,
    Payload,
    StringPayload,
    get_payload,
    payload_type,
)
from .streams import StreamReader

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = (
    "MultipartReader",
    "MultipartWriter",
    "BodyPartReader",
    "BadContentDispositionHeader",
    "BadContentDispositionParam",
    "parse_content_disposition",
    "content_disposition_filename",
)


if TYPE_CHECKING:
    from .client_reqrep import ClientResponse


class BadContentDispositionHeader(RuntimeWarning):
    pass


class BadContentDispositionParam(RuntimeWarning):
    pass


def parse_content_disposition(
    header: Optional[str],
) -> Tuple[Optional[str], Dict[str, str]]:
    def is_token(string: str) -> bool:
        return bool(string) and TOKEN >= set(string)

    def is_quoted(string: str) -> bool:
        return string[0] == string[-1] == '"'

    def is_rfc5987(string: str) -> bool:
        return is_token(string) and string.count("'") == 2

    def is_extended_param(string: str) -> bool:
        return string.endswith("*")

    def is_continuous_param(string: str) -> bool:
        pos = string.find("*") + 1
        if not pos:
            return False
        substring = string[pos:-1] if string.endswith("*") else string[pos:]
        return substring.isdigit()

    def unescape(text: str, *, chars: str = "".join(map(re.escape, CHAR))) -> str:
        return re.sub(f"\\\\([{chars}])", "\\1", text)

    if not header:
        return None, {}

    disptype, *parts = header.split(";")
    if not is_token(disptype):
        warnings.warn(BadContentDispositionHeader(header))
        return None, {}

    params: Dict[str, str] = {}
    while parts:
        item = parts.pop(0)

        if "=" not in item:
            warnings.warn(BadContentDispositionHeader(header))
            return None, {}

        key, value = item.split("=", 1)
        key = key.lower().strip()
        value = value.lstrip()

        if key in params:
            warnings.warn(BadContentDispositionHeader(header))
            return None, {}

        if not is_token(key):
            warnings.warn(BadContentDispositionParam(item))
            continue

        elif is_continuous_param(key):
            if is_quoted(value):
                value = unescape(value[1:-1])
            elif not is_token(value):
                warnings.warn(BadContentDispositionParam(item))
                continue

        elif is_extended_param(key):
            if is_rfc5987(value):
                encoding, _, value = value.split("'", 2)
                encoding = encoding or "utf-8"
            else:
                warnings.warn(BadContentDispositionParam(item))
                continue

            try:
                value = unquote(value, encoding, "strict")
            except UnicodeDecodeError:  # pragma: nocover
                warnings.warn(BadContentDispositionParam(item))
                continue

        else:
            failed = True
            if is_quoted(value):
                failed = False
                value = unescape(value[1:-1].lstrip("\\/"))
            elif is_token(value):
                failed = False
            elif parts:
                _value = f"{value};{parts[0]}"
                if is_quoted(_value):
                    parts.pop(0)
                    value = unescape(_value[1:-1].lstrip("\\/"))
                    failed = False

            if failed:
                warnings.warn(BadContentDispositionHeader(header))
                return None, {}

        params[key] = value

    return disptype.lower(), params


def content_disposition_filename(
    params: Mapping[str, str], name: str = "filename"
) -> Optional[str]:
    name_suf = "%s*" % name
    if not params:
        return None
    elif name_suf in params:
        return params[name_suf]
    elif name in params:
        return params[name]
    else:
        parts = []
        fnparams = sorted(
            (key, value) for key, value in params.items() if key.startswith(name_suf)
        )
        for num, (key, value) in enumerate(fnparams):
            _, tail = key.split("*", 1)
            if tail.endswith("*"):
                tail = tail[:-1]
            if tail == str(num):
                parts.append(value)
            else:
                break
        if not parts:
            return None
        value = "".join(parts)
        if "'" in value:
            encoding, _, value = value.split("'", 2)
            encoding = encoding or "utf-8"
            return unquote(value, encoding, "strict")
        return value


class MultipartResponseWrapper:
    def __init__(
        self,
        resp: "ClientResponse",
        stream: "MultipartReader",
    ) -> None:
        self.resp = resp
        self.stream = stream

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Union["MultipartReader", "BodyPartReader"]:
        part = await self.next()
        if part is None:
            raise StopAsyncIteration
        return part

    def at_eof(self) -> bool:
        return self.resp.content.at_eof()

    async def next(self) -> Optional[Union["MultipartReader", "BodyPartReader"]]:
        item = await self.stream.next()
        if self.stream.at_eof():
            await self.release()
        return item

    async def release(self) -> None:
        self.resp.release()


class BodyPartReader:
    chunk_size: int = 8192

    def __init__(
        self,
        boundary: bytes,
        headers: "CIMultiDictProxy[str]",
        content: StreamReader,
        *,
        subtype: str = "mixed",
        default_charset: Optional[str] = None,
    ) -> None:
        self.headers = headers
        self._boundary = boundary
        self._boundary_len = len(boundary) + 2
        self._content = content
        self._default_charset = default_charset
        self._at_eof = False
        self._is_form_data = subtype == "form-data"
        length = None if self._is_form_data else self.headers.get(CONTENT_LENGTH, None)
        self._length = int(length) if length is not None else None
        self._read_bytes = 0
        self._unread: Deque[bytes] = deque()
        self._prev_chunk: Optional[bytes] = None
        self._content_eof = 0
        self._cache: Dict[str, Any] = {}

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        part = await self.next()
        if part is None:
            raise StopAsyncIteration
        return part

    async def next(self) -> Optional[bytes]:
        item = await self.read()
        if not item:
            return None
        return item

    async def read(self, *, decode: bool = False) -> bytes:
        if self._at_eof:
            return b""
        data = bytearray()
        while not self._at_eof:
            data.extend(await self.read_chunk(self.chunk_size))
        if decode:
            return self.decode(data)
        return data

    async def read_chunk(self, size: int = chunk_size) -> bytes:
        if self._at_eof:
            return b""
        if self._length:
            chunk = await self._read_chunk_from_length(size)
        else:
            chunk = await self._read_chunk_from_stream(size)

        encoding = self.headers.get(CONTENT_TRANSFER_ENCODING)
        if encoding and encoding.lower() == "base64":
            stripped_chunk = b"".join(chunk.split())
            remainder = len(stripped_chunk) % 4

            while remainder != 0 and not self.at_eof():
                over_chunk_size = 4 - remainder
                over_chunk = b""

                if self._prev_chunk:
                    over_chunk = self._prev_chunk[:over_chunk_size]
                    self._prev_chunk = self._prev_chunk[len(over_chunk):]

                if len(over_chunk) != over_chunk_size:
                    over_chunk += await self._content.read(4 - len(over_chunk))

                if not over_chunk:
                    self._at_eof = True

                stripped_chunk += b"".join(over_chunk.split())
                chunk += over_chunk
                remainder = len(stripped_chunk) % 4

        self._read_bytes += len(chunk)
        if self._read_bytes == self._length:
            self._at_eof = True
        if self._at_eof:
            clrf = await self._content.readline()
            assert b"\r\n" == clrf
        return chunk

    async def _read_chunk_from_length(self, size: int) -> bytes:
        assert self._length is not None
        chunk_size = min(size, self._length - self._read_bytes)
        chunk = await self._content.read(chunk_size)
        if self._content.at_eof():
            self._at_eof = True
        return chunk

    async def _read_chunk_from_stream(self, size: int) -> bytes:
        assert size >= self._boundary_len
        first_chunk = self._prev_chunk is None
        if first_chunk:
            self._prev_chunk = await self._content.read(size)

        chunk = b""
        while len(chunk) < self._boundary_len:
            chunk += await self._content.read(size)
            self._content_eof += int(self._content.at_eof())
            assert self._content_eof < 3
            if self._content_eof:
                break
        if len(chunk) > size:
            self._content.unread_data(chunk[size:])
            chunk = chunk[:size]

        assert self._prev_chunk is not None
        window = self._prev_chunk + chunk
        sub = b"\r\n" + self._boundary
        if first_chunk:
            idx = window.find(sub)
        else:
            idx = window.find(sub, max(0, len(self._prev_chunk) - len(sub)))
        if idx >= 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self._content.unread_data(window[idx:])
            if size > idx:
                self._prev_chunk = self._prev_chunk[:idx]
            chunk = window[len(self._prev_chunk):idx]
            if not chunk:
                self._at_eof = True
        result = self._prev_chunk
        self._prev_chunk = chunk
        return result

    async def readline(self) -> bytes:
        if self._at_eof:
            return b""

        if self._unread:
            line = self._unread.popleft()
        else:
            line = await self._content.readline()

        if line.startswith(self._boundary):
            sline = line.rstrip(b"\r\n")
            boundary = self._boundary
            last_boundary = self._boundary + b"--"
            if sline == boundary or sline == last_boundary:
                self._at_eof = True
                self._unread.append(line)
                return b""
        else:
            next_line = await self._content.readline()
            if next_line.startswith(self._boundary):
                line = line[:-2]
            self._unread.append(next_line)

        return line

    async def release(self) -> None:
        if self._at_eof:
            return
        while not self._at_eof:
            await self.read_chunk(self.chunk_size)

    async def text(self, *, encoding: Optional[str] = None) -> str:
        data = await self.read(decode=True)
        encoding = encoding or self.get_charset(default="utf-8")
        return data.decode(encoding)

    async def json(self, *, encoding: Optional[str] = None) -> Optional[Dict[str, Any]]:
        data = await self.read(decode=True)
        if not data:
            return None
        encoding = encoding or self.get_charset(default="utf-8")
        return cast(Dict[str, Any], json.loads(data.decode(encoding)))

    async def form(self, *, encoding: Optional[str] = None) -> List[Tuple[str, str]]:
        data = await self.read(decode=True)
        if not data:
            return []
        if encoding is not None:
            real_encoding = encoding
        else:
            real_encoding = self.get_charset(default="utf-8")
        try:
            decoded_data = data.rstrip().decode(real_encoding)
        except UnicodeDecodeError:
            raise ValueError("data cannot be decoded with %s encoding" % real_encoding)

        return parse_qsl(
            decoded_data,
            keep_blank_values=True,
            encoding=real_encoding,
        )

    def at_eof(self) -> bool:
        return self._at_eof

    def decode(self, data: bytes) -> bytes:
        if CONTENT_TRANSFER_ENCODING in self.headers:
            data = self._decode_content_transfer(data)
        if not self._is_form_data and CONTENT_ENCODING in self.headers:
            return self._decode_content(data)
        return data

    def _decode_content(self, data: bytes) -> bytes:
        encoding = self.headers.get(CONTENT_ENCODING, "").lower()
        if encoding == "identity":
            return data
        if encoding in {"deflate", "gzip"}:
            return ZLibDecompressor(
                encoding=encoding,
                suppress_deflate_header=True,
            ).decompress_sync(data)

        raise RuntimeError(f"unknown content encoding: {encoding}")

    def _decode_content_transfer(self, data: bytes) -> bytes:
        encoding = self.headers.get(CONTENT_TRANSFER_ENCODING, "").lower()

        if encoding == "base64":
            return base64.b64decode(data)
        elif encoding == "quoted-printable":
            return binascii.a2b_qp(data)
        elif encoding in ("binary", "8bit", "7bit"):
            return data
        else:
            raise RuntimeError(f"unknown content transfer encoding: {encoding}")

    def get_charset(self, default: str) -> str:
        ctype = self.headers.get(CONTENT_TYPE, "")
        mimetype = parse_mimetype(ctype)
        return mimetype.parameters.get("charset", self._default_charset or default)

    @reify
    def name(self) -> Optional[str]:
        _, params = parse_content_disposition(self.headers.get(CONTENT_DISPOSITION))
        return content_disposition_filename(params, "name")

    @reify
    def filename(self) -> Optional[str]:
        _, params = parse_content_disposition(self.headers.get(CONTENT_DISPOSITION))
        return content_disposition_filename(params, "filename")


@payload_type(BodyPartReader, order=Order.try_first)
class BodyPartReaderPayload(Payload):
    _value: BodyPartReader

    def __init__(self, value: BodyPartReader, *args: Any, **kwargs: Any) -> None:
        super().__init__(value, *args, **kwargs)

        params: Dict[str, str] = {}
        if value.name is not None:
            params["name"] = value.name
        if value.filename is not None:
            params["filename"] = value.filename

        if params:
            self.set_content_disposition("attachment", True, **params)

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        raise TypeError("Unable to decode.")

    async def write(self, writer: Any) -> None:
        field = self._value
        chunk = await field.read_chunk(size=2**16)
        while chunk:
            await writer.write(field.decode(chunk))
            chunk = await field.read_chunk(size=2**16)


class MultipartReader:
    response_wrapper_cls: Type[MultipartResponseWrapper] = MultipartResponseWrapper
    multipart_reader_cls: Optional[Type["MultipartReader"]] = None
    part_reader_cls: Type[BodyPartReader] = BodyPartReader

    def __init__(self, headers: Mapping[str, str], content: StreamReader) -> None:
        self._mimetype = parse_mimetype(headers[CONTENT_TYPE])
        assert self._mimetype.type == "multipart"
        if "boundary" not in self._mimetype.parameters:
            raise ValueError(
                "boundary missed for Content-Type: %s" % headers[CONTENT_TYPE]
            )

        self.headers = headers
        self._boundary = ("--" + self._get_boundary()).encode()
        self._content = content
        self._default_charset: Optional[str] = None
        self._last_part: Optional[Union["MultipartReader", BodyPartReader]] = None
        self._at_eof = False
        self._at_bof = True
        self._unread: List[bytes] = []

    def __aiter__(self