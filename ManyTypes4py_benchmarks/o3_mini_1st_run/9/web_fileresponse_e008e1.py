import asyncio
import io
import os
import pathlib
import sys
from contextlib import suppress
from enum import Enum, auto
from mimetypes import MimeTypes
from stat import S_ISREG
from types import MappingProxyType
from typing import IO, Awaitable, Callable, Optional, Set, Tuple, Sequence, Any

from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import ETAG_ANY, ETag, must_be_empty_body
from .typedefs import LooseHeaders, PathLike
from .web_exceptions import (
    HTTPForbidden,
    HTTPNotFound,
    HTTPNotModified,
    HTTPPartialContent,
    HTTPPreconditionFailed,
    HTTPRequestRangeNotSatisfiable,
)
from .web_response import StreamResponse

__all__ = ('FileResponse',)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .web_request import BaseRequest

_OnChunkSent = Optional[Callable[[bytes], Awaitable[None]]]
NOSENDFILE: Final[bool] = bool(os.environ.get('AIOHTTP_NOSENDFILE'))
CONTENT_TYPES: Final[MimeTypes] = MimeTypes()
ENCODING_EXTENSIONS: Final = MappingProxyType({ext: CONTENT_TYPES.encodings_map[ext] for ext in ('.br', '.gz')})
FALLBACK_CONTENT_TYPE: Final[str] = 'application/octet-stream'
ADDITIONAL_CONTENT_TYPES: Final = MappingProxyType({
    'application/gzip': '.gz',
    'application/x-brotli': '.br',
    'application/x-bzip2': '.bz2',
    'application/x-compress': '.Z',
    'application/x-xz': '.xz'
})


class _FileResponseResult(Enum):
    """The result of the file response."""
    SEND_FILE = auto()
    NOT_ACCEPTABLE = auto()
    PRE_CONDITION_FAILED = auto()
    NOT_MODIFIED = auto()


CONTENT_TYPES.encodings_map.clear()
for content_type, extension in ADDITIONAL_CONTENT_TYPES.items():
    CONTENT_TYPES.add_type(content_type, extension)
_CLOSE_FUTURES: Set[asyncio.Future[Any]] = set()


class FileResponse(StreamResponse):
    """A response object can be used to send files."""

    def __init__(
        self,
        path: PathLike,
        chunk_size: int = 256 * 1024,
        status: int = 200,
        reason: Optional[str] = None,
        headers: Optional[LooseHeaders] = None
    ) -> None:
        super().__init__(status=status, reason=reason, headers=headers)
        self._path: pathlib.Path = pathlib.Path(path)
        self._chunk_size: int = chunk_size

    def _seek_and_read(self, fobj: IO[bytes], offset: int, chunk_size: int) -> bytes:
        fobj.seek(offset)
        return fobj.read(chunk_size)

    async def _sendfile_fallback(
        self,
        writer: AbstractStreamWriter,
        fobj: IO[bytes],
        offset: int,
        count: int
    ) -> AbstractStreamWriter:
        chunk_size: int = self._chunk_size
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        chunk: bytes = await loop.run_in_executor(None, self._seek_and_read, fobj, offset, chunk_size)
        remaining: int = count
        while chunk:
            await writer.write(chunk)
            remaining -= chunk_size
            if remaining <= 0:
                break
            chunk = await loop.run_in_executor(None, fobj.read, min(chunk_size, remaining))
        await writer.drain()
        return writer

    async def _sendfile(
        self,
        request: "BaseRequest",
        fobj: IO[bytes],
        offset: int,
        count: int
    ) -> AbstractStreamWriter:
        writer: AbstractStreamWriter = await super().prepare(request)
        assert writer is not None
        if NOSENDFILE or self.compression:
            return await self._sendfile_fallback(writer, fobj, offset, count)
        loop: asyncio.AbstractEventLoop = request._loop
        transport = request.transport
        assert transport is not None
        try:
            await loop.sendfile(transport, fobj, offset, count)
        except NotImplementedError:
            return await self._sendfile_fallback(writer, fobj, offset, count)
        await super().write_eof()
        return writer

    @staticmethod
    def _etag_match(etag_value: str, etags: Sequence[ETag], *, weak: bool) -> bool:
        if len(etags) == 1 and etags[0].value == ETAG_ANY:
            return True
        return any((etag.value == etag_value for etag in etags if weak or not etag.is_weak))

    async def _not_modified(
        self,
        request: "BaseRequest",
        etag_value: str,
        last_modified: float
    ) -> AbstractStreamWriter:
        self.set_status(HTTPNotModified.status_code)
        self._length_check = False
        self.etag = etag_value
        self.last_modified = last_modified
        return await super().prepare(request)

    async def _precondition_failed(self, request: "BaseRequest") -> AbstractStreamWriter:
        self.set_status(HTTPPreconditionFailed.status_code)
        self.content_length = 0
        return await super().prepare(request)

    def _make_response(
        self,
        request: "BaseRequest",
        accept_encoding: str
    ) -> Tuple[_FileResponseResult, Optional[IO[bytes]], os.stat_result, Optional[str]]:
        file_path: pathlib.Path = self._path
        for file_extension, file_encoding in ENCODING_EXTENSIONS.items():
            if file_encoding not in accept_encoding:
                continue
            compressed_path: pathlib.Path = file_path.with_suffix(file_path.suffix + file_extension)
            with suppress(OSError):
                st: os.stat_result = compressed_path.lstat()
                if S_ISREG(st.st_mode):
                    return ( _FileResponseResult.SEND_FILE, compressed_path.open('rb'), st, file_encoding )
        st: os.stat_result = file_path.stat()
        return ( _FileResponseResult.SEND_FILE if S_ISREG(st.st_mode) else _FileResponseResult.NOT_ACCEPTABLE,
                 file_path if S_ISREG(st.st_mode) else None,
                 st,
                 None )

    def _get_file_path_stat_encoding(
        self,
        accept_encoding: str
    ) -> Tuple[Optional[pathlib.Path], os.stat_result, Optional[str]]:
        file_path: pathlib.Path = self._path
        for file_extension, file_encoding in ENCODING_EXTENSIONS.items():
            if file_encoding not in accept_encoding:
                continue
            compressed_path: pathlib.Path = file_path.with_suffix(file_path.suffix + file_extension)
            with suppress(OSError):
                st: os.stat_result = compressed_path.lstat()
                if S_ISREG(st.st_mode):
                    return (compressed_path, st, file_encoding)
        st: os.stat_result = file_path.stat()
        return (file_path if S_ISREG(st.st_mode) else None, st, None)

    async def prepare(self, request: "BaseRequest") -> AbstractStreamWriter:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        accept_encoding: str = request.headers.get(hdrs.ACCEPT_ENCODING, '').lower()
        try:
            response_result, fobj, st, file_encoding = await loop.run_in_executor(
                None, self._make_response, request, accept_encoding
            )
        except PermissionError:
            self.set_status(HTTPForbidden.status_code)
            return await super().prepare(request)
        except OSError:
            self.set_status(HTTPNotFound.status_code)
            return await super().prepare(request)
        if response_result is _FileResponseResult.NOT_ACCEPTABLE:
            self.set_status(HTTPForbidden.status_code)
            return await super().prepare(request)
        if response_result is _FileResponseResult.PRE_CONDITION_FAILED:
            return await self._precondition_failed(request)
        if response_result is _FileResponseResult.NOT_MODIFIED:
            etag_value: str = f'{st.st_mtime_ns:x}-{st.st_size:x}'
            last_modified: float = st.st_mtime
            return await self._not_modified(request, etag_value, last_modified)
        assert fobj is not None
        try:
            return await self._prepare_open_file(request, fobj, st, file_encoding)
        finally:
            close_future: asyncio.Future[Any] = loop.run_in_executor(None, fobj.close)
            _CLOSE_FUTURES.add(close_future)
            close_future.add_done_callback(_CLOSE_FUTURES.remove)

    async def _prepare_open_file(
        self,
        request: "BaseRequest",
        fobj: IO[bytes],
        st: os.stat_result,
        file_encoding: Optional[str]
    ) -> AbstractStreamWriter:
        status: int = self._status
        file_size: int = st.st_size
        file_mtime: float = st.st_mtime
        count: int = file_size
        start: Optional[int] = None
        ifrange = getattr(request, 'if_range', None)
        if ifrange is None or file_mtime <= ifrange.timestamp():
            try:
                rng = request.http_range
                start = rng.start  # type: ignore
                end = rng.stop  # type: ignore
            except ValueError:
                self._headers[hdrs.CONTENT_RANGE] = f'bytes */{file_size}'
                self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                return await super().prepare(request)
            if start is not None:
                if start < 0 and (end is None):
                    start += file_size
                    if start < 0:
                        start = 0
                    count = file_size - start
                else:
                    count = min(end if end is not None else file_size, file_size) - start  # type: ignore
                if start >= file_size:
                    self._headers[hdrs.CONTENT_RANGE] = f'bytes */{file_size}'
                    self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                    return await super().prepare(request)
                status = HTTPPartialContent.status_code
                self.set_status(status)
        if hdrs.CONTENT_TYPE not in self._headers:
            if sys.version_info >= (3, 13):
                guesser = CONTENT_TYPES.guess_file_type  # type: ignore
            else:
                guesser = CONTENT_TYPES.guess_type
            self.content_type = guesser(self._path)[0] or FALLBACK_CONTENT_TYPE  # type: ignore
        if file_encoding:
            self._headers[hdrs.CONTENT_ENCODING] = file_encoding
            self._headers[hdrs.VARY] = hdrs.ACCEPT_ENCODING
            self._compression = False
        self.etag = f'{st.st_mtime_ns:x}-{st.st_size:x}'
        self.last_modified = file_mtime
        self.content_length = count
        self._headers[hdrs.ACCEPT_RANGES] = 'bytes'
        if status == HTTPPartialContent.status_code:
            real_start: int = start  # type: ignore
            self._headers[hdrs.CONTENT_RANGE] = f'bytes {real_start}-{real_start + count - 1}/{file_size}'
        if count == 0 or must_be_empty_body(request.method, status):
            return await super().prepare(request)
        offset: int = start if start is not None else 0
        return await self._sendfile(request, fobj, offset, count)