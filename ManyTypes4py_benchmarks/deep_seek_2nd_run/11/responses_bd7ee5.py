from __future__ import annotations
import hashlib
import http.cookies
import json
import os
import re
import stat
import typing
import warnings
from datetime import datetime
from email.utils import format_datetime, formatdate
from functools import partial
from mimetypes import guess_type
from secrets import token_hex
from urllib.parse import quote
import anyio
import anyio.to_thread
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, Headers, MutableHeaders
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

T = typing.TypeVar('T')

class Response:
    media_type: typing.Optional[str] = None
    charset: str = 'utf-8'

    def __init__(
        self,
        content: typing.Any = None,
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None
    ) -> None:
        self.status_code = status_code
        if media_type is not None:
            self.media_type = media_type
        self.background = background
        self.body = self.render(content)
        self.init_headers(headers)

    def render(self, content: typing.Any) -> bytes:
        if content is None:
            return b''
        if isinstance(content, (bytes, memoryview)):
            return content
        return content.encode(self.charset)

    def init_headers(self, headers: typing.Optional[typing.Mapping[str, str]] = None) -> None:
        if headers is None:
            raw_headers: typing.List[typing.Tuple[bytes, bytes]] = []
            populate_content_length = True
            populate_content_type = True
        else:
            raw_headers = [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in headers.items()]
            keys = [h[0] for h in raw_headers]
            populate_content_length = b'content-length' not in keys
            populate_content_type = b'content-type' not in keys
        body = getattr(self, 'body', None)
        if body is not None and populate_content_length and (not (self.status_code < 200 or self.status_code in (204, 304))):
            content_length = str(len(body))
            raw_headers.append((b'content-length', content_length.encode('latin-1')))
        content_type = self.media_type
        if content_type is not None and populate_content_type:
            if content_type.startswith('text/') and 'charset=' not in content_type.lower():
                content_type += '; charset=' + self.charset
            raw_headers.append((b'content-type', content_type.encode('latin-1')))
        self.raw_headers = raw_headers

    @property
    def headers(self) -> MutableHeaders:
        if not hasattr(self, '_headers'):
            self._headers = MutableHeaders(raw=self.raw_headers)
        return self._headers

    def set_cookie(
        self,
        key: str,
        value: str = '',
        max_age: typing.Optional[int] = None,
        expires: typing.Optional[typing.Union[datetime, str]] = None,
        path: str = '/',
        domain: typing.Optional[str] = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str = 'lax'
    ) -> None:
        cookie = http.cookies.SimpleCookie()
        cookie[key] = value
        if max_age is not None:
            cookie[key]['max-age'] = max_age
        if expires is not None:
            if isinstance(expires, datetime):
                cookie[key]['expires'] = format_datetime(expires, usegmt=True)
            else:
                cookie[key]['expires'] = expires
        if path is not None:
            cookie[key]['path'] = path
        if domain is not None:
            cookie[key]['domain'] = domain
        if secure:
            cookie[key]['secure'] = True
        if httponly:
            cookie[key]['httponly'] = True
        if samesite is not None:
            assert samesite.lower() in ['strict', 'lax', 'none'], "samesite must be either 'strict', 'lax' or 'none'"
            cookie[key]['samesite'] = samesite
        cookie_val = cookie.output(header='').strip()
        self.raw_headers.append((b'set-cookie', cookie_val.encode('latin-1')))

    def delete_cookie(
        self,
        key: str,
        path: str = '/',
        domain: typing.Optional[str] = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str = 'lax'
    ) -> None:
        self.set_cookie(key, max_age=0, expires=0, path=path, domain=domain, secure=secure, httponly=httponly, samesite=samesite)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        prefix = 'websocket.' if scope['type'] == 'websocket' else ''
        await send({'type': prefix + 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
        await send({'type': prefix + 'http.response.body', 'body': self.body})
        if self.background is not None:
            await self.background()

class HTMLResponse(Response):
    media_type = 'text/html'

class PlainTextResponse(Response):
    media_type = 'text/plain'

class JSONResponse(Response):
    media_type = 'application/json'

    def __init__(
        self,
        content: typing.Any,
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None
    ) -> None:
        super().__init__(content, status_code, headers, media_type, background)

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None, separators=(',', ':')).encode('utf-8')

class RedirectResponse(Response):
    def __init__(
        self,
        url: typing.Union[str, URL],
        status_code: int = 307,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        background: typing.Optional[BackgroundTask] = None
    ) -> None:
        super().__init__(content=b'', status_code=status_code, headers=headers, background=background)
        self.headers['location'] = quote(str(url), safe=":/%#?=@[]!$&'()*+,;")

Content = typing.Union[str, bytes, memoryview]
SyncContentStream = typing.Iterable[Content]
AsyncContentStream = typing.AsyncIterable[Content]
ContentStream = typing.Union[AsyncContentStream, SyncContentStream]

class StreamingResponse(Response):
    def __init__(
        self,
        content: ContentStream,
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None
    ) -> None:
        if isinstance(content, typing.AsyncIterable):
            self.body_iterator = content
        else:
            self.body_iterator = iterate_in_threadpool(content)
        self.status_code = status_code
        self.media_type = self.media_type if media_type is None else media_type
        self.background = background
        self.init_headers(headers)

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                break

    async def stream_response(self, send: Send) -> None:
        await send({'type': 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
        async for chunk in self.body_iterator:
            if not isinstance(chunk, (bytes, memoryview)):
                chunk = chunk.encode(self.charset)
            await send({'type': 'http.response.body', 'body': chunk, 'more_body': True})
        await send({'type': 'http.response.body', 'body': b'', 'more_body': False})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        spec_version = tuple(map(int, scope.get('asgi', {}).get('spec_version', '2.0').split('.')))
        if spec_version >= (2, 4):
            try:
                await self.stream_response(send)
            except OSError:
                raise ClientDisconnect()
        else:
            async with anyio.create_task_group() as task_group:
                async def wrap(func: typing.Callable[[], typing.Awaitable[None]]) -> None:
                    await func()
                    task_group.cancel_scope.cancel()
                task_group.start_soon(wrap, partial(self.stream_response, send))
                await wrap(partial(self.listen_for_disconnect, receive))
        if self.background is not None:
            await self.background()

class MalformedRangeHeader(Exception):
    def __init__(self, content: str = 'Malformed range header.') -> None:
        self.content = content

class RangeNotSatisfiable(Exception):
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size

_RANGE_PATTERN = re.compile('(\\d*)-(\\d*)')

class FileResponse(Response):
    chunk_size = 64 * 1024

    def __init__(
        self,
        path: str,
        status_code: int = 200,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
        media_type: typing.Optional[str] = None,
        background: typing.Optional[BackgroundTask] = None,
        filename: typing.Optional[str] = None,
        stat_result: typing.Optional[os.stat_result] = None,
        method: typing.Optional[str] = None,
        content_disposition_type: str = 'attachment'
    ) -> None:
        self.path = path
        self.status_code = status_code
        self.filename = filename
        if method is not None:
            warnings.warn("The 'method' parameter is not used, and it will be removed.", DeprecationWarning)
        if media_type is None:
            media_type = guess_type(filename or path)[0] or 'text/plain'
        self.media_type = media_type
        self.background = background
        self.init_headers(headers)
        self.headers.setdefault('accept-ranges', 'bytes')
        if self.filename is not None:
            content_disposition_filename = quote(self.filename)
            if content_disposition_filename != self.filename:
                content_disposition = f"{content_disposition_type}; filename*=utf-8''{content_disposition_filename}"
            else:
                content_disposition = f'{content_disposition_type}; filename="{self.filename}"'
            self.headers.setdefault('content-disposition', content_disposition)
        self.stat_result = stat_result
        if stat_result is not None:
            self.set_stat_headers(stat_result)

    def set_stat_headers(self, stat_result: os.stat_result) -> None:
        content_length = str(stat_result.st_size)
        last_modified = formatdate(stat_result.st_mtime, usegmt=True)
        etag_base = str(stat_result.st_mtime) + '-' + str(stat_result.st_size)
        etag = f'"{hashlib.md5(etag_base.encode(), usedforsecurity=False).hexdigest()}"'
        self.headers.setdefault('content-length', content_length)
        self.headers.setdefault('last-modified', last_modified)
        self.headers.setdefault('etag', etag)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        send_header_only = scope['method'].upper() == 'HEAD'
        if self.stat_result is None:
            try:
                stat_result = await anyio.to_thread.run_sync(os.stat, self.path)
                self.set_stat_headers(stat_result)
            except FileNotFoundError:
                raise RuntimeError(f'File at path {self.path} does not exist.')
            else:
                mode = stat_result.st_mode
                if not stat.S_ISREG(mode):
                    raise RuntimeError(f'File at path {self.path} is not a file.')
        else:
            stat_result = self.stat_result
        headers = Headers(scope=scope)
        http_range = headers.get('range')
        http_if_range = headers.get('if-range')
        if http_range is None or (http_if_range is not None and (not self._should_use_range(http_if_range))):
            await self._handle_simple(send, send_header_only)
        else:
            try:
                ranges = self._parse_range_header(http_range, stat_result.st_size)
            except MalformedRangeHeader as exc:
                return await PlainTextResponse(exc.content, status_code=400)(scope, receive, send)
            except RangeNotSatisfiable as exc:
                response = PlainTextResponse(status_code=416, headers={'Content-Range': f'*/{exc.max_size}'})
                return await response(scope, receive, send)
            if len(ranges) == 1:
                start, end = ranges[0]
                await self._handle_single_range(send, start, end, stat_result.st_size, send_header_only)
            else:
                await self._handle_multiple_ranges(send, ranges, stat_result.st_size, send_header_only)
        if self.background is not None:
            await self.background()

    async def _handle_simple(self, send: Send, send_header_only: bool) -> None:
        await send({'type': 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
        if send_header_only:
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        else:
            async with await anyio.open_file(self.path, mode='rb') as file:
                more_body = True
                while more_body:
                    chunk = await file.read(self.chunk_size)
                    more_body = len(chunk) == self.chunk_size
                    await send({'type': 'http.response.body', 'body': chunk, 'more_body': more_body})

    async def _handle_single_range(
        self,
        send: Send,
        start: int,
        end: int,
        file_size: int,
        send_header_only: bool
    ) -> None:
        self.headers['content-range'] = f'bytes {start}-{end - 1}/{file_size}'
        self.headers['content-length'] = str(end - start)
        await send({'type': 'http.response.start', 'status': 206, 'headers': self.raw_headers})
        if send_header_only:
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        else:
            async with await anyio.open_file(self.path, mode='rb') as file:
                await file.seek(start)
                more_body = True
                while more_body:
                    chunk = await file.read(min(self.chunk_size, end - start))
                    start += len(chunk)
                    more_body = len(chunk) == self.chunk_size and start < end
                    await send({'type': 'http.response.body', 'body': chunk, 'more_body': more_body})

    async def _handle_multiple_ranges(
        self,
        send: Send,
        ranges: typing.List[typing.Tuple[int, int]],
        file_size: int,
        send_header_only: bool
    ) -> None:
        boundary = token_hex(13)
        content_length, header_generator = self.generate_multipart(ranges, boundary, file_size, self.headers['content-type'])
        self.headers['content-range'] = f'multipart/byteranges; boundary={boundary}'
        self.headers['content-length'] = str(content_length)
        await send({'type': 'http.response.start', 'status': 206, 'headers': self.raw_headers})
        if send_header_only:
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        else:
            async with await anyio.open_file(self.path, mode='rb') as file:
                for start, end in ranges:
                    await send({'type': 'http.response.body', 'body': header_generator(start, end), 'more_body': True})
                    await file.seek(start)
                    while start < end:
                        chunk = await file.read(min(self.chunk_size, end - start))
                        start += len(chunk)
                        await send({'type': 'http.response.body', 'body': chunk, 'more_body': True})
                    await send({'type': 'http.response.body', 'body': b'\n', 'more_body': True})
                await send({'type': 'http.response.body', 'body': f'\n--{boundary}--\n'.encode('latin-1'), 'more_body': False})

    def _should_use_range(self, http_if_range: str) -> bool:
        return http_if_range == self.headers['last-modified'] or http_if_range == self.headers['etag']

    @staticmethod
    def _parse_range_header(http_range: str, file_size: int) -> typing.List[typing.Tuple[int, int]]:
        ranges = []
        try:
            units, range_ = http_range.split('=', 1)
        except ValueError:
            raise MalformedRangeHeader()
        units = units.strip().lower()
        if units != 'bytes':
            raise MalformedRangeHeader('Only support bytes range')
        ranges = [(int(_[0]) if _[0] else file_size - int(_[1]), int(_[1]) + 1 if _[0] and _[1] and (int(_[1]) < file_size) else file_size) for _ in _RANGE_PATTERN.findall(range_) if _ != ('', '')]
        if len(ranges) == 0:
            raise MalformedRangeHeader('Range header: range must be requested')
        if any((not 0 <= start < file_size for start, _ in ranges)):
           