import asyncio
import base64
import binascii
import hashlib
import json
import sys
from typing import Any, Final, Iterable, Optional, Tuple, Union, Dict, List, Callable, Awaitable, Set, cast
from multidict import CIMultiDict
from . import hdrs
from ._websocket.reader import WebSocketDataQueue
from ._websocket.writer import DEFAULT_LIMIT
from .abc import AbstractStreamWriter
from .client_exceptions import WSMessageTypeError
from .helpers import calculate_timeout_when, frozen_dataclass_decorator, set_exception, set_result
from .http import WS_CLOSED_MESSAGE, WS_CLOSING_MESSAGE, WS_KEY, WebSocketError, WebSocketReader, WebSocketWriter, WSCloseCode, WSMessage, WSMsgType, ws_ext_gen, ws_ext_parse
from .http_websocket import _INTERNAL_RECEIVE_TYPES, WSMessageError
from .log import ws_logger
from .streams import EofStream
from .typedefs import JSONDecoder, JSONEncoder
from .web_exceptions import HTTPBadRequest, HTTPException
from .web_request import BaseRequest
from .web_response import StreamResponse
if sys.version_info >= (3, 11):
    import asyncio as async_timeout
else:
    import async_timeout
__all__ = ('WebSocketResponse', 'WebSocketReady', 'WSMsgType')
THRESHOLD_CONNLOST_ACCESS: Final[int] = 5

@frozen_dataclass_decorator
class WebSocketReady:
    ok: bool
    protocol: Optional[str]

    def __bool__(self) -> bool:
        return self.ok

class WebSocketResponse(StreamResponse):
    _length_check: bool = False
    _ws_protocol: Optional[str] = None
    _writer: Optional[WebSocketWriter] = None
    _reader: Optional[WebSocketDataQueue] = None
    _closed: bool = False
    _closing: bool = False
    _conn_lost: int = 0
    _close_code: Optional[int] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _waiting: bool = False
    _close_wait: Optional[asyncio.Future[None]] = None
    _exception: Optional[Exception] = None
    _heartbeat_when: float = 0.0
    _heartbeat_cb: Optional[asyncio.TimerHandle] = None
    _pong_response_cb: Optional[asyncio.TimerHandle] = None
    _ping_task: Optional[asyncio.Task[None]] = None

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        receive_timeout: Optional[float] = None,
        autoclose: bool = True,
        autoping: bool = True,
        heartbeat: Optional[float] = None,
        protocols: Iterable[str] = (),
        compress: bool = True,
        max_msg_size: int = 4 * 1024 * 1024,
        writer_limit: int = DEFAULT_LIMIT,
    ) -> None:
        super().__init__(status=101)
        self._protocols: Tuple[str, ...] = tuple(protocols)
        self._timeout: float = timeout
        self._receive_timeout: Optional[float] = receive_timeout
        self._autoclose: bool = autoclose
        self._autoping: bool = autoping
        self._heartbeat: Optional[float] = heartbeat
        self._pong_heartbeat: float = heartbeat / 2.0 if heartbeat is not None else 0.0
        self._compress: bool = compress
        self._max_msg_size: int = max_msg_size
        self._writer_limit: int = writer_limit

    def _cancel_heartbeat(self) -> None:
        self._cancel_pong_response_cb()
        if self._heartbeat_cb is not None:
            self._heartbeat_cb.cancel()
            self._heartbeat_cb = None
        if self._ping_task is not None:
            self._ping_task.cancel()
            self._ping_task = None

    def _cancel_pong_response_cb(self) -> None:
        if self._pong_response_cb is not None:
            self._pong_response_cb.cancel()
            self._pong_response_cb = None

    def _reset_heartbeat(self) -> None:
        if self._heartbeat is None:
            return
        self._cancel_pong_response_cb()
        req = self._req
        timeout_ceil_threshold: float = req._protocol._timeout_ceil_threshold if req is not None else 5
        loop = self._loop
        assert loop is not None
        now: float = loop.time()
        when: float = calculate_timeout_when(now, self._heartbeat, timeout_ceil_threshold)
        self._heartbeat_when = when
        if self._heartbeat_cb is None:
            self._heartbeat_cb = loop.call_at(when, self._send_heartbeat)

    def _send_heartbeat(self) -> None:
        self._heartbeat_cb = None
        loop = self._loop
        assert loop is not None and self._writer is not None
        now: float = loop.time()
        if now < self._heartbeat_when:
            self._heartbeat_cb = loop.call_at(self._heartbeat_when, self._send_heartbeat)
            return
        req = self._req
        timeout_ceil_threshold: float = req._protocol._timeout_ceil_threshold if req is not None else 5
        when: float = calculate_timeout_when(now, self._pong_heartbeat, timeout_ceil_threshold)
        self._cancel_pong_response_cb()
        self._pong_response_cb = loop.call_at(when, self._pong_not_received)
        coro: Awaitable[None] = self._writer.send_frame(b'', WSMsgType.PING)
        if sys.version_info >= (3, 12):
            ping_task: asyncio.Task[None] = asyncio.Task(coro, loop=loop, eager_start=True)
        else:
            ping_task = loop.create_task(coro)
        if not ping_task.done():
            self._ping_task = ping_task
            ping_task.add_done_callback(self._ping_task_done)
        else:
            self._ping_task_done(ping_task)

    def _ping_task_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (exc := task.exception()):
            self._handle_ping_pong_exception(exc)
        self._ping_task = None

    def _pong_not_received(self) -> None:
        if self._req is not None and self._req.transport is not None:
            self._handle_ping_pong_exception(asyncio.TimeoutError(f'No PONG received after {self._pong_heartbeat} seconds'))

    def _handle_ping_pong_exception(self, exc: Exception) -> None:
        if self._closed:
            return
        self._set_closed()
        self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
        self._exception = exc
        if self._waiting and (not self._closing) and (self._reader is not None):
            self._reader.feed_data(WSMessageError(data=exc, extra=None))

    def _set_closed(self) -> None:
        self._closed = True
        self._cancel_heartbeat()

    async def prepare(self, request: BaseRequest) -> AbstractStreamWriter:
        if self._payload_writer is not None:
            return self._payload_writer
        protocol, writer = self._pre_start(request)
        payload_writer = await super().prepare(request)
        assert payload_writer is not None
        self._post_start(request, protocol, writer)
        await payload_writer.drain()
        return payload_writer

    def _handshake(self, request: BaseRequest) -> Tuple[CIMultiDict, Optional[str], int, bool]:
        headers = request.headers
        if 'websocket' != headers.get(hdrs.UPGRADE, '').lower().strip():
            raise HTTPBadRequest(text='No WebSocket UPGRADE hdr: {}\n Can "Upgrade" only to "WebSocket".'.format(headers.get(hdrs.UPGRADE)))
        if 'upgrade' not in headers.get(hdrs.CONNECTION, '').lower():
            raise HTTPBadRequest(text='No CONNECTION upgrade hdr: {}'.format(headers.get(hdrs.CONNECTION)))
        protocol: Optional[str] = None
        if hdrs.SEC_WEBSOCKET_PROTOCOL in headers:
            req_protocols: List[str] = [str(proto.strip()) for proto in headers[hdrs.SEC_WEBSOCKET_PROTOCOL].split(',')]
            for proto in req_protocols:
                if proto in self._protocols:
                    protocol = proto
                    break
            else:
                ws_logger.warning('Client protocols %r don’t overlap server-known ones %r', req_protocols, self._protocols)
        version: str = headers.get(hdrs.SEC_WEBSOCKET_VERSION, '')
        if version not in ('13', '8', '7'):
            raise HTTPBadRequest(text=f'Unsupported version: {version}')
        key: Optional[str] = headers.get(hdrs.SEC_WEBSOCKET_KEY)
        try:
            if not key or len(base64.b64decode(key)) != 16:
                raise HTTPBadRequest(text=f'Handshake error: {key!r}')
        except binascii.Error:
            raise HTTPBadRequest(text=f'Handshake error: {key!r}') from None
        accept_val: str = base64.b64encode(hashlib.sha1(key.encode() + WS_KEY).digest()).decode()
        response_headers: CIMultiDict = CIMultiDict({hdrs.UPGRADE: 'websocket', hdrs.CONNECTION: 'upgrade', hdrs.SEC_WEBSOCKET_ACCEPT: accept_val})
        notakeover: bool = False
        compress: int = 0
        if self._compress:
            extensions: Optional[str] = headers.get(hdrs.SEC_WEBSOCKET_EXTENSIONS)
            compress, notakeover = ws_ext_parse(extensions, isserver=True)
            if compress:
                enabledext: str = ws_ext_gen(compress=compress, isserver=True, server_notakeover=notakeover)
                response_headers[hdrs.SEC_WEBSOCKET_EXTENSIONS] = enabledext
        if protocol:
            response_headers[hdrs.SEC_WEBSOCKET_PROTOCOL] = protocol
        return (response_headers, protocol, compress, notakeover)

    def _pre_start(self, request: BaseRequest) -> Tuple[Optional[str], WebSocketWriter]:
        self._loop = request._loop
        headers, protocol, compress, notakeover = self._handshake(request)
        self.set_status(101)
        self.headers.update(headers)
        self.force_close()
        self._compress = compress
        transport = request._protocol.transport
        assert transport is not None
        writer: WebSocketWriter = WebSocketWriter(request._protocol, transport, compress=compress, notakeover=notakeover, limit=self._writer_limit)
        return (protocol, writer)

    def _post_start(self, request: BaseRequest, protocol: Optional[str], writer: WebSocketWriter) -> None:
        self._ws_protocol = protocol
        self._writer = writer
        self._reset_heartbeat()
        loop = self._loop
        assert loop is not None
        self._reader = WebSocketDataQueue(request._protocol, 2 ** 16, loop=loop)
        request.protocol.set_parser(WebSocketReader(self._reader, self._max_msg_size, compress=bool(self._compress)))
        request.protocol.keep_alive(False)

    def can_prepare(self, request: BaseRequest) -> WebSocketReady:
        if self._writer is not None:
            raise RuntimeError('Already started')
        try:
            _, protocol, _, _ = self._handshake(request)
        except HTTPException:
            return WebSocketReady(False, None)
        else:
            return WebSocketReady(True, protocol)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_code(self) -> Optional[int]:
        return self._close_code

    @property
    def ws_protocol(self) -> Optional[str]:
        return self._ws_protocol

    @property
    def compress(self) -> bool:
        return self._compress

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        writer = self._writer
        if writer is None:
            return default
        return writer.transport.get_extra_info(name, default)

    def exception(self) -> Optional[Exception]:
        return self._exception

    async def ping(self, message: bytes = b'') -> None:
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        await self._writer.send_frame(message, WSMsgType.PING)

    async def pong(self, message: bytes = b'') -> None:
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        await self._writer.send_frame(message, WSMsgType.PONG)

    async def send_frame(self, message: Union[str, bytes], opcode: int, compress: Optional[bool] = None) -> None:
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        await self._writer.send_frame(message, opcode, compress)

    async def send_str(self, data: str, compress: Optional[bool] = None) -> None:
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        if not isinstance(data, str):
            raise TypeError('data argument must be str (%r)' % type(data))
        await self._writer.send_frame(data.encode('utf-8'), WSMsgType.TEXT, compress=compress)

    async def send_bytes(self, data: Union[bytes, bytearray, memoryview], compress: Optional[bool] = None) -> None:
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('data argument must be byte-ish (%r)' % type(data))
        await self._writer.send_frame(data, WSMsgType.BINARY, compress=compress)

    async def send_json(self, data: Any, compress: Optional[bool] = None, *, dumps: JSONEncoder = json.dumps) -> None:
        await self.send_str(dumps(data), compress=compress)

    async def write_eof(self) -> None:
        if self._eof_sent:
            return
        if self._payload_writer is None:
            raise RuntimeError('Response has not been started')
        await self.close()
        self._eof_sent = True

    async def close(self, *, code: int = WSCloseCode.OK, message: bytes = b'', drain: bool = True) -> bool:
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        if self._closed:
            return False
        self._set_closed()
        try:
            await self._writer.close(code, message)
            writer = self._payload_writer
            assert writer is not None
            if drain:
                await writer.drain()
        except (asyncio.CancelledError, asyncio.TimeoutError):
            self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
            raise
        except Exception as exc:
            self._exception = exc
            self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
            return True
        reader = self._reader
        assert reader is not None
        if self._waiting:
            assert self._loop is not None
            assert self._close_wait is None
            self._close_wait = self._loop.create_future()
            reader.feed_data(WS_CLOSING_MESSAGE)
            await self._close_wait
        if self._closing:
            self._close_transport()
            return True
        try:
            async with async_timeout.timeout(self._timeout):
                while True:
                    msg = await reader.read()
                    if msg.type is WSMsgType.CLOSE:
                        self._set_code_close_transport(msg.data)
                        return True
        except asyncio.CancelledError:
            self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
            raise
        except Exception as exc:
            self._exception = exc
            self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
            return True

    def _set_closing(self, code: int) -> None:
        self._closing = True
        self._close_code = code
        self._cancel_heartbeat()

    def _set_code_close_transport(self, code: int) -> None:
        self._close_code = code
        self._close_transport()

    def _close_transport(self) -> None:
        if self._req is not None and self._req.transport is not None:
            self._req.transport.close()

    async def receive(self, timeout: Optional[float] = None) -> WSMessage:
        if self._reader is None:
            raise RuntimeError('Call .prepare() first')
        receive_timeout = timeout or self._receive_timeout
        while True:
            if self._waiting:
                raise RuntimeError('Concurrent call to receive() is not allowed')
            if self._closed:
                self._conn_lost += 1
                if self._conn_lost >= THRESHOLD_CONNLOST_ACCESS:
                    raise RuntimeError('WebSocket connection is closed.')
                return WS_CLOSED_MESSAGE
            elif self._closing:
                return WS_CLOSING_MESSAGE
            try:
                self._waiting = True
                try:
                    if receive_timeout:
                        async with async_timeout.timeout(re