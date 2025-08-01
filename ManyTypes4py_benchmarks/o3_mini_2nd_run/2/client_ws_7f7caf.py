"""WebSocket client for asyncio."""
import asyncio
import sys
from types import TracebackType
from typing import Any, Final, Optional, Type, Union, AsyncIterator, AsyncContextManager, Awaitable
from ._websocket.reader import WebSocketDataQueue
from .client_exceptions import ClientError, ServerTimeoutError, WSMessageTypeError
from .client_reqrep import ClientResponse
from .helpers import calculate_timeout_when, frozen_dataclass_decorator, set_result
from .http import WS_CLOSED_MESSAGE, WS_CLOSING_MESSAGE, WebSocketError, WSCloseCode, WSMessage, WSMsgType
from .http_websocket import _INTERNAL_RECEIVE_TYPES, WebSocketWriter, WSMessageError
from .streams import EofStream
from .typedefs import DEFAULT_JSON_DECODER, DEFAULT_JSON_ENCODER, JSONDecoder, JSONEncoder

if sys.version_info >= (3, 11):
    import asyncio as async_timeout
else:
    import async_timeout


@frozen_dataclass_decorator
class ClientWSTimeout:
    ws_receive: Optional[float] = None
    ws_close: Optional[float] = None


DEFAULT_WS_CLIENT_TIMEOUT: Final[ClientWSTimeout] = ClientWSTimeout(ws_receive=None, ws_close=10.0)


class ClientWebSocketResponse:
    def __init__(
        self,
        reader: WebSocketDataQueue,
        writer: WebSocketWriter,
        protocol: Any,
        response: ClientResponse,
        timeout: ClientWSTimeout,
        autoclose: bool,
        autoping: bool,
        loop: asyncio.AbstractEventLoop,
        *,
        heartbeat: Optional[float] = None,
        compress: int = 0,
        client_notakeover: bool = False
    ) -> None:
        self._response: ClientResponse = response
        self._conn: Any = response.connection
        self._writer: WebSocketWriter = writer
        self._reader: WebSocketDataQueue = reader
        self._protocol: Any = protocol
        self._closed: bool = False
        self._closing: bool = False
        self._close_code: Optional[int] = None
        self._timeout: ClientWSTimeout = timeout
        self._autoclose: bool = autoclose
        self._autoping: bool = autoping
        self._heartbeat: Optional[float] = heartbeat
        self._heartbeat_cb: Optional[asyncio.Handle] = None
        self._heartbeat_when: float = 0.0
        if heartbeat is not None:
            self._pong_heartbeat: float = heartbeat / 2.0
        else:
            self._pong_heartbeat = 0.0
        self._pong_response_cb: Optional[asyncio.Handle] = None
        self._loop: asyncio.AbstractEventLoop = loop
        self._waiting: bool = False
        self._close_wait: Optional[asyncio.Future[Any]] = None
        self._exception: Optional[BaseException] = None
        self._compress: int = compress
        self._client_notakeover: bool = client_notakeover
        self._ping_task: Optional[asyncio.Task[Any]] = None
        self._reset_heartbeat()

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
        loop: asyncio.AbstractEventLoop = self._loop
        conn: Any = self._conn
        timeout_ceil_threshold: Union[int, float] = (
            conn._connector._timeout_ceil_threshold if conn is not None else 5
        )
        now: float = loop.time()
        when: float = calculate_timeout_when(now, self._heartbeat, timeout_ceil_threshold)
        self._heartbeat_when = when
        if self._heartbeat_cb is None:
            self._heartbeat_cb = loop.call_at(when, self._send_heartbeat)

    def _send_heartbeat(self) -> None:
        self._heartbeat_cb = None
        loop: asyncio.AbstractEventLoop = self._loop
        now: float = loop.time()
        if now < self._heartbeat_when:
            self._heartbeat_cb = loop.call_at(self._heartbeat_when, self._send_heartbeat)
            return
        conn: Any = self._conn
        timeout_ceil_threshold: Union[int, float] = (
            conn._connector._timeout_ceil_threshold if conn is not None else 5
        )
        when: float = calculate_timeout_when(now, self._pong_heartbeat, timeout_ceil_threshold)
        self._cancel_pong_response_cb()
        self._pong_response_cb = loop.call_at(when, self._pong_not_received)
        coro: Awaitable[None] = self._writer.send_frame(b'', WSMsgType.PING)
        if sys.version_info >= (3, 12):
            ping_task: asyncio.Task[Any] = asyncio.Task(coro, loop=loop, eager_start=True)
        else:
            ping_task = loop.create_task(coro)
        if not ping_task.done():
            self._ping_task = ping_task
            ping_task.add_done_callback(self._ping_task_done)
        else:
            self._ping_task_done(ping_task)

    def _ping_task_done(self, task: asyncio.Task[Any]) -> None:
        """Callback for when the ping task completes."""
        if not task.cancelled() and (exc := task.exception()):
            self._handle_ping_pong_exception(exc)
        self._ping_task = None

    def _pong_not_received(self) -> None:
        self._handle_ping_pong_exception(ServerTimeoutError(f'No PONG received after {self._pong_heartbeat} seconds'))

    def _handle_ping_pong_exception(self, exc: Exception) -> None:
        """Handle exceptions raised during ping/pong processing."""
        if self._closed:
            return
        self._set_closed()
        self._close_code = WSCloseCode.ABNORMAL_CLOSURE
        self._exception = exc
        self._response.close()
        if self._waiting and (not self._closing):
            self._reader.feed_data(WSMessageError(data=exc, extra=None))

    def _set_closed(self) -> None:
        """Set the connection to closed.

        Cancel any heartbeat timers and set the closed flag.
        """
        self._closed = True
        self._cancel_heartbeat()

    def _set_closing(self) -> None:
        """Set the connection to closing.

        Cancel any heartbeat timers and set the closing flag.
        """
        self._closing = True
        self._cancel_heartbeat()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_code(self) -> Optional[int]:
        return self._close_code

    @property
    def protocol(self) -> Any:
        return self._protocol

    @property
    def compress(self) -> int:
        return self._compress

    @property
    def client_notakeover(self) -> bool:
        return self._client_notakeover

    def get_extra_info(self, name: str, default: Optional[Any] = None) -> Any:
        """extra info from connection transport"""
        conn: Any = self._response.connection
        if conn is None:
            return default
        transport: Any = conn.transport
        if transport is None:
            return default
        return transport.get_extra_info(name, default)

    def exception(self) -> Optional[BaseException]:
        return self._exception

    async def ping(self, message: bytes = b'') -> None:
        await self._writer.send_frame(message, WSMsgType.PING)

    async def pong(self, message: bytes = b'') -> None:
        await self._writer.send_frame(message, WSMsgType.PONG)

    async def send_frame(self, message: Union[bytes, str], opcode: int, compress: Optional[bool] = None) -> None:
        """Send a frame over the websocket."""
        await self._writer.send_frame(message, opcode, compress)

    async def send_str(self, data: str, compress: Optional[bool] = None) -> None:
        if not isinstance(data, str):
            raise TypeError('data argument must be str (%r)' % type(data))
        await self._writer.send_frame(data.encode('utf-8'), WSMsgType.TEXT, compress=compress)

    async def send_bytes(self, data: Union[bytes, bytearray, memoryview], compress: Optional[bool] = None) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('data argument must be byte-ish (%r)' % type(data))
        await self._writer.send_frame(data, WSMsgType.BINARY, compress=compress)

    async def send_json(self, data: Any, compress: Optional[bool] = None, *, dumps: JSONEncoder = DEFAULT_JSON_ENCODER) -> None:
        await self.send_str(dumps(data), compress=compress)

    async def close(self, *, code: int = WSCloseCode.OK, message: bytes = b'') -> bool:
        if self._waiting and (not self._closing):
            assert self._loop is not None
            self._close_wait = self._loop.create_future()
            self._set_closing()
            self._reader.feed_data(WS_CLOSING_MESSAGE)
            await self._close_wait
        if self._closed:
            return False
        self._set_closed()
        try:
            await self._writer.close(code, message)
        except asyncio.CancelledError:
            self._close_code = WSCloseCode.ABNORMAL_CLOSURE
            self._response.close()
            raise
        except Exception as exc:
            self._close_code = WSCloseCode.ABNORMAL_CLOSURE
            self._exception = exc
            self._response.close()
            return True
        if self._close_code:
            self._response.close()
            return True
        while True:
            try:
                async with async_timeout.timeout(self._timeout.ws_close):  # type: ignore
                    msg: WSMessage = await self._reader.read()
            except asyncio.CancelledError:
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                self._response.close()
                raise
            except Exception as exc:
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                self._exception = exc
                self._response.close()
                return True
            if msg.type is WSMsgType.CLOSE:
                self._close_code = msg.data
                self._response.close()
                return True

    async def receive(self, timeout: Optional[float] = None) -> WSMessage:
        receive_timeout: Optional[float] = timeout or self._timeout.ws_receive
        while True:
            if self._waiting:
                raise RuntimeError('Concurrent call to receive() is not allowed')
            if self._closed:
                return WS_CLOSED_MESSAGE
            elif self._closing:
                await self.close()
                return WS_CLOSED_MESSAGE
            try:
                self._waiting = True
                try:
                    if receive_timeout:
                        async with async_timeout.timeout(receive_timeout):
                            msg: WSMessage = await self._reader.read()
                    else:
                        msg = await self._reader.read()
                    self._reset_heartbeat()
                finally:
                    self._waiting = False
                    if self._close_wait:
                        set_result(self._close_wait, None)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                raise
            except EofStream:
                self._close_code = WSCloseCode.OK
                await self.close()
                return WS_CLOSED_MESSAGE
            except ClientError:
                self._set_closed()
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                return WS_CLOSED_MESSAGE
            except WebSocketError as exc:
                self._close_code = exc.code
                await self.close(code=exc.code)
                return WSMessageError(data=exc)
            except Exception as exc:
                self._exception = exc
                self._set_closing()
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                await self.close()
                return WSMessageError(data=exc)
            if msg.type not in _INTERNAL_RECEIVE_TYPES:
                return msg
            if msg.type is WSMsgType.CLOSE:
                self._set_closing()
                self._close_code = msg.data
                if not self._closed and self._autoclose:
                    await self.close()
            elif msg.type is WSMsgType.CLOSING:
                self._set_closing()
            elif msg.type is WSMsgType.PING and self._autoping:
                await self.pong(msg.data)
                continue
            elif msg.type is WSMsgType.PONG and self._autoping:
                continue
            return msg

    async def receive_str(self, *, timeout: Optional[float] = None) -> str:
        msg: WSMessage = await self.receive(timeout)
        if msg.type is not WSMsgType.TEXT:
            raise WSMessageTypeError(f'Received message {msg.type}:{msg.data!r} is not WSMsgType.TEXT')
        return msg.data  # type: ignore

    async def receive_bytes(self, *, timeout: Optional[float] = None) -> Union[bytes, bytearray, memoryview]:
        msg: WSMessage = await self.receive(timeout)
        if msg.type is not WSMsgType.BINARY:
            raise WSMessageTypeError(f'Received message {msg.type}:{msg.data!r} is not WSMsgType.BINARY')
        return msg.data  # type: ignore

    async def receive_json(self, *, loads: JSONDecoder = DEFAULT_JSON_DECODER, timeout: Optional[float] = None) -> Any:
        data: str = await self.receive_str(timeout=timeout)
        return loads(data)

    def __aiter__(self) -> AsyncIterator[WSMessage]:
        return self

    async def __anext__(self) -> WSMessage:
        msg: WSMessage = await self.receive()
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            raise StopAsyncIteration
        return msg

    async def __aenter__(self) -> "ClientWebSocketResponse":
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        await self.close()