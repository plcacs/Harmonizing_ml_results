import asyncio
import base64
import binascii
import hashlib
import json
import sys
from typing import Any, Final, Iterable, Optional, Tuple, Union
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
THRESHOLD_CONNLOST_ACCESS = 5

@frozen_dataclass_decorator
class WebSocketReady:

    def __bool__(self):
        return self.ok

class WebSocketResponse(StreamResponse):
    _length_check = False
    _ws_protocol = None
    _writer = None
    _reader = None
    _closed = False
    _closing = False
    _conn_lost = 0
    _close_code = None
    _loop = None
    _waiting = False
    _close_wait = None
    _exception = None
    _heartbeat_when = 0.0
    _heartbeat_cb = None
    _pong_response_cb = None
    _ping_task = None

    def __init__(self, *, timeout=10.0, receive_timeout=None, autoclose=True, autoping=True, heartbeat=None, protocols=(), compress=True, max_msg_size=4 * 1024 * 1024, writer_limit=DEFAULT_LIMIT):
        super().__init__(status=101)
        self._protocols = protocols
        self._timeout = timeout
        self._receive_timeout = receive_timeout
        self._autoclose = autoclose
        self._autoping = autoping
        self._heartbeat = heartbeat
        if heartbeat is not None:
            self._pong_heartbeat = heartbeat / 2.0
        self._compress = compress
        self._max_msg_size = max_msg_size
        self._writer_limit = writer_limit

    def _cancel_heartbeat(self):
        self._cancel_pong_response_cb()
        if self._heartbeat_cb is not None:
            self._heartbeat_cb.cancel()
            self._heartbeat_cb = None
        if self._ping_task is not None:
            self._ping_task.cancel()
            self._ping_task = None

    def _cancel_pong_response_cb(self):
        if self._pong_response_cb is not None:
            self._pong_response_cb.cancel()
            self._pong_response_cb = None

    def _reset_heartbeat(self):
        if self._heartbeat is None:
            return
        self._cancel_pong_response_cb()
        req = self._req
        timeout_ceil_threshold = req._protocol._timeout_ceil_threshold if req is not None else 5
        loop = self._loop
        assert loop is not None
        now = loop.time()
        when = calculate_timeout_when(now, self._heartbeat, timeout_ceil_threshold)
        self._heartbeat_when = when
        if self._heartbeat_cb is None:
            self._heartbeat_cb = loop.call_at(when, self._send_heartbeat)

    def _send_heartbeat(self):
        self._heartbeat_cb = None
        loop = self._loop
        assert loop is not None and self._writer is not None
        now = loop.time()
        if now < self._heartbeat_when:
            self._heartbeat_cb = loop.call_at(self._heartbeat_when, self._send_heartbeat)
            return
        req = self._req
        timeout_ceil_threshold = req._protocol._timeout_ceil_threshold if req is not None else 5
        when = calculate_timeout_when(now, self._pong_heartbeat, timeout_ceil_threshold)
        self._cancel_pong_response_cb()
        self._pong_response_cb = loop.call_at(when, self._pong_not_received)
        coro = self._writer.send_frame(b'', WSMsgType.PING)
        if sys.version_info >= (3, 12):
            ping_task = asyncio.Task(coro, loop=loop, eager_start=True)
        else:
            ping_task = loop.create_task(coro)
        if not ping_task.done():
            self._ping_task = ping_task
            ping_task.add_done_callback(self._ping_task_done)
        else:
            self._ping_task_done(ping_task)

    def _ping_task_done(self, task):
        """Callback for when the ping task completes."""
        if not task.cancelled() and (exc := task.exception()):
            self._handle_ping_pong_exception(exc)
        self._ping_task = None

    def _pong_not_received(self):
        if self._req is not None and self._req.transport is not None:
            self._handle_ping_pong_exception(asyncio.TimeoutError(f'No PONG received after {self._pong_heartbeat} seconds'))

    def _handle_ping_pong_exception(self, exc):
        """Handle exceptions raised during ping/pong processing."""
        if self._closed:
            return
        self._set_closed()
        self._set_code_close_transport(WSCloseCode.ABNORMAL_CLOSURE)
        self._exception = exc
        if self._waiting and (not self._closing) and (self._reader is not None):
            self._reader.feed_data(WSMessageError(data=exc, extra=None))

    def _set_closed(self):
        """Set the connection to closed.

        Cancel any heartbeat timers and set the closed flag.
        """
        self._closed = True
        self._cancel_heartbeat()

    async def prepare(self, request):
        if self._payload_writer is not None:
            return self._payload_writer
        protocol, writer = self._pre_start(request)
        payload_writer = await super().prepare(request)
        assert payload_writer is not None
        self._post_start(request, protocol, writer)
        await payload_writer.drain()
        return payload_writer

    def _handshake(self, request):
        headers = request.headers
        if 'websocket' != headers.get(hdrs.UPGRADE, '').lower().strip():
            raise HTTPBadRequest(text='No WebSocket UPGRADE hdr: {}\n Can "Upgrade" only to "WebSocket".'.format(headers.get(hdrs.UPGRADE)))
        if 'upgrade' not in headers.get(hdrs.CONNECTION, '').lower():
            raise HTTPBadRequest(text='No CONNECTION upgrade hdr: {}'.format(headers.get(hdrs.CONNECTION)))
        protocol = None
        if hdrs.SEC_WEBSOCKET_PROTOCOL in headers:
            req_protocols = [str(proto.strip()) for proto in headers[hdrs.SEC_WEBSOCKET_PROTOCOL].split(',')]
            for proto in req_protocols:
                if proto in self._protocols:
                    protocol = proto
                    break
            else:
                ws_logger.warning('Client protocols %r don’t overlap server-known ones %r', req_protocols, self._protocols)
        version = headers.get(hdrs.SEC_WEBSOCKET_VERSION, '')
        if version not in ('13', '8', '7'):
            raise HTTPBadRequest(text=f'Unsupported version: {version}')
        key = headers.get(hdrs.SEC_WEBSOCKET_KEY)
        try:
            if not key or len(base64.b64decode(key)) != 16:
                raise HTTPBadRequest(text=f'Handshake error: {key!r}')
        except binascii.Error:
            raise HTTPBadRequest(text=f'Handshake error: {key!r}') from None
        accept_val = base64.b64encode(hashlib.sha1(key.encode() + WS_KEY).digest()).decode()
        response_headers = CIMultiDict({hdrs.UPGRADE: 'websocket', hdrs.CONNECTION: 'upgrade', hdrs.SEC_WEBSOCKET_ACCEPT: accept_val})
        notakeover = False
        compress = 0
        if self._compress:
            extensions = headers.get(hdrs.SEC_WEBSOCKET_EXTENSIONS)
            compress, notakeover = ws_ext_parse(extensions, isserver=True)
            if compress:
                enabledext = ws_ext_gen(compress=compress, isserver=True, server_notakeover=notakeover)
                response_headers[hdrs.SEC_WEBSOCKET_EXTENSIONS] = enabledext
        if protocol:
            response_headers[hdrs.SEC_WEBSOCKET_PROTOCOL] = protocol
        return (response_headers, protocol, compress, notakeover)

    def _pre_start(self, request):
        self._loop = request._loop
        headers, protocol, compress, notakeover = self._handshake(request)
        self.set_status(101)
        self.headers.update(headers)
        self.force_close()
        self._compress = compress
        transport = request._protocol.transport
        assert transport is not None
        writer = WebSocketWriter(request._protocol, transport, compress=compress, notakeover=notakeover, limit=self._writer_limit)
        return (protocol, writer)

    def _post_start(self, request, protocol, writer):
        self._ws_protocol = protocol
        self._writer = writer
        self._reset_heartbeat()
        loop = self._loop
        assert loop is not None
        self._reader = WebSocketDataQueue(request._protocol, 2 ** 16, loop=loop)
        request.protocol.set_parser(WebSocketReader(self._reader, self._max_msg_size, compress=bool(self._compress)))
        request.protocol.keep_alive(False)

    def can_prepare(self, request):
        if self._writer is not None:
            raise RuntimeError('Already started')
        try:
            _, protocol, _, _ = self._handshake(request)
        except HTTPException:
            return WebSocketReady(False, None)
        else:
            return WebSocketReady(True, protocol)

    @property
    def closed(self):
        return self._closed

    @property
    def close_code(self):
        return self._close_code

    @property
    def ws_protocol(self):
        return self._ws_protocol

    @property
    def compress(self):
        return self._compress

    def get_extra_info(self, name, default=None):
        """Get optional transport information.

        If no value associated with ``name`` is found, ``default`` is returned.
        """
        writer = self._writer
        if writer is None:
            return default
        return writer.transport.get_extra_info(name, default)

    def exception(self):
        return self._exception

    async def ping(self, message=b''):
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        await self._writer.send_frame(message, WSMsgType.PING)

    async def pong(self, message=b''):
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        await self._writer.send_frame(message, WSMsgType.PONG)

    async def send_frame(self, message, opcode, compress=None):
        """Send a frame over the websocket."""
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        await self._writer.send_frame(message, opcode, compress)

    async def send_str(self, data, compress=None):
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        if not isinstance(data, str):
            raise TypeError('data argument must be str (%r)' % type(data))
        await self._writer.send_frame(data.encode('utf-8'), WSMsgType.TEXT, compress=compress)

    async def send_bytes(self, data, compress=None):
        if self._writer is None:
            raise RuntimeError('Call .prepare() first')
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('data argument must be byte-ish (%r)' % type(data))
        await self._writer.send_frame(data, WSMsgType.BINARY, compress=compress)

    async def send_json(self, data, compress=None, *, dumps=json.dumps):
        await self.send_str(dumps(data), compress=compress)

    async def write_eof(self):
        if self._eof_sent:
            return
        if self._payload_writer is None:
            raise RuntimeError('Response has not been started')
        await self.close()
        self._eof_sent = True

    async def close(self, *, code=WSCloseCode.OK, message=b'', drain=True):
        """Close websocket connection."""
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

    def _set_closing(self, code):
        """Set the close code and mark the connection as closing."""
        self._closing = True
        self._close_code = code
        self._cancel_heartbeat()

    def _set_code_close_transport(self, code):
        """Set the close code and close the transport."""
        self._close_code = code
        self._close_transport()

    def _close_transport(self):
        """Close the transport."""
        if self._req is not None and self._req.transport is not None:
            self._req.transport.close()

    async def receive(self, timeout=None):
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
                        async with async_timeout.timeout(receive_timeout):
                            msg = await self._reader.read()
                    else:
                        msg = await self._reader.read()
                    self._reset_heartbeat()
                finally:
                    self._waiting = False
                    if self._close_wait:
                        set_result(self._close_wait, None)
            except asyncio.TimeoutError:
                raise
            except EofStream:
                self._close_code = WSCloseCode.OK
                await self.close()
                return WS_CLOSED_MESSAGE
            except WebSocketError as exc:
                self._close_code = exc.code
                await self.close(code=exc.code)
                return WSMessageError(data=exc)
            except Exception as exc:
                self._exception = exc
                self._set_closing(WSCloseCode.ABNORMAL_CLOSURE)
                await self.close()
                return WSMessageError(data=exc)
            if msg.type not in _INTERNAL_RECEIVE_TYPES:
                return msg
            if msg.type is WSMsgType.CLOSE:
                self._set_closing(msg.data)
                if not self._closed and self._autoclose:
                    await self.close(drain=False)
            elif msg.type is WSMsgType.CLOSING:
                self._set_closing(WSCloseCode.OK)
            elif msg.type is WSMsgType.PING and self._autoping:
                await self.pong(msg.data)
                continue
            elif msg.type is WSMsgType.PONG and self._autoping:
                continue
            return msg

    async def receive_str(self, *, timeout=None):
        msg = await self.receive(timeout)
        if msg.type is not WSMsgType.TEXT:
            raise WSMessageTypeError(f'Received message {msg.type}:{msg.data!r} is not WSMsgType.TEXT')
        return msg.data

    async def receive_bytes(self, *, timeout=None):
        msg = await self.receive(timeout)
        if msg.type is not WSMsgType.BINARY:
            raise WSMessageTypeError(f'Received message {msg.type}:{msg.data!r} is not WSMsgType.BINARY')
        return msg.data

    async def receive_json(self, *, loads=json.loads, timeout=None):
        data = await self.receive_str(timeout=timeout)
        return loads(data)

    async def write(self, data):
        raise RuntimeError('Cannot call .write() for websocket')

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self.receive()
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            raise StopAsyncIteration
        return msg

    def _cancel(self, exc):
        self._closing = True
        self._cancel_heartbeat()
        if self._reader is not None:
            set_exception(self._reader, exc)