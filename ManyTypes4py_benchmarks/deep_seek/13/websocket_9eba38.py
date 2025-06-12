"""Implementation of the WebSocket protocol."""
import abc
import asyncio
import base64
import hashlib
import os
import sys
import struct
import tornado
from urllib.parse import urlparse
import warnings
import zlib
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen, httpclient, httputil
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.iostream import StreamClosedError, IOStream
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado import simple_httpclient
from tornado.queues import Queue
from tornado.tcpclient import TCPClient
from tornado.util import _websocket_mask
from typing import TYPE_CHECKING, cast, Any, Optional, Dict, Union, List, Awaitable, Callable, Tuple, Type, TypeVar, Generic
from types import TracebackType
import typing

if TYPE_CHECKING:
    from typing_extensions import Protocol

    class _Compressor(Protocol):
        def compress(self, data: bytes) -> bytes: ...
        def flush(self, mode: int) -> bytes: ...

    class _Decompressor(Protocol):
        unconsumed_tail: bytes
        def decompress(self, data: bytes, max_length: int) -> bytes: ...

    class _WebSocketDelegate(Protocol):
        def on_ws_connection_close(self, close_code: Optional[int] = None, close_reason: Optional[str] = None) -> None: ...
        def on_message(self, message: Union[str, bytes]) -> Optional[Awaitable[None]]: ...
        def on_ping(self, data: bytes) -> None: ...
        def on_pong(self, data: bytes) -> None: ...
        def log_exception(self, typ: Type[BaseException], value: BaseException, tb: TracebackType) -> None: ...

_default_max_message_size: int = 10 * 1024 * 1024

class WebSocketError(Exception):
    pass

class WebSocketClosedError(WebSocketError):
    pass

class _DecompressTooLargeError(Exception):
    pass

class _WebSocketParams:
    def __init__(
        self,
        ping_interval: Optional[float] = None,
        ping_timeout: Optional[float] = None,
        max_message_size: int = _default_max_message_size,
        compression_options: Optional[Dict[str, Any]] = None
    ) -> None:
        self.ping_interval: Optional[float] = ping_interval
        self.ping_timeout: Optional[float] = ping_timeout
        self.max_message_size: int = max_message_size
        self.compression_options: Optional[Dict[str, Any]] = compression_options

class WebSocketHandler(tornado.web.RequestHandler):
    def __init__(
        self,
        application: tornado.web.Application,
        request: httputil.HTTPServerRequest,
        **kwargs: Any
    ) -> None:
        super().__init__(application, request, **kwargs)
        self.ws_connection: Optional[WebSocketProtocol] = None
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self._on_close_called: bool = False
        self.open_args: Tuple[Any, ...] = ()
        self.open_kwargs: Dict[str, Any] = {}

    async def get(self, *args: Any, **kwargs: Any) -> None:
        self.open_args = args
        self.open_kwargs = kwargs
        if self.request.headers.get('Upgrade', '').lower() != 'websocket':
            self.set_status(400)
            log_msg = 'Can "Upgrade" only to "WebSocket".'
            self.finish(log_msg)
            gen_log.debug(log_msg)
            return
        headers = self.request.headers
        connection = [s.strip().lower() for s in headers.get('Connection', '').split(',')]
        if 'upgrade' not in connection:
            self.set_status(400)
            log_msg = '"Connection" must be "Upgrade".'
            self.finish(log_msg)
            gen_log.debug(log_msg)
            return
        origin: Optional[str]
        if 'Origin' in self.request.headers:
            origin = self.request.headers.get('Origin')
        else:
            origin = self.request.headers.get('Sec-Websocket-Origin', None)
        if origin is not None and (not self.check_origin(origin)):
            self.set_status(403)
            log_msg = 'Cross origin websockets not allowed'
            self.finish(log_msg)
            gen_log.debug(log_msg)
            return
        self.ws_connection = self.get_websocket_protocol()
        if self.ws_connection:
            await self.ws_connection.accept_connection(self)
        else:
            self.set_status(426, 'Upgrade Required')
            self.set_header('Sec-WebSocket-Version', '7, 8, 13')

    @property
    def ping_interval(self) -> Optional[float]:
        return self.settings.get('websocket_ping_interval', None)

    @property
    def ping_timeout(self) -> Optional[float]:
        return self.settings.get('websocket_ping_timeout', None)

    @property
    def max_message_size(self) -> int:
        return self.settings.get('websocket_max_message_size', _default_max_message_size)

    def write_message(self, message: Union[str, bytes, Dict[str, Any]], binary: bool = False) -> Future[None]:
        if self.ws_connection is None or self.ws_connection.is_closing():
            raise WebSocketClosedError()
        if isinstance(message, dict):
            message = tornado.escape.json_encode(message)
        message = tornado.escape.utf8(message)
        return self.ws_connection.write_message(message, binary=binary)

    def select_subprotocol(self, subprotocols: List[str]) -> Optional[str]:
        return None

    @property
    def selected_subprotocol(self) -> Optional[str]:
        assert self.ws_connection is not None
        return self.ws_connection.selected_subprotocol

    def get_compression_options(self) -> Optional[Dict[str, Any]]:
        return None

    def open(self, *args: Any, **kwargs: Any) -> Optional[Awaitable[None]]:
        pass

    def on_message(self, message: Union[str, bytes]) -> Optional[Awaitable[None]]:
        raise NotImplementedError()

    def ping(self, data: bytes = b'') -> None:
        data = utf8(data)
        if self.ws_connection is None or self.ws_connection.is_closing():
            raise WebSocketClosedError()
        self.ws_connection.write_ping(data)

    def on_pong(self, data: bytes) -> None:
        pass

    def on_ping(self, data: bytes) -> None:
        pass

    def on_close(self) -> None:
        pass

    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        if self.ws_connection:
            self.ws_connection.close(code, reason)
            self.ws_connection = None

    def check_origin(self, origin: str) -> bool:
        parsed_origin = urlparse(origin)
        origin = parsed_origin.netloc
        origin = origin.lower()
        host = self.request.headers.get('Host')
        return origin == host

    def set_nodelay(self, value: bool) -> None:
        assert self.ws_connection is not None
        self.ws_connection.set_nodelay(value)

    def on_connection_close(self) -> None:
        if self.ws_connection:
            self.ws_connection.on_connection_close()
            self.ws_connection = None
        if not self._on_close_called:
            self._on_close_called = True
            self.on_close()
            self._break_cycles()

    def on_ws_connection_close(self, close_code: Optional[int] = None, close_reason: Optional[str] = None) -> None:
        self.close_code = close_code
        self.close_reason = close_reason
        self.on_connection_close()

    def _break_cycles(self) -> None:
        if self.get_status() != 101 or self._on_close_called:
            super()._break_cycles()

    def get_websocket_protocol(self) -> Optional['WebSocketProtocol']:
        websocket_version = self.request.headers.get('Sec-WebSocket-Version')
        if websocket_version in ('7', '8', '13'):
            params = _WebSocketParams(
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                max_message_size=self.max_message_size,
                compression_options=self.get_compression_options()
            )
            return WebSocketProtocol13(self, False, params)
        return None

    def _detach_stream(self) -> IOStream:
        for method in ['write', 'redirect', 'set_header', 'set_cookie', 'set_status', 'flush', 'finish']:
            setattr(self, method, _raise_not_supported_for_websockets)
        return self.detach()

def _raise_not_supported_for_websockets(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError('Method not supported for Web Sockets')

class WebSocketProtocol(abc.ABC):
    def __init__(self, handler: WebSocketHandler) -> None:
        self.handler: WebSocketHandler = handler
        self.stream: Optional[IOStream] = None
        self.client_terminated: bool = False
        self.server_terminated: bool = False

    def _run_callback(
        self,
        callback: Callable[..., Optional[Awaitable[None]]],
        *args: Any,
        **kwargs: Any
    ) -> Optional[Future[None]]:
        try:
            result = callback(*args, **kwargs)
        except Exception:
            self.handler.log_exception(*sys.exc_info())
            self._abort()
            return None
        else:
            if result is not None:
                result = gen.convert_yielded(result)
                assert self.stream is not None
                self.stream.io_loop.add_future(result, lambda f: f.result())
            return result

    def on_connection_close(self) -> None:
        self._abort()

    def _abort(self) -> None:
        self.client_terminated = True
        self.server_terminated = True
        if self.stream is not None:
            self.stream.close()
        self.close()

    @abc.abstractmethod
    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None: ...

    @abc.abstractmethod
    def is_closing(self) -> bool: ...

    @abc.abstractmethod
    async def accept_connection(self, handler: WebSocketHandler) -> None: ...

    @abc.abstractmethod
    def write_message(self, message: bytes, binary: bool = False) -> Future[None]: ...

    @property
    @abc.abstractmethod
    def selected_subprotocol(self) -> Optional[str]: ...

    @abc.abstractmethod
    def write_ping(self, data: bytes) -> None: ...

    @abc.abstractmethod
    def _process_server_headers(self, key: str, headers: httputil.HTTPHeaders) -> None: ...

    @abc.abstractmethod
    def start_pinging(self) -> None: ...

    @abc.abstractmethod
    async def _receive_frame_loop(self) -> None: ...

    @abc.abstractmethod
    def set_nodelay(self, x: bool) -> None: ...

class _PerMessageDeflateCompressor:
    def __init__(
        self,
        persistent: bool,
        max_wbits: Optional[int],
        compression_options: Optional[Dict[str, Any]] = None
    ) -> None:
        if max_wbits is None:
            max_wbits = zlib.MAX_WBITS
        if not 8 <= max_wbits <= zlib.MAX_WBITS:
            raise ValueError('Invalid max_wbits value %r; allowed range 8-%d', max_wbits, zlib.MAX_WBITS)
        self._max_wbits: int = max_wbits
        if compression_options is None or 'compression_level' not in compression_options:
            self._compression_level: int = tornado.web.GZipContentEncoding.GZIP_LEVEL
        else:
            self._compression_level = compression_options['compression_level']
        if compression_options is None or 'mem_level' not in compression_options:
            self._mem_level: int = 8
        else:
            self._mem_level = compression_options['mem_level']
        if persistent:
            self._compressor: Optional[zlib._Compress] = self._create_compressor()
        else:
            self._compressor = None

    def _create_compressor(self) -> zlib._Compress:
        return zlib.compressobj(self._compression_level, zlib.DEFLATED, -self._max_wbits, self._mem_level)

    def compress(self, data: bytes) -> bytes:
        compressor = self._compressor or self._create_compressor()
        data = compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH)
        assert data.endswith(b'\x00\x00\xff\xff')
        return data[:-4]

class _PerMessageDeflateDecompressor:
    def __init__(
        self,
        persistent: bool,
        max_wbits: Optional[int],
        max_message_size: int,
        compression_options: Optional[Dict[str, Any]] = None
    ) -> None:
        self._max_message_size: int = max_message_size
        if max_wbits is None:
            max_wbits = zlib.MAX_WBITS
        if not 8 <= max_wbits <= zlib.MAX_WBITS:
            raise ValueError('Invalid max_wbits value %r; allowed range 8-%d', max_wbits, zlib.MAX_WBITS)
        self._max_wbits: int = max_wbits
        if persistent:
            self._decompressor: Optional[zlib._Decompress] = self._create_decompressor()
        else:
            self._decompressor = None

    def _create_decompressor(self) -> zlib._Decompress:
        return zlib.decompressobj(-self._max_wbits)

    def decompress(self, data: bytes) -> bytes:
        decompressor = self._decompressor or self._create_decompressor()
        result = decompressor.decompress(data + b'\x00\x00\xff\xff', self._max_message_size)
        if decompressor.unconsumed_tail:
            raise _DecompressTooLargeError()
        return result

class WebSocketProtocol13(WebSocketProtocol):
    FIN: int = 128
    RSV1: int = 64
    RSV2: int = 32
    RSV3: int = 16
    RSV_MASK: int = RSV1 | RSV2 | RSV3
    OPCODE_MASK: int = 15
    stream: Optional[IOStream] = None

    def __init__(
        self,
        handler: WebSocketHandler,
        mask_outgoing: bool,
        params: _WebSocketParams
    ) -> None:
        super().__init__(handler)
        self.mask_outgoing: bool = mask_outgoing
        self.params: _WebSocketParams = params
        self._final_frame: bool = False
        self._frame_opcode: Optional[int] = None
        self._masked_frame: Optional[bool] = None
        self._frame_mask: Optional[bytes] = None
        self._frame_length: Optional[int] = None
        self._fragmented_message_buffer: Optional[bytearray] = None
        self._fragmented_message_opcode: Optional[int] = None
        self._waiting: Optional[object] = None
        self._compression_options: Optional[Dict[str, Any]] = params.compression_options
        self._decompressor: Optional[_PerMessageDeflateDecompressor] = None
        self._compressor: Optional[_PerMessageDeflateCompressor] = None
        self._frame_compressed: bool = False
        self._message_bytes_in: int = 0
        self._message_bytes_out: int = 0
        self._wire_bytes_in: int = 0
        self._wire_bytes_out: int = 0
        self.ping_callback: Optional[PeriodicCallback] = None
        self.last_ping: float = 0.0
        self.last_pong: float = 0.0
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self._selected_subprotocol: Optional[str] = None

    @property
    def selected_subprotocol(self) -> Optional[str]:
        return self._selected_subprotocol

    @selected_subprotocol.setter
    def selected_subprotocol(self, value: Optional[str]) -> None:
        self._selected_subprotocol = value

    async def accept_connection(self, handler: WebSocketHandler) -> None:
        try:
            self._handle_websocket_headers(handler)
        except ValueError:
            handler.set_status(400)
            log_msg = 'Missing/Invalid WebSocket headers'
            handler.finish(log_msg)
            gen_log.debug(log_msg)
            return
        try:
            await self._accept_connection(handler)
        except asyncio.CancelledError:
            self._abort()
            return
        except ValueError:
            gen_log.debug('Malformed WebSocket request received', exc_info=True)
            self._abort()
            return

    def _handle_websocket_headers(self, handler: WebSocketHandler) -> None:
        fields = ('Host', 'Sec-Websocket-Key', 'Sec-Websocket-Version')
        if not all(map(lambda f: handler.request.headers.get(f), fields)):
            raise ValueError('Missing/Invalid WebSocket headers')

    @staticmethod
    def compute_accept_value(key: str) -> str:
        sha1 = hashlib.sha1()
        sha1.update(utf8(key))
        sha1.update(b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
        return native_str(base64.b64encode(sha1.digest()))

    def _challenge_response(self, handler: WebSocketHandler) -> str:
        return WebSocketProtocol13.com