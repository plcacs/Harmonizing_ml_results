#!/usr/bin/env python3
"""
Implementation of the WebSocket protocol.

`WebSockets <http://dev.w3.org/html5/websockets/>`_ allow for bidirectional
communication between the browser and server. WebSockets are supported in the
current versions of all major browsers.

This module implements the final version of the WebSocket protocol as
defined in `RFC 6455 <http://tools.ietf.org/html/rfc6455>`_.

.. versionchanged:: 4.0
   Removed support for the draft 76 protocol version.
"""

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
from typing import TYPE_CHECKING, cast, Any, Optional, Dict, Union, List, Awaitable, Callable, Tuple, Type

_default_max_message_size: int = 10 * 1024 * 1024

class WebSocketError(Exception):
    pass

class WebSocketClosedError(WebSocketError):
    """Raised by operations on a closed connection.

    .. versionadded:: 3.2
    """
    pass

class _DecompressTooLargeError(Exception):
    pass

class _WebSocketParams:
    def __init__(self, 
                 ping_interval: Optional[float] = None, 
                 ping_timeout: Optional[float] = None, 
                 max_message_size: int = _default_max_message_size, 
                 compression_options: Optional[Dict[str, Any]] = None) -> None:
        self.ping_interval: Optional[float] = ping_interval
        self.ping_timeout: Optional[float] = ping_timeout
        self.max_message_size: int = max_message_size
        self.compression_options: Optional[Dict[str, Any]] = compression_options

class WebSocketHandler(tornado.web.RequestHandler):
    def __init__(self, application: tornado.web.Application, request: httputil.HTTPServerRequest, **kwargs: Any) -> None:
        super().__init__(application, request, **kwargs)
        self.ws_connection: Optional[WebSocketProtocol] = None
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self._on_close_called: bool = False

    async def get(self, *args: Any, **kwargs: Any) -> None:
        self.open_args = args
        self.open_kwargs = kwargs
        if self.request.headers.get('Upgrade', '').lower() != 'websocket':
            self.set_status(400)
            log_msg: str = 'Can "Upgrade" only to "WebSocket".'
            self.finish(log_msg)
            gen_log.debug(log_msg)
            return
        headers = self.request.headers
        connection = map(lambda s: s.strip().lower(), headers.get('Connection', '').split(','))
        if 'upgrade' not in connection:
            self.set_status(400)
            log_msg = '"Connection" must be "Upgrade".'
            self.finish(log_msg)
            gen_log.debug(log_msg)
            return
        if 'Origin' in self.request.headers:
            origin: Optional[str] = self.request.headers.get('Origin')
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
        """The interval for websocket keep-alive pings.

        Set websocket_ping_interval = 0 to disable pings.
        """
        return self.settings.get('websocket_ping_interval', None)

    @property
    def ping_timeout(self) -> Optional[float]:
        """If no ping is received in this many seconds,
        close the websocket connection (VPNs, etc. can fail to cleanly close ws connections).
        Default is max of 3 pings or 30 seconds.
        """
        return self.settings.get('websocket_ping_timeout', None)

    @property
    def max_message_size(self) -> int:
        """Maximum allowed message size.

        If the remote peer sends a message larger than this, the connection
        will be closed.

        Default is 10MiB.
        """
        return self.settings.get('websocket_max_message_size', _default_max_message_size)

    def write_message(self, message: Union[str, Dict[str, Any]], binary: bool = False) -> asyncio.Future[Any]:
        """Sends the given message to the client of this Web Socket."""
        if self.ws_connection is None or self.ws_connection.is_closing():
            raise WebSocketClosedError()
        if isinstance(message, dict):
            message = tornado.escape.json_encode(message)
        return self.ws_connection.write_message(message, binary=binary)

    def select_subprotocol(self, subprotocols: List[str]) -> Optional[str]:
        """Override to implement subprotocol negotiation."""
        return None

    @property
    def selected_subprotocol(self) -> Optional[str]:
        """The subprotocol returned by `select_subprotocol`."""
        assert self.ws_connection is not None
        return self.ws_connection.selected_subprotocol

    def get_compression_options(self) -> Optional[Dict[str, Any]]:
        """Override to return compression options for the connection."""
        return None

    def open(self, *args: Any, **kwargs: Any) -> Any:
        """Invoked when a new WebSocket is opened."""
        pass

    def on_message(self, message: Union[str, bytes]) -> Any:
        """Handle incoming messages on the WebSocket."""
        raise NotImplementedError

    def ping(self, data: Union[str, bytes] = b'') -> None:
        """Send ping frame to the remote end."""
        data = utf8(data)
        if self.ws_connection is None or self.ws_connection.is_closing():
            raise WebSocketClosedError()
        self.ws_connection.write_ping(data)

    def on_pong(self, data: bytes) -> None:
        """Invoked when the response to a ping frame is received."""
        pass

    def on_ping(self, data: bytes) -> None:
        """Invoked when a ping frame is received."""
        pass

    def on_close(self) -> None:
        """Invoked when the WebSocket is closed."""
        pass

    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        """Closes this Web Socket."""
        if self.ws_connection:
            self.ws_connection.close(code, reason)
            self.ws_connection = None

    def check_origin(self, origin: str) -> bool:
        """Override to enable support for allowing alternate origins."""
        parsed_origin = urlparse(origin)
        origin = parsed_origin.netloc
        origin = origin.lower()
        host: Optional[str] = self.request.headers.get('Host')
        return origin == host

    def set_nodelay(self, value: bool) -> None:
        """Set the no-delay flag for this stream."""
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

    def get_websocket_protocol(self) -> Optional["WebSocketProtocol"]:
        websocket_version: Optional[str] = self.request.headers.get('Sec-WebSocket-Version')
        if websocket_version in ('7', '8', '13'):
            params = _WebSocketParams(ping_interval=self.ping_interval, 
                                      ping_timeout=self.ping_timeout, 
                                      max_message_size=self.max_message_size, 
                                      compression_options=self.get_compression_options())
            from __main__ import WebSocketProtocol13  # type: ignore
            return WebSocketProtocol13(self, False, params)
        return None

    def _detach_stream(self) -> IOStream:
        for method in ['write', 'redirect', 'set_header', 'set_cookie', 'set_status', 'flush', 'finish']:
            setattr(self, method, _raise_not_supported_for_websockets)
        return self.detach()

def _raise_not_supported_for_websockets(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError('Method not supported for Web Sockets')

class WebSocketProtocol(abc.ABC):
    def __init__(self, handler: WebSocketHandler) -> None:
        self.handler: WebSocketHandler = handler
        self.stream: Optional[IOStream] = None
        self.client_terminated: bool = False
        self.server_terminated: bool = False

    def _run_callback(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Optional[Awaitable[Any]]:
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
    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_closing(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    async def accept_connection(self, handler: WebSocketHandler) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def write_message(self, message: Any, binary: bool = False) -> Awaitable[None]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def selected_subprotocol(self) -> Optional[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def write_ping(self, data: bytes) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def _process_server_headers(self, key: str, headers: httputil.HTTPHeaders) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def start_pinging(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _receive_frame_loop(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_nodelay(self, x: bool) -> None:
        raise NotImplementedError()

class _PerMessageDeflateCompressor:
    def __init__(self, persistent: bool, max_wbits: Optional[int], compression_options: Optional[Dict[str, Any]] = None) -> None:
        if max_wbits is None:
            max_wbits = zlib.MAX_WBITS
        if not 8 <= max_wbits <= zlib.MAX_WBITS:
            raise ValueError('Invalid max_wbits value %r; allowed range 8-%d' % (max_wbits, zlib.MAX_WBITS))
        self._max_wbits: int = max_wbits
        if compression_options is None or 'compression_level' not in compression_options:
            self._compression_level: int = tornado.web.GZipContentEncoding.GZIP_LEVEL  # type: ignore
        else:
            self._compression_level = compression_options['compression_level']
        if compression_options is None or 'mem_level' not in compression_options:
            self._mem_level: int = 8
        else:
            self._mem_level = compression_options['mem_level']
        if persistent:
            self._compressor = self._create_compressor()
        else:
            self._compressor = None

    def _create_compressor(self) -> Any:
        return zlib.compressobj(self._compression_level, zlib.DEFLATED, -self._max_wbits, self._mem_level)

    def compress(self, data: bytes) -> bytes:
        compressor = self._compressor or self._create_compressor()
        data = compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH)
        assert data.endswith(b'\x00\x00\xff\xff')
        return data[:-4]

class _PerMessageDeflateDecompressor:
    def __init__(self, persistent: bool, max_wbits: Optional[int], max_message_size: int, compression_options: Optional[Dict[str, Any]] = None) -> None:
        self._max_message_size: int = max_message_size
        if max_wbits is None:
            max_wbits = zlib.MAX_WBITS
        if not 8 <= max_wbits <= zlib.MAX_WBITS:
            raise ValueError('Invalid max_wbits value %r; allowed range 8-%d' % (max_wbits, zlib.MAX_WBITS))
        self._max_wbits: int = max_wbits
        if persistent:
            self._decompressor = self._create_decompressor()
        else:
            self._decompressor = None

    def _create_decompressor(self) -> Any:
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
    OPCODE_MASK: int = 15
    RSV_MASK: int = RSV1 | RSV2 | RSV3
    stream: Optional[IOStream] = None

    def __init__(self, handler: WebSocketHandler, mask_outgoing: bool, params: _WebSocketParams) -> None:
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
        self._waiting: Optional[Any] = None
        self._compression_options: Optional[Dict[str, Any]] = params.compression_options
        self._decompressor: Optional[_PerMessageDeflateDecompressor] = None
        self._compressor: Optional[_PerMessageDeflateCompressor] = None
        self._frame_compressed: Optional[bool] = None
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
            log_msg: str = 'Missing/Invalid WebSocket headers'
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
        fields: Tuple[str, str, str] = ('Host', 'Sec-Websocket-Key', 'Sec-Websocket-Version')
        if not all(map(lambda f: handler.request.headers.get(f), fields)):
            raise ValueError('Missing/Invalid WebSocket headers')

    @staticmethod
    def compute_accept_value(key: str) -> str:
        sha1 = hashlib.sha1()
        sha1.update(utf8(key))
        sha1.update(b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
        return native_str(base64.b64encode(sha1.digest()))

    def _challenge_response(self, handler: WebSocketHandler) -> str:
        return WebSocketProtocol13.compute_accept_value(cast(str, handler.request.headers.get('Sec-Websocket-Key')))

    async def _accept_connection(self, handler: WebSocketHandler) -> None:
        subprotocol_header: Optional[str] = handler.request.headers.get('Sec-WebSocket-Protocol')
        if subprotocol_header:
            subprotocols: List[str] = [s.strip() for s in subprotocol_header.split(',')]
        else:
            subprotocols = []
        self.selected_subprotocol = handler.select_subprotocol(subprotocols)
        if self.selected_subprotocol:
            assert self.selected_subprotocol in subprotocols
            handler.set_header('Sec-WebSocket-Protocol', self.selected_subprotocol)
        extensions = self._parse_extensions_header(handler.request.headers)
        for ext in extensions:
            if ext[0] == 'permessage-deflate' and self._compression_options is not None:
                self._create_compressors('server', ext[1], self._compression_options)
                if 'client_max_window_bits' in ext[1] and ext[1]['client_max_window_bits'] is None:
                    del ext[1]['client_max_window_bits']
                handler.set_header('Sec-WebSocket-Extensions', httputil._encode_header('permessage-deflate', ext[1]))
                break
        handler.clear_header('Content-Type')
        handler.set_status(101)
        handler.set_header('Upgrade', 'websocket')
        handler.set_header('Connection', 'Upgrade')
        handler.set_header('Sec-WebSocket-Accept', self._challenge_response(handler))
        handler.finish()
        self.stream = handler._detach_stream()
        self.start_pinging()
        try:
            open_result = handler.open(*handler.open_args, **handler.open_kwargs)
            if open_result is not None:
                await open_result
        except Exception:
            handler.log_exception(*sys.exc_info())
            self._abort()
            return
        await self._receive_frame_loop()

    def _parse_extensions_header(self, headers: httputil.HTTPHeaders) -> List[Tuple[str, Dict[str, Optional[str]]]]:
        extensions_header: str = headers.get('Sec-WebSocket-Extensions', '')
        if extensions_header:
            return [httputil._parse_header(e.strip()) for e in extensions_header.split(',')]
        return []

    def _process_server_headers(self, key: str, headers: httputil.HTTPHeaders) -> None:
        assert headers['Upgrade'].lower() == 'websocket'
        assert headers['Connection'].lower() == 'upgrade'
        accept: str = self.compute_accept_value(key)
        assert headers['Sec-Websocket-Accept'] == accept
        extensions = self._parse_extensions_header(headers)
        for ext in extensions:
            if ext[0] == 'permessage-deflate' and self._compression_options is not None:
                self._create_compressors('client', ext[1])
            else:
                raise ValueError('unsupported extension %r' % (ext,))
        self.selected_subprotocol = headers.get('Sec-WebSocket-Protocol', None)

    def _get_compressor_options(self, side: str, agreed_parameters: Dict[str, Optional[str]], compression_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        options: Dict[str, Any] = dict(persistent=side + '_no_context_takeover' not in agreed_parameters)
        wbits_header: Optional[str] = agreed_parameters.get(side + '_max_window_bits', None)
        if wbits_header is None:
            options['max_wbits'] = zlib.MAX_WBITS
        else:
            options['max_wbits'] = int(wbits_header)
        options['compression_options'] = compression_options
        return options

    def _create_compressors(self, side: str, agreed_parameters: Dict[str, Optional[str]], compression_options: Optional[Dict[str, Any]] = None) -> None:
        allowed_keys = {'server_no_context_takeover', 'client_no_context_takeover', 'server_max_window_bits', 'client_max_window_bits'}
        for key in agreed_parameters:
            if key not in allowed_keys:
                raise ValueError('unsupported compression parameter %r' % key)
        other_side: str = 'client' if side == 'server' else 'server'
        self._compressor = _PerMessageDeflateCompressor(**self._get_compressor_options(side, agreed_parameters, compression_options))
        self._decompressor = _PerMessageDeflateDecompressor(max_message_size=self.params.max_message_size, **self._get_compressor_options(other_side, agreed_parameters, compression_options))

    def _write_frame(self, fin: bool, opcode: int, data: bytes, flags: int = 0) -> Awaitable[None]:
        data_len: int = len(data)
        if opcode & 8:
            if not fin:
                raise ValueError('control frames may not be fragmented')
            if data_len > 125:
                raise ValueError('control frame payloads may not exceed 125 bytes')
        finbit: int = self.FIN if fin else 0
        frame: bytes = struct.pack('B', finbit | opcode | flags)
        mask_bit: int = 128 if self.mask_outgoing else 0
        if data_len < 126:
            frame += struct.pack('B', data_len | mask_bit)
        elif data_len <= 65535:
            frame += struct.pack('!BH', 126 | mask_bit, data_len)
        else:
            frame += struct.pack('!BQ', 127 | mask_bit, data_len)
        if self.mask_outgoing:
            mask: bytes = os.urandom(4)
            data = mask + _websocket_mask(mask, data)
        frame += data
        self._wire_bytes_out += len(frame)
        return self.stream.write(frame)  # type: ignore

    def write_message(self, message: Union[str, Dict[str, Any]], binary: bool = False) -> asyncio.Future[Any]:
        if binary:
            opcode: int = 2
        else:
            opcode = 1
        if isinstance(message, dict):
            message = tornado.escape.json_encode(message)
        message_bytes: bytes = tornado.escape.utf8(message)
        assert isinstance(message_bytes, bytes)
        self._message_bytes_out += len(message_bytes)
        flags: int = 0
        if self._compressor:
            message_bytes = self._compressor.compress(message_bytes)
            flags |= self.RSV1
        try:
            fut: Awaitable[None] = self._write_frame(True, opcode, message_bytes, flags=flags)
        except StreamClosedError:
            raise WebSocketClosedError()

        async def wrapper() -> None:
            try:
                await fut
            except StreamClosedError:
                raise WebSocketClosedError()
        return asyncio.ensure_future(wrapper())

    def write_ping(self, data: bytes) -> None:
        assert isinstance(data, bytes)
        self._write_frame(True, 9, data)

    async def _receive_frame_loop(self) -> None:
        try:
            while not self.client_terminated:
                await self._receive_frame()
        except StreamClosedError:
            self._abort()
        self.handler.on_ws_connection_close(self.close_code, self.close_reason)

    async def _read_bytes(self, n: int) -> bytes:
        data: bytes = await self.stream.read_bytes(n)  # type: ignore
        self._wire_bytes_in += n
        return data

    async def _receive_frame(self) -> None:
        data: bytes = await self._read_bytes(2)
        header, mask_payloadlen = struct.unpack('BB', data)
        is_final_frame: bool = bool(header & self.FIN)
        reserved_bits: int = header & self.RSV_MASK
        opcode: int = header & self.OPCODE_MASK
        opcode_is_control: bool = bool(opcode & 8)
        if self._decompressor is not None and opcode != 0:
            self._frame_compressed = bool(reserved_bits & self.RSV1)
            reserved_bits &= ~self.RSV1
        if reserved_bits:
            self._abort()
            return
        is_masked: bool = bool(mask_payloadlen & 128)
        payloadlen: int = mask_payloadlen & 127
        if opcode_is_control and payloadlen >= 126:
            self._abort()
            return
        if payloadlen < 126:
            self._frame_length = payloadlen
        elif payloadlen == 126:
            data = await self._read_bytes(2)
            payloadlen = struct.unpack('!H', data)[0]
        elif payloadlen == 127:
            data = await self._read_bytes(8)
            payloadlen = struct.unpack('!Q', data)[0]
        new_len: int = payloadlen
        if self._fragmented_message_buffer is not None:
            new_len += len(self._fragmented_message_buffer)
        if new_len > self.params.max_message_size:
            self.close(1009, 'message too big')
            self._abort()
            return
        if is_masked:
            self._frame_mask = await self._read_bytes(4)
        data = await self._read_bytes(payloadlen)
        if is_masked:
            assert self._frame_mask is not None
            data = _websocket_mask(self._frame_mask, data)
        if opcode_is_control:
            if not is_final_frame:
                self._abort()
                return
        elif opcode == 0:
            if self._fragmented_message_buffer is None:
                self._abort()
                return
            self._fragmented_message_buffer.extend(data)
            if is_final_frame:
                opcode = self._fragmented_message_opcode  # type: ignore
                data = bytes(self._fragmented_message_buffer)
                self._fragmented_message_buffer = None
        else:
            if self._fragmented_message_buffer is not None:
                self._abort()
                return
            if not is_final_frame:
                self._fragmented_message_opcode = opcode
                self._fragmented_message_buffer = bytearray(data)
        if is_final_frame:
            handled_future: Optional[Awaitable[Any]] = self._handle_message(opcode, data)
            if handled_future is not None:
                await handled_future

    def _handle_message(self, opcode: int, data: bytes) -> Optional[Awaitable[Any]]:
        if self.client_terminated:
            return None
        if self._frame_compressed:
            assert self._decompressor is not None
            try:
                data = self._decompressor.decompress(data)
            except _DecompressTooLargeError:
                self.close(1009, 'message too big after decompression')
                self._abort()
                return None
        if opcode == 1:
            self._message_bytes_in += len(data)
            try:
                decoded: str = data.decode('utf-8')
            except UnicodeDecodeError:
                self._abort()
                return None
            return self._run_callback(self.handler.on_message, decoded)
        elif opcode == 2:
            self._message_bytes_in += len(data)
            return self._run_callback(self.handler.on_message, data)
        elif opcode == 8:
            self.client_terminated = True
            if len(data) >= 2:
                self.close_code = struct.unpack('>H', data[:2])[0]
            if len(data) > 2:
                self.close_reason = to_unicode(data[2:])
            self.close(self.close_code)
        elif opcode == 9:
            try:
                self._write_frame(True, 10, data)
            except StreamClosedError:
                self._abort()
            self._run_callback(self.handler.on_ping, data)
        elif opcode == 10:
            self.last_pong = IOLoop.current().time()
            return self._run_callback(self.handler.on_pong, data)
        else:
            self._abort()
        return None

    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        if not self.server_terminated:
            if not self.stream.closed():
                if code is None and reason is not None:
                    code = 1000
                if code is None:
                    close_data: bytes = b''
                else:
                    close_data = struct.pack('>H', code)
                if reason is not None:
                    close_data += utf8(reason)
                try:
                    self._write_frame(True, 8, close_data)
                except StreamClosedError:
                    self._abort()
            self.server_terminated = True
        if self.client_terminated:
            if self._waiting is not None:
                self.stream.io_loop.remove_timeout(self._waiting)  # type: ignore
                self._waiting = None
            self.stream.close()  # type: ignore
        elif self._waiting is None:
            self._waiting = self.stream.io_loop.add_timeout(self.stream.io_loop.time() + 5, self._abort)  # type: ignore
        if self.ping_callback:
            self.ping_callback.stop()
            self.ping_callback = None

    def is_closing(self) -> bool:
        return self.stream.closed() or self.client_terminated or self.server_terminated  # type: ignore

    @property
    def ping_interval(self) -> float:
        interval: Optional[float] = self.params.ping_interval
        if interval is not None:
            return interval
        return 0.0

    @property
    def ping_timeout(self) -> float:
        timeout: Optional[float] = self.params.ping_timeout
        if timeout is not None:
            return timeout
        assert self.ping_interval is not None
        return max(3 * self.ping_interval, 30)

    def start_pinging(self) -> None:
        assert self.ping_interval is not None
        if self.ping_interval > 0:
            self.last_ping = self.last_pong = IOLoop.current().time()
            self.ping_callback = PeriodicCallback(self.periodic_ping, self.ping_interval * 1000)
            self.ping_callback.start()

    def periodic_ping(self) -> None:
        if self.is_closing() and self.ping_callback is not None:
            self.ping_callback.stop()
            return
        now: float = IOLoop.current().time()
        since_last_pong: float = now - self.last_pong
        since_last_ping: float = now - self.last_ping
        assert self.ping_interval is not None
        assert self.ping_timeout is not None
        if since_last_ping < 2 * self.ping_interval and since_last_pong > self.ping_timeout:
            self.close()
            return
        self.write_ping(b'')
        self.last_ping = now

    def set_nodelay(self, x: bool) -> None:
        self.stream.set_nodelay(x)  # type: ignore

class WebSocketClientConnection(simple_httpclient._HTTPConnection):
    protocol: Optional[WebSocketProtocol] = None

    def __init__(self, 
                 request: httpclient.HTTPRequest, 
                 on_message_callback: Optional[Callable[[Optional[Union[str, bytes]]], Any]] = None, 
                 compression_options: Optional[Dict[str, Any]] = None, 
                 ping_interval: Optional[float] = None, 
                 ping_timeout: Optional[float] = None, 
                 max_message_size: int = _default_max_message_size, 
                 subprotocols: Optional[List[str]] = None, 
                 resolver: Optional[Resolver] = None) -> None:
        self.connect_future: Future["WebSocketClientConnection"] = Future()  # type: ignore
        self.read_queue: Queue[Optional[Union[str, bytes]]] = Queue(1)
        self.key: bytes = base64.b64encode(os.urandom(16))
        self._on_message_callback: Optional[Callable[[Optional[Union[str, bytes]]], Any]] = on_message_callback
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self.params: _WebSocketParams = _WebSocketParams(ping_interval=ping_interval, 
                                                         ping_timeout=ping_timeout, 
                                                         max_message_size=max_message_size, 
                                                         compression_options=compression_options)
        scheme, sep, rest = request.url.partition(':')
        scheme = {'ws': 'http', 'wss': 'https'}[scheme]
        request.url = scheme + sep + rest
        request.headers.update({'Upgrade': 'websocket', 
                                'Connection': 'Upgrade', 
                                'Sec-WebSocket-Key': to_unicode(self.key), 
                                'Sec-WebSocket-Version': '13'})
        if subprotocols is not None:
            request.headers['Sec-WebSocket-Protocol'] = ','.join(subprotocols)
        if compression_options is not None:
            request.headers['Sec-WebSocket-Extensions'] = 'permessage-deflate; client_max_window_bits'
        request.follow_redirects = False
        self.tcp_client: TCPClient = TCPClient(resolver=resolver)
        super().__init__(None, request, lambda: None, self._on_http_response, 104857600, self.tcp_client, 65536, 104857600)

    def __del__(self) -> None:
        if self.protocol is not None:
            warnings.warn('Unclosed WebSocketClientConnection', ResourceWarning)

    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        if self.protocol is not None:
            self.protocol.close(code, reason)
            self.protocol = None

    def on_connection_close(self) -> None:
        if not self.connect_future.done():
            self.connect_future.set_exception(StreamClosedError())
        self._on_message(None)
        self.tcp_client.close()
        super().on_connection_close()

    def on_ws_connection_close(self, close_code: Optional[int] = None, close_reason: Optional[str] = None) -> None:
        self.close_code = close_code
        self.close_reason = close_reason
        self.on_connection_close()

    def _on_http_response(self, response: httpclient.HTTPResponse) -> None:
        if not self.connect_future.done():
            if response.error:
                self.connect_future.set_exception(response.error)
            else:
                self.connect_future.set_exception(WebSocketError('Non-websocket response'))

    async def headers_received(self, start_line: httputil.ResponseStartLine, headers: httputil.HTTPHeaders) -> None:
        assert isinstance(start_line, httputil.ResponseStartLine)
        if start_line.code != 101:
            await super().headers_received(start_line, headers)
            return
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
            self._timeout = None
        self.headers: httputil.HTTPHeaders = headers
        self.protocol = self.get_websocket_protocol()
        assert self.protocol is not None
        self.protocol._process_server_headers(self.key.decode('utf-8'), self.headers)
        self.protocol.stream = self.connection.detach()
        IOLoop.current().add_callback(self.protocol._receive_frame_loop)
        self.protocol.start_pinging()
        self.final_callback = None
        future_set_result_unless_cancelled(self.connect_future, self)

    def write_message(self, message: Union[str, Dict[str, Any]], binary: bool = False) -> asyncio.Future[Any]:
        if self.protocol is None:
            raise WebSocketClosedError('Client connection has been closed')
        return self.protocol.write_message(message, binary=binary)

    def read_message(self, callback: Optional[Callable[[asyncio.Future[Any]], None]] = None) -> Awaitable[Optional[Union[str, bytes]]]:
        awaitable: Awaitable[Optional[Union[str, bytes]]] = self.read_queue.get()
        if callback is not None:
            self.io_loop.add_future(asyncio.ensure_future(awaitable), callback)
        return awaitable

    def on_message(self, message: Optional[Union[str, bytes]]) -> Any:
        return self._on_message(message)

    def _on_message(self, message: Optional[Union[str, bytes]]) -> Optional[Any]:
        if self._on_message_callback:
            self._on_message_callback(message)
            return None
        else:
            return self.read_queue.put(message)

    def ping(self, data: Union[str, bytes] = b'') -> None:
        data = utf8(data)
        if self.protocol is None:
            raise WebSocketClosedError()
        self.protocol.write_ping(data)

    def on_pong(self, data: bytes) -> None:
        pass

    def on_ping(self, data: bytes) -> None:
        pass

    def get_websocket_protocol(self) -> WebSocketProtocol:
        return WebSocketProtocol13(self, mask_outgoing=True, params=self.params)

    @property
    def selected_subprotocol(self) -> Optional[str]:
        return self.protocol.selected_subprotocol  # type: ignore

    def log_exception(self, typ: Any, value: Any, tb: Any) -> None:
        assert typ is not None
        assert value is not None
        app_log.error('Uncaught exception %s', value, exc_info=(typ, value, tb))

def websocket_connect(url: Union[str, httpclient.HTTPRequest],
                      callback: Optional[Callable[[asyncio.Future["WebSocketClientConnection"]], Any]] = None,
                      connect_timeout: Optional[float] = None,
                      on_message_callback: Optional[Callable[[Optional[Union[str, bytes]]], Any]] = None,
                      compression_options: Optional[Dict[str, Any]] = None,
                      ping_interval: Optional[float] = None,
                      ping_timeout: Optional[float] = None,
                      max_message_size: int = _default_max_message_size,
                      subprotocols: Optional[List[str]] = None,
                      resolver: Optional[Resolver] = None
                      ) -> Awaitable[WebSocketClientConnection]:
    if isinstance(url, httpclient.HTTPRequest):
        assert connect_timeout is None
        request: httpclient.HTTPRequest = url
        request.headers = httputil.HTTPHeaders(request.headers)
    else:
        request = httpclient.HTTPRequest(url, connect_timeout=connect_timeout)
    request = cast(httpclient.HTTPRequest, httpclient._RequestProxy(request, httpclient.HTTPRequest._DEFAULTS))
    conn: WebSocketClientConnection = WebSocketClientConnection(request, 
                                                                on_message_callback=on_message_callback, 
                                                                compression_options=compression_options, 
                                                                ping_interval=ping_interval, 
                                                                ping_timeout=ping_timeout, 
                                                                max_message_size=max_message_size, 
                                                                subprotocols=subprotocols, 
                                                                resolver=resolver)
    if callback is not None:
        IOLoop.current().add_future(conn.connect_future, callback)
    return conn.connect_future
