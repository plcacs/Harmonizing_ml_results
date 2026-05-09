class WebSocketProtocol(abc.ABC):
    """Base class for WebSocket protocol versions."""

    def __init__(self, handler: WebSocketHandler) -> None:
        self.handler: WebSocketHandler = handler
        self.stream: Optional[IOStream] = None
        self.client_terminated: bool = False
        self.server_terminated: bool = False

    def _run_callback(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Optional[Future[Any]]:
        """Runs the given callback with exception handling.

        If the callback is a coroutine, returns its Future. On error, aborts the
        websocket connection and returns None.
        """
        try:
            result: Any = callback(*args, **kwargs)
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
    def write_message(self, message: Union[str, bytes], binary: bool = False) -> Future[Any]:
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


class WebSocketProtocol13(WebSocketProtocol):
    """Implementation of the WebSocket protocol from RFC 6455.

    This class supports versions 7 and 8 of the protocol in addition to the
    final version 13.
    """
    FIN: int = 128
    RSV1: int = 64
    RSV2: int = 32
    RSV3: int = 16
    RSV_MASK: int = RSV1 | RSV2 | RSV3
    OPCODE_MASK: int = 15
    stream: Optional[IOStream] = None

    def __init__(self, handler: WebSocketHandler, mask_outgoing: bool, params: _WebSocketParams) -> None:
        WebSocketProtocol.__init__(self, handler)
        self.mask_outgoing: bool = mask_outgoing
        self.params: _WebSocketParams = params
        self._final_frame: bool = False
        self._frame_opcode: Optional[int] = None
        self._masked_frame: Optional[bytes] = None
        self._frame_mask: Optional[bytes] = None
        self._frame_length: Optional[int] = None
        self._fragmented_message_buffer: Optional[bytearray] = None
        self._fragmented_message_opcode: Optional[int] = None
        self._waiting: Optional[tornado.ioloop.Timeout] = None
        self._compression_options: Optional[Dict[str, Any]] = params.compression_options
        self._decompressor: Optional[_Decompressor] = None
        self._compressor: Optional[_Compressor] = None
        self._frame_compressed: Optional[bool] = None
        self._message_bytes_in: int = 0
        self._message_bytes_out: int = 0
        self._wire_bytes_in: int = 0
        self._wire_bytes_out: int = 0
        self.ping_callback: Optional[tornado.ioloop.PeriodicCallback] = None
        self.last_ping: float = 0.0
        self.last_pong: float = 0.0
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None

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
        """Verifies all invariant- and required headers

        If a header is missing or have an incorrect value ValueError will be
        raised
        """
        fields = ('Host', 'Sec-Websocket-Key', 'Sec-Websocket-Version')
        if not all(map(lambda f: handler.request.headers.get(f), fields)):
            raise ValueError('Missing/Invalid WebSocket headers')

    @staticmethod
    def compute_accept_value(key: str) -> str:
        """Computes the value for the Sec-WebSocket-Accept header,
        given the value for Sec-WebSocket-Key.
        """
        sha1 = hashlib.sha1()
        sha1.update(utf8(key))
        sha1.update(b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
        return native_str(base64.b64encode(sha1.digest()))

    def _challenge_response(self, handler: WebSocketHandler) -> str:
        return WebSocketProtocol13.compute_accept_value(cast(str, handler.request.headers.get('Sec-Websocket-Key')))

    async def _accept_connection(self, handler: WebSocketHandler) -> None:
        subprotocol_header = handler.request.headers.get('Sec-WebSocket-Protocol')
        if subprotocol_header:
            subprotocols = [s.strip() for s in subprotocol_header.split(',')]
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

    def _parse_extensions_header(self, headers: httputil.HTTPHeaders) -> List[Tuple[str, Dict[str, str]]]:
        extensions = headers.get('Sec-WebSocket-Extensions', '')
        if extensions:
            return [httputil._parse_header(e.strip()) for e in extensions.split(',')]
        return []

    def _process_server_headers(self, key: str, headers: httputil.HTTPHeaders) -> None:
        """Process the headers sent by the server to this client connection.

        'key' is the websocket handshake challenge/response key.
        """
        assert headers['Upgrade'].lower() == 'websocket'
        assert headers['Connection'].lower() == 'upgrade'
        accept = self.compute_accept_value(key)
        assert headers['Sec-Websocket-Accept'] == accept
        extensions = self._parse_extensions_header(headers)
        for ext in extensions:
            if ext[0] == 'permessage-deflate' and self._compression_options is not None:
                self._create_compressors('client', ext[1])
            else:
                raise ValueError('unsupported extension %r', ext)
        self.selected_subprotocol = headers.get('Sec-WebSocket-Protocol', None)

    def _get_compressor_options(self, side: str, agreed_parameters: Dict[str, str], compression_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Converts a websocket agreed_parameters set to keyword arguments
        for our compressor objects.
        """
        options = dict(persistent=side + '_no_context_takeover' not in agreed_parameters)
        wbits_header = agreed_parameters.get(side + '_max_window_bits', None)
        if wbits_header is None:
            options['max_wbits'] = zlib.MAX_WBITS
        else:
            options['max_wbits'] = int(wbits_header)
        options['compression_options'] = compression_options
        return options

    def _create_compressors(self, side: str, agreed_parameters: Dict[str, str], compression_options: Optional[Dict[str, Any]] = None) -> None:
        allowed_keys = {'server_no_context_takeover', 'client_no_context_takeover', 'server_max_window_bits', 'client_max_window_bits'}
        for key in agreed_parameters:
            if key not in allowed_keys:
                raise ValueError('unsupported compression parameter %r' % key)
        other_side = 'client' if side == 'server' else 'server'
        self._compressor = _PerMessageDeflateCompressor(**self._get_compressor_options(side, agreed_parameters, compression_options))
        self._decompressor = _PerMessageDeflateDecompressor(max_message_size=self.params.max_message_size, **self._get_compressor_options(other_side, agreed_parameters, compression_options))

    def _write_frame(self, fin: bool, opcode: int, data: bytes, flags: int = 0) -> Future[Any]:
        data_len = len(data)
        if opcode & 8:
            if not fin:
                raise ValueError('control frames may not be fragmented')
            if data_len > 125:
                raise ValueError('control frame payloads may not exceed 125 bytes')
        if fin:
            finbit = self.FIN
        else:
            finbit = 0
        frame = struct.pack('B', finbit | opcode | flags)
        if self.mask_outgoing:
            mask_bit = 128
        else:
            mask_bit = 0
        if data_len < 126:
            frame += struct.pack('B', data_len | mask_bit)
        elif data_len <= 65535:
            frame += struct.pack('!BH', 126 | mask_bit, data_len)
        else:
            frame += struct.pack('!BQ', 127 | mask_bit, data_len)
        if self.mask_outgoing:
            mask = os.urandom(4)
            data = mask + _websocket_mask(mask, data)
        frame += data
        self._wire_bytes_out += len(frame)
        return self.stream.write(frame)

    def write_message(self, message: Union[str, bytes], binary: bool = False) -> Future[Any]:
        """Sends the given message to the client of this Web Socket."""
        if binary:
            opcode = 2
        else:
            opcode = 1
        if isinstance(message, dict):
            message = tornado.escape.json_encode(message)
        message = tornado.escape.utf8(message)
        assert isinstance(message, bytes)
        self._message_bytes_out += len(message)
        flags = 0
        if self._compressor:
            message = self._compressor.compress(message)
            flags |= self.RSV1
        try:
            fut = self._write_frame(True, opcode, message, flags=flags)
        except StreamClosedError:
            raise WebSocketClosedError()

        async def wrapper() -> None:
            try:
                await fut
            except StreamClosedError:
                raise WebSocketClosedError()
        return asyncio.ensure_future(wrapper())

    def write_ping(self, data: bytes) -> None:
        """Send ping frame."""
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
        data = await self.stream.read_bytes(n)
        self._wire_bytes_in += n
        return data

    async def _receive_frame(self) -> None:
        data = await self._read_bytes(2)
        header, mask_payloadlen = struct.unpack('BB', data)
        is_final_frame = header & self.FIN
        reserved_bits = header & self.RSV_MASK
        opcode = header & self.OPCODE_MASK
        opcode_is_control = opcode & 8
        if self._decompressor is not None and opcode != 0:
            self._frame_compressed = bool(reserved_bits & self.RSV1)
            reserved_bits &= ~self.RSV1
        if reserved_bits:
            self._abort()
            return
        is_masked = bool(mask_payloadlen & 128)
        payloadlen = mask_payloadlen & 127
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
        new_len = payloadlen
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
                opcode = self._fragmented_message_opcode
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
            handled_future = self._handle_message(opcode, data)
            if handled_future is not None:
                await handled_future

    def _handle_message(self, opcode: int, data: bytes) -> Optional[Future[Any]]:
        """Execute on_message, returning its Future if it is a coroutine."""
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
                decoded = data.decode('utf-8')
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
        """Closes the WebSocket connection."""
        if not self.server_terminated:
            if not self.stream.closed():
                if code is None and reason is not None:
                    code = 1000
                if code is None:
                    close_data = b''
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
                self.stream.io_loop.remove_timeout(self._waiting)
                self._waiting = None
            self.stream.close()
        elif self._waiting is None:
            self._waiting = self.stream.io_loop.add_timeout(self.stream.io_loop.time() + 5, self._abort)
        if self.ping_callback:
            self.ping_callback.stop()
            self.ping_callback = None

    def is_closing(self) -> bool:
        """Return ``True`` if this connection is closing.

        The connection is considered closing if either side has
        initiated its closing handshake or if the stream has been
        shut down uncleanly.
        """
        return self.stream.closed() or self.client_terminated or self.server_terminated

    @property
    def ping_interval(self) -> Optional[int]:
        interval = self.params.ping_interval
        if interval is not None:
            return interval
        return 0

    @property
    def ping_timeout(self) -> Optional[int]:
        timeout = self.params.ping_timeout
        if timeout is not None:
            return timeout
        assert self.ping_interval is not None
        return max(3 * self.ping_interval, 30)

    def start_pinging(self) -> None:
        """Start sending periodic pings to keep the connection alive"""
        assert self.ping_interval is not None
        if self.ping_interval > 0:
            self.last_ping = self.last_pong = IOLoop.current().time()
            self.ping_callback = PeriodicCallback(self.periodic_ping, self.ping_interval * 1000)
            self.ping_callback.start()

    def periodic_ping(self) -> None:
        """Send a ping to keep the websocket alive

        Called periodically if the websocket_ping_interval is set and non-zero.
        """
        if self.is_closing() and self.ping_callback is not None:
            self.ping_callback.stop()
            return
        now = IOLoop.current().time()
        since_last_pong = now - self.last_pong
        since_last_ping = now - self.last_ping
        assert self.ping_interval is not None
        assert self.ping_timeout is not None
        if since_last_ping < 2 * self.ping_interval and since_last_pong > self.ping_timeout:
            self.close()
            return
        self.write_ping(b'')
        self.last_ping = now

    def set_nodelay(self, x: bool) -> None:
        self.stream.set_nodelay(x)


class WebSocketClientConnection(simple_httpclient._HTTPConnection):
    """WebSocket client connection.

    This class should not be instantiated directly; use the
    `websocket_connect` function instead.
    """
    protocol: Optional[WebSocketProtocol] = None

    def __init__(self, request: httpclient.HTTPRequest, on_message_callback: Optional[Callable[[Union[str, bytes]], None]] = None, compression_options: Optional[Dict[str, Any]] = None, ping_interval: Optional[int] = None, ping_timeout: Optional[int] = None, max_message_size: int = _default_max_message_size, subprotocols: Optional[List[str]] = None, resolver: Optional[Resolver] = None) -> None:
        self.connect_future: Future[Any] = Future()
        self.read_queue: Queue[Optional[Union[str, bytes]]] = Queue(1)
        self.key: bytes = base64.b64encode(os.urandom(16))
        self._on_message_callback: Optional[Callable[[Union[str, bytes]], None]] = on_message_callback
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self.params: _WebSocketParams = _WebSocketParams(ping_interval=ping_interval, ping_timeout=ping_timeout, max_message_size=max_message_size, compression_options=compression_options)
        scheme, sep, rest = request.url.partition(':')
        scheme = {'ws': 'http', 'wss': 'https'}[scheme]
        request.url = scheme + sep + rest
        request.headers.update({'Upgrade': 'websocket', 'Connection': 'Upgrade', 'Sec-WebSocket-Key': to_unicode(self.key), 'Sec-WebSocket-Version': '13'})
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
        """Closes the websocket connection.

        ``code`` and ``reason`` are documented under
        `WebSocketHandler.close`.

        .. versionadded:: 3.2

        .. versionchanged:: 4.0

           Added the ``code`` and ``reason`` arguments.
        """
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
        self.headers = headers
        self.protocol = self.get_websocket_protocol()
        self.protocol._process_server_headers(self.key, self.headers)
        self.protocol.stream = self.connection.detach()
        IOLoop.current().add_callback(self.protocol._receive_frame_loop)
        self.protocol.start_pinging()
        future_set_result_unless_cancelled(self.connect_future, self)

    def write_message(self, message: Union[str, bytes], binary: bool = False) -> Future[Any]:
        """Sends a message to the WebSocket server.

        If the stream is closed, raises `WebSocketClosedError`.
        Returns a `.Future` which can be used for flow control.

        .. versionchanged:: 5.0
           Exception raised on a closed stream changed from `.StreamClosedError`
           to `WebSocketClosedError`.
        """
        if self.protocol is None:
            raise WebSocketClosedError('Client connection has been closed')
        return self.protocol.write_message(message, binary=binary)

    def read_message(self, callback: Optional[Callable[[Optional[Union[str, bytes]]], None]] = None) -> Future[Optional[Union[str, bytes]]]:
        """Reads a message from the WebSocket server.

        If on_message_callback was specified at WebSocket
        initialization, this function will never return messages

        Returns a future whose result is the message, or None
        if the connection is closed.  If a callback argument
        is given it will be called with the future when it is
        ready.
        """
        awaitable = self.read_queue.get()
        if callback is not None:
            self.io_loop.add_future(asyncio.ensure_future(awaitable), callback)
        return awaitable

    def on_message(self, message: Optional[Union[str, bytes]]) -> None:
        return self._on_message(message)

    def _on_message(self, message: Optional[Union[str, bytes]]) -> None:
        if self._on_message_callback:
            self._on_message_callback(message)
            return
        else:
            self.read_queue.put(message)

    def ping(self, data: bytes = b'') -> None:
        """Send ping frame to the remote end.

        The data argument allows a small amount of data (up to 125
        bytes) to be sent as a part of the ping message. Note that not
        all websocket implementations expose this data to
        applications.

        Consider using the ``ping_interval`` argument to
        `websocket_connect` instead of sending pings manually.

        .. versionadded:: 5.1

        """
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
        """The subprotocol selected by the server.

        .. versionadded:: 5.1
        """
        return self.protocol.selected_subprotocol

    def log_exception(self, typ: Type[Exception], value: Exception, tb: TracebackType) -> None:
        assert typ is not None
        assert value is not None
        app_log.error('Uncaught exception %s', value, exc_info=(typ, value, tb))


def websocket_connect(url: str, callback: Optional[Callable[[WebSocketClientConnection], None]] = None, connect_timeout: Optional[int] = None, on_message_callback: Optional[Callable[[Union[str, bytes]], None]] = None, compression_options: Optional[Dict[str, Any]] = None, ping_interval: Optional[int] = None, ping_timeout: Optional[int] = None, max_message_size: int = _default_max_message_size, subprotocols: Optional[List[str]] = None, resolver: Optional[Resolver] = None) -> Future[WebSocketClientConnection]:
    """Client-side websocket support.

    Takes a url and returns a Future whose result is a
    `WebSocketClientConnection`.

    ``compression_options`` is interpreted in the same way as the
    return value of `.WebSocketHandler.get_compression_options`.

    The connection supports two styles of operation. In the coroutine
    style, the application typically calls
    `~.WebSocketClientConnection.read_message` in a loop::

        conn = yield websocket_connect(url)
        while True:
            msg = yield conn.read_message()
            if msg is None: break
            # Do something with msg

    In the callback style, pass an ``on_message_callback`` to
    ``websocket_connect``. In both styles, a message of ``None``
    indicates that the connection has been closed.

    ``subprotocols`` may be a list of strings specifying proposed
    subprotocols. The selected protocol may be found on the
    ``selected_subprotocol`` attribute of the connection object
    when the connection is complete.

    .. versionchanged:: 3.2
       Also accepts ``HTTPRequest`` objects in place of urls.

    .. versionchanged:: 4.1
       Added ``compression_options`` and ``on_message_callback``.

    .. versionchanged:: 4.5
       Added the ``ping_interval``, ``ping_timeout``, and ``max_message_size``
       arguments, which have the same meaning as in `WebSocketHandler`.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    .. versionchanged:: 5.1
       Added the ``subprotocols`` argument.

    .. versionchanged:: 6.3
       Added the ``resolver`` argument.
    """
    if isinstance(url, httpclient.HTTPRequest):
        assert connect_timeout is None
        request = url
        request.headers = httputil.HTTPHeaders(request.headers)
    else:
        request = httpclient.HTTPRequest(url, connect_timeout=connect_timeout)
    request = cast(httpclient.HTTPRequest, httpclient._RequestProxy(request, httpclient.HTTPRequest._DEFAULTS))
    conn = WebSocketClientConnection(request, on_message_callback=on_message_callback, compression_options=compression_options, ping_interval=ping_interval, ping_timeout=ping_timeout, max_message_size=max_message_size, subprotocols=subprotocols, resolver=resolver)
    if callback is not None:
        IOLoop.current().add_future(conn.connect_future, callback)
    return conn.connect_future
