import asyncio
from contextlib import suppress
from typing import Any, Optional, Tuple, Type, Union, cast
from .base_protocol import BaseProtocol
from .client_exceptions import ClientConnectionError, ClientOSError, ClientPayloadError, ServerDisconnectedError, SocketTimeoutError
from .helpers import _EXC_SENTINEL, EMPTY_BODY_STATUS_CODES, BaseTimerContext, set_exception, set_result
from .http import HttpResponseParser, RawResponseMessage, WebSocketReader
from .http_exceptions import HttpProcessingError
from .streams import EMPTY_PAYLOAD, DataQueue, StreamReader

class ResponseHandler(BaseProtocol, DataQueue[Tuple[RawResponseMessage, StreamReader]]):
    """Helper class to adapt between Protocol and StreamReader."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        BaseProtocol.__init__(self, loop=loop)
        DataQueue.__init__(self, loop)
        self._should_close: bool = False
        self._payload: Optional[StreamReader] = None
        self._skip_payload: bool = False
        self._payload_parser: Optional[Any] = None
        self._timer: Optional[BaseTimerContext] = None
        self._tail: bytes = b''
        self._upgraded: bool = False
        self._parser: Optional[HttpResponseParser] = None
        self._read_timeout: Optional[float] = None
        self._read_timeout_handle: Optional[asyncio.TimerHandle] = None
        self._timeout_ceil_threshold: int = 5
        self.closed: asyncio.Future[None] = self._loop.create_future()

    @property
    def upgraded(self) -> bool:
        return self._upgraded

    @property
    def should_close(self) -> bool:
        return bool(
            self._should_close or 
            (self._payload is not None and (not self._payload.is_eof())) or 
            self._upgraded or 
            (self._exception is not None) or 
            (self._payload_parser is not None) or 
            self._buffer or 
            self._tail
        )

    def force_close(self) -> None:
        self._should_close = True

    def close(self) -> None:
        transport = self.transport
        if transport is not None:
            transport.close()
            self.transport = None
            self._payload = None
            self._drop_timeout()

    def is_connected(self) -> bool:
        return self.transport is not None and (not self.transport.is_closing())

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._drop_timeout()
        original_connection_error = exc
        reraised_exc: Optional[Exception] = original_connection_error
        connection_closed_cleanly = original_connection_error is None
        if connection_closed_cleanly:
            set_result(self.closed, None)
        else:
            assert original_connection_error is not None
            set_exception(
                self.closed, 
                ClientConnectionError(f'Connection lost: {original_connection_error!s}'), 
                original_connection_error
            )
        if self._payload_parser is not None:
            with suppress(Exception):
                self._payload_parser.feed_eof()
        uncompleted: Optional[bytes] = None
        if self._parser is not None:
            try:
                uncompleted = self._parser.feed_eof()
            except Exception as underlying_exc:
                if self._payload is not None:
                    client_payload_exc_msg = f'Response payload is not completed: {underlying_exc!r}'
                    if not connection_closed_cleanly:
                        client_payload_exc_msg = f'{client_payload_exc_msg!s}. {original_connection_error!r}'
                    set_exception(
                        self._payload, 
                        ClientPayloadError(client_payload_exc_msg), 
                        underlying_exc
                    )
        if not self.is_eof():
            if isinstance(original_connection_error, OSError):
                reraised_exc = ClientOSError(*original_connection_error.args)
            if connection_closed_cleanly:
                reraised_exc = ServerDisconnectedError(uncompleted)
            underlying_non_eof_exc: Optional[Exception] = _EXC_SENTINEL if connection_closed_cleanly else original_connection_error
            assert underlying_non_eof_exc is not None
            assert reraised_exc is not None
            self.set_exception(reraised_exc, underlying_non_eof_exc)
        self._should_close = True
        self._parser = None
        self._payload = None
        self._payload_parser = None
        self._reading_paused = False
        super().connection_lost(reraised_exc)

    def eof_received(self) -> Optional[bool]:
        self._drop_timeout()
        return None

    def pause_reading(self) -> None:
        super().pause_reading()
        self._drop_timeout()

    def resume_reading(self) -> None:
        super().resume_reading()
        self._reschedule_timeout()

    def set_exception(self, exc: Exception, exc_cause: Any = _EXC_SENTINEL) -> None:
        self._should_close = True
        self._drop_timeout()
        super().set_exception(exc, exc_cause)

    def set_parser(self, parser: Any, payload: StreamReader) -> None:
        self._payload = payload
        self._payload_parser = parser
        self._drop_timeout()
        if self._tail:
            data, self._tail = (self._tail, b'')
            self.data_received(data)

    def set_response_params(
        self, 
        *, 
        timer: Optional[BaseTimerContext] = None, 
        skip_payload: bool = False, 
        read_until_eof: bool = False, 
        auto_decompress: bool = True, 
        read_timeout: Optional[float] = None, 
        read_bufsize: int = 2 ** 16, 
        timeout_ceil_threshold: int = 5, 
        max_line_size: int = 8190, 
        max_field_size: int = 8190
    ) -> None:
        self._skip_payload = skip_payload
        self._read_timeout = read_timeout
        self._timeout_ceil_threshold = timeout_ceil_threshold
        self._parser = HttpResponseParser(
            self, 
            self._loop, 
            read_bufsize, 
            timer=timer, 
            payload_exception=ClientPayloadError, 
            response_with_body=not skip_payload, 
            read_until_eof=read_until_eof, 
            auto_decompress=auto_decompress, 
            max_line_size=max_line_size, 
            max_field_size=max_field_size
        )
        if self._tail:
            data, self._tail = (self._tail, b'')
            self.data_received(data)

    def _drop_timeout(self) -> None:
        if self._read_timeout_handle is not None:
            self._read_timeout_handle.cancel()
            self._read_timeout_handle = None

    def _reschedule_timeout(self) -> None:
        timeout = self._read_timeout
        if self._read_timeout_handle is not None:
            self._read_timeout_handle.cancel()
        if timeout:
            self._read_timeout_handle = self._loop.call_later(timeout, self._on_read_timeout)
        else:
            self._read_timeout_handle = None

    def start_timeout(self) -> None:
        self._reschedule_timeout()

    @property
    def read_timeout(self) -> Optional[float]:
        return self._read_timeout

    @read_timeout.setter
    def read_timeout(self, read_timeout: Optional[float]) -> None:
        self._read_timeout = read_timeout

    def _on_read_timeout(self) -> None:
        exc = SocketTimeoutError('Timeout on reading data from socket')
        self.set_exception(exc)
        if self._payload is not None:
            set_exception(self._payload, exc)

    def data_received(self, data: bytes) -> None:
        self._reschedule_timeout()
        if not data:
            return
        if self._payload_parser is not None:
            eof, tail = self._payload_parser.feed_data(data)
            if eof:
                self._payload = None
                self._payload_parser = None
                if tail:
                    self.data_received(tail)
            return
        if self._upgraded or self._parser is None:
            self._tail += data
            return
        try:
            messages, upgraded, tail = self._parser.feed_data(data)
        except BaseException as underlying_exc:
            if self.transport is not None:
                self.transport.close()
            if isinstance(underlying_exc, HttpProcessingError):
                exc = HttpProcessingError(
                    code=underlying_exc.code, 
                    message=underlying_exc.message, 
                    headers=underlying_exc.headers
                )
            else:
                exc = HttpProcessingError()
            self.set_exception(exc, underlying_exc)
            return
        self._upgraded = upgraded
        payload: Optional[StreamReader] = None
        for message, payload in messages:
            if message.should_close:
                self._should_close = True
            self._payload = payload
            if self._skip_payload or message.code in EMPTY_BODY_STATUS_CODES:
                self.feed_data((message, EMPTY_PAYLOAD))
            else:
                self.feed_data((message, payload))
        if payload is not None:
            if payload is not EMPTY_PAYLOAD:
                payload.on_eof(self._drop_timeout)
            else:
                self._drop_timeout()
        if upgraded and tail:
            self.data_received(tail)
