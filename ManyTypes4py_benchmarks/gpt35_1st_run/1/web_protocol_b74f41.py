from typing import TYPE_CHECKING, Any, Awaitable, Callable, Deque, Generic, Optional, Sequence, Tuple, Type, TypeVar, Union

class RequestPayloadError(Exception):
    """Payload parsing error."""

class PayloadAccessError(Exception):
    """Payload was accessed after response was sent."""

class AccessLoggerWrapper(AbstractAsyncAccessLogger):
    """Wrap an AbstractAccessLogger so it behaves like an AbstractAsyncAccessLogger."""
    __slots__ = ('access_logger', '_loop')

    def __init__(self, access_logger: Union[AbstractAsyncAccessLogger, AbstractAccessLogger], loop: Any) -> None:
        self.access_logger = access_logger
        self._loop = loop
        super().__init__()

    async def log(self, request: Any, response: Any, request_start: float) -> None:
        self.access_logger.log(request, response, self._loop.time() - request_start)

    @property
    def enabled(self) -> bool:
        """Check if logger is enabled."""
        return self.access_logger.enabled

class RequestHandler(BaseProtocol, Generic[_Request]):
    """HTTP protocol implementation.

    RequestHandler handles incoming HTTP request. It reads request line,
    request headers and request payload and calls handle_request() method.
    By default it always returns with 404 response.

    RequestHandler handles errors in incoming request, like bad
    status line, bad headers or incomplete payload. If any error occurs,
    connection gets closed.

    keepalive_timeout -- number of seconds before closing
                         keep-alive connection

    tcp_keepalive -- TCP keep-alive is on, default is on

    logger -- custom logger object

    access_log_class -- custom class for access_logger

    access_log -- custom logging object

    access_log_format -- access log format string

    loop -- Optional event loop

    max_line_size -- Optional maximum header line size

    max_field_size -- Optional maximum header field size

    timeout_ceil_threshold -- Optional value to specify
                              threshold to ceil() timeout
                              values

    """
    __slots__ = ('_request_count', '_keepalive', '_manager', '_request_handler', '_request_factory', '_tcp_keepalive', '_next_keepalive_close_time', '_keepalive_handle', '_keepalive_timeout', '_lingering_time', '_messages', '_message_tail', '_handler_waiter', '_waiter', '_task_handler', '_upgrade', '_payload_parser', '_request_parser', 'logger', 'access_log', 'access_logger', '_close', '_force_close', '_current_request', '_timeout_ceil_threshold', '_request_in_progress')

    def __init__(self, manager: Any, *, loop: Any, keepalive_timeout: int = 3630, tcp_keepalive: bool = True, logger: Logger = server_logger, access_log_class: Type[_AnyAbstractAccessLogger] = AccessLogger, access_log: Logger = access_logger, access_log_format: str = AccessLogger.LOG_FORMAT, max_line_size: int = 8190, max_field_size: int = 8190, lingering_time: float = 10.0, read_bufsize: int = 2 ** 16, auto_decompress: bool = True, timeout_ceil_threshold: int = 5) -> None:
        super().__init__(loop)
        self._request_count = 0
        self._keepalive = False
        self._current_request = None
        self._manager = manager
        self._request_handler = manager.request_handler
        self._request_factory = manager.request_factory
        self._tcp_keepalive = tcp_keepalive
        self._next_keepalive_close_time = 0.0
        self._keepalive_handle = None
        self._keepalive_timeout = keepalive_timeout
        self._lingering_time = float(lingering_time)
        self._messages = deque()
        self._message_tail = b''
        self._waiter = None
        self._handler_waiter = None
        self._task_handler = None
        self._upgrade = False
        self._payload_parser = None
        self._request_parser = HttpRequestParser(self, loop, read_bufsize, max_line_size=max_line_size, max_field_size=max_field_size, payload_exception=RequestPayloadError, auto_decompress=auto_decompress)
        self._timeout_ceil_threshold = 5
        try:
            self._timeout_ceil_threshold = float(timeout_ceil_threshold)
        except (TypeError, ValueError):
            pass
        self.logger = logger
        self.access_log = access_log
        if access_log:
            if issubclass(access_log_class, AbstractAsyncAccessLogger):
                self.access_logger = access_log_class()
            else:
                access_logger = access_log_class(access_log, access_log_format)
                self.access_logger = AccessLoggerWrapper(access_logger, self._loop)
        else:
            self.access_logger = None
        self._close = False
        self._force_close = False
        self._request_in_progress = False
