"""Low level HTTP server."""
import asyncio
import warnings
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar, overload
from .abc import AbstractStreamWriter
from .http_parser import RawRequestMessage
from .streams import StreamReader
from .web_protocol import RequestHandler
from .web_request import BaseRequest
from .web_response import StreamResponse

__all__ = ('Server',)

_Request = TypeVar('_Request', bound=BaseRequest)
_RequestFactory = Callable[
    [RawRequestMessage, StreamReader, RequestHandler[_Request], AbstractStreamWriter, asyncio.Task[None]],
    _Request
]


class Server(Generic[_Request]):
    _loop: asyncio.AbstractEventLoop
    _connections: Dict[RequestHandler[_Request], Any]
    _kwargs: Dict[str, Any]
    requests_count: int
    request_handler: Callable[..., RequestHandler[_Request]]
    request_factory: _RequestFactory
    handler_cancellation: bool

    @overload
    def __init__(self, handler: Callable[..., RequestHandler[_Request]], *, debug: Any = None, handler_cancellation: bool = False, **kwargs: Any) -> None:
        ...

    @overload
    def __init__(self, handler: Callable[..., RequestHandler[_Request]], *, request_factory: _RequestFactory, debug: Any = None, handler_cancellation: bool = False, **kwargs: Any) -> None:
        ...

    def __init__(self, handler: Callable[..., RequestHandler[_Request]], *, request_factory: Optional[_RequestFactory] = None, debug: Any = None, handler_cancellation: bool = False, **kwargs: Any) -> None:
        if debug is not None:
            warnings.warn(
                'debug argument is no-op since 4.0 and scheduled for removal in 5.0',
                DeprecationWarning,
                stacklevel=2
            )
        self._loop = asyncio.get_running_loop()
        self._connections = {}  # type: Dict[RequestHandler[_Request], Any]
        self._kwargs = kwargs
        self.requests_count = 0
        self.request_handler = handler
        self.request_factory = request_factory or self._make_request
        self.handler_cancellation = handler_cancellation

    @property
    def connections(self) -> List[RequestHandler[_Request]]:
        return list(self._connections.keys())

    def connection_made(self, handler: RequestHandler[_Request], transport: Any) -> None:
        self._connections[handler] = transport

    def connection_lost(self, handler: RequestHandler[_Request], exc: Optional[BaseException] = None) -> None:
        if handler in self._connections:
            if getattr(handler, "_task_handler", None):
                handler._task_handler.add_done_callback(lambda f: self._connections.pop(handler, None))
            else:
                del self._connections[handler]

    def _make_request(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: RequestHandler[_Request],
        writer: AbstractStreamWriter,
        task: asyncio.Task[None]
    ) -> BaseRequest:
        return BaseRequest(message, payload, protocol, writer, task, self._loop)

    def pre_shutdown(self) -> None:
        for conn in self._connections.values():
            conn.close()

    async def shutdown(self, timeout: Optional[float] = None) -> None:
        coros = (conn.shutdown(timeout) for conn in self._connections.values())
        await asyncio.gather(*coros)
        self._connections.clear()

    def __call__(self) -> RequestHandler[_Request]:
        try:
            return RequestHandler(self, loop=self._loop, **self._kwargs)
        except TypeError:
            kwargs = {k: v for k, v in self._kwargs.items() if k in ['debug', 'access_log_class']}
            return RequestHandler(self, loop=self._loop, **kwargs)