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
_RequestFactory = Callable[[RawRequestMessage, StreamReader, 'RequestHandler[_Request]', AbstractStreamWriter, 'asyncio.Task[None]'], _Request]

class Server(Generic[_Request]):

    @overload
    def __init__(self, handler: Union[collections.abc.Awaitable[T], typing.Callable], *, debug: Union[None, typing.Callable, typing.Any, typing.Type]=None, handler_cancellation: bool=False, **kwargs) -> None:
        ...

    @overload
    def __init__(self, handler: Union[collections.abc.Awaitable[T], typing.Callable], *, request_factory, debug: Union[None, typing.Callable, typing.Any, typing.Type]=None, handler_cancellation: bool=False, **kwargs) -> None:
        ...

    def __init__(self, handler: Union[collections.abc.Awaitable[T], typing.Callable], *, request_factory=None, debug: Union[None, typing.Callable, typing.Any, typing.Type]=None, handler_cancellation: bool=False, **kwargs) -> None:
        if debug is not None:
            warnings.warn('debug argument is no-op since 4.0 and scheduled for removal in 5.0', DeprecationWarning, stacklevel=2)
        self._loop = asyncio.get_running_loop()
        self._connections = {}
        self._kwargs = kwargs
        self.requests_count = 0
        self.request_handler = handler
        self.request_factory = request_factory or self._make_request
        self.handler_cancellation = handler_cancellation

    @property
    def connections(self) -> list:
        return list(self._connections.keys())

    def connection_made(self, handler: Union[web_protocol.RequestHandler, typing.Callable[typing.Callable, None], asyncio.AbstractEventLoop], transport: Union[web_protocol.RequestHandler, typing.Callable[typing.Callable, None], asyncio.AbstractEventLoop]) -> None:
        self._connections[handler] = transport

    def connection_lost(self, handler: Union[typing.Callable, web_protocol.RequestHandler], exc: Union[None, BaseException, Exception, typing.Callable]=None) -> None:
        if handler in self._connections:
            if handler._task_handler:
                handler._task_handler.add_done_callback(lambda f: self._connections.pop(handler, None))
            else:
                del self._connections[handler]

    def _make_request(self, message: Union[web_protocol.RequestHandler, asyncio.AbstractEventLoop, bytes], payload: Union[web_protocol.RequestHandler, asyncio.AbstractEventLoop, bytes], protocol: Union[web_protocol.RequestHandler, asyncio.AbstractEventLoop, bytes], writer: Union[web_protocol.RequestHandler, asyncio.AbstractEventLoop, bytes], task: Union[web_protocol.RequestHandler, asyncio.AbstractEventLoop, bytes]) -> BaseRequest:
        return BaseRequest(message, payload, protocol, writer, task, self._loop)

    def pre_shutdown(self) -> None:
        for conn in self._connections:
            conn.close()

    async def shutdown(self, timeout=None):
        coros = (conn.shutdown(timeout) for conn in self._connections)
        await asyncio.gather(*coros)
        self._connections.clear()

    def __call__(self) -> RequestHandler:
        try:
            return RequestHandler(self, loop=self._loop, **self._kwargs)
        except TypeError:
            kwargs = {k: v for k, v in self._kwargs.items() if k in ['debug', 'access_log_class']}
            return RequestHandler(self, loop=self._loop, **kwargs)