import asyncio
import concurrent.futures
import errno
import os
import sys
import socket
import ssl
import stat
from tornado.concurrent import dummy_executor, run_on_executor
from tornado.ioloop import IOLoop
from tornado.util import Configurable, errno_from_exception
from typing import List, Callable, Any, Type, Dict, Union, Tuple, Awaitable, Optional

_client_ssl_defaults = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
_server_ssl_defaults = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
if hasattr(ssl, 'OP_NO_COMPRESSION'):
    _client_ssl_defaults.options |= ssl.OP_NO_COMPRESSION
    _server_ssl_defaults.options |= ssl.OP_NO_COMPRESSION

def bind_sockets(port: int, address: Optional[str] = None, family: int = socket.AF_UNSPEC, backlog: int = _DEFAULT_BACKLOG, flags: int = None, reuse_port: bool = False) -> List[socket.socket]:
    ...

def add_accept_handler(sock: socket.socket, callback: Callable[[socket.socket, Tuple[str, int]], None]) -> Callable[[], None]:
    ...

def is_valid_ip(ip: str) -> bool:
    ...

class Resolver(Configurable):
    ...

    def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> Awaitable[List[Tuple[int, Tuple[str, int]]]]:
        ...

    def close(self) -> None:
        ...

def _resolve_addr(host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[str, int]]]:
    ...

class DefaultExecutorResolver(Resolver):
    ...

    async def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[str, int]]]:
        ...

class DefaultLoopResolver(Resolver):
    ...

    async def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[str, int]]]:
        ...

class ExecutorResolver(Resolver):
    ...

    def initialize(self, executor: concurrent.futures.Executor = None, close_executor: bool = True) -> None:
        ...

    def close(self) -> None:
        ...

    @run_on_executor
    def resolve(self, host: str, port: int, family: int = socket.AF_UNSPEC) -> List[Tuple[int, Tuple[str, int]]]:
        ...

class BlockingResolver(ExecutorResolver):
    ...

    def initialize(self) -> None:
        ...

class ThreadedResolver(ExecutorResolver):
    ...

    def initialize(self, num_threads: int = 10) -> None:
        ...

class OverrideResolver(Resolver):
    ...

    def initialize(self, resolver: Resolver, mapping: Dict[Union[Tuple[str, int], Tuple[str, int, int]], str]) -> None:
        ...

    def close(self) -> None:
        ...

def ssl_options_to_context(ssl_options: Union[ssl.SSLContext, Dict[str, Any]], server_side: bool = None) -> ssl.SSLContext:
    ...

def ssl_wrap_socket(socket: socket.socket, ssl_options: Union[ssl.SSLContext, Dict[str, Any]], server_hostname: str = None, server_side: bool = None, **kwargs: Any) -> ssl.SSLSocket:
    ...
