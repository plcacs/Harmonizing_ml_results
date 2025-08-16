import errno
import os
import socket
import ssl
from tornado import gen
from tornado.log import app_log
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream, SSLIOStream
from tornado.netutil import bind_sockets, add_accept_handler, ssl_wrap_socket, _DEFAULT_BACKLOG
from tornado import process
from tornado.util import errno_from_exception
import typing
from typing import Union, Dict, Any, Iterable, Optional, Awaitable
if typing.TYPE_CHECKING:
    from typing import Callable, List

class TCPServer:
    def __init__(self, ssl_options: Optional[Union[ssl.SSLContext, Dict[str, Any]]] = None, max_buffer_size: Optional[int] = None, read_chunk_size: Optional[int] = None) -> None:
    
    def listen(self, port: int, address: Optional[str] = None, family: int = socket.AF_UNSPEC, backlog: int = _DEFAULT_BACKLOG, flags: Optional[int] = None, reuse_port: bool = False) -> None:
    
    def add_sockets(self, sockets: Iterable[socket.socket]) -> None:
    
    def add_socket(self, socket: socket.socket) -> None:
    
    def bind(self, port: int, address: Optional[str] = None, family: int = socket.AF_UNSPEC, backlog: int = _DEFAULT_BACKLOG, flags: Optional[int] = None, reuse_port: bool = False) -> None:
    
    def start(self, num_processes: Optional[int] = 1, max_restarts: Optional[int] = None) -> None:
    
    def stop(self) -> None:
    
    def handle_stream(self, stream: Union[IOStream, SSLIOStream], address: Tuple[str, int]) -> Awaitable[None]:
    
    def _handle_connection(self, connection: socket.socket, address: Tuple[str, int]) -> None:
