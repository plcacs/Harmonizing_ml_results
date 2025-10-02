"""A non-blocking, single-threaded TCP server."""
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
    def __init__(
        self,
        ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]] = None,
        max_buffer_size: Optional[int] = None,
        read_chunk_size: Optional[int] = None
    ) -> None:
        self.ssl_options = ssl_options
        self._sockets: Dict[int, socket.socket] = {}
        self._handlers: Dict[int, Callable[[], None]] = {}
        self._pending_sockets: List[socket.socket] = []
        self._started: bool = False
        self._stopped: bool = False
        self.max_buffer_size = max_buffer_size
        self.read_chunk_size = read_chunk_size
        if self.ssl_options is not None and isinstance(self.ssl_options, dict):
            if 'certfile' not in self.ssl_options:
                raise KeyError('missing key "certfile" in ssl_options')
            if not os.path.exists(self.ssl_options['certfile']):
                raise ValueError('certfile "%s" does not exist' % self.ssl_options['certfile'])
            if 'keyfile' in self.ssl_options and (not os.path.exists(self.ssl_options['keyfile'])):
                raise ValueError('keyfile "%s" does not exist' % self.ssl_options['keyfile'])

    def listen(
        self,
        port: int,
        address: Optional[str] = None,
        family: socket.AddressFamily = socket.AF_UNSPEC,
        backlog: int = _DEFAULT_BACKLOG,
        flags: Optional[int] = None,
        reuse_port: bool = False
    ) -> None:
        sockets = bind_sockets(port, address=address, family=family, backlog=backlog, flags=flags, reuse_port=reuse_port)
        self.add_sockets(sockets)

    def add_sockets(self, sockets: Iterable[socket.socket]) -> None:
        for sock in sockets:
            self._sockets[sock.fileno()] = sock
            self._handlers[sock.fileno()] = add_accept_handler(sock, self._handle_connection)

    def add_socket(self, socket: socket.socket) -> None:
        self.add_sockets([socket])

    def bind(
        self,
        port: int,
        address: Optional[str] = None,
        family: socket.AddressFamily = socket.AF_UNSPEC,
        backlog: int = _DEFAULT_BACKLOG,
        flags: Optional[int] = None,
        reuse_port: bool = False
    ) -> None:
        sockets = bind_sockets(port, address=address, family=family, backlog=backlog, flags=flags, reuse_port=reuse_port)
        if self._started:
            self.add_sockets(sockets)
        else:
            self._pending_sockets.extend(sockets)

    def start(self, num_processes: int = 1, max_restarts: Optional[int] = None) -> None:
        assert not self._started
        self._started = True
        if num_processes != 1:
            process.fork_processes(num_processes, max_restarts)
        sockets = self._pending_sockets
        self._pending_sockets = []
        self.add_sockets(sockets)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for fd, sock in self._sockets.items():
            assert sock.fileno() == fd
            self._handlers.pop(fd)()
            sock.close()

    def handle_stream(self, stream: Union[IOStream, SSLIOStream], address: Any) -> Optional[Awaitable[None]]:
        raise NotImplementedError()

    def _handle_connection(self, connection: socket.socket, address: Any) -> None:
        if self.ssl_options is not None:
            assert ssl, 'OpenSSL required for SSL'
            try:
                connection = ssl_wrap_socket(connection, self.ssl_options, server_side=True, do_handshake_on_connect=False)
            except ssl.SSLError as err:
                if err.args[0] == ssl.SSL_ERROR_EOF:
                    return connection.close()
                else:
                    raise
            except OSError as err:
                if errno_from_exception(err) in (errno.ECONNABORTED, errno.EINVAL):
                    return connection.close()
                else:
                    raise
        try:
            if self.ssl_options is not None:
                stream = SSLIOStream(connection, max_buffer_size=self.max_buffer_size, read_chunk_size=self.read_chunk_size)
            else:
                stream = IOStream(connection, max_buffer_size=self.max_buffer_size, read_chunk_size=self.read_chunk_size)
            future = self.handle_stream(stream, address)
            if future is not None:
                IOLoop.current().add_future(gen.convert_yielded(future), lambda f: f.result())
        except Exception:
            app_log.error('Error in connection callback', exc_info=True)
