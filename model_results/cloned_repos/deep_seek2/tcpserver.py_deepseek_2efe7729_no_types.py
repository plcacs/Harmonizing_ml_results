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
from typing import Union, Dict, Any, Iterable, Optional, Awaitable, Callable, List, Tuple
if typing.TYPE_CHECKING:
    from typing import Callable, List

class TCPServer:
    """A non-blocking, single-threaded TCP server.

    To use `TCPServer`, define a subclass which overrides the `handle_stream`
    method. For example, a simple echo server could be defined like this::

      from tornado.tcpserver import TCPServer
      from tornado.iostream import StreamClosedError

      class EchoServer(TCPServer):
          async def handle_stream(self, stream, address):
              while True:
                  try:
                      data = await stream.read_until(b"\\n") await
                      stream.write(data)
                  except StreamClosedError:
                      break

    To make this server serve SSL traffic, send the ``ssl_options`` keyword
    argument with an `ssl.SSLContext` object. For compatibility with older
    versions of Python ``ssl_options`` may also be a dictionary of keyword
    arguments for the `ssl.SSLContext.wrap_socket` method.::

       ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
       ssl_ctx.load_cert_chain(os.path.join(data_dir, "mydomain.crt"),
                               os.path.join(data_dir, "mydomain.key"))
       TCPServer(ssl_options=ssl_ctx)

    `TCPServer` initialization follows one of three patterns:

    1. `listen`: single-process::

            async def main():
                server = TCPServer()
                server.listen(8888)
                await asyncio.Event().wait()

            asyncio.run(main())

       While this example does not create multiple processes on its own, when
       the ``reuse_port=True`` argument is passed to ``listen()`` you can run
       the program multiple times to create a multi-process service.

    2. `add_sockets`: multi-process::

            sockets = bind_sockets(8888)
            tornado.process.fork_processes(0)
            async def post_fork_main():
                server = TCPServer()
                server.add_sockets(sockets)
                await asyncio.Event().wait()
            asyncio.run(post_fork_main())

       The `add_sockets` interface is more complicated, but it can be used with
       `tornado.process.fork_processes` to run a multi-process service with all
       worker processes forked from a single parent.  `add_sockets` can also be
       used in single-process servers if you want to create your listening
       sockets in some way other than `~tornado.netutil.bind_sockets`.

       Note that when using this pattern, nothing that touches the event loop
       can be run before ``fork_processes``.

    3. `bind`/`start`: simple **deprecated** multi-process::

            server = TCPServer()
            server.bind(8888)
            server.start(0)  # Forks multiple sub-processes
            IOLoop.current().start()

       This pattern is deprecated because it requires interfaces in the
       `asyncio` module that have been deprecated since Python 3.10. Support for
       creating multiple processes in the ``start`` method will be removed in a
       future version of Tornado.

    .. versionadded:: 3.1
       The ``max_buffer_size`` argument.

    .. versionchanged:: 5.0
       The ``io_loop`` argument has been removed.
    """

    def __init__(self, ssl_options=None, max_buffer_size=None, read_chunk_size=None):
        self.ssl_options = ssl_options
        self._sockets: Dict[int, socket.socket] = {}
        self._handlers: Dict[int, Callable[[], None]] = {}
        self._pending_sockets: List[socket.socket] = []
        self._started = False
        self._stopped = False
        self.max_buffer_size = max_buffer_size
        self.read_chunk_size = read_chunk_size
        if self.ssl_options is not None and isinstance(self.ssl_options, dict):
            if 'certfile' not in self.ssl_options:
                raise KeyError('missing key "certfile" in ssl_options')
            if not os.path.exists(self.ssl_options['certfile']):
                raise ValueError('certfile "%s" does not exist' % self.ssl_options['certfile'])
            if 'keyfile' in self.ssl_options and (not os.path.exists(self.ssl_options['keyfile'])):
                raise ValueError('keyfile "%s" does not exist' % self.ssl_options['keyfile'])

    def listen(self, port, address=None, family=socket.AF_UNSPEC, backlog=_DEFAULT_BACKLOG, flags=None, reuse_port=False):
        sockets = bind_sockets(port, address=address, family=family, backlog=backlog, flags=flags, reuse_port=reuse_port)
        self.add_sockets(sockets)

    def add_sockets(self, sockets):
        for sock in sockets:
            self._sockets[sock.fileno()] = sock
            self._handlers[sock.fileno()] = add_accept_handler(sock, self._handle_connection)

    def add_socket(self, socket):
        self.add_sockets([socket])

    def bind(self, port, address=None, family=socket.AF_UNSPEC, backlog=_DEFAULT_BACKLOG, flags=None, reuse_port=False):
        sockets = bind_sockets(port, address=address, family=family, backlog=backlog, flags=flags, reuse_port=reuse_port)
        if self._started:
            self.add_sockets(sockets)
        else:
            self._pending_sockets.extend(sockets)

    def start(self, num_processes=1, max_restarts=None):
        assert not self._started
        self._started = True
        if num_processes != 1:
            process.fork_processes(num_processes, max_restarts)
        sockets = self._pending_sockets
        self._pending_sockets = []
        self.add_sockets(sockets)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        for fd, sock in self._sockets.items():
            assert sock.fileno() == fd
            self._handlers.pop(fd)()
            sock.close()

    def handle_stream(self, stream, address):
        raise NotImplementedError()

    def _handle_connection(self, connection, address):
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
                stream: IOStream = SSLIOStream(connection, max_buffer_size=self.max_buffer_size, read_chunk_size=self.read_chunk_size)
            else:
                stream = IOStream(connection, max_buffer_size=self.max_buffer_size, read_chunk_size=self.read_chunk_size)
            future = self.handle_stream(stream, address)
            if future is not None:
                IOLoop.current().add_future(gen.convert_yielded(future), lambda f: f.result())
        except Exception:
            app_log.error('Error in connection callback', exc_info=True)