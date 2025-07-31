import errno
import socket
import ssl
from typing import Optional, Tuple, Callable, Union, Any
from ..util.connection import create_connection
from ..util.ssl_ import ssl_wrap_socket
from ..util import selectors
from ._common import DEFAULT_SELECTOR, is_readable, LoopAbort

__all__ = ['SyncBackend']
BUFSIZE = 65536

class SyncBackend(object):
    def __init__(self, connect_timeout: Optional[float] = None, read_timeout: Optional[float] = None) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout

    def connect(self, host: str, port: int, source_address: Optional[Tuple[str, int]] = None, socket_options: Optional[Any] = None) -> "SyncSocket":
        conn: socket.socket = create_connection(
            (host, port),
            self._connect_timeout,
            source_address=source_address,
            socket_options=socket_options
        )
        return SyncSocket(conn, self._read_timeout)

class SyncSocket(object):
    def __init__(self, sock: socket.socket, read_timeout: Optional[float]) -> None:
        self._sock = sock
        self._read_timeout = read_timeout
        self._sock.setblocking(False)

    def start_tls(self, server_hostname: str, ssl_context: ssl.SSLContext) -> "SyncSocket":
        self._sock.setblocking(True)
        wrapped: socket.socket = ssl_wrap_socket(self._sock, server_hostname=server_hostname, ssl_context=ssl_context)
        wrapped.setblocking(False)
        return SyncSocket(wrapped, self._read_timeout)

    def getpeercert(self, binary: bool = False) -> Union[dict, bytes]:
        return self._sock.getpeercert(binary_form=binary)

    def _wait(self, readable: bool, writable: bool) -> Tuple[bool, bool]:
        assert readable or writable
        s = DEFAULT_SELECTOR()
        flags = 0
        if readable:
            flags |= selectors.EVENT_READ
        if writable:
            flags |= selectors.EVENT_WRITE
        s.register(self._sock, flags)
        events = s.select(timeout=self._read_timeout)
        if not events:
            raise socket.timeout('XX FIXME timeout happened')
        _, event = events[0]
        return (bool(event & selectors.EVENT_READ), bool(event & selectors.EVENT_WRITE))

    def receive_some(self) -> bytes:
        while True:
            try:
                return self._sock.recv(BUFSIZE)
            except ssl.SSLWantReadError:
                self._wait(readable=True, writable=False)
            except ssl.SSLWantWriteError:
                self._wait(readable=False, writable=True)
            except (OSError, socket.error) as exc:
                if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    self._wait(readable=True, writable=False)
                else:
                    raise

    def send_and_receive_for_a_while(
        self,
        produce_bytes: Callable[[], Optional[bytes]],
        consume_bytes: Callable[[bytes], None]
    ) -> None:
        outgoing_finished = False
        outgoing: Optional[memoryview] = b''
        try:
            while True:
                if not outgoing_finished and (not outgoing):
                    b = produce_bytes()
                    if b is None:
                        outgoing = None
                        outgoing_finished = True
                    else:
                        outgoing = memoryview(b)
                want_read = False
                want_write = False
                try:
                    incoming = self._sock.recv(BUFSIZE)
                except ssl.SSLWantReadError:
                    want_read = True
                except ssl.SSLWantWriteError:
                    want_write = True
                except (OSError, socket.error) as exc:
                    if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                        want_read = True
                    else:
                        raise
                else:
                    consume_bytes(incoming)
                if not outgoing_finished and outgoing is not None:
                    try:
                        sent = self._sock.send(outgoing)
                        outgoing = outgoing[sent:]
                    except ssl.SSLWantReadError:
                        want_read = True
                    except ssl.SSLWantWriteError:
                        want_write = True
                    except (OSError, socket.error) as exc:
                        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                            want_write = True
                        else:
                            raise
                if want_read or want_write:
                    self._wait(want_read, want_write)
        except LoopAbort:
            pass

    def forceful_close(self) -> None:
        self._sock.close()

    def is_readable(self) -> bool:
        return is_readable(self._sock)

    def set_readable_watch_state(self, enabled: bool) -> None:
        pass