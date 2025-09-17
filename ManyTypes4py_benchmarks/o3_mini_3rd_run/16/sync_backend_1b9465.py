import errno
import socket
import ssl
from typing import Optional, Tuple, List, Callable, Union, Any
from ..util.connection import create_connection
from ..util.ssl_ import ssl_wrap_socket
from ..util import selectors
from ._common import DEFAULT_SELECTOR, is_readable, LoopAbort

__all__ = ['SyncBackend']
BUFSIZE: int = 65536

SocketOption = Tuple[int, int, int, Optional[int]]
SourceAddress = Optional[Tuple[str, int]]


class SyncBackend(object):
    def __init__(self, connect_timeout: Optional[float] = None, read_timeout: Optional[float] = None) -> None:
        self._connect_timeout: Optional[float] = connect_timeout
        self._read_timeout: Optional[float] = read_timeout

    def connect(
        self,
        host: str,
        port: int,
        source_address: SourceAddress = None,
        socket_options: Optional[List[SocketOption]] = None
    ) -> "SyncSocket":
        conn = create_connection(
            (host, port),
            self._connect_timeout,
            source_address=source_address,
            socket_options=socket_options
        )
        return SyncSocket(conn, self._read_timeout)


class SyncSocket(object):
    def __init__(self, sock: socket.socket, read_timeout: Optional[float]) -> None:
        self._sock: socket.socket = sock
        self._read_timeout: Optional[float] = read_timeout
        self._sock.setblocking(False)

    def start_tls(self, server_hostname: str, ssl_context: ssl.SSLContext) -> "SyncSocket":
        self._sock.setblocking(True)
        wrapped = ssl_wrap_socket(self._sock, server_hostname=server_hostname, ssl_context=ssl_context)
        wrapped.setblocking(False)
        return SyncSocket(wrapped, self._read_timeout)

    def getpeercert(self, binary: bool = False) -> Optional[Union[dict, bytes]]:
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
        consume_bytes: Callable[[bytes], Any]
    ) -> None:
        outgoing_finished: bool = False
        outgoing: Optional[memoryview] = None
        try:
            while True:
                if not outgoing_finished and (outgoing is None or not outgoing):
                    b: Optional[bytes] = produce_bytes()
                    if b is None:
                        outgoing = None
                        outgoing_finished = True
                    else:
                        outgoing = memoryview(b)
                want_read: bool = False
                want_write: bool = False
                try:
                    incoming: bytes = self._sock.recv(BUFSIZE)
                except ssl.SSLWantReadError:
                    want_read = True
                except ssl.SSLWantWriteError:
                    want_write = True
                except (OSError, socket.error) as exc:
                    if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                        want_read = True
                else:
                    consume_bytes(incoming)
                if not outgoing_finished and outgoing is not None:
                    try:
                        sent: int = self._sock.send(outgoing)
                        outgoing = outgoing[sent:]
                    except ssl.SSLWantReadError:
                        want_read = True
                    except ssl.SSLWantWriteError:
                        want_write = True
                    except (OSError, socket.error) as exc:
                        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                            want_write = True
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