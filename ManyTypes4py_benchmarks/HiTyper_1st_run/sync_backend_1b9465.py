import errno
import socket
import ssl
from ..util.connection import create_connection
from ..util.ssl_ import ssl_wrap_socket
from ..util import selectors
from ._common import DEFAULT_SELECTOR, is_readable, LoopAbort
__all__ = ['SyncBackend']
BUFSIZE = 65536

class SyncBackend(object):

    def __init__(self, connect_timeout=None, read_timeout: Any=None) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout

    def connect(self, host: Union[str, int], port: Union[str, int], source_address: Union[None, str, int]=None, socket_options: Union[None, str, int]=None) -> SyncSocket:
        conn = create_connection((host, port), self._connect_timeout, source_address=source_address, socket_options=socket_options)
        return SyncSocket(conn, self._read_timeout)

class SyncSocket(object):

    def __init__(self, sock: int, read_timeout: Any) -> None:
        self._sock = sock
        self._read_timeout = read_timeout
        self._sock.setblocking(False)

    def start_tls(self, server_hostname: Union[str, typing.Callable], ssl_context: Union[str, typing.Callable]) -> SyncSocket:
        self._sock.setblocking(True)
        wrapped = ssl_wrap_socket(self._sock, server_hostname=server_hostname, ssl_context=ssl_context)
        wrapped.setblocking(False)
        return SyncSocket(wrapped, self._read_timeout)

    def getpeercert(self, binary: bool=False) -> Union[str, bytes]:
        return self._sock.getpeercert(binary_form=binary)

    def _wait(self, readable: Union[bool, list[str]], writable: Union[bool, list[str]]) -> tuple:
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
        return (event & selectors.EVENT_READ, event & selectors.EVENT_WRITE)

    def receive_some(self) -> Union[bytes, int, typing.Sequence[str]]:
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

    def send_and_receive_for_a_while(self, produce_bytes: Union[bytes, typing.IO, int], consume_bytes: Union[int, bytes, typing.IO]) -> None:
        outgoing_finished = False
        outgoing = b''
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
                    consume_bytes(incoming)
                if not outgoing_finished:
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
                if want_read or want_write:
                    self._wait(want_read, want_write)
        except LoopAbort:
            pass

    def forceful_close(self) -> None:
        self._sock.close()

    def is_readable(self):
        return is_readable(self._sock)

    def set_readable_watch_state(self, enabled: bool) -> None:
        pass