import threading
import socket
import select

def consume_socket_content(sock: Union[int, list[typing.Deque], typing.Sequence[bytes]], timeout: float=0.5) -> bytes:
    chunks = 65536
    content = b''
    while True:
        more_to_read = select.select([sock], [], [], timeout)[0]
        if not more_to_read:
            break
        new_content = sock.recv(chunks)
        if not new_content:
            break
        content += new_content
    return content

class Server(threading.Thread):
    """Dummy server using for unit testing"""
    WAIT_EVENT_TIMEOUT = 5

    def __init__(self, handler: Union[None, int, str]=None, host: typing.Text='localhost', port: int=0, requests_to_handle: int=1, wait_to_close_event: Union[None, int, float]=None) -> None:
        super(Server, self).__init__()
        self.handler = handler or consume_socket_content
        self.handler_results = []
        self.host = host
        self.port = port
        self.requests_to_handle = requests_to_handle
        self.wait_to_close_event = wait_to_close_event
        self.ready_event = threading.Event()
        self.stop_event = threading.Event()

    @classmethod
    def text_response_server(cls: Union[int, str, None], text: Union[int, str, None], request_timeout: float=0.5, **kwargs) -> Server:

        def text_response_handler(sock: Any):
            request_content = consume_socket_content(sock, timeout=request_timeout)
            sock.send(text.encode('utf-8'))
            return request_content
        return Server(text_response_handler, **kwargs)

    @classmethod
    def basic_response_server(cls: typing.Type, **kwargs):
        return cls.text_response_server('HTTP/1.1 200 OK\r\n' + 'Content-Length: 0\r\n\r\n', **kwargs)

    def run(self) -> None:
        try:
            self.server_sock = self._create_socket_and_bind()
            self.port = self.server_sock.getsockname()[1]
            self.ready_event.set()
            self._handle_requests()
            if self.wait_to_close_event:
                self.wait_to_close_event.wait(self.WAIT_EVENT_TIMEOUT)
        finally:
            self.ready_event.set()
            self._close_server_sock_ignore_errors()
            self.stop_event.set()

    def _create_socket_and_bind(self):
        sock = socket.socket()
        sock.bind((self.host, self.port))
        sock.listen(0)
        return sock

    def _close_server_sock_ignore_errors(self) -> None:
        try:
            self.server_sock.close()
        except IOError:
            pass

    def _handle_requests(self) -> None:
        for _ in range(self.requests_to_handle):
            sock = self._accept_connection()
            if not sock:
                break
            handler_result = self.handler(sock)
            self.handler_results.append(handler_result)

    def _accept_connection(self) -> Union[None, str]:
        try:
            ready, _, _ = select.select([self.server_sock], [], [], self.WAIT_EVENT_TIMEOUT)
            if not ready:
                return None
            return self.server_sock.accept()[0]
        except (select.error, socket.error):
            return None

    def __enter__(self) -> tuple:
        self.start()
        self.ready_event.wait(self.WAIT_EVENT_TIMEOUT)
        return (self.host, self.port)

    def __exit__(self, exc_type: Union[typing.Type, None, Exception], exc_value: Union[Exception, None, BaseException, typing.Type], traceback: Union[Exception, None, BaseException, typing.Type]) -> bool:
        if exc_type is None:
            self.stop_event.wait(self.WAIT_EVENT_TIMEOUT)
        elif self.wait_to_close_event:
            self.wait_to_close_event.set()
        self._close_server_sock_ignore_errors()
        self.join()
        return False