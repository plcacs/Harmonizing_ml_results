    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def initialize(self, request_callback: Callable[[HTTPRequest], Awaitable[None]], no_keep_alive: bool = False, xheaders: bool = False, ssl_options: Union[ssl.SSLContext, Dict[str, Any], None] = None, protocol: Optional[str] = None, decompress_request: bool = False, chunk_size: Optional[int] = None, max_header_size: Optional[int] = None, idle_connection_timeout: Optional[int] = None, body_timeout: Optional[int] = None, max_body_size: Optional[int] = None, max_buffer_size: Optional[int] = None, trusted_downstream: Optional[List[str]] = None) -> None:

    def handle_stream(self, stream: iostream.IOStream, address: Tuple[str, int]) -> None:

    def start_request(self, server_conn: HTTP1ServerConnection, request_conn: HTTP1ConnectionParameters) -> Union[httputil.HTTPMessageDelegate, _CallableAdapter]:

    def on_close(self, server_conn: HTTP1ServerConnection) -> None:

    def headers_received(self, start_line: httputil.RequestStartLine, headers: httputil.HTTPHeaders) -> None:

    def data_received(self, chunk: bytes) -> None:

    def finish(self) -> None:

    def on_connection_close(self) -> None:

    def __init__(self, stream: iostream.IOStream, address: Tuple[str, int], protocol: Optional[str], trusted_downstream: Optional[List[str]]) -> None:

    def _apply_xheaders(self, headers: httputil.HTTPHeaders) -> None:

    def _unapply_xheaders(self) -> None:

    def __init__(self, delegate: httputil.HTTPMessageDelegate, request_conn: HTTP1ConnectionParameters) -> None:

    def headers_received(self, start_line: httputil.RequestStartLine, headers: httputil.HTTPHeaders) -> None:

    def data_received(self, chunk: bytes) -> None:

    def finish(self) -> None:

    def on_connection_close(self) -> None:
