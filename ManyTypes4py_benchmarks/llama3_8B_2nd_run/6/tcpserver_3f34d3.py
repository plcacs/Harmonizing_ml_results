class TCPServer:
    """A non-blocking, single-threaded TCP server."""

    def __init__(self, 
                 ssl_options: Optional[Union[ssl.SSLContext, dict]] = None, 
                 max_buffer_size: Optional[int] = None, 
                 read_chunk_size: Optional[int] = None) -> None:
        ...
