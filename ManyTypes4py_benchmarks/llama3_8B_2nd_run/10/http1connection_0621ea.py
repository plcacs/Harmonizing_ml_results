import asyncio
import logging
import re
import types
from tornado.concurrent import Future, future_add_done_callback, future_set_result_unless_cancelled
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple

class _QuietException(Exception):
    """Used to catch and log exceptions in the event that they are not caught by the delegate."""
    def __init__(self):
        pass

class HTTP1ConnectionParameters:
    """Parameters for `.HTTP1Connection` and `.HTTP1ServerConnection`."""
    def __init__(self, no_keep_alive: bool = False, chunk_size: int = None, max_header_size: int = None, header_timeout: float = None, max_body_size: int = None, body_timeout: float = None, decompress: bool = False) -> None:
        """
        :param bool no_keep_alive: If true, always close the connection after one request.
        :param int chunk_size: how much data to read into memory at once
        :param int max_header_size:  maximum amount of data for HTTP headers
        :param float header_timeout: how long to wait for all headers (seconds)
        :param int max_body_size: maximum amount of data for body
        :param float body_timeout: how long to wait while reading body (seconds)
        :param bool decompress: if true, decode incoming `Content-Encoding: gzip`
        """
        self.no_keep_alive = no_keep_alive
        self.chunk_size = chunk_size or 65536
        self.max_header_size = max_header_size or 65536
        self.header_timeout = header_timeout
        self.max_body_size = max_body_size
        self.body_timeout = body_timeout
        self.decompress = decompress

class HTTP1Connection(httputil.HTTPConnection):
    """Implements the HTTP/1.x protocol."""
    def __init__(self, stream: iostream.IOStream, is_client: bool, params: HTTP1ConnectionParameters = None, context: Optional[object] = None) -> None:
        """
        :param stream: an `.IOStream`
        :param bool is_client: client or server
        :param params: a `.HTTP1ConnectionParameters` instance or `None`
        :param context: an opaque application-defined object that can be accessed as `connection.context`
        """
        self.is_client = is_client
        self.stream = stream
        if params is None:
            params = HTTP1ConnectionParameters()
        self.params = params
        self.context = context
        self.no_keep_alive = params.no_keep_alive
        self._max_body_size = self.params.max_body_size if self.params.max_body_size is not None else self.stream.max_buffer_size
        self._body_timeout = self.params.body_timeout
        self._write_finished = False
        self._read_finished = False
        self._finish_future = Future()
        self._disconnect_on_finish = False
        self._clear_callbacks()
        self._request_start_line = None
        self._response_start_line = None
        self._request_headers = None
        self._chunking_output = False
        self._expected_content_remaining = None
        self._pending_write = None

    # ... rest of the code ...

def is_transfer_encoding_chunked(headers: httputil.HTTPHeaders) -> bool:
    """Returns true if the headers specify Transfer-Encoding: chunked.

    Raise httputil.HTTPInputError if any other transfer encoding is used.
    """
    if 'Transfer-Encoding' not in headers:
        return False
    if 'Content-Length' in headers:
        raise httputil.HTTPInputError('Message with both Transfer-Encoding and Content-Length')
    if headers.get('Transfer-Encoding', '').lower() == 'chunked':
        return True
    raise httputil.HTTPInputError('Unsupported Transfer-Encoding %s' % headers['Transfer-Encoding'])

def parse_int(s: str) -> int:
    """Parse a non-negative integer from a string."""
    if DIGITS.fullmatch(s) is None:
        raise ValueError('not an integer: %r' % s)
    return int(s)

def parse_hex_int(s: str) -> int:
    """Parse a non-negative hexadecimal integer from a string."""
    if HEXDIGITS.fullmatch(s) is None:
        raise ValueError('not a hexadecimal integer: %r' % s)
    return int(s, 16)
