#
# Copyright 2014 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Client and server implementations of HTTP/1.x.

.. versionadded:: 4.0
"""

import asyncio
import logging
import re
import types

from tornado.concurrent import (
    Future,
    future_add_done_callback,
    future_set_result_unless_cancelled,
)
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor


from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple, Any

CR_OR_LF_RE = re.compile(b"\r|\n")


class _QuietException(Exception):
    def __init__(self) -> None:
        pass


class _ExceptionLoggingContext:
    """Used with the ``with`` statement when calling delegate methods to
    log any exceptions with the given logger.  Any exceptions caught are
    converted to _QuietException
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        typ: Optional[Type[BaseException]],
        value: Optional[BaseException],
        tb: Optional[types.TracebackType],
    ) -> None:
        if value is not None:
            assert typ is not None
            self.logger.error("Uncaught exception", exc_info=(typ, value, tb))
            raise _QuietException


class HTTP1ConnectionParameters:
    """Parameters for `.HTTP1Connection` and `.HTTP1ServerConnection`."""

    def __init__(
        self,
        no_keep_alive: bool = False,
        chunk_size: Optional[int] = None,
        max_header_size: Optional[int] = None,
        header_timeout: Optional[float] = None,
        max_body_size: Optional[int] = None,
        body_timeout: Optional[float] = None,
        decompress: bool = False,
    ) -> None:
        """
        :arg bool no_keep_alive: If true, always close the connection after
            one request.
        :arg int chunk_size: how much data to read into memory at once
        :arg int max_header_size:  maximum amount of data for HTTP headers
        :arg float header_timeout: how long to wait for all headers (seconds)
        :arg int max_body_size: maximum amount of data for body
        :arg float body_timeout: how long to wait while reading body (seconds)
        :arg bool decompress: if true, decode incoming
            ``Content-Encoding: gzip``
        """
        self.no_keep_alive = no_keep_alive
        self.chunk_size = chunk_size or 65536
        self.max_header_size = max_header_size or 65536
        self.header_timeout = header_timeout
        self.max_body_size = max_body_size
        self.body_timeout = body_timeout
        self.decompress = decompress


class HTTP1Connection(httputil.HTTPConnection):
    """Implements the HTTP/1.x protocol.

    This class can be on its own for clients, or via `HTTP1ServerConnection`
    for servers.
    """

    def __init__(
        self,
        stream: iostream.IOStream,
        is_client: bool,
        params: Optional[HTTP1ConnectionParameters] = None,
        context: Optional[object] = None,
    ) -> None:
        """
        :arg stream: an `.IOStream`
        :arg bool is_client: client or server
        :arg params: a `.HTTP1ConnectionParameters` instance or ``None``
        :arg context: an opaque application-defined object that can be accessed
            as ``connection.context``.
        """
        self.is_client = is_client
        self.stream = stream
        if params is None:
            params = HTTP1ConnectionParameters()
        self.params = params
        self.context = context
        self.no_keep_alive = params.no_keep_alive
        # The body limits can be altered by the delegate, so save them
        # here instead of just referencing self.params later.
        self._max_body_size = (
            self.params.max_body_size
            if self.params.max_body_size is not None
            else self.stream.max_buffer_size
        )
        self._body_timeout = self.params.body_timeout
        # _write_finished is set to True when finish() has been called,
        # i.e. there will be no more data sent.  Data may still be in the
        # stream's write buffer.
        self._write_finished = False
        # True when we have read the entire incoming body.
        self._read_finished = False
        # _finish_future resolves when all data has been written and flushed
        # to the IOStream.
        self._finish_future = Future()  # type: Future[None]
        # If true, the connection should be closed after this request
        # (after the response has been written in the server side,
        # and after it has been read in the client)
        self._disconnect_on_finish = False
        self._clear_callbacks()
        # Save the start lines after we read or write them; they
        # affect later processing (e.g. 304 responses and HEAD methods
        # have content-length but no bodies)
        self._request_start_line = None  # type: Optional[httputil.RequestStartLine]
        self._response_start_line = None  # type: Optional[httputil.ResponseStartLine]
        self._request_headers = None  # type: Optional[httputil.HTTPHeaders]
        # True if we are writing output with chunked encoding.
        self._chunking_output = False
        # While reading a body with a content-length, this is the
        # amount left to read.
        self._expected_content_remaining = None  # type: Optional[int]
        # A Future for our outgoing writes, returned by IOStream.write.
        self._pending_write = None  # type: Optional[Future[None]]

    def read_response(self, delegate: httputil.HTTPMessageDelegate) -> Awaitable[bool]:
        """Read a single HTTP response.

        Typical client-mode usage is to write a request using `write_headers`,
        `write`, and `finish`, and then call ``read_response``.

        :arg delegate: a `.HTTPMessageDelegate`

        Returns a `.Future` that resolves to a bool after the full response has
        been read. The result is true if the stream is still open.
        """
        if self.params.decompress:
            delegate = _GzipMessageDelegate(delegate, self.params.chunk_size)
        return self._read_message(delegate)

    async def _read_message(self, delegate: httputil.HTTPMessageDelegate) -> bool:
        need_delegate_close = False
        try:
            header_future = self.stream.read_until_regex(
                b"\r?\n\r?\n", max_bytes=self.params.max_header_size
            )
            if self.params.header_timeout is None:
                header_data = await header_future
            else:
                try:
                    header_data = await gen.with_timeout(
                        self.stream.io_loop.time() + self.params.header_timeout,
                        header_future,
                        quiet_exceptions=iostream.StreamClosedError,
                    )
                except gen.TimeoutError:
                    self.close()
                    return False
            start_line_str, headers = self._parse_headers(header_data)
            if self.is_client:
                resp_start_line = httputil.parse_response_start_line(start_line_str)
                self._response_start_line = resp_start_line
                start_line = (
                    resp_start_line
                )  # type: Union[httputil.RequestStartLine, httputil.ResponseStartLine]
                # TODO: this will need to change to support client-side keepalive
                self._disconnect_on_finish = False
            else:
                req_start_line = httputil.parse_request_start_line(start_line_str)
                self._request_start_line = req_start_line
                self._request_headers = headers
                start_line = req_start_line
                self._disconnect_on_finish = not self._can_keep_alive(
                    req_start_line, headers
                )
            need_delegate_close = True
            with _ExceptionLoggingContext(app_log):
                header_recv_future = delegate.headers_received(start_line, headers)
                if header_recv_future is not None:
                    await header_recv_future
            if self.stream is None:
                # We've been detached.
                need_delegate_close = False
                return False
            skip_body = False
            if self.is_client:
                assert isinstance(start_line, httputil.ResponseStartLine)
                if (
                    self._request_start_line is not None
                    and self._request_start_line.method == "HEAD"
                ):
                    skip_body = True
                code = start_line.code
                if code == 304:
                    # 304 responses may include the content-length header
                    # but do not actually have a body.
                    # http://tools.ietf.org/html/rfc7230#section-3.3
                    skip_body = True
                if 100 <= code < 200:
                    # 1xx responses should never indicate the presence of
                    # a body.
                    if "Content-Length" in headers or "Transfer-Encoding" in headers:
                        raise httputil.HTTPInputError(
                            "Response code %d cannot have body" % code
                        )
                    # TODO: client delegates will get headers_received twice
                    # in the case of a 100-continue.  Document or change?
                    await self._read_message(delegate)
            else:
                if headers.get("Expect") == "100-continue" and not self._write_finished:
                    self.stream.write(b"HTTP/1.1 100 (Continue)\r\n\r\n")
            if not skip_body:
                body_future = self._read_body(
                    resp_start_line.code if self.is_client else 0, headers, delegate
                )
                if body_future is not None:
                    if self._body_timeout is None:
                        await body_future
                    else:
                        try:
                            await gen.with_timeout(
                                self.stream.io_loop.time() + self._body_timeout,
                                body_future,
                                quiet_exceptions=iostream.StreamClosedError,
                            )
                        except gen.TimeoutError:
                            gen_log.info("Timeout reading body from %s", self.context)
                            self.stream.close()
                            return False
            self._read_finished = True
            if not self._write_finished or self.is_client:
                need_delegate_close = False
                with _ExceptionLoggingContext(app_log):
                    delegate.finish()
            # If we're waiting for the application to produce an asynchronous
            # response, and we're not detached, register a close callback
            # on the stream (we didn't need one while we were reading)
            if (
                not self._finish_future.done()
                and self.stream is not None
                and not self.stream.closed()
            ):
                self.stream.set_close_callback(self._on_connection_close)
                await self._finish_future
            if self.is_client and self._disconnect_on_finish:
                self.close()
            if self.stream is None:
                return False
        except httputil.HTTPInputError as e:
            gen_log.info("Malformed HTTP message from %s: %s", self.context, e)
            if not self.is_client:
                await self.stream.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            self.close()
            return False
        finally:
            if need_delegate_close:
                with _ExceptionLoggingContext(app_log):
                    delegate.on_connection_close()
            header_future = None  # type: ignore
            self._clear_callbacks()
        return True

    def _clear_callbacks(self) -> None:
        """Clears the callback attributes.

        This allows the request handler to be garbage collected more
        quickly in CPython by breaking up reference cycles.
        """
        self._write_callback = None
        self._write_future = None  # type: Optional[Future[None]]
        self._close_callback = None  # type: Optional[Callable[[], None]]
        if self.stream is not None:
            self.stream.set_close_callback(None)

    def set_close_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Sets a callback that will be run when the connection is closed.

        Note that this callback is slightly different from
        `.HTTPMessageDelegate.on_connection_close`: The
        `.HTTPMessageDelegate` method is called when the connection is
        closed while receiving a message. This callback is used when
        there is not an active delegate (for example, on the server
        side this callback is used if the client closes the connection
        after sending its request but before receiving all the
        response.
        """
        self._close_callback = callback

    def _on_connection_close(self) -> None:
        # Note that this callback is only registered on the IOStream
        # when we have finished reading the request and are waiting for
        # the application to produce its response.
        if self._close_callback is not None:
            callback = self._close_callback
            self._close_callback = None
            callback()
        if not self._finish_future.done():
            future_set_result_unless_cancelled(self._finish_future, None)
        self._clear_callbacks()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
        self._clear_callbacks()
        if not self._finish_future.done():
            future_set_result_unless_cancelled(self._finish_future, None)

    def detach(self) -> iostream.IOStream:
        """Take control of the underlying stream.

        Returns the underlying `.IOStream` object and stops all further
        HTTP processing.  May only be called during
        `.HTTPMessageDelegate.headers_received`.  Intended for implementing
        protocols like websockets that tunnel over an HTTP handshake.
        """
        self._clear_callbacks()
        stream = self.stream
        self.stream = None  # type: ignore
        if not self._finish_future.done():
            future_set_result_unless_cancelled(self._finish_future, None)
        return stream

    def set_body_timeout(self, timeout: float) -> None:
        """Sets the body timeout for a single request.

        Overrides the value from `.HTTP1ConnectionParameters`.
        """
        self._body_timeout = timeout

    def set_max_body_size(self, max_body_size: int) -> None:
        """Sets the body size limit for a single request.

        Overrides the value from `.HTTP1ConnectionParameters`.
        """
        self._max_body_size = max_body_size

    def write_headers(
        self,
        start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine],
        headers: httputil.HTTPHeaders,
        chunk: Optional[bytes] = None,
    ) -> "Future[None]":
        """Implements `.HTTPConnection.write_headers`."""
        lines = []
        if self.is_client:
            assert isinstance(start_line, httputil.RequestStartLine)
            self._request_start_line = start_line
            lines.append(utf8(f"{start_line[0]} {start_line[1]} HTTP/1.1"))
            # Client requests with a non-empty body must have either a
            # Content-Length or a Transfer-Encoding. If Content-Length is not
            # present we'll add our Transfer-Encoding below.
            self._chunking_output = (
                start_line.method in ("POST", "PUT", "PATCH")
                and "Content-Length" not in headers
            )
        else:
            assert isinstance(start_line, httputil.ResponseStartLine)
            assert self._request_start_line is not None
            assert self._request_headers is not None
            self._response_start_line = start_line
            lines.append(utf8("HTTP/1.1 %d %s" % (start_line[1], start_line[2])))
            self._chunking_output = (
                # TODO: should this use
                # self._request_start_line.version or
                # start_line.version?
                self._request_start_line.version == "HTTP/1.1"
                # Omit payload header field for HEAD request.
                and self._request_start_line.method != "HEAD"
                # 1xx, 204 and 304 responses have no body (not even a zero-length
                # body), and so should not have either Content-Length or
                # Transfer-Encoding headers.
                and start_line.code not in (204, 304)
                and (start_line.code < 100 or start_line.code >= 200)
                # No need to chunk the output if a Content-Length is specified.
                and "Content-Length" not in headers
            )
            # If connection to a 1.1 client will be closed, inform client
            if (
                self._request_start_line.version == "HTTP/1.1"
                and self._disconnect_on_finish
            ):
                headers["Connection"] = "close"
            # If a 1.0 client asked for keep-alive, add the header.
            if (
                self._request_start_line.version == "HTTP/1.0"
                and self._request_headers.get("Connection", "").lower() == "keep-alive"
            ):
                headers["Connection"] = "Keep-Alive"
        if self._chunking_output:
            headers["Transfer-Encoding"] = "chunked"
        if not self.is_client and (
            self._request_start_line.method == "HEAD"
            or cast(httputil.ResponseStartLine, start_line).code == 304
        ):
            self._expected_content_remaining = 0
        elif "Content-Length" in headers:
            self._expected_content_remaining = parse_int(headers["Content-Length"])
        else:
            self._expected_content_remaining = None