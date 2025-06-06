from __future__ import absolute_import
from contextlib import contextmanager
import zlib
import io
import logging
from socket import timeout as SocketTimeout
from socket import error as SocketError

import h11

from .._collections import HTTPHeaderDict
from ..exceptions import ProtocolError, DecodeError, ReadTimeoutError
from ..packages.six import string_types as basestring, binary_type
from ..util.ssl_ import BaseSSLError

from typing import Optional, Union, Generator, Any, Iterator, IO

log = logging.getLogger(__name__)


class DeflateDecoder:
    def __init__(self) -> None:
        self._first_try: bool = True
        self._data: Optional[binary_type] = binary_type()
        self._obj: zlib.decompressobj = zlib.decompressobj()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def decompress(self, data: binary_type) -> binary_type:
        if not data:
            return data

        if not self._first_try:
            return self._obj.decompress(data)

        if self._data is not None:
            self._data += data
        try:
            decompressed = self._obj.decompress(data)
            if decompressed:
                self._first_try = False
                self._data = None
            return decompressed

        except zlib.error:
            self._first_try = False
            self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                if self._data is not None:
                    return self.decompress(self._data)
                return self.decompress(data)
            finally:
                self._data = None


class GzipDecoder:
    def __init__(self) -> None:
        self._obj: zlib.decompressobj = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def decompress(self, data: binary_type) -> binary_type:
        if not data:
            return data

        return self._obj.decompress(data)


def _get_decoder(mode: str) -> Union[GzipDecoder, DeflateDecoder]:
    if mode == "gzip":
        return GzipDecoder()

    return DeflateDecoder()


class HTTPResponse(io.IOBase):
    """
    HTTP Response container.

    Backwards-compatible to httplib's HTTPResponse but the response ``body`` is
    loaded and decoded on-demand when the ``data`` property is accessed.  This
    class is also compatible with the Python standard library's :mod:`io`
    module, and can hence be treated as a readable object in the context of that
    framework.

    Extra parameters for behaviour not present in httplib.HTTPResponse:

    :param preload_content:
        If True, the response's body will be preloaded during construction.

    :param decode_content:
        If True, attempts to decode specific content-encoding's based on headers
        (like 'gzip' and 'deflate') will be skipped and raw data will be used
        instead.

    :param retries:
        The retries contains the last :class:`~urllib3.util.retry.Retry` that
        was used during the request.
    """

    CONTENT_DECODERS: list = ["gzip", "deflate"]
    REDIRECT_STATUSES: list = [301, 302, 303, 307, 308]

    def __init__(
        self,
        body: Union[binary_type, IO[bytes], None] = "",
        headers: Optional[Union[HTTPHeaderDict, dict]] = None,
        status: int = 0,
        version: int = 0,
        reason: Optional[str] = None,
        strict: int = 0,
        preload_content: bool = True,
        decode_content: bool = True,
        original_response: Optional[Any] = None,
        pool: Optional[Any] = None,
        connection: Optional[Any] = None,
        retries: Optional[Any] = None,
        request_method: Optional[str] = None,
    ) -> None:
        if isinstance(headers, HTTPHeaderDict):
            self.headers: HTTPHeaderDict = headers
        else:
            self.headers = HTTPHeaderDict(headers)
        self.status: int = status
        self.version: int = version
        self.reason: Optional[str] = reason
        self.strict: int = strict
        self.decode_content: bool = decode_content
        self.retries: Optional[Any] = retries
        self._decoder: Optional[Union[GzipDecoder, DeflateDecoder]] = None
        self._body: Optional[Union[binary_type, str]] = None
        self._fp: Optional[Union[IO[bytes], binary_type]] = None
        self._original_response: Optional[Any] = original_response
        self._fp_bytes_read: int = 0
        self._buffer: binary_type = b""
        if body and isinstance(body, (basestring, binary_type)):
            self._body = body
        else:
            self._fp = body
        self._pool: Optional[Any] = pool
        self._connection: Optional[Any] = connection
        # If requested, preload the body.
        if preload_content and not self._body:
            self._body = self.read(decode_content=decode_content)

    def get_redirect_location(self) -> Union[Optional[str], bool]:
        """
        Should we redirect and where to?

        :returns: Truthy redirect location string if we got a redirect status
            code and valid location. ``None`` if redirect status and no
            location. ``False`` if not a redirect status code.
        """
        if self.status in self.REDIRECT_STATUSES:
            return self.headers.get("location")

        return False

    def release_conn(self) -> None:
        if not self._pool or not self._connection:
            return

        self._pool._put_conn(self._connection)
        self._connection = None

    @property
    def data(self) -> Optional[Union[binary_type, str]]:
        # For backwards-compat with earlier urllib3 0.4 and earlier.
        if self._body is not None:
            return self._body

        if self._fp:
            return self.read(cache_content=True)

    @property
    def connection(self) -> Optional[Any]:
        return self._connection

    def tell(self) -> int:
        """
        Obtain the number of bytes pulled over the wire so far. May differ from
        the amount of content returned by :meth:`HTTPResponse.read` if bytes
        are encoded on the wire (e.g, compressed).
        """
        return self._fp_bytes_read

    def _init_decoder(self) -> None:
        """
        Set-up the _decoder attribute if necessary.
        """
        # Note: content-encoding value should be case-insensitive, per RFC 7230
        # Section 3.2
        content_encoding: str = self.headers.get("content-encoding", "").lower()
        if self._decoder is None and content_encoding in self.CONTENT_DECODERS:
            self._decoder = _get_decoder(content_encoding)

    def _decode(
        self, data: binary_type, decode_content: bool, flush_decoder: bool
    ) -> binary_type:
        """
        Decode the data passed in and potentially flush the decoder.
        """
        try:
            if decode_content and self._decoder:
                data = self._decoder.decompress(data)
        except (IOError, zlib.error) as e:
            content_encoding: str = self.headers.get("content-encoding", "").lower()
            raise DecodeError(
                "Received response with content-encoding: %s, but "
                "failed to decode it." % content_encoding,
                e,
            )

        if flush_decoder and decode_content:
            data += self._flush_decoder()
        return data

    def _flush_decoder(self) -> binary_type:
        """
        Flushes the decoder. Should only be called if the decoder is actually
        being used.
        """
        if self._decoder:
            buf = self._decoder.decompress(b"")
            return buf + self._decoder.flush()

        return b""

    @contextmanager
    def _error_catcher(self) -> Generator[None, None, None]:
        """
        Catch low-level python exceptions, instead re-raising urllib3
        variants, so that low-level exceptions are not leaked in the
        high-level api.

        On exit, release the connection back to the pool.
        """
        clean_exit: bool = False
        try:
            try:
                yield

            except SocketTimeout:
                # FIXME: Ideally we'd like to include the url in the ReadTimeoutError but
                # there is yet no clean way to get at it from this context.
                raise ReadTimeoutError(self._pool, None, "Read timed out.")

            except BaseSSLError as e:
                # FIXME: Is there a better way to differentiate between SSLErrors?
                if "read operation timed out" not in str(e):  # Defensive:
                    # This shouldn't happen but just in case we're missing an edge
                    # case, let's avoid swallowing SSL errors.
                    raise

                raise ReadTimeoutError(self._pool, None, "Read timed out.")

            except (h11.ProtocolError, SocketError) as e:
                # This includes IncompleteRead.
                raise ProtocolError("Connection broken: %r" % e, e)

            except GeneratorExit:
                # We swallow GeneratorExit when it is emitted: this allows the
                # use of the error checker inside stream()
                pass
            # If no exception is thrown, we should avoid cleaning up
            # unnecessarily.
            clean_exit = True
        finally:
            # If we didn't terminate cleanly, we need to throw away our
            # connection.
            if not clean_exit:
                self.close()
            # If we hold the original response but it's finished now, we should
            # return the connection back to the pool.
            # XXX
            if False and self._original_response and getattr(self._original_response, "complete", False):
                self.release_conn()

    def read(
        self,
        amt: Optional[int] = None,
        decode_content: Optional[bool] = None,
        cache_content: bool = False,
    ) -> binary_type:
        """
        Similar to :meth:`httplib.HTTPResponse.read`, but with two additional
        parameters: ``decode_content`` and ``cache_content``.

        :param amt:
            How much of the content to read. If specified, caching is skipped
            because it doesn't make sense to cache partial content as the full
            response.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.

        :param cache_content:
            If True, will save the returned data such that the same result is
            returned despite of the state of the underlying file object. This
            is useful if you want the ``.data`` property to continue working
            after having ``.read()`` the file object. (Overridden if ``amt`` is
            set.)
        """
        # TODO: refactor this method to better handle buffered output.
        # This method is a weird one. We treat this read() like a buffered
        # read, meaning that it never reads "short" unless there is an EOF
        # condition at work. However, we have a decompressor in play here,
        # which means our read() returns decompressed data.
        #
        # This means the buffer can only meaningfully buffer decompressed data.
        # This makes this method prone to over-reading, and forcing too much
        # data into the buffer. That's unfortunate, but right now I'm not smart
        # enough to come up with a way to solve that problem.
        if self._fp is None and not self._buffer:
            return b""

        data: binary_type = self._buffer
        with self._error_catcher():
            if amt is None:
                chunks: list = []
                for chunk in self.stream(decode_content):
                    chunks.append(chunk)
                data += b"".join(chunks)
                self._buffer = b""
                # We only cache the body data for simple read calls.
                self._body = data
            else:
                data_len: int = len(data)
                chunks = [data]
                streamer = self.stream(decode_content)
                while data_len < amt:
                    try:
                        chunk: binary_type = next(streamer)
                    except StopIteration:
                        break

                    else:
                        chunks.append(chunk)
                        data_len += len(chunk)
                data = b"".join(chunks)
                self._buffer = data[amt:]
                data = data[:amt]
        return data

    def stream(
        self, decode_content: Optional[bool] = None
    ) -> Iterator[binary_type]:
        """
        A generator wrapper for the read() method.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        # Short-circuit evaluation for exhausted responses.
        if self._fp is None:
            return

        self._init_decoder()
        if decode_content is None:
            decode_content = self.decode_content
        with self._error_catcher():
            for raw_chunk in self._fp:
                self._fp_bytes_read += len(raw_chunk)
                decoded_chunk = self._decode(
                    raw_chunk, decode_content, flush_decoder=False
                )
                if decoded_chunk:
                    yield decoded_chunk

            # This branch is speculative: most decoders do not need to flush,
            # and so this produces no output. However, it's here because
            # anecdotally some platforms on which we do not test (like Jython)
            # do require the flush. For this reason, we exclude this from code
            # coverage. Happily, the code here is so simple that testing the
            # branch we don't enter is basically entirely unnecessary (it's
            # just a yield statement).
            final_chunk = self._decode(b"", decode_content, flush_decoder=True)
            if final_chunk:  # Platform-specific: Jython
                yield final_chunk

            self._fp = None

    @classmethod
    def from_base(cls, r: Any, **response_kw: Any) -> 'HTTPResponse':
        """
        Given an :class:`urllib3.base.Response` instance ``r``, return a
        corresponding :class:`urllib3.response.HTTPResponse` object.

        Remaining parameters are passed to the HTTPResponse constructor, along
        with ``original_response=r``.
        """
        # TODO: Huge hack.
        for kw in ("redirect", "assert_same_host", "enforce_content_length"):
            if kw in response_kw:
                response_kw.pop(kw)

        resp = cls(
            body=r.body,
            headers=r.headers,
            status=r.status_code,
            version=r.version,
            original_response=r,
            connection=r.body,  # type: ignore
            **response_kw
        )
        return resp

    # Backwards-compatibility methods for httplib.HTTPResponse
    def getheaders(self) -> HTTPHeaderDict:
        return self.headers

    def getheader(self, name: str, default: Optional[Any] = None) -> Optional[Any]:
        return self.headers.get(name, default)

    # Backwards compatibility for http.cookiejar
    def info(self) -> HTTPHeaderDict:
        return self.headers

    # Overrides from io.IOBase
    def close(self) -> None:
        if not self.closed:
            if self._fp is not None:
                self._fp.close()  # type: ignore
            self._buffer = b""
            self._fp = None
        if self._connection:
            self._connection.close()

    @property
    def closed(self) -> bool:
        # This method is required for `io` module compatibility.
        if self._fp is None and not self._buffer:
            return True

        elif hasattr(self._fp, "complete"):
            return self._fp.complete  # type: ignore

        else:
            return False

    def fileno(self) -> int:
        # This method is required for `io` module compatibility.
        if self._fp is None:
            raise IOError("HTTPResponse has no file to get a fileno from")

        elif hasattr(self._fp, "fileno"):
            return self._fp.fileno()  # type: ignore

        else:
            raise IOError(
                "The file-like object this HTTPResponse is wrapped "
                "around has no file descriptor"
            )

    def readable(self) -> bool:
        # This method is required for `io` module compatibility.
        return True

    def readinto(self, b: bytearray) -> int:
        # This method is required for `io` module compatibility.
        temp = self.read(len(b))

        if not temp:
            return 0

        b[: len(temp)] = temp
        return len(temp)
