#!/usr/bin/env python3
"""
requests_toolbelt.multipart.encoder
===================================

This holds all of the implementation details of the MultipartEncoder
"""

import contextlib
import io
import os
from uuid import uuid4
from typing import Any, Optional, Union, List, Tuple, Iterator, Callable, Generator, Mapping, IO
import requests
from .._compat import fields  # type: ignore


class FileNotSupportedError(Exception):
    """File not supported error."""


class MultipartEncoder:
    """
    The ``MultipartEncoder`` object is a generic interface to the engine that
    will create a ``multipart/form-data`` body for you.
    """

    def __init__(self, fields: Union[Mapping[str, Any], List[Tuple[str, Any]]],
                 boundary: Optional[str] = None, encoding: str = 'utf-8') -> None:
        self.boundary_value: str = boundary or uuid4().hex
        self.boundary: str = '--{}'.format(self.boundary_value)
        self.encoding: str = encoding
        self._encoded_boundary: bytes = b''.join([encode_with(self.boundary, self.encoding),
                                                   encode_with('\r\n', self.encoding)])
        self.fields: Union[Mapping[str, Any], List[Tuple[str, Any]]] = fields
        self.finished: bool = False
        self.parts: List[Part] = []
        self._iter_parts: Iterator[Part] = iter([])
        self._current_part: Optional[Part] = None
        self._len: Optional[int] = None
        self._buffer: CustomBytesIO = CustomBytesIO(encoding=encoding)
        self._prepare_parts()
        self._write_boundary()

    @property
    def len(self) -> int:
        """Length of the multipart/form-data body."""
        return self._len or self._calculate_length()

    def __repr__(self) -> str:
        return '<MultipartEncoder: {!r}>'.format(self.fields)

    def _calculate_length(self) -> int:
        """
        This uses the parts to calculate the length of the body.
        """
        boundary_len: int = len(self.boundary)
        self._len = sum((boundary_len + total_len(p) + 4 for p in self.parts)) + boundary_len + 4
        return self._len

    def _calculate_load_amount(self, read_size: int) -> int:
        """Calculate how many bytes need to be added to the buffer."""
        amount: int = read_size - total_len(self._buffer)
        return amount if amount > 0 else 0

    def _load(self, amount: int) -> None:
        """Load ``amount`` number of bytes into the buffer."""
        self._buffer.smart_truncate()
        part: Optional[Part] = self._current_part or self._next_part()
        while amount == -1 or amount > 0:
            written: int = 0
            if part and (not part.bytes_left_to_write()):
                written += self._write(b'\r\n')
                written += self._write_boundary()
                part = self._next_part()
            if not part:
                written += self._write_closing_boundary()
                self.finished = True
                break
            written += part.write_to(self._buffer, amount)
            if amount != -1:
                amount -= written

    def _next_part(self) -> Optional["Part"]:
        try:
            p: Part = next(self._iter_parts)
            self._current_part = p
        except StopIteration:
            p = None
        return p

    def _iter_fields(self) -> Iterator[fields.RequestField]:
        _fields: Union[Mapping[str, Any], List[Tuple[str, Any]]] = self.fields
        if hasattr(self.fields, 'items'):
            _fields = list(self.fields.items())
        for k, v in _fields:  # type: ignore
            file_name: Optional[Any] = None
            file_type: Optional[Any] = None
            file_headers: Optional[Any] = None
            if isinstance(v, (list, tuple)):
                if len(v) == 2:
                    file_name, file_pointer = v
                elif len(v) == 3:
                    file_name, file_pointer, file_type = v
                else:
                    file_name, file_pointer, file_type, file_headers = v
            else:
                file_pointer = v
            field_obj: fields.RequestField = fields.RequestField(name=k, data=file_pointer,
                                                                   filename=file_name, headers=file_headers)
            field_obj.make_multipart(content_type=file_type)
            yield field_obj

    def _prepare_parts(self) -> None:
        """Prepare parts using the provided fields."""
        enc: str = self.encoding
        self.parts = [Part.from_field(f, enc) for f in self._iter_fields()]
        self._iter_parts = iter(self.parts)

    def _write(self, bytes_to_write: bytes) -> int:
        """Write the bytes to the end of the buffer."""
        return self._buffer.append(bytes_to_write)

    def _write_boundary(self) -> int:
        """Write the boundary to the end of the buffer."""
        return self._write(self._encoded_boundary)

    def _write_closing_boundary(self) -> int:
        """Write the bytes necessary to finish a multipart/form-data body."""
        with reset(self._buffer):
            self._buffer.seek(-2, 2)  # type: ignore
            self._buffer.write(b'--\r\n')
        return 2

    def _write_headers(self, headers: str) -> int:
        """Write the current part's headers to the buffer."""
        return self._write(encode_with(headers, self.encoding))

    @property
    def content_type(self) -> str:
        return 'multipart/form-data; boundary={}'.format(self.boundary_value)

    def to_string(self) -> bytes:
        """Return the entirety of the data in the encoder."""
        return self.read()

    def read(self, size: int = -1) -> bytes:
        """Read data from the streaming encoder."""
        if self.finished:
            return self._buffer.read(size)
        bytes_to_load: Union[int, None] = size
        if bytes_to_load != -1 and bytes_to_load is not None:
            bytes_to_load = self._calculate_load_amount(int(size))
        self._load(bytes_to_load)  # type: ignore
        return self._buffer.read(size)


def IDENTITY(monitor: "MultipartEncoderMonitor") -> "MultipartEncoderMonitor":
    return monitor


class MultipartEncoderMonitor:
    """
    An object used to monitor the progress of a :class:`MultipartEncoder`.
    """

    def __init__(self, encoder: MultipartEncoder, callback: Optional[Callable[["MultipartEncoderMonitor"], Any]] = None) -> None:
        self.encoder: MultipartEncoder = encoder
        self.callback: Callable[["MultipartEncoderMonitor"], Any] = callback or IDENTITY
        self.bytes_read: int = 0
        self.len: int = self.encoder.len

    @classmethod
    def from_fields(cls, fields: Union[Mapping[str, Any], List[Tuple[str, Any]]],
                    boundary: Optional[str] = None, encoding: str = 'utf-8',
                    callback: Optional[Callable[["MultipartEncoderMonitor"], Any]] = None) -> "MultipartEncoderMonitor":
        encoder: MultipartEncoder = MultipartEncoder(fields, boundary, encoding)
        return cls(encoder, callback)

    @property
    def content_type(self) -> str:
        return self.encoder.content_type

    def to_string(self) -> bytes:
        return self.read()

    def read(self, size: int = -1) -> bytes:
        data: bytes = self.encoder.read(size)
        self.bytes_read += len(data)
        self.callback(self)
        return data


def encode_with(string: Optional[Union[str, bytes]], encoding: str) -> bytes:
    """Encoding ``string`` with ``encoding`` if necessary."""
    if not (string is None or isinstance(string, bytes)):
        return string.encode(encoding)  # type: ignore
    return string or b''  # type: ignore


def readable_data(data: Any, encoding: str) -> IO[bytes]:
    """Coerce the data to an object with a ``read`` method."""
    if hasattr(data, 'read'):
        return data  # type: ignore
    return CustomBytesIO(data, encoding)


def total_len(o: Any) -> int:
    if hasattr(o, '__len__'):
        return len(o)
    if hasattr(o, 'len'):
        return o.len  # type: ignore
    if hasattr(o, 'fileno'):
        try:
            fileno = o.fileno()  # type: ignore
        except io.UnsupportedOperation:
            pass
        else:
            return os.fstat(fileno).st_size  # type: ignore
    if hasattr(o, 'getvalue'):
        return len(o.getvalue())  # type: ignore
    return 0


@contextlib.contextmanager
def reset(buffer: io.BytesIO) -> Generator[None, None, None]:
    """Keep track of the buffer's current position and write to the end."""
    original_position: int = buffer.tell()
    buffer.seek(0, 2)
    yield
    buffer.seek(original_position, 0)


def coerce_data(data: Any, encoding: str) -> Union["CustomBytesIO", "FileWrapper"]:
    """Ensure that every object's __len__ behaves uniformly."""
    if not isinstance(data, CustomBytesIO):
        if hasattr(data, 'getvalue'):
            return CustomBytesIO(data.getvalue(), encoding)
        if hasattr(data, 'fileno'):
            return FileWrapper(data)
        if not hasattr(data, 'read'):
            return CustomBytesIO(data, encoding)
    return data  # type: ignore


def to_list(fields: Union[Mapping[str, Any], List[Tuple[str, Any]]]) -> List[Tuple[str, Any]]:
    if hasattr(fields, 'items'):
        return list(fields.items())  # type: ignore
    return list(fields)  # type: ignore


class Part:
    def __init__(self, headers: bytes, body: Union["CustomBytesIO", "FileWrapper", IO[bytes]]) -> None:
        self.headers: bytes = headers
        self.body: Union[CustomBytesIO, FileWrapper, IO[bytes]] = body
        self.headers_unread: bool = True
        self.len: int = len(self.headers) + total_len(self.body)

    @classmethod
    def from_field(cls, field: fields.RequestField, encoding: str) -> "Part":
        """Create a part from a Request Field generated by urllib3."""
        headers: bytes = encode_with(field.render_headers(), encoding)
        body = coerce_data(field.data, encoding)
        return cls(headers, body)

    def bytes_left_to_write(self) -> bool:
        """Determine if there are bytes left to write."""
        to_read: int = 0
        if self.headers_unread:
            to_read += len(self.headers)
        return to_read + total_len(self.body) > 0

    def write_to(self, buffer: "CustomBytesIO", size: int) -> int:
        """Write the requested amount of bytes to the buffer provided."""
        written: int = 0
        if self.headers_unread:
            written += buffer.append(self.headers)
            self.headers_unread = False
        while total_len(self.body) > 0 and (size == -1 or written < size):
            amount_to_read: int = size
            if size != -1:
                amount_to_read = size - written
            data = self.body.read(amount_to_read)
            written += buffer.append(data)
        return written


class CustomBytesIO(io.BytesIO):
    def __init__(self, buffer: Optional[Union[str, bytes]] = None, encoding: str = 'utf-8') -> None:
        buffer_bytes: bytes = encode_with(buffer, encoding) if buffer is not None else b''
        super(CustomBytesIO, self).__init__(buffer_bytes)
        self.encoding: str = encoding

    def _get_end(self) -> int:
        current_pos: int = self.tell()
        self.seek(0, 2)
        length: int = self.tell()
        self.seek(current_pos, 0)
        return length

    @property
    def len(self) -> int:
        length: int = self._get_end()
        return length - self.tell()

    def append(self, b: bytes) -> int:
        with reset(self):
            written: int = self.write(b)
        return written

    def smart_truncate(self) -> None:
        to_be_read: int = total_len(self)
        already_read: int = self._get_end() - to_be_read
        if already_read >= to_be_read:
            old_bytes: bytes = self.read()
            self.seek(0, 0)
            self.truncate()
            self.write(old_bytes)
            self.seek(0, 0)


class FileWrapper:
    def __init__(self, file_object: IO[bytes]) -> None:
        self.fd: IO[bytes] = file_object

    @property
    def len(self) -> int:
        return total_len(self.fd) - self.fd.tell()  # type: ignore

    def read(self, length: int = -1) -> bytes:
        return self.fd.read(length)


class FileFromURLWrapper:
    """
    File from URL wrapper.
    """
    def __init__(self, file_url: str, session: Optional[requests.Session] = None) -> None:
        self.session: requests.Session = session or requests.Session()
        requested_file: requests.Response = self._request_for_file(file_url)
        self.len: int = int(requested_file.headers['content-length'])
        self.raw_data: IO[bytes] = requested_file.raw  # type: ignore

    def _request_for_file(self, file_url: str) -> requests.Response:
        """Make call for file under provided URL."""
        response: requests.Response = self.session.get(file_url, stream=True)
        content_length: Optional[str] = response.headers.get('content-length', None)
        if content_length is None:
            error_msg: str = ('Data from provided URL {url} is not supported. '
                              'Lack of content-length Header in requested file response.'.format(url=file_url))
            raise FileNotSupportedError(error_msg)
        elif not content_length.isdigit():
            error_msg = ('Data from provided URL {url} is not supported. '
                         'content-length header value is not a digit.'.format(url=file_url))
            raise FileNotSupportedError(error_msg)
        return response

    def read(self, chunk_size: int) -> bytes:
        """Read file in chunks."""
        chunk_size = chunk_size if chunk_size >= 0 else self.len
        chunk: bytes = self.raw_data.read(chunk_size) or b''
        self.len -= len(chunk) if chunk else 0
        return chunk
