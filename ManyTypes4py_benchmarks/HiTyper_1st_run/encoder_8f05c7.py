"""

requests_toolbelt.multipart.encoder
===================================

This holds all of the implementation details of the MultipartEncoder

"""
import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields

class FileNotSupportedError(Exception):
    """File not supported error."""

class MultipartEncoder(object):
    """

    The ``MultipartEncoder`` object is a generic interface to the engine that
    will create a ``multipart/form-data`` body for you.

    The basic usage is:

    .. code-block:: python

        import requests
        from requests_toolbelt import MultipartEncoder

        encoder = MultipartEncoder({'field': 'value',
                                    'other_field', 'other_value'})
        r = requests.post('https://httpbin.org/post', data=encoder,
                          headers={'Content-Type': encoder.content_type})

    If you do not need to take advantage of streaming the post body, you can
    also do:

    .. code-block:: python

        r = requests.post('https://httpbin.org/post',
                          data=encoder.to_string(),
                          headers={'Content-Type': encoder.content_type})

    If you want the encoder to use a specific order, you can use an
    OrderedDict or more simply, a list of tuples:

    .. code-block:: python

        encoder = MultipartEncoder([('field', 'value'),
                                    ('other_field', 'other_value')])

    .. versionchanged:: 0.4.0

    You can also provide tuples as part values as you would provide them to
    requests' ``files`` parameter.

    .. code-block:: python

        encoder = MultipartEncoder({
            'field': ('file_name', b'{"a": "b"}', 'application/json',
                      {'X-My-Header': 'my-value'})
        ])

    .. warning::

        This object will end up directly in :mod:`httplib`. Currently,
        :mod:`httplib` has a hard-coded read size of **8192 bytes**. This
        means that it will loop until the file has been read and your upload
        could take a while. This is **not** a bug in requests. A feature is
        being considered for this object to allow you, the user, to specify
        what size should be returned on a read. If you have opinions on this,
        please weigh in on `this issue`_.

    .. _this issue:
        https://github.com/requests/toolbelt/issues/75

    """

    def __init__(self, fields, boundary=None, encoding='utf-8') -> None:
        self.boundary_value = boundary or uuid4().hex
        self.boundary = '--{}'.format(self.boundary_value)
        self.encoding = encoding
        self._encoded_boundary = b''.join([encode_with(self.boundary, self.encoding), encode_with('\r\n', self.encoding)])
        self.fields = fields
        self.finished = False
        self.parts = []
        self._iter_parts = iter([])
        self._current_part = None
        self._len = None
        self._buffer = CustomBytesIO(encoding=encoding)
        self._prepare_parts()
        self._write_boundary()

    @property
    def len(self) -> int:
        """Length of the multipart/form-data body.

        requests will first attempt to get the length of the body by calling
        ``len(body)`` and then by checking for the ``len`` attribute.

        On 32-bit systems, the ``__len__`` method cannot return anything
        larger than an integer (in C) can hold. If the total size of the body
        is even slightly larger than 4GB users will see an OverflowError. This
        manifested itself in `bug #80`_.

        As such, we now calculate the length lazily as a property.

        .. _bug #80:
            https://github.com/requests/toolbelt/issues/80
        """
        return self._len or self._calculate_length()

    def __repr__(self) -> str:
        return '<MultipartEncoder: {!r}>'.format(self.fields)

    def _calculate_length(self) -> bool:
        """
        This uses the parts to calculate the length of the body.

        This returns the calculated length so __len__ can be lazy.
        """
        boundary_len = len(self.boundary)
        self._len = sum((boundary_len + total_len(p) + 4 for p in self.parts)) + boundary_len + 4
        return self._len

    def _calculate_load_amount(self, read_size: Union[int, float, None]) -> Union[int, float]:
        """This calculates how many bytes need to be added to the buffer.

        When a consumer read's ``x`` from the buffer, there are two cases to
        satisfy:

            1. Enough data in the buffer to return the requested amount
            2. Not enough data

        This function uses the amount of unread bytes in the buffer and
        determines how much the Encoder has to load before it can return the
        requested amount of bytes.

        :param int read_size: the number of bytes the consumer requests
        :returns: int -- the number of bytes that must be loaded into the
            buffer before the read can be satisfied. This will be strictly
            non-negative
        """
        amount = read_size - total_len(self._buffer)
        return amount if amount > 0 else 0

    def _load(self, amount: int) -> None:
        """Load ``amount`` number of bytes into the buffer."""
        self._buffer.smart_truncate()
        part = self._current_part or self._next_part()
        while amount == -1 or amount > 0:
            written = 0
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

    def _next_part(self) -> None:
        try:
            p = self._current_part = next(self._iter_parts)
        except StopIteration:
            p = None
        return p

    def _iter_fields(self) -> typing.Generator:
        _fields = self.fields
        if hasattr(self.fields, 'items'):
            _fields = list(self.fields.items())
        for k, v in _fields:
            file_name = None
            file_type = None
            file_headers = None
            if isinstance(v, (list, tuple)):
                if len(v) == 2:
                    file_name, file_pointer = v
                elif len(v) == 3:
                    file_name, file_pointer, file_type = v
                else:
                    file_name, file_pointer, file_type, file_headers = v
            else:
                file_pointer = v
            field = fields.RequestField(name=k, data=file_pointer, filename=file_name, headers=file_headers)
            field.make_multipart(content_type=file_type)
            yield field

    def _prepare_parts(self) -> None:
        """This uses the fields provided by the user and creates Part objects.

        It populates the `parts` attribute and uses that to create a
        generator for iteration.
        """
        enc = self.encoding
        self.parts = [Part.from_field(f, enc) for f in self._iter_fields()]
        self._iter_parts = iter(self.parts)

    def _write(self, bytes_to_write: Union[int, list[bytes], bytes]) -> Union[bytes, int]:
        """Write the bytes to the end of the buffer.

        :param bytes bytes_to_write: byte-string (or bytearray) to append to
            the buffer
        :returns: int -- the number of bytes written
        """
        return self._buffer.append(bytes_to_write)

    def _write_boundary(self):
        """Write the boundary to the end of the buffer."""
        return self._write(self._encoded_boundary)

    def _write_closing_boundary(self) -> int:
        """Write the bytes necessary to finish a multipart/form-data body."""
        with reset(self._buffer):
            self._buffer.seek(-2, 2)
            self._buffer.write(b'--\r\n')
        return 2

    def _write_headers(self, headers: Union[str, bytes, dict]):
        """Write the current part's headers to the buffer."""
        return self._write(encode_with(headers, self.encoding))

    @property
    def content_type(self):
        return str('multipart/form-data; boundary={}'.format(self.boundary_value))

    def to_string(self) -> Union[str, bytes, typing.IO]:
        """Return the entirety of the data in the encoder.

        .. note::

            This simply reads all of the data it can. If you have started
            streaming or reading data from the encoder, this method will only
            return whatever data is left in the encoder.

        .. note::

            This method affects the internal state of the encoder. Calling
            this method will exhaust the encoder.

        :returns: the multipart message
        :rtype: bytes
        """
        return self.read()

    def read(self, size: int=-1) -> Union[bytes, str]:
        """Read data from the streaming encoder.

        :param int size: (optional), If provided, ``read`` will return exactly
            that many bytes. If it is not provided, it will return the
            remaining bytes.
        :returns: bytes
        """
        if self.finished:
            return self._buffer.read(size)
        bytes_to_load = size
        if bytes_to_load != -1 and bytes_to_load is not None:
            bytes_to_load = self._calculate_load_amount(int(size))
        self._load(bytes_to_load)
        return self._buffer.read(size)

def IDENTITY(monitor: Union[bool, typing.Optional, tuple]) -> Union[bool, typing.Optional, tuple]:
    return monitor

class MultipartEncoderMonitor(object):
    """
    An object used to monitor the progress of a :class:`MultipartEncoder`.

    The :class:`MultipartEncoder` should only be responsible for preparing and
    streaming the data. For anyone who wishes to monitor it, they shouldn't be
    using that instance to manage that as well. Using this class, they can
    monitor an encoder and register a callback. The callback receives the
    instance of the monitor.

    To use this monitor, you construct your :class:`MultipartEncoder` as you
    normally would.

    .. code-block:: python

        from requests_toolbelt import (MultipartEncoder,
                                       MultipartEncoderMonitor)
        import requests

        def callback(monitor):
            # Do something with this information
            pass

        m = MultipartEncoder(fields={'field0': 'value0'})
        monitor = MultipartEncoderMonitor(m, callback)
        headers = {'Content-Type': monitor.content_type}
        r = requests.post('https://httpbin.org/post', data=monitor,
                          headers=headers)

    Alternatively, if your use case is very simple, you can use the following
    pattern.

    .. code-block:: python

        from requests_toolbelt import MultipartEncoderMonitor
        import requests

        def callback(monitor):
            # Do something with this information
            pass

        monitor = MultipartEncoderMonitor.from_fields(
            fields={'field0': 'value0'}, callback
            )
        headers = {'Content-Type': montior.content_type}
        r = requests.post('https://httpbin.org/post', data=monitor,
                          headers=headers)

    """

    def __init__(self, encoder, callback=None) -> None:
        self.encoder = encoder
        self.callback = callback or IDENTITY
        self.bytes_read = 0
        self.len = self.encoder.len

    @classmethod
    def from_fields(cls: Union[typing.Callable[..., None], None, str], fields: Union[str, typing.AnyStr, None], boundary: Union[None, str, typing.AnyStr]=None, encoding: typing.Text='utf-8', callback: Union[None, typing.Callable[..., None], str]=None) -> Union[str, None, typing.Type]:
        encoder = MultipartEncoder(fields, boundary, encoding)
        return cls(encoder, callback)

    @property
    def content_type(self):
        return self.encoder.content_type

    def to_string(self) -> Union[str, bytes, typing.IO]:
        return self.read()

    def read(self, size: int=-1) -> Union[bytes, str]:
        string = self.encoder.read(size)
        self.bytes_read += len(string)
        self.callback(self)
        return string

def encode_with(string: str, encoding: Union[str, None]) -> Union[bytes, str]:
    """Encoding ``string`` with ``encoding`` if necessary.

    :param str string: If string is a bytes object, it will not encode it.
        Otherwise, this function will encode it with the provided encoding.
    :param str encoding: The encoding with which to encode string.
    :returns: encoded bytes object
    """
    if not (string is None or isinstance(string, bytes)):
        return string.encode(encoding)
    return string

def readable_data(data: Union[str, bytes], encoding: Union[str, bool]) -> Union[str, bytes, CustomBytesIO]:
    """Coerce the data to an object with a ``read`` method."""
    if hasattr(data, 'read'):
        return data
    return CustomBytesIO(data, encoding)

def total_len(o: Union[bytearray, str]) -> int:
    if hasattr(o, '__len__'):
        return len(o)
    if hasattr(o, 'len'):
        return o.len
    if hasattr(o, 'fileno'):
        try:
            fileno = o.fileno()
        except io.UnsupportedOperation:
            pass
        else:
            return os.fstat(fileno).st_size
    if hasattr(o, 'getvalue'):
        return len(o.getvalue())

@contextlib.contextmanager
def reset(buffer: Union[bytes, bytearray, memoryview]) -> typing.Generator:
    """Keep track of the buffer's current position and write to the end.

    This is a context manager meant to be used when adding data to the buffer.
    It eliminates the need for every function to be concerned with the
    position of the cursor in the buffer.
    """
    original_position = buffer.tell()
    buffer.seek(0, 2)
    yield
    buffer.seek(original_position, 0)

def coerce_data(data: Union[bytes, None, str], encoding: Union[str, bytes, list]) -> Union[CustomBytesIO, FileWrapper, bytes, None, str]:
    """Ensure that every object's __len__ behaves uniformly."""
    if not isinstance(data, CustomBytesIO):
        if hasattr(data, 'getvalue'):
            return CustomBytesIO(data.getvalue(), encoding)
        if hasattr(data, 'fileno'):
            return FileWrapper(data)
        if not hasattr(data, 'read'):
            return CustomBytesIO(data, encoding)
    return data

def to_list(fields: dict[str, typing.Any]) -> Union[list[list[tuple[typing.Union[str,typing.Any]]]], list[str]]:
    if hasattr(fields, 'items'):
        return list(fields.items())
    return list(fields)

class Part(object):

    def __init__(self, headers: Union[bytes, bytearray, tuple[typing.Union[int,bytearray]]], body: Union[bytes, dict, str]) -> None:
        self.headers = headers
        self.body = body
        self.headers_unread = True
        self.len = len(self.headers) + total_len(self.body)

    @classmethod
    def from_field(cls: Union[str, bytes, dict], field: Union[str, bytes], encoding: Union[str, bytes]):
        """Create a part from a Request Field generated by urllib3."""
        headers = encode_with(field.render_headers(), encoding)
        body = coerce_data(field.data, encoding)
        return cls(headers, body)

    def bytes_left_to_write(self) -> bool:
        """Determine if there are bytes left to write.

        :returns: bool -- ``True`` if there are bytes left to write, otherwise
            ``False``
        """
        to_read = 0
        if self.headers_unread:
            to_read += len(self.headers)
        return to_read + total_len(self.body) > 0

    def write_to(self, buffer: Union[int, io.BufferedWriter, io.BufferedReader], size: int) -> Union[bytearray, io.StringIO, io.TextIOWrapper, int]:
        """Write the requested amount of bytes to the buffer provided.

        The number of bytes written may exceed size on the first read since we
        load the headers ambitiously.

        :param CustomBytesIO buffer: buffer we want to write bytes to
        :param int size: number of bytes requested to be written to the buffer
        :returns: int -- number of bytes actually written
        """
        written = 0
        if self.headers_unread:
            written += buffer.append(self.headers)
            self.headers_unread = False
        while total_len(self.body) > 0 and (size == -1 or written < size):
            amount_to_read = size
            if size != -1:
                amount_to_read = size - written
            written += buffer.append(self.body.read(amount_to_read))
        return written

class CustomBytesIO(io.BytesIO):

    def __init__(self, buffer=None, encoding='utf-8') -> None:
        buffer = encode_with(buffer, encoding)
        super(CustomBytesIO, self).__init__(buffer)

    def _get_end(self) -> Union[int, dict, list[int]]:
        current_pos = self.tell()
        self.seek(0, 2)
        length = self.tell()
        self.seek(current_pos, 0)
        return length

    @property
    def len(self) -> int:
        length = self._get_end()
        return length - self.tell()

    def append(self, bytes: Union[bytes, int, list]) -> Union[bytes, int, io.BytesIO]:
        with reset(self):
            written = self.write(bytes)
        return written

    def smart_truncate(self) -> None:
        to_be_read = total_len(self)
        already_read = self._get_end() - to_be_read
        if already_read >= to_be_read:
            old_bytes = self.read()
            self.seek(0, 0)
            self.truncate()
            self.write(old_bytes)
            self.seek(0, 0)

class FileWrapper(object):

    def __init__(self, file_object) -> None:
        self.fd = file_object

    @property
    def len(self) -> int:
        return total_len(self.fd) - self.fd.tell()

    def read(self, length=-1) -> Union[bytes, str]:
        return self.fd.read(length)

class FileFromURLWrapper(object):
    """File from URL wrapper.

    The :class:`FileFromURLWrapper` object gives you the ability to stream file
    from provided URL in chunks by :class:`MultipartEncoder`.
    Provide a stateless solution for streaming file from one server to another.
    You can use the :class:`FileFromURLWrapper` without a session or with
    a session as demonstated by the examples below:

    .. code-block:: python
        # no session

        import requests
        from requests_toolbelt import MultipartEncoder, FileFromURLWrapper

        url = 'https://httpbin.org/image/png'
        streaming_encoder = MultipartEncoder(
            fields={
                'file': FileFromURLWrapper(url)
            }
        )
        r = requests.post(
            'https://httpbin.org/post', data=streaming_encoder,
            headers={'Content-Type': streaming_encoder.content_type}
        )

    .. code-block:: python
        # using a session

        import requests
        from requests_toolbelt import MultipartEncoder, FileFromURLWrapper

        session = requests.Session()
        url = 'https://httpbin.org/image/png'
        streaming_encoder = MultipartEncoder(
            fields={
                'file': FileFromURLWrapper(url, session=session)
            }
        )
        r = session.post(
            'https://httpbin.org/post', data=streaming_encoder,
            headers={'Content-Type': streaming_encoder.content_type}
        )

    """

    def __init__(self, file_url, session=None) -> None:
        self.session = session or requests.Session()
        requested_file = self._request_for_file(file_url)
        self.len = int(requested_file.headers['content-length'])
        self.raw_data = requested_file.raw

    def _request_for_file(self, file_url: str) -> Union[requests.models.Response, dict, dict[str, typing.Any]]:
        """Make call for file under provided URL."""
        response = self.session.get(file_url, stream=True)
        content_length = response.headers.get('content-length', None)
        if content_length is None:
            error_msg = 'Data from provided URL {url} is not supported. Lack of content-length Header in requested file response.'.format(url=file_url)
            raise FileNotSupportedError(error_msg)
        elif not content_length.isdigit():
            error_msg = 'Data from provided URL {url} is not supported. content-length header value is not a digit.'.format(url=file_url)
            raise FileNotSupportedError(error_msg)
        return response

    def read(self, chunk_size) -> Union[bytes, str]:
        """Read file in chunks."""
        chunk_size = chunk_size if chunk_size >= 0 else self.len
        chunk = self.raw_data.read(chunk_size) or b''
        self.len -= len(chunk) if chunk else 0
        return chunk