import asyncio
import enum
import io
import json
import mimetypes
import os
import sys
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import IO, TYPE_CHECKING, Any, Dict, Final, Iterable, Optional, TextIO, Tuple, Type, Union
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import _SENTINEL, content_disposition_header, guess_filename, parse_mimetype, sentinel
from .streams import StreamReader
from .typedefs import JSONEncoder, _CIMultiDict
__all__ = ('PAYLOAD_REGISTRY', 'get_payload', 'payload_type', 'Payload',
    'BytesPayload', 'StringPayload', 'IOBasePayload', 'BytesIOPayload',
    'BufferedReaderPayload', 'TextIOPayload', 'StringIOPayload',
    'JsonPayload', 'AsyncIterablePayload')
TOO_LARGE_BYTES_BODY: Final[int] = 2 ** 20
if TYPE_CHECKING:
    from typing import List


class LookupError(Exception):
    pass


class Order(str, enum.Enum):
    normal = 'normal'
    try_first = 'try_first'
    try_last = 'try_last'


def get_payload(data, *args: Any, **kwargs: Any):
    return PAYLOAD_REGISTRY.get(data, *args, **kwargs)


def register_payload(factory, type, *, order: Order=Order.normal):
    PAYLOAD_REGISTRY.register(factory, type, order=order)


class payload_type:

    def __init__(self, type, *, order: Order=Order.normal):
        self.type = type
        self.order = order

    def __call__(self, factory):
        register_payload(factory, self.type, order=self.order)
        return factory


PayloadType = Type['Payload']
_PayloadRegistryItem = Tuple[PayloadType, Any]


class PayloadRegistry:
    """Payload registry.

    note: we need zope.interface for more efficient adapter search
    """
    __slots__ = '_first', '_normal', '_last', '_normal_lookup'

    def __init__(self):
        self._first: List[_PayloadRegistryItem] = []
        self._normal: List[_PayloadRegistryItem] = []
        self._last: List[_PayloadRegistryItem] = []
        self._normal_lookup: Dict[Any, PayloadType] = {}

    def get(self, data, *args: Any, _CHAIN:
        'Type[chain[_PayloadRegistryItem]]'=chain, **kwargs: Any):
        if self._first:
            for factory, type_ in self._first:
                if isinstance(data, type_):
                    return factory(data, *args, **kwargs)
        if (lookup_factory := self._normal_lookup.get(type(data))):
            return lookup_factory(data, *args, **kwargs)
        if isinstance(data, Payload):
            return data
        for factory, type_ in _CHAIN(self._normal, self._last):
            if isinstance(data, type_):
                return factory(data, *args, **kwargs)
        raise LookupError()

    def register(self, factory, type, *, order: Order=Order.normal):
        if order is Order.try_first:
            self._first.append((factory, type))
        elif order is Order.normal:
            self._normal.append((factory, type))
            if isinstance(type, Iterable):
                for t in type:
                    self._normal_lookup[t] = factory
            else:
                self._normal_lookup[type] = factory
        elif order is Order.try_last:
            self._last.append((factory, type))
        else:
            raise ValueError(f'Unsupported order {order!r}')


class Payload(ABC):
    _default_content_type: str = 'application/octet-stream'
    _size: Optional[int] = None

    def __init__(self, value, headers=None, content_type=sentinel, filename
        =None, encoding=None, **kwargs: Any):
        self._encoding = encoding
        self._filename = filename
        self._headers: _CIMultiDict = CIMultiDict()
        self._value = value
        if content_type is not sentinel and content_type is not None:
            assert isinstance(content_type, str)
            self._headers[hdrs.CONTENT_TYPE] = content_type
        elif self._filename is not None:
            if sys.version_info >= (3, 13):
                guesser = mimetypes.guess_file_type
            else:
                guesser = mimetypes.guess_type
            content_type = guesser(self._filename)[0]
            if content_type is None:
                content_type = self._default_content_type
            self._headers[hdrs.CONTENT_TYPE] = content_type
        else:
            self._headers[hdrs.CONTENT_TYPE] = self._default_content_type
        if headers:
            self._headers.update(headers)

    @property
    def size(self):
        """Size of the payload."""
        return self._size

    @property
    def filename(self):
        """Filename of the payload."""
        return self._filename

    @property
    def headers(self):
        """Custom item headers"""
        return self._headers

    @property
    def _binary_headers(self):
        return ''.join([(k + ': ' + v + '\r\n') for k, v in self.headers.
            items()]).encode('utf-8') + b'\r\n'

    @property
    def encoding(self):
        """Payload encoding"""
        return self._encoding

    @property
    def content_type(self):
        """Content type"""
        return self._headers[hdrs.CONTENT_TYPE]

    def set_content_disposition(self, disptype, quote_fields=True, _charset
        ='utf-8', **params: str):
        """Sets ``Content-Disposition`` header."""
        self._headers[hdrs.CONTENT_DISPOSITION] = content_disposition_header(
            disptype, quote_fields=quote_fields, _charset=_charset, params=
            params)

    @abstractmethod
    def decode(self, encoding='utf-8', errors='strict'):
        """Return string representation of the value.

        This is named decode() to allow compatibility with bytes objects.
        """

    @abstractmethod
    async def write(self, writer: AbstractStreamWriter) ->None:
        """Write payload.

        writer is an AbstractStreamWriter instance:
        """


class BytesPayload(Payload):
    _value: bytes

    def __init__(self, value, *args: Any, **kwargs: Any):
        if 'content_type' not in kwargs:
            kwargs['content_type'] = 'application/octet-stream'
        super().__init__(value, *args, **kwargs)
        if isinstance(value, memoryview):
            self._size = value.nbytes
        elif isinstance(value, (bytes, bytearray)):
            self._size = len(value)
        else:
            raise TypeError(
                f'value argument must be byte-ish, not {type(value)!r}')
        if self._size > TOO_LARGE_BYTES_BODY:
            warnings.warn(
                'Sending a large body directly with raw bytes might lock the event loop. You should probably pass an io.BytesIO object instead'
                , ResourceWarning, source=self)

    def decode(self, encoding='utf-8', errors='strict'):
        return self._value.decode(encoding, errors)

    async def write(self, writer: AbstractStreamWriter) ->None:
        await writer.write(self._value)


class StringPayload(BytesPayload):

    def __init__(self, value, *args: Any, encoding: Optional[str]=None,
        content_type: Optional[str]=None, **kwargs: Any):
        if encoding is None:
            if content_type is None:
                real_encoding = 'utf-8'
                content_type = 'text/plain; charset=utf-8'
            else:
                mimetype = parse_mimetype(content_type)
                real_encoding = mimetype.parameters.get('charset', 'utf-8')
        else:
            if content_type is None:
                content_type = 'text/plain; charset=%s' % encoding
            real_encoding = encoding
        super().__init__(value.encode(real_encoding), *args, encoding=
            real_encoding, content_type=content_type, **kwargs)


class StringIOPayload(StringPayload):

    def __init__(self, value, *args: Any, **kwargs: Any):
        super().__init__(value.read(), *args, **kwargs)


class IOBasePayload(Payload):
    _value: io.IOBase

    def __init__(self, value, disposition='attachment', *args: Any, **
        kwargs: Any):
        if 'filename' not in kwargs:
            kwargs['filename'] = guess_filename(value)
        super().__init__(value, *args, **kwargs)
        if self._filename is not None and disposition is not None:
            if hdrs.CONTENT_DISPOSITION not in self.headers:
                self.set_content_disposition(disposition, filename=self.
                    _filename)

    async def write(self, writer: AbstractStreamWriter) ->None:
        loop = asyncio.get_event_loop()
        try:
            chunk = await loop.run_in_executor(None, self._value.read, 2 ** 16)
            while chunk:
                await writer.write(chunk)
                chunk = await loop.run_in_executor(None, self._value.read, 
                    2 ** 16)
        finally:
            await loop.run_in_executor(None, self._value.close)

    def decode(self, encoding='utf-8', errors='strict'):
        return ''.join(r.decode(encoding, errors) for r in self._value.
            readlines())


class TextIOPayload(IOBasePayload):
    _value: io.TextIOBase

    def __init__(self, value, *args: Any, encoding: Optional[str]=None,
        content_type: Optional[str]=None, **kwargs: Any):
        if encoding is None:
            if content_type is None:
                encoding = 'utf-8'
                content_type = 'text/plain; charset=utf-8'
            else:
                mimetype = parse_mimetype(content_type)
                encoding = mimetype.parameters.get('charset', 'utf-8')
        elif content_type is None:
            content_type = 'text/plain; charset=%s' % encoding
        super().__init__(value, *args, content_type=content_type, encoding=
            encoding, **kwargs)

    @property
    def size(self):
        try:
            return os.fstat(self._value.fileno()).st_size - self._value.tell()
        except OSError:
            return None

    def decode(self, encoding='utf-8', errors='strict'):
        return self._value.read()

    async def write(self, writer: AbstractStreamWriter) ->None:
        loop = asyncio.get_event_loop()
        try:
            chunk = await loop.run_in_executor(None, self._value.read, 2 ** 16)
            while chunk:
                data = chunk.encode(encoding=self._encoding
                    ) if self._encoding else chunk.encode()
                await writer.write(data)
                chunk = await loop.run_in_executor(None, self._value.read, 
                    2 ** 16)
        finally:
            await loop.run_in_executor(None, self._value.close)


class BytesIOPayload(IOBasePayload):
    _value: io.BytesIO

    @property
    def size(self):
        position = self._value.tell()
        end = self._value.seek(0, os.SEEK_END)
        self._value.seek(position)
        return end - position

    def decode(self, encoding='utf-8', errors='strict'):
        return self._value.read().decode(encoding, errors)


class BufferedReaderPayload(IOBasePayload):
    _value: io.BufferedIOBase

    @property
    def size(self):
        try:
            return os.fstat(self._value.fileno()).st_size - self._value.tell()
        except (OSError, AttributeError):
            return None

    def decode(self, encoding='utf-8', errors='strict'):
        return self._value.read().decode(encoding, errors)


class JsonPayload(BytesPayload):

    def __init__(self, value, encoding='utf-8', content_type=
        'application/json', dumps=json.dumps, *args: Any, **kwargs: Any):
        super().__init__(dumps(value).encode(encoding), *args, content_type
            =content_type, encoding=encoding, **kwargs)


if TYPE_CHECKING:
    from typing import AsyncIterable, AsyncIterator
    _AsyncIterator = AsyncIterator[bytes]
    _AsyncIterable = AsyncIterable[bytes]
else:
    from collections.abc import AsyncIterable, AsyncIterator
    _AsyncIterator = AsyncIterator
    _AsyncIterable = AsyncIterable


class AsyncIterablePayload(Payload):
    _iter: Optional[_AsyncIterator] = None
    _value: _AsyncIterable

    def __init__(self, value, *args: Any, **kwargs: Any):
        if not isinstance(value, AsyncIterable):
            raise TypeError(
                'value argument must support collections.abc.AsyncIterable interface, got {!r}'
                .format(type(value)))
        if 'content_type' not in kwargs:
            kwargs['content_type'] = 'application/octet-stream'
        super().__init__(value, *args, **kwargs)
        self._iter = value.__aiter__()

    async def write(self, writer: AbstractStreamWriter) ->None:
        if self._iter:
            try:
                while True:
                    chunk = await self._iter.__anext__()
                    await writer.write(chunk)
            except StopAsyncIteration:
                self._iter = None

    def decode(self, encoding='utf-8', errors='strict'):
        raise TypeError('Unable to decode.')


class StreamReaderPayload(AsyncIterablePayload):

    def __init__(self, value, *args: Any, **kwargs: Any):
        super().__init__(value.iter_any(), *args, **kwargs)


PAYLOAD_REGISTRY = PayloadRegistry()
PAYLOAD_REGISTRY.register(BytesPayload, (bytes, bytearray, memoryview))
PAYLOAD_REGISTRY.register(StringPayload, str)
PAYLOAD_REGISTRY.register(StringIOPayload, io.StringIO)
PAYLOAD_REGISTRY.register(TextIOPayload, io.TextIOBase)
PAYLOAD_REGISTRY.register(BytesIOPayload, io.BytesIO)
PAYLOAD_REGISTRY.register(BufferedReaderPayload, (io.BufferedReader, io.
    BufferedRandom))
PAYLOAD_REGISTRY.register(IOBasePayload, io.IOBase)
PAYLOAD_REGISTRY.register(StreamReaderPayload, StreamReader)
PAYLOAD_REGISTRY.register(AsyncIterablePayload, AsyncIterable, order=Order.
    try_last)
