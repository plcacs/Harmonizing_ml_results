from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
from typing import Optional, List, Tuple, Dict, Any, Union, Callable, Awaitable, BinaryIO

if typing.TYPE_CHECKING:
    import python_multipart as multipart
    from python_multipart.multipart import MultipartCallbacks, QuerystringCallbacks, parse_options_header
else:
    try:
        try:
            import python_multipart as multipart
            from python_multipart.multipart import parse_options_header
        except ModuleNotFoundError:
            import multipart
            from multipart.multipart import parse_options_header
    except ModuleNotFoundError:
        multipart = None
        parse_options_header = None

class FormMessage(Enum):
    FIELD_START = 1
    FIELD_NAME = 2
    FIELD_DATA = 3
    FIELD_END = 4
    END = 5

@dataclass
class MultipartPart:
    content_disposition: Optional[bytes] = None
    field_name: str = ''
    data: bytearray = field(default_factory=bytearray)
    file: Optional[UploadFile] = None
    item_headers: List[Tuple[bytes, bytes]] = field(default_factory=list)

def _user_safe_decode(src: bytes, codec: str) -> str:
    try:
        return src.decode(codec)
    except (UnicodeDecodeError, LookupError):
        return src.decode('latin-1')

class MultiPartException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

class FormParser:
    def __init__(self, headers: Headers, stream: typing.AsyncIterable[bytes]) -> None:
        assert multipart is not None, 'The `python-multipart` library must be installed to use form parsing.'
        self.headers: Headers = headers
        self.stream: typing.AsyncIterable[bytes] = stream
        self.messages: List[Tuple[FormMessage, bytes]] = []

    def on_field_start(self) -> None:
        message: Tuple[FormMessage, bytes] = (FormMessage.FIELD_START, b'')
        self.messages.append(message)

    def on_field_name(self, data: bytes, start: int, end: int) -> None:
        message: Tuple[FormMessage, bytes] = (FormMessage.FIELD_NAME, data[start:end])
        self.messages.append(message)

    def on_field_data(self, data: bytes, start: int, end: int) -> None:
        message: Tuple[FormMessage, bytes] = (FormMessage.FIELD_DATA, data[start:end])
        self.messages.append(message)

    def on_field_end(self) -> None:
        message: Tuple[FormMessage, bytes] = (FormMessage.FIELD_END, b'')
        self.messages.append(message)

    def on_end(self) -> None:
        message: Tuple[FormMessage, bytes] = (FormMessage.END, b'')
        self.messages.append(message)

    async def parse(self) -> FormData:
        callbacks: Dict[str, Callable[..., Any]] = {
            'on_field_start': self.on_field_start,
            'on_field_name': self.on_field_name,
            'on_field_data': self.on_field_data,
            'on_field_end': self.on_field_end,
            'on_end': self.on_end
        }
        parser: Any = multipart.QuerystringParser(callbacks)
        field_name: bytes = b''
        field_value: bytes = b''
        items: List[Tuple[str, str]] = []
        async for chunk in self.stream:
            if chunk:
                parser.write(chunk)
            else:
                parser.finalize()
            messages = list(self.messages)
            self.messages.clear()
            for message_type, message_bytes in messages:
                if message_type == FormMessage.FIELD_START:
                    field_name = b''
                    field_value = b''
                elif message_type == FormMessage.FIELD_NAME:
                    field_name += message_bytes
                elif message_type == FormMessage.FIELD_DATA:
                    field_value += message_bytes
                elif message_type == FormMessage.FIELD_END:
                    name = unquote_plus(field_name.decode('latin-1'))
                    value = unquote_plus(field_value.decode('latin-1'))
                    items.append((name, value))
        return FormData(items)

class MultiPartParser:
    max_file_size: int = 1024 * 1024

    def __init__(
        self,
        headers: Headers,
        stream: typing.AsyncIterable[bytes],
        *,
        max_files: int = 1000,
        max_fields: int = 1000,
        max_part_size: int = 1024 * 1024
    ) -> None:
        assert multipart is not None, 'The `python-multipart` library must be installed to use form parsing.'
        self.headers: Headers = headers
        self.stream: typing.AsyncIterable[bytes] = stream
        self.max_files: int = max_files
        self.max_fields: int = max_fields
        self.items: List[Tuple[str, Union[str, UploadFile]]] = []
        self._current_files: int = 0
        self._current_fields: int = 0
        self._current_partial_header_name: bytes = b''
        self._current_partial_header_value: bytes = b''
        self._current_part: MultipartPart = MultipartPart()
        self._charset: str = ''
        self._file_parts_to_write: List[Tuple[MultipartPart, bytes]] = []
        self._file_parts_to_finish: List[MultipartPart] = []
        self._files_to_close_on_error: List[BinaryIO] = []
        self.max_part_size: int = max_part_size

    def on_part_begin(self) -> None:
        self._current_part = MultipartPart()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        message_bytes: bytes = data[start:end]
        if self._current_part.file is None:
            if len(self._current_part.data) + len(message_bytes) > self.max_part_size:
                raise MultiPartException(f'Part exceeded maximum size of {int(self.max_part_size / 1024)}KB.')
            self._current_part.data.extend(message_bytes)
        else:
            self._file_parts_to_write.append((self._current_part, message_bytes))

    def on_part_end(self) -> None:
        if self._current_part.file is None:
            self.items.append((self._current_part.field_name, _user_safe_decode(self._current_part.data, self._charset)))
        else:
            self._file_parts_to_finish.append(self._current_part)
            self.items.append((self._current_part.field_name, self._current_part.file))

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._current_partial_header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._current_partial_header_value += data[start:end]

    def on_header_end(self) -> None:
        field: bytes = self._current_partial_header_name.lower()
        if field == b'content-disposition':
            self._current_part.content_disposition = self._current_partial_header_value
        self._current_part.item_headers.append((field, self._current_partial_header_value))
        self._current_partial_header_name = b''
        self._current_partial_header_value = b''

    def on_headers_finished(self) -> None:
        disposition: bytes
        options: Dict[bytes, bytes]
        disposition, options = parse_options_header(self._current_part.content_disposition)
        try:
            self._current_part.field_name = _user_safe_decode(options[b'name'], self._charset)
        except KeyError:
            raise MultiPartException('The Content-Disposition header field "name" must be provided.')
        if b'filename' in options:
            self._current_files += 1
            if self._current_files > self.max_files:
                raise MultiPartException(f'Too many files. Maximum number of files is {self.max_files}.')
            filename: str = _user_safe_decode(options[b'filename'], self._charset)
            tempfile: SpooledTemporaryFile = SpooledTemporaryFile(max_size=self.max_file_size)
            self._files_to_close_on_error.append(tempfile)
            self._current_part.file = UploadFile(
                file=tempfile,
                size=0,
                filename=filename,
                headers=Headers(raw=self._current_part.item_headers)
        else:
            self._current_fields += 1
            if self._current_fields > self.max_fields:
                raise MultiPartException(f'Too many fields. Maximum number of fields is {self.max_fields}.')
            self._current_part.file = None

    def on_end(self) -> None:
        pass

    async def parse(self) -> FormData:
        params: Dict[bytes, bytes]
        _, params = parse_options_header(self.headers['Content-Type'])
        charset: Union[bytes, str] = params.get(b'charset', 'utf-8')
        if isinstance(charset, bytes):
            charset = charset.decode('latin-1')
        self._charset = charset
        try:
            boundary: bytes = params[b'boundary']
        except KeyError:
            raise MultiPartException('Missing boundary in multipart.')
        callbacks: Dict[str, Callable[..., Any]] = {
            'on_part_begin': self.on_part_begin,
            'on_part_data': self.on_part_data,
            'on_part_end': self.on_part_end,
            'on_header_field': self.on_header_field,
            'on_header_value': self.on_header_value,
            'on_header_end': self.on_header_end,
            'on_headers_finished': self.on_headers_finished,
            'on_end': self.on_end
        }
        parser: Any = multipart.MultipartParser(boundary, callbacks)
        try:
            async for chunk in self.stream:
                parser.write(chunk)
                for part, data in self._file_parts_to_write:
                    assert part.file
                    await part.file.write(data)
                for part in self._file_parts_to_finish:
                    assert part.file
                    await part.file.seek(0)
                self._file_parts_to_write.clear()
                self._file_parts_to_finish.clear()
        except MultiPartException as exc:
            for file in self._files_to_close_on_error:
                file.close()
            raise exc
        parser.finalize()
        return FormData(self.items)
