"""Common IO api utilities"""
from __future__ import annotations
from abc import ABC, abstractmethod
import codecs
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
import dataclasses
import functools
import gzip
from io import BufferedIOBase, BytesIO, RawIOBase, StringIO, TextIOBase, TextIOWrapper
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import IO, TYPE_CHECKING, Any, AnyStr, DefaultDict, Generic, Literal, TypeVar, cast, overload, Union, Optional, List, Tuple
from urllib.parse import urljoin, urlparse as parse_url, uses_netloc, uses_params, uses_relative
import warnings
import zipfile
from pandas._typing import BaseBuffer, ReadCsvBuffer
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_bool, is_file_like, is_integer, is_list_like
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core.shared_docs import _shared_docs

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard('')
_RFC_3986_PATTERN = re.compile('^[A-Za-z][A-Za-z0-9+\\-+.]*://')
BaseBufferT = TypeVar('BaseBufferT', bound=BaseBuffer)

if TYPE_CHECKING:
    from types import TracebackType
    from pandas._typing import CompressionDict, CompressionOptions, FilePath, ReadBuffer, StorageOptions, WriteBuffer
    from pandas import MultiIndex

@dataclasses.dataclass
class IOArgs:
    should_close: bool = False

@dataclasses.dataclass
class IOHandles(Generic[AnyStr]):
    created_handles: List[IO[Any]] = dataclasses.field(default_factory=list)
    is_wrapped: bool = False

    def close(self) -> None:
        if self.is_wrapped:
            assert isinstance(self.handle, TextIOWrapper)
            self.handle.flush()
            self.handle.detach()
            self.created_handles.remove(self.handle)
        for handle in self.created_handles:
            handle.close()
        self.created_handles = []
        self.is_wrapped = False

    def __enter__(self) -> IOHandles:
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()

def is_url(url: Union[str, bytes]) -> bool:
    if not isinstance(url, str):
        return False
    return parse_url(url).scheme in _VALID_URLS

@overload
def _expand_user(filepath_or_buffer: str) -> str:
    ...

@overload
def _expand_user(filepath_or_buffer: Any) -> Any:
    ...

def _expand_user(filepath_or_buffer: Any) -> Any:
    if isinstance(filepath_or_buffer, str):
        return os.path.expanduser(filepath_or_buffer)
    return filepath_or_buffer

def validate_header_arg(header: Any) -> None:
    if header is None:
        return
    if is_integer(header):
        header = cast(int, header)
        if header < 0:
            raise ValueError('Passing negative integer to header is invalid. For no header, use header=None instead')
        return
    if is_list_like(header, allow_sets=False):
        header = cast(Sequence, header)
        if not all(map(is_integer, header)):
            raise ValueError('header must be integer or list of integers')
        if any((i < 0 for i in header)):
            raise ValueError('cannot specify multi-index header with negative integers')
        return
    if is_bool(header):
        raise TypeError('Passing a bool to header is invalid. Use header=None for no header or header=int or list-like of ints to specify the row(s) making up the column names')
    raise ValueError('header must be integer or list of integers')

@overload
def stringify_path(filepath_or_buffer: str, convert_file_like: bool = ...) -> str:
    ...

@overload
def stringify_path(filepath_or_buffer: Any, convert_file_like: bool = ...) -> Any:
    ...

def stringify_path(filepath_or_buffer: Any, convert_file_like: bool = False) -> Any:
    if not convert_file_like and is_file_like(filepath_or_buffer):
        return cast(BaseBufferT, filepath_or_buffer)
    if isinstance(filepath_or_buffer, os.PathLike):
        filepath_or_buffer = filepath_or_buffer.__fspath__()
    return _expand_user(filepath_or_buffer)

def urlopen(*args: Any, **kwargs: Any) -> Any:
    import urllib.request
    return urllib.request.urlopen(*args, **kwargs)

def is_fsspec_url(url: Union[str, bytes]) -> bool:
    return isinstance(url, str) and bool(_RFC_3986_PATTERN.match(url)) and (not url.startswith(('http://', 'https://')))

@doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'filepath_or_buffer')
def _get_filepath_or_buffer(filepath_or_buffer: Any, encoding: str = 'utf-8', compression: Optional[Union[str, Mapping[str, Any]]] = None, mode: str = 'r', storage_options: Optional[Mapping[str, Any]] = None) -> IOArgs:
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    compression_method, compression = get_compression_method(compression)
    compression_method = infer_compression(filepath_or_buffer, compression_method)
    if compression_method and hasattr(filepath_or_buffer, 'write') and ('b' not in mode):
        warnings.warn('compression has no effect when passing a non-binary object as input.', RuntimeWarning, stacklevel=find_stack_level())
        compression_method = None
    compression = dict(compression, method=compression_method)
    if 'w' in mode and compression_method in ['bz2', 'xz'] and (encoding in ['utf-16', 'utf-32']):
        warnings.warn(f'{compression} will not write the byte order mark for {encoding}', UnicodeWarning, stacklevel=find_stack_level())
    if 'a' in mode and compression_method in ['zip', 'tar']:
        warnings.warn("zip and tar do not support mode 'a' properly. This combination will result in multiple files with same name being added to the archive.", RuntimeWarning, stacklevel=find_stack_level())
    fsspec_mode = mode
    if 't' not in fsspec_mode and 'b' not in fsspec_mode:
        fsspec_mode += 'b'
    if isinstance(filepath_or_buffer, str) and is_url(filepath_or_buffer):
        storage_options = storage_options or {}
        import urllib.request
        req_info = urllib.request.Request(filepath_or_buffer, headers=storage_options)
        with urlopen(req_info) as req:
            content_encoding = req.headers.get('Content-Encoding', None)
            if content_encoding == 'gzip':
                compression = {'method': 'gzip'}
            reader = BytesIO(req.read())
        return IOArgs(filepath_or_buffer=reader, encoding=encoding, compression=compression, should_close=True, mode=fsspec_mode)
    if is_fsspec_url(filepath_or_buffer):
        assert isinstance(filepath_or_buffer, str)
        if filepath_or_buffer.startswith('s3a://'):
            filepath_or_buffer = filepath_or_buffer.replace('s3a://', 's3://')
        if filepath_or_buffer.startswith('s3n://'):
            filepath_or_buffer = filepath_or_buffer.replace('s3n://', 's3://')
        fsspec = import_optional_dependency('fsspec')
        err_types_to_retry_with_anon = []
        try:
            import_optional_dependency('botocore')
            from botocore.exceptions import ClientError, NoCredentialsError
            err_types_to_retry_with_anon = [ClientError, NoCredentialsError, PermissionError]
        except ImportError:
            pass
        try:
            file_obj = fsspec.open(filepath_or_buffer, mode=fsspec_mode, **storage_options or {}).open()
        except tuple(err_types_to_retry_with_anon):
            if storage_options is None:
                storage_options = {'anon': True}
            else:
                storage_options = dict(storage_options)
                storage_options['anon'] = True
            file_obj = fsspec.open(filepath_or_buffer, mode=fsspec_mode, **storage_options or {}).open()
        return IOArgs(filepath_or_buffer=file_obj, encoding=encoding, compression=compression, should_close=True, mode=fsspec_mode)
    elif storage_options:
        raise ValueError('storage_options passed with file object or non-fsspec file path')
    if isinstance(filepath_or_buffer, (str, bytes, mmap.mmap)):
        return IOArgs(filepath_or_buffer=_expand_user(filepath_or_buffer), encoding=encoding, compression=compression, should_close=False, mode=mode)
    if not (hasattr(filepath_or_buffer, 'read') or hasattr(filepath_or_buffer, 'write')):
        msg = f'Invalid file path or buffer object type: {type(filepath_or_buffer)}'
        raise ValueError(msg)
    return IOArgs(filepath_or_buffer=filepath_or_buffer, encoding=encoding, compression=compression, should_close=False, mode=mode)

def file_path_to_url(path: Union[str, os.PathLike]) -> str:
    from urllib.request import pathname2url
    return urljoin('file:', pathname2url(path))

extension_to_compression: Mapping[str, str] = {'.tar': 'tar', '.tar.gz': 'tar', '.tar.bz2': 'tar', '.tar.xz': 'tar', '.gz': 'gzip', '.bz2': 'bz2', '.zip': 'zip', '.xz': 'xz', '.zst': 'zstd'}
_supported_compressions: set[str] = set(extension_to_compression.values())

def get_compression_method(compression: Optional[Union[str, Mapping[str, Any]]]) -> Tuple[Optional[str], Mapping[str, Any]]:
    if isinstance(compression, Mapping):
        compression_args = dict(compression)
        try:
            compression_method = compression_args.pop('method')
        except KeyError as err:
            raise ValueError("If mapping, compression must have key 'method'") from err
    else:
        compression_args = {}
        compression_method = compression
    return (compression_method, compression_args)

@doc(compression_options=_shared_docs['compression_options'] % 'filepath_or_buffer')
def infer_compression(filepath_or_buffer: Any, compression: Optional[str]) -> Optional[str]:
    if compression is None:
        return None
    if compression == 'infer':
        if isinstance(filepath_or_buffer, str) and '::' in filepath_or_buffer:
            filepath_or_buffer = filepath_or_buffer.split('::')[0]
        filepath_or_buffer = stringify_path(filepath_or_buffer, convert_file_like=True)
        if not isinstance(filepath_or_buffer, str):
            return None
        for extension, compression in extension_to_compression.items():
            if filepath_or_buffer.lower().endswith(extension):
                return compression
        return None
    if compression in _supported_compressions:
        return compression
    valid = ['infer', None] + sorted(_supported_compressions)
    msg = f'Unrecognized compression type: {compression}\nValid compression types are {valid}'
    raise ValueError(msg)

def check_parent_directory(path: Union[str, os.PathLike]) -> None:
    parent = Path(path).parent
    if not parent.is_dir():
        raise OSError(f"Cannot save file into a non-existent directory: '{parent}'")

@overload
def get_handle(path_or_buf: Union[str, IO[Any]], mode: str, *, encoding: Optional[str] = ..., compression: Optional[Union[str, Mapping[str, Any]]] = ..., memory_map: bool = ..., is_text: bool, errors: Optional[str] = ..., storage_options: Optional[Mapping[str, Any]] = ...) -> IOHandles:
    ...

@overload
def get_handle(path_or_buf: Union[str, IO[Any]], mode: str, *, encoding: Optional[str] = ..., compression: Optional[Union[str, Mapping[str, Any]]] = ..., memory_map: bool = ..., is_text: bool = ..., errors: Optional[str] = ..., storage_options: Optional[Mapping[str, Any]] = ...) -> IOHandles:
    ...

@overload
def get_handle(path_or_buf: Union[str, IO[Any]], mode: str, *, encoding: Optional[str] = ..., compression: Optional[Union[str, Mapping[str, Any]]] = ..., memory_map: bool = ..., is_text: bool = ..., errors: Optional[str] = ..., storage_options: Optional[Mapping[str, Any]] = ...) -> IOHandles:
    ...

@doc(compression_options=_shared_docs['compression_options'] % 'path_or_buf')
def get_handle(path_or_buf: Union[str, IO[Any]], mode: str, *, encoding: Optional[str] = None, compression: Optional[Union[str, Mapping[str, Any]]] = None, memory_map: bool = False, is_text: bool = True, errors: Optional[str] = None, storage_options: Optional[Mapping[str, Any]] = None) -> IOHandles:
    encoding = encoding or 'utf-8'
    errors = errors or 'strict'
    if _is_binary_mode(path_or_buf, mode) and 'b' not in mode:
        mode += 'b'
    codecs.lookup(encoding)
    if isinstance(errors, str):
        codecs.lookup_error(errors)
    ioargs = _get_filepath_or_buffer(path_or_buf, encoding=encoding, compression=compression, mode=mode, storage_options=storage_options)
    handle = ioargs.filepath_or_buffer
    handle, memory_map, handles = _maybe_memory_map(handle, memory_map)
    is_path = isinstance(handle, str)
    compression_args = dict(ioargs.compression)
    compression = compression_args.pop('method')
    if 'r' not in mode and is_path:
        check_parent_directory(str(handle))
    if compression:
        if compression != 'zstd':
            ioargs.mode = ioargs.mode.replace('t', '')
        elif compression == 'zstd' and 'b' not in ioargs.mode:
            ioargs.mode += 'b'
        if compression == 'gzip':
            if isinstance(handle, str):
                handle = gzip.GzipFile(filename=handle, mode=ioargs.mode, **compression_args)
            else:
                handle = gzip.GzipFile(fileobj=handle, mode=ioargs.mode, **compression_args)
        elif compression == 'bz2':
            import bz2
            handle = bz2.BZ2File(handle, mode=ioargs.mode, **compression_args)
        elif compression == 'zip':
            handle = _BytesZipFile(handle, ioargs.mode, **compression_args)
            if handle.buffer.mode == 'r':
                handles.append(handle)
                zip_names = handle.buffer.namelist()
                if len(zip_names) == 1:
                    handle = handle.buffer.open(zip_names.pop())
                elif not zip_names:
                    raise ValueError(f'Zero files found in ZIP file {path_or_buf}')
                else:
                    raise ValueError(f'Multiple files found in ZIP file. Only one file per ZIP: {zip_names}')
        elif compression == 'tar':
            compression_args.setdefault('mode', ioargs.mode)
            if isinstance(handle, str):
                handle = _BytesTarFile(name=handle, **compression_args)
            else:
                handle = _BytesTarFile(fileobj=handle, **compression_args)
            assert isinstance(handle, _BytesTarFile)
            if 'r' in handle.buffer.mode:
                handles.append(handle)
                files = handle.buffer.getnames()
                if len(files) == 1:
                    file = handle.buffer.extractfile(files[0])
                    assert file is not None
                    handle = file
                elif not files:
                    raise ValueError(f'Zero files found in TAR archive {path_or_buf}')
                else:
                    raise ValueError(f'Multiple files found in TAR archive. Only one file per TAR archive: {files}')
        elif compression == 'xz':
            import lzma
            handle = lzma.LZMAFile(handle, ioargs.mode, **compression_args)
        elif compression == 'zstd':
            zstd = import_optional_dependency('zstandard')
            if 'r' in ioargs.mode:
                open_args = {'dctx': zstd.ZstdDecompressor(**compression_args)}
            else:
                open_args = {'cctx': zstd.ZstdCompressor(**compression_args)}
            handle = zstd.open(handle, mode=ioargs.mode, **open_args)
        else:
            msg = f'Unrecognized compression type: {compression}'
            raise ValueError(msg)
        assert not isinstance(handle, str)
        handles.append(handle)
    elif isinstance(handle, str):
        if ioargs.encoding and 'b' not in ioargs.mode:
            handle = open(handle, ioargs.mode, encoding=ioargs.encoding, errors=errors, newline='')
        else:
            handle = open(handle, ioargs.mode)
        handles.append(handle)
    is_wrapped = False
    if not is_text and ioargs.mode == 'rb' and isinstance(handle, TextIOBase):
        handle = _BytesIOWrapper(handle, encoding=ioargs.encoding)
    elif is_text and (compression or memory_map or _is_binary_mode(handle, ioargs.mode)):
        if not hasattr(handle, 'readable') or not hasattr(handle, 'writable') or (not hasattr(handle, 'seekable')):
            handle = _IOWrapper(handle)
        handle = TextIOWrapper(handle, encoding=ioargs.encoding, errors=errors, newline='')
        handles.append(handle)
        is_wrapped = not (isinstance(ioargs.filepath_or_buffer, str) or ioargs.should_close)
    if 'r' in ioargs.mode and (not hasattr(handle, 'read')):
        raise TypeError(f'Expected file path name or file-like object, got {type(ioargs.filepath_or_buffer)} type')
    handles.reverse()
    if ioargs.should_close:
        assert not isinstance(ioargs.filepath_or_buffer, str)
        handles.append(ioargs.filepath_or_buffer)
    return IOHandles(handle=handle, created_handles=handles, is_wrapped=is_wrapped, compression=ioargs.compression)

class _BufferedWriter(BytesIO, ABC):
    buffer: BytesIO = BytesIO()

    @abstractmethod
    def write_to_buffer(self) -> None:
        ...

    def close(self) -> None:
        if self.closed:
            return
        if self.getbuffer().nbytes:
            self.seek(0)
            with self.buffer:
                self.write_to_buffer()
        else:
            self.buffer.close()
        super().close()

class _BytesTarFile(_BufferedWriter):

    def __init__(self, name: Optional[str] = None, mode: str = 'r', fileobj: Optional[IO[bytes]] = None, archive_name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__()
        self.archive_name = archive_name
        self.name = name
        self.buffer = tarfile.TarFile.open(name=name, mode=self.extend_mode(mode), fileobj=fileobj, **kwargs)

    def extend_mode(self, mode: str) -> str:
        mode = mode.replace('b', '')
        if mode != 'w':
            return mode
        if self.name is not None:
            suffix = Path(self.name).suffix
            if suffix in ('.gz', '.xz', '.bz2'):
                mode = f'{mode}:{suffix[1:]}'
        return mode

    def infer_filename(self) -> Optional[str]:
        if self.name is None:
            return None
        filename = Path(self.name)
        if filename.suffix == '.tar':
            return filename.with_suffix('').name
        elif filename.suffix in ('.tar.gz', '.tar.bz2', '.tar.xz'):
            return filename.with_suffix('').with_suffix('').name
        return filename.name

    def write_to_buffer(self) -> None:
        archive_name = self.archive_name or self.infer_filename() or 'tar'
        tarinfo = tarfile.TarInfo(name=archive_name)
        tarinfo.size = len(self.getvalue())
        self.buffer.addfile(tarinfo, self)

class _BytesZipFile(_BufferedWriter):

    def __init__(self, file: Union[str, IO[bytes]], mode: str, archive_name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__()
        mode = mode.replace('b', '')
        self.archive_name = archive_name
        kwargs.setdefault('compression', zipfile.ZIP_DEFLATED)
        self.buffer = zipfile.ZipFile(file, mode, **kwargs)

    def infer_filename(self) -> Optional[str]:
        if isinstance(self.buffer.filename, (os.PathLike, str)):
            filename = Path(self.buffer.filename)
            if filename.suffix == '.zip':
                return filename.with_suffix('').name
            return filename.name
        return None

    def write_to_buffer(self) -> None:
        archive_name = self.archive_name or self.infer_filename() or 'zip'
        self.buffer.writestr(archive_name, self.getvalue())

class _IOWrapper:

    def __init__(self, buffer: IO[Any]) -> None:
        self.buffer = buffer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.buffer, name)

    def readable(self) -> bool:
        if hasattr(self.buffer, 'readable'):
            return self.buffer.readable()
        return True

    def seekable(self) -> bool:
        if hasattr(self.buffer, 'seekable'):
            return self.buffer.seekable()
        return True

    def writable(self) -> bool:
        if hasattr(self.buffer, 'writable'):
            return self.buffer.writable()
        return True

class _BytesIOWrapper:

    def __init__(self, buffer: TextIOBase, encoding: str = 'utf-8') -> None:
        self.buffer = buffer
        self.encoding = encoding
        self.overflow = b''

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.buffer, attr)

    def read(self, n: int = -1) -> bytes:
        assert self.buffer is not None
        bytestring = self.buffer.read(n).encode(self.encoding)
        combined_bytestring = self.overflow + bytestring
        if n is None or n < 0 or n >= len(combined_bytestring):
            self.overflow = b''
            return combined_bytestring
        else:
            to_return = combined_bytestring[:n]
            self.overflow = combined_bytestring[n:]
            return to_return

def _maybe_memory_map(handle: Union[str, IO[bytes]], memory_map: bool) -> Tuple[Union[str, IO[bytes]], bool, List[IO[bytes]]]:
    handles: List[IO[bytes]] = []
    memory_map &= hasattr(handle, 'fileno') or isinstance(handle, str)
    if not memory_map:
        return (handle, memory_map, handles)
    handle = cast(ReadCsvBuffer, handle)
    if isinstance(handle, str):
        handle = open(handle, 'rb')
        handles.append(handle)
    try:
        wrapped = _IOWrapper(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
    finally:
        for handle in reversed(handles):
            handle.close()
    return (wrapped, memory_map, [wrapped])

def file_exists(filepath_or_buffer: Any) -> bool:
    exists = False
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    if not isinstance(filepath_or_buffer, str):
        return exists
    try:
        exists = os.path.exists(filepath_or_buffer)
    except (TypeError, ValueError):
        pass
    return exists

def _is_binary_mode(handle: Any, mode: str) -> bool:
    if 't' in mode or 'b' in mode:
        return 'b' in mode
    text_classes = (codecs.StreamWriter, codecs.StreamReader, codecs.StreamReaderWriter)
    if issubclass(type(handle), text_classes):
        return False
    return isinstance(handle, _get_binary_io_classes()) or 'b' in getattr(handle, 'mode', mode)

@functools.lru_cache
def _get_binary_io_classes() -> Tuple[type, ...]:
    binary_classes: Tuple[type, ...] = (BufferedIOBase, RawIOBase)
    zstd = import_optional_dependency('zstandard', errors='ignore')
    if zstd is not None:
        with zstd.ZstdDecompressor().stream_reader(b'') as reader:
            binary_classes += (type(reader),)
    return binary_classes

def is_potential_multi_index(columns: Sequence[Hashable], index_col: Optional[Union[bool, Sequence[int]]] = None) -> bool:
    if index_col is None or isinstance(index_col, bool):
        index_columns = set()
    else:
        index_columns = set(index_col)
    return bool(len(columns) and (not isinstance(columns, ABCMultiIndex)) and all((isinstance(c, tuple) for c in columns if c not in index_columns)))

def dedup_names(names: Sequence[Hashable], is_potential_multiindex: bool) -> List[Hashable]:
    names = list(names)
    counts: DefaultDict[Hashable, int] = defaultdict(int)
    for i, col in enumerate(names):
        cur_count = counts[col]
        while cur_count > 0:
            counts[col] = cur_count + 1
            if is_potential_multiindex:
                assert isinstance(col, tuple)
                col = col[:-1] + (f'{col[-1]}.{cur_count}',)
            else:
                col = f'{col}.{cur_count}'
            cur_count = counts[col]
        names[i] = col
        counts[col] = cur_count + 1
    return names
