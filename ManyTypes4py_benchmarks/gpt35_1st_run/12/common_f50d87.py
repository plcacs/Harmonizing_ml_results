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
from typing import IO, TYPE_CHECKING, Any, AnyStr, DefaultDict, Generic, Literal, TypeVar, cast, overload
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
_VALID_URLS: set[str] = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard('')
_RFC_3986_PATTERN: re.Pattern = re.compile('^[A-Za-z][A-Za-z0-9+\\-+.]*://')
BaseBufferT = TypeVar('BaseBufferT', bound=BaseBuffer)
if TYPE_CHECKING:
    from types import TracebackType
    from pandas._typing import CompressionDict, CompressionOptions, FilePath, ReadBuffer, StorageOptions, WriteBuffer
    from pandas import MultiIndex

@dataclasses.dataclass
class IOArgs:
    """
    Return value of io/common.py:_get_filepath_or_buffer.
    """
    should_close: bool = False

@dataclasses.dataclass
class IOHandles(Generic[AnyStr]):
    """
    Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    created_handles: list = dataclasses.field(default_factory=list)
    is_wrapped: bool = False

    def close(self) -> None:
        """
        Close all created buffers.

        Note: If a TextIOWrapper was inserted, it is flushed and detached to
        avoid closing the potentially user-created buffer.
        """
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

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

def is_url(url: str) -> bool:
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str
        URL to check

    Returns
    -------
    bool
        If `url` has a valid protocol return True otherwise False.
    """
    if not isinstance(url, str):
        return False
    return parse_url(url).scheme in _VALID_URLS

@overload
def _expand_user(filepath_or_buffer: str) -> str:
    ...

@overload
def _expand_user(filepath_or_buffer: object) -> object:
    ...

def _expand_user(filepath_or_buffer: object) -> object:
    """
    Return the argument with an initial component of ~ or ~user
    replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : object
        Object to be converted if possible

    Returns
    -------
    object
        An expanded filepath or the input if not expandable
    """
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
def stringify_path(filepath_or_buffer: object, convert_file_like: bool = ...) -> object:
    ...

@overload
def stringify_path(filepath_or_buffer: object, convert_file_like: bool = ...) -> object:
    ...

def stringify_path(filepath_or_buffer: object, convert_file_like: bool = False) -> object:
    """
    Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object
        Object to be converted

    Returns
    -------
    object
        Maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    if not convert_file_like and is_file_like(filepath_or_buffer):
        return cast(BaseBufferT, filepath_or_buffer)
    if isinstance(filepath_or_buffer, os.PathLike):
        filepath_or_buffer = filepath_or_buffer.__fspath__()
    return _expand_user(filepath_or_buffer)

def urlopen(*args, **kwargs) -> Any:
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
    import urllib.request
    return urllib.request.urlopen(*args, **kwargs)

def is_fsspec_url(url: str) -> bool:
    """
    Returns true if the given URL looks like
    something fsspec can handle
    """
    return isinstance(url, str) and bool(_RFC_3986_PATTERN.match(url)) and (not url.startswith(('http://', 'https://')))

@doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'filepath_or_buffer')
def _get_filepath_or_buffer(filepath_or_buffer: object, encoding: str = 'utf-8', compression: Any = None, mode: str = 'r', storage_options: Any = None) -> IOArgs:
    """
    If the filepath_or_buffer is a url, translate and return the buffer.
    Otherwise passthrough.

    Parameters
    ----------
    filepath_or_buffer : object
        A url, filepath (str or pathlib.Path), or buffer
    encoding : str, optional
        The encoding to use to decode bytes, default is 'utf-8'
    mode : str, optional
        Mode to open the file with
    storage_options : Any, optional
        Storage options

    Returns
    -------
    IOArgs
        The dataclass IOArgs
    """
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

def file_path_to_url(path: str) -> str:
    """
    Converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : str
        A path in native format

    Returns
    -------
    str
        A valid FILE URL
    """
    from urllib.request import pathname2url
    return urljoin('file:', pathname2url(path))
extension_to_compression: dict[str, str] = {'.tar': 'tar', '.tar.gz': 'tar', '.tar.bz2': 'tar', '.tar.xz': 'tar', '.gz': 'gzip', '.bz2': 'bz2', '.zip': 'zip', '.xz': 'xz', '.zst': 'zstd'}
_supported_compressions: set[str] = set(extension_to_compression.values())

def get_compression_method(compression: Any) -> tuple[str, dict]:
    """
    Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of (str, dict)
        Compression method and compression arguments

    Raises
    ------
    ValueError
        If mapping is missing 'method' key
    """
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
def infer_compression(filepath_or_buffer: object, compression: Any) -> str:
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : object
        File path or object.
    compression : Any
        Compression method or 'infer'

    Returns
    -------
    str or None
        Compression method or None

    Raises
    ------
    ValueError
        If invalid compression specified
    """
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

def check_parent_directory(path: Path) -> None:
    """
    Check if parent directory of a file exists, raise OSError if it does not

    Parameters
    ----------
    path : Path
        Path to check parent directory of
    """
    parent = Path(path).parent
    if not parent.is_dir():
        raise OSError(f"Cannot save file into a non-existent directory: '{parent}'")

@overload
def get_handle(path_or_buf: object, mode: str, *, encoding: str = ..., compression: Any = ..., memory_map: bool = ..., is_text: bool, errors: str = ..., storage_options: Any = ...) -> IOHandles:
    ...

@overload
def get_handle(path_or_buf: object, mode: str, *, encoding: str = ..., compression: Any = ..., memory_map: bool = ..., is_text: bool = ..., errors: str = ..., storage_options: Any = ...) -> IOHandles:
    ...

@overload
def get_handle(path_or_buf: object, mode: str, *, encoding: str = ..., compression: Any = ..., memory_map: bool = ..., is_text: bool = ..., errors: str = ..., storage_options: Any = ...) -> IOHandles:
    ...

@doc(compression_options=_shared_docs['compression_options'] % 'path_or_buf')
def get_handle(path_or_buf: object, mode: str, *, encoding: str = None, compression: Any = None, memory_map: bool = False, is_text: bool = True, errors: str = None, storage_options: Any = None) -> IOHandles:
    """
    Get file handle for given path/buffer and mode.

    Parameters
    ----------
    path_or_buf : object
        File path or object.
    mode : str
        Mode to open path_or_buf with.
    encoding : str, optional
        Encoding to use.
    compression : Any, optional
        Compression method
    memory_map : bool, default False
        Whether to memory map the file/buffer
    is_text : bool, default True
        Whether the content is text or binary
    errors : str, default 'strict'
        How to handle encoding errors
    storage_options: Any, optional
        Storage options

    Returns
    -------
    IOHandles
        The dataclass IOHandles
    """
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
