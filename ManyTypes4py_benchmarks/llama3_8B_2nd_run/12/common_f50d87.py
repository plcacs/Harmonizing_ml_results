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
_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard('')
_RFC_3986_PATTERN = re.compile('^[A-Za-z][A-Za-z0-9+\\-+.]*://')

@dataclasses.dataclass
class IOArgs:
    """Return value of io/common.py:_get_filepath_or_buffer.
    """
    should_close: bool

@dataclasses.dataclass
class IOHandles(Generic[AnyStr]):
    """Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    handle: The file handle to be used.
    created_handles: All file handles that are created by get_handle
    is_wrapped: Whether a TextIOWrapper needs to be detached.
    """
    created_handles: list
    is_wrapped: bool

    def close(self) -> None:
        """Close all created buffers.

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

    def __enter__(self) -> 'IOHandles[AnyStr]':
        return self

    def __exit__(self, exc_type: type, exc_value: type, traceback: type) -> None:
        self.close()

def is_url(url: str) -> bool:
    """Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
    if not isinstance(url, str):
        return False
    return parse_url(url).scheme in _VALID_URLS

@overload
def _expand_user(filepath_or_buffer: str) -> str:
    ...

@overload
def _expand_user(filepath_or_buffer: str) -> str:
    ...

def _expand_user(filepath_or_buffer: AnyStr) -> str:
    """Return the argument with an initial component of ~ or ~user
    replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : object to be converted if possible

    Returns
    -------
    expanded_filepath_or_buffer : an expanded filepath or the
                                  input if not expandable
    """
    if isinstance(filepath_or_buffer, str):
        return os.path.expanduser(filepath_or_buffer)
    return filepath_or_buffer

def validate_header_arg(header: Any) -> None:
    """Check if the header is None, integer, or list-like of integers."""
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
def stringify_path(filepath_or_buffer: str, convert_file_like: bool) -> str:
    ...

@overload
def stringify_path(filepath_or_buffer: str, convert_file_like: bool) -> str:
    ...

def stringify_path(filepath_or_buffer: AnyStr, convert_file_like: bool) -> str:
    """Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object to be converted

    Returns
    -------
    str_filepath_or_buffer : maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    if not convert_file_like and is_file_like(filepath_or_buffer):
        return cast(BaseBuffer, filepath_or_buffer)
    if isinstance(filepath_or_buffer, os.PathLike):
        filepath_or_buffer = filepath_or_buffer.__fspath__()
    return _expand_user(filepath_or_buffer)

def file_path_to_url(path: str) -> str:
    """converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """
    from urllib.request import pathname2url
    return urljoin('file:', pathname2url(path))

extension_to_compression = {'.tar': 'tar', '.tar.gz': 'tar', '.tar.bz2': 'tar', '.tar.xz': 'tar', '.gz': 'gzip', '.bz2': 'bz2', '.zip': 'zip', '.xz': 'xz', '.zst': 'zstd'}
_supported_compressions = set(extension_to_compression.values())

def get_compression_method(compression: Any) -> tuple:
    """Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of ({compression method}, Optional[str]
              {compression arguments}, Dict[str, Any])

    Raises
    ------
    ValueError on mapping missing 'method' key
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

@overload
def infer_compression(filepath_or_buffer: str, compression: str) -> str:
    ...

@overload
def infer_compression(filepath_or_buffer: str, compression: str) -> str:
    ...

@overload
def infer_compression(filepath_or_buffer: str, compression: str) -> str:
    ...

@doc(compression_options=_shared_docs['compression_options'] % 'filepath_or_buffer')
def infer_compression(filepath_or_buffer: str, compression: str) -> str:
    """Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : str or file handle
        File path or object.
    {compression_options}

           May be a dict with key 'method' as compression mode
           and other keys as compression options if compression
           mode is 'zip'.

           Passing compression options as keys in dict is
           supported for compression modes 'gzip', 'bz2', 'zstd' and 'zip'.

        .. versionchanged:: 1.4.0 Z