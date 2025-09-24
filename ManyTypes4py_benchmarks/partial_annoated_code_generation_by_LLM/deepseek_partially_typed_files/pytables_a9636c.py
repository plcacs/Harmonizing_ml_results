"""
High level interface to PyTables for reading and writing pandas data structures
to disk
"""
from __future__ import annotations
from contextlib import suppress
import copy
from datetime import date, tzinfo
import itertools
import os
import re
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final, Literal, cast, overload
import warnings
import numpy as np
from pandas._config import config, get_option, using_string_dtype
from pandas._libs import lib, writers as libwriters
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import AttributeConflictWarning, ClosedFileError, IncompatibilityWarning, PerformanceWarning, PossibleDataLossError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import ensure_object, is_bool_dtype, is_complex_dtype, is_list_like, is_string_dtype, needs_i8_conversion
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, ExtensionDtype, PeriodDtype
from pandas.core.dtypes.missing import array_equivalent
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, PeriodIndex, RangeIndex, Series, StringDtype, TimedeltaIndex, concat, isna
from pandas.core.arrays import Categorical, DatetimeArray, PeriodArray
from pandas.core.arrays.datetimes import tz_to_dtype
from pandas.core.arrays.string_ import BaseStringArray
import pandas.core.common as com
from pandas.core.computation.pytables import PyTablesExpr, maybe_expression
from pandas.core.construction import array as pd_array, extract_array
from pandas.core.indexes.api import ensure_index
from pandas.io.common import stringify_path
from pandas.io.formats.printing import adjoin, pprint_thing
if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Sequence
    from types import TracebackType
    from tables import Col, File, Node
    from pandas._typing import AnyArrayLike, ArrayLike, AxisInt, DtypeArg, FilePath, Self, Shape, npt
    from pandas.core.internals import Block
_version: Final[str] = '0.15.2'
_default_encoding: Final[str] = 'UTF-8'

def _ensure_encoding(encoding: str | None) -> str:
    if encoding is None:
        encoding = _default_encoding
    return encoding

def _ensure_str(name: Any) -> Any:
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype. Non-string dtypes are passed through unchanged.

    https://github.com/pandas-dev/pandas/issues/13492
    """
    if isinstance(name, str):
        name = str(name)
    return name
Term: type[PyTablesExpr] = PyTablesExpr

def _ensure_term(where: Any, scope_level: int) -> Any:
    """
    Ensure that the where is a Term or a list of Term.

    This makes sure that we are capturing the scope of variables that are
    passed create the terms here with a frame_level=2 (we are 2 levels down)
    """
    level: int = scope_level + 1
    if isinstance(where, (list, tuple)):
        where = [Term(term, scope_level=level + 1) if maybe_expression(term) else term for term in where if term is not None]
    elif maybe_expression(where):
        where = Term(where, scope_level=level)
    return where if where is None or len(where) else None
incompatibility_doc: Final[str] = '\nwhere criteria is being ignored as this version [%s] is too old (or\nnot-defined), read the file in and write it out to a new file to upgrade (with\nthe copy_to method)\n'
attribute_conflict_doc: Final[str] = '\nthe [%s] attribute of the existing index is [%s] which conflicts with the new\n[%s], resetting the attribute to None\n'
performance_doc: Final[str] = '\nyour performance may suffer as PyTables will pickle object types that it cannot\nmap directly to c-types [inferred_type->%s,key->%s] [items->%s]\n'
_FORMAT_MAP: Final[dict[str, str]] = {'f': 'fixed', 'fixed': 'fixed', 't': 'table', 'table': 'table'}
_AXES_MAP: Final[dict[type, list[int]]] = {DataFrame: [0]}
dropna_doc: Final[str] = '\n: boolean\n    drop ALL nan rows when appending to a table\n'
format_doc: Final[str] = "\n: format\n    default format writing format, if None, then\n    put will default to 'fixed' and append will default to 'table'\n"
with config.config_prefix('io.hdf'):
    config.register_option('dropna_table', False, dropna_doc, validator=config.is_bool)
    config.register_option('default_format', None, format_doc, validator=config.is_one_of_factory(['fixed', 'table', None]))
_table_mod: Any | None = None
_table_file_open_policy_is_strict: bool = False

def _tables() -> Any:
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables
        _table_mod = tables
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = tables.file._FILE_OPEN_POLICY == 'strict'
    return _table_mod

def to_hdf(path_or_buf: Any, key: str, value: DataFrame | Series, mode: str = 'a', complevel: int | None = None, complib: str | None = None, append: bool = False, format: str | None = None, index: bool = True, min_itemsize: int | dict[str, int] | None = None, nan_rep: str | None = None, dropna: bool | None = None, data_columns: Literal[True] | list[str] | None = None, errors: str = 'strict', encoding: str = 'UTF-8') -> None:
    """store this object, close it if we opened it"""
    if append:
        f: Callable[[HDFStore], None] = lambda store: store.append(key, value, format=format, index=index, min_itemsize=min_itemsize, nan_rep=nan_rep, dropna=dropna, data_columns=data_columns, errors=errors, encoding=encoding)
    else:
        f = lambda store: store.put(key, value, format=format, index=index, min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=data_columns, errors=errors, encoding=encoding, dropna=dropna)
    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        path_or_buf = stringify_path(path_or_buf)
        with HDFStore(path_or_buf, mode=mode, complevel=complevel, complib=complib) as store:
            f(store)

def read_hdf(path_or_buf: Any, key: str | None = None, mode: str = 'r', errors: str = 'strict', where: Any = None, start: int | None = None, stop: int | None = None, columns: list[str] | None = None, iterator: bool = False, chunksize: int | None = None, **kwargs: Any) -> Any:
    """
    Read from the store, close it if we opened it.

    Retrieve pandas object stored in file, optionally based on where
    criteria.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path_or_buf : str, path object, pandas.HDFStore
        Any valid string path is acceptable. Only supports the local file system,
        remote URLs and file-like objects are not supported.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        Alternatively, pandas accepts an open :class:`pandas.HDFStore` object.

    key : object, optional
        The group identifier in the store. Can be omitted if the HDF file
        contains a single pandas object.
    mode : {'r', 'r+', 'a'}, default 'r'
        Mode to use when opening the file. Ignored if path_or_buf is a
        :class:`pandas.HDFStore`. Default is 'r'.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    where : list, optional
        A list of Term (or convertible) objects.
    start : int, optional
        Row number to start selection.
    stop : int, optional
        Row number to stop selection.
    columns : list, optional
        A list of columns names to return.
    iterator : bool, optional
        Return an iterator object.
    chunksize : int, optional
        Number of rows to include in an iteration when using an iterator.
    **kwargs
        Additional keyword arguments passed to HDFStore.

    Returns
    -------
    object
        The selected object. Return type depends on the object stored.

    See Also
    --------
    DataFrame.to_hdf : Write a HDF file from a DataFrame.
    HDFStore : Low-level access to HDF files.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 1.0, "a"]], columns=["x", "y", "z"])  # doctest: +SKIP
    >>> df.to_hdf("./store.h5", "data")  # doctest: +SKIP
    >>> reread = pd.read_hdf("./store.h5")  # doctest: +SKIP
    """
    if mode not in ['r', 'r+', 'a']:
        raise ValueError(f'mode {mode} is not allowed while performing a read. Allowed modes are r, r+ and a.')
    if where is not None:
        where = _ensure_term(where, scope_level=1)
    if isinstance(path_or_buf, HDFStore):
        if not path_or_buf.is_open:
            raise OSError('The HDFStore must be open for reading.')
        store: HDFStore = path_or_buf
        auto_close: bool = False
    else:
        path_or_buf = stringify_path(path_or_buf)
        if not isinstance(path_or_buf, str):
            raise NotImplementedError('Support for generic buffers has not been implemented.')
        try:
            exists: bool = os.path.exists(path_or_buf)
        except (TypeError, ValueError):
            exists = False
        if not exists:
            raise FileNotFoundError(f'File {path_or_buf} does not exist')
        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        auto_close = True
    try:
        if key is None:
            groups: list[Any] = store.groups()
            if len(groups) == 0:
                raise ValueError('Dataset(s) incompatible with Pandas data types, not table, or no datasets found in HDF5 file.')
            candidate_only_group: Any = groups[0]
            for group_to_check in groups[1:]:
                if not _is_metadata_of(group_to_check, candidate_only_group):
                    raise ValueError('key must be provided when HDF5 file contains multiple datasets.')
            key = candidate_only_group._v_pathname
        return store.select(key, where=where, start=start, stop=stop, columns=columns, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
    except (ValueError, TypeError, LookupError):
        if not isinstance(path_or_buf, HDFStore):
            with suppress(AttributeError):
                store.close()
        raise

def _is_metadata_of(group: Any, parent_group: Any) -> bool:
    """Check if a given group is a metadata group for a given parent_group."""
    if group._v_depth <= parent_group._v_depth:
        return False
    current: Any = group
    while current._v_depth > 1:
        parent: Any = current._v_parent
        if parent == parent_group and current._v_name == 'meta':
            return True
        current = current._v_parent
    return False

class HDFStore:
    """
    Dict-like IO interface for storing pandas objects in PyTables.

    Either Fixed or Table format.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path : str
        File path to HDF5 file.
    mode : {'a', 'w', 'r', 'r+'}, default 'a'

        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.
    complevel : int, 0-9, default None
        Specifies a compression level for data.
        A value of 0 or None disables compression.
    complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
        Specifies the compression library to be used.
        These additional compressors for Blosc are supported
        (default if no compressor specified: 'blosc:blosclz'):
        {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
         'blosc:zlib', 'blosc:zstd'}.
        Specifying a compression library which is not available issues
        a ValueError.
    fletcher32 : bool, default False
        If applying compression use the fletcher32 checksum.
    **kwargs
        These parameters will be passed to the PyTables open_file method.

    Examples
    --------
    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore("test.h5")
    >>> store["foo"] = bar  # write to HDF5
    >>> bar = store["foo"]  # retrieve
    >>> store.close()

    **Create or load HDF5 file in-memory**

    When passing the `driver` option to the PyTables open_file method through
    **kwargs, the HDF5 file is loaded or created in-memory and will only be
    written when closed:

    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore("test.h5", driver="H5FD_CORE")
    >>> store["foo"] = bar
    >>> store.close()  # only now, data is written to disk
    """
    _handle: File | None
    _mode: str

    def __init__(self, path: str, mode: str = 'a', complevel: int | None = None, complib: str | None = None, fletcher32: bool = False, **kwargs: Any) -> None:
        if 'format' in kwargs:
            raise ValueError('format is not a defined argument for HDFStore')
        tables: Any = import_optional_dependency('tables')
        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(f'complib only supports {tables.filters.all_complibs} compression.')
        if complib is None and complevel is not None:
            complib = tables.filters.default_complib
        self._path: str = stringify_path(path)
        if mode is None:
            mode = 'a'
        self._mode: str = mode
        self._handle: File | None = None
        self._complevel: int = complevel if complevel else 0
        self._complib: str | None = complib
        self._fletcher32: bool = fletcher32
        self._filters: Any = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self) -> str:
        return self._path

    @property
    def root(self) -> Any:
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None
        return self._handle.root

    @property
    def filename(self) -> str:
        return self._path

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: DataFrame | Series) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        return self.remove(key)

    def __getattr__(self, name: str) -> Any:
        """allow attribute access to get stores"""
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __contains__(self, key: str) -> bool:
        """
        check for existence of this key
        can match the exact pathname or the pathnm w/o the leading '/'
        """
        node: Node | None = self.get_node(key)
        if node is not None:
            name: str = node._v_pathname
            if key in (name, name[1:]):
                return True
        return False

    def __len__(self) -> int:
