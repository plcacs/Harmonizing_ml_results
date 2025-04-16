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
from typing import TYPE_CHECKING, Any, Final, Literal, cast, overload, Callable, Hashable, Iterator, Sequence, Union
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
_version: str = '0.15.2'
_default_encoding: str = 'UTF-8'


def _ensure_encoding(encoding):
    if encoding is None:
        encoding = _default_encoding
    return encoding


def _ensure_str(name):
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype. Non-string dtypes are passed through unchanged.

    https://github.com/pandas-dev/pandas/issues/13492
    """
    if isinstance(name, str):
        name = str(name)
    return name


Term = PyTablesExpr


def _ensure_term(where, scope_level):
    """
    Ensure that the where is a Term or a list of Term.

    This makes sure that we are capturing the scope of variables that are
    passed create the terms here with a frame_level=2 (we are 2 levels down)
    """
    level = scope_level + 1
    if isinstance(where, (list, tuple)):
        where = [(Term(term, scope_level=level + 1) if maybe_expression(
            term) else term) for term in where if term is not None]
    elif maybe_expression(where):
        where = Term(where, scope_level=level)
    return where if where is None or len(where) else None


incompatibility_doc: Final[str] = """
where criteria is being ignored as this version [%s] is too old (or
not-defined), read the file in and write it out to a new file to upgrade (with
the copy_to method)
"""
attribute_conflict_doc: Final[str] = """
the [%s] attribute of the existing index is [%s] which conflicts with the new
[%s], resetting the attribute to None
"""
performance_doc: Final[str] = """
your performance may suffer as PyTables will pickle object types that it cannot
map directly to c-types [inferred_type->%s,key->%s] [items->%s]
"""
_FORMAT_MAP: Final[dict[str, str]] = {'f': 'fixed', 'fixed': 'fixed', 't':
    'table', 'table': 'table'}
_AXES_MAP: Final[dict[type[Any], list[int]]] = {DataFrame: [0]}
dropna_doc: Final[str] = """
: boolean
    drop ALL nan rows when appending to a table
"""
format_doc: Final[str] = """
: format
    default format writing format, if None, then
    put will default to 'fixed' and append will default to 'table'
"""
with config.config_prefix('io.hdf'):
    config.register_option('dropna_table', False, dropna_doc, validator=
        config.is_bool)
    config.register_option('default_format', None, format_doc, validator=
        config.is_one_of_factory(['fixed', 'table', None]))
_table_mod: tables | None = None
_table_file_open_policy_is_strict: bool = False


def _tables():
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables
        _table_mod = tables
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (tables.file.
                _FILE_OPEN_POLICY == 'strict')
    assert _table_mod is not None
    return _table_mod


def to_hdf(path_or_buf, key, value, mode='a', complevel=None, complib=None,
    append=False, format=None, index=True, min_itemsize=None, nan_rep=None,
    dropna=None, data_columns=None, errors='strict', encoding='UTF-8'):
    """store this object, close it if we opened it"""
    if append:
        f = lambda store: store.append(key, value, format=format, index=
            index, min_itemsize=min_itemsize, nan_rep=nan_rep, dropna=
            dropna, data_columns=data_columns, errors=errors, encoding=encoding
            )
    else:
        f = lambda store: store.put(key, value, format=format, index=index,
            min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=
            data_columns, errors=errors, encoding=encoding, dropna=dropna)
    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        path_or_buf = stringify_path(path_or_buf)
        with HDFStore(path_or_buf, mode=mode, complevel=complevel, complib=
            complib) as store:
            f(store)


def read_hdf(path_or_buf, key=None, mode='r', errors='strict', where=None,
    start=None, stop=None, columns=None, iterator=False, chunksize=None, **
    kwargs: Any):
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
        raise ValueError(
            f'mode {mode} is not allowed while performing a read. Allowed modes are r, r+ and a.'
            )
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
            raise NotImplementedError(
                'Support for generic buffers has not been implemented.')
        try:
            exists = os.path.exists(path_or_buf)
        except (TypeError, ValueError):
            exists = False
        if not exists:
            raise FileNotFoundError(f'File {path_or_buf} does not exist')
        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        auto_close = True
    try:
        if key is None:
            groups = store.groups()
            if len(groups) == 0:
                raise ValueError(
                    'Dataset(s) incompatible with Pandas data types, not table, or no datasets found in HDF5 file.'
                    )
            candidate_only_group = groups[0]
            for group_to_check in groups[1:]:
                if not _is_metadata_of(group_to_check, candidate_only_group):
                    raise ValueError(
                        'key must be provided when HDF5 file contains multiple datasets.'
                        )
            key = candidate_only_group._v_pathname
        return store.select(key, where=where, start=start, stop=stop,
            columns=columns, iterator=iterator, chunksize=chunksize,
            auto_close=auto_close)
    except (ValueError, TypeError, LookupError):
        if not isinstance(path_or_buf, HDFStore):
            with suppress(AttributeError):
                store.close()
        raise


def _is_metadata_of(group, parent_group):
    """Check if a given group is a metadata group for a given parent_group."""
    if group._v_depth <= parent_group._v_depth:
        return False
    current = group
    while current._v_depth > 1:
        parent = current._v_parent
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

    def __init__(self, path, mode='a', complevel=None, complib=None,
        fletcher32=False, **kwargs: Any):
        if 'format' in kwargs:
            raise ValueError('format is not a defined argument for HDFStore')
        tables = import_optional_dependency('tables')
        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(
                f'complib only supports {tables.filters.all_complibs} compression.'
                )
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
        self._filters: tables.Filters | None = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self):
        return self._path

    @property
    def root(self):
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None
        return self._handle.root

    @property
    def filename(self):
        return self._path

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        return self.remove(key)

    def __getattr__(self, name):
        """allow attribute access to get stores"""
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __contains__(self, key):
        """
        check for existence of this key
        can match the exact pathname or the pathnm w/o the leading '/'
        """
        node = self.get_node(key)
        if node is not None:
            name = node._v_pathname
            if key in (name, name[1:]):
                return True
        return False

    def __len__(self):
        return len(self.groups())

    def __repr__(self):
        pstr = pprint_thing(self._path)
        return f'{type(self)}\nFile path: {pstr}\n'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def keys(self, include='pandas'):
        """
        Return a list of keys corresponding to objects stored in HDFStore.

        Parameters
        ----------

        include : str, default 'pandas'
                When kind equals 'pandas' return pandas objects.
                When kind equals 'native' return native HDF5 Table objects.

        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading '/').

        Raises
        ------
        raises ValueError if kind has an illegal value

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.close()  # doctest: +SKIP
        """
        if include == 'pandas':
            return [n._v_pathname for n in self.groups()]
        elif include == 'native':
            assert self._handle is not None
            return [n._v_pathname for n in self._handle.walk_nodes('/',
                classname='Table')]
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
            )

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        """
        iterate on key->group
        """
        for g in self.groups():
            yield g._v_pathname, g

    def open(self, mode='a', **kwargs: Any):
        """
        Open the file in the specified mode

        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
        tables_mod = _tables()
        if self._mode != mode:
            if self._mode in ['a', 'w'] and mode in ['r', 'r+']:
                pass
            elif mode in ['w']:
                if self.is_open:
                    raise PossibleDataLossError(
                        f'Re-opening the file [{self._path}] with mode [{self._mode}] will delete the current file!'
                        )
            self._mode = mode
        if self.is_open:
            self.close()
        if self._complevel and self._complevel > 0:
            self._filters = tables_mod.Filters(self._complevel, self.
                _complib, fletcher32=self._fletcher32)
        if _table_file_open_policy_is_strict and self.is_open:
            msg = (
                'Cannot open HDF5 file, which is already opened, even in read-only mode.'
                )
            raise ValueError(msg)
        self._handle = tables_mod.open_file(self._path, self._mode, **kwargs)

    def close(self):
        """
        Close the PyTables file handle
        """
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def is_open(self):
        """
        return a boolean indicating whether the file is open
        """
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def flush(self, fsync=False):
        """
        Force all buffered modifications to be written to disk.

        Parameters
        ----------
        fsync : bool (default False)
          call ``os.fsync()`` on the file handle to force writing to disk.

        Notes
        -----
        Without ``fsync=True``, flushing may not guarantee that the OS writes
        to disk. With fsync, the operation will block until the OS claims the
        file has been written; however, other caching layers may still
        interfere.
        """
        if self._handle is not None:
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def get(self, key):
        """
        Retrieve pandas object stored in file.

        Parameters
        ----------
        key : str
            Object to retrieve from file. Raises KeyError if not found.

        Returns
        -------
        object
            Same type as object stored in file.

        See Also
        --------
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        with patch_pickle():
            group = self.get_node(key)
            if group is None:
                raise KeyError(f'No object named {key} in the file')
            return self._read_group(group)

    def select(self, key, where=None, start=None, stop=None, columns=None,
        iterator=False, chunksize=None, auto_close=False):
        """
        Retrieve pandas object stored in file, optionally based on where criteria.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
            Object being retrieved from file.
        where : list or None
            List of Term (or convertible) objects, optional.
        start : int or None
            Row number to start selection.
        stop : int or None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number or rows to include in iteration, return an iterator.
        auto_close : bool or False
            Should automatically close the store when finished.

        Returns
        -------
        object
            Retrieved object from file.

        See Also
        --------
        HDFStore.select_as_coordinates : Returns the selection as an index.
        HDFStore.select_column : Returns a single column from the table.
        HDFStore.select_as_multiple : Retrieves pandas objects from multiple tables.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.select("/data1")  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        >>> store.select("/data1", where="columns == A")  # doctest: +SKIP
           A
        0  1
        1  3
        >>> store.close()  # doctest: +SKIP
        """
        group = self.get_node(key)
        if group is None:
            raise KeyError(f'No object named {key} in the file')
        where = _ensure_term(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        def func(_start, _stop, _where):
            return s.read(start=_start, stop=_stop, where=_where, columns=
                columns)
        it = TableIterator(self, s, func, where=where, nrows=s.nrows, start
            =start, stop=stop, iterator=iterator, chunksize=chunksize,
            auto_close=auto_close)
        return it.get_result()

    def select_as_coordinates(self, key, where=None, start=None, stop=None):
        """
        return the selection as an Index

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.


        Parameters
        ----------
        key : str
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection

        Returns
        -------
        Index

        """
        where = _ensure_term(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_coordinates with a table')
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def select_column(self, key, column, start=None, stop=None):
        """
        return a single column from the table. This is generally only useful to
        select an indexable

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
        column : str
            The column of interest.
        start : int or None, default None
        stop : int or None, default None

        Raises
        ------
        raises KeyError if the column is not found (or key is not a valid
            store)
        raises ValueError if the column can not be extracted individually (it
            is part of a data block)

        Returns
        -------
        Series
        """
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_column with a table')
        return tbl.read_column(column=column, start=start, stop=stop)

    def select_as_multiple(self, keys, where=None, selector=None, columns=
        None, start=None, stop=None, iterator=False, chunksize=None,
        auto_close=False):
        """
        Retrieve pandas objects from multiple tables.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        keys : list[str]
            a list of table keys
        selector : str | None, default None
            the table to apply the where criteria (defaults to keys[0]
                if not supplied)
        columns : list[str] | None, default None
            the columns to return
        start : int | None, default None
            row number to start selection
        stop : int | None, default None
            row number to stop selection
        iterator : bool, default False
            Whether to use the iterator.
        chunksize : int | None, default None
            Number of rows to include in an iteration, return an iterator.
        auto_close : bool, default False
            Should automatically close the store when finished.

        Raises
        ------
        raises KeyError if keys or selector is not found or keys is empty
        raises TypeError if keys is not a list or tuple
        raises ValueError if the tables are not ALL THE SAME DIMENSIONS

        Returns
        -------
        Any
            Combined DataFrame

        """
        where = _ensure_term(where, scope_level=1)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(key=keys, where=where, columns=columns,
                start=start, stop=stop, iterator=iterator, chunksize=
                chunksize, auto_close=auto_close)
        if not isinstance(keys, (list, tuple)):
            raise TypeError('keys must be a list/tuple')
        if not len(keys):
            raise ValueError('keys must have a non-zero length')
        if selector is None:
            selector = keys[0]
        tbls = [self.get_storer(k) for k in keys]
        s = self.get_storer(selector)
        nrows = None
        for t, k in itertools.chain([(s, selector)], zip(tbls, keys)):
            if t is None:
                raise KeyError(f'Invalid table [{k}]')
            if not t.is_table:
                raise TypeError(
                    f'object [{t.pathname}] is not a table, and cannot be used in all select as multiple'
                    )
            if nrows is None:
                nrows = t.nrows
            elif t.nrows != nrows:
                raise ValueError('all tables must have exactly the same nrows!'
                    )
        _tbls = [x for x in tbls if isinstance(x, Table)]
        axis = {t.non_index_axes[0][0] for t in _tbls}.pop()

        def func(_start, _stop, _where):
            objs = [t.read(where=_where, columns=columns, start=_start,
                stop=_stop) for t in tbls]
            return concat(objs, axis=axis, verify_integrity=False
                )._consolidate()
        it = TableIterator(self, s, func, where=where, nrows=nrows, start=
            start, stop=stop, iterator=iterator, chunksize=chunksize,
            auto_close=auto_close)
        return it.get_result(coordinates=True)

    def put(self, key, value, format=None, index=True, append=False,
        complib=None, complevel=None, min_itemsize=None, nan_rep=None,
        data_columns=None, encoding=None, errors='strict', track_times=True,
        dropna=False):
        """
        Store object in HDFStore.

        This method writes a pandas DataFrame or Series into an HDF5 file using
        either the fixed or table format. The `table` format allows additional
        operations like incremental appends and queries but may have performance
        trade-offs. The `fixed` format provides faster read/write operations but
        does not support appends or queries.

        Parameters
        ----------
        key : str
            Key of object to store in file.
        value : {Series, DataFrame}
            Value of object to store in file.
        format : 'fixed(f)|table(t)', default is 'fixed'
            Format to use when storing object in HDFStore. Value can be one of:

            ``'fixed'``
                Fixed format.  Fast writing/reading. Not-appendable, nor searchable.
            ``'table'``
                Table format.  Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        index : bool, default True
            Write DataFrame index as a column.
        append : bool, default False
            This will force Table format, append the input data to the existing.
        complib : str | None, default None
            This parameter is currently not accepted.
        complevel : int, 0-9, default None
            Specifies a compression level for data.
            A value of 0 or None disables compression.
        min_itemsize : int, dict, or None
            Dict of columns that specify minimum str sizes.
        nan_rep : Any
            Str to use as str nan representation.
        data_columns : list of columns or True, default None
            List of columns to create as indexed data columns, or True to use all columns.
            See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : str | None, default None
            Provide an encoding for strings.
        errors : str, default 'strict'
            The error handling scheme to use for encoding errors.
            The default is 'strict' meaning that encoding errors raise a
            UnicodeEncodeError.  Other possible values are 'ignore', 'replace' and
            'xmlcharrefreplace' as well as any other name registered with
            codecs.register_error that can handle UnicodeEncodeErrors.
        track_times : bool, default True
            Parameter is propagated to 'create_table' method of 'PyTables'.
            If set to False it enables to have the same h5 files (same hashes)
            independent on creation time.
        dropna : bool, default False, optional
            Remove missing values.

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        """
        if format is None:
            format = get_option('io.hdf.default_format') or 'fixed'
        format = self._validate_format(format)
        self._write_to_group(key, value, format=format, index=index, append
            =append, complib=complib, complevel=complevel, min_itemsize=
            min_itemsize, nan_rep=nan_rep, data_columns=data_columns,
            encoding=encoding, errors=errors, track_times=track_times,
            dropna=dropna)

    def remove(self, key, where=None, start=None, stop=None):
        """
        Remove pandas object partially by specifying the where condition

        Parameters
        ----------
        key : str
            Node to remove or delete rows from
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection

        Returns
        -------
        int | None
            number of rows removed (or None if not a Table)

        Raises
        ------
        KeyError
            if key is not a valid store

        """
        where = _ensure_term(where, scope_level=1)
        try:
            s = self.get_storer(key)
        except KeyError:
            raise
        except AssertionError:
            raise
        except Exception as err:
            if where is not None:
                raise ValueError(
                    'trying to remove a node with a non-None where clause!'
                    ) from err
            node = self.get_node(key)
            if node is not None:
                node._f_remove(recursive=True)
                return None
        if com.all_none(where, start, stop):
            s.group._f_remove(recursive=True)
            return None
        if not s.is_table:
            raise ValueError(
                'can only remove with where on objects written as tables')
        return s.delete(where=where, start=start, stop=stop)

    def append(self, key, value, format=None, axes=None, index=True, append
        =True, complib=None, complevel=None, columns=None, min_itemsize=
        None, nan_rep=None, chunksize=None, expectedrows=None, dropna=None,
        data_columns=None, encoding=None, errors='strict'):
        """
        Append to Table in file.

        Node must already exist and be Table format.

        Parameters
        ----------
        key : str
            Key of object to append.
        value : {Series, DataFrame}
            Value of object to append.
        format : 'table' is the default
            Format to use when storing object in HDFStore.  Value can be one of:

            ``'table'``
                Table format. Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        axes : default None
            This parameter is currently not accepted.
        index : bool | list[str], default True
            Write DataFrame index as a column.
        append : bool, default True
            Append the input data to the existing.
        complib : str | None, default None
            This parameter is currently not accepted.
        complevel : int, 0-9, default None
            Specifies a compression level for data.
            A value of 0 or None disables compression.
        columns : Any
            This parameter is currently not accepted, try data_columns.
        min_itemsize : int | dict[str, int] | None
            Dict of columns that specify minimum str sizes.
        nan_rep : Any
            Str to use as str nan representation.
        chunksize : int | None
            Size to chunk the writing.
        expectedrows : int | None
            Expected TOTAL row size of this table.
        dropna : bool | None, default False, optional
            Do not write an ALL nan row to the store settable
            by the option 'io.hdf.dropna_table'.
        data_columns : Literal[True] | list[str] | None, default None
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : str | None, default None
            Provide an encoding for str.
        errors : str, default 'strict'
            The error handling scheme to use for encoding errors.
            The default is 'strict' meaning that encoding errors raise a
            UnicodeEncodeError.  Other possible values are 'ignore', 'replace' and
            'xmlcharrefreplace' as well as any other name registered with
            codecs.register_error that can handle UnicodeEncodeErrors.

        See Also
        --------
        HDFStore.append_to_multiple : Append to multiple tables.

        Notes
        -----
        Does *not* check if data being appended overlaps with existing
        data in the table, so be careful

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df1, format="table")  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["A", "B"])
        >>> store.append("data", df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        0  5  6
        1  7  8
        """
        if columns is not None:
            raise TypeError(
                'columns is not a supported keyword in append, try data_columns'
                )
        if dropna is None:
            dropna = get_option('io.hdf.dropna_table')
        if format is None:
            format = get_option('io.hdf.default_format') or 'table'
        format = self._validate_format(format)
        self._write_to_group(key, value, format=format, axes=axes, index=
            index, append=append, complib=complib, complevel=complevel,
            min_itemsize=min_itemsize, nan_rep=nan_rep, chunksize=chunksize,
            expectedrows=expectedrows, dropna=dropna, data_columns=
            data_columns, encoding=encoding, errors=errors)

    def append_to_multiple(self, d, value, selector, data_columns=None,
        axes=None, dropna=False, **kwargs: Any):
        """
        Append to multiple tables

        Parameters
        ----------
        d : dict[str, Any]
            a dict of table_name to table_columns, None is acceptable as the
            values of one node (this will get all the remaining columns)
        value : Any
            a pandas object
        selector : str
            a string that designates the indexable table; all of its
            columns will be designed as data_columns, unless data_columns is
            passed, in which case these are used
        data_columns : Literal[True] | list[str] | None, default None
            Specify the columns to index.
        dropna : bool, default False
            Drop rows where all specified tables have NaN.
        **kwargs : Any
            Additional keyword arguments passed to append.

        Raises
        ------
        TypeError
            if axes is not accepted
        ValueError
            if d is not a dict or keys are invalid

        Notes
        -----
        axes parameter is currently not accepted

        """
        if axes is not None:
            raise TypeError(
                'axes is currently not accepted as a parameter to append_to_multiple; you can create the tables independently instead'
                )
        if not isinstance(d, dict):
            raise ValueError(
                'append_to_multiple must have a dictionary specified as the way to split the value'
                )
        if selector not in d:
            raise ValueError(
                'append_to_multiple requires a selector that is in passed dict'
                )
        axis = next(iter(set(range(value.ndim)) - set(_AXES_MAP[type(value)
            ])), None)
        remain_key: str | None = None
        remain_values: list[str] = []
        for k, v in d.items():
            if v is None:
                if remain_key is not None:
                    raise ValueError(
                        'append_to_multiple can only have one value in d that is None'
                        )
                remain_key = k
            else:
                remain_values.extend(v)
        if remain_key is not None:
            ordered = value.axes[axis]
            ordd = ordered.difference(Index(remain_values))
            ordd = sorted(ordered.get_indexer(ordd))
            d[remain_key] = ordered.take(ordd)
        if data_columns is None:
            data_columns = d[selector]
        if dropna:
            idxs = (value[cols].dropna(how='all').index for cols in d.values())
            valid_index = next(idxs)
            for index in idxs:
                valid_index = valid_index.intersection(index)
            value = value.loc[valid_index]
        min_itemsize = kwargs.pop('min_itemsize', None)
        for k, v in d.items():
            dc = data_columns if k == selector else None
            val = value.reindex(v, axis=axis)
            filtered = {key: value for key, value in min_itemsize.items() if
                key in v} if min_itemsize is not None else None
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **
                kwargs)

    def create_table_index(self, key, columns=None, optlevel=None, kind=None):
        """
        Create a pytables index on the table.

        Parameters
        ----------
        key : str
        columns : list[str] | bool | None
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int | None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str | None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError
            raises if the node is not a table

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.create_table_index("data", columns=["A"])
        >>> store.close()  # doctest: +SKIP
        """
        _tables()
        s = self.get_storer(key)
        if s is None:
            return
        if not isinstance(s, Table):
            raise TypeError('cannot create table index on a Fixed format store'
                )
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def groups(self):
        """
        Return a list of all the top-level nodes.

        Each node returned is not a pandas storage object.

        Returns
        -------
        list
            List of objects.

        See Also
        --------
        HDFStore.get_node : Returns the node with the key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> print(store.groups())  # doctest: +SKIP
        [/data (Group) ''
          children := ['axis0' (Array), 'axis1' (Array), 'block0_values' (Array),
          'block0_items' (Array)]]
        >>> store.close()  # doctest: +SKIP
        """
        tables_mod = _tables()
        self._check_if_open()
        assert self._handle is not None
        return [g for g in self._handle.walk_groups() if not isinstance(g,
            tables_mod.link.Link) and (getattr(g._v_attrs, 'pandas_type',
            None) or getattr(g, 'table', None) or isinstance(g, tables_mod.
            table.Table) and g._v_name != 'table')]

    def walk(self, where='/'):
        """
        Walk the pytables group hierarchy for pandas objects.

        This generator will yield the group path, subgroups and pandas object
        names for each group.

        Any non-pandas PyTables objects that are not a group will be ignored.

        The `where` group itself is listed first (preorder), then each of its
        child groups (following an alphanumerical order) is also traversed,
        following the same procedure.

        Parameters
        ----------
        where : str, default "/"
            Group where to start walking.

        Yields
        ------
        path : str
            Full path to a group (without trailing '/').
        groups : list
            Names (strings) of the groups contained in `path`.
        leaves : list
            Names (strings) of the pandas objects contained in `path`.

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data1", df1, format="table")  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["A", "B"])
        >>> store.append("data1", df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        >>> for group in store.walk():  # doctest: +SKIP
        ...     print(group)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        tables_mod = _tables()
        self._check_if_open()
        assert self._handle is not None
        assert tables_mod is not None
        for g in self._handle.walk_groups(where):
            if getattr(g._v_attrs, 'pandas_type', None) is not None:
                continue
            groups: list[str] = []
            leaves: list[str] = []
            for child in g._v_children.values():
                pandas_type = getattr(child._v_attrs, 'pandas_type', None)
                if pandas_type is None:
                    if isinstance(child, tables_mod.group.Group):
                        groups.append(child._v_name)
                else:
                    leaves.append(child._v_name)
            yield g._v_pathname.rstrip('/'), groups, leaves

    def get_node(self, key):
        """return the node with the key or None if it does not exist"""
        self._check_if_open()
        if not key.startswith('/'):
            key = '/' + key
        assert self._handle is not None
        tables_mod = _tables()
        assert tables_mod is not None
        try:
            node: tables_mod.Node = self._handle.get_node(self.root, key)
        except tables_mod.exceptions.NoSuchNodeError:
            return None
        assert isinstance(node, tables_mod.Node), type(node)
        return node

    def get_storer(self, key):
        """return the storer object for a key, raise if not in the file"""
        group = self.get_node(key)
        if group is None:
            raise KeyError(f'No object named {key} in the file')
        s = self._create_storer(group)
        s.infer_axes()
        return s

    def copy(self, file, mode='w', propindexes=True, keys=None, complib=
        None, complevel=None, fletcher32=False, overwrite=True):
        """
        Copy the existing store to a new file, updating in place.

        Parameters
        ----------
        file : FilePath
            Destination file path.
        mode : str, default 'w'
            Mode for the new HDFStore.
        propindexes : bool, default True
            Restore indexes in copied file.
        keys : list[str] | None, default None
            List of keys to include in the copy (defaults to all).
        overwrite : bool, default True
            Whether to overwrite (remove and replace) existing nodes in the new store.
        complib : str | None, default None
            Compression library for the new HDFStore.
        complevel : int | None, default None
            Compression level for the new HDFStore.
        fletcher32 : bool, default False
            Checksum for the new HDFStore.

        Returns
        -------
        HDFStore
            open file handle of the new store
        """
        new_store = HDFStore(file, mode=mode, complib=complib, complevel=
            complevel, fletcher32=fletcher32)
        if keys is None:
            keys = list(self.keys())
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        for k in keys:
            s = self.get_storer(k)
            if s is not None:
                if k in new_store:
                    if overwrite:
                        new_store.remove(k)
                data = self.select(k)
                if isinstance(s, Table):
                    index: bool | list[str] = False
                    if propindexes:
                        index = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(k, data, index=index, data_columns=
                        getattr(s, 'data_columns', None), encoding=s.encoding)
                else:
                    new_store.put(k, data, encoding=s.encoding)
        return new_store

    def info(self):
        """
        Print detailed information on the store.

        Returns
        -------
        str
            A String containing the python pandas class name, filepath to the HDF5
            file and all the object keys along with their respective dataframe shapes.

        See Also
        --------
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["C", "D"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data1", df1)  # doctest: +SKIP
        >>> store.put("data2", df2)  # doctest: +SKIP
        >>> print(store.info())  # doctest: +SKIP
        <class 'pandas.io.pytables.HDFStore'>
        File path: store.h5
        /data1            frame        (shape->[2,2])
        /data2            frame        (shape->[2,2])
        """
        path = pprint_thing(self._path)
        output = f'{type(self)}\nFile path: {path}\n'
        if self.is_open:
            lkeys = sorted(self.keys())
            if len(lkeys):
                keys: list[str] = []
                values: list[str] = []
                for k in lkeys:
                    try:
                        s = self.get_storer(k)
                        if s is not None:
                            keys.append(pprint_thing(s.pathname or k))
                            values.append(pprint_thing(s or
                                'invalid_HDFStore node'))
                    except AssertionError:
                        raise
                    except Exception as detail:
                        keys.append(k)
                        dstr = pprint_thing(detail)
                        values.append(f'[invalid_HDFStore node: {dstr}]')
                output += adjoin(12, keys, values)
            else:
                output += 'Empty'
        else:
            output += 'File is CLOSED'
        return output

    def _check_if_open(self):
        if not self.is_open:
            raise ClosedFileError(f'{self._path} file is not open!')

    def _validate_format(self, format):
        """validate / deprecate formats"""
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f'invalid HDFStore format specified [{format}]'
                ) from err
        return format

    def _create_storer(self, group, format=None, value=None, encoding=
        'UTF-8', errors='strict'):
        """return a suitable class to operate"""
        if value is not None and not isinstance(value, (Series, DataFrame)):
            raise TypeError('value must be None, Series, or DataFrame')
        pt = getattr(group._v_attrs, 'pandas_type', None)
        tt = getattr(group._v_attrs, 'table_type', None)
        if pt is None:
            if value is None:
                tables_mod = _tables()
                assert tables_mod is not None
                if getattr(group, 'table', None) or isinstance(group,
                    tables_mod.table.Table):
                    pt = 'frame_table'
                    tt = 'generic_table'
                else:
                    raise TypeError(
                        'cannot create a storer if the object is not existing nor a value are passed'
                        )
            else:
                if isinstance(value, Series):
                    pt = 'series'
                else:
                    pt = 'frame'
                if format == 'table':
                    pt += '_table'
        if 'table' not in pt:
            _STORER_MAP: dict[str, type[GenericFixed | Table]] = {'series':
                SeriesFixed, 'frame': FrameFixed}
            try:
                cls: type[GenericFixed | Table] = _STORER_MAP[pt]
            except KeyError as err:
                raise TypeError(
                    f'cannot properly create the storer for: [_STORER_MAP] [group->{group},value->{type(value)},format->{format}'
                    ) from err
            return cls(self, group, encoding=encoding, errors=errors)
        if tt is None:
            if value is not None:
                if pt == 'series_table':
                    index = getattr(value, 'index', None)
                    if index is not None:
                        if index.nlevels == 1:
                            tt = 'appendable_series'
                        elif index.nlevels > 1:
                            tt = 'appendable_multiseries'
                elif pt == 'frame_table':
                    index = getattr(value, 'index', None)
                    if index is not None:
                        if index.nlevels == 1:
                            tt = 'appendable_frame'
                        elif index.nlevels > 1:
                            tt = 'appendable_multiframe'
        _TABLE_MAP: dict[str, type[Table]] = {'generic_table': GenericTable,
            'appendable_series': AppendableSeriesTable,
            'appendable_multiseries': AppendableMultiSeriesTable,
            'appendable_frame': AppendableFrameTable,
            'appendable_multiframe': AppendableMultiFrameTable, 'worm':
            WORMTable}
        try:
            cls: type[Table] = _TABLE_MAP[tt]
        except KeyError as err:
            raise TypeError(
                f'cannot properly create the storer for: [_TABLE_MAP] [group->{group},value->{type(value)},format->{format}'
                ) from err
        return cls(self, group, encoding=encoding, errors=errors)

    def _write_to_group(self, key, value, format, axes, index, append,
        complib, complevel, fletcher32, min_itemsize, chunksize,
        expectedrows, dropna, nan_rep, data_columns, encoding, errors,
        track_times):
        if getattr(value, 'empty', None) and (format == 'table' or append):
            return
        group = self._identify_group(key, append)
        s = self._create_storer(group, format, value, encoding=encoding,
            errors=errors)
        if append:
            if (not s.is_table or s.is_table and format == 'fixed' and s.
                is_exists):
                raise ValueError('Can only append to Tables')
            if not s.is_exists:
                s.set_object_info()
        else:
            s.set_object_info()
        if not s.is_table and complib is not None:
            raise ValueError('Compression not supported on Fixed format stores'
                )
        s.write(obj=value, axes=axes, append=append, complib=complib,
            complevel=complevel, fletcher32=fletcher32, min_itemsize=
            min_itemsize, chunksize=chunksize, expectedrows=expectedrows,
            dropna=dropna, nan_rep=nan_rep, data_columns=data_columns,
            track_times=track_times)
        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def _read_group(self, group):
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def _identify_group(self, key, append):
        """Identify HDF5 group based on key, delete/create group if needed."""
        group = self.get_node(key)
        assert self._handle is not None
        if group is not None and not append:
            self._handle.remove_node(group, recursive=True)
            group = None
        if group is None:
            group = self._create_nodes_and_group(key)
        return group

    def _create_nodes_and_group(self, key):
        """Create nodes from key and return group name."""
        assert self._handle is not None
        paths = key.split('/')
        path = '/'
        for p in paths:
            if not len(p):
                continue
            new_path = path
            if not path.endswith('/'):
                new_path += '/'
            new_path += p
            group = self.get_node(new_path)
            if group is None:
                group = self._handle.create_group(path, p)
            path = new_path
        return group


class TableIterator:
    """
    Define the iteration interface on a table

    Parameters
    ----------
    store : HDFStore
    s     : the referred storer
    func  : the function to execute the query
    where : the where of the query
    nrows : the rows to iterate on
    start : the passed start value (default is None)
    stop  : the passed stop value (default is None)
    iterator : bool, default False
        Whether to use the default iterator.
    chunksize : int | None, default None
        Number of rows to include in iteration, return an iterator.
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """
    chunksize: int | None
    store: HDFStore
    s: GenericFixed | Table

    def __init__(self, store, s, func, where, nrows, start=None, stop=None,
        iterator=False, chunksize=None, auto_close=False):
        self.store: HDFStore = store
        self.s: GenericFixed | Table = s
        self.func: Callable[[int | None, int | None, Any], Any] = func
        self.where: Any = where
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)
        self.nrows: int = nrows
        self.start: int | None = start
        self.stop: int | None = stop
        self.coordinates: npt.NDArray[np.int64] | None = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize: int = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close: bool = auto_close

    def __iter__(self):
        current = self.start
        if self.coordinates is None:
            raise ValueError('Cannot iterate until get_result is called.')
        while current < self.stop:
            stop = min(current + self.chunksize, self.stop)
            value = self.func(None, None, self.coordinates[current:stop])
            current = stop
            if value is None or not len(value):
                continue
            yield value
        self.close()

    def close(self):
        if self.auto_close:
            self.store.close()

    def get_result(self, coordinates=False):
        if self.chunksize is not None:
            if not isinstance(self.s, Table):
                raise TypeError(
                    'can only use an iterator or chunksize on a table')
            self.coordinates = self.s.read_coordinates(where=self.where)
            return self
        if coordinates:
            if not isinstance(self.s, Table):
                raise TypeError('can only read_coordinates on a table')
            where = self.s.read_coordinates(where=self.where, start=self.
                start, stop=self.stop)
        else:
            where = self.where
        results = self.func(self.start, self.stop, where)
        self.close()
        return results


class IndexCol:
    """
    an index column description class

    Parameters
    ----------
    axis   : axis which I reference
    values : the ndarray like converted values
    kind   : a string description of this type
    typ    : the pytables type
    pos    : the position in the pytables

    """
    is_an_indexable: bool = True
    is_data_indexable: bool = True
    _info_fields: Final[list[str]] = ['freq', 'tz', 'index_name']

    def __init__(self, name, values=None, kind=None, typ=None, cname=None,
        axis=None, pos=None, freq=None, tz=None, index_name=None, ordered=
        None, table=None, meta=None, metadata=None):
        if not isinstance(name, str):
            raise ValueError('`name` must be a str.')
        self.values: Any = values
        self.kind: str | None = kind
        self.typ: Any = typ
        self.name: str = name
        self.cname: str = cname or name
        self.axis: AxisInt | None = axis
        self.pos: int | None = pos
        self.freq: Any = freq
        self.tz: tzinfo | None = tz
        self.index_name: Hashable | None = index_name
        self.ordered: bool | None = ordered
        self.table: Table | None = table
        self.meta: str | None = meta
        self.metadata: Any = metadata
        if pos is not None:
            self.set_pos(pos)
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def itemsize(self):
        return self.typ.itemsize

    @property
    def kind_attr(self):
        return f'{self.name}_kind'

    def set_pos(self, pos):
        """set the position of this column in the Table"""
        self.pos = pos
        if pos is not None and self.typ is not None:
            self.typ._v_pos = pos

    def __repr__(self):
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.axis,
            self.pos, self.kind)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name',
            'cname', 'axis', 'pos', 'kind'], temp)])

    def __eq__(self, other):
        """compare 2 col items"""
        if not isinstance(other, IndexCol):
            return False
        return all(getattr(self, a, None) == getattr(other, a, None) for a in
            ['name', 'cname', 'axis', 'pos'])

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def is_indexed(self):
        """return whether I am an indexed column"""
        if not hasattr(self.table, 'cols'):
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(self, values, nan_rep, encoding, errors):
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname].copy()
        val_kind = self.kind
        values = _maybe_convert(values, val_kind, encoding, errors)
        kwargs: dict[str, Any] = {}
        kwargs['name'] = self.index_name
        if self.freq is not None:
            kwargs['freq'] = self.freq
        factory: type[Index | DatetimeIndex | TimedeltaIndex | PeriodIndex]
        if lib.is_np_dtype(values.dtype, 'M') or isinstance(values.dtype,
            DatetimeTZDtype):
            factory = DatetimeIndex
        elif values.dtype == 'i8' and 'freq' in kwargs:
            factory = lambda x, **kwds: PeriodIndex.from_ordinals(x, freq=
                kwds.get('freq', None))._rename(kwds['name'])
        else:
            factory = Index
        try:
            new_pd_index = factory(values, **kwargs)
        except ValueError:
            if 'freq' in kwargs:
                kwargs['freq'] = None
            new_pd_index = factory(values, **kwargs)
        final_pd_index: Index
        if self.tz is not None and isinstance(new_pd_index, DatetimeIndex):
            final_pd_index = new_pd_index.tz_localize('UTC').tz_convert(self.tz
                )
        else:
            final_pd_index = new_pd_index
        return final_pd_index, final_pd_index

    def take_data(self):
        """return the values"""
        return self.values

    @property
    def attrs(self):
        return self.table._v_attrs

    @property
    def description(self):
        return self.table.description

    @property
    def col(self):
        """return my current col description"""
        return getattr(self.description, self.cname, None)

    @property
    def cvalues(self):
        """return my cython values"""
        return self.values

    def __iter__(self):
        return iter(self.values)

    def maybe_set_size(self, min_itemsize=None):
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
        if self.kind == 'string':
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)
            if min_itemsize is not None and self.typ.itemsize < min_itemsize:
                self.typ = _tables().StringCol(itemsize=min_itemsize, pos=
                    self.pos)

    def validate_names(self):
        pass

    def validate_and_set(self, handler, append):
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def validate_col(self, itemsize=None):
        """validate this column: return the compared against itemsize"""
        if self.kind == 'string':
            c = self.col
            if c is not None:
                if itemsize is None:
                    itemsize = self.itemsize
                if c.itemsize < itemsize:
                    raise ValueError(
                        f"""Trying to store a string with len [{itemsize}] in [{self.cname}] column but
this column has a limit of [{c.itemsize}]!
Consider using min_itemsize to preset the sizes on these columns"""
                        )
                return c.itemsize
        return None

    def validate_attr(self, append):
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(
                    f'incompatible kind in col [{existing_kind} - {self.kind}]'
                    )

    def update_info(self, info):
        """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
        for key in self._info_fields:
            value = getattr(self, key, None)
            idx = info.setdefault(self.name, {})
            existing_value = idx.get(key)
            if key in idx and value is not None and existing_value != value:
                if key in ['freq', 'index_name']:
                    ws = attribute_conflict_doc % (key, existing_value, value)
                    warnings.warn(ws, AttributeConflictWarning, stacklevel=
                        find_stack_level())
                    idx[key] = None
                    setattr(self, key, None)
                else:
                    raise ValueError(
                        f'invalid info for [{self.name}] for [{key}], existing_value [{existing_value}] conflicts with new value [{value}]'
                        )
            elif value is not None or existing_value is not None:
                idx[key] = value

    def set_info(self, info):
        """set my state from the passed info"""
        idx = info.get(self.name)
        if idx is not None:
            self.__dict__.update(idx)

    def set_attr(self):
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)

    def validate_metadata(self, handler):
        """validate that kind=category does not change the categories"""
        if self.meta == 'category':
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if (new_metadata is not None and cur_metadata is not None and 
                not array_equivalent(new_metadata, cur_metadata, strict_nan
                =True, dtype_equal=True)):
                raise ValueError(
                    'cannot append a categorical with different categories to the existing'
                    )

    def write_metadata(self, handler):
        """set the meta data"""
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)


class GenericIndexCol(IndexCol):
    """represent a generic pytables index column"""


class Fixed:
    """
    represent an object in my store
    facilitate read/write of various types of objects
    this is an abstract base class

    Parameters
    ----------
    parent : HDFStore
    group : Node
        The group node where the table resides.
    """
    pandas_kind: str
    format_type: str = 'fixed'
    obj_type: type[DataFrame | Series]
    ndim: int
    parent: HDFStore
    is_table: bool = False

    def __init__(self, parent, group, encoding='UTF-8', errors='strict'):
        assert isinstance(parent, HDFStore), type(parent)
        assert _table_mod is not None
        assert isinstance(group, _table_mod.Node), type(group)
        self.parent = parent
        self.group = group
        self.encoding = _ensure_encoding(encoding)
        self.errors = errors

    @property
    def is_old_version(self):
        return self.version[0] <= 0 and self.version[1] <= 10 and self.version[
            2] < 1

    @property
    def version(self):
        """compute and set our version"""
        version = getattr(self.group._v_attrs, 'pandas_version', None)
        if isinstance(version, str):
            version_tup = tuple(int(x) for x in version.split('.'))
            if len(version_tup) == 2:
                version_tup = version_tup + (0,)
            assert len(version_tup) == 3
            return version_tup
        else:
            return 0, 0, 0

    @property
    def pandas_type(self):
        return getattr(self.group._v_attrs, 'pandas_type', None)

    def __repr__(self):
        """return a pretty representation of myself"""
        self.infer_axes()
        s: Any = self.shape
        if s is not None:
            if isinstance(s, (list, tuple)):
                jshape = ','.join([pprint_thing(x) for x in s])
                s = f'[{jshape}]'
            return f'{self.pandas_type:12.12} (shape->{s})'
        return self.pandas_type

    def set_object_info(self):
        """set my pandas type & version"""
        self.attrs.pandas_type = str(self.pandas_kind)
        self.attrs.pandas_version = str(_version)

    def copy(self):
        new_self = copy.copy(self)
        return new_self

    @property
    def shape(self):
        return self.nrows

    @property
    def pathname(self):
        return self.group._v_pathname

    @property
    def _handle(self):
        return self.parent._handle

    @property
    def _filters(self):
        return self.parent._filters

    @property
    def _complevel(self):
        return self.parent._complevel

    @property
    def _fletcher32(self):
        return self.parent._fletcher32

    @property
    def attrs(self):
        return self.group._v_attrs

    def set_attrs(self):
        """set our object attributes"""

    def get_attrs(self):
        """get our object attributes"""

    @property
    def storable(self):
        return self.group

    @property
    def is_exists(self):
        return False

    @property
    def nrows(self):
        return getattr(self.storable, 'nrows', None)

    def validate(self, other):
        """validate against an existing storable"""
        if other is None:
            return None
        return True

    def validate_version(self, where=None):
        """are we trying to operate on an old version?"""

    def infer_axes(self):
        """
        infer the axes of my storer
        return a boolean indicating if we have a valid storer or not
        """
        s = self.storable
        if s is None:
            return False
        self.get_attrs()
        return True

    def read(self, where=None, columns=None, start=None, stop=None):
        raise NotImplementedError(
            'cannot read on an abstract storer: subclasses should implement')

    def write(self, obj, **kwargs: Any):
        raise NotImplementedError(
            'cannot write on an abstract storer: subclasses should implement')

    def delete(self, where=None, start=None, stop=None):
        """
        support fully deleting the node in its entirety (only) - where
        specification must be None
        """
        if com.all_none(where, start, stop):
            self._handle.remove_node(self.group, recursive=True)
            return None
        raise TypeError('cannot delete on an abstract storer')


class GenericFixed(Fixed):
    """a generified fixed version"""
    _index_type_map: Final[dict[type[Any], str]] = {DatetimeIndex:
        'datetime', PeriodIndex: 'period'}
    _reverse_index_map: Final[dict[str, type[Index]]] = {'datetime':
        DatetimeIndex, 'period': PeriodIndex}
    attributes: list[str] = []

    def _class_to_alias(self, cls):
        return self._index_type_map.get(cls, '')

    def _alias_to_class(self, alias):
        if isinstance(alias, type):
            return alias
        return self._reverse_index_map.get(alias, Index)

    def _get_index_factory(self, attrs):
        index_class: type[Index] = self._alias_to_class(getattr(attrs,
            'index_class', ''))
        factory: Callable[..., Index]
        if index_class == DatetimeIndex:

            def f(values, freq=None, tz=None):
                dta = DatetimeArray._simple_new(values.values, dtype=values
                    .dtype, freq=freq)
                result = DatetimeIndex._simple_new(dta, name=None)
                if tz is not None:
                    result = result.tz_localize('UTC').tz_convert(tz)
                return result
            factory = f
        elif index_class == PeriodIndex:

            def f(values, freq=None, tz=None):
                dtype = PeriodDtype(freq)
                parr = PeriodArray._simple_new(values, dtype=dtype)
                return PeriodIndex._simple_new(parr, name=None)
            factory = f
        else:
            factory = index_class
        kwargs: dict[str, Any] = {}
        if 'freq' in attrs:
            kwargs['freq'] = attrs['freq']
            if index_class is Index:
                factory = TimedeltaIndex
        if 'tz' in attrs:
            kwargs['tz'] = attrs['tz']
            assert index_class is DatetimeIndex
        return factory, kwargs

    def validate_read(self, columns, where):
        """
        raise if any keywords are passed which are not-None
        """
        if columns is not None:
            raise TypeError(
                'cannot pass a column specification when reading a Fixed format store. this store must be selected in its entirety'
                )
        if where is not None:
            raise TypeError(
                'cannot pass a where specification when reading from a Fixed format store. this store must be selected in its entirety'
                )

    @classmethod
    def get_object(cls, obj, transposed):
        """return the data for this obj"""
        return obj

    def validate_data_columns(self, data_columns, min_itemsize, non_index_axes
        ):
        """
        take the input data_columns and min_itemize and create a data
        columns spec
        """
        if min_itemsize is None:
            return []
        if not isinstance(min_itemsize, dict):
            return []
        q = self.queryables()
        for k in min_itemsize:
            if k == 'values':
                continue
            if k not in q:
                raise ValueError(
                    f'min_itemsize has the key [{k}] which is not an axis or data_column'
                    )
        return [c for c in data_columns if c in q] if data_columns else []

    def validate_multiindex(self, obj):
        """
        validate that we can store the multi-index; reset and return the
        new object
        """
        levels = com.fill_missing_names(obj.index.names)
        try:
            reset_obj = obj.reset_index()
        except ValueError as err:
            raise ValueError(
                'duplicate names/columns in the multi-index when storing as a table'
                ) from err
        assert isinstance(reset_obj, DataFrame)
        return reset_obj, levels

    def _get_blocks_and_items(self, frame, table_exists, new_non_index_axes,
        values_axes, data_columns):
        """Helper to clarify non-state-altering parts of _create_axes"""

        def get_blk_items(mgr):
            return [mgr.items.take(blk.mgr_locs) for blk in mgr.blocks]
        mgr = frame._mgr
        blocks: list[Block] = list(mgr.blocks)
        blk_items: list[Index] = get_blk_items(mgr)
        if len(data_columns):
            axis, axis_labels = new_non_index_axes[0]
            new_labels = Index(axis_labels).difference(Index(data_columns))
            mgr = frame.reindex(new_labels, axis=axis)._mgr
            blocks = list(mgr.blocks)
            blk_items = get_blk_items(mgr)
            for c in data_columns:
                mgr = frame.reindex([c], axis=axis)._mgr
                blocks.extend(mgr.blocks)
                blk_items.extend(get_blk_items(mgr))
        if table_exists:
            by_items: dict[tuple[Any, ...], tuple[Block, Index]] = {tuple(
                b_items.tolist()): (b, b_items) for b, b_items in zip(
                blocks, blk_items)}
            new_blocks: list[Block] = []
            new_blk_items: list[Index] = []
            for ea in values_axes:
                items = tuple(ea.values)
                try:
                    b, b_items = by_items.pop(items)
                    new_blocks.append(b)
                    new_blk_items.append(b_items)
                except (IndexError, KeyError) as err:
                    jitems = ','.join([pprint_thing(item) for item in items])
                    raise ValueError(
                        f'cannot match existing table structure for [{jitems}] with existing table [{{list(by_items.keys())}}]'
                        ) from err
            blocks = new_blocks
            blk_items = new_blk_items
        return blocks, blk_items

    def _create_axes(self, axes, obj, validate=True, nan_rep=None,
        data_columns=None, min_itemsize=None):
        """
        Create and return the axes.

        Parameters
        ----------
        axes: list | None
            The names or numbers of the axes to create.
        obj : DataFrame
            The object to create axes on.
        validate: bool, default True
            Whether to validate the obj against an existing object already written.
        nan_rep :
            A value to use for string column nan_rep.
        data_columns : list[str] | Literal[True] | None, default None
            Specify the columns that we want to create to allow indexing on.
            * True : Use all available columns.
            * None : Use no columns.
            * list[str] : Use the specified columns.
        min_itemsize: int | dict[str, int] | None, default None
            The min itemsize for a column in bytes.
        """
        if not isinstance(obj, DataFrame):
            group = self.group._v_name
            raise TypeError(
                f'cannot properly create the storer for: [group->{group},value->{type(obj)}]'
                )
        if axes is None:
            axes = [0]
        axes = [obj._get_axis_number(a) for a in axes]
        if self.infer_axes():
            table_exists = True
            axes = [a.axis for a in self.index_axes]
            data_columns = list(self.data_columns)
            nan_rep = self.nan_rep
        else:
            table_exists = False
        new_info: dict[str, dict[str, Any]] = self.info
        assert self.ndim == 2
        if len(axes) != self.ndim - 1:
            raise ValueError(
                'currently only support ndim-1 indexers in an AppendableTable')
        new_non_index_axes: list[tuple[AxisInt, Any]] = []
        if nan_rep is None:
            nan_rep = 'nan'
        idx = next(x for x in [0, 1] if x not in axes)
        a = obj.axes[idx]
        append_axis = list(a)
        if table_exists:
            indexer = len(new_non_index_axes)
            exist_axis = self.non_index_axes[indexer][1]
            if not array_equivalent(np.array(append_axis), np.array(
                exist_axis), strict_nan=True, dtype_equal=True):
                if array_equivalent(np.array(sorted(append_axis)), np.array
                    (sorted(exist_axis)), strict_nan=True, dtype_equal=True):
                    append_axis = exist_axis
        info = new_info.setdefault(idx, {})
        info['names'] = list(a.names)
        info['type'] = type(a).__name__
        new_non_index_axes.append((idx, append_axis))
        idx = axes[0]
        a = obj.axes[idx]
        axis_name = obj._get_axis_name(idx)
        new_index, new_index_kwargs = self._get_index_factory(getattr(self.
            attrs, 'index_cols', {}))
        new_index = _convert_index(axis_name, a, self.encoding, self.errors)
        new_index.axis = idx
        new_index.set_pos(0)
        new_index.update_info(new_info)
        new_index.maybe_set_size(min_itemsize)
        new_index_axes = [new_index]
        j = len(new_index_axes)
        assert len(new_non_index_axes) == 1
        for a in new_non_index_axes:
            obj = _reindex_axis(obj, a[0], a[1])
        transposed = new_index.axis == 1
        data_columns = self.validate_data_columns(data_columns,
            min_itemsize, new_non_index_axes)
        frame = self.get_object(obj, transposed)._consolidate()
        blocks, blk_items = self._get_blocks_and_items(frame, table_exists,
            new_non_index_axes, self.values_axes, data_columns)
        vaxes: list[IndexCol] = []
        for i, (blk, b_items) in enumerate(zip(blocks, blk_items)):
            klass: type[IndexCol | DataIndexableCol]
            name: str | None = None
            if data_columns and len(b_items) == 1 and b_items[0
                ] in data_columns:
                klass = DataIndexableCol
                name = b_items[0]
                if not (name is None or isinstance(name, str)):
                    raise ValueError(
                        'cannot have non-object label DataIndexableCol')
            else:
                klass = DataCol
            existing_col: IndexCol | None = None
            if table_exists and validate:
                try:
                    existing_col = self.values_axes[i]
                except (IndexError, KeyError) as err:
                    raise ValueError(
                        f'Incompatible appended table [{blocks}]with existing table [{self.values_axes}]'
                        ) from err
            new_name: str = name or f'values_block_{i}'
            data_converted, dtype_name = _get_data_and_dtype_name(blk.values)
            kind = _dtype_to_kind(dtype_name)
            tz: tzinfo | None = None
            if getattr(data_converted, 'tz', None) is not None:
                tz = _get_tz(data_converted.tz)
            meta: str | None = None
            metadata: Any = None
            if isinstance(data_converted.dtype, CategoricalDtype):
                ordered: bool | None = data_converted.ordered
                meta = 'category'
                metadata = np.asarray(data_converted.categories).ravel()
            if isinstance(existing_col, DataCol):
                cls_type = (DataIndexableCol if existing_col.
                    is_data_indexable else DataCol)
            else:
                cls_type = klass
            col = cls_type(name=new_name, cname=new_name, values=list(
                b_items), typ=klass._get_atom(data_converted), kind=kind,
                tz=tz, ordered=ordered, meta=meta, metadata=metadata, dtype
                =dtype_name, data=data_converted)
            col.update_info(new_info)
            vaxes.append(col)
            j += 1
        dcs: list[str] = [col.name for col in vaxes if col.is_data_indexable]
        new_table = Table(parent=self.parent, group=self.group, encoding=
            self.encoding, errors=self.errors, index_axes=new_index_axes,
            non_index_axes=new_non_index_axes, values_axes=vaxes,
            data_columns=dcs, info=new_info, nan_rep=nan_rep)
        if hasattr(self, 'levels'):
            new_table.levels = self.levels
        new_table.validate_min_itemsize(min_itemsize)
        if validate and table_exists:
            new_table.validate(self)
        return new_table


class SeriesFixed(GenericFixed):
    pandas_kind: Final[str] = 'series'
    attributes: Final[list[str]] = ['name']
    name: Hashable | None

    @property
    def shape(self):
        try:
            return len(self.group.values),
        except (TypeError, AttributeError):
            return None

    def read(self, where=None, columns=None, start=None, stop=None):
        self.validate_read(columns, where)
        index = self.read_index('index', start=start, stop=stop)
        values = self.read_array('values', start=start, stop=stop)
        result = Series(values, index=index, name=self.name, copy=False)
        if using_string_dtype() and isinstance(values, np.ndarray
            ) and is_string_array(values, skipna=True):
            result = result.astype(StringDtype(na_value=np.nan))
        return result

    def write(self, obj, **kwargs: Any):
        super().write(obj, **kwargs)
        self.write_index('index', obj.index)
        self.write_array('values', obj)
        self.attrs.name = obj.name


class BlockManagerFixed(GenericFixed):
    attributes: Final[list[str]] = ['ndim', 'nblocks']
    ndim: int
    nblocks: int

    def __init__(self, parent, group, encoding=None, errors='strict', ndim=
        0, nblocks=0):
        super().__init__(parent, group, encoding=encoding, errors=errors)
        self.ndim = ndim
        self.nblocks = nblocks

    @property
    def shape(self):
        try:
            ndim = self.ndim
            items = 0
            for i in range(self.nblocks):
                node = getattr(self.group, f'block{i}_items')
                shape = getattr(node, 'shape', None)
                if shape is not None:
                    items += shape[0]
            node = self.group.block0_values
            shape = getattr(node, 'shape', None)
            if shape is not None:
                shape = list(shape[0:ndim - 1])
            else:
                shape = []
            shape.append(items)
            return shape
        except AttributeError:
            return None

    def read(self, where=None, columns=None, start=None, stop=None):
        self.validate_version(where)
        if not self.infer_axes():
            return DataFrame()
        selection = Selection(self, where=where, start=start, stop=stop)
        values = selection.select()
        results: list[tuple[Index, Any]] = []
        for a in self.axes:
            a.set_info(self.info)
            res = a.convert(values, nan_rep=self.nan_rep, encoding=self.
                encoding, errors=self.errors)
            results.append(res)
        index = results[0][0]
        frames: list[DataFrame] = []
        for i, a in enumerate(self.axes):
            if a not in self.values_axes:
                continue
            index_vals, cvalues = results[i]
            if 'type' not in self.info.get(self.non_index_axes[0][0], {}):
                cols = Index(index_vals)
            else:
                cols = MultiIndex.from_tuples(index_vals)
            names = self.info.get('names')
            if names is not None:
                cols.set_names(names, inplace=True)
            if self.is_transposed:
                values = cvalues
                index_ = cols
                cols_ = Index(index, name=getattr(index, 'name', None))
            else:
                values = cvalues.T
                index_ = Index(index, name=getattr(index, 'name', None))
                cols_ = cols
            if values.ndim == 1 and isinstance(values, np.ndarray):
                values = values.reshape((1, values.shape[0]))
            if isinstance(values, (np.ndarray, DatetimeArray)):
                df = DataFrame(values.T, columns=cols_, index=index_, copy=
                    False)
            elif isinstance(values, Index):
                df = DataFrame(values, columns=cols_, index=index_)
            else:
                df = DataFrame._from_arrays([values], columns=cols_, index=
                    index_)
            if not (using_string_dtype() and values.dtype.kind == 'O'):
                assert (df.dtypes == values.dtype).all(), (df.dtypes,
                    values.dtype)
            if using_string_dtype() and isinstance(values, np.ndarray
                ) and is_string_array(values, skipna=True):
                df = df.astype(StringDtype(na_value=np.nan))
            frames.append(df)
        if len(frames) == 1:
            df = frames[0]
        else:
            df = concat(frames, axis=1)
            df = df.reindex(columns=self.axes[0].values)
        selection = Selection(self, where=where, start=start, stop=stop)
        df = self.process_axes(df, selection=selection, columns=columns)
        return df

    def write(self, obj, **kwargs: Any):
        self.set_attrs()

    def write_data(self, chunksize, dropna=False):
        """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
        names = self.dtype.names
        nrows = self.nrows_expected
        masks: list[npt.NDArray[np.bool_]] = []
        if dropna:
            for a in self.values_axes:
                mask = isna(a.data).all(axis=0)
                if isinstance(mask, np.ndarray):
                    masks.append(mask.astype('u1', copy=False))
        if len(masks):
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
            mask = mask.ravel()
        else:
            mask = None
        indexes = [a.cvalues for a in self.index_axes]
        nindexes = len(indexes)
        assert nindexes == 1, nindexes
        values = [a.take_data() for a in self.values_axes]
        values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in
            values]
        bvalues: list[np.ndarray] = []
        for i, v in enumerate(values):
            new_shape = (nrows,) + self.dtype[names[nindexes + i]].shape
            bvalues.append(v.reshape(new_shape))
        if chunksize is None:
            chunksize = 100000
        rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
        chunks = nrows // chunksize + 1
        for i in range(chunks):
            start_i = i * chunksize
            end_i = min((i + 1) * chunksize, nrows)
            if start_i >= end_i:
                break
            self.write_data_chunk(rows, indexes=[a[start_i:end_i] for a in
                indexes], mask=mask[start_i:end_i] if mask is not None else
                None, values=[v[start_i:end_i] for v in bvalues])

    def write_data_chunk(self, rows, indexes, mask, values):
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : a list of np.ndarray[int64]
        mask : npt.NDArray[np.bool_] | None
        values : a list of np.ndarray
        """
        for v in values:
            if not np.prod(v.shape):
                return
        nrows = indexes[0].shape[0]
        if nrows != len(rows):
            rows = np.empty(nrows, dtype=self.dtype)
        names = self.dtype.names
        nindexes = len(indexes)
        for i, idx in enumerate(indexes):
            rows[names[i]] = idx
        for i, v in enumerate(values):
            rows[names[i + nindexes]] = v
        if mask is not None:
            m = ~mask.ravel().astype(bool, copy=False)
            if not m.all():
                rows = rows[m]
        if len(rows):
            self.table.append(rows)
            self.table.flush()


class WORMTable(Table):
    """
    a write-once read-many table: this format DOES NOT ALLOW appending to a
    table. writing is a one-time operation the data are stored in a format
    that allows for searching the data on disk
    """
    table_type: Final[str] = 'worm'

    def read(self, where=None, columns=None, start=None, stop=None):
        """
        read the indices and the indexing array, calculate offset rows and return
        """
        raise NotImplementedError('WORMTable needs to implement read')

    def write(self, obj, **kwargs: Any):
        """
        write in a format that we can search later on (but cannot append
        to): write out the indices and the values using _write_array
        (e.g. a CArray) create an indexing table so that we can search
        """
        raise NotImplementedError('WORMTable needs to implement write')


class AppendableTable(Table):
    """support the new appendable table formats"""
    table_type: Final[str] = 'appendable'

    def write(self, obj, axes=None, append=False, complib=None, complevel=
        None, fletcher32=None, min_itemsize=None, chunksize=None,
        expectedrows=None, dropna=False, nan_rep=None, data_columns=None,
        track_times=True, **kwargs: Any):
        if not append and self.is_exists:
            self._handle.remove_node(self.group, 'table')
        table = self._create_axes(axes=axes, obj=obj, validate=append,
            min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=
            data_columns)
        for a in table.axes:
            a.validate_names()
        if not table.is_exists:
            options = table.create_description(complib=complib, complevel=
                complevel, fletcher32=fletcher32, expectedrows=expectedrows)
            table.set_attrs()
            options['track_times'] = track_times
            table._handle.create_table(table.group, **options)
        table.attrs.info = table.info
        for a in table.axes:
            a.validate_and_set(table, append)
        table.write_data(chunksize, dropna=dropna)

    def write_data(self, chunksize, dropna=False):
        """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
        names = self.dtype.names
        nrows = self.nrows_expected
        masks: list[npt.NDArray[np.bool_]] = []
        if dropna:
            for a in self.values_axes:
                mask = isna(a.data).all(axis=0)
                if isinstance(mask, np.ndarray):
                    masks.append(mask.astype('u1', copy=False))
        if len(masks):
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
            mask = mask.ravel()
        else:
            mask = None
        indexes = [a.cvalues for a in self.index_axes]
        nindexes = len(indexes)
        assert nindexes == 1, nindexes
        values = [a.take_data() for a in self.values_axes]
        values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in
            values]
        bvalues: list[np.ndarray] = []
        for i, v in enumerate(values):
            new_shape = (nrows,) + self.dtype[names[nindexes + i]].shape
            bvalues.append(v.reshape(new_shape))
        if chunksize is None:
            chunksize = 100000
        rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
        chunks = nrows // chunksize + 1
        for i in range(chunks):
            start_i = i * chunksize
            end_i = min((i + 1) * chunksize, nrows)
            if start_i >= end_i:
                break
            self.write_data_chunk(rows, indexes=[a[start_i:end_i] for a in
                indexes], mask=mask[start_i:end_i] if mask is not None else
                None, values=[v[start_i:end_i] for v in bvalues])

    def write_data_chunk(self, rows, indexes, mask, values):
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : a list of np.ndarray[int64]
        mask : npt.NDArray[np.bool_] | None
        values : a list of np.ndarray
        """
        for v in values:
            if not np.prod(v.shape):
                return
        nrows = indexes[0].shape[0]
        if nrows != len(rows):
            rows = np.empty(nrows, dtype=self.dtype)
        names = self.dtype.names
        nindexes = len(indexes)
        for i, idx in enumerate(indexes):
            rows[names[i]] = idx
        for i, v in enumerate(values):
            rows[names[i + nindexes]] = v
        if mask is not None:
            m = ~mask.ravel().astype(bool, copy=False)
            if not m.all():
                rows = rows[m]
        if len(rows):
            self.table.append(rows)
            self.table.flush()
