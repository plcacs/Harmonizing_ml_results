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
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    cast,
    overload,
    Callable,
    Hashable,
    Iterator,
    Sequence,
    Optional,
    List,
    Tuple,
    Union,
)
import warnings
import numpy as np
from pandas._config import config, get_option, using_string_dtype
from pandas._libs import lib, writers as libwriters
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
    AttributeConflictWarning,
    ClosedFileError,
    IncompatibilityWarning,
    PerformanceWarning,
    PossibleDataLossError,
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_complex_dtype,
    is_list_like,
    is_string_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    StringDtype,
    TimedeltaIndex,
    concat,
    isna,
)
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


def func_lygurqaw(encoding: Optional[str]) -> str:
    if encoding is None:
        encoding = _default_encoding
    return encoding


def func_1imi6w8q(name: Any) -> Any:
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype. Non-string dtypes are passed through unchanged.

    https://github.com/pandas-dev/pandas/issues/13492
    """
    if isinstance(name, str):
        name = str(name)
    return name


Term = PyTablesExpr


def func_64dq3psp(where: Optional[Union[Term, List[Optional[Term]]]], scope_level: int) -> Optional[Union[Term, List[Term]]]:
    """
    Ensure that the where is a Term or a list of Term.

    This makes sure that we are capturing the scope of variables that are
    passed create the terms here with a frame_level=2 (we are 2 levels down)
    """
    level = scope_level + 1
    if isinstance(where, (list, tuple)):
        where = [
            Term(term, scope_level=level + 1) if maybe_expression(term) else term
            for term in where
            if term is not None
        ]
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
_FORMAT_MAP: Final[Dict[str, str]] = {'f': 'fixed', 'fixed': 'fixed', 't': 'table', 'table': 'table'}
_AXES_MAP: Final[Dict[Any, List[int]]] = {DataFrame: [0]}
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
    config.register_option(
        'dropna_table',
        False,
        dropna_doc,
        validator=config.is_bool,
    )
    config.register_option(
        'default_format',
        None,
        format_doc,
        validator=config.is_one_of_factory(['fixed', 'table', None]),
    )
_table_mod: Optional[Any] = None
_table_file_open_policy_is_strict: bool = False


def func_q9kzi9o4() -> Any:
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables
        _table_mod = tables
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (tables.file._FILE_OPEN_POLICY == 'strict')
    return _table_mod


def func_w2x2fye9(
    path_or_buf: Union[str, File, DataFrame], 
    key: str, 
    value: Union[DataFrame, Series],
    mode: Literal['a'] = 'a', 
    complevel: Optional[int] = None,
    complib: Optional[str] = None, 
    append: bool = False, 
    format: Optional[str] = None, 
    index: bool = True, 
    min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
    nan_rep: Optional[str] = None, 
    dropna: Optional[bool] = None, 
    data_columns: Optional[Union[List[str], bool]] = None,
    errors: Literal['strict'] = 'strict', 
    encoding: str = 'UTF-8'
) -> None:
    """store this object, close it if we opened it"""
    if append:
        f: Callable[[Any], None] = lambda store: store.append(
            key,
            value,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            dropna=dropna,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
        )
    else:
        f = lambda store: store.put(
            key,
            value,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
            dropna=dropna,
        )
    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        path_or_buf = stringify_path(path_or_buf)
        with HDFStore(
            path_or_buf, mode=mode, complevel=complevel, complib=complib
        ) as store:
            f(store)


def func_f0qzhk05(
    path_or_buf: Union[str, File, HDFStore],
    key: Optional[str] = None, 
    mode: Literal['r', 'r+', 'a'] = 'r', 
    errors: Literal['strict'] = 'strict',
    where: Optional[Union[List[Term], Term]] = None, 
    start: Optional[int] = None, 
    stop: Optional[int] = None,
    columns: Optional[List[str]] = None, 
    iterator: bool = False, 
    chunksize: Optional[int] = None, 
    **kwargs: Any
) -> Any:
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
        where = func_64dq3psp(where, scope_level=1)
    if isinstance(path_or_buf, HDFStore):
        if not path_or_buf.is_open:
            raise OSError('The HDFStore must be open for reading.')
        store = path_or_buf
        auto_close = False
    else:
        path_or_buf = stringify_path(path_or_buf)
        if not isinstance(path_or_buf, str):
            raise NotImplementedError(
                'Support for generic buffers has not been implemented.'
            )
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
        return store.select(
            key,
            where=where,
            start=start,
            stop=stop,
            columns=columns,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
    except (ValueError, TypeError, LookupError):
        if not isinstance(path_or_buf, HDFStore):
            with suppress(AttributeError):
                store.close()
        raise


def func_7icgzlda(group: Node, parent_group: Node) -> bool:
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
    complevel : Optional[int], 0-9, default None
        Specifies a compression level for data.
        A value of 0 or None disables compression.
    complib : Optional[str], {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
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

    def __init__(
        self,
        path: str,
        mode: Literal['a', 'w', 'r', 'r+'] = 'a',
        complevel: Optional[int] = None,
        complib: Optional[str] = None,
        fletcher32: bool = False,
        **kwargs: Any,
    ) -> None:
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
        self._handle: Optional[Any] = None
        self._complevel: int = complevel if complevel else 0
        self._complib: Optional[str] = complib
        self._fletcher32: bool = fletcher32
        self._filters: Optional[Any] = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self) -> str:
        return self._path

    @property
    def func_xoujghg7(self) -> Node:
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None
        return self._handle.root

    @property
    def func_c2z2ymw9(self) -> str:
        return self._path

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Union[DataFrame, Series]) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        return self.remove(key)

    def __getattr__(self, name: str) -> Any:
        """allow attribute access to get stores"""
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __contains__(self, key: str) -> bool:
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

    def __len__(self) -> int:
        return len(self.groups())

    def __repr__(self) -> str:
        pstr = pprint_thing(self._path)
        return f'{type(self)}\nFile path: {pstr}\n'

    def __enter__(self) -> HDFStore:
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def func_ldc0l8ou(self, include: Literal['pandas', 'native'] = 'pandas') -> List[str]:
        """
        Return a list of keys corresponding to objects stored in HDFStore.

        Parameters
        ---------
        include : str, default 'pandas'
                When kind equals 'pandas' return pandas objects.
                When kind equals 'native' return native HDF5 Table objects.

        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading '/').
        """
        if include == 'pandas':
            return [n._v_pathname for n in self.groups()]
        elif include == 'native':
            assert self._handle is not None
            return [
                n._v_pathname
                for n in self._handle.walk_nodes('/', classname='Table')
            ]
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def func_vh9yy4lw(self) -> Iterator[Tuple[str, Node]]:
        """
        iterate on key->group
        """
        for g in self.groups():
            yield g._v_pathname, g

    def open(
        self, mode: Literal['a', 'w', 'r', 'r+'] = 'a', **kwargs: Any
    ) -> None:
        """
        Open the file in the specified mode

        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
        tables = func_q9kzi9o4()
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
            self._filters = func_q9kzi9o4().Filters(
                self._complevel, self._complib, fletcher32=self._fletcher32
            )
        if _table_file_open_policy_is_strict and self.is_open:
            msg = (
                'Cannot open HDF5 file, which is already opened, even in read-only mode.'
            )
            raise ValueError(msg)
        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def func_z55zjm5k(self) -> None:
        """
        Close the PyTables file handle
        """
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def func_3eujce5j(self) -> bool:
        """
        return a boolean indicating whether the file is open
        """
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def func_81x7zc5t(self, fsync: bool = False) -> None:
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

    def func_qzwi0szd(self, key: str) -> Any:
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

    def func_o9jwliii(
        self,
        key: str,
        where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        columns: Optional[List[str]] = None,
        iterator: bool = False,
        chunksize: Optional[int] = None,
        auto_close: bool = False,
    ) -> Any:
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
        start : int or None, default None
            Row number to start selection.
        stop : int or None, default None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number of rows to include in iteration, return an iterator.
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
        where = func_64dq3psp(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        def func_86pvfgyn(_start: Optional[int], _stop: Optional[int], _where: Optional[Any]) -> Any:
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        it = TableIterator(
            self,
            s,
            func_86pvfgyn,
            where=where,
            nrows=s.nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
        return it.get_result()

    def func_q65tcrqy(
        self, key: str, where: Optional[Union[List[Term], Term]] = None, start: Optional[int] = None, stop: Optional[int] = None
    ) -> Index:
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
        """
        where = func_64dq3psp(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_coordinates with a table')
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def func_8hhwipol(
        self, key: str, column: str, start: Optional[int] = None, stop: Optional[int] = None
    ) -> Series:
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

        """
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_column with a table')
        return tbl.read_column(column=column, start=start, stop=stop)

    def func_la0i3d8h(
        self,
        keys: Union[str, List[str], Tuple[str, ...]],
        where: Optional[Union[List[Term], Term]] = None,
        selector: Optional[str] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        iterator: bool = False,
        chunksize: Optional[int] = None,
        auto_close: bool = False,
    ) -> Any:
        """
        Retrieve pandas objects from multiple tables.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        keys : a list of the tables
        selector : the table to apply the where criteria (defaults to keys[0]
            if not supplied)
        columns : the columns I want back
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        iterator : bool, return an iterator, default False
        chunksize : nrows to include in iteration, return an iterator
        auto_close : bool, default False
            Should automatically close the store when finished.

        Raises
        ------
        raises KeyError if keys or selector is not found or keys is empty
        raises TypeError if keys is not a list or tuple
        raises ValueError if the tables are not ALL THE SAME DIMENSIONS
        """
        where = func_64dq3psp(where, scope_level=1)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(
                key=keys,
                where=where,
                columns=columns,
                start=start,
                stop=stop,
                iterator=iterator,
                chunksize=chunksize,
                auto_close=auto_close,
            )
        if not isinstance(keys, (list, tuple)):
            raise TypeError('keys must be a list/tuple')
        if not len(keys):
            raise ValueError('keys must have a non-zero length')
        if selector is None:
            selector = keys[0]
        tbls = [self.get_storer(k) for k in keys]
        s = self.get_storer(selector)
        nrows: Optional[int] = None
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
                raise ValueError('all tables must have exactly the same nrows!')
        _tbls = [x for x in tbls if isinstance(x, Table)]
        axis: Any = {t.non_index_axes[0][0] for t in _tbls}.pop()

        def func_86pvfgyn(_start: Optional[int], _stop: Optional[int], _where: Optional[Any]) -> DataFrame:
            objs = [t.read(where=_where, columns=columns, start=_start, stop=_stop) for t in tbls]
            return concat(objs, axis=axis, verify_integrity=False)._consolidate()

        it = TableIterator(
            self,
            s,
            func_86pvfgyn,
            where=where,
            nrows=nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
        return it.get_result(coordinates=True)

    def func_wwtmnm8a(
        self,
        key: str,
        value: Union[DataFrame, Series],
        format: Optional[str] = None,
        axes: Optional[List[int]] = None,
        index: bool = True,
        append: bool = False,
        complib: Optional[str] = None,
        complevel: Optional[int] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep: Optional[str] = None,
        data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None,
        errors: Literal['strict'] = 'strict',
        track_times: bool = True,
        dropna: bool = False,
    ) -> None:
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
        complib : Optional[str], default None
            This parameter is currently not accepted.
        complevel : Optional[int], 0-9, default None
            Specifies a compression level for data.
            A value of 0 or None disables compression.
        min_itemsize : Optional[Union[int, Dict[str, int]]]
            Dict of columns that specify minimum str sizes.
        nan_rep : Optional[str]
            Str to use as str nan representation.
        data_columns : Optional[Union[List[str], bool]], default None
            List of columns to create as data columns, or True to use all columns.
            See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : Optional[str], default None
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
        self._write_to_group(
            key,
            value,
            format=format,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
            track_times=track_times,
            dropna=dropna,
        )

    def func_30nemlpc(
        self,
        key: str,
        where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> Optional[int]:
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
        number of rows removed (or None if not a Table)

        Raises
        ------
        raises KeyError if key is not a valid store

        """
        where = func_64dq3psp(where, scope_level=1)
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
                'can only remove with where on objects written as tables'
            )
        return s.delete(where=where, start=start, stop=stop)

    def func_zee8uqzp(
        self,
        key: str,
        value: Union[DataFrame, Series],
        format: Optional[str] = None,
        index: bool = True,
        append: bool = False,
        complib: Optional[str] = None,
        complevel: Optional[int] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep: Optional[str] = None,
        chunksize: Optional[int] = None,
        expectedrows: Optional[int] = None,
        dropna: bool = False,
        data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None,
        errors: Literal['strict'] = 'strict',
    ) -> None:
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
        index : bool, default True
            Write DataFrame index as a column.
        append : bool, default True
            Append the input data to the existing.
        complib : Optional[str], default None
            This parameter is currently not accepted.
        complevel : Optional[int], 0-9, default None
            Specifies a compression level for data.
            A value of 0 or None disables compression.
        columns : Optional[List[str]], default None
            This parameter is currently not accepted, try data_columns.
        min_itemsize : Optional[Union[int, Dict[str, int]]]
            Dict of columns that specify minimum str sizes.
        nan_rep : Optional[str]
            Str to use as str nan representation.
        chunksize : Optional[int]
            Size to chunk the writing.
        expectedrows : Optional[int]
            Expected TOTAL row size of this table.
        dropna : bool, default False, optional
            Do not write an ALL nan row to the store settable
            by the option 'io.hdf.dropna_table'.
        data_columns : Optional[Union[List[str], bool]], default None
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : Optional[str], default None
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
        self._write_to_group(
            key,
            value,
            format=format,
            axes=axes,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
        )

    def func_5ol8yjd1(
        self,
        d: Dict[str, Optional[List[str]]],
        value: DataFrame,
        selector: str,
        data_columns: Optional[Union[List[str], bool]] = None,
        axes: Optional[List[int]] = None,
        dropna: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Append to multiple tables

        Parameters
        ----------
        d : a dict of table_name to table_columns, None is acceptable as the
            values of one node (this will get all the remaining columns)
        value : a pandas object
        selector : a string that designates the indexable table; all of its
            columns will be designed as data_columns, unless data_columns is
            passed, in which case these are used
        data_columns : Optional[List[str]], or True, default None
            Specify the columns that we want to create to allow indexing on.
        dropna : bool, default False
            If evaluates to True, drop rows from all tables if any single
            row in each table has all NaN. Default False.

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
        axis = next(iter(set(range(value.ndim)) - set(_AXES_MAP[type(value)])))
        remain_key: Optional[str] = None
        remain_values: List[str] = []
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
        min_itemsize: Optional[Dict[str, int]] = kwargs.pop('min_itemsize', None)
        for k, v in d.items():
            dc = data_columns if k == selector else None
            val = value.reindex(v, axis=axis)
            filtered = {key: value for key, value in min_itemsize.items() if
                key in v} if min_itemsize is not None else None
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)

    def func_ckoyybrz(
        self,
        key: str,
        columns: Optional[Union[List[str], bool]] = None,
        optlevel: Optional[int] = None,
        kind: Optional[str] = None,
    ) -> None:
        """
        Create a pytables index on the table.

        Parameters
        ----------
        key : str
        columns : Optional[Union[bool, List[str]]]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : Optional[int], default None
            Optimization level, if None, pytables defaults to 6.
        kind : Optional[str], default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError: raises if the node is not a table
        """
        func_q9kzi9o4()
        s = self.get_storer(key)
        if s is None:
            return
        if not isinstance(s, Table):
            raise TypeError('cannot create table index on a Fixed format store')
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def func_s8r4yw3o(self) -> List[Any]:
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
        >>> store.close()  # doctest: +SKIP
        [/data (Group) ''
          children := ['axis0' (Array), 'axis1' (Array), 'block0_values' (Array),
          'block0_items' (Array)]]
        """
        func_q9kzi9o4()
        self._check_if_open()
        assert self._handle is not None
        return [
            g
            for g in self._handle.walk_groups()
            if not isinstance(g, _table_mod.link.Link)
            and (
                getattr(g._v_attrs, 'pandas_type', None)
                or getattr(g, 'table', None)
                or (isinstance(g, _table_mod.table.Table) and g._v_name != 'table')
            )
        ]

    def func_hx6f1lva(
        self, where: str = '/'
    ) -> Iterator[Tuple[str, List[str], List[str]]]:
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
        groups : List[str]
            Names (strings) of the groups contained in `path`.
        leaves : List[str]
            Names (strings) of the pandas objects contained in `path`.

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df1, format="table")  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["A", "B"])
        >>> store.append("data", df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        >>> for group in store.walk():  # doctest: +SKIP
        ...     print(group)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        func_q9kzi9o4()
        self._check_if_open()
        assert self._handle is not None
        assert _table_mod is not None
        for g in self._handle.walk_groups(where):
            if getattr(g._v_attrs, 'pandas_type', None) is not None:
                continue
            groups: List[str] = []
            leaves: List[str] = []
            for child in g._v_children.values():
                pandas_type = getattr(child._v_attrs, 'pandas_type', None)
                if pandas_type is None:
                    if isinstance(child, _table_mod.group.Group):
                        groups.append(child._v_name)
                else:
                    leaves.append(child._v_name)
            yield g._v_pathname.rstrip('/'), groups, leaves

    def func_pfd805fb(self, key: str) -> Optional[Node]:
        """return the node with the key or None if it does not exist"""
        self._check_if_open()
        if not key.startswith('/'):
            key = '/' + key
        assert self._handle is not None
        assert _table_mod is not None
        try:
            node = self._handle.get_node(self.root, key)
        except _table_mod.exceptions.NoSuchNodeError:
            return None
        assert isinstance(node, _table_mod.Node), type(node)
        return node

    def func_yjb4p6om(self, key: str) -> Any:
        """return the storer object for a key, raise if not in the file"""
        group = self.get_node(key)
        if group is None:
            raise KeyError(f'No object named {key} in the file')
        s = self._create_storer(group)
        s.infer_axes()
        return s

    def func_4adgr5r9(
        self,
        file: str,
        mode: Literal['w', 'a', 'r', 'r+'] = 'w',
        propindexes: bool = True,
        keys: Optional[Union[List[str], Tuple[str, ...]]] = None,
        complib: Optional[str] = None,
        complevel: Optional[int] = None,
        fletcher32: bool = False,
        overwrite: bool = True,
    ) -> HDFStore:
        """
        Copy the existing store to a new file, updating in place.

        Parameters
        ----------
        propindexes : bool, default True
            Restore indexes in copied file.
        keys : List[str], optional
            List of keys to include in the copy (defaults to all).
        overwrite : bool, default True
            Whether to overwrite (remove and replace) existing nodes in the new store.
        mode, complib, complevel, fletcher32 same as in HDFStore.__init__

        Returns
        -------
        HDFStore
            Open file handle of the new store
        """
        new_store = HDFStore(file, mode=mode, complib=complib, complevel=complevel, fletcher32=fletcher32)
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
                    index = False
                    if propindexes:
                        index = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(
                        k,
                        data,
                        index=index,
                        data_columns=getattr(s, 'data_columns', None),
                        encoding=s.encoding,
                    )
                else:
                    new_store.put(k, data, encoding=s.encoding)
        return new_store

    def func_hcl4jnbg(self) -> str:
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
        >>> store.close()  # doctest: +SKIP
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
                keys: List[Any] = []
                values: List[str] = []
                for k in lkeys:
                    try:
                        s = self.get_storer(k)
                        if s is not None:
                            keys.append(pprint_thing(s.pathname or k))
                            values.append(pprint_thing(s or 'invalid_HDFStore node'))
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

    def func_j1l0p7ai(self) -> None:
        if not self.is_open:
            raise ClosedFileError(f'{self._path} file is not open!')

    def func_0aco1kn7(self, format: str) -> str:
        """validate / deprecate formats"""
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f'invalid HDFStore format specified [{format}]') from err
        return format

    def func_ykmxryjx(
        self,
        group: Node,
        format: Optional[str] = None,
        value: Optional[Union[DataFrame, Series]] = None,
        encoding: str = 'UTF-8',
        errors: str = 'strict',
    ) -> Any:
        """return a suitable class to operate"""
        if value is not None and not isinstance(value, (Series, DataFrame)):
            raise TypeError('value must be None, Series, or DataFrame')
        pt = getattr(group._v_attrs, 'pandas_type', None)
        tt = getattr(group._v_attrs, 'table_type', None)
        if pt is None:
            if value is None:
                func_q9kzi9o4()
                assert _table_mod is not None
                if getattr(group, 'table', None) or isinstance(group, _table_mod.table.Table):
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
            _STORER_MAP: Dict[str, type] = {'series': SeriesFixed, 'frame': FrameFixed}
            try:
                cls = _STORER_MAP[pt]
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
        _TABLE_MAP: Dict[str, type] = {
            'generic_table': GenericTable,
            'appendable_series': AppendableSeriesTable,
            'appendable_multiseries': AppendableMultiSeriesTable,
            'appendable_frame': AppendableFrameTable,
            'appendable_multiframe': AppendableMultiFrameTable,
            'worm': WORMTable,
        }
        try:
            cls = _TABLE_MAP[tt]
        except KeyError as err:
            raise TypeError(
                f'cannot properly create the storer for: [_TABLE_MAP] [group->{group},value->{type(value)},format->{format}'
            ) from err
        return cls(self, group, encoding=encoding, errors=errors)

    def func_xj4xi0yq(
        self,
        key: str,
        value: Union[DataFrame, Series],
        format: str,
        axes: Optional[List[int]] = None,
        index: bool = True,
        append: bool = False,
        complib: Optional[str] = None,
        complevel: Optional[int] = None,
        fletcher32: Optional[bool] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        chunksize: Optional[int] = None,
        expectedrows: Optional[int] = None,
        dropna: bool = False,
        nan_rep: Optional[str] = None,
        data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None,
        errors: Literal['strict'] = 'strict',
        track_times: bool = True,
    ) -> None:
        if getattr(value, 'empty', None) and (format == 'table' or append):
            return
        group = self._identify_group(key, append)
        s = self._create_storer(group, format, value, encoding=encoding, errors=errors)
        if append:
            if (not s.is_table or (s.is_table and format == 'fixed' and s.is_exists)):
                raise ValueError('Can only append to Tables')
            if not s.is_exists:
                s.set_object_info()
        else:
            s.set_object_info()
        if not s.is_table and complib:
            raise ValueError('Compression not supported on Fixed format stores')
        s.write(
            obj=value,
            axes=axes,
            append=append,
            complib=complib,
            complevel=complevel,
            fletcher32=fletcher32,
            min_itemsize=min_itemsize,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            nan_rep=nan_rep,
            data_columns=data_columns,
            track_times=track_times,
        )
        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def func_j0dqjvyn(self, group: Node) -> Any:
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def func_c5cfn3mh(self, key: str, append: bool) -> Node:
        """Identify HDF5 group based on key, delete/create group if needed."""
        group = self.get_node(key)
        assert self._handle is not None
        if group is not None and not append:
            self._handle.remove_node(group, recursive=True)
            group = None
        if group is None:
            group = self._create_nodes_and_group(key)
        return group

    def func_zsy7e2zp(self, key: str) -> Node:
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
    chunksize : Optional[int] = None
        the passed chunking value (default is 100000)
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """

    def __init__(
        self,
        store: HDFStore,
        s: Any,
        func: Callable[[Optional[int], Optional[int], Optional[Any]], Any],
        where: Optional[Any],
        nrows: int,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        iterator: bool = False,
        chunksize: Optional[int] = None,
        auto_close: bool = False,
    ) -> None:
        self.store = store
        self.s = s
        self.func = func
        self.where = where
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)
        self.nrows = nrows
        self.start = start
        self.stop = stop
        self.coordinates: Optional[Index] = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize: Optional[int] = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close = auto_close

    def __iter__(self) -> Iterator[Any]:
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

    def func_z55zjm5k(self) -> None:
        if self.auto_close:
            self.store.close()

    def func_n23j8dd6(self, coordinates: bool = False) -> Any:
        if self.chunksize is not None:
            if not isinstance(self.s, Table):
                raise TypeError('can only use an iterator or chunksize on a table')
            self.coordinates = self.s.read_coordinates(where=self.where)
            return self
        if coordinates:
            if not isinstance(self.s, Table):
                raise TypeError('can only read_coordinates on a table')
            where = self.s.read_coordinates(where=self.where, start=self.start, stop=self.stop)
        else:
            where = self.where
        results = self.func(self.start, self.stop, where)
        self.close()
        return results

    def close(self) -> None:
        self.func_z55zjm5k()

    def get_result(self, coordinates: bool = False) -> Any:
        return self.func_n23j8dd6(coordinates=coordinates)


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
    _info_fields: List[str] = ['freq', 'tz', 'index_name']

    def __init__(
        self,
        name: str,
        values: Optional[np.ndarray] = None,
        kind: Optional[str] = None,
        typ: Any = None,
        cname: Optional[str] = None,
        axis: Optional[int] = None,
        pos: Optional[int] = None,
        freq: Optional[str] = None,
        tz: Optional[str] = None,
        index_name: Optional[str] = None,
        ordered: Optional[bool] = None,
        table: Optional[Any] = None,
        meta: Optional[str] = None,
        metadata: Optional[np.ndarray] = None,
    ) -> None:
        if not isinstance(name, str):
            raise ValueError('`name` must be a str.')
        self.values: Optional[np.ndarray] = values
        self.kind: Optional[str] = kind
        self.typ: Any = typ
        self.name: str = name
        self.cname: str = cname or name
        self.axis: Optional[int] = axis
        self.pos: Optional[int] = pos
        self.freq: Optional[str] = freq
        self.tz: Optional[str] = tz
        self.index_name: Optional[str] = index_name
        self.ordered: Optional[bool] = ordered
        self.table: Optional[Any] = table
        self.meta: Optional[str] = meta
        self.metadata: Optional[np.ndarray] = metadata
        if pos is not None:
            self.set_pos(pos)
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def func_pcop8h9u(self) -> int:
        """return my itemsize"""
        return self.typ.itemsize

    @property
    def func_gqqnlz4j(self) -> str:
        return f'{self.name}_kind'

    def func_ceo8u8yl(self, pos: int) -> None:
        """set the position of this column in the Table"""
        self.pos = pos
        if pos is not None and self.typ is not None:
            self.typ._v_pos = pos

    def __repr__(self) -> str:
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.axis, self.pos, self.kind)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name', 'cname', 'axis', 'pos', 'kind'], temp)])

    def __eq__(self, other: Any) -> bool:
        """compare 2 col items"""
        if not isinstance(other, IndexCol):
            return False
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ['name', 'cname', 'axis', 'pos']
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def func_qba4ty6u(self) -> bool:
        """return whether I am an indexed column"""
        if not hasattr(self.table, 'cols'):
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def func_eeodmoc2(
        self,
        values: np.ndarray,
        nan_rep: Optional[str],
        encoding: str,
        errors: str,
    ) -> Tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname].copy()
        val_kind = self.kind
        values = _maybe_convert(values, val_kind, encoding, errors)
        kwargs: Dict[str, Any] = {}
        kwargs['name'] = self.index_name
        if self.freq is not None:
            kwargs['freq'] = self.freq
        factory: Callable[..., Index] = Index
        if lib.is_np_dtype(values.dtype, 'M') or isinstance(values.dtype, DatetimeTZDtype):
            factory = DatetimeIndex
        elif isinstance(self, PeriodDtype) and self.freq is not None:
            factory = lambda x, **kwds: PeriodIndex.from_ordinals(x, freq=kwds.get('freq', None))._rename(kwds['name'])
        try:
            new_pd_index = factory(values, **kwargs)
        except ValueError:
            if 'freq' in kwargs:
                kwargs['freq'] = None
            new_pd_index = factory(values, **kwargs)
        if self.tz is not None and isinstance(new_pd_index, DatetimeIndex):
            final_pd_index = new_pd_index.tz_localize('UTC').tz_convert(self.tz)
        else:
            final_pd_index = new_pd_index
        return final_pd_index, final_pd_index

    def func_grn3u58x(self) -> Optional[np.ndarray]:
        """return the values"""
        return self.values

    @property
    def func_oxrftv2c(self) -> Any:
        return self.table._v_attrs

    @property
    def func_bahbrdye(self) -> Any:
        return self.table.description

    @property
    def func_tbvkn3hi(self) -> Optional[Any]:
        """return my current col description"""
        return getattr(self.description, self.cname, None)

    @property
    def func_l03gavbl(self) -> Optional[np.ndarray]:
        """return my cython values"""
        return self.values

    def __iter__(self) -> Iterator[Any]:
        return iter(self.values)

    def func_4cszqa1g(self, min_itemsize: Optional[Union[int, Dict[str, int]]] = None) -> None:
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
        if self.kind == 'string':
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)
            if min_itemsize is not None and self.typ.itemsize < min_itemsize:
                self.typ = func_q9kzi9o4().StringCol(itemsize=min_itemsize, pos=self.pos)

    def func_pm6pqyjb(self) -> None:
        pass

    def func_qxkl0p5o(
        self,
        handler: Any,
        append: bool,
    ) -> None:
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def func_mf5h7wwa(self, itemsize: Optional[int] = None) -> Optional[int]:
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

    def func_9yxvmb78(self, append: bool) -> None:
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(
                    f'incompatible kind in col [{existing_kind} - {self.kind}]'
                )

    def func_s1dlau4x(self, info: Dict[str, Any]) -> None:
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
                    warnings.warn(ws, AttributeConflictWarning, stacklevel=find_stack_level())
                    idx[key] = None
                    setattr(self, key, None)
                else:
                    raise ValueError(
                        f'invalid info for [{self.name}] for [{key}], existing_value [{existing_value}] conflicts with new value [{value}]'
                    )
            elif value is not None or existing_value is not None:
                idx[key] = value

    def func_fp3ae04r(self, info: Dict[str, Any]) -> None:
        """set my state from the passed info"""
        idx = info.get(self.name)
        if idx is not None:
            self.__dict__.update(idx)

    def func_6z2vnl50(self) -> None:
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)

    def func_ku3plf7v(self, handler: Any) -> None:
        """validate that kind=category does not change the categories"""
        if self.meta == 'category':
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if (
                new_metadata is not None
                and cur_metadata is not None
                and not array_equivalent(new_metadata, cur_metadata, strict_nan=True, dtype_equal=True)
            ):
                raise ValueError(
                    'cannot append a categorical with different categories to the existing'
                )

    def func_soevodf0(self, handler: Any) -> None:
        """set the meta data"""
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)


class GenericIndexCol(IndexCol):
    """an index which is not represented in the data of the table"""

    @property
    def func_qba4ty6u(self) -> bool:
        return False

    def func_eeodmoc2(
        self,
        values: np.ndarray,
        nan_rep: Optional[str],
        encoding: str,
        errors: str,
    ) -> Tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep : str
        encoding : str
        errors : str

        Returns
        -------
        index : List-like to become an Index
        data : ndarray-like to become a column
        """
        index = RangeIndex(len(values))
        return index, index

    @classmethod
    def func_i1b00jvb(cls, values: np.ndarray) -> Any:
        """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
        dtype = values.dtype
        itemsize = dtype.itemsize
        shape = values.shape
        if values.ndim == 1:
            shape = 1, values.size
        if isinstance(values, Categorical):
            codes = values.codes
            atom = cls.get_atom_data(shape, kind=codes.dtype.name)
        elif lib.is_np_dtype(dtype, 'M') or isinstance(dtype, DatetimeTZDtype):
            atom = cls.get_atom_datetime64(shape)
        elif lib.is_np_dtype(dtype, 'm'):
            atom = cls.get_atom_timedelta64(shape)
        elif is_complex_dtype(dtype):
            atom = func_q9kzi9o4().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif is_string_dtype(dtype):
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)
        return atom

    @classmethod
    def func_e11nqklj(cls, shape: Tuple[int, ...], itemsize: int) -> Any:
        return func_q9kzi9o4().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def func_v0qksyk0(cls, kind: str) -> Any:
        """return the PyTables column class for this column"""
        if kind.startswith('uint'):
            k4 = kind[4:]
            col_name = f'UInt{k4}Col'
        elif kind.startswith('period'):
            col_name = 'Int64Col'
        else:
            kcap = kind.capitalize()
            col_name = f'{kcap}Col'
        return getattr(func_q9kzi9o4(), col_name)

    @classmethod
    def func_qph71b91(cls, shape: Tuple[int, ...], kind: str) -> Any:
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def func_8pml91ak(cls, shape: Tuple[int, ...]) -> Any:
        return func_q9kzi9o4().Int64Col(shape=shape[0])

    @classmethod
    def func_4uo6n0nb(cls, shape: Tuple[int, ...]) -> Any:
        return func_q9kzi9o4().Int64Col(shape=shape[0])

    @property
    def func_8mbgw4kh(self) -> Optional[int]:
        return getattr(self, 'nrows', None)

    @property
    def func_l03gavbl(self) -> Optional[np.ndarray]:
        """return my cython values"""
        return self.data

    def func_6z2vnl50(self) -> None:
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)


class GenericIndexCol(IndexCol):
    """an index which is not represented in the data of the table"""


class DataCol(IndexCol):
    """
    a data holding column, by definition this is not indexable

    Parameters
    ----------
    data   : the actual data
    cname  : the column name in the table to hold the data (typically
                values)
    meta   : a string description of the metadata
    metadata : the actual metadata
    """
    is_an_indexable: bool = False
    is_data_indexable: bool = False
    _info_fields: List[str] = ['tz', 'ordered']

    def __init__(
        self,
        name: str,
        values: Optional[np.ndarray] = None,
        kind: Optional[str] = None,
        typ: Any = None,
        cname: Optional[str] = None,
        pos: Optional[int] = None,
        tz: Optional[str] = None,
        ordered: Optional[bool] = None,
        table: Optional[Any] = None,
        meta: Optional[str] = None,
        metadata: Optional[np.ndarray] = None,
        dtype: Optional[str] = None,
        data: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            cname=cname,
            pos=pos,
            tz=tz,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
        )
        self.dtype: Optional[str] = dtype
        self.data: Optional[np.ndarray] = data

    @property
    def func_e8yqxdwc(self) -> str:
        return f'{self.name}_dtype'

    @property
    def func_dcmv510e(self) -> str:
        return f'{self.name}_meta'

    def __repr__(self) -> str:
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.dtype, self.kind, self.shape)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name', 'cname', 'dtype', 'kind', 'shape'], temp)])

    def __eq__(self, other: Any) -> bool:
        """compare 2 col items"""
        if not isinstance(other, DataCol):
            return False
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ['name', 'cname', 'dtype', 'pos']
        )

    def func_btxevbct(self, data: np.ndarray) -> None:
        assert data is not None
        assert self.dtype is None
        data, dtype_name = _get_data_and_dtype_name(data)
        self.data = data
        self.dtype = dtype_name
        self.kind = _dtype_to_kind(dtype_name)

    def func_grn3u58x(self) -> Optional[np.ndarray]:
        """return the data"""
        return self.data

    @classmethod
    def func_i1b00jvb(cls, values: np.ndarray) -> Any:
        """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
        dtype = values.dtype
        itemsize = dtype.itemsize
        shape = values.shape
        if values.ndim == 1:
            shape = 1, values.size
        if isinstance(values, Categorical):
            codes = values.codes
            atom = cls.get_atom_data(shape, kind=codes.dtype.name)
        elif lib.is_np_dtype(dtype, 'M') or isinstance(dtype, DatetimeTZDtype):
            atom = cls.get_atom_datetime64(shape)
        elif lib.is_np_dtype(dtype, 'm'):
            atom = cls.get_atom_timedelta64(shape)
        elif is_complex_dtype(dtype):
            atom = func_q9kzi9o4().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif is_string_dtype(dtype):
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)
        return atom

    @classmethod
    def func_e11nqklj(cls, shape: Tuple[int, ...], itemsize: int) -> Any:
        return func_q9kzi9o4().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def func_v0qksyk0(cls, kind: str) -> Any:
        """return the PyTables column class for this column"""
        if kind.startswith('uint'):
            k4 = kind[4:]
            col_name = f'UInt{k4}Col'
        elif kind.startswith('period'):
            col_name = 'Int64Col'
        else:
            kcap = kind.capitalize()
            col_name = f'{kcap}Col'
        return getattr(func_q9kzi9o4(), col_name)

    @classmethod
    def func_qph71b91(cls, shape: Tuple[int, ...], kind: str) -> Any:
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def func_8pml91ak(cls, shape: Tuple[int, ...]) -> Any:
        return func_q9kzi9o4().Int64Col()

    @classmethod
    def func_4uo6n0nb(cls, shape: Tuple[int, ...]) -> Any:
        return func_q9kzi9o4().Int64Col()

    @property
    def func_8mbgw4kh(self) -> Optional[Tuple[int, ...]]:
        """return my shape"""
        return getattr(self, 'shape', None)

    @property
    def func_l03gavbl(self) -> Optional[np.ndarray]:
        """return my cython values"""
        return self.data

    @classmethod
    def func_ad609m4b(cls, kind: str) -> bool:
        """
        Find the "kind" string describing the given dtype name.
        """
        if kind in ('datetime64', 'string') or 'datetime64' in kind:
            return True
        return False

    def func_9yxvmb78(self, append: bool) -> None:
        """validate that we have the same order as the existing & same dtype"""
        if append:
            existing_fields = getattr(self.attrs, self.kind_attr, None)
            if existing_fields is not None and existing_fields != list(self.values):
                raise ValueError(
                    'appended items do not match existing items in table!'
                )
            existing_dtype = getattr(self.attrs, self.dtype_attr, None)
            if existing_dtype is not None and existing_dtype != self.dtype:
                raise ValueError(
                    'appended items dtype do not match existing items dtype in table!'
                )

    def func_eeodmoc2(
        self,
        values: np.ndarray,
        nan_rep: Optional[str],
        encoding: str,
        errors: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep : str
        encoding : str
        errors : str

        Returns
        -------
        index : listlike to become an Index
        data : ndarraylike to become a column
        """
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname].copy()
        if self.dtype is None:
            converted, dtype_name = _get_data_and_dtype_name(values)
            kind = _dtype_to_kind(dtype_name)
        else:
            converted = values
            dtype_name = self.dtype
            kind = self.kind
        assert isinstance(converted, np.ndarray)
        meta = self.meta
        metadata = self.metadata
        ordered = self.ordered
        tz = self.tz
        assert dtype_name is not None
        dtype = dtype_name
        if dtype.startswith('datetime64'):
            converted = _set_tz(converted, tz, dtype)
        elif dtype == 'timedelta64':
            converted = np.asarray(converted, dtype='m8[ns]')
        elif dtype == 'date':
            try:
                converted = np.asarray([date.fromordinal(v) for v in converted], dtype=object)
            except ValueError:
                converted = np.asarray([date.fromtimestamp(v) for v in converted], dtype=object)
        elif meta == 'category':
            categories = metadata
            codes = converted.ravel()
            if categories is None:
                categories = Index([], dtype=np.float64)
            else:
                mask = isna(categories)
                if mask.any():
                    categories = categories[~mask]
                    codes[codes != -1] -= mask.astype(int).cumsum()._values
            converted = Categorical.from_codes(codes, categories=categories, ordered=ordered, validate=False)
        else:
            try:
                converted = converted.astype(dtype, copy=False)
            except TypeError:
                converted = converted.astype('O', copy=False)
        if kind == 'string':
            converted = _unconvert_string_array(converted, nan_rep=nan_rep, encoding=encoding, errors=errors)
        return self.values, converted

    def func_qph71b91(self, kind: str, encoding: str, errors: str) -> Callable[[np.ndarray], Any]:
        def converter(x: np.ndarray) -> np.ndarray:
            if kind == 'datetime64':
                return np.asarray(x, dtype='M8[ns]')
            elif 'datetime64' in kind:
                return np.asarray(x, dtype=kind)
            elif kind == 'string':
                return func_buaqa4gk(x, nan_rep=None, encoding=encoding, errors=errors)
            else:
                raise ValueError(f'invalid kind {kind}')
        return converter

    def func_ad609m4b(self, kind: str) -> bool:
        """check if kind requires conversion"""
        if kind in ('datetime64', 'string') or 'datetime64' in kind:
            return True
        return False

    def func_y322ga6n(
        self,
        name: str,
        index: Index,
        encoding: str,
        errors: str
    ) -> IndexCol:
        assert isinstance(name, str)
        index_name = index.name
        converted, dtype_name = _get_data_and_dtype_name(index)
        kind = _dtype_to_kind(dtype_name)
        atom = DataIndexableCol._get_atom(converted)
        if (
            lib.is_np_dtype(index.dtype, 'iu')
            or needs_i8_conversion(index.dtype)
            or is_bool_dtype(index.dtype)
        ):
            return IndexCol(
                name=name,
                values=converted,
                kind=kind,
                typ=atom,
                freq=getattr(index, 'freq', None),
                tz=getattr(index, 'tz', None),
                index_name=index_name,
            )
        if isinstance(index, MultiIndex):
            raise TypeError('MultiIndex not supported here!')
        inferred_type = lib.infer_dtype(index, skipna=False)
        values = np.asarray(index)
        if inferred_type == 'date':
            converted = np.asarray([date.fromordinal(v) for v in values], dtype=np.int32)
            return IndexCol(
                name=name,
                values=converted,
                kind='date',
                typ=func_q9kzi9o4().Time32Col(),
                index_name=index_name,
            )
        elif inferred_type == 'string':
            converted = _convert_string_array(values, encoding, errors)
            itemsize = converted.dtype.itemsize
            return IndexCol(
                name=name,
                values=converted,
                kind='string',
                typ=func_q9kzi9o4().StringCol(itemsize),
                index_name=index_name,
            )
        elif inferred_type in ['integer', 'floating']:
            return IndexCol(name=name, values=converted, kind=kind, typ=atom, index_name=index_name)
        else:
            assert isinstance(converted, np.ndarray) and converted.dtype == object
            assert kind == 'object', kind
            atom = func_q9kzi9o4().ObjectAtom()
            return IndexCol(
                name=name,
                values=converted,
                kind=kind,
                typ=atom,
                index_name=index_name,
            )

    def func_el8fwaj1(data: Any, encoding: str, errors: str) -> np.ndarray:
        """
        Inverse of _convert_string_array.

        Parameters
        ----------
        data : np.ndarray[fixed-length-string]
        nan_rep : str
        encoding : str
        errors : str
            Handler for encoding errors.

        Returns
        -------
        np.ndarray[object]
            Decoded data.
        """
        shape = data.shape
        data = np.asarray(data.ravel(), dtype=object)
        if len(data):
            itemsize = libwriters.max_len_string_array(ensure_object(data))
            dtype = f'U{itemsize}'
            if isinstance(data[0], bytes):
                ser = Series(data, copy=False).str.decode(encoding, errors=errors)
                data = ser.to_numpy()
                data.flags.writeable = True
            else:
                data = data.astype(dtype, copy=False).astype(object, copy=False)
        if nan_rep is None:
            nan_rep = 'nan'
        libwriters.string_array_replace_from_nan_rep(data, nan_rep)
        return data.reshape(shape)


class GenericFixed(Fixed):
    """a generified fixed version"""
    _index_type_map: Dict[Any, str] = {DatetimeIndex: 'datetime', PeriodIndex: 'period'}
    _reverse_index_map: Dict[str, type] = {v: k for k, v in _index_type_map.items()}
    attributes: List[str] = []

    @property
    def func_pcop8h9u(self) -> str:
        """Return string description"""
        return self.version[0] <= 0 and self.version[1] <= 10 and self.version[2] < 1

    @property
    def func_8xi8qani(self) -> Tuple[int, int, int]:
        """compute and set our version"""
        version = getattr(self.group._v_attrs, 'pandas_version', None)
        if isinstance(version, str):
            version_tup = tuple(int(x) for x in version.split('.'))
            if len(version_tup) == 2:
                version_tup = version_tup + (0,)
            assert len(version_tup) == 3
            return version_tup
        else:
            return (0, 0, 0)

    @property
    def func_vgmxe3i9(self) -> Optional[str]:
        return getattr(self.group._v_attrs, 'pandas_type', None)

    def __repr__(self) -> str:
        """return a pretty representation of myself"""
        self.infer_axes()
        s = self.shape
        if s is not None:
            if isinstance(s, (list, tuple)):
                jshape = ','.join([pprint_thing(x) for x in s])
                s = f'[{jshape}]'
            return f'{self.pandas_type:12.12} (shape->{s})'
        return self.pandas_type

    def func_glhze88r(self) -> None:
        """set our object attributes"""
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors

    def func_gr2vbxnc(self) -> None:
        """retrieve our attributes"""
        self.encoding = func_lygurqaw(getattr(self.attrs, 'encoding', None))
        self.errors = getattr(self.attrs, 'errors', 'strict')
        for n in self.attributes:
            setattr(self, n, getattr(self.attrs, n, None))

    def func_tgxf1jr6(
        self,
        obj: Union[DataFrame, Series],
        **kwargs: Any,
    ) -> None:
        """write in a format that we can search later on (but cannot append
        to): write out the indices and the values using _write_array
        (e.g. a CArray) create an indexing table so that we can search
        """
        raise NotImplementedError('GenericFixed cannot implement write')

    def func_0g6p4o97(
        self,
        key: str,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> np.ndarray:
        """read an array for the specified node (off of group"""
        import tables
        node = getattr(self.group, key)
        attrs = node._v_attrs
        transposed = getattr(attrs, 'transposed', False)
        if isinstance(node, tables.VLArray):
            ret = node[0][start:stop]
            dtype = getattr(attrs, 'value_type', None)
            if dtype is not None:
                ret = pd_array(ret, dtype=dtype)
        else:
            dtype = getattr(attrs, 'value_type', None)
            shape = getattr(attrs, 'shape', None)
            if shape is not None:
                ret = np.empty(shape, dtype=dtype)
            else:
                ret = node[start:stop]
            if dtype and dtype.startswith('datetime64'):
                tz = getattr(attrs, 'tz', None)
                ret = _set_tz(ret, tz, dtype)
            elif dtype == 'timedelta64':
                ret = np.asarray(ret, dtype='m8[ns]')
        if transposed:
            return ret.T
        else:
            return ret

    def func_kotm0c9l(
        self,
        key: str,
        index: Index,
    ) -> Any:
        """write a 0-len array"""
        raise NotImplementedError('GenericFixed cannot implement read')

    def func_djl8qbft(
        self,
        key: str,
        index: Index,
    ) -> Any:
        """write a 0-len array"""
        raise NotImplementedError('GenericFixed cannot implement write')

    def func_qzwi0szd(self, key: str) -> Any:
        """Retrieve pandas object stored in file"""
        return super().func_qzwi0szd(key)

    def func_pfd805fb(self, key: str) -> Optional[Node]:
        """return the node with the key or None if it does not exist"""
        return super().func_pfd805fb(key)

    def func_yjb4p6om(self, key: str) -> Any:
        """return the storer object for a key, raise if not in the file"""
        return super().func_yjb4p6om(key)

    def get_storer(self, key: str) -> Optional[Union[GenericFixed, FrameFixed, SeriesFixed, AppendableTable, Table, WORMTable]]:
        """Get the storer for a key, return None if it does not exist"""
        node = self.get_node(key)
        if node is None:
            return None
        return self._create_storer(node)

    def _create_storer(self, group: Node, format: Optional[str] = None, value: Optional[Union[DataFrame, Series]] = None, encoding: str = 'UTF-8', errors: str = 'strict') -> Any:
        return self.func_ykmxryjx(group, format=format, value=value, encoding=encoding, errors=errors)

    def _identify_group(
        self, key: str, append: bool
    ) -> Node:
        return self.func_c5cfn3mh(key, append)

    def _create_nodes_and_group(self, key: str) -> Node:
        return self.func_zsy7e2zp(key)

    def _validate_format(self, format: str) -> str:
        return self.func_0aco1kn7(format)

    def get_node(self, key: str) -> Optional[Node]:
        return self.func_pfd805fb(key)

    def get(
        self,
        key: str,
    ) -> Any:
        return self.func_qzwi0szd(key)

    def put(
        self,
        key: str,
        value: Union[DataFrame, Series],
        format: Optional[str] = None,
        index: bool = True,
        append: bool = False,
        complib: Optional[str] = None,
        complevel: Optional[int] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep: Optional[str] = None,
        data_columns: Optional[Union[List[str], bool]] = None,
        errors: Literal['strict'] = 'strict',
        track_times: bool = True,
        dropna: bool = False,
    ) -> None:
        self.func_zee8uqzp(
            key=key,
            value=value,
            format=format,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
            track_times=track_times,
            dropna=dropna,
        )

    def get_storer(self, key: str) -> Optional[Any]:
        return self.func_yjb4p6om(key)

    def append(
        self,
        key: str,
        value: Union[DataFrame, Series],
        format: Optional[str] = None,
        index: bool = True,
        append: bool = True,
        complib: Optional[str] = None,
        complevel: Optional[int] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        chunksize: Optional[int] = None,
        expectedrows: Optional[int] = None,
        dropna: bool = False,
        nan_rep: Optional[str] = None,
        data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None,
        errors: Literal['strict'] = 'strict',
        track_times: bool = True,
    ) -> None:
        self.func_zee8uqzp(
            key=key,
            value=value,
            format=format,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
            track_times=track_times,
        )

    def remove(
        self,
        key: str,
    ) -> None:
        self.func_30nemlpc(
            key=key,
            where=None,
            start=None,
            stop=None,
        )

    def append_to_multiple(
        self,
        d: Dict[str, Optional[List[str]]],
        value: DataFrame,
        selector: str,
        data_columns: Optional[Union[List[str], bool]] = None,
        axes: Optional[List[int]] = None,
        dropna: bool = False,
        **kwargs: Any,
    ) -> None:
        self.func_5ol8yjd1(
            d=d,
            value=value,
            selector=selector,
            data_columns=data_columns,
            axes=axes,
            dropna=dropna,
            **kwargs,
        )

    def create_index(
        self,
        key: str,
        columns: Optional[Union[List[str], bool]] = None,
        optlevel: Optional[int] = None,
        kind: Optional[str] = None,
    ) -> None:
        self.func_ckoyybrz(
            key=key,
            columns=columns,
            optlevel=optlevel,
            kind=kind,
        )

    def groups(self) -> List[Node]:
        return self.func_s8r4yw3o()

    def walk(
        self,
        where: str = '/',
    ) -> Iterator[Tuple[str, List[str], List[str]]]:
        return self.func_hx6f1lva(where)

    def keys(self) -> List[str]:
        return self.func_ldc0l8ou()

    def walkgroups(
        self,
        where: str = '/',
    ) -> Iterator[Tuple[str, List[str], List[str]]]:
        return self.func_hx6f1lva(where)

    def info(self) -> str:
        return self.func_hcl4jnbg()

    def close(self) -> None:
        self.func_z55zjm5k()

    def get_group_nodes(self, group: Node) -> List[Any]:
        return self.func_wwtmnm8a(group)

    def _check_if_open(self) -> None:
        return self.func_j1l0p7ai()

    def select(
        self,
        key: str,
        where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        columns: Optional[List[str]] = None,
        iterator: bool = False,
        chunksize: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        return func_f0qzhk05(
            path_or_buf=self,
            key=key,
            mode='r',
            errors='strict',
            where=where,
            start=start,
            stop=stop,
            columns=columns,
            iterator=iterator,
            chunksize=chunksize,
            **kwargs,
        )


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
    chunksize : Optional[int] = None
        the passed chunking value (default is 100000)
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """

    def __init__(
        self,
        store: HDFStore,
        s: Any,
        func: Callable[[Optional[int], Optional[int], Optional[Any]], Any],
        where: Optional[Any],
        nrows: int,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        iterator: bool = False,
        chunksize: Optional[int] = None,
        auto_close: bool = False,
    ) -> None:
        self.store = store
        self.s = s
        self.func = func
        self.where = where
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)
        self.nrows = nrows
        self.start = start
        self.stop = stop
        self.coordinates: Optional[Index] = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize: Optional[int] = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close = auto_close

    def __iter__(self) -> Iterator[Any]:
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

    def __next__(self) -> Any:
        return self.__iter__().__next__()

    def close(self) -> None:
        self.func_z55zjm5k()

    def get_result(self, coordinates: bool = False) -> Any:
        return self.func_n23j8dd6(coordinates=coordinates)

    def func_z55zjm5k(self) -> None:
        """close the iterator"""
        if self.auto_close:
            self.store.close()

    def func_n23j8dd6(self, coordinates: bool = False) -> Any:
        """generate the selection"""
        if self.chunksize is not None:
            if not isinstance(self.s, Table):
                raise TypeError('can only use an iterator or chunksize on a table')
            self.coordinates = self.s.read_coordinates(where=self.where)
            return self
        if coordinates:
            if not isinstance(self.s, Table):
                raise TypeError('can only read_coordinates on a table')
            where = self.s.read_coordinates(where=self.where, start=self.start, stop=self.stop)
        else:
            where = self.where
        results = self.func(self.start, self.stop, where)
        self.close()
        return results


class IndexCol:
    # ... (previous IndexCol code)

    pass  # Placeholder for additional methods


class GenericFixed(Fixed):
    """a generified fixed version"""
    _index_type_map: Dict[Any, str] = {DatetimeIndex: 'datetime', PeriodIndex: 'period'}
    _reverse_index_map: Dict[str, type] = {v: k for k, v in _index_type_map.items()}
    attributes: List[str] = []

    @property
    def func_pcop8h9u(self) -> bool:
        """Return whether the version is <= 0.10.1"""
        return self.func_8xi8qani()[0] <= 0 and self.func_8xi8qani()[1] <= 10 and self.func_8xi8qani()[2] < 1

    @property
    def func_8xi8qani(self) -> Tuple[int, int, int]:
        """compute and set our version"""
        return super().func_8xi8qani()

    @property
    def func_vgmxe3i9(self) -> Optional[str]:
        return super().func_vgmxe3i9

    def __repr__(self) -> str:
        return super().__repr__()

    def func_ceqp3p3c(self, info: Dict[str, Any]) -> None:
        """set/update the info for this indexable with the key/value"""
        super().func_s1dlau4x(info)

    def func_to7iirh7(
        self,
        columns: Optional[List[str]] = None,
        optlevel: Optional[int] = None,
        kind: Optional[str] = None,
    ) -> None:
        """
        Create a pytables index on the specified columns.

        Parameters
        ----------
        columns : Optional[Union[List[str], bool]]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : Optional[int], default None
            Optimization level, if None, pytables defaults to 6.
        kind : Optional[str], default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError: raises if trying to create an index on a complex-type column.

        Notes
        -----
        Cannot index Time64Col or ComplexCol.
        Pytables must be >= 3.0.
        """
        self.func_to7iirh7(columns, optlevel, kind)

    def get_attrs(self) -> None:
        """get our object attributes"""
        super().func_gr2vbxnc()

    def set_attrs(self) -> None:
        """set our object attributes"""
        super().func_glhze88r()

    def write(
        self,
        obj: Union[DataFrame, Series],
        **kwargs: Any,
    ) -> None:
        """write to the store"""
        super().func_tgxf1jr6(obj, **kwargs)

    def read(self) -> Any:
        """read from the store"""
        return super().func_j0dqjvyn(self.group)

    pass  # Placeholder for additional methods


class SeriesFixed(GenericFixed):
    pandas_kind: str = 'series'
    attributes: List[str] = ['name']

    @property
    def func_8mbgw4kh(self) -> Optional[float]:
        try:
            return len(self.group.values),
        except (TypeError, AttributeError):
            return None

    def func_8c0lfbv1(
        self,
        where: Optional[Union[List[Term], Term]] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> Series:
        """read the series from the store"""
        self.validate_read(columns, where)
        index = self.read_index('index', start=start, stop=stop)
        values = self.read_array('values', start=start, stop=stop)
        result = Series(values, index=index, name=self.name, copy=False)
        if using_string_dtype() and isinstance(values, np.ndarray) and is_string_array(values, skipna=True):
            result = result.astype(StringDtype(na_value=np.nan))
        return result

    def func_tgxf1jr6(
        self,
        obj: Series,
        **kwargs: Any,
    ) -> None:
        """write the series to the store"""
        super().write(obj, **kwargs)
        self.write_index('index', obj.index)
        self.write_array('values', obj)
        self.attrs.name = obj.name


class BlockManagerFixed(GenericFixed):
    """
    represent a block manager fixed format object.
    """
    attributes: List[str] = ['ndim', 'nblocks']
    pandas_kind: str = 'block_manager'
    obj_type: type = DataFrame

    @property
    def func_8mbgw4kh(self) -> Optional[List[int]]:
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
            return [items]
        except AttributeError:
            return None

    def func_8c0lfbv1(
        self,
        where: Optional[Union[List[Term], Term]] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> DataFrame:
        """read the block manager from the store"""
        self.validate_read(columns, where)
        selection = Selection(self, where=where, start=start, stop=stop)
        result = selection.select()
        frames: List[DataFrame] = []
        for a in self.axes:
            a.set_info(self.info)
            res = a.convert(result, nan_rep=self.nan_rep, encoding=self.encoding, errors=self.errors)
            frames.append(res)
        if len(frames) == 1:
            df = frames[0]
        else:
            df = concat(frames, axis=1)
        if len(frames) > 0:
            df = df.reindex(columns=self.axes[0].values)
        return df

    def func_tgxf1jr6(
        self,
        obj: DataFrame,
        **kwargs: Any,
    ) -> None:
        """write the block manager to the store"""
        super().write(obj, **kwargs)


class FrameFixed(BlockManagerFixed):
    """
    represent a frame fixed format object.
    """
    pandas_kind: str = 'frame'
    obj_type: type = DataFrame


class WORMTable(Table):
    """
    a write-once read-many table: this format DOES NOT ALLOW appending to a
    table. writing is a one-time operation the data are stored in a format
    that allows for searching the data on disk
    """
    table_type: str = 'worm'

    def func_8c0lfbv1(
        self,
        where: Optional[Union[List[Term], Term]] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> Any:
        """
        read the worm table from the store
        """
        raise NotImplementedError('WORMTable needs to implement read')

    def func_tgxf1jr6(
        self,
        obj: Any,
        **kwargs: Any,
    ) -> None:
        """
        write the worm table to the store
        """
        raise NotImplementedError('WORMTable needs to implement write')


class AppendableFrameTable(AppendableTable):
    """support the new appendable table formats"""
    pandas_kind: str = 'frame_table'
    table_type: str = 'appendable_frame'
    ndim: int = 2
    obj_type: type = DataFrame

    @property
    def func_pawazd2y(self) -> bool:
        return self.index_axes[0].axis == 1

    @classmethod
    def func_qyiagkk6(cls, obj: DataFrame, transposed: bool) -> DataFrame:
        """these are written transposed"""
        if transposed:
            obj = obj.T
        return obj

    def func_8c0lfbv1(
        self,
        where: Optional[Union[List[Term], Term]] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> DataFrame:
        """read the frame from the store"""
        self.validate_version(where)
        if not self.infer_axes():
            return None
        result = self._read_axes(where=where, start=start, stop=stop)
        info: Dict[str, Any] = self.info.get(self.non_index_axes[0][0], {}) if len(self.non_index_axes) else {}
        inds = [i for i, ax in enumerate(self.axes) if ax is self.index_axes[0]]
        assert len(inds) == 1
        ind = inds[0]
        index = result[ind][0]
        frames: List[DataFrame] = []
        for i, a in enumerate(self.axes):
            if a not in self.values_axes:
                continue
            index_vals, cvalues = result[i]
            if func_hcl4jnbg.get('type') != 'MultiIndex':
                cols = Index(index_vals)
            else:
                cols = MultiIndex.from_tuples(index_vals)
            names = func_hcl4jnbg.get('names')
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
                df = DataFrame(values.T, columns=cols_, index=index_, copy=False)
            elif isinstance(values, Index):
                df = DataFrame(values, columns=cols_, index=index_)
            else:
                df = DataFrame._from_arrays([values], columns=cols_, index=index_)
            if not (using_string_dtype() and values.dtype.kind == 'O'):
                assert (df.dtypes == values.dtype).all(), (df.dtypes, values.dtype)
            if using_string_dtype() and isinstance(values, np.ndarray) and is_string_array(values, skipna=True):
                df = df.astype(StringDtype(na_value=np.nan))
            frames.append(df)
        if len(frames) == 1:
            df = frames[0]
        else:
            df = concat(frames, axis=1)
        selection = Selection(self, where=where, start=start, stop=stop)
        df = self.process_axes(df, selection=selection, columns=columns)
        return df


class AppendableSeriesTable(AppendableFrameTable):
    """support the new appendable table formats"""
    pandas_kind: str = 'series_table'
    table_type: str = 'appendable_series'
    ndim: int = 2
    obj_type: type = Series

    @property
    def func_pawazd2y(self) -> bool:
        return False

    @classmethod
    def func_qyiagkk6(cls, obj: Series, transposed: bool) -> Series:
        """convert the series for writing"""
        return obj

    def func_tgxf1jr6(
        self,
        obj: Series,
        **kwargs: Any,
    ) -> None:
        """we are going to write this as a frame table"""
        if not isinstance(obj, DataFrame):
            name = obj.name or 'values'
            obj = obj.to_frame(name)
        super().write(obj=obj, data_columns=obj.columns.tolist(), **kwargs)

    def func_8c0lfbv1(
        self,
        where: Optional[Union[List[Term], Term]] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> Series:
        is_multi_index = self.is_multi_index
        if columns is not None and is_multi_index:
            assert isinstance(self.levels, list)
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)
        s = super().read(where=where, columns=columns, start=start, stop=stop)
        if is_multi_index:
            s.set_index(self.levels, inplace=True)
        s = s.iloc[:, 0]
        if s.name == 'values':
            s.name = None
        return s


class AppendableMultiSeriesTable(AppendableSeriesTable):
    """support the new appendable table formats"""
    pandas_kind: str = 'series_table'
    table_type: str = 'appendable_multiseries'
    ndim: int = 2
    obj_type: type = Series

    def func_tgxf1jr6(
        self,
        obj: Series,
        **kwargs: Any,
    ) -> None:
        """we are going to write this as a frame table"""
        name = obj.name or 'values'
        newobj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)
        cols = list(self.levels)
        cols.append(name)
        newobj.columns = Index(cols)
        super().write(obj=newobj, **kwargs)


class GenericTable(AppendableFrameTable):
    """a table that read/writes the generic pytables table format"""
    pandas_kind: str = 'frame_table'
    table_type: str = 'generic_table'
    ndim: int = 2
    obj_type: type = DataFrame

    @property
    def func_vgmxe3i9(self) -> str:
        return self.pandas_kind

    @property
    def func_9gr0vj9t(self) -> Any:
        return getattr(self.group, 'table', None) or self.group

    def func_gr2vbxnc(self) -> None:
        """retrieve our attributes"""
        self.non_index_axes = []
        self.nan_rep = None
        self.levels = []
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]
        self.data_columns = [a.name for a in self.values_axes]

    @cache_readonly
    def func_elmkvxwn(self) -> List[IndexCol]:
        """create/cache the indexables if they don't exist"""
        d = self.description
        md = self.read_metadata('index')
        meta = 'category' if md is not None else None
        index_col = IndexCol(
            name='index',
            axis=0,
            table=self.table,
            meta=meta,
            metadata=md,
        )
        _indexables: List[IndexCol] = [index_col]
        for i, n in enumerate(d._v_names):
            assert isinstance(n, str)
            atom = getattr(d, n)
            md = self.read_metadata(n)
            meta = 'category' if md is not None else None
            dc = DataIndexableCol(
                name=n,
                cname=n,
                values=[n],
                kind=_dtype_to_kind(str(atom.dtype.kind)),
                pos=i,
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
                dtype=str(atom.dtype),
                data=None,
            )
            _indexables.append(dc)
        return _indexables

    def func_tgxf1jr6(
        self,
        **kwargs: Any,
    ) -> None:
        """cannot write on a generic table"""
        raise NotImplementedError('cannot write on an generic table')

    def read(self) -> Any:
        """cannot read on a generic table"""
        raise NotImplementedError('GenericTable cannot implement read')
