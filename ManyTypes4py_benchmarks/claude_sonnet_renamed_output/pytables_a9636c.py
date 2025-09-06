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
    TYPE_CHECKING, Any, Callable, Dict, Final, Hashable, Iterator, List, Optional,
    Sequence, Tuple, Type, Union, overload
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
    AttributeConflictWarning, ClosedFileError, IncompatibilityWarning,
    PerformanceWarning, PossibleDataLossError
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
    ensure_object, is_bool_dtype, is_complex_dtype, is_list_like,
    is_string_dtype, needs_i8_conversion
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype, DatetimeTZDtype, ExtensionDtype, PeriodDtype
)
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
    DataFrame, DatetimeIndex, Index, MultiIndex, PeriodIndex, RangeIndex,
    Series, StringDtype, TimedeltaIndex, concat, isna
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
    from pandas._typing import AnyArrayLike, ArrayLike, AxisInt, DtypeArg, FilePath, Self, Shape
    import numpy.typing as npt

_version: Final[str] = '0.15.2'
_default_encoding: Final[str] = 'UTF-8'


def func_j7cb0xfa(encoding: Optional[str]) -> str:
    if encoding is None:
        encoding = _default_encoding
    return encoding


def func_6npcmgwl(name: Any) -> str:
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype. Non-string dtypes are passed through unchanged.

    https://github.com/pandas-dev/pandas/issues/13492
    """
    if isinstance(name, str):
        name = str(name)
    return name


Term = PyTablesExpr


def func_qkg3n79k(where: Optional[Union[Term, List[Term]]], scope_level: int) -> Optional[Union[Term, List[Term]]]:
    """
    Ensure that the where is a Term or a list of Term.

    This makes sure that we are capturing the scope of variables that are
    passed create the terms here with a frame_level=2 (we are 2 levels down)
    """
    level = scope_level + 1
    if isinstance(where, (list, tuple)):
        where = [
            (Term(term, scope_level=level + 1) if maybe_expression(term) else term)
            for term in where if term is not None
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
_AXES_MAP: Final[Dict[Type[DataFrame], List[int]]] = {DataFrame: [0]}
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
        'dropna_table', False, dropna_doc, validator=config.is_bool
    )
    config.register_option(
        'default_format', None, format_doc, validator=config.is_one_of_factory(['fixed', 'table', None])
    )
_table_mod: Optional[Any] = None
_table_file_open_policy_is_strict: bool = False


def func_glsq77za() -> Any:
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables
        _table_mod = tables
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (tables.file._FILE_OPEN_POLICY == 'strict')
    return _table_mod


def func_fgciumbb(path_or_buf: Union[str, "HDFStore"], key: Any, value: Any, mode: str = 'a', complevel: Optional[int] = None,
                complib: Optional[str] = None, append: bool = False, format: Optional[str] = None,
                index: bool = True, min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
                nan_rep: Optional[str] = None, data_columns: Optional[Union[List[str], bool]] = None,
                errors: str = 'strict', encoding: str = 'UTF-8') -> None:
    """store this object, close it if we opened it"""
    if append:
        f: Callable[[Any], Any] = lambda store: store.append(
            key, value, format=format, index=index, min_itemsize=min_itemsize,
            nan_rep=nan_rep, dropna=dropna, data_columns=data_columns,
            errors=errors, encoding=encoding
        )
    else:
        f = lambda store: store.put(
            key, value, format=format, index=index, min_itemsize=min_itemsize,
            nan_rep=nan_rep, data_columns=data_columns, errors=errors,
            encoding=encoding, dropna=dropna
        )
    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        path_or_buf = stringify_path(path_or_buf)
        with HDFStore(path_or_buf, mode=mode, complevel=complevel, complib=complib) as store:
            f(store)


def func_7y40qs0l(path_or_buf: Union[str, "HDFStore"], key: Optional[Any] = None, mode: str = 'r', errors: str = 'strict',
                where: Optional[Union[List[Term], Term]] = None, start: Optional[int] = None,
                stop: Optional[int] = None, columns: Optional[List[str]] = None,
                iterator: bool = False, chunksize: Optional[int] = None, **kwargs) -> Any:
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
    if mode not in {'r', 'r+', 'a'}:
        raise ValueError(
            f'mode {mode} is not allowed while performing a read. Allowed modes are r, r+ and a.'
        )
    if where is not None:
        where = func_qkg3n79k(where, scope_level=1)
    if isinstance(path_or_buf, HDFStore):
        if not path_or_buf.is_open:
            raise OSError('The HDFStore must be open for reading.')
        store: HDFStore = path_or_buf
        auto_close: bool = False
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
            key, where=where, start=start, stop=stop,
            columns=columns, iterator=iterator, chunksize=chunksize,
            auto_close=auto_close
        )
    except (ValueError, TypeError, LookupError):
        if not isinstance(path_or_buf, HDFStore):
            with suppress(AttributeError):
                store.close()
        raise


def func_fwuz3pa6(group: str, parent_group: Any) -> Any:
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

    def __init__(self, path: str, mode: str = 'a', complevel: Optional[int] = None,
                 complib: Optional[str] = None, fletcher32: bool = False,
                 **kwargs: Any) -> None:
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
    def func_725geno4(self) -> Any:
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None
        return self._handle.root

    @property
    def func_nzhyyr4n(self) -> str:
        return self._path

    def __getitem__(self, key: Any) -> Any:
        return self.get(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: Any) -> None:
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

    def __contains__(self, key: Any) -> bool:
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

    def __enter__(self) -> "HDFStore":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.close()

    def func_p58vk3vn(self, include: str = 'pandas') -> List[str]:
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
            return [n._v_pathname for n in self._handle.walk_nodes('/', classname='Table')]
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def func_7matvc4j(self) -> Iterator[Tuple[str, Any]]:
        """
        iterate on key->group
        """
        for g in self.groups():
            yield g._v_pathname, g

    def open(self, mode: str = 'a', **kwargs: Any) -> None:
        """
        Open the file in the specified mode

        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
        tables = func_glsq77za()
        if self._mode != mode:
            if self._mode in {'a', 'w'} and mode in {'r', 'r+'}:
                pass
            elif mode in {'w'}:
                if self.is_open:
                    raise PossibleDataLossError(
                        f'Re-opening the file [{self._path}] with mode [{self._mode}] will delete the current file!'
                    )
            self._mode = mode
        if self.is_open:
            self.close()
        if self._complevel and self._complevel > 0:
            self._filters = tables.Filters(
                self._complevel, self._complib, fletcher32=self._fletcher32
            )
        if _table_file_open_policy_is_strict and self.is_open:
            msg = (
                'Cannot open HDF5 file, which is already opened, even in read-only mode.'
            )
            raise ValueError(msg)
        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def func_7x2z67q5(self) -> None:
        """
        Close the PyTables file handle
        """
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def func_rmykwryd(self) -> bool:
        """
        return a boolean indicating whether the file is open
        """
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def func_cqzoy3n4(self, fsync: bool = False) -> None:
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

    def func_01vumk9r(self, key: Any) -> Any:
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

    def func_xufwq3f7(
        self, key: Any, where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None, stop: Optional[int] = None,
        columns: Optional[List[str]] = None, iterator: bool = False,
        chunksize: Optional[int] = None, auto_close: bool = False
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
        start : int or None
            Row number to start selection.
        stop : int, default None
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
        where = func_qkg3n79k(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        def func_e0clc2pl(_start: Optional[int], _stop: Optional[int], _where: Any) -> Any:
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        it = TableIterator(
            self, s, func, where=where, nrows=s.nrows,
            start=start, stop=stop, iterator=iterator,
            chunksize=chunksize, auto_close=auto_close
        )
        return it.get_result()

    def func_w19ft2e1(
        self, key: str, where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None, stop: Optional[int] = None
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
        where = func_qkg3n79k(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_coordinates with a table')
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def func_fhijx58a(
        self, key: str, column: str, start: Optional[int] = None,
        stop: Optional[int] = None
    ) -> Any:
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

    def func_xdlnzicv(
        self, keys: Union[List[str], Tuple[str, ...]], where: Optional[Union[List[Term], Term]] = None,
        selector: Optional[str] = None, columns: Optional[List[str]] = None,
        start: Optional[int] = None, stop: Optional[int] = None,
        iterator: bool = False, chunksize: Optional[int] = None,
        auto_close: bool = False
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
        where = func_qkg3n79k(where, scope_level=1)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(
                key=keys, where=where, columns=columns,
                start=start, stop=stop, iterator=iterator,
                chunksize=chunksize, auto_close=auto_close
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
        axis = {t.non_index_axes[0][0] for t in _tbls}.pop()

        def func_e0clc2pl(_start: Optional[int], _stop: Optional[int], _where: Any) -> DataFrame:
            objs = [t.read(where=_where, columns=columns, start=_start, stop=_stop) for t in tbls]
            return concat(objs, axis=axis, verify_integrity=False)._consolidate()

        it = TableIterator(
            self, s, func, where=where, nrows=nrows,
            start=start, stop=stop, iterator=iterator,
            chunksize=chunksize, auto_close=auto_close
        )
        return it.get_result(coordinates=True)

    def func_la2v1zd1(
        self, key: str, value: Union[Series, DataFrame],
        format: Optional[str] = None, axes: Optional[List[int]] = None,
        index: bool = True, append: bool = False, complib: Optional[str] = None,
        complevel: Optional[int] = None, min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep: Optional[str] = None, data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None, errors: str = 'strict',
        track_times: bool = True, dropna: bool = False
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
            key, value, format=format, index=index, append=append,
            complib=complib, complevel=complevel, min_itemsize=min_itemsize,
            nan_rep=nan_rep, data_columns=data_columns, encoding=encoding,
            errors=errors, track_times=track_times, dropna=dropna
        )

    def func_p2yix7g2(
        self, key: str, where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None, stop: Optional[int] = None
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
        where = func_qkg3n79k(where, scope_level=1)
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

    def func_ahag9i4l(
        self, key: str, value: Union[Series, DataFrame],
        format: Optional[str] = None, axes: Optional[List[int]] = None,
        index: bool = True, append: bool = True, complib: Optional[str] = None,
        complevel: Optional[int] = None, columns: Optional[List[str]] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep: Optional[str] = None, chunksize: Optional[int] = None,
        expectedrows: Optional[int] = None, dropna: bool = False,
        data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None, errors: str = 'strict'
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
        columns : default None
            This parameter is currently not accepted, try data_columns.
        min_itemsize : Optional[Union[int, Dict[str, int]]]
            Dict of columns that specify minimum str sizes.
        nan_rep : Optional[str]
            Str to use as str nan representation.
        chunksize : Optional[int]
            Size to chunk the writing.
        expectedrows : int
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
            key, value, format=format, axes=axes, index=index, append=append,
            complib=complib, complevel=complevel, min_itemsize=min_itemsize,
            nan_rep=nan_rep, chunksize=chunksize, expectedrows=expectedrows,
            dropna=dropna, data_columns=data_columns, encoding=encoding,
            errors=errors
        )

    def func_tx9jg86w(
        self, d: Dict[str, Optional[List[str]]],
        value: DataFrame, selector: str,
        data_columns: Optional[Union[List[str], bool]] = None,
        axes: Optional[List[int]] = None, dropna: bool = False,
        **kwargs: Any
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
        data_columns : Optional[Union[List[str], bool]], default None
            List of columns to create as data columns, or True to use all columns
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
        min_itemsize = kwargs.pop('min_itemsize', None)
        for k, v in d.items():
            dc = data_columns if k == selector else None
            val = value.reindex(v, axis=axis)
            filtered = {key: value for key, value in min_itemsize.items() if
                       key in v} if min_itemsize is not None else None
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)

    def func_iwsjwg6s(
        self, key: str, columns: Optional[Union[bool, List[str]]] = None,
        optlevel: Optional[int] = None, kind: Optional[str] = None
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
        func_glsq77za()
        s = self.get_storer(key)
        if s is None:
            return
        if not isinstance(s, Table):
            raise TypeError('cannot create table index on a Fixed format store')
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def func_f2t3b304(self) -> List[Any]:
        """
        Return a list of all the top-level nodes.

        Each node returned is not a pandas storage object.

        Returns
        -------
        list
            List of objects.

        See Also
        --------
        HDFStore.get_node : Returns the node with a key.

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
        func_glsq77za()
        self._check_if_open()
        assert self._handle is not None
        assert _table_mod is not None
        return [
            g for g in self._handle.walk_groups()
            if not isinstance(g, _table_mod.link.Link) and (
                getattr(g._v_attrs, 'pandas_type', None) or
                getattr(g, 'table', None) or
                (isinstance(g, _table_mod.table.Table) and g._v_name != 'table')
            )
        ]

    def func_0a4nq6tc(self, where: str = '/') -> Iterator[Tuple[str, List[str], List[str]]]:
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
        >>> store.put("data", df1, format="table")  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["A", "B"])
        >>> store.append("data", df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        >>> for group in store.walk():  # doctest: +SKIP
        ...     print(group)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        func_glsq77za()
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

    def func_jnyi4kd1(self, key: str) -> Optional[Any]:
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

    def func_sf8ijrk2(self, key: str) -> Any:
        """return the storer object for a key, raise if not in the file"""
        group = self.get_node(key)
        if group is None:
            raise KeyError(f'No object named {key} in the file')
        s = self._create_storer(group)
        s.infer_axes()
        return s

    def func_knn0xpsp(
        self, file: str, mode: str = 'w', propindexes: bool = True,
        keys: Optional[List[str]] = None, complib: Optional[str] = None,
        complevel: Optional[int] = None, fletcher32: bool = False,
        overwrite: bool = True
    ) -> "HDFStore":
        """
        Copy the existing store to a new file, updating in place.

        Parameters
        ----------
        propindexes : bool, default True
            Restore indexes in copied file.
        keys : list, optional
            List of keys to include in the copy (defaults to all).
        overwrite : bool, default True
            Whether to overwrite (remove and replace) existing nodes in the new store.
        mode, complib, complevel, fletcher32 same as in HDFStore.__init__

        Returns
        -------
        open file handle of the new store
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
                    index: Union[bool, List[str]] = False
                    if propindexes:
                        index = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(k, data, index=index, data_columns=getattr(s, 'data_columns', None),
                                     encoding=s.encoding)
                else:
                    new_store.put(k, data, encoding=s.encoding)
        return new_store

    def func_q31k3j0l(self) -> Dict[str, Any]:
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
                values: List[Any] = []
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
                        values.append(f'[invalid_HDFStore node: {pprint_thing(detail)}]')
                output += adjoin(12, keys, values)
            else:
                output += 'Empty'
        else:
            output += 'File is CLOSED'
        return output

    def func_porw7oai(self) -> None:
        if not self.is_open:
            raise ClosedFileError(f'{self._path} file is not open!')

    def func_4m9o9lif(self, format: str) -> str:
        """validate / deprecate formats"""
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f'invalid HDFStore format specified [{format}]') from err
        return format

    def func_4f0mooxy(
        self, group: Any, format: Optional[str] = None,
        value: Optional[Union[Series, DataFrame]] = None,
        encoding: str = 'UTF-8', errors: str = 'strict'
    ) -> Any:
        """return a suitable class to operate"""
        if value is not None and not isinstance(value, (Series, DataFrame)):
            raise TypeError('value must be None, Series, or DataFrame')
        pt = getattr(group._v_attrs, 'pandas_type', None)
        tt = getattr(group._v_attrs, 'table_type', None)
        if pt is None:
            if value is None:
                func_glsq77za()
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
            _STORER_MAP: Dict[str, Type[Fixed]] = {'series': SeriesFixed, 'frame': FrameFixed}
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
        _TABLE_MAP: Dict[str, Type[Table]] = {
            'generic_table': GenericTable,
            'appendable_series': AppendableSeriesTable,
            'appendable_multiseries': AppendableMultiSeriesTable,
            'appendable_frame': AppendableFrameTable,
            'appendable_multiframe': AppendableMultiFrameTable,
            'worm': WORMTable
        }
        try:
            cls = _TABLE_MAP[tt]
        except KeyError as err:
            raise TypeError(
                f'cannot properly create the storer for: [_TABLE_MAP] [group->{group},value->{type(value)},format->{format}'
            ) from err
        return cls(self, group, encoding=encoding, errors=errors)

    def func_ioo3epxi(
        self, key: str, value: Union[Series, DataFrame],
        format: Optional[str], axes: Optional[List[int]] = None,
        index: bool = True, append: bool = False, complib: Optional[str] = None,
        complevel: Optional[int] = None, fletcher32: Optional[bool] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        chunksize: Optional[int] = None, expectedrows: Optional[int] = None,
        dropna: bool = False, nan_rep: Optional[str] = None,
        data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None, errors: str = 'strict',
        track_times: bool = True
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
        s.write(obj=value, axes=axes, append=append, complib=complib,
                complevel=complevel, fletcher32=fletcher32, min_itemsize=min_itemsize,
                chunksize=chunksize, expectedrows=expectedrows, dropna=dropna,
                nan_rep=nan_rep, data_columns=data_columns, track_times=track_times)
        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def func_0bedw5kn(self, group: Any) -> Any:
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def func_95gbb1ca(
        self, key: str, append: bool
    ) -> Any:
        """Identify HDF5 group based on key, delete/create group if needed."""
        group = self.get_node(key)
        assert self._handle is not None
        if group is not None and not append:
            self._handle.remove_node(group, recursive=True)
            group = None
        if group is None:
            group = self._create_nodes_and_group(key)
        return group

    def func_eevwow0i(
        self, key: str
    ) -> Any:
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

    def func_iwsjwg6s(
        self, key: str, columns: Optional[Union[bool, List[str]]] = None,
        optlevel: Optional[int] = None, kind: Optional[str] = None
    ) -> None:
        """
        Create a pytables index on the specified columns.

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
        TypeError if trying to create an index on a complex-type column.

        Notes
        -----
        Cannot index Time64Col or ComplexCol.
        Pytables must be >= 3.0.
        """
        # This method is duplicated; already defined above as func_iwsjwg6s
        pass

    def func_cawsuv9p(self) -> None:
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)

    def func_zcf8n1m3(self) -> Any:
        """return my current col description"""
        return getattr(self.description, self.cname, None)

    def _check_if_open(self) -> None:
        if not self.is_open:
            raise ClosedFileError(f'File {self._path} is not open')

    def is_open(self) -> bool:
        return self.func_rmykwryd

    def groups(self) -> List[Any]:
        """
        Return a list of groups. Placeholder for actual implementation.
        """
        pass

    def get(self, key: Any) -> Any:
        """Placeholder for actual get implementation"""
        pass

    def put(
        self, key: str, value: Union[Series, DataFrame],
        format: Optional[str] = None, axes: Optional[List[int]] = None,
        index: bool = True, append: bool = False, complib: Optional[str] = None,
        complevel: Optional[int] = None, min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep: Optional[str] = None, data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None, errors: str = 'strict',
        track_times: bool = True, dropna: bool = False
    ) -> None:
        """Placeholder for actual put implementation"""
        pass

    def remove(self, key: Any) -> None:
        """Placeholder for actual remove implementation"""
        pass

    def select(
        self, key: Any, where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None, stop: Optional[int] = None,
        columns: Optional[List[str]] = None, iterator: bool = False,
        chunksize: Optional[int] = None, auto_close: bool = False
    ) -> Any:
        """Placeholder for actual select implementation"""
        pass

    def groups(self) -> List[Any]:
        """Placeholder method, should return list of groups"""
        pass

    def get_node(self, key: str) -> Optional[Node]:
        """Placeholder for get_node implementation"""
        pass

    def _create_storer(self, group: Any) -> Table:
        """Placeholder for create_storer implementation"""
        pass

    def _read_group(self, group: Any) -> Any:
        """Placeholder for _read_group implementation"""
        pass

    def _validate_format(self, format: str) -> str:
        return self.func_4m9o9lif(format)

    def _write_to_group(
        self, key: str, value: Union[Series, DataFrame],
        format: str, axes: Optional[List[int]] = None, index: bool = True,
        append: bool = False, complib: Optional[str] = None,
        complevel: Optional[int] = None, min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep: Optional[str] = None, data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None, errors: str = 'strict',
        track_times: bool = True, dropna: bool = False
    ) -> None:
        """Placeholder for _write_to_group implementation"""
        pass

    def get_storer(self, key: str) -> Optional[Table]:
        """Placeholder for get_storer implementation"""
        pass


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
    chunksize : Optional[int]
        the passed chunking value (default is 100000)
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """

    def __init__(
        self, store: HDFStore, s: Table,
        func: Callable[[Optional[Any], Optional[Any], Any], Any],
        where: Optional[Union[List[Term], Term]],
        nrows: int, start: Optional[int] = None,
        stop: Optional[int] = None, iterator: bool = False,
        chunksize: Optional[int] = None, auto_close: bool = False
    ) -> None:
        self.store: HDFStore = store
        self.s: Table = s
        self.func: Callable[[Optional[Any], Optional[Any], Any], Any] = func
        self.where: Optional[Union[List[Term], Term]] = where
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)
        self.nrows: int = nrows
        self.start: Optional[int] = start
        self.stop: Optional[int] = stop
        self.coordinates: Optional[Index] = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize: Optional[int] = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close: bool = auto_close

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

    def func_7x2z67q5(self) -> None:
        if self.auto_close:
            self.store.close()

    def func_0ynsariy(self, coordinates: bool = False) -> Any:
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
        self.func_7x2z67q5()


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
        self, name: str, values: Optional[np.ndarray] = None,
        kind: Optional[str] = None, typ: Optional[Any] = None,
        cname: Optional[str] = None, axis: Optional[int] = None,
        pos: Optional[int] = None, freq: Optional[Any] = None,
        tz: Optional[tzinfo] = None, index_name: Optional[str] = None,
        ordered: Optional[bool] = None, table: Optional[Any] = None,
        meta: Optional[str] = None, metadata: Optional[Any] = None
    ) -> None:
        if not isinstance(name, str):
            raise ValueError('`name` must be a str.')
        self.values: Optional[np.ndarray] = values
        self.kind: Optional[str] = kind
        self.typ: Optional[Any] = typ
        self.name: str = name
        self.cname: str = cname or name
        self.axis: Optional[int] = axis
        self.pos: Optional[int] = pos
        self.freq: Optional[Any] = freq
        self.tz: Optional[tzinfo] = tz
        self.index_name: Optional[str] = index_name
        self.ordered: Optional[bool] = ordered
        self.table: Optional[Any] = table
        self.meta: Optional[str] = meta
        self.metadata: Optional[Any] = metadata
        if pos is not None:
            self.set_pos(pos)
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def func_wfxcyp1y(self) -> int:
        return self.typ.itemsize if self.typ is not None else 0

    @property
    def func_38evnq6b(self) -> str:
        return f'{self.name}_kind'

    def func_axqnqk6s(self, pos: int) -> None:
        """set the position of this column in the Table"""
        self.pos = pos
        if self.typ is not None:
            self.typ._v_pos = pos

    def __repr__(self) -> str:
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.axis, self.pos, self.kind)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name', 'cname', 'axis', 'pos', 'kind'], temp)])

    def __eq__(self, other: Any) -> bool:
        """compare 2 col items"""
        if not isinstance(other, IndexCol):
            return NotImplemented
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ['name', 'cname', 'axis', 'pos']
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def func_7nrpmqrt(self) -> bool:
        """return whether I am an indexed column"""
        if not hasattr(self.table, 'cols'):
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def func_vs7kh4d7(
        self, values: np.ndarray, nan_rep: Optional[str],
        encoding: str, errors: str
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
        factory = Index
        if lib.is_np_dtype(values.dtype, 'M') or isinstance(values.dtype, DatetimeTZDtype):
            factory = DatetimeIndex
        elif isinstance(values.dtype, type) and isinstance(values.dtype, np.dtype) and values.dtype.kind == 'i' and len(kwargs.get('freq', '')):
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

    def func_ai2db2br(self) -> Optional[np.ndarray]:
        """return the values"""
        return self.values

    @property
    def func_qnpsejis(self) -> Any:
        return self.table._v_attrs

    @property
    def func_ee1r0rlf(self) -> Any:
        return self.table.description

    @property
    def func_zcf8n1m3(self) -> Optional[Any]:
        """return my current col description"""
        return getattr(self.description, self.cname, None)

    @property
    def func_nh4orhbm(self) -> Optional[np.ndarray]:
        """return my cython values"""
        return self.values

    def __iter__(self) -> Iterator[Any]:
        return iter(self.values)  # type: ignore

    def func_3awjy64u(self, min_itemsize: Optional[Union[int, Dict[str, int]]] = None) -> None:
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
        if self.kind == 'string':
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)
            if min_itemsize is not None and self.typ.itemsize < min_itemsize:
                self.typ = func_glsq77za().StringCol(itemsize=min_itemsize, pos=self.pos)

    def func_ua5cjx5w(self) -> None:
        pass

    def func_e7sj2gee(
        self, handler: Any, append: bool
    ) -> None:
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def func_mnomiqay(
        self, itemsize: Optional[int] = None
    ) -> Optional[int]:
        """
        validate this column: return the compared against itemsize
        """
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

    def func_2zwvbzc9(self, append: bool) -> None:
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(
                    f'incompatible kind in col [{existing_kind} - {self.kind}]'
                )

    def func_by8su2y8(self, info: Dict[str, Any]) -> None:
        """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
        for key in self._info_fields:
            value = getattr(self, key, None)
            idx = func_q31k3j0l.setdefault(self.name, {})
            existing_value = idx.get(key)
            if key in idx and value is not None and existing_value != value:
                if key in {'freq', 'index_name'}:
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

    def func_wnwavq9f(self, info: Dict[str, Any]) -> None:
        """set my state from the passed info"""
        idx = func_q31k3j0l.get(self.name)
        if idx is not None:
            self.__dict__.update(idx)

    def func_cawsuv9p(self) -> None:
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)

    def func_g1ib0z6c(
        self, handler: Any
    ) -> None:
        """validate that kind=category does not change the categories"""
        if self.meta == 'category':
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if (new_metadata is not None and cur_metadata is not None and
                not array_equivalent(new_metadata, cur_metadata, strict_nan=True, dtype_equal=True)):
                raise ValueError(
                    'cannot append a categorical with different categories to the existing'
                )

    def func_ttz1kgvr(self, handler: Any) -> None:
        """set the meta data"""
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)

    def validate_col(self) -> None:
        """Placeholder for validate_col implementation"""
        pass

    def validate_attr(self, append: bool) -> None:
        """Placeholder for validate_attr implementation"""
        pass

    def validate_metadata(self, handler: Any) -> None:
        """Placeholder for validate_metadata implementation"""
        pass

    def write_metadata(self, handler: Any) -> None:
        """Placeholder for write_metadata implementation"""
        pass

    def set_attr(self) -> None:
        """Placeholder for set_attr implementation"""
        pass


class GenericIndexCol(IndexCol):
    """an index which is not represented in the data of the table"""

    @property
    def func_7nrpmqrt(self) -> bool:
        return False

    def func_vs7kh4d7(
        self, values: np.ndarray, nan_rep: Optional[str],
        encoding: str, errors: str
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
        index : listlike to become an Index
        data : ndarraylike to become a column
        """
        assert isinstance(values, np.ndarray), type(values)
        index = RangeIndex(len(values))
        return index, index

    def func_cawsuv9p(self) -> None:
        pass


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
        self, name: str, values: Optional[np.ndarray] = None,
        kind: Optional[str] = None, typ: Optional[Any] = None,
        cname: Optional[str] = None, pos: Optional[int] = None,
        tz: Optional[tzinfo] = None, ordered: Optional[bool] = None,
        table: Optional[Any] = None, meta: Optional[str] = None,
        metadata: Optional[Any] = None, dtype: Optional[str] = None,
        data: Optional[np.ndarray] = None
    ) -> None:
        super().__init__(
            name=name, values=values, kind=kind, typ=typ, cname=cname, pos=pos,
            tz=tz, ordered=ordered, table=table, meta=meta, metadata=metadata
        )
        self.dtype: Optional[str] = dtype
        self.data: Optional[np.ndarray] = data

    @property
    def func_ih8e0mf9(self) -> str:
        return f'{self.name}_dtype'

    @property
    def func_6az1xgy3(self) -> str:
        return f'{self.name}_meta'

    def __repr__(self) -> str:
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.dtype, self.kind, self.shape)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name', 'cname', 'dtype', 'kind', 'shape'], temp)])

    def __eq__(self, other: Any) -> bool:
        """compare 2 col items"""
        if not isinstance(other, DataCol):
            return NotImplemented
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ['name', 'cname', 'dtype', 'pos']
        )

    def func_7gmptw6v(self, data: Any) -> None:
        assert data is not None
        assert self.dtype is None
        data, dtype_name = _get_data_and_dtype_name(data)
        self.data = data
        self.dtype = dtype_name
        self.kind = _dtype_to_kind(dtype_name)

    def func_ai2db2br(self) -> Optional[np.ndarray]:
        """return the data"""
        return self.data

    @classmethod
    def func_h05b3une(cls, values: np.ndarray) -> Any:
        """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
        dtype = values.dtype
        itemsize = dtype.itemsize
        shape = values.shape
        if values.ndim == 1:
            shape = (1, values.size)
        if isinstance(values, Categorical):
            codes = values.codes
            atom = cls.get_atom_data(shape, kind=codes.dtype.name)
        elif lib.is_np_dtype(dtype, 'M') or isinstance(dtype, DatetimeTZDtype):
            atom = cls.get_atom_datetime64(shape)
        elif lib.is_np_dtype(dtype, 'm'):
            atom = cls.get_atom_timedelta64(shape)
        elif is_complex_dtype(dtype):
            atom = func_glsq77za().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif is_string_dtype(dtype):
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)
        return atom

    @classmethod
    def func_xh1fnifs(cls, shape: Tuple[int, ...], itemsize: int) -> Any:
        return func_glsq77za().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def func_grw4pba5(cls, kind: str) -> Any:
        """return the PyTables column class for this column"""
        if kind.startswith('uint'):
            k4 = kind[4:]
            col_name = f'UInt{k4}Col'
        elif kind.startswith('period'):
            col_name = 'Int64Col'
        else:
            kcap = kind.capitalize()
            col_name = f'{kcap}Col'
        return getattr(func_glsq77za(), col_name)

    @classmethod
    def func_zz9p81hr(cls, shape: Tuple[int, ...], kind: str) -> Any:
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def func_nfreqdvb(cls, shape: Tuple[int, ...]) -> Any:
        return func_glsq77za().Int64Col(shape=shape[0])

    @classmethod
    def func_4z4weuss(cls, shape: Tuple[int, ...]) -> Any:
        return func_glsq77za().Int64Col(shape=shape[0])

    @property
    def func_g1e1unjr(self) -> Optional[Tuple[int, ...]]:
        """return my cython values"""
        return getattr(self.data, 'shape', None)

    @property
    def func_nh4orhbm(self) -> Optional[np.ndarray]:
        """return my cython values"""
        return self.data

    @classmethod
    def func_48qowxm4(cls, obj: DataFrame, table_exists: bool,
                      new_non_index_axes: List[Tuple[int, List[str]]],
                      values_axes: List[DataCol], data_columns: List[str]
                      ) -> Tuple[List[Any], List[List[str]]]:
        """
        take the input data_columns and min_itemize and create a data
        columns spec
        """
        mgr = obj._mgr

        def func_8zs08dlx(mgr: Any) -> List[np.ndarray]:
            return [mgr.items.take(blk.mgr_locs) for blk in mgr.blocks]

        blocks = list(mgr.blocks)
        blk_items = func_8zs08dlx(mgr)
        if len(data_columns):
            axis, axis_labels = new_non_index_axes[0]
            new_labels = Index(axis_labels).difference(Index(data_columns))
            mgr = obj.reindex(new_labels, axis=axis)._mgr
            blocks = list(mgr.blocks)
            blk_items = func_8zs08dlx(mgr)
            for c in data_columns:
                mgr = obj.reindex([c], axis=axis)._mgr
                blocks.extend(mgr.blocks)
                blk_items.extend(func_8zs08dlx(mgr))
        if table_exists:
            by_items = {tuple(b_items.tolist()): (b, b_items) for b, b_items in zip(blocks, blk_items)}
            new_blocks: List[Any] = []
            new_blk_items: List[List[str]] = []
            for ea in values_axes:
                items = tuple(ea.values)
                try:
                    b, b_items = by_items.pop(items)
                    new_blocks.append(b)
                    new_blk_items.append(b_items)
                except (IndexError, KeyError) as err:
                    jitems = ','.join([pprint_thing(item) for item in items])
                    raise ValueError(
                        f'cannot match existing table structure for [{jitems}] on appending data'
                    ) from err
            blocks = new_blocks
            blk_items = new_blk_items
        return blocks, blk_items

    def func_e0clc2pl(
        self, _start: Optional[int], _stop: Optional[int], _where: Any
    ) -> Any:
        return self.s.read(start=_start, stop=_stop, where=_where, columns=self.columns)

    def func_boemdmu7(
        self, key: str, append: bool
    ) -> Any:
        """write a 0-len array"""
        arr = np.empty((1,) * self.value.ndim, dtype=self.value.dtype)
        self._handle.create_array(self.group, key, arr)
        node = getattr(self.group, key)
        node._v_attrs.value_type = str(self.value.dtype)
        node._v_attrs.shape = self.value.shape

    def func_RD1CS57W(self, key: str, index: Any) -> Any:
        """
        Write out a metadata array to the key as a fixed-format Series.

        Parameters
        ----------
        key : str
        values : ndarray
        """
        self.parent.put(self._get_metadata_path(key), Series(index, copy=False), format='table',
                       encoding=self.encoding, errors=self.errors, nan_rep=self.nan_rep)

    def func_s5ulwoqp(self, key: str) -> Optional[Series]:
        """return the meta data array for this key"""
        if getattr(getattr(self.group, 'meta', None), key, None) is not None:
            return self.parent.select(self._get_metadata_path(key))
        return None

    def validate_read(
        self, columns: Optional[List[str]], where: Optional[Union[List[Term], Term]]
    ) -> None:
        if columns is not None:
            raise TypeError(
                'cannot pass a column specification when reading a Fixed format store. this store must be selected in its entirety'
            )
        if where is not None:
            raise TypeError(
                'cannot pass a where specification when reading from a Fixed format store. this store must be selected in its entirety'
            )

    def read_index_node(
        self, node: Any, start: Optional[int] = None, stop: Optional[int] = None
    ) -> Index:
        data = node[start:stop]
        if 'shape' in node._v_attrs and np.prod(node._v_attrs.shape) == 0:
            data = np.empty(node._v_attrs.shape, dtype=node._v_attrs.value_type)
        kind = node._v_attrs.kind
        name = None
        if 'name' in node._v_attrs:
            name = func_6npcmgwl(node._v_attrs.name)
        attrs = node._v_attrs
        factory, kwargs = self._get_index_factory(attrs)
        if kind in {'date', 'object'}:
            index = factory(_unconvert_index(data, kind, encoding=self.encoding, errors=self.errors),
                            dtype=object, **kwargs)
        else:
            index = factory(_unconvert_index(data, kind, encoding=self.encoding, errors=self.errors), **kwargs)
        index.name = name
        return index

    def _get_metadata_path(self, key: str) -> str:
        group = self.group._v_pathname
        return f'{group}/meta/{key}/meta'

    def create_description(self, complib: Optional[str], complevel: Optional[int],
                           fletcher32: Optional[bool], expectedrows: Optional[int]) -> Dict[str, Any]:
        if expectedrows is None:
            expectedrows = max(self.nrows_expected, 10000)
        d: Dict[str, Any] = {'name': 'table', 'expectedrows': expectedrows}
        d['description'] = {a.cname: a.typ for a in self.axes}
        if complib:
            if complevel is None:
                complevel = self._complevel or 9
            filters = func_glsq77za().Filters(complevel=complevel, complib=complib, fletcher32=fletcher32 or self._fletcher32)
            d['filters'] = filters
        elif self._filters is not None:
            d['filters'] = self._filters
        d['track_times'] = True
        return d

    def remove_rows(
        self, start: int, stop: int
    ) -> int:
        """Placeholder for actual remove_rows implementation"""
        pass

    def read_index(
        self, key: str, start: Optional[int] = None, stop: Optional[int] = None
    ) -> Index:
        """Placeholder for actual read_index implementation"""
        pass

    def read_array(
        self, key: str, start: Optional[int] = None, stop: Optional[int] = None,
        fill_value: Optional[Any] = None
    ) -> np.ndarray:
        """Placeholder for read_array implementation"""
        pass

    def remove_node(self, node: Any, recursive: bool = True) -> None:
        """Placeholder for remove_node implementation"""
        pass

    def get_storer(self, k: str) -> Optional[Table]:
        """Placeholder for get_storer implementation"""
        pass

    def remove_node_recursive(self, node: Any, recursive: bool = True) -> None:
        """Placeholder for remove_node_recursive implementation"""
        pass

    def write_array_empty(self, key: str, value: np.ndarray) -> None:
        """Placeholder for write_array_empty implementation"""
        pass

    def _create_nodes_and_group(self, key: str) -> Any:
        """Placeholder for _create_nodes_and_group implementation"""
        pass

    def validate_min_itemsize(self, min_itemsize: Optional[Union[int, Dict[str, int]]]) -> None:
        """Placeholder for validate_min_itemsize implementation"""
        pass

    def validate(
        self, other: Table
    ) -> None:
        """Placeholder for validate implementation"""
        pass

    def validate_version(
        self, where: Optional[Union[List[Term], Term]] = None
    ) -> None:
        """Placeholder for validate_version implementation"""
        pass

    def read_metadata(self, key: str) -> Optional[Any]:
        """Placeholder for read_metadata implementation"""
        pass

    def write_data(
        self, chunksize: Optional[int], dropna: bool = False
    ) -> None:
        """Placeholder for write_data implementation"""
        pass

    def read_multi_index(self, key: str, start: Optional[int] = None, stop: Optional[int] = None) -> MultiIndex:
        """Placeholder for read_multi_index implementation"""
        pass

    def read_metadata_key(self, key: str) -> Optional[Any]:
        """Placeholder for read_metadata_key implementation"""
        pass

    def get_object(self, obj: Any, transposed: bool) -> Any:
        """Placeholder for get_object implementation"""
        pass

    def _create_storer(
        self, group: Any, format: Optional[str] = None,
        value: Optional[Union[Series, DataFrame]] = None,
        encoding: str = 'UTF-8', errors: str = 'strict'
    ) -> Table:
        """Placeholder for _create_storer implementation"""
        pass

    def _read_group(self, group: Any) -> Any:
        """Placeholder for _read_group implementation"""
        pass

    def _write_to_group(
        self, key: str, value: Union[Series, DataFrame],
        format: str, axes: Optional[List[int]] = None, index: bool = True,
        append: bool = False, complib: Optional[str] = None,
        complevel: Optional[int] = None, min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep: Optional[str] = None, data_columns: Optional[Union[List[str], bool]] = None,
        encoding: Optional[str] = None, errors: str = 'strict',
        track_times: bool = True, dropna: bool = False
    ) -> None:
        """Placeholder for _write_to_group implementation"""
        pass

    def get(
        self, key: Any
    ) -> Any:
        """Placeholder for get implementation"""
        pass

    def keys(self) -> List[str]:
        """Placeholder for keys implementation"""
        pass

    def remove(
        self, key: Any
    ) -> None:
        """Placeholder for remove implementation"""
        pass

    def append(
        self, key: str, value: Any,
        data_columns: Optional[List[str]] = None, min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        **kwargs: Any
    ) -> None:
        """Placeholder for append implementation"""
        pass

    def close(self) -> None:
        """Placeholder for close implementation"""
        pass

    def info(self) -> str:
        """Placeholder for info implementation"""
        pass


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
    chunksize : Optional[int]
        the passed chunking value (default is 100000)
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """

    def __init__(
        self, store: HDFStore, s: Table,
        func: Callable[[Optional[Any], Optional[Any], Any], Any],
        where: Optional[Union[List[Term], Term]],
        nrows: int, start: Optional[int] = None,
        stop: Optional[int] = None, iterator: bool = False,
        chunksize: Optional[int] = None, auto_close: bool = False
    ) -> None:
        self.store: HDFStore = store
        self.s: Table = s
        self.func: Callable[[Optional[Any], Optional[Any], Any], Any] = func
        self.where: Optional[Union[List[Term], Term]] = where
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)
        self.nrows: int = nrows
        self.start: Optional[int] = start
        self.stop: Optional[int] = stop
        self.coordinates: Optional[Index] = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize: Optional[int] = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close: bool = auto_close

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

    def func_7x2z67q5(self) -> None:
        if self.auto_close:
            self.store.close()

    def func_0ynsariy(self, coordinates: bool = False) -> Any:
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
        self.func_7x2z67q5()


class IndexCol:
    # ... (rest of the IndexCol class as above)
    pass

class GenericIndexCol(IndexCol):
    # ... (rest of the GenericIndexCol class as above)
    pass

class DataCol(IndexCol):
    # ... (rest of the DataCol class as above)
    pass

class WORMTable(Table):
    """
    a write-once read-many table: this format DOES NOT ALLOW appending to a
    table. writing is a one-time operation the data are stored in a format
    that allows for searching the data on disk
    """
    table_type: ClassVar[str] = 'worm'

    def func_i30vrwbe(
        self, where: Optional[Union[List[Term], Term]] = None, start: Optional[int] = None,
        stop: Optional[int] = None
    ) -> Any:
        """
        read the indices and the indexing array, calculate offset rows and return
        """
        raise NotImplementedError('WORMTable needs to implement read')

    def func_ddw250ff(
        self, obj: Any, **kwargs: Any
    ) -> None:
        """
        write in a format that we can search later on (but cannot append
        to): write out the indices and the values using _write_array
        (e.g. a CArray) create an indexing table so that we can search
        """
        raise NotImplementedError('WORMTable needs to implement write')


class AppendableTable(Table):
    """support the new appendable table formats"""
    table_type: ClassVar[str] = 'appendable'

    pandas_kind: ClassVar[str] = 'frame_table'
    ndim: ClassVar[int] = 2
    obj_type: Type[Any] = DataFrame

    def func_ddw250ff(
        self, obj: DataFrame, axes: Optional[List[int]] = None,
        append: bool = False, complib: Optional[str] = None,
        complevel: Optional[int] = None, fletcher32: Optional[bool] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        chunksize: Optional[int] = None, expectedrows: Optional[int] = None,
        dropna: bool = False, nan_rep: Optional[str] = None,
        data_columns: Optional[Union[List[str], bool]] = None,
        **kwargs: Any
    ) -> None:
        if not append and self.is_exists:
            self._handle.remove_node(self.group, 'table')
        table = self._create_axes(axes=axes, obj=obj, validate=append,
                                  min_itemsize=min_itemsize, nan_rep=nan_rep,
                                  data_columns=data_columns)
        for a in table.axes:
            a.validate_names()
        if not table.is_exists:
            options = self.create_description(complib=complib, complevel=complevel,
                                              fletcher32=fletcher32, expectedrows=expectedrows)
            table._handle.create_table(self.group, **options)
        table.attrs.info = table.info
        for a in table.axes:
            a.validate_and_set(table, append)
        self.write_data(chunksize=chunksize, dropna=dropna)

    def func_0gluxa1d(self, chunksize: Optional[int], dropna: bool = False) -> None:
        """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
        names = self.dtype.names
        nrows = self.nrows_expected
        masks: List[np.ndarray] = []
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
        values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in values]
        bvalues = []
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
            self.write_data_chunk(rows, indexes=[a[start_i:end_i] for a in indexes],
                                  mask=mask[start_i:end_i] if mask is not None else None,
                                  values=[v[start_i:end_i] for v in bvalues])

    def func_tcuo9frk(
        self, rows: np.ndarray, indexes: List[np.ndarray],
        mask: Optional[np.ndarray], values: List[np.ndarray]
    ) -> None:
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : List[np.ndarray]
            array of the indexes
        mask : Optional[np.ndarray]
            an array of the masks
        values : List[np.ndarray]
            array of the values
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

    def func_dbpfbtgg(
        self, where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None, stop: Optional[int] = None
    ) -> Optional[int]:
        """
        support fully deleting the node in its entirety (only) - where
        specification must be None
        """
        if com.all_none(where, start, stop):
            self._handle.remove_node(self.group, recursive=True)
            return None
        raise TypeError('cannot delete on an abstract storer')

    def is_exists(self) -> bool:
        """Placeholder for is_exists implementation"""
        pass

    @property
    def groups(self) -> List[Any]:
        """Placeholder for groups implementation"""
        pass

    def append_rows(self, rows: np.ndarray) -> None:
        """Placeholder for append_rows implementation"""
        pass


class AppendableFrameTable(AppendableTable):
    """a frame with a multi-index"""
    table_type: ClassVar[str] = 'appendable_multiframe'
    obj_type: ClassVar[Type[DataFrame]] = DataFrame
    ndim: ClassVar[int] = 2

    def func_ddw250ff(
        self, obj: DataFrame,
        data_columns: Optional[Union[List[str], bool]] = None,
        **kwargs: Any
    ) -> None:
        """we are going to write this as a frame table"""
        if data_columns is None:
            data_columns = []
        elif data_columns is True:
            data_columns = obj.columns.tolist()
        obj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)
        cols = list(self.levels)
        cols.append('values')
        obj.columns = Index(cols)
        super().write(obj=obj, data_columns=data_columns, **kwargs)

    def func_i30vrwbe(
        self, where: Optional[Union[List[Term], Term]] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None, stop: Optional[int] = None
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

    def validate_multiindex(self, obj: DataFrame) -> Tuple[DataFrame, List[Any]]:
        """Placeholder for validate_multiindex implementation"""
        pass


class GenericTable(AppendableFrameTable):
    """a table that read/writes the generic pytables table format"""
    pandas_kind: ClassVar[str] = 'frame_table'
    table_type: ClassVar[str] = 'generic_table'
    ndim: ClassVar[int] = 2
    obj_type: Type[DataFrame] = DataFrame

    def func_ddw250ff(self, **kwargs: Any) -> None:
        raise NotImplementedError('cannot write on an generic table')


class AppendableMultiSeriesTable(AppendableSeriesTable):
    """support the new appendable table formats"""
    pandas_kind: ClassVar[str] = 'series_table'
    table_type: ClassVar[str] = 'appendable_multiseries'
    ndim: ClassVar[int] = 2
    obj_type: ClassVar[Type[Series]] = Series

    def func_ddw250ff(self, obj: Series, **kwargs: Any) -> None:
        """we are going to write this as a frame table"""
        name = obj.name or 'values'
        newobj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)
        newobj.columns = Index([*self.levels, name])
        super().write(obj=newobj, **kwargs)


def func_fobedmu7(obj: Any, axis: int, labels: List[Any], other: Optional[Index] = None) -> Any:
    """Placeholder for func_fobedmu7 implementation"""
    pass


def func_vmqkbat2(tz: Optional[tzinfo]) -> Optional[str]:
    """for a tz-aware type, return an encoded zone"""
    if tz is not None:
        return timezones.get_timezone(tz)
    return None

def func_04gv14qo(values: np.ndarray, tz: Optional[tzinfo], datetime64_dtype: str) -> DatetimeArray:
    """
    Coerce the values to a DatetimeArray with appropriate tz.

    Parameters
    ----------
    values : np.ndarray[int64]
    tz : str, tzinfo, or None
    datetime64_dtype : str, e.g. "datetime64[ns]", "datetime64[25s]"
    """
    assert values.dtype == 'i8', values.dtype
    unit, _ = np.datetime_data(datetime64_dtype)
    dtype = tz_to_dtype(tz=tz, unit=unit)
    dta = DatetimeArray._from_sequence(values, dtype=dtype)
    return dta

def func_79e9ube9(
    name: str, index: Index, encoding: str, errors: str
) -> IndexCol:
    assert isinstance(name, str)
    index_name = index.name
    converted, dtype_name = _get_data_and_dtype_name(index)
    kind = _dtype_to_kind(dtype_name)
    atom = DataIndexableCol.func_h05b3une(converted)
    if lib.is_np_dtype(index.dtype, 'iu') or needs_i8_conversion(index.dtype) or is_bool_dtype(index.dtype):
        return IndexCol(
            name=name, values=converted, kind=kind, typ=atom,
            freq=getattr(index, 'freq', None),
            tz=getattr(index, 'tz', None),
            index_name=index_name
        )
    if isinstance(index, MultiIndex):
        raise TypeError('MultiIndex not supported here!')
    inferred_type = lib.infer_dtype(index, skipna=False)
    values = np.asarray(index)
    if inferred_type == 'date':
        converted = np.asarray([date.fromordinal(v) for v in values], dtype=np.int32)
        return IndexCol(
            name=name, values=converted, kind='date',
            typ=func_glsq77za().Time32Col(), index_name=index_name
        )
    elif inferred_type == 'string':
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(
            name=name, values=converted, kind='string',
            typ=func_glsq77za().StringCol(itemsize=itemsize),
            index_name=index_name
        )
    elif inferred_type in {'integer', 'floating'}:
        return IndexCol(
            name=name, values=converted, kind=kind, typ=atom,
            index_name=index_name
        )
    else:
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == 'object', kind
        atom = func_glsq77za().ObjectAtom()
        return IndexCol(
            name=name, values=converted, kind=kind, typ=atom,
            index_name=index_name
        )

def func_tyrrdh2r(
    data: np.ndarray, kind: str, encoding: str, errors: str
) -> np.ndarray:
    """
    Inverse of _convert_string_array.

    Parameters
    ----------
    data : np.ndarray[fixed-length-string]
    kind : str
        kind of the dtype
    encoding : str
    errors : str

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

def func_nw3v6hwe(
    name: str, bvalues: np.ndarray,
    existing_col: Optional[DataCol], min_itemsize: Optional[Union[int, Dict[str, int]]],
    nan_rep: Optional[str], encoding: str, errors: str,
    columns: List[str]
) -> np.ndarray:
    """
    Convert the passed data into a storable form and a dtype string.
    """
    if isinstance(bvalues.dtype, StringDtype):
        bvalues = bvalues.to_numpy()
    if bvalues.dtype != object:
        return bvalues
    bvalues = cast(np.ndarray, bvalues)
    dtype_name = bvalues.dtype.name
    inferred_type = lib.infer_dtype(bvalues, skipna=False)
    if inferred_type == 'date':
        raise TypeError('[date] is not implemented as a table column')
    if inferred_type == 'datetime':
        raise TypeError(
            'too many timezones in this block, create separate data columns'
        )
    if not (inferred_type == 'string' or dtype_name == 'object'):
        return bvalues
    mask = isna(bvalues)
    data = bvalues.copy()
    data[mask] = nan_rep
    if existing_col is not None and mask.any() and len(nan_rep) > existing_col.func_wfxcyp1y:
        raise ValueError('NaN representation is too large for existing column size')
    inferred_type = lib.infer_dtype(data, skipna=False)
    if inferred_type != 'string':
        for i in range(data.shape[0]):
            col = data[i]
            inferred_type = lib.infer_dtype(col, skipna=False)
            if inferred_type != 'string':
                error_column_label = columns[i] if len(columns) > i else f'No.{i}'
                raise TypeError(
                    f"""Cannot serialize the column [{error_column_label}]
because its data contents are not [string] but [{inferred_type}] object dtype"""
                )
    data_converted = _convert_string_array(data, encoding, errors).reshape(data.shape)
    itemsize = data_converted.itemsize
    if isinstance(min_itemsize, dict):
        min_itemsize = int(min_itemsize.get(name, min_itemsize.get('values', 0)))
    itemsize = max(min_itemsize or 0, itemsize)
    if existing_col is not None:
        eci = existing_col.func_mnomiqay(itemsize)
        if eci is not None and eci > itemsize:
            itemsize = eci
    data_converted = data_converted.astype(f'|S{itemsize}', copy=False)
    return data_converted

def func_rd1cs57w(
    data: np.ndarray, encoding: str, errors: str
) -> np.ndarray:
    """
    Take a string-like that is object dtype and coerce to a fixed size string type.

    Parameters
    ----------
    data : np.ndarray[object]
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[fixed-length-string]
        Decoded data.
    """
    if len(data):
        data = Series(data.ravel(), copy=False).str.encode(encoding, errors=errors)._values.reshape(data.shape)
    ensured = ensure_object(data.ravel())
    itemsize = max(1, libwriters.max_len_string_array(ensured))
    data = np.asarray(data, dtype=f'S{itemsize}')
    return data

def func_pjv0io2t(
    data: np.ndarray, nan_rep: Optional[str],
    encoding: str, errors: str
) -> np.ndarray:
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

def func_19m5iaiv(
    values: np.ndarray, val_kind: str, encoding: str, errors: str
) -> np.ndarray:
    """
    Convert the data from this selection to the appropriate pandas type.

    Parameters
    ----------
    values : np.ndarray
    val_kind : str
    encoding : str
    errors : str

    Returns
    -------
    np.ndarray
    """
    assert isinstance(val_kind, str), type(val_kind)
    if _need_convert(val_kind):
        conv = _get_converter(val_kind, encoding, errors)
        values = conv(values)
    return values

def func_mby3wq9i(kind: str, encoding: str, errors: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Find the "kind" string describing the given dtype name.
    """
    if kind == 'datetime64':
        return lambda x: np.asarray(x, dtype='M8[ns]')
    elif 'datetime64' in kind:
        return lambda x: np.asarray(x, dtype=kind)
    elif kind == 'string':
        return lambda x: func_pjv0io2t(x, nan_rep=None, encoding=encoding, errors=errors)
    else:
        raise ValueError(f'invalid kind {kind}')

def func_v6defya3(kind: str) -> bool:
    if kind in {'datetime64', 'string'} or 'datetime64' in kind:
        return True
    return False

def func_awrzbszi(name: str, version: Tuple[int, int, int]) -> str:
    """
    Prior to 0.10.1, we named values blocks like: values_block_0 and the
    name values_0, adjust the given name if necessary.

    Parameters
    ----------
    name : str
    version : Tuple[int, int, int]

    Returns
    -------
    str
    """
    if isinstance(version, str) or len(version) < 3:
        raise ValueError(
            'Version is incorrect, expected sequence of 3 integers.'
        )
    if version[0] == 0 and version[1] <= 10 and version[2] == 0:
        m = re.search(r'^values_block_(\d+)', name)
        if m:
            grp = m.groups()[0]
            name = f'values_{grp}'
    return name

def func_wgpibb1z(kind: str) -> str:
    """
    Find the "kind" string describing the given dtype name.
    """
    if kind.startswith(('string', 'bytes')):
        kind_desc = 'string'
    elif kind.startswith('float'):
        kind_desc = 'float'
    elif kind.startswith('complex'):
        kind_desc = 'complex'
    elif kind.startswith(('int', 'uint')):
        kind_desc = 'integer'
    elif kind.startswith('datetime64'):
        kind_desc = kind
    elif kind.startswith('timedelta'):
        kind_desc = 'timedelta64'
    elif kind.startswith('bool'):
        kind_desc = 'bool'
    elif kind.startswith('category'):
        kind_desc = 'category'
    elif kind.startswith('period'):
        kind_desc = 'integer'
    elif kind == 'object':
        kind_desc = 'object'
    elif kind == 'str':
        kind_desc = 'str'
    else:
        raise ValueError(f'cannot interpret dtype of [{kind}]')
    return kind_desc

def func_o4yhj26v(data: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Convert the passed data into a storable form and a dtype string.
    """
    if isinstance(data, Categorical):
        data = data.codes
    if isinstance(data.dtype, DatetimeTZDtype):
        dtype_name = f'datetime64[{data.dtype.unit}]'
    else:
        dtype_name = data.dtype.name
    if data.dtype.kind in {'m', 'M'}:
        data = np.asarray(data.view('i8'))
    elif isinstance(data, PeriodIndex):
        data = data.asi8
    data = np.asarray(data)
    return data, dtype_name

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
    format_type: ClassVar[str] = 'fixed'
    is_table: ClassVar[bool] = False

    def __init__(self, parent: HDFStore, group: Any, encoding: str = 'UTF-8', errors: str = 'strict') -> None:
        assert isinstance(parent, HDFStore), type(parent)
        assert _table_mod is not None
        assert isinstance(group, _table_mod.Node), type(group)
        self.parent: HDFStore = parent
        self.group: Any = group
        self.encoding: str = func_j7cb0xfa(encoding)
        self.errors: str = errors

    @property
    def func_ip1ednpa(self) -> bool:
        return self.version[0] <= 0 and self.version[1] <= 10 and self.version[2] < 1

    @property
    def func_2q0ii6k8(self) -> Tuple[int, int, int]:
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
    def func_ofm8lfvd(self) -> Optional[str]:
        return getattr(self.group._v_attrs, 'pandas_type', None)

    def __repr__(self) -> str:
        """return a pretty representation of myself"""
        if not self.infer_axes():
            shape = None
        else:
            shape = self.shape
        jdc = 'N/A' if shape is None else f'(shape->{shape})'
        return f'{self.pandas_type:12.12} {jdc}'

    def func_pbn4cdeq(self) -> None:
        """set our pandas type & version"""
        self.attrs.pandas_type = str(self.pandas_kind)
        self.attrs.pandas_version = str(_version)

    def func_knn0xpsp(self) -> Fixed:
        new_self = copy.copy(self)
        return new_self

    @property
    def func_g1e1unjr(self) -> Optional[int]:
        return self.nrows

    @property
    def func_p5aabyab(self) -> str:
        return self.group._v_pathname

    @property
    def func_t0u4p9as(self) -> Any:
        return self.parent._handle

    @property
    def func_s3l3ex4m(self) -> Optional[Any]:
        return self.parent._filters

    @property
    def func_1gjo68ha(self) -> int:
        return self.parent._complevel

    @property
    def func_w1hbyr18(self) -> bool:
        return self.parent._fletcher32

    @property
    def func_qnpsejis(self) -> Any:
        return self.group._v_attrs

    def func_yftc3utz(self) -> None:
        """set our object attributes"""
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors

    def func_fctclga9(self) -> None:
        """retrieve our attributes"""
        self.encoding = func_j7cb0xfa(getattr(self.attrs, 'encoding', None))
        self.errors = getattr(self.attrs, 'errors', 'strict')

    @property
    def func_v3zznlp6(self) -> Any:
        """return my storable"""
        return self.group

    @property
    def func_4c3l5j1y(self) -> bool:
        return False

    @property
    def func_my1i0gml(self) -> Optional[int]:
        return getattr(self.storable, 'nrows', None)

    def func_lyez4o51(self, other: Optional[Fixed]) -> Optional[Any]:
        """validate against an existing storer"""
        if other is None:
            return None
        return True

    def func_1ig3p4fh(self, where: Optional[Union[List[Term], Term]] = None) -> None:
        """are we trying to operate on an old version?"""
        if where is not None:
            if self.is_old_version:
                ws = incompatibility_doc % '.'.join([str(x) for x in self.version])
                warnings.warn(ws, IncompatibilityWarning, stacklevel=find_stack_level())

    def func_840qwob8(self) -> bool:
        """
        infer the axes of my storer
        return a boolean indicating if we have a valid storer or not
        """
        s = self.storable
        if s is None:
            return False
        self.get_attrs()
        return True

    def func_i30vrwbe(self, where: Optional[Union[List[Term], Term]] = None,
                     columns: Optional[List[str]] = None, start: Optional[int] = None,
                     stop: Optional[int] = None) -> Any:
        """Cannot read on an abstract storer: subclasses should implement"""
        raise NotImplementedError

    def func_ddw250ff(self, obj: Any, **kwargs: Any) -> None:
        """Cannot write on an abstract storer: subclasses should implement"""
        raise NotImplementedError

    def func_dbpfbtgg(
        self, where: Optional[Union[List[Term], Term]] = None,
        start: Optional[int] = None, stop: Optional[int] = None
    ) -> Any:
        """
        support fully deleting the node in its entirety (only) - where
        specification must be None
        """
        if com.all_none(where, start, stop):
            self._handle.remove_node(self.group, recursive=True)
            return None
        raise TypeError('cannot delete on an abstract storer')
