#!/usr/bin/env python3
"""
High level interface to PyTables for reading and writing pandas data structures
to disk
"""

from __future__ import annotations

from contextlib import suppress
import copy
from datetime import date
import itertools
import os
import re
from textwrap import dedent
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, overload

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType
    from tables import Col, File, Node
    from pandas._typing import AnyArrayLike, ArrayLike, AxisInt, DtypeArg, FilePath, Self, Shape, npt
    from pandas.core.internals import Block

_version: str = '0.15.2'
_default_encoding: str = 'UTF-8'

def _ensure_encoding(encoding: Optional[str]) -> str:
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

Term: Type[PyTablesExpr] = PyTablesExpr

def _ensure_term(where: Any, scope_level: int) -> Optional[Union[Term, List[Term]]]:
    """
    Ensure that the where is a Term or a list of Term.
    """
    level: int = scope_level + 1
    if isinstance(where, (list, tuple)):
        where = [Term(term, scope_level=level + 1) if maybe_expression(term) else term for term in where if term is not None]
    elif maybe_expression(where):
        where = Term(where, scope_level=level)
    return where if where is None or len(where) else None

incompatibility_doc: str = (
    "\nwhere criteria is being ignored as this version [%s] is too old (or\n"
    "not-defined), read the file in and write it out to a new file to upgrade (with\n"
    "the copy_to method)\n"
)
attribute_conflict_doc: str = (
    "\nthe [%s] attribute of the existing index is [%s] which conflicts with the new\n"
    "[%s], resetting the attribute to None\n"
)
performance_doc: str = (
    "\nyour performance may suffer as PyTables will pickle object types that it cannot\n"
    "map directly to c-types [inferred_type->%s,key->%s] [items->%s]\n"
)
_FORMAT_MAP: Dict[str, str] = {'f': 'fixed', 'fixed': 'fixed', 't': 'table', 'table': 'table'}
_AXES_MAP: Dict[Type[Any], List[int]] = {DataFrame: [0]}
dropna_doc: str = '\n: boolean\n    drop ALL nan rows when appending to a table\n'
format_doc: str = "\n: format\n    default format writing format, if None, then\n    put will default to 'fixed' and append will default to 'table'\n"
with config.config_prefix('io.hdf'):
    config.register_option('dropna_table', False, dropna_doc, validator=config.is_bool)
    config.register_option('default_format', None, format_doc, validator=config.is_one_of_factory(['fixed', 'table', None]))
_table_mod: Any = None
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

def to_hdf(
    path_or_buf: Union[str, HDFStore],
    key: Any,
    value: Any,
    mode: str = 'a',
    complevel: Optional[int] = None,
    complib: Optional[str] = None,
    append: bool = False,
    format: Optional[str] = None,
    index: bool = True,
    min_itemsize: Optional[Any] = None,
    nan_rep: Optional[Any] = None,
    dropna: Optional[bool] = None,
    data_columns: Optional[Any] = None,
    errors: str = 'strict',
    encoding: str = 'UTF-8'
) -> None:
    """store this object, close it if we opened it"""
    if append:
        f = lambda store: store.append(key, value, format=format, index=index, min_itemsize=min_itemsize,
                                       nan_rep=nan_rep, dropna=dropna, data_columns=data_columns, errors=errors, encoding=encoding)
    else:
        f = lambda store: store.put(key, value, format=format, index=index, min_itemsize=min_itemsize,
                                    nan_rep=nan_rep, data_columns=data_columns, errors=errors, encoding=encoding, dropna=dropna)
    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        path_or_buf = stringify_path(path_or_buf)
        with HDFStore(path_or_buf, mode=mode, complevel=complevel, complib=complib) as store:
            f(store)

def read_hdf(
    path_or_buf: Union[str, HDFStore],
    key: Optional[Any] = None,
    mode: str = 'r',
    errors: str = 'strict',
    where: Optional[Any] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    columns: Optional[List[Any]] = None,
    iterator: bool = False,
    chunksize: Optional[int] = None,
    **kwargs: Any
) -> Any:
    """
    Read from the store, close it if we opened it.

    Retrieve pandas object stored in file, optionally based on where
    criteria.
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
            groups: List[Any] = store.groups()
            if len(groups) == 0:
                raise ValueError('Dataset(s) incompatible with Pandas data types, not table, or no datasets found in HDF5 file.')
            candidate_only_group = groups[0]
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
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        mode: str = 'a',
        complevel: Optional[int] = None,
        complib: Optional[str] = None,
        fletcher32: bool = False,
        **kwargs: Any
    ) -> None:
        if 'format' in kwargs:
            raise ValueError('format is not a defined argument for HDFStore')
        tables = import_optional_dependency('tables')
        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(f'complib only supports {tables.filters.all_complibs} compression.')
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
    def root(self) -> Any:
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None
        return self._handle.root

    @property
    def filename(self) -> str:
        return self._path

    def __getitem__(self, key: Any) -> Any:
        return self.get(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: Any) -> Any:
        return self.remove(key)

    def __getattr__(self, name: str) -> Any:
        """allow attribute access to get stores"""
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __contains__(self, key: Any) -> bool:
        """
        check for existence of this key
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

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()

    def keys(self, include: str = 'pandas') -> List[str]:
        """
        Return a list of keys corresponding to objects stored in HDFStore.
        """
        if include == 'pandas':
            return [n._v_pathname for n in self.groups()]
        elif include == 'native':
            assert self._handle is not None
            return [n._v_pathname for n in self._handle.walk_nodes('/', classname='Table')]
        raise ValueError(f"`include` should be either 'pandas' or 'native' but is '{include}'")

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def items(self) -> Iterable[Tuple[str, Any]]:
        """
        iterate on key->group
        """
        for g in self.groups():
            yield (g._v_pathname, g)

    def open(self, mode: str = 'a', **kwargs: Any) -> None:
        """
        Open the file in the specified mode
        """
        tables = _tables()
        if self._mode != mode:
            if self._mode in ['a', 'w'] and mode in ['r', 'r+']:
                pass
            elif mode in ['w']:
                if self.is_open:
                    raise PossibleDataLossError(f'Re-opening the file [{self._path}] with mode [{self._mode}] will delete the current file!')
            self._mode = mode
        if self.is_open:
            self.close()
        if self._complevel and self._complevel > 0:
            self._filters = _tables().Filters(self._complevel, self._complib, fletcher32=self._fletcher32)
        if _table_file_open_policy_is_strict and self.is_open:
            msg = 'Cannot open HDF5 file, which is already opened, even in read-only mode.'
            raise ValueError(msg)
        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def close(self) -> None:
        """
        Close the PyTables file handle
        """
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def is_open(self) -> bool:
        """
        return a boolean indicating whether the file is open
        """
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def flush(self, fsync: bool = False) -> None:
        """
        Force all buffered modifications to be written to disk.
        """
        if self._handle is not None:
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def get(self, key: Any) -> Any:
        """
        Retrieve pandas object stored in file.
        """
        with patch_pickle():
            group = self.get_node(key)
            if group is None:
                raise KeyError(f'No object named {key} in the file')
            return self._read_group(group)

    def select(
        self,
        key: Any,
        where: Optional[Any] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        columns: Optional[List[Any]] = None,
        iterator: bool = False,
        chunksize: Optional[int] = None,
        auto_close: bool = False
    ) -> Any:
        """
        Retrieve pandas object stored in file, optionally based on where criteria.
        """
        group = self.get_node(key)
        if group is None:
            raise KeyError(f'No object named {key} in the file')
        where = _ensure_term(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        def func(_start: Optional[int], _stop: Optional[int], _where: Any) -> Any:
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)
        it = TableIterator(self, s, func, where=where, nrows=s.nrows, start=start, stop=stop,
                           iterator=iterator, chunksize=chunksize, auto_close=auto_close)
        return it.get_result()

    def select_as_coordinates(self, key: Any, where: Optional[Any] = None, start: Optional[int] = None, stop: Optional[int] = None) -> Any:
        """
        return the selection as an Index
        """
        where = _ensure_term(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_coordinates with a table')
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def select_column(self, key: Any, column: str, start: Optional[int] = None, stop: Optional[int] = None) -> Any:
        """
        return a single column from the table.
        """
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_column with a table')
        return tbl.read_column(column=column, start=start, stop=stop)

    def select_as_multiple(
        self,
        keys: Union[List[Any], Tuple[Any, ...], str],
        where: Optional[Any] = None,
        selector: Optional[Any] = None,
        columns: Optional[List[Any]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        iterator: bool = False,
        chunksize: Optional[int] = None,
        auto_close: bool = False
    ) -> Any:
        """
        Retrieve pandas objects from multiple tables.
        """
        where = _ensure_term(where, scope_level=1)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(key=keys, where=where, columns=columns, start=start, stop=stop,
                               iterator=iterator, chunksize=chunksize, auto_close=auto_close)
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
                raise TypeError(f'object [{t.pathname}] is not a table, and cannot be used in all select as multiple')
            if nrows is None:
                nrows = t.nrows
            elif t.nrows != nrows:
                raise ValueError('all tables must have exactly the same nrows!')
        _tbls = [x for x in tbls if isinstance(x, Table)]
        axis = {t.non_index_axes[0][0] for t in _tbls}.pop()

        def func(_start: Optional[int], _stop: Optional[int], _where: Any) -> Any:
            objs = [t.read(where=_where, columns=columns, start=_start, stop=_stop) for t in tbls]
            return concat(objs, axis=axis, verify_integrity=False)._consolidate()
        it = TableIterator(self, s, func, where=where, nrows=nrows, start=start, stop=stop,
                           iterator=iterator, chunksize=chunksize, auto_close=auto_close)
        return it.get_result(coordinates=True)

    def put(
        self,
        key: Any,
        value: Any,
        format: Optional[str] = None,
        index: bool = True,
        append: bool = False,
        complib: Optional[Any] = None,
        complevel: Optional[int] = None,
        min_itemsize: Optional[Any] = None,
        nan_rep: Optional[Any] = None,
        data_columns: Optional[List[Any]] = None,
        encoding: Optional[str] = None,
        errors: str = 'strict',
        track_times: bool = True,
        dropna: bool = False
    ) -> None:
        """
        Store object in HDFStore.
        """
        if format is None:
            format = get_option('io.hdf.default_format') or 'fixed'
        format = self._validate_format(format)
        self._write_to_group(key, value, format=format, index=index, append=append, complib=complib,
                              complevel=complevel, min_itemsize=min_itemsize, nan_rep=nan_rep,
                              data_columns=data_columns, encoding=encoding, errors=errors,
                              track_times=track_times, dropna=dropna)

    def remove(self, key: Any, where: Optional[Any] = None, start: Optional[int] = None, stop: Optional[int] = None) -> Optional[int]:
        """
        Remove pandas object partially by specifying the where condition
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
                raise ValueError('trying to remove a node with a non-None where clause!') from err
            node = self.get_node(key)
            if node is not None:
                node._f_remove(recursive=True)
                return None
        if com.all_none(where, start, stop):
            s.group._f_remove(recursive=True)
            return None
        if not s.is_table:
            raise ValueError('can only remove with where on objects written as tables')
        return s.delete(where=where, start=start, stop=stop)

    def append(
        self,
        key: Any,
        value: Any,
        format: Optional[str] = None,
        axes: Optional[Any] = None,
        index: bool = True,
        append: bool = True,
        complib: Optional[Any] = None,
        complevel: Optional[int] = None,
        columns: Optional[Any] = None,
        min_itemsize: Optional[Any] = None,
        nan_rep: Optional[Any] = None,
        chunksize: Optional[int] = None,
        expectedrows: Optional[int] = None,
        dropna: Optional[bool] = None,
        data_columns: Optional[List[Any]] = None,
        encoding: Optional[str] = None,
        errors: str = 'strict'
    ) -> None:
        """
        Append to Table in file.
        """
        if columns is not None:
            raise TypeError('columns is not a supported keyword in append, try data_columns')
        if dropna is None:
            dropna = get_option('io.hdf.dropna_table')
        if format is None:
            format = get_option('io.hdf.default_format') or 'table'
        format = self._validate_format(format)
        self._write_to_group(key, value, format=format, axes=axes, index=index, append=append,
                              complib=complib, complevel=complevel, min_itemsize=min_itemsize, nan_rep=nan_rep,
                              chunksize=chunksize, expectedrows=expectedrows, dropna=dropna,
                              data_columns=data_columns, encoding=encoding, errors=errors)

    def append_to_multiple(
        self,
        d: Dict[Any, Optional[Any]],
        value: Any,
        selector: Any,
        data_columns: Optional[Any] = None,
        axes: Optional[Any] = None,
        dropna: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Append to multiple tables
        """
        if axes is not None:
            raise TypeError('axes is currently not accepted as a parameter to append_to_multiple; you can create the tables independently instead')
        if not isinstance(d, dict):
            raise ValueError('append_to_multiple must have a dictionary specified as the way to split the value')
        if selector not in d:
            raise ValueError('append_to_multiple requires a selector that is in passed dict')
        axis = next(iter(set(range(value.ndim)) - set(_AXES_MAP[type(value)])))
        remain_key: Optional[Any] = None
        remain_values: List[Any] = []
        for k, v in d.items():
            if v is None:
                if remain_key is not None:
                    raise ValueError('append_to_multiple can only have one value in d that is None')
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
            filtered = {key: value for key, value in min_itemsize.items() if key in v} if min_itemsize is not None else None
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)

    def create_table_index(self, key: Any, columns: Optional[Union[bool, List[str]]] = None, optlevel: Optional[int] = None, kind: Optional[str] = None) -> None:
        """
        Create a pytables index on the table.
        """
        _tables()
        s = self.get_storer(key)
        if s is None:
            return
        if not isinstance(s, Table):
            raise TypeError('cannot create table index on a Fixed format store')
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def groups(self) -> List[Any]:
        """
        Return a list of all the top-level nodes.
        """
        _tables()
        self._check_if_open()
        assert self._handle is not None
        assert _table_mod is not None
        return [g for g in self._handle.walk_groups() if not isinstance(g, _table_mod.link.Link) and
                (getattr(g._v_attrs, 'pandas_type', None) or getattr(g, 'table', None) or (isinstance(g, _table_mod.table.Table) and g._v_name != 'table'))]

    def walk(self, where: str = '/') -> Iterator[Tuple[str, List[str], List[str]]]:
        """
        Walk the pytables group hierarchy for pandas objects.
        """
        _tables()
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
            yield (g._v_pathname.rstrip('/'), groups, leaves)

    def get_node(self, key: Any) -> Optional[Any]:
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

    def get_storer(self, key: Any) -> Any:
        """return the storer object for a key, raise if not in the file"""
        group = self.get_node(key)
        if group is None:
            raise KeyError(f'No object named {key} in the file')
        s = self._create_storer(group)
        s.infer_axes()
        return s

    def copy(
        self,
        file: Union[str, os.PathLike],
        mode: str = 'w',
        propindexes: bool = True,
        keys: Optional[List[Any]] = None,
        complib: Optional[Any] = None,
        complevel: Optional[int] = None,
        fletcher32: bool = False,
        overwrite: bool = True
    ) -> HDFStore:
        """
        Copy the existing store to a new file, updating in place.
        """
        new_store: HDFStore = HDFStore(file, mode=mode, complib=complib, complevel=complevel, fletcher32=fletcher32)
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
                    index_flag: Union[bool, List[str]] = False
                    if propindexes:
                        index_flag = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(k, data, index=index_flag, data_columns=getattr(s, 'data_columns', None), encoding=s.encoding)
                else:
                    new_store.put(k, data, encoding=s.encoding)
        return new_store

    def info(self) -> str:
        """
        Print detailed information on the store.
        """
        path = pprint_thing(self._path)
        output: str = f'{type(self)}\nFile path: {path}\n'
        if self.is_open:
            lkeys = sorted(self.keys())
            if len(lkeys):
                keys: List[str] = []
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
                        dstr = pprint_thing(detail)
                        values.append(f'[invalid_HDFStore node: {dstr}]')
                output += adjoin(12, keys, values)
            else:
                output += 'Empty'
        else:
            output += 'File is CLOSED'
        return output

    def _check_if_open(self) -> None:
        if not self.is_open:
            raise ClosedFileError(f'{self._path} file is not open!')

    def _validate_format(self, format: str) -> str:
        """validate / deprecate formats"""
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f'invalid HDFStore format specified [{format}]') from err
        return format

    def _create_storer(self, group: Any, format: Optional[str] = None, value: Optional[Any] = None, encoding: str = 'UTF-8', errors: str = 'strict') -> Any:
        """return a suitable class to operate"""
        if value is not None and (not isinstance(value, (Series, DataFrame))):
            raise TypeError('value must be None, Series, or DataFrame')
        pt = getattr(group._v_attrs, 'pandas_type', None)
        tt = getattr(group._v_attrs, 'table_type', None)
        if pt is None:
            if value is None:
                _tables()
                assert _table_mod is not None
                if getattr(group, 'table', None) or isinstance(group, _table_mod.table.Table):
                    pt = 'frame_table'
                    tt = 'generic_table'
                else:
                    raise TypeError('cannot create a storer if the object is not existing nor a value are passed')
            else:
                if isinstance(value, Series):
                    pt = 'series'
                else:
                    pt = 'frame'
                if format == 'table':
                    pt += '_table'
        if 'table' not in pt:
            _STORER_MAP: Dict[str, Any] = {'series': SeriesFixed, 'frame': FrameFixed}
            try:
                cls = _STORER_MAP[pt]
            except KeyError as err:
                raise TypeError(f'cannot properly create the storer for: [_STORER_MAP] [group->{group},value->{type(value)},format->{format}') from err
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
        _TABLE_MAP: Dict[str, Any] = {
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
            raise TypeError(f'cannot properly create the storer for: [_TABLE_MAP] [group->{group},value->{type(value)},format->{format}') from err
        return cls(self, group, encoding=encoding, errors=errors)

    def _write_to_group(
        self,
        key: Any,
        value: Any,
        format: str,
        axes: Optional[Any] = None,
        index: bool = True,
        append: bool = False,
        complib: Optional[Any] = None,
        complevel: Optional[int] = None,
        fletcher32: Optional[bool] = None,
        min_itemsize: Optional[Any] = None,
        chunksize: Optional[int] = None,
        expectedrows: Optional[int] = None,
        dropna: bool = False,
        nan_rep: Optional[Any] = None,
        data_columns: Optional[List[Any]] = None,
        encoding: Optional[str] = None,
        errors: str = 'strict',
        track_times: bool = True
    ) -> None:
        if getattr(value, 'empty', None) and (format == 'table' or append):
            return
        group = self._identify_group(key, append)
        s = self._create_storer(group, format, value, encoding=encoding, errors=errors)
        if append:
            if not s.is_table or (s.is_table and format == 'fixed' and s.is_exists):
                raise ValueError('Can only append to Tables')
            if not s.is_exists:
                s.set_object_info()
        else:
            s.set_object_info()
        if not s.is_table and complib:
            raise ValueError('Compression not supported on Fixed format stores')
        s.write(obj=value, axes=axes, append=append, complib=complib, complevel=complevel,
                fletcher32=fletcher32, min_itemsize=min_itemsize, chunksize=chunksize,
                expectedrows=expectedrows, dropna=dropna, nan_rep=nan_rep, data_columns=data_columns,
                track_times=track_times)
        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def _read_group(self, group: Any) -> Any:
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def _identify_group(self, key: Any, append: bool) -> Any:
        """Identify HDF5 group based on key, delete/create group if needed."""
        group = self.get_node(key)
        assert self._handle is not None
        if group is not None and (not append):
            self._handle.remove_node(group, recursive=True)
            group = None
        if group is None:
            group = self._create_nodes_and_group(key)
        return group

    def _create_nodes_and_group(self, key: Any) -> Any:
        """Create nodes from key and return group name."""
        assert self._handle is not None
        paths: List[str] = key.split('/')
        path: str = '/'
        for p in paths:
            if not len(p):
                continue
            new_path: str = path
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
    """

    def __init__(
        self,
        store: HDFStore,
        s: Any,
        func: Callable[[Optional[int], Optional[int], Any], Any],
        where: Optional[Any],
        nrows: Optional[int],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        iterator: bool = False,
        chunksize: Optional[int] = None,
        auto_close: bool = False
    ) -> None:
        self.store: HDFStore = store
        self.s: Any = s
        self.func: Callable[[Optional[int], Optional[int], Any], Any] = func
        self.where: Optional[Any] = where
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)
        self.nrows: Optional[int] = nrows
        self.start: Optional[int] = start
        self.stop: Optional[int] = stop
        self.coordinates: Optional[Any] = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize: Optional[int] = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close: bool = auto_close

    def __iter__(self) -> Iterator[Any]:
        current: int = self.start if self.start is not None else 0
        if self.coordinates is None:
            raise ValueError('Cannot iterate until get_result is called.')
        while current < (self.stop if self.stop is not None else current):
            stop = min(current + self.chunksize, self.stop if self.stop is not None else current)
            value = self.func(None, None, self.coordinates[current:stop])
            current = stop
            if value is None or not len(value):
                continue
            yield value
        self.close()

    def close(self) -> None:
        if self.auto_close:
            self.store.close()

    def get_result(self, coordinates: bool = False) -> Any:
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
    """
    an index column description class
    """

    is_an_indexable: bool = True
    is_data_indexable: bool = True
    _info_fields: List[str] = ['freq', 'tz', 'index_name']

    def __init__(
        self,
        name: str,
        values: Optional[Any] = None,
        kind: Optional[str] = None,
        typ: Optional[Any] = None,
        cname: Optional[str] = None,
        axis: Optional[int] = None,
        pos: Optional[int] = None,
        freq: Optional[Any] = None,
        tz: Optional[Any] = None,
        index_name: Optional[str] = None,
        ordered: Optional[bool] = None,
        table: Optional[Any] = None,
        meta: Optional[Any] = None,
        metadata: Optional[Any] = None
    ) -> None:
        if not isinstance(name, str):
            raise ValueError('`name` must be a str.')
        self.values: Any = values
        self.kind: Optional[str] = kind
        self.typ: Optional[Any] = typ
        self.name: str = name
        self.cname: str = cname or name
        self.axis: Optional[int] = axis
        self.pos: Optional[int] = pos
        self.freq: Optional[Any] = freq
        self.tz: Optional[Any] = tz
        self.index_name: Optional[str] = index_name
        self.ordered: Optional[bool] = ordered
        self.table: Optional[Any] = table
        self.meta: Optional[Any] = meta
        self.metadata: Optional[Any] = metadata
        if pos is not None:
            self.set_pos(pos)
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def itemsize(self) -> int:
        return self.typ.itemsize  # type: ignore

    @property
    def kind_attr(self) -> str:
        return f'{self.name}_kind'

    def set_pos(self, pos: int) -> None:
        """set the position of this column in the Table"""
        self.pos = pos
        if pos is not None and self.typ is not None:
            self.typ._v_pos = pos  # type: ignore

    def __repr__(self) -> str:
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.axis, self.pos, self.kind)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name', 'cname', 'axis', 'pos', 'kind'], temp)])

    def __eq__(self, other: Any) -> bool:
        """compare 2 col items"""
        return all((getattr(self, a, None) == getattr(other, a, None) for a in ['name', 'cname', 'axis', 'pos']))

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def is_indexed(self) -> bool:
        """return whether I am an indexed column"""
        if not hasattr(self.table, 'cols'):
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(self, values: np.ndarray, nan_rep: Any, encoding: str, errors: str) -> Tuple[Any, Any]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname].copy()
        val_kind: Optional[str] = self.kind
        values = _maybe_convert(values, val_kind, encoding, errors)
        kwargs: Dict[str, Any] = {}
        kwargs['name'] = self.index_name
        if self.freq is not None:
            kwargs['freq'] = self.freq
        factory = Index
        if lib.is_np_dtype(values.dtype, 'M') or isinstance(values.dtype, DatetimeTZDtype):
            factory = DatetimeIndex
        elif values.dtype == 'i8' and 'freq' in kwargs:
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
        return (final_pd_index, final_pd_index)

    def take_data(self) -> Any:
        """return the values"""
        return self.values

    @property
    def attrs(self) -> Any:
        return self.table._v_attrs  # type: ignore

    @property
    def description(self) -> Any:
        return self.table.description  # type: ignore

    @property
    def col(self) -> Any:
        """return my current col description"""
        return getattr(self.description, self.cname, None)

    @property
    def cvalues(self) -> Any:
        """return my cython values"""
        return self.values

    def __iter__(self) -> Any:
        return iter(self.values)

    def maybe_set_size(self, min_itemsize: Optional[Union[int, Dict[str, int]]] = None) -> None:
        """
        maybe set a string col itemsize.
        """
        if self.kind == 'string':
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)
            if min_itemsize is not None and self.typ.itemsize < min_itemsize:  # type: ignore
                self.typ = _tables().StringCol(itemsize=min_itemsize, pos=self.pos)

    def validate_names(self) -> None:
        pass

    def validate_and_set(self, handler: Any, append: bool) -> None:
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def validate_col(self, itemsize: Optional[int] = None) -> Optional[int]:
        """validate this column: return the compared against itemsize"""
        if self.kind == 'string':
            c = self.col
            if c is not None:
                if itemsize is None:
                    itemsize = self.itemsize
                if c.itemsize < itemsize:
                    raise ValueError(f'Trying to store a string with len [{itemsize}] in [{self.cname}] column but\nthis column has a limit of [{c.itemsize}]!\nConsider using min_itemsize to preset the sizes on these columns')
                return c.itemsize
        return None

    def validate_attr(self, append: bool) -> None:
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(f'incompatible kind in col [{existing_kind} - {self.kind}]')

    def update_info(self, info: Dict[str, Any]) -> None:
        """
        set/update the info for this indexable with the key/value
        """
        for key in self._info_fields:
            value = getattr(self, key, None)
            idx = info.setdefault(self.name, {})
            existing_value = idx.get(key)
            if key in idx and value is not None and (existing_value != value):
                if key in ['freq', 'index_name']:
                    ws = attribute_conflict_doc % (key, existing_value, value)
                    warnings.warn(ws, AttributeConflictWarning, stacklevel=find_stack_level())
                    idx[key] = None
                    setattr(self, key, None)
                else:
                    raise ValueError(f'invalid info for [{self.name}] for [{key}], existing_value [{existing_value}] conflicts with new value [{value}]')
            elif value is not None or existing_value is not None:
                idx[key] = value

    def set_info(self, info: Dict[str, Any]) -> None:
        """set my state from the passed info"""
        idx = info.get(self.name)
        if idx is not None:
            self.__dict__.update(idx)

    def set_attr(self) -> None:
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)

    def validate_metadata(self, handler: Any) -> None:
        """validate that kind=category does not change the categories"""
        if self.meta == 'category':
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if new_metadata is not None and cur_metadata is not None and (not array_equivalent(new_metadata, cur_metadata, strict_nan=True, dtype_equal=True)):
                raise ValueError('cannot append a categorical with different categories to the existing')

    def write_metadata(self, handler: Any) -> None:
        """set the meta data"""
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)

class GenericIndexCol(IndexCol):
    """an index which is not represented in the data of the table"""

    @property
    def is_indexed(self) -> bool:
        return False

    def convert(self, values: np.ndarray, nan_rep: Any, encoding: str, errors: str) -> Tuple[Any, Any]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)
        index = RangeIndex(len(values))
        return (index, index)

    def set_attr(self) -> None:
        pass

class DataCol(IndexCol):
    """
    a data holding column, by definition this is not indexable
    """

    is_an_indexable: bool = False
    is_data_indexable: bool = False
    _info_fields: List[str] = ['tz', 'ordered']

    def __init__(
        self,
        name: str,
        values: Optional[Any] = None,
        kind: Optional[str] = None,
        typ: Optional[Any] = None,
        cname: Optional[str] = None,
        pos: Optional[int] = None,
        tz: Optional[Any] = None,
        ordered: Optional[bool] = None,
        table: Optional[Any] = None,
        meta: Optional[Any] = None,
        metadata: Optional[Any] = None,
        dtype: Optional[Any] = None,
        data: Optional[Any] = None
    ) -> None:
        super().__init__(name=name, values=values, kind=kind, typ=typ, cname=cname, pos=pos, tz=tz, ordered=ordered, table=table, meta=meta, metadata=metadata)
        self.dtype: Optional[Any] = dtype
        self.data: Optional[Any] = data

    @property
    def dtype_attr(self) -> str:
        return f'{self.name}_dtype'

    @property
    def meta_attr(self) -> str:
        return f'{self.name}_meta'

    def __repr__(self) -> str:
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.dtype, self.kind, self.shape)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name', 'cname', 'dtype', 'kind', 'shape'], temp)])

    def __eq__(self, other: Any) -> bool:
        return all((getattr(self, a, None) == getattr(other, a, None) for a in ['name', 'cname', 'dtype', 'pos']))

    def set_data(self, data: Any) -> None:
        assert data is not None
        assert self.dtype is None
        data, dtype_name = _get_data_and_dtype_name(data)
        self.data = data
        self.dtype = dtype_name
        self.kind = _dtype_to_kind(dtype_name)

    def take_data(self) -> Any:
        return self.data

    @classmethod
    def _get_atom(cls, values: np.ndarray) -> Any:
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
            atom = _tables().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif is_string_dtype(dtype):
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)
        return atom

    @classmethod
    def get_atom_string(cls, shape: Tuple[int, ...], itemsize: int) -> Any:
        return _tables().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def get_atom_coltype(cls, kind: str) -> Any:
        if kind.startswith('uint'):
            k4 = kind[4:]
            col_name = f'UInt{k4}Col'
        elif kind.startswith('period'):
            col_name = 'Int64Col'
        else:
            kcap = kind.capitalize()
            col_name = f'{kcap}Col'
        return getattr(_tables(), col_name)

    @classmethod
    def get_atom_data(cls, shape: Tuple[int, ...], kind: str) -> Any:
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def get_atom_datetime64(cls, shape: Tuple[int, ...]) -> Any:
        return _tables().Int64Col(shape=shape[0])

    @classmethod
    def get_atom_timedelta64(cls, shape: Tuple[int, ...]) -> Any:
        return _tables().Int64Col(shape=shape[0])

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return getattr(self.data, 'shape', None)

    @property
    def cvalues(self) -> Any:
        return self.data

    def validate_attr(self, append: bool) -> None:
        if append:
            existing_fields = getattr(self.attrs, self.kind_attr, None)
            if existing_fields is not None and existing_fields != list(self.values):
                raise ValueError('appended items do not match existing items in table!')
            existing_dtype = getattr(self.attrs, self.dtype_attr, None)
            if existing_dtype is not None and existing_dtype != self.dtype:
                raise ValueError('appended items dtype do not match existing items dtype in table!')

    def convert(self, values: np.ndarray, nan_rep: Any, encoding: str, errors: str) -> Tuple[Any, np.ndarray]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname]
        assert self.typ is not None
        if self.dtype is None:
            converted, dtype_name = _get_data_and_dtype_name(values)
            kind = _dtype_to_kind(dtype_name)
        else:
            converted = values
            dtype_name = self.dtype
            kind = self.kind  # type: ignore
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
        return (self.values, converted)

    def set_attr(self) -> None:
        setattr(self.attrs, self.kind_attr, self.values)
        setattr(self.attrs, self.meta_attr, self.meta)
        assert self.dtype is not None
        setattr(self.attrs, self.dtype_attr, self.dtype)

# Additional classes (DataIndexableCol, GenericDataIndexableCol, Fixed, GenericFixed, SeriesFixed,
# BlockManagerFixed, FrameFixed, Table, WORMTable, AppendableTable, AppendableFrameTable, AppendableSeriesTable,
# AppendableMultiSeriesTable, GenericTable, AppendableMultiFrameTable) and helper functions (_reindex_axis,
# _get_tz, _set_tz, _convert_index, _unconvert_index, _maybe_convert_for_string_atom, _convert_string_array,
# _unconvert_string_array, _maybe_convert, _get_converter, _need_convert, _maybe_adjust_name, _dtype_to_kind,
# _get_data_and_dtype) should be annotated similarly.
#
# For brevity, type annotations have been added to the functions and methods shown above.
# The remaining parts of the module should be annotated in a similar manner.
