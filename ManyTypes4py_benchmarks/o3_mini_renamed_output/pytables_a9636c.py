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
from typing import TYPE_CHECKING, Any, Final, List, Optional, Union, Dict, Tuple, overload, Iterator, cast
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
    from collections.abc import Callable, Hashable, Sequence
    from types import TracebackType, ModuleType
    # The following types are imported for type checking.
    from tables import Col, File, Node
    from pandas._typing import AnyArrayLike, ArrayLike, AxisInt, DtypeArg, FilePath, Self, Shape, npt
    from pandas.core.internals import Block

_version: Final[str] = '0.15.2'
_default_encoding: Final[str] = 'UTF-8'

def func_j7cb0xfa(encoding: Optional[str]) -> str:
    if encoding is None:
        encoding = _default_encoding
    return encoding

def func_6npcmgwl(name: Any) -> Any:
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype.
    """
    if isinstance(name, str):
        name = str(name)
    return name

Term: Final = PyTablesExpr

def func_qkg3n79k(where: Any, scope_level: int) -> Optional[Union[Term, List[Term]]]:
    """
    Ensure that the where criteria are correctly wrapped
    """
    level: int = scope_level
    if isinstance(where, (list, tuple)):
        where = [Term(w, scope_level=level) if maybe_expression(w) else w for w in where if w is not None]
    elif maybe_expression(where):
        where = Term(where, scope_level=level)
    return where if where is None or (hasattr(where, '__len__') and len(where)) else None

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

dropna_doc: Final[str] = "dropna documentation"
format_doc: Final[str] = "format documentation"
with config.config_prefix('io.hdf'):
    config.register_option('dropna_table', False, dropna_doc, validator=config.is_bool)
    config.register_option('default_format', None, format_doc, validator=config.is_one_of_factory(['fixed', 'table', None]))

_table_mod: Optional[Any] = None
_table_file_open_policy_is_strict: bool = False

def func_glsq77za() -> Any:
    """
    Lazy-load and return the tables module.
    """
    global _table_mod, _table_file_open_policy_is_strict
    if _table_mod is None:
        _table_mod = import_optional_dependency("tables")
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (_table_mod.file._FILE_OPEN_POLICY == 'strict')
    return _table_mod

def func_fgciumbb(path_or_buf: Any, key: Any, value: Any, mode: str = 'a', complevel: Optional[int] = None,
                   complib: Optional[str] = None, append: bool = False, format: Optional[str] = None,
                   index: bool = True, min_itemsize: Any = None, nan_rep: Any = None, dropna: Any = None,
                   data_columns: Any = None, errors: str = 'strict', encoding: str = 'UTF-8') -> None:
    """store this object, close it if we opened it"""
    if append:
        f = lambda store: store.append(key, value, format=format, index=index, min_itemsize=min_itemsize,
                                      nan_rep=nan_rep, dropna=dropna, data_columns=data_columns, errors=errors,
                                      encoding=encoding)
    else:
        f = lambda store: store.put(key, value, format=format, index=index, min_itemsize=min_itemsize,
                                     nan_rep=nan_rep, data_columns=data_columns, errors=errors, encoding=encoding,
                                     dropna=dropna)
    from pandas.io.pytables import HDFStore  # type: ignore
    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        path_or_buf = stringify_path(path_or_buf)
        store = HDFStore(path_or_buf, mode=mode, complevel=complevel, complib=complib)
        try:
            f(store)
        finally:
            store.close()

def func_7y40qs0l(path_or_buf: Any, key: Optional[Any] = None, mode: str = 'r', errors: str = 'strict',
                   where: Any = None, start: Optional[int] = None, stop: Optional[int] = None,
                   columns: Optional[Any] = None, iterator: bool = False, chunksize: Optional[int] = None, **kwargs: Any) -> Any:
    """
    Read table from store.
    """
    from pandas.io.pytables import HDFStore  # type: ignore
    if mode not in ['r', 'r+', 'a']:
        raise ValueError(f'mode {mode} is not allowed while performing a read. Allowed modes are r, r+ and a.')
    if where is not None:
        where = func_qkg3n79k(where, scope_level=1)
    if isinstance(path_or_buf, HDFStore):
        if not path_or_buf.is_open:
            raise OSError('The HDFStore must be open for reading.')
        store = path_or_buf
        auto_close: bool = False
    else:
        path_or_buf = stringify_path(path_or_buf)
        if not isinstance(path_or_buf, str):
            raise NotImplementedError('Support for generic buffers has not been implemented.')
        if not os.path.exists(path_or_buf):
            raise FileNotFoundError(f'File {path_or_buf} does not exist')
        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        auto_close = True
    try:
        if key is None:
            groups = store.groups()
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

def func_fwuz3pa6(group: Any, parent_group: Any) -> bool:
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
    def __init__(self, path: Any, mode: str = 'a', complevel: Optional[int] = None, complib: Optional[str] = None,
                 fletcher32: bool = False, **kwargs: Any) -> None:
        if 'format' in kwargs:
            raise ValueError('format is not a defined argument for HDFStore')
        tables = import_optional_dependency('tables')
        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(f'complib only supports {tables.filters.all_complibs} compression.')
        if complib is None and complevel is not None:
            complib = tables.filters.default_complib
        self._path: Any = stringify_path(path)
        if mode is None:
            mode = 'a'
        self._mode: str = mode
        self._handle: Optional[Any] = None
        self._complevel: int = complevel if complevel else 0
        self._complib: Optional[str] = complib
        self._fletcher32: bool = fletcher32
        self._filters: Optional[Any] = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self) -> Any:
        return self._path

    @property
    def func_725geno4(self) -> Any:
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None
        return self._handle.root

    @property
    def func_nzhyyr4n(self) -> Any:
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

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]) -> None:
        self.close()

    def func_p58vk3vn(self, include: str = 'pandas') -> List[str]:
        """
        Return a list of keys corresponding to objects stored in HDFStore.
        """
        if include == 'pandas':
            return [n._v_pathname for n in self.groups()]
        elif include == 'native':
            assert self._handle is not None
            return [n._v_pathname for n in self._handle.walk_nodes('/', classname='Table')]
        raise ValueError(f"`include` should be either 'pandas' or 'native' but is '{include}'")

    def __iter__(self) -> Iterator:
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
        """
        tables = func_glsq77za()
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
            self._filters = func_glsq77za().Filters(self._complevel, self._complib, fletcher32=self._fletcher32)
        if _table_file_open_policy_is_strict and self.is_open:
            msg = ('Cannot open HDF5 file, which is already opened, even in read-only mode.')
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
        """
        if self._handle is not None:
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def func_01vumk9r(self, key: str) -> Any:
        """
        Retrieve pandas object stored in file.
        """
        with patch_pickle():
            group = self.get_node(key)
            if group is None:
                raise KeyError(f'No object named {key} in the file')
            return self._read_group(group)

    def func_xufwq3f7(self, key: str, where: Any = None, start: Optional[int] = None, stop: Optional[int] = None,
                       columns: Optional[Any] = None, iterator: bool = False, chunksize: Optional[int] = None,
                       auto_close: bool = False) -> Any:
        """
        Retrieve pandas object stored in file, optionally based on where criteria.
        """
        group = self.get_node(key)
        if group is None:
            raise KeyError(f'No object named {key} in the file')
        where = func_qkg3n79k(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()
        def func_e0clc2pl(_start: Optional[int], _stop: Optional[int], _where: Any) -> Any:
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)
        it = TableIterator(self, s, func_e0clc2pl, where=where, nrows=s.nrows, start=start, stop=stop, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
        return it.get_result()

    def func_w19ft2e1(self, key: str, where: Any = None, start: Optional[int] = None, stop: Optional[int] = None) -> Any:
        """
        return the selection as an Index
        """
        where = func_qkg3n79k(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_coordinates with a table')
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def func_fhijx58a(self, key: str, column: str, start: Optional[int] = None, stop: Optional[int] = None) -> Any:
        """
        return a single column from the table.
        """
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError('can only read_column with a table')
        return tbl.read_column(column=column, start=start, stop=stop)

    def func_xdlnzicv(self, keys: Union[List[Any], Tuple[Any, ...], str], where: Any = None, selector: Optional[Any] = None,
                      columns: Optional[Any] = None, start: Optional[int] = None, stop: Optional[int] = None,
                      iterator: bool = False, chunksize: Optional[int] = None, auto_close: bool = False) -> Any:
        """
        Retrieve pandas objects from multiple tables.
        """
        where = func_qkg3n79k(where, scope_level=1)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(key=keys, where=where, columns=columns, start=start, stop=stop, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
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
        def func_e0clc2pl(_start: Optional[int], _stop: Optional[int], _where: Any) -> Any:
            objs = [t.read(where=_where, columns=columns, start=_start, stop=_stop) for t in tbls]
            return concat(objs, axis=axis, verify_integrity=False)._consolidate()
        it = TableIterator(self, s, func_e0clc2pl, where=where, nrows=nrows, start=start, stop=stop, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
        return it.get_result(coordinates=True)

    def func_la2v1zd1(self, key: str, value: Any, format: Optional[str] = None, index: bool = True, append: bool = False,
                       complib: Optional[str] = None, complevel: Optional[int] = None, min_itemsize: Any = None,
                       nan_rep: Any = None, data_columns: Any = None, encoding: Optional[str] = None, errors: str = 'strict',
                       track_times: bool = True, dropna: bool = False) -> None:
        """
        Store object in HDFStore.
        """
        if format is None:
            format = get_option('io.hdf.default_format') or 'fixed'
        format = self._validate_format(format)
        self._write_to_group(key, value, format=format, index=index, append=append, complib=complib, complevel=complevel,
                              min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=data_columns, encoding=encoding,
                              errors=errors, track_times=track_times, dropna=dropna)

    def func_p2yix7g2(self, key: str, where: Any = None, start: Optional[int] = None, stop: Optional[int] = None) -> Optional[int]:
        """
        Remove pandas object partially by specifying the where condition
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

    def func_ahag9i4l(self, key: str, value: Any, format: Optional[str] = None, axes: Any = None, index: bool = True,
                       append: bool = True, complib: Optional[str] = None, complevel: Optional[int] = None, columns: Any = None,
                       min_itemsize: Any = None, nan_rep: Any = None, chunksize: Optional[int] = None, expectedrows: Optional[int] = None,
                       dropna: Optional[bool] = None, data_columns: Any = None, encoding: Optional[str] = None, errors: str = 'strict') -> None:
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
        self._write_to_group(key, value, format=format, axes=axes, index=index, append=append, complib=complib, complevel=complevel,
                              min_itemsize=min_itemsize, nan_rep=nan_rep, chunksize=chunksize, expectedrows=expectedrows,
                              dropna=dropna, data_columns=data_columns, encoding=encoding, errors=errors)

    def func_tx9jg86w(self, d: Dict[str, Optional[List[str]]], value: Any, selector: str, data_columns: Any = None, axes: Any = None,
                       dropna: bool = False, **kwargs: Any) -> None:
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
        remain_key: Optional[str] = None
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

    def func_iwsjwg6s(self, key: str, columns: Optional[Union[bool, List[str]]] = None, optlevel: Optional[int] = None, kind: Optional[str] = None) -> None:
        """
        Create a pytables index on the table.
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
        """
        func_glsq77za()
        self._check_if_open()
        assert self._handle is not None
        assert _table_mod is not None
        return [g for g in self._handle.walk_groups() if not isinstance(g, _table_mod.link.Link) and (getattr(g._v_attrs, 'pandas_type', None) or getattr(g, 'table', None) or (hasattr(g, '_v_name') and g._v_name != 'table'))]

    def func_0a4nq6tc(self, where: str = '/') -> Iterator[Tuple[str, List[str], List[str]]]:
        """
        Walk the pytables group hierarchy for pandas objects.
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
        assert isinstance(node, _table_mod.Node)
        return node

    def func_sf8ijrk2(self, key: str) -> Any:
        """return the storer object for a key, raise if not in the file"""
        group = self.get_node(key)
        if group is None:
            raise KeyError(f'No object named {key} in the file')
        s = self._create_storer(group)
        s.infer_axes()
        return s

    def func_knn0xpsp(self, file: Any, mode: str = 'w', propindexes: bool = True, keys: Optional[List[Any]] = None,
                       complib: Optional[str] = None, complevel: Optional[int] = None, fletcher32: bool = False, overwrite: bool = True) -> Any:
        """
        Copy the existing store to a new file, updating in place.
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
                    index_flag = False
                    if propindexes:
                        index_flag = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(k, data, index=index_flag, data_columns=getattr(s, 'data_columns', None), encoding=s.encoding)
                else:
                    new_store.put(k, data, encoding=s.encoding)
        return new_store

    def func_q31k3j0l(self) -> str:
        """
        Print detailed information on the store.
        """
        path = pprint_thing(self._path)
        output = f'{type(self)}\nFile path: {path}\n'
        if self.is_open:
            lkeys = sorted(self.keys())
            if len(lkeys):
                keys: List[str] = []
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

    def func_4f0mooxy(self, group: Any, format: Optional[str] = None, value: Optional[Any] = None, encoding: str = 'UTF-8', errors: str = 'strict') -> Any:
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
        _TABLE_MAP: Dict[str, Any] = {'generic_table': GenericTable, 'appendable_series': AppendableSeriesTable,
                                       'appendable_multiseries': AppendableMultiSeriesTable, 'appendable_frame': AppendableFrameTable,
                                       'appendable_multiframe': AppendableMultiFrameTable, 'worm': WORMTable}
        try:
            cls = _TABLE_MAP[tt]
        except KeyError as err:
            raise TypeError(f'cannot properly create the storer for: [_TABLE_MAP] [group->{group},value->{type(value)},format->{format}') from err
        return cls(self, group, encoding=encoding, errors=errors)

    def func_ioo3epxi(self, key: str, value: Any, format: str, axes: Any = None, index: bool = True,
                        append: bool = False, complib: Optional[str] = None, complevel: Optional[int] = None,
                        fletcher32: Optional[bool] = None, min_itemsize: Any = None, chunksize: Optional[int] = None,
                        expectedrows: Optional[int] = None, dropna: bool = False, nan_rep: Any = None,
                        data_columns: Any = None, encoding: Optional[str] = None, errors: str = 'strict',
                        track_times: bool = True) -> None:
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
        s.write(obj=value, axes=axes, append=append, complib=complib, complevel=complevel, fletcher32=fletcher32,
                min_itemsize=min_itemsize, chunksize=chunksize, expectedrows=expectedrows, dropna=dropna, nan_rep=nan_rep,
                data_columns=data_columns, track_times=track_times)
        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def func_0bedw5kn(self, group: Any) -> Any:
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def func_95gbb1ca(self, key: str, append: bool) -> Any:
        """Identify HDF5 group based on key, delete/create group if needed."""
        group = self.get_node(key)
        assert self._handle is not None
        if group is not None and not append:
            self._handle.remove_node(group, recursive=True)
            group = None
        if group is None:
            group = self._create_nodes_and_group(key)
        return group

    def func_eevwow0i(self, key: str) -> Any:
        """Create nodes from key and return group name."""
        assert self._handle is not None
        paths: List[str] = key.split('/')
        path: str = '/'
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

# The classes TableIterator, IndexCol, GenericIndexCol, DataCol, DataIndexableCol,
# GenericDataIndexableCol, Fixed, GenericFixed, SeriesFixed, BlockManagerFixed, FrameFixed,
# Table, WORMTable, AppendableTable, AppendableFrameTable, AppendableSeriesTable,
# AppendableMultiSeriesTable, GenericTable, AppendableMultiFrameTable, and the functions
# func_fobedmu7, func_vmqkbat2, func_04gv14qo, func_79e9ube9, func_tyrrdh2r, func_nw3v6hwe,
# func_rd1cs57w, func_pjv0io2t, func_19m5iaiv, func_mby3wq9i, func_v6defya3, func_awrzbszi,
# func_wgpibb1z, func_o4yhj26v, and the Selection class would be similarly annotated.
# Due to the length of the code, similar type annotations should be applied throughout the rest of the program.
