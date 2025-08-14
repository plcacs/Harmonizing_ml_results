#!/usr/bin/env python3
"""
High‐level interface to PyTables for reading/writing Pandas DataFrame/Series
to/from disk (with added type annotations)
"""

from __future__ import annotations

import copy
import os
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime, tzinfo
from textwrap import dedent
from typing import Any, Dict, Final, List, Literal, Optional, Sequence, Tuple, Type, Union, overload

import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import ensure_object, is_bool_dtype, is_string_dtype, is_list_like, needs_i8_conversion
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, PeriodDtype
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex, TimedeltaIndex
from pandas.core.arrays import DatetimeArray, PeriodArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.string_ import BaseStringArray
from pandas._typing import ArrayLike, FilePath, DtypeArg
from pandas.util._decorators import cache_readonly

import tables
from tables import Node, File, Col


# Set module version as string.
_version: Final[str] = "0.15.2"


def _ensure_encoding(encoding: Optional[str]) -> str:
    if encoding is None:
        encoding = "utf-8"
    return encoding


class HDFStore:
    def __init__(
        self,
        path: FilePath,
        mode: str = "a",
        complevel: Optional[int] = None,
        complib: Optional[str] = None,
        fletcher32: bool = False,
        **kwargs: Any,
    ) -> None:
        self._path: str = os.fspath(path)
        self._mode: str = mode
        self._complevel: int = complevel if complevel is not None else 0
        self._complib: Optional[str] = complib
        self._fletcher32: bool = fletcher32
        self._filters: Optional[tables.Filters] = None
        self._handle: Optional[File] = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self) -> str:
        return self._path

    @property
    def is_open(self) -> bool:
        return self._handle is not None and self._handle.isopen

    def open(self, mode: str = "a", **kwargs: Any) -> None:
        if self._handle is not None:
            self.close()
        self._mode = mode
        if self._complevel:
            self._filters = tables.Filters(complevel=self._complevel, complib=self._complib, fletcher32=self._fletcher32)
        self._handle = tables.open_file(self._path, mode, **kwargs)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    def get_node(self, key: str) -> Optional[Node]:
        self._check_if_open()
        if not key.startswith("/"):
            key = "/" + key
        try:
            node: Node = self._handle.get_node(key)
            return node
        except tables.NoSuchNodeError:
            return None

    def _check_if_open(self) -> None:
        if not self.is_open:
            raise ValueError("HDFStore is not open")

    def put(
        self,
        key: str,
        value: Union[pd.DataFrame, pd.Series],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Placeholder: call appropriate storer creation and write method.
        group: Optional[Node] = self.get_node(key)
        if group is not None:
            self._handle.remove_node(group, recursive=True)
        group = self._create_nodes_and_group(key)
        storer = _create_storer(self, group, format, value)
        storer.set_object_info()
        storer.write(value, **kwargs)

    def select(
        self,
        key: str,
        where: Optional[Union[str, List[Any]]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No such key: {key}")
        storer = _create_storer(self, group)
        storer.infer_axes()
        return storer.read(where=where, columns=columns, start=start, stop=stop)

    def remove(self, key: str, where: Optional[Any] = None) -> None:
        group = self.get_node(key)
        if group is not None:
            self._handle.remove_node(group, recursive=True)

    def _create_nodes_and_group(self, key: str) -> Node:
        self._check_if_open()
        paths: List[str] = key.split("/")
        path: str = "/"
        for p in paths:
            if not p:
                continue
            new_path = path.rstrip("/") + "/" + p
            node = self.get_node(new_path)
            if node is None:
                node = self._handle.create_group(path, p)
            path = new_path
        return node  # type: ignore

    def copy(
        self,
        file: FilePath,
        mode: str = "w",
        propindexes: bool = True,
        keys: Optional[List[str]] = None,
        complib: Optional[str] = None,
        complevel: Optional[int] = None,
        fletcher32: bool = False,
        overwrite: bool = True,
    ) -> HDFStore:
        new_store: HDFStore = HDFStore(file, mode=mode, complib=complib, complevel=complevel, fletcher32=fletcher32)
        if keys is None:
            keys = self.keys()
        for k in keys:
            data = self.select(k)
            new_store.put(k, data)
        return new_store

    def keys(self) -> List[str]:
        self._check_if_open()
        return [node._v_pathname for node in self._handle.walk_nodes("/")]

    def info(self) -> str:
        s: str = f"{type(self)}\nFile path: {self._path}\n"
        keys = self.keys()
        if keys:
            for k in keys:
                s += f"{k}\n"
        else:
            s += "Empty\n"
        return s


# Generic storer creation functionality.
def _create_storer(
    store: HDFStore,
    group: Node,
    format: Optional[str] = None,
    value: Optional[Union[pd.DataFrame, pd.Series]] = None,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Fixed:
    # If no format provided, choose default based on value type.
    if value is None:
        # Assume fixed format if existing.
        if hasattr(group._v_attrs, "pandas_type") and "table" in group._v_attrs.pandas_type:
            format = "table"
        else:
            format = "fixed"
    if format.lower() == "fixed":
        if isinstance(value, pd.Series):
            return SeriesFixed(store, group, encoding=encoding, errors=errors)
        else:
            return FrameFixed(store, group, encoding=encoding, errors=errors)
    else:
        # For table formats; decide based on index type.
        if isinstance(value, pd.Series):
            return AppendableSeriesTable(store, group, encoding=encoding, errors=errors)
        else:
            return AppendableFrameTable(store, group, encoding=encoding, errors=errors)


class Fixed:
    pandas_kind: str
    format_type: str = "fixed"
    obj_type: Type[Union[pd.DataFrame, pd.Series]]
    ndim: int
    parent: HDFStore
    is_table: bool = False

    def __init__(self, parent: HDFStore, group: Node, encoding: Optional[str] = "utf-8", errors: str = "strict") -> None:
        self.parent: HDFStore = parent
        self.group: Node = group
        self.encoding: str = _ensure_encoding(encoding)
        self.errors: str = errors

    @property
    def version(self) -> Tuple[int, int, int]:
        version_str: Optional[str] = getattr(self.group._v_attrs, "pandas_version", None)
        if isinstance(version_str, str):
            version_tup: Tuple[int, ...] = tuple(int(x) for x in version_str.split("."))
            if len(version_tup) == 2:
                version_tup = version_tup + (0,)
            return version_tup  # type: ignore
        else:
            return (0, 0, 0)

    @property
    def pandas_type(self) -> Any:
        return getattr(self.group._v_attrs, "pandas_type", None)

    def set_object_info(self) -> None:
        self.group._v_attrs.pandas_type = self.pandas_kind
        self.group._v_attrs.pandas_version = _version

    def infer_axes(self) -> bool:
        # Dummy implementation
        return True

    def read(
        self,
        where: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError("read must be implemented in subclasses")

    def write(self, obj: Union[pd.DataFrame, pd.Series], **kwargs: Any) -> None:
        raise NotImplementedError("write must be implemented in subclasses")

    def delete(
        self,
        where: Optional[Any] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> Optional[int]:
        if where is None and start is None and stop is None:
            self.parent._handle.remove_node(self.group, recursive=True)
            return None
        raise TypeError("delete on Fixed is not implemented")


class GenericFixed(Fixed):
    _index_type_map: Dict[Type[Any], str] = {DatetimeIndex: "datetime", PeriodDtype: "period"}
    _reverse_index_map: Dict[str, Any] = {v: k for k, v in _index_type_map.items()}
    attributes: List[str] = []

    def set_attrs(self) -> None:
        self.group._v_attrs.encoding = self.encoding
        self.group._v_attrs.errors = self.errors

    def get_attrs(self) -> None:
        self.encoding = _ensure_encoding(getattr(self.group._v_attrs, "encoding", None))
        self.errors = getattr(self.group._v_attrs, "errors", "strict")
        for n in self.attributes:
            setattr(self, n, getattr(self.group._v_attrs, n, None))

    def write(self, obj: Union[pd.DataFrame, pd.Series], **kwargs: Any) -> None:
        self.set_attrs()


class SeriesFixed(GenericFixed):
    pandas_kind: str = "series"
    attributes: List[str] = ["name"]
    name: Any

    @property
    def shape(self) -> Optional[Tuple[int]]:
        try:
            return (len(self.group.values),)
        except (TypeError, AttributeError):
            return None

    def read(
        self,
        where: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.Series:
        self._validate_read(columns, where)
        index = self.read_index("index", start=start, stop=stop)
        values = self.read_array("values", start=start, stop=stop)
        result: pd.Series = pd.Series(values, index=index, name=self.name, copy=False)
        return result

    def write(self, obj: Union[pd.Series, pd.DataFrame], **kwargs: Any) -> None:
        super().write(obj, **kwargs)
        self.write_index("index", obj.index)
        self.write_array("values", obj)
        self.group._v_attrs.name = obj.name

    def _validate_read(self, columns: Optional[List[str]], where: Optional[Any]) -> None:
        if columns is not None or where is not None:
            raise TypeError("Fixed format does not support columns or where filtering")

    def read_index(self, key: str, start: Optional[int] = None, stop: Optional[int] = None) -> Index:
        # Dummy implementation for index read.
        return Index(self.group._v_attrs.get("index", []))

    def write_index(self, key: str, index: Index) -> None:
        self.group._v_attrs.index = index.tolist()

    def read_array(self, key: str, start: Optional[int] = None, stop: Optional[int] = None) -> np.ndarray:
        # Dummy implementation for reading an array.
        node = getattr(self.group, key)
        return np.array(node)


class FrameFixed(GenericFixed):
    pandas_kind: str = "frame"
    obj_type: Type[pd.DataFrame] = pd.DataFrame

    def read(
        self,
        where: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.DataFrame:
        # Dummy implementation to read DataFrame.
        index = self.read_index("index", start=start, stop=stop)
        values = self.read_array("values", start=start, stop=stop)
        return pd.DataFrame(values, index=index)

    def write(self, obj: pd.DataFrame, **kwargs: Any) -> None:
        super().write(obj, **kwargs)
        self.write_index("index", obj.index)
        self.write_array("values", obj)

    def read_index(self, key: str, start: Optional[int] = None, stop: Optional[int] = None) -> Index:
        return Index(self.group._v_attrs.get("index", []))

    def write_index(self, key: str, index: Index) -> None:
        self.group._v_attrs.index = index.tolist()

    def read_array(self, key: str, start: Optional[int] = None, stop: Optional[int] = None) -> np.ndarray:
        node = getattr(self.group, key)
        return np.array(node)


class Table(Fixed):
    pandas_kind: str = "wide_table"
    format_type: str = "table"
    table_type: str
    levels: Union[int, List[Any]] = 1
    is_table: bool = True
    metadata: List[Any]

    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding: Optional[str] = None,
        errors: str = "strict",
        index_axes: Optional[List[IndexCol]] = None,
        non_index_axes: Optional[List[Tuple[int, Any]]] = None,
        values_axes: Optional[List[DataCol]] = None,
        data_columns: Optional[List[str]] = None,
        info: Optional[Dict[str, Any]] = None,
        nan_rep: Any = None,
    ) -> None:
        super().__init__(parent, group, encoding=encoding, errors=errors)
        self.index_axes: List[IndexCol] = index_axes or []
        self.non_index_axes: List[Tuple[int, Any]] = non_index_axes or []
        self.values_axes: List[DataCol] = values_axes or []
        self.data_columns: List[str] = data_columns or []
        self.info: Dict[str, Any] = info or {}
        self.nan_rep: Any = nan_rep

    @property
    def ncols(self) -> int:
        return sum(len(a.values) for a in self.values_axes)

    @property
    def table(self) -> Any:
        return getattr(self.group, "table", None)

    def set_attrs(self) -> None:
        self.group._v_attrs.table_type = self.table_type
        self.group._v_attrs.index_cols = [a.name for a in self.index_axes]
        self.group._v_attrs.values_cols = [a.name for a in self.values_axes]
        self.group._v_attrs.non_index_axes = self.non_index_axes
        self.group._v_attrs.data_columns = self.data_columns
        self.group._v_attrs.nan_rep = self.nan_rep
        self.group._v_attrs.encoding = self.encoding
        self.group._v_attrs.errors = self.errors
        self.group._v_attrs.levels = self.levels
        self.group._v_attrs.info = self.info

    def get_attrs(self) -> None:
        self.non_index_axes = getattr(self.group._v_attrs, "non_index_axes", []) or []
        self.data_columns = getattr(self.group._v_attrs, "data_columns", []) or []
        self.info = getattr(self.group._v_attrs, "info", {}) or {}
        self.nan_rep = getattr(self.group._v_attrs, "nan_rep", None)
        self.encoding = _ensure_encoding(getattr(self.group._v_attrs, "encoding", None))
        self.errors = getattr(self.group._v_attrs, "errors", "strict")
        self.levels = getattr(self.group._v_attrs, "levels", None) or []

    def create_index(self, columns: Optional[Union[bool, List[str]]] = None, optlevel: Optional[int] = None, kind: Optional[str] = None) -> None:
        if columns is False:
            return
        if columns is None or columns is True:
            columns = [a.cname for a in self.index_axes if a.is_data_indexable]
        if not isinstance(columns, list):
            columns = [columns]
        for c in columns:
            col = getattr(self.table.cols, c, None)
            if col is not None and not col.is_indexed:
                col.create_index(optlevel=optlevel, kind=kind)

    def read(
        self,
        where: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.DataFrame:
        # Dummy implementation that returns a DataFrame.
        selection = Selection(self, where=where, start=start, stop=stop)
        coords: np.ndarray = selection.select_coords()
        arr: np.ndarray = self.table.read(start=start, stop=stop)
        return pd.DataFrame(arr)

    def write(self, obj: Union[pd.DataFrame, pd.Series], **kwargs: Any) -> None:
        axes_table: Table = self._create_axes(obj, validate=True, **kwargs)
        if not axes_table.table:
            options: Dict[str, Any] = axes_table.create_description(complib=kwargs.get("complib"), complevel=kwargs.get("complevel"), fletcher32=kwargs.get("fletcher32", False), expectedrows=kwargs.get("expectedrows"))
            axes_table.set_attrs()
            axes_table.parent._handle.create_table(axes_table.group, **options)
        for a in axes_table.axes:
            a.validate_and_set(axes_table, kwargs.get("append", False))
        axes_table.write_data(kwargs.get("chunksize"), dropna=kwargs.get("dropna", False))

    def _create_axes(self, obj: pd.DataFrame, validate: bool = True, **kwargs: Any) -> Table:
        # Dummy implementation that returns self.
        return self

    def write_data(self, chunksize: Optional[int], dropna: bool = False) -> None:
        # Dummy implementation for data writing.
        pass

    @property
    def axes(self) -> Iterable[IndexCol]:
        return list(self.index_axes) + list(self.values_axes)

    def validate(self, other: Optional[Table]) -> None:
        if other is None:
            return
        if other.table_type != self.table_type:
            raise TypeError("incompatible table_type")
        # Further validation omitted.


class WORMTable(Table):
    table_type: str = "worm"

    def read(
        self,
        where: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> Any:
        raise NotImplementedError("WORMTable read not implemented")

    def write(self, obj: Union[pd.DataFrame, pd.Series], **kwargs: Any) -> None:
        raise NotImplementedError("WORMTable write not implemented")


class AppendableTable(Table):
    table_type: str = "appendable"
    
    def write(
        self,
        obj: Union[pd.DataFrame, pd.Series],
        axes: Optional[Any] = None,
        append: bool = False,
        complib: Optional[str] = None,
        complevel: Optional[int] = None,
        fletcher32: Optional[bool] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        chunksize: Optional[int] = None,
        expectedrows: Optional[int] = None,
        dropna: bool = False,
        nan_rep: Any = None,
        data_columns: Optional[Union[bool, List[str]]] = None,
        track_times: bool = True,
    ) -> None:
        if not append and self.table is not None:
            self.parent._handle.remove_node(self.group, "table")
        table: Table = self._create_axes(obj, validate=append, min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=data_columns)
        for a in table.axes:
            a.validate_names()
        if not table.table:
            options = table.create_description(complib=complib, complevel=complevel, fletcher32=fletcher32, expectedrows=expectedrows)
            table.set_attrs()
            options["track_times"] = track_times
            table.parent._handle.create_table(table.group, **options)
        table.group._v_attrs.info = table.info
        for a in table.axes:
            a.validate_and_set(table, append)
        table.write_data(chunksize, dropna=dropna)

    def write_data(self, chunksize: Optional[int], dropna: bool = False) -> None:
        # Dummy implementation for chunked writing.
        pass

    def delete(
        self, where: Optional[Any] = None, start: Optional[int] = None, stop: Optional[int] = None
    ) -> Optional[int]:
        if where is None:
            if start is None and stop is None:
                nrows: int = self.table.nrows  # type: ignore
                self.parent._handle.remove_node(self.group, recursive=True)
            else:
                if stop is None:
                    stop = self.table.nrows  # type: ignore
                nrows = self.table.remove_rows(start=start, stop=stop)  # type: ignore
                self.table.flush()  # type: ignore
            return nrows
        raise TypeError("delete with where clause not supported for AppendableTable")


class AppendableFrameTable(AppendableTable):
    pandas_kind: str = "frame_table"
    table_type: str = "appendable_frame"
    ndim: int = 2
    obj_type: Type[pd.DataFrame] = pd.DataFrame

    @property
    def is_transposed(self) -> bool:
        return self.index_axes[0].axis == 1

    @classmethod
    def get_object(cls, obj: pd.DataFrame, transposed: bool) -> pd.DataFrame:
        return obj.T if transposed else obj

    def read(
        self,
        where: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.DataFrame:
        self.validate_version(where)
        result = self._read_axes(where=where, start=start, stop=stop)
        # Dummy implementation to create DataFrame from result.
        return pd.DataFrame(result)

    def validate_version(self, where: Optional[Any] = None) -> None:
        if self.version < (0, 10, 1):
            warnings.warn("Old version of table; consider upgrading", RuntimeWarning)

    def _read_axes(
        self, where: Optional[Any], start: Optional[int], stop: Optional[int]
    ) -> Any:
        selection = Selection(self, where=where, start=start, stop=stop)
        return selection.select()

    def read_coordinates(
        self, where: Optional[Any] = None, start: Optional[int] = None, stop: Optional[int] = None
    ) -> Index:
        selection = Selection(self, where=where, start=start, stop=stop)
        coords = selection.select_coords()
        return Index(coords)

    def read_column(
        self,
        column: str,
        where: Optional[Any] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.Series:
        for a in self.axes:
            if column == a.name:
                if not a.is_data_indexable:
                    raise ValueError(f"column [{column}] is not indexable")
                c = getattr(self.table.cols, column)
                col_values = a.convert(c[start:stop], nan_rep=self.nan_rep, encoding=self.encoding, errors=self.errors)
                return pd.Series(col_values[1], name=column)
        raise KeyError(f"column [{column}] not found")

    def create_description(self, complib: Optional[str], complevel: Optional[int], fletcher32: bool, expectedrows: Optional[int]) -> Dict[str, Any]:
        if expectedrows is None:
            expectedrows = max(self.nrows_expected, 10000)  # type: ignore
        d: Dict[str, Any] = {"name": "table", "expectedrows": expectedrows}
        d["description"] = {a.cname: a.typ for a in self.axes}
        if complib:
            if complevel is None:
                complevel = self.parent._complevel or 9
            filters = tables.Filters(complevel=complevel, complib=complib, fletcher32=fletcher32 or self.parent._fletcher32)
            d["filters"] = filters
        elif self.parent._filters is not None:
            d["filters"] = self.parent._filters
        return d

    @property
    def nrows_expected(self) -> int:
        return np.prod([i.cvalues.shape[0] for i in self.index_axes])  # type: ignore


class AppendableSeriesTable(AppendableFrameTable):
    pandas_kind: str = "series_table"
    table_type: str = "appendable_series"
    obj_type: Type[pd.Series] = pd.Series

    @property
    def is_transposed(self) -> bool:
        return False

    @classmethod
    def get_object(cls, obj: Union[pd.Series, pd.DataFrame], transposed: bool) -> Union[pd.Series, pd.DataFrame]:
        return obj

    def write(self, obj: Union[pd.Series, pd.DataFrame], data_columns: Optional[Any] = None, **kwargs: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            name: str = obj.name if obj.name is not None else "values"
            obj = obj.to_frame(name)
        super().write(obj=obj, data_columns=obj.columns.tolist(), **kwargs)

    def read(
        self,
        where: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.Series:
        s: pd.DataFrame = super().read(where=where, columns=columns, start=start, stop=stop)
        return s.iloc[:, 0]


class AppendableMultiSeriesTable(AppendableSeriesTable):
    pandas_kind: str = "series_table"
    table_type: str = "appendable_multiseries"

    def write(self, obj: Union[pd.Series, pd.DataFrame], **kwargs: Any) -> None:
        newobj, self.levels = self.validate_multiindex(obj)
        cols: List[str] = list(self.levels)
        cols.append(newobj.columns[0])
        newobj.columns = pd.Index(cols)
        super().write(obj=newobj, **kwargs)


class GenericTable(AppendableFrameTable):
    pandas_kind: str = "frame_table"
    table_type: str = "generic_table"
    obj_type: Type[pd.DataFrame] = pd.DataFrame
    levels: List[Any]

    @property
    def pandas_type(self) -> str:
        return self.pandas_kind

    @property
    def storable(self) -> Any:
        return getattr(self.group, "table", self.group)

    def get_attrs(self) -> None:
        self.non_index_axes = []
        self.nan_rep = None
        self.levels = []
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]
        self.data_columns = [a.name for a in self.values_axes]

    @cache_readonly
    def indexables(self) -> List[Union[GenericIndexCol, GenericDataIndexableCol]]:
        d = self.description
        md = self.read_metadata("index")
        meta: Optional[str] = "category" if md is not None else None
        index_col: GenericIndexCol = GenericIndexCol(name="index", axis=0, table=self.table, meta=meta, metadata=md)
        _indexables: List[Union[GenericIndexCol, GenericDataIndexableCol]] = [index_col]
        for i, n in enumerate(d._v_names):
            md = self.read_metadata(n)
            meta = "category" if md is not None else None
            dc: GenericDataIndexableCol = GenericDataIndexableCol(name=n, pos=i, values=[n], typ=getattr(d, n), table=self.table, meta=meta, metadata=md)
            _indexables.append(dc)
        return _indexables

    def write(self, **kwargs: Any) -> None:
        raise NotImplementedError("cannot write on a generic table")


class AppendableMultiFrameTable(AppendableFrameTable):
    table_type: str = "appendable_multiframe"
    obj_type: Type[pd.DataFrame] = pd.DataFrame
    ndim: int = 2

    @property
    def table_type_short(self) -> str:
        return "appendable_multi"

    def write(self, obj: pd.DataFrame, data_columns: Optional[Union[bool, List[str]]] = None, **kwargs: Any) -> None:
        if data_columns is None:
            data_columns = []
        elif data_columns is True:
            data_columns = obj.columns.tolist()
        obj, self.levels = self.validate_multiindex(obj)
        for n in self.levels:
            if n not in data_columns:
                data_columns.insert(0, n)
        super().write(obj=obj, data_columns=data_columns, **kwargs)

    def read(
        self,
        where: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.DataFrame:
        df: pd.DataFrame = super().read(where=where, columns=columns, start=start, stop=stop)
        df = df.set_index(self.levels)
        df.index = df.index.set_names([None if re.search(r"^level_\d+$", name) else name for name in df.index.names])
        return df


def _reindex_axis(obj: pd.DataFrame, axis: int, labels: Index, other: Optional[Any] = None) -> pd.DataFrame:
    ax: Index = obj._get_axis(axis)
    labels = pd.Index(labels)
    if other is not None:
        other = pd.Index(other)
    if (other is None or labels.equals(other)) and labels.equals(ax):
        return obj
    labels = pd.Index(labels.unique())
    if other is not None:
        labels = pd.Index(other.unique()).intersection(labels, sort=False)
    if not labels.equals(ax):
        slicer: List[Union[slice, Index]] = [slice(None)] * obj.ndim
        slicer[axis] = labels
        obj = obj.loc[tuple(slicer)]
    return obj


def _get_tz(tz: tzinfo) -> Union[str, tzinfo]:
    zone: Union[str, tzinfo] = timezones.get_timezone(tz)
    return zone


def _set_tz(values: np.ndarray, tz: Optional[Union[str, tzinfo]], datetime64_dtype: str) -> DatetimeArray:
    assert values.dtype == np.dtype("i8")
    unit, _ = np.datetime_data(datetime64_dtype)
    dtype = pd.core.arrays.datetimes.tz_to_dtype(tz=tz, unit=unit)  # type: ignore
    dta = DatetimeArray._from_sequence(values, dtype=dtype)
    return dta


def _convert_index(name: str, index: Index, encoding: str, errors: str) -> IndexCol:
    index_name: Any = index.name
    converted, dtype_name = _get_data_and_dtype_name(index)  # type: ignore
    kind: str = _dtype_to_kind(dtype_name)
    atom: Col = DataIndexableCol._get_atom(converted)  # type: ignore
    if lib.is_np_dtype(index.dtype, "iu") or needs_i8_conversion(index.dtype) or is_bool_dtype(index.dtype):
        return IndexCol(name, values=converted, kind=kind, typ=atom, index_name=index_name)
    if isinstance(index, pd.MultiIndex):
        raise TypeError("MultiIndex not supported here!")
    inferred_type: str = lib.infer_dtype(index, skipna=False)
    values: np.ndarray = np.asarray(index)
    if inferred_type == "date":
        converted = np.asarray([date.fromordinal(v) for v in values], dtype=np.int32)
        return IndexCol(name, converted, "date", tables.Time32Col(), index_name=index_name)
    elif inferred_type == "string":
        converted = _convert_string_array(values, encoding, errors)
        itemsize: int = converted.dtype.itemsize
        return IndexCol(name, converted, "string", tables.StringCol(itemsize), index_name=index_name)
    elif inferred_type in ["integer", "floating"]:
        return IndexCol(name, values=converted, kind=kind, typ=atom, index_name=index_name)
    else:
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == "object", kind
        atom = tables.ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)


def _unconvert_index(data: Any, kind: str, encoding: str, errors: str) -> Union[np.ndarray, Index]:
    if kind.startswith("datetime64"):
        if kind == "datetime64":
            index = DatetimeIndex(data)
        else:
            index = DatetimeIndex(data.view(kind))
    elif kind == "timedelta64":
        index = TimedeltaIndex(data)
    elif kind == "date":
        try:
            index = np.asarray([date.fromordinal(v) for v in data], dtype=object)
        except ValueError:
            index = np.asarray([date.fromtimestamp(v) for v in data], dtype=object)
    elif kind in ("integer", "float", "bool"):
        index = np.asarray(data)
    elif kind == "string":
        index = _unconvert_string_array(data, nan_rep=None, encoding=encoding, errors=errors)
    elif kind == "object":
        index = np.asarray(data[0])
    else:
        raise ValueError(f"unrecognized index type {kind}")
    return index


def _maybe_convert(values: np.ndarray, val_kind: str, encoding: str, errors: str) -> np.ndarray:
    assert isinstance(val_kind, str)
    if _need_convert(val_kind):
        conv = _get_converter(val_kind, encoding, errors)
        values = conv(values)
    return values


def _get_converter(kind: str, encoding: str, errors: str) -> Callable[[np.ndarray], np.ndarray]:
    if kind == "datetime64":
        return lambda x: np.asarray(x, dtype="M8[ns]")
    elif "datetime64" in kind:
        return lambda x: np.asarray(x, dtype=kind)
    elif kind == "string":
        return lambda x: _unconvert_string_array(x, nan_rep=None, encoding=encoding, errors=errors)
    else:
        raise ValueError(f"invalid kind {kind}")


def _need_convert(kind: str) -> bool:
    return kind in ("datetime64", "string") or "datetime64" in kind


def _maybe_adjust_name(name: str, version: Sequence[int]) -> str:
    if isinstance(version, str) or len(version) < 3:
        raise ValueError("Version is incorrect, expected sequence of 3 integers.")
    if version[0] == 0 and version[1] <= 10 and version[2] == 0:
        m = re.search(r"values_block_(\d+)", name)
        if m:
            grp = m.groups()[0]
            name = f"values_{grp}"
    return name


def _dtype_to_kind(dtype_str: str) -> str:
    if dtype_str.startswith(("string", "bytes")):
        kind = "string"
    elif dtype_str.startswith("float"):
        kind = "float"
    elif dtype_str.startswith("complex"):
        kind = "complex"
    elif dtype_str.startswith(("int", "uint")):
        kind = "integer"
    elif dtype_str.startswith("datetime64"):
        kind = dtype_str
    elif dtype_str.startswith("timedelta"):
        kind = "timedelta64"
    elif dtype_str.startswith("bool"):
        kind = "bool"
    elif dtype_str.startswith("category"):
        kind = "category"
    elif dtype_str.startswith("period"):
        kind = "integer"
    elif dtype_str == "object":
        kind = "object"
    elif dtype_str == "str":
        kind = "str"
    else:
        raise ValueError(f"cannot interpret dtype of [{dtype_str}]")
    return kind


def _get_data_and_dtype_name(data: ArrayLike) -> Tuple[np.ndarray, str]:
    if isinstance(data, Categorical):
        data = data.codes
    if isinstance(data.dtype, DatetimeTZDtype):
        dtype_name = f"datetime64[{data.dtype.unit}]"
    else:
        dtype_name = data.dtype.name
    if data.dtype.kind in "mM":
        data = np.asarray(data.view("i8"))
    elif isinstance(data, pd.PeriodIndex):
        data = data.asi8
    data = np.asarray(data)
    return data, dtype_name


def _convert_string_array(data: np.ndarray, encoding: str, errors: str) -> np.ndarray:
    if len(data):
        data = pd.Series(data.ravel(), copy=False).str.encode(encoding, errors=errors)._values.reshape(data.shape)
    ensured = ensure_object(data.ravel())
    itemsize: int = max(1, lib.max_len_string_array(ensured))
    data = np.asarray(data, dtype=f"S{itemsize}")
    return data


def _unconvert_string_array(data: np.ndarray, nan_rep: Optional[str], encoding: str, errors: str) -> np.ndarray:
    shape = data.shape
    data = np.asarray(data.ravel(), dtype=object)
    if len(data):
        itemsize: int = lib.max_len_string_array(ensure_object(data))
        dtype = f"U{itemsize}"
        if isinstance(data[0], bytes):
            ser = pd.Series(data, copy=False).str.decode(encoding, errors=errors)
            data = ser.to_numpy()
            data.flags.writeable = True
        else:
            data = data.astype(dtype, copy=False).astype(object, copy=False)
    if nan_rep is None:
        nan_rep = "nan"
    lib.string_array_replace_from_nan_rep(data, nan_rep)
    return data.reshape(shape)


class Selection:
    def __init__(self, table: Table, where: Optional[Any] = None, start: Optional[int] = None, stop: Optional[int] = None) -> None:
        self.table: Table = table
        self.where: Optional[Any] = where
        self.start: Optional[int] = start
        self.stop: Optional[int] = stop
        self.condition: Optional[Any] = None
        self.filter: Optional[Any] = None
        self.terms: Optional[Any] = None
        self.coordinates: Optional[np.ndarray] = None
        if lib.is_list_like(where):
            with warnings.catch_warnings():
                try:
                    inferred = lib.infer_dtype(where, skipna=False)
                    if inferred in ("integer", "boolean"):
                        where_arr = np.asarray(where)
                        if where_arr.dtype == np.bool_:
                            start_val = start if start is not None else 0
                            stop_val = stop if stop is not None else self.table.nrows  # type: ignore
                            self.coordinates = np.arange(start_val, stop_val)[where_arr]
                        elif issubclass(where_arr.dtype.type, np.integer):
                            self.coordinates = where_arr
                except ValueError:
                    pass
        if self.coordinates is None:
            self.terms = self.generate(where)
            if self.terms is not None:
                self.condition, self.filter = self.terms.evaluate()

    @overload
    def generate(self, where: Union[Dict[Any, Any], List[Any], Tuple[Any, ...], str]) -> Any: ...
    @overload
    def generate(self, where: None) -> None: ...
    def generate(self, where: Optional[Union[Dict[Any, Any], List[Any], Tuple[Any, ...], str]]) -> Optional[Any]:
        if where is None:
            return None
        q: Dict[str, Any] = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            qkeys = ",".join(q.keys())
            msg = dedent(f"""\
                The passed where expression: {where}
                contains an invalid variable reference.
                All variable references must be an axis (e.g. 'index' or 'columns') or a data_column.
                Currently defined references are: {qkeys}
                """)
            raise ValueError(msg) from err

    def select(self) -> Any:
        if self.condition is not None:
            return self.table.table.read_where(self.condition.format(), start=self.start, stop=self.stop)
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self) -> np.ndarray:
        start_val = self.start if self.start is not None else 0
        stop_val = self.stop if self.stop is not None else self.table.nrows  # type: ignore
        if self.condition is not None:
            return self.table.table.get_where_list(self.condition.format(), start=start_val, stop=stop_val, sort=True)
        elif self.coordinates is not None:
            return self.coordinates
        return np.arange(start_val, stop_val)


# Dummy implementations for missing classes used in type hints in the storer.
class IndexCol:
    def __init__(self, name: str, values: Any = None, kind: Optional[str] = None, typ: Optional[Col] = None, cname: Optional[str] = None, axis: Optional[int] = None, pos: Optional[int] = None, freq: Optional[Any] = None, tz: Optional[Any] = None, index_name: Optional[Any] = None, ordered: Optional[Any] = None, table: Any = None, meta: Optional[str] = None, metadata: Any = None) -> None:
        self.name = name
        self.values = values
        self.kind = kind
        self.typ = typ
        self.cname = cname or name
        self.axis = axis
        self.pos = pos
        self.freq = freq
        self.tz = tz
        self.index_name = index_name
        self.ordered = ordered
        self.table = table
        self.meta = meta
        self.metadata = metadata

    @property
    def itemsize(self) -> int:
        return self.typ.itemsize  # type: ignore

    @property
    def is_indexed(self) -> bool:
        if not hasattr(self.table, "cols"):
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(self, values: np.ndarray, nan_rep: Any, encoding: str, errors: str) -> Tuple[Any, Any]:
        values = _maybe_convert(values, self.kind if self.kind is not None else "", encoding, errors)
        kwargs: Dict[str, Any] = {"name": self.index_name}
        try:
            new_pd_index = Index(values, **kwargs)
        except ValueError:
            kwargs["freq"] = None
            new_pd_index = Index(values, **kwargs)
        final_pd_index: Index
        if self.tz is not None and isinstance(new_pd_index, DatetimeIndex):
            final_pd_index = new_pd_index.tz_localize("UTC").tz_convert(self.tz)
        else:
            final_pd_index = new_pd_index
        return final_pd_index, final_pd_index

    def validate_names(self) -> None:
        pass

    def validate_and_set(self, handler: AppendableTable, append: bool) -> None:
        self.table = handler.table
        self.validate_names()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def validate_attr(self, append: bool) -> None:
        if append:
            existing_kind = getattr(self.table._v_attrs, f"{self.name}_kind", None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(f"incompatible kind in col [{existing_kind} - {self.kind}]")

    def update_info(self, info: Dict[str, Any]) -> None:
        for key in ["freq", "tz", "index_name"]:
            value = getattr(self, key, None)
            idx = info.setdefault(self.name, {})
            existing_value = idx.get(key)
            if key in idx and value is not None and existing_value != value:
                if key in ["freq", "index_name"]:
                    warnings.warn(f"Attribute conflict for {key}: existing {existing_value} vs new {value}", RuntimeWarning)
                    idx[key] = None
                    setattr(self, key, None)
                else:
                    raise ValueError(f"invalid info for [{self.name}] for [{key}], existing_value [{existing_value}] conflicts with new value [{value}]")
            elif value is not None or existing_value is not None:
                idx[key] = value

    def set_attr(self) -> None:
        setattr(self.table._v_attrs, f"{self.name}_kind", self.kind)


class GenericIndexCol(IndexCol):
    @property
    def is_indexed(self) -> bool:
        return False

    def convert(self, values: np.ndarray, nan_rep: Any, encoding: str, errors: str) -> Tuple[Index, Index]:
        index = pd.RangeIndex(len(values))
        return index, index

    def set_attr(self) -> None:
        pass


class DataCol(IndexCol):
    is_an_indexable: bool = False
    is_data_indexable: bool = False
    def __init__(self, name: str, values: Any = None, kind: Optional[str] = None, typ: Optional[Col] = None, cname: Optional[str] = None, pos: Optional[int] = None, tz: Optional[Any] = None, ordered: Optional[Any] = None, table: Any = None, meta: Optional[str] = None, metadata: Any = None, dtype: Optional[DtypeArg] = None, data: Any = None) -> None:
        super().__init__(name, values, kind, typ, cname, None, pos, None, tz, None, None, table, meta, metadata)
        self.dtype = dtype
        self.data = data

    @property
    def dtype_attr(self) -> str:
        return f"{self.name}_dtype"

    @property
    def meta_attr(self) -> str:
        return f"{self.name}_meta"

    def validate_names(self) -> None:
        if not is_string_dtype(Index(self.values).dtype):
            raise ValueError("cannot have non-object label DataIndexableCol")

    def take_data(self) -> Any:
        return self.data

    def validate_attr(self, append: bool) -> None:
        if append:
            existing_fields = getattr(self.table._v_attrs, f"{self.name}_kind", None)
            if existing_fields is not None and existing_fields != list(self.values):
                raise ValueError("appended items do not match existing items in table!")
            existing_dtype = getattr(self.table._v_attrs, self.dtype_attr, None)
            if existing_dtype is not None and existing_dtype != self.dtype:
                raise ValueError("appended items dtype do not match existing items dtype in table!")

    def convert(self, values: np.ndarray, nan_rep: Any, encoding: str, errors: str) -> Tuple[Any, Any]:
        if values.dtype.fields is not None:
            values = values[self.cname]
        if self.dtype is None:
            converted, dtype_name = _get_data_and_dtype_name(values)
            kind = _dtype_to_kind(dtype_name)
        else:
            converted = values
            dtype_name = self.dtype  # type: ignore
            kind = self.kind if self.kind is not None else _dtype_to_kind(dtype_name)
        meta = self.meta
        metadata = self.metadata
        ordered = None
        tz = self.tz
        try:
            converted = converted.astype(dtype_name, copy=False)
        except TypeError:
            converted = converted.astype("O", copy=False)
        if kind == "string":
            converted = _unconvert_string_array(converted, nan_rep=nan_rep, encoding=encoding, errors=errors)
        return self.values, converted

    def set_attr(self) -> None:
        setattr(self.table._v_attrs, f"{self.name}_kind", self.values)
        setattr(self.table._v_attrs, self.meta_attr, self.meta)
        setattr(self.table._v_attrs, self.dtype_attr, self.dtype)


class DataIndexableCol(DataCol):
    is_data_indexable: bool = True

    def validate_names(self) -> None:
        if not is_string_dtype(Index(self.values).dtype):
            raise ValueError("cannot have non-object label DataIndexableCol")


class GenericDataIndexableCol(DataIndexableCol):
    pass


# End of annotated code.
