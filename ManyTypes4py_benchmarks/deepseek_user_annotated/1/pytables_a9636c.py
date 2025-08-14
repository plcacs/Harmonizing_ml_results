from __future__ import annotations

from contextlib import suppress
import copy
from datetime import (
    date,
    tzinfo,
)
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
    Optional,
    Union,
    Dict,
    List,
    Tuple,
    Set,
    Iterable,
    Callable,
    Hashable,
    Iterator,
    Sequence,
    TypeVar,
    Generic,
    Mapping,
    MutableMapping,
    Type,
    cast,
)
import warnings
import numpy as np
import pandas as pd
from pandas._config import (
    config,
    get_option,
    using_string_dtype,
)
from pandas._libs import (
    lib,
    writers as libwriters,
)
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
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    PeriodArray,
)
from pandas.core.arrays.datetimes import tz_to_dtype
from pandas.core.arrays.string_ import BaseStringArray
import pandas.core.common as com
from pandas.core.computation.pytables import (
    PyTablesExpr,
    maybe_expression,
)
from pandas.core.construction import (
    array as pd_array,
    extract_array,
)
from pandas.core.indexes.api import ensure_index
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
    adjoin,
    pprint_thing,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterator,
        Sequence,
    )
    from types import TracebackType
    from tables import (
        Col,
        File,
        Node,
    )
    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        AxisInt,
        DtypeArg,
        FilePath,
        Self,
        Shape,
        npt,
    )
    from pandas.core.internals import Block

# versioning attribute
_version: Final[str] = "0.15.2"

# encoding
_default_encoding: Final[str] = "UTF-8"

def _ensure_encoding(encoding: str | None) -> str:
    if encoding is None:
        encoding = _default_encoding
    return encoding

def _ensure_str(name: Any) -> str:
    if isinstance(name, str):
        name = str(name)
    return name

Term = PyTablesExpr

def _ensure_term(where: Any, scope_level: int) -> Any:
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

# formats
_FORMAT_MAP: Final[Dict[str, str]] = {"f": "fixed", "fixed": "fixed", "t": "table", "table": "table"}

# axes map
_AXES_MAP: Final[Dict[Type[DataFrame], List[int]]] = {DataFrame: [0]}

# register our configuration options
dropna_doc: Final[str] = """
: boolean
    drop ALL nan rows when appending to a table
"""
format_doc: Final[str] = """
: format
    default format writing format, if None, then
    put will default to 'fixed' and append will default to 'table'
"""

with config.config_prefix("io.hdf"):
    config.register_option("dropna_table", False, dropna_doc, validator=config.is_bool)
    config.register_option(
        "default_format",
        None,
        format_doc,
        validator=config.is_one_of_factory(["fixed", "table", None]),
    )

# oh the troubles to reduce import time
_table_mod: Any = None
_table_file_open_policy_is_strict: bool = False

def _tables() -> Any:
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables
        _table_mod = tables
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (
                tables.file._FILE_OPEN_POLICY == "strict"
            )
    return _table_mod

def to_hdf(
    path_or_buf: FilePath | "HDFStore",
    key: str,
    value: DataFrame | Series,
    mode: str = "a",
    complevel: int | None = None,
    complib: str | None = None,
    append: bool = False,
    format: str | None = None,
    index: bool = True,
    min_itemsize: int | Dict[str, int] | None = None,
    nan_rep: Any = None,
    dropna: bool | None = None,
    data_columns: Literal[True] | List[str] | None = None,
    errors: str = "strict",
    encoding: str = "UTF-8",
) -> None:
    if append:
        f = lambda store: store.append(
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

def read_hdf(
    path_or_buf: FilePath | "HDFStore",
    key: str | None = None,
    mode: str = "r",
    errors: str = "strict",
    where: str | List[Any] | None = None,
    start: int | None = None,
    stop: int | None = None,
    columns: List[str] | None = None,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwargs: Any,
) -> Any:
    if mode not in ["r", "r+", "a"]:
        raise ValueError(
            f"mode {mode} is not allowed while performing a read. "
            f"Allowed modes are r, r+ and a."
        )
    if where is not None:
        where = _ensure_term(where, scope_level=1)

    if isinstance(path_or_buf, HDFStore):
        if not path_or_buf.is_open:
            raise OSError("The HDFStore must be open for reading.")
        store = path_or_buf
        auto_close = False
    else:
        path_or_buf = stringify_path(path_or_buf)
        if not isinstance(path_or_buf, str):
            raise NotImplementedError(
                "Support for generic buffers has not been implemented."
            )
        try:
            exists = os.path.exists(path_or_buf)
        except (TypeError, ValueError):
            exists = False

        if not exists:
            raise FileNotFoundError(f"File {path_or_buf} does not exist")

        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        auto_close = True

    try:
        if key is None:
            groups = store.groups()
            if len(groups) == 0:
                raise ValueError(
                    "Dataset(s) incompatible with Pandas data types, "
                    "not table, or no datasets found in HDF5 file."
                )
            candidate_only_group = groups[0]
            for group_to_check in groups[1:]:
                if not _is_metadata_of(group_to_check, candidate_only_group):
                    raise ValueError(
                        "key must be provided when HDF5 "
                        "file contains multiple datasets."
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

def _is_metadata_of(group: "Node", parent_group: "Node") -> bool:
    if group._v_depth <= parent_group._v_depth:
        return False
    current = group
    while current._v_depth > 1:
        parent = current._v_parent
        if parent == parent_group and current._v_name == "meta":
            return True
        current = current._v_parent
    return False

class HDFStore:
    _handle: Optional["File"]
    _mode: str

    def __init__(
        self,
        path: str,
        mode: str = "a",
        complevel: int | None = None,
        complib: str | None = None,
        fletcher32: bool = False,
        **kwargs: Any,
    ) -> None:
        if "format" in kwargs:
            raise ValueError("format is not a defined argument for HDFStore")

        tables = import_optional_dependency("tables")

        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(
                f"complib only supports {tables.filters.all_complibs} compression."
            )

        if complib is None and complevel is not None:
            complib = tables.filters.default_complib

        self._path = stringify_path(path)
        if mode is None:
            mode = "a"
        self._mode = mode
        self._handle = None
        self._complevel = complevel if complevel else 0
        self._complib = complib
        self._fletcher32 = fletcher32
        self._filters = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self) -> str:
        return self._path

    @property
    def root(self) -> "Node":
        self._check_if_open()
        assert self._handle is not None
        return self._handle.root

    @property
    def filename(self) -> str:
        return self._path

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> int | None:
        return self.remove(key)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __contains__(self, key: str) -> bool:
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
        return f"{type(self)}\nFile path: {pstr}\n"

    def __enter__(self) -> "HDFStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def keys(self, include: str = "pandas") -> List[str]:
        if include == "pandas":
            return [n._v_pathname for n in self.groups()]
        elif include == "native":
            assert self._handle is not None
            return [
                n._v_pathname for n in self._handle.walk_nodes("/", classname="Table")
            ]
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def items(self) -> Iterator[Tuple[str, List[Any]]]:
        for g in self.groups():
            yield g._v_pathname, g

    def open(self, mode: str = "a", **kwargs: Any) -> None:
        tables = _tables()

        if self._mode != mode:
            if self._mode in ["a", "w"] and mode in ["r", "r+"]:
                pass
            elif mode in ["w"]:
                if self.is_open:
                    raise PossibleDataLossError(
                        f"Re-opening the file [{self._path}] with mode [{self._mode}] "
                        "will delete the current file!"
                    )
            self._mode = mode

        if self.is_open:
            self.close()

        if self._complevel and self._complevel > 0:
            self._filters = _tables().Filters(
                self._complevel, self._complib, fletcher32=self._fletcher32
            )

        if _table_file_open_policy_is_strict and self.is_open:
            msg = (
                "Cannot open HDF5 file, which is already opened, "
                "even in read-only mode."
            )
            raise ValueError(msg)

        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def is_open(self) -> bool:
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def flush(self, fsync: bool = False) -> None:
        if self._handle is not None:
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def get(self, key: str) -> Any:
        with patch_pickle():
            group = self.get_node(key)
            if group is None:
                raise KeyError(f"No object named {key} in the file")
            return self._read_group(group)

    def select(
        self,
        key: str,
        where: Any = None,
        start: int | None = None,
        stop: int | None = None,
        columns: List[str] | None = None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ) -> Any:
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")

        s = self._create_storer(group)
        s.infer_axes()

        def func(_start: int | None, _stop: int | None, _where: Any) -> Any:
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=s.nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        return it.get_result()

    def select_as_coordinates(
        self,
        key: str,
        where: Any = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        where = _ensure_term(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_coordinates with a table")
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def select_column(
        self,
        key: str,
        column: str,
        start: int | None = None,
        stop: int | None = None,
    ) -> Any:
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_column with a table")
        return tbl.read_column(column=column, start=start