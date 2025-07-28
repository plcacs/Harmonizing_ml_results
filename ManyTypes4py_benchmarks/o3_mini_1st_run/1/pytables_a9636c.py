from __future__ import annotations
from contextlib import suppress
import copy
from datetime import date, tzinfo
import itertools
import os
import re
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final, Literal, cast, overload, Callable, List, Tuple, Optional, Dict, Union
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
    from collections.abc import Iterator, Sequence
    from types import TracebackType
    from tables import Col, File, Node
    from pandas._typing import AnyArrayLike, ArrayLike, AxisInt, DtypeArg, FilePath, Self, Shape, npt
    from pandas.core.internals import Block

_version: Final[str] = '0.15.2'
_default_encoding: Final[str] = 'UTF-8'

def _ensure_encoding(encoding: Optional[str]) -> str:
    if encoding is None:
        encoding = _default_encoding
    return encoding

def _ensure_str(name: Any) -> Any:
    """
    Ensure that an index / column name is a str.
    """
    if isinstance(name, str):
        name = str(name)
    return name

def _get_tz(tz: Union[str, tzinfo]) -> Any:
    zone = timezones.get_timezone(tz)
    return zone

def _set_tz(values: np.ndarray, tz: Optional[Union[str, tzinfo]], datetime64_dtype: str) -> DatetimeArray:
    """
    Coerce the values to a DatetimeArray with appropriate tz.
    """
    assert values.dtype == 'i8', values.dtype
    unit, _ = np.datetime_data(datetime64_dtype)
    dtype = tz_to_dtype(tz=tz, unit=unit)
    dta = DatetimeArray._from_sequence(values, dtype=dtype)
    return dta

def _convert_index(name: str, index: Any, encoding: str, errors: str) -> IndexCol:
    """
    Convert an index into an IndexCol.
    """
    assert isinstance(name, str)
    index_name: Any = index.name
    converted, dtype_name = _get_data_and_dtype_name(index)
    kind: str = _dtype_to_kind(dtype_name)
    atom = DataIndexableCol._get_atom(converted)
    if lib.is_np_dtype(index.dtype, 'iu') or needs_i8_conversion(index.dtype) or is_bool_dtype(index.dtype):
        return IndexCol(name, values=converted, kind=kind, typ=atom, index_name=index_name)
    if isinstance(index, MultiIndex):
        raise TypeError('MultiIndex not supported here!')
    inferred_type: str = lib.infer_dtype(index, skipna=False)
    values: np.ndarray = np.asarray(index)
    if inferred_type == 'date':
        converted = np.asarray([v.toordinal() for v in values], dtype=np.int32)
        return IndexCol(name, converted, 'date', _tables().Time32Col(), index_name=index_name)
    elif inferred_type == 'string':
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(name, converted, 'string', _tables().StringCol(itemsize), index_name=index_name)
    elif inferred_type in ['integer', 'floating']:
        return IndexCol(name, values=converted, kind=kind, typ=atom, index_name=index_name)
    else:
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == 'object', kind
        atom = _tables().ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)

def _unconvert_index(data: np.ndarray, kind: str, encoding: str, errors: str) -> np.ndarray:
    """
    Inverse of _convert_index.
    """
    if kind.startswith('datetime64'):
        if kind == 'datetime64':
            index = DatetimeIndex(data)
        else:
            index = DatetimeIndex(data.view(kind))
    elif kind == 'timedelta64':
        index = TimedeltaIndex(data)
    elif kind == 'date':
        try:
            index = np.asarray([date.fromordinal(v) for v in data], dtype=object)
        except ValueError:
            index = np.asarray([date.fromtimestamp(v) for v in data], dtype=object)
    elif kind in ('integer', 'float', 'bool'):
        index = np.asarray(data)
    elif kind in 'string':
        index = _unconvert_string_array(data, nan_rep=None, encoding=encoding, errors=errors)
    elif kind == 'object':
        index = np.asarray(data[0])
    else:
        raise ValueError(f'unrecognized index type {kind}')
    return index

def _maybe_convert_for_string_atom(
    name: str,
    bvalues: np.ndarray,
    existing_col: Optional[Any],
    min_itemsize: Optional[Union[int, Dict[str, int]]],
    nan_rep: Any,
    encoding: str,
    errors: str,
    columns: Optional[List[str]]
) -> np.ndarray:
    """
    maybe set a string col itemsize.
    """
    if isinstance(bvalues.dtype, StringDtype):
        bvalues = bvalues.to_numpy()
    if bvalues.dtype != object:
        return bvalues
    bvalues = cast(np.ndarray, bvalues)
    dtype_name = bvalues.dtype.name
    inferred_type: str = lib.infer_dtype(bvalues, skipna=False)
    if inferred_type == 'date':
        raise TypeError('[date] is not implemented as a table column')
    if inferred_type == 'datetime':
        raise TypeError('too many timezones in this block, create separate data columns')
    if not (inferred_type == 'string' or dtype_name == 'object'):
        return bvalues
    mask = isna(bvalues)
    data = bvalues.copy()
    data[mask] = nan_rep
    if existing_col and mask.any() and (len(nan_rep) > existing_col.itemsize):
        raise ValueError('NaN representation is too large for existing column size')
    inferred_type = lib.infer_dtype(data, skipna=False)
    if inferred_type != 'string':
        for i in range(data.shape[0]):
            col = data[i]
            inferred_type = lib.infer_dtype(col, skipna=False)
            if inferred_type != 'string':
                error_column_label = columns[i] if columns and i < len(columns) else f'No.{i}'
                raise TypeError(f'Cannot serialize the column [{error_column_label}]\nbecause its data contents are not [string] but [{inferred_type}] object dtype')
    data_converted = _convert_string_array(data, encoding, errors).reshape(data.shape)
    itemsize = data_converted.itemsize
    if isinstance(min_itemsize, dict):
        min_itemsize = int(min_itemsize.get(name) or min_itemsize.get('values') or 0)
    itemsize = max(min_itemsize or 0, itemsize)
    if existing_col is not None:
        eci = existing_col.validate_col(itemsize)
        if eci is not None and eci > itemsize:
            itemsize = eci
    data_converted = data_converted.astype(f'|S{itemsize}', copy=False)
    return data_converted

def _convert_string_array(data: np.ndarray, encoding: str, errors: str) -> np.ndarray:
    """
    Convert an object dtype array to fixed-length string dtype.
    """
    if len(data):
        data = Series(data.ravel(), copy=False).str.encode(encoding, errors)._values.reshape(data.shape)
    ensured = ensure_object(data.ravel())
    itemsize = max(1, libwriters.max_len_string_array(ensured))
    data = np.asarray(data, dtype=f'S{itemsize}')
    return data

def _unconvert_string_array(data: np.ndarray, nan_rep: Optional[str], encoding: str, errors: str) -> np.ndarray:
    """
    Inverse of _convert_string_array.
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

def _maybe_convert(values: Any, val_kind: str, encoding: str, errors: str) -> Any:
    """
    Convert the data from this selection to the appropriate pandas type.
    """
    assert isinstance(val_kind, str), type(val_kind)
    if _need_convert(val_kind):
        conv = _get_converter(val_kind, encoding, errors)
        values = conv(values)
    return values

def _get_converter(kind: str, encoding: str, errors: str) -> Callable[[Any], Any]:
    if kind == 'datetime64':
        return lambda x: np.asarray(x, dtype='M8[ns]')
    elif 'datetime64' in kind:
        return lambda x: np.asarray(x, dtype=kind)
    elif kind == 'string':
        return lambda x: _unconvert_string_array(x, nan_rep=None, encoding=encoding, errors=errors)
    else:
        raise ValueError(f'invalid kind {kind}')

def _need_convert(kind: str) -> bool:
    if kind in ('datetime64', 'string') or 'datetime64' in kind:
        return True
    return False

def _maybe_adjust_name(name: str, version: Tuple[int, int, int]) -> str:
    """
    Adjust the given name if necessary for versions < 0.10.1.
    """
    if isinstance(version, str) or len(version) < 3:
        raise ValueError('Version is incorrect, expected sequence of 3 integers.')
    if version[0] == 0 and version[1] <= 10 and (version[2] == 0):
        m = re.search('values_block_(\\d+)', name)
        if m:
            grp = m.groups()[0]
            name = f'values_{grp}'
    return name

def _dtype_to_kind(dtype_str: str) -> str:
    """
    Find the "kind" string describing the given dtype name.
    """
    if dtype_str.startswith(('string', 'bytes')):
        kind = 'string'
    elif dtype_str.startswith('float'):
        kind = 'float'
    elif dtype_str.startswith('complex'):
        kind = 'complex'
    elif dtype_str.startswith(('int', 'uint')):
        kind = 'integer'
    elif dtype_str.startswith('datetime64'):
        kind = dtype_str
    elif dtype_str.startswith('timedelta'):
        kind = 'timedelta64'
    elif dtype_str.startswith('bool'):
        kind = 'bool'
    elif dtype_str.startswith('category'):
        kind = 'category'
    elif dtype_str.startswith('period'):
        kind = 'integer'
    elif dtype_str == 'object':
        kind = 'object'
    elif dtype_str == 'str':
        kind = 'str'
    else:
        raise ValueError(f'cannot interpret dtype of [{dtype_str}]')
    return kind

def _get_data_and_dtype_name(data: Any) -> Tuple[np.ndarray, str]:
    """
    Convert the passed data into a storable form and a dtype string.
    """
    if isinstance(data, Categorical):
        data = data.codes
    if isinstance(data.dtype, DatetimeTZDtype):
        dtype_name = f'datetime64[{data.dtype.unit}]'
    else:
        dtype_name = data.dtype.name
    if data.dtype.kind in 'mM':
        data = np.asarray(data.view('i8'))
    elif isinstance(data, PeriodIndex):
        data = data.asi8
    data = np.asarray(data)
    return (data, dtype_name)

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
        if is_list_like(where):
            with suppress(ValueError):
                inferred = lib.infer_dtype(where, skipna=False)
                if inferred in ('integer', 'boolean'):
                    where_arr = np.asarray(where)
                    if where_arr.dtype == np.bool_:
                        s: int = self.start if self.start is not None else 0
                        e: int = self.stop if self.stop is not None else self.table.nrows
                        self.coordinates = np.arange(s, e)[where_arr]
                    elif issubclass(where_arr.dtype.type, np.integer):
                        if self.start is not None and (where_arr < self.start).any() or (self.stop is not None and (where_arr >= self.stop).any()):
                            raise ValueError('where must have index locations >= start and < stop')
                        self.coordinates = where_arr
        if self.coordinates is None:
            self.terms = self.generate(where)
            if self.terms is not None:
                self.condition, self.filter = self.terms.evaluate()

    @overload
    def generate(self, where: Any) -> Optional[PyTablesExpr]:
        ...
    @overload
    def generate(self, where: Any) -> Optional[PyTablesExpr]:
        ...

    def generate(self, where: Any) -> Optional[PyTablesExpr]:
        """
        where can be a dict, list, tuple, or string.
        """
        if where is None:
            return None
        q: Dict[str, Any] = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            qkeys = ','.join(q.keys())
            msg = dedent(f"                The passed where expression: {where}\n                            contains an invalid variable reference\n                            all of the variable references must be a reference to\n                            an axis (e.g. 'index' or 'columns'), or a data_column\n                            The currently defined references are: {qkeys}\n                ")
            raise ValueError(msg) from err

    def select(self) -> Any:
        if self.condition is not None:
            return self.table.table.read_where(self.condition.format(), start=self.start, stop=self.stop)
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self) -> np.ndarray:
        start: int = self.start if self.start is not None else 0
        nrows: int = self.table.nrows
        if start < 0:
            start += nrows
        stop: int = self.stop if self.stop is not None else nrows
        if stop < 0:
            stop += nrows
        if self.condition is not None:
            return self.table.table.get_where_list(self.condition.format(), start=start, stop=stop, sort=True)
        elif self.coordinates is not None:
            return self.coordinates
        return np.arange(start, stop)