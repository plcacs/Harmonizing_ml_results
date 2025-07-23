from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
from enum import Enum
import itertools
from typing import TYPE_CHECKING, Any, cast, final, overload, Union, Optional, Tuple, Dict, List, Set
import warnings
import numpy as np
from pandas._libs import lib, parsers
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import import_optional_dependency
from pandas.errors import ParserError, ParserWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_bool_dtype, is_dict_like, is_float_dtype, is_integer, is_integer_dtype, is_list_like, is_object_dtype, is_string_dtype
from pandas.core.dtypes.missing import isna
from pandas import DataFrame, DatetimeIndex, StringDtype
from pandas.core import algorithms
from pandas.core.arrays import ArrowExtensionArray, BaseMaskedArray, BooleanArray, FloatingArray, IntegerArray
from pandas.core.indexes.api import Index, MultiIndex, default_index, ensure_index_from_sequences
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index
if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pandas._typing import ArrayLike, DtypeArg, Hashable, HashableT, Scalar, SequenceT

class ParserBase:

    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2

    def __init__(self, kwds: Dict[str, Any]) -> None:
        self._implicit_index: bool = False
        self.names: Optional[List[str]] = kwds.get('names')
        self.orig_names: Optional[List[str]] = None
        self.index_col: Optional[Union[int, List[int]]] = kwds.get('index_col', None)
        self.unnamed_cols: Set[str] = set()
        self.index_names: Optional[List[str]] = None
        self.col_names: Optional[List[str]] = None
        parse_dates = kwds.pop('parse_dates', False)
        if parse_dates is None or lib.is_bool(parse_dates):
            parse_dates = bool(parse_dates)
        elif not isinstance(parse_dates, list):
            raise TypeError("Only booleans and lists are accepted for the 'parse_dates' parameter")
        self.parse_dates: Union[bool, List[Union[int, str]]] = parse_dates
        self.date_parser: Any = kwds.pop('date_parser', lib.no_default)
        self.date_format: Optional[Union[str, Dict[str, str]]] = kwds.pop('date_format', None)
        self.dayfirst: bool = kwds.pop('dayfirst', False)
        self.na_values: Optional[Union[Dict[str, Any], List[Any]]] = kwds.get('na_values')
        self.na_fvalues: Optional[Union[Dict[str, Any], List[Any]]] = kwds.get('na_fvalues')
        self.na_filter: bool = kwds.get('na_filter', False)
        self.keep_default_na: bool = kwds.get('keep_default_na', True)
        self.dtype: Optional[Union[DtypeArg, Dict[str, DtypeArg]]] = copy(kwds.get('dtype', None))
        self.converters: Optional[Dict[str, Callable]] = kwds.get('converters')
        self.dtype_backend: Any = kwds.get('dtype_backend')
        self.true_values: Optional[List[str]] = kwds.get('true_values')
        self.false_values: Optional[List[str]] = kwds.get('false_values')
        self.cache_dates: bool = kwds.pop('cache_dates', True)
        self.header: Optional[Union[int, List[int]]] = kwds.get('header')
        if is_list_like(self.header, allow_sets=False):
            if kwds.get('usecols'):
                raise ValueError('cannot specify usecols when specifying a multi-index header')
            if kwds.get('names'):
                raise ValueError('cannot specify names when specifying a multi-index header')
            if self.index_col is not None:
                if is_integer(self.index_col):
                    self.index_col = [self.index_col]
                elif not (is_list_like(self.index_col, allow_sets=False) and all(map(is_integer, self.index_col))):
                    raise ValueError('index_col must only contain integers of column positions when specifying a multi-index header')
                else:
                    self.index_col = list(self.index_col)
        self._first_chunk: bool = True
        self.usecols, self.usecols_dtype = _validate_usecols_arg(kwds['usecols'])
        self.on_bad_lines: ParserBase.BadLineHandleMethod = kwds.get('on_bad_lines', self.BadLineHandleMethod.ERROR)

    def close(self) -> None:
        pass

    @final
    def _should_parse_dates(self, i: int) -> bool:
        if isinstance(self.parse_dates, bool):
            return self.parse_dates
        else:
            if self.index_names is not None:
                name = self.index_names[i]
            else:
                name = None
            j = i if self.index_col is None else self.index_col[i]
            return j in self.parse_dates or (name is not None and name in self.parse_dates)

    @final
    def _extract_multi_indexer_columns(self, header: List[List[str]], index_names: Optional[List[str]], passed_names: bool = False) -> Tuple[List[Tuple[Hashable, ...]], Optional[List[str]], Optional[List[Optional[str]]], bool]:
        if len(header) < 2:
            return (header[0], index_names, None, passed_names)
        ic = self.index_col
        if ic is None:
            ic = []
        if not isinstance(ic, (list, tuple, np.ndarray)):
            ic = [ic]
        sic = set(ic)
        index_names = header.pop(-1)
        index_names, _, _ = self._clean_index_names(index_names, self.index_col)
        field_count = len(header[0])
        if not all((len(header_iter) == field_count for header_iter in header[1:])):
            raise ParserError('Header rows must have an equal number of columns.')

        def extract(r: List[str]) -> Tuple[str, ...]:
            return tuple((r[i] for i in range(field_count) if i not in sic))
        columns = list(zip(*(extract(r) for r in header)))
        names = columns.copy()
        for single_ic in sorted(ic):
            names.insert(single_ic, single_ic)
        if len(ic):
            col_names = [r[ic[0]] if r[ic[0]] is not None and r[ic[0]] not in self.unnamed_cols else None for r in header]
        else:
            col_names = [None] * len(header)
        passed_names = True
        return (names, index_names, col_names, passed_names)

    @final
    def _maybe_make_multi_index_columns(self, columns: List[Tuple[Hashable, ...]], col_names: Optional[List[str]] = None) -> Union[List[Tuple[Hashable, ...]], MultiIndex]:
        if is_potential_multi_index(columns):
            columns_mi = cast('Sequence[tuple[Hashable, ...]]', columns)
            return MultiIndex.from_tuples(columns_mi, names=col_names)
        return columns

    @final
    def _make_index(self, alldata: List[ArrayLike], columns: List[str], indexnamerow: Optional[List[str]] = None) -> Tuple[Optional[Union[Index, MultiIndex]], List[str]]:
        if isinstance(self.index_col, list) and len(self.index_col):
            to_remove = []
            indexes = []
            for idx in self.index_col:
                if isinstance(idx, str):
                    raise ValueError(f'Index {idx} invalid')
                to_remove.append(idx)
                indexes.append(alldata[idx])
            for i in sorted(to_remove, reverse=True):
                alldata.pop(i)
                if not self._implicit_index:
                    columns.pop(i)
            index = self._agg_index(indexes)
            if indexnamerow:
                coffset = len(indexnamerow) - len(columns)
                index = index.set_names(indexnamerow[:coffset])
        else:
            index = None
        columns = self._maybe_make_multi_index_columns(columns, self.col_names)
        return (index, columns)

    @final
    def _clean_mapping(self, mapping: Any) -> Any:
        if not isinstance(mapping, dict):
            return mapping
        clean: Dict[Any, Any] = {}
        assert self.orig_names is not None
        for col, v in mapping.items():
            if isinstance(col, int) and col not in self.orig_names:
                col = self.orig_names[col]
            clean[col] = v
        if isinstance(mapping, defaultdict):
            remaining_cols = set(self.orig_names) - set(clean.keys())
            clean.update({col: mapping[col] for col in remaining_cols})
        return clean

    @final
    def _agg_index(self, index: List[ArrayLike]) -> Union[Index, MultiIndex]:
        arrays = []
        converters = self._clean_mapping(self.converters)
        clean_dtypes = self._clean_mapping(self.dtype)
        if self.index_names is not None:
            names = self.index_names
        else:
            names = itertools.cycle([None])
        for i, (arr, name) in enumerate(zip(index, names)):
            if self._should_parse_dates(i):
                arr = date_converter(arr, col=self.index_names[i] if self.index_names is not None else None, dayfirst=self.dayfirst, cache_dates=self.cache_dates, date_format=self.date_format)
            if self.na_filter:
                col_na_values = self.na_values
                col_na_fvalues = self.na_fvalues
            else:
                col_na_values = set()
                col_na_fvalues = set()
            if isinstance(self.na_values, dict):
                assert self.index_names is not None
                col_name = self.index_names[i]
                if col_name is not None:
                    col_na_values, col_na_fvalues = get_na_values(col_name, self.na_values, self.na_fvalues, self.keep_default_na)
                else:
                    col_na_values, col_na_fvalues = (set(), set())
            cast_type = None
            index_converter = False
            if self.index_names is not None:
                if isinstance(clean_dtypes, dict):
                    cast_type = clean_dtypes.get(self.index_names[i], None)
                if isinstance(converters, dict):
                    index_converter = converters.get(self.index_names[i]) is not None
            try_num_bool = not (cast_type and is_string_dtype(cast_type) or index_converter)
            arr, _ = self._infer_types(arr, col_na_values | col_na_fvalues, cast_type is None, try_num_bool)
            if cast_type is not None:
                idx = Index(arr, name=name, dtype=cast_type)
            else:
                idx = ensure_index_from_sequences([arr], [name])
            arrays.append(idx)
        if len(arrays) == 1:
            return arrays[0]
        else:
            return MultiIndex.from_arrays(arrays)

    @final
    def _set_noconvert_dtype_columns(self, col_indices: List[int], names: List[str]) -> Set[int]:
        noconvert_columns = set()
        if self.usecols_dtype == 'integer':
            usecols = sorted(self.usecols)
        elif callable(self.usecols) or self.usecols_dtype not in ('empty', None):
            usecols = col_indices
        else:
            usecols = None

        def _set(x: Union[int, str]) -> int:
            if usecols is not None and is_integer(x):
                x = usecols[x]
            if not is_integer(x):
                x = col_indices[names.index(x)]
            return x
        if isinstance(self.parse_dates, list):
            validate_parse_dates_presence(self.parse_dates, names)
            for val in self.parse_dates:
                noconvert_columns.add(_set(val))
        elif self.parse_dates:
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    noconvert_columns.add(_set(k))
            elif self.index_col is not None:
                noconvert_columns.add(_set(self.index_col))
        return noconvert_columns

    @final
    def _infer_types(self, values: np.ndarray, na_values: Set[Any], no_dtype_specified: bool, try_num_bool: bool = True) -> Tuple[Union[np.ndarray, ExtensionArray], int]:
        na_count = 0
        if issubclass(values.dtype.type, (np.number, np.bool_)):
            na_values = np.array([val for val in na_values if not isinstance(val, str)])
            mask = algorithms.isin(values, na_values)
            na_count = mask.astype('uint8', copy=False).sum()
            if na_count > 0:
                if is_integer_dtype(values):
                    values = values.astype(np.float64)
                np.putmask(values, mask, np.nan)
            return (values, na_count)
        dtype_backend = self.dtype_backend
        non_default_dtype_backend = no_dtype_specified and dtype_backend is not lib.no_default
        if try_num_bool and is_object_dtype(values.dtype):
            try:
                result, result_mask = lib.maybe_convert_numeric(values, na_values, False, convert_to_masked_nullable=non_default_dtype_backend)
            except (ValueError, TypeError):
                na_count = parsers.sanitize_objects(values, na_values)
                result = values
            else:
                if non_default_dtype_backend:
                    if result_mask is None:
                        result_mask = np.zeros(result.shape, dtype=np.bool_)
                    if result_mask.all():
                        result = IntegerArray(np.ones(result_mask.shape, dtype=np.int64), result_mask)
                    elif is_integer_dtype(result):
                        result = IntegerArray(result, result_mask)
                    elif is_bool_dtype(result):
                        result = BooleanArray(result, result_mask)
                    elif is_float_dtype(result):
                        result = FloatingArray(result, result_mask)
                    na_count = result_mask.sum()
                else:
                    na_count = isna(result).sum()
        else:
            result = values
            if values.dtype == np.object_:
                na_count = parsers.sanitize_objects(values, na_values)
        if result.dtype == np.object_ and try_num_bool:
            result, bool_mask = libops.maybe_convert_bool(np.asarray(values), true_values=self.true_values, false_values=self.false_values, convert_to_masked_nullable=non_default_dtype_backend)
            if result.dtype == np.bool_ and non_default_dtype_backend:
                if bool_mask is None:
                    bool_mask = np.zeros(result.shape, dtype=np.bool_)
                result = BooleanArray(result, bool_mask)
            elif result.dtype == np.object_ and non_default_dtype_backend:
                if not lib.is_datetime_array(result, skipna=True):
                    dtype = StringDtype()
                    cls = dtype.construct_array_type()
                    result = cls._from_sequence(values, dtype=dtype)
        if dtype_backend == 'pyarrow':
            pa = import_optional_dependency('pyarrow')
            if isinstance(result, np.ndarray):
                result = ArrowExtensionArray(pa.array(result, from_pandas=True))
            elif isinstance(result, BaseMaskedArray):
                if result._mask.all():
                    result = ArrowExtensionArray(pa.array([None] * len(result)))
                else:
                    result = ArrowExtensionArray(pa.array(result._data, mask=result._mask))
            else:
                result = ArrowExtensionArray(pa.array(result.to_numpy(), from_pandas=True))
        return (result, na_count)

    @overload
    def _do_date_conversions(self, names: List[str], data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        ...

    @overload
    def _do_date_conversions(self, names: List[str], data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        ...

    @final
    def _do_date_conversions(self, names: List[str], data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if not isinstance(self.parse_dates, list):
            return data
        for colspec in self.parse_dates:
            if isinstance(colspec, int) and colspec not in data:
                colspec = names[colspec]
            if isinstance(self.index_col, list) and colspec in self.index_col or (isinstance(self.index_names, list) and colspec in self.index_names):
                continue
            result = date_converter(data[colspec], col=colspec, dayfirst=self.dayfirst, cache_dates=self.cache_dates, date_format=self.date_format)
            data[colspec] = result
        return data

    @final
    def _check_data_length(self, columns: List[str], data: List[ArrayLike]) -> None:
        if not self.index_col and len(columns) != len(data) and columns:
            empty_str = is_object_dtype(data[-1]) and data[-1] == ''
            empty_str_or_na = empty_str | isna(data[-1])
            if len(columns) == len(data) - 1 and np.all(empty_str_or_na):
                return
            warnings.warn('Length of header or names does not match length of data. This leads to a loss of data with index_col=False.', ParserWarning, stacklevel=find_stack_level())

    @final
    def _validate_usecols_names(self, usecols: Iterable[Union[int, str]], names: Iterable[str]) -> Iterable[Union[int, str]]:
        missing = [c for c in usecols if c not in names]
        if len(missing) > 0:
            raise ValueError(f'Usecols do not match columns, columns expected but not found: {missing}')
        return usecols

    @final
    def _clean_index_names(self, columns: List[str], index_col: Optional[Union[int, List[int]]]) -> Tuple[Optional[List[str]], List[str], Optional[Union[int, List[int]]]]:
        if not is_index_col(index_col):
            return (None, columns, index_col)
        columns = list(columns)
        if not columns:
            return ([None] * len(index_col), columns, index_col)
        cp_cols = list(columns)
        index_names = []
        index_col = list(index_col)
        for i, c in enumerate(index_col):
            if isinstance(c, str):
                index_names.append(c)
                for j, name in enumerate(cp_cols):
                    if name == c:
                        index_col[i] = j
                        columns.remove(name)
                        break
            else:
                name = cp_cols[c]
                columns.remove(name)
                index_names.append(name)
        for i, name in enumerate(index_names):
            if isinstance(name, str) and name in self.unnamed_cols:
                index_names[i] = None
        return (index_names, columns, index_col)

    @final
    def _get_empty_meta(self, columns: List[str], dtype: Optional[Union[DtypeArg, Dict[str, DtypeArg]]] = None) -> Tuple[Index, List[str], Dict[str, Series]]:
        columns = list(columns)
        index_col = self.index_col
        index_names = self.index_names
        if not is_dict_like(dtype):
            dtype_dict = defaultdict(lambda: dtype)
        else:
            dtype = cast(dict, dtype)
            dtype_dict = defaultdict(lambda: None, {columns[k] if is_integer(k) else k: v for k, v in dtype.items()})
        if (index_col is None or index_col is False) or index_names is None:
            index = default_index(0)
        else:
            data = [Index([], name=name, dtype=dtype_dict[name]) for name in index_names]
            if len(data) == 1:
                index = data[0]
            else:
                index = MultiIndex.from_arrays(data)
            index_col.sort()
            for i, n in enumerate(index_col):
                columns.pop(n - i)
        col_dict = {col_name: Series([], dtype=dtype_dict[col_name]) for col_name in columns}
        return (index, columns, col_dict)

def date_converter(date_col: ArrayLike, col: Optional[str], dayfirst: bool = False, cache_dates: bool = True, date_format: Optional[Union[str, Dict[str, str]]] = None) -> ArrayLike:
    if date_col.dtype.kind in 'Mm':
        return date_col
    date_fmt = date_format.get(col) if isinstance(date_format, dict) else date_format
    str_objs = lib.ensure_string_array(np.asarray(date_col))
    try:
        result = tools.to_datetime(str_objs, format=date_fmt, utc=False, dayfirst=dayfirst, cache=cache_dates)
    except (ValueError, TypeError):
        return str_objs
    if isinstance(result, DatetimeIndex):
        arr = result.to_numpy()
        arr.flags.writeable = True
        return arr
    return result._values

parser_defaults: Dict[str, Any] = {'delimiter': None, 'escapechar': None, 'quotechar': '"', 'quoting': csv.QUOTE_MINIMAL, 'doublequote': True, 'skipinitialspace': False, 'lineterminator': None, 'header': 'infer', 'index_col': None, 'names': None, 'skiprows': None, 'skipfooter': 0, 'nrows': None, 'na_values': None, 'keep_default_na': True, 'true_values': None, 'false_values': None, 'converters': None, 'dtype': None, 'cache_dates': True, 'thousands': None, 'comment': None, 'decimal': '.', 'parse_dates': False, 'dayfirst': False, 'date_format': None, 'usecols': None, 'chunksize': None, 'encoding': None, 'compression': None, 'skip_blank_lines': True, 'encoding_errors': 'strict', 'on_bad_lines': ParserBase.BadLineHandleMethod.ERROR, 'dtype_backend': lib.no_default}

def get_na_values(col: str, na_values: Union[Dict[str, Any], List[Any]], na_fvalues: Union[Dict[str, Any], List[Any]], keep_default_na: bool) -> Tuple[Set[Any], Set[Any]]:
    if isinstance(na_values, dict):
        if col in na_values:
            return (na_values[col], na_fvalues[col])
        else:
            if keep_default_na:
                return (STR_NA_VALUES, set())
            return (set(), set())
    else:
        return (na_values, na_fvalues)

def is_index_col(col: Optional[Union[int, List[int]]]) -> bool:
    return col is not None and col is not False

def validate_parse_dates_presence(parse_dates: List[Union[int, str]], columns: List[str]) -> Set[str]:
    if not isinstance(parse_dates, list):
        return set()
    missing = set()
    unique_cols = set()
    for col in parse_dates:
        if isinstance(col, str):
            if col not in columns:
                missing.add(col)
            else:
                unique_cols.add(col)
        elif col in columns:
            unique_cols.add(col)
        else:
            unique_cols.add(columns[col])
    if missing:
        missing_cols = ', '.join(sorted(missing))
        raise ValueError(f"Missing column provided to 'parse_dates': '{missing_cols}'")
    return unique_cols

def _validate_usecols_arg(usecols: Optional[Union[Iterable[Union[int, str]], Callable[[str], bool]]]) -> Tuple[Optional[Union[Set[Union[int, str]], Callable[[str], bool]]], Optional[str]]:
    msg = "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
    if usecols is not None:
        if callable(usecols):
            return (usecols, None)
        if not is_list_like(usecols):
            raise ValueError(msg)
        usecols_dtype = lib.infer_dtype(usecols, skipna=False)
        if usecols_dtype not in ('empty', 'integer', 'string'):
            raise ValueError(msg)
        usecols = set(usecols)
        return (usecols, usecols_dtype)
    return (usecols, None)

@overload
def evaluate_callable_usecols(usecols: Callable[[str], bool], names: List[str]) -> Set[int]:
    ...

@overload
def evaluate_callable_usecols(usecols: Optional[Union[Iterable[Union[int, str]], Callable[[str], bool]]], names: List[str]) -> Optional[Union[Set[Union[int, str]], Callable[[str], bool]]]:
    ...

def evaluate_callable_usecols(usecols: Optional[Union[Iterable[Union[int, str]], Callable[[str], bool]]], names: List[str]) -> Optional[Union[Set[Union[int, str]], Callable[[str], bool]]]:
    if callable(usecols):
        return {i for i, name in enumerate(names) if usecols(name)}
    return usecols
