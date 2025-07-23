from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
from enum import Enum
import itertools
from typing import TYPE_CHECKING, Any, cast, final, overload, Optional, Union, Dict, List, Set, Tuple, Callable, Iterable, Mapping, Sequence
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
    from collections.abc import Callable as CallableT, Iterable as IterableT, Mapping as MappingT, Sequence as SequenceT
    from pandas._typing import ArrayLike, DtypeArg, Hashable, HashableT, Scalar, SequenceT

class ParserBase:

    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2

    def __init__(self, kwds: Dict[str, Any]) -> None:
        self._implicit_index: bool = False
        self.names: Optional[Sequence[Hashable]] = kwds.get('names')
        self.orig_names: Optional[Sequence[Hashable]] = None
        self.index_col: Optional[Union[int, Sequence[int], str, Sequence[str]]] = kwds.get('index_col', None)
        self.unnamed_cols: Set[str] = set()
        self.index_names: Optional[Sequence[Hashable]] = None
        self.col_names: Optional[Sequence[Hashable]] = None
        parse_dates: Union[bool, List[Union[int, str]]] = kwds.pop('parse_dates', False)
        if parse_dates is None or lib.is_bool(parse_dates):
            parse_dates = bool(parse_dates)
        elif not isinstance(parse_dates, list):
            raise TypeError("Only booleans and lists are accepted for the 'parse_dates' parameter")
        self.parse_dates: Union[bool, List[Union[int, str]]] = parse_dates
        self.date_parser: Optional[Callable] = kwds.pop('date_parser', lib.no_default)
        self.date_format: Optional[Union[str, Dict[str, str]]] = kwds.pop('date_format', None)
        self.dayfirst: bool = kwds.pop('dayfirst', False)
        self.na_values: Optional[Union[Sequence[str], Dict[str, Sequence[str]]]] = kwds.get('na_values')
        self.na_fvalues: Optional[Union[Sequence[float], Dict[str, Sequence[float]]]] = kwds.get('na_fvalues')
        self.na_filter: bool = kwds.get('na_filter', False)
        self.keep_default_na: bool = kwds.get('keep_default_na', True)
        self.dtype: Optional[Union[DtypeArg, Dict[Hashable, DtypeArg]]] = copy(kwds.get('dtype', None))
        self.converters: Optional[Dict[Hashable, Callable]] = kwds.get('converters')
        self.dtype_backend: Optional[str] = kwds.get('dtype_backend')
        self.true_values: Optional[List[str]] = kwds.get('true_values')
        self.false_values: Optional[List[str]] = kwds.get('false_values')
        self.cache_dates: bool = kwds.pop('cache_dates', True)
        self.header: Optional[Union[int, Sequence[int]]] = kwds.get('header')
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
        self.usecols: Optional[Union[Set[Hashable], Callable[[Hashable], bool]]] = None
        self.usecols_dtype: Optional[str] = None
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
    def _extract_multi_indexer_columns(
        self,
        header: List[List[Hashable]],
        index_names: Optional[List[Hashable]],
        passed_names: bool = False
    ) -> Tuple[List[Hashable], Optional[List[Hashable]], Optional[List[Hashable]], bool]:
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

        def extract(r: List[Hashable]) -> Tuple[Hashable, ...]:
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
    def _maybe_make_multi_index_columns(
        self,
        columns: Sequence[Hashable],
        col_names: Optional[Sequence[Hashable]] = None
    ) -> Union[Sequence[Hashable], MultiIndex]:
        if is_potential_multi_index(columns):
            columns_mi = cast('Sequence[tuple[Hashable, ...]]', columns)
            return MultiIndex.from_tuples(columns_mi, names=col_names)
        return columns

    @final
    def _make_index(
        self,
        alldata: List[ArrayLike],
        columns: List[Hashable],
        indexnamerow: Optional[List[Hashable]] = None
    ) -> Tuple[Optional[Union[Index, MultiIndex]], List[Hashable]]:
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
        clean = {}
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
    def _set_noconvert_dtype_columns(
        self,
        col_indices: List[int],
        names: List[Hashable]
    ) -> Set[int]:
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
    def _infer_types(
        self,
        values: np.ndarray,
        na_values: Set[Any],
        no_dtype_specified: bool,
        try_num_bool: bool = True
    ) -> Tuple[Union[np.ndarray, BaseMaskedArray, ArrowExtensionArray], int]:
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
    def _do_date_conversions(self, names: Sequence[Hashable], data: Dict[Hashable, ArrayLike]) -> Dict[Hashable, ArrayLike]: ...

    @overload
    def _do_date_conversions(self, names: Sequence[Hashable], data: List[ArrayLike]) -> List[ArrayLike]: ...

    @final
    def _do_date_conversions(
        self,
        names: Sequence[Hashable],
        data: Union[Dict[Hashable, ArrayLike], List[ArrayLike]]
    ) -> Union[Dict[Hashable, ArrayLike], List[ArrayLike]]:
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
    def _check_data_length(self, columns: Sequence[Hashable], data: List[ArrayLike]) -> None:
        if not self.index_col and len(columns) !=