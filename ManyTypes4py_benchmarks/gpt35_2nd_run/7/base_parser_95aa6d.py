from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
from enum import Enum
import itertools
from typing import TYPE_CHECKING, Any, cast, final, overload, Set, Dict, Tuple, Union, List, Callable, Iterable, Mapping, Sequence

class ParserBase:

    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2

    def __init__(self, kwds: Dict[str, Any]):
        self._implicit_index: bool = False
        self.names: Union[List[str], None] = kwds.get('names')
        self.orig_names: Union[List[str], None] = None
        self.index_col: Union[int, List[int], None] = kwds.get('index_col', None)
        self.unnamed_cols: Set[int] = set()
        self.index_names: Union[List[str], None] = None
        self.col_names: Union[List[str], None] = None
        parse_dates = kwds.pop('parse_dates', False)
        if parse_dates is None or lib.is_bool(parse_dates):
            parse_dates = bool(parse_dates)
        elif not isinstance(parse_dates, list):
            raise TypeError("Only booleans and lists are accepted for the 'parse_dates' parameter")
        self.parse_dates: Union[bool, List[Union[int, str]]] = parse_dates
        self.date_parser = kwds.pop('date_parser', lib.no_default)
        self.date_format: Union[str, None] = kwds.pop('date_format', None)
        self.dayfirst: bool = kwds.pop('dayfirst', False)
        self.na_values: Union[List[str], None] = kwds.get('na_values')
        self.na_fvalues: Union[List[float], None] = kwds.get('na_fvalues')
        self.na_filter: bool = kwds.get('na_filter', False)
        self.keep_default_na: bool = kwds.get('keep_default_na', True)
        self.dtype: Union[Dict[str, Any], None] = copy(kwds.get('dtype', None))
        self.converters: Union[Dict[str, Callable], None] = kwds.get('converters')
        self.dtype_backend: Union[str, None] = kwds.get('dtype_backend')
        self.true_values: Union[List[str], None] = kwds.get('true_values')
        self.false_values: Union[List[str], None] = kwds.get('false_values')
        self.cache_dates: bool = kwds.pop('cache_dates', True)
        self.header: Union[int, None] = kwds.get('header')
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
    def _extract_multi_indexer_columns(self, header: List[List[str]], index_names: List[str], passed_names: bool = False) -> Tuple[List[str], List[str], Union[List[str], None], bool]:
        ...

    @final
    def _maybe_make_multi_index_columns(self, columns: List[str], col_names: Union[List[str], None]) -> Union[List[str], MultiIndex]:
        ...

    @final
    def _make_index(self, alldata: List[Any], columns: List[str], indexnamerow: Union[List[str], None]) -> Tuple[Union[Index, None], Union[List[str], MultiIndex]]:
        ...

    @final
    def _clean_mapping(self, mapping: Union[Dict[Union[int, str], Any], Any]) -> Any:
        ...

    @final
    def _agg_index(self, index: List[Any]) -> Union[Index, MultiIndex]:
        ...

    @final
    def _set_noconvert_dtype_columns(self, col_indices: List[int], names: List[str]) -> Set[int]:
        ...

    @final
    def _infer_types(self, values: np.ndarray, na_values: Set[str], no_dtype_specified: bool, try_num_bool: bool = True) -> Tuple[np.ndarray, int]:
        ...

    @overload
    def _do_date_conversions(self, names: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @final
    def _do_date_conversions(self, names: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @final
    def _check_data_length(self, columns: List[str], data: List[Any]) -> None:
        ...

    @final
    def _validate_usecols_names(self, usecols: Iterable, names: Iterable) -> Iterable:
        ...

    @final
    def _clean_index_names(self, columns: List[str], index_col: Union[int, List[int], bool]) -> Tuple[Union[List[str], None], List[str], Union[int, List[int], bool]]:
        ...

    @final
    def _get_empty_meta(self, columns: List[str], dtype: Union[Dict[str, Any], None]) -> Tuple[Union[Index, MultiIndex], List[str], Dict[str, Series]]:
        ...

def date_converter(date_col: np.ndarray, col: str, dayfirst: bool = False, cache_dates: bool = True, date_format: Union[str, None] = None) -> np.ndarray:
    ...

def get_na_values(col: str, na_values: Union[Dict[str, Set[str]], Set[str]], na_fvalues: Union[Dict[str, Set[float]], Set[float]], keep_default_na: bool) -> Tuple[Set[str], Set[float]]:
    ...

def is_index_col(col: Union[int, List[int], bool]) -> bool:
    ...

def validate_parse_dates_presence(parse_dates: List[Union[int, str]], columns: List[str]) -> Set[str]:
    ...

def _validate_usecols_arg(usecols: Union[List[Union[int, str]], Callable, None]) -> Tuple[Union[Set[Union[int, str]], None], Union[str, None]]:
    ...

def evaluate_callable_usecols(usecols: Union[List[Union[int, str]], Callable], names: List[str]) -> Union[Set[int], List[Union[int, str]]]:
    ...
