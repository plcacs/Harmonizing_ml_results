from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
from enum import Enum
import itertools
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast, final, overload
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
    _implicit_index: bool
    _first_chunk: bool
    keep_default_na: bool
    dayfirst: bool
    cache_dates: bool
    usecols_dtype: Optional[str]

    def __init__(self, kwds):
        self._implicit_index = False
        self.names = kwds.get('names')
        self.orig_names: Optional[Sequence[Hashable]] = None
        self.index_col = kwds.get('index_col', None)
        self.unnamed_cols: Set = set()
        self.index_names: Optional[Sequence[Hashable]] = None
        self.col_names: Optional[Sequence[Hashable]] = None
        parse_dates = kwds.pop('parse_dates', False)
        if parse_dates is None or lib.is_bool(parse_dates):
            parse_dates = bool(parse_dates)
        elif not isinstance(parse_dates, list):
            raise TypeError("Only booleans and lists are accepted for the 'parse_dates' parameter")
        self.parse_dates: Union[bool, List] = parse_dates
        self.date_parser = kwds.pop('date_parser', lib.no_default)
        self.date_format = kwds.pop('date_format', None)
        self.dayfirst = kwds.pop('dayfirst', False)
        self.na_values = kwds.get('na_values')
        self.na_fvalues = kwds.get('na_fvalues')
        self.na_filter = kwds.get('na_filter', False)
        self.keep_default_na = kwds.get('keep_default_na', True)
        self.dtype = copy(kwds.get('dtype', None))
        self.converters = kwds.get('converters')
        self.dtype_backend = kwds.get('dtype_backend')
        self.true_values = kwds.get('true_values')
        self.false_values = kwds.get('false_values')
        self.cache_dates = kwds.pop('cache_dates', True)
        self.header = kwds.get('header')
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
        self._first_chunk = True
        self.usecols, self.usecols_dtype = _validate_usecols_arg(kwds['usecols'])
        self.on_bad_lines = kwds.get('on_bad_lines', self.BadLineHandleMethod.ERROR)

    def close(self):
        pass

    @final
    def _should_parse_dates(self, i):
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
    def _extract_multi_indexer_columns(self, header, index_names, passed_names=False):
        """
        Extract and return the names, index_names, col_names if the column
        names are a MultiIndex.

        Parameters
        ----------
        header: list of lists
            The header rows
        index_names: list, optional
            The names of the future index
        passed_names: bool, default False
            A flag specifying if names where passed

        """
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

        def extract(r):
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
    def _maybe_make_multi_index_columns(self, columns, col_names=None):
        if is_potential_multi_index(columns):
            columns_mi = cast('Sequence[Tuple[Hashable, ...]]', columns)
            return MultiIndex.from_tuples(columns_mi, names=col_names)
        return columns

    @final
    def _make_index(self, alldata, columns, indexnamerow=None):
        index: Optional[Index]
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
    def _clean_mapping(self, mapping):
        """converts col numbers to names"""
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
    def _agg_index(self, index):
        arrays = []
        converters = self._clean_mapping(self.converters)
        clean_dtypes = self._clean_mapping(self.dtype)
        if self.index_names is not None:
            names: Iterable = self.index_names
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
    def _set_noconvert_dtype_columns(self, col_indices, names):
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions. If usecols is specified, the positions of the columns
        not to cast is relative to the usecols not to all columns.

        Parameters
        ----------
        col_indices: The indices specifying order and positions of the columns
        names: The column names which order is corresponding with the order
               of col_indices

        Returns
        -------
        A set of integers containing the positions of the columns not to convert.
        """
        usecols: Optional[Union[List[int], List[str]]]
        noconvert_columns = set()
        if self.usecols_dtype == 'integer':
            usecols = sorted(self.usecols)
        elif callable(self.usecols) or self.usecols_dtype not in ('empty', None):
            usecols = col_indices
        else:
            usecols = None

        def _set(x):
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
    def _infer_types(self, values, na_values, no_dtype_specified, try_num_bool=True):
        """
        Infer types of values, possibly casting

        Parameters
        ----------
        values : ndarray
        na_values : set
        no_dtype_specified: Specifies if we want to cast explicitly
        try_num_bool : bool, default try
           try to cast values to numeric (first preference) or boolean

        Returns
        -------
        converted : ndarray or ExtensionArray
        na_count : int
        """
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
        result: ArrayLike
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
    def _do_date_conversions(self, names, data):
        ...

    @overload
    def _do_date_conversions(self, names, data):
        ...

    @final
    def _do_date_conversions(self, names, data):
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
    def _check_data_length(self, columns, data):
        """Checks if length of data is equal to length of column names.

        One set of trailing commas is allowed. self.index_col not False
        results in a ParserError previously when lengths do not match.

        Parameters
        ----------
        columns: list of column names
        data: list of array-likes containing the data column-wise.
        """
        if not self.index_col and len(columns) != len(data) and columns:
            empty_str = is_object_dtype(data[-1]) and data[-1] == ''
            empty_str_or_na = empty_str | isna(data[-1])
            if len(columns) == len(data) - 1 and np.all(empty_str_or_na):
                return
            warnings.warn('Length of header or names does not match length of data. This leads to a loss of data with index_col=False.', ParserWarning, stacklevel=find_stack_level())

    @final
    def _validate_usecols_names(self, usecols, names):
        """
        Validates that all usecols are present in a given
        list of names. If not, raise a ValueError that
        shows what usecols are missing.

        Parameters
        ----------
        usecols : iterable of usecols
            The columns to validate are present in names.
        names : iterable of names
            The column names to check against.

        Returns
        -------
        usecols : iterable of usecols
            The `usecols` parameter if the validation succeeds.

        Raises
        ------
        ValueError : Columns were missing. Error message will list them.
        """
        missing = [c for c in usecols if c not in names]
        if len(missing) > 0:
            raise ValueError(f'Usecols do not match columns, columns expected but not found: {missing}')
        return usecols

    @final
    def _clean_index_names(self, columns, index_col):
        if not is_index_col(index_col):
            return (None, list(columns), list(index_col) if index_col is not None else [])
        columns = list(columns)
        if not columns:
            return ([None] * len(index_col), columns, list(index_col))
        cp_cols = list(columns)
        index_names: List[Optional[Union[str, int]]] = []
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
    def _get_empty_meta(self, columns, dtype=None):
        columns = list(columns)
        index_col = self.index_col
        index_names = self.index_names
        dtype_dict: defaultdict[Hashable, Any]
        if not is_dict_like(dtype):
            dtype_dict = defaultdict(lambda: dtype)
        else:
            dtype = cast(dict, dtype)
            dtype_dict = defaultdict(lambda: None, {columns[k] if is_integer(k) else k: v for k, v in dtype.items()})
        index: Index
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

def date_converter(date_col, col, dayfirst=False, cache_dates=True, date_format=None):
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
parser_defaults = {'delimiter': None, 'escapechar': None, 'quotechar': '"', 'quoting': csv.QUOTE_MINIMAL, 'doublequote': True, 'skipinitialspace': False, 'lineterminator': None, 'header': 'infer', 'index_col': None, 'names': None, 'skiprows': None, 'skipfooter': 0, 'nrows': None, 'na_values': None, 'keep_default_na': True, 'true_values': None, 'false_values': None, 'converters': None, 'dtype': None, 'cache_dates': True, 'thousands': None, 'comment': None, 'decimal': '.', 'parse_dates': False, 'dayfirst': False, 'date_format': None, 'usecols': None, 'chunksize': None, 'encoding': None, 'compression': None, 'skip_blank_lines': True, 'encoding_errors': 'strict', 'on_bad_lines': ParserBase.BadLineHandleMethod.ERROR, 'dtype_backend': lib.no_default}

def get_na_values(col, na_values, na_fvalues, keep_default_na):
    """
    Get the NaN values for a given column.

    Parameters
    ----------
    col : str
        The name of the column.
    na_values : array-like, dict
        The object listing the NaN values as strings.
    na_fvalues : array-like, dict
        The object listing the NaN values as floats.
    keep_default_na : bool
        If `na_values` is a dict, and the column is not mapped in the
        dictionary, whether to return the default NaN values or the empty set.

    Returns
    -------
    nan_tuple : A length-two tuple composed of

        1) na_values : the string NaN values for that column.
        2) na_fvalues : the float NaN values for that column.
    """
    if isinstance(na_values, dict):
        if col in na_values:
            return (na_values[col], na_fvalues[col])
        else:
            if keep_default_na:
                return (STR_NA_VALUES, set())
            return (set(), set())
    else:
        return (na_values, na_fvalues)

def is_index_col(col):
    return col is not None and col is not False

def validate_parse_dates_presence(parse_dates, columns):
    """
    Check if parse_dates are in columns.

    If user has provided names for parse_dates, check if those columns
    are available.

    Parameters
    ----------
    columns : list
        List of names of the dataframe.

    Returns
    -------
    The names of the columns which will get parsed later if a list
    is given as specification.

    Raises
    ------
    ValueError
        If column to parse_date is not in dataframe.

    """
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

def _validate_usecols_arg(usecols):
    """
    Validate the 'usecols' parameter.

    Checks whether or not the 'usecols' parameter contains all integers
    (column selection by index), strings (column by name) or is a callable.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    usecols : list-like, callable, or None
        List of columns to use when parsing or a callable that can be used
        to filter a list of table columns.

    Returns
    -------
    usecols_tuple : tuple
        A tuple of (verified_usecols, usecols_dtype).

        'verified_usecols' is either a set if an array-like is passed in or
        'usecols' if a callable or None is passed in.

        'usecols_dtype` is the inferred dtype of 'usecols' if an array-like
        is passed in or None if a callable or None is passed in.
    """
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
def evaluate_callable_usecols(usecols, names):
    ...

@overload
def evaluate_callable_usecols(usecols, names):
    ...

def evaluate_callable_usecols(usecols, names):
    """
    Check whether or not the 'usecols' parameter
    is a callable.  If so, enumerates the 'names'
    parameter and returns a set of indices for
    each entry in 'names' that evaluates to True.
    If not a callable, returns 'usecols'.
    """
    if callable(usecols):
        return {i for i, name in enumerate(names) if usecols(name)}
    return usecols