from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import TYPE_CHECKING, Literal, cast, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas._typing import AlignJoin, DtypeObj, F, Scalar, npt
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import ensure_object, is_bool_dtype, is_extension_array_dtype, is_integer, is_list_like, is_numeric_dtype, is_object_dtype, is_re
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCMultiIndex, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import Callable as CallableABC, Hashable, Iterator
    from pandas._typing import NpDtype
    from pandas import DataFrame, Index, Series

_shared_docs: Dict[str, str] = {}
_cpython_optimized_encoders: Tuple[str, ...] = ('utf-8', 'utf8', 'latin-1', 'latin1', 'iso-8859-1', 'mbcs', 'ascii')
_cpython_optimized_decoders: Tuple[str, ...] = _cpython_optimized_encoders + ('utf-16', 'utf-32')

T = TypeVar('T')

def func_6rxv26hk(forbidden: Optional[List[str]], name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to forbid specific types for a method of StringMethods.
    """
    forbidden = [] if forbidden is None else forbidden
    allowed_types = {'string', 'empty', 'bytes', 'mixed', 'mixed-integer'} - set(forbidden)

    def decorator(func: F) -> F:
        func_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._inferred_dtype not in allowed_types:
                msg = f"Cannot use .str.{func_name} with values of inferred dtype '{self._inferred_dtype}'."
                raise TypeError(msg)
            return func(self, *args, **kwargs)
        wrapper.__name__ = func_name
        return cast(F, wrapper)
    return decorator

def func_1keyjoio(name: str, docstring: str) -> Callable[[Any], Any]:
    @func_6rxv26hk(['bytes'], name=name)
    def wrapper(self: Any) -> Any:
        result = getattr(self._data.array, f'_str_{name}')()
        return self._wrap_result(result, returns_string=name not in ('isnumeric', 'isdecimal'))
    wrapper.__doc__ = docstring
    return wrapper

class StringMethods(NoNewAttributesMixin):
    def __init__(self, data: Union[ABCSeries, ABCIndex]) -> None:
        from pandas.core.arrays.string_ import StringDtype
        self._inferred_dtype = self._validate(data)
        self._is_categorical = isinstance(data.dtype, CategoricalDtype)
        self._is_string = isinstance(data.dtype, StringDtype)
        self._data = data
        self._index = None
        self._name = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name
        self._parent = data._values.categories if self._is_categorical else data
        self._orig = data
        self._freeze()

    @staticmethod
    def _validate(data: Union[ABCSeries, ABCIndex]) -> str:
        """
        Auxiliary function for StringMethods, infers and checks dtype of data.
        """
        if isinstance(data, ABCMultiIndex):
            raise AttributeError('Can only use .str accessor with Index, not MultiIndex')
        allowed_types = ['string', 'empty', 'bytes', 'mixed', 'mixed-integer']
        data = extract_array(data)
        values = getattr(data, 'categories', data)
        inferred_dtype = lib.infer_dtype(values, skipna=True)
        if inferred_dtype not in allowed_types:
            raise AttributeError(f'Can only use .str accessor with string values, not {inferred_dtype}')
        return inferred_dtype

    def __getitem__(self, key: Any) -> Any:
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> None:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(self, result: Any, name: Optional[str] = None, expand: Optional[bool] = None, fill_value: Any = np.nan, returns_string: bool = True, dtype: Optional[DtypeObj] = None) -> Any:
        from pandas import Index, MultiIndex
        if not hasattr(result, 'ndim') or not hasattr(result, 'dtype'):
            if isinstance(result, ABCDataFrame):
                result = result.__finalize__(self._orig, name='str')
            return result
        assert result.ndim < 3
        if expand is None:
            expand = result.ndim != 1
        elif expand is True and not isinstance(self._orig, ABCIndex):
            if isinstance(result.dtype, ArrowDtype):
                import pyarrow as pa
                from pandas.compat import pa_version_under11p0
                from pandas.core.arrays.arrow.array import ArrowExtensionArray
                value_lengths = pa.compute.list_value_length(result._pa_array)
                max_len = pa.compute.max(value_lengths).as_py()
                min_len = pa.compute.min(value_lengths).as_py()
                if result._hasna:
                    result = ArrowExtensionArray(result._pa_array.fill_null([None] * max_len))
                if min_len < max_len:
                    if not pa_version_under11p0:
                        result = ArrowExtensionArray(pa.compute.list_slice(result._pa_array, start=0, stop=max_len, return_fixed_size_list=True))
                    else:
                        all_null = np.full(max_len, fill_value=None, dtype=object)
                        values = result.to_numpy()
                        new_values = []
                        for row in values:
                            if len(row) < max_len:
                                nulls = all_null[:max_len - len(row)]
                                row = np.append(row, nulls)
                            new_values.append(row)
                        pa_type = result._pa_array.type
                        result = ArrowExtensionArray(pa.array(new_values, type=pa_type))
                if name is None:
                    name = range(max_len)
                result = pa.compute.list_flatten(result._pa_array).to_numpy().reshape(len(result), max_len)
                result = {label: ArrowExtensionArray(pa.array(res)) for label, res in zip(name, result.T)}
            elif is_object_dtype(result):
                def func_nyqy5cn6(x: Any) -> Any:
                    if is_list_like(x):
                        return x
                    else:
                        return [x]
                result = [func_nyqy5cn6(x) for x in result]
                if result and not self._is_string:
                    max_len = max(len(x) for x in result)
                    result = [(x * max_len if len(x) == 0 or x[0] is np.nan else x) for x in result]
        if not isinstance(expand, bool):
            raise ValueError('expand must be True or False')
        if expand is False:
            if name is None:
                name = getattr(result, 'name', None)
            if name is None:
                name = self._orig.name
        if isinstance(self._orig, ABCIndex):
            if is_bool_dtype(result):
                return result
            if expand:
                result = list(result)
                out = MultiIndex.from_tuples(result, names=name)
                if out.nlevels == 1:
                    out = out.get_level_values(0)
                return out
            else:
                return Index(result, name=name, dtype=dtype)
        else:
            index = self._orig.index
            _dtype = dtype
            vdtype = getattr(result, 'dtype', None)
            if _dtype is not None:
                pass
            elif self._is_string:
                if is_bool_dtype(vdtype):
                    _dtype = result.dtype
                elif returns_string:
                    _dtype = self._orig.dtype
                else:
                    _dtype = vdtype
            elif vdtype is not None:
                _dtype = vdtype
            if expand:
                cons = self._orig._constructor_expanddim
                result = cons(result, columns=name, index=index, dtype=_dtype)
            else:
                cons = self._orig._constructor
                result = cons(result, name=name, index=index, dtype=_dtype)
            result = result.__finalize__(self._orig, method='str')
            if name is not None and result.ndim == 1:
                result.name = name
            return result

    def _get_series_list(self, others: Any) -> List[ABCSeries]:
        """
        Auxiliary function for :meth:`str.cat`.
        """
        from pandas import DataFrame, Series
        idx = self._orig if isinstance(self._orig, ABCIndex) else self._orig.index
        if isinstance(others, ABCSeries):
            return [others]
        elif isinstance(others, ABCIndex):
            return [Series(others, index=idx, dtype=others.dtype)]
        elif isinstance(others, ABCDataFrame):
            return [others[x] for x in others]
        elif isinstance(others, np.ndarray) and others.ndim == 2:
            others = DataFrame(others, index=idx)
            return [others[x] for x in others]
        elif is_list_like(others, allow_sets=False):
            try:
                others = list(others)
            except TypeError:
                pass
            else:
                if all(isinstance(x, (ABCSeries, ABCIndex, ExtensionArray)) or isinstance(x, np.ndarray) and x.ndim == 1 for x in others):
                    los = []
                    while others:
                        los = los + self._get_series_list(others.pop(0))
                    return los
                elif all(not is_list_like(x) for x in others):
                    return [Series(others, index=idx)]
        raise TypeError('others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])')

    @func_6rxv26hk(['bytes', 'mixed', 'mixed-integer'])
    def cat(self, others: Optional[Any] = None, sep: Optional[str] = None, na_rep: Optional[str] = None, join: str = 'left') -> Any:
        """
        Concatenate strings in the Series/Index with given separator.
        """
        from pandas import Index, Series, concat
        if isinstance(others, str):
            raise ValueError('Did you mean to supply a `sep` keyword?')
        if sep is None:
            sep = ''
        if isinstance(self._orig, ABCIndex):
            data = Series(self._orig, index=self._orig, dtype=self._orig.dtype)
        else:
            data = self._orig
        if others is None:
            data = ensure_object(data)
            na_mask = isna(data)
            if na_rep is None and na_mask.any():
                return sep.join(data[~na_mask])
            elif na_rep is not None and na_mask.any():
                return sep.join(np.where(na_mask, na_rep, data))
            else:
                return sep.join(data)
        try:
            others = self._get_series_list(others)
        except ValueError as err:
            raise ValueError('If `others` contains arrays or lists (or other list-likes without an index), these must all be of the same length as the calling Series/Index.') from err
        if any(not data.index.equals(x.index) for x in others):
            others = concat(others, axis=1, join=join if join == 'inner' else 'outer', keys=range(len(others)), sort=False)
            data, others = data.align(others, join=join)
            others = [others[x] for x in others]
        all_cols = [ensure_object(x) for x in [data] + others]
        na_masks = np.array([isna(x) for x in all_cols])
        union_mask = np.logical_or.reduce(na_masks, axis=0)
        if na_rep is None and union_mask.any():
            result = np.empty(len(data), dtype=object)
            np.putmask(result, union_mask, np.nan)
            not_masked = ~union_mask
            result[not_masked] = cat_safe([x[not_masked] for x in all_cols], sep)
        elif na_rep is not None and union_mask.any():
            all_cols = [np.where(nm, na_rep, col) for nm, col in zip(na_masks, all_cols)]
            result = cat_safe(all_cols, sep)
        else:
            result = cat_safe(all_cols, sep)
        if isinstance(self._orig.dtype, CategoricalDtype):
            dtype = self._orig.dtype.categories.dtype
        else:
            dtype = self._orig.dtype
        if isinstance(self._orig, ABCIndex):
            if isna(result).all():
                dtype = object
            out = Index(result, dtype=dtype, name=self._orig.name)
        else:
            res_ser = Series(result, dtype=dtype, index=data.index, name=self._orig.name, copy=False)
            out = res_ser.__finalize__(self._orig, method='str_cat')
        return out

    # ... [rest of the methods with their type annotations]

def cat_safe(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`.
    """
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ['string', 'empty']:
                raise TypeError(f'Concatenation requires list-likes containing only strings (or missing values). Offending values found in column {dtype}') from None
    return result

def cat_core(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`
    """
    if sep == '':
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)

def _result_dtype(arr: Union[ABCSeries, ABCIndex]) -> DtypeObj:
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object

def _get_single_group_name(regex: re.Pattern) -> Optional[str]:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None

def _get_group_names(regex: re.Pattern) -> Union[range, List[Any]]:
    """
    Get named groups from compiled regex.
    """
    rng = range(regex.groups)
    names = {v: k for k, v in regex.groupindex.items()}
    if not names:
        return rng
    result = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == 'i' and lib.is_range_indexer(arr, len(arr)):
        return rng
    return result

def str_extractall(arr: Union[ABCSeries, ABCIndex], pat: str, flags: int = 0) -> Any:
    regex = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError('pattern contains no capture groups')
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns = _get_group_names(regex)
    match_list = []
    index_list = []
    is_mi = arr.index.nlevels > 1
    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)
            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = [(np.nan if group == '' else group) for group in match_tuple]
                match_list.append(na_tuple)
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)
    from pandas import MultiIndex
    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ['match'])
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(match_list, index=index, columns=columns, dtype=dtype)
    return result
