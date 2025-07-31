from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import TYPE_CHECKING, Literal, cast, Any, List, Tuple, Optional, Union
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
    from collections.abc import Callable, Hashable, Iterator
    from pandas._typing import NpDtype
    from pandas import DataFrame, Index, Series

_shared_docs: dict[str, Any] = {}
_cpython_optimized_encoders: Tuple[str, ...] = ('utf-8', 'utf8', 'latin-1', 'latin1', 'iso-8859-1', 'mbcs', 'ascii')
_cpython_optimized_decoders: Tuple[str, ...] = _cpython_optimized_encoders + ('utf-16', 'utf-32')


def forbid_nonstring_types(forbidden: Optional[List[str]], name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to forbid specific types for a method of StringMethods.
    """
    forbidden = [] if forbidden is None else forbidden
    allowed_types = {'string', 'empty', 'bytes', 'mixed', 'mixed-integer'} - set(forbidden)

    def _forbid_nonstring_types(func: F) -> F:
        func_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            if self._inferred_dtype not in allowed_types:
                msg = f"Cannot use .str.{func_name} with values of inferred dtype '{self._inferred_dtype}'."
                raise TypeError(msg)
            return func(self, *args, **kwargs)
        wrapper.__name__ = func_name
        return cast(F, wrapper)
    return _forbid_nonstring_types


def _map_and_wrap(name: str, docstring: str) -> Any:
    @forbid_nonstring_types(['bytes'], name=name)
    def wrapper(self: Any) -> Any:
        result = getattr(self._data.array, f'_str_{name}')()
        return self._wrap_result(result, returns_string=name not in ('isnumeric', 'isdecimal'))
    wrapper.__doc__ = docstring
    return wrapper


class StringMethods(NoNewAttributesMixin):
    """
    Vectorized string functions for Series and Index.
    """
    def __init__(self, data: Any) -> None:
        from pandas.core.arrays.string_ import StringDtype
        self._inferred_dtype = self._validate(data)
        self._is_categorical = isinstance(data.dtype, CategoricalDtype)
        self._is_string = isinstance(data.dtype, StringDtype)
        self._data = data
        self._index = self._name = None
        from pandas.core.dtypes.generic import ABCSeries  # Local import for type checking
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name
        self._parent = data._values.categories if self._is_categorical else data
        self._orig = data
        self._freeze()

    @staticmethod
    def _validate(data: Any) -> str:
        from pandas.core.dtypes.generic import ABCMultiIndex
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

    def __iter__(self) -> Any:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(self, result: Any, name: Optional[Any] = None, expand: Optional[bool] = None, fill_value: Any = np.nan, returns_string: bool = True, dtype: Optional[Any] = None) -> Any:
        from pandas import Index, MultiIndex
        if not hasattr(result, 'ndim') or not hasattr(result, 'dtype'):
            if isinstance(result, ABCDataFrame):
                result = result.__finalize__(self._orig, name='str')
            return result
        assert result.ndim < 3
        if expand is None:
            expand = result.ndim != 1
        elif expand is True and (not isinstance(self._orig, ABCIndex)):
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
                def cons_row(x: Any) -> List[Any]:
                    if is_list_like(x):
                        return x
                    else:
                        return [x]
                result = [cons_row(x) for x in result]
                if result and (not self._is_string):
                    max_len = max((len(x) for x in result))
                    result = [x * max_len if len(x) == 0 or x[0] is np.nan else x for x in result]
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

    def _get_series_list(self, others: Any) -> List[Any]:
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
                if all((isinstance(x, (ABCSeries, ABCIndex, ExtensionArray)) or (isinstance(x, np.ndarray) and x.ndim == 1) for x in others)):
                    los: List[Any] = []
                    while others:
                        los = los + self._get_series_list(others.pop(0))
                    return los
                elif all((not is_list_like(x) for x in others)):
                    return [Series(others, index=idx)]
        raise TypeError('others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])')

    @forbid_nonstring_types(['bytes', 'mixed', 'mixed-integer'])
    def cat(self, others: Any = None, sep: Optional[str] = None, na_rep: Optional[str] = None, join: AlignJoin = 'left') -> Any:
        from pandas import Index, Series, concat
        if isinstance(others, str):
            raise ValueError('Did you mean to supply a `sep` keyword?')
        if sep is None:
            sep = ''
        if isinstance(self._orig, ABCIndex):
            from pandas import Series as pdSeries
            data = pdSeries(self._orig, index=self._orig, dtype=self._orig.dtype)
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
        if any((not data.index.equals(x.index) for x in others)):
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
            from pandas import Index as pdIndex
            out = pdIndex(result, dtype=dtype, name=self._orig.name)
        else:
            res_ser = Series(result, dtype=dtype, index=data.index, name=self._orig.name, copy=False)
            out = res_ser.__finalize__(self._orig, method='str_cat')
        return out

    @forbid_nonstring_types(['bytes'])
    def split(self, pat: Optional[str] = None, *, n: int = -1, expand: bool = False, regex: Optional[bool] = None) -> Any:
        if regex is False and is_re(pat):
            raise ValueError('Cannot use a compiled regex as replacement pattern with regex=False')
        if is_re(pat):
            regex = True
        result = self._data.array._str_split(pat, n, expand, regex)
        if self._data.dtype == 'category':
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @forbid_nonstring_types(['bytes'])
    def rsplit(self, pat: Optional[str] = None, *, n: int = -1, expand: bool = False) -> Any:
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @forbid_nonstring_types(['bytes'])
    def partition(self, sep: str = ' ', expand: bool = True) -> Any:
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == 'category':
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @forbid_nonstring_types(['bytes'])
    def rpartition(self, sep: str = ' ', expand: bool = True) -> Any:
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == 'category':
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    def get(self, i: Any) -> Any:
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def join(self, sep: str) -> Any:
        result = self._data.array._str_join(sep)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def contains(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default, regex: bool = True) -> Any:
        if regex and re.compile(pat).groups:
            warnings.warn('This pattern is interpreted as a regular expression, and has match groups. To actually get the groups, use str.extract.', UserWarning, stacklevel=find_stack_level())
        result = self._data.array._str_contains(pat, case, flags, na, regex)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def match(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any:
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any:
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def replace(self, pat: Union[str, re.Pattern, dict], repl: Optional[Union[str, Callable[[Any], str]]] = None, n: int = -1, case: Optional[bool] = None, flags: int = 0, regex: bool = False) -> Any:
        if isinstance(pat, dict) and repl is not None:
            raise ValueError('repl cannot be used when pat is a dictionary')
        if not isinstance(pat, dict) and (not (isinstance(repl, str) or callable(repl))):
            raise TypeError('repl must be a string or callable')
        is_compiled_re = is_re(pat)
        if regex or regex is None:
            if is_compiled_re and (case is not None or flags != 0):
                raise ValueError('case and flags cannot be set when pat is a compiled regex')
        elif is_compiled_re:
            raise ValueError('Cannot use a compiled regex as replacement pattern with regex=False')
        elif callable(repl):
            raise ValueError('Cannot use a callable replacement when regex=False')
        if case is None:
            case = True
        res_output = self._data
        if not isinstance(pat, dict):
            pat = {pat: repl}
        for key, value in pat.items():
            result = res_output.array._str_replace(key, value, n=n, case=case, flags=flags, regex=regex)
            res_output = self._wrap_result(result)
        return res_output

    @forbid_nonstring_types(['bytes'])
    def repeat(self, repeats: Union[int, List[int]]) -> Any:
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def pad(self, width: int, side: str = 'left', fillchar: str = ' ') -> Any:
        if not isinstance(fillchar, str):
            msg = f'fillchar must be a character, not {type(fillchar).__name__}'
            raise TypeError(msg)
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        if not is_integer(width):
            msg = f'width must be of integer type, not {type(width).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_pad'] % {'side': 'left and right', 'method': 'center'})
    @forbid_nonstring_types(['bytes'])
    def center(self, width: int, fillchar: str = ' ') -> Any:
        return self.pad(width, side='both', fillchar=fillchar)

    @Appender(_shared_docs['str_pad'] % {'side': 'right', 'method': 'ljust'})
    @forbid_nonstring_types(['bytes'])
    def ljust(self, width: int, fillchar: str = ' ') -> Any:
        return self.pad(width, side='right', fillchar=fillchar)

    @Appender(_shared_docs['str_pad'] % {'side': 'left', 'method': 'rjust'})
    @forbid_nonstring_types(['bytes'])
    def rjust(self, width: int, fillchar: str = ' ') -> Any:
        return self.pad(width, side='left', fillchar=fillchar)

    @forbid_nonstring_types(['bytes'])
    def zfill(self, width: int) -> Any:
        if not is_integer(width):
            msg = f'width must be of integer type, not {type(width).__name__}'
            raise TypeError(msg)
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        return self._wrap_result(result)

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> Any:
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def slice_replace(self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> Any:
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def decode(self, encoding: str, errors: str = 'strict') -> Any:
        if encoding in _cpython_optimized_decoders:
            f = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        arr = self._data.array
        result = arr._str_map(f)
        dtype = 'str' if get_option('future.infer_string') else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(['bytes'])
    def encode(self, encoding: str, errors: str = 'strict') -> Any:
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    @Appender(_shared_docs['str_strip'] % {'side': 'left and right sides', 'method': 'strip', 'position': 'leading and trailing'})
    @forbid_nonstring_types(['bytes'])
    def strip(self, to_strip: Optional[str] = None) -> Any:
        result = self._data.array._str_strip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_strip'] % {'side': 'left side', 'method': 'lstrip', 'position': 'leading'})
    @forbid_nonstring_types(['bytes'])
    def lstrip(self, to_strip: Optional[str] = None) -> Any:
        result = self._data.array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_strip'] % {'side': 'right side', 'method': 'rstrip', 'position': 'trailing'})
    @forbid_nonstring_types(['bytes'])
    def rstrip(self, to_strip: Optional[str] = None) -> Any:
        result = self._data.array._str_rstrip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_removefix'] % {'side': 'prefix', 'other_side': 'suffix'})
    @forbid_nonstring_types(['bytes'])
    def removeprefix(self, prefix: str) -> Any:
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_removefix'] % {'side': 'suffix', 'other_side': 'prefix'})
    @forbid_nonstring_types(['bytes'])
    def removesuffix(self, suffix: str) -> Any:
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def wrap(self, width: int, expand_tabs: bool = True, tabsize: int = 8, replace_whitespace: bool = True, drop_whitespace: bool = True, initial_indent: str = '', subsequent_indent: str = '', fix_sentence_endings: bool = False, break_long_words: bool = True, break_on_hyphens: bool = True, max_lines: Optional[int] = None, placeholder: str = ' [...]') -> Any:
        result = self._data.array._str_wrap(width=width, expand_tabs=expand_tabs, tabsize=tabsize, replace_whitespace=replace_whitespace, drop_whitespace=drop_whitespace, initial_indent=initial_indent, subsequent_indent=subsequent_indent, fix_sentence_endings=fix_sentence_endings, break_long_words=break_long_words, break_on_hyphens=break_on_hyphens, max_lines=max_lines, placeholder=placeholder)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def get_dummies(self, sep: str = '|', dtype: Optional[Any] = None) -> Any:
        from pandas.core.frame import DataFrame
        if dtype is not None and (not (is_numeric_dtype(dtype) or is_bool_dtype(dtype))):
            raise ValueError("Only numeric or boolean dtypes are supported for 'dtype'")
        result, name = self._data.array._str_get_dummies(sep, dtype)
        if is_extension_array_dtype(dtype):
            return self._wrap_result(DataFrame(result, columns=name, dtype=dtype), name=name, returns_string=False)
        return self._wrap_result(result, name=name, expand=True, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def translate(self, table: dict) -> Any:
        result = self._data.array._str_translate(table)
        dtype = object if self._data.dtype == 'object' else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(['bytes'])
    def count(self, pat: str, flags: int = 0) -> Any:
        result = self._data.array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def startswith(self, pat: Union[str, Tuple[str, ...]], na: Any = lib.no_default) -> Any:
        if not isinstance(pat, (str, tuple)):
            msg = f'expected a string or tuple, not {type(pat).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_startswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def endswith(self, pat: Union[str, Tuple[str, ...]], na: Any = lib.no_default) -> Any:
        if not isinstance(pat, (str, tuple)):
            msg = f'expected a string or tuple, not {type(pat).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_endswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def findall(self, pat: str, flags: int = 0) -> Any:
        result = self._data.array._str_findall(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def extract(self, pat: str, flags: int = 0, expand: bool = True) -> Any:
        from pandas import DataFrame
        if not isinstance(expand, bool):
            raise ValueError('expand must be True or False')
        regex = re.compile(pat, flags=flags)
        if regex.groups == 0:
            raise ValueError('pattern contains no capture groups')
        if not expand and regex.groups > 1 and isinstance(self._data, ABCIndex):
            raise ValueError('only one regex group is supported with Index')
        obj = self._data
        result_dtype = _result_dtype(obj)
        returns_df = regex.groups > 1 or expand
        if returns_df:
            name = None
            columns = _get_group_names(regex)
            if obj.array.size == 0:
                result = DataFrame(columns=columns, dtype=result_dtype)
            else:
                result_list = self._data.array._str_extract(pat, flags=flags, expand=returns_df)
                if isinstance(obj, ABCSeries):
                    result_index = obj.index
                else:
                    result_index = None
                result = DataFrame(result_list, columns=columns, index=result_index, dtype=result_dtype)
        else:
            name = _get_single_group_name(regex)
            result = self._data.array._str_extract(pat, flags=flags, expand=returns_df)
        return self._wrap_result(result, name=name, dtype=result_dtype)

    @forbid_nonstring_types(['bytes'])
    def extractall(self, pat: str, flags: int = 0) -> Any:
        return str_extractall(self._orig, pat, flags)


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


def _result_dtype(arr: Any) -> Any:
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object


def _get_single_group_name(regex: re.Pattern) -> Optional[str]:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None


def _get_group_names(regex: re.Pattern) -> List[Union[int, str]]:
    rng = range(regex.groups)
    names = {v: k for k, v in regex.groupindex.items()}
    if not names:
        return list(rng)
    result = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == 'i' and lib.is_range_indexer(arr, len(arr)):
        return list(rng)
    return result


def str_extractall(arr: Union[ABCIndex, Any], pat: str, flags: int = 0) -> Any:
    regex = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError('pattern contains no capture groups')
    from pandas import MultiIndex, DataFrame
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns = _get_group_names(regex)
    match_list: List[List[Any]] = []
    index_list: List[Tuple[Any, ...]] = []
    is_mi = arr.index.nlevels > 1
    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)
            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = [np.nan if group == '' else group for group in match_tuple]
                match_list.append(na_tuple)
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)
    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ['match'])
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(match_list, index=index, columns=columns, dtype=dtype)
    return result