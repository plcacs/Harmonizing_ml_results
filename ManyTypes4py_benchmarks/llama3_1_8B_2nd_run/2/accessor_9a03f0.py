from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import TYPE_CHECKING, Literal, cast
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

_shared_docs = {}
_cpython_optimized_encoders = ('utf-8', 'utf8', 'latin-1', 'latin1', 'iso-8859-1', 'mbcs', 'ascii')
_cpython_optimized_decoders = _cpython_optimized_encoders + ('utf-16', 'utf-32')

def forbid_nonstring_types(forbidden: list[str] | None, name: str | None = None) -> Callable:
    """
    Decorator to forbid specific types for a method of StringMethods.

    For calling `.str.{method}` on a Series or Index, it is necessary to first
    initialize the :class:`StringMethods` object, and then call the method.
    However, different methods allow different input types, and so this can not
    be checked during :meth:`StringMethods.__init__`, but must be done on a
    per-method basis. This decorator exists to facilitate this process, and
    make it explicit which (inferred) types are disallowed by the method.

    :meth:`StringMethods.__init__` allows the *union* of types its different
    methods allow (after skipping NaNs; see :meth:`StringMethods._validate`),
    namely: ['string', 'empty', 'bytes', 'mixed', 'mixed-integer'].

    The default string types ['string', 'empty'] are allowed for all methods.
    For the additional types ['bytes', 'mixed', 'mixed-integer'], each method
    then needs to forbid the types it is not intended for.

    Parameters
    ----------
    forbidden : list-of-str or None
        List of forbidden non-string types, may be one or more of
        `['bytes', 'mixed', 'mixed-integer']`.
    name : str, default None
        Name of the method to use in the error message. By default, this is
        None, in which case the name from the method being wrapped will be
        copied. However, for working with further wrappers (like _pat_wrapper
        and _noarg_wrapper), it is necessary to specify the name.

    Returns
    -------
    func : wrapper
        The method to which the decorator is applied, with an added check that
        enforces the inferred type to not be in the list of forbidden types.

    Raises
    ------
    TypeError
        If the inferred type of the underlying data is in `forbidden`.
    """
    forbidden = [] if forbidden is None else forbidden
    allowed_types = {'string', 'empty', 'bytes', 'mixed', 'mixed-integer'} - set(forbidden)

    def _forbid_nonstring_types(func: F) -> F:
        func_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self: StringMethods, *args, **kwargs) -> F:
            if self._inferred_dtype not in allowed_types:
                msg = f"Cannot use .str.{func_name} with values of inferred dtype '{self._inferred_dtype}'."
                raise TypeError(msg)
            return func(self, *args, **kwargs)
        wrapper.__name__ = func_name
        return cast(F, wrapper)
    return _forbid_nonstring_types

def _map_and_wrap(name: str, docstring: str) -> F:
    """
    Wrapper function for mapping a string method to a pandas method.
    """
    @forbid_nonstring_types(['bytes'], name=name)
    def wrapper(self: StringMethods) -> F:
        result = getattr(self._data.array, f'_str_{name}')()
        return self._wrap_result(result, returns_string=name not in ('isnumeric', 'isdecimal'))
    wrapper.__doc__ = docstring
    return wrapper

class StringMethods(NoNewAttributesMixin):
    """
    Vectorized string functions for Series and Index.

    NAs stay NA unless handled otherwise by a particular method.
    Patterned after Python's string methods, with some inspiration from
    R's stringr package.

    Parameters
    ----------
    data : Series or Index
        The content of the Series or Index.

    See Also
    --------
    Series.str : Vectorized string functions for Series.
    Index.str : Vectorized string functions for Index.

    Examples
    --------
    >>> s = pd.Series(["A_Str_Series"])
    >>> s
    0    A_Str_Series
    dtype: object

    >>> s.str.split("_")
    0    [A, Str, Series]
    dtype: object

    >>> s.str.replace("_", "")
    0    AStrSeries
    dtype: object
    """

    def __init__(self, data: Series | Index) -> None:
        from pandas.core.arrays.string_ import StringDtype
        self._inferred_dtype = self._validate(data)
        self._is_categorical = isinstance(data.dtype, CategoricalDtype)
        self._is_string = isinstance(data.dtype, StringDtype)
        self._data = data
        self._index = self._name = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name
        self._parent = data
        self._orig = data
        self._freeze()

    @staticmethod
    def _validate(data: Series | Index) -> str:
        """
        Auxiliary function for StringMethods, infers and checks dtype of data.

        This is a "first line of defence" at the creation of the StringMethods-
        object, and just checks that the dtype is in the
        *union* of the allowed types over all string methods below; this
        restriction is then refined on a per-method basis using the decorator
        @forbid_nonstring_types (more info in the corresponding docstring).

        This really should exclude all series/index with any non-string values,
        but that isn't practical for performance reasons until we have a str
        dtype (GH 9343 / 13877)

        Parameters
        ----------
        data : The content of the Series

        Returns
        -------
        dtype : inferred dtype of data
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

    def __getitem__(self, key: Hashable) -> Series | Index:
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator[None]:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(self, result: object, name: str | None = None, expand: bool | None = None, fill_value: Scalar | None = None, returns_string: bool = True, dtype: DtypeObj | None = None) -> Series | Index:
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

                def cons_row(x: object) -> list[str]:
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

    def _get_series_list(self, others: Series | Index | np.ndarray | list[str] | list[object]) -> list[Series]:
        """
        Auxiliary function for :meth:`str.cat`. Turn potentially mixed input
        into a list of Series (elements without an index must match the length
        of the calling Series/Index).

        Parameters
        ----------
        others : Series, DataFrame, np.ndarray, list-like or list-like of
            Objects that are either Series, Index or np.ndarray (1-dim).

        Returns
        -------
        list of Series
            Others transformed into list of Series.
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
                if all((isinstance(x, (ABCSeries, ABCIndex, ExtensionArray)) or (isinstance(x, np.ndarray) and x.ndim == 1) for x in others)):
                    los = []
                    while others:
                        los = los + self._get_series_list(others.pop(0))
                    return los
                elif all((not is_list_like(x) for x in others)):
                    return [Series(others, index=idx)]
        raise TypeError('others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])')

    @forbid_nonstring_types(['bytes'])
    def cat(self, others: Series | Index | np.ndarray | list[str] | list[object], sep: str | None = None, na_rep: str | None = None, join: Literal['left', 'right', 'outer', 'inner'] = 'left') -> Series | Index:
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
            out = Index(result, dtype=dtype, name=self._orig.name)
        else:
            res_ser = Series(result, dtype=dtype, index=data.index, name=self._orig.name, copy=False)
            out = res_ser.__finalize__(self._orig, method='str_cat')
        return out

    _shared_docs['str_split'] = '\n    Split strings around given separator/delimiter.\n\n    Splits the string in the Series/Index from the %(side)s,\n    at the specified delimiter string.\n\n    Parameters\n    ----------\n    pat : str%(pat_regex)s, optional\n        %(pat_description)s.\n        If not specified, split on whitespace.\n    n : int, default -1 (all)\n        Limit number of splits in output.\n        ``None``, 0 and -1 will be interpreted as return all splits.\n    expand : bool, default False\n        Expand the split strings into separate columns.\n\n        - If ``True``, return DataFrame/MultiIndex expanding dimensionality.\n        - If ``False``, return Series/Index, containing lists of strings.\n    %(regex_argument)s\n    Returns\n    -------\n    Series, Index, DataFrame or MultiIndex\n        Type matches caller unless ``expand=True`` (see Notes).\n    %(raises_split)s\n    See Also\n    --------\n    Series.str.split : Split strings around given separator/delimiter.\n    Series.str.rsplit : Splits string around given separator/delimiter,\n        starting from the right.\n    Series.str.join : Join lists contained as elements in the Series/Index\n        with passed delimiter.\n    str.split : Standard library version for split.\n    str.rsplit : Standard library version for rsplit.\n\n    Notes\n    -----\n    The handling of the `n` keyword depends on the number of found splits:\n\n    - If found splits > `n`,  make first `n` splits only\n    - If found splits <= `n`, make all splits\n    - If for a certain row the number of found splits < `n`,\n      append `None` for padding up to `n` if ``expand=True``\n\n    If using ``expand=True``, Series and Index callers return DataFrame and\n    MultiIndex objects, respectively.\n    %(regex_pat_note)s\n    Examples\n    --------\n    >>> s = pd.Series(\n    ...     [\n    ...         "this is a regular sentence",\n    ...         "https://docs.python.org/3/tutorial/index.html",\n    ...         np.nan\n    ...     ]\n    ... )\n    >>> s\n    0                       this is a regular sentence\n    1    https://docs.python.org/3/tutorial/index.html\n    2                                              NaN\n    dtype: object\n\n    In the default setting, the string is split by whitespace.\n\n    >>> s.str.split()\n    0                   [this, is, a, regular, sentence]\n    1    [https://docs.python.org/3/tutorial/index.html]\n    2                                                NaN\n    dtype: object\n\n    Without the `n` parameter, the outputs of `rsplit` and `split`\n    are identical.\n\n    >>> s.str.rsplit()\n    0                   [this, is, a, regular, sentence]\n    1    [https://docs.python.org/3/tutorial/index.html]\n    2                                                NaN\n    dtype: object\n\n    The `n` parameter can be used to limit the number of splits on the\n    delimiter. The outputs of `split` and `rsplit` are different.\n\n    >>> s.str.split(n=2)\n    0                     [this, is, a regular sentence]\n    1    [https://docs.python.org/3/tutorial/index.html]\n    2                                                NaN\n    dtype: object\n\n    >>> s.str.rsplit(n=2)\n    0                     [this is a, regular, sentence]\n    1    [https://docs.python.org/3/tutorial/index.html]\n    2                                                NaN\n    dtype: object\n\n    The `pat` parameter can be used to split by other characters.\n\n    >>> s.str.split(pat="/")\n    0                         [this is a regular sentence]\n    1    [https:, , docs.python.org, 3, tutorial, index...\n    2                                                  NaN\n    dtype: object\n\n    When using ``expand=True``, the split elements will expand out into\n    separate columns. If NaN is present, it is propagated throughout\n    the columns during the split.\n\n    >>> s.str.split(expand=True)\n                                                   0     1     2        3         4\n    0                                           this    is     a  regular  sentence\n    1  https://docs.python.org/3/tutorial/index.html  None  None     None      None\n    2                                            NaN   NaN   NaN      NaN       NaN\n\n    For slightly more complex use cases like splitting the html document name\n    from a url, a combination of parameter settings can be used.\n\n    >>> s.str.rsplit("/", n=1, expand=True)\n                                        0           1\n    0          this is a regular sentence        None\n    1  https://docs.python.org/3/tutorial  index.html\n    2                                 NaN         NaN\n    %(regex_examples)s'

    @Appender(_shared_docs['str_split'] % {'side': 'beginning', 'pat_regex': ' or compiled regex', 'pat_description': 'String or regular expression to split on', 'regex_argument': '\n    regex : bool, default None\n        Determines if the passed-in pattern is a regular expression:\n\n        - If ``True``, assumes the passed-in pattern is a regular expression\n        - If ``False``, treats the pattern as a literal string.\n        - If ``None`` and `pat` length is 1, treats `pat` as a literal string.\n        - If ``None`` and `pat` length is not 1, treats `pat` as a regular expression.\n        - Cannot be set to False if `pat` is a compiled regex\n\n        .. versionadded:: 1.4.0\n         ', 'raises_split': '\n                      Raises\n                      ------\n                      ValueError\n                          * if `regex` is False and `pat` is a compiled regex\n                      ', 'regex_pat_note': '\n    Use of `regex =False` with a `pat` as a compiled regex will raise an error.\n            ', 'method': 'split', 'regex_examples': '\n    Remember to escape special characters when explicitly using regular expressions.\n\n    >>> s = pd.Series(["foo and bar plus baz"])\n    >>> s.str.split(r"and|plus", expand=True)\n        0   1   2\n    0 foo bar baz\n\n    Regular expressions can be used to handle urls or file names.\n    When `pat` is a string and ``regex=None`` (the default), the given `pat` is compiled\n    as a regex only if ``len(pat) != 1``.\n\n    >>> s = pd.Series([\'foojpgbar.jpg\'])\n    >>> s.str.split(r".", expand=True)\n               0    1\n    0  foojpgbar  jpg\n\n    >>> s.str.split(r"\\.jpg", expand=True)\n               0 1\n    0  foojpgbar\n\n    When ``regex=True``, `pat` is interpreted as a regex\n\n    >>> s.str.split(r"\\.jpg", regex=True, expand=True)\n               0 1\n    0  foojpgbar\n\n    A compiled regex can be passed as `pat`\n\n    >>> import re\n    >>> s.str.split(re.compile(r"\\.jpg"), expand=True)\n               0 1\n    0  foojpgbar\n\n    When ``regex=False``, `pat` is interpreted as the string itself\n\n    >>> s.str.split(r"\\.jpg", regex=False, expand=True)\n                   0\n    0  foojpgbar.jpg\n    '})
    @forbid_nonstring_types(['bytes'])
    def split(self, pat: str | None, n: int = -1, expand: bool = False, regex: bool | None = None) -> Series | Index:
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

    @Appender(_shared_docs['str_split'] % {'side': 'end', 'pat_regex': '', 'pat_description': 'String to split on', 'regex_argument': '', 'raises_split': '', 'regex_pat_note': '', 'method': 'rsplit', 'regex_examples': ''})
    @forbid_nonstring_types(['bytes'])
    def rsplit(self, pat: str | None, n: int = -1, expand: bool = False) -> Series | Index:
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)
    _shared_docs['str_partition'] = "\n    Split the string at the %(side)s occurrence of `sep`.\n\n    This method splits the string at the %(side)s occurrence of `sep`,\n    and returns 3 elements containing the part before the separator,\n    the separator itself, and the part after the separator.\n    If the separator is not found, return %(return)s.\n\n    Parameters\n    ----------\n    sep : str, default whitespace\n        String to split on.\n    expand : bool, default True\n        If True, return DataFrame/MultiIndex expanding dimensionality.\n        If False, return Series/Index.\n\n    Returns\n    -------\n    DataFrame/MultiIndex or Series/Index of objects\n        Returns appropriate type based on `expand` parameter with strings\n        split based on the `sep` parameter.\n\n    See Also\n    --------\n    %(also)s\n    Series.str.split : Split strings around given separators.\n    str.partition : Standard library version.\n\n    Examples\n    --------\n\n    >>> s = pd.Series(['Linda van der Berg', 'George Pitt-Rivers'])\n    >>> s\n    0    Linda van der Berg\n    1    George Pitt-Rivers\n    dtype: object\n\n    >>> s.str.partition()\n            0  1             2\n    0   Linda     van der Berg\n    1  George      Pitt-Rivers\n\n    To partition by the last space instead of the first one:\n\n    >>> s.str.rpartition()\n                   0  1            2\n    0  Linda van der            Berg\n    1         George     Pitt-Rivers\n\n    To partition by something different than a space:\n\n    >>> s.str.partition('-')\n                        0  1       2\n    0  Linda van der Berg\n    1         George Pitt  -  Rivers\n\n    To return a Series containing tuples instead of a DataFrame:\n\n    >>> s.str.partition('-', expand=False)\n    0    (Linda van der Berg, , )\n    1    (George Pitt, -, Rivers)\n    dtype: object\n\n    Also available on indices:\n\n    >>> idx = pd.Index(['X 123', 'Y 999'])\n    >>> idx\n    Index(['X 123', 'Y 999'], dtype='object')\n\n    Which will create a MultiIndex:\n\n    >>> idx.str.partition()\n    MultiIndex([('X', ' ', '123'),\n                ('Y', ' ', '999')],\n               )\n\n    Or an index with tuples with ``expand=False``:\n\n    >>> idx.str.partition(expand=False)\n    Index([('X', ' ', '123'), ('Y', ' ', '999')], dtype='object')\n    "

    @Appender(_shared_docs['str_partition'] % {'side': 'first', 'return': '3 elements containing the string itself, followed by two empty strings', 'also': 'rpartition : Split the string at the last occurrence of `sep`.'})
    @forbid_nonstring_types(['bytes'])
    def partition(self, sep: str = ' ', expand: bool = True) -> Series | Index:
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == 'category':
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @Appender(_shared_docs['str_partition'] % {'side': 'last', 'return': '3 elements containing two empty strings, followed by the string itself', 'also': 'partition : Split the string at the first occurrence of `sep`.'})
    @forbid_nonstring_types(['bytes'])
    def rpartition(self, sep: str = ' ', expand: bool = True) -> Series | Index:
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == 'category':
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    def get(self, i: Hashable) -> Series | Index:
        """
        Extract element from each component at specified position or with specified key.

        Extract element from lists, tuples, dict, or strings in each element in the
        Series/Index.

        Parameters
        ----------
        i : int or hashable dict label
            Position or key of element to extract.

        Returns
        -------
        Series or Index
            Series or Index where each value is the extracted element from
            the corresponding input component.

        See Also
        --------
        Series.str.extract : Extract capture groups in the regex as columns
            in a DataFrame.

        Examples
        --------
        >>> s = pd.Series(
        ...     [
        ...         "String",
        ...         (1, 2, 3),
        ...         ["a", "b", "c"],
        ...         123,
        ...         -456,
        ...         {1: "Hello", "2": "World"},
        ...     ]
        ... )
        >>> s
        0                        String
        1                     (1, 2, 3)
        2                     [a, b, c]
        3                           123
        4                          -456
        5    {1: 'Hello', '2': 'World'}
        dtype: object

        >>> s.str.get(1)
        0        t
        1        2
        2        b
        3      NaN
        4      NaN
        5    Hello
        dtype: object

        >>> s.str.get(-1)
        0      g
        1      3
        2      c
        3    NaN
        4    NaN
        5    None
        dtype: object

        Return element with given key

        >>> s = pd.Series(
        ...     [
        ...         {"name": "Hello", "value": "World"},
        ...         {"name": "Goodbye", "value": "Planet"},
        ...     ]
        ... )
        >>> s.str.get("name")
        0      Hello
        1    Goodbye
        dtype: object
        """
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def join(self, sep: str) -> Series | Index:
        """
        Join lists contained as elements in the Series/Index with passed delimiter.

        If the elements of a Series are lists themselves, join the content of these
        lists using the delimiter passed to the function.
        This function is an equivalent to :meth:`str.join`.

        Parameters
        ----------
        sep : str
            Delimiter to use between list entries.

        Returns
        -------
        Series/Index: object
            The list entries concatenated by intervening occurrences of the
            delimiter.

        Raises
        ------
        AttributeError
            If the supplied Series contains neither strings nor lists.

        See Also
        --------
        str.join : Standard library version of this method.
        Series.str.split : Split strings around given separator/delimiter.

        Notes
        -----
        If any of the list items is not a string object, the result of the join
        will be `NaN`.

        Examples
        --------
        Example with a list that contains non-string elements.

        >>> s = pd.Series(
        ...     [
        ...         ["lion", "elephant", "zebra"],
        ...         [1.1, 2.2, 3.3],
        ...         ["cat", np.nan, "dog"],
        ...         ["cow", 4.5, "goat"],
        ...         ["duck", ["swan", "fish"], "guppy"],
        ...     ]
        ... )
        >>> s
        0        [lion, elephant, zebra]
        1                [1.1, 2.2, 3.3]
        2                [cat, nan, dog]
        3               [cow, 4.5, goat]
        4    [duck, [swan, fish], guppy]
        dtype: object

        Join all lists using a '-'. The lists containing object(s) of types other
        than str will produce a NaN.

        >>> s.str.join("-")
        0    lion-elephant-zebra
        1                    NaN
        2                    NaN
        3                    NaN
        4                    NaN
        dtype: object
        """
        result = self._data.array._str_join(sep)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def contains(self, pat: str, case: bool = True, flags: int = 0, na: Scalar | None = lib.no_default, regex: bool = True) -> Series | Index:
        if regex is False and is_re(pat):
            raise ValueError('Cannot use a compiled regex as replacement pattern with regex=False')
        if is_re(pat):
            regex = True
        result = self._data.array._str_contains(pat, case, flags, na, regex)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def match(self, pat: str, case: bool = True, flags: int = 0, na: Scalar | None = lib.no_default) -> Series | Index:
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Scalar | None = lib.no_default) -> Series | Index:
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def replace(self, pat: str | dict[str, str], repl: str | None = None, n: int = -1, case: bool | None = None, flags: int = 0, regex: bool = False) -> Series | Index:
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
    def repeat(self, repeats: int | list[int]) -> Series | Index:
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def pad(self, width: int, side: str = 'left', fillchar: str = ' ') -> Series | Index:
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

    _shared_docs['str_pad'] = "\n    Pad %(side)s side of strings in the Series/Index.\n\n    Equivalent to :meth:`str.%(method)s`.\n\n    Parameters\n    ----------\n    width : int\n        Minimum width of resulting string; additional characters will be filled\n        with ``fillchar``.\n    fillchar : str\n        Additional character for filling, default is whitespace.\n\n    Returns\n    -------\n    Series/Index of objects.\n        A Series or Index where the strings are modified by :meth:`str.%(method)s`.\n\n    See Also\n    --------\n    Series.str.rjust : Fills the left side of strings with an arbitrary\n        character.\n    Series.str.ljust : Fills the right side of strings with an arbitrary\n        character.\n    Series.str.center : Fills both sides of strings with an arbitrary\n        character.\n    Series.str.zfill : Pad strings in the Series/Index by prepending '0'\n        character.\n\n    Examples\n    --------\n    For Series.str.center:\n\n    >>> ser = pd.Series(['dog', 'bird', 'mouse'])\n    >>> ser.str.center(8, fillchar='.')\n    0   ..dog...\n    1   ..bird..\n    2   .mouse..\n    dtype: object\n\n    For Series.str.ljust:\n\n    >>> ser = pd.Series(['dog', 'bird', 'mouse'])\n    >>> ser.str.ljust(8, fillchar='.')\n    0   dog.....\n    1   bird....\n    2   mouse...\n    dtype: object\n\n    For Series.str.rjust:\n\n    >>> ser = pd.Series(['dog', 'bird', 'mouse'])\n    >>> ser.str.rjust(8, fillchar='.')\n    0   .....dog\n    1   ....bird\n    2   ...mouse\n    dtype: object\n    "

    @Appender(_shared_docs['str_pad'] % {'side': 'left and right', 'method': 'center'})
    @forbid_nonstring_types(['bytes'])
    def center(self, width: int, fillchar: str = ' ') -> Series | Index:
        return self.pad(width, side='both', fillchar=fillchar)

    @Appender(_shared_docs['str_pad'] % {'side': 'right', 'method': 'ljust'})
    @forbid_nonstring_types(['bytes'])
    def ljust(self, width: int, fillchar: str = ' ') -> Series | Index:
        return self.pad(width, side='right', fillchar=fillchar)

    @Appender(_shared_docs['str_pad'] % {'side': 'left', 'method': 'rjust'})
    @forbid_nonstring_types(['bytes'])
    def rjust(self, width: int, fillchar: str = ' ') -> Series | Index:
        return self.pad(width, side='left', fillchar=fillchar)

    @forbid_nonstring_types(['bytes'])
    def zfill(self, width: int) -> Series | Index:
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        return self._wrap_result(result)

    def slice(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> Series | Index:
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def slice_replace(self, start: int | None = None, stop: int | None = None, repl: str | None = None) -> Series | Index:
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def decode(self, encoding: str, errors: str = 'strict') -> Series | Index:
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
    def encode(self, encoding: str, errors: str = 'strict') -> Series | Index:
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    _shared_docs['str_strip'] = "\n    Remove %(position)s characters.\n\n    Strip whitespaces (including newlines) or a set of specified characters\n    from each string in the Series/Index from %(side)s.\n    Replaces any non-strings in Series with NaNs.\n    Equivalent to :meth:`str.%(method)s`.\n\n    Parameters\n    ----------\n    to_strip : str or None, default None\n        Specifying the set of characters to be removed.\n        All combinations of this set of characters will be stripped.\n        If None then whitespaces are removed.\n\n    Returns\n    -------\n    Series or Index of object\n        Series or Index with the strings being stripped from the %(side)s.\n\n    See Also\n    --------\n    Series.str.strip : Remove leading and trailing characters in Series/Index.\n    Series.str.lstrip : Remove leading characters in Series/Index.\n    Series.str.rstrip : Remove trailing characters in Series/Index.\n\n    Examples\n    --------\n    >>> s = pd.Series(['1. Ant.  ', '2. Bee!\\n', '3. Cat?\\t', np.nan, 10, True])\n    >>> s\n    0    1. Ant.\n    1    2. Bee!\\n\n    2    3. Cat?\\t\n    3          NaN\n    4           10\n    5         True\n    dtype: object\n\n    >>> s.str.strip()\n    0    1. Ant.\n    1    2. Bee!\n    2    3. Cat?\n    3        NaN\n    4        NaN\n    5        NaN\n    dtype: object\n\n    >>> s.str.lstrip('123.')\n    0    Ant.\n    1    Bee!\\n\n    2    Cat?\\t\n    3       NaN\n    4       NaN\n    5       NaN\n    dtype: object\n\n    >>> s.str.rstrip('.!? \\n\\t')\n    0    1. Ant\n    1    2. Bee\n    2    3. Cat\n    3       NaN\n    4       NaN\n    5       NaN\n    dtype: object\n\n    >>> s.str.strip('123.!? \\n\\t')\n    0    Ant\n    1    Bee\n    2    Cat\n    3    NaN\n    4    NaN\n    5    NaN\n    dtype: object\n    "

    @Appender(_shared_docs['str_strip'] % {'side': 'left and right sides', 'method': 'strip', 'position': 'leading and trailing'})
    @forbid_nonstring_types(['bytes'])
    def strip(self, to_strip: str | None = None) -> Series | Index:
        result = self._data.array._str_strip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_strip'] % {'side': 'left side', 'method': 'lstrip', 'position': 'leading'})
    @forbid_nonstring_types(['bytes'])
    def lstrip(self, to_strip: str | None = None) -> Series | Index:
        result = self._data.array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_strip'] % {'side': 'right side', 'method': 'rstrip', 'position': 'trailing'})
    @forbid_nonstring_types(['bytes'])
    def rstrip(self, to_strip: str | None = None) -> Series | Index:
        result = self._data.array._str_rstrip(to_strip)
        return self._wrap_result(result)

    _shared_docs['str_removefix'] = '\n    Remove a %(side)s from an object series.\n\n    If the %(side)s is not present, the original string will be returned.\n\n    Parameters\n    ----------\n    %(side)s : str\n        Remove the %(side)s of the string.\n\n    Returns\n    -------\n    Series/Index: object\n        The Series or Index with given %(side)s removed.\n\n    See Also\n    --------\n    Series.str.remove%(other_side)s : Remove a %(other_side)s from an object series.\n\n    Examples\n    --------\n    >>> s = pd.Series(["str_foo", "str_bar", "no_prefix"])\n    >>> s\n    0    str_foo\n    1    str_bar\n    2    no_prefix\n    dtype: object\n    >>> s.str.removeprefix("str_")\n    0    foo\n    1    bar\n    2    no_prefix\n    dtype: object\n\n    >>> s = pd.Series(["foo_str", "bar_str", "no_suffix"])\n    >>> s\n    0    foo_str\n    1    bar_str\n    2    no_suffix\n    dtype: object\n    >>> s.str.removesuffix("_str")\n    0    foo\n    1    bar\n    2    no_suffix\n    dtype: object\n    '

    @Appender(_shared_docs['str_removefix'] % {'side': 'prefix', 'other_side': 'suffix'})
    @forbid_nonstring_types(['bytes'])
    def removeprefix(self, prefix: str) -> Series | Index:
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_removefix'] % {'side': 'suffix', 'other_side': 'prefix'})
    @forbid_nonstring_types(['bytes'])
    def removesuffix(self, suffix: str) -> Series | Index:
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def wrap(self, width: int, expand_tabs: bool = True, tabsize: int = 8, replace_whitespace: bool = True, drop_whitespace: bool = True, initial_indent: str = '', subsequent_indent: str = '', fix_sentence_endings: bool = False, break_long_words: bool = True, break_on_hyphens: bool = True, max_lines: int | None = None, placeholder: str = ' [...]') -> Series | Index:
        result = self._data.array._str_wrap(width=width, expand_tabs=expand_tabs, tabsize=tabsize, replace_whitespace=replace_whitespace, drop_whitespace=drop_whitespace, initial_indent=initial_indent, subsequent_indent=subsequent_indent, fix_sentence_endings=fix_sentence_endings, break_long_words=break_long_words, break_on_hyphens=break_on_hyphens, max_lines=max_lines, placeholder=placeholder)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def get_dummies(self, sep: str = '|', dtype: DtypeObj | None = None) -> DataFrame:
        from pandas import DataFrame
        if dtype is not None and (not (is_numeric_dtype(dtype) or is_bool_dtype(dtype))):
            raise ValueError("Only numeric or boolean dtypes are supported for 'dtype'")
        result, name = self._data.array._str_get_dummies(sep, dtype)
        if is_extension_array_dtype(dtype):
            return self._wrap_result(DataFrame(result, columns=name, dtype=dtype), name=name, returns_string=False)
        return self._wrap_result(result, name=name, expand=True, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def translate(self, table: dict[str, str]) -> Series | Index:
        result = self._data.array._str_translate(table)
        dtype = object if self._data.dtype == 'object' else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(['bytes'])
    def count(self, pat: str, flags: int = 0) -> Series | Index:
        result = self._data.array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def startswith(self, pat: str | tuple[str, ...], na: Scalar | None = lib.no_default) -> Series | Index:
        if not isinstance(pat, (str, tuple)):
            msg = f'expected a string or tuple, not {type(pat).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_startswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def endswith(self, pat: str | tuple[str, ...], na: Scalar | None = lib.no_default) -> Series | Index:
        if not isinstance(pat, (str, tuple)):
            msg = f'expected a string or tuple, not {type(pat).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_endswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def findall(self, pat: str, flags: int = 0) -> Series | Index:
        result = self._data.array._str_findall(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def extract(self, pat: str, flags: int = 0, expand: bool = True) -> Series | Index:
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
    def extractall(self, pat: str, flags: int = 0) -> DataFrame:
        return str_extractall(self._orig, pat, flags)

    _shared_docs['find'] = '\n    Return %(side)s indexes in each strings in the Series/Index.\n\n    Each of returned indexes corresponds to the position where the\n    substring is fully contained between [start:end]. Return -1 on\n    failure. Equivalent to standard :meth:`str.%(method)s`.\n\n    Parameters\n    ----------\n    sub : str\n        Substring being searched.\n    start : int\n        Left edge index.\n    end : int\n        Right edge index.\n\n    Returns\n    -------\n    Series or Index of int.\n        A Series (if the input is a Series) or an Index (if the input is an\n        Index) of the %(side)s indexes corresponding to the positions where the\n        substring is found in each string of the input.\n\n    See Also\n    --------\n    %(also)s\n\n    Examples\n    --------\n    For Series.str.find:\n\n    >>> ser = pd.Series(["_cow_", "duck_", "do_v_e"])\n    >>> ser.str.find("_")\n    0   0\n    1   4\n    2   2\n    dtype: int64\n    '

    @Appender(_shared_docs['find'] % {'side': 'lowest', 'method': 'find', 'also': 'rfind : Return highest indexes in each strings.'})
    @forbid_nonstring_types(['bytes'])
    def find(self, sub: str, start: int = 0, end: int | None = None) -> Series | Index:
        if not isinstance(sub, str):
            msg = f'expected a string object, not {type(sub).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_find(sub, start, end)
        return self._wrap_result(result, returns_string=False)

    @Appender(_shared_docs['find'] % {'side': 'highest', 'method': 'rfind', 'also': 'find : Return lowest indexes in each strings.'})
    @forbid_nonstring_types(['bytes'])
    def rfind(self, sub: str, start: int = 0, end: int | None = None) -> Series | Index:
        if not isinstance(sub, str):
            msg = f'expected a string object, not {type(sub).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_rfind(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def normalize(self, form: str) -> Series | Index:
        result = self._data.array._str_normalize(form)
        return self._wrap_result(result)

    def index(self, sub: str, start: int = 0, end: int | None = None) -> Series | Index:
        if not isinstance(sub, str):
            msg = f'expected a string object, not {type(sub).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def rindex(self, sub: str, start: int = 0, end: int | None = None) -> Series | Index:
        if not isinstance(sub, str):
            msg = f'expected a string object, not {type(sub).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def len(self) -> Series | Index:
        result = self._data.array._str_len()
        return self._wrap_result(result, returns_string=False)
    _shared_docs['casemethods'] = "\n    Convert strings in the Series/Index to %(type)s.\n    %(version)s\n    Equivalent to :meth:`str.%(method)s`.\n\n    Returns\n    -------\n    Series or Index of objects\n        A Series or Index where the strings are modified by :meth:`str.%(method)s`.\n\n    See Also\n    --------\n    Series.str.lower : Converts all characters to lowercase.\n    Series.str.upper : Converts all characters to uppercase.\n    Series.str.title : Converts first character of each word to uppercase and\n        remaining to lowercase.\n    Series.str.capitalize : Converts first character to uppercase and\n        remaining to lowercase.\n    Series.str.swapcase : Converts uppercase to lowercase and lowercase to\n        uppercase.\n    Series.str.casefold: Removes all case distinctions in the string.\n\n    Examples\n    --------\n    >>> s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])\n    >>> s\n    0                 lower\n    1              CAPITALS\n    2    this is a sentence\n    3              SwApCaSe\n    dtype: object\n\n    >>> s.str.lower()\n    0                 lower\n    1              capitals\n    2    this is a sentence\n    3              swapcase\n    dtype: object\n\n    >>> s.str.upper()\n    0                 LOWER\n    1              CAPITALS\n    2    THIS IS A SENTENCE\n    3              SWAPCASE\n    dtype: object\n\n    >>> s.str.title()\n    0                 Lower\n    1              Capitals\n    2    This Is A Sentence\n    3              Swapcase\n    dtype: object\n\n    >>> s.str.capitalize()\n    0                 Lower\n    1              Capitals\n    2    This is a sentence\n    3              Swapcase\n    dtype: object\n\n    >>> s.str.swapcase()\n    0                 LOWER\n    1              capitals\n    2    THIS IS A SENTENCE\n    3              sWaPcAsE\n    dtype: object\n    "
    _doc_args = {}
    _doc_args['lower'] = {'type': 'lowercase', 'method': 'lower', 'version': ''}
    _doc_args['upper'] = {'type': 'uppercase', 'method': 'upper', 'version': ''}
    _doc_args['title'] = {'type': 'titlecase', 'method': 'title', 'version': ''}
    _doc_args['capitalize'] = {'type': 'be capitalized', 'method': 'capitalize', 'version': ''}
    _doc_args['swapcase'] = {'type': 'be swapcased', 'method': 'swapcase', 'version': ''}
    _doc_args['casefold'] = {'type': 'be casefolded', 'method': 'casefold', 'version': ''}

    @Appender(_shared_docs['casemethods'] % _doc_args['lower'])
    @forbid_nonstring_types(['bytes'])
    def lower(self) -> Series | Index:
        result = self._data.array._str_lower()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['upper'])
    @forbid_nonstring_types(['bytes'])
    def upper(self) -> Series | Index:
        result = self._data.array._str_upper()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['title'])
    @forbid_nonstring_types(['bytes'])
    def title(self) -> Series | Index:
        result = self._data.array._str_title()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['capitalize'])
    @forbid_nonstring_types(['bytes'])
    def capitalize(self) -> Series | Index:
        result = self._data.array._str_capitalize()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['swapcase'])
    @forbid_nonstring_types(['bytes'])
    def swapcase(self) -> Series | Index:
        result = self._data.array._str_swapcase()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['casefold'])
    @forbid_nonstring_types(['bytes'])
    def casefold(self) -> Series | Index:
        result = self._data.array._str_casefold()
        return self._wrap_result(result)
    _shared_docs['ismethods'] = '\n    Check whether all characters in each string are %(type)s.\n\n    This is equivalent to running the Python string method\n    :meth:`str.%(method)s` for each element of the Series/Index. If a string\n    has zero characters, ``False`` is returned for that check.\n\n    Returns\n    -------\n    Series or Index of bool\n        Series or Index of boolean values with the same length as the original\n        Series/Index.\n    '
    _shared_docs['isalpha'] = "\n    See Also\n    --------\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n\n    >>> s1 = pd.Series(['one', 'one1', '1', ''])\n    >>> s1.str.isalpha()\n    0     True\n    1    False\n    2    False\n    3    False\n    dtype: bool\n    "
    _shared_docs['isnumeric'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n    The ``s.str.isnumeric`` method is the same as ``s3.str.isdigit`` but\n    also includes other characters that can represent quantities such as\n    unicode fractions.\n\n    >>> s1 = pd.Series(['one', 'one1', '1', ''])\n    >>> s1.str.isnumeric()\n    0    False\n    1    False\n    2     True\n    3    False\n    dtype: bool\n    "
    _shared_docs['isalnum'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n    >>> s1 = pd.Series(['one', 'one1', '1', ''])\n    >>> s1.str.isalnum()\n    0     True\n    1     True\n    2     True\n    3    False\n    dtype: bool\n\n    Note that checks against characters mixed with any additional punctuation\n    or whitespace will evaluate to false for an alphanumeric check.\n\n    >>> s2 = pd.Series(['A B', '1.5', '3,000'])\n    >>> s2.str.isalnum()\n    0    False\n    1    False\n    2    False\n    dtype: bool\n    "
    _shared_docs['isdecimal'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n    The ``s3.str.isdecimal`` method checks for characters used to form\n    numbers in base 10.\n\n    >>> s3 = pd.Series(['23', '³', '⅕', ''])\n    >>> s3.str.isdecimal()\n    0     True\n    1    False\n    2    False\n    3    False\n    dtype: bool\n    "
    _shared_docs['isdigit'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n    Similar to ``str.isdecimal`` but also includes special digits, like\n    superscripted and subscripted digits in unicode.\n\n    >>> s3 = pd.Series(['23', '³', '⅕', ''])\n    >>> s3.str.isdigit()\n    0     True\n    1     True\n    2    False\n    3    False\n    dtype: bool\n    "
    _shared_docs['isspace'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n\n    >>> s4 = pd.Series([' ', '\\t\\r\\n ', ''])\n    >>> s4.str.isspace()\n    0     True\n    1     True\n    2    False\n    dtype: bool\n    "
    _shared_docs['islower'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n\n    >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])\n    >>> s5.str.islower()\n    0     True\n    1    False\n    2    False\n    3    False\n    dtype: bool\n    "
    _shared_docs['isupper'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n\n    >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])\n    >>> s5.str.isupper()\n    0    False\n    1    False\n    2     True\n    3    False\n    dtype: bool\n    "
    _shared_docs['istitle'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isascii : Check whether all characters are ascii.\n    Series.str.isupper : Check whether all characters are uppercase.\n\n    Examples\n    --------\n    The ``s5.str.istitle`` method checks for whether all words are in title\n    case (whether only the first letter of each word is capitalized). Words are\n    assumed to be as any sequence of non-numeric characters separated by\n    whitespace characters.\n\n    >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])\n    >>> s5.str.istitle()\n    0    False\n    1     True\n    2    False\n    3    False\n    dtype: bool\n    "
    _shared_docs['isascii'] = "\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    ------------\n    The ``s5.str.isascii`` method checks for whether all characters are ascii\n    characters, which includes digits 0-9, capital and lowercase letters A-Z,\n    and some other special characters.\n\n    >>> s5 = pd.Series(['ö', 'see123', 'hello world', ''])\n    >>> s5.str.isascii()\n    0    False\n    1     True\n    2     True\n    3     True\n    dtype: bool\n    "
    _doc_args['isalnum'] = {'type': 'alphanumeric', 'method': 'isalnum'}
    _doc_args['isalpha'] = {'type': 'alphabetic', 'method': 'isalpha'}
    _doc_args['isdigit'] = {'type': 'digits', 'method': 'isdigit'}
    _doc_args['isspace'] = {'type': 'whitespace', 'method': 'isspace'}
    _doc_args['islower'] = {'type': 'lowercase', 'method': 'islower'}
    _doc_args['isascii'] = {'type': 'ascii', 'method': 'isascii'}
    _doc_args['isupper'] = {'type': 'uppercase', 'method': 'isupper'}
    _doc_args['istitle'] = {'type': 'titlecase', 'method': 'istitle'}
    _doc_args['isnumeric'] = {'type': 'numeric', 'method': 'isnumeric'}
    _doc_args['isdecimal'] = {'type': 'decimal', 'method': 'isdecimal'}

    isalnum = _map_and_wrap('isalnum', docstring=_shared_docs['ismethods'] % _doc_args['isalnum'] + _shared_docs['isalnum'])
    isalpha = _map_and_wrap('isalpha', docstring=_shared_docs['ismethods'] % _doc_args['isalpha'] + _shared_docs['isalpha'])
    isdigit = _map_and_wrap('isdigit', docstring=_shared_docs['ismethods'] % _doc_args['isdigit'] + _shared_docs['isdigit'])
    isspace = _map_and_wrap('isspace', docstring=_shared_docs['ismethods'] % _doc_args['isspace'] + _shared_docs['isspace'])
    islower = _map_and_wrap('islower', docstring=_shared_docs['ismethods'] % _doc_args['islower'] + _shared_docs['islower'])
    isascii = _map_and_wrap('isascii', docstring=_shared_docs['ismethods'] % _doc_args['isascii'] + _shared_docs['isascii'])
    isupper = _map_and_wrap('isupper', docstring=_shared_docs['ismethods'] % _doc_args['isupper'] + _shared_docs['isupper'])
    istitle = _map_and_wrap('istitle', docstring=_shared_docs['ismethods'] % _doc_args['istitle'] + _shared_docs['istitle'])
    isnumeric = _map_and_wrap('isnumeric', docstring=_shared_docs['ismethods'] % _doc_args['isnumeric'] + _shared_docs['isnumeric'])
    isdecimal = _map_and_wrap('isdecimal', docstring=_shared_docs['ismethods'] % _doc_args['isdecimal'] + _shared_docs['isdecimal'])

def cat_safe(list_of_columns: list[npt.ndarray], sep: str) -> npt.ndarray:
    """
    Auxiliary function for :meth:`str.cat`.

    Same signature as cat_core, but handles TypeErrors in concatenation, which
    happen if the arrays in list_of columns have the wrong dtypes or content.

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    nd.array
        The concatenation of list_of_columns with sep.
    """
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ['string', 'empty']:
                raise TypeError(f'Concatenation requires list-likes containing only strings (or missing values). Offending values found in column {dtype}') from None
    return result

def cat_core(list_of_columns: list[npt.ndarray], sep: str) -> npt.ndarray:
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    nd.array
        The concatenation of list_of_columns with sep.
    """
    if sep == '':
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)

def _result_dtype(arr: Series | Index) -> DtypeObj:
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object

def _get_single_group_name(regex: re.Pattern) -> str | None:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None

def _get_group_names(regex: re.Pattern) -> list[str]:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : compiled regex

    Returns
    -------
    list of column labels
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

def str_extractall(arr: Series | Index, pat: str, flags: int = 0) -> DataFrame:
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
                na_tuple = [np.nan if group == '' else group for group in match_tuple]
                match_list.append(na_tuple)
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)
    from pandas import MultiIndex
    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ['match'])
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(match_list, index=index, columns=columns, dtype=dtype)
    return result
