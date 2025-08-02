from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Callable,
    List,
    Set,
    Dict,
    Tuple,
    Any,
)
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas._typing import AlignJoin, DtypeObj, F, Scalar, npt
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_extension_array_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_re,
)
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCMultiIndex, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator
    from pandas._typing import NpDtype
    from pandas import DataFrame, Index, Series

_shared_docs: Dict[str, str] = {}
_cpython_optimized_encoders: Tuple[str, ...] = (
    'utf-8',
    'utf8',
    'latin-1',
    'latin1',
    'iso-8859-1',
    'mbcs',
    'ascii',
)
_cpython_optimized_decoders: Tuple[str, ...] = _cpython_optimized_encoders + (
    'utf-16',
    'utf-32',
)


def func_6rxv26hk(
    forbidden: Optional[List[str]], name: Optional[str] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to forbid specific types for a method of StringMethods.
    """
    forbidden = [] if forbidden is None else forbidden
    allowed_types: Set[str] = {'string', 'empty', 'bytes', 'mixed', 'mixed-integer'} - set(
        forbidden
    )

    def func_67wcsjpn(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name: str = func.__name__ if name is None else name

        @wraps(func)
        def func_cillb8qt(self: StringMethods, *args: Any, **kwargs: Any) -> Any:
            if self._inferred_dtype not in allowed_types:
                msg: str = (
                    f"Cannot use .str.{func_name} with values of inferred dtype '{self._inferred_dtype}'."
                )
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        func_cillb8qt.__name__ = func_name
        return cast(Callable[..., Any], func_cillb8qt)

    return func_67wcsjpn


def func_1keyjoio(name: str, docstring: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    @func_6rxv26hk(['bytes'], name=name)
    def func_cillb8qt(self: StringMethods) -> Any:
        result = getattr(self._data.array, f'_str_{name}')()
        return self._wrap_result(result, returns_string=name not in ('isnumeric', 'isdecimal'))

    func_cillb8qt.__doc__ = docstring
    return func_cillb8qt


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

    def __init__(self, data: Union[Series, Index]) -> None:
        from pandas.core.arrays.string_ import StringDtype

        self._inferred_dtype: str = self._validate(data)
        self._is_categorical: bool = isinstance(data.dtype, CategoricalDtype)
        self._is_string: bool = isinstance(data.dtype, StringDtype)
        self._data: Union[Series, Index] = data
        self._index: Optional[Index] = None
        self._name: Optional[Hashable] = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name
        self._parent: Union[Series, Index, ExtensionArray] = (
            data._values.categories if self._is_categorical else data
        )
        self._orig: Union[Series, Index] = data
        self._freeze()

    @staticmethod
    def func_7jhjjinp(data: Union[Series, Index]) -> str:
        """
        Auxiliary function for StringMethods, infers and checks dtype of data.
        """
        if isinstance(data, ABCMultiIndex):
            raise AttributeError('Can only use .str accessor with Index, not MultiIndex')
        allowed_types: List[str] = ['string', 'empty', 'bytes', 'mixed', 'mixed-integer']
        data_extracted: ExtensionArray = extract_array(data)
        values: Union[ExtensionArray, Any] = getattr(data_extracted, 'categories', data_extracted)
        inferred_dtype: str = lib.infer_dtype(values, skipna=True)
        if inferred_dtype not in allowed_types:
            raise AttributeError(
                f'Can only use .str accessor with string values, not {inferred_dtype}'
            )
        return inferred_dtype

    def __getitem__(self, key: Any) -> Any:
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator[Any]:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def func_22k1fd5l(
        self,
        result: Any,
        name: Optional[str] = None,
        expand: Optional[bool] = None,
        fill_value: float = np.nan,
        returns_string: bool = True,
        dtype: Optional[DtypeObj] = None,
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        from pandas import Index, MultiIndex, DataFrame

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
                max_len: int = pa.compute.max(value_lengths).as_py()
                min_len: int = pa.compute.min(value_lengths).as_py()
                if result._hasna:
                    result = ArrowExtensionArray(result._pa_array.fill_null([None] * max_len))
                if min_len < max_len:
                    if not pa_version_under11p0:
                        result = ArrowExtensionArray(
                            pa.compute.list_slice(
                                result._pa_array, start=0, stop=max_len, return_fixed_size_list=True
                            )
                        )
                    else:
                        all_null = np.full(max_len, fill_value=None, dtype=object)
                        values = result.to_numpy()
                        new_values: List[Any] = []
                        for row in values:
                            if len(row) < max_len:
                                nulls = all_null[: max_len - len(row)]
                                row = np.append(row, nulls)
                            new_values.append(row)
                        pa_type = result._pa_array.type
                        result = ArrowExtensionArray(pa.array(new_values, type=pa_type))
                if name is None:
                    name = range(max_len)
                result = pa.compute.list_flatten(result._pa_array).to_numpy().reshape(len(result), max_len)
                result = {label: ArrowExtensionArray(pa.array(res)) for label, res in zip(name, result.T)}
            elif is_object_dtype(result):
                def func_nyqy5cn6(x: Any) -> List[Any]:
                    if is_list_like(x):
                        return x
                    else:
                        return [x]

                result = [func_nyqy5cn6(x) for x in result]
                if result and not self._is_string:
                    max_len = max(len(x) for x in result)
                    result = [
                        (x * max_len if len(x) == 0 or x[0] is np.nan else x) for x in result
                    ]

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

    def func_8sbkfk15(self, others: Any) -> List[Series]:
        """
        Auxiliary function for :meth:`str.cat`. Turn potentially mixed input
        into a list of Series (elements without an index must match the length
        of the calling Series/Index).
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
            others_df = DataFrame(others, index=idx)
            return [others_df[x] for x in others_df]
        elif is_list_like(others, allow_sets=False):
            try:
                others = list(others)
            except TypeError:
                pass
            else:
                if all(
                    isinstance(x, (ABCSeries, ABCIndex, ExtensionArray))
                    or (isinstance(x, np.ndarray) and x.ndim == 1)
                    for x in others
                ):
                    los: List[Series] = []
                    while others:
                        los = los + self._get_series_list(others.pop(0))
                    return los
                elif all(not is_list_like(x) for x in others):
                    return [Series(others, index=idx)]
        raise TypeError(
            'others must be Series, Index, DataFrame, np.ndarray or list-like '
            '(either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])'
        )

    @func_6rxv26hk(['bytes', 'mixed', 'mixed-integer'])
    def func_h4z6w7g9(
        self,
        others: Optional[Any] = None,
        sep: Optional[str] = None,
        na_rep: Optional[str] = None,
        join: AlignJoin = 'left',
    ) -> Union[str, Series, Index]:
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
            raise ValueError(
                'If `others` contains arrays or lists (or other list-likes without an index), these must all be of the same length as the calling Series/Index.'
            ) from err
        if any(not data.index.equals(x.index) for x in others):
            others = concat(
                others,
                axis=1,
                join=join if join == 'inner' else 'outer',
                keys=range(len(others)),
                sort=False,
            )
            data, others = data.align(others, join=join)
            others = [others[x] for x in others]
        all_cols: List[np.ndarray] = [ensure_object(x) for x in [data] + others]
        na_masks: np.ndarray = np.array([isna(x) for x in all_cols])
        union_mask: np.ndarray = np.logical_or.reduce(na_masks, axis=0)
        if na_rep is None and union_mask.any():
            result: np.ndarray = np.empty(len(data), dtype=object)
            np.putmask(result, union_mask, np.nan)
            not_masked: np.ndarray = ~union_mask
            result[not_masked] = cat_safe([x[not_masked] for x in all_cols], sep)
        elif na_rep is not None and union_mask.any():
            all_cols = [np.where(nm, na_rep, col) for nm, col in zip(na_masks, all_cols)]
            result = cat_safe(all_cols, sep)
        else:
            result = cat_safe(all_cols, sep)

        if isinstance(self._orig.dtype, CategoricalDtype):
            dtype: Optional[DtypeObj] = self._orig.dtype.categories.dtype
        else:
            dtype = self._orig.dtype

        if isinstance(self._orig, ABCIndex):
            if isna(result).all():
                dtype = object
            out: Union[Index, MultiIndex] = Index(result, dtype=dtype, name=self._orig.name)
            if isinstance(out, MultiIndex) and out.nlevels == 1:
                out = out.get_level_values(0)
            return out if not isinstance(out, MultiIndex) else out
        else:
            index = self._orig.index
            _dtype: Optional[DtypeObj] = dtype
            vdtype = getattr(result, 'dtype', None)
            if _dtype is None:
                if self._is_string:
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

    def func_8sbkfk15(self, others: Any) -> List[Series]:
        pass  # Placeholder for actual implementation

    def _wrap_result(
        self,
        result: Any,
        name: Optional[str] = None,
        expand: bool = False,
        returns_string: bool = True,
        dtype: Optional[DtypeObj] = None,
        fill_value: Optional[Any] = None,
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        # Placeholder for the actual _wrap_result implementation
        pass

    def _get_series_list(self, others: Any) -> List[Series]:
        # Placeholder for the actual _get_series_list implementation
        pass

    def _validate(self, data: Union[Series, Index]) -> str:
        # Placeholder for the actual _validate implementation
        pass

    def _freeze(self) -> None:
        # Placeholder for the actual _freeze implementation
        pass

    @Appender(_shared_docs['str_split'] % {'side': 'beginning', 'pat_regex': ' or compiled regex', 'pat_description': 'String or regular expression to split on', 'regex_argument': """
        regex : bool, default None
            Determines if the passed-in pattern is a regular expression:

            - If ``True``, assumes the passed-in pattern is a regular expression
            - If ``False``, treats the pattern as a literal string.
            - If ``None`` and `pat` length is 1, treats `pat` as a literal string.
            - If ``None`` and `pat` length is not 1, treats `pat` as a regular expression.
            - Cannot be set to False if `pat` is a compiled regex

            .. versionadded:: 1.4.0
             """
        , 'raises_split': """
                          Raises
                          ------
                          ValueError
                              * if `regex` is False and `pat` is a compiled regex
                          """
        , 'regex_pat_note': """
        Use of `regex =False` with a `pat` as a compiled regex will raise an error.
                """
        , 'method': 'split', 'regex_examples': """
        Remember to escape special characters when explicitly using regular expressions.

        >>> s = pd.Series(["foo and bar plus baz"])
        >>> s.str.split(r"and|plus", expand=True)
            0   1   2
        0 foo bar baz

        Regular expressions can be used to handle urls or file names.
        When `pat` is a string and ``regex=None`` (the default), the given `pat` is compiled
        as a regex only if ``len(pat) != 1``.

        >>> s = pd.Series(['foojpgbar.jpg'])
        >>> s.str.split(r".", expand=True)
                   0    1
        0  foojpgbar  jpg

        >>> s.str.split(r"\\.jpg", expand=True)
                   0 1
        0  foojpgbar

        When ``regex=True``, `pat` is interpreted as a regex

        >>> s.str.split(r"\\.jpg", regex=True, expand=True)
                   0 1
        0  foojpgbar

        A compiled regex can be passed as `pat`

        >>> import re
        >>> s.str.split(re.compile(r"\\.jpg"), expand=True)
                   0 1
        0  foojpgbar

        When ``regex=False``, `pat` is interpreted as the string itself

        >>> s.str.split(r"\\.jpg", regex=False, expand=True)
                       0
        0  foojpgbar.jpg
        """
        })
    @func_6rxv26hk(['bytes'])
    def func_pxzn8qcb(
        self,
        pat: Optional[str] = None,
        *,
        n: int = -1,
        expand: bool = False,
        regex: Optional[bool] = None,
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        if regex is False and is_re(pat):
            raise ValueError(
                'Cannot use a compiled regex as replacement pattern with regex=False'
            )
        if is_re(pat):
            regex = True
        result = self._data.array._str_split(pat, n, expand, regex)
        if self._data.dtype == 'category':
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(_shared_docs['str_split'] % {'side': 'end', 'pat_regex': '', 'pat_description': 'String to split on', 'regex_argument': '', 'raises_split': '', 'regex_pat_note': '', 'method': 'rsplit', 'regex_examples': ''})
    @func_6rxv26hk(['bytes'])
    def func_puopbp51(
        self, pat: Optional[str] = None, *, n: int = -1, expand: bool = False
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    _shared_docs['str_partition'] = """
        Split the string at the %(side)s occurrence of `sep`.

        This method splits the string at the %(side)s occurrence of `sep`,
        and returns 3 elements containing the part before the separator,
        the separator itself, and the part after the separator.
        If the separator is not found, return %(return)s.

        Parameters
        ----------
        sep : str, default whitespace
            String to split on.
        expand : bool, default True
            If True, return DataFrame/MultiIndex expanding dimensionality.
            If False, return Series/Index.

        Returns
        -------
        DataFrame/MultiIndex or Series/Index of objects
            Returns appropriate type based on `expand` parameter with strings
            split based on the `sep` parameter.

        See Also
        --------
        %(also)s
        Series.str.split : Split strings around given separators.
        str.partition : Standard library version.

        Examples
        --------

        >>> s = pd.Series(['Linda van der Berg', 'George Pitt-Rivers'])
        >>> s
        0    Linda van der Berg
        1    George Pitt-Rivers
        dtype: object

        >>> s.str.partition()
                0  1             2
        0   Linda     van der Berg
        1  George      Pitt-Rivers

        To partition by the last space instead of the first one:

        >>> s.str.rpartition()
                       0  1            2
        0  Linda van der            Berg
        1         George     Pitt-Rivers

        To partition by something different than a space:

        >>> s.str.partition('-')
                            0  1       2
        0  Linda van der Berg
        1         George Pitt  -  Rivers

        To return a Series containing tuples instead of a DataFrame:

        >>> s.str.partition('-', expand=False)
        0    (Linda van der Berg, , )
        1    (George Pitt, -, Rivers)
        dtype: object

        Also available on indices:

        >>> idx = pd.Index(['X 123', 'Y 999'])
        >>> idx
        Index(['X 123', 'Y 999'], dtype='object')

        Which will create a MultiIndex:

        >>> idx.str.partition()
        MultiIndex([('X', ' ', '123'),
                    ('Y', ' ', '999')],
                   )

        Or an index with tuples with ``expand=False``:

        >>> idx.str.partition(expand=False)
        Index([('X', ' ', '123'), ('Y', ' ', '999')], dtype='object')
        """

    @Appender(_shared_docs['str_partition'] % {'side': 'first', 'return': '3 elements containing the string itself, followed by two empty strings', 'also': 'rpartition : Split the string at the last occurrence of `sep`.'})
    @func_6rxv26hk(['bytes'])
    def func_0vysn2pk(
        self, sep: str = ' ', expand: bool = True
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == 'category':
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(_shared_docs['str_partition'] % {'side': 'last', 'return': '3 elements containing two empty strings, followed by the string itself', 'also': 'partition : Split the string at the first occurrence of `sep`.'})
    @func_6rxv26hk(['bytes'])
    def func_ssvvrh11(
        self, sep: str = ' ', expand: bool = True
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == 'category':
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    def func_o8ajmpt2(self, i: Union[int, Hashable]) -> Series:
        """
        Extract element from each component at specified position or with specified key.
        """
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_o29cu1tj(self, sep: str) -> Union[Series, Index]:
        """
        Join lists contained as elements in the Series/Index with passed delimiter.
        """
        result = self._data.array._str_join(sep)
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_msyx4up8(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Any = lib.no_default,
        regex: bool = True,
    ) -> Union[Series, Index]:
        """
        Test if pattern or regex is contained within a string of a Series or Index.
        """
        if regex and re.compile(pat).groups:
            warnings.warn(
                'This pattern is interpreted as a regular expression, and has match groups. '
                'To actually get the groups, use str.extract.',
                UserWarning,
                stacklevel=find_stack_level(),
            )
        result = self._data.array._str_contains(pat, case, flags, na, regex)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @func_6rxv26hk(['bytes'])
    def func_owt6woiz(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Any = lib.no_default,
    ) -> Union[Series, Index]:
        """
        Determine if each string starts with a match of a regular expression.
        """
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @func_6rxv26hk(['bytes'])
    def func_xeur9qtk(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Any = lib.no_default,
    ) -> Union[Series, Index]:
        """
        Determine if each string entirely matches a regular expression.
        """
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @func_6rxv26hk(['bytes'])
    def func_ap9qb488(
        self,
        pat: Union[str, Dict[str, str]],
        repl: Optional[Union[str, Callable[[re.Match], str]]] = None,
        n: int = -1,
        case: Optional[bool] = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Union[Series, Index]:
        """
        Replace each occurrence of pattern/regex in the Series/Index.
        """
        if isinstance(pat, dict) and repl is not None:
            raise ValueError('repl cannot be used when pat is a dictionary')
        if not isinstance(pat, dict) and not (isinstance(repl, str) or callable(repl)):
            raise TypeError('repl must be a string or callable')
        is_compiled_re: bool = is_re(pat)
        if regex or regex is None:
            if is_compiled_re and (case is not None or flags != 0):
                raise ValueError(
                    'case and flags cannot be set when pat is a compiled regex'
                )
        elif is_compiled_re:
            raise ValueError(
                'Cannot use a compiled regex as replacement pattern with regex=False'
            )
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

    @func_6rxv26hk(['bytes'])
    def func_5huo21j2(self, repeats: Union[int, List[int]]) -> Union[Series, Index]:
        """
        Duplicate each string in the Series or Index.
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_14o7lkcs(
        self,
        width: int,
        side: Literal['left', 'right', 'both'] = 'left',
        fillchar: str = ' ',
    ) -> Union[Series, Index]:
        """
        Pad strings in the Series/Index up to width.
        """
        if not isinstance(fillchar, str):
            msg: str = f'fillchar must be a character, not {type(fillchar).__name__}'
            raise TypeError(msg)
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        if not is_integer(width):
            msg: str = f'width must be of integer type, not {type(width).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    _shared_docs['str_pad'] = """
        Pad %(side)s side of strings in the Series/Index.

        Equivalent to :meth:`str.%(method)s`.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be filled
            with ``fillchar``.
        fillchar : str
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of objects.
            A Series or Index where the strings are modified by :meth:`str.%(method)s`.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character.
        Series.str.center : Fills both sides of strings with an arbitrary
            character.
        Series.str.zfill : Pad strings in the Series/Index by prepending '0'
            character.

        Examples
        --------
        For Series.str.center:

        >>> ser = pd.Series(['dog', 'bird', 'mouse'])
        >>> ser.str.center(8, fillchar='.')
        0   ..dog...
        1   ..bird..
        2   .mouse..
        dtype: object

        For Series.str.ljust:

        >>> ser = pd.Series(['dog', 'bird', 'mouse'])
        >>> ser.str.ljust(8, fillchar='.')
        0   dog.....
        1   bird....
        2   mouse...
        dtype: object

        For Series.str.rjust:

        >>> ser = pd.Series(['dog', 'bird', 'mouse'])
        >>> ser.str.rjust(8, fillchar='.')
        0   .....dog
        1   ....bird
        2   ...mouse
        dtype: object
    """

    @Appender(_shared_docs['str_pad'] % {'side': 'left and right', 'method': 'center'})
    @func_6rxv26hk(['bytes'])
    def func_dif9bz67(
        self, width: int, fillchar: str = ' '
    ) -> Union[Series, Index]:
        return self.func_14o7lkcs(width, side='both', fillchar=fillchar)

    @Appender(_shared_docs['str_pad'] % {'side': 'right', 'method': 'ljust'})
    @func_6rxv26hk(['bytes'])
    def func_zj2dg9je(
        self, width: int, fillchar: str = ' '
    ) -> Union[Series, Index]:
        return self.func_14o7lkcs(width, side='right', fillchar=fillchar)

    @Appender(_shared_docs['str_pad'] % {'side': 'left', 'method': 'rjust'})
    @func_6rxv26hk(['bytes'])
    def func_f9ka5swm(
        self, width: int, fillchar: str = ' '
    ) -> Union[Series, Index]:
        return self.func_14o7lkcs(width, side='left', fillchar=fillchar)

    @func_6rxv26hk(['bytes'])
    def func_bmknq9c2(self, width: int) -> Union[Series, Index]:
        """
        Pad strings in the Series/Index by prepending '0' characters.
        """
        if not is_integer(width):
            msg: str = f'width must be of integer type, not {type(width).__name__}'
            raise TypeError(msg)
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        dtype: Optional[str] = 'str' if get_option('future.infer_string') else None
        return self._wrap_result(result, dtype=dtype)

    def func_561af8o7(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Union[Series, Index]:
        """
        Slice substrings from each element in the Series or Index.
        """
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_ajpxf1yt(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        repl: Optional[str] = None,
    ) -> Union[Series, Index]:
        """
        Replace a positional slice of a string with another value.
        """
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def func_3i222q23(self, encoding: str, errors: str = 'strict') -> Union[Series, Index]:
        """
        Decode character string in the Series/Index using indicated encoding.
        """
        if encoding in _cpython_optimized_decoders:
            f: Callable[[bytes], str] = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        arr = self._data.array
        result = arr._str_map(f)
        dtype: Optional[str] = 'str' if get_option('future.infer_string') else None
        return self._wrap_result(result, dtype=dtype)

    @func_6rxv26hk(['bytes'])
    def func_mxui4s5t(
        self, encoding: str, errors: str = 'strict'
    ) -> Union[Series, Index]:
        """
        Encode character string in the Series/Index using indicated encoding.
        """
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    def func_561af8o7(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> Union[Series, Index]:
        """
        Slice substrings from each element in the Series or Index.
        """
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    def func_3i222q23(self, encoding: str, errors: str = 'strict') -> Union[Series, Index]:
        """
        Decode character string in the Series/Index using indicated encoding.
        """
        if encoding in _cpython_optimized_decoders:
            f: Callable[[bytes], str] = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        arr = self._data.array
        result = arr._str_map(f)
        dtype: Optional[str] = 'str' if get_option('future.infer_string') else None
        return self._wrap_result(result, dtype=dtype)

    @func_6rxv26hk(['bytes'])
    def func_mxui4s5t(
        self, encoding: str, errors: str = 'strict'
    ) -> Union[Series, Index]:
        """
        Encode character string in the Series/Index using indicated encoding.
        """
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    @func_6rxv26hk(['bytes'])
    def func_qse3a4t8(self, sep: str = '|', dtype: Optional[DtypeObj] = None) -> DataFrame:
        """
        Return DataFrame of dummy/indicator variables for Series.
        """
        from pandas.core.frame import DataFrame

        if dtype is not None and not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)):
            raise ValueError("Only numeric or boolean dtypes are supported for 'dtype'")
        result, name = self._data.array._str_get_dummies(sep, dtype)
        if is_extension_array_dtype(dtype):
            return self._wrap_result(DataFrame(result, columns=name, dtype=dtype), name=name, returns_string=False)
        return self._wrap_result(result, name=name, expand=True, returns_string=False)

    @func_6rxv26hk(['bytes'])
    def func_6tpy1roj(
        self,
        repeats: Union[int, List[int]],
    ) -> Union[Series, Index]:
        """
        Duplicate each string in the Series or Index.
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_14o7lkcs(
        self,
        width: int,
        side: Literal['left', 'right', 'both'] = 'left',
        fillchar: str = ' ',
    ) -> Union[Series, Index]:
        """
        Pad strings in the Series/Index up to width.
        """
        if not isinstance(fillchar, str):
            msg: str = (
                f'fillchar must be a character, not {type(fillchar).__name__}'
            )
            raise TypeError(msg)
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        if not is_integer(width):
            msg: str = f'width must be of integer type, not {type(width).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    def len(self) -> Series:
        """
        Compute the length of each element in the Series/Index.
        """
        result = self._data.array._str_len()
        return self._wrap_result(result, returns_string=False)

    _shared_docs['casemethods'] = """
        Convert strings in the Series/Index to %(type)s.
        %(version)s
        Equivalent to :meth:`str.%(method)s`.

        Returns
        -------
        Series or Index of objects
            A Series or Index where the strings are modified by :meth:`str.%(method)s`.

        See Also
        --------
        Series.str.lower : Convert all characters in each string to lowercase.
        Series.str.upper : Convert all characters in each string to uppercase.
        Series.str.title : Convert each string to title case (capitalizing the first
            letter of each word).
        Series.str.capitalize : Convert first character to uppercase and
            remaining to lowercase.
        Series.str.swapcase : Convert uppercase to lowercase and lowercase to
            uppercase.
        Series.str.casefold: Remove all case distinctions in the string.

        Examples
        --------
        >>> s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object

        >>> s.str.lower()
        0                 lower
        1              capitals
        2    this is a sentence
        3              swapcase
        dtype: object

        >>> s.str.upper()
        0                 LOWER
        1              CAPITALS
        2    THIS IS A SENTENCE
        3              SWAPCASE
        dtype: object

        >>> s.str.title()
        0                 Lower
        1              Capitals
        2    This Is A Sentence
        3              Swapcase
        dtype: object

        >>> s.str.capitalize()
        0                 Lower
        1              Capitals
        2    This is a sentence
        3              Swapcase
        dtype: object

        >>> s.str.swapcase()
        0                 LOWER
        1              capitals
        2    THIS IS A SENTENCE
        3              sWaPcAsE
        dtype: object
    """
    _doc_args: Dict[str, Dict[str, Any]] = {}
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

    _shared_docs['isalpha'] = """
        See Also
        --------
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        --------

        >>> s1 = pd.Series(['one', 'one1', '1', ''])
        >>> s1.str.isalpha()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
    """

    _shared_docs['isnumeric'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        --------

        The ``s.str.isnumeric`` method is the same as ``s3.str.isdigit`` but
        also includes other characters that can represent quantities such as
        unicode fractions.

        >>> s1 = pd.Series(['one', 'one1', '1', ''])
        >>> s1.str.isnumeric()
        0    False
        1    False
        2     True
        3    False
        dtype: bool
    """

    _shared_docs['isalnum'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        --------

        >>> s1 = pd.Series(['one', 'one1', '1', ''])
        >>> s1.str.isalnum()
        0     True
        1     True
        2     True
        3    False
        dtype: bool

        Note that checks against characters mixed with any additional punctuation
        or whitespace will evaluate to false for an alphanumeric check.

        >>> s2 = pd.Series(['A B', '1.5', '3,000'])
        >>> s2.str.isalnum()
        0    False
        1    False
        2    False
        dtype: bool
    """

    _shared_docs['isdecimal'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        --------

        The ``s3.str.isdecimal`` method checks for characters used to form
        numbers in base 10.

        >>> s3 = pd.Series(['23', '³', '⅕', ''])
        >>> s3.str.isdecimal()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
    """

    _shared_docs['isdigit'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        --------

        Similar to ``str.isdecimal`` but also includes special digits, like
        superscripted and subscripted digits in unicode.

        >>> s3 = pd.Series(['23', '³', '⅕', ''])
        >>> s3.str.isdigit()
        0     True
        1     True
        2    False
        3    False
        dtype: bool
    """

    _shared_docs['isspace'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        --------

        >>> s4 = pd.Series([' ', '\t\r\n ', ''])
        >>> s4.str.isspace()
        0     True
        1     True
        2    False
        dtype: bool
    """

    _shared_docs['islower'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        --------

        >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s5.str.islower()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
    """

    _shared_docs['isupper'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        --------

        >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s5.str.isupper()
        0    False
        1    False
        2     True
        3    False
        dtype: bool
    """

    _shared_docs['istitle'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.

        Examples
        --------

        The ``s5.str.istitle`` method checks for whether all words are in title
        case (whether only the first letter of each word is capitalized). Words are
        assumed to be as any sequence of non-numeric characters separated by
        whitespace characters.

        >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s5.str.istitle()
        0    False
        1     True
        2    False
        3    False
        dtype: bool
    """

    _shared_docs['isascii'] = """
        See Also
        --------
        Series.str.isalpha : Check whether all characters are alphabetic.
        Series.str.isnumeric : Check whether all characters are numeric.
        Series.str.isalnum : Check whether all characters are alphanumeric.
        Series.str.isdigit : Check whether all characters are digits.
        Series.str.isdecimal : Check whether all characters are decimal.
        Series.str.isspace : Check whether all characters are whitespace.
        Series.str.islower : Check whether all characters are lowercase.
        Series.str.isascii : Check whether all characters are ascii.
        Series.str.isupper : Check whether all characters are uppercase.
        Series.str.istitle : Check whether all characters are titlecase.

        Examples
        ------------

        The ``s5.str.isascii`` method checks for whether all characters are ascii
        characters, which includes digits 0-9, capital and lowercase letters A-Z,
        and some other special characters.

        >>> s5 = pd.Series(['ö', 'see123', 'hello world', ''])
        >>> s5.str.isascii()
        0    False
        1     True
        2     True
        3     True
        dtype: bool
    """

    isalnum: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'isalnum',
        docstring=_shared_docs['ismethods'] % _doc_args['isalnum'] + _shared_docs['isalnum'],
    )
    isalpha: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'isalpha',
        docstring=_shared_docs['ismethods'] % _doc_args['isalpha'] + _shared_docs['isalpha'],
    )
    isdigit: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'isdigit',
        docstring=_shared_docs['ismethods'] % _doc_args['isdigit'] + _shared_docs['isdigit'],
    )
    isspace: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'isspace',
        docstring=_shared_docs['ismethods'] % _doc_args['isspace'] + _shared_docs['isspace'],
    )
    islower: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'islower',
        docstring=_shared_docs['ismethods'] % _doc_args['islower'] + _shared_docs['islower'],
    )
    isascii: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'isascii',
        docstring=_shared_docs['ismethods'] % _doc_args['isascii'] + _shared_docs['isascii'],
    )
    isupper: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'isupper',
        docstring=_shared_docs['ismethods'] % _doc_args['isupper'] + _shared_docs['isupper'],
    )
    istitle: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'istitle',
        docstring=_shared_docs['ismethods'] % _doc_args['istitle'] + _shared_docs['istitle'],
    )
    isnumeric: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'isnumeric',
        docstring=_shared_docs['ismethods'] % _doc_args['isnumeric'] + _shared_docs['isnumeric'],
    )
    isdecimal: Callable[..., Union[Series, Index]] = func_1keyjoio(
        'isdecimal',
        docstring=_shared_docs['ismethods'] % _doc_args['isdecimal'] + _shared_docs['isdecimal'],
    )

    @func_6rxv26hk(['bytes'])
    def func_if5cuvdp(
        self,
        width: int,
        expand_tabs: bool = True,
        tabsize: int = 8,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        initial_indent: str = '',
        subsequent_indent: str = '',
        fix_sentence_endings: bool = False,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
        max_lines: Optional[int] = None,
        placeholder: str = ' [...]',
    ) -> Union[Series, Index]:
        """
        Wrap strings in Series/Index at specified line width.
        """
        result = self._data.array._str_wrap(
            width=width,
            expand_tabs=expand_tabs,
            tabsize=tabsize,
            replace_whitespace=replace_whitespace,
            drop_whitespace=drop_whitespace,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            fix_sentence_endings=fix_sentence_endings,
            break_long_words=break_long_words,
            break_on_hyphens=break_on_hyphens,
            max_lines=max_lines,
            placeholder=placeholder,
        )
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_ycc1g20c(
        self, sep: str = '|', dtype: Optional[DtypeObj] = None
    ) -> DataFrame:
        """
        Return DataFrame of dummy/indicator variables for Series.
        """
        from pandas.core.frame import DataFrame

        if dtype is not None and not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)):
            raise ValueError("Only numeric or boolean dtypes are supported for 'dtype'")
        result, name = self._data.array._str_get_dummies(sep, dtype)
        if is_extension_array_dtype(dtype):
            return self._wrap_result(
                DataFrame(result, columns=name, dtype=dtype), name=name, returns_string=False
            )
        return self._wrap_result(result, name=name, expand=True, returns_string=False)

    @func_6rxv26hk(['bytes'])
    def func_6tpy1roj(
        self, repeats: Union[int, List[int]]
    ) -> Union[Series, Index]:
        """
        Duplicate each string in the Series or Index.
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_14o7lkcs(
        self,
        width: int,
        side: Literal['left', 'right', 'both'] = 'left',
        fillchar: str = ' ',
    ) -> Union[Series, Index]:
        """
        Pad strings in the Series/Index up to width.
        """
        if not isinstance(fillchar, str):
            msg: str = (
                f'fillchar must be a character, not {type(fillchar).__name__}'
            )
            raise TypeError(msg)
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        if not is_integer(width):
            msg: str = f'width must be of integer type, not {type(width).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_strip'] % {'side': 'left and right sides', 'method': 'strip', 'position': 'leading and trailing'})
    @func_6rxv26hk(['bytes'])
    def func_or3doxm0(
        self, to_strip: Optional[str] = None
    ) -> Union[Series, Index]:
        result = self._data.array._str_strip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_strip'] % {'side': 'left side', 'method': 'lstrip', 'position': 'leading'})
    @func_6rxv26hk(['bytes'])
    def func_q6apsig1(
        self, to_strip: Optional[str] = None
    ) -> Union[Series, Index]:
        result = self._data.array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_strip'] % {'side': 'right side', 'method': 'rstrip', 'position': 'trailing'})
    @func_6rxv26hk(['bytes'])
    def func_rub1xbls(
        self, to_strip: Optional[str] = None
    ) -> Union[Series, Index]:
        result = self._data.array._str_rstrip(to_strip)
        return self._wrap_result(result)

    _shared_docs['str_removefix'] = """
        Remove a %(side)s from an object series.

        If the %(side)s is not present, the original string will be returned.

        Parameters
        ----------
        %(side)s : str
            Remove the %(side)s of the string.

        Returns
        -------
        Series/Index: object
            The Series or Index with given %(side)s removed.

        See Also
        --------
        Series.str.remove%(other_side)s : Remove a %(other_side)s from an object series.

        Examples
        --------
        >>> s = pd.Series(["str_foo", "str_bar", "no_prefix"])
        >>> s
        0    str_foo
        1    str_bar
        2    no_prefix
        dtype: object
        >>> s.str.removeprefix("str_")
        0    foo
        1    bar
        2    no_prefix
        dtype: object

        >>> s = pd.Series(["foo_str", "bar_str", "no_suffix"])
        >>> s
        0    foo_str
        1    bar_str
        2    no_suffix
        dtype: object
        >>> s.str.removesuffix("_str")
        0    foo
        1    bar
        2    no_suffix
        dtype: object
    """

    @Appender(_shared_docs['str_removefix'] % {'side': 'prefix', 'other_side': 'suffix'})
    @func_6rxv26hk(['bytes'])
    def func_pqctj78t(
        self, prefix: str
    ) -> Union[Series, Index]:
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @Appender(_shared_docs['str_removefix'] % {'side': 'suffix', 'other_side': 'prefix'})
    @func_6rxv26hk(['bytes'])
    def func_na04ilkf(
        self, suffix: str
    ) -> Union[Series, Index]:
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_if5cuvdp(
        self,
        width: int,
        expand_tabs: bool = True,
        tabsize: int = 8,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        initial_indent: str = '',
        subsequent_indent: str = '',
        fix_sentence_endings: bool = False,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
        max_lines: Optional[int] = None,
        placeholder: str = ' [...]',
    ) -> Union[Series, Index]:
        """
        Wrap strings in Series/Index at specified line width.
        """
        result = self._data.array._str_wrap(
            width=width,
            expand_tabs=expand_tabs,
            tabsize=tabsize,
            replace_whitespace=replace_whitespace,
            drop_whitespace=drop_whitespace,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            fix_sentence_endings=fix_sentence_endings,
            break_long_words=break_long_words,
            break_on_hyphens=break_on_hyphens,
            max_lines=max_lines,
            placeholder=placeholder,
        )
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_ycc1g20c(
        self, sep: str = '|', dtype: Optional[DtypeObj] = None
    ) -> DataFrame:
        """
        Return DataFrame of dummy/indicator variables for Series.
        """
        from pandas.core.frame import DataFrame

        if dtype is not None and not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)):
            raise ValueError("Only numeric or boolean dtypes are supported for 'dtype'")
        result, name = self._data.array._str_get_dummies(sep, dtype)
        if is_extension_array_dtype(dtype):
            return self._wrap_result(
                DataFrame(result, columns=name, dtype=dtype), name=name, returns_string=False
            )
        return self._wrap_result(result, name=name, expand=True, returns_string=False)

    @func_6rxv26hk(['bytes'])
    def func_6tpy1roj(
        self, repeats: Union[int, List[int]]
    ) -> Union[Series, Index]:
        """
        Duplicate each string in the Series or Index.
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @func_6rxv26hk(['bytes'])
    def func_14o7lkcs(
        self,
        width: int,
        side: Literal['left', 'right', 'both'] = 'left',
        fillchar: str = ' ',
    ) -> Union[Series, Index]:
        """
        Pad strings in the Series/Index up to width.
        """
        if not isinstance(fillchar, str):
            msg: str = (
                f'fillchar must be a character, not {type(fillchar).__name__}'
            )
            raise TypeError(msg)
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        if not is_integer(width):
            msg: str = f'width must be of integer type, not {type(width).__name__}'
            raise TypeError(msg)
        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    _shared_docs['str_removefix'] = """
        Remove a %(side)s from an object series.

        If the %(side)s is not present, the original string will be returned.

        Parameters
        ----------
        %(side)s : str
            Remove the %(side)s of the string.

        Returns
        -------
        Series/Index: object
            The Series or Index with given %(side)s removed.

        See Also
        --------
        Series.str.remove%(other_side)s : Remove a %(other_side)s from an object series.

        Examples
        --------
        >>> s = pd.Series(["str_foo", "str_bar", "no_prefix"])
        >>> s
        0    str_foo
        1    str_bar
        2    no_prefix
        dtype: object
        >>> s.str.removeprefix("str_")
        0    foo
        1    bar
        2    no_prefix
        dtype: object

        >>> s = pd.Series(["foo_str", "bar_str", "no_suffix"])
        >>> s
        0    foo_str
        1    bar_str
        2    no_suffix
        dtype: object
        >>> s.str.removesuffix("_str")
        0    foo
        1    bar
        2    no_suffix
        dtype: object
    """

    def func_pl2oya31(
        list_of_columns: List[np.ndarray], sep: str
    ) -> np.ndarray:
        """
        Auxiliary function for :meth:`str.cat`.

        Same signature as cat_core, but handles TypeErrors in concatenation, which
        happen if the arrays in list_of columns have the wrong dtypes or content.

        Parameters
        ----------
        list_of_columns : list-of-ndarray
            List of arrays to be concatenated with sep;
            these arrays may not contain NaNs!
        sep : str
            The separator string for concatenating the columns.

        Returns
        -------
        np.ndarray
            The concatenation of list_of_columns with sep.
        """
        try:
            result = cat_core(list_of_columns, sep)
        except TypeError:
            for column in list_of_columns:
                dtype = lib.infer_dtype(column, skipna=True)
                if dtype not in ['string', 'empty']:
                    raise TypeError(
                        f'Concatenation requires list-likes containing only strings (or missing values). Offending values found in column {dtype}'
                    ) from None
            result = np.array(list_of_columns[0])  # Fallback in case
        return result

    def func_au58jps4(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
        """
        Auxiliary function for :meth:`str.cat`

        Parameters
        ----------
        list_of_columns : list-of-ndarray
            List of arrays to be concatenated with sep;
            these arrays may not contain NaNs!
        sep : str
            The separator string for concatenating the columns.

        Returns
        -------
        np.ndarray
            The concatenation of list_of_columns with sep.
        """
        if sep == '':
            arr_of_cols = np.asarray(list_of_columns, dtype=object)
            return np.sum(arr_of_cols, axis=0)
        list_with_sep: List[Union[np.ndarray, str]] = [sep] * (2 * len(list_of_columns) - 1)
        list_with_sep[::2] = list_of_columns
        arr_with_sep = np.asarray(list_with_sep, dtype=object)
        return np.sum(arr_with_sep, axis=0)

    def func_sd2ysqrk(arr: ExtensionArray) -> Optional[DtypeObj]:
        from pandas.core.arrays.string_ import StringDtype
        if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
            return arr.dtype
        return object

    def func_p2th25pv(regex: re.Pattern) -> Optional[str]:
        if regex.groupindex:
            return next(iter(regex.groupindex))
        else:
            return None

    def func_wvdsvtgm(regex: re.Pattern) -> List[Any]:
        """
        Get named groups from compiled regex.

        Unnamed groups are numbered.

        Parameters
        ----------
        regex : re.Pattern
            Compiled regex pattern.

        Returns
        -------
        List[Any]
            List of group names or numbers.
        """
        rng = range(regex.groups)
        names: Dict[int, str] = {v: k for k, v in regex.groupindex.items()}
        if not names:
            return list(rng)
        result: List[Any] = [names.get(1 + i, i) for i in rng]
        arr = np.array(result)
        if arr.dtype.kind == 'i' and lib.is_range_indexer(arr, len(arr)):
            return list(rng)
        return result

    def func_wnqevzpp(arr: Series, pat: str, flags: int = 0) -> DataFrame:
        """
        Extract capture groups in the regex `pat` from the Series.

        Parameters
        ----------
        arr : Series
            The Series to extract from.
        pat : str
            Regular expression pattern with capturing groups.
        flags : int, default 0
            Flags for regex compilation.

        Returns
        -------
        DataFrame
            Extracted groups as a DataFrame.
        """
        regex = re.compile(pat, flags=flags)
        if regex.groups == 0:
            raise ValueError('pattern contains no capture groups')
        if isinstance(arr, ABCIndex):
            arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
        columns = func_wvdsvtgm(regex)
        match_list: List[Tuple[Any, ...]] = []
        index_list: List[Tuple[Any, ...]] = []
        is_mi: bool = arr.index.nlevels > 1
        for subject_key, subject in arr.items():
            if isinstance(subject, str):
                if not is_mi:
                    subject_key = (subject_key,)
                for match_i, match_tuple in enumerate(regex.findall(subject)):
                    if isinstance(match_tuple, str):
                        match_tuple = (match_tuple,)
                    na_tuple = tuple(np.nan if group == '' else group for group in match_tuple)
                    match_list.append(na_tuple)
                    result_key = tuple(subject_key + (match_i,))
                    index_list.append(result_key)
        from pandas import MultiIndex

        index = MultiIndex.from_tuples(index_list, names=arr.index.names + ['match'])
        dtype = func_sd2ysqrk(arr)
        result = arr._constructor_expanddim(match_list, index=index, columns=columns, dtype=dtype)
        return result

    @Appender(_shared_docs['casemethods'] % _doc_args['lower'])
    @func_6rxv26hk(['bytes'])
    def func_6h3aihch(self) -> Union[Series, Index]:
        """
        Convert strings in the Series/Index to lowercase.
        """
        result = self._data.array._str_lower()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['upper'])
    @func_6rxv26hk(['bytes'])
    def func_diefblpt(self) -> Union[Series, Index]:
        """
        Convert strings in the Series/Index to uppercase.
        """
        result = self._data.array._str_upper()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['title'])
    @func_6rxv26hk(['bytes'])
    def func_smrqilpm(self) -> Union[Series, Index]:
        """
        Convert strings in the Series/Index to titlecase.
        """
        result = self._data.array._str_title()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['capitalize'])
    @func_6rxv26hk(['bytes'])
    def func_17vcgcww(self) -> Union[Series, Index]:
        """
        Convert strings in the Series/Index to capitalized.
        """
        result = self._data.array._str_capitalize()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['swapcase'])
    @func_6rxv26hk(['bytes'])
    def func_14f2cj0w(self) -> Union[Series, Index]:
        """
        Convert strings in the Series/Index to swapcase.
        """
        result = self._data.array._str_swapcase()
        return self._wrap_result(result)

    @Appender(_shared_docs['casemethods'] % _doc_args['casefold'])
    @func_6rxv26hk(['bytes'])
    def func_025a2q0d(self) -> Union[Series, Index]:
        """
        Convert strings in the Series/Index to casefolded.
        """
        result = self._data.array._str_casefold()
        return self._wrap_result(result)
    

def func_pl2oya31(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`.

    Same signature as cat_core, but handles TypeErrors in concatenation, which
    happen if the arrays in list_of columns have the wrong dtypes or content.

    Parameters
    ----------
    list_of_columns : list-of-ndarray
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : str
        The separator string for concatenating the columns.

    Returns
    -------
    np.ndarray
        The concatenation of list_of_columns with sep.
    """
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ['string', 'empty']:
                raise TypeError(
                    f'Concatenation requires list-likes containing only strings (or missing values). Offending values found in column {dtype}'
                ) from None
        result = np.array(list_of_columns[0])  # Fallback in case
    return result


def func_au58jps4(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list-of-ndarray
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : str
        The separator string for concatenating the columns.

    Returns
    -------
    np.ndarray
        The concatenation of list_of_columns with sep.
    """
    if sep == '':
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep: List[Union[np.ndarray, str]] = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)


def func_sd2ysqrk(arr: ExtensionArray) -> Optional[DtypeObj]:
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object


def func_p2th25pv(regex: re.Pattern) -> Optional[str]:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None


def func_wvdsvtgm(regex: re.Pattern) -> List[Any]:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : re.Pattern
        Compiled regex pattern.

    Returns
    -------
    List[Any]
        List of group names or numbers.
    """
    rng = range(regex.groups)
    names: Dict[int, str] = {v: k for k, v in regex.groupindex.items()}
    if not names:
        return list(rng)
    result: List[Any] = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == 'i' and lib.is_range_indexer(arr, len(arr)):
        return list(rng)
    return result


def func_wnqevzpp(arr: Series, pat: str, flags: int = 0) -> DataFrame:
    """
    Extract capture groups in the regex `pat` from the Series.

    Parameters
    ----------
    arr : Series
        The Series to extract from.
    pat : str
        Regular expression pattern with capturing groups.
    flags : int, default 0
        Flags for regex compilation.

    Returns
    -------
    DataFrame
        Extracted groups as a DataFrame.
    """
    regex = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError('pattern contains no capture groups')
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns = func_wvdsvtgm(regex)
    match_list: List[Tuple[Any, ...]] = []
    index_list: List[Tuple[Any, ...]] = []
    is_mi: bool = arr.index.nlevels > 1
    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)
            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = tuple(np.nan if group == '' else group for group in match_tuple)
                match_list.append(na_tuple)
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)
    from pandas import MultiIndex

    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ['match'])
    dtype = func_sd2ysqrk(arr)
    result = arr._constructor_expanddim(match_list, index=index, columns=columns, dtype=dtype)
    return result
