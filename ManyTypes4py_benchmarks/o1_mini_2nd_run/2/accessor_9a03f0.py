from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Literal,
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
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from pandas import DataFrame, Index, Series
    from pandas._typing import NpDtype

_shared_docs: dict = {}
_cpython_optimized_encoders: Tuple[str, ...] = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
_cpython_optimized_decoders: Tuple[str, ...] = _cpython_optimized_encoders + (
    "utf-16",
    "utf-32",
)

def forbid_nonstring_types(
    forbidden: Optional[List[str]], name: Optional[str] = None
) -> Callable[[F], F]:
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
    allowed_types: Set[str] = {
        "string",
        "empty",
        "bytes",
        "mixed",
        "mixed-integer",
    } - set(forbidden)

    def _forbid_nonstring_types(func: F) -> F:
        func_name: str = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self: StringMethods, *args: Any, **kwargs: Any) -> Any:
            if self._inferred_dtype not in allowed_types:
                msg = (
                    f"Cannot use .str.{func_name} with values of inferred dtype "
                    f"'{self._inferred_dtype}'."
                )
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        wrapper.__name__ = func_name
        return cast(F, wrapper)

    return _forbid_nonstring_types

def _map_and_wrap(name: str, docstring: str) -> Callable[[StringMethods], Any]:
    @forbid_nonstring_types(["bytes"], name=name)
    def wrapper(self: StringMethods) -> Any:
        result = getattr(self._data.array, f"_str_{name}")()
        return self._wrap_result(
            result, returns_string=name not in ("isnumeric", "isdecimal")
        )

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
        self._parent: Union[Index, Series] = (
            data._values.categories if self._is_categorical else data
        )
        self._orig: Union[Series, Index] = data
        self._freeze()

    @staticmethod
    def _validate(data: Union[Series, Index]) -> str:
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
            raise AttributeError(
                "Can only use .str accessor with Index, not MultiIndex"
            )
        allowed_types = ["string", "empty", "bytes", "mixed", "mixed-integer"]
        data_extracted = extract_array(data)
        values = getattr(data_extracted, "categories", data_extracted)
        inferred_dtype: str = lib.infer_dtype(values, skipna=True)
        if inferred_dtype not in allowed_types:
            raise AttributeError(
                f"Can only use .str accessor with string values, not {inferred_dtype}"
            )
        return inferred_dtype

    def __getitem__(self, key: Any) -> Any:
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator[str]:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(
        self,
        result: Any,
        name: Optional[Hashable] = None,
        expand: Optional[bool] = None,
        fill_value: Scalar = np.nan,
        returns_string: bool = True,
        dtype: Optional[DtypeObj] = None,
    ) -> Union[Series, Index, DataFrame, Any]:
        from pandas import Index, MultiIndex, DataFrame

        if not hasattr(result, "ndim") or not hasattr(result, "dtype"):
            if isinstance(result, ABCDataFrame):
                result = result.__finalize__(self._orig, name="str")
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
                    result = ArrowExtensionArray(
                        result._pa_array.fill_null([None] * max_len)
                    )
                if min_len < max_len:
                    if not pa_version_under11p0:
                        result = ArrowExtensionArray(
                            pa.compute.list_slice(
                                result._pa_array,
                                start=0,
                                stop=max_len,
                                return_fixed_size_list=True,
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
                result = pa.compute.list_flatten(result._pa_array).to_numpy().reshape(
                    len(result), max_len
                )
                result = {
                    label: ArrowExtensionArray(pa.array(res))
                    for label, res in zip(name, result.T)
                }
            elif is_object_dtype(result):
                def cons_row(x: Any) -> List[Any]:
                    if is_list_like(x):
                        return list(x)
                    else:
                        return [x]

                result = [cons_row(x) for x in result]
                if result and (not self._is_string):
                    max_len = max((len(x) for x in result))
                    result = [
                        x * max_len if len(x) == 0 or x[0] is np.nan else x
                        for x in result
                    ]

        if not isinstance(expand, bool):
            raise ValueError("expand must be True or False")
        if expand is False:
            if name is None:
                name = getattr(result, "name", None)
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
            _dtype_inner: Optional[DtypeObj] = dtype
            vdtype = getattr(result, "dtype", None)
            if _dtype_inner is not None:
                pass
            elif self._is_string:
                if is_bool_dtype(vdtype):
                    _dtype_inner = result.dtype
                elif returns_string:
                    _dtype_inner = self._orig.dtype
                else:
                    _dtype_inner = vdtype
            elif vdtype is not None:
                _dtype_inner = vdtype
            if expand:
                cons = self._orig._constructor_expanddim
                result = cons(result, columns=name, index=index, dtype=_dtype_inner)
            else:
                cons = self._orig._constructor
                result = cons(result, name=name, index=index, dtype=_dtype_inner)
            result = result.__finalize__(self._orig, method="str")
            if name is not None and result.ndim == 1:
                result.name = name
            return result

    def _get_series_list(self, others: Any) -> List[Series]:
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

        idx: Union[Index, Series] = (
            self._orig if isinstance(self._orig, ABCIndex) else self._orig.index
        )
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
                    isinstance(
                        x,
                        (
                            ABCSeries,
                            ABCIndex,
                            ExtensionArray,
                        ),
                    )
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
            "others must be Series, Index, DataFrame, np.ndarray or list-like "
            "(either containing only strings or containing only objects of type "
            "Series/Index/np.ndarray[1-dim])"
        )

    @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
    def cat(
        self,
        others: Optional[Any] = None,
        sep: Optional[str] = None,
        na_rep: Optional[str] = None,
        join: AlignJoin = "left",
    ) -> Union[str, Series, Index]:
        """
        Concatenate strings in the Series/Index with given separator.

        If `others` is specified, this function concatenates the Series/Index
        and elements of `others` element-wise.
        If `others` is not passed, then all values in the Series/Index are
        concatenated into a single string with a given `sep`.

        Parameters
        ----------
        others : Series, Index, DataFrame, np.ndarray or list-like
            Series, Index, DataFrame, np.ndarray (one- or two-dimensional) and
            other list-likes of strings must have the same length as the
            calling Series/Index, with the exception of indexed objects (i.e.
            Series/Index/DataFrame) if `join` is not None.

            If others is a list-like that contains a combination of Series,
            Index or np.ndarray (1-dim), then all elements will be unpacked and
            must satisfy the above criteria individually.

            If others is None, the method returns the concatenation of all
            strings in the calling Series/Index.
        sep : str, default ''
            The separator between the different elements/columns. By default
            the empty string `''` is used.
        na_rep : str or None, default None
            Representation that is inserted for all missing values:

            - If `na_rep` is None, and `others` is None, missing values in the
              Series/Index are omitted from the result.
            - If `na_rep` is None, and `others` is not None, a row containing a
              missing value in any of the columns (before concatenation) will
              have a missing value in the result.
        join : {'left', 'right', 'outer', 'inner'}, default 'left'
            Determines the join-style between the calling Series/Index and any
            Series/Index/DataFrame in `others` (objects without an index need
            to match the length of the calling Series/Index). To disable
            alignment, use `.values` on any Series/Index/DataFrame in `others`.

        Returns
        -------
        str, Series or Index
            If `others` is None, `str` is returned, otherwise a `Series/Index`
            (same type as caller) of objects is returned.

        See Also
        --------
        split : Split each string in the Series/Index.
        join : Join lists contained as elements in the Series/Index.

        Examples
        --------
        When not passing `others`, all values are concatenated into a single
        string:

        >>> s = pd.Series(["a", "b", np.nan, "d"])
        >>> s.str.cat(sep=" ")
        'a b d'

        By default, NA values in the Series are ignored. Using `na_rep`, they
        can be given a representation:

        >>> s.str.cat(sep=" ", na_rep="?")
        'a b ? d'

        If `others` is specified, corresponding values are concatenated with
        the separator. Result will be a Series of strings.

        >>> s.str.cat(["A", "B", "C", "D"], sep=",")
        0    a,A
        1    b,B
        2    NaN
        3    d,D
        dtype: object

        Missing values will remain missing in the result, but can again be
        represented using `na_rep`

        >>> s.str.cat(["A", "B", "C", "D"], sep=",", na_rep="-")
        0    a,A
        1    b,B
        2    -,C
        3    d,D
        dtype: object

        If `sep` is not specified, the values are concatenated without
        separation.

        >>> s.str.cat(["A", "B", "C", "D"], na_rep="-")
        0    aA
        1    bB
        2    -C
        3    dD
        dtype: object

        Series with different indexes can be aligned before concatenation. The
        `join`-keyword works as in other methods.

        >>> t = pd.Series(["d", "a", "e", "c"], index=[3, 0, 4, 2])
        >>> s.str.cat(t, join="left", na_rep="-")
        0    aa
        1    b-
        2    -c
        3    dd
        dtype: object
        >>>
        >>> s.str.cat(t, join="outer", na_rep="-")
        0    aa
        1    b-
        2    -c
        3    dd
        4    -e
        dtype: object
        >>>
        >>> s.str.cat(t, join="inner", na_rep="-")
        0    aa
        2    -c
        3    dd
        dtype: object
        >>>
        >>> s.str.cat(t, join="right", na_rep="-")
        3    dd
        0    aa
        4    -e
        2    -c
        dtype: object

        For more examples, see :ref:`here <text.concatenate>`.
        """
        from pandas import Index, Series, concat

        if isinstance(others, str):
            raise ValueError("Did you mean to supply a `sep` keyword?")
        sep = "" if sep is None else sep
        if isinstance(self._orig, ABCIndex):
            data = Series(self._orig, index=self._orig, dtype=self._orig.dtype)
        else:
            data = self._orig
        if others is None:
            data_object = ensure_object(data)
            na_mask = isna(data_object)
            if na_rep is None and na_mask.any():
                return sep.join(data_object[~na_mask].astype(str))
            elif na_rep is not None and na_mask.any():
                joined = np.where(na_mask, na_rep, data_object)
                return sep.join(joined.astype(str))
            else:
                return sep.join(data_object.astype(str))
        try:
            others_series_list = self._get_series_list(others)
        except ValueError as err:
            raise ValueError(
                "If `others` contains arrays or lists (or other list-likes without an index), "
                "these must all be of the same length as the calling Series/Index."
            ) from err
        if any(not data.index.equals(x.index) for x in others_series_list):
            others_concat = concat(
                others_series_list,
                axis=1,
                join=join if join == "inner" else "outer",
                keys=range(len(others_series_list)),
                sort=False,
            )
            data, others_aligned = data.align(others_concat, join=join)
            others_series_list = [others_aligned[x] for x in others_aligned]
        all_cols: List[Series] = [ensure_object(x) for x in [data] + others_series_list]
        na_masks: np.ndarray = np.array([isna(x) for x in all_cols])
        union_mask: np.ndarray = np.logical_or.reduce(na_masks, axis=0)
        if na_rep is None and union_mask.any():
            result: Any = np.empty(len(data), dtype=object)
            np.putmask(result, union_mask, np.nan)
            not_masked = ~union_mask
            result[not_masked] = cat_safe(
                [x[not_masked].astype(str) for x in all_cols], sep
            )
        elif na_rep is not None and union_mask.any():
            all_cols = [
                np.where(nm, na_rep, col.astype(str)) for nm, col in zip(na_masks, all_cols)
            ]
            result = cat_safe(all_cols, sep)
        else:
            result = cat_safe([x.astype(str) for x in all_cols], sep)
        if isinstance(self._orig.dtype, CategoricalDtype):
            dtype = self._orig.dtype.categories.dtype
        else:
            dtype = self._orig.dtype
        if isinstance(self._orig, ABCIndex):
            if isna(result).all():
                dtype = object
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
            _dtype_inner: Optional[DtypeObj] = dtype
            vdtype = getattr(result, "dtype", None)
            if _dtype_inner is None:
                if self._is_string:
                    if is_bool_dtype(vdtype):
                        _dtype_inner = result.dtype
                    elif returns_string:
                        _dtype_inner = self._orig.dtype
                    else:
                        _dtype_inner = vdtype
                elif vdtype is not None:
                    _dtype_inner = vdtype
            if expand:
                cons = self._orig._constructor_expanddim
                result = cons(result, columns=name, index=index, dtype=_dtype_inner)
            else:
                cons = self._orig._constructor
                result = cons(result, name=name, index=index, dtype=_dtype_inner)
            result = result.__finalize__(self._orig, method="str_cat")
            if name is not None and result.ndim == 1:
                result.name = name
            return result

    @forbid_nonstring_types(["bytes"])
    def split(
        self,
        pat: Optional[str] = None,
        *,
        n: int = -1,
        expand: bool = False,
        regex: Optional[bool] = None,
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        if regex is False and is_re(pat):
            raise ValueError("Cannot use a compiled regex as replacement pattern with regex=False")
        if is_re(pat):
            regex = True
        result = self._data.array._str_split(pat, n, expand, regex)
        dtype: Optional[DtypeObj] = (
            self._data.dtype.categories.dtype
            if self._data.dtype == "category"
            else object
            if self._data.dtype == object
            else None
        )
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def rsplit(
        self,
        pat: Optional[str] = None,
        *,
        n: int = -1,
        expand: bool = False,
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        result = self._data.array._str_rsplit(pat, n=n)
        dtype: Optional[DtypeObj] = (
            self._data.dtype.categories.dtype
            if self._data.dtype == "category"
            else object
            if self._data.dtype == object
            else None
        )
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    def find(self, sub: str, start: int = 0, end: Optional[int] = None) -> Series:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_find(sub, start, end)
        return self._wrap_result(result, returns_string=False)

    def rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> Series:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_rfind(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def index(self, sub: str, start: int = 0, end: Optional[int] = None) -> Series:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> Series:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def count(self, pat: str, flags: int = 0) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def contains(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Optional[Any] = lib.no_default,
        regex: bool = True,
    ) -> Series:
        """
        Docstring...
        """
        if regex and re.compile(pat).groups:
            warnings.warn(
                "This pattern is interpreted as a regular expression, and has match groups. "
                "To actually get the groups, use str.extract.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        result = self._data.array._str_contains(pat, case, flags, na, regex)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Optional[Any] = lib.no_default,
    ) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def fullmatch(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Optional[Any] = lib.no_default,
    ) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat: Union[str, re.Pattern, dict],
        repl: Optional[Union[str, Callable[[re.Match], str]]] = None,
        n: int = -1,
        case: Optional[bool] = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Union[Series, Index]:
        """
        Docstring...
        """
        if isinstance(pat, dict) and repl is not None:
            raise ValueError("repl cannot be used when pat is a dictionary")
        if not isinstance(pat, dict) and not (
            isinstance(repl, str) or callable(repl)
        ):
            raise TypeError("repl must be a string or callable")
        is_compiled_re = is_re(pat)
        if regex or regex is None:
            if is_compiled_re and (case is not None or flags != 0):
                raise ValueError(
                    "case and flags cannot be set when pat is a compiled regex"
                )
        elif is_compiled_re:
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )
        elif callable(repl):
            raise ValueError("Cannot use a callable replacement when regex=False")
        if case is None:
            case = True
        res_output = self._data
        if not isinstance(pat, dict):
            pat = {pat: repl}
        for key, value in pat.items():
            result = res_output.array._str_replace(
                key, value, n=n, case=case, flags=flags, regex=regex
            )
            res_output = self._wrap_result(result)
        return res_output

    @forbid_nonstring_types(["bytes"])
    def repeat(self, repeats: Union[int, List[int], np.ndarray]) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Series:
        """
        Docstring...
        """
        if not isinstance(fillchar, str):
            msg = f"fillchar must be a character, not {type(fillchar).__name__}"
            raise TypeError(msg)
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def center(self, width: int, fillchar: str = " ") -> Series:
        """
        Docstring...
        """
        return self.pad(width, side="both", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def ljust(self, width: int, fillchar: str = " ") -> Series:
        """
        Docstring...
        """
        return self.pad(width, side="right", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def rjust(self, width: int, fillchar: str = " ") -> Series:
        """
        Docstring...
        """
        return self.pad(width, side="left", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def zfill(self, width: int) -> Series:
        """
        Docstring...
        """
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        f: Callable[[str], str] = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        dtype: Optional[str] = "str" if get_option("future.infer_string") else None
        return self._wrap_result(result, dtype=dtype)

    def slice(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    def slice_replace(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        repl: Optional[str] = None,
    ) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def decode(
        self, encoding: str, errors: str = "strict"
    ) -> Union[Series, Index]:
        """
        Docstring...
        """
        if encoding in _cpython_optimized_decoders:
            f: Callable[[bytes], str] = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        arr = self._data.array
        result = arr._str_map(f)
        dtype: Optional[str] = "str" if get_option("future.infer_string") else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def encode(
        self, encoding: str, errors: str = "strict"
    ) -> Union[Series, Index]:
        """
        Docstring...
        """
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    def findall(self, pat: str, flags: int = 0) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_findall(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def partition(
        self, sep: str = " ", expand: bool = True
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        """
        Docstring...
        """
        result = self._data.array._str_partition(sep, expand)
        dtype: Optional[DtypeObj] = (
            self._data.dtype.categories.dtype
            if self._data.dtype == "category"
            else object
            if self._data.dtype == object
            else None
        )
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def rpartition(
        self, sep: str = " ", expand: bool = True
    ) -> Union[Series, Index, DataFrame, MultiIndex]:
        """
        Docstring...
        """
        result = self._data.array._str_rpartition(sep, expand)
        dtype: Optional[DtypeObj] = (
            self._data.dtype.categories.dtype
            if self._data.dtype == "category"
            else object
            if self._data.dtype == object
            else None
        )
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    def get(self, i: Union[int, Hashable]) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    def normalize(self, form: Literal["NFC", "NFKC", "NFD", "NFKD"]) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_normalize(form)
        return self._wrap_result(result)

    def translate(self, table: dict) -> Union[Series, Index]:
        """
        Docstring...
        """
        result = self._data.array._str_translate(table)
        dtype: Optional[DtypeObj] = (
            object if self._data.dtype == "object" else None
        )
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def get_dummies(
        self, sep: str = "|", dtype: Optional[Union[type, str, "_dtype"]] = None
    ) -> DataFrame:
        """
        Docstring...
        """
        from pandas.core.frame import DataFrame

        if dtype is not None and not (
            is_numeric_dtype(dtype) or is_bool_dtype(dtype)
        ):
            raise ValueError(
                "Only numeric or boolean dtypes are supported for 'dtype'"
            )
        result, name = self._data.array._str_get_dummies(sep, dtype)
        if is_extension_array_dtype(dtype):
            return self._wrap_result(
                DataFrame(result, columns=name, dtype=dtype),
                name=name,
                returns_string=False,
            )
        return self._wrap_result(result, name=name, expand=True, returns_string=False)

    def extract(
        self,
        pat: str,
        flags: int = 0,
        expand: bool = True,
    ) -> Union[DataFrame, Series, Index]:
        """
        Docstring...
        """
        from pandas import DataFrame

        regex: re.Pattern = re.compile(pat, flags=flags)
        if regex.groups == 0:
            raise ValueError("pattern contains no capture groups")
        if not expand and regex.groups > 1 and isinstance(self._data, ABCIndex):
            raise ValueError("only one regex group is supported with Index")
        obj = self._data
        result_dtype: Union[DtypeObj, str] = _result_dtype(obj)
        returns_df = regex.groups > 1 or expand
        if returns_df:
            name: Optional[Hashable] = None
            columns: List[Union[int, str]] = _get_group_names(regex)
            if obj.array.size == 0:
                result = DataFrame(columns=columns, dtype=result_dtype)
            else:
                result_list: List[Tuple[Any, ...]] = self._data.array._str_extract(
                    pat, flags=flags, expand=returns_df
                )
                if isinstance(obj, ABCSeries):
                    result_index = obj.index
                else:
                    result_index = None
                result = DataFrame(
                    result_list,
                    columns=columns,
                    index=result_index,
                    dtype=result_dtype,
                )
        else:
            name: Optional[str] = _get_single_group_name(regex)
            result = self._data.array._str_extract(pat, flags=flags, expand=returns_df)
        return self._wrap_result(result, name=name, dtype=result_dtype)

    def extractall(self, pat: str, flags: int = 0) -> DataFrame:
        """
        Docstring...
        """
        return str_extractall(self._orig, pat, flags)

    @forbid_nonstring_types(["bytes"])
    def wrap(
        self,
        width: int,
        expand_tabs: bool = True,
        tabsize: int = 8,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        initial_indent: str = "",
        subsequent_indent: str = "",
        fix_sentence_endings: bool = False,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
        max_lines: Optional[int] = None,
        placeholder: str = " [...]",
    ) -> Series:
        """
        Docstring...
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

    @forbid_nonstring_types(["bytes"])
    def remove_prefix(self, prefix: str) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def remove_suffix(self, suffix: str) -> Series:
        """
        Docstring...
        """
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def get_dummies(
        self, sep: str = "|", dtype: Optional[Union[type, str, Any]] = None
    ) -> DataFrame:
        """
        Docstring...
        """
        from pandas.core.frame import DataFrame

        if dtype is not None and not (
            is_numeric_dtype(dtype) or is_bool_dtype(dtype)
        ):
            raise ValueError(
                "Only numeric or boolean dtypes are supported for 'dtype'"
            )
        result, name = self._data.array._str_get_dummies(sep, dtype)
        if is_extension_array_dtype(dtype):
            return self._wrap_result(
                DataFrame(result, columns=name, dtype=dtype),
                name=name,
                returns_string=False,
            )
        return self._wrap_result(result, name=name, expand=True, returns_string=False)

def cat_safe(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
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
            if dtype not in ["string", "empty"]:
                raise TypeError(
                    f"Concatenation requires list-likes containing only strings "
                    f"(or missing values). Offending values found in column {dtype}"
                ) from None
    return result

def cat_core(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
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
    if sep == "":
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)

def _result_dtype(arr: ExtensionArray) -> Union[DtypeObj, str]:
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
        return list(rng)
    result: List[Union[int, str]] = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == "i" and lib.is_range_indexer(arr, len(arr)):
        return list(rng)
    return result

def str_extractall(
    arr: Union[Series, Index],
    pat: str,
    flags: int = 0,
) -> DataFrame:
    regex: re.Pattern = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns: List[Union[int, str]] = _get_group_names(regex)
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
                na_tuple: Tuple[Any, ...] = tuple(
                    np.nan if group == "" else group for group in match_tuple
                )
                match_list.append(na_tuple)
                result_key: Tuple[Any, ...] = tuple(subject_key + (match_i,))
                index_list.append(result_key)
    from pandas import MultiIndex

    index = MultiIndex.from_tuples(
        index_list, names=arr.index.names + ["match"]
    )
    dtype: Union[DtypeObj, str] = _result_dtype(arr)
    result = arr._constructor_expanddim(
        match_list, index=index, columns=columns, dtype=dtype
    )
    return result

# Note: The shared_docs and _doc_args dictionaries, as well as other methods, should also be annotated similarly.
