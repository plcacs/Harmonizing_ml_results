from __future__ import annotations

import codecs
from functools import wraps
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Iterator,
    Literal,
    Optional,
    TypeVar,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import lib
from pandas._typing import (
    AlignJoin,
    DtypeObj,
    F,
    NpDtype,
    Scalar,
    npt,
)
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
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
)
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

F = TypeVar("F", bound=Callable[..., Any])

_shared_docs: dict[str, str] = {}
_cpython_optimized_encoders: tuple[str, ...] = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
_cpython_optimized_decoders: tuple[str, ...] = _cpython_optimized_encoders + ("utf-16", "utf-32")


def forbid_nonstring_types(
    forbidden: Optional[list[str]], name: Optional[str] = None
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
    # deal with None
    forbidden = [] if forbidden is None else forbidden

    allowed_types: set[str] = {"string", "empty", "bytes", "mixed", "mixed-integer"} - set(forbidden)

    def _forbid_nonstring_types(func: F) -> F:
        func_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self: StringMethods, *args: Any, **kwargs: Any) -> Any:
            if self._inferred_dtype not in allowed_types:
                msg = (
                    f"Cannot use .str.{func_name} with values of "
                    f"inferred dtype '{self._inferred_dtype}'."
                )
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        wrapper.__name__ = func_name
        return cast(F, wrapper)

    return _forbid_nonstring_types


def _map_and_wrap(name: Optional[str], docstring: Optional[str]) -> Callable[[StringMethods], Any]:
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

    # Note: see the docstring in pandas.core.strings.__init__
    # for an explanation of the implementation.
    # TODO: Dispatch all the methods
    # Currently the following are not dispatched to the array
    # * cat
    # * extractall

    def __init__(self, data: Series | Index) -> None:
        from pandas.core.arrays.string_ import StringDtype

        self._inferred_dtype: str = self._validate(data)
        self._is_categorical: bool = isinstance(data.dtype, CategoricalDtype)
        self._is_string: bool = isinstance(data.dtype, StringDtype)
        self._data: Series | Index = data

        self._index: Optional[Index] = None
        self._name: Optional[Hashable] = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name

        # ._values.categories works for both Series/Index
        self._parent: Series | Index | ExtensionArray = (
            data._values.categories if self._is_categorical else data
        )
        # save orig to blow up categoricals to the right type
        self._orig: Series | Index = data
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
            raise AttributeError(
                "Can only use .str accessor with Index, not MultiIndex"
            )

        # see _libs/lib.pyx for list of inferred types
        allowed_types: list[str] = ["string", "empty", "bytes", "mixed", "mixed-integer"]

        data = extract_array(data)

        values = getattr(data, "categories", data)  # categorical / normal

        inferred_dtype: str = lib.infer_dtype(values, skipna=True)

        if inferred_dtype not in allowed_types:
            raise AttributeError(
                f"Can only use .str accessor with string values, not {inferred_dtype}"
            )
        return inferred_dtype

    def __getitem__(self, key: int | slice | Any) -> Any:
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(
        self,
        result: Any,
        name: Optional[Hashable] = None,
        expand: Optional[bool] = None,
        fill_value: Any = np.nan,
        returns_string: bool = True,
        dtype: Optional[DtypeObj] = None,
    ) -> Series | Index | DataFrame | Any:
        from pandas import (
            Index,
            MultiIndex,
            Series,
        )

        if not hasattr(result, "ndim") or not hasattr(result, "dtype"):
            if isinstance(result, ABCDataFrame):
                result = result.__finalize__(self._orig, name="str")
            return result
        assert result.ndim < 3

        # We can be wrapping a string / object / categorical result, in which
        # case we'll want to return the same dtype as the input.
        # Or we can be wrapping a numeric output, in which case we don't want
        # to return a StringArray.
        # Ideally the array method returns the right array type.
        if expand is None:
            # infer from ndim if expand is not specified
            expand = result.ndim != 1
        elif expand is True and not isinstance(self._orig, ABCIndex):
            # required when expand=True is explicitly specified
            # not needed when inferred
            if isinstance(result.dtype, ArrowDtype):
                import pyarrow as pa

                from pandas.compat import pa_version_under11p0

                from pandas.core.arrays.arrow.array import ArrowExtensionArray

                value_lengths = pa.compute.list_value_length(result._pa_array)
                max_len = pa.compute.max(value_lengths).as_py()
                min_len = pa.compute.min(value_lengths).as_py()
                if result._hasna:
                    # ArrowExtensionArray.fillna doesn't work for list scalars
                    result = ArrowExtensionArray(
                        result._pa_array.fill_null([None] * max_len)
                    )
                if min_len < max_len:
                    # append nulls to each scalar list element up to max_len
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
                        new_values: list[Any] = []
                        for row in values:
                            if len(row) < max_len:
                                nulls = all_null[: max_len - len(row)]
                                row = np.append(row, nulls)
                            new_values.append(row)
                        pa_type = result._pa_array.type
                        result = ArrowExtensionArray(pa.array(new_values, type=pa_type))
                if name is None:
                    name = range(max_len)
                result = (
                    pa.compute.list_flatten(result._pa_array)
                    .to_numpy()
                    .reshape(len(result), max_len)
                )
                result = {
                    label: ArrowExtensionArray(pa.array(res))
                    for label, res in zip(name, result.T)
                }
            elif is_object_dtype(result):
                def cons_row(x: Any) -> list[Any]:
                    if is_list_like(x):
                        return x
                    else:
                        return [x]

                result = [cons_row(x) for x in result]
                if result and not self._is_string:
                    # propagate nan values to match longest sequence (GH 18450)
                    max_len = max(len(x) for x in result)
                    result = [
                        x * max_len if len(x) == 0 or x[0] is np.nan else x
                        for x in result
                    ]

        if not isinstance(expand, bool):
            raise ValueError("expand must be True or False")

        if expand is False:
            # if expand is False, result should have the same name
            # as the original otherwise specified
            if name is None:
                name = getattr(result, "name", None)
            if name is None:
                # do not use logical or, _orig may be a DataFrame
                # which has "name" column
                name = self._orig.name

        # Wait until we are sure result is a Series or Index before
        # checking attributes (GH 12180)
        if isinstance(self._orig, ABCIndex):
            # if result is a boolean np.array, return the np.array
            # instead of wrapping it into a boolean Index (GH 8875)
            if is_bool_dtype(result):
                return result

            if expand:
                result = list(result)
                out: Index = MultiIndex.from_tuples(result, names=name)
                if out.nlevels == 1:
                    # We had all tuples of length-one, which are
                    # better represented as a regular Index.
                    out = out.get_level_values(0)
                return out
            else:
                return Index(result, name=name, dtype=dtype)
        else:
            index = self._orig.index
            # This is a mess.
            _dtype: DtypeObj | str | None = dtype
            vdtype = getattr(result, "dtype", None)
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
                # Must be a Series
                cons = self._orig._constructor
                result = cons(result, name=name, index=index, dtype=_dtype)
            result = result.__finalize__(self._orig, method="str")
            if name is not None and result.ndim == 1:
                # __finalize__ might copy over the original name, but we may
                # want the new name (e.g. str.extract).
                result.name = name
            return result

    def _get_series_list(self, others: Any) -> list[Series]:
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
        from pandas import (
            DataFrame,
            Series,
        )

        # self._orig is either Series or Index
        idx: Series | Index = self._orig if isinstance(self._orig, ABCIndex) else self._orig.index

        # Generally speaking, all objects without an index inherit the index
        # `idx` of the calling Series/Index - i.e. must have matching length.
        # Objects with an index (i.e. Series/Index/DataFrame) keep their own.
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
                others = list(others)  # ensure iterators do not get read twice etc
            except TypeError:
                # e.g. ser.str, raise below
                pass
            else:
                # in case of list-like `others`, all elements must be
                # either Series/Index/np.ndarray (1-dim)...
                if all(
                    isinstance(x, (ABCSeries, ABCIndex, ExtensionArray))
                    or (isinstance(x, np.ndarray) and x.ndim == 1)
                    for x in others
                ):
                    los: list[Series] = []
                    while others:  # iterate through list and append each element
                        los = los + self._get_series_list(others.pop(0))
                    return los
                # ... or just strings
                elif all(not is_list_like(x) for x in others):
                    return [Series(others, index=idx)]
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarray "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )

    @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
    def cat(
        self,
        others: Any = None,
        sep: Optional[str] = None,
        na_rep: Optional[str] = None,
        join: AlignJoin = "left",
    ) -> str | Series | Index:
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
        from pandas import (
            Index,
            Series,
            concat,
            DataFrame,
        )

        if isinstance(others, str):
            raise ValueError("Did you mean to supply a `sep` keyword?")
        if sep is None:
            sep = ""

        if isinstance(self._orig, ABCIndex):
            data = Series(self._orig, index=self._orig, dtype=self._orig.dtype)
        else:  # Series
            data = self._orig

        # concatenate Series/Index with itself if no "others"
        if others is None:
            # error: Incompatible types in assignment (expression has type
            # "ndarray", variable has type "Series")
            data = ensure_object(data)  # type: ignore[assignment]
            na_mask = isna(data)
            if na_rep is None and na_mask.any():
                return sep.join(data[~na_mask])
            elif na_rep is not None and na_mask.any():
                return sep.join(np.where(na_mask, na_rep, data))
            else:
                return sep.join(data)

        try:
            # turn anything in "others" into lists of Series
            others = self._get_series_list(others)
        except ValueError as err:  # do not catch TypeError raised by _get_series_list
            raise ValueError(
                "If `others` contains arrays or lists (or other "
                "list-likes without an index), these must all be "
                "of the same length as the calling Series/Index."
            ) from err

        # align if required
        if any(not data.index.equals(x.index) for x in others):
            # Need to add keys for uniqueness in case of duplicate columns
            others_df = concat(
                others,
                axis=1,
                join=(join if join == "inner" else "outer"),
                keys=range(len(others)),
                sort=False,
            )
            data, others_aligned = data.align(others_df, join=join)
            others = [others_aligned[x] for x in others_aligned]  # again list of Series

        all_cols: list[Series] = [ensure_object(x) for x in [data] + others]
        na_masks = np.array([isna(x) for x in all_cols])
        union_mask = np.logical_or.reduce(na_masks, axis=0)

        if na_rep is None and union_mask.any():
            # no na_rep means NaNs for all rows where any column has a NaN
            # only necessary if there are actually any NaNs
            result: npt.NDArray[Any] = np.empty(len(data), dtype=object)
            np.putmask(result, union_mask, np.nan)

            not_masked = ~union_mask
            result[not_masked] = cat_safe([x[not_masked] for x in all_cols], sep)
        elif na_rep is not None and union_mask.any():
            # fill NaNs with na_rep in case there are actually any NaNs
            all_cols = [
                np.where(nm, na_rep, col) for nm, col in zip(na_masks, all_cols)
            ]
            result = cat_safe(all_cols, sep)
        else:
            # no NaNs - can just concatenate
            result = cat_safe(all_cols, sep)

        out: Index | Series
        if isinstance(self._orig.dtype, CategoricalDtype):
            # We need to infer the new categories.
            dtype = self._orig.dtype.categories.dtype
        else:
            dtype = self._orig.dtype
        if isinstance(self._orig, ABCIndex):
            # add dtype for case that result is all-NA
            if isna(result).all():
                dtype = object  # type: ignore[assignment]

            out = Index(result, dtype=dtype, name=self._orig.name)
        else:
            res_ser = Series(
                result, dtype=dtype, index=data.index, name=self._orig.name, copy=False
            )
            out = res_ser.__finalize__(self._orig, method="str_cat")
        return out

    _shared_docs["str_split"] = r"""
    Split strings around given separator/delimiter.

    Splits the string in the Series/Index from the %(side)s,
    at the specified delimiter string.

    Parameters
    ----------
    pat : str%(pat_regex)s, optional
        %(pat_description)s.
        If not specified, split on whitespace.
    n : int, default -1 (all)
        Limit number of splits in output.
        ``None``, 0 and -1 will be interpreted as return all splits.
    expand : bool, default False
        Expand the split strings into separate columns.

        - If ``True``, return DataFrame/MultiIndex expanding dimensionality.
        - If ``False``, return Series/Index, containing lists of strings.
    %(regex_argument)s
    Returns
    -------
    Series, Index, DataFrame or MultiIndex
        Type matches caller unless ``expand=True`` (see Notes).
    %(raises_split)s
    See Also
    --------
    Series.str.split : Split strings around given separator/delimiter.
    Series.str.rsplit : Splits string around given separator/delimiter,
        starting from the right.
    Series.str.join : Join lists contained as elements in the Series/Index
        with passed delimiter.
    str.split : Standard library version for split.
    str.rsplit : Standard library version for rsplit.

    Notes
    -----
    The handling of the `n` keyword depends on the number of found splits:

    - If found splits > `n`,  make first `n` splits only
    - If found splits <= `n`, make all splits
    - If for a certain row the number of found splits < `n`,
      append `None` for padding up to `n` if ``expand=True``

    If using ``expand=True``, Series and Index callers return DataFrame and
    MultiIndex objects, respectively.
    %(regex_pat_note)s
    Examples
    --------
    >>> s = pd.Series(
    ...     [
    ...         "this is a regular sentence",
    ...         "https://docs.python.org/3/tutorial/index.html",
    ...         np.nan
    ...     ]
    ... )
    >>> s
    0                       this is a regular sentence
    1    https://docs.python.org/3/tutorial/index.html
    2                                              NaN
    dtype: object

    In the default setting, the string is split by whitespace.

    >>> s.str.split()
    0                   [this, is, a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    Without the `n` parameter, the outputs of `rsplit` and `split`
    are identical.

    >>> s.str.rsplit()
    0                   [this, is, a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    The `n` parameter can be used to limit the number of splits on the
    delimiter. The outputs of `split` and `rsplit` are different.

    >>> s.str.split(n=2)
    0                     [this, is, a regular sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    >>> s.str.rsplit(n=2)
    0                     [this is a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    The `pat` parameter can be used to split by other characters.

    >>> s.str.split(pat="/")
    0                         [this is a regular sentence]
    1    [https:, , docs.python.org, 3, tutorial, index...
    2                                                  NaN
    dtype: object

    When using ``expand=True``, the split elements will expand out into
    separate columns. If NaN is present, it is propagated throughout
    the columns during the split.

    >>> s.str.split(expand=True)
                                                   0     1     2        3         4
    0                                       this    is     a  regular  sentence
    1  https://docs.python.org/3/tutorial/index.html  None  None     None      None
    2                                        NaN   NaN   NaN      NaN       NaN

    For slightly more complex use cases like splitting the html document name
    from a url, a combination of parameter settings can be used.

    >>> s.str.rsplit("/", n=1, expand=True)
                                        0           1
    0          this is a regular sentence        None
    1  https://docs.python.org/3/tutorial  index.html
    2                                 NaN         NaN
    %(regex_examples)s"""

    @Appender(
        _shared_docs["str_split"]
        % {
            "side": "beginning",
            "pat_regex": " or compiled regex",
            "pat_description": "String or regular expression to split on",
            "regex_argument": """
    regex : bool, default None
        Determines if the passed-in pattern is a regular expression:

        - If ``True``, assumes the passed-in pattern is a regular expression
        - If ``False``, treats the pattern as a literal string.
        - If ``None`` and `pat` length is 1, treats `pat` as a literal string.
        - If ``None`` and `pat` length is not 1, treats `pat` as a regular expression.
        - Cannot be set to False if `pat` is a compiled regex

        .. versionadded:: 1.4.0
         """,
            "raises_split": """
                      Raises
                      ------
                      ValueError
                          * if `regex` is False and `pat` is a compiled regex
                      """,
            "regex_pat_note": """
    Use of `regex =False` with a `pat` as a compiled regex will raise an error.
            """,
            "method": "split",
            "regex_examples": r"""
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

    >>> s.str.split(r"\.jpg", expand=True)
               0 1
    0  foojpgbar

    When ``regex=True``, `pat` is interpreted as a regex

    >>> s.str.split(r"\.jpg", regex=True, expand=True)
               0 1
    0  foojpgbar

    A compiled regex can be passed as `pat`

    >>> import re
    >>> s.str.split(re.compile(r"\.jpg"), expand=True)
               0 1
    0  foojpgbar

    When ``regex=False``, `pat` is interpreted as the string itself

    >>> s.str.split(r"\.jpg", regex=False, expand=True)
                   0
    0  foojpgbar.jpg
    """,
        }
    )
    @forbid_nonstring_types(["bytes"])
    def split(
        self,
        pat: str | re.Pattern | None = None,
        *,
        n: int = -1,
        expand: bool = False,
        regex: bool | None = None,
    ) -> DataFrame | Index | Series | list[str] | Any:
        if regex is False and is_re(pat):
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )
        if is_re(pat):
            regex = True
        result = self._data.array._str_split(pat, n, expand, regex)
        if self._data.dtype == "category":
            dtype: Optional[DtypeObj] = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(
        _shared_docs["str_split"]
        % {
            "side": "end",
            "pat_regex": "",
            "pat_description": "String to split on",
            "regex_argument": "",
            "raises_split": "",
            "regex_pat_note": "",
            "method": "rsplit",
            "regex_examples": "",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rsplit(self, pat: str | re.Pattern | None = None, *, n: int = -1, expand: bool = False) -> DataFrame | Index | Series | list[str] | Any:
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    _shared_docs["str_partition"] = """
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
    1     George Pitt  -  Rivers

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

    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "first",
            "return": "3 elements containing the string itself, followed by two "
            "empty strings",
            "also": "rpartition : Split the string at the last occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def partition(self, sep: str = " ", expand: bool = True) -> DataFrame | Index | Series:
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "last",
            "return": "3 elements containing two empty strings, followed by the "
            "string itself",
            "also": "partition : Split the string at the first occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rpartition(self, sep: str = " ", expand: bool = True) -> DataFrame | Index | Series:
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    def get(self, i: int | Hashable) -> Series | Index:
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

    @forbid_nonstring_types(["bytes"])
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

    @forbid_nonstring_types(["bytes"])
    def contains(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | None = None,
        regex: bool = True,
    ) -> Series | Index:
        r"""
        Test if pattern or regex is contained within a string of a Series or Index.

        Return boolean Series or Index based on whether a given pattern or regex is
        contained within a string of a Series or Index.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Flags to pass through to the re module, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.
        regex : bool, default True
            If True, assumes the pat is a regular expression.

            If False, treats the pat as a literal string.

        Returns
        -------
        Series or Index of boolean values
            A Series or Index of boolean values indicating whether the
            given pattern is contained within the string of each element
            of the Series or Index.

        See Also
        --------
        match : Analogous, but stricter, relying on re.match instead of re.search.
        Series.str.startswith : Test if the start of each string element matches a
            pattern.
        Series.str.endswith : Same as startswith, but tests the end of string.

        Examples
        --------
        Returning a Series of booleans using only a literal pattern.

        >>> s1 = pd.Series(["Mouse", "dog", "house and parrot", "23", np.nan])
        >>> s1.str.contains("og", regex=False)
        0    False
        1     True
        2    False
        3    False
        4      NaN
        dtype: object

        Returning an Index of booleans using only a literal pattern.

        >>> ind = pd.Index(["Mouse", "dog", "house and parrot", "23.0", np.nan])
        >>> ind.str.contains("23", regex=False)
        Index([False, False, False, True, nan], dtype='object')

        Specifying case sensitivity using `case`.

        >>> s1.str.contains("oG", case=True, regex=True)
        0    False
        1    False
        2    False
        3    False
        4      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN` replaces NaN values
        with `False`. If Series or Index does not contain NaN values
        the resultant dtype will be `bool`, otherwise, an `object` dtype.

        >>> s1.str.contains("og", na=False, regex=True)
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        Returning 'house' or 'dog' when either expression occurs in a string.

        >>> s1.str.contains("house|dog", regex=True)
        0    False
        1     True
        2     True
        3    False
        4      NaN
        dtype: object

        Ignoring case sensitivity using `flags` with regex.

        >>> import re
        >>> s1.str.contains("PARROT", flags=re.IGNORECASE, regex=True)
        0    False
        1    False
        2     True
        3    False
        4      NaN
        dtype: object

        Returning any digit using regular expression.

        >>> s1.str.contains("\\d", regex=True)
        0    False
        1    False
        2    False
        3     True
        4      NaN
        dtype: object

        Ensure `pat` is a not a literal pattern when `regex` is set to True.
        Note in the following example one might expect only `s2[1]` and `s2[3]` to
        return `True`. However, '.0' as a regex matches any character
        followed by a 0.

        >>> s2 = pd.Series(["40", "40.0", "41", "41.0", "35"])
        >>> s2.str.contains(".0", regex=True)
        0     True
        1     True
        2    False
        3     True
        4    False
        dtype: bool
        """
        if regex and pat and re.compile(pat).groups:
            warnings.warn(
                "This pattern is interpreted as a regular expression, and has "
                "match groups. To actually get the groups, use str.extract.",
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
        na: Scalar | None = None,
    ) -> Series | Index:
        """
        Determine if each string starts with a match of a regular expression.

        Determines whether each string in the Series or Index starts with a
        match to a specified regular expression. This function is especially
        useful for validating prefixes, such as ensuring that codes, tags, or
        identifiers begin with a specific pattern.

        Parameters
        ----------
        pat : str
            Character sequence.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.

        Returns
        -------
        Series/Index/array of boolean values
            A Series, Index, or array of boolean values indicating whether the start
            of each string matches the pattern. The result will be of the same type
            as the input.

        See Also
        --------
        fullmatch : Stricter matching that requires the entire string to match.
        contains : Analogous, but less strict, relying on re.search instead of re.match.
        extract : Extract matched groups.

        Examples
        --------
        >>> ser = pd.Series(["horse", "eagle", "donkey"])
        >>> ser.str.match("e")
        0   False
        1   True
        2   False
        dtype: bool
        """
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def fullmatch(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | None = None,
    ) -> Series | Index:
        """
        Determine if each string entirely matches a regular expression.

        Checks if each string in the Series or Index fully matches the
        specified regular expression pattern. This function is useful when the
        requirement is for an entire string to conform to a pattern, such as
        validating formats like phone numbers or email addresses.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.

        Returns
        -------
        Series/Index/array of boolean values
            The function returns a Series, Index, or array of boolean values,
            where True indicates that the entire string matches the regular
            expression pattern and False indicates that it does not.

        See Also
        --------
        match : Similar, but also returns `True` when only a *prefix* of the string
            matches the regular expression.
        extract : Extract matched groups.

        Examples
        --------
        >>> ser = pd.Series(["cat", "duck", "dove"])
        >>> ser.str.fullmatch(r"d.+")
        0   False
        1    True
        2    True
        dtype: bool
        """
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat: str | re.Pattern | dict[str, str],
        repl: str | Callable[[re.Match], str] | None = None,
        n: int = -1,
        case: Optional[bool] = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Series | Index:
        r"""
        Replace each occurrence of pattern/regex in the Series/Index.

        Equivalent to :meth:`str.replace` or :func:`re.sub`, depending on
        the regex value.

        Parameters
        ----------
        pat : str, compiled regex, or a dict
            String can be a character sequence or regular expression.
            Dictionary contains <key : value> pairs of strings to be replaced
            along with the updated value.
        repl : str or callable
            Replacement string or a callable. The callable is passed the regex
            match object and must return a replacement string to be used.
            Must have a value of None if `pat` is a dict
            See :func:`re.sub`.
        n : int, default -1 (all)
            Number of replacements to make from start.
        case : bool, default None
            Determines if replace is case sensitive:

            - If True, case sensitive (the default if `pat` is a string)
            - Set to False for case insensitive
            - Cannot be set if `pat` is a compiled regex.

        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE. Cannot be set if `pat` is a compiled
            regex.
        regex : bool, default False
            Determines if the passed-in pattern is a regular expression:

            - If True, assumes the passed-in pattern is a regular expression.
            - If False, treats the pattern as a literal string
            - Cannot be set to False if `pat` is a compiled regex or `repl` is
              a callable.

        Returns
        -------
        Series or Index of object
            A copy of the object with all matching occurrences of `pat` replaced by
            `repl`.

        Raises
        ------
        ValueError
            * if `regex` is False and `repl` is a callable or `pat` is a compiled regex
            * if `pat` is a compiled regex and `case` or `flags` is set
            * if `pat` is a dictionary and `repl` is not None.

        See Also
        --------
        Series.str.replace : Method to replace occurrences of a substring with another
            substring.
        Series.str.extract : Extract substrings using a regular expression.
        Series.str.findall : Find all occurrences of a pattern or regex in each string.
        Series.str.split : Split each string by a specified delimiter or pattern.

        Notes
        -----
        When `pat` is a compiled regex, all flags should be included in the
        compiled regex. Use of `case`, `flags`, or `regex=False` with a compiled
        regex will raise an error.

        Examples
        --------
        When `pat` is a dictionary, every key in `pat` is replaced
        with its corresponding value:

        >>> pd.Series(["A", "B", np.nan]).str.replace(pat={"A": "a", "B": "b"})
        0    a
        1    b
        2    NaN
        dtype: object

        When `pat` is a string and `regex` is True, the given `pat`
        is compiled as a regex. When `repl` is a string, it replaces matching
        regex patterns as with :meth:`re.sub`. NaN value(s) in the Series are
        left as is:

        >>> pd.Series(["foo", "fuz", np.nan]).str.replace("f.", "ba", regex=True)
        0    bao
        1    baz
        2    NaN
        dtype: object

        When `pat` is a string and `regex` is False, every `pat` is replaced with
        `repl` as with :meth:`str.replace`:

        >>> pd.Series(["f.o", "fuz", np.nan]).str.replace("f.", "ba", regex=False)
        0    bao
        1    fuz
        2    NaN
        dtype: object

        When `repl` is a callable, it is called on every `pat` using
        :func:`re.sub`. The callable should expect one positional argument
        (a regex object) and return a string.

        To get the idea:

        >>> pd.Series(["foo", "fuz", np.nan]).str.replace("f", repr, regex=True)
        0    <re.Match object; span=(0, 1), match='f'>oo
        1    <re.Match object; span=(0, 1), match='f'>uz
        2                                            NaN
        dtype: object

        Reverse every lowercase alphabetic word:

        >>> repl = lambda m: m.group(0)[::-1]
        >>> ser = pd.Series(["foo 123", "bar baz", np.nan])
        >>> ser.str.replace(r"[a-z]+", repl, regex=True)
        0    oof 123
        1    rab zab
        2        NaN
        dtype: object

        Using regex groups (extract second group and swap case):

        >>> pat = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
        >>> repl = lambda m: m.group("two").swapcase()
        >>> ser = pd.Series(["One Two Three", "Foo Bar Baz"])
        >>> ser.str.replace(pat, repl, regex=True)
        0    tWO
        1    bAR
        dtype: object

        Using a compiled regex with flags

        >>> import re
        >>> regex_pat = re.compile(r"FUZ", flags=re.IGNORECASE)
        >>> pd.Series(["foo", "fuz", np.nan]).str.replace(regex_pat, "bar", regex=True)
        0    foo
        1    bar
        2    NaN
        dtype: object
        """
        if isinstance(pat, dict) and repl is not None:
            raise ValueError("repl cannot be used when pat is a dictionary")

        # Check whether repl is valid (GH 13438, GH 15055)
        if not isinstance(pat, dict) and not (isinstance(repl, str) or callable(repl)):
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

        res_output: Series | Index = self._data
        if not isinstance(pat, dict):
            pat = {pat: repl}

        for key, value in pat.items():
            result = res_output.array._str_replace(
                key, value, n=n, case=case, flags=flags, regex=regex
            )
            res_output = self._wrap_result(result)

        return res_output

    @forbid_nonstring_types(["bytes"])
    def repeat(self, repeats: int | list[int] | np.ndarray) -> Series | Index:
        """
        Duplicate each string in the Series or Index.

        Duplicates each string in the Series or Index, either by applying the
        same repeat count to all elements or by using different repeat values
        for each element.

        Parameters
        ----------
        repeats : int or sequence of int
            Same value for all (int) or different value per (sequence).

        Returns
        -------
        Series or pandas.Index
            Series or Index of repeated string objects specified by
            input parameter repeats.

        See Also
        --------
        Series.str.lower : Convert all characters in each string to lowercase.
        Series.str.upper : Convert all characters in each string to uppercase.
        Series.str.title : Convert each string to title case (capitalizing the first
            letter of each word).
        Series.str.strip : Remove leading and trailing whitespace from each string.
        Series.str.replace : Replace occurrences of a substring with another substring
            in each string.
        Series.str.ljust : Left-justify each string in the Series/Index by padding with
            a specified character.
        Series.str.rjust : Right-justify each string in the Series/Index by padding with
            a specified character.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"])
        >>> s
        0    a
        1    b
        2    c
        dtype: object

        Single int repeats string in Series

        >>> s.str.repeat(repeats=2)
        0    aa
        1    bb
        2    cc
        dtype: object

        Sequence of int repeats corresponding string in Series

        >>> s.str.repeat(repeats=[1, 2, 3])
        0      a
        1     bb
        2    ccc
        dtype: object
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Series | Index:
        """
        Pad strings in the Series/Index up to width.

        This function pads strings in a Series or Index to a specified width,
        filling the extra space with a character of your choice. It provides
        flexibility in positioning the padding, allowing it to be added to the
        left, right, or both sides. This is useful for formatting strings to
        align text or ensure consistent string lengths in data processing.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be filled
            with character defined in `fillchar`.
        side : {'left', 'right', 'both'}, default 'left'
            Side from which to fill resulting string.
        fillchar : str, default ' '
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series or Index of object
            Returns Series or Index with minimum number of char in object.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='left')``.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='right')``.
        Series.str.center : Fills both sides of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='both')``.
        Series.str.zfill : Pad strings in the Series/Index by prepending '0'
            character. Equivalent to ``Series.str.pad(side='left', fillchar='0')``.

        Examples
        --------
        >>> s = pd.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        dtype: object

        >>> s.str.pad(width=10)
        0       caribou
        1         tiger
        dtype: object

        >>> s.str.pad(width=10, side="right", fillchar="-")
        0    caribou---
        1    tiger-----
        dtype: object

        >>> s.str.pad(width=10, side="both", fillchar="-")
        0    -caribou--
        1    --tiger---
        dtype: object
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

    _shared_docs["str_pad"] = """
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

    @Appender(_shared_docs["str_pad"] % {"side": "left and right", "method": "center"})
    @forbid_nonstring_types(["bytes"])
    def center(self, width: int, fillchar: str = " ") -> Series | Index:
        return self.pad(width, side="both", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "right", "method": "ljust"})
    @forbid_nonstring_types(["bytes"])
    def ljust(self, width: int, fillchar: str = " ") -> Series | Index:
        return self.pad(width, side="right", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "left", "method": "rjust"})
    @forbid_nonstring_types(["bytes"])
    def rjust(self, width: int, fillchar: str = " ") -> Series | Index:
        return self.pad(width, side="left", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def zfill(self, width: int) -> Series | Index:
        """
        Pad strings in the Series/Index by prepending '0' characters.

        Strings in the Series/Index are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the Series/Index with length greater or equal to `width` are
        unchanged.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less
            than `width` be prepended with '0' characters.

        Returns
        -------
        Series/Index of objects.
            A Series or Index where the strings are prepended with '0' characters.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character.
        Series.str.pad : Fills the specified sides of strings with an arbitrary
            character.
        Series.str.center : Fills both sides of strings with an arbitrary
            character.

        Notes
        -----
        Differs from :meth:`str.zfill` which has special handling
        for '+'/'-' in the string.

        Examples
        --------
        >>> s = pd.Series(["-1", "1", "1000", 10, np.nan])
        >>> s
        0      -1
        1       1
        2    1000
        3      10
        4     NaN
        dtype: object

        Note that ``10`` and ``NaN`` are not strings, therefore they are
        converted to ``NaN``. The minus sign in ``'-1'`` is treated as a
        special character and the zero is added to the right of it
        (:meth:`str.zfill` would have moved it to the left). ``1000``
        remains unchanged as it is longer than `width`.

        >>> s.str.zfill(3)
        0     -01
        1     001
        2    1000
        3     NaN
        4     NaN
        dtype: object
        """
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        dtype = "str" if get_option("future.infer_string") else None
        return self._wrap_result(result, dtype=dtype)

    _shared_docs["str_removefix"] = r"""
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

    @Appender(
        _shared_docs["str_removefix"] % {"side": "prefix", "other_side": "suffix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removeprefix(self, prefix: str) -> Series | Index:
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_removefix"] % {"side": "suffix", "other_side": "prefix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removesuffix(self, suffix: str) -> Series | Index:
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    def normalize(self, form: str) -> Series | Index:
        """
        Return the Unicode normal form for the strings in the Series/Index.

        For more information on the forms, see the
        :func:`unicodedata.normalize`.

        Parameters
        ----------
        form : {'NFC', 'NFKC', 'NFD', 'NFKD'}
            Unicode form.

        Returns
        -------
        Series/Index of objects
            A Series or Index of strings in the same Unicode form specified by `form`.
            The returned object retains the same type as the input (Series or Index),
            and contains the normalized strings.

        See Also
        --------
        Series.str.upper : Convert all characters in each string to uppercase.
        Series.str.lower : Convert all characters in each string to lowercase.
        Series.str.title : Convert each string to title case (capitalizing the first
            letter of each word).
        Series.str.strip : Remove leading and trailing whitespace from each string.
        Series.str.replace : Replace occurrences of a substring with another substring
            in each string.

        Examples
        --------
        >>> ser = pd.Series(["ñ"])
        >>> ser.str.normalize("NFC") == ser.str.normalize("NFD")
        0   False
        dtype: bool
        """
        result = self._data.array._str_normalize(form)
        return self._wrap_result(result)

    _shared_docs["index"] = """
    Return %(side)s indexes in each string in Series/Index.

    Each of returned indexes corresponds to the position where the
    substring is fully contained between [start:end]. This is the same
    as ``str.%(similar)s`` except instead of returning -1, it raises a
    ValueError when the substring is not found. Equivalent to standard
    ``str.%(method)s``.

    Parameters
    ----------
    sub : str
        Substring being searched.
    start : int
        Left edge index.
    end : int
        Right edge index.

    Returns
    -------
    Series or Index of object
        Returns a Series or an Index of the %(side)s indexes
        in each string of the input.

    See Also
    --------
    %(also)s

    Examples
    --------
    For Series.str.index:

    >>> ser = pd.Series(["horse", "eagle", "donkey"])
    >>> ser.str.index("e")
    0   4
    1   0
    2   4
    dtype: int64

    For Series.str.rindex:

    >>> ser = pd.Series(["Deer", "eagle", "Sheep"])
    >>> ser.str.rindex("e")
    0   2
    1   4
    2   3
    dtype: int64
    """

    @Appender(
        _shared_docs["index"]
        % {
            "side": "lowest",
            "similar": "find",
            "method": "index",
            "also": "rindex : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def index(self, sub: str, start: int = 0, end: Optional[int] = None) -> Series | Index:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["index"]
        % {
            "side": "highest",
            "similar": "rfind",
            "method": "rindex",
            "also": "index : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> Series | Index:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def len(self) -> Series | Index:
        """
        Compute the length of each element in the Series/Index.

        The element may be a sequence (such as a string, tuple or list) or a collection
        (such as a dictionary).

        Returns
        -------
        Series or Index of int
            A Series or Index of integer values indicating the length of each
            element in the Series or Index.

        See Also
        --------
        str.len : Python built-in function returning the length of an object.
        Series.size : Returns the length of the Series.

        Examples
        --------
        Returns the length (number of characters) in a string. Returns the
        number of entries for dictionaries, lists or tuples.

        >>> s = pd.Series(
        ...     ["dog", "", 5, {"foo": "bar"}, [2, 3, 5, 7], ("one", "two", "three")]
        ... )
        >>> s
        0                  dog
        1
        2                    5
        3       {'foo': 'bar'}
        4         [2, 3, 5, 7]
        5    (one, two, three)
        dtype: object
        >>> s.str.len()
        0    3.0
        1    0.0
        2    NaN
        3    1.0
        4    4.0
        5    3.0
        dtype: float64
        """
        result = self._data.array._str_len()
        return self._wrap_result(result, returns_string=False)

    _shared_docs["casemethods"] = """
    Convert strings in the Series/Index to %(type)s.
    %(version)s
    Equivalent to :meth:`str.%(method)s`.

    Returns
    -------
    Series or Index of objects
        A Series or Index where the strings are modified by :meth:`str.%(method)s`.

    See Also
    --------
    Series.str.lower : Converts all characters to lowercase.
    Series.str.upper : Converts all characters to uppercase.
    Series.str.title : Converts first character of each word to uppercase and
        remaining to lowercase.
    Series.str.capitalize : Converts first character to uppercase and
        remaining to lowercase.
    Series.str.swapcase : Converts uppercase to lowercase and lowercase to
        uppercase.
    Series.str.casefold : Removes all case distinctions in the string.

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

    # Types:
    #   cases:
    #       upper, lower, title, capitalize, swapcase, casefold
    #   boolean:
    #     isalpha, isnumeric isalnum isdigit isdecimal isspace islower
    #     isupper istitle isascii
    # _doc_args holds dict of strings to use in substituting casemethod docs
    _doc_args_methods: dict[str, dict[str, str]] = {}
    _doc_args_methods["lower"] = {"type": "lowercase", "method": "lower", "version": ""}
    _doc_args_methods["upper"] = {"type": "uppercase", "method": "upper", "version": ""}
    _doc_args_methods["title"] = {"type": "titlecase", "method": "title", "version": ""}
    _doc_args_methods["capitalize"] = {
        "type": "be capitalized",
        "method": "capitalize",
        "version": "",
    }
    _doc_args_methods["swapcase"] = {
        "type": "be swapcased",
        "method": "swapcase",
        "version": "",
    }
    _doc_args_methods["casefold"] = {
        "type": "be casefolded",
        "method": "casefold",
        "version": "",
    }

    def isalnum(self) -> Series | Index:
        """
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
        return self._map_and_wrap("isalnum", _shared_docs["isalpha"])

    def _map_and_wrap(self, name: str, docstring: str) -> Callable[[StringMethods], Any]:
        return _map_and_wrap(name, docstring)

    isalpha = _map_and_wrap(
        "isalpha",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["isalpha"]
        + _shared_docs["isalpha"],
    )
    isdigit = _map_and_wrap(
        "isdigit",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["isdigit"]
        + _shared_docs["isdigit"],
    )
    isspace = _map_and_wrap(
        "isspace",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["isspace"]
        + _shared_docs["isspace"],
    )
    islower = _map_and_wrap(
        "islower",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["islower"]
        + _shared_docs["islower"],
    )
    isascii = _map_and_wrap(
        "isascii",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["isascii"]
        + _shared_docs["isascii"],
    )
    isupper = _map_and_wrap(
        "isupper",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["isupper"]
        + _shared_docs["isupper"],
    )
    istitle = _map_and_wrap(
        "istitle",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["istitle"]
        + _shared_docs["istitle"],
    )
    isnumeric = _map_and_wrap(
        "isnumeric",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["isnumeric"]
        + _shared_docs["isnumeric"],
    )
    isdecimal = _map_and_wrap(
        "isdecimal",
        docstring=_shared_docs["ismethods"] % _doc_args_methods["isdecimal"]
        + _shared_docs["isdecimal"],
    )

def cat_safe(list_of_columns: list[npt.NDArray[np.object_]], sep: str) -> npt.NDArray[np.object_]:
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
        # if there are any non-string values (wrong dtype or hidden behind
        # object dtype), np.sum will fail; catch and return with better message
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ["string", "empty"]:
                raise TypeError(
                    "Concatenation requires list-likes containing only "
                    "strings (or missing values). Offending values found in "
                    f"column {dtype}"
                ) from None
    return result


def cat_core(list_of_columns: list[npt.NDArray[Any]], sep: str) -> npt.NDArray[Any]:
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
        # no need to interleave sep if it is empty
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)


def _result_dtype(arr: Series | Index) -> DtypeObj | None:
    # workaround #27953
    # ideally we just pass `dtype=arr.dtype` unconditionally, but this fails
    # when the list of values is empty.
    from pandas.core.arrays.string_ import StringDtype

    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object


def _get_single_group_name(regex: re.Pattern) -> Hashable | None:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None


def _get_group_names(regex: re.Pattern) -> list[Hashable] | range:
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
    result: list[Hashable] = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == "i" and lib.is_range_indexer(arr, len(arr)):
        return rng
    return result


def str_extractall(arr: Series | Index, pat: str, flags: int = 0) -> DataFrame:
    """
    Extract all capture groups in the regex `pat` as columns in DataFrame.
    """
    regex = re.compile(pat, flags=flags)
    # the regex must contain capture groups.
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")

    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)

    columns = _get_group_names(regex)
    match_list: list[tuple[Any, ...]] = []
    index_list: list[tuple[Any, ...]] = []
    is_mi: bool = arr.index.nlevels > 1

    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)

            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = [np.nan if group == "" else group for group in match_tuple]
                match_list.append(tuple(na_tuple))
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)

    from pandas import MultiIndex, DataFrame

    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ["match"])
    dtype = _result_dtype(arr)

    if len(match_list) == 0:
        result = DataFrame(columns=columns, dtype=dtype)
    else:
        result = DataFrame(
            match_list, columns=columns, index=index, dtype=dtype
        )

    return result
