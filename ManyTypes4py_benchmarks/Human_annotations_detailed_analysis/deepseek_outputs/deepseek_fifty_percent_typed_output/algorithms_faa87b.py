"""
Generic data algorithms. This module is experimental at the moment and not
intended for public consumption
"""

from __future__ import annotations

import decimal
import operator
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._libs import (
    algos,
    hashtable as htable,
    iNaT,
    lib,
)
from pandas._libs.missing import NA
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    ArrayLikeT,
    AxisInt,
    DtypeObj,
    TakeIndexer,
    npt,
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike,
    np_find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_float64,
    ensure_object,
    ensure_platform_int,
    is_bool_dtype,
    is_complex_dtype,
    is_dict_like,
    is_extension_array_dtype,
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_signed_integer_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype,
    CategoricalDtype,
    ExtensionDtype,
    NumpyEADtype,
)
from pandas.core.dtypes.generic import (
    ABCDatetimeArray,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCNumpyExtensionArray,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
)

from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import validate_indices

if TYPE_CHECKING:
    from pandas._typing import (
        ListLike,
        NumpySorter,
        NumpyValueArrayLike,
    )

    from pandas import (
        Categorical,
        Index,
        Series,
    )
    from pandas.core.arrays import (
        BaseMaskedArray,
        ExtensionArray,
    )


# --------------- #
# dtype access    #
# --------------- #
def _ensure_data(values: AnyArrayLike) -> np.ndarray:
    """
    routine to ensure that our data is of the correct
    input dtype for lower-level routines

    This will coerce:
    - ints -> int64
    - uint -> uint64
    - bool -> uint8
    - datetimelike -> i8
    - datetime64tz -> i8 (in local tz)
    - categorical -> codes

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    np.ndarray
    """

    if not isinstance(values, ABCMultiIndex):
        # extract_array would raise
        values = extract_array(values, extract_numpy=True)

    if is_object_dtype(values.dtype):
        return ensure_object(np.asarray(values))

    elif isinstance(values.dtype, BaseMaskedDtype):
        # i.e. BooleanArray, FloatingArray, IntegerArray
        values = cast("BaseMaskedArray", values)
        if not values._hasna:
            # No pd.NAs -> We can avoid an object-dtype cast (and copy) GH#41816
            #  recurse to avoid re-implementing logic for eg bool->uint8
            return _ensure_data(values._data)
        return np.asarray(values)

    elif isinstance(values.dtype, CategoricalDtype):
        # NB: cases that go through here should NOT be using _reconstruct_data
        #  on the back-end.
        values = cast("Categorical", values)
        return values.codes

    elif is_bool_dtype(values.dtype):
        if isinstance(values, np.ndarray):
            # i.e. actually dtype == np.dtype("bool")
            return np.asarray(values).view("uint8")
        else:
            # e.g. Sparse[bool, False]  # TODO: no test cases get here
            return np.asarray(values).astype("uint8", copy=False)

    elif is_integer_dtype(values.dtype):
        return np.asarray(values)

    elif is_float_dtype(values.dtype):
        # Note: checking `values.dtype == "float128"` raises on Windows and 32bit
        # error: Item "ExtensionDtype" of "Union[Any, ExtensionDtype, dtype[Any]]"
        # has no attribute "itemsize"
        if values.dtype.itemsize in [2, 12, 16]:  # type: ignore[union-attr]
            # we dont (yet) have float128 hashtable support
            return ensure_float64(values)
        return np.asarray(values)

    elif is_complex_dtype(values.dtype):
        return cast(np.ndarray, values)

    # datetimelike
    elif needs_i8_conversion(values.dtype):
        npvalues = values.view("i8")
        npvalues = cast(np.ndarray, npvalues)
        return npvalues

    # we have failed, return object
    values = np.asarray(values, dtype=object)
    return ensure_object(values)


def _reconstruct_data(
    values: ArrayLikeT, dtype: DtypeObj, original: AnyArrayLike
) -> ArrayLikeT:
    """
    reverse of _ensure_data

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    dtype : np.dtype or ExtensionDtype
    original : AnyArrayLike

    Returns
    -------
    ExtensionArray or np.ndarray
    """
    if isinstance(values, ABCExtensionArray) and values.dtype == dtype:
        # Catch DatetimeArray/TimedeltaArray
        return values

    if not isinstance(dtype, np.dtype):
        # i.e. ExtensionDtype; note we have ruled out above the possibility
        #  that values.dtype == dtype
        cls = dtype.construct_array_type()

        # error: Incompatible types in assignment (expression has type
        # "ExtensionArray", variable has type "ndarray[Any, Any]")
        values = cls._from_sequence(values, dtype=dtype)  # type: ignore[assignment]

    else:
        values = values.astype(dtype, copy=False)

    return values


def _ensure_arraylike(values: Any, func_name: str) -> ArrayLike:
    """
    ensure that we are arraylike if not already
    """
    if not isinstance(
        values,
        (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray, ABCNumpyExtensionArray),
    ):
        # GH#52986
        if func_name != "isin-targets":
            # Make an exception for the comps argument in isin.
            raise TypeError(
                f"{func_name} requires a Series, Index, "
                f"ExtensionArray, np.ndarray or NumpyExtensionArray "
                f"got {type(values).__name__}."
            )

        inferred = lib.infer_dtype(values, skipna=False)
        if inferred in ["mixed", "string", "mixed-integer"]:
            # "mixed-integer" to ensure we do not cast ["ss", 42] to str GH#22160
            if isinstance(values, tuple):
                values = list(values)
            values = construct_1d_object_array_from_listlike(values)
        else:
            values = np.asarray(values)
    return values


_hashtables = {
    "complex128": htable.Complex128HashTable,
    "complex64": htable.Complex64HashTable,
    "float64": htable.Float64HashTable,
    "float32": htable.Float32HashTable,
    "uint64": htable.UInt64HashTable,
    "uint32": htable.UInt32HashTable,
    "uint16": htable.UInt16HashTable,
    "uint8": htable.UInt8HashTable,
    "int64": htable.Int64HashTable,
    "int32": htable.Int32HashTable,
    "int16": htable.Int16HashTable,
    "int8": htable.Int8HashTable,
    "string": htable.StringHashTable,
    "object": htable.PyObjectHashTable,
}


def _get_hashtable_algo(
    values: AnyArrayLike,
) -> tuple[type[htable.HashTable], np.ndarray]:
    """
    Parameters
    ----------
    values : np.ndarray

    Returns
    -------
    htable : HashTable subclass
    values : ndarray
    """
    values = _ensure_data(values)

    ndtype = _check_object_for_strings(values)
    hashtable = _hashtables[ndtype]
    return hashtable, values


def _check_object_for_strings(values: np.ndarray) -> str:
    """
    Check if we can use string hashtable instead of object hashtable.

    Parameters
    ----------
    values : ndarray

    Returns
    -------
    str
    """
    ndtype = values.dtype.name
    if ndtype == "object":
        # it's cheaper to use a String Hash Table than Object; we infer
        # including nulls because that is the only difference between
        # StringHashTable and ObjectHashtable
        if lib.is_string_array(values, skipna=False):
            ndtype = "string"
    return ndtype


# --------------- #
# top-level algos #
# --------------- #


def unique(values: AnyArrayLike) -> ArrayLike:
    """
    Return unique values based on a hash table.

    Uniques are returned in order of appearance. This does NOT sort.

    Significantly faster than numpy.unique for long enough sequences.
    Includes NA values.

    Parameters
    ----------
    values : 1d array-like
        The input array-like object containing values from which to extract
        unique values.

    Returns
    -------
    numpy.ndarray, ExtensionArray or NumpyExtensionArray

        The return can be:

        * Index : when the input is an Index
        * Categorical : when the input is a Categorical dtype
        * ndarray : when the input is a Series/ndarray

        Return numpy.ndarray, ExtensionArray or NumpyExtensionArray.

    See Also
    --------
    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Examples
    --------
    >>> pd.unique(pd.Series([2, 1, 3, 3]))
    array([2, 1, 3])

    >>> pd.unique(pd.Series([2] + [1] * 5))
    array([2, 1])

    >>> pd.unique(pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")]))
    array(['2016-01-01T00:00:00'], dtype='datetime64[s]')

    >>> pd.unique(
    ...     pd.Series(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ],
    ...         dtype="M8[ns, US/Eastern]",
    ...     )
    ... )
    <DatetimeArray>
    ['2016-01-01 00:00:00-05:00']
    Length: 1, dtype: datetime64[ns, US/Eastern]

    >>> pd.unique(
    ...     pd.Index(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ],
    ...         dtype="M8[ns, US/Eastern]",
    ...     )
    ... )
    DatetimeIndex(['2016-01-01 00:00:00-05:00'],
            dtype='datetime64[ns, US/Eastern]',
            freq=None)

    >>> pd.unique(np.array(list("baabc"), dtype="O"))
    array(['b', 'a', 'c'], dtype=object)

    An unordered Categorical will return categories in the
    order of appearance.

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"), categories=list("abc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    An ordered Categorical preserves the category ordering.

    >>> pd.unique(
    ...     pd.Series(
    ...         pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ...     )
    ... )
    ['b', 'a', 'c']
    Categories (3, object): ['a' < 'b' < 'c']

    An array of tuples

    >>> pd.unique(pd.Series([("a", "b"), ("b", "a"), ("a", "c"), ("b", "a")]).values)
    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)

    An NumpyExtensionArray of complex

    >>> pd.unique(pd.array([1 + 1j, 2, 3]))
    <NumpyExtensionArray>
    [(1+1j), (2+0j), (3+0j)]
    Length: 3, dtype: complex128
    """
    return unique_with_mask(values)


def nunique_ints(values: AnyArrayLike) -> int:
    """
    Return the number of unique values for integer array-likes.

    Significantly faster than pandas.unique for long enough sequences.
    No checks are done to ensure input is integral.

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    int : The number of unique values in ``values``
    """
    if len(values) == 0:
        return 0
    values = _ensure_data(values)
    # bincount requires intp
    result = (np.bincount(values.ravel().astype("intp")) != 0).sum()
    return result


def unique_with_mask(
    values: AnyArrayLike, mask: npt.NDArray[np.bool_] | None = None
) -> ArrayLike | tuple[ArrayLike, npt.NDArray[np.bool_]]:
    """See algorithms.unique for docs. Takes a mask for masked arrays."""
    values = _ensure_arraylike(values, func_name="unique")

    if isinstance(values.dtype, ExtensionDtype):
        # Dispatch to extension dtype's unique.
        return values.unique()

    if isinstance(values, ABCIndex):
        # Dispatch to Index's unique.
        return values.unique()

    original = values
    hashtable, values = _get_hashtable_algo(values)

    table = hashtable(len(values))
    if mask is None:
        uniques = table.unique(values)
        uniques = _reconstruct_data(uniques, original.dtype, original)
        return uniques

    else:
        uniques, mask = table.unique(values, mask=mask)
        uniques = _reconstruct_data(uniques, original.dtype, original)
        assert mask is not None  # for mypy
        return uniques, mask.astype("bool")


unique1d = unique


_MINIMUM_COMP_ARR_LEN = 1_000_000


def isin(comps: ListLike, values: AnyArrayLike) -> npt.NDArray[np.bool_]:
    """
    Compute the isin boolean array.

    Parameters
    ----------
    comps : list-like
    values : list-like

    Returns
    -------
    ndarray[bool]
        Same length as `comps`.
    """
    if not is_list_like(comps):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a `{type(comps).__name__}`"
        )
    if not is_list_like(values):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a `{type(values).__name__}`"
        )

    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray)):
        orig_values = list(values)
        values = _ensure_arraylike(orig_values, func_name="isin-targets")

        if (
            len(values) > 0
            and values.dtype.kind in "iufcb"
            and not is_signed_integer_dtype(comps)
        ):
            # GH#46485 Use object to avoid upcast to float64 later
            # TODO: Share with _find_common_type_compat
            values = construct_1d_object_array_from_listlike(orig_values)

    elif isinstance(values, ABCMultiIndex):
        # Avoid raising in extract_array
        values = np.array(values)
    else:
        values = extract_array(values, extract_numpy=True, extract_range=True)

    comps_array = _ensure_arraylike(comps, func_name="isin")
    comps_array = extract_array(comps_array, extract_numpy=True)
    if not isinstance(comps_array, np.ndarray):
        # i.e. Extension Array
        return comps_array.isin(values)

    elif needs_i8_conversion(comps_array.dtype):
        # Dispatch to DatetimeLikeArrayMixin.isin
        return pd_array(comps_array).isin(values)
    elif needs_i8_conversion(values.dtype) and not is_object_dtype(comps_array.dtype):
        # e.g. comps_array are integers and values are datetime64s
        return np.zeros(comps_array.shape, dtype=bool)
        # TODO: not quite right ... Sparse/Categorical
    elif needs_i8_conversion(values.dtype):
        return isin(comps_array, values.astype(object))

    elif isinstance(values.dtype, ExtensionDtype):
        return isin(np.asarray(comps_array), np.asarray(values))

    # GH16012
    # Ensure np.isin doesn't get object types or it *may* throw an exception
    # Albeit hashmap has O(1) look