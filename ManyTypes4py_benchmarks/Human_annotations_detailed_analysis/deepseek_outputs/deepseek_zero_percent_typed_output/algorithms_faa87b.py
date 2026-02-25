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
    Literal, 
    cast, 
    Any, 
    Union, 
    Optional, 
    Tuple, 
    List, 
    Dict, 
    Callable, 
    TypeVar, 
    Sequence, 
    overload
)
import warnings
import numpy as np
from pandas._libs import algos, hashtable as htable, iNaT, lib
from pandas._libs.missing import NA
from pandas._typing import (
    AnyArrayLike, 
    ArrayLike, 
    ArrayLikeT, 
    AxisInt, 
    DtypeObj, 
    TakeIndexer, 
    npt
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike, 
    np_find_common_type
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
    needs_i8_conversion
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype, 
    CategoricalDtype, 
    ExtensionDtype, 
    NumpyEADtype
)
from pandas.core.dtypes.generic import (
    ABCDatetimeArray, 
    ABCExtensionArray, 
    ABCIndex, 
    ABCMultiIndex, 
    ABCNumpyExtensionArray, 
    ABCSeries, 
    ABCTimedeltaArray
)
from pandas.core.dtypes.missing import isna, na_value_for_dtype
from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
    array as pd_array, 
    ensure_wrapped_if_datetimelike, 
    extract_array
)
from pandas.core.indexers import validate_indices

if TYPE_CHECKING:
    from pandas._typing import ListLike, NumpySorter, NumpyValueArrayLike
    from pandas import Categorical, Index, Series
    from pandas.core.arrays import BaseMaskedArray, ExtensionArray

T = TypeVar('T')

def _ensure_data(values: ArrayLike) -> np.ndarray:
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
        values = extract_array(values, extract_numpy=True)
    if is_object_dtype(values.dtype):
        return ensure_object(np.asarray(values))
    elif isinstance(values.dtype, BaseMaskedDtype):
        values = cast('BaseMaskedArray', values)
        if not values._hasna:
            return _ensure_data(values._data)
        return np.asarray(values)
    elif isinstance(values.dtype, CategoricalDtype):
        values = cast('Categorical', values)
        return values.codes
    elif is_bool_dtype(values.dtype):
        if isinstance(values, np.ndarray):
            return np.asarray(values).view('uint8')
        else:
            return np.asarray(values).astype('uint8', copy=False)
    elif is_integer_dtype(values.dtype):
        return np.asarray(values)
    elif is_float_dtype(values.dtype):
        if values.dtype.itemsize in [2, 12, 16]:
            return ensure_float64(values)
        return np.asarray(values)
    elif is_complex_dtype(values.dtype):
        return cast(np.ndarray, values)
    elif needs_i8_conversion(values.dtype):
        npvalues = values.view('i8')
        npvalues = cast(np.ndarray, npvalues)
        return npvalues
    values = np.asarray(values, dtype=object)
    return ensure_object(values)

def _reconstruct_data(
    values: np.ndarray, 
    dtype: Union[np.dtype, ExtensionDtype], 
    original: AnyArrayLike
) -> Union[ABCExtensionArray, np.ndarray]:
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
        return values
    if not isinstance(dtype, np.dtype):
        cls = dtype.construct_array_type()
        values = cls._from_sequence(values, dtype=dtype)
    else:
        values = values.astype(dtype, copy=False)
    return values

def _ensure_arraylike(values: Any, func_name: str) -> ArrayLike:
    """
    ensure that we are arraylike if not already
    """
    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray, ABCNumpyExtensionArray)):
        if func_name != 'isin-targets':
            raise TypeError(f'{func_name} requires a Series, Index, ExtensionArray, np.ndarray or NumpyExtensionArray got {type(values).__name__}.')
        inferred = lib.infer_dtype(values, skipna=False)
        if inferred in ['mixed', 'string', 'mixed-integer']:
            if isinstance(values, tuple):
                values = list(values)
            values = construct_1d_object_array_from_listlike(values)
        else:
            values = np.asarray(values)
    return values

_hashtables: Dict[str, Any] = {
    'complex128': htable.Complex128HashTable, 
    'complex64': htable.Complex64HashTable, 
    'float64': htable.Float64HashTable, 
    'float32': htable.Float32HashTable, 
    'uint64': htable.UInt64HashTable, 
    'uint32': htable.UInt32HashTable, 
    'uint16': htable.UInt16HashTable, 
    'uint8': htable.UInt8HashTable, 
    'int64': htable.Int64HashTable, 
    'int32': htable.Int32HashTable, 
    'int16': htable.Int16HashTable, 
    'int8': htable.Int8HashTable, 
    'string': htable.StringHashTable, 
    'object': htable.PyObjectHashTable
}

def _get_hashtable_algo(values: np.ndarray) -> Tuple[Any, np.ndarray]:
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
    return (hashtable, values)

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
    if ndtype == 'object':
        if lib.is_string_array(values, skipna=False):
            ndtype = 'string'
    return ndtype

def unique(values: ArrayLike) -> Union[np.ndarray, ABCExtensionArray]:
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

def nunique_ints(values: ArrayLike) -> int:
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
    result = (np.bincount(values.ravel().astype('intp')) != 0).sum()
    return result

def unique_with_mask(
    values: ArrayLike, 
    mask: Optional[np.ndarray] = None
) -> Union[np.ndarray, ABCExtensionArray, Tuple[np.ndarray, np.ndarray]]:
    """See algorithms.unique for docs. Takes a mask for masked arrays."""
    values = _ensure_arraylike(values, func_name='unique')
    if isinstance(values.dtype, ExtensionDtype):
        return values.unique()
    if isinstance(values, ABCIndex):
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
        assert mask is not None
        return (uniques, mask.astype('bool'))

unique1d = unique

_MINIMUM_COMP_ARR_LEN = 1000000

def isin(
    comps: ArrayLike, 
    values: ArrayLike
) -> np.ndarray:
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
        raise TypeError(f'only list-like objects are allowed to be passed to isin(), you passed a `{type(comps).__name__}`')
    if not is_list_like(values):
        raise TypeError(f'only list-like objects are allowed to be passed to isin(), you passed a `{type(values).__name__}`')
    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray)):
        orig_values = list(values)
        values = _ensure_arraylike(orig_values, func_name='isin-targets')
        if len(values) > 0 and values.dtype.kind in 'iufcb' and (not is_signed_integer_dtype(comps)):
            values = construct_1d_object_array_from_listlike(orig_values)
    elif isinstance(values, ABCMultiIndex):
        values = np.array(values)
    else:
        values = extract_array(values, extract_numpy=True, extract_range=True)
    comps_array = _ensure_arraylike(comps, func_name='isin')
    comps_array = extract_array(comps_array, extract_numpy=True)
    if not isinstance(comps_array, np.ndarray):
        return comps_array.isin(values)
    elif needs_i8_conversion(comps_array.dtype):
        return pd_array(comps_array).isin(values)
    elif needs_i8_conversion(values.dtype) and (not is_object_dtype(comps_array.dtype)):
        return np.zeros(comps_array.shape, dtype=bool)
    elif needs_i8_conversion(values.dtype):
        return isin(comps_array, values.astype(object))
    elif isinstance(values.dtype, ExtensionDtype):
        return isin(np.asarray(comps_array), np.asarray(values))
    if len(comps_array) > _MINIMUM_COMP_ARR_LEN and len(values) <= 26 and (comps_array.dtype != object) and (not any((v is NA for v in values))):
        if isna(values).any():

            def f(c: np.ndarray, v: np.ndarray) -> np.ndarray:
                return np.logical_or(np.isin(c, v).ravel(), np.isnan(c))
        else:
            f = lambda a, b: np.isin(a, b).ravel()
    else:
        common = np_find_common_type(values.dtype, comps_array.dtype)
        values = values.astype(common, copy=False)
        comps_array = comps_array.astype(common, copy=False)
        f = htable.ismember
    return f(comps_array, values)

def factorize_array(
    values: np.ndarray, 
    use_na_sentinel: bool = True, 
    size_hint: Optional[int] = None, 
    na_value: Any = None, 
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factorize a numpy array to codes and uniques.

    This doesn't do any coercion of types or unboxing before factorization.

    Parameters
    ----------
    values : ndarray
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    size_hint : int, optional
        Passed through to the hashtable's 'get_labels' method
    na_value : object, optional
        A value in `values` to consider missing. Note: only use this
        parameter when you know that you don't have any values pandas would
        consider missing in the array (NaN for float data, iNaT for
        datetimes, etc.).
    mask : ndarray[bool], optional
        If not None, the mask is used as indicator for missing values
        (True = missing, False = valid) instead of `na_value` or
        condition "val != val".

    Returns
    -------
    codes : ndarray[np.intp]
    uniques : ndarray
    """
    original = values
    if values.dtype.kind in '