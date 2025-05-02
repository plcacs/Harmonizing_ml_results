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
    Sequence, 
    TypeVar, 
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
ArrayLikeOrScalar = Union[ArrayLike, Any]

def _ensure_data(values: ArrayLike) -> np.ndarray:
    """
    routine to ensure that our data is of the correct
    input dtype for lower-level routines
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
    """
    ndtype = values.dtype.name
    if ndtype == 'object':
        if lib.is_string_array(values, skipna=False):
            ndtype = 'string'
    return ndtype

def unique(values: ArrayLike) -> Union[np.ndarray, ABCExtensionArray]:
    """
    Return unique values based on a hash table.
    """
    return unique_with_mask(values)

def nunique_ints(values: ArrayLike) -> int:
    """
    Return the number of unique values for integer array-likes.
    """
    if len(values) == 0:
        return 0
    values = _ensure_data(values)
    result = (np.bincount(values.ravel().astype('intp')) != 0).sum()
    return result

def unique_with_mask(
    values: ArrayLike, 
    mask: Optional[np.ndarray] = None
) -> Union[
    Tuple[np.ndarray, np.ndarray], 
    Union[np.ndarray, ABCExtensionArray]
]:
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

_MINIMUM_COMP_ARR_LEN: int = 1000000

def isin(comps: ArrayLike, values: ArrayLike) -> np.ndarray:
    """
    Compute the isin boolean array.
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
            f = lambda a: np.ndarray, b: np.ndarray: np.isin(a, b).ravel()
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
    """
    original = values
    if values.dtype.kind in 'mM':
        na_value = iNaT
    hash_klass, values = _get_hashtable_algo(values)
    table = hash_klass(size_hint or len(values))
    uniques, codes = table.factorize(values, na_sentinel=-1, na_value=na_value, mask=mask, ignore_na=use_na_sentinel)
    uniques = _reconstruct_data(uniques, original.dtype, original)
    codes = ensure_platform_int(codes)
    return (codes, uniques)

@doc(values=dedent("    values : sequence\n        A 1-D sequence. Sequences that aren't pandas objects are\n        coerced to ndarrays before factorization.\n    "), sort=dedent('    sort : bool, default False\n        Sort `uniques` and shuffle `codes` to maintain the\n        relationship.\n    '), size_hint=dedent('    size_hint : int, optional\n        Hint to the hashtable sizer.\n    '))
def factorize(
    values: ArrayLike,
    sort: bool = False,
    use_na_sentinel: bool = True,
    size_hint: Optional[int] = None
) -> Tuple[np.ndarray, Union[np.ndarray, ABCIndex, ABCExtensionArray]]:
    """
    Encode the object as an enumerated type or categorical variable.
    """
    if isinstance(values, (ABCIndex, ABCSeries)):
        return values.factorize(sort=sort, use_na_sentinel=use_na_sentinel)
    values = _ensure_arraylike(values, func_name='factorize')
    original = values
    if isinstance(values, (ABCDatetimeArray, ABCTimedeltaArray)) and values.freq is not None:
        codes, uniques = values.factorize(sort=sort)
        return (codes, uniques)
    elif not isinstance(values, np.ndarray):
        codes, uniques = values.factorize(use_na_sentinel=use_na_sentinel)
    else:
        values = np.asarray(values)
        if not use_na_sentinel and values.dtype == object:
            null_mask = isna(values)
            if null_mask.any():
                na_value = na_value_for_dtype(values.dtype, compat=False)
                values = np.where(null_mask, na_value, values)
        codes, uniques = factorize_array(values, use_na_sentinel=use_na_sentinel, size_hint=size_hint)
    if sort and len(uniques) > 0:
        uniques, codes = safe_sort(uniques, codes, use_na_sentinel=use_na_sentinel, assume_unique=True, verify=False)
    uniques = _reconstruct_data(uniques, original.dtype, original)
    return (codes, uniques)

def value_counts_internal(
    values: ArrayLike,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins: Optional[int] = None,
    dropna: bool = True
) -> Any:
    from pandas import Index, Series
    index_name = getattr(values, 'name', None)
    name = 'proportion' if normalize else 'count'
    if bins is not None:
        from pandas.core.reshape.tile import cut
        if isinstance(values, Series):
            values = values._values
        try:
            ii = cut(values, bins, include_lowest=True)
        except TypeError as err:
            raise TypeError('bins argument only works with numeric data.') from err
        result = ii.value_counts(dropna=dropna)
        result.name = name
        result = result[result.index.notna()]
        result.index = result.index.astype('interval')
        result = result.sort_index()
        if dropna and (result._values == 0).all():
            result = result.iloc[0:0]
        counts = np.array([len(ii)])
    elif is_extension_array_dtype(values):
        result = Series(values, copy=False)._values.value_counts(dropna=dropna)
        result.name = name
        result.index.name = index_name
        counts = result._values
        if not isinstance(counts, np.ndarray):
            counts = np.asarray(counts)
    elif isinstance(values, ABCMultiIndex):
        levels = list(range(values.nlevels))
        result = Series(index=values, name=name).groupby(level=levels, dropna=dropna).size()
        result.index.names = values.names
        counts = result._values
    else:
        values = _ensure_arraylike(values, func_name='value_counts')
        keys, counts, _ = value_counts_arraylike(values, dropna)
        if keys.dtype == np.float16:
            keys = keys.astype(np.float32)
        idx = Index(keys, dtype=keys.dtype, name=index_name)
        result = Series(counts, index=idx, name=name, copy=False)
    if sort:
        result = result.sort_values(ascending=ascending)
    if normalize:
        result = result / counts.sum()
    return result

def value_counts_arraylike(
    values: np.ndarray,
    dropna: bool,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Parameters
    ----------
    values : np.ndarray
    dropna : bool
    mask : np.ndarray[bool] or None, default None

    Returns
    -------
    uniques : np.ndarray
    counts : np.ndarray[np.int64]
    """
    original = values
    values = _ensure_data(values)
    keys, counts, na_counter = htable.value_count(values, dropna, mask=mask)
    if needs_i8_conversion(original.dtype):
        if dropna:
            mask = keys != iNaT
            keys, counts = (keys[mask], counts[mask])
    res_keys = _reconstruct_data(keys, original.dtype, original)
    return (res_keys, counts, na_counter)

def duplicated(
    values: ArrayLike,
    keep: Literal['first', 'last', False] = 'first',
    mask: Optional[np.ndarray] = None
