from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, cast
import numpy as np
import numpy.typing as npt
from pandas._libs.hashing import hash_object_array
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCExtensionArray, ABCIndex, ABCMultiIndex, ABCSeries
if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Iterator
    from pandas._typing import ArrayLike, npt
    from pandas import DataFrame, Index, MultiIndex, Series
_default_hash_key: str = '0123456789123456'

def combine_hash_arrays(arrays, num_items):
    """
    Parameters
    ----------
    arrays : Iterator[np.ndarray]
    num_items : int

    Returns
    -------
    np.ndarray[uint64]

    Should be the same as CPython's tupleobject.c
    """
    try:
        first = next(arrays)
    except StopIteration:
        return np.array([], dtype=np.uint64)
    arrays = itertools.chain([first], arrays)
    mult: np.uint64 = np.uint64(1000003)
    out: npt.NDArray[np.uint64] = np.zeros_like(first) + np.uint64(3430008)
    last_i: int = 0
    for i, a in enumerate(arrays):
        inverse_i: int = num_items - i
        out ^= a
        out *= mult
        mult += np.uint64(82520 + inverse_i + inverse_i)
        last_i = i
    assert last_i + 1 == num_items, 'Fed in wrong num_items'
    out += np.uint64(97531)
    return out

def hash_pandas_object(obj, index=True, encoding='utf8', hash_key=_default_hash_key, categorize=True):
    """
    Return a data hash of the Index/Series/DataFrame.

    Parameters
    ----------
    obj : Index, Series, or DataFrame
    index : bool, default True
        Include the index in the hash (if Series/DataFrame).
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    Series of uint64
        Same length as the object.

    Examples
    --------
    >>> pd.util.hash_pandas_object(pd.Series([1, 2, 3]))
    0    14639053686158035780
    1     3869563279212530728
    2      393322362522515241
    dtype: uint64
    """
    from pandas import Series
    if hash_key is None:
        hash_key = _default_hash_key
    if isinstance(obj, ABCMultiIndex):
        return Series(hash_tuples(obj, encoding, hash_key), dtype='uint64', copy=False)
    elif isinstance(obj, ABCIndex):
        h: npt.NDArray[np.uint64] = hash_array(obj._values, encoding, hash_key, categorize).astype('uint64', copy=False)
        ser: Series = Series(h, index=obj, dtype='uint64', copy=False)
    elif isinstance(obj, ABCSeries):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype('uint64', copy=False)
        if index:
            index_iter: Iterator[npt.NDArray[np.uint64]] = (hash_pandas_object(obj.index, index=False, encoding=encoding, hash_key=hash_key, categorize=categorize)._values for _ in [None])
            arrays: Iterator[npt.NDArray[np.uint64]] = itertools.chain([h], index_iter)
            h = combine_hash_arrays(arrays, 2)
        ser = Series(h, index=obj.index, dtype='uint64', copy=False)
    elif isinstance(obj, ABCDataFrame):
        hashes: Iterator[npt.NDArray[np.uint64]] = (hash_array(series._values, encoding, hash_key, categorize) for _, series in obj.items())
        num_items: int = len(obj.columns)
        if index:
            index_hash_generator: Iterator[npt.NDArray[np.uint64]] = (hash_pandas_object(obj.index, index=False, encoding=encoding, hash_key=hash_key, categorize=categorize)._values for _ in [None])
            num_items += 1
            _hashes: Iterator[npt.NDArray[np.uint64]] = itertools.chain(hashes, index_hash_generator)
            hashes = (x for x in _hashes)
        h = combine_hash_arrays(hashes, num_items)
        ser = Series(h, index=obj.index, dtype='uint64', copy=False)
    else:
        raise TypeError(f'Unexpected type for hashing {type(obj)}')
    return ser

def hash_tuples(vals, encoding='utf8', hash_key=_default_hash_key):
    """
    Hash an MultiIndex / listlike-of-tuples efficiently.

    Parameters
    ----------
    vals : MultiIndex or listlike-of-tuples
    encoding : str, default 'utf8'
    hash_key : str, default _default_hash_key

    Returns
    -------
    ndarray[np.uint64] of hashed values
    """
    if not is_list_like(vals):
        raise TypeError('must be convertible to a list-of-tuples')
    from pandas import Categorical, MultiIndex
    if not isinstance(vals, ABCMultiIndex):
        mi: MultiIndex = MultiIndex.from_tuples(vals)
    else:
        mi = vals
    cat_vals: list[Categorical] = [Categorical._simple_new(mi.codes[level], CategoricalDtype(categories=mi.levels[level], ordered=False)) for level in range(mi.nlevels)]
    hashes: Iterator[npt.NDArray[np.uint64]] = (cat._hash_pandas_object(encoding=encoding, hash_key=hash_key, categorize=False) for cat in cat_vals)
    h: npt.NDArray[np.uint64] = combine_hash_arrays(hashes, len(cat_vals))
    return h

def hash_array(vals, encoding='utf8', hash_key=_default_hash_key, categorize=True):
    """
    Given a 1d array, return an array of deterministic integers.

    Parameters
    ----------
    vals : ndarray or ExtensionArray
        The input array to hash.
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    ndarray[np.uint64, ndim=1]
        Hashed values, same length as the vals.

    See Also
    --------
    util.hash_pandas_object : Return a data hash of the Index/Series/DataFrame.
    util.hash_tuples : Hash an MultiIndex / listlike-of-tuples efficiently.

    Examples
    --------
    >>> pd.util.hash_array(np.array([1, 2, 3]))
    array([ 6238072747940578789, 15839785061582574730,  2185194620014831856],
      dtype=uint64)
    """
    if not hasattr(vals, 'dtype'):
        raise TypeError('must pass a ndarray-like')
    if isinstance(vals, ABCExtensionArray):
        return vals._hash_pandas_object(encoding=encoding, hash_key=hash_key, categorize=categorize)
    if not isinstance(vals, np.ndarray):
        raise TypeError(f'hash_array requires np.ndarray or ExtensionArray, not {type(vals).__name__}. Use hash_pandas_object instead.')
    return _hash_ndarray(vals, encoding, hash_key, categorize)

def _hash_ndarray(vals, encoding='utf8', hash_key=_default_hash_key, categorize=True):
    """
    See hash_array.__doc__.
    """
    dtype: np.dtype = vals.dtype
    if np.issubdtype(dtype, np.complex128):
        hash_real: npt.NDArray[np.uint64] = _hash_ndarray(vals.real, encoding, hash_key, categorize)
        hash_imag: npt.NDArray[np.uint64] = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
        return hash_real + 23 * hash_imag
    if dtype == bool:
        vals = vals.astype('u8')
    elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        vals = vals.view('i8').astype('u8', copy=False)
    elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
        vals = vals.view(f'u{vals.dtype.itemsize}').astype('u8')
    else:
        if categorize:
            from pandas import Categorical, Index, factorize
            codes, categories = factorize(vals, sort=False)
            dtype = CategoricalDtype(categories=Index(categories), ordered=False)
            cat: Categorical = Categorical._simple_new(codes, dtype)
            return cat._hash_pandas_object(encoding=encoding, hash_key=hash_key, categorize=False)
        try:
            vals = hash_object_array(vals, hash_key, encoding)
        except TypeError:
            vals = hash_object_array(vals.astype(str).astype(object), hash_key, encoding)
    vals ^= vals >> 30
    vals *= np.uint64(13787848793156543929)
    vals ^= vals >> 27
    vals *= np.uint64(10723151780598845931)
    vals ^= vals >> 31
    return vals