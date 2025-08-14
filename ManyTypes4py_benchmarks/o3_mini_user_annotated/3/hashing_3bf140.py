#!/usr/bin/env python3
"""
data hash pandas / numpy objects
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Iterator, Optional, Union

import numpy as np
from numpy import typing as npt

from pandas._libs.hashing import hash_object_array

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    from pandas._typing import ArrayLike

    from pandas import DataFrame, Index, MultiIndex, Series

# 16 byte long hashing key
_default_hash_key: str = "0123456789123456"


def combine_hash_arrays(
    arrays: Iterator[npt.NDArray[np.uint64]], num_items: int
) -> npt.NDArray[np.uint64]:
    """
    Parameters
    ----------
    arrays : Iterator[np.ndarray]
    num_items : int

    Returns
    -------
    np.ndarray[uint64]
    """
    try:
        first: npt.NDArray[np.uint64] = next(arrays)
    except StopIteration:
        return np.array([], dtype=np.uint64)

    arrays = itertools.chain([first], arrays)

    mult: np.uint64 = np.uint64(1000003)
    out: npt.NDArray[np.uint64] = np.zeros_like(first) + np.uint64(0x345678)
    last_i: int = 0
    for i, a in enumerate(arrays):
        inverse_i: int = num_items - i
        out ^= a
        out *= mult
        mult += np.uint64(82520 + inverse_i + inverse_i)
        last_i = i
    assert last_i + 1 == num_items, "Fed in wrong num_items"
    out += np.uint64(97531)
    return out


def hash_pandas_object(
    obj: Union[ABCIndex, ABCDataFrame, ABCSeries],
    index: bool = True,
    encoding: str = "utf8",
    hash_key: Optional[str] = _default_hash_key,
    categorize: bool = True,
) -> "Series":
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
    """
    from pandas import Series

    if hash_key is None:
        hash_key = _default_hash_key

    if isinstance(obj, ABCMultiIndex):
        return Series(
            hash_tuples(obj, encoding, hash_key),
            dtype="uint64",
            copy=False,
        )
    elif isinstance(obj, ABCIndex):
        h: npt.NDArray[np.uint64] = hash_array(obj._values, encoding, hash_key, categorize).astype(
            "uint64", copy=False
        )
        ser: "Series" = Series(h, index=obj, dtype="uint64", copy=False)
    elif isinstance(obj, ABCSeries):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype("uint64", copy=False)
        if index:
            index_iter = (
                hash_pandas_object(
                    obj.index,
                    index=False,
                    encoding=encoding,
                    hash_key=hash_key,
                    categorize=categorize,
                )._values for _ in [None]
            )
            arrays: Iterator[npt.NDArray[np.uint64]] = itertools.chain([h], index_iter)
            h = combine_hash_arrays(arrays, 2)
        ser = Series(h, index=obj.index, dtype="uint64", copy=False)
    elif isinstance(obj, ABCDataFrame):
        hashes = (
            hash_array(series._values, encoding, hash_key, categorize)
            for _, series in obj.items()
        )
        num_items: int = len(obj.columns)
        if index:
            index_hash_generator = (
                hash_pandas_object(
                    obj.index,
                    index=False,
                    encoding=encoding,
                    hash_key=hash_key,
                    categorize=categorize,
                )._values for _ in [None]
            )
            num_items += 1
            _hashes = itertools.chain(hashes, index_hash_generator)
            hashes = (x for x in _hashes)
        h = combine_hash_arrays(hashes, num_items)
        ser = Series(h, index=obj.index, dtype="uint64", copy=False)
    else:
        raise TypeError(f"Unexpected type for hashing {type(obj)}")
    return ser


def hash_tuples(
    vals: Union["MultiIndex", Iterable[tuple[Hashable, ...]]],
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,
) -> npt.NDArray[np.uint64]:
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
        raise TypeError("must be convertible to a list-of-tuples")

    from pandas import Categorical, MultiIndex

    if not isinstance(vals, ABCMultiIndex):
        mi: "MultiIndex" = MultiIndex.from_tuples(vals)  # type: ignore[arg-type]
    else:
        mi = vals

    cat_vals = [
        Categorical._simple_new(
            mi.codes[level],
            CategoricalDtype(categories=mi.levels[level], ordered=False),
        )
        for level in range(mi.nlevels)
    ]

    hashes = (
        cat._hash_pandas_object(encoding=encoding, hash_key=hash_key, categorize=False)
        for cat in cat_vals
    )
    h = combine_hash_arrays(hashes, len(cat_vals))
    return h


def hash_array(
    vals: ArrayLike,
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,
    categorize: bool = True,
) -> npt.NDArray[np.uint64]:
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
    """
    if not hasattr(vals, "dtype"):
        raise TypeError("must pass a ndarray-like")

    if isinstance(vals, ABCExtensionArray):
        return vals._hash_pandas_object(
            encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    if not isinstance(vals, np.ndarray):
        raise TypeError(
            "hash_array requires np.ndarray or ExtensionArray, not "
            f"{type(vals).__name__}. Use hash_pandas_object instead."
        )

    return _hash_ndarray(vals, encoding, hash_key, categorize)


def _hash_ndarray(
    vals: np.ndarray,
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,
    categorize: bool = True,
) -> npt.NDArray[np.uint64]:
    """
    See hash_array.__doc__.
    """
    dtype = vals.dtype

    if np.issubdtype(dtype, np.complex128):
        hash_real = _hash_ndarray(vals.real, encoding, hash_key, categorize)
        hash_imag = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
        return hash_real + 23 * hash_imag

    if dtype == bool:
        vals = vals.astype("u8")
    elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        vals = vals.view("i8").astype("u8", copy=False)
    elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
        vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
    else:
        if categorize:
            from pandas import Categorical, Index, factorize

            codes, categories = factorize(vals, sort=False)
            dtype_cat = CategoricalDtype(categories=Index(categories), ordered=False)
            from pandas import Categorical as PandasCategorical  # type: ignore
            cat = PandasCategorical._simple_new(codes, dtype_cat)
            return cat._hash_pandas_object(
                encoding=encoding, hash_key=hash_key, categorize=False
            )

        try:
            vals = hash_object_array(vals, hash_key, encoding)
        except TypeError:
            vals = hash_object_array(
                vals.astype(str).astype(object), hash_key, encoding
            )

    vals ^= vals >> 30
    vals *= np.uint64(0xBF58476D1CE4E5B9)
    vals ^= vals >> 27
    vals *= np.uint64(0x94D049BB133111EB)
    vals ^= vals >> 31
    return vals
