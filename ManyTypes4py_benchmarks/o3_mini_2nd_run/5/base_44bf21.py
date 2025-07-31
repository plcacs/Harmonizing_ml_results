#!/usr/bin/env python
from __future__ import annotations
from collections.abc import Iterable
from itertools import zip_longest
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

# Depending on your environment, these imports may need to be adjusted.
from pandas import Index, MultiIndex
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.arrays.categorical import Categorical


def maybe_sequence_to_range(sequence: Sequence[Any]) -> Union[Sequence[Any], range]:
    """
    Convert a 1D, non-pandas sequence to a range if possible.

    Returns the input if not possible.

    Parameters
    ----------
    sequence : 1D sequence
    names : sequence of str

    Returns
    -------
    Any : input or range
    """
    if isinstance(sequence, (range,)) or hasattr(sequence, "ndim"):
        return sequence
    elif len(sequence) == 1 or np.array(sequence).dtype.kind != 'i':
        return sequence
    elif isinstance(sequence, (list, tuple)) and (not (hasattr(sequence, "dtype"))):
        return sequence
    if len(sequence) == 0:
        return range(0)
    try:
        np_sequence = np.asarray(sequence, dtype=np.int64)
    except OverflowError:
        return sequence
    diff = np_sequence[1] - np_sequence[0]
    if diff == 0:
        return sequence
    elif len(sequence) == 2 or all(np_sequence[i+1]-np_sequence[i] == diff for i in range(len(np_sequence) - 1)):
        return range(np_sequence[0], np_sequence[-1] + diff, diff)
    else:
        return sequence


def ensure_index_from_sequences(sequences: Sequence[Any], names: Optional[Sequence[Any]] = None) -> Union[Index, MultiIndex]:
    """
    Construct an index from sequences of data.

    A single sequence returns an Index. Many sequences returns a
    MultiIndex.

    Parameters
    ----------
    sequences : sequence of sequences
    names : sequence of str

    Returns
    -------
    index : Index or MultiIndex

    Examples
    --------
    >>> ensure_index_from_sequences([[1, 2, 4]], names=["name"])
    Index([1, 2, 4], dtype='int64', name='name')

    >>> ensure_index_from_sequences([["a", "a"], ["a", "b"]], names=["L1", "L2"])
    MultiIndex([('a', 'a'),
                ('a', 'b')],
               names=['L1', 'L2'])

    See Also
    --------
    ensure_index
    """
    from pandas.core.indexes.api import default_index
    if len(sequences) == 0:
        return default_index(0)
    elif len(sequences) == 1:
        _name: Optional[Any] = names[0] if names is not None else None
        return Index(maybe_sequence_to_range(sequences[0]), name=_name)
    else:
        return MultiIndex.from_arrays(sequences, names=names)


def ensure_index(index_like: Any, copy: bool = False) -> Union[Index, MultiIndex]:
    """
    Ensure that we have an index from some index-like object.

    Parameters
    ----------
    index_like : sequence
        An Index or other sequence
    copy : bool, default False

    Returns
    -------
    index : Index or MultiIndex

    See Also
    --------
    ensure_index_from_sequences

    Examples
    --------
    >>> ensure_index(["a", "b"])
    Index(['a', 'b'], dtype='object')

    >>> ensure_index([("a", "a"), ("b", "c")])
    Index([('a', 'a'), ('b', 'c')], dtype='object')

    >>> ensure_index([["a", "a"], ["b", "c"]])
    MultiIndex([('a', 'b'),
                ('a', 'c')],
               )
    """
    from pandas.core.indexes.api import default_index
    if isinstance(index_like, Index):
        return index_like.copy() if copy else index_like
    # Assuming ABCSeries exists in the pandas namespace.
    try:
        from pandas import Series as ABCSeries  # type: ignore
    except ImportError:
        ABCSeries = object  # type: ignore
    if isinstance(index_like, ABCSeries):
        name = getattr(index_like, "name", None)
        return Index(index_like, name=name, copy=copy)
    if isinstance(index_like, Iterable) and not isinstance(index_like, (str, bytes)):
        if isinstance(index_like, list):
            if type(index_like) is not list:
                index_like = list(index_like)
            if len(index_like) and all(isinstance(x, (list, tuple, np.ndarray)) for x in index_like):
                return MultiIndex.from_arrays(index_like)
            else:
                return Index(index_like, copy=copy, tupleize_cols=False)
        else:
            return Index(index_like, copy=copy)
    else:
        return Index(index_like, copy=copy)


def trim_front(strings: Sequence[str]) -> Sequence[str]:
    """
    Trims leading spaces evenly among all strings.

    Examples
    --------
    >>> trim_front([" a", " b"])
    ['a', 'b']

    >>> trim_front([" a", " "])
    ['a', '']
    """
    if not strings:
        return strings
    smallest_leading_space = min((len(x) - len(x.lstrip()) for x in strings))
    if smallest_leading_space > 0:
        strings = [x[smallest_leading_space:] for x in strings]
    return strings


def _validate_join_method(method: str) -> None:
    if method not in ['left', 'right', 'inner', 'outer']:
        raise ValueError(f'do not recognize join method {method}')


def maybe_extract_name(name: Optional[Any], obj: Any, cls: Type[Any]) -> Any:
    """
    If no name is passed, then extract it from data, validating hashability.
    """
    if name is None and isinstance(obj, (Index,)):
        name = obj.name
    try:
        hash(name)
    except Exception as e:
        raise TypeError(f'{cls.__name__}.name must be a hashable type') from e
    return name


def get_unanimous_names(*indexes: Index) -> Tuple[Any, ...]:
    """
    Return common name if all indices agree, otherwise None (level-by-level).

    Parameters
    ----------
    indexes : list of Index objects

    Returns
    -------
    tuple
        A tuple representing the unanimous 'names' found.
    """
    name_tups = (tuple(i.names) for i in indexes)
    name_sets = ({*ns} for ns in zip_longest(*name_tups))
    names = tuple((ns.pop() if len(ns) == 1 else None for ns in name_sets))
    return names


def _unpack_nested_dtype(other: Index) -> Union[np.dtype[Any], ExtensionDtype]:
    """
    When checking if our dtype is comparable with another, we need
    to unpack CategoricalDtype to look at its categories.dtype.

    Parameters
    ----------
    other : Index

    Returns
    -------
    np.dtype or ExtensionDtype
    """
    dtype = other.dtype
    if hasattr(dtype, "categories") and isinstance(dtype, Categorical):
        return dtype.categories.dtype
    elif hasattr(dtype, "pyarrow_dtype"):
        try:
            import pyarrow as pa
        except ImportError:
            pass
        else:
            if pa.types.is_dictionary(dtype.pyarrow_dtype):
                other = other[:0].astype(ExtensionDtype(dtype.pyarrow_dtype.value_type))
    return other.dtype


def _maybe_try_sort(result: Any, sort: Any) -> Any:
    if sort is not False:
        try:
            result = np.sort(result)
        except TypeError as err:
            if sort is True:
                raise err
            import warnings
            from pandas.util._exceptions import find_stack_level
            warnings.warn(f'{err}, sort order is undefined for incomparable objects.', RuntimeWarning, stacklevel=find_stack_level())
    return result


def get_values_for_csv(
    values: Any,
    *,
    date_format: Optional[str],
    na_rep: str = 'nan',
    quoting: Optional[Any] = None,
    float_format: Optional[Callable[[float], str]] = None,
    decimal: str = '.'
) -> Any:
    """
    Convert to types which can be consumed by the standard library's
    csv.writer.writerows.
    """
    from pandas.core.arrays.categorical import Categorical  # type: ignore

    if isinstance(values, Categorical) and values.categories.dtype.kind in 'Mm':
        values = np.take(values.categories._values, np.array(values._codes, dtype=np.intp))
    # Ensure datetimelike types are wrapped appropriately.
    # The functions below (e.g., _format_native_types) are assumed to exist.
    if hasattr(values, "_format_native_types"):
        if getattr(values, "ndim", 1) == 1:
            result = values._format_native_types(na_rep=na_rep, date_format=date_format)
            result = result.astype(object, copy=False)
            return result
        else:
            results_converted: List[Any] = []
            for i in range(len(values)):
                result = values[i, :]._format_native_types(na_rep=na_rep, date_format=date_format)
                results_converted.append(result.astype(object, copy=False))
            return np.vstack(results_converted)
    elif hasattr(values.dtype, "name") and values.dtype.name == "period":
        # Assuming we have a PeriodArray with _format_native_types
        res = values._format_native_types(na_rep=na_rep, date_format=date_format)
        return res
    elif hasattr(values.dtype, "name") and values.dtype.name.startswith("interval"):
        mask = np.isnan(values)
        if not quoting:
            result = np.array(values).astype(str)
        else:
            result = np.array(values, dtype=object, copy=True)
        result[mask] = na_rep
        return result
    elif isinstance(values, np.ndarray) and values.dtype.kind == 'f':
        if float_format is None and decimal == '.':
            mask = np.isnan(values)
            if not quoting:
                values = values.astype(str)
            else:
                values = np.array(values, dtype='object')
            values[mask] = na_rep
            values = values.astype(object, copy=False)
            return values
        else:
            from pandas.io.formats.format import FloatArrayFormatter
            formatter = FloatArrayFormatter(values, na_rep=na_rep, float_format=float_format, decimal=decimal, quoting=quoting, fixed_width=False)
            res = formatter.get_result_as_array()
            res = res.astype(object, copy=False)
            return res
    elif hasattr(values, "dtype") and getattr(values.dtype, "kind", None) == 'O':
        mask = np.isnan(values)
        itemsize = len(na_rep)
        if values.dtype != np.dtype('object') and (not quoting) and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype('U1').itemsize < itemsize:
                values = values.astype(f'<U{itemsize}')
        else:
            values = np.array(values, dtype='object')
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values
    else:
        mask = np.isnan(values)
        values = np.array(values, dtype='object')
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values
