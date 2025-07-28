from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any, Optional, List, Tuple, Union, Dict
import numpy as np
import warnings

# These imports are assumed to be available in the pandas context.
from pandas import Index, Series
from pandas.core.arrays.base import ExtensionArray
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.api import default_index
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.missing import isna
from pandas.core.dtypes.dtypes import SparseDtype
from pandas._libs import lib, algos
from pandas._libs.lib import find_stack_level


def maybe_sequence_to_range(sequence: Sequence[Any]) -> Union[Sequence[Any], range]:
    if isinstance(sequence, (range, ExtensionArray)):
        return sequence
    elif len(sequence) == 1 or lib.infer_dtype(sequence, skipna=False) != 'integer':
        return sequence
    elif isinstance(sequence, (Series, Index)):
        # If data is wrapped in a Series or Index, return as‐is.
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
    elif len(sequence) == 2 or lib.is_sequence_range(np_sequence, diff):
        return range(int(np_sequence[0]), int(np_sequence[-1] + diff), int(diff))
    else:
        return sequence


def ensure_index_from_sequences(sequences: Sequence[Sequence[Any]], names: Optional[Sequence[str]] = None) -> Union[Index, MultiIndex]:
    """
    Construct an index from sequences of data.

    A single sequence returns an Index. Many sequences returns a MultiIndex.
    """
    if len(sequences) == 0:
        return default_index(0)
    elif len(sequences) == 1:
        if names is not None:
            names = names[0]
        return Index(maybe_sequence_to_range(sequences[0]), name=names)
    else:
        return MultiIndex.from_arrays(sequences, names=names)


def ensure_index(index_like: Any, copy: bool = False) -> Union[Index, MultiIndex]:
    """
    Ensure that we have an Index from some index‐like object.
    """
    if isinstance(index_like, Index):
        if copy:
            index_like = index_like.copy()
        return index_like
    if isinstance(index_like, Series):
        name = index_like.name
        return Index(index_like, name=name, copy=copy)
    if isinstance(index_like, Iterable) and not isinstance(index_like, (str, bytes)):
        index_like = list(index_like)
    if isinstance(index_like, list):
        if type(index_like) is not list:
            index_like = list(index_like)
        if len(index_like) and lib.is_all_arraylike(index_like):
            return MultiIndex.from_arrays(index_like)
        else:
            return Index(index_like, copy=copy, tupleize_cols=False)
    else:
        return Index(index_like, copy=copy)


def trim_front(strings: List[str]) -> List[str]:
    """
    Trims leading spaces evenly among all strings.
    """
    if not strings:
        return strings
    smallest_leading_space = min((len(x) - len(x.lstrip()) for x in strings))
    if smallest_leading_space > 0:
        strings = [x[smallest_leading_space:] for x in strings]
    return strings


def _validate_join_method(method: str) -> None:
    """
    Validate that the join method is one of the acceptable strings.
    """
    if method not in ['left', 'right', 'inner', 'outer']:
        raise ValueError(f'do not recognize join method {method}')


def maybe_extract_name(name: Optional[Any], obj: Any, cls: Any) -> Any:
    """
    If no name is passed, extract it from data, validating hashability.
    """
    if name is None and isinstance(obj, (Index, Series)):
        name = obj.name
    # For simplicity, assume that name must be hashable; here we rely on Python's hash.
    try:
        hash(name)
    except Exception as err:
        raise TypeError(f'{cls.__name__}.name must be a hashable type') from err
    return name


def get_unanimous_names(*indexes: Index) -> Tuple[Any, ...]:
    """
    Return common name if all indices agree, otherwise None (level-by-level).
    """
    name_tups = (tuple(i.names) for i in indexes)
    name_sets = ({*ns} for ns in zip(*name_tups))
    names = tuple((ns.pop() if len(ns) == 1 else None for ns in name_sets))
    return names


def _unpack_nested_dtype(other: Index) -> Any:
    """
    Unpack CategoricalDtype to look at its categories.dtype.
    """
    dtype = other.dtype
    from pandas.api.types import CategoricalDtype
    if isinstance(dtype, CategoricalDtype):
        return dtype.categories.dtype
    elif hasattr(dtype, "pyarrow_dtype"):
        from pandas.core.arrays.arrow import ArrowDtype
        import pyarrow as pa
        if pa.types.is_dictionary(dtype.pyarrow_dtype):
            other = other[:0].astype(ArrowDtype(dtype.pyarrow_dtype.value_type))
    return other.dtype


def _maybe_try_sort(result: np.ndarray, sort: Optional[bool]) -> np.ndarray:
    """
    If sort is requested, try to safely sort the result.
    """
    if sort is not False:
        try:
            result = algos.safe_sort(result)
        except TypeError as err:
            if sort is True:
                raise
            warnings.warn(f'{err}, sort order is undefined for incomparable objects.',
                          RuntimeWarning, stacklevel=find_stack_level())
    return result


def get_values_for_csv(
    values: Any,
    *,
    date_format: Optional[str],
    na_rep: str = 'nan',
    quoting: Optional[Any] = None,
    float_format: Optional[str] = None,
    decimal: str = '.'
) -> np.ndarray:
    """
    Convert to types which can be consumed by the standard library's csv.writer.writerows.
    """
    from pandas import Categorical
    from pandas.core.arrays.datetimes import DatetimeArray, TimedeltaArray
    from pandas.core.dtypes.dtypes import PeriodDtype, IntervalDtype
    from pandas.io.formats.format import FloatArrayFormatter

    if isinstance(values, Categorical) and values.categories.dtype.kind in 'Mm':
        values = algos.take_nd(values.categories._values,
                               np.asarray(values._codes, dtype=np.intp),
                               fill_value=na_rep)
    # Assume ensure_wrapped_if_datetimelike is available in pandas.core.indexes.base
    from pandas.core.indexes.base import ensure_wrapped_if_datetimelike
    values = ensure_wrapped_if_datetimelike(values)

    if isinstance(values, (DatetimeArray, TimedeltaArray)):
        if values.ndim == 1:
            result = values._format_native_types(na_rep=na_rep, date_format=date_format)
            result = result.astype(object, copy=False)
            return result
        results_converted: List[np.ndarray] = []
        for i in range(len(values)):
            result = values[i, :]._format_native_types(na_rep=na_rep, date_format=date_format)
            results_converted.append(result.astype(object, copy=False))
        return np.vstack(results_converted)
    elif isinstance(values.dtype, PeriodDtype):
        values = values  # type: ignore
        res = values._format_native_types(na_rep=na_rep, date_format=date_format)
        return res
    elif isinstance(values.dtype, IntervalDtype):
        values = values  # type: ignore
        mask = values.isna()
        if not quoting:
            result = np.asarray(values).astype(str)
        else:
            result = np.array(values, dtype=object, copy=True)
        result[mask] = na_rep
        return result
    elif values.dtype.kind == 'f' and (not isinstance(values.dtype, SparseDtype)):
        if float_format is None and decimal == '.':
            mask = isna(values)
            if not quoting:
                values = values.astype(str)
            else:
                values = np.array(values, dtype='object')
            values[mask] = na_rep
            values = values.astype(object, copy=False)
            return values
        formatter = FloatArrayFormatter(values, na_rep=na_rep, float_format=float_format,
                                        decimal=decimal, quoting=quoting, fixed_width=False)
        res = formatter.get_result_as_array()
        res = res.astype(object, copy=False)
        return res
    elif isinstance(values, ExtensionArray):
        mask = isna(values)
        new_values = np.asarray(values.astype(object))
        new_values[mask] = na_rep
        return new_values
    else:
        mask = isna(values)
        itemsize = lib.writers.word_len(na_rep)
        if values.dtype != np.dtype("object") and (not quoting) and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype("U1").itemsize < itemsize:
                values = values.astype(f"<U{itemsize}")
        else:
            values = np.array(values, dtype="object")
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values

# End of annotated functions.
