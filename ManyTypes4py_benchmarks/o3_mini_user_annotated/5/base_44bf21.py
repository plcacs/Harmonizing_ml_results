from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any, Hashable, Optional

import numpy as np
import pandas._libs.lib as lib
from pandas._libs import algos
from pandas._libs.tslibs import NaT
from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import is_object_dtype, is_numeric_dtype, is_string_dtype
from pandas.core.arrays.categorical import Categorical
from pandas._typing import ArrayLike, Axes, DtypeObj, NDArray

# Assume Index is defined elsewhere in the module.
# Also assume ExtensionArray is defined elsewhere.
Index: Any

def maybe_sequence_to_range(sequence: Sequence[Any]) -> Any | range:
    """
    Convert a 1D, non-pandas sequence to a range if possible.
    Returns the input if not possible.
    """
    if isinstance(sequence, (range,)) or hasattr(sequence, "dtype"):
        return sequence
    elif len(sequence) == 1 or lib.infer_dtype(sequence, skipna=False) != "integer":
        return sequence
    elif isinstance(sequence, (list, tuple)) and not (
        hasattr(sequence, "dtype") and is_object_dtype(sequence.dtype)
    ):
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
        return range(np_sequence[0], np_sequence[-1] + diff, diff)
    else:
        return sequence


def ensure_index_from_sequences(sequences: Sequence[Sequence[Any]], names: Optional[Sequence[str]] = None) -> Index:
    """
    Construct an index from sequences of data.
    A single sequence returns an Index. Many sequences returns a MultiIndex.
    """
    from pandas.core.indexes.api import default_index
    from pandas.core.indexes.multi import MultiIndex

    if len(sequences) == 0:
        return default_index(0)
    elif len(sequences) == 1:
        if names is not None:
            names = names[0]
        return Index(maybe_sequence_to_range(sequences[0]), name=names)
    else:
        # TODO: Apply maybe_sequence_to_range to sequences?
        return MultiIndex.from_arrays(sequences, names=names)


def ensure_index(index_like: Axes, copy: bool = False) -> Index:
    """
    Ensure that we have an index from some index-like object.
    """
    from pandas.core.dtypes.common import is_list_like

    if isinstance(index_like, Index):
        if copy:
            index_like = index_like.copy()
        return index_like

    from pandas import Series

    if isinstance(index_like, Series):
        name = index_like.name
        return Index(index_like, name=name, copy=copy)

    if isinstance(index_like, Iterable) and not isinstance(index_like, (str, bytes)):
        index_like = list(index_like)

    if isinstance(index_like, list):
        if type(index_like) is not list:
            index_like = list(index_like)
        if len(index_like) and lib.is_all_arraylike(index_like):
            from pandas.core.indexes.multi import MultiIndex
            return MultiIndex.from_arrays(index_like)
        else:
            return Index(index_like, copy=copy, tupleize_cols=False)
    else:
        return Index(index_like, copy=copy)


def trim_front(strings: list[str]) -> list[str]:
    """
    Trims leading spaces evenly among all strings.
    """
    if not strings:
        return strings
    smallest_leading_space = min(len(x) - len(x.lstrip()) for x in strings)
    if smallest_leading_space > 0:
        strings = [x[smallest_leading_space:] for x in strings]
    return strings


def _validate_join_method(method: str) -> None:
    if method not in ["left", "right", "inner", "outer"]:
        raise ValueError(f"do not recognize join method {method}")


def maybe_extract_name(name: Any, obj: Any, cls: type[Any]) -> Hashable:
    """
    If no name is passed, then extract it from data, validating hashability.
    """
    if name is None and isinstance(obj, (Index, )):
        name = getattr(obj, "name", None)
    if not isinstance(name, Hashable):
        raise TypeError(f"{cls.__name__}.name must be a hashable type")
    return name


def get_unanimous_names(*indexes: Index) -> tuple[Hashable, ...]:
    """
    Return common name if all indices agree, otherwise None (level-by-level).
    """
    name_tups = (tuple(i.names) for i in indexes)
    name_sets = ({*ns} for ns in zip_longest(*name_tups))
    names = tuple(ns.pop() if len(ns) == 1 else None for ns in name_sets)
    return names


def _unpack_nested_dtype(other: Index) -> DtypeObj:
    """
    When checking if our dtype is comparable with another, we need
    to unpack CategoricalDtype to look at its categories.dtype.
    """
    dtype = other.dtype
    from pandas.core.dtypes.dtypes import CategoricalDtype, ArrowDtype
    if isinstance(dtype, CategoricalDtype):
        return dtype.categories.dtype
    elif isinstance(dtype, ArrowDtype):
        import pyarrow as pa
        if pa.types.is_dictionary(dtype.pyarrow_dtype):
            other = other[:0].astype(ArrowDtype(dtype.pyarrow_dtype.value_type))
    return other.dtype


def _maybe_try_sort(result: Index | ArrayLike, sort: bool | None) -> Index | ArrayLike:
    if sort is not False:
        try:
            result = algos.safe_sort(result)
        except TypeError as err:
            if sort is True:
                raise
            import warnings
            from pandas.util._exceptions import find_stack_level
            warnings.warn(
                f"{err}, sort order is undefined for incomparable objects.",
                RuntimeWarning,
                stacklevel=find_stack_level(),
            )
    return result


def get_values_for_csv(
    values: ArrayLike,
    *,
    date_format: Optional[str] = None,
    na_rep: str = "nan",
    quoting: Any = None,
    float_format: Any = None,
    decimal: str = ".",
) -> NDArray[np.object_]:
    """
    Convert to types which can be consumed by the standard library's
    csv.writer.writerows.
    """
    from pandas.core.arrays.categorical import Categorical
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.arrays.timedeltas import TimedeltaArray
    from pandas.core.arrays.period import PeriodArray
    from pandas.core.arrays.interval import IntervalArray
    from pandas.core.dtypes.dtypes import SparseDtype
    from pandas.io.formats import writers

    if isinstance(values, Categorical) and values.categories.dtype.kind in "Mm":
        values = algos.take_nd(
            values.categories._values,
            np.asarray(values._codes, dtype=np.intp),
            fill_value=na_rep,
        )
    from pandas.core.arrays.datetimelike import ensure_wrapped_if_datetimelike
    values = ensure_wrapped_if_datetimelike(values)
    if isinstance(values, (DatetimeArray, TimedeltaArray)):
        if values.ndim == 1:
            result = values._format_native_types(na_rep=na_rep, date_format=date_format)
            result = result.astype(object, copy=False)
            return result
        results_converted = []
        for i in range(len(values)):
            result = values[i, :]._format_native_types(
                na_rep=na_rep, date_format=date_format
            )
            results_converted.append(result.astype(object, copy=False))
        return np.vstack(results_converted)
    elif isinstance(values.dtype, PeriodArray().dtype.__class__):
        values = cast(PeriodArray, values)
        res = values._format_native_types(na_rep=na_rep, date_format=date_format)
        return res
    elif isinstance(values.dtype, IntervalArray().dtype.__class__):
        values = cast(IntervalArray, values)
        mask = values.isna()
        if not quoting:
            result = np.asarray(values).astype(str)
        else:
            result = np.array(values, dtype=object, copy=True)
        result[mask] = na_rep
        return result
    elif values.dtype.kind == "f" and not isinstance(values.dtype, SparseDtype):
        mask = lib.isna(values)
        if not quoting:
            values = values.astype(str)
        else:
            values = np.array(values, dtype="object")
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values
    elif hasattr(values, "dtype") and hasattr(values.dtype, "kind") and values.dtype.kind not in "fMiO":
        mask = lib.isna(values)
        itemsize = writers.word_len(na_rep)
        if values.dtype != np.dtype("object") and not quoting and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype("U1").itemsize < itemsize:
                values = values.astype(f"<U{itemsize}")
        else:
            values = np.array(values, dtype="object")
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values
    elif hasattr(values, "dtype") and hasattr(values.dtype, "kind") and is_object_dtype(values.dtype):
        mask = lib.isna(values)
        new_values = np.asarray(values.astype(object))
        new_values[mask] = na_rep
        return new_values
    else:
        mask = lib.isna(values)
        itemsize = writers.word_len(na_rep)
        if values.dtype != np.dtype("object") and not quoting and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype("U1").itemsize < itemsize:
                values = values.astype(f"<U{itemsize}")
        else:
            values = np.array(values, dtype="object")
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values