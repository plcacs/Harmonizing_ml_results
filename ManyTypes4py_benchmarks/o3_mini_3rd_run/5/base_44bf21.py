from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from pandas.core.indexes.base import Index  # type: ignore
from pandas.core.indexes.multi import MultiIndex  # type: ignore
from pandas.core.arrays.extension import ExtensionArray  # type: ignore
from pandas.core.arrays.categorical import Categorical  # type: ignore

def maybe_sequence_to_range(sequence: Sequence[Any]) -> Union[range, Sequence[Any]]:
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
    if isinstance(sequence, (range, ExtensionArray)):
        return sequence
    elif len(sequence) == 1 or np.array(sequence).dtype.name != 'int64':
        return sequence
    elif isinstance(sequence, (list, tuple)) and (not (isinstance(sequence, np.ndarray))):
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
    elif len(sequence) == 2 or np.all(np.diff(np_sequence) == diff):
        return range(np_sequence[0], np_sequence[-1] + diff, diff)
    else:
        return sequence

def ensure_index_from_sequences(sequences: Sequence[Sequence[Any]], 
                                names: Optional[Sequence[str]] = None) -> Union[Index, MultiIndex]:
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
    from pandas.core.indexes.api import default_index  # type: ignore
    if len(sequences) == 0:
        return default_index(0)
    elif len(sequences) == 1:
        if names is not None:
            names = names[0]
        return Index(maybe_sequence_to_range(sequences[0]), name=names)
    else:
        return MultiIndex.from_arrays(list(sequences), names=names)

def ensure_index(index_like: Any, copy: bool = False) -> Index:
    """
    Ensure that we have an index from some index-like object.

    Parameters
    ----------
    index_like : sequence
        An Index or other sequence.
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
    from pandas import Series  # type: ignore
    if isinstance(index_like, Index):
        if copy:
            index_like = index_like.copy()
        return index_like
    if isinstance(index_like, Series):
        name = index_like.name
        return Index(index_like, name=name, copy=copy)
    if hasattr(index_like, '__iter__') and not isinstance(index_like, (str, bytes)):
        if isinstance(index_like, list):
            if type(index_like) is not list:
                index_like = list(index_like)
            if len(index_like) and all(hasattr(x, '__iter__') for x in index_like):
                return MultiIndex.from_arrays(index_like)
            else:
                return Index(index_like, copy=copy, tupleize_cols=False)
        else:
            return Index(index_like, copy=copy)
    else:
        return Index(index_like, copy=copy)

def trim_front(strings: Sequence[str]) -> List[str]:
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
        return list(strings)
    smallest_leading_space = min((len(x) - len(x.lstrip()) for x in strings))
    if smallest_leading_space > 0:
        strings = [x[smallest_leading_space:] for x in strings]
    return list(strings)

def _validate_join_method(method: str) -> None:
    if method not in ['left', 'right', 'inner', 'outer']:
        raise ValueError(f'do not recognize join method {method}')

def maybe_extract_name(name: Any, obj: Any, cls: type) -> Any:
    """
    If no name is passed, then extract it from data, validating hashability.
    """
    if name is None and isinstance(obj, (Index,)):
        name = obj.name
    if not isinstance(name, (int, str, tuple)) and name is not None:
        raise TypeError(f'{cls.__name__}.name must be a hashable type')
    return name

def get_unanimous_names(*indexes: Index) -> Tuple[Optional[Any], ...]:
    """
    Return common name if all indices agree, otherwise None (level-by-level).

    Parameters
    ----------
    indexes : list of Index objects

    Returns
    -------
    Tuple[Optional[Any], ...]
        A tuple containing the object's names.
    """
    name_tups = (tuple(i.names) for i in indexes)
    name_sets = ({*ns} for ns in zip(*name_tups))
    names = tuple((ns.pop() if len(ns) == 1 else None for ns in name_sets))
    return names

def _unpack_nested_dtype(other: Index) -> Any:
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
    from pandas.core.dtypes.dtypes import CategoricalDtype  # type: ignore
    from pandas.core.arrays.arrow.dtype import ArrowDtype  # type: ignore
    if isinstance(dtype, CategoricalDtype):
        return dtype.categories.dtype
    elif isinstance(dtype, ArrowDtype):
        import pyarrow as pa
        if pa.types.is_dictionary(dtype.pyarrow_dtype):
            other = other[:0].astype(ArrowDtype(dtype.pyarrow_dtype.value_type))
    return other.dtype

def _maybe_try_sort(result: Any, sort: Optional[bool]) -> Any:
    if sort is not False:
        try:
            result = np.sort(result)
        except TypeError as err:
            if sort is True:
                raise
            import warnings
            from pandas.core.common import find_stack_level  # type: ignore
            warnings.warn(f'{err}, sort order is undefined for incomparable objects.', RuntimeWarning, stacklevel=find_stack_level())
    return result

def get_values_for_csv(values: Any, *, date_format: str, na_rep: str = 'nan', 
                       quoting: Optional[Any] = None, float_format: Optional[Any] = None, 
                       decimal: str = '.') -> np.ndarray:
    """
    Convert to types which can be consumed by the standard library's
    csv.writer.writerows.
    """
    from pandas.core.arrays.interval import IntervalDtype  # type: ignore
    from pandas.core.dtypes.cast import np_can_hold_element  # type: ignore
    from pandas.core.dtypes.common import is_object_dtype, is_numeric_dtype, is_string_dtype  # type: ignore
    from pandas._libs import writers  # type: ignore
    from pandas.core.arrays.datetimelike import DatetimeArray, TimedeltaArray  # type: ignore
    from pandas.core.dtypes.dtypes import PeriodDtype  # type: ignore
    from pandas.core.construction import ensure_wrapped_if_datetimelike  # type: ignore

    if isinstance(values, Categorical) and values.categories.dtype.kind in 'Mm':
        values = np.take(values.categories._values, np.array(values._codes, dtype=np.intp))
    values = ensure_wrapped_if_datetimelike(values)
    if isinstance(values, (DatetimeArray, TimedeltaArray)):
        if values.ndim == 1:
            result = values._format_native_types(na_rep=na_rep, date_format=date_format)
            result = result.astype(object, copy=False)
            return result
        results_converted = []
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
    elif values.dtype.kind == 'f' and (not hasattr(values.dtype, 'type')):
        if float_format is None and decimal == '.':
            mask = np.isnan(values)
            if not quoting:
                values = values.astype(str)
            else:
                values = np.array(values, dtype='object')
            values[mask] = na_rep
            values = values.astype(object, copy=False)
            return values
        from pandas.io.formats.format import FloatArrayFormatter  # type: ignore
        formatter = FloatArrayFormatter(values, na_rep=na_rep, float_format=float_format, decimal=decimal, quoting=quoting, fixed_width=False)
        res = formatter.get_result_as_array()
        res = res.astype(object, copy=False)
        return res
    elif isinstance(values, ExtensionArray):
        mask = np.array([v is None or (hasattr(v, 'isna') and v.isna()) for v in values])
        new_values = np.asarray(values.astype(object))
        new_values[mask] = na_rep
        return new_values
    else:
        mask = np.isnan(values) if is_numeric_dtype(values.dtype) else np.array([x is None for x in values])
        itemsize = writers.word_len(na_rep)
        if values.dtype != np.dtype('O') and (not quoting) and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype('U1').itemsize < itemsize:
                values = values.astype(f'<U{itemsize}')
        else:
            values = np.array(values, dtype='object')
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values