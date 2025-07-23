from __future__ import annotations
from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, cast, final, overload
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import NaT, algos as libalgos, index as libindex, lib, writers
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import is_datetime_array, no_default
from pandas._libs.tslibs import IncompatibleFrequency, OutOfBoundsDatetime, Timestamp, tz_compare
from pandas._typing import AnyAll, ArrayLike, Axes, Axis, AxisInt, DropKeep, Dtype, DtypeObj, F, IgnoreRaise, IndexLabel, IndexT, JoinHow, Level, NaPosition, ReindexMethod, Self, Shape, SliceType, npt
from pandas.compat.numpy import function as nv
from pandas.errors import DuplicateLabelError, InvalidIndexError
from pandas.util._decorators import Appender, cache_readonly, doc, set_module
from pandas.util._exceptions import find_stack_level, rewrite_exception
from pandas.core.dtypes.astype import astype_array, astype_is_view
from pandas.core.dtypes.cast import LossySetitemError, can_hold_element, common_dtype_categorical_compat, find_result_type, infer_dtype_from, maybe_cast_pointwise_result, np_can_hold_element
from pandas.core.dtypes.common import ensure_int64, ensure_object, ensure_platform_int, is_any_real_numeric_dtype, is_bool_dtype, is_ea_or_datetimelike_dtype, is_float, is_hashable, is_integer, is_iterator, is_list_like, is_numeric_dtype, is_object_dtype, is_scalar, is_signed_integer_dtype, is_string_dtype, needs_i8_conversion, pandas_dtype, validate_all_hashable
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype, DatetimeTZDtype, ExtensionDtype, IntervalDtype, PeriodDtype, SparseDtype
from pandas.core.dtypes.generic import ABCCategoricalIndex, ABCDataFrame, ABCDatetimeIndex, ABCIntervalIndex, ABCMultiIndex, ABCPeriodIndex, ABCRangeIndex, ABCSeries, ABCTimedeltaIndex
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import array_equivalent, is_valid_na_for_dtype, isna
from pandas.core import arraylike, nanops, ops
from pandas.core.accessor import Accessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import setitem_datetimelike_compat, validate_putmask
from pandas.core.arrays import ArrowExtensionArray, BaseMaskedArray, Categorical, DatetimeArray, ExtensionArray, TimedeltaArray
from pandas.core.arrays.string_ import StringArray, StringDtype
from pandas.core.base import IndexOpsMixin, PandasObject
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike, extract_array, sanitize_array
from pandas.core.indexers import disallow_ndim_indexing, is_valid_positional_slice
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.sorting import ensure_key_mapped, get_group_index_sorter, nargsort
from pandas.core.strings.accessor import StringMethods
from pandas.io.formats.printing import PrettyDict, default_pprint, format_object_summary, pprint_thing
if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Sequence
    from pandas import CategoricalIndex, DataFrame, MultiIndex, Series
    from pandas.core.arrays import IntervalArray, PeriodArray
__all__ = ['Index']
_unsortable_types = frozenset(('mixed', 'mixed-integer'))
_index_doc_kwargs = {'klass': 'Index', 'inplace': '', 'target_klass': 'Index', 'raises_section': '', 'unique': 'Index', 'duplicated': 'np.ndarray'}
_index_shared_docs = {}
str_t = str
_dtype_obj = np.dtype('object')
_masked_engines = {'Complex128': libindex.MaskedComplex128Engine, 'Complex64': libindex.MaskedComplex64Engine, 'Float64': libindex.MaskedFloat64Engine, 'Float32': libindex.MaskedFloat32Engine, 'UInt64': libindex.MaskedUInt64Engine, 'UInt32': libindex.MaskedUInt32Engine, 'UInt16': libindex.MaskedUInt16Engine, 'UInt8': libindex.MaskedUInt8Engine, 'Int64': libindex.MaskedInt64Engine, 'Int32': libindex.MaskedInt32Engine, 'Int16': libindex.MaskedInt16Engine, 'Int8': libindex.MaskedInt8Engine, 'boolean': libindex.MaskedBoolEngine, 'double[pyarrow]': libindex.MaskedFloat64Engine, 'float64[pyarrow]': libindex.MaskedFloat64Engine, 'float32[pyarrow]': libindex.MaskedFloat32Engine, 'float[pyarrow]': libindex.MaskedFloat32Engine, 'uint64[pyarrow]': libindex.MaskedUInt64Engine, 'uint32[pyarrow]': libindex.MaskedUInt32Engine, 'uint16[pyarrow]': libindex.MaskedUInt16Engine, 'uint8[pyarrow]': libindex.MaskedUInt8Engine, 'int64[pyarrow]': libindex.MaskedInt64Engine, 'int32[pyarrow]': libindex.MaskedInt32Engine, 'int16[pyarrow]': libindex.MaskedInt16Engine, 'int8[pyarrow]': libindex.MaskedInt8Engine, 'bool[pyarrow]': libindex.MaskedBoolEngine}

def _maybe_return_indexers(meth: F) -> F:
    """
    Decorator to simplify 'return_indexers' checks in Index.join.
    """

    @functools.wraps(meth)
    def join(self, other, *, how='left', level=None, return_indexers=False, sort=False):
        join_index, lidx, ridx = meth(self, other, how=how, level=level, sort=sort)
        if not return_indexers:
            return join_index
        if lidx is not None:
            lidx = ensure_platform_int(lidx)
        if ridx is not None:
            ridx = ensure_platform_int(ridx)
        return (join_index, lidx, ridx)
    return cast(F, join)

def _new_Index(cls, d: dict[str, Any]) -> Index:
    """
    This is called upon unpickling, rather than the default which doesn't
    have arguments and breaks __new__.
    """
    if issubclass(cls, ABCPeriodIndex):
        from pandas.core.indexes.period import _new_PeriodIndex
        return _new_PeriodIndex(cls, **d)
    if issubclass(cls, ABCMultiIndex):
        if 'labels' in d and 'codes' not in d:
            d['codes'] = d.pop('labels')
        d['verify_integrity'] = False
    elif 'dtype' not in d and 'data' in d:
        d['dtype'] = d['data'].dtype
    return cls.__new__(cls, **d)

@set_module('pandas')
class Index(IndexOpsMixin, PandasObject):
    """
    Immutable sequence used for indexing and alignment.

    The basic object storing axis labels for all pandas objects.

    .. versionchanged:: 2.0.0

       Index can hold all numpy numeric dtypes (except float16). Previously only
       int64/uint64/float64 dtypes were accepted.

    Parameters
    ----------
    data : array-like (1-dimensional)
        An array-like structure containing the data for the index. This could be a
        Python list, a NumPy array, or a pandas Series.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Index. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.
    copy : bool, default False
        Copy input data.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.

    See Also
    --------
    RangeIndex : Index implementing a monotonic integer range.
    CategoricalIndex : Index of :class:`Categorical` s.
    MultiIndex : A multi-level, or hierarchical Index.
    IntervalIndex : An Index of :class:`Interval` s.
    DatetimeIndex : Index of datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    PeriodIndex : Index of Period data.

    Notes
    -----
    An Index instance can **only** contain hashable objects.
    An Index instance *can not* hold numpy float16 dtype.

    Examples
    --------
    >>> pd.Index([1, 2, 3])
    Index([1, 2, 3], dtype='int64')

    >>> pd.Index(list("abc"))
    Index(['a', 'b', 'c'], dtype='object')

    >>> pd.Index([1, 2, 3], dtype="uint8")
    Index([1, 2, 3], dtype='uint8')
    """
    __pandas_priority__: ClassVar[int] = 2000

    @final
    def _left_indexer_unique(self, other: Index) -> np.ndarray:
        sv = self._get_join_target()
        ov = other._get_join_target()
        return libjoin.left_join_indexer_unique(sv, ov)

    @final
    def _left_indexer(self, other: Index) -> tuple[Index, np.ndarray, np.ndarray]:
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.left_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return (joined, lidx, ridx)

    @final
    def _inner_indexer(self, other: Index) -> tuple[Index, np.ndarray, np.ndarray]:
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.inner_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return (joined, lidx, ridx)

    @final
    def _outer_indexer(self, other: Index) -> tuple[Index, np.ndarray, np.ndarray]:
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.outer_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return (joined, lidx, ridx)
    _typ: ClassVar[str] = 'index'
    _data_cls: ClassVar[tuple[type[np.ndarray], type[ExtensionArray]]] = (np.ndarray, ExtensionArray)
    _id: Any = None
    _name: Any = None
    _no_setting_name: bool = False
    _comparables: ClassVar[list[str]] = ['name']
    _attributes: ClassVar[list[str]] = ['name']

    @cache_readonly
    def _can_hold_strings(self) -> bool:
        return not is_numeric_dtype(self.dtype)
    _engine_types: ClassVar[dict[np.dtype, Any]] = {np.dtype(np.int8): libindex.Int8Engine, np.dtype(np.int16): libindex.Int16Engine, np.dtype(np.int32): libindex.Int32Engine, np.dtype(np.int64): libindex.Int64Engine, np.dtype(np.uint8): libindex.UInt8Engine, np.dtype(np.uint16): libindex.UInt16Engine, np.dtype(np.uint32): libindex.UInt32Engine, np.dtype(np.uint64): libindex.UInt64Engine, np.dtype(np.float32): libindex.Float32Engine, np.dtype(np.float64): libindex.Float64Engine, np.dtype(np.complex64): libindex.Complex64Engine, np.dtype(np.complex128): libindex.Complex128Engine}

    @property
    def _engine_type(self) -> Any:
        return self._engine_types.get(self.dtype, libindex.ObjectEngine)
    _supports_partial_string_indexing: bool = False
    _accessors: ClassVar[set[str]] = {'str'}
    str: Accessor = Accessor('str', StringMethods)
    _references: BlockValuesRefs | None = None

    def __new__(cls, data: Any = None, dtype: Dtype | None = None, copy: bool = False, name: Any = None, tupleize_cols: bool = True) -> Index:
        from pandas.core.indexes.range import RangeIndex
        name = maybe_extract_name(name, data, cls)
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        data_dtype = getattr(data, 'dtype', None)
        refs = None
        if not copy and isinstance(data, (ABCSeries, Index)):
            refs = data._references
        if isinstance(data, (range, RangeIndex)):
            result = RangeIndex(start=data, copy=copy, name=name)
            if dtype is not None:
                return result.astype(dtype, copy=False)
            return result
        elif is_ea_or_datetimelike_dtype(dtype):
            if isinstance(data, (set, frozenset)):
                data = list(data)
        elif is_ea_or_datetimelike_dtype(data_dtype):
            pass
        elif isinstance(data, (np.ndarray, ABCMultiIndex)):
            if isinstance(data, ABCMultiIndex):
                data = data._values
            if data.dtype.kind not in 'iufcbmM':
                data = com.asarray_tuplesafe(data, dtype=_dtype_obj)
        elif isinstance(data, (ABCSeries, Index)):
            pass
        elif is_scalar(data):
            raise cls._raise_scalar_data_error(data)
        elif hasattr(data, '__array__'):
            return cls(np.asarray(data), dtype=dtype, copy=copy, name=name)
        elif not is_list_like(data) and (not isinstance(data, memoryview)):
            raise cls._raise_scalar_data_error(data)
        else:
            if tupleize_cols:
                if is_iterator(data):
                    data = list(data)
                if data and all((isinstance(e, tuple) for e in data)):
                    from pandas.core.indexes.multi import MultiIndex
                    return MultiIndex.from_tuples(data, names=name)
            if not isinstance(data, (list, tuple)):
                data = list(data)
            if len(data) == 0:
                data = np.array(data, dtype=object)
            if len(data) and isinstance(data[0], tuple):
                data = com.asarray_tuplesafe(data, dtype=_dtype_obj)
        try:
            arr = sanitize_array(data, None, dtype=dtype, copy=copy)
        except ValueError as err:
            if 'index must be specified when data is not list-like' in str(err):
                raise cls._raise_scalar_data_error(data) from err
            if 'Data must be 1-dimensional' in str(err):
                raise ValueError('Index data must be 1-dimensional') from err
            raise
        arr = ensure_wrapped_if_datetimelike(arr)
        klass = cls._dtype_to_subclass(arr.dtype)
        arr = klass._ensure_array(arr, arr.dtype, copy=False)
        return klass._simple_new(arr, name, refs=refs)

    @classmethod
    def _ensure_array(cls, data: np.ndarray, dtype: DtypeObj, copy: bool) -> np.ndarray:
        """
        Ensure we have a valid array to pass to _simple_new.
        """
        if data.ndim > 1:
            raise ValueError('Index data must be 1-dimensional')
        elif dtype == np.float16:
            raise NotImplementedError('float16 indexes are not supported')
        if copy:
            data = data.copy()
        return data

    @final
    @classmethod
    def _dtype_to_subclass(cls, dtype: DtypeObj) -> type[Index]:
        if isinstance(dtype, ExtensionDtype):
            return dtype.index_class
        if dtype.kind == 'M':
            from pandas import DatetimeIndex
            return DatetimeIndex
        elif dtype.kind == 'm':
            from pandas import TimedeltaIndex
            return TimedeltaIndex
        elif dtype.kind == 'O':
            return Index
        elif issubclass(dtype.type, str) or is_numeric_dtype(dtype):
            return Index
        raise NotImplementedError(dtype)

    @classmethod
    def _simple_new(cls, values: Any, name: Any | None = None, refs: BlockValuesRefs | None = None) -> Self:
        """
        We require that we have a dtype compat for the values. If we are passed
        a non-dtype compat, then coerce using the constructor.

        Must be careful not to recurse.
        """
        assert isinstance(values, cls._data_cls), type(values)
        result = object.__new__(cls)
        result._data = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        if refs is not None:
            result._references = refs
        else:
            result._references = BlockValuesRefs()
        result._references.add_index_reference(result)
        return result

    @classmethod
    def _with_infer(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Constructor that uses the 1.0.x behavior