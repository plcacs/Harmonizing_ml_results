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

def _maybe_return_indexers(meth):
    """
    Decorator to simplify 'return_indexers' checks in Index.join.
    """
    @functools.wraps(meth)
    def join(self: Self, other: Index, *, how: Literal['left', 'right', 'inner', 'outer'] = 'left', level: AxisInt | Level = no_default, return_indexers: bool = False, sort: bool = False) -> tuple[Index, np.ndarray, np.ndarray]:
        join_index, lidx, ridx = meth(self, other, how=how, level=level, sort=sort)
        if not return_indexers:
            return join_index
        if lidx is not None:
            lidx = ensure_platform_int(lidx)
        if ridx is not None:
            ridx = ensure_platform_int(ridx)
        return (join_index, lidx, ridx)
    return cast(F, join)

def _new_Index(cls: type[Index], d: dict) -> Index:
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
    __pandas_priority__ = 2000

    @final
    def _left_indexer_unique(self: Self, other: Index) -> tuple[Index, np.ndarray, np.ndarray]:
        sv = self._get_join_target()
        ov = other._get_join_target()
        return libjoin.left_join_indexer_unique(sv, ov)

    @final
    def _left_indexer(self: Self, other: Index) -> tuple[Index, np.ndarray, np.ndarray]:
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.left_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return (joined, lidx, ridx)

    @final
    def _inner_indexer(self: Self, other: Index) -> tuple[Index, np.ndarray, np.ndarray]:
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.inner_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return (joined, lidx, ridx)

    @final
    def _outer_indexer(self: Self, other: Index) -> tuple[Index, np.ndarray, np.ndarray]:
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.outer_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return (joined, lidx, ridx)
    _typ = 'index'
    _data_cls = (np.ndarray, ExtensionArray)
    _id = None
    _name = None
    _no_setting_name = False
    _comparables = ['name']
    _attributes = ['name']

    @cache_readonly
    def _can_hold_strings(self: Self) -> bool:
        return not is_numeric_dtype(self.dtype)
    _engine_types = {np.dtype(np.int8): libindex.Int8Engine, np.dtype(np.int16): libindex.Int16Engine, np.dtype(np.int32): libindex.Int32Engine, np.dtype(np.int64): libindex.Int64Engine, np.dtype(np.uint8): libindex.UInt8Engine, np.dtype(np.uint16): libindex.UInt16Engine, np.dtype(np.uint32): libindex.UInt32Engine, np.dtype(np.uint64): libindex.UInt64Engine, np.dtype(np.float32): libindex.Float32Engine, np.dtype(np.float64): libindex.Float64Engine, np.dtype(np.complex64): libindex.Complex64Engine, np.dtype(np.complex128): libindex.Complex128Engine}

    @property
    def _engine_type(self: Self) -> libindex.IndexEngine:
        return self._engine_types.get(self.dtype, libindex.ObjectEngine)
    _supports_partial_string_indexing = False
    _accessors = {'str'}
    str = Accessor('str', StringMethods)
    _references = None

    def __new__(cls: type[Index], data: ArrayLike, dtype: DtypeObj | None = None, copy: bool = False, name: IndexLabel | None = None, tupleize_cols: bool = True) -> Index:
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
    def _ensure_array(cls: type[Index], data: np.ndarray, dtype: DtypeObj, copy: bool) -> np.ndarray:
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
    def _dtype_to_subclass(cls: type[Index], dtype: DtypeObj) -> type[Index]:
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
    def _simple_new(cls: type[Index], values: np.ndarray, name: IndexLabel | None, refs: BlockValuesRefs | None) -> Index:
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
    def _with_infer(cls: type[Index], *args, **kwargs) -> Index:
        """
        Constructor that uses the 1.0.x behavior inferring numeric dtypes
        for ndarray[object] inputs.
        """
        result = cls(*args, **kwargs)
        if result.dtype == _dtype_obj and (not result._is_multi):
            values = lib.maybe_convert_objects(result._values)
            if values.dtype.kind in 'iufb':
                return Index(values, name=result.name)
        return result

    @cache_readonly
    def _constructor(self: Self) -> type[Index]:
        return type(self)

    @final
    def _maybe_check_unique(self: Self) -> None:
        """
        Check that an Index has no duplicates.

        This is typically only called via
        `NDFrame.flags.allows_duplicate_labels.setter` when it's set to
        True (duplicates aren't allowed).

        Raises
        ------
        DuplicateLabelError
            When the index is not unique.
        """
        if not self.is_unique:
            msg = 'Index has duplicates.'
            duplicates = self._format_duplicate_message()
            msg += f'\n{duplicates}'
            raise DuplicateLabelError(msg)

    @final
    def _format_duplicate_message(self: Self) -> str:
        """
        Construct the DataFrame for a DuplicateLabelError.

        This returns a DataFrame indicating the labels and positions
        of duplicates in an index. This should only be called when it's
        already known that duplicates are present.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "a"])
        >>> idx._format_duplicate_message()
            positions
        label
        a        [0, 2]
        """
        from pandas import Series
        duplicates = self[self.duplicated(keep='first')].unique()
        assert len(duplicates)
        out = Series(np.arange(len(self)), copy=False).groupby(self, observed=False).agg(list)[duplicates]
        if self._is_multi:
            out.index = type(self).from_tuples(out.index)
        if self.nlevels == 1:
            out = out.rename_axis('label')
        return out.to_frame(name='positions')

    def _shallow_copy(self: Self, values: np.ndarray, name: IndexLabel | None = no_default) -> Index:
        """
        Create a new Index with the same class as the caller, don't copy the
        data, use the same object attributes with passed in attributes taking
        precedence.

        *this is an internal non-public method*

        Parameters
        ----------
        values : the values to create the new Index, optional
        name : Label, defaults to self.name
        """
        name = self._name if name is no_default else name
        return self._simple_new(values, name=name, refs=self._references)

    def _view(self: Self) -> Index:
        """
        fastpath to make a shallow copy, i.e. new object with same data.
        """
        result = self._simple_new(self._values, name=self._name, refs=self._references)
        result._cache = self._cache
        return result

    @final
    def _rename(self: Self, name: IndexLabel) -> Index:
        """
        fastpath for rename if new name is already validated.
        """
        result = self._view()
        result._name = name
        return result

    @final
    def is_(self: Self, other: object) -> bool:
        """
        More flexible, faster check like ``is`` but that works through views.

        Note: this is *not* the same as ``Index.identical()``, which checks
        that metadata is also the same.

        Parameters
        ----------
        other : object
            Other object to compare against.

        Returns
        -------
        bool
            True if both have same underlying data, False otherwise.

        See Also
        --------
        Index.identical : Works like ``Index.is_`` but also checks metadata.

        Examples
        --------
        >>> idx1 = pd.Index(["1", "2", "3"])
        >>> idx1.is_(idx1.view())
        True

        >>> idx1.is_(idx1.copy())
        False
        """
        if self is other:
            return True
        elif not hasattr(other, '_id'):
            return False
        elif self._id is None or other._id is None:
            return False
        else:
            return self._id is other._id

    @final
    def _reset_identity(self: Self) -> None:
        """
        Initializes or resets ``_id`` attribute with new object.
        """
        self._id = object()

    @final
    def _cleanup(self: Self) -> None:
        if '_engine' in self._cache:
            self._engine.clear_mapping()

    @cache_readonly
    def _engine(self: Self) -> libindex.IndexEngine:
        target_values = self._get_engine_target()
        if isinstance(self._values, ArrowExtensionArray) and self.dtype.kind in 'Mm':
            import pyarrow as pa
            pa_type = self._values._pa_array.type
            if pa.types.is_timestamp(pa_type):
                target_values = self._values._to_datetimearray()
                return libindex.DatetimeEngine(target_values._ndarray)
            elif pa.types.is_duration(pa_type):
                target_values = self._values._to_timedeltaarray()
                return libindex.TimedeltaEngine(target_values._ndarray)
        if isinstance(target_values, ExtensionArray):
            if isinstance(target_values, (BaseMaskedArray, ArrowExtensionArray)):
                try:
                    return _masked_engines[target_values.dtype.name](target_values)
                except KeyError:
                    pass
            elif self._engine_type is libindex.ObjectEngine:
                return libindex.ExtensionEngine(target_values)
        target_values = cast(np.ndarray, target_values)
        if target_values.dtype == bool:
            return libindex.BoolEngine(target_values)
        elif target_values.dtype == np.complex64:
            return libindex.Complex64Engine(target_values)
        elif target_values.dtype == np.complex128:
            return libindex.Complex128Engine(target_values)
        elif needs_i8_conversion(self.dtype):
            target_values = self._data._ndarray
        elif is_string_dtype(self.dtype) and (not is_object_dtype(self.dtype)):
            return libindex.StringObjectEngine(target_values, self.dtype.na_value)
        return self._engine_type(target_values)

    @cache_readonly
    def _dir_additions_for_owner(self: Self) -> set[str]:
        """
        Add the string-like labels to the owner dataframe/series dir output.

        If this is a MultiIndex, it's first level values are used.
        """
        return {c for c in self.unique(level=0)[:get_option('display.max_dir_items')] if isinstance(c, str) and c.isidentifier()}

    def __len__(self: Self) -> int:
        """
        Return the length of the Index.
        """
        return len(self._data)

    def __array__(self: Self, dtype: DtypeObj | None = None, copy: bool | None = None) -> np.ndarray:
        """
        The array interface, return my values.
        """
        if copy is None:
            return np.asarray(self._data, dtype=dtype)
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self: Self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if any((isinstance(other, (ABCSeries, ABCDataFrame)) for other in inputs)):
            return NotImplemented
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
        if 'out' in kwargs:
            return arraylike.dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        if method == 'reduce':
            result = arraylike.dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
            if result is not NotImplemented:
                return result
        new_inputs = [x if x is not self else x._values for x in inputs]
        result = getattr(ufunc, method)(*new_inputs, **kwargs)
        if ufunc.nout == 2:
            return tuple((self.__array_wrap__(x) for x in result))
        elif method == 'reduce':
            result = lib.item_from_zerodim(result)
            return result
        elif is_scalar(result):
            return result
        if result.dtype == np.float16:
            result = result.astype(np.float32)
        return self.__array_wrap__(result)

    @final
    def __array_wrap__(self: Self, result: Any, context: Any | None = None, return_scalar: bool = False) -> Index:
        """
        Gets called after a ufunc and other functions e.g. np.split.
        """
        result = lib.item_from_zerodim(result)
        if not isinstance(result, Index) and is_bool_dtype(result.dtype) or np.ndim(result) > 1:
            return result
        return Index(result, name=self.name)

    @cache_readonly
    def dtype(self: Self) -> np.dtype:
        """
        Return the dtype object of the underlying data.

        See Also
        --------
        Index.inferred_type: Return a string of the type inferred from the values.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.dtype
        dtype('int64')
        """
        return self._data.dtype

    @final
    def ravel(self: Self) -> Index:
        """
        Return a view on self.

        Parameters
        ----------
        order : {'K', 'A', 'C', 'F'}, default 'C'
            Specify the memory layout of the view. This parameter is not
            implemented currently.

        Returns
        -------
        Index
            A view on self.

        See Also
        --------
        numpy.ndarray.ravel : Return a flattened array.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=["a", "b", "c"])
        >>> s.index.ravel()
        Index(['a', 'b', 'c'], dtype='object')
        """
        return self[:]

    def view(self: Self, cls: type[np.ndarray] | None = None) -> Index:
        """
        Return a view of the Index with the specified dtype or a new Index instance.

        This method returns a view of the calling Index object if no arguments are
        provided. If a dtype is specified through the `cls` argument, it attempts
        to return a view of the Index with the specified dtype. Note that viewing
        the Index as a different dtype reinterprets the underlying data, which can
        lead to unexpected results for non-numeric or incompatible dtype conversions.

        Parameters
        ----------
        cls : data-type or ndarray sub-class, optional
            Data-type descriptor of the returned view, e.g., float32 or int16.
            Omitting it results in the view having the same data-type as `self`.
            This argument can also be specified as an ndarray sub-class,
            e.g., np.int64 or np.float32 which then specifies the type of
            the returned object.

        Returns
        -------
        Index or ndarray
            A view of the Index. If `cls` is None, the returned object is an Index
            view with the same dtype as the calling object. If a numeric `cls` is
            specified an ndarray view with the new dtype is returned.

        Raises
        ------
        ValueError
            If attempting to change to a dtype in a way that is not compatible with
            the original dtype's memory layout, for example, viewing an 'int64' Index
            as 'str'.

        See Also
        --------
        Index.copy : Returns a copy of the Index.
        numpy.ndarray.view : Returns a new view of array with the same data.

        Examples
        --------
        >>> idx = pd.Index([-1, 0, 1])
        >>> idx.view()
        Index([-1, 0, 1], dtype='int64')

        >>> idx.view(np.uint64)
        array([18446744073709551615,                    0,                    1],
          dtype=uint64)

        Viewing as 'int32' or 'float32' reinterprets the memory, which may lead to
        unexpected behavior:

        >>> idx.view("float32")
        array([   nan,    nan, 0.e+00, 0.e+00, 1.e-45, 0.e+00], dtype=float32)
        """
        if cls is not None:
            dtype = cls
            if isinstance(cls, str):
                dtype = pandas_dtype(cls)
            if needs_i8_conversion(dtype):
                idx_cls = self._dtype_to_subclass(dtype)
                arr = self.array.view(dtype)
                if isinstance(arr, ExtensionArray):
                    return idx_cls._simple_new(arr, name=self.name, refs=self._references)
                return arr
            result = self._data.view(cls)
        else:
            result = self._view()
        if isinstance(result, Index):
            result._id = self._id
        return result

    def astype(self: Self, dtype: DtypeObj | None = None, copy: bool = True) -> Index:
        """
        Create an Index with values cast to dtypes.

        The class of a new Index is determined by dtype. When conversion is
        impossible, a TypeError exception is raised.

        Parameters
        ----------
        dtype : numpy dtype or pandas type
            Note that any signed integer `dtype` is treated as ``'int64'``,
            and any unsigned integer `dtype` is treated as ``'uint64'``,
            regardless of the size.
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

        Returns
        -------
        Index
           NA/NaN values replaced with `value`.

        See Also
        --------
        Index.dtype: Return the dtype object of the underlying data.
        Index.dtypes: Return the dtype object of the underlying data.
        Index.convert_dtypes: Convert columns to the best possible dtypes.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.astype("float")
        Index([1.0, 2.0, 3.0], dtype='float64')
        """
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if self.dtype == dtype:
            return self.copy() if copy else self
        values = self._data
        if isinstance(values, ExtensionArray):
            with rewrite_exception(type(values).__name__, type(self).__name__):
                new_values = values.astype(dtype, copy=copy)
        elif isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            new_values = cls._from_sequence(self, dtype=dtype, copy=copy)
        else:
            new_values = astype_array(values, dtype=dtype, copy=copy)
        result = Index(new_values, name=self.name, dtype=new_values.dtype, copy=False)
        if not copy and self._references is not None and astype_is_view(self.dtype, dtype):
            result._references = self._references
            result._references.add_index_reference(result)
        return result
    _index_shared_docs['take'] = "\n        Return a new %(klass)s of the values selected by the indices.\n\n        For internal compatibility with numpy arrays.\n\n        Parameters\n        ----------\n        indices : array-like\n            Indices to be taken.\n        axis : int, optional\n            The axis over which to select values, always 0.\n        allow_fill : bool, default True\n            How to handle negative values in `indices`.\n\n            * False: negative values in `indices` indicate positional indices\n              from the right (the default). This is similar to\n              :func:`numpy.take`.\n\n            * True: negative values in `indices` indicate\n              missing values. These values are set to `fill_value`. Any other\n              other negative values raise a ``ValueError``.\n\n        fill_value : scalar, default None\n            If allow_fill=True and fill_value is not None, indices specified by\n            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.\n        **kwargs\n            Required for compatibility with numpy.\n\n        Returns\n        -------\n        Index\n            An index formed of elements at the given indices. Will be the same\n            type as self, except for RangeIndex.\n\n        See Also\n        --------\n        numpy.ndarray.take: Return an array formed from the\n            elements of a at the given indices.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['a', 'b', 'c'])\n        >>> idx.take([2, 2, 1, 2])\n        Index(['c', 'c', 'b', 'c'], dtype='object')\n        "

    @Appender(_index_shared_docs['take'] % _index_doc_kwargs)
    def take(self: Self, indices: ArrayLike, axis: AxisInt = 0, allow_fill: bool = True, fill_value: Any | None = None, **kwargs: Any) -> Index:
        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError('Expected indices to be array-like')
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)
        if indices.ndim == 1 and lib.is_range_indexer(indices, len(self)):
            return self.copy()
        values = self._values
        if isinstance(values, np.ndarray):
            taken = algos.take(values, indices, allow_fill=allow_fill, fill_value=self._na_value)
        else:
            taken = values.take(indices, allow_fill=allow_fill, fill_value=self._na_value)
        return self._constructor._simple_new(taken, name=self.name)

    @final
    def _maybe_disallow_fill(self: Self, allow_fill: bool, fill_value: Any | None, indices: ArrayLike) -> bool:
        """
        We only use pandas-style take when allow_fill is True _and_
        fill_value is not None.
        """
        if allow_fill and fill_value is not None:
            if self._can_hold_na:
                if (indices < -1).any():
                    raise ValueError('When allow_fill=True and fill_value is not None, all indices must be >= -1')
            else:
                cls_name = type(self).__name__
                raise ValueError(f'Unable to fill values because {cls_name} cannot contain NA')
        else:
            allow_fill = False
        return allow_fill
    _index_shared_docs['repeat'] = "\n        Repeat elements of a %(klass)s.\n\n        Returns a new %(klass)s where each element of the current %(klass)s\n        is repeated consecutively a given number of times.\n\n        Parameters\n        ----------\n        repeats : int or array of ints\n            The number of repetitions for each element. This should be a\n            non-negative integer. Repeating 0 times will return an empty\n            %(klass)s.\n        axis : None\n            Must be ``None``. Has no effect but is accepted for compatibility\n            with numpy.\n\n        Returns\n        -------\n        %(klass)s\n            Newly created %(klass)s with repeated elements.\n\n        See Also\n        --------\n        Series.repeat : Equivalent function for Series.\n        numpy.repeat : Similar method for :class:`numpy.ndarray`.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['a', 'b', 'c'])\n        >>> idx\n        Index(['a', 'b', 'c'], dtype='object')\n        >>> idx.repeat(2)\n        Index(['a', 'a', 'b', 'b', 'c', 'c'], dtype='object')\n        >>> idx.repeat([1, 2, 3])\n        Index(['a', 'b', 'b', 'c', 'c', 'c'], dtype='object')\n        "

    @Appender(_index_shared_docs['repeat'] % _index_doc_kwargs)
    def repeat(self: Self, repeats: ArrayLike, axis: AxisInt | None = None) -> Index:
        repeats = ensure_platform_int(repeats)
        nv.validate_repeat((), {'axis': axis})
        res_values = self._values.repeat(repeats)
        return self._constructor._simple_new(res_values, name=self.name)

    def copy(self: Self, name: IndexLabel | None = None, deep: bool = False) -> Index:
        """
        Make a copy of this object.

        Name is set on the new object.

        Parameters
        ----------
        name : Label, optional
            Set name for new object.
        deep : bool, default False
            If True attempts to make a deep copy of the Index.
                Else makes a shallow copy.

        Returns
        -------
        Index
            Index refer to new object which is a copy of this object.

        See Also
        --------
        Index.delete: Make new Index with passed location(-s) deleted.
        Index.drop: Make new Index with passed list of labels deleted.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> new_idx = idx.copy()
        >>> idx is new_idx
        False
        """
        name = self._validate_names(name=name, deep=deep)[0]
        if deep:
            new_data = self._data.copy()
            new_index = type(self)._simple_new(new_data, name=name)
        else:
            new_index = self._rename(name=name)
        return new_index

    @final
    def __copy__(self: Self, **kwargs: Any) -> Index:
        return self.copy(**kwargs)

    @final
    def __deepcopy__(self: Self, memo: dict | None = None) -> Index:
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        return self.copy(deep=True)

    @final
    def __repr__(self: Self) -> str:
        """
        Return a string representation for this object.
        """
        klass_name = type(self).__name__
        data = self._format_data()
        attrs = self._format_attrs()
        attrs_str = [f'{k}={v}' for k, v in attrs]
        prepr = ', '.join(attrs_str)
        return f'{klass_name}({data}{prepr})'

    @property
    def _formatter_func(self: Self) -> Callable[[Any], str]:
        """
        Return the formatter function.
        """
        return default_pprint

    @final
    def _format_data(self: Self, name: IndexLabel | None = None) -> str:
        """
        Return the formatted data as a unicode string.
        """
        is_justify = True
        if self.inferred_type == 'string':
            is_justify = False
        elif isinstance(self.dtype, CategoricalDtype):
            self = cast('CategoricalIndex', self)
            if is_object_dtype(self.categories.dtype):
                is_justify = False
        elif isinstance(self, ABCRangeIndex):
            return ''
        return format_object_summary(self, self._formatter_func, is_justify=is_justify, name=name, line_break_each_value=self._is_multi)

    def _format_attrs(self: Self) -> list[tuple[str, str]]:
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs = []
        if not self._is_multi:
            attrs.append(('dtype', f"'{self.dtype}'"))
        if self.name is not None:
            attrs.append(('name', default_pprint(self.name)))
        elif self._is_multi and any((x is not None for x in self.names)):
            attrs.append(('names', default_pprint(self.names)))
        max_seq_items = get_option('display.max_seq_items') or len(self)
        if len(self) > max_seq_items:
            attrs.append(('length', len(self)))
        return attrs

    @final
    def _get_level_names(self: Self) -> list[Level | None]:
        """
        Return a name or list of names with None replaced by the level number.
        """
        if self._is_multi:
            return maybe_sequence_to_range([level if name is None else name for level, name in enumerate(self.names)])
        else:
            return range(1) if self.name is None else [self.name]

    @final
    def _mpl_repr(self: Self) -> np.ndarray:
        if isinstance(self.dtype, np.dtype) and self.dtype.kind != 'M':
            return cast(np.ndarray, self.values)
        return self.astype(object, copy=False)._values
    _default_na_rep = 'NaN'

    @final
    def _format_flat(self: Self, *, include_name: bool = True, formatter: Callable[[Any], str] | None = None) -> list[str]:
        """
        Render a string representation of the Index.
        """
        header = []
        if include_name:
            header.append(pprint_thing(self.name, escape_chars=('\t', '\r', '\n')) if self.name is not None else '')
        if formatter is not None:
            return header + list(self.map(formatter))
        return self._format_with_header(header=header, na_rep=self._default_na_rep)

    def _format_with_header(self: Self, *, header: list[str], na_rep: str) -> list[str]:
        from pandas.io.formats.format import format_array
        values = self._values
        if is_object_dtype(values.dtype) or is_string_dtype(values.dtype) or isinstance(self.dtype, (IntervalDtype, CategoricalDtype)):
            justify = 'all'
        else:
            justify = 'left'
        formatted = format_array(values, None, justify=justify)
        result = trim_front(formatted)
        return header + result

    def _get_values_for_csv(self: Self, *, na_rep: str = 'nan', decimal: str = '.', float_format: str | None = None, date_format: str | None = None, quoting: str | None = None) -> np.ndarray:
        return get_values_for_csv(self._values, na_rep=na_rep, decimal=decimal, float_format=float_format, date_format=date_format, quoting=quoting)

    def _summary(self: Self, name: IndexLabel | None = None) -> str:
        """
        Return a summarized representation.

        Parameters
        ----------
        name : str
            name to use in the summary representation

        Returns
        -------
        String with a summarized representation of the index
        """
        if len(self) > 0:
            head = self[0]
            if hasattr(head, 'format') and (not isinstance(head, str)):
                head = head.format()
            elif needs_i8_conversion(self.dtype):
                head = self._formatter_func(head).replace("'", '')
            tail = self[-1]
            if hasattr(tail, 'format') and (not isinstance(tail, str)):
                tail = tail.format()
            elif needs_i8_conversion(self.dtype):
                tail = self._formatter_func(tail).replace("'", '')
            index_summary = f', {head} to {tail}'
        else:
            index_summary = ''
        if name is None:
            name = type(self).__name__
        return f'{name}: {len(self)} entries{index_summary}'

    def to_flat_index(self: Self) -> Index:
        """
        Identity method.

        This is implemented for compatibility with subclass implementations
        when chaining.

        Returns
        -------
        pd.Index
            Caller.

        See Also
        --------
        MultiIndex.to_flat_index : Subclass implementation.
        """
        return self

    @final
    def to_series(self: Self, index: Index | None = None, name: IndexLabel | None = None) -> Series:
        """
        Create a Series with both index and values equal to the index keys.

        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Name of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.

        See Also
        --------
        Index.to_frame : Convert an Index to a DataFrame.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(["Ant", "Bear", "Cow"], name="animal")

        By default, the original index and original name is reused.

        >>> idx.to_series()
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: animal, dtype: object

        To enforce a new index, specify new labels to ``index``:

        >>> idx.to_series(index=[0, 1, 2])
        0     Ant
        1    Bear
        2     Cow
        Name: animal, dtype: object

        To override the name of the resulting column, specify ``name``:

        >>> idx.to_series(name="zoo")
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: zoo, dtype: object
        """
        from pandas import Series
        if index is None:
            index = self._view()
        if name is None:
            name = self.name
        return Series(self._values.copy(), index=index, name=name)

    def to_frame(self: Self, index: bool = True, name: IndexLabel | None = lib.no_default) -> DataFrame:
        """
        Create a DataFrame with a column containing the Index.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original Index.

        name : object, defaults to index.name
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(["Ant", "Bear", "Cow"], name="animal")
        >>> idx.to_frame()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
            animal
        0   Ant
        1  Bear
        2   Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name="zoo")
            zoo
        0   Ant
        1  Bear
        2   Cow
        """
        from pandas import DataFrame
        if name is lib.no_default:
            result_name = self._get_level_names()
        else:
            result_name = Index([name])
        result = DataFrame(self, copy=False)
        result.columns = result_name
        if index:
            result.index = self
        return result

    @property
    def name(self: Self) -> IndexLabel | None:
        """
        Return Index or MultiIndex name.

        See Also
        --------
        Index.set_names: Able to set new names partially and by level.
        Index.rename: Able to set new names partially and by level.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3], name="x")
        >>> idx
        Index([1, 2, 3], dtype='int64', name='x')
        >>> idx.name
        'x'
        """
        return self._name

    @name.setter
    def name(self: Self, value: IndexLabel) -> None:
        if self._no_setting_name:
            raise RuntimeError("Cannot set name on a level of a MultiIndex. Use 'MultiIndex.set_names' instead.")
        maybe_extract_name(value, None, type(self))
        self._name = value

    @final
    def _validate_names(self: Self, name: IndexLabel | None = None, names: list[IndexLabel] | None = None, deep: bool = False) -> tuple[IndexLabel, list[IndexLabel]]:
        """
        Handles the quirks of having a singular 'name' parameter for general
        Index and plural 'names' parameter for MultiIndex.
        """
        from copy import deepcopy
        if names is not None and name is not None:
            raise TypeError('Can only provide one of `names` and `name`')
        if names is None and name is None:
            new_names = deepcopy(self.names) if deep else self.names
        elif names is not None:
            if not is_list_like(names):
                raise TypeError('Names must be a list-like')
            new_names = names
        else:
            new_names = name
        if len(new_names) != len(self.names):
            raise ValueError(f'Length of new names must be {len(self.names)}, got {len(new_names)}')
        validate_all_hashable(*new_names, error_name=f'{type(self).__name__}.name')
        return new_names

    def _get_default_index_names(self: Self, names: IndexLabel | None = None, default: str | None = None) -> list[IndexLabel]:
        """
        Get names of index.

        Parameters
        ----------
        names : int, str or 1-dimensional list, default None
            Index names to set.
        default : str
            Default name of index.

        Raises
        ------
        TypeError
            if names not str or list-like
        """
        from pandas.core.indexes.multi import MultiIndex
        if names is not None:
            if isinstance(names, (int, str)):
                names = [names]
        if not isinstance(names, list) and names is not None:
            raise ValueError('Index names must be str or 1-dimensional list')
        if not names:
            if isinstance(self, MultiIndex):
                names = com.fill_missing_names(self.names)
            else:
                names = [default] if self.name is None else [self.name]
        return names

    def _get_names(self: Self) -> FrozenList[IndexLabel]:
        """
        Get names on index.

        This method returns a FrozenList containing the names of the object.
        It's primarily intended for internal use.

        Returns
        -------
        FrozenList
            A FrozenList containing the object's names, contains None if the object
            does not have a name.

        See Also
        --------
        Index.name : Index name as a string, or None for MultiIndex.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3], name="x")
        >>> idx.names
        FrozenList(['x'])

        >>> idx = pd.Index([1, 2, 3], name=("x", "y"))
        >>> idx.names
        FrozenList([('x', 'y')])

        If the index does not have a name set:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx.names
        FrozenList([None])
        """
        return FrozenList((self.name,))

    def _set_names(self: Self, values: list[IndexLabel], *, level: AxisInt | None = None) -> None:
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None

        Raises
        ------
        TypeError if each name is not hashable.
        """
        if not is_list_like(values):
            raise ValueError('Names must be a list-like')
        if len(values) != 1:
            raise ValueError(f'Length of new names must be 1, got {len(values)}')
        validate_all_hashable(*values, error_name=f'{type(self).__name__}.name')
        self._name = values[0]
    names = property(fset=_set_names, fget=_get_names)

    @overload
    def set_names(self: Self, names: list[IndexLabel], *, level: AxisInt | None = ..., inplace: bool = ...) -> Index | None:
        ...

    @overload
    def set_names(self: Self, names: list[IndexLabel], *, level: AxisInt | None, inplace: bool) -> Index | None:
        ...

    @overload
    def set_names(self: Self, names: list[IndexLabel], *, level: AxisInt | None, inplace: bool = ...) -> Index | None:
        ...

    def set_names(self: Self, names: list[IndexLabel], *, level: AxisInt | None = None, inplace: bool = False) -> Index | None:
        """
        Set Index or MultiIndex name.

        Able to set new names partially and by level.

        Parameters
        ----------

        names : label or list of label or dict-like for MultiIndex
            Name(s) to set.

            .. versionchanged:: 1.3.0

        level : int, label or list of int or label, optional
            If the index is a MultiIndex and names is not dict-like, level(s) to set
            (None for all levels). Otherwise level must be None.

            .. versionchanged:: 1.3.0

        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.rename : Able to set new names without level.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names("quarter")
        Index([1, 2, 3, 4], dtype='int64', name='quarter')

        >>> idx = pd.MultiIndex.from_product([["python", "cobra"], [2018, 2019]])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   )
        >>> idx = idx.set_names(["kind", "year"])
        >>> idx.set_names("species", level=0)
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])

        When renaming levels with a dict, levels can not be passed.

        >>> idx.set_names({"kind": "snake"})
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['snake', 'year'])
        """
        if level is not None and (not isinstance(self, ABCMultiIndex)):
            raise ValueError('Level must be None for non-MultiIndex')
        if level is not None and (not is_list_like(level)) and is_list_like(names):
            raise TypeError('Names must be a string when a single level is provided.')
        if not is_list_like(names) and level is None and (self.nlevels > 1):
            raise TypeError('Must pass list-like as `names`.')
        if is_dict_like(names) and (not isinstance(self, ABCMultiIndex)):
            raise TypeError('Can only pass dict-like as `names` for MultiIndex.')
        if is_dict_like(names) and level is not None:
            raise TypeError('Can not pass level for dictlike `names`.')
        if isinstance(self, ABCMultiIndex) and is_dict_like(names) and (level is None):
            level, names_adjusted = ([], [])
            for i, name in enumerate(self.names):
                if name in names.keys():
                    level.append(i)
                    names_adjusted.append(names[name])
            names = names_adjusted
        if not is_list_like(names):
            names = [names]
        if level is not None and (not is_list_like(level)):
            level = [level]
        if inplace:
            idx = self
        else:
            idx = self._view()
        idx._set_names(names, level=level)
        if not inplace:
            return idx
        return None

    @overload
    def rename(self: Self, name: IndexLabel, *, inplace: bool = ...) -> Index | None:
        ...

    @overload
    def rename(self: Self, name: IndexLabel, *, inplace: bool) -> Index | None:
        ...

    def rename(self: Self, name: IndexLabel, *, inplace: bool = False) -> Index | None:
        """
        Alter Index or MultiIndex name.

        Able to set new names without level. Defaults to returning new index.
        Length of names must match number of levels in MultiIndex.

        Parameters
        ----------
        name : label or list of labels
            Name(s) to set.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.set_names : Able to set new names partially and by level.

        Examples
        --------
        >>> idx = pd.Index(["A", "C", "A", "B"], name="score")
        >>> idx.rename("grade")
        Index(['A', 'C', 'A', 'B'], dtype='object', name='grade')

        >>> idx = pd.MultiIndex.from_product(
        ...     [["python", "cobra"], [2018, 2019]], names=["kind", "year"]
        ... )
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['kind', 'year'])
        >>> idx.rename(["species", "year"])
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])
        >>> idx.rename("species")
        Traceback (most recent call last):
        TypeError: Must pass list-like as `names`.
        """
        return self.set_names([name], inplace=inplace)

    @property
    def nlevels(self: Self) -> int:
        """
        Number of levels.
        """
        return 1

    def _sort_levels_monotonic(self: Self) -> Self:
        """
        Compat with MultiIndex.
        """
        return self

    @final
    def _validate_index_level(self: Self, level: AxisInt | IndexLabel) -> None:
        """
        Validate index level.

        For single-level Index getting level number is a no-op, but some
        verification must be done like in MultiIndex.

        """
        if isinstance(level, int):
            if level < 0 and level != -1:
                raise IndexError(f'Too many levels: Index has only 1 level, {level} is not a valid level number')
            if level > 0:
                raise IndexError(f'Too many levels: Index has only 1 level, not {level + 1}')
        elif level != self.name:
            raise KeyError(f'Requested level ({level}) does not match index name ({self.name})')

    def _get_level_number(self: Self, level: AxisInt | IndexLabel) -> int:
        self._validate_index_level(level)
        return 0

    def sortlevel(self: Self, level: AxisInt | IndexLabel = None, ascending: bool = True, sort_remaining: bool | None = None, na_position: str = 'first') -> Self:
        """
        For internal compatibility with the Index API.

        Sort the Index. This is for compat with MultiIndex

        Parameters
        ----------
        ascending : bool, default True
            False to sort in descending order
        na_position : {'first' or 'last'}, default 'first'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.

            .. versionadded:: 2.1.0

        level, sort_remaining are compat parameters

        Returns
        -------
        Index
        """
        if not isinstance(ascending, (list, bool)):
            raise TypeError('ascending must be a single bool value ora list of bool values of length 1')
        if isinstance(ascending, list):
            if len(ascending) != 1:
                raise TypeError('ascending must be a list of bool values of length 1')
            ascending = ascending[0]
        if not isinstance(ascending, bool):
            raise TypeError('ascending must be a bool value')
        return self.sort_values(return_indexer=True, ascending=ascending, na_position=na_position)

    def _get_level_values(self: Self, level: AxisInt | IndexLabel) -> Self:
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatibility.

        Parameters
        ----------
        level : int or str
            It is either the integer position or the name of the level.

        Returns
        -------
        Index
            Calling object, as there is only one level in the Index.

        See Also
        --------
        MultiIndex.get_level_values : Get values for a level of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------
        >>> idx = pd.Index(list("abc"))
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')

        Get level values by supplying `level` as integer:

        >>> idx.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object')
        """
        self._validate_index_level(level)
        return self
    get_level_values = _get_level_values

    @final
    def droplevel(self: Self, level: AxisInt | IndexLabel = 0) -> Self:
        """
        Return index with requested level(s) removed.

        If resulting index has only 1 level left, the result will be
        of Index type, not MultiIndex. The original index is not modified inplace.

        Parameters
        ----------
        level : int, str, or list-like, default 0
            If a string is given, must be the name of a level
            If list-like, elements must be names or indexes of levels.

        Returns
        -------
        Index or MultiIndex
            Returns an Index or MultiIndex object, depending on the resulting index
            after removing the requested level(s).

        See Also
        --------
        Index.dropna : Return Index without NA/NaN values.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ...     [[1, 2], [3, 4], [5, 6]], names=["x", "y", "z"]
        ... )
        >>> mi
        MultiIndex([(1, 3, 5),
                    (2, 4, 6)],
                   names=['x', 'y', 'z'])

        >>> mi.droplevel()
        MultiIndex([(3, 5),
                    (4, 6)],
                   names=['y', 'z'])

        >>> mi.droplevel(2)
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel("z")
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel(["x", "y"])
        Index([5, 6], dtype='int64', name='z')
        """
        if not isinstance(level, (tuple, list)):
            level = [level]
        levnums = sorted((self._get_level_number(lev) for lev in level), reverse=True)
        return self._drop_level_numbers(levnums)

    @final
    def _drop_level_numbers(self: Self, levnums: list[int]) -> Self:
        """
        Drop MultiIndex levels by level _number_, not name.
        """
        if not levnums and (not isinstance(self, ABCMultiIndex)):
            return self
        if len(levnums) >= self.nlevels:
            raise ValueError(f'Cannot remove {len(levnums)} levels from an index with {self.nlevels} levels: at least one level must be left.')
        self = cast('MultiIndex', self)
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)
        for i in levnums:
            new_levels.pop(i)
            new_codes.pop(i)
            new_names.pop(i)
        if len(new_levels) == 1:
            lev = new_levels[0]
            if len(lev) == 0:
                if len(new_codes[0]) == 0:
                    result = lev[:0]
                else:
                    res_values = algos.take(lev._values, new_codes[0], allow_fill=True)
                    result = lev._constructor._simple_new(res_values, name=new_names[0])
            else:
                mask = new_codes[0] == -1
                result = new_levels[0].take(new_codes[0])
                if mask.any():
                    result = result.putmask(mask, np.nan)
                result._name = new_names[0]
            return result
        else:
            from pandas.core.indexes.multi import MultiIndex
            return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    @cache_readonly
    @final
    def _can_hold_na(self: Self) -> bool:
        if isinstance(self.dtype, ExtensionDtype):
            return self.dtype._can_hold_na
        if self.dtype.kind in 'iub':
            return False
        return True

    @property
    def is_monotonic_increasing(self: Self) -> bool:
        """
        Return a boolean if the values are equal or increasing.

        Returns
        -------
        bool

        See Also
        --------
        Index.is_monotonic_decreasing : Check if the values are equal or decreasing.

        Examples
        --------
        >>> pd.Index([1, 2, 3]).is_monotonic_increasing
        True
        >>> pd.Index([1, 2, 2]).is_monotonic_increasing
        True
        >>> pd.Index([1, 3, 2]).is_monotonic_increasing
        False
        """
        return self._engine.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self: Self) -> bool:
        """
        Return a boolean if the values are equal or decreasing.

        Returns
        -------
        bool

        See Also
        --------
        Index.is_monotonic_increasing : Check if the values are equal or increasing.

        Examples
        --------
        >>> pd.Index([3, 2, 1]).is_monotonic_decreasing
        True
        >>> pd.Index([3, 2, 2]).is_monotonic_decreasing
        True
        >>> pd.Index([3, 1, 2]).is_monotonic_decreasing
        False
        """
        return self._engine.is_monotonic_decreasing

    @final
    @property
    def _is_strictly_monotonic_increasing(self: Self) -> bool:
        """
        Return if the index is strictly monotonic increasing
        (only increasing) values.

        Examples
        --------
        >>> Index([1, 2, 3])._is_strictly_monotonic_increasing
        True
        >>> Index([1, 2, 2])._is_strictly_monotonic_increasing
        False
        >>> Index([1, 3, 2])._is_strictly_monotonic_increasing
        False
        """
        return self.is_unique and self.is_monotonic_increasing

    @final
    @property
    def _is_strictly_monotonic_decreasing(self: Self) -> bool:
        """
        Return if the index is strictly monotonic decreasing
        (only decreasing) values.

        Examples
        --------
        >>> Index([3, 2, 1])._is_strictly_monotonic_decreasing
        True
        >>> Index([3, 2, 2])._is_strictly_monotonic_decreasing
        False
        >>> Index([3, 1, 2])._is_strictly_monotonic_decreasing
        False
        """
        return self.is_unique and self.is_monotonic_decreasing

    @cache_readonly
    def is_unique(self: Self) -> bool:
        """
        Return if the index has unique values.

        Returns
        -------
        bool

        See Also
        --------
        Index.has_duplicates : Inverse method that checks if it has duplicate values.

        Examples
        --------
        >>> idx = pd.Index([1, 5, 7, 7])
        >>> idx.is_unique
        False

        >>> idx = pd.Index([1, 5, 7])
        >>> idx.is_unique
        True

        >>> idx = pd.Index(["Watermelon", "Orange", "Apple", "Watermelon"]).astype(
        ...     "category"
        ... )
        >>> idx.is_unique
        False

        >>> idx = pd.Index(["Orange", "Apple", "Watermelon"]).astype("category")
        >>> idx.is_unique
        True
        """
        return self._engine.is_unique

    @final
    @property
    def has_duplicates(self: Self) -> bool:
        """
        Check if the Index has duplicate values.

        Returns
        -------
        bool
            Whether or not the Index has duplicate values.

        See Also
        --------
        Index.is_unique : Inverse method that checks if it has unique values.

        Examples
        --------
        >>> idx = pd.Index([1, 5, 7, 7])
        >>> idx.has_duplicates
        True

        >>> idx = pd.Index([1, 5, 7])
        >>> idx.has_duplicates
        False

        >>> idx = pd.Index(["Watermelon", "Orange", "Apple", "Watermelon"]).astype(
        ...     "category"
        ... )
        >>> idx.has_duplicates
        True

        >>> idx = pd.Index(["Orange", "Apple", "Watermelon"]).astype("category")
        >>> idx.has_duplicates
        False
        """
        return not self.is_unique

    @cache_readonly
    def inferred_type(self: Self) -> str:
        """
        Return a string of the type inferred from the values.

        See Also
        --------
        Index.dtype : Return the dtype object of the underlying data.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.inferred_type
        'integer'
        """
        return lib.infer_dtype(self._values, skipna=False)

    @cache_readonly
    @final
    def _is_all_dates(self: Self) -> bool:
        """
        Whether or not the index values only consist of dates.
        """
        if needs_i8_conversion(self.dtype):
            return True
        elif self.dtype != _dtype_obj:
            return False
        elif self._is_multi:
            return False
        return is_datetime_array(ensure_object(self._values))

    @final
    @cache_readonly
    def _is_multi(self: Self) -> bool:
        """
        Cached check equivalent to isinstance(self, MultiIndex)
        """
        return isinstance(self, ABCMultiIndex)

    def __reduce__(self: Self) -> tuple[type[Index], tuple[Any, Any], None]:
        d = {'data': self._data, 'name': self.name}
        return (_new_Index, (type(self), d), None)

    @cache_readonly
    def _na_value(self: Self) -> Any:
        """The expected NA value to use with this index."""
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            if dtype.kind in 'mM':
                return NaT
            return np.nan
        return dtype.na_value

    @cache_readonly
    def _isnan(self: Self) -> np.ndarray[np.bool_]:
        """
        Return if each value is NaN.
        """
        if self._can_hold_na:
            return isna(self)
        else:
            values = np.empty(len(self), dtype=np.bool_)
            values.fill(False)
            return values

    @cache_readonly
    def hasnans(self: Self) -> bool:
        """
        Return True if there are any NaNs.

        Enables various performance speedups.

        Returns
        -------
        bool

        See Also
        --------
        Index.isna : Detect missing values.
        Index.dropna : Return Index without NA/NaN values.
        Index.fillna : Fill NA/NaN values with the specified value.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=["a", "b", None])
        >>> s
        a    1
        b    2
        None 3
        dtype: int64
        >>> s.index.hasnans
        True
        """
        if self._can_hold_na:
            return bool(self._isnan.any())
        else:
            return False

    @final
    def isna(self: Self) -> np.ndarray[np.bool_]:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as ``None``, :attr:`numpy.NaN` or :attr:`pd.NaT`, get
        mapped to ``True`` values.
        Everything else get mapped to ``False`` values. Characters such as
        empty strings `''` or :attr:`numpy.inf` are not considered NA values.

        Returns
        -------
        numpy.ndarray[bool]
            A boolean array of whether my values are NA.

        See Also
        --------
        Index.notna : Boolean inverse of isna.
        Index.dropna : Omit entries with missing values.
        isna : Top-level isna.
        Series.isna : Detect missing values in Series object.

        Examples
        --------
        Show which entries in a pandas.Index are NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.nan])
        >>> idx
        Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.isna()
        array([False, False,  True])

        Empty strings are not considered NA values. None is considered an NA
        value.

        >>> idx = pd.Index(["black", "", "red", None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.isna()
        array([False, False, False,  True])

        For datetimes, `NaT` (Not a Time) is considered as an NA value.

        >>> idx = pd.DatetimeIndex(
        ...     [pd.Timestamp("1940-04-25"), pd.Timestamp(""), None, pd.NaT]
        ... )
        >>> idx
        DatetimeIndex(['1940-04-25', 'NaT', 'NaT', 'NaT'],
                      dtype='datetime64[s]', freq=None)
        >>> idx.isna()
        array([False,  True,  True,  True])
        """
        return self._isnan
    isnull = isna

    @final
    def notna(self: Self) -> np.ndarray[np.bool_]:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to ``True``. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.
        NA values, such as None or :attr:`numpy.NaN`, get mapped to ``False``
        values.

        Returns
        -------
        numpy.ndarray[bool]
            Boolean array to indicate which entries are not NA.

        See Also
        --------
        Index.notnull : Alias of notna.
        Index.isna: Inverse of notna.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in an Index are not NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.nan])
        >>> idx
        Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.notna()
        array([ True,  True, False])

        Empty strings are not considered NA values. None is considered a NA
        value.

        >>> idx = pd.Index(["black", "", "red", None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.notna()
        array([ True,  True,  True, False])
        """
        return ~self.isna()
    notnull = notna

    def fillna(self: Self, value: Any) -> Self:
        """
        Fill NA/NaN values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill holes (e.g. 0).
            This value cannot be a list-likes.

        Returns
        -------
        Index
           NA/NaN values replaced with `value`.

        See Also
        --------
        DataFrame.fillna : Fill NaN values of a DataFrame.
        Series.fillna : Fill NaN Values of a Series.

        Examples
        --------
        >>> idx = pd.Index([np.nan, np.nan, 3])
        >>> idx.fillna(0)
        Index([0.0, 0.0, 3.0], dtype='float64')
        """
        if not is_scalar(value):
            raise TypeError(f"'value' must be a scalar, passed: {type(value).__name__}")
        if self.hasnans:
            result = self.putmask(self._isnan, value)
            return Index._with_infer(result, name=self.name)
        return self._view()

    def dropna(self: Self, how: str = 'any') -> Self:
        """
        Return Index without NA/NaN values.

        Parameters
        ----------
        how : {'any', 'all'}, default 'any'
            If the Index is a MultiIndex, drop the value when any or all levels
            are NaN.

        Returns
        -------
        Index
            Returns an Index object after removing NA/NaN values.

        See Also
        --------
        Index.fillna : Fill NA/NaN values with the specified value.
        Index.isna : Detect missing values.

        Examples
        --------
        >>> idx = pd.Index([1, np.nan, 3])
        >>> idx.dropna()
        Index([1.0, 3.0], dtype='float64')
        """
        if how not in ('any', 'all'):
            raise ValueError(f'invalid how option: {how}')
        if self.hasnans:
            res_values = self._values[~self._isnan]
            return type(self)._simple_new(res_values, name=self.name)
        return self._view()

    def unique(self: Self, level: AxisInt | IndexLabel | None = None) -> Self:
        """
        Return unique values in the index.

        Unique values are returned in order of appearance, this does NOT sort.

        Parameters
        ----------
        level : int or hashable, optional
            Only return values from specified level (for MultiIndex).
            If int, gets the level by integer position, else by level name.

        Returns
        -------
        Index
            Unique values in the index.

        See Also
        --------
        unique : Numpy array of unique values in that column.
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> idx = pd.Index([1, 1, 2, 3, 3])
        >>> idx.unique()
        Index([1, 2, 3], dtype='int64')
        """
        if level is not None:
            self._validate_index_level(level)
        if self.is_unique:
            return self._view()
        result = super().unique()
        return self._shallow_copy(result)

    def drop_duplicates(self: Self, *, keep: str = 'first') -> Self:
        """
        Return Index with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        Returns
        -------
        Index
            A new Index object with the duplicate values removed.

        See Also
        --------
        Series.drop_duplicates : Equivalent method on Series.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Index.duplicated : Related method on Index, indicating duplicate
            Index values.

        Examples
        --------
        Generate an pandas.Index with duplicate values.

        >>> idx = pd.Index(["llama", "cow", "llama", "beetle", "llama", "hippo"])

        The `keep` parameter controls  which duplicate values are removed.
        The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> idx.drop_duplicates(keep="first")
        Index(['llama', 'cow', 'beetle', 'hippo'], dtype='object')

        The value 'last' keeps the last occurrence for each set of duplicated
        entries.

        >>> idx.drop_duplicates(keep="last")
        Index(['cow', 'beetle', 'llama', 'hippo'], dtype='object')

        The value ``False`` discards all sets of duplicated entries.

        >>> idx.drop_duplicates(keep=False)
        Index(['cow', 'beetle', 'hippo'], dtype='object')
        """
        if self.is_unique:
            return self._view()
        return super().drop_duplicates(keep=keep)

    def duplicated(self: Self, keep: str = 'first') -> np.ndarray[np.bool_]:
        """
        Indicate duplicate index values.

        Duplicated values are indicated as ``True`` values in the resulting
        array. Either all duplicates, all except the first, or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            The value or values in a set of duplicates to mark as missing.

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        np.ndarray[bool]
            A numpy array of boolean values indicating duplicate index values.

        See Also
        --------
        Series.duplicated : Equivalent method on pandas.Series.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Index.drop_duplicates : Remove duplicate values from Index.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set to False and all others to True:

        >>> idx = pd.Index(["llama", "cow", "llama", "beetle", "llama"])
        >>> idx.duplicated()
        array([False, False,  True, False,  True])

        which is equivalent to

        >>> idx.duplicated(keep="first")
        array([False, False,  True, False,  True])

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> idx.duplicated(keep="last")
        array([ True, False,  True, False, False])

        By setting keep on ``False``, all duplicates are True:

        >>> idx.duplicated(keep=False)
        array([ True, False,  True, False,  True])
        """
        if self.is_unique:
            return np.zeros(len(self), dtype=bool)
        return self._duplicated(keep=keep)

    def __iadd__(self: Self, other: Self) -> Self:
        return self + other

    @final
    def __bool__(self: Self) -> NoReturn:
        raise ValueError(f'The truth value of a {type(self).__name__} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().')

    def _get_reconciled_name_object(self: Self, other: Self) -> Self:
        """
        If the result of a set operation will be self,
        return self, unless the name changes, in which
        case make a shallow copy of self.
        """
        name = get_op_result_name(self, other)
        if self.name is not name:
            return self.rename(name)
        return self

    @final
    def _validate_sort_keyword(self: Self, sort: bool | None) -> None:
        """
        Raise if we have a get_indexer `method` that is not supported or valid.
        """
        if sort not in [None, False, True]:
            raise ValueError(f"The 'sort' keyword only takes the values of None, True, or False; {sort} was passed.")

    @final
    def _dti_setop_align_tzs(self: Self, other: Self, setop: str) -> tuple[Self, Self]:
        """
        With mismatched timezones, cast both to UTC.
        """
        if isinstance(self, ABCDatetimeIndex) and isinstance(other, ABCDatetimeIndex) and (self.tz is not None) and (other.tz is not None):
            left = self.tz_convert('UTC')
            right = other.tz_convert('UTC')
            return (left, right)
        return (self, other)

    @final
    def union(self: Self, other: Self, sort: bool | None) -> Self:
        """
        Form the union of two Index objects.

        If the Index objects are incompatible, both Index objects will be
        cast to dtype('object') first.

        Parameters
        ----------
        other : Index or array-like
            Index or an array-like object containing elements to form the union
            with the original Index.
        sort : bool or None, default None
            Whether to sort the resulting Index.

            * None : Sort the result, except when

              1. `self` and `other` are equal.
              2. `self` or `other` has length 0.
              3. Some values in `self` or `other` cannot be compared.
                 A RuntimeWarning is issued in this case.

            * False : do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index
            Returns a new Index object with all unique elements from both the original
            Index and the `other` Index.

        See Also
        --------
        Index.unique : Return unique values in the index.
        Index.intersection : Form the intersection of two Index objects.
        Index.difference : Return a new Index with elements of index not in `other`.

        Examples
        --------
        Union matching dtypes

        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.union(idx2)
        Index([1, 2, 3, 4, 5, 6], dtype='int64')

        Union mismatched dtypes

        >>> idx1 = pd.Index(["a", "b", "c", "d"])
        >>> idx2 = pd.Index([1, 2, 3, 4])
        >>> idx1.union(idx2)
        Index(['a', 'b', 'c', 'd', 1, 2, 3, 4], dtype='object')

        MultiIndex case

        >>> idx1 = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ... )
        >>> idx1
        MultiIndex([(1,  'Red'),
            (1, 'Blue'),
            (2,  'Red'),
            (2, 'Blue')],
           )
        >>> idx2 = pd.MultiIndex.from_arrays(
        ...     [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
        ... )
        >>> idx2
        MultiIndex([(3,   'Red'),
            (3, 'Green'),
            (2,   'Red'),
            (2, 'Green')],
           )
        >>> idx1.union(idx2)
        MultiIndex([(1,  'Blue'),
            (1,   'Red'),
            (2,  'Blue'),
            (2, 'Green'),
            (2,   'Red'),
            (3, 'Green'),
            (3,   'Red')],
           )
        >>> idx1.union(idx2, sort=False)
        MultiIndex([(1,   'Red'),
            (1,  'Blue'),
            (2,   'Red'),
            (2,  'Blue'),
            (3,   'Red'),
            (3, 'Green'),
            (2, 'Green')],
           )
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)
        if self.dtype != other.dtype:
            self, other = self._dti_setop_align_tzs(other, 'union')
        if self.equals(other):
            if not self.is_unique:
                result = self.unique()._get_reconciled_name_object(other)
            else:
                result = self._get_reconciled_name_object(other)
            if sort is True:
                result = result.sort_values()
            return result
        if len(self) == 0 or len(other) == 0:
            try:
                return self._join_empty(other, how, sort)
            except TypeError:
                pass
        if self.dtype != other.dtype:
            dtype = self._find_common_type_compat(other)
            this = self.astype(dtype, copy=False)
            other = other.astype(dtype, copy=False)
            return this.union(other, sort=sort)
        elif not self._should_compare(other):
            if isinstance(self, ABCMultiIndex):
                return self[:0].rename(result_name)
            return Index([], name=result_name)
        elif self.is_monotonic_increasing and other.is_monotonic_increasing and self._can_use_libjoin and other._can_use_libjoin and (self.is_unique or other.is_unique):
            try:
                return self._join_monotonic(other, how='union')
            except TypeError:
                pass
        elif not self.is_unique or not other.is_unique:
            return self._join_non_unique(other, how='union', sort=sort)
        return self._join_via_get_indexer(other, how='union', sort=sort)

    def _join_empty(self: Self, other: Self, how: str, sort: bool) -> tuple[Self, np.ndarray, np.ndarray]:
        assert len(self) == 0 or len(other) == 0
        _validate_join_method(how)
        if len(other):
            how = cast(JoinHow, {'left': 'right', 'right': 'left'}.get(how, how))
            join_index, ridx, lidx = other._join_empty(self, how, sort)
        elif how in ['left', 'outer']:
            if sort and (not self.is_monotonic_increasing):
                lidx = self.argsort()
                join_index = self.take(lidx)
            else:
                lidx = None
                join_index = self._view()
            ridx = np.broadcast_to(np.intp(-1), len(join_index))
        else:
            join_index = other._view()
            lidx = np.array([], dtype=np.intp)
            ridx = None
        return (join_index, lidx, ridx)

    @final
    def _join_via_get_indexer(self: Self, other: Self, how: str, sort: bool) -> tuple[Self, np.ndarray, np.ndarray]:
        if how == 'left':
            if sort:
                join_index, lindexer = self.sort_values(return_indexer=True)
                rindexer = other.get_indexer_for(join_index)
                return (join_index, lindexer, rindexer)
            else:
                join_index = self
        elif how == 'right':
            if sort:
                join_index, rindexer = other.sort_values(return_indexer=True)
                lindexer = self.get_indexer_for(join_index)
                return (join_index, lindexer, rindexer)
            else:
                join_index = other
        elif how == 'inner':
            join_index = self.intersection(other, sort=sort)
        elif how == 'outer':
            try:
                join_index = self.union(other, sort=sort)
            except TypeError:
                join_index = self.union(other)
                try:
                    join_index = _maybe_try_sort(join_index, sort)
                except TypeError:
                    pass
        names = other.names if how == 'right' else self.names
        if join_index.names != names:
            join_index = join_index.set_names(names)
        if join_index is self:
            lindexer = None
        else:
            lindexer = self.get_indexer_for(join_index)
        if join_index is other:
            rindexer = None
        else:
            rindexer = other.get_indexer_for(join_index)
        return (join_index, lindexer, rindexer)

    @final
    def _join_multi(self: Self, other: Self) -> tuple[Self, np.ndarray, np.ndarray]:
        from pandas.core.indexes.multi import MultiIndex
        self_names_list = list(self.names)
        other_names_list = list(other.names)
        self_names_order = self_names_list.index
        other_names_order = other_names_list.index
        self_names = set(self_names_list)
        other_names = set(other_names_list)
        overlap = self_names & other_names
        if not overlap:
            raise ValueError('cannot join with no overlapping index names')
        if isinstance(self, MultiIndex) and isinstance(other, MultiIndex):
            ldrop_names = sorted(self_names - overlap, key=self_names_order)
            rdrop_names = sorted(other_names - overlap, key=other_names_order)
            if not len(ldrop_names + rdrop_names):
                self_jnlevels = self
                other_jnlevels = other.reorder_levels(self.names)
            else:
                self_jnlevels = self.droplevel(ldrop_names)
                other_jnlevels = other.droplevel(rdrop_names)
            join_idx, lidx, ridx = self_jnlevels.join(other_jnlevels, how=how, return_indexers=True)
            dropped_names = ldrop_names + rdrop_names
            levels, codes, names = restore_dropped_levels_multijoin(self, other, dropped_names, join_idx, lidx, ridx)
            multi_join_idx = MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=False)
            multi_join_idx = multi_join_idx.remove_unused_levels()
            if how == 'right':
                level_order = other_names_list + ldrop_names
            else:
                level_order = self_names_list + rdrop_names
            multi_join_idx = multi_join_idx.reorder_levels(level_order)
            return (multi_join_idx, lidx, ridx)
        jl = next(iter(overlap))
        flip_order = False
        if isinstance(self, MultiIndex):
            self, other = (other, self)
            flip_order = True
            flip = {'right': 'left', 'left': 'right'}
            how = flip.get(how, how)
        assert isinstance(self, MultiIndex)
        level = other.names.index(jl)
        result = self._join_level(other, level, how=how)
        if flip_order:
            return (result[0], result[2], result[1])
        return result

    @final
    def _join_non_unique(self: Self, other: Self, how: str, sort: bool) -> tuple[Self, np.ndarray, np.ndarray]:
        from pandas.core.reshape.merge import get_join_indexers_non_unique
        assert self.dtype == other.dtype
        left_idx, right_idx = get_join_indexers_non_unique(self._values, other._values, how=how, sort=sort)
        if how == 'right':
            join_index = other.take(right_idx)
        else:
            join_index = self.take(left_idx)
        if how == 'outer':
            mask = left_idx == -1
            if mask.any():
                right = other.take(right_idx)
                join_index = join_index.putmask(mask, right)
        if isinstance(join_index, ABCMultiIndex) and how == 'outer':
            join_index = join_index._sort_levels_monotonic()
        return (join_index, left_idx, right_idx)

    @final
    def _join_level(self: Self, other: Self, level: AxisInt | IndexLabel, how: str, keep_order: bool = True) -> tuple[Self, np.ndarray, np.ndarray]:
        """
        The join method *only* affects the level of the resulting
        MultiIndex. Otherwise it just exactly aligns the Index data to the
        labels of the level in the MultiIndex.

        If 