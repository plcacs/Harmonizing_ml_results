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

def _new_Index(cls, d):
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
    def _left_indexer_unique(self, other):
        sv = self._get_join_target()
        ov = other._get_join_target()
        return libjoin.left_join_indexer_unique(sv, ov)

    @final
    def _left_indexer(self, other):
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.left_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return (joined, lidx, ridx)

    @final
    def _inner_indexer(self, other):
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.inner_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return (joined, lidx, ridx)

    @final
    def _outer_indexer(self, other):
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
    def _can_hold_strings(self):
        return not is_numeric_dtype(self.dtype)
    _engine_types = {np.dtype(np.int8): libindex.Int8Engine, np.dtype(np.int16): libindex.Int16Engine, np.dtype(np.int32): libindex.Int32Engine, np.dtype(np.int64): libindex.Int64Engine, np.dtype(np.uint8): libindex.UInt8Engine, np.dtype(np.uint16): libindex.UInt16Engine, np.dtype(np.uint32): libindex.UInt32Engine, np.dtype(np.uint64): libindex.UInt64Engine, np.dtype(np.float32): libindex.Float32Engine, np.dtype(np.float64): libindex.Float64Engine, np.dtype(np.complex64): libindex.Complex64Engine, np.dtype(np.complex128): libindex.Complex128Engine}

    @property
    def _engine_type(self):
        return self._engine_types.get(self.dtype, libindex.ObjectEngine)
    _supports_partial_string_indexing = False
    _accessors = {'str'}
    str = Accessor('str', StringMethods)
    _references = None

    def __new__(cls, data=None, dtype=None, copy=False, name=None, tupleize_cols=True):
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
    def _ensure_array(cls, data, dtype, copy):
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
    def _dtype_to_subclass(cls, dtype):
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
    def _simple_new(cls, values, name=None, refs=None):
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
    def _with_infer(cls, *args, **kwargs):
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
    def _constructor(self):
        return type(self)

    @final
    def _maybe_check_unique(self):
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
    def _format_duplicate_message(self):
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

    def _shallow_copy(self, values, name=no_default):
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

    def _view(self):
        """
        fastpath to make a shallow copy, i.e. new object with same data.
        """
        result = self._simple_new(self._values, name=self._name, refs=self._references)
        result._cache = self._cache
        return result

    @final
    def _rename(self, name):
        """
        fastpath for rename if new name is already validated.
        """
        result = self._view()
        result._name = name
        return result

    @final
    def is_(self, other):
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
    def _reset_identity(self):
        """
        Initializes or resets ``_id`` attribute with new object.
        """
        self._id = object()

    @final
    def _cleanup(self):
        if '_engine' in self._cache:
            self._engine.clear_mapping()

    @cache_readonly
    def _engine(self):
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

    @final
    @cache_readonly
    def _dir_additions_for_owner(self):
        """
        Add the string-like labels to the owner dataframe/series dir output.

        If this is a MultiIndex, it's first level values are used.
        """
        return {c for c in self.unique(level=0)[:get_option('display.max_dir_items')] if isinstance(c, str) and c.isidentifier()}

    def __len__(self):
        """
        Return the length of the Index.
        """
        return len(self._data)

    def __array__(self, dtype=None, copy=None):
        """
        The array interface, return my values.
        """
        if copy is None:
            return np.asarray(self._data, dtype=dtype)
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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
    def __array_wrap__(self, result, context=None, return_scalar=False):
        """
        Gets called after a ufunc and other functions e.g. np.split.
        """
        result = lib.item_from_zerodim(result)
        if not isinstance(result, Index) and is_bool_dtype(result.dtype) or np.ndim(result) > 1:
            return result
        return Index(result, name=self.name)

    @cache_readonly
    def dtype(self):
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
    def ravel(self, order='C'):
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

    def view(self, cls=None):
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

    def astype(self, dtype, copy=True):
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
            Index with values cast to specified dtype.

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
    def take(self, indices, axis=0, allow_fill=True, fill_value=None, **kwargs):
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
    def _maybe_disallow_fill(self, allow_fill, fill_value, indices):
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
    def repeat(self, repeats, axis=None):
        repeats = ensure_platform_int(repeats)
        nv.validate_repeat((), {'axis': axis})
        res_values = self._values.repeat(repeats)
        return self._constructor._simple_new(res_values, name=self.name)

    def copy(self, name=None, deep=False):
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
    def __copy__(self, **kwargs):
        return self.copy(**kwargs)

    @final
    def __deepcopy__(self, memo=None):
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        return self.copy(deep=True)

    @final
    def __repr__(self):
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
    def _formatter_func(self):
        """
        Return the formatter function.
        """
        return default_pprint

    @final
    def _format_data(self, name=None):
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

    def _format_attrs(self):
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
    def _get_level_names(self):
        """
        Return a name or list of names with None replaced by the level number.
        """
        if self._is_multi:
            return maybe_sequence_to_range([level if name is None else name for level, name in enumerate(self.names)])
        else:
            return range(1) if self.name is None else [self.name]

    @final
    def _mpl_repr(self):
        if isinstance(self.dtype, np.dtype) and self.dtype.kind != 'M':
            return cast(np.ndarray, self.values)
        return self.astype(object, copy=False)._values
    _default_na_rep = 'NaN'

    @final
    def _format_flat(self, *, include_name, formatter=None):
        """
        Render a string representation of the Index.
        """
        header = []
        if include_name:
            header.append(pprint_thing(self.name, escape_chars=('\t', '\r', '\n')) if self.name is not None else '')
        if formatter is not None:
            return header + list(self.map(formatter))
        return self._format_with_header(header=header, na_rep=self._default_na_rep)

    def _format_with_header(self, *, header, na_rep):
        from pandas.io.formats.format import format_array
        values = self._values
        if is_object_dtype(values.dtype) or is_string_dtype(values.dtype) or isinstance(self.dtype, (IntervalDtype, CategoricalDtype)):
            justify = 'all'
        else:
            justify = 'left'
        formatted = format_array(values, None, justify=justify)
        result = trim_front(formatted)
        return header + result

    def _get_values_for_csv(self, *, na_rep='', decimal='.', float_format=None, date_format=None, quoting=None):
        return get_values_for_csv(self._values, na_rep=na_rep, decimal=decimal, float_format=float_format, date_format=date_format, quoting=quoting)

    def _summary(self, name=None):
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

    def to_flat_index(self):
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
    def to_series(self, index=None, name=None):
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

    def to_frame(self, index=True, name=lib.no_default):
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
    def name(self):
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
        Index([1, 2, 3], dtype='int64',  name='x')
        >>> idx.name
        'x'
        """
        return self._name

    @name.setter
    def name(self, value):
        if self._no_setting_name:
            raise RuntimeError("Cannot set name on a level of a MultiIndex. Use 'MultiIndex.set_names' instead.")
        maybe_extract_name(value, None, type(self))
        self._name = value

    @final
    def _validate_names(self, name=None, names=None, deep=False):
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
                raise TypeError('Must pass list-like as `names`.')
            new_names = names
        elif not is_list_like(name):
            new_names = [name]
        else:
            new_names = name
        if len(new_names) != len(self.names):
            raise ValueError(f'Length of new names must be {len(self.names)}, got {len(new_names)}')
        validate_all_hashable(*new_names, error_name=f'{type(self).__name__}.name')
        return new_names

    def _get_default_index_names(self, names=None, default=None):
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

    def _get_names(self):
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

    def _set_names(self, values, *, level=None):
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
    def set_names(self, names, *, level=..., inplace=...):
        ...

    @overload
    def set_names(self, names, *, level=..., inplace):
        ...

    @overload
    def set_names(self, names, *, level=..., inplace=...):
        ...

    def set_names(self, names, *, level=None, inplace=False):
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
    def rename(self, name, *, inplace=...):
        ...

    @overload
    def rename(self, name, *, inplace):
        ...

    def rename(self, name, *, inplace=False):
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
    def nlevels(self):
        """
        Number of levels.
        """
        return 1

    def _sort_levels_monotonic(self):
        """
        Compat with MultiIndex.
        """
        return self

    @final
    def _validate_index_level(self, level):
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

    def _get_level_number(self, level):
        self._validate_index_level(level)
        return 0

    def sortlevel(self, level=None, ascending=True, sort_remaining=None, na_position='first'):
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

    def _get_level_values(self, level):
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
    def droplevel(self, level=0):
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
    def _drop_level_numbers(self, levnums):
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
    def _can_hold_na(self):
        if isinstance(self.dtype, ExtensionDtype):
            return self.dtype._can_hold_na
        if self.dtype.kind in 'iub':
            return False
        return True

    @property
    def is_monotonic_increasing(self):
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
    def is_monotonic_decreasing(self):
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
    def _is_strictly_monotonic_increasing(self):
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
    def _is_strictly_monotonic_decreasing(self):
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
    def is_unique(self):
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
    def has_duplicates(self):
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
    def inferred_type(self):
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
    def _is_all_dates(self):
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
    def _is_multi(self):
        """
        Cached check equivalent to isinstance(self, MultiIndex)
        """
        return isinstance(self, ABCMultiIndex)

    def __reduce__(self):
        d = {'data': self._data, 'name': self.name}
        return (_new_Index, (type(self), d), None)

    @cache_readonly
    def _na_value(self):
        """The expected NA value to use with this index."""
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            if dtype.kind in 'mM':
                return NaT
            return np.nan
        return dtype.na_value

    @cache_readonly
    def _isnan(self):
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
    def hasnans(self):
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
    def isna(self):
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
    def notna(self):
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

    def fillna(self, value):
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

    def dropna(self, how='any'):
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

    def unique(self, level=None):
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

    def drop_duplicates(self, *, keep='first'):
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

    def duplicated(self, keep='first'):
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

    def __iadd__(self, other):
        return self + other

    @final
    def __bool__(self):
        raise ValueError(f'The truth value of a {type(self).__name__} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().')

    def _get_reconciled_name_object(self, other):
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
    def _validate_sort_keyword(self, sort):
        if sort not in [None, False, True]:
            raise ValueError(f"The 'sort' keyword only takes the values of None, True, or False; {sort} was passed.")

    @final
    def _dti_setop_align_tzs(self, other, setop):
        """
        With mismatched timezones, cast both to UTC.
        """
        if isinstance(self, ABCDatetimeIndex) and isinstance(other, ABCDatetimeIndex) and (self.tz is not None) and (other.tz is not None):
            left = self.tz_convert('UTC')
            right = other.tz_convert('UTC')
            return (left, right)
        return (self, other)

    @final
    def union(self, other, sort=None):
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
            if isinstance(self, ABCMultiIndex) and (not is_object_dtype(_unpack_nested_dtype(other))) and (len(other) > 0):
                raise NotImplementedError('Can only union MultiIndex with MultiIndex or Index of tuples, try mi.to_flat_index().union(other) instead.')
            self, other = self._dti_setop_align_tzs(other, 'union')
            dtype = self._find_common_type_compat(other)
            left = self.astype(dtype, copy=False)
            right = other.astype(dtype, copy=False)
            return left.union(right, sort=sort)
        elif not len(other) or self.equals(other):
            result = self._get_reconciled_name_object(other)
            if sort is True:
                return result.sort_values()
            return result
        elif not len(self):
            result = other._get_reconciled_name_object(self)
            if sort is True:
                return result.sort_values()
            return result
        result = self._union(other, sort=sort)
        return self._wrap_setop_result(other, result)

    def _union(self, other, sort):
        """
        Specific union logic should go here. In subclasses, union behavior
        should be overwritten here rather than in `self.union`.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default False
            Whether to sort the resulting index.

            * True : sort the result
            * False : do not sort the result.
            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.

        Returns
        -------
        Index
        """
        lvals = self._values
        rvals = other._values
        if sort in (None, True) and (self.is_unique or other.is_unique) and self._can_use_libjoin and other._can_use_libjoin:
            try:
                return self._outer_indexer(other)[0]
            except (TypeError, IncompatibleFrequency):
                value_list = list(lvals)
                value_set = set(lvals)
                value_list.extend((x for x in rvals if x not in value_set))
                return np.array(value_list, dtype=object)
        elif not other.is_unique:
            result_dups = algos.union_with_duplicates(self, other)
            return _maybe_try_sort(result_dups, sort)
        if self._index_as_unique:
            indexer = self.get_indexer(other)
            missing = (indexer == -1).nonzero()[0]
        else:
            missing = algos.unique1d(self.get_indexer_non_unique(other)[1])
        if self._is_multi:
            result = self.append(other.take(missing))
        elif len(missing) > 0:
            other_diff = rvals.take(missing)
            result = concat_compat((lvals, other_diff))
        else:
            result = lvals
        if not self.is_monotonic_increasing or not other.is_monotonic_increasing:
            result = _maybe_try_sort(result, sort)
        return result

    @final
    def _wrap_setop_result(self, other, result):
        name = get_op_result_name(self, other)
        if isinstance(result, Index):
            if result.name != name:
                result = result.rename(name)
        else:
            result = self._shallow_copy(result, name=name)
        return result

    @final
    def intersection(self, other, sort=False):
        """
        Form the intersection of two Index objects.

        This returns a new Index with elements common to the index and `other`.

        Parameters
        ----------
        other : Index or array-like
            An Index or an array-like object containing elements to form the
            intersection with the original Index.
        sort : True, False or None, default False
            Whether to sort the resulting index.

            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.
            * False : do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index
            Returns a new Index object with elements common to both the original Index
            and the `other` Index.

        See Also
        --------
        Index.union : Form the union of two Index objects.
        Index.difference : Return a new Index with elements of index not in other.
        Index.isin : Return a boolean array where the index values are in values.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.intersection(idx2)
        Index([3, 4], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)
        if self.dtype != other.dtype:
            self, other = self._dti_setop_align_tzs(other, 'intersection')
        if self.equals(other):
            if not self.is_unique:
                result = self.unique()._get_reconciled_name_object(other)
            else:
                result = self._get_reconciled_name_object(other)
            if sort is True:
                result = result.sort_values()
            return result
        if len(self) == 0 or len(other) == 0:
            if self._is_multi or other._is_multi:
                return self[:0].rename(result_name)
            dtype = self._find_common_type_compat(other)
            if self.dtype == dtype:
                if len(self) == 0:
                    return self[:0].rename(result_name)
                else:
                    return other[:0].rename(result_name)
            return Index([], dtype=dtype, name=result_name)
        elif not self._should_compare(other):
            if isinstance(self, ABCMultiIndex):
                return self[:0].rename(result_name)
            return Index([], name=result_name)
        elif self.dtype != other.dtype:
            dtype = self._find_common_type_compat(other)
            this = self.astype(dtype, copy=False)
            other = other.astype(dtype, copy=False)
            return this.intersection(other, sort=sort)
        result = self._intersection(other, sort=sort)
        return self._wrap_intersection_result(other, result)

    def _intersection(self, other, sort=False):
        """
        intersection specialized to the case with matching dtypes.
        """
        if self._can_use_libjoin and other._can_use_libjoin:
            try:
                res_indexer, indexer, _ = self._inner_indexer(other)
            except TypeError:
                pass
            else:
                if is_numeric_dtype(self.dtype):
                    res = algos.unique1d(res_indexer)
                else:
                    result = self.take(indexer)
                    res = result.drop_duplicates()
                return ensure_wrapped_if_datetimelike(res)
        res_values = self._intersection_via_get_indexer(other, sort=sort)
        res_values = _maybe_try_sort(res_values, sort)
        return res_values

    def _wrap_intersection_result(self, other, result):
        return self._wrap_setop_result(other, result)

    @final
    def _intersection_via_get_indexer(self, other, sort):
        """
        Find the intersection of two Indexes using get_indexer.

        Returns
        -------
        np.ndarray or ExtensionArray or MultiIndex
            The returned array will be unique.
        """
        left_unique = self.unique()
        right_unique = other.unique()
        indexer = left_unique.get_indexer_for(right_unique)
        mask = indexer != -1
        taker = indexer.take(mask.nonzero()[0])
        if sort is False:
            taker = np.sort(taker)
        if isinstance(left_unique, ABCMultiIndex):
            result = left_unique.take(taker)
        else:
            result = left_unique.take(taker)._values
        return result

    @final
    def difference(self, other, sort=None):
        """
        Return a new Index with elements of index not in `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
            Index object or an array-like object containing elements to be compared
            with the elements of the original Index.
        sort : bool or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index
            Returns a new Index object containing elements that are in the original
            Index but not in the `other` Index.

        See Also
        --------
        Index.symmetric_difference : Compute the symmetric difference of two Index
            objects.
        Index.intersection : Form the intersection of two Index objects.

        Examples
        --------
        >>> idx1 = pd.Index([2, 1, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2)
        Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Index([2, 1], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)
        if self.equals(other):
            return self[:0].rename(result_name)
        if len(other) == 0:
            result = self.unique().rename(result_name)
            if sort is True:
                return result.sort_values()
            return result
        if not self._should_compare(other):
            result = self.unique().rename(result_name)
            if sort is True:
                return result.sort_values()
            return result
        result = self._difference(other, sort=sort)
        return self._wrap_difference_result(other, result)

    def _difference(self, other, sort):
        this = self
        if isinstance(self, ABCCategoricalIndex) and self.hasnans and other.hasnans:
            this = this.dropna()
        other = other.unique()
        the_diff = this[other.get_indexer_for(this) == -1]
        the_diff = the_diff if this.is_unique else the_diff.unique()
        the_diff = _maybe_try_sort(the_diff, sort)
        return the_diff

    def _wrap_difference_result(self, other, result):
        return self._wrap_setop_result(other, result)

    def symmetric_difference(self, other, result_name=None, sort=None):
        """
        Compute the symmetric difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
            Index or an array-like object with elements to compute the symmetric
            difference with the original Index.
        result_name : str
            A string representing the name of the resulting Index, if desired.
        sort : bool or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index
            Returns a new Index object containing elements that appear in either the
            original Index or the `other` Index, but not both.

        See Also
        --------
        Index.difference : Return a new Index with elements of index not in other.
        Index.union : Form the union of two Index objects.
        Index.intersection : Form the intersection of two Index objects.

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([2, 3, 4, 5])
        >>> idx1.symmetric_difference(idx2)
        Index([1, 5], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name_update = self._convert_can_do_setop(other)
        if result_name is None:
            result_name = result_name_update
        if self.dtype != other.dtype:
            self, other = self._dti_setop_align_tzs(other, 'symmetric_difference')
        if not self._should_compare(other):
            return self.union(other, sort=sort).rename(result_name)
        elif self.dtype != other.dtype:
            dtype = self._find_common_type_compat(other)
            this = self.astype(dtype, copy=False)
            that = other.astype(dtype, copy=False)
            return this.symmetric_difference(that, sort=sort).rename(result_name)
        this = self.unique()
        other = other.unique()
        indexer = this.get_indexer_for(other)
        common_indexer = indexer.take((indexer != -1).nonzero()[0])
        left_indexer = np.setdiff1d(np.arange(this.size), common_indexer, assume_unique=True)
        left_diff = this.take(left_indexer)
        right_indexer = (indexer == -1).nonzero()[0]
        right_diff = other.take(right_indexer)
        res_values = left_diff.append(right_diff)
        result = _maybe_try_sort(res_values, sort)
        if not self._is_multi:
            return Index(result, name=result_name, dtype=res_values.dtype)
        else:
            left_diff = cast('MultiIndex', left_diff)
            if len(result) == 0:
                return left_diff.remove_unused_levels().set_names(result_name)
            return result.set_names(result_name)

    @final
    def _assert_can_do_setop(self, other):
        if not is_list_like(other):
            raise TypeError('Input must be Index or array-like')
        return True

    def _convert_can_do_setop(self, other):
        if not isinstance(other, Index):
            other = Index(other, name=self.name)
            result_name = self.name
        else:
            result_name = get_op_result_name(self, other)
        return (other, result_name)

    def get_loc(self, key):
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label
            The key to check its location if it is present in the index.

        Returns
        -------
        int if unique index, slice if monotonic index, else mask
            Integer location, slice or boolean mask.

        See Also
        --------
        Index.get_slice_bound : Calculate slice bound that corresponds to
            given label.
        Index.get_indexer : Computes indexer and mask for new index given
            the current index.
        Index.get_non_unique : Returns indexer and masks for new index given
            the current index.
        Index.get_indexer_for : Returns an indexer even when non-unique.

        Examples
        --------
        >>> unique_index = pd.Index(list("abc"))
        >>> unique_index.get_loc("b")
        1

        >>> monotonic_index = pd.Index(list("abbc"))
        >>> monotonic_index.get_loc("b")
        slice(1, 3, None)

        >>> non_monotonic_index = pd.Index(list("abcb"))
        >>> non_monotonic_index.get_loc("b")
        array([False,  True, False,  True])
        """
        casted_key = self._maybe_cast_indexer(key)
        try:
            return self._engine.get_loc(casted_key)
        except KeyError as err:
            if isinstance(casted_key, slice) or (isinstance(casted_key, abc.Iterable) and any((isinstance(x, slice) for x in casted_key))):
                raise InvalidIndexError(key) from err
            raise KeyError(key) from err
        except TypeError:
            self._check_indexing_error(key)
            raise

    @final
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : Index
            An iterable containing the values to be used for computing indexer.
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        limit : int, optional
            Maximum number of consecutive labels in ``target`` to match for
            inexact matches.
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

        Returns
        -------
        np.ndarray[np.intp]
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.

        See Also
        --------
        Index.get_indexer_for : Returns an indexer even when non-unique.
        Index.get_non_unique : Returns indexer and masks for new index given
            the current index.

        Notes
        -----
        Returns -1 for unmatched values, for further explanation see the
        example below.

        Examples
        --------
        >>> index = pd.Index(["c", "a", "b"])
        >>> index.get_indexer(["a", "b", "x"])
        array([ 1,  2, -1])

        Notice that the return value is an array of locations in ``index``
        and ``x`` is marked by -1, as it is not in ``index``.
        """
        method = clean_reindex_fill_method(method)
        orig_target = target
        target = self._maybe_cast_listlike_indexer(target)
        self._check_indexing_method(method, limit, tolerance)
        if not self._index_as_unique:
            raise InvalidIndexError(self._requires_unique_msg)
        if len(target) == 0:
            return np.array([], dtype=np.intp)
        if not self._should_compare(target) and (not self._should_partial_index(target)):
            return self._get_indexer_non_comparable(target, method=method, unique=True)
        if isinstance(self.dtype, CategoricalDtype):
            assert self.dtype == target.dtype
            indexer = self._engine.get_indexer(target.codes)
            if self.hasnans and target.hasnans:
                target_nans = isna(orig_target)
                loc = self.get_loc(np.nan)
                mask = target.isna()
                indexer[target_nans] = loc
                indexer[mask & ~target_nans] = -1
            return indexer
        if isinstance(target.dtype, CategoricalDtype):
            categories_indexer = self.get_indexer(target.categories)
            indexer = algos.take_nd(categories_indexer, target.codes, fill_value=-1)
            if (not self._is_multi and self.hasnans) and target.hasnans:
                loc = self.get_loc(np.nan)
                mask = target.isna()
                indexer[mask] = loc
            return ensure_platform_int(indexer)
        pself, ptarget = self._maybe_downcast_for_indexing(target)
        if pself is not self or ptarget is not target:
            return pself.get_indexer(ptarget, method=method, limit=limit, tolerance=tolerance)
        if self.dtype == target.dtype and self.equals(target):
            return np.arange(len(target), dtype=np.intp)
        if self.dtype != target.dtype and (not self._should_partial_index(target)):
            dtype = self._find_common_type_compat(target)
            this = self.astype(dtype, copy=False)
            target = target.astype(dtype, copy=False)
            return this._get_indexer(target, method=method, limit=limit, tolerance=tolerance)
        return self._get_indexer(target, method, limit, tolerance)

    def _get_indexer(self, target, method=None, limit=None, tolerance=None):
        if tolerance is not None:
            tolerance = self._convert_tolerance(tolerance, target)
        if method in ['pad', 'backfill']:
            indexer = self._get_fill_indexer(target, method, limit, tolerance)
        elif method == 'nearest':
            indexer = self._get_nearest_indexer(target, limit, tolerance)
        else:
            if target._is_multi and self._is_multi:
                engine = self._engine
                tgt_values = engine._extract_level_codes(target)
            else:
                tgt_values = target._get_engine_target()
            indexer = self._engine.get_indexer(tgt_values)
        return ensure_platform_int(indexer)

    @final
    def _should_partial_index(self, target):
        """
        Should we attempt partial-matching indexing?
        """
        if isinstance(self.dtype, IntervalDtype):
            if isinstance(target.dtype, IntervalDtype):
                return False
            return self.left._should_compare(target)
        return False

    @final
    def _check_indexing_method(self, method, limit=None, tolerance=None):
        """
        Raise if we have a get_indexer `method` that is not supported or valid.
        """
        if method not in [None, 'bfill', 'backfill', 'pad', 'ffill', 'nearest']:
            raise ValueError('Invalid fill method')
        if self._is_multi:
            if method == 'nearest':
                raise NotImplementedError("method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365")
            if method in ('pad', 'backfill'):
                if tolerance is not None:
                    raise NotImplementedError('tolerance not implemented yet for MultiIndex')
        if isinstance(self.dtype, (IntervalDtype, CategoricalDtype)):
            if method is not None:
                raise NotImplementedError(f'method {method} not yet implemented for {type(self).__name__}')
        if method is None:
            if tolerance is not None:
                raise ValueError('tolerance argument only valid if doing pad, backfill or nearest reindexing')
            if limit is not None:
                raise ValueError('limit argument only valid if doing pad, backfill or nearest reindexing')

    def _convert_tolerance(self, tolerance, target):
        tolerance = np.asarray(tolerance)
        if target.size != tolerance.size and tolerance.size > 1:
            raise ValueError('list-like tolerance size must match target index size')
        elif is_numeric_dtype(self) and (not np.issubdtype(tolerance.dtype, np.number)):
            if tolerance.ndim > 0:
                raise ValueError(f'tolerance argument for {type(self).__name__} with dtype {self.dtype} must contain numeric elements if it is list type')
            raise ValueError(f'tolerance argument for {type(self).__name__} with dtype {self.dtype} must be numeric if it is a scalar: {tolerance!r}')
        return tolerance

    @final
    def _get_fill_indexer(self, target, method, limit=None, tolerance=None):
        if self._is_multi:
            if not (self.is_monotonic_increasing or self.is_monotonic_decreasing):
                raise ValueError('index must be monotonic increasing or decreasing')
            encoded = self.append(target)._engine.values
            self_encoded = Index(encoded[:len(self)])
            target_encoded = Index(encoded[len(self):])
            return self_encoded._get_fill_indexer(target_encoded, method, limit, tolerance)
        if self.is_monotonic_increasing and target.is_monotonic_increasing:
            target_values = target._get_engine_target()
            own_values = self._get_engine_target()
            if not isinstance(target_values, np.ndarray) or not isinstance(own_values, np.ndarray):
                raise NotImplementedError
            if method == 'pad':
                indexer = libalgos.pad(own_values, target_values, limit=limit)
            else:
                indexer = libalgos.backfill(own_values, target_values, limit=limit)
        else:
            indexer = self._get_fill_indexer_searchsorted(target, method, limit)
        if tolerance is not None and len(self):
            indexer = self._filter_indexer_tolerance(target, indexer, tolerance)
        return indexer

    @final
    def _get_fill_indexer_searchsorted(self, target, method, limit=None):
        """
        Fallback pad/backfill get_indexer that works for monotonic decreasing
        indexes and non-monotonic targets.
        """
        if limit is not None:
            raise ValueError(f'limit argument for {method!r} method only well-defined if index and target are monotonic')
        side = 'left' if method == 'pad' else 'right'
        indexer = self.get_indexer(target)
        nonexact = indexer == -1
        indexer[nonexact] = self._searchsorted_monotonic(target[nonexact], side)
        if side == 'left':
            indexer[nonexact] -= 1
        else:
            indexer[indexer == len(self)] = -1
        return indexer

    @final
    def _get_nearest_indexer(self, target, limit, tolerance):
        """
        Get the indexer for the nearest index labels; requires an index with
        values that can be subtracted from each other (e.g., not strings or
        tuples).
        """
        if not len(self):
            return self._get_fill_indexer(target, 'pad')
        left_indexer = self.get_indexer(target, 'pad', limit=limit)
        right_indexer = self.get_indexer(target, 'backfill', limit=limit)
        left_distances = self._difference_compat(target, left_indexer)
        right_distances = self._difference_compat(target, right_indexer)
        op = operator.lt if self.is_monotonic_increasing else operator.le
        indexer = np.where(op(left_distances, right_distances) | (right_indexer == -1), left_indexer, right_indexer)
        if tolerance is not None:
            indexer = self._filter_indexer_tolerance(target, indexer, tolerance)
        return indexer

    @final
    def _filter_indexer_tolerance(self, target, indexer, tolerance):
        distance = self._difference_compat(target, indexer)
        return np.where(distance <= tolerance, indexer, -1)

    @final
    def _difference_compat(self, target, indexer):
        if isinstance(self.dtype, PeriodDtype):
            own_values = cast('PeriodArray', self._data)._ndarray
            target_values = cast('PeriodArray', target._data)._ndarray
            diff = own_values[indexer] - target_values
        else:
            diff = self._values[indexer] - target._values
        return abs(diff)

    @final
    def _validate_positional_slice(self, key):
        """
        For positional indexing, a slice must have either int or None
        for each of start, stop, and step.
        """
        self._validate_indexer('positional', key.start, 'iloc')
        self._validate_indexer('positional', key.stop, 'iloc')
        self._validate_indexer('positional', key.step, 'iloc')

    def _convert_slice_indexer(self, key, kind):
        """
        Convert a slice indexer.

        By definition, these are labels unless 'iloc' is passed in.
        Floats are not allowed as the start, step, or stop of the slice.

        Parameters
        ----------
        key : label of the slice bound
        kind : {'loc', 'getitem'}
        """
        start, stop, step = (key.start, key.stop, key.step)
        is_index_slice = is_valid_positional_slice(key)
        if kind == 'getitem':
            if is_index_slice:
                return key
            elif self.dtype.kind in 'iu':
                self._validate_indexer('slice', key.start, 'getitem')
                self._validate_indexer('slice', key.stop, 'getitem')
                self._validate_indexer('slice', key.step, 'getitem')
                return key
        is_positional = is_index_slice and self._should_fallback_to_positional
        if is_positional:
            try:
                if start is not None:
                    self.get_loc(start)
                if stop is not None:
                    self.get_loc(stop)
                is_positional = False
            except KeyError:
                pass
        if com.is_null_slice(key):
            indexer = key
        elif is_positional:
            if kind == 'loc':
                raise TypeError('Slicing a positional slice with .loc is not allowed, Use .loc with labels or .iloc with positions instead.')
            indexer = key
        else:
            indexer = self.slice_indexer(start, stop, step)
        return indexer

    @final
    def _raise_invalid_indexer(self, form, key, reraise=lib.no_default):
        """
        Raise consistent invalid indexer message.
        """
        msg = f'cannot do {form} indexing on {type(self).__name__} with these indexers [{key}] of type {type(key).__name__}'
        if reraise is not lib.no_default:
            raise TypeError(msg) from reraise
        raise TypeError(msg)

    @final
    def _validate_can_reindex(self, indexer):
        """
        Check if we are allowing reindexing with this particular indexer.

        Parameters
        ----------
        indexer : an integer ndarray

        Raises
        ------
        ValueError if its a duplicate axis
        """
        if not self._index_as_unique and len(indexer):
            raise ValueError('cannot reindex on an axis with duplicate labels')

    def reindex(self, target, method=None, level=None, limit=None, tolerance=None):
        """
        Create index with target's values.

        Parameters
        ----------
        target : an iterable
            An iterable containing the values to be used for creating the new index.
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        level : int, optional
            Level of multiindex.
        limit : int, optional
            Maximum number of consecutive labels in ``target`` to match for
            inexact matches.
        tolerance : int or float, optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

        Returns
        -------
        new_index : pd.Index
            Resulting index.
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index.

        Raises
        ------
        TypeError
            If ``method`` passed along with ``level``.
        ValueError
            If non-unique multi-index
        ValueError
            If non-unique index and ``method`` or ``limit`` passed.

        See Also
        --------
        Series.reindex : Conform Series to new index with optional filling logic.
        DataFrame.reindex : Conform DataFrame to new index with optional filling logic.

        Examples
        --------
        >>> idx = pd.Index(["car", "bike", "train", "tractor"])
        >>> idx
        Index(['car', 'bike', 'train', 'tractor'], dtype='object')
        >>> idx.reindex(["car", "bike"])
        (Index(['car', 'bike'], dtype='object'), array([0, 1]))
        """
        preserve_names = not hasattr(target, 'name')
        if is_iterator(target):
            target = list(target)
        if not isinstance(target, Index) and len(target) == 0:
            if level is not None and self._is_multi:
                idx = self.levels[level]
            else:
                idx = self
            target = idx[:0]
        else:
            target = ensure_index(target)
        if level is not None and (isinstance(self, ABCMultiIndex) or isinstance(target, ABCMultiIndex)):
            if method is not None:
                raise TypeError('Fill method not supported if level passed')
            target, indexer, _ = self._join_level(target, level, how='right', keep_order=not self._is_multi)
        elif self.equals(target):
            indexer = None
        elif self._index_as_unique:
            indexer = self.get_indexer(target, method=method, limit=limit, tolerance=tolerance)
        elif self._is_multi:
            raise ValueError('cannot handle a non-unique multi-index!')
        elif not self.is_unique:
            raise ValueError('cannot reindex on an axis with duplicate labels')
        else:
            indexer, _ = self.get_indexer_non_unique(target)
        target = self._wrap_reindex_result(target, indexer, preserve_names)
        return (target, indexer)

    def _wrap_reindex_result(self, target, indexer, preserve_names):
        target = self._maybe_preserve_names(target, preserve_names)
        return target

    def _maybe_preserve_names(self, target, preserve_names):
        if preserve_names and target.nlevels == 1 and (target.name != self.name):
            target = target.copy(deep=False)
            target.name = self.name
        return target

    @final
    def _reindex_non_unique(self, target):
        """
        Create a new index with target's values (move/add/delete values as
        necessary) use with non-unique Index and a possibly non-unique target.

        Parameters
        ----------
        target : an iterable

        Returns
        -------
        new_index : pd.Index
            Resulting index.
        indexer : np.ndarray[np.intp]
            Indices of output values in original index.
        new_indexer : np.ndarray[np.intp] or None

        """
        target = ensure_index(target)
        if len(target) == 0:
            return (self[:0], np.array([], dtype=np.intp), None)
        indexer, missing = self.get_indexer_non_unique(target)
        check = indexer != -1
        new_labels = self.take(indexer[check])
        new_indexer = None
        if len(missing):
            length = np.arange(len(indexer), dtype=np.intp)
            missing = ensure_platform_int(missing)
            missing_labels = target.take(missing)
            missing_indexer = length[~check]
            cur_labels = self.take(indexer[check]).values
            cur_indexer = length[check]
            new_labels = np.empty((len(indexer),), dtype=object)
            new_labels[cur_indexer] = cur_labels
            new_labels[missing_indexer] = missing_labels
            if not len(self):
                new_indexer = np.arange(0, dtype=np.intp)
            elif target.is_unique:
                new_indexer = np.arange(len(indexer), dtype=np.intp)
                new_indexer[cur_indexer] = np.arange(len(cur_labels))
                new_indexer[missing_indexer] = -1
            else:
                indexer[~check] = -1
                new_indexer = np.arange(len(self.take(indexer)), dtype=np.intp)
                new_indexer[~check] = -1
        if not isinstance(self, ABCMultiIndex):
            new_index = Index(new_labels, name=self.name)
        else:
            new_index = type(self).from_tuples(new_labels, names=self.names)
        return (new_index, indexer, new_indexer)

    @overload
    def join(self, other, *, how=..., level=..., return_indexers, sort=...):
        ...

    @overload
    def join(self, other, *, how=..., level=..., return_indexers=..., sort=...):
        ...

    @overload
    def join(self, other, *, how=..., level=..., return_indexers=..., sort=...):
        ...

    @final
    @_maybe_return_indexers
    def join(self, other, *, how='left', level=None, return_indexers=False, sort=False):
        """
        Compute join_index and indexers to conform data structures to the new index.

        Parameters
        ----------
        other : Index
            The other index on which join is performed.
        how : {'left', 'right', 'inner', 'outer'}
        level : int or level name, default None
            It is either the integer position or the name of the level.
        return_indexers : bool, default False
            Whether to return the indexers or not for both the index objects.
        sort : bool, default False
            Sort the join keys lexicographically in the result Index. If False,
            the order of the join keys depends on the join type (how keyword).

        Returns
        -------
        join_index, (left_indexer, right_indexer)
            The new index.

        See Also
        --------
        DataFrame.join : Join columns with `other` DataFrame either on index
            or on a key.
        DataFrame.merge : Merge DataFrame or named Series objects with a
            database-style join.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx2 = pd.Index([4, 5, 6])
        >>> idx1.join(idx2, how="outer")
        Index([1, 2, 3, 4, 5, 6], dtype='int64')
        >>> idx1.join(other=idx2, how="outer", return_indexers=True)
        (Index([1, 2, 3, 4, 5, 6], dtype='int64'),
        array([ 0,  1,  2, -1, -1, -1]), array([-1, -1, -1,  0,  1,  2]))
        """
        other = ensure_index(other)
        sort = sort or how == 'outer'
        if isinstance(self, ABCDatetimeIndex) and isinstance(other, ABCDatetimeIndex):
            if (self.tz is None) ^ (other.tz is None):
                raise TypeError('Cannot join tz-naive with tz-aware DatetimeIndex')
        if not self._is_multi and (not other._is_multi):
            pself, pother = self._maybe_downcast_for_indexing(other)
            if pself is not self or pother is not other:
                return pself.join(pother, how=how, level=level, return_indexers=True, sort=sort)
        if level is None and (self._is_multi or other._is_multi):
            if self.names == other.names:
                pass
            else:
                return self._join_multi(other, how=how)
        if level is not None and (self._is_multi or other._is_multi):
            return self._join_level(other, level, how=how)
        if len(self) == 0 or len(other) == 0:
            try:
                return self._join_empty(other, how, sort)
            except TypeError:
                pass
        if self.dtype != other.dtype:
            dtype = self._find_common_type_compat(other)
            this = self.astype(dtype, copy=False)
            other = other.astype(dtype, copy=False)
            return this.join(other, how=how, return_indexers=True)
        elif isinstance(self, ABCCategoricalIndex) and isinstance(other, ABCCategoricalIndex) and (not self.ordered) and (not self.categories.equals(other.categories)):
            other = Index(other._values.reorder_categories(self.categories))
        _validate_join_method(how)
        if self.is_monotonic_increasing and other.is_monotonic_increasing and self._can_use_libjoin and other._can_use_libjoin and (self.is_unique or other.is_unique):
            try:
                return self._join_monotonic(other, how=how)
            except TypeError:
                pass
        elif not self.is_unique or not other.is_unique:
            return self._join_non_unique(other, how=how, sort=sort)
        return self._join_via_get_indexer(other, how, sort)

    def _join_empty(self, other, how, sort):
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
    def _join_via_get_indexer(self, other, how, sort):
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
    def _join_multi(self, other, how):
        from pandas.core.indexes.multi import MultiIndex
        from pandas.core.reshape.merge import restore_dropped_levels_multijoin
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
        level = other.names.index(jl)
        result = self._join_level(other, level, how=how)
        if flip_order:
            return (result[0], result[2], result[1])
        return result

    @final
    def _join_non_unique(self, other, how='left', sort=False):
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
    def _join_level(self, other, level, how='left', keep_order=True):
        """
        The join method *only* affects the level of the resulting
        MultiIndex. Otherwise it just exactly aligns the Index data to the
        labels of the level in the MultiIndex.

        If ```keep_order == True```, the order of the data indexed by the
        MultiIndex will not be changed; otherwise, it will tie out
        with `other`.
        """
        from pandas.core.indexes.multi import MultiIndex

        def _get_leaf_sorter(labels):
            """
            Returns sorter for the inner most level while preserving the
            order of higher levels.

            Parameters
            ----------
            labels : list[np.ndarray]
                Each ndarray has signed integer dtype, not necessarily identical.

            Returns
            -------
            np.ndarray[np.intp]
            """
            if labels[0].size == 0:
                return np.empty(0, dtype=np.intp)
            if len(labels) == 1:
                return get_group_index_sorter(ensure_platform_int(labels[0]))
            tic = labels[0][:-1] != labels[0][1:]
            for lab in labels[1:-1]:
                tic |= lab[:-1] != lab[1:]
            starts = np.hstack(([True], tic, [True])).nonzero()[0]
            lab = ensure_int64(labels[-1])
            return lib.get_level_sorter(lab, ensure_platform_int(starts))
        if isinstance(self, MultiIndex) and isinstance(other, MultiIndex):
            raise TypeError('Join on level between two MultiIndex objects is ambiguous')
        left, right = (self, other)
        flip_order = not isinstance(self, MultiIndex)
        if flip_order:
            left, right = (right, left)
            flip = {'right': 'left', 'left': 'right'}
            how = flip.get(how, how)
        assert isinstance(left, MultiIndex)
        level = left._get_level_number(level)
        old_level = left.levels[level]
        if not right.is_unique:
            raise NotImplementedError('Index._join_level on non-unique index is not implemented')
        new_level, left_lev_indexer, right_lev_indexer = old_level.join(right, how=how, return_indexers=True)
        if left_lev_indexer is None:
            if keep_order or len(left) == 0:
                left_indexer = None
                join_index = left
            else:
                left_indexer = _get_leaf_sorter(left.codes[:level + 1])
                join_index = left[left_indexer]
        else:
            left_lev_indexer = ensure_platform_int(left_lev_indexer)
            rev_indexer = lib.get_reverse_indexer(left_lev_indexer, len(old_level))
            old_codes = left.codes[level]
            taker = old_codes[old_codes != -1]
            new_lev_codes = rev_indexer.take(taker)
            new_codes = list(left.codes)
            new_codes[level] = new_lev_codes
            new_levels = list(left.levels)
            new_levels[level] = new_level
            if keep_order:
                left_indexer = np.arange(len(left), dtype=np.intp)
                left_indexer = cast(np.ndarray, left_indexer)
                mask = new_lev_codes != -1
                if not mask.all():
                    new_codes = [lab[mask] for lab in new_codes]
                    left_indexer = left_indexer[mask]
            elif level == 0:
                max_new_lev = 0 if len(new_lev_codes) == 0 else new_lev_codes.max()
                ngroups = 1 + max_new_lev
                left_indexer, counts = libalgos.groupsort_indexer(new_lev_codes, ngroups)
                left_indexer = left_indexer[counts[0]:]
                new_codes = [lab[left_indexer] for lab in new_codes]
            else:
                mask = new_lev_codes != -1
                mask_all = mask.all()
                if not mask_all:
                    new_codes = [lab[mask] for lab in new_codes]
                left_indexer = _get_leaf_sorter(new_codes[:level + 1])
                new_codes = [lab[left_indexer] for lab in new_codes]
                if not mask_all:
                    left_indexer = mask.nonzero()[0][left_indexer]
            join_index = MultiIndex(levels=new_levels, codes=new_codes, names=left.names, verify_integrity=False)
        if right_lev_indexer is not None:
            right_indexer = right_lev_indexer.take(join_index.codes[level])
        else:
            right_indexer = join_index.codes[level]
        if flip_order:
            left_indexer, right_indexer = (right_indexer, left_indexer)
        left_indexer = None if left_indexer is None else ensure_platform_int(left_indexer)
        right_indexer = None if right_indexer is None else ensure_platform_int(right_indexer)
        return (join_index, left_indexer, right_indexer)

    def _join_monotonic(self, other, how='left'):
        assert other.dtype == self.dtype
        assert self._can_use_libjoin and other._can_use_libjoin
        if self.equals(other):
            ret_index = other if how == 'right' else self
            return (ret_index, None, None)
        if how == 'left':
            if other.is_unique:
                join_index = self
                lidx = None
                ridx = self._left_indexer_unique(other)
            else:
                join_array, lidx, ridx = self._left_indexer(other)
                join_index, lidx, ridx = self._wrap_join_result(join_array, other, lidx, ridx, how)
        elif how == 'right':
            if self.is_unique:
                join_index = other
                lidx = other._left_indexer_unique(self)
                ridx = None
            else:
                join_array, ridx, lidx = other._left_indexer(self)
                join_index, lidx, ridx = self._wrap_join_result(join_array, other, lidx, ridx, how)
        elif how == 'inner':
            join_array, lidx, ridx = self._inner_indexer(other)
            join_index, lidx, ridx = self._wrap_join_result(join_array, other, lidx, ridx, how)
        elif how == 'outer':
            join_array, lidx, ridx = self._outer_indexer(other)
            join_index, lidx, ridx = self._wrap_join_result(join_array, other, lidx, ridx, how)
        lidx = None if lidx is None else ensure_platform_int(lidx)
        ridx = None if ridx is None else ensure_platform_int(ridx)
        return (join_index, lidx, ridx)

    def _wrap_join_result(self, joined, other, lidx, ridx, how):
        assert other.dtype == self.dtype
        if lidx is not None and lib.is_range_indexer(lidx, len(self)):
            lidx = None
        if ridx is not None and lib.is_range_indexer(ridx, len(other)):
            ridx = None
        if lidx is None:
            join_index = self
        elif ridx is None:
            join_index = other
        else:
            join_index = self._constructor._with_infer(joined, dtype=self.dtype)
        names = other.names if how == 'right' else self.names
        if join_index.names != names:
            join_index = join_index.set_names(names)
        return (join_index, lidx, ridx)

    @final
    @cache_readonly
    def _can_use_libjoin(self):
        """
        Whether we can use the fastpaths implemented in _libs.join.

        This is driven by whether (in monotonic increasing cases that are
        guaranteed not to have NAs) we can convert to a np.ndarray without
        making a copy. If we cannot, this negates the performance benefit
        of using libjoin.
        """
        if not self.is_monotonic_increasing:
            return False
        if type(self) is Index:
            return isinstance(self.dtype, np.dtype) or isinstance(self._values, (ArrowExtensionArray, BaseMaskedArray)) or (isinstance(self.dtype, StringDtype) and self.dtype.storage == 'python')
        return not isinstance(self, (ABCIntervalIndex, ABCMultiIndex))

    @property
    def values(self):
        """
        Return an array representing the data in the Index.

        .. warning::

           We recommend using :attr:`Index.array` or
           :meth:`Index.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        array: numpy.ndarray or ExtensionArray

        See Also
        --------
        Index.array : Reference to the underlying data.
        Index.to_numpy : A NumPy array representing the underlying data.

        Examples
        --------
        For :class:`pandas.Index`:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.values
        array([1, 2, 3])

        For :class:`pandas.IntervalIndex`:

        >>> idx = pd.interval_range(start=0, end=5)
        >>> idx.values
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]]
        Length: 5, dtype: interval[int64, right]
        """
        data = self._data
        if isinstance(data, np.ndarray):
            data = data.view()
            data.flags.writeable = False
        return data

    @cache_readonly
    @doc(IndexOpsMixin.array)
    def array(self):
        array = self._data
        if isinstance(array, np.ndarray):
            from pandas.core.arrays.numpy_ import NumpyExtensionArray
            array = NumpyExtensionArray(array)
        return array

    @property
    def _values(self):
        """
        The best array representation.

        This is an ndarray or ExtensionArray.

        ``_values`` are consistent between ``Series`` and ``Index``.

        It may differ from the public '.values' method.

        index             | values          | _values       |
        ----------------- | --------------- | ------------- |
        Index             | ndarray         | ndarray       |
        CategoricalIndex  | Categorical     | Categorical   |
        DatetimeIndex     | ndarray[M8ns]   | DatetimeArray |
        DatetimeIndex[tz] | ndarray[M8ns]   | DatetimeArray |
        PeriodIndex       | ndarray[object] | PeriodArray   |
        IntervalIndex     | IntervalArray   | IntervalArray |

        See Also
        --------
        values : Values
        """
        return self._data

    def _get_engine_target(self):
        """
        Get the ndarray or ExtensionArray that we can pass to the IndexEngine
        constructor.
        """
        vals = self._values
        if isinstance(vals, StringArray):
            return vals._ndarray
        if isinstance(vals, ArrowExtensionArray) and self.dtype.kind in 'Mm':
            import pyarrow as pa
            pa_type = vals._pa_array.type
            if pa.types.is_timestamp(pa_type):
                vals = vals._to_datetimearray()
                return vals._ndarray.view('i8')
            elif pa.types.is_duration(pa_type):
                vals = vals._to_timedeltaarray()
                return vals._ndarray.view('i8')
        if type(self) is Index and isinstance(self._values, ExtensionArray) and (not isinstance(self._values, BaseMaskedArray)) and (not (isinstance(self._values, ArrowExtensionArray) and is_numeric_dtype(self.dtype) and (self.dtype.kind != 'O'))):
            return self._values.astype(object)
        return vals

    @final
    def _get_join_target(self):
        """
        Get the ndarray or ExtensionArray that we can pass to the join
        functions.
        """
        if isinstance(self._values, BaseMaskedArray):
            return self._values._data
        elif isinstance(self._values, ArrowExtensionArray):
            return self._values.to_numpy()
        target = self._get_engine_target()
        if not isinstance(target, np.ndarray):
            raise ValueError('_can_use_libjoin should return False.')
        return target

    def _from_join_target(self, result):
        """
        Cast the ndarray returned from one of the libjoin.foo_indexer functions
        back to type(self._data).
        """
        if isinstance(self.values, BaseMaskedArray):
            return type(self.values)(result, np.zeros(result.shape, dtype=np.bool_))
        elif isinstance(self.values, (ArrowExtensionArray, StringArray)):
            return type(self.values)._from_sequence(result, dtype=self.dtype)
        return result

    @doc(IndexOpsMixin._memory_usage)
    def memory_usage(self, deep=False):
        result = self._memory_usage(deep=deep)
        if '_engine' in self._cache:
            result += self._engine.sizeof(deep=deep)
        return result

    @final
    def where(self, cond, other=None):
        """
        Replace values where the condition is False.

        The replacement is taken from other.

        Parameters
        ----------
        cond : bool array-like with the same length as self
            Condition to select the values on.
        other : scalar, or array-like, default None
            Replacement if the condition is False.

        Returns
        -------
        pandas.Index
            A copy of self with values replaced from other
            where the condition is False.

        See Also
        --------
        Series.where : Same method for Series.
        DataFrame.where : Same method for DataFrame.

        Examples
        --------
        >>> idx = pd.Index(["car", "bike", "train", "tractor"])
        >>> idx
        Index(['car', 'bike', 'train', 'tractor'], dtype='object')
        >>> idx.where(idx.isin(["car", "train"]), "other")
        Index(['car', 'other', 'train', 'other'], dtype='object')
        """
        if isinstance(self, ABCMultiIndex):
            raise NotImplementedError('.where is not supported for MultiIndex operations')
        cond = np.asarray(cond, dtype=bool)
        return self.putmask(~cond, other)

    @final
    @classmethod
    def _raise_scalar_data_error(cls, data):
        raise TypeError(f'{cls.__name__}(...) must be called with a collection of some kind, {(repr(data) if not isinstance(data, np.generic) else str(data))} was passed')

    def _validate_fill_value(self, value):
        """
        Check if the value can be inserted into our array without casting,
        and convert it to an appropriate native type if necessary.

        Raises
        ------
        TypeError
            If the value cannot be inserted into an array of this dtype.
        """
        dtype = self.dtype
        if isinstance(dtype, np.dtype) and dtype.kind not in 'mM':
            try:
                return np_can_hold_element(dtype, value)
            except LossySetitemError as err:
                raise TypeError from err
        elif not can_hold_element(self._values, value):
            raise TypeError
        return value

    @cache_readonly
    def _is_memory_usage_qualified(self):
        """
        Return a boolean if we need a qualified .info display.
        """
        return is_object_dtype(self.dtype) or (is_string_dtype(self.dtype) and self.dtype.storage == 'python')

    def __contains__(self, key):
        """
        Return a boolean indicating whether the provided key is in the index.

        Parameters
        ----------
        key : label
            The key to check if it is present in the index.

        Returns
        -------
        bool
            Whether the key search is in the index.

        Raises
        ------
        TypeError
            If the key is not hashable.

        See Also
        --------
        Index.isin : Returns an ndarray of boolean dtype indicating whether the
            list-like key is in the index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')

        >>> 2 in idx
        True
        >>> 6 in idx
        False
        """
        hash(key)
        try:
            return key in self._engine
        except (OverflowError, TypeError, ValueError):
            return False

    @final
    def __setitem__(self, key, value):
        raise TypeError('Index does not support mutable operations')

    def __getitem__(self, key):
        """
        Override numpy.ndarray's __getitem__ method to work as desired.

        This function adds lists and Series as valid boolean indexers
        (ndarrays only supports ndarray with dtype=bool).

        If resulting ndim != 1, plain ndarray is returned instead of
        corresponding `Index` subclass.

        """
        getitem = self._data.__getitem__
        if is_integer(key) or is_float(key):
            key = com.cast_scalar_indexer(key)
            return getitem(key)
        if isinstance(key, slice):
            return self._getitem_slice(key)
        if com.is_bool_indexer(key):
            if isinstance(getattr(key, 'dtype', None), ExtensionDtype):
                key = key.to_numpy(dtype=bool, na_value=False)
            else:
                key = np.asarray(key, dtype=bool)
            if not isinstance(self.dtype, ExtensionDtype):
                if len(key) == 0 and len(key) != len(self):
                    raise ValueError('The length of the boolean indexer cannot be 0 when the Index has length greater than 0.')
        result = getitem(key)
        if result.ndim > 1:
            disallow_ndim_indexing(result)
        return self._constructor._simple_new(result, name=self._name)

    def _getitem_slice(self, slobj):
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._data[slobj]
        result = type(self)._simple_new(res, name=self._name, refs=self._references)
        if '_engine' in self._cache:
            reverse = slobj.step is not None and slobj.step < 0
            result._engine._update_from_sliced(self._engine, reverse=reverse)
        return result

    @final
    def _can_hold_identifiers_and_holds_name(self, name):
        """
        Faster check for ``name in self`` when we know `name` is a Python
        identifier (e.g. in NDFrame.__getattr__, which hits this to support
        . key lookup). For indexes that can't hold identifiers (everything
        but object & categorical) we just return False.

        https://github.com/pandas-dev/pandas/issues/19764
        """
        if is_object_dtype(self.dtype) or is_string_dtype(self.dtype) or isinstance(self.dtype, CategoricalDtype):
            return name in self
        return False

    def append(self, other):
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices
            Single Index or a collection of indices, which can be either a list or a
            tuple.

        Returns
        -------
        Index
            Returns a new Index object resulting from appending the provided other
            indices to the original Index.

        See Also
        --------
        Index.insert : Make new Index inserting new item at location.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.append(pd.Index([4]))
        Index([1, 2, 3, 4], dtype='int64')
        """
        to_concat = [self]
        if isinstance(other, (list, tuple)):
            to_concat += list(other)
        else:
            to_concat.append(other)
        for obj in to_concat:
            if not isinstance(obj, Index):
                raise TypeError('all inputs must be Index')
        names = {obj.name for obj in to_concat}
        name = None if len(names) > 1 else self.name
        return self._concat(to_concat, name)

    def _concat(self, to_concat, name):
        """
        Concatenate multiple Index objects.
        """
        to_concat_vals = [x._values for x in to_concat]
        result = concat_compat(to_concat_vals)
        return Index._with_infer(result, name=name)

    def putmask(self, mask, value):
        """
        Return a new Index of the values set with the mask.

        Parameters
        ----------
        mask : np.ndarray[bool]
            Array of booleans denoting where values in the original
            data are not ``NA``.
        value : scalar
            Scalar value to use to fill holes (e.g. 0).
            This value cannot be a list-likes.

        Returns
        -------
        Index
            A new Index of the values set with the mask.

        See Also
        --------
        numpy.putmask : Changes elements of an array
            based on conditional and input values.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx2 = pd.Index([5, 6, 7])
        >>> idx1.putmask([True, False, False], idx2)
        Index([5, 2, 3], dtype='int64')
        """
        mask, noop = validate_putmask(self._values, mask)
        if noop:
            return self.copy()
        if self.dtype != object and is_valid_na_for_dtype(value, self.dtype):
            value = self._na_value
        try:
            converted = self._validate_fill_value(value)
        except (LossySetitemError, ValueError, TypeError) as err:
            if is_object_dtype(self.dtype):
                raise err
            dtype = self._find_common_type_compat(value)
            return self.astype(dtype).putmask(mask, value)
        values = self._values.copy()
        if isinstance(values, np.ndarray):
            converted = setitem_datetimelike_compat(values, mask.sum(), converted)
            np.putmask(values, mask, converted)
        else:
            values._putmask(mask, value)
        return self._shallow_copy(values)

    def equals(self, other):
        """
        Determine if two Index object are equal.

        The things that are being compared are:

        * The elements inside the Index object.
        * The order of the elements inside the Index object.

        Parameters
        ----------
        other : Any
            The other object to compare against.

        Returns
        -------
        bool
            True if "other" is an Index and it has the same elements and order
            as the calling index; False otherwise.

        See Also
        --------
        Index.identical: Checks that object attributes and types are also equal.
        Index.has_duplicates: Check if the Index has duplicate values.
        Index.is_unique: Return if the index has unique values.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx1
        Index([1, 2, 3], dtype='int64')
        >>> idx1.equals(pd.Index([1, 2, 3]))
        True

        The elements inside are compared

        >>> idx2 = pd.Index(["1", "2", "3"])
        >>> idx2
        Index(['1', '2', '3'], dtype='object')

        >>> idx1.equals(idx2)
        False

        The order is compared

        >>> ascending_idx = pd.Index([1, 2, 3])
        >>> ascending_idx
        Index([1, 2, 3], dtype='int64')
        >>> descending_idx = pd.Index([3, 2, 1])
        >>> descending_idx
        Index([3, 2, 1], dtype='int64')
        >>> ascending_idx.equals(descending_idx)
        False

        The dtype is *not* compared

        >>> int64_idx = pd.Index([1, 2, 3], dtype="int64")
        >>> int64_idx
        Index([1, 2, 3], dtype='int64')
        >>> uint64_idx = pd.Index([1, 2, 3], dtype="uint64")
        >>> uint64_idx
        Index([1, 2, 3], dtype='uint64')
        >>> int64_idx.equals(uint64_idx)
        True
        """
        if self.is_(other):
            return True
        if not isinstance(other, Index):
            return False
        if len(self) != len(other):
            return False
        if isinstance(self.dtype, StringDtype) and self.dtype.na_value is np.nan and (other.dtype != self.dtype):
            return other.equals(self.astype(object))
        if is_object_dtype(self.dtype) and (not is_object_dtype(other.dtype)):
            return other.equals(self)
        if isinstance(other, ABCMultiIndex):
            return other.equals(self)
        if isinstance(self._values, ExtensionArray):
            if not isinstance(other, type(self)):
                return False
            earr = cast(ExtensionArray, self._data)
            return earr.equals(other._data)
        if isinstance(other.dtype, ExtensionDtype):
            return other.equals(self)
        return array_equivalent(self._values, other._values)

    @final
    def identical(self, other):
        """
        Similar to equals, but checks that object attributes and types are also equal.

        Parameters
        ----------
        other : Index
            The Index object you want to compare with the current Index object.

        Returns
        -------
        bool
            If two Index objects have equal elements and same type True,
            otherwise False.

        See Also
        --------
        Index.equals: Determine if two Index object are equal.
        Index.has_duplicates: Check if the Index has duplicate values.
        Index.is_unique: Return if the index has unique values.

        Examples
        --------
        >>> idx1 = pd.Index(["1", "2", "3"])
        >>> idx2 = pd.Index(["1", "2", "3"])
        >>> idx2.identical(idx1)
        True

        >>> idx1 = pd.Index(["1", "2", "3"], name="A")
        >>> idx2 = pd.Index(["1", "2", "3"], name="B")
        >>> idx2.identical(idx1)
        False
        """
        return self.equals(other) and all((getattr(self, c, None) == getattr(other, c, None) for c in self._comparables)) and (type(self) == type(other)) and (self.dtype == other.dtype)

    @final
    def asof(self, label):
        """
        Return the label from the index, or, if not present, the previous one.

        Assuming that the index is sorted, return the passed index label if it
        is in the index, or return the previous index label if the passed one
        is not in the index.

        Parameters
        ----------
        label : object
            The label up to which the method returns the latest index label.

        Returns
        -------
        object
            The passed label if it is in the index. The previous label if the
            passed label is not in the sorted index or `NaN` if there is no
            such label.

        See Also
        --------
        Series.asof : Return the latest value in a Series up to the
            passed index.
        merge_asof : Perform an asof merge (similar to left join but it
            matches on nearest key rather than equal key).
        Index.get_loc : An `asof` is a thin wrapper around `get_loc`
            with method='pad'.

        Examples
        --------
        `Index.asof` returns the latest index label up to the passed label.

        >>> idx = pd.Index(["2013-12-31", "2014-01-02", "2014-01-03"])
        >>> idx.asof("2014-01-01")
        '2013-12-31'

        If the label is in the index, the method returns the passed label.

        >>> idx.asof("2014-01-02")
        '2014-01-02'

        If all of the labels in the index are later than the passed label,
        NaN is returned.

        >>> idx.asof("1999-01-02")
        nan

        If the index is not sorted, an error is raised.

        >>> idx_not_sorted = pd.Index(["2013-12-31", "2015-01-02", "2014-01-03"])
        >>> idx_not_sorted.asof("2013-12-31")
        Traceback (most recent call last):
        ValueError: index must be monotonic increasing or decreasing
        """
        self._searchsorted_monotonic(label)
        try:
            loc = self.get_loc(label)
        except (KeyError, TypeError) as err:
            indexer = self.get_indexer([label], method='pad')
            if indexer.ndim > 1 or indexer.size > 1:
                raise TypeError('asof requires scalar valued input') from err
            loc = indexer.item()
            if loc == -1:
                return self._na_value
        else:
            if isinstance(loc, slice):
                loc = loc.indices(len(self))[-1]
        return self[loc]

    def asof_locs(self, where, mask):
        """
        Return the locations (indices) of labels in the index.

        As in the :meth:`pandas.Index.asof`, if the label (a particular entry in
        ``where``) is not in the index, the latest index label up to the
        passed label is chosen and its index returned.

        If all of the labels in the index are later than a label in ``where``,
        -1 is returned.

        ``mask`` is used to ignore ``NA`` values in the index during calculation.

        Parameters
        ----------
        where : Index
            An Index consisting of an array of timestamps.
        mask : np.ndarray[bool]
            Array of booleans denoting where values in the original
            data are not ``NA``.

        Returns
        -------
        np.ndarray[np.intp]
            An array of locations (indices) of the labels from the index
            which correspond to the return values of :meth:`pandas.Index.asof`
            for every element in ``where``.

        See Also
        --------
        Index.asof : Return the label from the index, or, if not present, the
            previous one.

        Examples
        --------
        >>> idx = pd.date_range("2023-06-01", periods=3, freq="D")
        >>> where = pd.DatetimeIndex(
        ...     ["2023-05-30 00:12:00", "2023-06-01 00:00:00", "2023-06-02 23:59:59"]
        ... )
        >>> mask = np.ones(3, dtype=bool)
        >>> idx.asof_locs(where, mask)
        array([-1,  0,  1])

        We can use ``mask`` to ignore certain values in the index during calculation.

        >>> mask[1] = False
        >>> idx.asof_locs(where, mask)
        array([-1,  0,  0])
        """
        locs = self._values[mask].searchsorted(where._values, side='right')
        locs = np.where(locs > 0, locs - 1, 0)
        result = np.arange(len(self), dtype=np.intp)[mask].take(locs)
        first_value = self._values[mask.argmax()]
        result[(locs == 0) & (where._values < first_value)] = -1
        return result

    @overload
    def sort_values(self, *, return_indexer=..., ascending=..., na_position=..., key=...):
        ...

    @overload
    def sort_values(self, *, return_indexer, ascending=..., na_position=..., key=...):
        ...

    @overload
    def sort_values(self, *, return_indexer=..., ascending=..., na_position=..., key=...):
        ...

    def sort_values(self, *, return_indexer=False, ascending=True, na_position='last', key=None):
        """
        Return a sorted copy of the index.

        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

        Returns
        -------
        sorted_index : pandas.Index
            Sorted copy of the index.
        indexer : numpy.ndarray, optional
            The indices that the index itself was sorted by.

        See Also
        --------
        Series.sort_values : Sort values of a Series.
        DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([10, 100, 1, 1000])
        >>> idx
        Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2]))
        """
        if key is None and (ascending and self.is_monotonic_increasing or (not ascending and self.is_monotonic_decreasing)):
            if return_indexer:
                indexer = np.arange(len(self), dtype=np.intp)
                return (self.copy(), indexer)
            else:
                return self.copy()
        if not isinstance(self, ABCMultiIndex):
            _as = nargsort(items=self, ascending=ascending, na_position=na_position, key=key)
        else:
            idx = cast(Index, ensure_key_mapped(self, key))
            _as = idx.argsort(na_position=na_position)
            if not ascending:
                _as = _as[::-1]
        sorted_index = self.take(_as)
        if return_indexer:
            return (sorted_index, _as)
        else:
            return sorted_index

    def shift(self, periods=1, freq=None):
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or str, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.Index
            Shifted index.

        See Also
        --------
        Series.shift : Shift values of Series.

        Notes
        -----
        This method is only implemented for datetime-like index classes,
        i.e., DatetimeIndex, PeriodIndex and TimedeltaIndex.

        Examples
        --------
        Put the first 5 month starts of 2011 into an index.

        >>> month_starts = pd.date_range("1/1/2011", periods=5, freq="MS")
        >>> month_starts
        DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01',
                       '2011-05-01'],
                      dtype='datetime64[ns]', freq='MS')

        Shift the index by 10 days.

        >>> month_starts.shift(10, freq="D")
        DatetimeIndex(['2011-01-11', '2011-02-11', '2011-03-11', '2011-04-11',
                       '2011-05-11'],
                      dtype='datetime64[ns]', freq=None)

        The default value of `freq` is the `freq` attribute of the index,
        which is 'MS' (month start) in this example.

        >>> month_starts.shift(10)
        DatetimeIndex(['2011-11-01', '2011-12-01', '2012-01-01', '2012-02-01',
                       '2012-03-01'],
                      dtype='datetime64[ns]', freq='MS')
        """
        raise NotImplementedError(f'This method is only implemented for DatetimeIndex, PeriodIndex and TimedeltaIndex; Got type {type(self).__name__}')

    def argsort(self, *args, **kwargs):
        """
        Return the integer indices that would sort the index.

        Parameters
        ----------
        *args
            Passed to `numpy.ndarray.argsort`.
        **kwargs
            Passed to `numpy.ndarray.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Integer indices that would sort the index if used as
            an indexer.

        See Also
        --------
        numpy.argsort : Similar method for NumPy arrays.
        Index.sort_values : Return sorted copy of Index.

        Examples
        --------
        >>> idx = pd.Index(["b", "a", "d", "c"])
        >>> idx
        Index(['b', 'a', 'd', 'c'], dtype='object')

        >>> order = idx.argsort()
        >>> order
        array([1, 0, 3, 2])

        >>> idx[order]
        Index(['a', 'b', 'c', 'd'], dtype='object')
        """
        return self._data.argsort(*args, **kwargs)

    def _check_indexing_error(self, key):
        if not is_scalar(key):
            raise InvalidIndexError(key)

    @cache_readonly
    def _should_fallback_to_positional(self):
        """
        Should an integer key be treated as positional?
        """
        return self.inferred_type not in {'integer', 'mixed-integer', 'floating', 'complex'}
    _index_shared_docs['get_indexer_non_unique'] = "\n        Compute indexer and mask for new index given the current index.\n\n        The indexer should be then used as an input to ndarray.take to align the\n        current data to the new index.\n\n        Parameters\n        ----------\n        target : %(target_klass)s\n            An iterable containing the values to be used for computing indexer.\n\n        Returns\n        -------\n        indexer : np.ndarray[np.intp]\n            Integers from 0 to n - 1 indicating that the index at these\n            positions matches the corresponding target values. Missing values\n            in the target are marked by -1.\n        missing : np.ndarray[np.intp]\n            An indexer into the target of the values not found.\n            These correspond to the -1 in the indexer array.\n\n        See Also\n        --------\n        Index.get_indexer : Computes indexer and mask for new index given\n            the current index.\n        Index.get_indexer_for : Returns an indexer even when non-unique.\n\n        Examples\n        --------\n        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])\n        >>> index.get_indexer_non_unique(['b', 'b'])\n        (array([1, 3, 4, 1, 3, 4]), array([], dtype=int64))\n\n        In the example below there are no matched values.\n\n        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])\n        >>> index.get_indexer_non_unique(['q', 'r', 't'])\n        (array([-1, -1, -1]), array([0, 1, 2]))\n\n        For this reason, the returned ``indexer`` contains only integers equal to -1.\n        It demonstrates that there's no match between the index and the ``target``\n        values at these positions. The mask [0, 1, 2] in the return value shows that\n        the first, second, and third elements are missing.\n\n        Notice that the return value is a tuple contains two items. In the example\n        below the first item is an array of locations in ``index``. The second\n        item is a mask shows that the first and third elements are missing.\n\n        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])\n        >>> index.get_indexer_non_unique(['f', 'b', 's'])\n        (array([-1,  1,  3,  4, -1]), array([0, 2]))\n        "

    @Appender(_index_shared_docs['get_indexer_non_unique'] % _index_doc_kwargs)
    def get_indexer_non_unique(self, target):
        target = self._maybe_cast_listlike_indexer(target)
        if not self._should_compare(target) and (not self._should_partial_index(target)):
            return self._get_indexer_non_comparable(target, method=None, unique=False)
        pself, ptarget = self._maybe_downcast_for_indexing(target)
        if pself is not self or ptarget is not target:
            return pself.get_indexer_non_unique(ptarget)
        if self.dtype != target.dtype:
            dtype = self._find_common_type_compat(target)
            this = self.astype(dtype, copy=False)
            that = target.astype(dtype, copy=False)
            return this.get_indexer_non_unique(that)
        if self._is_multi and target._is_multi:
            engine = self._engine
            tgt_values = engine._extract_level_codes(target)
        else:
            tgt_values = target._get_engine_target()
        indexer, missing = self._engine.get_indexer_non_unique(tgt_values)
        return (ensure_platform_int(indexer), ensure_platform_int(missing))

    @final
    def get_indexer_for(self, target):
        """
        Guaranteed return of an indexer even when non-unique.

        This dispatches to get_indexer or get_indexer_non_unique
        as appropriate.

        Parameters
        ----------
        target : Index
            An iterable containing the values to be used for computing indexer.

        Returns
        -------
        np.ndarray[np.intp]
            List of indices.

        See Also
        --------
        Index.get_indexer : Computes indexer and mask for new index given
            the current index.
        Index.get_non_unique : Returns indexer and masks for new index given
            the current index.

        Examples
        --------
        >>> idx = pd.Index([np.nan, "var1", np.nan])
        >>> idx.get_indexer_for([np.nan])
        array([0, 2])
        """
        if self._index_as_unique:
            return self.get_indexer(target)
        indexer, _ = self.get_indexer_non_unique(target)
        return indexer

    def _get_indexer_strict(self, key, axis_name):
        """
        Analogue to get_indexer that raises if any elements are missing.
        """
        keyarr = key
        if not isinstance(keyarr, Index):
            keyarr = com.asarray_tuplesafe(keyarr)
        if self._index_as_unique:
            indexer = self.get_indexer_for(keyarr)
            keyarr = self.reindex(keyarr)[0]
        else:
            keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
        self._raise_if_missing(keyarr, indexer, axis_name)
        keyarr = self.take(indexer)
        if isinstance(key, Index):
            keyarr.name = key.name
        if lib.is_np_dtype(keyarr.dtype, 'mM') or isinstance(keyarr.dtype, DatetimeTZDtype):
            if isinstance(key, list) or (isinstance(key, type(self)) and key.freq is None):
                keyarr = keyarr._with_freq(None)
        return (keyarr, indexer)

    def _raise_if_missing(self, key, indexer, axis_name):
        """
        Check that indexer can be used to return a result.

        e.g. at least one element was found,
        unless the list of keys was actually empty.

        Parameters
        ----------
        key : list-like
            Targeted labels (only used to show correct error message).
        indexer: array-like of booleans
            Indices corresponding to the key,
            (with -1 indicating not found).
        axis_name : str

        Raises
        ------
        KeyError
            If at least one key was requested but none was found.
        """
        if len(key) == 0:
            return
        missing_mask = indexer < 0
        nmissing = missing_mask.sum()
        if nmissing:
            if nmissing == len(indexer):
                raise KeyError(f'None of [{key}] are in the [{axis_name}]')
            not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
            raise KeyError(f'{not_found} not in index')

    @overload
    def _get_indexer_non_comparable(self, target, method, unique=...):
        ...

    @overload
    def _get_indexer_non_comparable(self, target, method, unique):
        ...

    @overload
    def _get_indexer_non_comparable(self, target, method, unique=True):
        ...

    @final
    def _get_indexer_non_comparable(self, target, method, unique=True):
        """
        Called from get_indexer or get_indexer_non_unique when the target
        is of a non-comparable dtype.

        For get_indexer lookups with method=None, get_indexer is an _equality_
        check, so non-comparable dtypes mean we will always have no matches.

        For get_indexer lookups with a method, get_indexer is an _inequality_
        check, so non-comparable dtypes mean we will always raise TypeError.

        Parameters
        ----------
        target : Index
        method : str or None
        unique : bool, default True
            * True if called from get_indexer.
            * False if called from get_indexer_non_unique.

        Raises
        ------
        TypeError
            If doing an inequality check, i.e. method is not None.
        """
        if method is not None:
            other_dtype = _unpack_nested_dtype(target)
            raise TypeError(f'Cannot compare dtypes {self.dtype} and {other_dtype}')
        no_matches = -1 * np.ones(target.shape, dtype=np.intp)
        if unique:
            return no_matches
        else:
            missing = np.arange(len(target), dtype=np.intp)
            return (no_matches, missing)

    @property
    def _index_as_unique(self):
        """
        Whether we should treat this as unique for the sake of
        get_indexer vs get_indexer_non_unique.

        For IntervalIndex compat.
        """
        return self.is_unique
    _requires_unique_msg = 'Reindexing only valid with uniquely valued Index objects'

    @final
    def _maybe_downcast_for_indexing(self, other):
        """
        When dealing with an object-dtype Index and a non-object Index, see
        if we can upcast the object-dtype one to improve performance.
        """
        if isinstance(self, ABCDatetimeIndex) and isinstance(other, ABCDatetimeIndex):
            if self.tz is not None and other.tz is not None and (not tz_compare(self.tz, other.tz)):
                return (self.tz_convert('UTC'), other.tz_convert('UTC'))
        elif self.inferred_type == 'date' and isinstance(other, ABCDatetimeIndex):
            try:
                return (type(other)(self), other)
            except OutOfBoundsDatetime:
                return (self, other)
        elif self.inferred_type == 'timedelta' and isinstance(other, ABCTimedeltaIndex):
            return (type(other)(self), other)
        elif self.dtype.kind == 'u' and other.dtype.kind == 'i':
            if other.min() >= 0:
                return (self, other.astype(self.dtype))
        elif self._is_multi and (not other._is_multi):
            try:
                other = type(self).from_tuples(other)
            except (TypeError, ValueError):
                self = Index(self._values)
        if not is_object_dtype(self.dtype) and is_object_dtype(other.dtype):
            other, self = other._maybe_downcast_for_indexing(self)
        return (self, other)

    @final
    def _find_common_type_compat(self, target):
        """
        Implementation of find_common_type that adjusts for Index-specific
        special cases.
        """
        target_dtype, _ = infer_dtype_from(target)
        if self.dtype == 'uint64' or target_dtype == 'uint64':
            if is_signed_integer_dtype(self.dtype) or is_signed_integer_dtype(target_dtype):
                return _dtype_obj
        dtype = find_result_type(self.dtype, target)
        dtype = common_dtype_categorical_compat([self, target], dtype)
        return dtype

    @final
    def _should_compare(self, other):
        """
        Check if `self == other` can ever have non-False entries.
        """
        if other.inferred_type == 'boolean' and is_any_real_numeric_dtype(self.dtype) or (self.inferred_type == 'boolean' and is_any_real_numeric_dtype(other.dtype)):
            return False
        dtype = _unpack_nested_dtype(other)
        return self._is_comparable_dtype(dtype) or is_object_dtype(dtype) or is_string_dtype(dtype)

    def _is_comparable_dtype(self, dtype):
        """
        Can we compare values of the given dtype to our own?
        """
        if self.dtype.kind == 'b':
            return dtype.kind == 'b'
        elif is_numeric_dtype(self.dtype):
            return is_numeric_dtype(dtype)
        return True

    @final
    def groupby(self, values):
        """
        Group the index labels by a given array of values.

        Parameters
        ----------
        values : array
            Values used to determine the groups.

        Returns
        -------
        dict
            {group name -> group labels}
        """
        if isinstance(values, ABCMultiIndex):
            values = values._values
        values = Categorical(values)
        result = values._reverse_indexer()
        result = {k: self.take(v) for k, v in result.items()}
        return PrettyDict(result)

    def map(self, mapper, na_action=None):
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Union[Index, MultiIndex]
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.

        See Also
        --------
        Index.where : Replace values where the condition is False.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.map({1: "a", 2: "b", 3: "c"})
        Index(['a', 'b', 'c'], dtype='object')

        Using `map` with a function:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx.map("I am a {}".format)
        Index(['I am a 1', 'I am a 2', 'I am a 3'], dtype='object')

        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.map(lambda x: x.upper())
        Index(['A', 'B', 'C'], dtype='object')
        """
        from pandas.core.indexes.multi import MultiIndex
        new_values = self._map_values(mapper, na_action=na_action)
        if new_values.size and isinstance(new_values[0], tuple):
            if isinstance(self, MultiIndex):
                names = self.names
            elif self.name:
                names = [self.name] * len(new_values[0])
            else:
                names = None
            return MultiIndex.from_tuples(new_values, names=names)
        dtype = None
        if not new_values.size:
            dtype = self.dtype
        same_dtype = lib.infer_dtype(new_values, skipna=False) == self.inferred_type
        if same_dtype:
            new_values = maybe_cast_pointwise_result(new_values, self.dtype, same_dtype=same_dtype)
        return Index._with_infer(new_values, dtype=dtype, copy=False, name=self.name)

    @final
    def _transform_index(self, func, *, level=None):
        """
        Apply function to all values found in index.

        This includes transforming multiindex entries separately.
        Only apply function to one level of the MultiIndex if level is specified.
        """
        if isinstance(self, ABCMultiIndex):
            values = [self.get_level_values(i).map(func) if i == level or level is None else self.get_level_values(i) for i in range(self.nlevels)]
            return type(self).from_arrays(values)
        else:
            items = [func(x) for x in self]
            return Index(items, name=self.name, tupleize_cols=False)

    def isin(self, values, level=None):
        """
        Return a boolean array where the index values are in `values`.

        Compute boolean array of whether each index value is found in the
        passed set of values. The length of the returned boolean array matches
        the length of the index.

        Parameters
        ----------
        values : set or list-like
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index is a
            `MultiIndex`).

        Returns
        -------
        np.ndarray[bool]
            NumPy array of boolean values.

        See Also
        --------
        Series.isin : Same for Series.
        DataFrame.isin : Same method for DataFrames.

        Notes
        -----
        In the case of `MultiIndex` you must either specify `values` as a
        list-like object containing tuples that are the same length as the
        number of levels, or specify `level`. Otherwise it will raise a
        ``ValueError``.

        If `level` is specified:

        - if it is the name of one *and only one* index level, use that level;
        - otherwise it should be a number indicating level position.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.

        >>> idx.isin([1, 4])
        array([ True, False, False])

        >>> midx = pd.MultiIndex.from_arrays(
        ...     [[1, 2, 3], ["red", "blue", "green"]], names=["number", "color"]
        ... )
        >>> midx
        MultiIndex([(1,   'red'),
                    (2,  'blue'),
                    (3, 'green')],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> midx.isin(["red", "orange", "yellow"], level="color")
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> midx.isin([(1, "red"), (3, "red")])
        array([ True, False, False])
        """
        if level is not None:
            self._validate_index_level(level)
        return algos.isin(self._values, values)

    def _get_string_slice(self, key):
        raise NotImplementedError

    def slice_indexer(self, start=None, end=None, step=None):
        """
        Compute the slice indexer for input labels and step.

        Index needs to be ordered and unique.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, default None
            If None, defaults to 1.

        Returns
        -------
        slice
            A slice object.

        Raises
        ------
        KeyError : If key does not exist, or key is not unique and index is
            not ordered.

        See Also
        --------
        Index.slice_locs : Computes slice locations for input labels.
        Index.get_slice_bound : Retrieves slice bound that corresponds to given label.

        Notes
        -----
        This function assumes that the data is sorted, so use at your own peril.

        Examples
        --------
        This is a method on all index types. For example you can do:

        >>> idx = pd.Index(list("abcd"))
        >>> idx.slice_indexer(start="b", end="c")
        slice(1, 3, None)

        >>> idx = pd.MultiIndex.from_arrays([list("abcd"), list("efgh")])
        >>> idx.slice_indexer(start="b", end=("c", "g"))
        slice(1, 3, None)
        """
        start_slice, end_slice = self.slice_locs(start, end, step=step)
        if not is_scalar(start_slice):
            raise AssertionError('Start slice bound is non-scalar')
        if not is_scalar(end_slice):
            raise AssertionError('End slice bound is non-scalar')
        return slice(start_slice, end_slice, step)

    def _maybe_cast_indexer(self, key):
        """
        If we have a float key and are not a floating index, then try to cast
        to an int if equivalent.
        """
        return key

    def _maybe_cast_listlike_indexer(self, target):
        """
        Analogue to maybe_cast_indexer for get_indexer instead of get_loc.
        """
        target_index = ensure_index(target)
        if not hasattr(target, 'dtype') and self.dtype == object and (target_index.dtype == 'string'):
            target_index = Index(target, dtype=self.dtype)
        return target_index

    @final
    def _validate_indexer(self, form, key, kind):
        """
        If we are positional indexer, validate that we have appropriate
        typed bounds must be an integer.
        """
        if not lib.is_int_or_none(key):
            self._raise_invalid_indexer(form, key)

    def _maybe_cast_slice_bound(self, label, side):
        """
        This function should be overloaded in subclasses that allow non-trivial
        casting on label-slice bounds, e.g. datetime-like indices allowing
        strings containing formatted datetimes.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
        if is_numeric_dtype(self.dtype):
            return self._maybe_cast_indexer(label)
        if (is_float(label) or is_integer(label)) and label not in self:
            self._raise_invalid_indexer('slice', label)
        return label

    def _searchsorted_monotonic(self, label, side='left'):
        if self.is_monotonic_increasing:
            return self.searchsorted(label, side=side)
        elif self.is_monotonic_decreasing:
            pos = self[::-1].searchsorted(label, side='right' if side == 'left' else 'left')
            return len(self) - pos
        raise ValueError('index must be monotonic increasing or decreasing')

    def get_slice_bound(self, label, side):
        """
        Calculate slice bound that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
            The label for which to calculate the slice bound.
        side : {'left', 'right'}
            if 'left' return leftmost position of given label.
            if 'right' return one-past-the-rightmost position of given label.

        Returns
        -------
        int
            Index of label.

        See Also
        --------
        Index.get_loc : Get integer location, slice or boolean mask for requested
            label.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.get_slice_bound(3, "left")
        3

        >>> idx.get_slice_bound(3, "right")
        4

        If ``label`` is non-unique in the index, an error will be raised.

        >>> idx_duplicate = pd.Index(["a", "b", "a", "c", "d"])
        >>> idx_duplicate.get_slice_bound("a", "left")
        Traceback (most recent call last):
        KeyError: Cannot get left slice bound for non-unique label: 'a'
        """
        if side not in ('left', 'right'):
            raise ValueError(f"Invalid value for side kwarg, must be either 'left' or 'right': {side}")
        original_label = label
        label = self._maybe_cast_slice_bound(label, side)
        try:
            slc = self.get_loc(label)
        except KeyError as err:
            try:
                return self._searchsorted_monotonic(label, side)
            except ValueError:
                raise err from None
        if isinstance(slc, np.ndarray):
            assert is_bool_dtype(slc.dtype)
            slc = lib.maybe_booleans_to_slice(slc.view('u1'))
            if isinstance(slc, np.ndarray):
                raise KeyError(f'Cannot get {side} slice bound for non-unique label: {original_label!r}')
        if isinstance(slc, slice):
            if side == 'left':
                return slc.start
            else:
                return slc.stop
        elif side == 'right':
            return slc + 1
        else:
            return slc

    def slice_locs(self, start=None, end=None, step=None):
        """
        Compute slice locations for input labels.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, defaults None
            If None, defaults to 1.

        Returns
        -------
        tuple[int, int]
            Returns a tuple of two integers representing the slice locations for the
            input labels within the index.

        See Also
        --------
        Index.get_loc : Get location for a single label.

        Notes
        -----
        This method only works if the index is monotonic or unique.

        Examples
        --------
        >>> idx = pd.Index(list("abcd"))
        >>> idx.slice_locs(start="b", end="c")
        (1, 3)

        >>> idx = pd.Index(list("bcde"))
        >>> idx.slice_locs(start="a", end="c")
        (0, 2)
        """
        inc = step is None or step >= 0
        if not inc:
            start, end = (end, start)
        if isinstance(start, (str, datetime)) and isinstance(end, (str, datetime)):
            try:
                ts_start = Timestamp(start)
                ts_end = Timestamp(end)
            except (ValueError, TypeError):
                pass
            else:
                if not tz_compare(ts_start.tzinfo, ts_end.tzinfo):
                    raise ValueError('Both dates must have the same UTC offset')
        start_slice = None
        if start is not None:
            start_slice = self.get_slice_bound(start, 'left')
        if start_slice is None:
            start_slice = 0
        end_slice = None
        if end is not None:
            end_slice = self.get_slice_bound(end, 'right')
        if end_slice is None:
            end_slice = len(self)
        if not inc:
            end_slice, start_slice = (start_slice - 1, end_slice - 1)
            if end_slice == -1:
                end_slice -= len(self)
            if start_slice == -1:
                start_slice -= len(self)
        return (start_slice, end_slice)

    def delete(self, loc):
        """
        Make new Index with passed location(-s) deleted.

        Parameters
        ----------
        loc : int or list of int
            Location of item(-s) which will be deleted.
            Use a list of locations to delete more than one value at the same time.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        See Also
        --------
        numpy.delete : Delete any rows and column from NumPy array (ndarray).

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.delete(1)
        Index(['a', 'c'], dtype='object')

        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.delete([0, 2])
        Index(['b'], dtype='object')
        """
        values = self._values
        if isinstance(values, np.ndarray):
            res_values = np.delete(values, loc)
        else:
            res_values = values.delete(loc)
        return self._constructor._simple_new(res_values, name=self.name)

    def insert(self, loc, item):
        """
        Make new Index inserting new item at location.

        Follows Python numpy.insert semantics for negative values.

        Parameters
        ----------
        loc : int
            The integer location where the new item will be inserted.
        item : object
            The new item to be inserted into the Index.

        Returns
        -------
        Index
            Returns a new Index object resulting from inserting the specified item at
            the specified location within the original Index.

        See Also
        --------
        Index.append : Append a collection of Indexes together.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.insert(1, "x")
        Index(['a', 'x', 'b', 'c'], dtype='object')
        """
        item = lib.item_from_zerodim(item)
        if is_valid_na_for_dtype(item, self.dtype) and self.dtype != object:
            item = self._na_value
        arr = self._values
        try:
            if isinstance(arr, ExtensionArray):
                res_values = arr.insert(loc, item)
                return type(self)._simple_new(res_values, name=self.name)
            else:
                item = self._validate_fill_value(item)
        except (TypeError, ValueError, LossySetitemError):
            dtype = self._find_common_type_compat(item)
            if dtype == self.dtype:
                raise
            return self.astype(dtype).insert(loc, item)
        if arr.dtype != object or not isinstance(item, (tuple, np.datetime64, np.timedelta64)):
            casted = arr.dtype.type(item)
            new_values = np.insert(arr, loc, casted)
        else:
            new_values = np.insert(arr, loc, None)
            loc = loc if loc >= 0 else loc - 1
            new_values[loc] = item
        out = Index(new_values, dtype=new_values.dtype, name=self.name)
        return out

    def drop(self, labels, errors='raise'):
        """
        Make new Index with passed list of labels deleted.

        Parameters
        ----------
        labels : array-like or scalar
            Array-like object or a scalar value, representing the labels to be removed
            from the Index.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        Raises
        ------
        KeyError
            If not all of the labels are found in the selected axis

        See Also
        --------
        Index.dropna : Return Index without NA/NaN values.
        Index.drop_duplicates : Return Index with duplicate values removed.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.drop(["a"])
        Index(['b', 'c'], dtype='object')
        """
        if not isinstance(labels, Index):
            arr_dtype = 'object' if self.dtype == 'object' else None
            labels = com.index_labels_to_array(labels, dtype=arr_dtype)
        indexer = self.get_indexer_for(labels)
        mask = indexer == -1
        if mask.any():
            if errors != 'ignore':
                raise KeyError(f'{labels[mask].tolist()} not found in axis')
            indexer = indexer[~mask]
        return self.delete(indexer)

    @final
    def infer_objects(self, copy=True):
        """
        If we have an object dtype, try to infer a non-object dtype.

        Parameters
        ----------
        copy : bool, default True
            Whether to make a copy in cases where no inference occurs.
        """
        if self._is_multi:
            raise NotImplementedError('infer_objects is not implemented for MultiIndex. Use index.to_frame().infer_objects() instead.')
        if self.dtype != object:
            return self.copy() if copy else self
        values = self._values
        values = cast('npt.NDArray[np.object_]', values)
        res_values = lib.maybe_convert_objects(values, convert_non_numeric=True)
        if copy and res_values is values:
            return self.copy()
        result = Index(res_values, name=self.name)
        if not copy and res_values is values and (self._references is not None):
            result._references = self._references
            result._references.add_index_reference(result)
        return result

    @final
    def diff(self, periods=1):
        """
        Computes the difference between consecutive values in the Index object.

        If periods is greater than 1, computes the difference between values that
        are `periods` number of positions apart.

        Parameters
        ----------
        periods : int, optional
            The number of positions between the current and previous
            value to compute the difference with. Default is 1.

        Returns
        -------
        Index
            A new Index object with the computed differences.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.Index([10, 20, 30, 40, 50])
        >>> idx.diff()
        Index([nan, 10.0, 10.0, 10.0, 10.0], dtype='float64')

        """
        return Index(self.to_series().diff(periods))

    def round(self, decimals=0):
        """
        Round each value in the Index to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.

        Returns
        -------
        Index
            A new Index with the rounded values.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.Index([10.1234, 20.5678, 30.9123, 40.4567, 50.7890])
        >>> idx.round(decimals=2)
        Index([10.12, 20.57, 30.91, 40.46, 50.79], dtype='float64')

        """
        return self._constructor(self.to_series().round(decimals))

    def _cmp_method(self, other, op):
        """
        Wrapper used to dispatch comparison operations.
        """
        if self.is_(other):
            if op in {operator.eq, operator.le, operator.ge}:
                arr = np.ones(len(self), dtype=bool)
                if self._can_hold_na and (not isinstance(self, ABCMultiIndex)):
                    arr[self.isna()] = False
                return arr
            elif op is operator.ne:
                arr = np.zeros(len(self), dtype=bool)
                if self._can_hold_na and (not isinstance(self, ABCMultiIndex)):
                    arr[self.isna()] = True
                return arr
        if isinstance(other, (np.ndarray, Index, ABCSeries, ExtensionArray)) and len(self) != len(other):
            raise ValueError('Lengths must match to compare')
        if not isinstance(other, ABCMultiIndex):
            other = extract_array(other, extract_numpy=True)
        else:
            other = np.asarray(other)
        if is_object_dtype(self.dtype) and isinstance(other, ExtensionArray):
            result = op(self._values, other)
        elif isinstance(self._values, ExtensionArray):
            result = op(self._values, other)
        elif is_object_dtype(self.dtype) and (not isinstance(self, ABCMultiIndex)):
            result = ops.comp_method_OBJECT_ARRAY(op, self._values, other)
        else:
            result = ops.comparison_op(self._values, other, op)
        return result

    @final
    def _logical_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.logical_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    @final
    def _construct_result(self, result, name):
        if isinstance(result, tuple):
            return (Index(result[0], name=name, dtype=result[0].dtype), Index(result[1], name=name, dtype=result[1].dtype))
        return Index(result, name=name, dtype=result.dtype)

    def _arith_method(self, other, op):
        if isinstance(other, Index) and is_object_dtype(other.dtype) and (type(other) is not Index):
            return NotImplemented
        return super()._arith_method(other, op)

    @final
    def _unary_method(self, op):
        result = op(self._values)
        return Index(result, name=self.name)

    def __abs__(self):
        return self._unary_method(operator.abs)

    def __neg__(self):
        return self._unary_method(operator.neg)

    def __pos__(self):
        return self._unary_method(operator.pos)

    def __invert__(self):
        return self._unary_method(operator.inv)

    def any(self, *args, **kwargs):
        """
        Return whether any element is Truthy.

        Parameters
        ----------
        *args
            Required for compatibility with numpy.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        bool or array-like (if axis is specified)
            A single element array-like may be converted to bool.

        See Also
        --------
        Index.all : Return whether all elements are True.
        Series.all : Return whether all elements are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        >>> index = pd.Index([0, 1, 2])
        >>> index.any()
        True

        >>> index = pd.Index([0, 0, 0])
        >>> index.any()
        False
        """
        nv.validate_any(args, kwargs)
        self._maybe_disable_logical_methods('any')
        vals = self._values
        if not isinstance(vals, np.ndarray):
            return vals._reduce('any')
        return np.any(vals)

    def all(self, *args, **kwargs):
        """
        Return whether all elements are Truthy.

        Parameters
        ----------
        *args
            Required for compatibility with numpy.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        bool or array-like (if axis is specified)
            A single element array-like may be converted to bool.

        See Also
        --------
        Index.any : Return whether any element in an Index is True.
        Series.any : Return whether any element in a Series is True.
        Series.all : Return whether all elements in a Series are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        True, because nonzero integers are considered True.

        >>> pd.Index([1, 2, 3]).all()
        True

        False, because ``0`` is considered False.

        >>> pd.Index([0, 1, 2]).all()
        False
        """
        nv.validate_all(args, kwargs)
        self._maybe_disable_logical_methods('all')
        vals = self._values
        if not isinstance(vals, np.ndarray):
            return vals._reduce('all')
        return np.all(vals)

    @final
    def _maybe_disable_logical_methods(self, opname):
        """
        raise if this Index subclass does not support any or all.
        """
        if isinstance(self, ABCMultiIndex):
            raise TypeError(f'cannot perform {opname} with {type(self).__name__}')

    @Appender(IndexOpsMixin.argmin.__doc__)
    def argmin(self, axis=None, skipna=True, *args, **kwargs):
        nv.validate_argmin(args, kwargs)
        nv.validate_minmax_axis(axis)
        if not self._is_multi and self.hasnans:
            if not skipna:
                raise ValueError('Encountered an NA value with skipna=False')
            elif self._isnan.all():
                raise ValueError('Encountered all NA values')
        return super().argmin(skipna=skipna)

    @Appender(IndexOpsMixin.argmax.__doc__)
    def argmax(self, axis=None, skipna=True, *args, **kwargs):
        nv.validate_argmax(args, kwargs)
        nv.validate_minmax_axis(axis)
        if not self._is_multi and self.hasnans:
            if not skipna:
                raise ValueError('Encountered an NA value with skipna=False')
            elif self._isnan.all():
                raise ValueError('Encountered all NA values')
        return super().argmax(skipna=skipna)

    def min(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return the minimum value of the Index.

        Parameters
        ----------
        axis : {None}
            Dummy argument for consistency with Series.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value of the object.
        Series.min : Return the minimum value in a Series.
        DataFrame.min : Return the minimum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.min()
        1

        >>> idx = pd.Index(["c", "b", "a"])
        >>> idx.min()
        'a'

        For a MultiIndex, the minimum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([("a", "b"), (2, 1)])
        >>> idx.min()
        ('a', 1)
        """
        nv.validate_min(args, kwargs)
        nv.validate_minmax_axis(axis)
        if not len(self):
            return self._na_value
        if len(self) and self.is_monotonic_increasing:
            first = self[0]
            if not isna(first):
                return first
        if not self._is_multi and self.hasnans:
            mask = self._isnan
            if not skipna or mask.all():
                return self._na_value
        if not self._is_multi and (not isinstance(self._values, np.ndarray)):
            return self._values._reduce(name='min', skipna=skipna)
        return nanops.nanmin(self._values, skipna=skipna)

    def max(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return the maximum value of the Index.

        Parameters
        ----------
        axis : int, optional
            For compatibility with NumPy. Only 0 or None are allowed.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        Index.min : Return the minimum value in an Index.
        Series.max : Return the maximum value in a Series.
        DataFrame.max : Return the maximum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.max()
        3

        >>> idx = pd.Index(["c", "b", "a"])
        >>> idx.max()
        'c'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([("a", "b"), (2, 1)])
        >>> idx.max()
        ('b', 2)
        """
        nv.validate_max(args, kwargs)
        nv.validate_minmax_axis(axis)
        if not len(self):
            return self._na_value
        if len(self) and self.is_monotonic_increasing:
            last = self[-1]
            if not isna(last):
                return last
        if not self._is_multi and self.hasnans:
            mask = self._isnan
            if not skipna or mask.all():
                return self._na_value
        if not self._is_multi and (not isinstance(self._values, np.ndarray)):
            return self._values._reduce(name='max', skipna=skipna)
        return nanops.nanmax(self._values, skipna=skipna)

    @final
    @property
    def shape(self):
        """
        Return a tuple of the shape of the underlying data.

        See Also
        --------
        Index.size: Return the number of elements in the underlying data.
        Index.ndim: Number of dimensions of the underlying data, by definition 1.
        Index.dtype: Return the dtype object of the underlying data.
        Index.values: Return an array representing the data in the Index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.shape
        (3,)
        """
        return (len(self),)

def maybe_sequence_to_range(sequence):
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
    elif len(sequence) == 1 or lib.infer_dtype(sequence, skipna=False) != 'integer':
        return sequence
    elif isinstance(sequence, (ABCSeries, Index)) and (not (isinstance(sequence.dtype, np.dtype) and sequence.dtype.kind == 'i')):
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

def ensure_index_from_sequences(sequences, names=None):
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
    from pandas.core.indexes.multi import MultiIndex
    if len(sequences) == 0:
        return default_index(0)
    elif len(sequences) == 1:
        if names is not None:
            names = names[0]
        return Index(maybe_sequence_to_range(sequences[0]), name=names)
    else:
        return MultiIndex.from_arrays(sequences, names=names)

def ensure_index(index_like, copy=False):
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
    if isinstance(index_like, Index):
        if copy:
            index_like = index_like.copy()
        return index_like
    if isinstance(index_like, ABCSeries):
        name = index_like.name
        return Index(index_like, name=name, copy=copy)
    if is_iterator(index_like):
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

def trim_front(strings):
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

def _validate_join_method(method):
    if method not in ['left', 'right', 'inner', 'outer']:
        raise ValueError(f'do not recognize join method {method}')

def maybe_extract_name(name, obj, cls):
    """
    If no name is passed, then extract it from data, validating hashability.
    """
    if name is None and isinstance(obj, (Index, ABCSeries)):
        name = obj.name
    if not is_hashable(name):
        raise TypeError(f'{cls.__name__}.name must be a hashable type')
    return name

def get_unanimous_names(*indexes):
    """
    Return common name if all indices agree, otherwise None (level-by-level).

    Parameters
    ----------
    indexes : list of Index objects

    Returns
    -------
    list
        A list representing the unanimous 'names' found.
    """
    name_tups = (tuple(i.names) for i in indexes)
    name_sets = ({*ns} for ns in zip_longest(*name_tups))
    names = tuple((ns.pop() if len(ns) == 1 else None for ns in name_sets))
    return names

def _unpack_nested_dtype(other):
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
    if isinstance(dtype, CategoricalDtype):
        return dtype.categories.dtype
    elif isinstance(dtype, ArrowDtype):
        import pyarrow as pa
        if pa.types.is_dictionary(dtype.pyarrow_dtype):
            other = other[:0].astype(ArrowDtype(dtype.pyarrow_dtype.value_type))
    return other.dtype

def _maybe_try_sort(result, sort):
    if sort is not False:
        try:
            result = algos.safe_sort(result)
        except TypeError as err:
            if sort is True:
                raise
            warnings.warn(f'{err}, sort order is undefined for incomparable objects.', RuntimeWarning, stacklevel=find_stack_level())
    return result

def get_values_for_csv(values, *, date_format, na_rep='nan', quoting=None, float_format=None, decimal='.'):
    """
    Convert to types which can be consumed by the standard library's
    csv.writer.writerows.
    """
    if isinstance(values, Categorical) and values.categories.dtype.kind in 'Mm':
        values = algos.take_nd(values.categories._values, ensure_platform_int(values._codes), fill_value=na_rep)
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
        values = cast('PeriodArray', values)
        res = values._format_native_types(na_rep=na_rep, date_format=date_format)
        return res
    elif isinstance(values.dtype, IntervalDtype):
        values = cast('IntervalArray', values)
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
        from pandas.io.formats.format import FloatArrayFormatter
        formatter = FloatArrayFormatter(values, na_rep=na_rep, float_format=float_format, decimal=decimal, quoting=quoting, fixed_width=False)
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
        itemsize = writers.word_len(na_rep)
        if values.dtype != _dtype_obj and (not quoting) and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype('U1').itemsize < itemsize:
                values = values.astype(f'<U{itemsize}')
        else:
            values = np.array(values, dtype='object')
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values