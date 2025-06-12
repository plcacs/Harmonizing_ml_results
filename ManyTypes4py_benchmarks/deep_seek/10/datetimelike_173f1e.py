from __future__ import annotations
from datetime import datetime, timedelta
from functools import wraps
import operator
from typing import TYPE_CHECKING, Any, Literal, Union, cast, final, overload, Optional, List, Tuple, Dict, Set, Callable, Iterable, Iterator, Sequence, TypeVar, Generic, Mapping
import warnings
import numpy as np
from pandas._config import using_string_dtype
from pandas._config.config import get_option
from pandas._libs import algos, lib
from pandas._libs.tslibs import BaseOffset, IncompatibleFrequency, NaT, NaTType, Period, Resolution, Tick, Timedelta, Timestamp, add_overflowsafe, astype_overflowsafe, get_unit_from_dtype, iNaT, ints_to_pydatetime, ints_to_pytimedelta, periods_per_day, to_offset
from pandas._libs.tslibs.fields import RoundTo, round_nsint64
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import ArrayLike, AxisInt, DatetimeLikeScalar, Dtype, DtypeObj, F, InterpolateOptions, NpDtype, PositionalIndexer2D, PositionalIndexerTuple, ScalarIndexer, Self, SequenceIndexer, TakeIndexer, TimeAmbiguous, TimeNonexistent, npt
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError, InvalidComparison, PerformanceWarning
from pandas.util._decorators import Appender, Substitution, cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_all_strings, is_integer_dtype, is_list_like, is_object_dtype, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype, DatetimeTZDtype, ExtensionDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCCategorical, ABCMultiIndex
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna
from pandas.core import algorithms, missing, nanops, ops
from pandas.core.algorithms import isin, map_array, unique1d
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray, ravel_compat
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import array as pd_array, ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexers import check_array_indexer, check_setitem_lengths
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import invalid_comparison, make_invalid_op
from pandas.tseries import frequencies

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pandas import Index
    from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray

DTScalarOrNaT = Union[DatetimeLikeScalar, NaTType]

def _make_unpacked_invalid_op(op_name: str) -> Callable:
    op = make_invalid_op(op_name)
    return unpack_zerodim_and_defer(op_name)(op)

def _period_dispatch(meth: F) -> F:
    """
    For PeriodArray methods, dispatch to DatetimeArray and re-wrap the results
    in PeriodArray.  We cannot use ._ndarray directly for the affected
    methods because the i8 data has different semantics on NaT values.
    """

    @wraps(meth)
    def new_meth(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self.dtype, PeriodDtype):
            return meth(self, *args, **kwargs)
        arr = self.view('M8[ns]')
        result = meth(arr, *args, **kwargs)
        if result is NaT:
            return NaT
        elif isinstance(result, Timestamp):
            return self._box_func(result._value)
        res_i8 = result.view('i8')
        return self._from_backing_data(res_i8)
    return cast(F, new_meth)

class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray):
    """
    Shared Base/Mixin class for DatetimeArray, TimedeltaArray, PeriodArray

    Assumes that __new__/__init__ defines:
        _ndarray

    and that inheriting subclass implements:
        freq
    """

    @cache_readonly
    def _can_hold_na(self) -> bool:
        return True

    def __init__(self, data: Any, dtype: Optional[Dtype] = None, freq: Any = None, copy: bool = False) -> None:
        raise AbstractMethodError(self)

    @property
    def _scalar_type(self) -> type:
        """
        The scalar associated with this datelike

        * PeriodArray : Period
        * DatetimeArray : Timestamp
        * TimedeltaArray : Timedelta
        """
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value: str) -> Union[Period, Timestamp, Timedelta, NaTType]:
        """
        Construct a scalar type from a string.

        Parameters
        ----------
        value : str

        Returns
        -------
        Period, Timestamp, or Timedelta, or NaT
            Whatever the type of ``self._scalar_type`` is.

        Notes
        -----
        This should call ``self._check_compatible_with`` before
        unboxing the result.
        """
        raise AbstractMethodError(self)

    def _unbox_scalar(self, value: Union[Period, Timestamp, Timedelta, NaTType]) -> int:
        """
        Unbox the integer value of a scalar `value`.

        Parameters
        ----------
        value : Period, Timestamp, Timedelta, or NaT
            Depending on subclass.

        Returns
        -------
        int

        Examples
        --------
        >>> arr = pd.array(np.array(["1970-01-01"], "datetime64[ns]"))
        >>> arr._unbox_scalar(arr[0])
        numpy.datetime64('1970-01-01T00:00:00.000000000')
        """
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other: Any) -> None:
        """
        Verify that `self` and `other` are compatible.

        * DatetimeArray verifies that the timezones (if any) match
        * PeriodArray verifies that the freq matches
        * Timedelta has no verification

        In each case, NaT is considered compatible.

        Parameters
        ----------
        other

        Raises
        ------
        Exception
        """
        raise AbstractMethodError(self)

    def _box_func(self, x: int) -> Any:
        """
        box function to get object from internal representation
        """
        raise AbstractMethodError(self)

    def _box_values(self, values: np.ndarray) -> np.ndarray:
        """
        apply box func to passed values
        """
        return lib.map_infer(values, self._box_func, convert=False)

    def __iter__(self) -> Iterator:
        if self.ndim > 1:
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self) -> np.ndarray:
        """
        Integer representation of the values.

        Returns
        -------
        ndarray
            An ndarray with int64 dtype.
        """
        return self._ndarray.view('i8')

    def _format_native_types(self, *, na_rep: str = 'NaT', date_format: Optional[str] = None) -> np.ndarray:
        """
        Helper method for astype when converting to strings.

        Returns
        -------
        ndarray[str]
        """
        raise AbstractMethodError(self)

    def _formatter(self, boxed: bool = False) -> Callable:
        return "'{}'".format

    def __array__(self, dtype: Optional[Dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        if is_object_dtype(dtype):
            if copy is False:
                raise ValueError('Unable to avoid copy while creating an array as requested.')
            return np.array(list(self), dtype=object)
        if copy is True:
            return np.array(self._ndarray, dtype=dtype)
        return self._ndarray

    @overload
    def __getitem__(self, key: ScalarIndexer) -> DTScalarOrNaT: ...

    @overload
    def __getitem__(self, key: SequenceIndexer) -> Self: ...

    def __getitem__(self, key: Any) -> Union[Self, DTScalarOrNaT]:
        """
        This getitem defers to the underlying array, which by-definition can
        only handle list-likes, slices, and integer scalars
        """
        result = cast('Union[Self, DTScalarOrNaT]', super().__getitem__(key))
        if lib.is_scalar(result):
            return result
        else:
            result = cast(Self, result)
        result._freq = self._get_getitem_freq(key)
        return result

    def _get_getitem_freq(self, key: Any) -> Optional[BaseOffset]:
        """
        Find the `freq` attribute to assign to the result of a __getitem__ lookup.
        """
        is_period = isinstance(self.dtype, PeriodDtype)
        if is_period:
            freq = self.freq
        elif self.ndim != 1:
            freq = None
        else:
            key = check_array_indexer(self, key)
            freq = None
            if isinstance(key, slice):
                if self.freq is not None and key.step is not None:
                    freq = key.step * self.freq
                else:
                    freq = self.freq
            elif key is Ellipsis:
                freq = self.freq
            elif com.is_bool_indexer(key):
                new_key = lib.maybe_booleans_to_slice(key.view(np.uint8))
                if isinstance(new_key, slice):
                    return self._get_getitem_freq(new_key)
        return freq

    def __setitem__(self, key: Any, value: Any) -> None:
        no_op = check_setitem_lengths(key, value, self)
        super().__setitem__(key, value)
        if no_op:
            return
        self._maybe_clear_freq()

    def _maybe_clear_freq(self) -> None:
        pass

    def astype(self, dtype: Dtype, copy: bool = True) -> ArrayLike:
        dtype = pandas_dtype(dtype)
        if dtype == object:
            if self.dtype.kind == 'M':
                self = cast('DatetimeArray', self)
                i8data = self.asi8
                converted = ints_to_pydatetime(i8data, tz=self.tz, box='timestamp', reso=self._creso)
                return converted
            elif self.dtype.kind == 'm':
                return ints_to_pytimedelta(self._ndarray, box=True)
            return self._box_values(self.asi8.ravel()).reshape(self.shape)
        elif is_string_dtype(dtype):
            if isinstance(dtype, ExtensionDtype):
                arr_object = self._format_native_types(na_rep=dtype.na_value)
                cls = dtype.construct_array_type()
                return cls._from_sequence(arr_object, dtype=dtype, copy=False)
            else:
                return self._format_native_types()
        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        elif dtype.kind in 'iu':
            values = self.asi8
            if dtype != np.int64:
                raise TypeError(f"Converting from {self.dtype} to {dtype} is not supported. Do obj.astype('int64').astype(dtype) instead")
            if copy:
                values = values.copy()
            return values
        elif dtype.kind in 'mM' and self.dtype != dtype or dtype.kind == 'f':
            msg = f'Cannot cast {type(self).__name__} to dtype {dtype}'
            raise TypeError(msg)
        else:
            return np.asarray(self, dtype=dtype)

    @overload
    def view(self) -> Self: ...

    @overload
    def view(self, dtype: Dtype) -> ArrayLike: ...

    def view(self, dtype: Optional[Dtype] = None) -> Union[Self, ArrayLike]:
        return super().view(dtype)

    def _validate_comparison_value(self, other: Any) -> Any:
        if isinstance(other, str):
            try:
                other = self._scalar_from_string(other)
            except (ValueError, IncompatibleFrequency) as err:
                raise InvalidComparison(other) from err
        if isinstance(other, self._recognized_scalars) or other is NaT:
            other = self._scalar_type(other)
            try:
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                raise InvalidComparison(other) from err
        elif not is_list_like(other):
            raise InvalidComparison(other)
        elif len(other) != len(self):
            raise ValueError('Lengths must match')
        else:
            try:
                other = self._validate_listlike(other, allow_object=True)
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                if is_object_dtype(getattr(other, 'dtype', None)):
                    pass
                else:
                    raise InvalidComparison(other) from err
        return other

    def _validate_scalar(self, value: Any, *, allow_listlike: bool = False, unbox: bool = True) -> Any:
        """
        Validate that the input value can be cast to our scalar_type.

        Parameters
        ----------
        value : object
        allow_listlike: bool, default False
            When raising an exception, whether the message should say
            listlike inputs are allowed.
        unbox : bool, default True
            Whether to unbox the result before returning.  Note: unbox=False
            skips the setitem compatibility check.

        Returns
        -------
        self._scalar_type or NaT
        """
        if isinstance(value, self._scalar_type):
            pass
        elif isinstance(value, str):
            try:
                value = self._scalar_from_string(value)
            except ValueError as err:
                msg = self._validation_error_message(value, allow_listlike)
                raise TypeError(msg) from err
        elif is_valid_na_for_dtype(value, self.dtype):
            value = NaT
        elif isna(value):
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)
        elif isinstance(value, self._recognized_scalars):
            value = self._scalar_type(value)
        else:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)
        if not unbox:
            return value
        return self._unbox_scalar(value)

    def _validation_error_message(self, value: Any, allow_listlike: bool = False) -> str:
        """
        Construct an exception message on validation error.

        Some methods allow only scalar inputs, while others allow either scalar
        or listlike.

        Parameters
        ----------
        allow_listlike: bool, default False

        Returns
        -------
        str
        """
        if hasattr(value, 'dtype') and getattr(value, 'ndim', 0) > 0:
            msg_got = f'{value.dtype} array'
        else:
            msg_got = f"'{type(value).__name__}'"
        if allow_listlike:
            msg = f"value should be a '{self._scalar_type.__name__}', 'NaT', or array of those. Got {msg_got} instead."
        else:
            msg = f"value should be a '{self._scalar_type.__name__}' or 'NaT'. Got {msg_got} instead."
        return msg

    def _validate_listlike(self, value: Any, allow_object: bool = False) -> Any:
        if isinstance(value, type(self)):
            if self.dtype.kind in 'mM' and (not allow_object) and (self.unit != value.unit):
                value = value.as_unit(self.unit, round_ok=False)
            return value
        if isinstance(value, list) and len(value) == 0:
            return type(self)._from_sequence([], dtype=self.dtype)
        if hasattr(value, 'dtype') and value.dtype == object:
            if lib.infer_dtype(value) in self._infer_matches:
                try:
                    value = type(self)._from_sequence(value)
                except (ValueError, TypeError) as err:
                    if allow_object:
                        return value
                    msg = self._validation_error_message(value, True)
                    raise TypeError(msg) from err
        value = extract_array(value, extract_numpy=True)
        value = pd_array(value)
        value = extract_array(value, extract_numpy=True)
        if is_all_strings(value):
            try:
                value = type(self)._from_sequence(value, dtype=self.dtype)
            except ValueError:
                pass
        if isinstance(value.dtype, CategoricalDtype):
            if value.categories.dtype == self.dtype:
                value = value._internal_get_values()
                value = extract_array(value, extract_numpy=True)
        if allow_object and is_object_dtype(value.dtype):
            pass
        elif not type(self)._is_recognized_dtype(value.dtype):
            msg = self._validation_error_message(value, True)
            raise TypeError(msg)
        if self.dtype.kind in 'mM' and (not allow_object):
            value = value.as_unit(self.unit, round_ok=False)
        return value

    def _validate_setitem_value(self, value: Any) ->