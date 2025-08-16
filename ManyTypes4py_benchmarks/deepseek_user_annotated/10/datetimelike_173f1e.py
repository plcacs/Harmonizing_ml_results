from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from functools import wraps
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    cast,
    final,
    overload,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Generic,
)
import warnings

import numpy as np
from numpy.typing import NDArray

from pandas._config import using_string_dtype
from pandas._config.config import get_option

from pandas._libs import (
    algos,
    lib,
)
from pandas._libs.tslibs import (
    BaseOffset,
    IncompatibleFrequency,
    NaT,
    NaTType,
    Period,
    Resolution,
    Tick,
    Timedelta,
    Timestamp,
    add_overflowsafe,
    astype_overflowsafe,
    get_unit_from_dtype,
    iNaT,
    ints_to_pydatetime,
    ints_to_pytimedelta,
    periods_per_day,
    to_offset,
)
from pandas._libs.tslibs.fields import (
    RoundTo,
    round_nsint64,
)
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import (
    ArrayLike,
    AxisInt,
    DatetimeLikeScalar,
    Dtype,
    DtypeObj,
    F,
    InterpolateOptions,
    NpDtype,
    PositionalIndexer2D,
    PositionalIndexerTuple,
    ScalarIndexer,
    Self,
    SequenceIndexer,
    TakeIndexer,
    TimeAmbiguous,
    TimeNonexistent,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    AbstractMethodError,
    InvalidComparison,
    PerformanceWarning,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_all_strings,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCCategorical,
    ABCMultiIndex,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    algorithms,
    missing,
    nanops,
    ops,
)
from pandas.core.algorithms import (
    isin,
    map_array,
    unique1d,
)
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,
    ravel_compat,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    check_setitem_lengths,
)
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import (
    invalid_comparison,
    make_invalid_op,
)

from pandas.tseries import frequencies

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )

    from pandas import Index
    from pandas.core.arrays import (
        DatetimeArray,
        PeriodArray,
        TimedeltaArray,
    )

DTScalarOrNaT = Union[DatetimeLikeScalar, NaTType]

T = TypeVar('T', bound='DatetimeLikeArrayMixin')

def _make_unpacked_invalid_op(op_name: str) -> Callable[[Any], Any]:
    op = make_invalid_op(op_name)
    return unpack_zerodim_and_defer(op_name)(op)

def _period_dispatch(meth: F) -> F:
    @wraps(meth)
    def new_meth(self: 'PeriodArray', *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self.dtype, PeriodDtype):
            return meth(self, *args, **kwargs)

        arr = self.view("M8[ns]")
        result = meth(arr, *args, **kwargs)
        if result is NaT:
            return NaT
        elif isinstance(result, Timestamp):
            return self._box_func(result._value)

        res_i8 = result.view("i8")
        return self._from_backing_data(res_i8)

    return cast(F, new_meth)

class DatetimeLikeArrayMixin(
    OpsMixin, NDArrayBackedExtensionArray
):
    _infer_matches: tuple[str, ...]
    _is_recognized_dtype: Callable[[DtypeObj], bool]
    _recognized_scalars: tuple[type, ...]
    _ndarray: np.ndarray
    freq: BaseOffset | None

    @cache_readonly
    def _can_hold_na(self) -> bool:
        return True

    def __init__(
        self, data: Any, dtype: Dtype | None = None, freq: Any = None, copy: bool = False
    ) -> None:
        raise AbstractMethodError(self)

    @property
    def _scalar_type(self) -> type[DatetimeLikeScalar]:
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value: str) -> DTScalarOrNaT:
        raise AbstractMethodError(self)

    def _unbox_scalar(
        self, value: DTScalarOrNaT
    ) -> np.int64 | np.datetime64 | np.timedelta64:
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other: DTScalarOrNaT) -> None:
        raise AbstractMethodError(self)

    def _box_func(self, x: Any) -> Any:
        raise AbstractMethodError(self)

    def _box_values(self, values: Any) -> np.ndarray:
        return lib.map_infer(values, self._box_func, convert=False)

    def __iter__(self) -> Iterator[Any]:
        if self.ndim > 1:
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        return self._ndarray.view("i8")

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format: str | None = None
    ) -> npt.NDArray[np.object_]:
        raise AbstractMethodError(self)

    def _formatter(self, boxed: bool = False) -> Callable[[object], str]:
        return "'{}'".format

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if is_object_dtype(dtype):
            if copy is False:
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
            return np.array(list(self), dtype=object)

        if copy is True:
            return np.array(self._ndarray, dtype=dtype)
        return self._ndarray

    @overload
    def __getitem__(self, key: ScalarIndexer) -> DTScalarOrNaT: ...

    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | PositionalIndexerTuple,
    ) -> Self: ...

    def __getitem__(self, key: PositionalIndexer2D) -> Self | DTScalarOrNaT:
        result = cast("Union[Self, DTScalarOrNaT]", super().__getitem__(key))
        if lib.is_scalar(result):
            return result
        else:
            result = cast(Self, result)
        result._freq = self._get_getitem_freq(key)
        return result

    def _get_getitem_freq(self, key: Any) -> BaseOffset | None:
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

    def __setitem__(
        self,
        key: int | Sequence[int] | Sequence[bool] | slice,
        value: NaTType | Any | Sequence[Any],
    ) -> None:
        no_op = check_setitem_lengths(key, value, self)
        super().__setitem__(key, value)
        if no_op:
            return
        self._maybe_clear_freq()

    def _maybe_clear_freq(self) -> None:
        pass

    def astype(self, dtype: Dtype, copy: bool = True) -> Any:
        dtype = pandas_dtype(dtype)

        if dtype == object:
            if self.dtype.kind == "M":
                self = cast("DatetimeArray", self)
                i8data = self.asi8
                converted = ints_to_pydatetime(
                    i8data,
                    tz=self.tz,
                    box="timestamp",
                    reso=self._creso,
                )
                return converted

            elif self.dtype.kind == "m":
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
        elif dtype.kind in "iu":
            values = self.asi8
            if dtype != np.int64:
                raise TypeError(
                    f"Converting from {self.dtype} to {dtype} is not supported. "
                    "Do obj.astype('int64').astype(dtype) instead"
                )

            if copy:
                values = values.copy()
            return values
        elif (dtype.kind in "mM" and self.dtype != dtype) or dtype.kind == "f":
            msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
            raise TypeError(msg)
        else:
            return np.asarray(self, dtype=dtype)

    @overload
    def view(self) -> Self: ...

    @overload
    def view(self, dtype: Literal["M8[ns]"]) -> 'DatetimeArray': ...

    @overload
    def view(self, dtype: Literal["m8[ns]"]) -> 'TimedeltaArray': ...

    @overload
    def view(self, dtype: Dtype | None = ...) -> ArrayLike: ...

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
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
            raise ValueError("Lengths must match")

        else:
            try:
                other = self._validate_listlike(other, allow_object=True)
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                if is_object_dtype(getattr(other, "dtype", None)):
                    pass
                else:
                    raise InvalidComparison(other) from err

        return other

    def _validate_scalar(
        self,
        value: Any,
        *,
        allow_listlike: bool = False,
        unbox: bool = True,
    ) -> Any:
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
        if hasattr(value, "dtype") and getattr(value, "ndim", 0) > 0:
            msg_got = f"{value.dtype} array"
        else:
            msg_got = f"'{type(value).__name__}'"
        if allow_listlike:
            msg = (
                f"value should be a '{self._scalar_type.__name__}', 'NaT', "
                f"or array of those. Got {msg_got} instead."
            )
        else:
            msg = (
                f"value should be a '{self._scalar_type.__name__}' or 'NaT'. "
                f"Got {msg_got} instead."
            )
        return msg

    def _validate_listlike(self, value: Any, allow_object: bool = False) -> Any:
        if isinstance(value, type(self)):
            if self.dtype.kind in "mM" and not allow_object and self.unit != value.unit:
                value = value.as_unit(self.unit, round_ok=False)
            return value

        if isinstance(value, list) and len(value) == 0:
            return type(self)._from_sequence([], dtype=self.dtype)

        if hasattr(value, "dtype") and value.dtype == object:
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

        if self.dtype.kind in "mM" and not allow_object:
            value = value.as_unit(self.unit, round_ok=False)
        return value

    def _validate_setitem_value(self, value: Any) -> Any:
        if is_list_like(value):
            value = self._validate_listlike(value)
        else:
            return self._validate_scalar(value, allow_listlike=True)

        return self._unbox(value)

    @final
    def _unbox(self, other: Any) -> np.int64 | np.datetime64 | np.timedelta64 | np.ndarray:
        if lib.is_scalar(other):
            other = self._unbox_scalar(other)
        else:
            self._check_compatible_with(other)
            other = other._ndarray
        return other

    @ravel_compat
    def map(self, mapper: Callable, na_action: Literal["ignore"] | None = None) -> Any:
        from pandas import Index

        result = map_array(self, mapper, na_action=na_action)
        result = Index(result)

        if isinstance(result, ABCMultiIndex):
            return result.to_numpy()
        else:
            return result.array

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        if values.dtype.kind in "fiuc":
            return np.zeros(self.shape, dtype=bool)

        values = ensure_wrapped_if_datetimelike(values)

        if not isinstance(values, type(self)):
            if values.dtype == object:
                values = lib.maybe_convert_objects(
                    values,
                    convert_non_numeric=True,
                    dtype_if_all_nat=self.dtype,
                )
                if values.dtype != object:
                    return self.isin(values)
                else:
                   