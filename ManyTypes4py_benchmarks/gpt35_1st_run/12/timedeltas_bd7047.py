from __future__ import annotations
from datetime import timedelta
import operator
from typing import TYPE_CHECKING, cast
import numpy as np
from pandas._libs import lib, tslibs
from pandas._libs.tslibs import NaT, NaTType, Tick, Timedelta, astype_overflowsafe, get_supported_dtype, iNaT, is_supported_dtype, periods_per_second
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import get_timedelta_days, get_timedelta_field
from pandas._libs.tslibs.timedeltas import array_to_timedelta64, floordiv_object_array, ints_to_pytimedelta, parse_timedelta_unit, truediv_object_array
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_endpoints
from pandas.core.dtypes.common import TD64NS_DTYPE, is_float_dtype, is_integer_dtype, is_object_dtype, is_scalar, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas.core import nanops, roperator
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer
if TYPE_CHECKING:
    from collections.abc import Iterator
    from pandas._typing import AxisInt, DateTimeErrorChoices, DtypeObj, NpDtype, Self, npt
    from pandas import DataFrame
import textwrap

def _field_accessor(name: str, alias: str, docstring: str) -> property:

class TimedeltaArray(dtl.TimelikeOps):
    _typ: str = 'timedeltaarray'
    _internal_fill_value: np.timedelta64 = np.timedelta64('NaT', 'ns')
    _recognized_scalars: tuple = (timedelta, np.timedelta64, Tick)
    _is_recognized_dtype: callable = lambda x: lib.is_np_dtype(x, 'm')
    _infer_matches: tuple = ('timedelta', 'timedelta64')

    @property
    def _scalar_type(self) -> Timedelta:
    __array_priority__: int = 1000
    _other_ops: list = []
    _bool_ops: list = []
    _field_ops: list = ['days', 'seconds', 'microseconds', 'nanoseconds']
    _datetimelike_ops: list = _field_ops + _bool_ops + ['unit', 'freq']
    _datetimelike_methods: list = ['to_pytimedelta', 'total_seconds', 'round', 'floor', 'ceil', 'as_unit']

    def _box_func(self, x: np.ndarray) -> np.ndarray:

    @property
    def dtype(self) -> np.dtype:

    @classmethod
    def _validate_dtype(cls, values: np.ndarray, dtype: np.dtype) -> np.dtype:

    @classmethod
    def _simple_new(cls, values: np.ndarray, freq: Tick = None, dtype: np.dtype = TD64NS_DTYPE) -> TimedeltaArray:

    @classmethod
    def _from_sequence(cls, data: list, *, dtype: np.dtype = None, copy: bool = False) -> TimedeltaArray:

    @classmethod
    def _from_sequence_not_strict(cls, data: list, *, dtype: np.dtype = None, copy: bool = False, freq: Tick = lib.no_default, unit: str = None) -> TimedeltaArray:

    @classmethod
    def _generate_range(cls, start, end, periods, freq, closed=None, *, unit=None) -> TimedeltaArray:

    def _unbox_scalar(self, value) -> np.timedelta64:

    def _scalar_from_string(self, value) -> Timedelta:

    def _check_compatible_with(self, other):

    def astype(self, dtype, copy=True) -> TimedeltaArray:

    def __iter__(self) -> Iterator:

    def sum(self, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, skipna=True, min_count=0) -> np.ndarray:

    def std(self, *, axis=None, dtype=None, out=None, ddof=1, keepdims=False, skipna=True) -> np.ndarray:

    def _accumulate(self, name, *, skipna=True, **kwargs):

    def _formatter(self, boxed=False):

    def _format_native_types(self, *, na_rep='NaT', date_format=None, **kwargs) -> np.ndarray:

    def _add_offset(self, other):

    def __mul__(self, other) -> TimedeltaArray:
    __rmul__ = __mul__

    def _scalar_divlike_op(self, other, op):

    def _cast_divlike_op(self, other):

    def _vector_divlike_op(self, other, op):

    def __truediv__(self, other) -> TimedeltaArray:
    __rtruediv__ = __truediv__

    def __floordiv__(self, other) -> TimedeltaArray:
    __rfloordiv__ = __floordiv__

    def __mod__(self, other) -> TimedeltaArray:
    __rmod__ = __mod__

    def __divmod__(self, other) -> tuple:
    __rdivmod__ = __divmod__

    def __neg__(self) -> TimedeltaArray:

    def __pos__(self) -> TimedeltaArray:

    def __abs__(self) -> TimedeltaArray:

    def total_seconds(self) -> np.ndarray:

    def to_pytimedelta(self) -> np.ndarray:

    @property
    def days(self) -> property:

    @property
    def seconds(self) -> property:

    @property
    def microseconds(self) -> property:

    @property
    def nanoseconds(self) -> property:

    @property
    def components(self) -> DataFrame:

def sequence_to_td64ns(data, copy=False, unit=None, errors='raise') -> tuple:

def _ints_to_td64ns(data, unit='ns') -> tuple:

def _objects_to_td64ns(data, unit=None, errors='raise') -> np.ndarray:

def _validate_td64_dtype(dtype) -> np.dtype:
