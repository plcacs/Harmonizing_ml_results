from __future__ import annotations

import functools
import operator
import re
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
    Union,
    Optional,
    Dict,
    List,
    Tuple,
    Callable,
    Sequence,
    Iterator,
    TypeVar,
    Generic,
)
import unicodedata

import numpy as np
import numpy.typing as npt

from pandas._libs import lib
from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
    timezones,
)
from pandas.compat import (
    pa_version_under10p1,
    pa_version_under11p0,
    pa_version_under13p0,
)
from pandas.util._decorators import doc

from pandas.core.dtypes.cast import (
    can_hold_element,
    infer_dtype_from_scalar,
)
from pandas.core.dtypes.common import (
    CategoricalDtype,
    is_array_like,
    is_bool_dtype,
    is_float_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    algorithms as algos,
    missing,
    ops,
    roperator,
)
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionArraySupportsAnyAll,
)
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
    validate_indices,
)
from pandas.core.nanops import check_below_min_count
from pandas.core.strings.base import BaseStringArrayMethods

from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset

if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pandas.core.dtypes.dtypes import ArrowDtype

    ARROW_CMP_FUNCS: Dict[str, Callable] = {
        "eq": pc.equal,
        "ne": pc.not_equal,
        "lt": pc.less,
        "gt": pc.greater,
        "le": pc.less_equal,
        "ge": pc.greater_equal,
    }

    ARROW_LOGICAL_FUNCS: Dict[str, Callable] = {
        "and_": pc.and_kleene,
        "rand_": lambda x, y: pc.and_kleene(y, x),
        "or_": pc.or_kleene,
        "ror_": lambda x, y: pc.or_kleene(y, x),
        "xor": pc.xor,
        "rxor": lambda x, y: pc.xor(y, x),
    }

    ARROW_BIT_WISE_FUNCS: Dict[str, Callable] = {
        "and_": pc.bit_wise_and,
        "rand_": lambda x, y: pc.bit_wise_and(y, x),
        "or_": pc.bit_wise_or,
        "ror_": lambda x, y: pc.bit_wise_or(y, x),
        "xor": pc.bit_wise_xor,
        "rxor": lambda x, y: pc.bit_wise_xor(y, x),
    }

    def cast_for_truediv(
        arrow_array: pa.ChunkedArray, pa_object: Union[pa.Array, pa.Scalar]
    ) -> Tuple[pa.ChunkedArray, Union[pa.Array, pa.Scalar]]:
        if pa.types.is_integer(arrow_array.type) and pa.types.is_integer(
            pa_object.type
        ):
            return pc.cast(arrow_array, pa.float64(), safe=False), pc.cast(
                pa_object, pa.float64(), safe=False
            )
        return arrow_array, pa_object

    def floordiv_compat(
        left: Union[pa.ChunkedArray, pa.Array, pa.Scalar],
        right: Union[pa.ChunkedArray, pa.Array, pa.Scalar],
    ) -> pa.ChunkedArray:
        if pa.types.is_integer(left.type) and pa.types.is_integer(right.type):
            divided = pc.divide_checked(left, right)
            if pa.types.is_signed_integer(divided.type):
                has_remainder = pc.not_equal(pc.multiply(divided, right), left)
                has_one_negative_operand = pc.less(
                    pc.bit_wise_xor(left, right),
                    pa.scalar(0, type=divided.type),
                )
                result = pc.if_else(
                    pc.and_(
                        has_remainder,
                        has_one_negative_operand,
                    ),
                    pc.subtract(divided, pa.scalar(1, type=divided.type)),
                    divided,
                )
            else:
                result = divided
            result = result.cast(left.type)
        else:
            divided = pc.divide(left, right)
            result = pc.floor(divided)
        return result

    ARROW_ARITHMETIC_FUNCS: Dict[str, Callable] = {
        "add": pc.add_checked,
        "radd": lambda x, y: pc.add_checked(y, x),
        "sub": pc.subtract_checked,
        "rsub": lambda x, y: pc.subtract_checked(y, x),
        "mul": pc.multiply_checked,
        "rmul": lambda x, y: pc.multiply_checked(y, x),
        "truediv": lambda x, y: pc.divide(*cast_for_truediv(x, y)),
        "rtruediv": lambda x, y: pc.divide(*cast_for_truediv(y, x)),
        "floordiv": lambda x, y: floordiv_compat(x, y),
        "rfloordiv": lambda x, y: floordiv_compat(y, x),
        "mod": NotImplemented,
        "rmod": NotImplemented,
        "divmod": NotImplemented,
        "rdivmod": NotImplemented,
        "pow": pc.power_checked,
        "rpow": lambda x, y: pc.power_checked(y, x),
    }

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from pandas._libs.missing import NAType
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        Iterator,
        NpDtype,
        NumpySorter,
        NumpyValueArrayLike,
        PositionalIndexer,
        Scalar,
        Self,
        SortKind,
        TakeIndexer,
        TimeAmbiguous,
        TimeNonexistent,
        npt,
    )

    from pandas.core.dtypes.dtypes import ExtensionDtype

    from pandas import Series
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.arrays.timedeltas import TimedeltaArray

Self = TypeVar("Self", bound="ArrowExtensionArray")

def get_unit_from_pa_dtype(pa_dtype: pa.DataType) -> str:
    if pa_version_under11p0:
        unit = str(pa_dtype).split("[", 1)[-1][:-1]
        if unit not in ["s", "ms", "us", "ns"]:
            raise ValueError(pa_dtype)
        return unit
    return pa_dtype.unit

def to_pyarrow_type(
    dtype: Union[ArrowDtype, pa.DataType, Dtype, None],
) -> Optional[pa.DataType]:
    if isinstance(dtype, ArrowDtype):
        return dtype.pyarrow_dtype
    elif isinstance(dtype, pa.DataType):
        return dtype
    elif isinstance(dtype, DatetimeTZDtype):
        return pa.timestamp(dtype.unit, dtype.tz)
    elif dtype:
        try:
            return pa.from_numpy_dtype(dtype)
        except pa.ArrowNotImplementedError:
            pass
    return None

class ArrowExtensionArray(
    OpsMixin,
    ExtensionArraySupportsAnyAll,
    ArrowStringArrayMixin,
    BaseStringArrayMethods,
):
    _pa_array: pa.ChunkedArray
    _dtype: ArrowDtype

    def __init__(self, values: Union[pa.Array, pa.ChunkedArray]) -> None:
        if pa_version_under10p1:
            msg = "pyarrow>=10.0.1 is required for PyArrow backed ArrowExtensionArray."
            raise ImportError(msg)
        if isinstance(values, pa.Array):
            self._pa_array = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._pa_array = values
        else:
            raise ValueError(
                f"Unsupported type '{type(values)}' for ArrowExtensionArray"
            )
        self._dtype = ArrowDtype(self._pa_array.type)

    @classmethod
    def _from_sequence(
        cls, scalars: Sequence[Any], *, dtype: Optional[Dtype] = None, copy: bool = False
    ) -> Self:
        pa_type = to_pyarrow_type(dtype)
        pa_array = cls._box_pa_array(scalars, pa_type=pa_type, copy=copy)
        arr = cls(pa_array)
        return arr

    @classmethod
    def _from_sequence_of_strings(
        cls, strings: Sequence[str], *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        pa_type = to_pyarrow_type(dtype)
        if (
            pa_type is None
            or pa.types.is_binary(pa_type)
            or pa.types.is_string(pa_type)
            or pa.types.is_large_string(pa_type)
        ):
            scalars = strings
        elif pa.types.is_timestamp(pa_type):
            from pandas.core.tools.datetimes import to_datetime
            scalars = to_datetime(strings, errors="raise")
        elif pa.types.is_date(pa_type):
            from pandas.core.tools.datetimes import to_datetime
            scalars = to_datetime(strings, errors="raise").date
        elif pa.types.is_duration(pa_type):
            from pandas.core.tools.timedeltas import to_timedelta
            scalars = to_timedelta(strings, errors="raise")
            if pa_type.unit != "ns":
                mask = isna(scalars)
                if not isinstance(strings, (pa.Array, pa.ChunkedArray)):
                    strings = pa.array(strings, type=pa.string(), from_pandas=True)
                strings = pc.if_else(mask, None, strings)
                try:
                    scalars = strings.cast(pa.int64())
                except pa.ArrowInvalid:
                    pass
        elif pa.types.is_time(pa_type):
            from pandas.core.tools.times import to_time
            scalars = to_time(strings, errors="coerce")
        elif pa.types.is_boolean(pa_type):
            if isinstance(strings, (pa.Array, pa.ChunkedArray)):
                scalars = strings
            else:
                scalars = pa.array(strings, type=pa.string(), from_pandas=True)
            scalars = pc.if_else(pc.equal(scalars, "1.0"), "1", scalars)
            scalars = pc.if_else(pc.equal(scalars, "0.0"), "0", scalars)
            scalars = scalars.cast(pa.bool_())
        elif (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_decimal(pa_type)
        ):
            from pandas.core.tools.numeric import to_numeric
            scalars = to_numeric(strings, errors="raise")
        else:
            raise NotImplementedError(
                f"Converting strings to {pa_type} is not implemented."
            )
        return cls._from_sequence(scalars, dtype=pa_type, copy=copy)

    @classmethod
    def _box_pa(
        cls, value: Any, pa_type: Optional[pa.DataType] = None
    ) -> Union[pa.Array, pa.ChunkedArray, pa.Scalar]:
        if isinstance(value, pa.Scalar) or not is_list_like(value):
            return cls._box_pa_scalar(value, pa_type)
        return cls._box_pa_array(value, pa_type)

    @classmethod
    def _box_pa_scalar(cls, value: Any, pa_type: Optional[pa.DataType] = None) -> pa.Scalar:
        if isinstance(value, pa.Scalar):
            pa_scalar = value
        elif isna(value):
            pa_scalar = pa.scalar(None, type=pa_type)
        else:
            if isinstance(value, Timedelta):
                if pa_type is None:
                    pa_type = pa.duration(value.unit)
                elif value.unit != pa_type.unit:
                    value = value.as_unit(pa_type.unit)
                value = value._value
            elif isinstance(value, Timestamp):
                if pa_type is None:
                    pa_type = pa.timestamp(value.unit, tz=value.tz)
                elif value.unit != pa_type.unit:
                    value = value.as_unit(pa_type.unit)
                value = value._value
            pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)
        if pa_type is not None and pa_scalar.type != pa_type:
            pa_scalar = pa_scalar.cast(pa_type)
        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value: Sequence[Any], pa_type: Optional[pa.DataType] = None, copy: bool = False
    ) -> Union[pa.Array, pa.ChunkedArray]:
        if isinstance(value, cls):
            pa_array = value._pa_array
        elif isinstance(value, (pa.Array, pa.ChunkedArray)):
            pa_array = value
        elif isinstance(value, BaseMaskedArray):
            if copy:
                value = value.copy()
            pa_array = value.__arrow_array__()
        else:
            if (
                isinstance(value, np.ndarray)
                and pa_type is not None
                and (
                    pa.types.is_large_binary(pa_type)
                    or pa.types.is_large_string(pa_type)
                )
            ):
                value = value.tolist()
            elif copy and is_array_like(value):
                value = value.copy()
            if (
                pa_type is not None
                and pa.types.is_duration(pa_type)
                and (not isinstance(value, np.ndarray) or value.dtype.kind not in "mi")
            ):
                from pandas.core.tools.timedeltas import to_timedelta
                value = to_timedelta(value, unit=pa_type.unit).as_unit(pa_type.unit)
                value = value.to_numpy()
            try:
                pa_array = pa.array(value, type=pa_type, from_pandas=True)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                pa_array = pa.array(value, from_pandas=True)
            if pa_type is None and pa.types.is_duration(pa_array.type):
                from pandas.core.tools.timedeltas import to_timedelta
                value = to_timedelta(value)
                value = value.to_numpy()
                pa_array = pa.array(value, type=pa_type, from_pandas=True)
            if pa.types.is_duration(pa_array.type) and pa_array.null_count > 0:
                arr = cls(pa_array)
                arr = arr.fillna(arr.dtype.na_value)
                pa_array = arr._pa_array
        if pa_type is not None and pa_array.type != pa_type:
            if pa.types.is_dictionary(pa_type):
                pa_array = pa_array.dictionary_encode()
                if pa_array.type != pa_type:
                    pa_array = pa_array.cast(pa_type)
            else:
                try:
                    pa_array = pa_array.cast(pa_type)
                except (pa.ArrowNotImplementedError, pa.ArrowTypeError):
                    if pa.types.is_string(pa_array.type) or pa.types.is_large_string(
                        pa_array.type
                    ):
                        dtype = ArrowDtype(pa_type)
                        return cls._from_sequence_of_strings(
                            value, dtype=dtype
                        )._pa_array
                    else:
                        raise
        return pa_array

    def __getitem__(self, item: PositionalIndexer) -> Any:
        item = check_array_indexer(self, item)
        if isinstance(item, np.ndarray):
            if not len(item):
                if (
                    isinstance(self._dtype, StringDtype)
                    and self._dtype.storage == "pyarrow"
                ):
                    pa_dtype = pa.string()
                else:
                    pa_dtype = self._dtype.pyarrow_dtype
                return type(self)(pa.chunked_array([], type=pa_dtype))
            elif item.dtype.kind in "iu":
                return self.take(item)
            elif item.dtype.kind == "b":
                return type(self)(self._pa_array.filter(item))
            else:
                raise IndexError(
                    "Only integers, slices and integer or "
                    "boolean arrays are valid indices."
                )
        elif isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)
        if item is Ellipsis:
            item = slice(None)
        if is_scalar(item) and not is_integer(item):
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        if isinstance(item, slice):
            if item.start == item.stop:
                pass
            elif (
                item.stop is not None
                and item.stop < -len(self)
                and item.step is not None
                and item.step < 0
            ):
                item = slice(item.start, None,