from __future__ import annotations

from datetime import timedelta
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
    overload,
    Union,
    Optional,
    Dict,
    List,
    Tuple,
    Callable,
)
import warnings

import numpy as np
from numpy.typing import NDArray

from pandas._libs import (
    algos as libalgos,
    lib,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Timedelta,
    add_overflowsafe,
    astype_overflowsafe,
    dt64arr_to_periodarr as c_dt64arr_to_periodarr,
    get_unit_from_dtype,
    iNaT,
    parsing,
    period as libperiod,
    to_offset,
)
from pandas._libs.tslibs.dtypes import (
    FreqGroup,
    PeriodDtypeBase,
)
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import (
    Tick,
    delta_to_tick,
)
from pandas._libs.tslibs.period import (
    DIFFERENT_FREQ,
    IncompatibleFrequency,
    Period,
    get_period_field_arr,
    period_asfreq_arr,
)
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

from pandas.core.dtypes.common import (
    ensure_object,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCPeriodIndex,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com

if TYPE_CHECKING:
    from collections.abc import (
        Sequence as ABCSequence,
    )

    from pandas._typing import (
        AnyArrayLike,
        Dtype,
        FillnaOptions,
        NpDtype,
        NumpySorter,
        NumpyValueArrayLike,
        Self,
        npt,
    )

    from pandas.core.dtypes.dtypes import ExtensionDtype

    from pandas.core.arrays import (
        DatetimeArray,
        TimedeltaArray,
    )
    from pandas.core.arrays.base import ExtensionArray


BaseOffsetT = TypeVar("BaseOffsetT", bound=BaseOffset)

_shared_doc_kwargs: Dict[str, str] = {
    "klass": "PeriodArray",
}

def _field_accessor(name: str, docstring: str | None = None) -> property:
    def f(self: PeriodArray) -> NDArray[np.int64]:
        base = self.dtype._dtype_code
        result = get_period_field_arr(name, self.asi8, base)
        return result

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)

class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):
    _typ: str = "periodarray"
    _internal_fill_value: np.int64 = np.int64(iNaT)
    _recognized_scalars: Tuple[type[Period], ...] = (Period,)
    _is_recognized_dtype: Callable[[Any], bool] = lambda x: isinstance(x, PeriodDtype)
    _infer_matches: Tuple[str, ...] = ("period",)

    @property
    def _scalar_type(self) -> type[Period]:
        return Period

    _other_ops: List[str] = []
    _bool_ops: List[str] = ["is_leap_year"]
    _object_ops: List[str] = ["start_time", "end_time", "freq"]
    _field_ops: List[str] = [
        "year", "month", "day", "hour", "minute", "second", "weekofyear", 
        "weekday", "week", "dayofweek", "day_of_week", "dayofyear", 
        "day_of_year", "quarter", "qyear", "days_in_month", "daysinmonth"
    ]
    _datetimelike_ops: List[str] = _field_ops + _object_ops + _bool_ops
    _datetimelike_methods: List[str] = ["strftime", "to_timestamp", "asfreq"]

    _dtype: PeriodDtype

    def __init__(
        self, 
        values: Any, 
        dtype: Dtype | None = None, 
        copy: bool = False
    ) -> None:
        # Implementation remains the same
        pass

    @classmethod
    def _simple_new(
        cls,
        values: npt.NDArray[np.int64],
        dtype: PeriodDtype,
    ) -> Self:
        # Implementation remains the same
        pass

    @classmethod
    def _from_sequence(
        cls,
        scalars: Any,
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> Self:
        # Implementation remains the same
        pass

    @classmethod
    def _from_sequence_of_strings(
        cls, 
        strings: Sequence[str], 
        *, 
        dtype: ExtensionDtype, 
        copy: bool = False
    ) -> Self:
        # Implementation remains the same
        pass

    @classmethod
    def _from_datetime64(
        cls, 
        data: npt.NDArray[np.datetime64], 
        freq: str | Tick, 
        tz: Any = None
    ) -> Self:
        # Implementation remains the same
        pass

    @classmethod
    def _generate_range(
        cls, 
        start: Period | None, 
        end: Period | None, 
        periods: int | None, 
        freq: str | Tick | None
    ) -> Tuple[npt.NDArray[np.int64], BaseOffset]:
        # Implementation remains the same
        pass

    @classmethod
    def _from_fields(cls, *, fields: Dict[str, Any], freq: Any) -> Self:
        # Implementation remains the same
        pass

    def _unbox_scalar(self, value: Period | NaTType) -> np.int64:
        # Implementation remains the same
        pass

    def _scalar_from_string(self, value: str) -> Period:
        # Implementation remains the same
        pass

    def _check_compatible_with(
        self, 
        other: Period | NaTType | PeriodArray
    ) -> None:
        # Implementation remains the same
        pass

    @cache_readonly
    def dtype(self) -> PeriodDtype:
        return self._dtype

    @property
    def freq(self) -> BaseOffset:
        return self.dtype.freq

    @property
    def freqstr(self) -> str:
        return PeriodDtype(self.freq)._freqstr

    def __array__(
        self, 
        dtype: NpDtype | None = None, 
        copy: bool | None = None
    ) -> np.ndarray:
        # Implementation remains the same
        pass

    def __arrow_array__(self, type: Any = None) -> Any:
        # Implementation remains the same
        pass

    year = _field_accessor("year")
    month = _field_accessor("month")
    day = _field_accessor("day")
    hour = _field_accessor("hour")
    minute = _field_accessor("minute")
    second = _field_accessor("second")
    weekofyear = _field_accessor("week")
    week = weekofyear
    day_of_week = _field_accessor("day_of_week")
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor("day_of_year")
    quarter = _field_accessor("quarter")
    qyear = _field_accessor("qyear")
    days_in_month = _field_accessor("days_in_month")
    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> npt.NDArray[np.bool_]:
        return isleapyear_arr(np.asarray(self.year))

    def to_timestamp(
        self, 
        freq: str | BaseOffset | None = None, 
        how: str = "start"
    ) -> DatetimeArray:
        # Implementation remains the same
        pass

    def _box_func(self, x: int) -> Period | NaTType:
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    def asfreq(self, freq: str | BaseOffset | None = None, how: str = "E") -> Self:
        # Implementation remains the same
        pass

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]:
        # Implementation remains the same
        pass

    def _format_native_types(
        self, 
        *, 
        na_rep: str | float = "NaT", 
        date_format: str | None = None, 
        **kwargs: Any
    ) -> npt.NDArray[np.object_]:
        # Implementation remains the same
        pass

    def astype(
        self, 
        dtype: Dtype, 
        copy: bool = True
    ) -> ExtensionArray | np.ndarray:
        # Implementation remains the same
        pass

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # Implementation remains the same
        pass

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # Implementation remains the same
        pass

    def _addsub_int_array_or_scalar(
        self, 
        other: np.ndarray | int, 
        op: Callable[[Any, Any], Any]
    ) -> Self:
        # Implementation remains the same
        pass

    def _add_offset(self, other: BaseOffset) -> Self:
        # Implementation remains the same
        pass

    def _add_timedeltalike_scalar(self, other: Any) -> Self:
        # Implementation remains the same
        pass

    def _add_timedelta_arraylike(
        self, 
        other: TimedeltaArray | npt.NDArray[np.timedelta64]
    ) -> Self:
        # Implementation remains the same
        pass

    def _check_timedeltalike_freq_compat(
        self, 
        other: Any
    ) -> np.ndarray | int:
        # Implementation remains the same
        pass

    def _reduce(
        self, 
        name: str, 
        *, 
        skipna: bool = True, 
        keepdims: bool = False, 
        **kwargs: Any
    ) -> Self | Any:
        # Implementation remains the same
        pass

def raise_on_incompatible(
    left: PeriodArray, 
    right: Any
) -> IncompatibleFrequency:
    # Implementation remains the same
    pass

def period_array(
    data: Sequence[Period | str | None] | AnyArrayLike,
    freq: str | Tick | BaseOffset | None = None,
    copy: bool = False,
) -> PeriodArray:
    # Implementation remains the same
    pass

@overload
def validate_dtype_freq(dtype: Any, freq: BaseOffsetT) -> BaseOffsetT: ...

@overload
def validate_dtype_freq(
    dtype: Any, 
    freq: timedelta | str | None
) -> BaseOffset: ...

def validate_dtype_freq(
    dtype: Any, 
    freq: BaseOffsetT | BaseOffset | timedelta | str | None
) -> BaseOffsetT:
    # Implementation remains the same
    pass

def dt64arr_to_periodarr(
    data: Any, 
    freq: Any, 
    tz: Any = None
) -> Tuple[npt.NDArray[np.int64], BaseOffset]:
    # Implementation remains the same
    pass

def _get_ordinal_range(
    start: Any, 
    end: Any, 
    periods: Any, 
    freq: Any, 
    mult: int = 1
) -> Tuple[np.ndarray, BaseOffset]:
    # Implementation remains the same
    pass

def _range_from_fields(
    year: Any = None,
    month: Any = None,
    quarter: Any = None,
    day: Any = None,
    hour: Any = None,
    minute: Any = None,
    second: Any = None,
    freq: Any = None,
) -> Tuple[np.ndarray, BaseOffset]:
    # Implementation remains the same
    pass

def _make_field_arrays(*fields: Any) -> List[np.ndarray]:
    # Implementation remains the same
    pass
