from __future__ import annotations
from typing import (
    Any,
    ClassVar,
    Iterator,
    List,
    Literal,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
import numpy as np
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    Period,
    PeriodIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
from pandas._libs import NaTType
from pandas._libs.tslibs import BaseOffset
from pandas.core.arrays import (
    DatetimeArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.dtypes.dtypes import PeriodDtype
import pytest

_T = TypeVar("_T")
_DTArray = TypeVar("_DTArray", DatetimeArray, TimedeltaArray, PeriodArray)

@pytest.fixture
def freqstr(request: pytest.FixtureRequest) -> str: ...
@pytest.fixture
def period_index(freqstr: str) -> PeriodIndex: ...
@pytest.fixture
def datetime_index(freqstr: str) -> DatetimeIndex: ...
@pytest.fixture
def timedelta_index() -> TimedeltaIndex: ...

class SharedTests:
    index_cls: ClassVar[Type[Union[DatetimeIndex, TimedeltaIndex, PeriodIndex]]]
    array_cls: ClassVar[Type[Union[DatetimeArray, TimedeltaArray, PeriodArray]]]
    scalar_type: ClassVar[Type[Union[Timestamp, Timedelta, Period]]]
    example_dtype: ClassVar[Union[str, PeriodDtype]]
    
    @pytest.fixture
    def arr1d(self) -> Union[DatetimeArray, TimedeltaArray, PeriodArray]: ...
    
    def test_compare_len1_raises(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    @pytest.mark.parametrize("result", [
        pd.date_range("2020", periods=3),
        pd.date_range("2020", periods=3, tz="UTC"),
        pd.timedelta_range("0 days", periods=3),
        pd.period_range("2020Q1", periods=3, freq="Q")
    ])
    def test_compare_with_Categorical(
        self, 
        result: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex]
    ) -> None: ...
    
    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_compare_categorical_dtype(
        self,
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray],
        as_index: bool,
        reverse: bool,
        ordered: bool
    ) -> None: ...
    
    def test_take(self) -> None: ...
    
    @pytest.mark.parametrize(
        "fill_value", 
        [2, 2.0, Timestamp(2021, 1, 1, 12).time]
    )
    def test_take_fill_raises(
        self,
        fill_value: Any,
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_take_fill(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_take_fill_str(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_concat_same_type(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_unbox_scalar(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_check_compatible_with(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_scalar_from_string(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_reduce_invalid(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_method_doesnt_change_orig(self, method: str) -> None: ...
    
    def test_searchsorted(self) -> None: ...
    
    @pytest.mark.parametrize("box", [None, "index", "series"])
    def test_searchsorted_castable_strings(
        self,
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray],
        box: Union[None, str],
        string_storage: str
    ) -> None: ...
    
    def test_getitem_near_implementation_bounds(self) -> None: ...
    
    def test_getitem_2d(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_iter_2d(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_repr_2d(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_setitem(self) -> None: ...
    
    @pytest.mark.parametrize(
        "box", 
        [pd.Index, pd.Series, np.array, list, NumpyExtensionArray]
    )
    def test_setitem_object_dtype(
        self,
        box: Type[Union[Index, Series, np.ndarray, list, NumpyExtensionArray]],
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_setitem_strs(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    @pytest.mark.parametrize("as_index", [True, False])
    def test_setitem_categorical(
        self,
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray],
        as_index: bool
    ) -> None: ...
    
    def test_setitem_raises(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    @pytest.mark.parametrize("box", [list, np.array, pd.Index, pd.Series])
    def test_setitem_numeric_raises(
        self,
        box: Type[Union[list, np.ndarray, Index, Series]],
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_inplace_arithmetic(self) -> None: ...
    
    def test_shift_fill_int_deprecated(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_median(
        self, 
        arr1d: Union[DatetimeArray, TimedeltaArray, PeriodArray]
    ) -> None: ...
    
    def test_from_integer_array(self) -> None: ...

class TestDatetimeArray(SharedTests):
    index_cls: ClassVar[Type[DatetimeIndex]] = DatetimeIndex
    array_cls: ClassVar[Type[DatetimeArray]] = DatetimeArray
    scalar_type: ClassVar[Type[Timestamp]] = Timestamp
    example_dtype: ClassVar[str] = "M8[ns]"
    
    @pytest.fixture
    def arr1d(
        self, 
        tz_naive_fixture: Union[None, str], 
        freqstr: str
    ) -> DatetimeArray: ...
    
    def test_round(
        self, 
        arr1d: DatetimeArray
    ) -> None: ...
    
    def test_array_interface(
        self, 
        datetime_index: DatetimeIndex
    ) -> None: ...
    
    def test_array_object_dtype(
        self, 
        arr1d: DatetimeArray
    ) -> None: ...
    
    def test_array_tz(
        self, 
        arr1d: DatetimeArray
    ) -> None: ...
    
    def test_array_i8_dtype(
        self, 
        arr1d: DatetimeArray
    ) -> None: ...
    
    def test_from_array_keeps_base(self) -> None: ...
    
    def test_from_dti(
        self, 
        arr1d: DatetimeArray
    ) -> None: ...
    
    def test_astype_object(
        self, 
        arr1d: DatetimeArray
    ) -> None: ...
    
    @pytest.mark.filterwarnings(
        "ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning"
    )
    def test_to_period(
        self, 
        datetime_index: DatetimeIndex, 
        freqstr: str
    ) -> None: ...
    
    def test_to_period_2d(
        self, 
        arr1d: DatetimeArray
    ) -> None: ...
    
    @pytest.mark.parametrize("propname", DatetimeArray._bool_ops)
    def test_bool_properties(
        self, 
        arr1d: DatetimeArray, 
        propname: str
    ) -> None: ...
    
    @pytest.mark.parametrize("propname", DatetimeArray._field_ops)
    def test_int_properties(
        self, 
        arr1d: DatetimeArray, 
        propname: str
    ) -> None: ...
    
    def test_take_fill_valid(
        self, 
        arr1d: DatetimeArray, 
        fixed_now_ts: Timestamp
    ) -> None: ...
    
    def test_concat_same_type_invalid(
        self, 
        arr1d: DatetimeArray
    ) -> None: ...
    
    def test_concat_same_type_different_freq(
        self, 
        unit: str
    ) -> None: ...
    
    def test_strftime(
        self, 
        arr1d: DatetimeArray, 
        using_infer_string: bool
    ) -> None: ...
    
    def test_strftime_nat(
        self, 
        using_infer_string: bool
    ) -> None: ...

class TestTimedeltaArray(SharedTests):
    index_cls: ClassVar[Type[TimedeltaIndex]] = TimedeltaIndex
    array_cls: ClassVar[Type[TimedeltaArray]] = TimedeltaArray
    scalar_type: ClassVar[Type[Timedelta]] = Timedelta
    example_dtype: ClassVar[str] = "m8[ns]"
    
    def test_from_tdi(self) -> None: ...
    
    def test_astype_object(self) -> None: ...
    
    def test_to_pytimedelta(
        self, 
        timedelta_index: TimedeltaIndex
    ) -> None: ...
    
    def test_total_seconds(
        self, 
        timedelta_index: TimedeltaIndex
    ) -> None: ...
    
    @pytest.mark.parametrize("propname", TimedeltaArray._field_ops)
    def test_int_properties(
        self, 
        timedelta_index: TimedeltaIndex, 
        propname: str
    ) -> None: ...
    
    def test_array_interface(
        self, 
        timedelta_index: TimedeltaIndex
    ) -> None: ...
    
    def test_take_fill_valid(
        self, 
        timedelta_index: TimedeltaIndex, 
        fixed_now_ts: Timestamp
    ) -> None: ...

@pytest.mark.filterwarnings(
    "ignore:Period with BDay freq is deprecated:FutureWarning"
)
@pytest.mark.filterwarnings(
    "ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning"
)
class TestPeriodArray(SharedTests):
    index_cls: ClassVar[Type[PeriodIndex]] = PeriodIndex
    array_cls: ClassVar[Type[PeriodArray]] = PeriodArray
    scalar_type: ClassVar[Type[Period]] = Period
    example_dtype: ClassVar[PeriodDtype] = PeriodIndex([], freq="W").dtype
    
    @pytest.fixture
    def arr1d(
        self, 
        period_index: PeriodIndex
    ) -> PeriodArray: ...
    
    def test_from_pi(
        self, 
        arr1d: PeriodArray
    ) -> None: ...
    
    def test_astype_object(
        self, 
        arr1d: PeriodArray
    ) -> None: ...
    
    def test_take_fill_valid(
        self, 
        arr1d: PeriodArray
    ) -> None: ...
    
    @pytest.mark.parametrize("how", ["S", "E"])
    def test_to_timestamp(
        self, 
        how: str, 
        arr1d: PeriodArray
    ) -> None: ...
    
    def test_to_timestamp_roundtrip_bday(self) -> None: ...
    
    def test_to_timestamp_out_of_bounds(self) -> None: ...
    
    @pytest.mark.parametrize("propname", PeriodArray._bool_ops)
    def test_bool_properties(
        self, 
        arr1d: PeriodArray, 
        propname: str
    ) -> None: ...
    
    @pytest.mark.parametrize("propname", PeriodArray._field_ops)
    def test_int_properties(
        self, 
        arr1d: PeriodArray, 
        propname: str
    ) -> None: ...
    
    def test_array_interface(
        self, 
        arr1d: PeriodArray
    ) -> None: ...
    
    def test_strftime(
        self, 
        arr1d: PeriodArray, 
        using_infer_string: bool
    ) -> None: ...
    
    def test_strftime_nat(
        self, 
        using_infer_string: bool
    ) -> None: ...

@pytest.mark.parametrize(
    "arr,casting_nats",
    [
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (NaTType, np.timedelta64)
        ),
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (NaTType, np.datetime64)
        ),
        (
            pd.period_range("2000-01-01", periods=3, freq="D")._data,
            (NaTType,)
        )
    ],
    ids=lambda x: type(x).__name__
)
def test_casting_nat_setitem_array(
    arr: Union[DatetimeArray, TimedeltaArray, PeriodArray],
    casting_nats: Tuple[Union[Type[NaTType], Type[np.datetime64], Type[np.timedelta64]], ...]
) -> None: ...

@pytest.mark.parametrize(
    "arr,non_casting_nats",
    [
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (np.datetime64, int)
        ),
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (np.timedelta64, int)
        ),
        (
            pd.period_range("2000-01-01", periods=3, freq="D")._data,
            (np.datetime64, np.timedelta64, int)
        )
    ],
    ids=lambda x: type(x).__name__
)
def test_invalid_nat_setitem_array(
    arr: Union[DatetimeArray, TimedeltaArray, PeriodArray],
    non_casting_nats: Tuple[Union[Type[np.datetime64], Type[np.timedelta64], Type[int]], ...]
) -> None: ...

@pytest.mark.parametrize(
    "arr",
    [
        pd.date_range("2000", periods=4).array,
        pd.timedelta_range("2000", periods=4).array
    ]
)
def test_to_numpy_extra(
    arr: Union[DatetimeArray, TimedeltaArray]
) -> None: ...

@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D")
    ]
)
@pytest.mark.parametrize(
    "klass",
    [
        list,
        np.array,
        pd.array,
        pd.Series,
        pd.Index,
        pd.Categorical,
        pd.CategoricalIndex
    ]
)
def test_searchsorted_datetimelike_with_listlike(
    values: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex],
    klass: Type[Union[list, np.ndarray, pd.array, Series, Index, Categorical, CategoricalIndex]],
    as_index: bool
) -> None: ...

@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D")
    ]
)
@pytest.mark.parametrize(
    "arg",
    [
        [1, 2],
        ["a", "b"],
        [Timestamp("2020-01-01", tz="Europe/London")] * 2
    ]
)
def test_searchsorted_datetim