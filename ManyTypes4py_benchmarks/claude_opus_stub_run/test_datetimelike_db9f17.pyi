from __future__ import annotations

import numpy as np
import pytest

from pandas._libs import NaT, Timestamp
from pandas._libs.tslibs import to_offset

import pandas as pd
from pandas import DatetimeIndex, Period, PeriodIndex, TimedeltaIndex
from pandas.core.arrays import DatetimeArray, NumpyExtensionArray, PeriodArray, TimedeltaArray


@pytest.fixture(params=["D", "B", "W", "ME", "QE", "YE"])
def freqstr(request: pytest.FixtureRequest) -> str: ...

@pytest.fixture
def period_index(freqstr: str) -> PeriodIndex: ...

@pytest.fixture
def datetime_index(freqstr: str) -> DatetimeIndex: ...

@pytest.fixture
def timedelta_index() -> TimedeltaIndex: ...


class SharedTests:
    index_cls: type
    array_cls: type
    scalar_type: type
    example_dtype: object

    @pytest.fixture
    def arr1d(self) -> DatetimeArray | TimedeltaArray | PeriodArray: ...

    def test_compare_len1_raises(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    @pytest.mark.parametrize("result", [
        pd.date_range("2020", periods=3),
        pd.date_range("2020", periods=3, tz="UTC"),
        pd.timedelta_range("0 days", periods=3),
        pd.period_range("2020Q1", periods=3, freq="Q"),
    ])
    def test_compare_with_Categorical(self, result: DatetimeIndex | TimedeltaIndex | PeriodIndex) -> None: ...

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_compare_categorical_dtype(
        self,
        arr1d: DatetimeArray | TimedeltaArray | PeriodArray,
        as_index: bool,
        reverse: bool,
        ordered: bool,
    ) -> None: ...

    def test_take(self) -> None: ...

    @pytest.mark.parametrize("fill_value", [2, 2.0, Timestamp(2021, 1, 1, 12).time])
    def test_take_fill_raises(self, fill_value: object, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_take_fill(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    @pytest.mark.filterwarnings("ignore:Period with BDay freq is deprecated:FutureWarning")
    def test_take_fill_str(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_concat_same_type(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_unbox_scalar(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_check_compatible_with(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_scalar_from_string(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_reduce_invalid(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_method_doesnt_change_orig(self, method: str) -> None: ...

    def test_searchsorted(self) -> None: ...

    @pytest.mark.parametrize("box", [None, "index", "series"])
    def test_searchsorted_castable_strings(
        self,
        arr1d: DatetimeArray | TimedeltaArray | PeriodArray,
        box: str | None,
        string_storage: str,
    ) -> None: ...

    def test_getitem_near_implementation_bounds(self) -> None: ...

    def test_getitem_2d(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_iter_2d(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_repr_2d(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_setitem(self) -> None: ...

    @pytest.mark.parametrize("box", [pd.Index, pd.Series, np.array, list, NumpyExtensionArray])
    def test_setitem_object_dtype(
        self,
        box: type,
        arr1d: DatetimeArray | TimedeltaArray | PeriodArray,
    ) -> None: ...

    def test_setitem_strs(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    @pytest.mark.parametrize("as_index", [True, False])
    def test_setitem_categorical(
        self,
        arr1d: DatetimeArray | TimedeltaArray | PeriodArray,
        as_index: bool,
    ) -> None: ...

    def test_setitem_raises(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    @pytest.mark.parametrize("box", [list, np.array, pd.Index, pd.Series])
    def test_setitem_numeric_raises(
        self,
        arr1d: DatetimeArray | TimedeltaArray | PeriodArray,
        box: type,
    ) -> None: ...

    def test_inplace_arithmetic(self) -> None: ...

    def test_shift_fill_int_deprecated(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_median(self, arr1d: DatetimeArray | TimedeltaArray | PeriodArray) -> None: ...

    def test_from_integer_array(self) -> None: ...


class TestDatetimeArray(SharedTests):
    index_cls = DatetimeIndex
    array_cls = DatetimeArray
    scalar_type = Timestamp
    example_dtype: str

    @pytest.fixture
    def arr1d(self, tz_naive_fixture: object, freqstr: str) -> DatetimeArray: ...

    def test_round(self, arr1d: DatetimeArray) -> None: ...

    def test_array_interface(self, datetime_index: DatetimeIndex) -> None: ...

    def test_array_object_dtype(self, arr1d: DatetimeArray) -> None: ...

    def test_array_tz(self, arr1d: DatetimeArray) -> None: ...

    def test_array_i8_dtype(self, arr1d: DatetimeArray) -> None: ...

    def test_from_array_keeps_base(self) -> None: ...

    def test_from_dti(self, arr1d: DatetimeArray) -> None: ...

    def test_astype_object(self, arr1d: DatetimeArray) -> None: ...

    @pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
    def test_to_period(self, datetime_index: DatetimeIndex, freqstr: str) -> None: ...

    def test_to_period_2d(self, arr1d: DatetimeArray) -> None: ...

    @pytest.mark.parametrize("propname", DatetimeArray._bool_ops)
    def test_bool_properties(self, arr1d: DatetimeArray, propname: str) -> None: ...

    @pytest.mark.parametrize("propname", DatetimeArray._field_ops)
    def test_int_properties(self, arr1d: DatetimeArray, propname: str) -> None: ...

    def test_take_fill_valid(self, arr1d: DatetimeArray, fixed_now_ts: Timestamp) -> None: ...

    def test_concat_same_type_invalid(self, arr1d: DatetimeArray) -> None: ...

    def test_concat_same_type_different_freq(self, unit: str) -> None: ...

    def test_strftime(self, arr1d: DatetimeArray, using_infer_string: bool) -> None: ...

    def test_strftime_nat(self, using_infer_string: bool) -> None: ...


class TestTimedeltaArray(SharedTests):
    index_cls = TimedeltaIndex
    array_cls = TimedeltaArray
    scalar_type = pd.Timedelta
    example_dtype: str

    def test_from_tdi(self) -> None: ...

    def test_astype_object(self) -> None: ...

    def test_to_pytimedelta(self, timedelta_index: TimedeltaIndex) -> None: ...

    def test_total_seconds(self, timedelta_index: TimedeltaIndex) -> None: ...

    @pytest.mark.parametrize("propname", TimedeltaArray._field_ops)
    def test_int_properties(self, timedelta_index: TimedeltaIndex, propname: str) -> None: ...

    def test_array_interface(self, timedelta_index: TimedeltaIndex) -> None: ...

    def test_take_fill_valid(self, timedelta_index: TimedeltaIndex, fixed_now_ts: Timestamp) -> None: ...


@pytest.mark.filterwarnings("ignore:Period with BDay freq is deprecated:FutureWarning")
@pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
class TestPeriodArray(SharedTests):
    index_cls = PeriodIndex
    array_cls = PeriodArray
    scalar_type = Period
    example_dtype: object

    @pytest.fixture
    def arr1d(self, period_index: PeriodIndex) -> PeriodArray: ...

    def test_from_pi(self, arr1d: PeriodArray) -> None: ...

    def test_astype_object(self, arr1d: PeriodArray) -> None: ...

    def test_take_fill_valid(self, arr1d: PeriodArray) -> None: ...

    @pytest.mark.parametrize("how", ["S", "E"])
    def test_to_timestamp(self, how: str, arr1d: PeriodArray) -> None: ...

    def test_to_timestamp_roundtrip_bday(self) -> None: ...

    def test_to_timestamp_out_of_bounds(self) -> None: ...

    @pytest.mark.parametrize("propname", PeriodArray._bool_ops)
    def test_bool_properties(self, arr1d: PeriodArray, propname: str) -> None: ...

    @pytest.mark.parametrize("propname", PeriodArray._field_ops)
    def test_int_properties(self, arr1d: PeriodArray, propname: str) -> None: ...

    def test_array_interface(self, arr1d: PeriodArray) -> None: ...

    def test_strftime(self, arr1d: PeriodArray, using_infer_string: bool) -> None: ...

    def test_strftime_nat(self, using_infer_string: bool) -> None: ...


@pytest.mark.parametrize(
    "arr,casting_nats",
    [
        (TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data, (NaT, np.timedelta64("NaT", "ns"))),
        (pd.date_range("2000-01-01", periods=3, freq="D")._data, (NaT, np.datetime64("NaT", "ns"))),
        (pd.period_range("2000-01-01", periods=3, freq="D")._data, (NaT,)),
    ],
    ids=lambda x: type(x).__name__,
)
def test_casting_nat_setitem_array(
    arr: TimedeltaArray | DatetimeArray | PeriodArray,
    casting_nats: tuple[object, ...],
) -> None: ...


@pytest.mark.parametrize(
    "arr,non_casting_nats",
    [
        (TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data, (np.datetime64("NaT", "ns"), NaT._value)),
        (pd.date_range("2000-01-01", periods=3, freq="D")._data, (np.timedelta64("NaT", "ns"), NaT._value)),
        (pd.period_range("2000-01-01", periods=3, freq="D")._data, (np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns"), NaT._value)),
    ],
    ids=lambda x: type(x).__name__,
)
def test_invalid_nat_setitem_array(
    arr: TimedeltaArray | DatetimeArray | PeriodArray,
    non_casting_nats: tuple[object, ...],
) -> None: ...


@pytest.mark.parametrize(
    "arr",
    [
        pd.date_range("2000", periods=4).array,
        pd.timedelta_range("2000", periods=4).array,
    ],
)
def test_to_numpy_extra(arr: DatetimeArray | TimedeltaArray) -> None: ...


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "klass",
    [list, np.array, pd.array, pd.Series, pd.Index, pd.Categorical, pd.CategoricalIndex],
)
def test_searchsorted_datetimelike_with_listlike(
    values: DatetimeIndex | TimedeltaIndex | PeriodIndex | DatetimeArray | TimedeltaArray | PeriodArray,
    klass: type,
    as_index: bool,
) -> None: ...


@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "arg",
    [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2],
)
def test_searchsorted_datetimelike_with_listlike_invalid_dtype(
    values: DatetimeIndex | TimedeltaIndex | PeriodIndex,
    arg: list[object],
) -> None: ...


@pytest.mark.parametrize("klass", [list, tuple, np.array, pd.Series])
def test_period_index_construction_from_strings(klass: type) -> None: ...


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
def test_from_pandas_array(dtype: str) -> None: ...