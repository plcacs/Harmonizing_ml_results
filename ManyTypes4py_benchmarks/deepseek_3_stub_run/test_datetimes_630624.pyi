"""
Tests for DatetimeArray
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pytest
from pandas._libs.tslibs import tzinfo
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray, TimedeltaArray
from pandas.core.arrays.datetimes import DatetimeArray as DTA
from pandas.core.arrays.timedeltas import TimedeltaArray as TDA
from pandas.tseries.offsets import DateOffset, Tick

class TestNonNano:
    @pytest.fixture
    def unit(self, request: pytest.FixtureRequest) -> str:
        """Fixture returning parametrized time units"""
        ...

    @pytest.fixture
    def dtype(self, unit: str, tz_naive_fixture: Optional[tzinfo]) -> Union[np.dtype, DatetimeTZDtype]:
        ...

    @pytest.fixture
    def dta_dti(self, unit: str, dtype: Union[np.dtype, DatetimeTZDtype]) -> Tuple[DatetimeArray, pd.DatetimeIndex]:
        ...

    @pytest.fixture
    def dta(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> DatetimeArray:
        ...

    def test_non_nano(self, unit: str, dtype: Union[np.dtype, DatetimeTZDtype]) -> None:
        ...

    @pytest.mark.parametrize('field', DatetimeArray._field_ops + DatetimeArray._bool_ops)
    def test_fields(self, unit: str, field: str, dtype: Union[np.dtype, DatetimeTZDtype], dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> None:
        ...

    def test_normalize(self, unit: str) -> None:
        ...

    def test_simple_new_requires_match(self, unit: str) -> None:
        ...

    def test_std_non_nano(self, unit: str) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:Converting to PeriodArray.*:UserWarning')
    def test_to_period(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> None:
        ...

    def test_iter(self, dta: DatetimeArray) -> None:
        ...

    def test_astype_object(self, dta: DatetimeArray) -> None:
        ...

    def test_to_pydatetime(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> None:
        ...

    @pytest.mark.parametrize('meth', ['time', 'timetz', 'date'])
    def test_time_date(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex], meth: str) -> None:
        ...

    def test_format_native_types(self, unit: str, dtype: Union[np.dtype, DatetimeTZDtype], dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> None:
        ...

    def test_repr(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex], unit: str) -> None:
        ...

    def test_compare_mismatched_resolutions(self, comparison_op: Any) -> None:
        ...

    def test_add_mismatched_reso_doesnt_downcast(self) -> None:
        ...

    @pytest.mark.parametrize('scalar', [timedelta(hours=2), pd.Timedelta(hours=2), np.timedelta64(2, 'h'), np.timedelta64(2 * 3600 * 1000, 'ms'), pd.offsets.Minute(120), pd.offsets.Hour(2)])
    def test_add_timedeltalike_scalar_mismatched_reso(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex], scalar: Union[timedelta, pd.Timedelta, np.timedelta64, DateOffset]) -> None:
        ...

    def test_sub_datetimelike_scalar_mismatch(self) -> None:
        ...

    def test_sub_datetime64_reso_mismatch(self) -> None:
        ...

class TestDatetimeArrayComparisons:
    def test_cmp_dt64_arraylike_tznaive(self, comparison_op: Any) -> None:
        ...

class TestDatetimeArray:
    def test_astype_ns_to_ms_near_bounds(self) -> None:
        ...

    def test_astype_non_nano_tznaive(self) -> None:
        ...

    def test_astype_non_nano_tzaware(self) -> None:
        ...

    def test_astype_to_same(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['datetime64[ns]', 'datetime64[ns, UTC]'])
    @pytest.mark.parametrize('other', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, CET]'])
    def test_astype_copies(self, dtype: str, other: str) -> None:
        ...

    @pytest.mark.parametrize('dtype', [int, np.int32, np.int64, 'uint32', 'uint64'])
    def test_astype_int(self, dtype: Union[type, str]) -> None:
        ...

    def test_astype_to_sparse_dt64(self) -> None:
        ...

    def test_tz_setter_raises(self) -> None:
        ...

    def test_setitem_str_impute_tz(self, tz_naive_fixture: Optional[tzinfo]) -> None:
        ...

    def test_setitem_different_tz_raises(self) -> None:
        ...

    def test_setitem_clears_freq(self) -> None:
        ...

    @pytest.mark.parametrize('obj', [pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-01').to_datetime64(), pd.Timestamp('2021-01-01').to_pydatetime()])
    def test_setitem_objects(self, obj: Union[pd.Timestamp, np.datetime64, datetime]) -> None:
        ...

    def test_repeat_preserves_tz(self) -> None:
        ...

    def test_value_counts_preserves_tz(self) -> None:
        ...

    @pytest.mark.parametrize('method', ['pad', 'backfill'])
    def test_fillna_preserves_tz(self, method: str) -> None:
        ...

    def test_fillna_2d(self) -> None:
        ...

    def test_array_interface_tz(self) -> None:
        ...

    def test_array_interface(self) -> None:
        ...

    @pytest.mark.parametrize('index', [True, False])
    def test_searchsorted_different_tz(self, index: bool) -> None:
        ...

    @pytest.mark.parametrize('index', [True, False])
    def test_searchsorted_tzawareness_compat(self, index: bool) -> None:
        ...

    @pytest.mark.parametrize('other', [1, np.int64(1), 1.0, np.timedelta64('NaT'), pd.Timedelta(days=2), 'invalid', np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9, np.arange(10).view('timedelta64[ns]') * 24 * 3600 * 10 ** 9, pd.Timestamp('2021-01-01').to_period('D')])
    @pytest.mark.parametrize('index', [True, False])
    def test_searchsorted_invalid_types(self, other: Any, index: bool) -> None:
        ...

    def test_shift_fill_value(self) -> None:
        ...

    def test_shift_value_tzawareness_mismatch(self) -> None:
        ...

    def test_shift_requires_tzmatch(self) -> None:
        ...

    def test_tz_localize_t2d(self) -> None:
        ...

    @pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern', 'pytz/US/Eastern'])
    def test_iter_zoneinfo_fold(self, tz: str) -> None:
        ...

    @pytest.mark.parametrize('freq', ['2M', '2SM', '2sm', '2Q', '2Q-SEP', '1Y', '2Y-MAR', '2m', '2q-sep', '2y'])
    def test_date_range_frequency_M_Q_Y_raises(self, freq: str) -> None:
        ...

    @pytest.mark.parametrize('freq_depr', ['2MIN', '2nS', '2Us'])
    def test_date_range_uppercase_frequency_deprecated(self, freq_depr: str) -> None:
        ...

    @pytest.mark.parametrize('freq', ['2ye-mar', '2ys', '2qe', '2qs-feb', '2bqs', '2sms', '2bms', '2cbme', '2me'])
    def test_date_range_lowercase_frequency_raises(self, freq: str) -> None:
        ...

    def test_date_range_lowercase_frequency_deprecated(self) -> None:
        ...

    @pytest.mark.parametrize('freq', ['1A', '2A-MAR', '2a-mar'])
    def test_date_range_frequency_A_raises(self, freq: str) -> None:
        ...

    @pytest.mark.parametrize('freq', ['2H', '2CBH', '2S'])
    def test_date_range_uppercase_frequency_raises(self, freq: str) -> None:
        ...

def test_factorize_sort_without_freq() -> None:
    ...