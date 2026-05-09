from datetime import datetime
from functools import partial
from zoneinfo import ZoneInfo
from numpy import float64, int64
from pandas import (
    DataFrame,
    Series,
    DatetimeIndex,
    Period,
    Timestamp,
    Timedelta,
    Index,
    isna,
    notna,
)
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range
from pandas.core.resample import _get_timestamp_range_edges
from pandas.tseries.offsets import Minute
from pytest import fixture, mark, raises
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

@fixture
def simple_date_range_series() -> Callable[[str, str, str], Series]:
    ...

def test_custom_grouper(unit: str) -> None:
    ...

def test_custom_grouper_df(unit: str) -> None:
    ...

@mark.parametrize('closed, expected', [('right', Callable[[Series], Series]), ('left', Callable[[Series], Series])])
def test_resample_basic(closed: str, expected: Callable[[Series], Series], unit: str) -> None:
    ...

def test_resample_integerarray(unit: str) -> None:
    ...

def test_resample_basic_grouper(unit: str) -> None:
    ...

@mark.filterwarnings("ignore:The 'convention' keyword in Series.resample:FutureWarning")
@mark.parametrize('keyword, value', [('label', str), ('closed', str), ('convention', str)])
def test_resample_string_kwargs(keyword: str, value: str, unit: str) -> None:
    ...

def test_resample_how(downsample_method: str, unit: str) -> None:
    ...

def test_resample_how_ohlc(unit: str) -> None:
    ...

def test_resample_how_callables(unit: str) -> None:
    ...

def test_resample_rounding(unit: str) -> None:
    ...

def test_resample_basic_from_daily(unit: str) -> None:
    ...

def test_resample_upsampling_picked_but_not_correct(unit: str) -> None:
    ...

@mark.parametrize('f', ['sum', 'mean', 'prod', 'min', 'max', 'var'])
def test_resample_frame_basic_cy_funcs(f: str, unit: str) -> None:
    ...

@mark.parametrize('freq', ['YE', 'ME'])
def test_resample_frame_basic_M_A(freq: str, unit: str) -> None:
    ...

def test_resample_upsample(unit: str) -> None:
    ...

def test_resample_how_method(unit: str) -> None:
    ...

def test_resample_extra_index_point(unit: str) -> None:
    ...

def test_upsample_with_limit(unit: str) -> None:
    ...

@mark.parametrize('freq', ['1D', '10h', '5Min', '10s'])
@mark.parametrize('rule', ['YE', '3ME', '15D', '30h', '15Min', '30s'])
def test_nearest_upsample_with_limit(tz_aware_fixture: Any, freq: str, rule: str, unit: str) -> None:
    ...

def test_resample_ohlc(unit: str) -> None:
    ...

def test_resample_ohlc_result(unit: str) -> None:
    ...

def test_resample_ohlc_result_odd_period(unit: str) -> None:
    ...

def test_resample_ohlc_dataframe(unit: str) -> None:
    ...

def test_resample_reresample(unit: str) -> None:
    ...

@mark.parametrize('freq, expected_kwargs', [['YE-DEC', Dict[str, Any]], ['YE-JUN', Dict[str, Any]], ['ME', Dict[str, Any]]])
def test_resample_timestamp_to_period(simple_date_range_series: Callable[[str, str, str], Series], freq: str, expected_kwargs: Dict[str, Any], unit: str) -> None:
    ...

def test_ohlc_5min(unit: str) -> None:
    ...

def test_downsample_non_unique(unit: str) -> None:
    ...

def test_asfreq_non_unique(unit: str) -> None:
    ...

@mark.parametrize('freq', ['min', '5min', '15min', '30min', '4h', '12h'])
def test_resample_anchored_ticks(freq: str, unit: str) -> None:
    ...

@mark.parametrize('end', [1, 2])
def test_resample_single_group(end: int, unit: str) -> None:
    ...

def test_resample_single_group_std(unit: str) -> None:
    ...

def test_resample_offset(unit: str) -> None:
    ...

@mark.parametrize('kwargs', [{'origin': '1999-12-31 23:57:00'}, {'origin': Timestamp}, {'origin': 'epoch', 'offset': '2m'}, {'origin': '1999-12-31 12:02:00'}, {'offset': '-3m'}])
def test_resample_origin(kwargs: Dict[str, Any], unit: str) -> None:
    ...

@mark.parametrize('origin', ['invalid_value', 'epch', 'startday', 'startt', '2000-30-30', object()])
def test_resample_bad_origin(origin: Any, unit: str) -> None:
    ...

@mark.parametrize('offset', ['invalid_value', '12dayys', '2000-30-30', object()])
def test_resample_bad_offset(offset: Any, unit: str) -> None:
    ...

def test_resample_origin_prime_freq(unit: str) -> None:
    ...

def test_resample_origin_with_tz(unit: str) -> None:
    ...

def test_resample_origin_with_day_freq_on_dst(unit: str) -> None:
    ...

def test_resample_dst_midnight_last_nonexistent() -> None:
    ...

def test_resample_daily_anchored(unit: str) -> None:
    ...

def test_resample_to_period_monthly_buglet(unit: str) -> None:
    ...

def test_period_with_agg() -> None:
    ...

def test_resample_segfault(unit: str) -> None:
    ...

def test_resample_dtype_preservation(unit: str) -> None:
    ...

def test_resample_dtype_coercion(unit: str) -> None:
    ...

def test_weekly_resample_buglet(unit: str) -> None:
    ...

def test_monthly_resample_error(unit: str) -> None:
    ...

def test_nanosecond_resample_error() -> None:
    ...

def test_resample_anchored_intraday(unit: str) -> None:
    ...

def test_resample_anchored_intraday2(unit: str) -> None:
    ...

def test_resample_anchored_intraday3(simple_date_range_series: Callable[[str, str, str], Series], unit: str) -> None:
    ...

@mark.parametrize('freq', ['MS', 'BMS', 'QS-MAR', 'YS-DEC', 'YS-JUN'])
def test_resample_anchored_monthstart(simple_date_range_series: Callable[[str, str, str], Series], freq: str, unit: str) -> None:
    ...

@mark.parametrize('label, sec', [[None, 2.0], ['right', '4.2']])
def test_resample_anchored_multiday(label: Optional[str], sec: float) -> None:
    ...

def test_corner_cases(unit: str) -> None:
    ...

def test_corner_cases_date(simple_date_range_series: Callable[[str, str, str], Series], unit: str) -> None:
    ...

def test_anchored_lowercase_buglet(unit: str) -> None:
    ...

def test_upsample_apply_functions(unit: str) -> None:
    ...

def test_resample_not_monotonic(unit: str) -> None:
    ...

@mark.parametrize('dtype', ['int64', 'int32', 'float64', pytest.param('float32', marks=pytest.mark.xfail(reason='Empty groups cause x.mean() to return float64'))])
def test_resample_median_bug_1688(dtype: str, unit: str) -> None:
    ...

def test_how_lambda_functions(simple_date_range_series: Callable[[str, str, str], Series], unit: str) -> None:
    ...

def test_resample_unequal_times(unit: str) -> None:
    ...

def test_resample_consistency(unit: str) -> None:
    ...

@mark.parametrize('dates', [list, list, list])
def test_resample_timegrouper(dates: List[datetime], unit: str) -> None:
    ...

@mark.parametrize('dates', [list, list, list])
def test_resample_timegrouper2(dates: List[datetime], unit: str) -> None:
    ...

def test_resample_nunique(unit: str) -> None:
    ...

def test_resample_nunique_preserves_column_level_names(unit: str) -> None:
    ...

@mark.parametrize('func', [lambda x: x.nunique(), lambda x: x.agg(Series.nunique), lambda x: x.agg('nunique')])
def test_resample_nunique_with_date_gap(func: Callable[[Any], Any], unit: str) -> None:
    ...

def test_resample_group_info(unit: str) -> None:
    ...

def test_resample_size(unit: str) -> None:
    ...

def test_resample_across_dst() -> None:
    ...

def test_groupby_with_dst_time_change(unit: str) -> None:
    ...

def test_resample_dst_anchor(unit: str) -> None:
    ...

def test_resample_dst_anchor2(unit: str) -> None:
    ...

def test_resample_with_nat(unit: str) -> None:
    ...

def test_resample_datetime_values(unit: str) -> None:
    ...

def test_resample_apply_with_additional_args(unit: str) -> None:
    ...

def test_resample_apply_with_additional_args2() -> None:
    ...

@mark.parametrize('k', [1, 2, 3])
@mark.parametrize('n1, freq1, n2, freq2', [(30, 's', 0.5, 'Min'), (60, 's', 1, 'Min'), (3600, 's', 1, 'h'), (60, 'Min', 1, 'h'), (21600, 's', 0.25, 'D'), (86400, 's', 1, 'D'), (43200, 's', 0.5, 'D'), (1440, 'Min', 1, 'D'), (12, 'h', 0.5, 'D'), (24, 'h', 1, 'D')])
def test_resample_equivalent_offsets(n1: int, freq1: str, n2: float, freq2: str, k: int, unit: str) -> None:
    ...

@mark.parametrize('first,last,freq,exp_first,exp_last', [('19910905', '19920406', 'D', '19910905', '19920407'), ('19910905 00:00', '19920406 06:00', 'D', '19910905', '19920407'), ('19910905 06:00', '19920406 06:00', 'h', '19910905 06:00', '19920406 07:00'), ('19910906', '19920406', 'ME', '19910831', '19920430'), ('19910831', '19920430', 'ME', '19910831', '19920531'), ('1991-08', '1992-04', 'ME', '19910831', '19920531')])
def test_get_timestamp_range_edges(first: str, last: str, freq: str, exp_first: str, exp_last: str, unit: str) -> None:
    ...

@mark.parametrize('duplicates', [True, False])
def test_resample_apply_product(duplicates: bool, unit: str) -> None:
    ...

@mark.parametrize('first,last,freq_in,freq_out,exp_last', [('2020-03-28', '2020-03-31', 'D', '24h', '2020-03-30 01:00'), ('2020-03-28', '2020-10-27', 'D', '24h', '2020-10-27 00:00'), ('2020-10-25', '2020-10-27', 'D', '24h', '2020-10-26 23:00'), ('2020-03-28', '2020-03-31', '24h', 'D', '2020-03-30 00:00'), ('2020-03-28', '2020-10-27', '24h', 'D', '2020-10-27 00:00'), ('2020-10-25', '2020-10-27', '24h', 'D', '2020-10-26 00:00')])
def test_resample_calendar_day_with_dst(first: str, last: str, freq_in: str, freq_out: str, exp_last: str, unit: str) -> None:
    ...

@mark.parametrize('func', ['min', 'max', 'first', 'last'])
def test_resample_aggregate_functions_min_count(func: str, unit: str) -> None:
    ...

def test_resample_unsigned_int(any_unsigned_int_numpy_dtype: Any, unit: str) -> None:
    ...

def test_long_rule_non_nano() -> None:
    ...

def test_resample_empty_series_with_tz() -> None:
    ...

@mark.parametrize('freq', ['2M', '2m', '2Q', '2Q-SEP', '2q-sep', '1Y', '2Y-MAR'])
def test_resample_M_Q_Y_raises(freq: str) -> None:
    ...

@mark.parametrize('freq', ['2BM', '1bm', '1BQ', '2BQ-MAR', '2bq=-mar'])
def test_resample_BM_BQ_raises(freq: str) -> None:
    ...

@mark.parametrize('freq,freq_depr,data', [('1W-SUN', '1w-sun', ['2013-01-06']), ('1D', '1d', ['2013-01-01']), ('1B', '1b', ['2013-01-01']), ('1C', '1c', ['2013-01-01'])])
def test_resample_depr_lowercase_frequency(freq: str, freq_depr: str, data: List[str]) -> None:
    ...

def test_resample_ms_closed_right(unit: str) -> None:
    ...

@mark.parametrize('freq', ['B', 'C'])
def test_resample_c_b_closed_right(freq: str, unit: str) -> None:
    ...

def test_resample_b_55282(unit: str) -> None:
    ...

@td.skip_if_no('pyarrow')
@mark.parametrize('tz', [None, pytest.param('UTC', marks=pytest.mark.xfail(condition=is_platform_windows(), reason='TODO: Set ARROW_TIMEZONE_DATABASE env var in CI'))])
def test_arrow_timestamp_resample(tz: Optional[str]) -> None:
    ...

@mark.parametrize('freq', ['1A', '2A-MAR'])
def test_resample_A_raises(freq: str) -> None:
    ...