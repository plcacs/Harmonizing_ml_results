from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.indexes.datetimes import DatetimeIndex

@pytest.fixture
def dti() -> DatetimeIndex:
    ...

@pytest.fixture
def _test_series(dti: DatetimeIndex) -> Series[float]:
    ...

@pytest.fixture
def test_frame(dti: DatetimeIndex, _test_series: Series[float]) -> DataFrame:
    ...

def test_str(_test_series: Series[float]) -> None:
    ...

def test_api(_test_series: Series[float]) -> None:
    ...

def test_groupby_resample_api() -> None:
    ...

def test_groupby_resample_on_api() -> None:
    ...

def test_resample_group_keys() -> None:
    ...

def test_pipe(test_frame: DataFrame, _test_series: Series[float]) -> None:
    ...

def test_getitem(test_frame: DataFrame) -> None:
    ...

@pytest.mark.parametrize('key', [['D'], ['A', 'D']])
def test_select_bad_cols(key: List[str], test_frame: DataFrame) -> None:
    ...

def test_attribute_access(test_frame: DataFrame) -> None:
    ...

@pytest.mark.parametrize('attr', ['groups', 'ngroups', 'indices'])
def test_api_compat_before_use(attr: str) -> None:
    ...

def tests_raises_on_nuisance(test_frame: DataFrame, using_infer_string: bool) -> None:
    ...

def test_downsample_but_actually_upsampling() -> None:
    ...

def test_combined_up_downsampling_of_irregular() -> None:
    ...

def test_transform_series(_test_series: Series[float]) -> None:
    ...

@pytest.mark.parametrize('on', [None, 'date'])
def test_transform_frame(on: Optional[str]) -> None:
    ...

@pytest.mark.parametrize('func', [lambda x: x.resample('20min', group_keys=False), lambda x: x.groupby(pd.Grouper(freq='20min'), group_keys=False)], ids=['resample', 'groupby'])
def test_apply_without_aggregation(func: Any, _test_series: Series[float]) -> None:
    ...

def test_apply_without_aggregation2(_test_series: Series[float]) -> None:
    ...

def test_agg_consistency() -> None:
    ...

def test_agg_consistency_int_str_column_mix() -> None:
    ...

@pytest.fixture
def index() -> DatetimeIndex:
    ...

@pytest.fixture
def df(index: DatetimeIndex) -> DataFrame:
    ...

@pytest.fixture
def df_col(df: DataFrame) -> DataFrame:
    ...

@pytest.fixture
def df_mult(df_col: DataFrame, index: DatetimeIndex) -> DataFrame:
    ...

@pytest.fixture
def a_mean(df: DataFrame) -> Series[float]:
    ...

@pytest.fixture
def a_std(df: DataFrame) -> Series[float]:
    ...

@pytest.fixture
def a_sum(df: DataFrame) -> Series[float]:
    ...

@pytest.fixture
def b_mean(df: DataFrame) -> Series[float]:
    ...

@pytest.fixture
def b_std(df: DataFrame) -> Series[float]:
    ...

@pytest.fixture
def b_sum(df: DataFrame) -> Series[float]:
    ...

@pytest.fixture
def df_resample(df: DataFrame) -> Any:
    ...

@pytest.fixture
def df_col_resample(df_col: DataFrame) -> Any:
    ...

@pytest.fixture
def df_mult_resample(df_mult: DataFrame) -> Any:
    ...

@pytest.fixture
def df_grouper_resample(df: DataFrame) -> Any:
    ...

@pytest.fixture(params=['df_resample', 'df_col_resample', 'df_mult_resample', 'df_grouper_resample'])
def cases(request: Any) -> Any:
    ...

def test_agg_mixed_column_aggregation(cases: Any, a_mean: Series[float], a_std: Series[float], b_mean: Series[float], b_std: Series[float], request: Any) -> None:
    ...

@pytest.mark.parametrize('agg', [{'func': {'A': np.mean, 'B': lambda x: np.std(x, ddof=1)}}, {'A': ('A', np.mean), 'B': ('B', lambda x: np.std(x, ddof=1))}, {'A': NamedAgg('A', np.mean), 'B': NamedAgg('B', lambda x: np.std(x, ddof=1))}])
def test_agg_both_mean_std_named_result(cases: Any, a_mean: Series[float], b_std: Series[float], agg: Dict[str, Any]) -> None:
    ...

def test_agg_both_mean_std_dict_of_list(cases: Any, a_mean: Series[float], a_std: Series[float]) -> None:
    ...

@pytest.mark.parametrize('agg', [{'func': ['mean', 'sum']}, {'mean': 'mean', 'sum': 'sum'}])
def test_agg_both_mean_sum(cases: Any, a_mean: Series[float], a_sum: Series[float], agg: Dict[str, Union[List[str], Dict[str, str]]]) -> None:
    ...

@pytest.mark.parametrize('agg', [{'func': {'result1': np.sum, 'result2': np.mean}}, {'A': ('result1', np.sum), 'B': ('result2', np.mean)}, {'A': NamedAgg('result1', np.sum), 'B': NamedAgg('result2', np.mean)}])
def test_agg_no_column(cases: Any, agg: Dict[str, Union[List[str], Dict[str, str]]]) -> None:
    ...

@pytest.mark.parametrize('cols, agg', [[None, {'A': ['sum', 'std'], 'B': ['mean', 'std']}], [['A', 'B'], {'A': ['sum', 'std'], 'B': ['mean', 'std']}]])
def test_agg_specificationerror_nested(cases: Any, cols: Optional[List[str]], agg: Dict[str, List[str]], a_sum: Series[float], a_std: Series[float], b_mean: Series[float], b_std: Series[float]) -> None:
    ...

@pytest.mark.parametrize('agg', [{'A': ['sum', 'std']}, {'A': ['sum', 'std'], 'B': ['mean', 'std']}])
def test_agg_specificationerror_series(cases: Any, agg: Dict[str, List[str]]) -> None:
    ...

def test_agg_specificationerror_invalid_names(cases: Any) -> None:
    ...

def test_agg_nested_dicts() -> None:
    ...

def test_try_aggregate_non_existing_column() -> None:
    ...

def test_agg_list_like_func_with_args() -> None:
    ...

def test_selection_api_validation() -> None:
    ...

@pytest.mark.parametrize('col_name', ['t2', 't2x', 't2q', 'T_2M', 't2p', 't2m', 't2m1', 'T2M'])
def test_agg_with_datetime_index_list_agg_func(col_name: str) -> None:
    ...

def test_resample_agg_readonly() -> None:
    ...

@pytest.mark.parametrize('start,end,freq,data,resample_freq,origin,closed,exp_data,exp_end,exp_periods', [('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end', None, [0, 18, 27, 63], '20001002 00:26:00', 4), ('20200101 8:26:35', '20200101 9:31:58', '77s', [1] * 51, '7min', 'end', 'right', [1, 6, 5, 6, 5, 6, 5, 6, 5, 6], '2020-01-01 09:30:45', 10), ('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end', 'left', [0, 18, 27, 39, 24], '20001002 00:43:00', 5), ('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end_day', None, [3, 15, 45, 45], '2000-10-02 00:29:00', 4)])
def test_end_and_end_day_origin(start: str, end: str, freq: str, data: List[int], resample_freq: str, origin: str, closed: Optional[str], exp_data: List[int], exp_end: str, exp_periods: int) -> None:
    ...

@pytest.mark.parametrize('method, numeric_only, expected_data', [('sum', True, {'num': [25]}), ('sum', False, {'cat': ['cat_1cat_2'], 'num': [25]}), ('sum', lib.no_default, {'cat': ['cat_1cat_2'], 'num': [25]}), ('prod', True, {'num': [100]}), ('prod', False, "can't multiply sequence"), ('prod', lib.no_default, "can't multiply sequence"), ('min', True, {'num': [5]}), ('min', False, {'cat': ['cat_1'], 'num': [5]}), ('min', lib.no_default, {'cat': ['cat_1'], 'num': [5]}), ('max', True, {'num': [20]}), ('max', False, {'cat': ['cat_2'], 'num': [20]}), ('max', lib.no_default, {'cat': ['cat_2'], 'num': [20]}), ('first', True, {'num': [5]}), ('first', False, {'cat': ['cat_1'], 'num': [5]}), ('first', lib.no_default, {'cat': ['cat_1'], 'num': [5]}), ('last', True, {'num': [20]}), ('last', False, {'cat': ['cat_2'], 'num': [20]}), ('last', lib.no_default, {'cat': ['cat_2'], 'num': [20]}), ('mean', True, {'num': [12.5]}), ('mean', False, 'Could not convert'), ('mean', lib.no_default, 'Could not convert'), ('median', True, {'num': [12.5]}), ('median', False, "Cannot convert \\['cat_1' 'cat_2'\\] to numeric"), ('median', lib.no_default, "Cannot convert \\['cat_1' 'cat_2'\\] to numeric"), ('std', True, {'num': [10.606601717798213]}), ('std', False, 'could not convert string to float'), ('std', lib.no_default, 'could not convert string to float'), ('var', True, {'num': [112.5]}), ('var', False, 'could not convert string to float'), ('var', lib.no_default, 'could not convert string to float'), ('sem', True, {'num': [7.5]}), ('sem', False, 'could not convert string to float'), ('sem', lib.no_default, 'could not convert string to float')])
def test_frame_downsample_method(method: str, numeric_only: Union[bool, Any], expected_data: Union[Dict[str, List[Any]], str], using_infer_string: bool) -> None:
    ...

@pytest.mark.parametrize('method, numeric_only, expected_data', [('sum', True, ()), ('sum', False, ['cat_1cat_2']), ('sum', lib.no_default, ['cat_1cat_2']), ('prod', True, ()), ('prod', False, ()), ('prod', lib.no_default, ()), ('min', True, ()), ('min', False, ['cat_1']), ('min', lib.no_default, ['cat_1']), ('max', True, ()), ('max', False, ['cat_2']), ('max', lib.no_default, ['cat_2']), ('first', True, ()), ('first', False, ['cat_1']), ('first', lib.no_default, ['cat_1']), ('last', True, ()), ('last', False, ['cat_2']), ('last', lib.no_default, ['cat_2'])])
def test_series_downsample_method(method: str, numeric_only: Union[bool, Any], expected_data: Union[Tuple, List[str]], using_infer_string: bool) -> None:
    ...

def test_resample_empty() -> None:
    ...