import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, concat, date_range, timedelta_range
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
from typing import Union, Optional, List, Dict, Any

@pytest.fixture
def by_row(request) -> Union[bool, str]:
    ...

def test_series_map_box_timedelta(by_row: Union[bool, str]) -> None:
    ...

def test_apply(datetime_series: Series, by_row: Union[bool, str]) -> None:
    ...

def test_apply_map_same_length_inference_bug() -> None:
    ...

def test_apply_args() -> None:
    ...

@pytest.mark.parametrize('args, kwargs, increment', [((), {}, 0), ((), {'a': 1}, 1), ((2, 3), {}, 32), ((1,), {'c': 2}, 201)])
def test_agg_args(args: tuple, kwargs: dict, increment: int) -> None:
    ...

def test_agg_mapping_func_deprecated() -> None:
    ...

def test_series_apply_map_box_timestamps(by_row: Union[bool, str]) -> None:
    ...

def test_apply_box_dt64() -> None:
    ...

def test_apply_box_dt64tz() -> None:
    ...

def test_apply_box_td64() -> None:
    ...

def test_apply_box_period() -> None:
    ...

def test_apply_datetimetz(by_row: Union[bool, str]) -> None:
    ...

def test_apply_categorical(by_row: Union[bool, str], using_infer_string: Optional[bool] = None) -> None:
    ...

@pytest.mark.parametrize('series', [['1-1', '1-1', np.nan], ['1-1', '1-2', np.nan]])
def test_apply_categorical_with_nan_values(series: List[str], by_row: Union[bool, str]) -> None:
    ...

def test_apply_empty_integer_series_with_datetime_index(by_row: Union[bool, str]) -> None:
    ...

def test_apply_dataframe_iloc() -> None:
    ...

def test_transform(string_series: Series, by_row: Union[bool, str]) -> None:
    ...

@pytest.mark.parametrize('op', series_transform_kernels)
def test_transform_partial_failure(op: str, request: pytest.FixtureRequest) -> None:
    ...

def test_transform_partial_failure_valueerror() -> None:
    ...

def test_demo() -> None:
    ...

@pytest.mark.parametrize('func', [str, lambda x: str(x)])
def test_apply_map_evaluate_lambdas_the_same(string_series: Series, func: Union[type, callable], by_row: Union[bool, str]) -> None:
    ...

def test_agg_evaluate_lambdas(string_series: Series) -> None:
    ...

@pytest.mark.parametrize('op_name', ['agg', 'apply'])
def test_with_nested_series(datetime_series: Series, op_name: str) -> None:
    ...

def test_replicate_describe(string_series: Series) -> None:
    ...

def test_reduce(string_series: Series) -> None:
    ...

@pytest.mark.parametrize('how, kwds', [('agg', {}), ('apply', {'by_row': 'compat'}), ('apply', {'by_row': False})])
def test_non_callable_aggregates(how: str, kwds: dict) -> None:
    ...

def test_series_apply_no_suffix_index(by_row: Union[bool, str]) -> None:
    ...

@pytest.mark.parametrize('dti, exp', [(Series([1, 2], index=pd.DatetimeIndex([0, 31536000000])), DataFrame(np.repeat([[1, 2]], 2, axis=0), dtype='int64')), (Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts'), DataFrame(np.repeat([[1, 2]], 10, axis=0), dtype='int64'))])
@pytest.mark.parametrize('aware', [True, False])
def test_apply_series_on_date_time_index_aware_series(dti: Series, exp: DataFrame, aware: bool) -> None:
    ...

@pytest.mark.parametrize('by_row, expected', [('compat', Series(np.ones(10), dtype='int64')), (False, 1)])
def test_apply_scalar_on_date_time_index_aware_series(by_row: Union[bool, str], expected: Union[Series, int]) -> None:
    ...

def test_apply_to_timedelta(by_row: Union[bool, str]) -> None:
    ...

@pytest.mark.parametrize('ops, names', [([np.sum], ['sum']), ([np.sum, np.mean], ['sum', 'mean']), (np.array([np.sum]), ['sum']), (np.array([np.sum, np.mean]), ['sum', 'mean'])])
@pytest.mark.parametrize('how, kwargs', [['agg', {}], ['apply', {'by_row': 'compat'}], ['apply', {'by_row': False}]])
def test_apply_listlike_reducer(string_series: Series, ops: Union[List[callable], np.ndarray], names: List[str], how: str, kwargs: dict) -> None:
    ...

@pytest.mark.parametrize('ops', [{'A': np.sum}, {'A': np.sum, 'B': np.mean}, Series({'A': np.sum}), Series({'A': np.sum, 'B': np.mean})])
@pytest.mark.parametrize('how, kwargs', [['agg', {}], ['apply', {'by_row': 'compat'}], ['apply', {'by_row': False}]])
def test_apply_dictlike_reducer(string_series: Series, ops: Union[Dict[str, callable], Series], how: str, kwargs: dict, by_row: Union[bool, str]) -> None:
    ...

@pytest.mark.parametrize('ops, names', [([np.sqrt], ['sqrt']), ([np.abs, np.sqrt], ['absolute', 'sqrt']), (np.array([np.sqrt]), ['sqrt']), (np.array([np.abs, np.sqrt]), ['absolute', 'sqrt'])])
def test_apply_listlike_transformer(string_series: Series, ops: Union[List[callable], np.ndarray], names: List[str], by_row: Union[bool, str]) -> None:
    ...

@pytest.mark.parametrize('ops, expected', [([lambda x: x], DataFrame({'<lambda>': [1, 2, 3]})), ([lambda x: x.sum()], Series([6], index=['<lambda>']))])
def test_apply_listlike_lambda(ops: List[callable], expected: Union[DataFrame, Series], by_row: Union[bool, str]) -> None:
    ...

@pytest.mark.parametrize('ops', [{'A': np.sqrt}, {'A': np.sqrt, 'B': np.exp}, Series({'A': np.sqrt}), Series({'A': np.sqrt, 'B': np.exp})])
def test_apply_dictlike_transformer(string_series: Series, ops: Union[Dict[str, callable], Series], by_row: Union[bool, str]) -> None:
    ...

@pytest.mark.parametrize('ops, expected', [({'a': lambda x: x}, Series([1, 2, 3], index=MultiIndex.from_arrays([['a'] * 3, range(3)]))), ({'a': lambda x: x.sum()}, Series([6], index=['a']))])
def test_apply_dictlike_lambda(ops: Dict[str, callable], by_row: Union[bool, str], expected: Series) -> None:
    ...

def test_apply_retains_column_name(by_row: Union[bool, str]) -> None:
    ...

def test_apply_type() -> None:
    ...

def test_series_apply_unpack_nested_data() -> None:
    ...