from typing import List, Dict, Union

def test_agg_args(args: Tuple, kwargs: Dict, increment: int):
def test_agg_mapping_func_deprecated():
def test_apply(datetime_series: Series, by_row: Union[bool, str]):
def test_apply_args():
def test_apply_categorical(by_row: Union[bool, str], using_infer_string: bool):
def test_apply_categorical_with_nan_values(series: List, by_row: Union[bool, str]):
def test_apply_dataframe_iloc():
def test_apply_map_same_length_inference_bug():
def test_apply_map_box_timedelta(by_row: Union[bool, str]):
def test_apply_box_dt64():
def test_apply_box_dt64tz():
def test_apply_box_td64():
def test_apply_box_period():
def test_apply_datetimetz(by_row: Union[bool, str]):
def test_apply_scalar_on_date_time_index_aware_series(by_row: Union[bool, str]):
def test_apply_series_on_date_time_index_aware_series(dti: Series, exp: DataFrame, aware: bool):
def test_apply_to_timedelta(by_row: Union[bool, str]):
def test_demo():
def test_reduce(string_series: Series):
def test_replicate_describe(string_series: Series):
def test_series_apply_no_suffix_index(by_row: Union[bool, str]):
def test_transform(string_series: Series, by_row: Union[bool, str]):
def test_transform_partial_failure(op: str, request: Any):
def test_transform_partial_failure_valueerror():
def test_with_nested_series(datetime_series: Series, op_name: str):
def test_with_nested_series(datetime_series: Series, op_name: str):
