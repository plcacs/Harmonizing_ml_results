from typing import List, Tuple, Any, Union

def test_values_consistent(arr: np.ndarray, expected_type: Any, dtype: str, using_infer_string: bool):
def test_numpy_array(arr: np.ndarray):
def test_numpy_array_all_dtypes(any_numpy_dtype: Any):
def test_array(arr: Any, attr: str, index_or_series: Any):
def test_array_multiindex_raises():
def test_to_numpy(arr: Any, expected_type: Any, dtype: str, using_infer_string: bool):
def test_to_numpy_copy(arr: np.ndarray, as_series: bool, using_infer_string: bool):
def test_to_numpy_dtype(as_series: bool):
def test_to_numpy_na_value_numpy_dtype(index_or_series: Any, values: List[Any], dtype: str, na_value: Any, expected: List[Any]):
def test_to_numpy_multiindex_series_na_value(data: List[Any], multiindex: List[Tuple[Any, Any]], dtype: Any, na_value: Any, expected: List[Any]):
def test_to_numpy_kwargs_raises():
def test_to_numpy_dataframe_na_value(data: dict, dtype: Any, na_value: Any):
def test_to_numpy_dataframe_single_block(data: dict, expected_data: List[List[Any]]):
def test_to_numpy_dataframe_single_block_no_mutate():
def test_asarray_object_dt64(self, tz: Union[None, str]):
def test_asarray_tz_naive():
def test_asarray_tz_aware():
