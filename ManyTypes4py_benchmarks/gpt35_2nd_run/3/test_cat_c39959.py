from typing import Any, Union, List, Tuple

def test_str_cat_name(index_or_series: Any, other: Union[None, Any]):
def test_str_cat(index_or_series: Any, infer_string: bool):
def test_str_cat_raises_intuitive_error(index_or_series: Any):
def test_str_cat_categorical(index_or_series: Any, dtype_caller: str, dtype_target: str, sep: Union[str, None], infer_string: bool):
def test_str_cat_wrong_dtype_raises(box: Any, data: List[Union[int, float, str]]):
def test_str_cat_mixed_inputs(index_or_series: Any):
def test_str_cat_align_indexed(index_or_series: Any, join_type: str):
def test_str_cat_align_mixed_inputs(join_type: str):
def test_str_cat_all_na(index_or_series: Any, index_or_series2: Any):
def test_str_cat_special_cases():
def test_cat_on_filtered_index():
def test_cat_different_classes(klass: Union[Tuple, List, np.ndarray, Series, Index]):
def test_cat_on_series_dot_str():
