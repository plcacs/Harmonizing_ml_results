from typing import Any, Union, List, Dict

def _update_data(data: Union[pd.DataFrame, np.ndarray, Any], idx: int, jdx: int, value: Any) -> Any:
def _check_equals(data1: Any, data2: Any) -> bool:
def test_copy_mode_assign(data: List[Union[str, Dict[str, str]]]):
def test_copy_mode_copy(data: List[Union[Dict[str, str], List[str]]]):
def test_copy_mode_deepcopy(data: List[Union[Dict[str, str], List[str]]]):
def test_copy_mode_invalid_string():
def test_infer_mode_copy(input_data: Any):
def test_infer_mode_deepcopy(data: Union[List[str], List[List[str]], Dict[str, str], Dict[str, str]]):
def test_infer_mode_assign():
def test_is_memory_dataset(ds_or_type: Union[str, type], expected_result: bool):
