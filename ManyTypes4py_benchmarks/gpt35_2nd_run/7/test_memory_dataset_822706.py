from typing import Any, List, Dict

def _update_data(data: Any, idx: int, jdx: int, value: Any) -> Any:
def _check_equals(data1: Any, data2: Any) -> bool:
def test_copy_mode_assign(data: List[Dict[str, str]]) -> None:
def test_copy_mode_copy(data: List[Dict[str, str]]) -> None:
def test_copy_mode_deepcopy(data: List[Dict[str, str]]) -> None:
def test_copy_mode_invalid_string() -> None:
def test_infer_mode_copy(input_data: Any) -> None:
def test_infer_mode_deepcopy(data: Any) -> None:
def test_infer_mode_assign() -> None:
def test_is_memory_dataset(ds_or_type: Any, expected_result: bool) -> None:
