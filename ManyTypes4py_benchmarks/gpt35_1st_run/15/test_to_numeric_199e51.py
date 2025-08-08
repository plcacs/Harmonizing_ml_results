from typing import Any

def test_really_large_scalar(large_val: int, signed: bool, transform: Any, errors: Any) -> None:
def test_really_large_in_arr(large_val: int, signed: bool, transform: Any, multiple_elts: bool, errors: Any) -> None:
def test_really_large_in_arr_consistent(large_val: int, signed: bool, multiple_elts: bool, errors: Any) -> None:
def test_downcast_limits(dtype: str, downcast: str, min_max: list) -> None:
def test_downcast_nullable_numeric(data: list, input_dtype: str, downcast: str, expected_dtype: str) -> None:
def test_to_numeric_dtype_backend(val: Any, dtype: str) -> None:
def test_to_numeric_dtype_backend_na(val: Any, dtype: str) -> None:
def test_to_numeric_dtype_backend_downcasting(val: Any, dtype: str, downcast: str) -> None:
def test_to_numeric_dtype_backend_downcasting_uint(smaller: str, dtype_backend: str) -> None:
def test_to_numeric_dtype_backend_already_nullable(dtype: str) -> None:
def test_to_numeric_dtype_backend_error(dtype_backend: str) -> None:
