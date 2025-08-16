from typing import List, Any, Callable

class TestDataFrameLogicalOperators:

    def test_logical_operators_nans(self, left: List[Any], right: List[Any], op: Callable, expected: List[Any], frame_or_series: Any) -> None:

    def test_logical_ops_empty_frame(self) -> None:

    def test_logical_ops_bool_frame(self) -> None:

    def test_logical_ops_int_frame(self) -> None:

    def test_logical_ops_invalid(self, using_infer_string: bool) -> None:

    def test_logical_operators(self) -> None:

    def test_logical_with_nas(self) -> None:

    def test_logical_ops_categorical_columns(self) -> None:

    def test_int_dtype_different_index_not_bool(self) -> None:

    def test_different_dtypes_different_index_raises(self) -> None:
