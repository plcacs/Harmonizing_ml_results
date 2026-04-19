from typing import Any, Callable, Literal, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import pytest
from pandas import DataFrame, Series

def _frame() -> DataFrame: ...
def _frame2() -> DataFrame: ...
def _mixed(_frame: DataFrame) -> DataFrame: ...
def _mixed2(_frame2: DataFrame) -> DataFrame: ...
def _integer() -> DataFrame: ...
def _integer_integers(_integer: DataFrame) -> DataFrame: ...
def _integer2() -> DataFrame: ...
def _array(_frame: DataFrame) -> NDArray[np.float64]: ...
def _array2(_frame2: DataFrame) -> NDArray[np.float64]: ...
def _array_mixed(_mixed: DataFrame) -> NDArray[np.int32]: ...
def _array_mixed2(_mixed2: DataFrame) -> NDArray[np.int32]: ...

class TestExpressions:
    @staticmethod
    def call_op(
        df: Union[DataFrame, Series],
        other: Union[DataFrame, Series],
        flex: bool,
        opname: str,
    ) -> Tuple[Union[DataFrame, Series], Union[DataFrame, Series]]: ...
    def test_run_arithmetic(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        flex: bool,
        arith: Literal["add", "sub", "mul", "mod", "truediv", "floordiv"],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...
    def test_run_binary(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        flex: bool,
        comparison_op: Callable[[Any, Any], Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...
    def test_invalid(self) -> None: ...
    def test_binary_ops(
        self,
        request: pytest.FixtureRequest,
        opname: Literal["add", "sub", "mul", "truediv", "pow"],
        op_str: Literal["+", "-", "*", "/", "**"],
        left_fix: Literal["_array", "_array2", "_array_mixed", "_array_mixed2"],
        right_fix: Literal["_array", "_array2", "_array_mixed", "_array_mixed2"],
    ) -> None: ...
    def test_comparison_ops(
        self,
        request: pytest.FixtureRequest,
        comparison_op: Callable[[Any, Any], Any],
        left_fix: Literal["_array", "_array2", "_array_mixed", "_array_mixed2"],
        right_fix: Literal["_array", "_array2", "_array_mixed", "_array_mixed2"],
    ) -> None: ...
    def test_where(
        self,
        request: pytest.FixtureRequest,
        cond: bool,
        fixture: Literal["_frame", "_frame2", "_mixed", "_mixed2"],
    ) -> None: ...
    def test_bool_ops_raise_on_arithmetic(
        self,
        op_str: Literal["/", "//", "**"],
        opname: Literal["truediv", "floordiv", "pow"],
    ) -> None: ...
    def test_bool_ops_warn_on_arithmetic(
        self,
        op_str: Literal["+", "*", "-"],
        opname: Literal["add", "mul", "sub"],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...
    def test_bool_ops_column_name_dtype(self, test_input: DataFrame, expected: DataFrame) -> None: ...
    def test_frame_series_axis(
        self,
        axis: Literal[0, 1],
        arith: Literal["add", "sub", "mul", "mod", "truediv", "floordiv"],
        _frame: DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...
    def test_python_semantics_with_numexpr_installed(
        self,
        op: Literal["__mod__", "__rmod__", "__floordiv__", "__rfloordiv__"],
        box_with_array: Callable[[NDArray[np.int64]], Any],
        scalar: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...