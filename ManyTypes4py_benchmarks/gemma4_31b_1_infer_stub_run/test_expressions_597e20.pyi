import operator
import numpy as np
import pytest
from pandas.core.api import DataFrame
from pandas.core.series import Series
from typing import Any, Tuple, Union, Callable, overload

def _frame() -> DataFrame: ...
def _frame2() -> DataFrame: ...
def _mixed(_frame: DataFrame) -> DataFrame: ...
def _mixed2(_frame2: DataFrame) -> DataFrame: ...
def _integer() -> DataFrame: ...
def _integer_integers(_integer: DataFrame) -> DataFrame: ...
def _integer2() -> DataFrame: ...
def _array(_frame: DataFrame) -> np.ndarray: ...
def _array2(_frame2: DataFrame) -> np.ndarray: ...
def _array_mixed(_mixed: DataFrame) -> np.ndarray: ...
def _array_mixed2(_mixed2: DataFrame) -> np.ndarray: ...

class TestExpressions:
    @staticmethod
    def call_op(
        df: Union[DataFrame, Series], 
        other: Union[DataFrame, Series, Any], 
        flex: bool, 
        opname: str
    ) -> Tuple[Union[DataFrame, Series, np.ndarray], Union[DataFrame, Series, np.ndarray]]: ...

    def test_run_arithmetic(
        self, 
        request: Any, 
        fixture: str, 
        flex: bool, 
        arith: str, 
        monkeypatch: Any
    ) -> None: ...

    def test_run_binary(
        self, 
        request: Any, 
        fixture: str, 
        flex: bool, 
        comparison_op: Callable, 
        monkeypatch: Any
    ) -> None: ...

    def test_invalid(self) -> None: ...

    def test_binary_ops(
        self, 
        request: Any, 
        opname: str, 
        op_str: str, 
        left_fix: str, 
        right_fix: str
    ) -> None: ...

    def test_comparison_ops(
        self, 
        request: Any, 
        comparison_op: Callable, 
        left_fix: str, 
        right_fix: str
    ) -> None: ...

    def test_where(
        self, 
        request: Any, 
        cond: bool, 
        fixture: str
    ) -> None: ...

    def test_bool_ops_raise_on_arithmetic(self, op_str: str, opname: str) -> None: ...

    def test_bool_ops_warn_on_arithmetic(
        self, 
        op_str: str, 
        opname: str, 
        monkeypatch: Any
    ) -> None: ...

    def test_bool_ops_column_name_dtype(self, test_input: DataFrame, expected: DataFrame) -> None: ...

    def test_frame_series_axis(
        self, 
        axis: int, 
        arith: str, 
        _frame: DataFrame, 
        monkeypatch: Any
    ) -> None: ...

    def test_python_semantics_with_numexpr_installed(
        self, 
        op: str, 
        box_with_array: Callable[[np.ndarray], Union[DataFrame, Series]], 
        scalar: int, 
        monkeypatch: Any
    ) -> None: ...