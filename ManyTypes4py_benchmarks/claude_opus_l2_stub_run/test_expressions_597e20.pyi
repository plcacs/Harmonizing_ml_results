import numpy as np
import pytest
from pandas.core.api import DataFrame


@pytest.fixture
def _frame() -> DataFrame: ...

@pytest.fixture
def _frame2() -> DataFrame: ...

@pytest.fixture
def _mixed(_frame: DataFrame) -> DataFrame: ...

@pytest.fixture
def _mixed2(_frame2: DataFrame) -> DataFrame: ...

@pytest.fixture
def _integer() -> DataFrame: ...

@pytest.fixture
def _integer_integers(_integer: DataFrame) -> DataFrame: ...

@pytest.fixture
def _integer2() -> DataFrame: ...

@pytest.fixture
def _array(_frame: DataFrame) -> np.ndarray: ...

@pytest.fixture
def _array2(_frame2: DataFrame) -> np.ndarray: ...

@pytest.fixture
def _array_mixed(_mixed: DataFrame) -> np.ndarray: ...

@pytest.fixture
def _array_mixed2(_mixed2: DataFrame) -> np.ndarray: ...

class TestExpressions:
    @staticmethod
    def call_op(
        df: DataFrame | object,
        other: DataFrame | object,
        flex: bool,
        opname: str,
    ) -> tuple[object, object]: ...

    def test_run_arithmetic(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        flex: bool,
        arith: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...

    def test_run_binary(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        flex: bool,
        comparison_op: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...

    def test_invalid(self) -> None: ...

    def test_binary_ops(
        self,
        request: pytest.FixtureRequest,
        opname: str,
        op_str: str,
        left_fix: str,
        right_fix: str,
    ) -> None: ...

    def test_comparison_ops(
        self,
        request: pytest.FixtureRequest,
        comparison_op: object,
        left_fix: str,
        right_fix: str,
    ) -> None: ...

    def test_where(
        self,
        request: pytest.FixtureRequest,
        cond: bool,
        fixture: str,
    ) -> None: ...

    def test_bool_ops_raise_on_arithmetic(
        self, op_str: str, opname: str
    ) -> None: ...

    def test_bool_ops_warn_on_arithmetic(
        self, op_str: str, opname: str, monkeypatch: pytest.MonkeyPatch
    ) -> None: ...

    def test_bool_ops_column_name_dtype(
        self, test_input: DataFrame, expected: DataFrame
    ) -> None: ...

    def test_frame_series_axis(
        self,
        axis: int,
        arith: str,
        _frame: DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...

    def test_python_semantics_with_numexpr_installed(
        self,
        op: str,
        box_with_array: type,
        scalar: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None: ...