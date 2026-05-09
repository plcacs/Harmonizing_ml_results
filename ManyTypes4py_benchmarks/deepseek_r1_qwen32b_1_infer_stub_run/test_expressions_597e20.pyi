from __future__ import annotations
from typing import Any, Union, Tuple, Callable, List, Optional
import numpy as np
from pandas import DataFrame
import pytest

@pytest.fixture
def _frame() -> DataFrame:
    ...

@pytest.fixture
def _frame2() -> DataFrame:
    ...

@pytest.fixture
def _mixed(_frame: DataFrame) -> DataFrame:
    ...

@pytest.fixture
def _mixed2(_frame2: DataFrame) -> DataFrame:
    ...

@pytest.fixture
def _integer() -> DataFrame:
    ...

@pytest.fixture
def _integer_integers(_integer: DataFrame) -> DataFrame:
    ...

@pytest.fixture
def _integer2() -> DataFrame:
    ...

@pytest.fixture
def _array(_frame: DataFrame) -> np.ndarray:
    ...

@pytest.fixture
def _array2(_frame2: DataFrame) -> np.ndarray:
    ...

@pytest.fixture
def _array_mixed(_mixed: DataFrame) -> np.ndarray:
    ...

@pytest.fixture
def _array_mixed2(_mixed2: DataFrame) -> np.ndarray:
    ...

class TestExpressions:
    @staticmethod
    def call_op(df: DataFrame, other: Any, flex: bool, opname: str) -> Tuple[Union[DataFrame, np.ndarray], Union[DataFrame, np.ndarray]]:
        ...

    @pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
    @pytest.mark.parametrize('flex', [True, False])
    @pytest.mark.parametrize('arith', ['add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'])
    def test_run_arithmetic(self, request: pytest.FixtureRequest, fixture: str, flex: bool, arith: str, monkeypatch: pytest.MonkeyPatch) -> None:
        ...

    @pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
    @pytest.mark.parametrize('flex', [True, False])
    def test_run_binary(self, request: pytest.FixtureRequest, fixture: str, flex: bool, comparison_op: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
        ...

    def test_invalid(self) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:invalid value encountered in:RuntimeWarning')
    @pytest.mark.parametrize('opname,op_str', [('add', '+'), ('sub', '-'), ('mul', '*'), ('truediv', '/'), ('pow', '**')])
    @pytest.mark.parametrize('left_fix,right_fix', [('_array', '_array2'), ('_array_mixed', '_array_mixed2')])
    def test_binary_ops(self, request: pytest.FixtureRequest, opname: str, op_str: str, left_fix: str, right_fix: str) -> None:
        ...

    @pytest.mark.parametrize('left_fix,right_fix', [('_array', '_array2'), ('_array_mixed', '_array_mixed2')])
    def test_comparison_ops(self, request: pytest.FixtureRequest, comparison_op: Callable, left_fix: str, right_fix: str) -> None:
        ...

    @pytest.mark.parametrize('cond', [True, False])
    @pytest.mark.parametrize('fixture', ['_frame', '_frame2', '_mixed', '_mixed2'])
    def test_where(self, request: pytest.FixtureRequest, cond: bool, fixture: str) -> None:
        ...

    @pytest.mark.parametrize('op_str,opname', [('/', 'truediv'), ('//', 'floordiv'), ('**', 'pow')])
    def test_bool_ops_raise_on_arithmetic(self, op_str: str, opname: str) -> None:
        ...

    @pytest.mark.parametrize('op_str,opname', [('+', 'add'), ('*', 'mul'), ('-', 'sub')])
    def test_bool_ops_warn_on_arithmetic(self, op_str: str, opname: str, monkeypatch: pytest.MonkeyPatch) -> None:
        ...

    @pytest.mark.parametrize('test_input,expected', [(DataFrame([[0, 1, 2, 'aa'], [0, 1, 2, 'aa']], columns=['a', 'b', 'c', 'dtype']), DataFrame([[False, False], [False, False]], columns=['a', 'dtype'])), (DataFrame([[0, 3, 2, 'aa'], [0, 4, 2, 'aa'], [0, 1, 1, 'bb']], columns=['a', 'b', 'c', 'dtype']), DataFrame([[False, False], [False, False], [False, False]], columns=['a', 'dtype']))])
    def test_bool_ops_column_name_dtype(self, test_input: DataFrame, expected: DataFrame) -> None:
        ...

    @pytest.mark.parametrize('arith', ('add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'))
    @pytest.mark.parametrize('axis', (0, 1))
    def test_frame_series_axis(self, axis: int, arith: str, _frame: DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        ...

    @pytest.mark.parametrize('op', ['__mod__', '__rmod__', '__floordiv__', '__rfloordiv__'])
    @pytest.mark.parametrize('scalar', [-5, 5])
    def test_python_semantics_with_numexpr_installed(self, op: str, box_with_array: Union[DataFrame, np.ndarray], scalar: int, monkeypatch: pytest.MonkeyPatch) -> None:
        ...