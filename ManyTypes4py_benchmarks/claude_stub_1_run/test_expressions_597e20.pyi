```pyi
from typing import Any, Callable, Literal
import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import DataFrame
from pandas.core.computation import expressions as expr

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
def _array(_frame: DataFrame) -> np.ndarray[Any, np.dtype[np.floating[Any]]]: ...

@pytest.fixture
def _array2(_frame2: DataFrame) -> np.ndarray[Any, np.dtype[np.floating[Any]]]: ...

@pytest.fixture
def _array_mixed(_mixed: DataFrame) -> np.ndarray[Any, np.dtype[Any]]: ...

@pytest.fixture
def _array_mixed2(_mixed2: DataFrame) -> np.ndarray[Any, np.dtype[Any]]: ...

class TestExpressions:
    @staticmethod
    def call_op(df: Any, other: Any, flex: bool, opname: str) -> tuple[Any, Any]: ...
    @pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
    @pytest.mark.parametrize('flex', [True, False])
    @pytest.mark.parametrize('arith', ['add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'])
    def test_run_arithmetic(self, request: Any, fixture: str, flex: bool, arith: str, monkeypatch: Any) -> None: ...
    @pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
    @pytest.mark.parametrize('flex', [True, False])
    def test_run_binary(self, request: Any, fixture: str, flex: bool, comparison_op: Any, monkeypatch: Any) -> None: ...
    def test_invalid(self) -> None: ...
    @pytest.mark.filterwarnings('ignore:invalid value encountered in:RuntimeWarning')
    @pytest.mark.parametrize('opname,op_str', [('add', '+'), ('sub', '-'), ('mul', '*'), ('truediv', '/'), ('pow', '**')])
    @pytest.mark.parametrize('left_fix,right_fix', [('_array', '_array2'), ('_array_mixed', '_array_mixed2')])
    def test_binary_ops(self, request: Any, opname: str, op_str: str, left_fix: str, right_fix: str) -> None: ...
    @pytest.mark.parametrize('left_fix,right_fix', [('_array', '_array2'), ('_array_mixed', '_array_mixed2')])
    def test_comparison_ops(self, request: Any, comparison_op: Any, left_fix: str, right_fix: str) -> None: ...
    @pytest.mark.parametrize('cond', [True, False])
    @pytest.mark.parametrize('fixture', ['_frame', '_frame2', '_mixed', '_mixed2'])
    def test_where(self, request: Any, cond: bool, fixture: str) -> None: ...
    @pytest.mark.parametrize('op_str,opname', [('/', 'truediv'), ('//', 'floordiv'), ('**', 'pow')])
    def test_bool_ops_raise_on_arithmetic(self, op_str: str, opname: str) -> None: ...
    @pytest.mark.parametrize('op_str,opname', [('+', 'add'), ('*', 'mul'), ('-', 'sub')])
    def test_bool_ops_warn_on_arithmetic(self, op_str: str, opname: str, monkeypatch: Any) -> None: ...
    @pytest.mark.parametrize('test_input,expected', [(DataFrame([[0, 1, 2, 'aa'], [0, 1, 2, 'aa']], columns=['a', 'b', 'c', 'dtype']), DataFrame([[False, False], [False, False]], columns=['a', 'dtype'])), (DataFrame([[0, 3, 2, 'aa'], [0, 4, 2, 'aa'], [0, 1, 1, 'bb']], columns=['a', 'b', 'c', 'dtype']), DataFrame([[False, False], [False, False], [False, False]], columns=['a', 'dtype']))])
    def test_bool_ops_column_name_dtype(self, test_input: DataFrame, expected: DataFrame) -> None: ...
    @pytest.mark.parametrize('arith', ('add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'))
    @pytest.mark.parametrize('axis', (0, 1))
    def test_frame_series_axis(self, axis: int, arith: str, _frame: DataFrame, monkeypatch: Any) -> None: ...
    @pytest.mark.parametrize('op', ['__mod__', '__rmod__', '__floordiv__', '__rfloordiv__'])
    @pytest.mark.parametrize('scalar', [-5, 5])
    def test_python_semantics_with_numexpr_installed(self, op: str, box_with_array: Any, scalar: int, monkeypatch: Any) -> None: ...
```