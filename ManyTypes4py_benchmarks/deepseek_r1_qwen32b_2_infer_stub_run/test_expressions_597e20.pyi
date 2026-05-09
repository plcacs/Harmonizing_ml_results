import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import DataFrame
from pandas.core.computation import expressions as expr
from pytest import MonkeyPatch, SubRequest

_frame: DataFrame[float64] = ...
_frame2: DataFrame[float64] = ...
_mixed: DataFrame[Union[float64, float32, int64, int32]] = ...
_mixed2: DataFrame[Union[float64, float32, int64, int32]] = ...
_integer: DataFrame[int64] = ...
_integer_integers: DataFrame[int64] = ...
_integer2: DataFrame[int64] = ...
_array: np.ndarray[float64] = ...
_array2: np.ndarray[float64] = ...
_array_mixed: np.ndarray[int32] = ...
_array_mixed2: np.ndarray[int32] = ...

class TestExpressions:
    @staticmethod
    def call_op(df: DataFrame, other: Union[DataFrame, np.ndarray, int, float], flex: bool, opname: str) -> tuple[Union[DataFrame, np.ndarray], Union[DataFrame, np.ndarray]]:
        ...

    @pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
    @pytest.mark.parametrize('flex', [True, False])
    @pytest.mark.parametrize('arith', ['add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'])
    def test_run_arithmetic(self, request: SubRequest, fixture: str, flex: bool, arith: str, monkeypatch: MonkeyPatch) -> None:
        ...

    @pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
    @pytest.mark.parametrize('flex', [True, False])
    def test_run_binary(self, request: SubRequest, fixture: str, flex: bool, comparison_op: callable, monkeypatch: MonkeyPatch) -> None:
        ...

    def test_invalid(self) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:invalid value encountered in:RuntimeWarning')
    @pytest.mark.parametrize('opname,op_str', [('add', '+'), ('sub', '-'), ('mul', '*'), ('truediv', '/'), ('pow', '**')])
    @pytest.mark.parametrize('left_fix,right_fix', [('_array', '_array2'), ('_array_mixed', '_array_mixed2')])
    def test_binary_ops(self, request: SubRequest, opname: str, op_str: str, left_fix: str, right_fix: str) -> None:
        ...

    @pytest.mark.parametrize('left_fix,right_fix', [('_array', '_array2'), ('_array_mixed', '_array_mixed2')])
    def test_comparison_ops(self, request: SubRequest, comparison_op: callable, left_fix: str, right_fix: str) -> None:
        ...

    @pytest.mark.parametrize('cond', [True, False])
    @pytest.mark.parametrize('fixture', ['_frame', '_frame2', '_mixed', '_mixed2'])
    def test_where(self, request: SubRequest, cond: bool, fixture: str) -> None:
        ...

    @pytest.mark.parametrize('op_str,opname', [('/', 'truediv'), ('//', 'floordiv'), ('**', 'pow')])
    def test_bool_ops_raise_on_arithmetic(self, op_str: str, opname: str) -> None:
        ...

    @pytest.mark.parametrize('op_str,opname', [('+', 'add'), ('*', 'mul'), ('-', 'sub')])
    def test_bool_ops_warn_on_arithmetic(self, op_str: str, opname: str, monkeypatch: MonkeyPatch) -> None:
        ...

    @pytest.mark.parametrize('test_input,expected', [(DataFrame([[0, 1, 2, 'aa'], [0, 1, 2, 'aa']], columns=['a', 'b', 'c', 'dtype']), DataFrame([[False, False], [False, False]], columns=['a', 'dtype'])), (DataFrame([[0, 3, 2, 'aa'], [0, 4, 2, 'aa'], [0, 1, 1, 'bb']], columns=['a', 'b', 'c', 'dtype']), DataFrame([[False, False], [False, False], [False, False]], columns=['a', 'dtype']))])
    def test_bool_ops_column_name_dtype(self, test_input: DataFrame, expected: DataFrame) -> None:
        ...

    @pytest.mark.parametrize('arith', ('add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'))
    @pytest.mark.parametrize('axis', (0, 1))
    def test_frame_series_axis(self, axis: int, arith: str, _frame: DataFrame, monkeypatch: MonkeyPatch) -> None:
        ...

    @pytest.mark.parametrize('op', ['__mod__', '__rmod__', '__floordiv__', '__rfloordiv__'])
    @pytest.mark.parametrize('scalar', [-5, 5])
    def test_python_semantics_with_numexpr_installed(self, op: str, box_with_array: callable, scalar: int, monkeypatch: MonkeyPatch) -> None:
        ...