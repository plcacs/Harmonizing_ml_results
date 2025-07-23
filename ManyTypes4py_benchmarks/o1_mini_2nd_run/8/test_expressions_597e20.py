import operator
import re
from typing import Any, Callable, Tuple, Union

import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import DataFrame, Series
from pandas.core.computation import expressions as expr


@pytest.fixture
def _frame() -> DataFrame:
    return DataFrame(
        np.random.default_rng(2).standard_normal((10001, 4)),
        columns=list('ABCD'),
        dtype='float64'
    )


@pytest.fixture
def _frame2() -> DataFrame:
    return DataFrame(
        np.random.default_rng(2).standard_normal((100, 4)),
        columns=list('ABCD'),
        dtype='float64'
    )


@pytest.fixture
def _mixed(_frame: DataFrame) -> DataFrame:
    return DataFrame({
        'A': _frame['A'],
        'B': _frame['B'].astype('float32'),
        'C': _frame['C'].astype('int64'),
        'D': _frame['D'].astype('int32')
    })


@pytest.fixture
def _mixed2(_frame2: DataFrame) -> DataFrame:
    return DataFrame({
        'A': _frame2['A'],
        'B': _frame2['B'].astype('float32'),
        'C': _frame2['C'].astype('int64'),
        'D': _frame2['D'].astype('int32')
    })


@pytest.fixture
def _integer() -> DataFrame:
    return DataFrame(
        np.random.default_rng(2).integers(1, 100, size=(10001, 4)),
        columns=list('ABCD'),
        dtype='int64'
    )


@pytest.fixture
def _integer_integers(_integer: DataFrame) -> DataFrame:
    return _integer * np.random.default_rng(2).integers(0, 2, size=_integer.shape)


@pytest.fixture
def _integer2() -> DataFrame:
    return DataFrame(
        np.random.default_rng(2).integers(1, 100, size=(101, 4)),
        columns=list('ABCD'),
        dtype='int64'
    )


@pytest.fixture
def _array(_frame: DataFrame) -> np.ndarray:
    return _frame['A'].to_numpy()


@pytest.fixture
def _array2(_frame2: DataFrame) -> np.ndarray:
    return _frame2['A'].to_numpy()


@pytest.fixture
def _array_mixed(_mixed: DataFrame) -> np.ndarray:
    return _mixed['D'].to_numpy()


@pytest.fixture
def _array_mixed2(_mixed2: DataFrame) -> np.ndarray:
    return _mixed2['D'].to_numpy()


@pytest.mark.skipif(not expr.USE_NUMEXPR, reason='not using numexpr')
class TestExpressions:

    @staticmethod
    def call_op(
        df: Union[DataFrame, Series, np.ndarray],
        other: Union[DataFrame, Series, np.ndarray],
        flex: bool,
        opname: str
    ) -> Tuple[Union[DataFrame, Series, np.ndarray], Union[DataFrame, Series, np.ndarray]]:
        if flex:
            op: Callable[[Any, Any], Any] = lambda x, y: getattr(x, opname)(y)  # type: ignore
            op.__name__ = opname
        else:
            op = getattr(operator, opname)  # type: Callable[[Any, Any], Any]
        with option_context('compute.use_numexpr', False):
            expected = op(df, other)
        expr.get_test_result()
        result = op(df, other)
        return (result, expected)

    @pytest.mark.parametrize(
        'fixture',
        ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2']
    )
    @pytest.mark.parametrize('flex', [True, False])
    @pytest.mark.parametrize('arith', ['add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'])
    def test_run_arithmetic(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        flex: bool,
        arith: str,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df: Union[DataFrame, Series, np.ndarray] = request.getfixturevalue(fixture)
        with monkeypatch.context() as m:
            m.setattr(expr, '_MIN_ELEMENTS', 0)
            result, expected = self.call_op(df, df, flex, arith)
            if arith == 'truediv':
                assert all((x.kind == 'f' for x in expected.dtypes.values))  # type: ignore
            tm.assert_equal(expected, result)
            for i in range(len(df.columns)):
                left = df.iloc[:, i]
                right = df.iloc[:, i]
                result, expected = self.call_op(left, right, flex, arith)
                if arith == 'truediv':
                    assert expected.dtype.kind == 'f'  # type: ignore
                tm.assert_equal(expected, result)

    @pytest.mark.parametrize(
        'fixture',
        ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2']
    )
    @pytest.mark.parametrize('flex', [True, False])
    def test_run_binary(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        flex: bool,
        comparison_op: Callable[[Any, Any], Any],
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        tests solely that the result is the same whether or not numexpr is
        enabled.  Need to test whether the function does the correct thing
        elsewhere.
        """
        df: Union[DataFrame, Series, np.ndarray] = request.getfixturevalue(fixture)
        arith: str = comparison_op.__name__
        with option_context('compute.use_numexpr', False):
            other: Union[DataFrame, Series, np.ndarray] = df + 1  # type: ignore
        with monkeypatch.context() as m:
            m.setattr(expr, '_MIN_ELEMENTS', 0)
            expr.set_test_mode(True)
            result, expected = self.call_op(df, other, flex, arith)
            used_numexpr: bool = expr.get_test_result()
            assert used_numexpr, 'Did not use numexpr as expected.'
            tm.assert_equal(expected, result)
            for i in range(len(df.columns)):
                binary_comp = other.iloc[:, i] + 1  # type: ignore
                self.call_op(df.iloc[:, i], binary_comp, flex, 'add')

    def test_invalid(self) -> None:
        array: np.ndarray = np.random.default_rng(2).standard_normal(1000001)
        array2: np.ndarray = np.random.default_rng(2).standard_normal(100)
        result: bool = expr._can_use_numexpr(operator.add, None, array, array, 'evaluate')
        assert not result
        result = expr._can_use_numexpr(operator.add, '+', array2, array2, 'evaluate')
        assert not result
        result = expr._can_use_numexpr(operator.add, '+', array, array2, 'evaluate')
        assert result

    @pytest.mark.filterwarnings('ignore:invalid value encountered in:RuntimeWarning')
    @pytest.mark.parametrize(
        'opname,op_str',
        [('add', '+'), ('sub', '-'), ('mul', '*'), ('truediv', '/'), ('pow', '**')]
    )
    @pytest.mark.parametrize(
        'left_fix,right_fix',
        [('_array', '_array2'), ('_array_mixed', '_array_mixed2')]
    )
    def test_binary_ops(
        self,
        request: pytest.FixtureRequest,
        opname: str,
        op_str: str,
        left_fix: str,
        right_fix: str
    ) -> None:
        left: np.ndarray = request.getfixturevalue(left_fix)
        right: np.ndarray = request.getfixturevalue(right_fix)

        def testit(
            left_arr: np.ndarray,
            right_arr: np.ndarray,
            opname_inner: str,
            op_str_inner: str
        ) -> None:
            if opname_inner == 'pow':
                left_arr = np.abs(left_arr)
            op_func: Callable[[Any, Any], Any] = getattr(operator, opname_inner)  # type: ignore
            result = expr.evaluate(op_func, left_arr, left_arr, use_numexpr=True)
            expected = expr.evaluate(op_func, left_arr, left_arr, use_numexpr=False)
            tm.assert_numpy_array_equal(result, expected)
            can_use = expr._can_use_numexpr(op_func, op_str_inner, right_arr, right_arr, 'evaluate')
            assert not can_use

        with option_context('compute.use_numexpr', False):
            testit(left, right, opname, op_str)
        expr.set_numexpr_threads(1)
        testit(left, right, opname, op_str)
        expr.set_numexpr_threads()
        testit(left, right, opname, op_str)

    @pytest.mark.parametrize(
        'left_fix,right_fix',
        [('_array', '_array2'), ('_array_mixed', '_array_mixed2')]
    )
    def test_comparison_ops(
        self,
        request: pytest.FixtureRequest,
        comparison_op: Callable[[Any, Any], Any],
        left_fix: str,
        right_fix: str
    ) -> None:
        left: np.ndarray = request.getfixturevalue(left_fix)
        right: np.ndarray = request.getfixturevalue(right_fix)

        def testit() -> None:
            f12 = left + 1  # type: np.ndarray
            f22 = right + 1  # type: np.ndarray
            op = comparison_op
            result = expr.evaluate(op, left, f12, use_numexpr=True)
            expected = expr.evaluate(op, left, f12, use_numexpr=False)
            tm.assert_numpy_array_equal(result, expected)
            can_use = expr._can_use_numexpr(op, op, right, f22, 'evaluate')
            assert not can_use

        with option_context('compute.use_numexpr', False):
            testit()
        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize(
        'cond',
        [True, False]
    )
    @pytest.mark.parametrize(
        'fixture',
        ['_frame', '_frame2', '_mixed', '_mixed2']
    )
    def test_where(
        self,
        request: pytest.FixtureRequest,
        cond: bool,
        fixture: str
    ) -> None:
        df: DataFrame = request.getfixturevalue(fixture)

        def testit() -> None:
            c: np.ndarray = np.empty(df.shape, dtype=np.bool_)
            c.fill(cond)
            result = expr.where(c, df.values, df.values + 1)
            expected = np.where(c, df.values, df.values + 1)
            tm.assert_numpy_array_equal(result, expected)

        with option_context('compute.use_numexpr', False):
            testit()
        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize(
        'op_str,opname',
        [('/', 'truediv'), ('//', 'floordiv'), ('**', 'pow')]
    )
    def test_bool_ops_raise_on_arithmetic(
        self,
        op_str: str,
        opname: str
    ) -> None:
        df: DataFrame = DataFrame({
            'a': np.random.default_rng(2).random(10) > 0.5,
            'b': np.random.default_rng(2).random(10) > 0.5
        })
        msg: str = f"operator '{opname}' not implemented for bool dtypes"
        f: Callable[[Any, Any], Any] = getattr(operator, opname)
        err_msg: str = re.escape(msg)
        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, df)
        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, df.b)
        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, True)
        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df.a)
        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df)
        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, True)

    @pytest.mark.parametrize(
        'op_str,opname',
        [('+', 'add'), ('*', 'mul'), ('-', 'sub')]
    )
    def test_bool_ops_warn_on_arithmetic(
        self,
        op_str: str,
        opname: str,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n: int = 10
        df: DataFrame = DataFrame({
            'a': np.random.default_rng(2).random(n) > 0.5,
            'b': np.random.default_rng(2).random(n) > 0.5
        })
        subs: dict[str, str] = {'+': '|', '*': '&', '-': '^'}
        sub_funcs: dict[str, str] = {'|': 'or_', '&': 'and_', '^': 'xor'}
        f: Callable[[Any, Any], Any] = getattr(operator, opname)
        fe: Callable[[Any, Any], Any] = getattr(operator, sub_funcs[subs[op_str]])
        if op_str == '-':
            return
        msg: str = 'operator is not supported by numexpr'
        with monkeypatch.context() as m:
            m.setattr(expr, '_MIN_ELEMENTS', 5)
            with option_context('compute.use_numexpr', True):
                with tm.assert_produces_warning(UserWarning, match=msg):
                    r = f(df, df)
                    e = fe(df, df)
                    tm.assert_frame_equal(r, e)
                with tm.assert_produces_warning(UserWarning, match=msg):
                    r = f(df.a, df.b)
                    e = fe(df.a, df.b)
                    tm.assert_series_equal(r, e)
                with tm.assert_produces_warning(UserWarning, match=msg):
                    r = f(df.a, True)
                    e = fe(df.a, True)
                    tm.assert_series_equal(r, e)
                with tm.assert_produces_warning(UserWarning, match=msg):
                    r = f(False, df.a)
                    e = fe(False, df.a)
                    tm.assert_series_equal(r, e)
                with tm.assert_produces_warning(UserWarning, match=msg):
                    r = f(False, df)
                    e = fe(False, df)
                    tm.assert_frame_equal(r, e)
                with tm.assert_produces_warning(UserWarning, match=msg):
                    r = f(df, True)
                    e = fe(df, True)
                    tm.assert_frame_equal(r, e)

    @pytest.mark.parametrize(
        'test_input,expected',
        [
            (
                DataFrame(
                    [[0, 1, 2, 'aa'], [0, 1, 2, 'aa']],
                    columns=['a', 'b', 'c', 'dtype']
                ),
                DataFrame(
                    [[False, False], [False, False]],
                    columns=['a', 'dtype']
                )
            ),
            (
                DataFrame(
                    [[0, 3, 2, 'aa'], [0, 4, 2, 'aa'], [0, 1, 1, 'bb']],
                    columns=['a', 'b', 'c', 'dtype']
                ),
                DataFrame(
                    [[False, False], [False, False], [False, False]],
                    columns=['a', 'dtype']
                )
            )
        ]
    )
    def test_bool_ops_column_name_dtype(
        self,
        test_input: DataFrame,
        expected: DataFrame
    ) -> None:
        result: DataFrame = test_input.loc[:, ['a', 'dtype']].ne(test_input.loc[:, ['a', 'dtype']])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'arith',
        ('add', 'sub', 'mul', 'mod', 'truediv', 'floordiv')
    )
    @pytest.mark.parametrize(
        'axis',
        (0, 1)
    )
    def test_frame_series_axis(
        self,
        axis: int,
        arith: str,
        _frame: DataFrame,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df: DataFrame = _frame
        if axis == 1:
            other: Series = df.iloc[0, :]
        else:
            other = df.iloc[:, 0]
        with monkeypatch.context() as m:
            m.setattr(expr, '_MIN_ELEMENTS', 0)
            op_func: Callable[[Any, Any], Any] = getattr(df, arith)  # type: ignore
            with option_context('compute.use_numexpr', False):
                expected = op_func(other, axis=axis)  # type: ignore
            result = op_func(other, axis=axis)  # type: ignore
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        'op',
        ['__mod__', '__rmod__', '__floordiv__', '__rfloordiv__']
    )
    @pytest.mark.parametrize(
        'scalar',
        [-5, 5]
    )
    def test_python_semantics_with_numexpr_installed(
        self,
        op: str,
        box_with_array: Callable[[np.ndarray], Union[DataFrame, Series]],
        scalar: int,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with monkeypatch.context() as m:
            m.setattr(expr, '_MIN_ELEMENTS', 0)
            data: np.ndarray = np.arange(-50, 50)
            obj: Union[DataFrame, Series] = box_with_array(data)
            method: Callable[[Any], Any] = getattr(obj, op)
            result = method(scalar)
            with option_context('compute.use_numexpr', False):
                expected = method(scalar)
            tm.assert_equal(result, expected)
            for i, elem in enumerate(data):
                if isinstance(obj, DataFrame):
                    scalar_result = result.iloc[i, 0]
                else:
                    scalar_result = result[i]
                try:
                    expected_val = getattr(int(elem), op)(scalar)
                except ZeroDivisionError:
                    pass
                else:
                    assert scalar_result == expected_val
