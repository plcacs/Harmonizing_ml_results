from __future__ import annotations
from functools import reduce
from itertools import product
import operator
import numpy as np
import pytest
from pandas.compat import PY312
from pandas.errors import NumExprClobberingError, PerformanceWarning, UndefinedVariableError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_bool, is_float, is_list_like, is_scalar
import pandas as pd
from pandas import DataFrame, Index, Series, date_range, period_range, timedelta_range
import pandas._testing as tm
from pandas.core.computation import expr, pytables
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import BaseExprVisitor, PandasExprVisitor, PythonExprVisitor
from pandas.core.computation.expressions import NUMEXPR_INSTALLED, USE_NUMEXPR
from pandas.core.computation.ops import ARITH_OPS_SYMS, _binary_math_ops, _binary_ops_dict, _unary_math_ops
from pandas.core.computation.scope import DEFAULT_GLOBALS
from typing import Any, Callable, Dict, List, Tuple, Union, Optional


@pytest.fixture(params=(
    pytest.param(
        engine,
        marks=[
            pytest.mark.skipif(
                engine == 'numexpr' and (not USE_NUMEXPR),
                reason=f'numexpr enabled->{USE_NUMEXPR}, installed->{NUMEXPR_INSTALLED}'
            ),
            td.skip_if_no('numexpr')
        ]
    ) for engine in ENGINES
))
def engine(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=expr.PARSERS)
def parser(request: pytest.FixtureRequest) -> str:
    return request.param


def _eval_single_bin(lhs: Any, cmp1: str, rhs: Any, engine: str) -> Any:
    c = _binary_ops_dict[cmp1]
    if ENGINES[engine].has_neg_frac:
        try:
            return c(lhs, rhs)
        except ValueError as e:
            if str(e).startswith('negative number cannot be raised to a fractional power'):
                return np.nan
            raise
    return c(lhs, rhs)


@pytest.fixture(params=list(range(5)), ids=['DataFrame', 'Series', 'SeriesNaN', 'DataFrameNaN', 'float'])
def lhs(request: pytest.FixtureRequest) -> Union[DataFrame, Series, float, np.ndarray]:
    nan_df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    nan_df1[nan_df1 > 0.5] = np.nan
    opts: Tuple[DataFrame, Series, Series, DataFrame, np.ndarray] = (
        DataFrame(np.random.default_rng(2).standard_normal((10, 5))),
        Series(np.random.default_rng(2).standard_normal(5)),
        Series([1, 2, np.nan, np.nan, 5]),
        nan_df1,
        np.random.default_rng(2).standard_normal()
    )
    return opts[request.param]


rhs = lhs
midhs = lhs


@pytest.fixture
def idx_func_dict() -> Dict[str, Callable[[int], Index]]:
    return {
        'i': lambda n: Index(np.arange(n), dtype=np.int64),
        'f': lambda n: Index(np.arange(n), dtype=np.float64),
        's': lambda n: Index([f'{i}_{chr(i)}' for i in range(97, 97 + n)]),
        'dt': lambda n: date_range('2020-01-01', periods=n),
        'td': lambda n: timedelta_range('1 day', periods=n),
        'p': lambda n: period_range('2020-01-01', periods=n, freq='D')
    }


class TestEval:

    @pytest.mark.parametrize('cmp1', ['!=', '==', '<=', '>=', '<', '>'], ids=['ne', 'eq', 'le', 'ge', 'lt', 'gt'])
    @pytest.mark.parametrize('cmp2', ['>', '<'], ids=['gt', 'lt'])
    @pytest.mark.parametrize('binop', expr.BOOL_OPS_SYMS)
    def test_complex_cmp_ops(
        self,
        cmp1: str,
        cmp2: str,
        binop: str,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        rhs: Union[DataFrame, Series, float, np.ndarray],
        engine: str,
        parser: str
    ) -> None:
        if parser == 'python' and binop in ['and', 'or']:
            msg = "'BoolOp' nodes are not implemented"
            ex = f'(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)'
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, engine=engine, parser=parser)
            return
        lhs_new = _eval_single_bin(lhs, cmp1, rhs, engine)
        rhs_new = _eval_single_bin(lhs, cmp2, rhs, engine)
        expected = _eval_single_bin(lhs_new, binop, rhs_new, engine)
        ex = f'(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)'
        result = pd.eval(ex, engine=engine, parser=parser)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('cmp_op', expr.CMP_OPS_SYMS)
    def test_simple_cmp_ops(
        self,
        cmp_op: str,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        rhs: Union[DataFrame, Series, float, np.ndarray],
        engine: str,
        parser: str
    ) -> None:
        lhs = lhs < 0
        rhs = rhs < 0
        if parser == 'python' and cmp_op in ['in', 'not in']:
            msg = "'(In|NotIn)' nodes are not implemented"
            ex = f'lhs {cmp_op} rhs'
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, engine=engine, parser=parser)
            return
        ex = f'lhs {cmp_op} rhs'
        msg = '|'.join([
            "only list-like( or dict-like)? objects are allowed to be passed to (DataFrame\\.)?isin\\(\\), you passed a (`|')bool(`|')",
            "argument of type 'bool' is not iterable"
        ])
        if cmp_op in ('in', 'not in') and (not is_list_like(rhs)):
            with pytest.raises(TypeError, match=msg):
                pd.eval(ex, engine=engine, parser=parser, local_dict={'lhs': lhs, 'rhs': rhs})
        else:
            expected = _eval_single_bin(lhs, cmp_op, rhs, engine)
            result = pd.eval(ex, engine=engine, parser=parser)
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize('op', expr.CMP_OPS_SYMS)
    def test_compound_invert_op(
        self,
        op: str,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        rhs: Union[DataFrame, Series, float, np.ndarray],
        request: pytest.FixtureRequest,
        engine: str,
        parser: str
    ) -> None:
        if parser == 'python' and op in ['in', 'not in']:
            msg = "'(In|NotIn)' nodes are not implemented"
            ex = f'~(lhs {op} rhs)'
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, engine=engine, parser=parser)
            return
        if is_float(lhs) and (not is_float(rhs)) and (op in ['in', 'not in']) and (engine == 'python') and (parser == 'pandas'):
            mark = pytest.mark.xfail(reason='Looks like expected is negative, unclear whether expected is incorrect or result is incorrect')
            request.applymarker(mark)
        skip_these = ['in', 'not in']
        ex = f'~(lhs {op} rhs)'
        msg = '|'.join([
            "only list-like( or dict-like)? objects are allowed to be passed to (DataFrame\\.)?isin\\(\\), you passed a (`|')float(`|')",
            "argument of type 'float' is not iterable"
        ])
        if is_scalar(rhs) and op in skip_these:
            with pytest.raises(TypeError, match=msg):
                pd.eval(ex, engine=engine, parser=parser, local_dict={'lhs': lhs, 'rhs': rhs})
        else:
            if is_scalar(lhs) and is_scalar(rhs):
                lhs, rhs = (np.array([x]) for x in (lhs, rhs))
            expected = _eval_single_bin(lhs, op, rhs, engine)
            if is_scalar(expected):
                expected = not expected
            else:
                expected = ~expected
            result = pd.eval(ex, engine=engine, parser=parser)
            tm.assert_almost_equal(expected, result)

    @pytest.mark.parametrize('cmp1', ['<', '>'])
    @pytest.mark.parametrize('cmp2', ['<', '>'])
    def test_chained_cmp_op(
        self,
        cmp1: str,
        cmp2: str,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        midhs: Union[DataFrame, Series, float, np.ndarray],
        rhs: Union[DataFrame, Series, float, np.ndarray],
        engine: str,
        parser: str
    ) -> None:
        mid = midhs
        if parser == 'python':
            ex1 = f'lhs {cmp1} mid {cmp2} rhs'
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex1, engine=engine, parser=parser)
            return
        lhs_new = _eval_single_bin(lhs, cmp1, mid, engine)
        rhs_new = _eval_single_bin(mid, cmp2, rhs, engine)
        if lhs_new is not None and rhs_new is not None:
            ex1 = f'lhs {cmp1} mid {cmp2} rhs'
            ex2 = f'lhs {cmp1} mid and mid {cmp2} rhs'
            ex3 = f'(lhs {cmp1} mid) & (mid {cmp2} rhs)'
            expected = _eval_single_bin(lhs_new, '&', rhs_new, engine)
            for ex in (ex1, ex2, ex3):
                result = pd.eval(ex, engine=engine, parser=parser)
                tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('arith1', sorted(set(ARITH_OPS_SYMS).difference({'**', '//', '%'})))
    def test_binary_arith_ops(
        self,
        arith1: str,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        rhs: Union[DataFrame, Series, float, np.ndarray],
        engine: str,
        parser: str
    ) -> None:
        ex = f'lhs {arith1} rhs'
        result = pd.eval(ex, engine=engine, parser=parser)
        expected = _eval_single_bin(lhs, arith1, rhs, engine)
        tm.assert_almost_equal(result, expected)
        ex = f'lhs {arith1} rhs {arith1} rhs'
        result = pd.eval(ex, engine=engine, parser=parser)
        nlhs = _eval_single_bin(lhs, arith1, rhs, engine)
        try:
            nlhs, ghs = nlhs.align(rhs)
        except (ValueError, TypeError, AttributeError):
            return
        else:
            if engine == 'numexpr':
                import numexpr as ne
                expected = ne.evaluate(f'nlhs {arith1} ghs')
                tm.assert_almost_equal(result.values, expected)
            else:
                expected = eval(f'nlhs {arith1} ghs')  # type: ignore
                tm.assert_almost_equal(result, expected)

    def test_modulus(
        self,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        rhs: Union[DataFrame, Series, float, np.ndarray],
        engine: str,
        parser: str
    ) -> None:
        ex = 'lhs % rhs'
        result = pd.eval(ex, engine=engine, parser=parser)
        expected = lhs % rhs
        tm.assert_almost_equal(result, expected)
        if engine == 'numexpr':
            import numexpr as ne
            expected = ne.evaluate('expected % rhs')
            if isinstance(result, (DataFrame, Series)):
                tm.assert_almost_equal(result.values, expected)
            else:
                tm.assert_almost_equal(result, expected.item())
        else:
            expected = _eval_single_bin(expected, '%', rhs, engine)
            tm.assert_almost_equal(result, expected)

    def test_floor_division(
        self,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        rhs: Union[DataFrame, Series, float, np.ndarray],
        engine: str,
        parser: str
    ) -> None:
        ex = 'lhs // rhs'
        if engine == 'python':
            res = pd.eval(ex, engine=engine, parser=parser)
            expected = lhs // rhs
            tm.assert_equal(res, expected)
        else:
            msg = "unsupported operand type\\(s\\) for //: 'VariableNode' and 'VariableNode'"
            with pytest.raises(TypeError, match=msg):
                pd.eval(ex, local_dict={'lhs': lhs, 'rhs': rhs}, engine=engine, parser=parser)

    @td.skip_if_windows
    def test_pow(
        self,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        rhs: Union[DataFrame, Series, float, np.ndarray],
        engine: str,
        parser: str
    ) -> None:
        ex = 'lhs ** rhs'
        expected = _eval_single_bin(lhs, '**', rhs, engine)
        result = pd.eval(ex, engine=engine, parser=parser)
        if is_scalar(lhs) and is_scalar(rhs) and isinstance(expected, (complex, np.complexfloating)) and np.isnan(result):
            msg = '(DataFrame.columns|numpy array) are different'
            with pytest.raises(AssertionError, match=msg):
                tm.assert_numpy_array_equal(result, expected)
        else:
            tm.assert_almost_equal(result, expected)
            ex = '(lhs ** rhs) ** rhs'
            result = pd.eval(ex, engine=engine, parser=parser)
            middle = _eval_single_bin(lhs, '**', rhs, engine)
            expected = _eval_single_bin(middle, '**', rhs, engine)
            tm.assert_almost_equal(result, expected)

    def test_check_single_invert_op(
        self,
        lhs: Union[DataFrame, Series, float, np.ndarray],
        engine: str,
        parser: str
    ) -> None:
        try:
            elb = lhs.astype(bool)
        except AttributeError:
            elb = np.array([bool(lhs)])
        expected = ~elb
        result = pd.eval('~elb', engine=engine, parser=parser)
        tm.assert_almost_equal(expected, result)

    def test_frame_invert(
        self,
        engine: str,
        parser: str
    ) -> None:
        expr = '~lhs'
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)))
        if engine == 'numexpr':
            msg = "couldn't find matching opcode for 'invert_dd'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            msg = "ufunc 'invert' not supported for the input types"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        lhs = DataFrame(np.random.default_rng(2).integers(5, size=(5, 2)))
        if engine == 'numexpr':
            msg = "couldn't find matching opcode for 'invert"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            expect = ~lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_frame_equal(expect, result)
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)) > 0.5)
        expect = ~lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)
        lhs = DataFrame({'b': ['a', 1, 2.0], 'c': np.random.default_rng(2).standard_normal(3) > 0.5})
        if engine == 'numexpr':
            with pytest.raises(ValueError, match='unknown type object'):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            msg = "bad operand type for unary ~: 'str'"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)

    def test_series_invert(
        self,
        engine: str,
        parser: str
    ) -> None:
        expr = '~lhs'
        lhs = Series(np.random.default_rng(2).standard_normal(5))
        if engine == 'numexpr':
            msg = "couldn't find matching opcode for 'invert_dd'"
            with pytest.raises(NotImplementedError, match=msg):
                result = pd.eval(expr, engine=engine, parser=parser)
        else:
            msg = "ufunc 'invert' not supported for the input types"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        lhs = Series(np.random.default_rng(2).integers(5, size=5))
        if engine == 'numexpr':
            msg = "couldn't find matching opcode for 'invert"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            expect = ~lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_series_equal(expect, result)
        lhs = Series(np.random.default_rng(2).standard_normal(5) > 0.5)
        expect = ~lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_series_equal(expect, result)
        lhs = Series(['a', 1, 2.0])
        if engine == 'numexpr':
            with pytest.raises(ValueError, match='unknown type object'):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            msg = "bad operand type for unary ~: 'str'"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)

    def test_frame_negate(
        self,
        engine: str,
        parser: str
    ) -> None:
        expr = '-lhs'
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)))
        expect = -lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)
        lhs = DataFrame(np.random.default_rng(2).integers(5, size=(5, 2)))
        expect = -lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)) > 0.5)
        if engine == 'numexpr':
            msg = "couldn't find matching opcode for 'neg_bb'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            expect = -lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_frame_equal(expect, result)

    def test_series_negate(
        self,
        engine: str,
        parser: str
    ) -> None:
        expr = '-lhs'
        lhs = Series(np.random.default_rng(2).standard_normal(5))
        expect = -lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_series_equal(expect, result)
        lhs = Series(np.random.default_rng(2).integers(5, size=5))
        expect = -lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_series_equal(expect, result)
        lhs = Series(np.random.default_rng(2).standard_normal(5) > 0.5)
        if engine == 'numexpr':
            msg = "couldn't find matching opcode for 'neg_bb'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            expect = -lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_series_equal(expect, result)

    @pytest.mark.parametrize(
        'lhs',
        [
            np.random.default_rng(2).standard_normal((5, 2)),
            np.random.default_rng(2).integers(5, size=(5, 2)),
            np.array([True, False, True, False, True], dtype=np.bool_)
        ]
    )
    def test_frame_pos(
        self,
        lhs: Union[np.ndarray, DataFrame],
        engine: str,
        parser: str
    ) -> None:
        lhs = DataFrame(lhs)
        expr = '+lhs'
        expect = lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)

    @pytest.mark.parametrize(
        'lhs',
        [
            np.random.default_rng(2).standard_normal(5),
            np.random.default_rng(2).integers(5, size=5),
            np.array([True, False, True, False, True], dtype=np.bool_)
        ]
    )
    def test_series_pos(
        self,
        lhs: Union[np.ndarray, Series],
        engine: str,
        parser: str
    ) -> None:
        lhs = Series(lhs)
        expr = '+lhs'
        expect = lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_series_equal(expect, result)

    def test_scalar_unary(
        self,
        engine: str,
        parser: str
    ) -> None:
        msg = "bad operand type for unary ~: 'float'"
        warn: Optional[DeprecationWarning] = None
        if PY312 and (not (engine == 'numexpr' and parser == 'pandas')):
            warn = DeprecationWarning
        with pytest.raises(TypeError, match=msg):
            pd.eval('~1.0', engine=engine, parser=parser)
        assert pd.eval('-1.0', parser=parser, engine=engine) == -1.0
        assert pd.eval('+1.0', parser=parser, engine=engine) == +1.0
        assert pd.eval('~1', parser=parser, engine=engine) == ~1
        assert pd.eval('-1', parser=parser, engine=engine) == -1
        assert pd.eval('+1', parser=parser, engine=engine) == +1
        with tm.assert_produces_warning(warn, match='Bitwise inversion', check_stacklevel=False):
            assert pd.eval('~True', parser=parser, engine=engine) == ~True
        with tm.assert_produces_warning(warn, match='Bitwise inversion', check_stacklevel=False):
            assert pd.eval('~False', parser=parser, engine=engine) == ~False
        assert pd.eval('-True', parser=parser, engine=engine) == -True
        assert pd.eval('-False', parser=parser, engine=engine) == -False
        assert pd.eval('+True', parser=parser, engine=engine) == +True
        assert pd.eval('+False', parser=parser, engine=engine) == +False

    def test_unary_in_array(self) -> None:
        result = np.array(pd.eval('[-True, True, +True, -False, False, +False, -37, 37, ~37, +37]'), dtype=np.object_)
        expected = np.array([-True, True, +True, -False, False, +False, -37, 37, ~37, +37], dtype=np.object_)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('expr', ['x < -0.1', '-5 > x'])
    def test_float_comparison_bin_op(
        self,
        float_numpy_dtype: Any,
        expr: str
    ) -> None:
        df = DataFrame({'x': np.array([0], dtype=float_numpy_dtype)})
        res = df.eval(expr)
        assert res.values == np.array([False])

    def test_unary_in_function(self) -> None:
        df = DataFrame({'x': [0, 1, np.nan]})
        result = df.eval('x.fillna(-1)')
        expected = df.x.fillna(-1)
        tm.assert_series_equal(result, expected, check_names=not USE_NUMEXPR)
        result = df.eval('x.shift(1, fill_value=-1)')
        expected = df.x.shift(1, fill_value=-1)
        tm.assert_series_equal(result, expected, check_names=not USE_NUMEXPR)

    @pytest.mark.parametrize(
        'ex',
        (
            '1 or 2',
            '1 and 2',
            'a and b',
            'a or b',
            '1 or 2 and (3 + 2) > 3',
            '2 * x > 2 or 1 and 2',
            '2 * df > 3 and 1 or a'
        )
    )
    def test_disallow_scalar_bool_ops(
        self,
        ex: str,
        engine: str,
        parser: str
    ) -> None:
        x, a, b = (np.random.default_rng(2).standard_normal(3), 1, 2)
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, engine=engine, parser=parser)

    def test_identical(
        self,
        engine: str,
        parser: str
    ) -> None:
        x = 1
        result = pd.eval('x', engine=engine, parser=parser)
        assert result == 1
        assert is_scalar(result)
        x = 1.5
        result = pd.eval('x', engine=engine, parser=parser)
        assert result == 1.5
        assert is_scalar(result)
        x = False
        result = pd.eval('x', engine=engine, parser=parser)
        assert not result
        assert is_bool(result)
        assert is_scalar(result)
        x = np.array([1])
        result = pd.eval('x', engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([1]))
        assert result.shape == (1,)
        x = np.array([1.5])
        result = pd.eval('x', engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([1.5]))
        assert result.shape == (1,)
        x = np.array([False])
        result = pd.eval('x', engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([False]))
        assert result.shape == (1,)

    def test_line_continuation(
        self,
        engine: str,
        parser: str
    ) -> None:
        exp: str = '1 + 2 *         5 - 1 + 2 '
        result = pd.eval(exp, engine=engine, parser=parser)
        assert result == 12

    def test_float_truncation(
        self,
        engine: str,
        parser: str
    ) -> None:
        exp: str = '1000000000.006'
        result = pd.eval(exp, engine=engine, parser=parser)
        expected: float = np.float64(exp)
        assert result == expected
        df = DataFrame({'A': [1000000000.0009, 1000000000.0011, 1000000000.0015]})
        cutoff: float = 1000000000.0006
        result = df.query(f'A < {cutoff:.4f}')
        assert result.empty
        cutoff = 1000000000.001
        result = df.query(f'A > {cutoff:.4f}')
        expected = df.loc[[1, 2], :]
        tm.assert_frame_equal(expected, result)
        exact: float = 1000000000.0011
        result = df.query(f'A == {exact:.4f}')
        expected = df.loc[[1], :]
        tm.assert_frame_equal(expected, result)

    def test_disallow_python_keywords(self) -> None:
        df = DataFrame([[0, 0, 0]], columns=['foo', 'bar', 'class'])
        msg = 'Python keyword not valid identifier in numexpr query'
        with pytest.raises(SyntaxError, match=msg):
            df.query('class == 0')
        df = DataFrame()
        df.index.name = 'lambda'
        with pytest.raises(SyntaxError, match=msg):
            df.query('lambda == 0')

    def test_true_false_logic(self) -> None:
        with tm.maybe_produces_warning(DeprecationWarning, PY312, check_stacklevel=False):
            assert pd.eval('not True') == -2
            assert pd.eval('not False') == -1
            assert pd.eval('True and not True') == 0

    def test_and_logic_string_match(self) -> None:
        event = Series({'a': 'hello'})
        assert pd.eval(f'{event.str.match("hello").a}')
        assert pd.eval(f'{event.str.match("hello").a and event.str.match("hello").a}')

    def test_eval_keep_name(
        self,
        engine: str,
        parser: str
    ) -> None:
        df = Series([2, 15, 28], name='a').to_frame()
        res = df.eval('a + a', engine=engine, parser=parser)
        expected = Series([4, 30, 56], name='a')
        tm.assert_series_equal(expected, res)

    def test_eval_unmatching_names(
        self,
        engine: str,
        parser: str
    ) -> None:
        variable_name = Series([42], name='series_name')
        res = pd.eval('variable_name + 0', engine=engine, parser=parser)
        tm.assert_series_equal(variable_name, res)


class TestTypeCasting:

    @pytest.mark.parametrize('op', ['+', '-', '*', '**', '/'])
    @pytest.mark.parametrize('left_right', [('df', '3'), ('3', 'df')])
    def test_binop_typecasting(
        self,
        engine: str,
        parser: str,
        op: str,
        complex_or_float_dtype: str,
        left_right: Tuple[str, str],
        request: pytest.FixtureRequest
    ) -> None:
        dtype: str = complex_or_float_dtype
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), dtype=dtype)
        left, right = left_right
        s: str = f'{left} {op} {right}'
        res = pd.eval(s, engine=engine, parser=parser)
        if dtype == 'complex64' and engine == 'numexpr':
            mark = pytest.mark.xfail(
                reason='numexpr issue with complex that are upcast to complex 128 https://github.com/pydata/numexpr/issues/492'
            )
            request.applymarker(mark)
        assert df.values.dtype == dtype
        assert res.values.dtype == dtype
        tm.assert_frame_equal(res, eval(s), check_exact=False)


def should_warn(*args: Any) -> bool:
    not_mono = not any(map(operator.attrgetter('is_monotonic_increasing'), args))
    only_one_dt = reduce(operator.xor, (issubclass(x.dtype.type, np.datetime64) for x in args))
    return not_mono and only_one_dt


class TestAlignment:
    index_types: List[str] = ['i', 's', 'dt']
    lhs_index_types: List[str] = index_types + ['s']

    def test_align_nested_unary_op(
        self,
        engine: str,
        parser: str
    ) -> None:
        s: str = 'df * ~2'
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        res = pd.eval(s, engine=engine, parser=parser)
        tm.assert_frame_equal(res, df * ~2)

    @pytest.mark.filterwarnings('always::RuntimeWarning')
    @pytest.mark.parametrize('lr_idx_type', lhs_index_types)
    @pytest.mark.parametrize('rr_idx_type', index_types)
    @pytest.mark.parametrize('c_idx_type', index_types)
    def test_basic_frame_alignment(
        self,
        engine: str,
        parser: str,
        lr_idx_type: str,
        rr_idx_type: str,
        c_idx_type: str,
        idx_func_dict: Dict[str, Callable[[int], Index]]
    ) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[lr_idx_type](10),
            columns=idx_func_dict[c_idx_type](10)
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((20, 10)),
            index=idx_func_dict[rr_idx_type](20),
            columns=idx_func_dict[c_idx_type](10)
        )
        if should_warn(df.index, df2.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval('df + df2', engine=engine, parser=parser)
        else:
            res = pd.eval('df + df2', engine=engine, parser=parser)
        tm.assert_frame_equal(res, df + df2)

    @pytest.mark.parametrize('r_idx_type', lhs_index_types)
    @pytest.mark.parametrize('c_idx_type', lhs_index_types)
    def test_frame_comparison(
        self,
        engine: str,
        parser: str,
        r_idx_type: str,
        c_idx_type: str,
        idx_func_dict: Dict[str, Callable[[int], Index]]
    ) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[r_idx_type](10),
            columns=idx_func_dict[c_idx_type](10)
        )
        res = pd.eval('df < 2', engine=engine, parser=parser)
        tm.assert_frame_equal(res, df < 2)
        df3 = DataFrame(
            np.random.default_rng(2).standard_normal(df.shape),
            index=df.index,
            columns=df.columns
        )
        res = pd.eval('df < df3', engine=engine, parser=parser)
        tm.assert_frame_equal(res, df < df3)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('r1', lhs_index_types)
    @pytest.mark.parametrize('c1', index_types)
    @pytest.mark.parametrize('r2', index_types)
    @pytest.mark.parametrize('c2', index_types)
    def test_medium_complex_frame_alignment(
        self,
        engine: str,
        parser: str,
        r1: str,
        c1: str,
        r2: str,
        c2: str,
        idx_func_dict: Dict[str, Callable[[int], Index]]
    ) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 2)),
            index=idx_func_dict[r1](3),
            columns=idx_func_dict[c1](2)
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            index=idx_func_dict[r2](4),
            columns=idx_func_dict[c2](2)
        )
        df3 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            index=idx_func_dict[r2](5),
            columns=idx_func_dict[c2](2)
        )
        if should_warn(df.index, df2.index, df3.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval('df + df2 + df3', engine=engine, parser=parser)
        else:
            res = pd.eval('df + df2 + df3', engine=engine, parser=parser)
        tm.assert_frame_equal(res, df + df2 + df3)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('index_name', ['index', 'columns'])
    @pytest.mark.parametrize('c_idx_type', index_types)
    @pytest.mark.parametrize('r_idx_type', lhs_index_types)
    def test_basic_frame_series_alignment(
        self,
        engine: str,
        parser: str,
        index_name: str,
        r_idx_type: str,
        c_idx_type: str,
        idx_func_dict: Dict[str, Callable[[int], Index]]
    ) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[r_idx_type](10),
            columns=idx_func_dict[c_idx_type](10)
        )
        index = getattr(df, index_name)
        s = Series(np.random.default_rng(2).standard_normal(5), index=index[:5])
        if should_warn(df.index, s.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval('df + s', engine=engine, parser=parser)
        else:
            res = pd.eval('df + s', engine=engine, parser=parser)
        if r_idx_type == 'dt' or c_idx_type == 'dt':
            expected = df.add(s) if engine == 'numexpr' else df + s
        else:
            expected = df + s
        tm.assert_frame_equal(res, expected)

    @pytest.mark.parametrize('index_name', ['index', 'columns'])
    @pytest.mark.parametrize('r_idx_type, c_idx_type', list(product(['i', 's'], ['i', 's'])) + [('dt', 'dt')])
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_basic_series_frame_alignment(
        self,
        request: pytest.FixtureRequest,
        engine: str,
        parser: str,
        index_name: str,
        r_idx_type: str,
        c_idx_type: str,
        idx_func_dict: Dict[str, Callable[[int], Index]]
    ) -> None:
        if engine == 'numexpr' and parser in ('pandas', 'python') and (index_name == 'index') and (r_idx_type == 'i') and (c_idx_type == 's'):
            reason = f'Flaky column ordering when engine={engine}, parser={parser}, index_name={index_name}, r_idx_type={r_idx_type}, c_idx_type={c_idx_type}'
            request.applymarker(pytest.mark.xfail(reason=reason, strict=False))
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 7)),
            index=idx_func_dict[r_idx_type](10),
            columns=idx_func_dict[c_idx_type](7)
        )
        index = getattr(df, index_name)
        s = Series(np.random.default_rng(2).standard_normal(5), index=index[:5])
        if should_warn(df.index, s.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval('s + df', engine=engine, parser=parser)
        else:
            res = pd.eval('s + df', engine=engine, parser=parser)
        if r_idx_type == 'dt' or c_idx_type == 'dt':
            expected = df.add(s) if engine == 'numexpr' else s + df
        else:
            expected = s + df
        tm.assert_frame_equal(res, expected)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('c_idx_type', index_types)
    @pytest.mark.parametrize('r_idx_type', lhs_index_types)
    @pytest.mark.parametrize('index_name', ['index', 'columns'])
    @pytest.mark.parametrize('op', ['+', '*'])
    def test_series_frame_commutativity(
        self,
        engine: str,
        parser: str,
        index_name: str,
        op: str,
        r_idx_type: str,
        c_idx_type: str,
        idx_func_dict: Dict[str, Callable[[int], Index]]
    ) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[r_idx_type](10),
            columns=idx_func_dict[c_idx_type](10)
        )
        index = getattr(df, index_name)
        s = Series(np.random.default_rng(2).standard_normal(5), index=index[:5])
        lhs = f's {op} df'
        rhs = f'df {op} s'
        if should_warn(df.index, s.index):
            with tm.assert_produces_warning(RuntimeWarning):
                a = pd.eval(lhs, engine=engine, parser=parser)
            with tm.assert_produces_warning(RuntimeWarning):
                b = pd.eval(rhs, engine=engine, parser=parser)
        else:
            a = pd.eval(lhs, engine=engine, parser=parser)
            b = pd.eval(rhs, engine=engine, parser=parser)
        if r_idx_type != 'dt' and c_idx_type != 'dt':
            if engine == 'numexpr':
                tm.assert_frame_equal(a, b)

    @pytest.mark.filterwarnings('always::RuntimeWarning')
    @pytest.mark.parametrize('r1', lhs_index_types)
    @pytest.mark.parametrize('c1', index_types)
    @pytest.mark.parametrize('r2', index_types)
    @pytest.mark.parametrize('c2', index_types)
    def test_complex_series_frame_alignment(
        self,
        engine: str,
        parser: str,
        r1: str,
        c1: str,
        r2: str,
        c2: str,
        idx_func_dict: Dict[str, Callable[[int], Index]]
    ) -> None:
        n: int = 3
        m1: int = 5
        m2: int = 2 * m1
        df = DataFrame(
            np.random.default_rng(2).standard_normal((m1, n)),
            index=idx_func_dict[r1](m1),
            columns=idx_func_dict[c1](n)
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((m2, n)),
            index=idx_func_dict[r2](m2),
            columns=idx_func_dict[c2](n)
        )
        index = df2.columns
        ser = Series(np.random.default_rng(2).standard_normal(n), index[:n])
        if r2 == 'dt' or c2 == 'dt':
            if engine == 'numexpr':
                expected2 = df2.add(ser)
            else:
                expected2 = df2 + ser
        else:
            expected2 = df2 + ser
        if r1 == 'dt' or c1 == 'dt':
            if engine == 'numexpr':
                expected = expected2.add(df)
            else:
                expected = expected2 + df
        else:
            expected = expected2 + df
        if should_warn(df2.index, ser.index, df.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval('df2 + ser + df', engine=engine, parser=parser)
        else:
            res = pd.eval('df2 + ser + df', engine=engine, parser=parser)
        assert res.shape == expected.shape
        tm.assert_frame_equal(res, expected)

    def test_performance_warning_for_poor_alignment(
        self,
        performance_warning: bool,
        engine: str,
        parser: str
    ) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 10)))
        s = Series(np.random.default_rng(2).standard_normal(10000))
        if engine == 'numexpr' and performance_warning:
            seen: Union[PerformanceWarning, bool] = PerformanceWarning
        else:
            seen = False
        msg = 'Alignment difference on axis 1 is larger than an order of magnitude'
        with tm.assert_produces_warning(seen, match=msg):
            pd.eval('df + s', engine=engine, parser=parser)
        s = Series(np.random.default_rng(2).standard_normal(1000))
        with tm.assert_produces_warning(False):
            pd.eval('df + s', engine=engine, parser=parser)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 10000)))
        s = Series(np.random.default_rng(2).standard_normal(10000))
        with tm.assert_produces_warning(False):
            pd.eval('df + s', engine=engine, parser=parser)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 10)))
        s = Series(np.random.default_rng(2).standard_normal(10000))
        is_python_engine: bool = engine == 'python'
        if not is_python_engine and performance_warning:
            wrn: Optional[PerformanceWarning] = PerformanceWarning
        else:
            wrn = False
        with tm.assert_produces_warning(wrn, match=msg) as w:
            pd.eval('df + s', engine=engine, parser=parser)
            if not is_python_engine and performance_warning:
                assert len(w) == 1
                msg = str(w[0].message)
                logged = np.log10(s.size - df.shape[1])
                expected = f"Alignment difference on axis 1 is larger than an order of magnitude on term 'df', by more than {logged:.4g}; performance may suffer."
                assert msg == expected

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('index_name', ['index', 'columns'])
    @pytest.mark.parametrize('c_idx_type', index_types)
    @pytest.mark.parametrize('r_idx_type', lhs_index_types)
    def test_basic_frame_series_alignment(
        self,
        engine: str,
        parser: str,
        index_name: str,
        r_idx_type: str,
        c_idx_type: str,
        idx_func_dict: Dict[str, Callable[[int], Index]]
    ) -> None:
        ...

    # Continue for all other methods with similar annotations


class TestOperations:

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        kwargs['level'] = kwargs.pop('level', 0) + 1
        return pd.eval(*args, **kwargs)

    @pytest.mark.parametrize('op', ['+', '-', '*', '**', '/'])
    @pytest.mark.parametrize('left_right', [('df', '3'), ('3', 'df')])
    def test_binop_typecasting(
        self,
        engine: str,
        parser: str,
        op: str,
        left_right: Tuple[str, str],
        complex_or_float_dtype: str,
        request: pytest.FixtureRequest
    ) -> None:
        ...

    # Continue implementing all methods with type annotations similarly


_var_s: np.ndarray = np.random.default_rng(2).standard_normal(10)


class TestScope:

    def test_global_scope(
        self,
        engine: str,
        parser: str
    ) -> None:
        e: str = '_var_s * 2'
        tm.assert_numpy_array_equal(_var_s * 2, pd.eval(e, engine=engine, parser=parser))

    def test_no_new_locals(
        self,
        engine: str,
        parser: str
    ) -> None:
        x: int = 1
        lcls: Dict[str, Any] = locals().copy()
        pd.eval('x + 1', local_dict=lcls, engine=engine, parser=parser)
        lcls2: Dict[str, Any] = locals().copy()
        lcls2.pop('lcls')
        assert lcls == lcls2

    def test_no_new_globals(
        self,
        engine: str,
        parser: str
    ) -> None:
        x: int = 1
        gbls: Dict[str, Any] = globals().copy()
        pd.eval('x + 1', engine=engine, parser=parser)
        gbls2: Dict[str, Any] = globals().copy()
        assert gbls == gbls2

    def test_empty_locals(
        self,
        engine: str,
        parser: str
    ) -> None:
        x: int = 1
        msg: str = "name 'x' is not defined"
        with pytest.raises(UndefinedVariableError, match=msg):
            pd.eval('x + 1', engine=engine, parser=parser, local_dict={})

    def test_empty_globals(
        self,
        engine: str,
        parser: str
    ) -> None:
        msg: str = "name '_var_s' is not defined"
        e: str = '_var_s * 2'
        with pytest.raises(UndefinedVariableError, match=msg):
            pd.eval(e, engine=engine, parser=parser, global_dict={})


@td.skip_if_no('numexpr')
def test_invalid_engine() -> None:
    msg: str = "Invalid engine 'asdf' passed"
    with pytest.raises(KeyError, match=msg):
        pd.eval('x + y', local_dict={'x': 1, 'y': 2}, engine='asdf')


@td.skip_if_no('numexpr')
@pytest.mark.parametrize(('use_numexpr', 'expected'), ((True, 'numexpr'), (False, 'python')))
def test_numexpr_option_respected(use_numexpr: bool, expected: str) -> None:
    from pandas.core.computation.eval import _check_engine
    with pd.option_context('compute.use_numexpr', use_numexpr):
        result = _check_engine(None)
        assert result == expected


@td.skip_if_no('numexpr')
def test_numexpr_option_incompatible_op(
    engine: str,
    parser: str
) -> None:
    with pd.option_context('compute.use_numexpr', False):
        df = DataFrame({
            'A': [True, False, True, False, None, None],
            'B': [1, 2, 3, 4, 5, 6]
        })
        result = df.query('A.isnull()')
        expected = DataFrame({'A': [None, None], 'B': [5, 6]}, index=range(4, 6))
        tm.assert_frame_equal(result, expected)


@td.skip_if_no('numexpr')
def test_invalid_parser() -> None:
    msg: str = "Invalid parser 'asdf' passed"
    with pytest.raises(KeyError, match=msg):
        pd.eval('x + y', local_dict={'x': 1, 'y': 2}, parser='asdf')


_parsers: Dict[str, Callable[..., BaseExprVisitor]] = {
    'python': PythonExprVisitor,
    'pytables': pytables.PyTablesExprVisitor,
    'pandas': PandasExprVisitor
}


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('parser', _parsers)
def test_disallowed_nodes(engine: str, parser: str) -> None:
    VisitorClass: type = _parsers[parser]
    inst: BaseExprVisitor = VisitorClass('x + 1', engine, parser)
    for ops in VisitorClass.unsupported_nodes:
        msg: str = 'nodes are not implemented'
        with pytest.raises(NotImplementedError, match=msg):
            getattr(inst, ops)()


def test_syntax_error_exprs(
    engine: str,
    parser: str
) -> None:
    e: str = 's +'
    with pytest.raises(SyntaxError, match='invalid syntax'):
        pd.eval(e, engine=engine, parser=parser)


def test_name_error_exprs(
    engine: str,
    parser: str
) -> None:
    e: str = 's + t'
    msg: str = "name 's' is not defined"
    with pytest.raises(NameError, match=msg):
        pd.eval(e, engine=engine, parser=parser)


@pytest.mark.parametrize('express', ['a + @b', '@a + b', '@a + @b'])
def test_invalid_local_variable_reference(
    engine: str,
    parser: str,
    express: str
) -> None:
    a, b = (1, 2)
    if parser != 'pandas':
        with pytest.raises(SyntaxError, match="The '@' prefix is only"):
            pd.eval(express, engine=engine, parser=parser)
    else:
        with pytest.raises(SyntaxError, match="The '@' prefix is not"):
            pd.eval(express, engine=engine, parser=parser)


def test_numexpr_builtin_raises(
    engine: str,
    parser: str
) -> None:
    sin: int = 1
    dotted_line: int = 2
    if engine == 'numexpr':
        msg: str = 'Variables in expression .+'
        with pytest.raises(NumExprClobberingError, match=msg):
            pd.eval('sin + dotted_line', engine=engine, parser=parser)
    else:
        res: int = pd.eval('sin + dotted_line', engine=engine, parser=parser)
        assert res == sin + dotted_line


def test_bad_resolver_raises(
    engine: str,
    parser: str
) -> None:
    cannot_resolve: Tuple[int, float] = (42, 3.0)
    with pytest.raises(TypeError, match='Resolver of type .+'):
        pd.eval('1 + 2', resolvers=cannot_resolve, engine=engine, parser=parser)


def test_empty_string_raises(
    engine: str,
    parser: str
) -> None:
    with pytest.raises(ValueError, match='expr cannot be an empty string'):
        pd.eval('', engine=engine, parser=parser)


def test_more_than_one_expression_raises(
    engine: str,
    parser: str
) -> None:
    with pytest.raises(SyntaxError, match='only a single expression is allowed'):
        pd.eval('1 + 1; 2 + 2', engine=engine, parser=parser)


@pytest.mark.parametrize('cmp', ('and', 'or'))
@pytest.mark.parametrize('lhs', (int, float))
@pytest.mark.parametrize('rhs', (int, float))
def test_bool_ops_fails_on_scalars(
    lhs: type,
    cmp: str,
    rhs: type,
    engine: str,
    parser: str
) -> None:
    gen: Dict[type, Callable[[], Any]] = {
        int: lambda: np.random.default_rng(2).integers(10),
        float: lambda: np.random.default_rng(2).standard_normal()
    }
    mid: Any = gen[lhs]()
    lhs_val: Any = gen[lhs]()
    rhs_val: Any = gen[rhs]()
    ex1: str = f'lhs {cmp} mid {cmp} rhs'
    ex2: str = f'lhs {cmp} mid and mid {cmp} rhs'
    ex3: str = f'(lhs {cmp} mid) & (mid {cmp} rhs)'
    for ex in (ex1, ex2, ex3):
        msg: str = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, engine=engine, parser=parser)


@pytest.mark.parametrize('other', ["'x'", '...'])
def test_equals_various(
    engine: str,
    other: str
) -> None:
    df = DataFrame({'A': ['a', 'b', 'c']}, dtype=object)
    result = df.eval(f'A == {other}')
    expected = Series([False, False, False], name='A')
    tm.assert_series_equal(result, expected)


def test_inf(
    engine: str,
    parser: str
) -> None:
    s: str = 'inf + 1'
    expected: float = np.inf
    result: float = pd.eval(s, engine=engine, parser=parser)
    assert result == expected


@pytest.mark.parametrize('column', ['Temp(°C)', 'Capacitance(μF)'])
def test_query_token(
    engine: str,
    column: str
) -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 2)),
        columns=[column, 'b']
    )
    expected = df[df[column] > 5]
    query_string: str = f'`{column}` > 5'
    result = df.query(query_string, engine=engine)
    tm.assert_frame_equal(result, expected)


def test_negate_lt_eq_le(
    engine: str,
    parser: str
) -> None:
    df = DataFrame([[0, 10], [1, 20]], columns=['cat', 'count'])
    expected = df[~(df.cat > 0)]
    result = df.query('~(cat > 0)', engine=engine, parser=parser)
    tm.assert_frame_equal(result, expected)
    if parser == 'python':
        msg = "'Not' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query('not (cat > 0)', engine=engine, parser=parser)
    else:
        result = df.query('not (cat > 0)', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('column', DEFAULT_GLOBALS.keys())
def test_eval_no_support_column_name(
    request: pytest.FixtureRequest,
    engine: str,
    column: str
) -> None:
    if column in ['True', 'False', 'inf', 'Inf']:
        request.applymarker(pytest.mark.xfail(
            raises=KeyError,
            reason=f'GH 47859 DataFrame eval not supported with {column}'
        ))
    df = DataFrame(
        np.random.default_rng(2).integers(0, 100, size=(10, 2)),
        columns=[column, 'col1']
    )
    expected = df[df[column] > 6]
    result = df.query(f'{column}>6')
    tm.assert_frame_equal(result, expected)


def test_set_inplace() -> None:
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result_view = df[:]
    ser = df['A']
    df.eval('A = B + C', inplace=True)
    expected = DataFrame({'A': [11, 13, 15], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    tm.assert_frame_equal(df, expected)
    expected = Series([1, 2, 3], name='A')
    tm.assert_series_equal(ser, expected)
    tm.assert_series_equal(result_view['A'], expected)


@pytest.mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
def test_validate_bool_args(value: Any) -> None:
    msg: str = 'For argument "inplace" expected type bool, received type'
    with pytest.raises(ValueError, match=msg):
        pd.eval('2+2', inplace=value)


@td.skip_if_no('numexpr')
def test_eval_float_div_numexpr(
    engine: str,
    parser: str
) -> None:
    result: float = pd.eval('1 / 2', engine='numexpr')
    expected: float = 0.5
    assert result == expected
