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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator

@pytest.fixture(params=(pytest.param(engine, marks=[pytest.mark.skipif(engine == 'numexpr' and (not USE_NUMEXPR), reason=f'numexpr enabled->{USE_NUMEXPR}, installed->{NUMEXPR_INSTALLED}'), td.skip_if_no('numexpr')]) for engine in ENGINES))
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
def lhs(request: pytest.FixtureRequest) -> Union[DataFrame, Series, float]:
    nan_df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    nan_df1[nan_df1 > 0.5] = np.nan
    opts = (DataFrame(np.random.default_rng(2).standard_normal((10, 5))), Series(np.random.default_rng(2).standard_normal(5)), Series([1, 2, np.nan, np.nan, 5]), nan_df1, np.random.default_rng(2).standard_normal())
    return opts[request.param]

rhs: Union[DataFrame, Series, float] = lhs
midhs: Union[DataFrame, Series, float] = lhs

@pytest.fixture
def idx_func_dict() -> Dict[str, Callable[[int], Index]]:
    return {'i': lambda n: Index(np.arange(n), dtype=np.int64), 'f': lambda n: Index(np.arange(n), dtype=np.float64), 's': lambda n: Index([f'{i}_{chr(i)}' for i in range(97, 97 + n)]), 'dt': lambda n: date_range('2020-01-01', periods=n), 'td': lambda n: timedelta_range('1 day', periods=n), 'p': lambda n: period_range('2020-01-01', periods=n, freq='D')}

class TestEval:
    @pytest.mark.parametrize('cmp1', ['!=', '==', '<=', '>=', '<', '>'], ids=['ne', 'eq', 'le', 'ge', 'lt', 'gt'])
    @pytest.mark.parametrize('cmp2', ['>', '<'], ids=['gt', 'lt'])
    @pytest.mark.parametrize('binop', expr.BOOL_OPS_SYMS)
    def test_complex_cmp_ops(self, cmp1: str, cmp2: str, binop: str, lhs: Union[DataFrame, Series, float], rhs: Union[DataFrame, Series, float], engine: str, parser: str) -> None:
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
    def test_simple_cmp_ops(self, cmp_op: str, lhs: Union[DataFrame, Series, float], rhs: Union[DataFrame, Series, float], engine: str, parser: str) -> None:
        lhs = lhs < 0
        rhs = rhs < 0
        if parser == 'python' and cmp_op in ['in', 'not in']:
            msg = "'(In|NotIn)' nodes are not implemented"
            ex = f'lhs {cmp_op} rhs'
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, engine=engine, parser=parser)
            return
        ex = f'lhs {cmp_op} rhs'
        msg = '|'.join(["only list-like( or dict-like)? objects are allowed to be passed to (DataFrame\\.)?isin\\(\\), you passed a (`|')bool(`|')", "argument of type 'bool' is not iterable"])
        if cmp_op in ('in', 'not in') and (not is_list_like(rhs)):
            with pytest.raises(TypeError, match=msg):
                pd.eval(ex, engine=engine, parser=parser, local_dict={'lhs': lhs, 'rhs': rhs})
        else:
            expected = _eval_single_bin(lhs, cmp_op, rhs, engine)
            result = pd.eval(ex, engine=engine, parser=parser)
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize('op', expr.CMP_OPS_SYMS)
    def test_compound_invert_op(self, op: str, lhs: Union[DataFrame, Series, float], rhs: Union[DataFrame, Series, float], request: pytest.FixtureRequest, engine: str, parser: str) -> None:
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
        msg = '|'.join(["only list-like( or dict-like)? objects are allowed to be passed to (DataFrame\\.)?isin\\(\\), you passed a (`|')float(`|')", "argument of type 'float' is not iterable"])
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
    def test_chained_cmp_op(self, cmp1: str, cmp2: str, lhs: Union[DataFrame, Series, float], midhs: Union[DataFrame, Series, float], rhs: Union[DataFrame, Series, float], engine: str, parser: str) -> None:
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
    def test_binary_arith_ops(self, arith1: str, lhs: Union[DataFrame, Series, float], rhs: Union[DataFrame, Series, float], engine: str, parser: str) -> None:
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
                expected = eval(f'nlhs {arith1} ghs')
                tm.assert_almost_equal(result, expected)

    def test_modulus(self, lhs: Union[DataFrame, Series, float], rhs: Union[DataFrame, Series, float], engine: str, parser: str) -> None:
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

    def test_floor_division(self, lhs: Union[DataFrame, Series, float], rhs: Union[DataFrame, Series, float], engine: str, parser: str) -> None:
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
    def test_pow(self, lhs: Union[DataFrame, Series, float], rhs: Union[DataFrame, Series, float], engine: str, parser: str) -> None:
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

    def test_check_single_invert_op(self, lhs: Union[DataFrame, Series, float], engine: str, parser: str) -> None:
        try:
            elb = lhs.astype(bool)
        except AttributeError:
            elb = np.array([bool(lhs)])
        expected = ~elb
        result = pd.eval('~elb', engine=engine, parser=parser)
        tm.assert_almost_equal(expected, result)

    def test_frame_invert(self, engine: str, parser: str) -> None:
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

    def test_series_invert(self, engine: str, parser: str) -> None:
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
        lhs = Series(np.random.default_rng(2).