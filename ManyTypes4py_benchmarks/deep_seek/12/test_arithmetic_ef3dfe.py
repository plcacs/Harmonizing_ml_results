from datetime import date, timedelta, timezone
from decimal import Decimal
import operator
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import Categorical, DatetimeTZDtype, Index, Series, Timedelta, bdate_range, date_range, isna
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED

@pytest.fixture(autouse=True, params=[0, 1000000], ids=['numexpr', 'python'])
def switch_numexpr_min_elements(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', request.param)
        yield

def _permute(obj: Series) -> Series:
    return obj.take(np.random.default_rng(2).permutation(len(obj)))

class TestSeriesFlexArithmetic:

    @pytest.mark.parametrize('ts', [(lambda x: x, lambda x: x * 2, False), (lambda x: x, lambda x: x[::2], False), (lambda x: x, lambda x: 5, True), (lambda x: Series(range(10), dtype=np.float64), lambda x: Series(range(10), dtype=np.float64), True)])
    @pytest.mark.parametrize('opname', ['add', 'sub', 'mul', 'floordiv', 'truediv', 'pow'])
    def test_flex_method_equivalence(self, opname: str, ts: Tuple[Callable[[Series], Series], Callable[[Series], Any], bool]) -> None:
        tser = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20), name='ts')
        series = ts[0](tser)
        other = ts[1](tser)
        check_reverse = ts[2]
        op = getattr(Series, opname)
        alt = getattr(operator, opname)
        result = op(series, other)
        expected = alt(series, other)
        tm.assert_almost_equal(result, expected)
        if check_reverse:
            rop = getattr(Series, 'r' + opname)
            result = rop(series, other)
            expected = alt(other, series)
            tm.assert_almost_equal(result, expected)

    def test_flex_method_subclass_metadata_preservation(self, all_arithmetic_operators: str) -> None:
        class MySeries(Series):
            _metadata = ['x']

            @property
            def _constructor(self) -> 'MySeries':
                return MySeries
        opname = all_arithmetic_operators
        op = getattr(Series, opname)
        m = MySeries([1, 2, 3], name='test')
        m.x = 42
        result = op(m, 1)
        assert result.x == 42

    def test_flex_add_scalar_fill_value(self) -> None:
        ser = Series([0, 1, np.nan, 3, 4, 5])
        exp = ser.fillna(0).add(2)
        res = ser.add(2, fill_value=0)
        tm.assert_series_equal(res, exp)
    pairings: List[Tuple[Callable[[Series, Series], Series], Callable[[Any, Any], Any], int]] = [(Series.div, operator.truediv, 1), (Series.rdiv, ops.rtruediv, 1)]
    for op in ['add', 'sub', 'mul', 'pow', 'truediv', 'floordiv']:
        fv = 0
        lop = getattr(Series, op)
        lequiv = getattr(operator, op)
        rop = getattr(Series, 'r' + op)
        requiv = lambda x, y, op=op: getattr(operator, op)(y, x)
        pairings.append((lop, lequiv, fv))
        pairings.append((rop, requiv, fv))

    @pytest.mark.parametrize('op, equiv_op, fv', pairings)
    def test_operators_combine(self, op: Callable[[Series, Series], Series], equiv_op: Callable[[Any, Any], Any], fv: int) -> None:
        def _check_fill(meth: Callable[[Series, Series], Series], op: Callable[[Any, Any], Any], a: Series, b: Series, fill_value: int = 0) -> None:
            exp_index = a.index.union(b.index)
            a = a.reindex(exp_index)
            b = b.reindex(exp_index)
            amask = isna(a)
            bmask = isna(b)
            exp_values = []
            for i in range(len(exp_index)):
                with np.errstate(all='ignore'):
                    if amask[i]:
                        if bmask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(fill_value, b[i]))
                    elif bmask[i]:
                        if amask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(a[i], fill_value))
                    else:
                        exp_values.append(op(a[i], b[i]))
            result = meth(a, b, fill_value=fill_value)
            expected = Series(exp_values, exp_index)
            tm.assert_series_equal(result, expected)
        a = Series([np.nan, 1.0, 2.0, 3.0, np.nan], index=np.arange(5))
        b = Series([np.nan, 1, np.nan, 3, np.nan, 4.0], index=np.arange(6))
        result = op(a, b)
        exp = equiv_op(a, b)
        tm.assert_series_equal(result, exp)
        _check_fill(op, equiv_op, a, b, fill_value=fv)
        op(a, b, axis=0)

class TestSeriesArithmetic:

    def test_add_series_with_period_index(self) -> None:
        rng = pd.period_range('1/1/2000', '1/1/2010', freq='Y')
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        result = ts + ts[::2]
        expected = ts + ts
        expected.iloc[1::2] = np.nan
        tm.assert_series_equal(result, expected)
        result = ts + _permute(ts[::2])
        tm.assert_series_equal(result, expected)
        msg = 'Input has different freq=D from Period\\(freq=Y-DEC\\)'
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts + ts.asfreq('D', how='end')

    @pytest.mark.parametrize('target_add,input_value,expected_value', [('!', ['hello', 'world'], ['hello!', 'world!']), ('m', ['hello', 'world'], ['hellom', 'worldm'])])
    def test_string_addition(self, target_add: str, input_value: List[str], expected_value: List[str]) -> None:
        a = Series(input_value)
        result = a + target_add
        expected = Series(expected_value)
        tm.assert_series_equal(result, expected)

    def test_divmod(self) -> None:
        a = Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
        b = Series([2, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
        result = a.divmod(b)
        expected = divmod(a, b)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])
        result = a.rdivmod(b)
        expected = divmod(b, a)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    @pytest.mark.parametrize('index', [None, range(9)])
    def test_series_integer_mod(self, index: Optional[range]) -> None:
        s1 = Series(range(1, 10))
        s2 = Series('foo', index=index)
        msg = "not all arguments converted during string formatting|'mod' not supported"
        with pytest.raises(TypeError, match=msg):
            s2 % s1

    def test_add_with_duplicate_index(self) -> None:
        s1 = Series([1, 2], index=[1, 1])
        s2 = Series([10, 10], index=[1, 2])
        result = s1 + s2
        expected = Series([11, 12, np.nan], index=[1, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_add_na_handling(self) -> None:
        ser = Series([Decimal('1.3'), Decimal('2.3')], index=[date(2012, 1, 1), date(2012, 1, 2)])
        result = ser + ser.shift(1)
        result2 = ser.shift(1) + ser
        assert isna(result.iloc[0])
        assert isna(result2.iloc[0])

    def test_add_corner_cases(self, datetime_series: Series) -> None:
        empty = Series([], index=Index([]), dtype=np.float64)
        result = datetime_series + empty
        assert np.isnan(result).all()
        result = empty + empty
        assert len(result) == 0

    def test_add_float_plus_int(self, datetime_series: Series) -> None:
        int_ts = datetime_series.astype(int)[:-5]
        added = datetime_series + int_ts
        expected = Series(datetime_series.values[:-5] + int_ts.values, index=datetime_series.index[:-5], name='ts')
        tm.assert_series_equal(added[:-5], expected)

    def test_mul_empty_int_corner_case(self) -> None:
        s1 = Series([], [], dtype=np.int32)
        s2 = Series({'x': 0.0})
        tm.assert_series_equal(s1 * s2, Series([np.nan], index=['x']))

    def test_sub_datetimelike_align(self) -> None:
        dt = Series(date_range('2012-1-1', periods=3, freq='D'))
        dt.iloc[2] = np.nan
        dt2 = dt[::-1]
        expected = Series([timedelta(0), timedelta(0), pd.NaT])
        result = dt2 - dt
        tm.assert_series_equal(result, expected)
        expected = Series(expected, name=0)
        result = (dt2.to_frame() - dt.to_frame())[0]
        tm.assert_series_equal(result, expected)

    def test_alignment_doesnt_change_tz(self) -> None:
        dti = date_range('2016-01-01', periods=10, tz='CET')
        dti_utc = dti.tz_convert('UTC')
        ser = Series(10, index=dti)
        ser_utc = Series(10, index=dti_utc)
        ser * ser_utc
        assert ser.index is dti
        assert ser_utc.index is dti_utc

    def test_alignment_categorical(self) -> None:
        cat = Categorical(['3z53', '3z53', 'LoJG', 'LoJG', 'LoJG', 'N503'])
        ser1 = Series(2, index=cat)
        ser2 = Series(2, index=cat[:-1])
        result = ser1 * ser2
        exp_index = ['3z53'] * 4 + ['LoJG'] * 9 + ['N503']
        exp_index = pd.CategoricalIndex(exp_index, categories=cat.categories)
        exp_values = [4.0] * 13 + [np.nan]
        expected = Series(exp_values, exp_index)
        tm.assert_series_equal(result, expected)

    def test_arithmetic_with_duplicate_index(self) -> None:
        index = [2, 2, 3, 3, 4]
        ser = Series(np.arange(1, 6, dtype='int64'), index=index)
        other = Series(np.arange(5, dtype='int64'), index=index)
        result = ser - other
        expected = Series(1, index=[2, 2, 3, 3, 4])
        tm.assert_series_equal(result, expected)
        ser = Series(date_range('20130101 09:00:00', periods=5), index=index)
        other = Series(date_range('20130101', periods=5), index=index)
        result = ser - other
        expected = Series(Timedelta('9 hours'), index=[2, 2, 3, 3, 4])
        tm.assert_series_equal(result, expected)

    def test_masked_and_non_masked_propagate_na(self) -> None:
        ser1 = Series([0, np.nan], dtype='float')
        ser2 = Series([0, 1], dtype='Int64')
        result = ser1 * ser2
        expected = Series([0, pd.NA], dtype='Float64')
        tm.assert_series_equal(result, expected)

    def test_mask_div_propagate_na_for_non_na_dtype(self) -> None:
        ser1 = Series([15, pd.NA, 5, 4], dtype='Int64')
        ser2 = Series([15, 5, np.nan, 4])
        result = ser1 / ser2
        expected = Series([1.0, pd.NA, pd.NA, 1.0], dtype='Float64')
        tm.assert_series_equal(result, expected)
        result = ser2 / ser1
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('val, dtype', [(3, 'Int64'), (3.5, 'Float64')])
    def test_add_list_to_masked_array(self, val: Union[int, float], dtype: str) -> None:
        ser = Series([1, None, 3], dtype='Int64')
        result = ser + [1, None, val]
        expected = Series([2, None, 3 + val], dtype=dtype)
        tm.assert_series_equal(result, expected)
        result = [1, None, val] + ser
        tm.assert_series_equal(result, expected)

    def test_add_list_to_masked_array_boolean(self, request: pytest.FixtureRequest) -> None:
        warning = UserWarning if request.node.callspec.id == 'numexpr' and NUMEXPR_INSTALLED else None
        ser = Series([True, None, False], dtype='boolean')
        msg = 'operator is not supported by numexpr for the bool dtype'
        with tm.assert_produces_warning(warning, match=msg):
            result = ser + [True, None, True]
        expected = Series([True, None, True], dtype='boolean')
        tm.assert_series_equal(result, expected)
        with tm.assert_produces_warning(warning, match=msg):
            result = [True, None, True] + ser
        tm.assert_series_equal(result, expected)

class TestSeriesFlexComparison:

    @pytest.mark.parametrize('axis', [0, None, 'index'])
    def test_comparison_flex_basic(self, axis: Union[int, str, None], comparison_op: Callable[[Any, Any], bool]) -> None:
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))
        result = getattr(left, comparison_op.__name__)(right, axis=axis)
        expected = comparison_op(left, right)
        tm.assert_series_equal(result, expected)

    def test_comparison_bad_axis(self, comparison_op: Callable[[Any, Any], bool]) -> None:
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))
        msg = 'No axis named 1 for object type'
        with pytest.raises(ValueError, match=msg):
            getattr(left, comparison_op.__name__)(right, axis=1)

    @pytest.mark.parametrize('values, op', [([False, False, True, False], 'eq'), ([True, True, False, True], 'ne'), ([False, False, True, False], 'le'), ([False, False, False, False], 'lt'), ([False, True, True, False], 'ge'), ([False, True, False, False], 'gt')])
    def test_comparison_flex_alignment(self, values: List[bool], op: str) -> None:
        left = Series([1, 3, 2], index=list('abc'))
        right = Series([2, 2, 2], index=list('bcd'))
        result = getattr(left, op)(right)
        expected = Series(values, index=list('abcd'))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('values, op, fill_value', [([False, False, True, True], 'eq', 2), ([True, True, False, False], 'ne', 2), ([False, False, True, True], 'le', 0), ([False, False, False, True], 'lt', 0), ([True, True, True, False], 'ge', 0), ([True, True, False, False], 'gt', 0)])
    def test_comparison_flex_alignment_fill(self, values: List[bool], op: str, fill_value: int) -> None:
        left = Series([1, 3, 2], index=list('abc'))
        right = Series([2, 2