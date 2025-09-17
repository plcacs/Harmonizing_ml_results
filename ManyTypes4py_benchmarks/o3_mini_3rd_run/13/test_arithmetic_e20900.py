#!/usr/bin/env python3
from collections import deque
from datetime import datetime, timezone
from enum import Enum
import functools
import operator
import re
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pytest
from pandas.compat import HAS_PYARROW
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import _check_mixed_float, _check_mixed_int

@pytest.fixture
def simple_frame() -> DataFrame:
    """
    Fixture for simple 3x3 DataFrame

    Columns are ['one', 'two', 'three'], index is ['a', 'b', 'c'].

       one  two  three
    a  1.0  2.0    3.0
    b  4.0  5.0    6.0
    c  7.0  8.0    9.0
    """
    arr = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])
    return DataFrame(arr, columns=['one', 'two', 'three'], index=['a', 'b', 'c'])

@pytest.fixture(autouse=True, params=[0, 100], ids=['numexpr', 'python'])
def switch_numexpr_min_elements(request: Any, monkeypatch: Any) -> int:
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', request.param)
        yield request.param

class DummyElement:
    def __init__(self, value: Any, dtype: Union[str, np.dtype]) -> None:
        self.value = value
        self.dtype = np.dtype(dtype)

    def __array__(self, dtype: Union[None, Any] = None, copy: Union[None, bool] = None) -> np.ndarray:
        return np.array(self.value, dtype=self.dtype)

    def __str__(self) -> str:
        return f'DummyElement({self.value}, {self.dtype})'

    def __repr__(self) -> str:
        return str(self)

    def astype(self, dtype: Union[str, np.dtype], copy: bool = False) -> "DummyElement":
        self.dtype = np.dtype(dtype)
        return self

    def view(self, dtype: Union[str, np.dtype]) -> "DummyElement":
        return type(self)(self.value.view(dtype), dtype)

    def any(self, axis: Any = None) -> bool:
        return bool(self.value)

class TestFrameComparisons:
    def test_comparison_with_categorical_dtype(self) -> None:
        df: DataFrame = DataFrame({'A': ['foo', 'bar', 'baz']})
        exp: DataFrame = DataFrame({'A': [True, False, False]})
        res: DataFrame = df == 'foo'
        tm.assert_frame_equal(res, exp)
        df['A'] = df['A'].astype('category')
        res = df == 'foo'
        tm.assert_frame_equal(res, exp)

    def test_frame_in_list(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), columns=list('ABCD'))
        msg: str = 'The truth value of a DataFrame is ambiguous'
        with pytest.raises(ValueError, match=msg):
            _ = df in [None]

    @pytest.mark.parametrize('arg, arg2', [
        [{'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)},
         {'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}],
        [{'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)},
         {'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}],
        [{'a': pd.date_range('20010101', periods=10), 'b': pd.date_range('20010101', periods=10)},
         {'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}],
        [{'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)},
         {'a': pd.date_range('20010101', periods=10), 'b': pd.date_range('20010101', periods=10)}]
    ])
    def test_comparison_invalid(self, arg: Any, arg2: Any) -> None:
        x: DataFrame = DataFrame(arg)
        y: DataFrame = DataFrame(arg2)
        result = x == y
        expected = DataFrame({col: x[col] == y[col] for col in x.columns}, index=x.index, columns=x.columns)
        tm.assert_frame_equal(result, expected)
        result = x != y
        expected = DataFrame({col: x[col] != y[col] for col in x.columns}, index=x.index, columns=x.columns)
        tm.assert_frame_equal(result, expected)
        msgs: List[str] = ['Invalid comparison between dtype=datetime64\\[ns\\] and ndarray', 'invalid type promotion', "The DTypes <class 'numpy.dtype\\[.*\\]'> and <class 'numpy.dtype\\[.*\\]'> do not have a common DType."]
        msg: str = '|'.join(msgs)
        with pytest.raises(TypeError, match=msg):
            _ = x >= y
        with pytest.raises(TypeError, match=msg):
            _ = x > y
        with pytest.raises(TypeError, match=msg):
            _ = x < y
        with pytest.raises(TypeError, match=msg):
            _ = x <= y

    @pytest.mark.parametrize('left, right', [
        ('gt', 'lt'), ('lt', 'gt'), ('ge', 'le'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')
    ])
    def test_timestamp_compare(self, left: str, right: str) -> None:
        df: DataFrame = DataFrame({
            'dates1': pd.date_range('20010101', periods=10),
            'dates2': pd.date_range('20010102', periods=10),
            'intcol': np.random.default_rng(2).integers(1000000000, size=10),
            'floatcol': np.random.default_rng(2).standard_normal(10),
            'stringcol': [chr(100 + i) for i in range(10)]
        })
        df.loc[np.random.default_rng(2).random(len(df)) > 0.5, 'dates2'] = pd.NaT
        left_f = getattr(operator, left)
        right_f = getattr(operator, right)
        if left in ['eq', 'ne']:
            expected = left_f(df, pd.Timestamp('20010109'))
            result = right_f(pd.Timestamp('20010109'), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg: str = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'Timestamp'"
            with pytest.raises(TypeError, match=msg):
                _ = left_f(df, pd.Timestamp('20010109'))
            with pytest.raises(TypeError, match=msg):
                _ = right_f(pd.Timestamp('20010109'), df)
        if left in ['eq', 'ne']:
            expected = left_f(df, pd.Timestamp('nat'))
            result = right_f(pd.Timestamp('nat'), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'NaTType'"
            with pytest.raises(TypeError, match=msg):
                _ = left_f(df, pd.Timestamp('nat'))
            with pytest.raises(TypeError, match=msg):
                _ = right_f(pd.Timestamp('nat'), df)

    def test_mixed_comparison(self) -> None:
        df: DataFrame = DataFrame([['1989-08-01', 1], ['1989-08-01', 2]])
        other: DataFrame = DataFrame([['a', 'b'], ['c', 'd']])
        result: DataFrame = df == other
        assert not result.any().any()
        result = df != other
        assert result.all().all()

    def test_df_boolean_comparison_error(self) -> None:
        df: DataFrame = DataFrame(np.arange(6).reshape((3, 2)))
        expected: DataFrame = DataFrame([[False, False], [True, False], [False, False]])
        result: DataFrame = df == (2, 2)
        tm.assert_frame_equal(result, expected)
        result = df == [2, 2]
        tm.assert_frame_equal(result, expected)

    def test_df_float_none_comparison(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((8, 3)), index=range(8), columns=['A', 'B', 'C'])
        result = df.__eq__(None)
        assert not result.any().any()

    def test_df_string_comparison(self) -> None:
        df: DataFrame = DataFrame([{'a': 1, 'b': 'foo'}, {'a': 2, 'b': 'bar'}])
        mask_a = df.a > 1
        tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
        tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])
        mask_b = df.b == 'foo'
        tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
        tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])

class TestFrameFlexComparisons:
    def test_bool_flex_frame(self, comparison_op: Any) -> None:
        data: np.ndarray = np.random.default_rng(2).standard_normal((5, 3))
        other_data: np.ndarray = np.random.default_rng(2).standard_normal((5, 3))
        df: DataFrame = DataFrame(data)
        other: DataFrame = DataFrame(other_data)
        ndim_5: np.ndarray = np.ones(df.shape + (1, 3))
        assert df.eq(df).values.all()
        assert not df.ne(df).values.any()
        f = getattr(df, comparison_op.__name__)
        o = comparison_op
        tm.assert_frame_equal(f(other), o(df, other))
        part_o = other.loc[3:, 1:].copy()
        rs = f(part_o)
        xp = o(df, part_o.reindex(index=df.index, columns=df.columns))
        tm.assert_frame_equal(rs, xp)
        tm.assert_frame_equal(f(other.values), o(df, other.values))
        tm.assert_frame_equal(f(0), o(df, 0))
        msg: str = 'Unable to coerce to Series/DataFrame'
        tm.assert_frame_equal(f(np.nan), o(df, np.nan))
        with pytest.raises(ValueError, match=msg):
            _ = f(ndim_5)

    @pytest.mark.parametrize('box', [np.array, Series])
    def test_bool_flex_series(self, box: Callable[[Any], Union[np.ndarray, Series]]) -> None:
        data: np.ndarray = np.random.default_rng(2).standard_normal((5, 3))
        df: DataFrame = DataFrame(data)
        idx_ser = box(np.random.default_rng(2).standard_normal(5))
        col_ser = box(np.random.default_rng(2).standard_normal(3))
        idx_eq = df.eq(idx_ser, axis=0)
        col_eq = df.eq(col_ser)
        idx_ne = df.ne(idx_ser, axis=0)
        col_ne = df.ne(col_ser)
        tm.assert_frame_equal(col_eq, df == Series(col_ser))
        tm.assert_frame_equal(col_eq, -col_ne)
        tm.assert_frame_equal(idx_eq, -idx_ne)
        tm.assert_frame_equal(idx_eq, df.T.eq(idx_ser).T)
        tm.assert_frame_equal(col_eq, df.eq(list(col_ser)))
        tm.assert_frame_equal(idx_eq, df.eq(Series(idx_ser), axis=0))
        tm.assert_frame_equal(idx_eq, df.eq(list(idx_ser), axis=0))
        idx_gt = df.gt(idx_ser, axis=0)
        col_gt = df.gt(col_ser)
        idx_le = df.le(idx_ser, axis=0)
        col_le = df.le(col_ser)
        tm.assert_frame_equal(col_gt, df > Series(col_ser))
        tm.assert_frame_equal(col_gt, -col_le)
        tm.assert_frame_equal(idx_gt, -idx_le)
        tm.assert_frame_equal(idx_gt, df.T.gt(idx_ser).T)
        idx_ge = df.ge(idx_ser, axis=0)
        col_ge = df.ge(col_ser)
        idx_lt = df.lt(idx_ser, axis=0)
        col_lt = df.lt(col_ser)
        tm.assert_frame_equal(col_ge, df >= Series(col_ser))
        tm.assert_frame_equal(col_ge, -col_lt)
        tm.assert_frame_equal(idx_ge, -idx_lt)
        tm.assert_frame_equal(idx_ge, df.T.ge(idx_ser).T)
        idx_ser = Series(np.random.default_rng(2).standard_normal(5))
        col_ser = Series(np.random.default_rng(2).standard_normal(3))

    def test_bool_flex_frame_na(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df.loc[0, 0] = np.nan
        rs = df.eq(df)
        assert not rs.loc[0, 0]
        rs = df.ne(df)
        assert rs.loc[0, 0]
        rs = df.gt(df)
        assert not rs.loc[0, 0]
        rs = df.lt(df)
        assert not rs.loc[0, 0]
        rs = df.ge(df)
        assert not rs.loc[0, 0]
        rs = df.le(df)
        assert not rs.loc[0, 0]

    def test_bool_flex_frame_complex_dtype(self) -> None:
        arr: np.ndarray = np.array([np.nan, 1, 6, np.nan])
        arr2: np.ndarray = np.array([2j, np.nan, 7, None])
        df: DataFrame = DataFrame({'a': arr})
        df2: DataFrame = DataFrame({'a': arr2})
        msg: str = '|'.join(["'>' not supported between instances of '.*' and 'complex'", 'unorderable types: .*complex\\(\\)'])
        with pytest.raises(TypeError, match=msg):
            _ = df.gt(df2)
        with pytest.raises(TypeError, match=msg):
            _ = df['a'].gt(df2['a'])
        with pytest.raises(TypeError, match=msg):
            _ = df.values > df2.values
        rs = df.ne(df2)
        assert rs.values.all()
        arr3: np.ndarray = np.array([2j, np.nan, None])
        df3: DataFrame = DataFrame({'a': arr3})
        with pytest.raises(TypeError, match=msg):
            _ = df3.gt(2j)
        with pytest.raises(TypeError, match=msg):
            _ = df3['a'].gt(2j)
        with pytest.raises(TypeError, match=msg):
            _ = df3.values > 2j

    def test_bool_flex_frame_object_dtype(self) -> None:
        df1: DataFrame = DataFrame({'col': ['foo', np.nan, 'bar']}, dtype=object)
        df2: DataFrame = DataFrame({'col': ['foo', datetime.now(), 'bar']}, dtype=object)
        result = df1.ne(df2)
        exp: DataFrame = DataFrame({'col': [False, True, False]})
        tm.assert_frame_equal(result, exp)

    def test_flex_comparison_nat(self) -> None:
        df: DataFrame = DataFrame([pd.NaT])
        result = df == pd.NaT
        assert result.iloc[0, 0].item() is False
        result = df.eq(pd.NaT)
        assert result.iloc[0, 0].item() is False
        result = df != pd.NaT
        assert result.iloc[0, 0].item() is True
        result = df.ne(pd.NaT)
        assert result.iloc[0, 0].item() is True

    def test_df_flex_cmp_constant_return_types(self, comparison_op: Any) -> None:
        df: DataFrame = DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        const = 2
        result = getattr(df, comparison_op.__name__)(const).dtypes.value_counts()
        tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))

    def test_df_flex_cmp_constant_return_types_empty(self, comparison_op: Any) -> None:
        df: DataFrame = DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        const = 2
        empty: DataFrame = df.iloc[:0]
        result = getattr(empty, comparison_op.__name__)(const).dtypes.value_counts()
        tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))

    def test_df_flex_cmp_ea_dtype_with_ndarray_series(self) -> None:
        ii: pd.IntervalIndex = pd.IntervalIndex.from_breaks([1, 2, 3])
        df: DataFrame = DataFrame({'A': ii, 'B': ii})
        ser: Series = Series([0, 0])
        res = df.eq(ser, axis=0)
        expected: DataFrame = DataFrame({'A': [False, False], 'B': [False, False]})
        tm.assert_frame_equal(res, expected)
        ser2: Series = Series([1, 2], index=['A', 'B'])
        res2 = df.eq(ser2, axis=1)
        tm.assert_frame_equal(res2, expected)

class TestFrameFlexArithmetic:
    def test_floordiv_axis0(self) -> None:
        arr: np.ndarray = np.arange(3)
        ser: Series = Series(arr)
        df: DataFrame = DataFrame({'A': ser, 'B': ser})
        result: DataFrame = df.floordiv(ser, axis=0)
        expected: DataFrame = DataFrame({col: df[col] // ser for col in df.columns})
        tm.assert_frame_equal(result, expected)
        result2: DataFrame = df.floordiv(ser.values, axis=0)
        tm.assert_frame_equal(result2, expected)

    def test_df_add_td64_columnwise(self) -> None:
        dti = pd.date_range('2016-01-01', periods=10)
        tdi = pd.timedelta_range('1', periods=10)
        tser: Series = Series(tdi)
        df: DataFrame = DataFrame({0: dti, 1: tdi})
        result: DataFrame = df.add(tser, axis=0)
        expected: DataFrame = DataFrame({0: dti + tdi, 1: tdi + tdi})
        tm.assert_frame_equal(result, expected)

    def test_df_add_flex_filled_mixed_dtypes(self) -> None:
        dti = pd.date_range('2016-01-01', periods=3)
        ser: Series = Series(['1 Day', 'NaT', '2 Days'], dtype='timedelta64[ns]')
        df: DataFrame = DataFrame({'A': dti, 'B': ser})
        other: DataFrame = DataFrame({'A': ser, 'B': ser})
        fill = pd.Timedelta(days=1).to_timedelta64()
        result: DataFrame = df.add(other, fill_value=fill)
        expected: DataFrame = DataFrame({
            'A': Series(['2016-01-02', '2016-01-03', '2016-01-05'], dtype='datetime64[ns]'),
            'B': ser * 2
        })
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_frame(self, all_arithmetic_operators: Any, float_frame: DataFrame, mixed_float_frame: DataFrame) -> None:
        op: str = all_arithmetic_operators
        def f(x: Any, y: Any) -> Any:
            if op.startswith('__r'):
                return getattr(operator, op.replace('__r', '__'))(y, x)
            return getattr(operator, op)(x, y)
        result: DataFrame = getattr(float_frame, op)(2 * float_frame)
        expected: DataFrame = f(float_frame, 2 * float_frame)
        tm.assert_frame_equal(result, expected)
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={'C': None})

    @pytest.mark.parametrize('op', ['__add__', '__sub__', '__mul__'])
    def test_arith_flex_frame_mixed(self, op: str, int_frame: DataFrame, mixed_int_frame: DataFrame, mixed_float_frame: DataFrame, switch_numexpr_min_elements: int) -> None:
        f = getattr(operator, op)
        result: DataFrame = getattr(mixed_int_frame, op)(2 + mixed_int_frame)
        expected: DataFrame = f(mixed_int_frame, 2 + mixed_int_frame)
        dtype: Any = None
        if op in ['__sub__']:
            dtype = {'B': 'uint64', 'C': None}
        elif op in ['__add__', '__mul__']:
            dtype = {'C': None}
        if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
            dtype['A'] = (2 + mixed_int_frame)['A'].dtype
        tm.assert_frame_equal(result, expected)
        _check_mixed_int(result, dtype=dtype)
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={'C': None})
        result = getattr(int_frame, op)(2 * int_frame)
        expected = f(int_frame, 2 * int_frame)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dim', range(3, 6))
    def test_arith_flex_frame_raise(self, all_arithmetic_operators: Any, float_frame: DataFrame, dim: int) -> None:
        op: str = all_arithmetic_operators
        arr: np.ndarray = np.ones((1,) * dim)
        msg: str = 'Unable to coerce to Series/DataFrame'
        with pytest.raises(ValueError, match=msg):
            _ = getattr(float_frame, op)(arr)

    def test_arith_flex_frame_corner(self, float_frame: DataFrame) -> None:
        const_add: DataFrame = float_frame.add(1)
        tm.assert_frame_equal(const_add, float_frame + 1)
        result: DataFrame = float_frame.add(float_frame[:0])
        expected: DataFrame = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)
        result = float_frame[:0].add(float_frame)
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)
        with pytest.raises(NotImplementedError, match='fill_value'):
            _ = float_frame.add(float_frame.iloc[0], fill_value=3)
        with pytest.raises(NotImplementedError, match='fill_value'):
            _ = float_frame.add(float_frame.iloc[0], axis='index', fill_value=3)

    @pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'mod'])
    def test_arith_flex_series_ops(self, simple_frame: DataFrame, op: str) -> None:
        df: DataFrame = simple_frame
        row: Series = df.xs('a')
        col: Series = df['two']
        f = getattr(df, op)
        op_func = getattr(operator, op)
        tm.assert_frame_equal(f(row), op_func(df, row))
        tm.assert_frame_equal(f(col, axis=0), op_func(df.T, col).T)

    def test_arith_flex_series(self, simple_frame: DataFrame) -> None:
        df: DataFrame = simple_frame
        row: Series = df.xs('a')
        col: Series = df['two']
        tm.assert_frame_equal(df.add(row, axis=None), df + row)
        tm.assert_frame_equal(df.div(row), df / row)
        tm.assert_frame_equal(df.div(col, axis=0), (df.T / col).T)

    def test_arith_flex_series_broadcasting(self, any_real_numpy_dtype: str) -> None:
        df: DataFrame = DataFrame(np.arange(3 * 2).reshape((3, 2)), dtype=any_real_numpy_dtype)
        expected: DataFrame = DataFrame([[np.nan, np.inf], [1.0, 1.5], [1.0, 1.25]])
        if any_real_numpy_dtype == 'float32':
            expected = expected.astype(any_real_numpy_dtype)
        result: DataFrame = df.div(df[0], axis='index')
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_zero_len_raises(self) -> None:
        ser_len0: Series = Series([], dtype=object)
        df_len0: DataFrame = DataFrame(columns=['A', 'B'])
        df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        with pytest.raises(NotImplementedError, match='fill_value'):
            _ = df.add(ser_len0, fill_value='E')
        with pytest.raises(NotImplementedError, match='fill_value'):
            _ = df_len0.sub(df['A'], axis=None, fill_value=3)

    def test_flex_add_scalar_fill_value(self) -> None:
        dat: np.ndarray = np.array([0, 1, np.nan, 3, 4, 5], dtype='float')
        df: DataFrame = DataFrame({'foo': dat}, index=range(6))
        exp: DataFrame = df.fillna(0).add(2)
        res: DataFrame = df.add(2, fill_value=0)
        tm.assert_frame_equal(res, exp)

    def test_sub_alignment_with_duplicate_index(self) -> None:
        df1: DataFrame = DataFrame([1, 2, 3, 4, 5], index=[1, 2, 1, 2, 3])
        df2: DataFrame = DataFrame([1, 2, 3], index=[1, 2, 3])
        expected: DataFrame = DataFrame([0, 2, 0, 2, 2], index=[1, 1, 2, 2, 3])
        result: DataFrame = df1.sub(df2)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op', ['__add__', '__mul__', '__sub__', '__truediv__'])
    def test_arithmetic_with_duplicate_columns(self, op: str) -> None:
        df: DataFrame = DataFrame({'A': np.arange(10), 'B': np.random.default_rng(2).random(10)})
        expected: DataFrame = getattr(df, op)(df)
        expected.columns = ['A', 'A']
        df.columns = ['A', 'A']
        result: DataFrame = getattr(df, op)(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('level', [0, None])
    def test_broadcast_multiindex(self, level: Union[int, None]) -> None:
        df1: DataFrame = DataFrame({'A': [0, 1, 2], 'B': [1, 2, 3]})
        df1.columns = df1.columns.set_names('L1')
        df2: DataFrame = DataFrame({('A', 'C'): [0, 0, 0], ('A', 'D'): [0, 0, 0]})
        df2.columns = df2.columns.set_names(['L1', 'L2'])
        result: DataFrame = df1.add(df2, level=level)
        expected: DataFrame = DataFrame({('A', 'C'): [0, 1, 2], ('A', 'D'): [0, 1, 2]})
        expected.columns = expected.columns.set_names(['L1', 'L2'])
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations(self) -> None:
        df: DataFrame = DataFrame({
            2010: [1, 2, 3],
            2020: [3, 4, 5]
        }, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        series: Series = Series([0.4], index=MultiIndex.from_product([['b'], ['a']], names=['mod', 'scen']))
        expected: DataFrame = DataFrame({
            2010: [1.4, 2.4, 3.4],
            2020: [3.4, 4.4, 5.4]
        }, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        result: DataFrame = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_series_index_to_frame_index(self) -> None:
        df: DataFrame = DataFrame({2010: [1], 2020: [3]}, index=MultiIndex.from_product([['a'], ['b']], names=['scen', 'mod']))
        series: Series = Series([10.0, 20.0, 30.0], index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        expected: DataFrame = DataFrame({2010: [11.0, 21, 31.0], 2020: [13.0, 23.0, 33.0]}, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        result: DataFrame = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_no_align(self) -> None:
        df: DataFrame = DataFrame({2010: [1, 2, 3], 2020: [3, 4, 5]}, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        series: Series = Series([0.4], index=MultiIndex.from_product([['c'], ['a']], names=['mod', 'scen']))
        expected: DataFrame = DataFrame({2010: np.nan, 2020: np.nan}, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'b', 2), ('a', 'c', np.nan)], names=['scen', 'mod', 'id']))
        result: DataFrame = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_part_align(self) -> None:
        df: DataFrame = DataFrame({2010: [1, 2, 3], 2020: [3, 4, 5]}, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'c', 2)], names=['scen', 'mod', 'id']))
        series: Series = Series([0.4], index=MultiIndex.from_product([['b'], ['a']], names=['mod', 'scen']))
        expected: DataFrame = DataFrame({2010: [1.4, 2.4, np.nan], 2020: [3.4, 4.4, np.nan]}, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'c', 2)], names=['scen', 'mod', 'id']))
        result: DataFrame = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

class TestFrameArithmetic:
    def test_td64_op_nat_casting(self) -> None:
        ser: Series = Series(['NaT', 'NaT'], dtype='timedelta64[ns]')
        df: DataFrame = DataFrame([[1, 2], [3, 4]])
        result: DataFrame = df * ser
        expected: DataFrame = DataFrame({0: ser, 1: ser})
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_rowlike_broadcasts(self) -> None:
        arr: np.ndarray = np.arange(6).reshape(3, 2)
        df: DataFrame = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        rowlike: np.ndarray = arr[[1], :]
        assert rowlike.shape == (1, df.shape[1])
        expected: DataFrame = DataFrame([[2, 4], [4, 6], [6, 8]], columns=df.columns, index=df.index, dtype=arr.dtype)
        result: DataFrame = df + rowlike
        tm.assert_frame_equal(result, expected)
        result = rowlike + df
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_collike_broadcasts(self) -> None:
        arr: np.ndarray = np.arange(6).reshape(3, 2)
        df: DataFrame = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        collike: np.ndarray = arr[:, [1]]
        assert collike.shape == (df.shape[0], 1)
        expected: DataFrame = DataFrame([[1, 2], [5, 6], [9, 10]], columns=df.columns, index=df.index, dtype=arr.dtype)
        result: DataFrame = df + collike
        tm.assert_frame_equal(result, expected)
        result = collike + df
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_rowlike_broadcasts(self, request: Any, all_arithmetic_operators: Any) -> None:
        opname: str = all_arithmetic_operators
        arr: np.ndarray = np.arange(6).reshape(3, 2)
        df: DataFrame = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        rowlike: np.ndarray = arr[[1], :]
        assert rowlike.shape == (1, df.shape[1])
        exvals: List[Any] = [
            getattr(df.loc['A'], opname)(rowlike.squeeze()),
            getattr(df.loc['B'], opname)(rowlike.squeeze()),
            getattr(df.loc['C'], opname)(rowlike.squeeze())
        ]
        expected: DataFrame = DataFrame(exvals, columns=df.columns, index=df.index)
        result: DataFrame = getattr(df, opname)(rowlike)
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_collike_broadcasts(self, request: Any, all_arithmetic_operators: Any) -> None:
        opname: str = all_arithmetic_operators
        arr: np.ndarray = np.arange(6).reshape(3, 2)
        df: DataFrame = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        collike: np.ndarray = arr[:, [1]]
        assert collike.shape == (df.shape[0], 1)
        exvals: dict = {
            True: getattr(df[True], opname)(collike.squeeze()),
            False: getattr(df[False], opname)(collike.squeeze())
        }
        dtype: Any = None
        if opname in ['__rmod__', '__rfloordiv__']:
            dtype = np.common_type(*(x.values for x in exvals.values()))
        expected: DataFrame = DataFrame(exvals, columns=df.columns, index=df.index, dtype=dtype)
        result: DataFrame = getattr(df, opname)(collike)
        tm.assert_frame_equal(result, expected)

    def test_df_bool_mul_int(self) -> None:
        df: DataFrame = DataFrame([[False, True], [False, False]])
        result: DataFrame = df * 1
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == 'i').all()
        result = 1 * df
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == 'i').all()

    def test_arith_mixed(self) -> None:
        left: DataFrame = DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3]})
        result: DataFrame = left + left
        expected: DataFrame = DataFrame({'A': ['aa', 'bb', 'cc'], 'B': [2, 4, 6]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('col', ['A', 'B'])
    def test_arith_getitem_commute(self, all_arithmetic_functions: Callable[[Any, Any], Any], col: str) -> None:
        df: DataFrame = DataFrame({'A': [1.1, 3.3], 'B': [2.5, -3.9]})
        result: Series = all_arithmetic_functions(df, 1)[col]
        expected: Series = all_arithmetic_functions(df[col], 1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('values', [
        [1, 2], (1, 2), np.array([1, 2]), range(1, 3), deque([1, 2])
    ])
    def test_arith_alignment_non_pandas_object(self, values: Any) -> None:
        df: DataFrame = DataFrame({'A': [1, 1], 'B': [1, 1]})
        expected: DataFrame = DataFrame({'A': [2, 2], 'B': [3, 3]})
        result: DataFrame = df + values
        tm.assert_frame_equal(result, expected)

    def test_arith_non_pandas_object(self) -> None:
        df: DataFrame = DataFrame(np.arange(1, 10, dtype='f8').reshape(3, 3),
                                  columns=['one', 'two', 'three'],
                                  index=['a', 'b', 'c'])
        val1: np.ndarray = df.xs('a').values
        added: DataFrame = DataFrame(df.values + val1, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val1, added)
        added = DataFrame((df.values.T + val1).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val1, axis=0), added)
        val2: List[Any] = list(df['two'])
        added = DataFrame(df.values + val2, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val2, added)
        added = DataFrame((df.values.T + val2).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val2, axis='index'), added)
        val3: np.ndarray = np.random.default_rng(2).random(df.shape)
        added = DataFrame(df.values + val3, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val3), added)

    def test_operations_with_interval_categories_index(self, all_arithmetic_operators: Any) -> None:
        op = all_arithmetic_operators
        ind: pd.CategoricalIndex = pd.CategoricalIndex(pd.interval_range(start=0.0, end=2.0))
        data: List[int] = [1, 2]
        df: DataFrame = DataFrame([data], columns=ind)
        num: int = 10
        result: DataFrame = getattr(df, op)(num)
        expected: DataFrame = DataFrame([[getattr(n, op)(num) for n in data]], columns=ind)
        tm.assert_frame_equal(result, expected)

    def test_frame_with_frame_reindex(self) -> None:
        df: DataFrame = DataFrame({
            'foo': [pd.Timestamp('2019'), pd.Timestamp('2020')],
            'bar': [pd.Timestamp('2018'), pd.Timestamp('2021')]
        }, columns=['foo', 'bar'], dtype='M8[ns]')
        df2: DataFrame = df[['foo']]
        result: DataFrame = df - df2
        expected: DataFrame = DataFrame({'foo': [pd.Timedelta(0), pd.Timedelta(0)], 'bar': [np.nan, np.nan]}, columns=['bar', 'foo'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value, dtype', [
        (1, 'i8'),
        (1.0, 'f8'),
        (2 ** 63, 'f8'),
        (1j, 'complex128'),
        (2 ** 63, 'complex128'),
        (True, 'bool'),
        (np.timedelta64(20, 'ns'), '<m8[ns]'),
        (np.datetime64(20, 'ns'), '<M8[ns]')
    ])
    @pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv, operator.mod, operator.pow], ids=lambda x: x.__name__)
    def test_binop_other(self, op: Callable[[Any, Any], Any], value: Any, dtype: str, switch_numexpr_min_elements: int) -> None:
        skip = {(operator.truediv, 'bool'), (operator.pow, 'bool'), (operator.add, 'bool'), (operator.mul, 'bool')}
        elem: DummyElement = DummyElement(value, dtype)
        df: DataFrame = DataFrame({'A': [elem.value, elem.value]}, dtype=elem.dtype)
        invalid = {
            (operator.pow, '<M8[ns]'),
            (operator.mod, '<M8[ns]'),
            (operator.truediv, '<M8[ns]'),
            (operator.mul, '<M8[ns]'),
            (operator.add, '<M8[ns]'),
            (operator.pow, '<m8[ns]'),
            (operator.mul, '<m8[ns]'),
            (operator.sub, 'bool'),
            (operator.mod, 'complex128')
        }
        if (op, dtype) in invalid:
            warn: Any = None
            if dtype == '<M8[ns]' and op == operator.add or (dtype == '<m8[ns]' and op == operator.mul):
                msg = None
            elif dtype == 'complex128':
                msg = "ufunc 'remainder' not supported for the input types"
            elif op is operator.sub:
                msg = 'numpy boolean subtract, the `-` operator, is '
                if dtype == 'bool' and expr.USE_NUMEXPR and (switch_numexpr_min_elements == 0):
                    warn = UserWarning
            else:
                msg = f'cannot perform __{op.__name__}__ with this index type: (DatetimeArray|TimedeltaArray)'
            with pytest.raises(TypeError, match=msg):
                with tm.assert_produces_warning(warn, match='evaluating in Python'):
                    _ = op(df, elem.value)
        elif (op, dtype) in skip:
            if op in [operator.add, operator.mul]:
                if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                    warn = UserWarning
                else:
                    warn = None
                with tm.assert_produces_warning(warn, match='evaluating in Python'):
                    _ = op(df, elem.value)
            else:
                msg = "operator '.*' not implemented for .* dtypes"
                with pytest.raises(NotImplementedError, match=msg):
                    _ = op(df, elem.value)
        else:
            with tm.assert_produces_warning(None):
                result = op(df, elem.value).dtypes
                expected = op(df, value).dtypes
            tm.assert_series_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes(self) -> None:
        midx: MultiIndex = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2: MultiIndex = MultiIndex.from_arrays([Series([1, 2], dtype='Int8'), Series([3, 4])])
        left: DataFrame = DataFrame([[1, 2], [3, 4]], columns=midx)
        right: DataFrame = DataFrame([[1, 2], [3, 4]], columns=midx2)
        result: DataFrame = left - right
        expected: DataFrame = DataFrame([[0, 0], [0, 0]], columns=midx)
        tm.assert_frame_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes_different_order(self) -> None:
        midx: MultiIndex = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2: MultiIndex = MultiIndex.from_arrays([Series([2, 1], dtype='Int8'), Series([4, 3])])
        left: DataFrame = DataFrame([[1, 2], [3, 4]], columns=midx)
        right: DataFrame = DataFrame([[1, 2], [3, 4]], columns=midx2)
        result: DataFrame = left - right
        expected: DataFrame = DataFrame([[-1, 1], [-1, 1]], columns=midx)
        tm.assert_frame_equal(result, expected)

def test_frame_with_zero_len_series_corner_cases() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal(6).reshape(3, 2), columns=['A', 'B'])
    ser: Series = Series(dtype=np.float64)
    result: DataFrame = df + ser
    expected: DataFrame = DataFrame(df.values * np.nan, columns=df.columns)
    tm.assert_frame_equal(result, expected)
    with pytest.raises(ValueError, match='not aligned'):
        _ = df == ser
    df2: DataFrame = DataFrame(df.values.view('M8[ns]'), columns=df.columns)
    with pytest.raises(ValueError, match='not aligned'):
        _ = df2 == ser

def test_zero_len_frame_with_series_corner_cases() -> None:
    df: DataFrame = DataFrame(columns=['A', 'B'], dtype=np.float64)
    ser: Series = Series([1, 2], index=['A', 'B'])
    result: DataFrame = df + ser
    expected: DataFrame = df
    tm.assert_frame_equal(result, expected)

def test_frame_single_columns_object_sum_axis_1() -> None:
    data: dict = {'One': Series(['A', 1.2, np.nan])}
    df: DataFrame = DataFrame(data)
    result: Series = df.sum(axis=1)
    expected: Series = Series(['A', 1.2, 0])
    tm.assert_series_equal(result, expected)

class TestFrameArithmeticUnsorted:
    def test_frame_add_tz_mismatch_converts_to_utc(self) -> None:
        rng = pd.date_range('1/1/2011', periods=10, freq='h', tz='US/Eastern')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal(len(rng)), index=rng, columns=['a'])
        df_moscow: DataFrame = df.tz_convert('Europe/Moscow')
        result: DataFrame = df + df_moscow
        assert result.index.tz is timezone.utc
        result = df_moscow + df
        assert result.index.tz is timezone.utc

    def test_align_frame(self) -> None:
        rng = pd.period_range('1/1/2000', '1/1/2010', freq='Y')
        ts: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng)
        result: DataFrame = ts + ts[::2]
        expected: DataFrame = ts + ts
        expected.iloc[1::2] = np.nan
        tm.assert_frame_equal(result, expected)
        half: DataFrame = ts[::2]
        result = ts + half.take(np.random.default_rng(2).permutation(len(half)))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv])
    def test_operators_none_as_na(self, op: Callable[[Any, Any], Any]) -> None:
        df: DataFrame = DataFrame({'col1': [2, 5.0, 123, None], 'col2': [1, 2, 3, 4]}, dtype=object)
        filled: DataFrame = df.fillna(np.nan)
        result: DataFrame = op(df, 3)
        expected: DataFrame = op(filled, 3).astype(object)
        expected[pd.isna(expected)] = np.nan
        tm.assert_frame_equal(result, expected)
        result = op(df, df)
        expected = op(filled, filled).astype(object)
        expected[pd.isna(expected)] = np.nan
        tm.assert_frame_equal(result, expected)
        result = op(df, df.fillna(7))
        tm.assert_frame_equal(result, expected)
        result = op(df.fillna(7), df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op,res', [('__eq__', False), ('__ne__', True)])
    @pytest.mark.filterwarnings('ignore:elementwise:FutureWarning')
    def test_logical_typeerror_with_non_valid(self, op: str, res: bool, float_frame: DataFrame) -> None:
        result: DataFrame = getattr(float_frame, op)('foo')
        assert bool(result.all().all()) is res

    @pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'div', 'truediv'])
    def test_binary_ops_align(self, op: str) -> None:
        index: MultiIndex = MultiIndex.from_product([list('abc'), ['one', 'two', 'three'], [1, 2, 3]], names=['first', 'second', 'third'])
        df: DataFrame = DataFrame(np.arange(27 * 3).reshape(27, 3), index=index, columns=['value1', 'value2', 'value3']).sort_index()
        idx = pd.IndexSlice
        opa = getattr(operator, op, None)
        if opa is None:
            return
        x: Series = Series([1.0, 10.0, 100.0], index=[1, 2, 3])
        result: DataFrame = getattr(df, op)(x, level='third', axis=0)
        expected: DataFrame = pd.concat([opa(df.loc[idx[:, :, i], :], v) for i, v in x.items()]).sort_index()
        tm.assert_frame_equal(result, expected)
        x = Series([1.0, 10.0], index=['two', 'three'])
        result = getattr(df, op)(x, level='second', axis=0)
        expected = pd.concat([opa(df.loc[idx[:, i], :], v) for i, v in x.items()]).reindex_like(df).sort_index()
        tm.assert_frame_equal(result, expected)

    def test_binary_ops_align_series_dataframe(self) -> None:
        midx: MultiIndex = MultiIndex.from_product([['A', 'B'], ['a', 'b']])
        df: DataFrame = DataFrame(np.ones((2, 4), dtype='int64'), columns=midx)
        s: Series = Series({'a': 1, 'b': 2})
        df2: DataFrame = df.copy()
        df2.columns.names = ['lvl0', 'lvl1']
        s2: Series = s.copy()
        s2.index.name = 'lvl1'
        res1: DataFrame = df.mul(s, axis=1, level=1)
        res2: DataFrame = df.mul(s2, axis=1, level=1)
        res3: DataFrame = df2.mul(s, axis=1, level=1)
        res4: DataFrame = df2.mul(s2, axis=1, level=1)
        res5: DataFrame = df2.mul(s, axis=1, level='lvl1')
        res6: DataFrame = df2.mul(s2, axis=1, level='lvl1')
        exp: DataFrame = DataFrame(np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype='int64'), columns=midx)
        for res in [res1, res2]:
            tm.assert_frame_equal(res, exp)
        exp.columns.names = ['lvl0', 'lvl1']
        for res in [res3, res4, res5, res6]:
            tm.assert_frame_equal(res, exp)

    def test_add_with_dti_mismatched_tzs(self) -> None:
        base = pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], tz='UTC')
        idx1 = base.tz_convert('Asia/Tokyo')[:2]
        idx2 = base.tz_convert('US/Eastern')[1:]
        df1: DataFrame = DataFrame({'A': [1, 2]}, index=idx1)
        df2: DataFrame = DataFrame({'A': [1, 1]}, index=idx2)
        exp: DataFrame = DataFrame({'A': [np.nan, 3, np.nan]}, index=base)
        tm.assert_frame_equal(df1 + df2, exp)

    def test_combineFrame(self, float_frame: DataFrame, mixed_float_frame: DataFrame, mixed_int_frame: DataFrame) -> None:
        frame_copy: DataFrame = float_frame.reindex(float_frame.index[::2])
        del frame_copy['D']
        frame_copy.loc[:frame_copy.index[4], 'C'] = np.nan
        added: DataFrame = float_frame + frame_copy
        indexer: Index = added['A'].dropna().index
        exp: Series = (float_frame['A'] * 2).copy()
        tm.assert_series_equal(added['A'].dropna(), exp.loc[indexer])
        exp.loc[~exp.index.isin(indexer)] = np.nan
        tm.assert_series_equal(added['A'], exp.loc[added['A'].index])
        assert np.isnan(added['C'].reindex(frame_copy.index)[:5]).all()
        assert np.isnan(added['D']).all()
        self_added: DataFrame = float_frame + float_frame
        tm.assert_index_equal(self_added.index, float_frame.index)
        added_rev: DataFrame = frame_copy + float_frame
        assert np.isnan(added['D']).all()
        assert np.isnan(added_rev['D']).all()
        plus_empty: DataFrame = float_frame + DataFrame()
        assert np.isnan(plus_empty.values).all()
        empty_plus: DataFrame = DataFrame() + float_frame
        assert np.isnan(empty_plus.values).all()
        empty_empty: DataFrame = DataFrame() + DataFrame()
        assert empty_empty.empty
        reverse: DataFrame = float_frame.reindex(columns=float_frame.columns[::-1])
        tm.assert_frame_equal(reverse + float_frame, float_frame * 2)
        added = float_frame + mixed_float_frame
        _check_mixed_float(added, dtype='float64')
        added = mixed_float_frame + float_frame
        _check_mixed_float(added, dtype='float64')
        added = mixed_float_frame + mixed_float_frame
        _check_mixed_float(added, dtype={'C': None})
        added = float_frame + mixed_int_frame
        _check_mixed_float(added, dtype='float64')

    def test_combine_series(self, float_frame: DataFrame, mixed_float_frame: DataFrame, mixed_int_frame: DataFrame) -> None:
        series: Series = float_frame.xs(float_frame.index[0])
        added: DataFrame = float_frame + series
        for key, s in added.items():
            tm.assert_series_equal(s, float_frame[key] + series[key])
        larger_series: Series = Series(series.to_dict() | {'E': 1})
        larger_added: DataFrame = float_frame + larger_series
        for key, s in float_frame.items():
            tm.assert_series_equal(larger_added[key], s + series[key])
        assert 'E' in larger_added
        assert np.isnan(larger_added['E']).all()
        added = mixed_float_frame + series
        assert np.all(added.dtypes == series.dtype)
        added = mixed_float_frame + series.astype('float32')
        _check_mixed_float(added, dtype={'C': None})
        added = mixed_float_frame + series.astype('float16')
        _check_mixed_float(added, dtype={'C': None})
        added = mixed_int_frame + (100 * series).astype('int64')
        _check_mixed_int(added, dtype={'A': 'int64', 'B': 'float64', 'C': 'int64', 'D': 'int64'})
        added = mixed_int_frame + (100 * series).astype('int32')
        _check_mixed_int(added, dtype={'A': 'int32', 'B': 'float64', 'C': 'int32', 'D': 'int64'})

    def test_combine_timeseries(self, datetime_frame: DataFrame) -> None:
        ts: Series = datetime_frame['A']
        added: DataFrame = datetime_frame.add(ts, axis='index')
        for key, col in datetime_frame.items():
            result: Series = col + ts
            tm.assert_series_equal(added[key], result, check_names=False)
            assert added[key].name == key
            if col.name == ts.name:
                assert result.name == 'A'
            else:
                assert result.name is None
        smaller_frame: DataFrame = datetime_frame[:-5]
        smaller_added: DataFrame = smaller_frame.add(ts, axis='index')
        tm.assert_index_equal(smaller_added.index, datetime_frame.index)
        smaller_ts: Series = ts[:-5]
        smaller_added2: DataFrame = datetime_frame.add(smaller_ts, axis='index')
        tm.assert_frame_equal(smaller_added, smaller_added2)
        result = datetime_frame.add(ts[:0], axis='index')
        expected: DataFrame = DataFrame(np.nan, index=datetime_frame.index, columns=datetime_frame.columns)
        tm.assert_frame_equal(result, expected)
        result = datetime_frame[:0].add(ts, axis='index')
        expected = DataFrame(np.nan, index=datetime_frame.index, columns=datetime_frame.columns)
        tm.assert_frame_equal(result, expected)
        frame: DataFrame = datetime_frame[:1].reindex(columns=[])
        result = frame.mul(ts, axis='index')
        assert len(result) == len(ts)

    def test_combineFunc(self, float_frame: DataFrame, mixed_float_frame: DataFrame) -> None:
        result: DataFrame = float_frame * 2
        tm.assert_numpy_array_equal(result.values, float_frame.values * 2)
        result = mixed_float_frame * 2
        for c, s in result.items():
            tm.assert_numpy_array_equal(s.values, mixed_float_frame[c].values * 2)
        _check_mixed_float(result, dtype={'C': None})
        result = DataFrame() * 2
        assert result.index.equals(DataFrame().index)
        assert len(result.columns) == 0

    @pytest.mark.parametrize('func', [operator.eq, operator.ne, operator.lt, operator.gt, operator.ge, operator.le])
    def test_comparisons(self, simple_frame: DataFrame, float_frame: DataFrame, func: Callable[[Any, Any], Any]) -> None:
        df1: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((30, 4)), columns=Index(list('ABCD'), dtype=object), index=pd.date_range('2000-01-01', periods=30, freq='B'))
        df2: DataFrame = df1.copy()
        row: Series = simple_frame.xs('a')
        ndim_5: np.ndarray = np.ones(df1.shape + (1, 1, 1))
        result: DataFrame = func(df1, df2)
        tm.assert_numpy_array_equal(result.values, func(df1.values, df2.values))
        msg: str = 'Unable to coerce to Series/DataFrame, dimension must be <= 2: (30, 4, 1, 1, 1)'
        with pytest.raises(ValueError, match=re.escape(msg)):
            _ = func(df1, ndim_5)
        result2: DataFrame = func(simple_frame, row)
        tm.assert_numpy_array_equal(result2.values, func(simple_frame.values, row.values))
        result3: DataFrame = func(float_frame, 0)
        tm.assert_numpy_array_equal(result3.values, func(float_frame.values, 0))
        msg = 'Can only compare identically-labeled \\(both index and columns\\) DataFrame objects'
        with pytest.raises(ValueError, match=msg):
            _ = func(simple_frame, simple_frame[:2])

    def test_strings_to_numbers_comparisons_raises(self, compare_operators_no_eq_ne: Any) -> None:
        df: DataFrame = DataFrame({x: {'x': 'foo', 'y': 'bar', 'z': 'baz'} for x in ['a', 'b', 'c']})
        f = getattr(operator, compare_operators_no_eq_ne)
        msg: str = '|'.join(["'[<>]=?' not supported between instances of 'str' and 'int'", 'Invalid comparison between dtype=str and int'])
        with pytest.raises(TypeError, match=msg):
            _ = f(df, 0)

    def test_comparison_protected_from_errstate(self) -> None:
        missing_df: DataFrame = DataFrame(np.ones((10, 4), dtype=np.float64), columns=Index(list('ABCD'), dtype=object))
        missing_df.loc[missing_df.index[0], 'A'] = np.nan
        with np.errstate(invalid='ignore'):
            expected = missing_df.values < 0
        with np.errstate(invalid='raise'):
            result = (missing_df < 0).values
        tm.assert_numpy_array_equal(result, expected)

    def test_boolean_comparison(self) -> None:
        df: DataFrame = DataFrame(np.arange(6).reshape((3, 2)))
        b: np.ndarray = np.array([2, 2])
        b_r: np.ndarray = np.atleast_2d([2, 2])
        b_c: np.ndarray = b_r.T
        lst: List[int] = [2, 2, 2]
        tup: Tuple[int, int, int] = tuple(lst)
        expected: DataFrame = DataFrame([[False, False], [False, True], [True, True]])
        result: DataFrame = df > b
        tm.assert_frame_equal(result, expected)
        result = df.values > b
        tm.assert_numpy_array_equal(result, expected.values)
        msg1d: str = 'Unable to coerce to Series, length must be 2: given 3'
        msg2d: str = 'Unable to coerce to DataFrame, shape must be'
        msg2db: str = 'operands could not be broadcast together with shapes'
        with pytest.raises(ValueError, match=msg1d):
            _ = df > lst
        with pytest.raises(ValueError, match=msg1d):
            _ = df > tup
        result = df > b_r
        tm.assert_frame_equal(result, expected)
        result = df.values > b_r
        tm.assert_numpy_array_equal(result, expected.values)
        with pytest.raises(ValueError, match=msg2d):
            _ = df > b_c
        with pytest.raises(ValueError, match=msg2db):
            _ = df.values > b_c
        expected = DataFrame([[False, False], [True, False], [False, False]])
        result = df == b
        tm.assert_frame_equal(result, expected)
        with pytest.raises(ValueError, match=msg1d):
            _ = df == lst
        with pytest.raises(ValueError, match=msg1d):
            _ = df == tup
        result = df == b_r
        tm.assert_frame_equal(result, expected)
        result = df.values == b_r
        tm.assert_numpy_array_equal(result, expected.values)
        with pytest.raises(ValueError, match=msg2d):
            _ = df == b_c
        df = DataFrame(np.arange(6).reshape((3, 2)), columns=list('AB'), index=list('abc'))
        expected.index = df.index
        expected.columns = df.columns
        with pytest.raises(ValueError, match=msg1d):
            _ = df == lst
        with pytest.raises(ValueError, match=msg1d):
            _ = df == tup

    def test_inplace_ops_alignment(self) -> None:
        columns: List[str] = list('abcdefg')
        X_orig: DataFrame = DataFrame(np.arange(10 * len(columns)).reshape(-1, len(columns)), columns=columns, index=range(10))
        Z: DataFrame = 100 * X_orig.iloc[:, 1:-1].copy()
        block1: List[str] = list('bedcf')
        subs: List[str] = list('bcdef')
        X: DataFrame = X_orig.copy()
        result1: DataFrame = (X[block1] + Z).reindex(columns=subs)
        X[block1] += Z
        result2: DataFrame = X.reindex(columns=subs)
        X = X_orig.copy()
        result3: DataFrame = (X[block1] + Z[block1]).reindex(columns=subs)
        X[block1] += Z[block1]
        result4: DataFrame = X.reindex(columns=subs)
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)
        tm.assert_frame_equal(result1, result4)
        X = X_orig.copy()
        result1 = (X[block1] - Z).reindex(columns=subs)
        X[block1] -= Z
        result2 = X.reindex(columns=subs)
        X = X_orig.copy()
        result3 = (X[block1] - Z[block1]).reindex(columns=subs)
        X[block1] -= Z[block1]
        result4 = X.reindex(columns=subs)
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)
        tm.assert_frame_equal(result1, result4)

    def test_inplace_ops_identity(self) -> None:
        s_orig: Series = Series([1, 2, 3])
        df_orig: DataFrame = DataFrame(np.random.default_rng(2).integers(0, 5, size=10).reshape(-1, 5))
        s: Series = s_orig.copy()
        s2: Series = s
        s += 1
        tm.assert_series_equal(s, s2)
        tm.assert_series_equal(s_orig + 1, s)
        assert s is s2
        assert s._mgr is s2._mgr
        df: DataFrame = df_orig.copy()
        df2: DataFrame = df
        df += 1
        tm.assert_frame_equal(df, df2)
        tm.assert_frame_equal(df_orig + 1, df)
        assert df is df2
        assert df._mgr is df2._mgr
        s = s_orig.copy()
        s2 = s
        s += 1.5
        tm.assert_series_equal(s, s2)
        tm.assert_series_equal(s_orig + 1.5, s)
        df = df_orig.copy()
        df2 = df
        df += 1.5
        tm.assert_frame_equal(df, df2)
        tm.assert_frame_equal(df_orig + 1.5, df)
        assert df is df2
        assert df._mgr is df2._mgr
        arr: np.ndarray = np.random.default_rng(2).integers(0, 10, size=5)
        df_orig = DataFrame({'A': arr.copy(), 'B': 'foo'})
        df = df_orig.copy()
        df2 = df
        df['A'] += 1
        expected = DataFrame({'A': arr.copy() + 1, 'B': 'foo'})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df2, expected)
        assert df._mgr is df2._mgr
        df = df_orig.copy()
        df2 = df
        df['A'] += 1.0
        expected = DataFrame({'A': arr.copy() + 1.0, 'B': 'foo'})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df2, expected)
        assert df._mgr is df2._mgr

    @pytest.mark.parametrize('op', ['add', 'and', pytest.param('div', marks=pytest.mark.xfail(raises=AttributeError, reason='__idiv__ not implemented')), 'floordiv', 'mod', 'mul', 'or', 'pow', 'sub', 'truediv', 'xor'])
    def test_inplace_ops_identity2(self, op: str) -> None:
        df: DataFrame = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1, 2, 3]})
        operand: int = 2
        if op in ('and', 'or', 'xor'):
            df['a'] = [True, False, True]
        df_copy: DataFrame = df.copy()
        iop: str = f'__i{op}__'
        op_func: str = f'__{op}__'
        getattr(df, iop)(operand)
        expected: DataFrame = getattr(df_copy, op_func)(operand)
        tm.assert_frame_equal(df, expected)
        expected_id: int = id(df)
        assert id(df) == expected_id

    @pytest.mark.parametrize('val', [
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3], dtype=np.int64),
        range(1, 4)
    ])
    def test_alignment_non_pandas(self, val: Any) -> None:
        index: List[str] = ['A', 'B', 'C']
        columns: List[str] = ['X', 'Y', 'Z']
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
        align = DataFrame._align_for_op
        expected: DataFrame = DataFrame({'X': val, 'Y': val, 'Z': val}, index=df.index)
        tm.assert_frame_equal(align(df, val, axis=0)[1], expected)
        expected = DataFrame({'X': [1, 1, 1], 'Y': [2, 2, 2], 'Z': [3, 3, 3]}, index=df.index)
        tm.assert_frame_equal(align(df, val, axis=1)[1], expected)

    @pytest.mark.parametrize('val', [
        [1, 2],
        (1, 2),
        np.array([1, 2]),
        range(1, 3)
    ])
    def test_alignment_non_pandas_length_mismatch(self, val: Any) -> None:
        index: List[str] = ['A', 'B', 'C']
        columns: List[str] = ['X', 'Y', 'Z']
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
        align = DataFrame._align_for_op
        msg: str = 'Unable to coerce to Series, length must be 3: given 2'
        with pytest.raises(ValueError, match=msg):
            _ = align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            _ = align(df, val, axis=1)

    def test_alignment_non_pandas_index_columns(self) -> None:
        index: List[str] = ['A', 'B', 'C']
        columns: List[str] = ['X', 'Y', 'Z']
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
        align = DataFrame._align_for_op
        val: np.ndarray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tm.assert_frame_equal(align(df, val, axis=0)[1], DataFrame(val, index=df.index, columns=df.columns))
        tm.assert_frame_equal(align(df, val, axis=1)[1], DataFrame(val, index=df.index, columns=df.columns))
        msg: str = 'Unable to coerce to DataFrame, shape must be'
        val = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match=msg):
            _ = align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            _ = align(df, val, axis=1)
        val = np.zeros((3, 3, 3))
        msg = re.escape('Unable to coerce to Series/DataFrame, dimension must be <= 2: (3, 3, 3)')
        with pytest.raises(ValueError, match=msg):
            _ = align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            _ = align(df, val, axis=1)

    def test_no_warning(self, all_arithmetic_operators: Any) -> None:
        df: DataFrame = DataFrame({'A': [0.0, 0.0], 'B': [0.0, None]})
        b: Series = df['B']
        with tm.assert_produces_warning(None):
            _ = getattr(df, all_arithmetic_operators)(b)

    def test_dunder_methods_binary(self, all_arithmetic_operators: Any) -> None:
        df: DataFrame = DataFrame({'A': [0.0, 0.0], 'B': [0.0, None]})
        b: Series = df['B']
        with pytest.raises(TypeError, match='takes 2 positional arguments'):
            _ = getattr(df, all_arithmetic_operators)(b, 0)

    def test_align_int_fill_bug(self) -> None:
        X: np.ndarray = np.arange(10 * 10, dtype='float64').reshape(10, 10)
        Y: np.ndarray = np.ones((10, 1), dtype=int)
        df1: DataFrame = DataFrame(X)
        df1['0.X'] = Y.squeeze()
        df2: DataFrame = df1.astype(float)
        result: DataFrame = df1 - df1.mean()
        expected: DataFrame = df2 - df2.mean()
        tm.assert_frame_equal(result, expected)

def test_pow_with_realignment() -> None:
    left: DataFrame = DataFrame({'A': [0, 1, 2]})
    right: DataFrame = DataFrame(index=[0, 1, 2])
    result: DataFrame = left ** right
    expected: DataFrame = DataFrame({'A': [np.nan, 1.0, np.nan]})
    tm.assert_frame_equal(result, expected)

def test_dataframe_series_extension_dtypes() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).integers(0, 100, (10, 3)), columns=['a', 'b', 'c'])
    ser: Series = Series([1, 2, 3], index=['a', 'b', 'c'])
    expected: DataFrame = DataFrame(df.to_numpy('int64') + ser.to_numpy('int64').reshape(-1, 3), columns=df.columns, dtype='Int64')
    df_ea: DataFrame = df.astype('Int64')
    result: DataFrame = df_ea + ser
    tm.assert_frame_equal(result, expected)
    result = df_ea + ser.astype('Int64')
    tm.assert_frame_equal(result, expected)

def test_dataframe_blockwise_slicelike() -> None:
    arr: np.ndarray = np.random.default_rng(2).integers(0, 1000, (100, 10))
    df1: DataFrame = DataFrame(arr)
    df2: DataFrame = df1.copy().astype({1: 'float', 3: 'float', 7: 'float'})
    df2.iloc[0, [1, 3, 7]] = np.nan
    df3: DataFrame = df1.copy().astype({5: 'float'})
    df3.iloc[0, [5]] = np.nan
    df4: DataFrame = df1.copy().astype({2: 'float', 3: 'float', 4: 'float'})
    df4.iloc[0, np.arange(2, 5)] = np.nan
    df5: DataFrame = df1.copy().astype({4: 'float', 5: 'float', 6: 'float'})
    df5.iloc[0, np.arange(4, 7)] = np.nan
    for left, right in [(df1, df2), (df2, df3), (df4, df5)]:
        res: DataFrame = left + right
        expected: DataFrame = DataFrame({i: left[i] + right[i] for i in left.columns})
        tm.assert_frame_equal(res, expected)

@pytest.mark.parametrize('df, col_dtype', [
    (DataFrame([[1.0, 2.0], [4.0, 5.0]], columns=list('ab')), 'float64'),
    (DataFrame([[1.0, 'b'], [4.0, 'b']], columns=list('ab')).astype({'b': object}), 'object')
])
def test_dataframe_operation_with_non_numeric_types(df: DataFrame, col_dtype: str) -> None:
    expected: DataFrame = DataFrame([[0.0, np.nan], [3.0, np.nan]], columns=list('ab'))
    expected = expected.astype({'b': col_dtype})
    result: DataFrame = df + Series([-1.0], index=list('a'))
    tm.assert_frame_equal(result, expected)

def test_arith_reindex_with_duplicates() -> None:
    df1: DataFrame = DataFrame(data=[[0]], columns=['second'])
    df2: DataFrame = DataFrame(data=[[0, 0, 0]], columns=['first', 'second', 'second'])
    result: DataFrame = df1 + df2
    expected: DataFrame = DataFrame([[np.nan, 0, 0]], columns=['first', 'second', 'second'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('to_add', [
    [Series([1, 1])],
    [Series([1, 1]), Series([1, 1])]
])
def test_arith_list_of_arraylike_raise(to_add: List[Any]) -> None:
    df: DataFrame = DataFrame({'x': [1, 2], 'y': [1, 2]})
    msg: str = f'Unable to coerce list of {type(to_add[0])} to Series/DataFrame'
    with pytest.raises(ValueError, match=msg):
        _ = df + to_add
    with pytest.raises(ValueError, match=msg):
        _ = to_add + df

def test_inplace_arithmetic_series_update() -> None:
    df: DataFrame = DataFrame({'A': [1, 2, 3]})
    df_orig: DataFrame = df.copy()
    series: Series = df['A']
    vals: Any = series._values
    series += 1
    assert series._values is not vals
    tm.assert_frame_equal(df, df_orig)

def test_arithmetic_multiindex_align() -> None:
    """
    Regression test for: https://github.com/pandas-dev/pandas/issues/33765
    """
    df1: DataFrame = DataFrame([[1]], index=['a'], columns=MultiIndex.from_product([[0], [1]], names=['a', 'b']))
    df2: DataFrame = DataFrame([[1]], index=['a'], columns=Index([0], name='a'))
    expected: DataFrame = DataFrame([[0]], index=['a'], columns=MultiIndex.from_product([[0], [1]], names=['a', 'b']))
    result: DataFrame = df1 - df2
    tm.assert_frame_equal(result, expected)

def test_arithmetic_multiindex_column_align() -> None:
    df1: DataFrame = DataFrame(data=100, columns=MultiIndex.from_product([['1A', '1B'], ['2A', '2B']], names=['Lev1', 'Lev2']), index=['C1', 'C2'])
    df2: DataFrame = DataFrame(data=np.array([[0.1, 0.25], [0.2, 0.45]]), columns=MultiIndex.from_product([['1A', '1B']], names=['Lev1']), index=['C1', 'C2'])
    expected: DataFrame = DataFrame(data=np.array([[10.0, 10.0, 25.0, 25.0], [20.0, 20.0, 45.0, 45.0]]), columns=MultiIndex.from_product([['1A', '1B'], ['2A', '2B']], names=['Lev1', 'Lev2']), index=['C1', 'C2'])
    result: DataFrame = df1 * df2
    tm.assert_frame_equal(result, expected)

def test_arithmetic_multiindex_column_align_with_fillvalue() -> None:
    df1: DataFrame = DataFrame(data=[[1.0, 2.0]], columns=MultiIndex.from_tuples([('A', 'one'), ('A', 'two')]))
    df2: DataFrame = DataFrame(data=[[3.0, 4.0]], columns=MultiIndex.from_tuples([('B', 'one'), ('B', 'two')]))
    expected: DataFrame = DataFrame(data=[[1.0, 2.0, 3.0, 4.0]], columns=MultiIndex.from_tuples([('A', 'one'), ('A', 'two'), ('B', 'one'), ('B', 'two')]))
    result: DataFrame = df1.add(df2, fill_value=0)
    tm.assert_frame_equal(result, expected)

def test_bool_frame_mult_float() -> None:
    df: DataFrame = DataFrame(True, list('ab'), list('cd'))
    result: DataFrame = df * 1.0
    expected: DataFrame = DataFrame(np.ones((2, 2)), list('ab'), list('cd'))
    tm.assert_frame_equal(result, expected)

def test_frame_sub_nullable_int(any_int_ea_dtype: str) -> None:
    series1: Series = Series([1, 2, None], dtype=any_int_ea_dtype)
    series2: Series = Series([1, 2, 3], dtype=any_int_ea_dtype)
    expected: DataFrame = DataFrame([0, 0, None], dtype=any_int_ea_dtype)
    result: DataFrame = series1.to_frame() - series2.to_frame()
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning')
def test_frame_op_subclass_nonclass_constructor() -> None:
    class SubclassedSeries(Series):
        @property
        def _constructor(self) -> Callable[..., "SubclassedSeries"]:
            return SubclassedSeries

        @property
        def _constructor_expanddim(self) -> type:
            return SubclassedDataFrame

    class SubclassedDataFrame(DataFrame):
        _metadata: List[str] = ['my_extra_data']
        def __init__(self, my_extra_data: Any, *args: Any, **kwargs: Any) -> None:
            self.my_extra_data = my_extra_data
            super().__init__(*args, **kwargs)

        @property
        def _constructor(self) -> Callable[..., "SubclassedDataFrame"]:
            return functools.partial(type(self), self.my_extra_data)

        @property
        def _constructor_sliced(self) -> type:
            return SubclassedSeries

    sdf: SubclassedDataFrame = SubclassedDataFrame('some_data', {'A': [1, 2, 3], 'B': [4, 5, 6]})
    result: SubclassedDataFrame = sdf * 2
    expected: SubclassedDataFrame = SubclassedDataFrame('some_data', {'A': [2, 4, 6], 'B': [8, 10, 12]})
    tm.assert_frame_equal(result, expected)
    result = sdf + sdf
    tm.assert_frame_equal(result, expected)

def test_enum_column_equality() -> None:
    Cols = Enum('Cols', 'col1 col2')
    q1: DataFrame = DataFrame({Cols.col1: [1, 2, 3]})
    q2: DataFrame = DataFrame({Cols.col1: [1, 2, 3]})
    result: Series = q1[Cols.col1] == q2[Cols.col1]
    expected: Series = Series([True, True, True], name=Cols.col1)
    tm.assert_series_equal(result, expected)

def test_mixed_col_index_dtype(using_infer_string: bool) -> None:
    df1: DataFrame = DataFrame(columns=list('abc'), data=1.0, index=[0])
    df2: DataFrame = DataFrame(columns=list('abc'), data=0.0, index=[0])
    df1.columns = df2.columns.astype('string')
    result: DataFrame = df1 + df2
    expected: DataFrame = DataFrame(columns=list('abc'), data=1.0, index=[0])
    if using_infer_string:
        if HAS_PYARROW:
            dtype: str = 'string[pyarrow]'
        else:
            dtype = 'string'
        expected.columns = expected.columns.astype(dtype)
    tm.assert_frame_equal(result, expected)