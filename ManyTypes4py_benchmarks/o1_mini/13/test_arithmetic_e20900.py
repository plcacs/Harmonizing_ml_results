from collections import deque
from datetime import datetime, timezone
from enum import Enum
import functools
import operator
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas.compat import HAS_PYARROW
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import _check_mixed_float, _check_mixed_int


@pytest.fixture
def simple_frame() -> pd.DataFrame:
    """
    Fixture for simple 3x3 DataFrame

    Columns are ['one', 'two', 'three'], index is ['a', 'b', 'c'].

       one  two  three
    a  1.0  2.0    3.0
    b  4.0  5.0    6.0
    c  7.0  8.0    9.0
    """
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return DataFrame(arr, columns=['one', 'two', 'three'], index=['a', 'b', 'c'])


@pytest.fixture(autouse=True, params=[0, 100], ids=['numexpr', 'python'])
def switch_numexpr_min_elements(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', request.param)
        yield request.param


class DummyElement:

    def __init__(self, value: Any, dtype: Union[str, np.dtype]) -> None:
        self.value: Any = value
        self.dtype: np.dtype = np.dtype(dtype)

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        return np.array(self.value, dtype=self.dtype)

    def __str__(self) -> str:
        return f'DummyElement({self.value}, {self.dtype})'

    def __repr__(self) -> str:
        return str(self)

    def astype(self, dtype: Union[str, np.dtype], copy: bool = False) -> 'DummyElement':
        self.dtype = dtype
        return self

    def view(self, dtype: Union[str, np.dtype]) -> 'DummyElement':
        return type(self)(self.value.view(dtype), dtype)

    def any(self, axis: Optional[Any] = None) -> bool:
        return bool(self.value)


class TestFrameComparisons:

    def test_comparison_with_categorical_dtype(self) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': ['foo', 'bar', 'baz']})
        exp: pd.DataFrame = pd.DataFrame({'A': [True, False, False]})
        res: pd.DataFrame = df == 'foo'
        tm.assert_frame_equal(res, exp)
        df['A'] = df['A'].astype('category')
        res = df == 'foo'
        tm.assert_frame_equal(res, exp)

    def test_frame_in_list(self) -> None:
        df: pd.DataFrame = pd.DataFrame(np.random.default_rng(2).standard_normal((6, 4)), columns=list('ABCD'))
        msg: str = 'The truth value of a DataFrame is ambiguous'
        with pytest.raises(ValueError, match=msg):
            df in [None]

    @pytest.mark.parametrize(
        'arg, arg2',
        [
            [
                {
                    'a': np.random.default_rng(2).integers(10, size=10),
                    'b': pd.date_range('20010101', periods=10)
                },
                {
                    'a': np.random.default_rng(2).integers(10, size=10),
                    'b': np.random.default_rng(2).integers(10, size=10)
                }
            ],
            [
                {
                    'a': np.random.default_rng(2).integers(10, size=10),
                    'b': np.random.default_rng(2).integers(10, size=10)
                },
                {
                    'a': np.random.default_rng(2).integers(10, size=10),
                    'b': pd.date_range('20010101', periods=10)
                }
            ],
            [
                {
                    'a': pd.date_range('20010101', periods=10),
                    'b': pd.date_range('20010101', periods=10)
                },
                {
                    'a': np.random.default_rng(2).integers(10, size=10),
                    'b': np.random.default_rng(2).integers(10, size=10)
                }
            ],
            [
                {
                    'a': np.random.default_rng(2).integers(10, size=10),
                    'b': pd.date_range('20010101', periods=10)
                },
                {
                    'a': pd.date_range('20010101', periods=10),
                    'b': pd.date_range('20010101', periods=10)
                }
            ],
        ]
    )
    def test_comparison_invalid(self, arg: Dict[str, Any], arg2: Dict[str, Any]) -> None:
        x: pd.DataFrame = pd.DataFrame(arg)
        y: pd.DataFrame = pd.DataFrame(arg2)
        result: pd.DataFrame = x == y
        expected: pd.DataFrame = pd.DataFrame(
            {col: x[col] == y[col] for col in x.columns},
            index=x.index,
            columns=x.columns
        )
        tm.assert_frame_equal(result, expected)
        result = x != y
        expected = pd.DataFrame(
            {col: x[col] != y[col] for col in x.columns},
            index=x.index,
            columns=x.columns
        )
        tm.assert_frame_equal(result, expected)
        msgs: List[str] = [
            'Invalid comparison between dtype=datetime64\\[ns\\] and ndarray',
            'invalid type promotion',
            "The DTypes <class 'numpy.dtype\\[.*\\]'> and <class 'numpy.dtype\\[.*\\]'> do not have a common DType."
        ]
        msg: str = '|'.join(msgs)
        with pytest.raises(TypeError, match=msg):
            x >= y
        with pytest.raises(TypeError, match=msg):
            x > y
        with pytest.raises(TypeError, match=msg):
            x < y
        with pytest.raises(TypeError, match=msg):
            x <= y

    @pytest.mark.parametrize(
        'left, right',
        [
            ('gt', 'lt'),
            ('lt', 'gt'),
            ('ge', 'le'),
            ('le', 'ge'),
            ('eq', 'eq'),
            ('ne', 'ne')
        ]
    )
    def test_timestamp_compare(self, left: str, right: str) -> None:
        df: pd.DataFrame = pd.DataFrame({
            'dates1': pd.date_range('20010101', periods=10),
            'dates2': pd.date_range('20010102', periods=10),
            'intcol': np.random.default_rng(2).integers(1000000000, size=10),
            'floatcol': np.random.default_rng(2).standard_normal(10),
            'stringcol': [chr(100 + i) for i in range(10)]
        })
        df.loc[np.random.default_rng(2).random(len(df)) > 0.5, 'dates2'] = pd.NaT
        left_f: Callable = getattr(operator, left)
        right_f: Callable = getattr(operator, right)
        if left in ['eq', 'ne']:
            expected: pd.DataFrame = left_f(df, pd.Timestamp('20010109'))
            result: pd.DataFrame = right_f(pd.Timestamp('20010109'), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg: str = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'Timestamp'"
            with pytest.raises(TypeError, match=msg):
                left_f(df, pd.Timestamp('20010109'))
            with pytest.raises(TypeError, match=msg):
                right_f(pd.Timestamp('20010109'), df)
        if left in ['eq', 'ne']:
            expected = left_f(df, pd.Timestamp('nat'))
            result = right_f(pd.Timestamp('nat'), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'NaTType'"
            with pytest.raises(TypeError, match=msg):
                left_f(df, pd.Timestamp('nat'))
            with pytest.raises(TypeError, match=msg):
                right_f(pd.Timestamp('nat'), df)

    def test_mixed_comparison(self) -> None:
        df: pd.DataFrame = pd.DataFrame([['1989-08-01', 1], ['1989-08-01', 2]])
        other: pd.DataFrame = pd.DataFrame([['a', 'b'], ['c', 'd']])
        result: pd.DataFrame = df == other
        assert not result.any().any()
        result = df != other
        assert result.all().all()

    def test_df_boolean_comparison_error(self) -> None:
        df: pd.DataFrame = pd.DataFrame(np.arange(6).reshape((3, 2)))
        expected: pd.DataFrame = pd.DataFrame([[False, False], [True, False], [False, False]])
        result: pd.DataFrame = df == (2, 2)
        tm.assert_frame_equal(result, expected)
        result = df == [2, 2]
        tm.assert_frame_equal(result, expected)

    def test_df_float_none_comparison(self) -> None:
        df: pd.DataFrame = pd.DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)),
            index=range(8),
            columns=['A', 'B', 'C']
        )
        result: pd.DataFrame = df.__eq__(None)
        assert not result.any().any()

    def test_df_string_comparison(self) -> None:
        df: pd.DataFrame = pd.DataFrame([{'a': 1, 'b': 'foo'}, {'a': 2, 'b': 'bar'}])
        mask_a: pd.Series = df.a > 1
        tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
        tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])
        mask_b: pd.Series = df.b == 'foo'
        tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
        tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])


class TestFrameFlexComparisons:

    def test_bool_flex_frame(self, comparison_op: Callable) -> None:
        data: np.ndarray = np.random.default_rng(2).standard_normal((5, 3))
        other_data: np.ndarray = np.random.default_rng(2).standard_normal((5, 3))
        df: pd.DataFrame = pd.DataFrame(data)
        other: pd.DataFrame = pd.DataFrame(other_data)
        ndim_5: np.ndarray = np.ones(df.shape + (1, 3))
        assert df.eq(df).values.all()
        assert not df.ne(df).values.any()
        f: Callable = getattr(df, comparison_op.__name__)
        o: Callable = comparison_op
        tm.assert_frame_equal(f(other), o(df, other))
        part_o: pd.DataFrame = other.loc[3:, 1:].copy()
        rs: pd.DataFrame = f(part_o)
        xp: pd.DataFrame = o(df, part_o.reindex(index=df.index, columns=df.columns))
        tm.assert_frame_equal(rs, xp)
        tm.assert_frame_equal(f(other.values), o(df, other.values))
        tm.assert_frame_equal(f(0), o(df, 0))
        msg: str = 'Unable to coerce to Series/DataFrame'
        tm.assert_frame_equal(f(np.nan), o(df, np.nan))
        with pytest.raises(ValueError, match=msg):
            f(ndim_5)

    @pytest.mark.parametrize('box', [np.array, Series])
    def test_bool_flex_series(self, box: Callable[[Any], Union[np.ndarray, pd.Series]]) -> None:
        data: np.ndarray = np.random.default_rng(2).standard_normal((5, 3))
        df: pd.DataFrame = pd.DataFrame(data)
        idx_ser: Union[np.ndarray, pd.Series] = box(np.random.default_rng(2).standard_normal(5))
        col_ser: Union[np.ndarray, pd.Series] = box(np.random.default_rng(2).standard_normal(3))
        idx_eq: pd.DataFrame = df.eq(idx_ser, axis=0)
        col_eq: pd.DataFrame = df.eq(col_ser)
        idx_ne: pd.DataFrame = df.ne(idx_ser, axis=0)
        col_ne: pd.DataFrame = df.ne(col_ser)
        tm.assert_frame_equal(col_eq, df == Series(col_ser))
        tm.assert_frame_equal(col_eq, -col_ne)
        tm.assert_frame_equal(idx_eq, -idx_ne)
        tm.assert_frame_equal(idx_eq, df.T.eq(idx_ser).T)
        tm.assert_frame_equal(col_eq, df.eq(list(col_ser)))
        tm.assert_frame_equal(idx_eq, df.eq(Series(idx_ser), axis=0))
        tm.assert_frame_equal(idx_eq, df.eq(list(idx_ser), axis=0))
        idx_gt: pd.DataFrame = df.gt(idx_ser, axis=0)
        col_gt: pd.DataFrame = df.gt(col_ser)
        idx_le: pd.DataFrame = df.le(idx_ser, axis=0)
        col_le: pd.DataFrame = df.le(col_ser)
        tm.assert_frame_equal(col_gt, df > Series(col_ser))
        tm.assert_frame_equal(col_gt, -col_le)
        tm.assert_frame_equal(idx_gt, -idx_le)
        tm.assert_frame_equal(idx_gt, df.T.gt(idx_ser).T)
        idx_ge: pd.DataFrame = df.ge(idx_ser, axis=0)
        col_ge: pd.DataFrame = df.ge(col_ser)
        idx_lt: pd.DataFrame = df.lt(idx_ser, axis=0)
        col_lt: pd.DataFrame = df.lt(col_ser)
        tm.assert_frame_equal(col_ge, df >= Series(col_ser))
        tm.assert_frame_equal(col_ge, -col_lt)
        tm.assert_frame_equal(idx_ge, -idx_lt)
        tm.assert_frame_equal(idx_ge, df.T.ge(idx_ser).T)
        idx_ser = Series(np.random.default_rng(2).standard_normal(5))
        col_ser = Series(np.random.default_rng(2).standard_normal(3))

    def test_bool_flex_frame_na(self) -> None:
        df: pd.DataFrame = pd.DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df.loc[0, 0] = np.nan
        rs: pd.DataFrame = df.eq(df)
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
        df: pd.DataFrame = pd.DataFrame({'a': arr})
        df2: pd.DataFrame = pd.DataFrame({'a': arr2})
        msg: str = '|'.join([
            "'>' not supported between instances of '.*' and 'complex'",
            'unorderable types: .*complex\\(\\)'
        ])
        with pytest.raises(TypeError, match=msg):
            df.gt(df2)
        with pytest.raises(TypeError, match=msg):
            df['a'].gt(df2['a'])
        with pytest.raises(TypeError, match=msg):
            df.values > df2.values
        rs: pd.DataFrame = df.ne(df2)
        assert rs.values.all()
        arr3: np.ndarray = np.array([2j, np.nan, None])
        df3: pd.DataFrame = pd.DataFrame({'a': arr3})
        with pytest.raises(TypeError, match=msg):
            df3.gt(2j)
        with pytest.raises(TypeError, match=msg):
            df3['a'].gt(2j)
        with pytest.raises(TypeError, match=msg):
            df3.values > 2j

    def test_bool_flex_frame_object_dtype(self) -> None:
        df1: pd.DataFrame = pd.DataFrame({'col': ['foo', np.nan, 'bar']}, dtype=object)
        df2: pd.DataFrame = pd.DataFrame({'col': ['foo', datetime.now(), 'bar']}, dtype=object)
        result: pd.DataFrame = df1.ne(df2)
        exp: pd.DataFrame = pd.DataFrame({'col': [False, True, False]})
        tm.assert_frame_equal(result, exp)

    def test_flex_comparison_nat(self) -> None:
        df: pd.DataFrame = pd.DataFrame([pd.NaT])
        result: pd.DataFrame = df == pd.NaT
        assert result.iloc[0, 0].item() is False
        result = df.eq(pd.NaT)
        assert result.iloc[0, 0].item() is False
        result = df != pd.NaT
        assert result.iloc[0, 0].item() is True
        result = df.ne(pd.NaT)
        assert result.iloc[0, 0].item() is True

    def test_df_flex_cmp_constant_return_types(self, comparison_op: Callable) -> None:
        df: pd.DataFrame = pd.DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        const: int = 2
        result: pd.Series = getattr(df, comparison_op.__name__)(const).dtypes.value_counts()
        tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))

    def test_df_flex_cmp_constant_return_types_empty(self, comparison_op: Callable) -> None:
        df: pd.DataFrame = pd.DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        const: int = 2
        empty: pd.DataFrame = df.iloc[:0]
        result: pd.Series = getattr(empty, comparison_op.__name__)(const).dtypes.value_counts()
        tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))

    def test_df_flex_cmp_ea_dtype_with_ndarray_series(self) -> None:
        ii: pd.IntervalIndex = pd.IntervalIndex.from_breaks([1, 2, 3])
        df: pd.DataFrame = pd.DataFrame({'A': ii, 'B': ii})
        ser: pd.Series = Series([0, 0])
        res: pd.DataFrame = df.eq(ser, axis=0)
        expected: pd.DataFrame = pd.DataFrame({'A': [False, False], 'B': [False, False]})
        tm.assert_frame_equal(res, expected)
        ser2: pd.Series = Series([1, 2], index=['A', 'B'])
        res2: pd.DataFrame = df.eq(ser2, axis=1)
        tm.assert_frame_equal(res2, expected)


class TestFrameFlexArithmetic:

    def test_floordiv_axis0(self) -> None:
        arr: np.ndarray = np.arange(3)
        ser: pd.Series = Series(arr)
        df: pd.DataFrame = pd.DataFrame({'A': ser, 'B': ser})
        result: pd.DataFrame = df.floordiv(ser, axis=0)
        expected: pd.DataFrame = pd.DataFrame({col: df[col] // ser for col in df.columns})
        tm.assert_frame_equal(result, expected)
        result2: pd.DataFrame = df.floordiv(ser.values, axis=0)
        tm.assert_frame_equal(result2, expected)

    def test_df_add_td64_columnwise(self) -> None:
        dti: pd.DatetimeIndex = pd.date_range('2016-01-01', periods=10)
        tdi: pd.TimedeltaIndex = pd.timedelta_range('1', periods=10)
        tser: pd.Series = Series(tdi)
        df: pd.DataFrame = pd.DataFrame({0: dti, 1: tdi})
        result: pd.DataFrame = df.add(tser, axis=0)
        expected: pd.DataFrame = pd.DataFrame({0: dti + tdi, 1: tdi + tdi})
        tm.assert_frame_equal(result, expected)

    def test_df_add_flex_filled_mixed_dtypes(self) -> None:
        dti: pd.DatetimeIndex = pd.date_range('2016-01-01', periods=3)
        ser: pd.Series = Series(['1 Day', 'NaT', '2 Days'], dtype='timedelta64[ns]')
        df: pd.DataFrame = pd.DataFrame({'A': dti, 'B': ser})
        other: pd.DataFrame = pd.DataFrame({'A': ser, 'B': ser})
        fill: np.timedelta64 = pd.Timedelta(days=1).to_timedelta64()
        result: pd.DataFrame = df.add(other, fill_value=fill)
        expected: pd.DataFrame = pd.DataFrame({
            'A': Series(['2016-01-02', '2016-01-03', '2016-01-05'], dtype='datetime64[ns]'),
            'B': ser * 2
        })
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_frame(self, all_arithmetic_operators: Callable, float_frame: pd.DataFrame, mixed_float_frame: pd.DataFrame) -> None:
        op: Callable = all_arithmetic_operators

        def f(x: Any, y: Any) -> Any:
            if op.__name__.startswith('__r'):
                return getattr(operator, op.__name__.replace('__r', '__'))(y, x)
            return op(x, y)

        result: pd.DataFrame = getattr(float_frame, op.__name__)(2 * float_frame)
        expected: pd.DataFrame = f(float_frame, 2 * float_frame)
        tm.assert_frame_equal(result, expected)
        result = getattr(mixed_float_frame, op.__name__)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={'C': None})

    @pytest.mark.parametrize('op', ['__add__', '__sub__', '__mul__'])
    def test_arith_flex_frame_mixed(
        self,
        op: str,
        int_frame: pd.DataFrame,
        mixed_int_frame: pd.DataFrame,
        mixed_float_frame: pd.DataFrame,
        switch_numexpr_min_elements: int
    ) -> None:
        f: Callable = getattr(operator, op)
        result: pd.DataFrame = getattr(mixed_int_frame, op)(2 + mixed_int_frame)
        expected: pd.DataFrame = f(mixed_int_frame, 2 + mixed_int_frame)
        dtype: Optional[Dict[str, Union[str, None]]] = None
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
    def test_arith_flex_frame_raise(self, all_arithmetic_operators: Callable, float_frame: pd.DataFrame, dim: int) -> None:
        op: Callable = all_arithmetic_operators
        arr: np.ndarray = np.ones((1,) * dim)
        msg: str = 'Unable to coerce to Series/DataFrame'
        with pytest.raises(ValueError, match=msg):
            getattr(float_frame, op.__name__)(arr)

    def test_arith_flex_frame_corner(self, float_frame: pd.DataFrame) -> None:
        const_add: pd.DataFrame = float_frame.add(1)
        tm.assert_frame_equal(const_add, float_frame + 1)
        result: pd.DataFrame = float_frame.add(float_frame[:0])
        expected: pd.DataFrame = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)
        result = float_frame[:0].add(float_frame)
        tm.assert_frame_equal(result, expected)
        with pytest.raises(NotImplementedError, match='fill_value'):
            float_frame.add(float_frame.iloc[0], fill_value=3)
        with pytest.raises(NotImplementedError, match='fill_value'):
            float_frame.add(float_frame.iloc[0], axis='index', fill_value=3)

    @pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'mod'])
    def test_arith_flex_series_ops(self, simple_frame: pd.DataFrame, op: str) -> None:
        df: pd.DataFrame = simple_frame
        row: pd.Series = df.xs('a')
        col: pd.Series = df['two']
        f: Callable = getattr(df, op)
        operator_func: Callable = getattr(operator, op)
        tm.assert_frame_equal(f(row), operator_func(df, row))
        tm.assert_frame_equal(f(col, axis=0), operator_func(df.T, col).T)

    def test_arith_flex_series(self, simple_frame: pd.DataFrame) -> None:
        df: pd.DataFrame = simple_frame
        row: pd.Series = df.xs('a')
        col: pd.Series = df['two']
        tm.assert_frame_equal(df.add(row, axis=None), df + row)
        tm.assert_frame_equal(df.div(row), df / row)
        tm.assert_frame_equal(df.div(col, axis=0), (df.T / col).T)

    def test_arith_flex_series_broadcasting(self, any_real_numpy_dtype: str) -> None:
        df: pd.DataFrame = pd.DataFrame(np.arange(3 * 2).reshape((3, 2)), dtype=any_real_numpy_dtype)
        expected: pd.DataFrame = pd.DataFrame([[np.nan, np.inf], [1.0, 1.5], [1.0, 1.25]], columns=df.columns, index=df.index)
        if any_real_numpy_dtype == 'float32':
            expected = expected.astype(any_real_numpy_dtype)
        result: pd.DataFrame = df.div(df[0], axis='index')
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_zero_len_raises(self) -> None:
        ser_len0: pd.Series = Series([], dtype=object)
        df_len0: pd.DataFrame = pd.DataFrame(columns=['A', 'B'])
        df: pd.DataFrame = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        with pytest.raises(NotImplementedError, match='fill_value'):
            df.add(ser_len0, fill_value='E')
        with pytest.raises(NotImplementedError, match='fill_value'):
            df_len0.sub(df['A'], axis=None, fill_value=3)

    def test_flex_add_scalar_fill_value(self) -> None:
        dat: np.ndarray = np.array([0, 1, np.nan, 3, 4, 5], dtype='float')
        df: pd.DataFrame = pd.DataFrame({'foo': dat}, index=range(6))
        exp: pd.DataFrame = df.fillna(0).add(2)
        res: pd.DataFrame = df.add(2, fill_value=0)
        tm.assert_frame_equal(res, exp)

    def test_sub_alignment_with_duplicate_index(self) -> None:
        df1: pd.DataFrame = pd.DataFrame([1, 2, 3, 4, 5], index=[1, 2, 1, 2, 3])
        df2: pd.DataFrame = pd.DataFrame([1, 2, 3], index=[1, 2, 3])
        expected: pd.DataFrame = pd.DataFrame([0, 2, 0, 2, 2], index=[1, 1, 2, 2, 3])
        result: pd.DataFrame = df1.sub(df2)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op', ['__add__', '__mul__', '__sub__', '__truediv__'])
    def test_arithmetic_with_duplicate_columns(self, op: str) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': np.arange(10), 'B': np.random.default_rng(2).random(10)})
        expected: pd.DataFrame = getattr(df, op)(df)
        expected.columns = ['A', 'A']
        df.columns = ['A', 'A']
        result: pd.DataFrame = getattr(df, op)(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('level', [0, None])
    def test_broadcast_multiindex(self, level: Optional[int]) -> None:
        df1: pd.DataFrame = pd.DataFrame({'A': [0, 1, 2], 'B': [1, 2, 3]})
        df1.columns = df1.columns.set_names('L1')
        df2: pd.DataFrame = pd.DataFrame({('A', 'C'): [0, 0, 0], ('A', 'D'): [0, 0, 0]})
        df2.columns = df2.columns.set_names(['L1', 'L2'])
        result: pd.DataFrame = df1.add(df2, level=level)
        expected: pd.DataFrame = pd.DataFrame({('A', 'C'): [0, 1, 2], ('A', 'D'): [0, 1, 2]})
        expected.columns = expected.columns.set_names(['L1', 'L2'])
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            2010: [1, 2, 3],
            2020: [3, 4, 5]
        }, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        series: pd.Series = Series([0.4], index=MultiIndex.from_product([['b'], ['a']], names=['mod', 'scen']))
        expected: pd.DataFrame = pd.DataFrame({
            2010: [1.4, 2.4, 3.4],
            2020: [3.4, 4.4, 5.4]
        }, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        result: pd.DataFrame = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_series_index_to_frame_index(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            2010: [1],
            2020: [3]
        }, index=MultiIndex.from_product([['a'], ['b']], names=['scen', 'mod']))
        series: pd.Series = Series([10.0, 20.0, 30.0], index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        expected: pd.DataFrame = pd.DataFrame({
            2010: [11.0, 21, 31.0],
            2020: [13.0, 23.0, 33.0]
        }, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        result: pd.DataFrame = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_no_align(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            2010: [1, 2, 3],
            2020: [3, 4, 5]
        }, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
        series: pd.Series = Series([0.4], index=MultiIndex.from_product([['c'], ['a']], names=['mod', 'scen']))
        expected: pd.DataFrame = pd.DataFrame({
            2010: [np.nan, 3, np.nan],
            2020: [np.nan, 5, np.nan]
        }, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'b', 2), ('a', 'c', np.nan)], names=['scen', 'mod', 'id']))
        result: pd.DataFrame = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_part_align(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            2010: [1, 2, 3],
            2020: [3, 4, 5]
        }, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'c', 2)], names=['scen', 'mod', 'id']))
        series: pd.Series = Series([0.4], index=MultiIndex.from_product([['b'], ['a']], names=['mod', 'scen']))
        expected: pd.DataFrame = pd.DataFrame({
            2010: [1.4, 2.4, np.nan],
            2020: [3.4, 4.4, np.nan]
        }, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'c', 2)], names=['scen', 'mod', 'id']))
        result: pd.DataFrame = df.add(series, axis=0)
        tm.assert_frame_equal(result, expected)


class TestFrameArithmetic:

    def test_td64_op_nat_casting(self) -> None:
        ser: pd.Series = Series(['NaT', 'NaT'], dtype='timedelta64[ns]')
        df: pd.DataFrame = pd.DataFrame([[1, 2], [3, 4]])
        result: pd.DataFrame = df * ser
        expected: pd.DataFrame = pd.DataFrame({'0': ser, '1': ser})
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_rowlike_broadcasts(self) -> None:
        arr: np.ndarray = np.arange(6).reshape(3, 2)
        df: pd.DataFrame = pd.DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        rowlike: np.ndarray = arr[[1], :]
        assert rowlike.shape == (1, df.shape[1])
        expected: pd.DataFrame = pd.DataFrame([[2, 4], [4, 6], [6, 8]], columns=df.columns, index=df.index, dtype=arr.dtype)
        result: pd.DataFrame = df + rowlike
        tm.assert_frame_equal(result, expected)
        result = rowlike + df
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_collike_broadcasts(self) -> None:
        arr: np.ndarray = np.arange(6).reshape(3, 2)
        df: pd.DataFrame = pd.DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        collike: np.ndarray = arr[:, [1]]
        assert collike.shape == (df.shape[0], 1)
        expected: pd.DataFrame = pd.DataFrame([[1, 2], [5, 6], [9, 10]], columns=df.columns, index=df.index, dtype=arr.dtype)
        result: pd.DataFrame = df + collike
        tm.assert_frame_equal(result, expected)
        result = collike + df
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_rowlike_broadcasts(self, request: Any, all_arithmetic_operators: Callable) -> None:
        opname: str = all_arithmetic_operators.__name__
        arr: np.ndarray = np.arange(6).reshape(3, 2)
        df: pd.DataFrame = pd.DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        rowlike: np.ndarray = arr[[1], :]
        assert rowlike.shape == (1, df.shape[1])
        exvals: List[Any] = [
            getattr(df.loc['A'], opname)(rowlike.squeeze()),
            getattr(df.loc['B'], opname)(rowlike.squeeze()),
            getattr(df.loc['C'], opname)(rowlike.squeeze())
        ]
        expected: pd.DataFrame = pd.DataFrame(exvals, columns=df.columns, index=df.index)
        result: pd.DataFrame = getattr(df, opname)(rowlike)
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_collike_broadcasts(self, request: Any, all_arithmetic_operators: Callable) -> None:
        opname: str = all_arithmetic_operators.__name__
        arr: np.ndarray = np.arange(6).reshape(3, 2)
        df: pd.DataFrame = pd.DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
        collike: np.ndarray = arr[:, [1]]
        assert collike.shape == (df.shape[0], 1)
        exvals: Dict[bool, Any] = {
            True: getattr(df[True], opname)(collike.squeeze()),
            False: getattr(df[False], opname)(collike.squeeze())
        }
        dtype: Optional[np.dtype] = None
        if opname in ['__rmod__', '__rfloordiv__']:
            dtype = np.common_type(*(x.values for x in exvals.values()))
        expected: pd.DataFrame = pd.DataFrame(exvals, columns=df.columns, index=df.index, dtype=dtype)
        result: pd.DataFrame = getattr(df, opname)(collike)
        tm.assert_frame_equal(result, expected)

    def test_df_bool_mul_int(self) -> None:
        df: pd.DataFrame = pd.DataFrame([[False, True], [False, False]])
        result: pd.DataFrame = df * 1
        kinds: pd.Series = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == 'i').all()
        result = 1 * df
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == 'i').all()

    def test_arith_mixed(self) -> None:
        left: pd.DataFrame = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3]})
        result: pd.DataFrame = left + left
        expected: pd.DataFrame = pd.DataFrame({'A': ['aa', 'bb', 'cc'], 'B': [2, 4, 6]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('col', ['A', 'B'])
    def test_arith_getitem_commute(self, all_arithmetic_functions: Callable, col: str) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': [1.1, 3.3], 'B': [2.5, -3.9]})
        result: pd.Series = all_arithmetic_functions(df, 1)[col]
        expected: pd.Series = all_arithmetic_functions(df[col], 1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('values', [[1, 2], (1, 2), np.array([1, 2]), range(1, 3), deque([1, 2])])
    def test_arith_alignment_non_pandas_object(self, values: Union[List[int], Tuple[int, int], np.ndarray, range, deque]) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': [1, 1], 'B': [1, 1]})
        expected: pd.DataFrame = pd.DataFrame({'A': [2, 2], 'B': [3, 3]})
        result: pd.DataFrame = df + values
        tm.assert_frame_equal(result, expected)
        result = values + df
        tm.assert_frame_equal(result, expected)

    def test_arith_non_pandas_object(self) -> None:
        df: pd.DataFrame = pd.DataFrame(
            np.arange(1, 10, dtype='f8').reshape(3, 3),
            columns=['one', 'two', 'three'],
            index=['a', 'b', 'c']
        )
        val1: np.ndarray = df.xs('a').values
        added: pd.DataFrame = pd.DataFrame(df.values + val1, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val1, added)
        added = pd.DataFrame((df.values.T + val1).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val1, axis=0), added)
        val2: List[float] = list(df['two'])
        added = pd.DataFrame(df.values + val2, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val2, added)
        added = pd.DataFrame((df.values.T + val2).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val2, axis='index'), added)
        val3: np.ndarray = np.random.default_rng(2).random(df.shape)
        added = pd.DataFrame(df.values + val3, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val3), added)

    def test_operations_with_interval_categories_index(self, all_arithmetic_operators: Callable) -> None:
        op: Callable = all_arithmetic_operators
        ind: pd.CategoricalIndex = pd.CategoricalIndex(pd.interval_range(start=0.0, end=2.0))
        data: List[int] = [1, 2]
        df: pd.DataFrame = pd.DataFrame([data], columns=ind)
        num: int = 10
        result: pd.DataFrame = getattr(df, op.__name__)(num)
        expected: pd.DataFrame = pd.DataFrame([[getattr(n, op.__name__)(num) for n in data]], columns=ind)
        tm.assert_frame_equal(result, expected)

    def test_frame_with_frame_reindex(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            'foo': [pd.Timestamp('2019'), pd.Timestamp('2020')],
            'bar': [pd.Timestamp('2018'), pd.Timestamp('2021')]
        }, columns=['foo', 'bar'], dtype='M8[ns]')
        df2: pd.DataFrame = df[['foo']]
        result: pd.DataFrame = df - df2
        expected: pd.DataFrame = pd.DataFrame({
            'foo': [pd.Timedelta(0), pd.Timedelta(0)],
            'bar': [np.nan, np.nan]
        }, columns=['bar', 'foo'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'value, dtype',
        [
            (1, 'i8'),
            (1.0, 'f8'),
            (2 ** 63, 'f8'),
            (1j, 'complex128'),
            (2 ** 63, 'complex128'),
            (True, 'bool'),
            (np.timedelta64(20, 'ns'), '<m8[ns]'),
            (np.datetime64(20, 'ns'), '<M8[ns]')
        ]
    )
    @pytest.mark.parametrize(
        'op',
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.mod,
            operator.pow
        ],
        ids=lambda x: x.__name__
    )
    def test_binop_other(
        self,
        op: Callable,
        value: Any,
        dtype: str,
        switch_numexpr_min_elements: int
    ) -> None:
        skip: set = {
            (operator.truediv, 'bool'),
            (operator.pow, 'bool'),
            (operator.add, 'bool'),
            (operator.mul, 'bool')
        }
        elem: DummyElement = DummyElement(value, dtype)
        df: pd.DataFrame = pd.DataFrame({'A': [elem.value, elem.value]}, dtype=elem.dtype)
        invalid: set = {
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
            warn: Optional[Callable] = None
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
                    op(df, elem.value)
        elif (op, dtype) in skip:
            if op in [operator.add, operator.mul]:
                if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                    warn: Optional[Callable] = UserWarning
                else:
                    warn = None
                with tm.assert_produces_warning(warn, match='evaluating in Python'):
                    op(df, elem.value)
            else:
                msg: str = "operator '.*' not implemented for .* dtypes"
                with pytest.raises(NotImplementedError, match=msg):
                    op(df, elem.value)
        else:
            with tm.assert_produces_warning(None):
                result: pd.Series = op(df, elem.value).dtypes
                expected: pd.Series = op(df, value).dtypes
            tm.assert_series_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes(self) -> None:
        midx: pd.MultiIndex = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2: pd.MultiIndex = MultiIndex.from_arrays([Series([1, 2], dtype='Int8'), Series([3, 4])])
        left: pd.DataFrame = pd.DataFrame([[1, 2], [3, 4]], columns=midx)
        right: pd.DataFrame = pd.DataFrame([[1, 2], [3, 4]], columns=midx2)
        result: pd.DataFrame = left - right
        expected: pd.DataFrame = pd.DataFrame([[0, 0], [0, 0]], columns=midx)
        tm.assert_frame_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes_different_order(self) -> None:
        midx: pd.MultiIndex = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2: pd.MultiIndex = MultiIndex.from_arrays([Series([2, 1], dtype='Int8'), Series([4, 3])])
        left: pd.DataFrame = pd.DataFrame([[1, 2], [3, 4]], columns=midx)
        right: pd.DataFrame = pd.DataFrame([[1, 2], [3, 4]], columns=midx2)
        result: pd.DataFrame = left - right
        expected: pd.DataFrame = pd.DataFrame([[-1, 1], [-1, 1]], columns=midx)
        tm.assert_frame_equal(result, expected)


def test_frame_with_zero_len_series_corner_cases() -> None:
    df: pd.DataFrame = pd.DataFrame(np.random.default_rng(2).standard_normal(6).reshape(3, 2), columns=['A', 'B'])
    ser: pd.Series = Series(dtype=np.float64)
    result: pd.DataFrame = df + ser
    expected: pd.DataFrame = pd.DataFrame(df.values * np.nan, columns=df.columns)
    tm.assert_frame_equal(result, expected)
    with pytest.raises(ValueError, match='not aligned'):
        df == ser
    df2: pd.DataFrame = pd.DataFrame(df.values.view('M8[ns]'), columns=df.columns)
    with pytest.raises(ValueError, match='not aligned'):
        df2 == ser


def test_zero_len_frame_with_series_corner_cases() -> None:
    df: pd.DataFrame = pd.DataFrame(columns=['A', 'B'], dtype=np.float64)
    ser: pd.Series = Series([1, 2], index=['A', 'B'])
    result: pd.DataFrame = df + ser
    expected: pd.DataFrame = df
    tm.assert_frame_equal(result, expected)


def test_frame_single_columns_object_sum_axis_1() -> None:
    data: Dict[str, pd.Series] = {'One': Series(['A', 1.2, np.nan])}
    df: pd.DataFrame = pd.DataFrame(data)
    result: pd.Series = df.sum(axis=1)
    expected: pd.Series = Series(['A', 1.2, 0], name='One')
    tm.assert_series_equal(result, expected)


class TestFrameArithmeticUnsorted:

    def test_frame_add_tz_mismatch_converts_to_utc(self) -> None:
        rng: pd.DatetimeIndex = pd.date_range('1/1/2011', periods=10, freq='h', tz='US/Eastern')
        df: pd.DataFrame = pd.DataFrame(
            np.random.default_rng(2).standard_normal(len(rng)),
            index=rng,
            columns=['a']
        )
        df_moscow: pd.DataFrame = df.tz_convert('Europe/Moscow')
        result: pd.DataFrame = df + df_moscow
        assert result.index.tz is timezone.utc
        result = df_moscow + df
        assert result.index.tz is timezone.utc

    def test_align_frame(self) -> None:
        rng: pd.PeriodIndex = pd.period_range('1/1/2000', '1/1/2010', freq='Y')
        ts: pd.DataFrame = pd.DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng)
        result: pd.DataFrame = ts + ts[::2]
        expected: pd.DataFrame = ts + ts
        expected.iloc[1::2] = np.nan
        tm.assert_frame_equal(result, expected)
        half: pd.DataFrame = ts[::2]
        result = ts + half.take(np.random.default_rng(2).permutation(len(half)))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv])
    def test_operators_none_as_na(self, op: Callable, float_frame: pd.DataFrame, mixed_float_frame: pd.DataFrame, mixed_int_frame: pd.DataFrame) -> None:
        df: pd.DataFrame = pd.DataFrame({'col1': [2, 5.0, 123, None], 'col2': [1, 2, 3, 4]}, dtype=object)
        filled: pd.DataFrame = df.fillna(np.nan)
        result: pd.DataFrame = op(df, 3)
        expected: pd.DataFrame = op(filled, 3).astype(object)
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

    @pytest.mark.parametrize('op,res', [
        ('__eq__', False),
        ('__ne__', True)
    ])
    @pytest.mark.filterwarnings('ignore:elementwise:FutureWarning')
    def test_logical_typeerror_with_non_valid(self, op: str, res: bool, float_frame: pd.DataFrame) -> None:
        result: pd.DataFrame = getattr(float_frame, op)('foo')
        assert bool(result.all().all()) is res

    @pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'div', 'truediv'])
    def test_binary_ops_align(self, op: str) -> None:
        index: pd.MultiIndex = MultiIndex.from_product(
            [['a', 'b', 'c'], ['one', 'two', 'three'], [1, 2, 3]],
            names=['first', 'second', 'third']
        )
        df: pd.DataFrame = pd.DataFrame(
            np.arange(27 * 3).reshape(27, 3),
            index=index,
            columns=['value1', 'value2', 'value3']
        ).sort_index()
        idx: pd.IndexSlice = pd.IndexSlice
        opa: Optional[Callable] = getattr(operator, op, None)
        if opa is None:
            return
        x: pd.Series = Series([1.0, 10.0, 100.0], [1, 2, 3])
        result: pd.DataFrame = getattr(df, op)(x, level='third', axis=0)
        expected: pd.DataFrame = pd.concat([
            opa(df.loc[idx[:, :, i], :], v) for i, v in x.items()
        ]).sort_index()
        tm.assert_frame_equal(result, expected)
        x = Series([1.0, 10.0], ['two', 'three'])
        result = getattr(df, op)(x, level='second', axis=0)
        expected = pd.concat([
            opa(df.loc[idx[:, i], :], v) for i, v in x.items()
        ]).reindex_like(df).sort_index()
        tm.assert_frame_equal(result, expected)

    def test_binary_ops_align_series_dataframe(self) -> None:
        midx: pd.MultiIndex = MultiIndex.from_product([['A', 'B'], ['a', 'b']])
        df: pd.DataFrame = pd.DataFrame(np.ones((2, 4), dtype='int64'), columns=midx)
        s: pd.Series = Series({'a': 1, 'b': 2})
        df2: pd.DataFrame = df.copy()
        df2.columns.names = ['lvl0', 'lvl1']
        s2: pd.Series = s.copy()
        s2.index.name = 'lvl1'
        res1: pd.DataFrame = df.mul(s, axis=1, level='lvl1')
        res2: pd.DataFrame = df.mul(s2, axis=1, level='lvl1')
        res3: pd.DataFrame = df2.mul(s, axis=1, level='lvl1')
        res4: pd.DataFrame = df2.mul(s2, axis=1, level='lvl1')
        res5: pd.DataFrame = df2.mul(s, axis=1, level='lvl1')
        res6: pd.DataFrame = df2.mul(s2, axis=1, level='lvl1')
        exp: pd.DataFrame = pd.DataFrame(
            np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype='int64'),
            columns=midx
        )
        for res in [res1, res2]:
            tm.assert_frame_equal(res, exp)
        exp.columns.names = ['lvl0', 'lvl1']
        for res in [res3, res4, res5, res6]:
            tm.assert_frame_equal(res, exp)

    def test_add_with_dti_mismatched_tzs(self) -> None:
        base: pd.DatetimeIndex = pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], tz='UTC')
        idx1: pd.DatetimeIndex = base.tz_convert('Asia/Tokyo')[:2]
        idx2: pd.DatetimeIndex = base.tz_convert('US/Eastern')[1:]
        df1: pd.DataFrame = pd.DataFrame({'A': [1, 2]}, index=idx1)
        df2: pd.DataFrame = pd.DataFrame({'A': [1, 1]}, index=idx2)
        exp: pd.DataFrame = pd.DataFrame({'A': [np.nan, 3, np.nan]}, index=base)
        tm.assert_frame_equal(df1 + df2, exp)

    def test_combineFrame(
        self,
        float_frame: pd.DataFrame,
        mixed_float_frame: pd.DataFrame,
        mixed_int_frame: pd.DataFrame
    ) -> None:
        frame_copy: pd.DataFrame = float_frame.reindex(float_frame.index[::2])
        del frame_copy['D']
        frame_copy.loc[:frame_copy.index[4], 'C'] = np.nan
        added: pd.DataFrame = float_frame + frame_copy
        indexer: pd.Index = added['A'].dropna().index
        exp: pd.Series = (float_frame['A'] * 2).copy()
        tm.assert_series_equal(added['A'].dropna(), exp.loc[indexer])
        exp.loc[~exp.index.isin(indexer)] = np.nan
        tm.assert_series_equal(added['A'], exp.loc[added['A'].index])
        assert np.isnan(added['C'].reindex(frame_copy.index)[:5]).all()
        assert np.isnan(added['D']).all()
        self_added: pd.DataFrame = float_frame + float_frame
        tm.assert_index_equal(self_added.index, float_frame.index)
        added_rev: pd.DataFrame = frame_copy + float_frame
        assert np.isnan(added['D']).all()
        assert np.isnan(added_rev['D']).all()
        plus_empty: pd.DataFrame = float_frame + pd.DataFrame()
        assert np.isnan(plus_empty.values).all()
        empty_plus: pd.DataFrame = pd.DataFrame() + float_frame
        assert np.isnan(empty_plus.values).all()
        empty_empty: pd.DataFrame = pd.DataFrame() + pd.DataFrame()
        assert empty_empty.empty
        reverse: pd.DataFrame = float_frame.reindex(columns=float_frame.columns[::-1])
        tm.assert_frame_equal(reverse + float_frame, float_frame * 2)
        added: pd.DataFrame = float_frame + mixed_float_frame
        _check_mixed_float(added, dtype='float64')
        added = mixed_float_frame + float_frame
        _check_mixed_float(added, dtype='float64')
        added = mixed_float_frame + mixed_float_frame
        _check_mixed_float(added, dtype={'C': None})
        added = float_frame + mixed_int_frame
        _check_mixed_float(added, dtype='float64')

    def test_combine_series(
        self,
        float_frame: pd.DataFrame,
        mixed_float_frame: pd.DataFrame,
        mixed_int_frame: pd.DataFrame
    ) -> None:
        series: pd.Series = float_frame.xs(float_frame.index[0])
        added: pd.DataFrame = float_frame + series
        for key, s in added.items():
            tm.assert_series_equal(s, float_frame[key] + series[key])
        larger_series: pd.Series = Series(series.to_dict())
        larger_series['E'] = 1
        larger_series = Series(larger_series)
        larger_added: pd.DataFrame = float_frame + larger_series
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

    def test_combine_timeseries(self, datetime_frame: pd.DataFrame) -> None:
        ts: pd.Series = datetime_frame['A']
        added: pd.DataFrame = datetime_frame.add(ts, axis='index')
        for key, col in datetime_frame.items():
            result: pd.Series = col + ts
            tm.assert_series_equal(added[key], result, check_names=False)
            assert added[key].name == key
            if col.name == ts.name:
                assert result.name == 'A'
            else:
                assert result.name is None
        smaller_frame: pd.DataFrame = datetime_frame[:-5]
        smaller_added: pd.DataFrame = smaller_frame.add(ts, axis='index')
        tm.assert_index_equal(smaller_added.index, datetime_frame.index)
        smaller_ts: pd.Series = ts[:-5]
        smaller_added2: pd.DataFrame = datetime_frame.add(smaller_ts, axis='index')
        tm.assert_frame_equal(smaller_added, smaller_added2)
        result: pd.DataFrame = datetime_frame.add(ts[:0], axis='index')
        expected: pd.DataFrame = pd.DataFrame(np.nan, index=datetime_frame.index, columns=datetime_frame.columns)
        tm.assert_frame_equal(result, expected)
        result = datetime_frame[:0].add(ts, axis='index')
        tm.assert_frame_equal(result, expected)
        frame: pd.DataFrame = datetime_frame[:1].reindex(columns=[])
        result = frame.mul(ts, axis='index')
        assert len(result) == len(ts)

    def test_combineFunc(
        self,
        float_frame: pd.DataFrame,
        mixed_float_frame: pd.DataFrame
    ) -> None:
        result: pd.DataFrame = float_frame * 2
        tm.assert_numpy_array_equal(result.values, float_frame.values * 2)
        result = mixed_float_frame * 2
        for c, s in result.items():
            tm.assert_numpy_array_equal(s.values, mixed_float_frame[c].values * 2)
        _check_mixed_float(result, dtype={'C': None})
        result = pd.DataFrame() * 2
        assert result.index.equals(pd.DataFrame().index)
        assert len(result.columns) == 0

    @pytest.mark.parametrize(
        'func',
        [
            operator.eq,
            operator.ne,
            operator.lt,
            operator.gt,
            operator.ge,
            operator.le
        ]
    )
    def test_comparisons(self, simple_frame: pd.DataFrame, float_frame: pd.DataFrame, func: Callable) -> None:
        df1: pd.DataFrame = pd.DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=pd.date_range('2000-01-01', periods=30, freq='B')
        )
        df2: pd.DataFrame = df1.copy()
        row: pd.Series = simple_frame.xs('a')
        ndim_5: np.ndarray = np.ones(df1.shape + (1, 1, 1))
        result: pd.DataFrame = func(df1, df2)
        tm.assert_numpy_array_equal(result.values, func(df1.values, df2.values))
        msg: str = 'Unable to coerce to Series/DataFrame, dimension must be <= 2: (30, 4, 1, 1, 1)'
        with pytest.raises(ValueError, match=re.escape(msg)):
            func(df1, ndim_5)
        result2: pd.DataFrame = func(simple_frame, row)
        tm.assert_numpy_array_equal(result2.values, func(simple_frame.values, row.values))
        result3: pd.DataFrame = func(float_frame, 0)
        tm.assert_numpy_array_equal(result3.values, func(float_frame.values, 0))
        msg = 'Can only compare identically-labeled \\(both index and columns\\) DataFrame objects'
        with pytest.raises(ValueError, match=msg):
            func(simple_frame, simple_frame[:2])

    def test_strings_to_numbers_comparisons_raises(self, compare_operators_no_eq_ne: Callable) -> None:
        df: pd.DataFrame = pd.DataFrame({
            x: {'x': 'foo', 'y': 'bar', 'z': 'baz'} for x in ['a', 'b', 'c']
        })
        f: Callable = getattr(operator, compare_operators_no_eq_ne)
        msg: str = '|'.join([
            "'[<>]=?' not supported between instances of 'str' and 'int'",
            'Invalid comparison between dtype=str and int'
        ])
        with pytest.raises(TypeError, match=msg):
            f(df, 0)

    def test_comparison_protected_from_errstate(self) -> None:
        missing_df: pd.DataFrame = pd.DataFrame(
            np.ones((10, 4), dtype=np.float64),
            columns=Index(list('ABCD'), dtype=object)
        )
        missing_df.loc[missing_df.index[0], 'A'] = np.nan
        with np.errstate(invalid='ignore'):
            expected: np.ndarray = missing_df.values < 0
        with np.errstate(invalid='raise'):
            result: np.ndarray = (missing_df < 0).values
        tm.assert_numpy_array_equal(result, expected)

    def test_boolean_comparison(self) -> None:
        df: pd.DataFrame = pd.DataFrame(np.arange(6).reshape((3, 2)))
        b: np.ndarray = np.array([2, 2])
        b_r: np.ndarray = np.atleast_2d([2, 2])
        b_c: np.ndarray = b_r.T
        lst: List[int] = [2, 2, 2]
        tup: Tuple[int, ...] = tuple(lst)
        expected: pd.DataFrame = pd.DataFrame([[False, False], [False, True], [True, True]])
        result: pd.DataFrame = df > b
        tm.assert_frame_equal(result, expected)
        result = df.values > b
        tm.assert_numpy_array_equal(result, expected.values)
        msg1d: str = 'Unable to coerce to Series, length must be 2: given 3'
        msg2d: str = 'Unable to coerce to DataFrame, shape must be'
        msg2db: str = 'operands could not be broadcast together with shapes'
        with pytest.raises(ValueError, match=msg1d):
            df > lst
        with pytest.raises(ValueError, match=msg1d):
            df > tup
        result = df > b_r
        tm.assert_frame_equal(result, expected)
        result = df.values > b_r
        tm.assert_numpy_array_equal(result, expected.values)
        with pytest.raises(ValueError, match=msg2d):
            df > b_c
        with pytest.raises(ValueError, match=msg2db):
            df.values > b_c
        expected = pd.DataFrame([[False, False], [True, False], [False, False]])
        result = df == b
        tm.assert_frame_equal(result, expected)
        with pytest.raises(ValueError, match=msg1d):
            df == lst
        with pytest.raises(ValueError, match=msg1d):
            df == tup
        result = df == b_r
        tm.assert_frame_equal(result, expected)
        result = df.values == b_r
        tm.assert_numpy_array_equal(result, expected.values)
        with pytest.raises(ValueError, match=msg2d):
            df == b_c
        assert df.values.shape != b_c.shape
        df = pd.DataFrame(np.arange(6).reshape((3, 2)), columns=list('AB'), index=list('abc'))
        expected.index = df.index
        expected.columns = df.columns
        with pytest.raises(ValueError, match=msg1d):
            df == lst
        with pytest.raises(ValueError, match=msg1d):
            df == tup

    def test_inplace_ops_alignment(self) -> None:
        columns: List[str] = list('abcdefg')
        X_orig: pd.DataFrame = pd.DataFrame(
            np.arange(10 * len(columns)).reshape(-1, len(columns)),
            columns=columns,
            index=range(10)
        )
        Z: pd.DataFrame = 100 * X_orig.iloc[:, 1:-1].copy()
        block1: List[str] = list('bedcf')
        subs: List[str] = list('bcdef')
        X: pd.DataFrame = X_orig.copy()
        result1: pd.DataFrame = (X[block1] + Z).reindex(columns=subs)
        X[block1] += Z
        result2: pd.DataFrame = X.reindex(columns=subs)
        X = X_orig.copy()
        result3: pd.DataFrame = (X[block1] + Z[block1]).reindex(columns=subs)
        X[block1] += Z[block1]
        result4: pd.DataFrame = X.reindex(columns=subs)
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
        s_orig: pd.Series = Series([1, 2, 3])
        df_orig: pd.DataFrame = pd.DataFrame(np.random.default_rng(2).integers(0, 5, size=10).reshape(-1, 5))
        s: pd.Series = s_orig.copy()
        s2: pd.Series = s
        s += 1
        tm.assert_series_equal(s, s2)
        tm.assert_series_equal(s_orig + 1, s)
        assert s is s2
        assert s._mgr is s2._mgr
        df: pd.DataFrame = df_orig.copy()
        df2: pd.DataFrame = df
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
        assert s is s2
        assert s._mgr is s2._mgr
        df = df_orig.copy()
        df2 = df
        df += 1.5
        tm.assert_frame_equal(df, df2)
        tm.assert_frame_equal(df_orig + 1.5, df)
        assert df is df2
        assert df._mgr is df2._mgr
        arr: np.ndarray = np.random.default_rng(2).integers(0, 10, size=5)
        df_orig = pd.DataFrame({'A': arr.copy(), 'B': 'foo'})
        df = df_orig.copy()
        df2 = df
        df['A'] += 1
        expected: pd.DataFrame = pd.DataFrame({'A': arr.copy() + 1, 'B': 'foo'})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df2, expected)
        assert df._mgr is df2._mgr
        df = df_orig.copy()
        df2 = df
        df['A'] += 1.5
        expected = pd.DataFrame({'A': arr.copy() + 1.5, 'B': 'foo'})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df2, expected)
        assert df._mgr is df2._mgr

    @pytest.mark.parametrize('op', ['add', 'and', pytest.param('div', marks=pytest.mark.xfail(raises=AttributeError, reason='__idiv__ not implemented')), 'floordiv', 'mod', 'mul', 'or', 'pow', 'sub', 'truediv', 'xor'])
    def test_inplace_ops_identity2(self, op: str) -> None:
        df: pd.DataFrame = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1, 2, 3]})
        operand: Union[int, float, complex] = 2
        if op in ('and', 'or', 'xor'):
            df['a'] = [True, False, True]
        df_copy: pd.DataFrame = df.copy()
        iop: str = f'__i{op}__'
        op_method: str = f'__{op}__'
        getattr(df, iop)(operand)
        expected: pd.DataFrame = getattr(df_copy, op_method)(operand)
        tm.assert_frame_equal(df, expected)
        expected = id(df)
        assert id(df) == expected

    @pytest.mark.parametrize('val', [[1, 2, 3], (1, 2, 3), np.array([1, 2, 3], dtype=np.int64), range(1, 4)])
    def test_alignment_non_pandas(self, val: Union[List[int], Tuple[int, ...], np.ndarray, range]) -> None:
        index: List[str] = ['A', 'B', 'C']
        columns: List[str] = ['X', 'Y', 'Z']
        df: pd.DataFrame = pd.DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
        align = DataFrame._align_for_op
        expected: pd.DataFrame = pd.DataFrame({'X': val, 'Y': val, 'Z': val}, index=df.index)
        tm.assert_frame_equal(align(df, val, axis=0)[1], expected)
        expected = pd.DataFrame({'X': [1, 1, 1], 'Y': [2, 2, 2], 'Z': [3, 3, 3]}, index=df.index)
        tm.assert_frame_equal(align(df, val, axis=1)[1], expected)

    @pytest.mark.parametrize('val', [[1, 2], (1, 2), np.array([1, 2]), range(1, 3)])
    def test_alignment_non_pandas_length_mismatch(self, val: Union[List[int], Tuple[int, ...], np.ndarray, range]) -> None:
        index: List[str] = ['A', 'B', 'C']
        columns: List[str] = ['X', 'Y', 'Z']
        df: pd.DataFrame = pd.DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
        align = DataFrame._align_for_op
        msg: str = 'Unable to coerce to Series, length must be 3: given 2'
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

    def test_alignment_non_pandas_index_columns(self) -> None:
        index: List[str] = ['A', 'B', 'C']
        columns: List[str] = ['X', 'Y', 'Z']
        df: pd.DataFrame = pd.DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
        align = DataFrame._align_for_op
        val: np.ndarray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tm.assert_frame_equal(align(df, val, axis=0)[1], pd.DataFrame(val, index=df.index, columns=df.columns))
        tm.assert_frame_equal(align(df, val, axis=1)[1], pd.DataFrame(val, index=df.index, columns=df.columns))
        msg: str = 'Unable to coerce to DataFrame, shape must be'
        val = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)
        val = np.zeros((3, 3, 3))
        msg = re.escape('Unable to coerce to Series/DataFrame, dimension must be <= 2: (3, 3, 3)')
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

    def test_no_warning(self, all_arithmetic_operators: Callable) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': [0.0, 0.0], 'B': [0.0, None]})
        b: pd.Series = df['B']
        with tm.assert_produces_warning(None):
            getattr(df, all_arithmetic_operators.__name__)(b)

    def test_dunder_methods_binary(self, all_arithmetic_operators: Callable) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': [0.0, 0.0], 'B': [0.0, None]})
        b: pd.Series = df['B']
        with pytest.raises(TypeError, match='takes 2 positional arguments'):
            getattr(df, all_arithmetic_operators.__name__)(b, 0)

    def test_align_int_fill_bug(self) -> None:
        X: np.ndarray = np.arange(10 * 10, dtype='float64').reshape(10, 10)
        Y: np.ndarray = np.ones((10, 1), dtype=int)
        df1: pd.DataFrame = pd.DataFrame(X)
        df1['0.X'] = Y.squeeze()
        df2: pd.DataFrame = df1.astype(float)
        result: pd.DataFrame = df1 - df1.mean()
        expected: pd.DataFrame = df2 - df2.mean()
        tm.assert_frame_equal(result, expected)


def test_pow_with_realignment() -> None:
    left: pd.DataFrame = pd.DataFrame({'A': [0, 1, 2]})
    right: pd.DataFrame = pd.DataFrame(index=[0, 1, 2])
    result: pd.DataFrame = left ** right
    expected: pd.DataFrame = pd.DataFrame({'A': [np.nan, 1.0, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_dataframe_series_extension_dtypes() -> None:
    df: pd.DataFrame = pd.DataFrame(
        np.random.default_rng(2).integers(0, 100, (10, 3)),
        columns=['a', 'b', 'c']
    )
    ser: pd.Series = Series([1, 2, 3], index=['a', 'b', 'c'])
    expected: pd.DataFrame = pd.DataFrame(
        df.to_numpy('int64') + ser.to_numpy('int64').reshape(-1, 3),
        columns=df.columns,
        dtype='Int64'
    )
    df_ea: pd.DataFrame = df.astype('Int64')
    result: pd.DataFrame = df_ea + ser
    tm.assert_frame_equal(result, expected)
    result = df_ea + ser.astype('Int64')
    tm.assert_frame_equal(result, expected)


def test_dataframe_blockwise_slicelike() -> None:
    arr: np.ndarray = np.random.default_rng(2).integers(0, 1000, (100, 10))
    df1: pd.DataFrame = pd.DataFrame(arr)
    df2: pd.DataFrame = df1.copy().astype({1: 'float', 3: 'float', 7: 'float'})
    df2.iloc[0, [1, 3, 7]] = np.nan
    df3: pd.DataFrame = df1.copy().astype({5: 'float'})
    df3.iloc[0, [5]] = np.nan
    df4: pd.DataFrame = df1.copy().astype({2: 'float', 3: 'float', 4: 'float'})
    df4.iloc[0, np.arange(2, 5)] = np.nan
    df5: pd.DataFrame = df1.copy().astype({4: 'float', 5: 'float', 6: 'float'})
    df5.iloc[0, np.arange(4, 7)] = np.nan
    for left, right in [(df1, df2), (df2, df3), (df4, df5)]:
        res: pd.DataFrame = left + right
        expected: pd.DataFrame = pd.DataFrame({i: left[i] + right[i] for i in left.columns})
        tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    'df, col_dtype',
    [
        (
            pd.DataFrame([[1.0, 2.0], [4.0, 5.0]], columns=list('ab')),
            'float64'
        ),
        (
            pd.DataFrame([[1.0, 'b'], [4.0, 'b']], columns=list('ab')).astype({'b': object}),
            'object'
        )
    ]
)
def test_dataframe_operation_with_non_numeric_types(df: pd.DataFrame, col_dtype: str) -> None:
    expected: pd.DataFrame = pd.DataFrame([[0.0, np.nan], [3.0, np.nan]], columns=list('ab'))
    expected = expected.astype({'b': col_dtype})
    result: pd.DataFrame = df + Series([-1.0], index=list('a'))
    tm.assert_frame_equal(result, expected)
    result = df + Series([-1.0], index=list('a'))
    tm.assert_frame_equal(result, expected)


def test_arith_reindex_with_duplicates() -> None:
    df1: pd.DataFrame = pd.DataFrame(data=[[0]], columns=['second'])
    df2: pd.DataFrame = pd.DataFrame(data=[[0, 0, 0]], columns=['first', 'second', 'second'])
    result: pd.DataFrame = df1 + df2
    expected: pd.DataFrame = pd.DataFrame([[np.nan, 0, 0]], columns=['first', 'second', 'second'])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('to_add', [
    [Series([1, 1])],
    [Series([1, 1]), Series([1, 1])]
])
def test_arith_list_of_arraylike_raise(to_add: List[pd.Series]) -> None:
    df: pd.DataFrame = pd.DataFrame({'x': [1, 2], 'y': [1, 2]})
    msg: str = f'Unable to coerce list of {type(to_add[0])} to Series/DataFrame'
    with pytest.raises(ValueError, match=msg):
        df + to_add
    with pytest.raises(ValueError, match=msg):
        to_add + df


def test_inplace_arithmetic_series_update() -> None:
    df: pd.DataFrame = pd.DataFrame({'A': [1, 2, 3]})
    df_orig: pd.DataFrame = df.copy()
    series: pd.Series = df['A']
    vals: np.ndarray = series._values
    series += 1
    assert series._values is not vals
    tm.assert_frame_equal(df, df_orig)


def test_arithmetic_multiindex_align() -> None:
    """
    Regression test for: https://github.com/pandas-dev/pandas/issues/33765
    """
    df1: pd.DataFrame = pd.DataFrame(
        [[1]], 
        index=['a'], 
        columns=MultiIndex.from_product([[0], [1]], names=['a', 'b'])
    )
    df2: pd.DataFrame = pd.DataFrame(
        [[1]], 
        index=['a'], 
        columns=Index([0], name='a')
    )
    expected: pd.DataFrame = pd.DataFrame(
        [[0]], 
        index=['a'], 
        columns=MultiIndex.from_product([[0], [1]], names=['a', 'b'])
    )
    result: pd.DataFrame = df1 - df2
    tm.assert_frame_equal(result, expected)


def test_arithmetic_multiindex_column_align() -> None:
    df1: pd.DataFrame = pd.DataFrame(
        data=100, 
        columns=MultiIndex.from_product([['1A', '1B'], ['2A', '2B']], names=['Lev1', 'Lev2']), 
        index=['C1', 'C2']
    )
    df2: pd.DataFrame = pd.DataFrame(
        data=np.array([[0.1, 0.25], [0.2, 0.45]]), 
        columns=MultiIndex.from_product([['1A', '1B']], names=['Lev1']), 
        index=['C1', 'C2']
    )
    expected: pd.DataFrame = pd.DataFrame(
        data=np.array([[10.0, 10.0, 25.0, 25.0], [20.0, 20.0, 45.0, 45.0]]),
        columns=MultiIndex.from_product([['1A', '1B'], ['2A', '2B']], names=['Lev1', 'Lev2']),
        index=['C1', 'C2']
    )
    result: pd.DataFrame = df1 * df2
    tm.assert_frame_equal(result, expected)


def test_arithmetic_multiindex_column_align_with_fillvalue() -> None:
    df1: pd.DataFrame = pd.DataFrame(
        data=[[1.0, 2.0]], 
        columns=MultiIndex.from_tuples([('A', 'one'), ('A', 'two')])
    )
    df2: pd.DataFrame = pd.DataFrame(
        data=[[3.0, 4.0]], 
        columns=MultiIndex.from_tuples([('B', 'one'), ('B', 'two')])
    )
    expected: pd.DataFrame = pd.DataFrame(
        data=[[1.0, 2.0, 3.0, 4.0]],
        columns=MultiIndex.from_tuples([('A', 'one'), ('A', 'two'), ('B', 'one'), ('B', 'two')])
    )
    result: pd.DataFrame = df1.add(df2, fill_value=0)
    tm.assert_frame_equal(result, expected)


def test_bool_frame_mult_float() -> None:
    df: pd.DataFrame = pd.DataFrame(True, list('ab'), list('cd'))
    result: pd.DataFrame = df * 1.0
    expected: pd.DataFrame = pd.DataFrame(np.ones((2, 2)), list('ab'), list('cd'))
    tm.assert_frame_equal(result, expected)


def test_frame_sub_nullable_int(any_int_ea_dtype: str) -> None:
    series1: pd.Series = Series([1, 2, None], dtype=any_int_ea_dtype)
    series2: pd.Series = Series([1, 2, 3], dtype=any_int_ea_dtype)
    expected: pd.DataFrame = pd.DataFrame([0, 0, None], dtype=any_int_ea_dtype)
    result: pd.DataFrame = series1.to_frame() - series2.to_frame()
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings('ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning')
def test_frame_op_subclass_nonclass_constructor() -> None:

    class SubclassedSeries(Series):

        @property
        def _constructor(self) -> Callable[..., 'SubclassedSeries']:
            return SubclassedSeries

        @property
        def _constructor_expanddim(self) -> Callable[..., 'SubclassedDataFrame']:
            return SubclassedDataFrame

    class SubclassedDataFrame(DataFrame):
        _metadata: List[str] = ['my_extra_data']

        def __init__(self, my_extra_data: str, *args: Any, **kwargs: Any) -> None:
            self.my_extra_data: str = my_extra_data
            super().__init__(*args, **kwargs)

        @property
        def _constructor(self) -> Callable[..., 'SubclassedDataFrame']:
            return functools.partial(type(self), self.my_extra_data)

        @property
        def _constructor_sliced(self) -> Callable[..., 'SubclassedSeries']:
            return SubclassedSeries

    sdf: SubclassedDataFrame = SubclassedDataFrame('some_data', {'A': [1, 2, 3], 'B': [4, 5, 6]})
    result: SubclassedDataFrame = sdf * 2
    expected: SubclassedDataFrame = SubclassedDataFrame('some_data', {'A': [2, 4, 6], 'B': [8, 10, 12]})
    tm.assert_frame_equal(result, expected)
    result = sdf + sdf
    tm.assert_frame_equal(result, expected)


def test_enum_column_equality() -> None:
    Cols = Enum('Cols', 'col1 col2')
    q1: pd.DataFrame = pd.DataFrame({Cols.col1: [1, 2, 3]})
    q2: pd.DataFrame = pd.DataFrame({Cols.col1: [1, 2, 3]})
    result: pd.Series = q1[Cols.col1] == q2[Cols.col1]
    expected: pd.Series = Series([True, True, True], name=Cols.col1)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('using_infer_string', [True, False])
@pytest.mark.parametrize('df, col_dtype', [
    (
        pd.DataFrame([[1.0, 2.0], [4.0, 5.0]], columns=list('ab')),
        'float64'
    ),
    (
        pd.DataFrame([[1.0, 'b'], [4.0, 'b']], columns=list('ab')).astype({'b': object}),
        'object'
    )
])
def test_mixed_col_index_dtype(using_infer_string: bool, df: pd.DataFrame, col_dtype: str) -> None:
    expected: pd.DataFrame = pd.DataFrame([[0.0, np.nan], [3.0, np.nan]], columns=list('ab'))
    result: pd.DataFrame = df + Series([-1.0], index=list('a'))
    tm.assert_frame_equal(result, expected)
    result = df + Series([-1.0], index=list('a'))
    tm.assert_frame_equal(result, expected)
    if using_infer_string:
        if HAS_PYARROW:
            dtype: str = 'string[pyarrow]'
        else:
            dtype = 'string'
        expected.columns = expected.columns.astype(dtype)
    tm.assert_frame_equal(result, expected)
