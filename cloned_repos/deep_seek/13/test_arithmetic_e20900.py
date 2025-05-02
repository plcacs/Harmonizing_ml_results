from collections import deque
from datetime import datetime, timezone
from enum import Enum
import functools
import operator
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return DataFrame(arr, columns=['one', 'two', 'three'], index=['a', 'b', 'c'])

@pytest.fixture(autouse=True, params=[0, 100], ids=['numexpr', 'python'])
def switch_numexpr_min_elements(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int:
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', request.param)
        yield request.param

class DummyElement:
    def __init__(self, value: Any, dtype: str) -> None:
        self.value = value
        self.dtype = np.dtype(dtype)

    def __array__(self, dtype: Optional[str] = None, copy: Optional[bool] = None) -> np.ndarray:
        return np.array(self.value, dtype=self.dtype)

    def __str__(self) -> str:
        return f'DummyElement({self.value}, {self.dtype})'

    def __repr__(self) -> str:
        return str(self)

    def astype(self, dtype: str, copy: bool = False) -> 'DummyElement':
        self.dtype = dtype
        return self

    def view(self, dtype: str) -> 'DummyElement':
        return type(self)(self.value.view(dtype), dtype)

    def any(self, axis: Optional[int] = None) -> bool:
        return bool(self.value)

class TestFrameComparisons:
    def test_comparison_with_categorical_dtype(self) -> None:
        df = DataFrame({'A': ['foo', 'bar', 'baz']})
        exp = DataFrame({'A': [True, False, False]})
        res = df == 'foo'
        tm.assert_frame_equal(res, exp)
        df['A'] = df['A'].astype('category')
        res = df == 'foo'
        tm.assert_frame_equal(res, exp)

    def test_frame_in_list(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), columns=list('ABCD'))
        msg = 'The truth value of a DataFrame is ambiguous'
        with pytest.raises(ValueError, match=msg):
            df in [None]

    @pytest.mark.parametrize('arg, arg2', [[{'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}], [{'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}], [{'a': pd.date_range('20010101', periods=10), 'b': pd.date_range('20010101', periods=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}], [{'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}, {'a': pd.date_range('20010101', periods=10), 'b': pd.date_range('20010101', periods=10)}]])
    def test_comparison_invalid(self, arg: Dict[str, Union[np.ndarray, pd.DatetimeIndex]], arg2: Dict[str, Union[np.ndarray, pd.DatetimeIndex]]) -> None:
        x = DataFrame(arg)
        y = DataFrame(arg2)
        result = x == y
        expected = DataFrame({col: x[col] == y[col] for col in x.columns}, index=x.index, columns=x.columns)
        tm.assert_frame_equal(result, expected)
        result = x != y
        expected = DataFrame({col: x[col] != y[col] for col in x.columns}, index=x.index, columns=x.columns)
        tm.assert_frame_equal(result, expected)
        msgs = ['Invalid comparison between dtype=datetime64\\[ns\\] and ndarray', 'invalid type promotion', "The DTypes <class 'numpy.dtype\\[.*\\]'> and <class 'numpy.dtype\\[.*\\]'> do not have a common DType."]
        msg = '|'.join(msgs)
        with pytest.raises(TypeError, match=msg):
            x >= y
        with pytest.raises(TypeError, match=msg):
            x > y
        with pytest.raises(TypeError, match=msg):
            x < y
        with pytest.raises(TypeError, match=msg):
            x <= y

    @pytest.mark.parametrize('left, right', [('gt', 'lt'), ('lt', 'gt'), ('ge', 'le'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
    def test_timestamp_compare(self, left: str, right: str) -> None:
        df = DataFrame({'dates1': pd.date_range('20010101', periods=10), 'dates2': pd.date_range('20010102', periods=10), 'intcol': np.random.default_rng(2).integers(1000000000, size=10), 'floatcol': np.random.default_rng(2).standard_normal(10), 'stringcol': [chr(100 + i) for i in range(10)]})
        df.loc[np.random.default_rng(2).random(len(df)) > 0.5, 'dates2'] = pd.NaT
        left_f = getattr(operator, left)
        right_f = getattr(operator, right)
        if left in ['eq', 'ne']:
            expected = left_f(df, pd.Timestamp('20010109'))
            result = right_f(pd.Timestamp('20010109'), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'Timestamp'"
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
        df = DataFrame([['1989-08-01', 1], ['1989-08-01', 2]])
        other = DataFrame([['a', 'b'], ['c', 'd']])
        result = df == other
        assert not result.any().any()
        result = df != other
        assert result.all().all()

    def test_df_boolean_comparison_error(self) -> None:
        df = DataFrame(np.arange(6).reshape((3, 2)))
        expected = DataFrame([[False, False], [True, False], [False, False]])
        result = df == (2, 2)
        tm.assert_frame_equal(result, expected)
        result = df == [2, 2]
        tm.assert_frame_equal(result, expected)

    def test_df_float_none_comparison(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((8, 3)), index=range(8), columns=['A', 'B', 'C'])
        result = df.__eq__(None)
        assert not result.any().any()

    def test_df_string_comparison(self) -> None:
        df = DataFrame([{'a': 1, 'b': 'foo'}, {'a': 2, 'b': 'bar'}])
        mask_a = df.a > 1
        tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
        tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])
        mask_b = df.b == 'foo'
        tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
        tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])

class TestFrameFlexComparisons:
    def test_bool_flex_frame(self, comparison_op: Any) -> None:
        data = np.random.default_rng(2).standard_normal((5, 3))
        other_data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        other = DataFrame(other_data)
        ndim_5 = np.ones(df.shape + (1, 3))
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
        msg = 'Unable to coerce to Series/DataFrame'
        tm.assert_frame_equal(f(np.nan), o(df, np.nan))
        with pytest.raises(ValueError, match=msg):
            f(ndim_5)

    @pytest.mark.parametrize('box', [np.array, Series])
    def test_bool_flex_series(self, box: Any) -> None:
        data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
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
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
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
        arr = np.array([np.nan, 1, 6, np.nan])
        arr2 = np.array([2j, np.nan, 7, None])
        df = DataFrame({'a': arr})
        df2 = DataFrame({'a': arr2})
        msg = '|'.join(["'>' not supported between instances of '.*' and 'complex'", 'unorderable types: .*complex\\(\\)'])
        with pytest.raises(TypeError, match=msg):
            df.gt(df2)
        with pytest.raises(TypeError, match=msg):
            df['a'].gt(df2['a'])
        with pytest.raises(TypeError, match=msg):
            df.values > df2.values
        rs = df.ne(df2)
        assert rs.values.all()
        arr3 = np.array([2j, np.nan, None])
        df3 = DataFrame({'a': arr3})
        with pytest.raises(TypeError, match=msg):
            df3.gt(2j)
        with pytest.raises(TypeError, match=msg):
            df3['a'].gt(2j)
        with pytest.raises(TypeError, match=msg):
            df3.values > 2j

    def test_bool_flex_frame_object_dtype(self) -> None:
        df1 = DataFrame({'col': ['foo', np.nan, 'bar']}, dtype=object)
        df2 = DataFrame({'col': ['foo', datetime.now(), 'bar']}, dtype=object)
        result = df1.ne(df2)
        exp = DataFrame({'col': [False, True, False]})
        tm.assert_frame_equal(result, exp)

    def test_flex_comparison_nat(self) -> None:
        df = DataFrame([pd.NaT])
        result = df == pd.NaT
        assert result.iloc[0, 0].item() is False
        result = df.eq(pd.NaT)
        assert result.iloc[0, 0].item() is False
        result = df != pd.NaT
        assert result.iloc[0, 0].item() is True
        result = df.ne(pd.NaT)
        assert result.iloc[0, 0].item() is True

    def test_df_flex_cmp_constant_return_types(self, comparison_op: Any) -> None:
        df = DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        const = 2
        result = getattr(df, comparison_op.__name__)(const).dtypes.value_counts()
        tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))

    def test_df_flex_cmp_constant_return_types_empty(self, comparison_op: Any) -> None:
        df = DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        const = 2
        empty = df.iloc[:0]
        result = getattr(empty, comparison_op.__name__)(const).dtypes.value_counts()
        tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))

    def test_df_flex_cmp_ea_dtype_with_ndarray_series(self) -> None:
        ii = pd.IntervalIndex.from_breaks([1, 2, 3])
        df = DataFrame({'A': ii, 'B': ii})
        ser = Series([0, 0])
        res = df.eq(ser, axis=0)
        expected = DataFrame({'A': [False, False], 'B': [False, False]})
        tm.assert_frame_equal(res, expected)
        ser2 = Series([1, 2], index=['A', 'B'])
        res2 = df.eq(ser2, axis=1)
        tm.assert