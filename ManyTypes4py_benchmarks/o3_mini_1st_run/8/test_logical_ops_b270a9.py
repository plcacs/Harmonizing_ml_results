from typing import Any, Callable, List, Union
import operator
import re
import numpy as np
import pytest
from pandas import CategoricalIndex, DataFrame, Interval, Series, isnull
import pandas._testing as tm

class TestDataFrameLogicalOperators:

    @pytest.mark.parametrize(
        'left, right, op, expected', [
            (
                [True, False, np.nan],
                [True, False, True],
                operator.and_,
                [True, False, False]
            ),
            (
                [True, False, True],
                [True, False, np.nan],
                operator.and_,
                [True, False, False]
            ),
            (
                [True, False, np.nan],
                [True, False, True],
                operator.or_,
                [True, False, False]
            ),
            (
                [True, False, True],
                [True, False, np.nan],
                operator.or_,
                [True, False, True]
            )
        ]
    )
    def test_logical_operators_nans(
        self,
        left: List[Any],
        right: List[Any],
        op: Callable[[Any, Any], Any],
        expected: List[Any],
        frame_or_series: Callable[[List[Any]], Union[DataFrame, Series]]
    ) -> None:
        result = op(frame_or_series(left), frame_or_series(right))
        expected_conv = frame_or_series(expected)
        tm.assert_equal(result, expected_conv)

    def test_logical_ops_empty_frame(self) -> None:
        df: DataFrame = DataFrame(index=[1])
        result: DataFrame = df & df
        tm.assert_frame_equal(result, df)
        result = df | df
        tm.assert_frame_equal(result, df)
        df2: DataFrame = DataFrame(index=[1, 2])
        result = df & df2
        tm.assert_frame_equal(result, df2)
        dfa: DataFrame = DataFrame(index=[1], columns=['A'])
        result = dfa & dfa
        expected: DataFrame = DataFrame(False, index=[1], columns=['A'])
        tm.assert_frame_equal(result, expected)

    def test_logical_ops_bool_frame(self) -> None:
        df1a_bool: DataFrame = DataFrame(True, index=[1], columns=['A'])
        result: DataFrame = df1a_bool & df1a_bool
        tm.assert_frame_equal(result, df1a_bool)
        result = df1a_bool | df1a_bool
        tm.assert_frame_equal(result, df1a_bool)

    def test_logical_ops_int_frame(self) -> None:
        df1a_int: DataFrame = DataFrame(1, index=[1], columns=['A'])
        df1a_bool: DataFrame = DataFrame(True, index=[1], columns=['A'])
        result: DataFrame = df1a_int | df1a_bool
        tm.assert_frame_equal(result, df1a_bool)
        res_ser: Series = df1a_int['A'] | df1a_bool['A']
        tm.assert_series_equal(res_ser, df1a_bool['A'])

    def test_logical_ops_invalid(self, using_infer_string: bool) -> None:
        df1: DataFrame = DataFrame(1.0, index=[1], columns=['A'])
        df2: DataFrame = DataFrame(True, index=[1], columns=['A'])
        msg: str = re.escape("unsupported operand type(s) for |: 'float' and 'bool'")
        with pytest.raises(TypeError, match=msg):
            df1 | df2
        df1 = DataFrame('foo', index=[1], columns=['A'])
        df2 = DataFrame(True, index=[1], columns=['A'])
        if using_infer_string and df1['A'].dtype.storage == 'pyarrow':
            msg = "operation 'or_' not supported for dtype 'str'"
        else:
            msg = re.escape("unsupported operand type(s) for |: 'str' and 'bool'")
        with pytest.raises(TypeError, match=msg):
            df1 | df2

    def test_logical_operators(self) -> None:

        def _check_bin_op(op: Callable[[Any, Any], Any]) -> None:
            result: DataFrame = op(df1, df2)
            expected: DataFrame = DataFrame(
                op(df1.values, df2.values),
                index=df1.index,
                columns=df1.columns
            )
            assert result.values.dtype == np.bool_
            tm.assert_frame_equal(result, expected)

        def _check_unary_op(op: Callable[[Any], Any]) -> None:
            result: DataFrame = op(df1)
            expected: DataFrame = DataFrame(
                op(df1.values),
                index=df1.index,
                columns=df1.columns
            )
            assert result.values.dtype == np.bool_
            tm.assert_frame_equal(result, expected)

        df1_dict: dict = {
            'a': {'a': True, 'b': False, 'c': False, 'd': True, 'e': True},
            'b': {'a': False, 'b': True, 'c': False, 'd': False, 'e': False},
            'c': {'a': False, 'b': False, 'c': True, 'd': False, 'e': False},
            'd': {'a': True, 'b': False, 'c': False, 'd': True, 'e': True},
            'e': {'a': True, 'b': False, 'c': False, 'd': True, 'e': True}
        }
        df2_dict: dict = {
            'a': {'a': True, 'b': False, 'c': True, 'd': False, 'e': False},
            'b': {'a': False, 'b': True, 'c': False, 'd': False, 'e': False},
            'c': {'a': True, 'b': False, 'c': True, 'd': False, 'e': False},
            'd': {'a': False, 'b': False, 'c': False, 'd': True, 'e': False},
            'e': {'a': False, 'b': False, 'c': False, 'd': False, 'e': True}
        }
        df1: DataFrame = DataFrame(df1_dict)
        df2: DataFrame = DataFrame(df2_dict)
        _check_bin_op(operator.and_)
        _check_bin_op(operator.or_)
        _check_bin_op(operator.xor)
        _check_unary_op(operator.inv)

    def test_logical_with_nas(self) -> None:
        d: DataFrame = DataFrame({'a': [np.nan, False], 'b': [True, True]})
        result: Series = d['a'] | d['b']
        expected: Series = Series([False, True])
        tm.assert_series_equal(result, expected)
        result = d['a'].fillna(False) | d['b']
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)
        result = d['a'].fillna(False) | d['b']
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

    def test_logical_ops_categorical_columns(self) -> None:
        intervals: List[Interval] = [Interval(1, 2), Interval(3, 4)]
        data: DataFrame = DataFrame(
            [[1, np.nan], [2, np.nan]],
            columns=CategoricalIndex(intervals, categories=intervals + [Interval(5, 6)])
        )
        mask: DataFrame = DataFrame(
            [[False, False], [False, False]],
            columns=data.columns,
            dtype=bool
        )
        result: DataFrame = mask | isnull(data)
        expected: DataFrame = DataFrame(
            [[False, True], [False, True]],
            columns=CategoricalIndex(intervals, categories=intervals + [Interval(5, 6)])
        )
        tm.assert_frame_equal(result, expected)

    def test_int_dtype_different_index_not_bool(self) -> None:
        df1: DataFrame = DataFrame([1, 2, 3], index=[10, 11, 23], columns=['a'])
        df2: DataFrame = DataFrame([10, 20, 30], index=[11, 10, 23], columns=['a'])
        result: DataFrame = np.bitwise_xor(df1, df2)
        expected: DataFrame = DataFrame([21, 8, 29], index=[10, 11, 23], columns=['a'])
        tm.assert_frame_equal(result, expected)
        result = df1 ^ df2
        tm.assert_frame_equal(result, expected)

    def test_different_dtypes_different_index_raises(self) -> None:
        df1: DataFrame = DataFrame([1, 2], index=['a', 'b'])
        df2: DataFrame = DataFrame([3, 4], index=['b', 'c'])
        with pytest.raises(TypeError, match='unsupported operand type'):
            _ = df1 & df2
