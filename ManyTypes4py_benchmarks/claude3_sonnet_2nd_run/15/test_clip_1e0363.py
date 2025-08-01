import numpy as np
import pytest
from pandas import DataFrame, Series
import pandas._testing as tm
from typing import Any, List, Optional, Union, Tuple, Sequence, cast

class TestDataFrameClip:

    def test_clip(self, float_frame: DataFrame) -> None:
        median = float_frame.median().median()
        original = float_frame.copy()
        double = float_frame.clip(upper=median, lower=median)
        assert not (double.values != median).any()
        assert (float_frame.values == original.values).all()

    def test_inplace_clip(self, float_frame: DataFrame) -> None:
        median = float_frame.median().median()
        frame_copy = float_frame.copy()
        return_value = frame_copy.clip(upper=median, lower=median, inplace=True)
        assert return_value is None
        assert not (frame_copy.values != median).any()

    def test_dataframe_clip(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        for lb, ub in [(-1, 1), (1, -1)]:
            clipped_df = df.clip(lb, ub)
            lb, ub = (min(lb, ub), max(ub, lb))
            lb_mask = df.values <= lb
            ub_mask = df.values >= ub
            mask = ~lb_mask & ~ub_mask
            assert (clipped_df.values[lb_mask] == lb).all()
            assert (clipped_df.values[ub_mask] == ub).all()
            assert (clipped_df.values[mask] == df.values[mask]).all()

    def test_clip_mixed_numeric(self) -> None:
        df = DataFrame({'A': [1, 2, 3], 'B': [1.0, np.nan, 3.0]})
        result = df.clip(1, 2)
        expected = DataFrame({'A': [1, 2, 2], 'B': [1.0, np.nan, 2.0]})
        tm.assert_frame_equal(result, expected)
        df = DataFrame([[1, 2, 3.4], [3, 4, 5.6]], columns=['foo', 'bar', 'baz'])
        expected = df.dtypes
        result = df.clip(upper=3).dtypes
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_clip_against_series(self, inplace: bool) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        lb = Series(np.random.default_rng(2).standard_normal(1000))
        ub = lb + 1
        original = df.copy()
        clipped_df = df.clip(lb, ub, axis=0, inplace=inplace)
        if inplace:
            clipped_df = df
        for i in range(2):
            lb_mask = original.iloc[:, i] <= lb
            ub_mask = original.iloc[:, i] >= ub
            mask = ~lb_mask & ~ub_mask
            result = clipped_df.loc[lb_mask, i]
            tm.assert_series_equal(result, lb[lb_mask], check_names=False)
            assert result.name == i
            result = clipped_df.loc[ub_mask, i]
            tm.assert_series_equal(result, ub[ub_mask], check_names=False)
            assert result.name == i
            tm.assert_series_equal(clipped_df.loc[mask, i], df.loc[mask, i])

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('lower', [[2, 3, 4], np.asarray([2, 3, 4])])
    @pytest.mark.parametrize('axis,res', [(0, [[2.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 7.0, 7.0]]), (1, [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])])
    def test_clip_against_list_like(self, inplace: bool, lower: Union[List[int], np.ndarray], axis: int, res: List[List[float]]) -> None:
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        original = DataFrame(arr, columns=['one', 'two', 'three'], index=['a', 'b', 'c'])
        result = original.clip(lower=lower, upper=[5, 6, 7], axis=axis, inplace=inplace)
        expected = DataFrame(res, columns=original.columns, index=original.index)
        if inplace:
            result = original
        tm.assert_frame_equal(result, expected, check_exact=True)

    @pytest.mark.parametrize('axis', [0, 1, None])
    def test_clip_against_frame(self, axis: Optional[int]) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        lb = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        ub = lb + 1
        clipped_df = df.clip(lb, ub, axis=axis)
        lb_mask = df <= lb
        ub_mask = df >= ub
        mask = ~lb_mask & ~ub_mask
        tm.assert_frame_equal(clipped_df[lb_mask], lb[lb_mask])
        tm.assert_frame_equal(clipped_df[ub_mask], ub[ub_mask])
        tm.assert_frame_equal(clipped_df[mask], df[mask])

    def test_clip_against_unordered_columns(self) -> None:
        df1 = DataFrame(np.random.default_rng(2).standard_normal((1000, 4)), columns=['A', 'B', 'C', 'D'])
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1000, 4)), columns=['D', 'A', 'B', 'C'])
        df3 = DataFrame(df2.values - 1, columns=['B', 'D', 'C', 'A'])
        result_upper = df1.clip(lower=0, upper=df2)
        expected_upper = df1.clip(lower=0, upper=df2[df1.columns])
        result_lower = df1.clip(lower=df3, upper=3)
        expected_lower = df1.clip(lower=df3[df1.columns], upper=3)
        result_lower_upper = df1.clip(lower=df3, upper=df2)
        expected_lower_upper = df1.clip(lower=df3[df1.columns], upper=df2[df1.columns])
        tm.assert_frame_equal(result_upper, expected_upper)
        tm.assert_frame_equal(result_lower, expected_lower)
        tm.assert_frame_equal(result_lower_upper, expected_lower_upper)

    def test_clip_with_na_args(self, float_frame: DataFrame) -> None:
        """Should process np.nan argument as None"""
        tm.assert_frame_equal(float_frame.clip(np.nan), float_frame)
        tm.assert_frame_equal(float_frame.clip(upper=np.nan, lower=np.nan), float_frame)
        df = DataFrame({'col_0': [1, 2, 3], 'col_1': [4, 5, 6], 'col_2': [7, 8, 9]})
        result = df.clip(lower=[4, 5, np.nan], axis=0)
        expected = DataFrame({'col_0': Series([4, 5, 3], dtype='float'), 'col_1': [4, 5, 6], 'col_2': [7, 8, 9]})
        tm.assert_frame_equal(result, expected)
        result = df.clip(lower=[4, 5, np.nan], axis=1)
        expected = DataFrame({'col_0': [4, 4, 4], 'col_1': [5, 5, 6], 'col_2': [7, 8, 9]})
        tm.assert_frame_equal(result, expected)
        data = {'col_0': [9, -3, 0, -1, 5], 'col_1': [-2, -7, 6, 8, -5]}
        df = DataFrame(data)
        t = Series([2, -4, np.nan, 6, 3])
        result = df.clip(lower=t, axis=0)
        expected = DataFrame({'col_0': [9, -3, 0, 6, 5], 'col_1': [2, -4, 6, 8, 3]}, dtype='float')
        tm.assert_frame_equal(result, expected)

    def test_clip_int_data_with_float_bound(self) -> None:
        df = DataFrame({'a': [1, 2, 3]})
        result = df.clip(lower=1.5)
        expected = DataFrame({'a': [1.5, 2.0, 3.0]})
        tm.assert_frame_equal(result, expected)

    def test_clip_with_list_bound(self) -> None:
        df = DataFrame([1, 5])
        expected = DataFrame([3, 5])
        result = df.clip([3])
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([1, 3])
        result = df.clip(upper=[3])
        tm.assert_frame_equal(result, expected)
