import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import CategoricalDtype, DataFrame, NaT, Series, Timestamp
import pandas._testing as tm
from typing import Any, Dict, List, Tuple, Union

class TestUpdate:

    def test_update(self) -> None:
        s: Series = Series([1.5, np.nan, 3.0, 4.0, np.nan])
        s2: Series = Series([np.nan, 3.5, np.nan, 5.0])
        s.update(s2)
        expected: Series = Series([1.5, 3.5, 3.0, 5.0, np.nan])
        tm.assert_series_equal(s, expected)
        df: DataFrame = DataFrame([{'a': 1}, {'a': 3, 'b': 2}])
        df['c'] = np.nan
        df['c'] = df['c'].astype(object)
        df_orig: DataFrame = df.copy()
        with tm.raises_chained_assignment_error():
            df['c'].update(Series(['foo'], index=[0]))
        expected = df_orig
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        'other, dtype, expected, raises',
        [
            ([61, 63], 'int32', Series([10, 61, 12], dtype='int32'), False),
            ([61, 63], 'int64', Series([10, 61, 12]), False),
            ([61, 63], float, Series([10.0, 61.0, 12.0]), False),
            ([61, 63], object, Series([10, 61, 12], dtype=object), False),
            ([61.0, 63.0], 'int32', Series([10, 61, 12], dtype='int32'), False),
            ([61.0, 63.0], 'int64', Series([10, 61, 12]), False),
            ([61.0, 63.0], float, Series([10.0, 61.0, 12.0]), False),
            ([61.0, 63.0], object, Series([10, 61.0, 12], dtype=object), False),
            ([61.1, 63.1], 'int32', Series([10.0, 61.1, 12.0]), True),
            ([61.1, 63.1], 'int64', Series([10.0, 61.1, 12.0]), True),
            ([61.1, 63.1], float, Series([10.0, 61.1, 12.0]), False),
            ([61.1, 63.1], object, Series([10, 61.1, 12], dtype=object), False),
            ([(61,), (63,)], 'int32', Series([10, (61,), 12]), True),
            ([(61,), (63,)], 'int64', Series([10, (61,), 12]), True),
            ([(61,), (63,)], float, Series([10.0, (61,), 12.0]), True),
            ([(61,), (63,)], object, Series([10, (61,), 12]), False),
        ]
    )
    def test_update_dtypes(
        self,
        other: Union[List[int], List[float], List[Tuple[int, ...]]],
        dtype: Union[str, type, CategoricalDtype],
        expected: Series,
        raises: bool
    ) -> None:
        ser: Series = Series([10, 11, 12], dtype=dtype)
        other_series: Series = Series(other, index=[1, 3])
        if raises:
            with pytest.raises(TypeError, match='Invalid value'):
                ser.update(other_series)
        else:
            ser.update(other_series)
            tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        'values, other, expected',
        [
            (
                {'a': 1, 'b': 2, 'c': 3, 'd': 4},
                {'b': 5, 'c': np.nan},
                {'a': 1, 'b': 5, 'c': 3, 'd': 4}
            ),
            (
                [1, 2, 3, 4],
                [np.nan, 5, 1],
                [1, 5, 1, 4]
            )
        ]
    )
    def test_update_from_non_series(
        self,
        values: Union[Dict[str, int], List[int]],
        other: Union[Dict[str, Any], List[Any]],
        expected: Union[Dict[str, int], List[int]]
    ) -> None:
        series: Series = Series(values)
        series.update(other)
        expected_series: Series = Series(expected)
        tm.assert_series_equal(series, expected_series)

    @pytest.mark.parametrize(
        'data, other, expected, dtype',
        [
            (
                ['a', None],
                [None, 'b'],
                ['a', 'b'],
                'string[python]'
            ),
            pytest.param(
                ['a', None],
                [None, 'b'],
                ['a', 'b'],
                'string[pyarrow]',
                marks=td.skip_if_no('pyarrow')
            ),
            (
                [1, None],
                [None, 2],
                [1, 2],
                'Int64'
            ),
            (
                [True, None],
                [None, False],
                [True, False],
                'boolean'
            ),
            (
                ['a', None],
                [None, 'b'],
                ['a', 'b'],
                CategoricalDtype(categories=['a', 'b'])
            ),
            (
                [Timestamp(year=2020, month=1, day=1, tz='Europe/London'), NaT],
                [NaT, Timestamp(year=2020, month=1, day=1, tz='Europe/London')],
                [Timestamp(year=2020, month=1, day=1, tz='Europe/London')] * 2,
                'datetime64[ns, Europe/London]'
            )
        ]
    )
    def test_update_extension_array_series(
        self,
        data: List[Any],
        other: List[Any],
        expected: List[Any],
        dtype: Union[str, CategoricalDtype]
    ) -> None:
        result: Series = Series(data, dtype=dtype)
        other_series: Series = Series(other, dtype=dtype)
        expected_series: Series = Series(expected, dtype=dtype)
        result.update(other_series)
        tm.assert_series_equal(result, expected_series)

    def test_update_with_categorical_type(self) -> None:
        dtype: CategoricalDtype = CategoricalDtype(['a', 'b', 'c', 'd'])
        s1: Series = Series(['a', 'b', 'c'], index=[1, 2, 3], dtype=dtype)
        s2: Series = Series(['b', 'a'], index=[1, 2], dtype=dtype)
        s1.update(s2)
        result: Series = s1
        expected: Series = Series(['b', 'a', 'c'], index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)
