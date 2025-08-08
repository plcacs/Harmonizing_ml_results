import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from typing import List, Union

class TestSeriesSubclassing:

    @pytest.mark.parametrize('idx_method, indexer, exp_data, exp_idx', [['loc', ['a', 'b'], [1, 2], 'ab'], ['iloc', [2, 3], [3, 4], 'cd']])
    def test_indexing_sliced(self, idx_method: str, indexer: List[Union[str, int]], exp_data: List[int], exp_idx: str) -> None:
        s: Series = tm.SubclassedSeries([1, 2, 3, 4], index=list('abcd'))
        res: Series = getattr(s, idx_method)[indexer]
        exp: Series = tm.SubclassedSeries(exp_data, index=list(exp_idx))
        tm.assert_series_equal(res, exp)

    def test_to_frame(self) -> None:
        s: Series = tm.SubclassedSeries([1, 2, 3, 4], index=list('abcd'), name='xxx')
        res: DataFrame = s.to_frame()
        exp: DataFrame = tm.SubclassedDataFrame({'xxx': [1, 2, 3, 4]}, index=list('abcd'))
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self) -> None:
        s: Series = tm.SubclassedSeries([1, 2, 3, 4], index=[list('aabb'), list('xyxy')])
        res: DataFrame = s.unstack()
        exp: DataFrame = tm.SubclassedDataFrame({'x': [1, 3], 'y': [2, 4]}, index=['a', 'b'])
        tm.assert_frame_equal(res, exp)

    def test_subclass_empty_repr(self) -> None:
        sub_series: Series = tm.SubclassedSeries()
        assert 'SubclassedSeries' in repr(sub_series)

    def test_asof(self) -> None:
        N: int = 3
        rng: pd.DatetimeIndex = pd.date_range('1/1/1990', periods=N, freq='53s')
        s: Series = tm.SubclassedSeries({'A': [np.nan, np.nan, np.nan]}, index=rng)
        result: Series = s.asof(rng[-2:])
        assert isinstance(result, tm.SubclassedSeries)

    def test_explode(self) -> None:
        s: Series = tm.SubclassedSeries([[1, 2, 3], 'foo', [], [3, 4]])
        result: Series = s.explode()
        assert isinstance(result, tm.SubclassedSeries)

    def test_equals(self) -> None:
        s1: Series = pd.Series([1, 2, 3])
        s2: Series = tm.SubclassedSeries([1, 2, 3])
        assert s1.equals(s2)
        assert s2.equals(s1)

class SubclassedSeries(pd.Series):

    @property
    def _constructor(self) -> callable:

        def _new(*args, **kwargs) -> Union[Series, 'SubclassedSeries']:
            if self.name == 'test':
                return pd.Series(*args, **kwargs)
            return SubclassedSeries(*args, **kwargs)
        return _new

def test_constructor_from_dict() -> None:
    result: Series = SubclassedSeries({'a': 1, 'b': 2, 'c': 3})
    assert isinstance(result, SubclassedSeries)
