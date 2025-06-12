import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from typing import Any, Callable, List

pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning')

class TestSeriesSubclassing:

    @pytest.mark.parametrize(
        'idx_method, indexer, exp_data, exp_idx',
        [
            ['loc', ['a', 'b'], [1, 2], 'ab'],
            ['iloc', [2, 3], [3, 4], 'cd']
        ]
    )
    def test_indexing_sliced(
        self,
        idx_method: str,
        indexer: List[Any],
        exp_data: List[int],
        exp_idx: str
    ) -> None:
        s: 'SubclassedSeries' = tm.SubclassedSeries([1, 2, 3, 4], index=list('abcd'))
        res: 'SubclassedSeries' = getattr(s, idx_method)[indexer]
        exp: 'SubclassedSeries' = tm.SubclassedSeries(exp_data, index=list(exp_idx))
        tm.assert_series_equal(res, exp)

    def test_to_frame(self) -> None:
        s: 'SubclassedSeries' = tm.SubclassedSeries([1, 2, 3, 4], index=list('abcd'), name='xxx')
        res: pd.DataFrame = s.to_frame()
        exp: tm.SubclassedDataFrame = tm.SubclassedDataFrame({'xxx': [1, 2, 3, 4]}, index=list('abcd'))
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self) -> None:
        s: 'SubclassedSeries' = tm.SubclassedSeries([1, 2, 3, 4], index=[list('aabb'), list('xyxy')])
        res: pd.DataFrame = s.unstack()
        exp: tm.SubclassedDataFrame = tm.SubclassedDataFrame({'x': [1, 3], 'y': [2, 4]}, index=['a', 'b'])
        tm.assert_frame_equal(res, exp)

    def test_subclass_empty_repr(self) -> None:
        sub_series: 'SubclassedSeries' = tm.SubclassedSeries()
        assert 'SubclassedSeries' in repr(sub_series)

    def test_asof(self) -> None:
        N: int = 3
        rng: pd.DatetimeIndex = pd.date_range('1/1/1990', periods=N, freq='53s')
        s: 'SubclassedSeries' = tm.SubclassedSeries({'A': [np.nan, np.nan, np.nan]}, index=rng)
        result: 'SubclassedSeries' = s.asof(rng[-2:])
        assert isinstance(result, tm.SubclassedSeries)

    def test_explode(self) -> None:
        s: 'SubclassedSeries' = tm.SubclassedSeries([[1, 2, 3], 'foo', [], [3, 4]])
        result: 'SubclassedSeries' = s.explode()
        assert isinstance(result, tm.SubclassedSeries)

    def test_equals(self) -> None:
        s1: pd.Series = pd.Series([1, 2, 3])
        s2: 'SubclassedSeries' = tm.SubclassedSeries([1, 2, 3])
        assert s1.equals(s2)
        assert s2.equals(s1)

class SubclassedSeries(pd.Series):

    @property
    def _constructor(self) -> Callable[..., pd.Series]:
        def _new(*args: Any, **kwargs: Any) -> pd.Series:
            if self.name == 'test':
                return pd.Series(*args, **kwargs)
            return SubclassedSeries(*args, **kwargs)
        return _new

def test_constructor_from_dict() -> None:
    result: 'SubclassedSeries' = SubclassedSeries({'a': 1, 'b': 2, 'c': 3})
    assert isinstance(result, SubclassedSeries)
