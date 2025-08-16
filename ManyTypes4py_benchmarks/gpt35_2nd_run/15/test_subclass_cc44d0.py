import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.sparse import SparseArray
from pandas.core.arrays.string_ import StringArray
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.arrays.timedelta import TimedeltaArray
from pandas.core.arrays.numeric import NumericArray
from pandas.core.arrays.period import PeriodArray
from pandas.core.arrays.base import ExtensionOpsMixin
from pandas.core.arrays.base import ExtensionScalarOpsMixin
from pandas.core.arrays.base import ExtensionArrayOpsMixin
from pandas.core.arrays.base import ExtensionArrayNumericOpsMixin
from pandas.core.arrays.base import ExtensionArrayDatetimeOpsMixin
from pandas.core.arrays.base import ExtensionArrayComparisonOpsMixin
from pandas.core.arrays.base import ExtensionArrayArithmeticOpsMixin
from pandas.core.arrays.base import ExtensionArrayGroupBy
from pandas.core.arrays.base import ExtensionArraySetOpsMixin
from pandas.core.arrays.base import ExtensionArrayFillna
from pandas.core.arrays.base import ExtensionArrayTake
from pandas.core.arrays.base import ExtensionArrayShift
from pandas.core.arrays.base import ExtensionArrayMethods
from pandas.core.arrays.base import ExtensionArrayBackfill
from pandas.core.arrays.base import ExtensionArrayRolling
from pandas.core.arrays.base import ExtensionArrayResampling
from pandas.core.arrays.base import ExtensionArrayDatetime
from pandas.core.arrays.base import ExtensionArrayPeriod
from pandas.core.arrays.base import ExtensionArrayTimedelta
from pandas.core.arrays.base import ExtensionArraySparse
from pandas.core.arrays.base import ExtensionArrayCategorical
from pandas.core.arrays.base import ExtensionArraySetitem
from pandas.core.arrays.base import ExtensionArrayGetitem
from pandas.core.arrays.base import ExtensionArrayGroupby
from pandas.core.arrays.base import ExtensionArrayIndexOps
from pandas.core.arrays.base import ExtensionArrayIndexer
from pandas.core.arrays.base import ExtensionArrayOps
from pandas.core.arrays.base import ExtensionArraySetOps
from pandas.core.arrays.base import ExtensionArrayTakeOps
from pandas.core.arrays.base import ExtensionArrayUnaryOps
from pandas.core.arrays.base import ExtensionArrayBinaryOps
from pandas.core.arrays.base import ExtensionArrayComparisonOps
from pandas.core.arrays.base import ExtensionArrayArithmeticOps
from pandas.core.arrays.base import ExtensionArrayReductionOps
from pandas.core.arrays.base import ExtensionArrayDatetimeOps
from pandas.core.arrays.base import ExtensionArraySetitemOps
from pandas.core.arrays.base import ExtensionArrayGetitemOps
from pandas.core.arrays.base import ExtensionArrayGroupbyOps
from pandas.core.arrays.base import ExtensionArrayIndexOps
from pandas.core.arrays.base import ExtensionArrayIndexerOps
from pandas.core.arrays.base import ExtensionArrayOpsMixin
from pandas.core.arrays.base import ExtensionArraySetOpsMixin
from pandas.core.arrays.base import ExtensionArrayTakeOpsMixin
from pandas.core.arrays.base import ExtensionArrayUnaryOpsMixin
from pandas.core.arrays.base import ExtensionArrayBinaryOpsMixin
from pandas.core.arrays.base import ExtensionArrayComparisonOpsMixin
from pandas.core.arrays.base import ExtensionArrayArithmeticOpsMixin
from pandas.core.arrays.base import ExtensionArrayReductionOpsMixin
from pandas.core.arrays.base import ExtensionArrayDatetimeOpsMixin
from pandas.core.arrays.base import ExtensionArraySetitemOpsMixin
from pandas.core.arrays.base import ExtensionArrayGetitemOpsMixin
from pandas.core.arrays.base import ExtensionArrayGroupbyOpsMixin
from pandas.core.arrays.base import ExtensionArrayIndexOpsMixin
from pandas.core.arrays.base import ExtensionArrayIndexerOpsMixin
from pandas.core.arrays.base import ExtensionArrayOpsMixin
from pandas.core.arrays.base import ExtensionArraySetOpsMixin
from pandas.core.arrays.base import ExtensionArrayTakeOpsMixin
from pandas.core.arrays.base import ExtensionArrayUnaryOpsMixin
from pandas.core.arrays.base import ExtensionArrayBinaryOpsMixin
from pandas.core.arrays.base import ExtensionArrayComparisonOpsMixin
from pandas.core.arrays.base import ExtensionArrayArithmeticOpsMixin
from pandas.core.arrays.base import ExtensionArrayReductionOpsMixin
from pandas.core.arrays.base import ExtensionArrayDatetimeOpsMixin
from pandas.core.arrays.base import ExtensionArraySetitemOpsMixin
from pandas.core.arrays.base import ExtensionArrayGetitemOpsMixin
from pandas.core.arrays.base import ExtensionArrayGroupbyOpsMixin
from pandas.core.arrays.base import ExtensionArrayIndexOpsMixin
from pandas.core.arrays.base import ExtensionArrayIndexerOpsMixin

pytestmark: pytest.mark.filterwarnings('ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning')

class TestSeriesSubclassing:

    @pytest.mark.parametrize('idx_method, indexer, exp_data, exp_idx', [['loc', ['a', 'b'], [1, 2], 'ab'], ['iloc', [2, 3], [3, 4], 'cd']])
    def test_indexing_sliced(self, idx_method: str, indexer: list, exp_data: list, exp_idx: str) -> None:
        s: SubclassedSeries = tm.SubclassedSeries([1, 2, 3, 4], index=list('abcd'))
        res: SubclassedSeries = getattr(s, idx_method)[indexer]
        exp: SubclassedSeries = tm.SubclassedSeries(exp_data, index=list(exp_idx))
        tm.assert_series_equal(res, exp)

    def test_to_frame(self) -> None:
        s: SubclassedSeries = tm.SubclassedSeries([1, 2, 3, 4], index=list('abcd'), name='xxx')
        res: DataFrame = s.to_frame()
        exp: DataFrame = tm.SubclassedDataFrame({'xxx': [1, 2, 3, 4]}, index=list('abcd'))
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self) -> None:
        s: SubclassedSeries = tm.SubclassedSeries([1, 2, 3, 4], index=[list('aabb'), list('xyxy')])
        res: DataFrame = s.unstack()
        exp: DataFrame = tm.SubclassedDataFrame({'x': [1, 3], 'y': [2, 4]}, index=['a', 'b'])
        tm.assert_frame_equal(res, exp)

    def test_subclass_empty_repr(self) -> None:
        sub_series: SubclassedSeries = tm.SubclassedSeries()
        assert 'SubclassedSeries' in repr(sub_series)

    def test_asof(self) -> None:
        N: int = 3
        rng: DatetimeIndex = pd.date_range('1/1/1990', periods=N, freq='53s')
        s: SubclassedSeries = tm.SubclassedSeries({'A': [np.nan, np.nan, np.nan]}, index=rng)
        result: SubclassedSeries = s.asof(rng[-2:])
        assert isinstance(result, SubclassedSeries)

    def test_explode(self) -> None:
        s: SubclassedSeries = tm.SubclassedSeries([[1, 2, 3], 'foo', [], [3, 4]])
        result: SubclassedSeries = s.explode()
        assert isinstance(result, SubclassedSeries)

    def test_equals(self) -> None:
        s1: Series = pd.Series([1, 2, 3])
        s2: SubclassedSeries = tm.SubclassedSeries([1, 2, 3])
        assert s1.equals(s2)
        assert s2.equals(s1)

class SubclassedSeries(pd.Series):

    @property
    def _constructor(self) -> callable:

        def _new(*args, **kwargs) -> NDFrame:
            if self.name == 'test':
                return pd.Series(*args, **kwargs)
            return SubclassedSeries(*args, **kwargs)
        return _new

def test_constructor_from_dict() -> None:
    result: SubclassedSeries = SubclassedSeries({'a': 1, 'b': 2, 'c': 3})
    assert isinstance(result, SubclassedSeries)
