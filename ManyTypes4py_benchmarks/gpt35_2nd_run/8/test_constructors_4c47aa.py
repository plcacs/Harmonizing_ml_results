from functools import partial
import numpy as np
import pytest
from pandas import Categorical, CategoricalDtype, CategoricalIndex, Index, Interval, IntervalIndex, date_range, notna, period_range, timedelta_range
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com

class ConstructorTests:
    @pytest.mark.parametrize('breaks_and_expected_subtype', [([3, 14, 15, 92, 653], np.int64), (np.arange(10, dtype='int64'), np.int64), (Index(np.arange(-10, 11, dtype=np.int64)), np.int64), (Index(np.arange(10, 31, dtype=np.uint64)), np.uint64), (Index(np.arange(20, 30, 0.5), dtype=np.float64), np.float64), (date_range('20180101', periods=10), 'M8[ns]'), (date_range('20180101', periods=10, tz='US/Eastern'), 'datetime64[ns, US/Eastern]'), (timedelta_range('1 day', periods=10), 'm8[ns]')])
    @pytest.mark.parametrize('name', [None, 'foo'])
    def test_constructor(self, constructor, breaks_and_expected_subtype, closed, name):
    
    @pytest.mark.parametrize('breaks, subtype', [(Index([0, 1, 2, 3, 4], dtype=np.int64), 'float64'), (Index([0, 1, 2, 3, 4], dtype=np.int64), 'datetime64[ns]'), (Index([0, 1, 2, 3, 4], dtype=np.int64), 'timedelta64[ns]'), (Index([0, 1, 2, 3, 4], dtype=np.float64), 'int64'), (date_range('2017-01-01', periods=5), 'int64'), (timedelta_range('1 day', periods=5), 'int64')])
    def test_constructor_dtype(self, constructor, breaks, subtype):
    
    @pytest.mark.parametrize('breaks', [Index([0, 1, 2, 3, 4], dtype=np.int64), Index([0, 1, 2, 3, 4], dtype=np.uint64), Index([0, 1, 2, 3, 4], dtype=np.float64), date_range('2017-01-01', periods=5), timedelta_range('1 day', periods=5)])
    def test_constructor_pass_closed(self, constructor, breaks):
    
    @pytest.mark.parametrize('breaks', [[np.nan] * 2, [np.nan] * 4, [np.nan] * 50])
    def test_constructor_nan(self, constructor, breaks, closed):
    
    @pytest.mark.parametrize('breaks', [[], np.array([], dtype='int64'), np.array([], dtype='uint64'), np.array([], dtype='float64'), np.array([], dtype='datetime64[ns]'), np.array([], dtype='timedelta64[ns]')])
    def test_constructor_empty(self, constructor, breaks, closed):
    
    @pytest.mark.parametrize('breaks', [tuple('0123456789'), list('abcdefghij'), np.array(list('abcdefghij'), dtype=object), np.array(list('abcdefghij'), dtype='<U1')])
    def test_constructor_string(self, constructor, breaks):
    
    @pytest.mark.parametrize('cat_constructor', [Categorical, CategoricalIndex])
    def test_constructor_categorical_valid(self, constructor, cat_constructor):
    
    def test_generic_errors(self, constructor):
    
class TestFromArrays(ConstructorTests):
    
    def test_mixed_float_int(self, left_subtype, right_subtype):
    
    @pytest.mark.parametrize('interval_cls', [IntervalArray, IntervalIndex])
    def test_from_arrays_mismatched_datetimelike_resos(self, interval_cls):
    
class TestFromBreaks(ConstructorTests):
    
    def test_length_one(self):
    
    def test_left_right_dont_share_data(self, breaks):
    
class TestFromTuples(ConstructorTests):
    
    def test_na_tuples(self):
    
class TestClassConstructors(ConstructorTests):
    
    def test_override_inferred_closed(self, constructor, data, closed):
    
    def test_index_object_dtype(self, values_constructor):
    
    def test_index_mixed_closed(self):
    
@pytest.mark.parametrize('timezone', ['UTC', 'US/Pacific', 'GMT'])
def test_interval_index_subtype(timezone, inclusive_endpoints_fixture):
    
def test_dtype_closed_mismatch():
    
@pytest.mark.parametrize('dtype', ['Float64', pytest.param('float64[pyarrow]', marks=td.skip_if_no('pyarrow'))])
def test_ea_dtype(dtype):
