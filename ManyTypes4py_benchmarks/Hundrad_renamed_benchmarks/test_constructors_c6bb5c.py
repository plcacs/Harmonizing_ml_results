import array
from collections import OrderedDict, abc, defaultdict, namedtuple
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import date, datetime, timedelta
import functools
import re
import zoneinfo
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, IntervalDtype, NumpyEADtype, PeriodDtype
import pandas as pd
from pandas import Categorical, CategoricalIndex, DataFrame, DatetimeIndex, Index, Interval, MultiIndex, Period, RangeIndex, Series, Timedelta, Timestamp, cut, date_range, isna
import pandas._testing as tm
from pandas.arrays import DatetimeArray, IntervalArray, PeriodArray, SparseArray, TimedeltaArray
MIXED_FLOAT_DTYPES = ['float16', 'float32', 'float64']
MIXED_INT_DTYPES = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
    'int32', 'int64']


class TestDataFrameConstructors:

    def func_tpwclf00(self):
        arr = np.arange(12).reshape(4, 3)
        df = DataFrame(arr, dtype=str)
        expected = DataFrame(arr.astype(str), dtype='str')
        tm.assert_frame_equal(df, expected)

    def func_vrjydxp1(self):
        dti = date_range('2016-01-01', periods=6, tz='US/Pacific')
        dta = dti._data.reshape(3, 2)
        df = DataFrame(dta)
        expected = DataFrame({(0): dta[:, 0], (1): dta[:, 1]})
        tm.assert_frame_equal(df, expected)
        assert len(df._mgr.blocks) == 1

    def func_2kwmah1s(self):
        dt = Timestamp('2019-11-03 01:00:00-0700').tz_convert(
            'America/Los_Angeles')
        dt = dt.as_unit('ns')
        df = DataFrame({'dt': dt}, index=[0])
        expected = DataFrame({'dt': [dt]})
        tm.assert_frame_equal(df, expected, check_index_type=False)
        df = DataFrame({'dt': dt, 'value': [1]})
        expected = DataFrame({'dt': [dt], 'value': [1]})
        tm.assert_frame_equal(df, expected)

    def func_0nay5or8(self):
        arr = np.array([[1, np.nan], [2, 3]])
        msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype='i8')
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype='i8', name=0)

    def func_d5t8g1wd(self):
        df = DataFrame([datetime.now(), datetime.now()])
        assert df[0].dtype == np.dtype('M8[us]')

    def func_iimta2od(self):
        naive = DatetimeIndex(['2013-1-1 13:00', '2013-1-2 14:00'], name='B')
        idx = naive.tz_localize('US/Pacific')
        expected = Series(np.array(idx.tolist(), dtype='object'), name='B')
        assert expected.dtype == idx.dtype
        result = Series(idx)
        tm.assert_series_equal(result, expected)

    def func_tuljgg7b(self):
        col_underscore = '_b'
        df = DataFrame({'a': [1, 2], col_underscore: [3, 4]})
        d = df.to_dict(orient='records')
        ref_d = [{'a': 1, col_underscore: 3}, {'a': 2, col_underscore: 4}]
        assert ref_d == d

    def func_sbwfqzc0(self):
        col_with_num = '1_b'
        df = DataFrame({'a': [1, 2], col_with_num: [3, 4]})
        d = df.to_dict(orient='records')
        ref_d = [{'a': 1, col_with_num: 3}, {'a': 2, col_with_num: 4}]
        assert ref_d == d

    def func_w1nial83(self, frame_or_series):
        nat = np.datetime64('NaT', 'ns')
        arr = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            arr = arr.reshape(1, 1)
        msg = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        with pytest.raises(TypeError, match=msg):
            frame_or_series(arr, dtype='m8[ns]')

    @pytest.mark.parametrize('kind', ['m', 'M'])
    def func_pg9t5ler(self, kind, frame_or_series):
        if kind == 'M':
            dtype = 'M8[ns]'
            scalar_type = Timestamp
        else:
            dtype = 'm8[ns]'
            scalar_type = Timedelta
        arr = np.arange(6, dtype='i8').view(dtype).reshape(3, 2)
        if frame_or_series is Series:
            arr = arr[:, 0]
        obj = frame_or_series(arr, dtype=object)
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        obj = frame_or_series(frame_or_series(arr), dtype=object)
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        obj = frame_or_series(frame_or_series(arr), dtype=NumpyEADtype(object))
        assert obj._mgr.blocks[0].values.dtype == object
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)
        if frame_or_series is DataFrame:
            sers = [Series(x) for x in arr]
            obj = frame_or_series(sers, dtype=object)
            assert obj._mgr.blocks[0].values.dtype == object
            assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type
                )

    def func_td3z3906(self):
        x = Series(range(5), name=1)
        y = Series(range(5), name=0)
        result = DataFrame(x, columns=[0])
        expected = DataFrame([], columns=[0])
        tm.assert_frame_equal(result, expected)
        result = DataFrame(y, columns=[1])
        expected = DataFrame([], columns=[1])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('constructor', [lambda : DataFrame(), lambda :
        DataFrame(None), lambda : DataFrame(()), lambda : DataFrame([]), lambda
        : DataFrame(_ for _ in []), lambda : DataFrame(range(0)), lambda :
        DataFrame(data=None), lambda : DataFrame(data=()), lambda :
        DataFrame(data=[]), lambda : DataFrame(data=(_ for _ in [])), lambda :
        DataFrame(data=range(0))])
    def func_qehxi1is(self, constructor):
        expected = DataFrame()
        result = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected)

    def func_6mzehorb(self):
        expected = DataFrame(index=RangeIndex(0), columns=RangeIndex(0))
        result = DataFrame({})
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected, check_index_type=True)

    @pytest.mark.parametrize('emptylike,expected_index,expected_columns', [
        ([[]], RangeIndex(1), RangeIndex(0)), ([[], []], RangeIndex(2),
        RangeIndex(0)), ([(_ for _ in [])], RangeIndex(1), RangeIndex(0))])
    def func_7qk2mbwx(self, emptylike, expected_index, expected_columns):
        expected = DataFrame(index=expected_index, columns=expected_columns)
        result = DataFrame(emptylike)
        tm.assert_frame_equal(result, expected)

    def func_x9ql5afk(self, float_string_frame, using_infer_string):
        dtype = 'str' if using_infer_string else np.object_
        assert float_string_frame['foo'].dtype == dtype

    def func_qhlbq8ho(self):
        msg = 'could not convert string to float'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': ['a', 'b', 'c']}, dtype=np.float64)
        df = DataFrame(np.ones((4, 2)))
        df['foo'] = np.ones((4, 2)).tolist()
        msg = 'Expected a 1D array, got an array with shape \\(4, 2\\)'
        with pytest.raises(ValueError, match=msg):
            df['test'] = np.ones((4, 2))
        df['foo2'] = np.ones((4, 2)).tolist()

    def func_6rol1gn5(self):
        orig_df = DataFrame({'col1': [1.0], 'col2': [2.0], 'col3': [3.0]})
        new_df = DataFrame(orig_df, dtype=float, copy=True)
        new_df['col1'] = 200.0
        assert orig_df['col1'][0] == 1.0

    def func_5z8c5vjk(self):
        df = DataFrame([[1, 2]])
        should_be_view = DataFrame(df, dtype=df[0].dtype)
        should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 1

    def func_qee4ed39(self):
        df = DataFrame([[1, 2], [3, 4]], dtype='int64')
        df2 = DataFrame(df.values, dtype=df[0].dtype)
        assert df2._mgr.blocks[0].values.flags.c_contiguous

    def func_218psdlo(self, using_infer_string):
        arr = np.array(['a', 'b'], dtype='object')
        df = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def func_4fyaqlkl(self, using_infer_string):
        arr = np.array([['a', 'b'], ['c', 'd']], dtype='object')
        df = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == 'pyarrow':
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)
        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def func_3i77x8e2(self):
        df = DataFrame([[1, '2'], [None, 'a']], dtype=object)
        assert df.loc[1, 0] is None
        assert df.loc[0, 1] == '2'

    def func_vaprppjm(self):
        a = DataFrame()
        b = np.empty((0, 0))
        with pytest.raises(ValueError, match='shape=\\(1, 0, 0\\)'):
            DataFrame([a])
        with pytest.raises(ValueError, match='shape=\\(1, 0, 0\\)'):
            DataFrame([b])
        a = DataFrame({'A': [1, 2]})
        with pytest.raises(ValueError, match='shape=\\(2, 2, 1\\)'):
            DataFrame([a, a])

    @pytest.mark.parametrize('typ, ad', [['float', {}], ['float', {'A': 1,
        'B': 'foo', 'C': 'bar'}], ['int', {}]])
    def func_5neeim46(self, typ, ad):
        if typ == 'int':
            dtypes = MIXED_INT_DTYPES
            arrays = [np.array(np.random.default_rng(2).random(10), dtype=d
                ) for d in dtypes]
        elif typ == 'float':
            dtypes = MIXED_FLOAT_DTYPES
            arrays = [np.array(np.random.default_rng(2).integers(10, size=
                10), dtype=d) for d in dtypes]
        for d, a in zip(dtypes, arrays):
            assert a.dtype == d
        ad.update(dict(zip(dtypes, arrays)))
        df = DataFrame(ad)
        dtypes = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
        for d in dtypes:
            if d in df:
                assert df.dtypes[d] == d

    def func_ovjbp9b4(self):
        a = np.random.default_rng(2).random(10).astype(np.complex64)
        b = np.random.default_rng(2).random(10).astype(np.complex128)
        df = DataFrame({'a': a, 'b': b})
        assert a.dtype == df.a.dtype
        assert b.dtype == df.b.dtype

    def func_35iwqrho(self, string_dtype):
        df = DataFrame({'A': ['x', None]}, dtype=string_dtype)
        result = df.isna()
        expected = DataFrame({'A': [False, True]})
        tm.assert_frame_equal(result, expected)
        assert df.iloc[1, 0] is None
        df = DataFrame({'A': ['x', np.nan]}, dtype=string_dtype)
        assert np.isnan(df.iloc[1, 0])

    def func_83b2oq6p(self, float_frame):
        rec = float_frame.to_records(index=False)
        rec.dtype.names = list(rec.dtype.names)[::-1]
        index = float_frame.index
        df = DataFrame(rec)
        tm.assert_index_equal(df.columns, Index(rec.dtype.names))
        df2 = DataFrame(rec, index=index)
        tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
        tm.assert_index_equal(df2.index, index)
        rng = np.arange(len(rec))[::-1]
        df3 = DataFrame(rec, index=rng, columns=['C', 'B'])
        expected = DataFrame(rec, index=rng).reindex(columns=['C', 'B'])
        tm.assert_frame_equal(df3, expected)

    def func_annt4gt2(self):
        df = DataFrame({(0): np.ones(10, dtype=bool), (1): np.zeros(10,
            dtype=bool)})
        assert df.values.dtype == np.bool_

    def func_o9ialiee(self):
        values = np.array([(2 ** 64 - i) for i in range(1, 10)], dtype=np.
            uint64)
        result = DataFrame({'a': values})
        assert result['a'].dtype == np.uint64
        data_scores = [(6311132704823138710, 273), (2685045978526272070, 23
            ), (8921811264899370420, 45), (17019687244989530680, 270), (
            9930107427299601010, 273)]
        dtype = [('uid', 'u8'), ('score', 'u8')]
        data = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls = DataFrame(data)
        assert df_crawls['uid'].dtype == np.uint64

    @pytest.mark.parametrize('values', [np.array([2 ** 64], dtype=object),
        np.array([2 ** 65]), [2 ** 64 + 1], np.array([-2 ** 63 - 4], dtype=
        object), np.array([-2 ** 64 - 1]), [-2 ** 65 - 2]])
    def func_f1qagkd5(self, values):
        value = values[0]
        result = DataFrame(values)
        assert result[0].dtype == object
        assert result[0][0] == value

    @pytest.mark.parametrize('values', [np.array([1], dtype=np.uint16), np.
        array([1], dtype=np.uint32), np.array([1], dtype=np.uint64), [np.
        uint16(1)], [np.uint32(1)], [np.uint64(1)]])
    def func_okzhf14a(self, values):
        value = values[0]
        result = DataFrame(values)
        assert result[0].dtype == value.dtype
        assert result[0][0] == value

    def func_i9sjc98g(self):
        nitems = 100
        nums = list(range(nitems))
        np.random.default_rng(2).shuffle(nums)
        expected = [f'A{i:d}' for i in nums]
        df = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        assert expected == list(df.columns)

    def func_i8da26om(self):
        datetime_series = Series(np.arange(30, dtype=np.float64), index=
            date_range('2020-01-01', periods=30))
        datetime_series_short = datetime_series[5:]
        frame = DataFrame({'col1': datetime_series, 'col2':
            datetime_series_short})
        assert len(datetime_series) == 30
        assert len(datetime_series_short) == 25
        tm.assert_series_equal(frame['col1'], datetime_series.rename('col1'))
        exp = Series(np.concatenate([[np.nan] * 5, datetime_series_short.
            values]), index=datetime_series.index, name='col2')
        tm.assert_series_equal(exp, frame['col2'])
        frame = DataFrame({'col1': datetime_series, 'col2':
            datetime_series_short}, columns=['col2', 'col3', 'col4'])
        assert len(frame) == len(datetime_series_short)
        assert 'col1' not in frame
        assert isna(frame['col3']).all()
        assert len(DataFrame()) == 0
        msg = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})

    def func_98mml5ff(self):
        frame = DataFrame({'A': {'1': 1, '2': 2}})
        tm.assert_index_equal(frame.index, Index(['1', '2']))

    def func_c9uinhew(self):
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx)
        assert frame.index is idx

    def func_gl0z11by(self):
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx, columns=idx)
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def func_bhxd83t2(self):
        frame = DataFrame({'A': [], 'B': []}, columns=['A', 'B'])
        tm.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def func_e1ynqbk2(self):
        frame_none = DataFrame({'a': None}, index=[0])
        frame_none_list = DataFrame({'a': [None]}, index=[0])
        assert frame_none._get_value(0, 'a') is None
        assert frame_none_list._get_value(0, 'a') is None
        tm.assert_frame_equal(frame_none, frame_none_list)

    def func_jos9r8vx(self):
        msg = 'If using all scalar values, you must pass an index'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': 0.7})
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': 0.7}, columns=['a'])

    @pytest.mark.parametrize('scalar', [2, np.nan, None, 'D'])
    def func_t0qmvyd8(self, scalar):
        result = DataFrame({'a': scalar}, columns=['b'])
        expected = DataFrame(columns=['b'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value', [4, np.nan, None, float('nan')])
    def func_0bufgzhk(self, value):
        cols = [1, value, 3]
        idx = ['a', value]
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values(1).sort_values('a', axis=1)
        expected = DataFrame(np.arange(6, dtype='int64').reshape(2, 3),
            index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values('a', axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('value', [np.nan, None, float('nan')])
    def func_aqjgs5gh(self, value):
        cols = Index([(11, 21), (value, 22), (13, value)])
        idx = Index([('a', value), (value, 2)])
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values((11, 21)).sort_values(('a',
            value), axis=1)
        expected = DataFrame(np.arange(6, dtype='int64').reshape(2, 3),
            index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values(('a', value), axis=1)
        tm.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    def func_yhmv4bvd(self):
        datetime_series = Series(np.arange(10, dtype=np.float64), index=
            date_range('2020-01-01', periods=10))
        datetime_series_short = datetime_series[:5]
        d = {'b': datetime_series_short, 'a': datetime_series}
        frame = DataFrame(data=d)
        expected = DataFrame(data=d, columns=list('ba'))
        tm.assert_frame_equal(frame, expected)

    def func_0p85c41f(self):
        result = DataFrame({np.nan: [1, 2], (2): [2, 3]}, columns=[np.nan, 2])
        expected = DataFrame([[1, 2], [2, 3]], columns=[np.nan, 2])
        tm.assert_frame_equal(result, expected)

    def func_cm7mk4cu(self):
        tuples = [(2, 3), (3, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()
        tuples = [(3, 3), (2, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

    def func_31q2a65t(self):
        df = DataFrame([[1]], columns=[[1]], index=[1, 2])
        expected = DataFrame([1, 1], index=Index([1, 2], dtype='int64'),
            columns=MultiIndex(levels=[[1]], codes=[[0]]))
        tm.assert_frame_equal(df, expected)
        df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
        expected = DataFrame([1, 1], index=MultiIndex(levels=[[1, 2]],
            codes=[[0, 1]]), columns=MultiIndex(levels=[[1]], codes=[[0]]))
        tm.assert_frame_equal(df, expected)

    def func_pohs8x7k(self):
        msg = 'Empty data passed with indices specified.'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), index=[1])
        msg = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})
        msg = 'Shape of passed values is \\(4, 3\\), indices imply \\(3, 3\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.arange(12).reshape((4, 3)), columns=['foo', 'bar',
                'baz'], index=date_range('2000-01-01', periods=3))
        arr = np.array([[4, 5, 6]])
        msg = 'Shape of passed values is \\(1, 3\\), indices imply \\(1, 4\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)
        arr = np.array([4, 5, 6])
        msg = 'Shape of passed values is \\(3, 1\\), indices imply \\(1, 4\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)
        with pytest.raises(ValueError, match='Must pass 2-d input'):
            DataFrame(np.zeros((3, 3, 3)), columns=['A', 'B', 'C'], index=[1])
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(1, 3\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.random.default_rng(2).random((2, 3)), columns=['A',
                'B', 'C'], index=[1])
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(2, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.random.default_rng(2).random((2, 3)), columns=['A',
                'B'], index=[1, 2])
        msg = '2 columns passed, passed data had 10 columns'
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=('ones', 'twos'))
        msg = 'If using all scalar values, you must pass an index'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': False, 'b': True})

    def func_vsiyhfmu(self, dict_subclass):
        data = {'col1': dict_subclass((x, 10.0 * x) for x in range(10)),
            'col2': dict_subclass((x, 20.0 * x) for x in range(10))}
        df = DataFrame(data)
        refdf = DataFrame({col: dict(val.items()) for col, val in data.items()}
            )
        tm.assert_frame_equal(refdf, df)
        data = dict_subclass(data.items())
        df = DataFrame(data)
        tm.assert_frame_equal(refdf, df)

    def func_8f5jwpp4(self, float_frame):
        data = {}
        float_frame.loc[:float_frame.index[10], 'B'] = np.nan
        for k, v in float_frame.items():
            dct = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        frame = DataFrame(data)
        expected = frame.reindex(index=float_frame.index)
        tm.assert_frame_equal(float_frame, expected)

    def func_2k9j3ftr(self):
        expected = np.array([[4.0, 3.0, 2.0, 1.0]])
        df = DataFrame({'d': [4.0], 'c': [3.0], 'b': [2.0], 'a': [1.0]},
            columns=['d', 'c', 'b', 'a'])
        tm.assert_numpy_array_equal(df.values, expected)

    def func_hw0on4dz(self, using_infer_string):
        test_data = {'A': {'1': 1, '2': 2}, 'B': {'1': '1', '2': '2', '3': '3'}
            }
        frame = DataFrame(test_data, dtype=float)
        assert len(frame) == 3
        assert frame['B'].dtype == np.float64
        assert frame['A'].dtype == np.float64
        frame = DataFrame(test_data)
        assert len(frame) == 3
        assert frame['B'
            ].dtype == np.object_ if not using_infer_string else 'str'
        assert frame['A'].dtype == np.float64

    def func_8oyoqyyg(self):
        test_data = {'A': dict(zip(range(20), [f'word_{i}' for i in range(
            20)])), 'B': dict(zip(range(15), np.random.default_rng(2).
            standard_normal(15)))}
        with pytest.raises(ValueError, match='could not convert string'):
            DataFrame(test_data, dtype=float)

    def func_f6qrtwt6(self):
        d = {'Col1': {'Row1': 'A String', 'Row2': np.nan}}
        df = DataFrame(d)
        assert isinstance(df['Col1']['Row2'], float)

    def func_hmpigcwf(self):
        dm = DataFrame([[1, 2], ['a', 'b']], index=[1, 2], columns=[1, 2])
        assert isinstance(dm[1][1], int)

    def func_8cutfyxq(self):
        data = {'a': (1, 2, 3), 'b': (4, 5, 6)}
        result = DataFrame(data)
        expected = DataFrame({k: list(v) for k, v in data.items()})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def func_3q0hw9ht(self):
        data = {'a': range(3), 'b': range(3, 6)}
        result = DataFrame(data)
        expected = DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def func_ronaisc3(self):
        data = {'a': iter(range(3)), 'b': reversed(range(3))}
        result = DataFrame(data)
        expected = DataFrame({'a': [0, 1, 2], 'b': [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def func_fjc98g26(self):
        data = {'a': (i for i in range(3)), 'b': (i for i in reversed(range
            (3)))}
        result = DataFrame(data)
        expected = DataFrame({'a': [0, 1, 2], 'b': [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def func_ayskhbre(self):
        d = {('a', 'a'): {('i', 'i'): 0, ('i', 'j'): 1, ('j', 'i'): 2}, (
            'b', 'a'): {('i', 'i'): 6, ('i', 'j'): 5, ('j', 'i'): 4}, ('b',
            'c'): {('i', 'i'): 7, ('i', 'j'): 8, ('j', 'i'): 9}}
        _d = sorted(d.items())
        df = DataFrame(d)
        expected = DataFrame([x[1] for x in _d], index=MultiIndex.
            from_tuples([x[0] for x in _d])).T
        expected.index = MultiIndex.from_tuples(expected.index)
        tm.assert_frame_equal(df, expected)
        d['z'] = {'y': 123.0, ('i', 'i'): 111, ('i', 'j'): 111, ('j', 'i'): 111
            }
        _d.insert(0, ('z', d['z']))
        expected = DataFrame([x[1] for x in _d], index=Index([x[0] for x in
            _d], tupleize_cols=False)).T
        expected.index = Index(expected.index, tupleize_cols=False)
        df = DataFrame(d)
        df = df.reindex(columns=expected.columns, index=expected.index)
        tm.assert_frame_equal(df, expected)

    def func_vpkpavr1(self):
        dates_as_str = ['1984-02-19', '1988-11-06', '1989-12-03', '1990-03-15']

        def func_l3xayzsg(constructor):
            return {i: {constructor(s): 2 * i} for i, s in enumerate(
                dates_as_str)}
        data_datetime64 = func_l3xayzsg(np.datetime64)
        data_datetime = func_l3xayzsg(lambda x: datetime.strptime(x,
            '%Y-%m-%d'))
        data_Timestamp = func_l3xayzsg(Timestamp)
        expected = DataFrame([[0, None, None, None], [None, 2, None, None],
            [None, None, 4, None], [None, None, None, 6]], index=[Timestamp
            (dt) for dt in dates_as_str])
        result_datetime64 = DataFrame(data_datetime64)
        result_datetime = DataFrame(data_datetime)
        assert result_datetime.index.unit == 'us'
        result_datetime.index = result_datetime.index.as_unit('s')
        result_Timestamp = DataFrame(data_Timestamp)
        tm.assert_frame_equal(result_datetime64, expected)
        tm.assert_frame_equal(result_datetime, expected)
        tm.assert_frame_equal(result_Timestamp, expected)

    @pytest.mark.parametrize('klass,name', [(lambda x: np.timedelta64(x,
        'D'), 'timedelta64'), (lambda x: timedelta(days=x), 'pytimedelta'),
        (lambda x: Timedelta(x, 'D'), 'Timedelta[ns]'), (lambda x:
        Timedelta(x, 'D').as_unit('s'), 'Timedelta[s]')])
    def func_aba9pemj(self, klass, name):
        td_as_int = [1, 2, 3, 4]
        data = {i: {klass(s): 2 * i} for i, s in enumerate(td_as_int)}
        expected = DataFrame([{(0): 0, (1): None, (2): None, (3): None}, {(
            0): None, (1): 2, (2): None, (3): None}, {(0): None, (1): None,
            (2): 4, (3): None}, {(0): None, (1): None, (2): None, (3): 6}],
            index=[Timedelta(td, 'D') for td in td_as_int])
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def func_nwui1a3v(self):
        a = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
        b = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
        df = DataFrame({'a': a, 'b': b})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype
        df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(
            object).tolist()})
        assert df['a'].dtype == a.dtype
        assert df['b'].dtype == b.dtype

    def func_gxheib4g(self, ea_scalar_and_dtype):
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df = DataFrame({'a': ea_scalar}, index=[0])
        assert df['a'].dtype == ea_dtype
        expected = DataFrame(index=[0], columns=['a'], data=ea_scalar)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('data,dtype', [(Period('2020-01'), PeriodDtype
        ('M')), (Interval(left=0, right=5), IntervalDtype('int64', 'right')
        ), (Timestamp('2011-01-01', tz='US/Eastern'), DatetimeTZDtype(unit=
        's', tz='US/Eastern'))])
    def func_u4rcxc4g(self, data, dtype):
        df = DataFrame(index=range(2), columns=['a', 'b'], data=data)
        assert df['a'].dtype == dtype
        assert df['b'].dtype == dtype
        arr = pd.array([data] * 2, dtype=dtype)
        expected = DataFrame({'a': arr, 'b': arr})
        tm.assert_frame_equal(df, expected)

    def func_llmdehfi(self):
        rng = pd.period_range('1/1/2000', periods=5)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)),
            columns=rng)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def func_dq4uh104(self, empty):
        mat = empty((2, 3), dtype=float)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        frame = DataFrame(empty((3,)), columns=['A'], index=[1, 2, 3])
        assert len(frame.index) == 3
        assert len(frame.columns) == 1
        if empty is not np.ones:
            msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
            with pytest.raises(IntCastingNaNError, match=msg):
                DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype
                    =np.int64)
            return
        else:
            frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2],
                dtype=np.int64)
            assert frame.values.dtype == np.int64
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(1, 3\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=['A', 'B', 'C'], index=[1])
        msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(2, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=['A', 'B'], index=[1, 2])
        with pytest.raises(ValueError, match='Must pass 2-d input'):
            DataFrame(empty((3, 3, 3)), columns=['A', 'B', 'C'], index=[1])
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, columns=['A', 'B', 'C'])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        frame = DataFrame(empty((0, 3)))
        assert len(frame.index) == 0
        frame = DataFrame(empty((3, 0)))
        assert len(frame.columns) == 0

    def func_pv63rtxp(self):
        self._check_basic_constructor(np.ones)
        frame = DataFrame(['foo', 'bar'], index=[0, 1], columns=['A'])
        assert len(frame) == 2

    def func_i7dbektp(self):
        self._check_basic_constructor(ma.masked_all)
        mat = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1.0 == frame['A'][1]
        assert 2.0 == frame['C'][2]
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert np.all(~np.asarray(frame == frame))

    @pytest.mark.filterwarnings(
        'ignore:elementwise comparison failed:DeprecationWarning')
    def func_qlrokko3(self):
        mat = ma.masked_all((2, 3), dtype=int)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype
            =np.float64)
        assert frame.values.dtype == np.float64
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1 == frame['A'][1]
        assert 2 == frame['C'][2]
        mat = ma.masked_all((2, 3), dtype='M8[ns]')
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert isna(frame).values.all()
        msg = 'datetime64\\[ns\\] values and dtype=int64 is not supported'
        with pytest.raises(TypeError, match=msg):
            DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype=np.
                int64)
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert 1 == frame['A'].astype('i8')[1]
        assert 2 == frame['C'].astype('i8')[2]
        mat = ma.masked_all((2, 3), dtype=bool)
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2], dtype
            =object)
        assert frame.values.dtype == object
        mat2 = ma.copy(mat)
        mat2[0, 0] = True
        mat2[1, 2] = False
        frame = DataFrame(mat2, columns=['A', 'B', 'C'], index=[1, 2])
        assert frame['A'][1] is True
        assert frame['C'][2] is False

    def func_itjc0l4v(self):
        mat_hard = ma.masked_all((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
        expected = DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]},
            columns=['A', 'B'], index=[1, 2], dtype=float)
        tm.assert_frame_equal(result, expected)
        mat_hard = ma.ones((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
        expected = DataFrame({'A': [1.0, 1.0], 'B': [1.0, 1.0]}, columns=[
            'A', 'B'], index=[1, 2], dtype=float)
        tm.assert_frame_equal(result, expected)

    def func_r8z1489c(self):
        data = np.ma.array(np.ma.zeros(5, dtype=[('date', '<f8'), ('price',
            '<f8')]), mask=[False] * 5)
        data = data.view(mrecords.mrecarray)
        with pytest.raises(TypeError, match='Pass \\{name: data\\[name\\]'):
            DataFrame(data, dtype=int)

    def func_5qifx8sx(self):
        df = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize('data, index, columns, dtype, expected', [(
        None, list(range(10)), ['a', 'b'], object, np.object_), (None, None,
        ['a', 'b'], 'int64', np.dtype('int64')), (None, list(range(10)), [
        'a', 'b'], int, np.dtype('float64')), ({}, None, ['foo', 'bar'],
        None, np.object_), ({'b': 1}, list(range(10)), list('abc'), int, np
        .dtype('float64'))])
    def func_n2r8qrq1(self, data, index, columns, dtype, expected):
        df = DataFrame(data, index, columns, dtype)
        assert df.values.dtype == expected

    @pytest.mark.parametrize('data,input_dtype,expected_dtype', (([True, 
        False, None], 'boolean', pd.BooleanDtype), ([1.0, 2.0, None],
        'Float64', pd.Float64Dtype), ([1, 2, None], 'Int64', pd.Int64Dtype),
        (['a', 'b', 'c'], 'string', pd.StringDtype)))
    def func_qywkpqds(self, data, input_dtype, expected_dtype):
        df = DataFrame({'a': data}, dtype=input_dtype)
        assert df['a'].dtype == expected_dtype()

    def func_cxs6evzi(self, using_infer_string):
        data = {'int': 1, 'bool': True, 'float': 3.0, 'complex': 4.0j,
            'object': 'foo'}
        df = DataFrame(data, index=np.arange(10))
        assert df['int'].dtype == np.int64
        assert df['bool'].dtype == np.bool_
        assert df['float'].dtype == np.float64
        assert df['complex'].dtype == np.complex128
        assert df['object'
            ].dtype == np.object_ if not using_infer_string else 'str'

    def func_5v3qhxh9(self):
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10),
            'b': True})
        exp = DataFrame({'a': df['a'].values, 'b': [True] * 10})
        tm.assert_frame_equal(df, exp)
        with pytest.raises(ValueError, match='must pass an index'):
            DataFrame({'a': False, 'b': True})

    def func_i9k6ai2u(self, float_frame):
        df = DataFrame(float_frame)
        tm.assert_frame_equal(df, float_frame)
        df_casted = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    def func_cmmp3fwv(self):
        actual = DataFrame(DataFrame(), dtype='object')
        expected = DataFrame([], dtype='object')
        tm.assert_frame_equal(actual, expected)

    def func_och5ju6a(self, float_frame):
        arr = np.random.default_rng(2).standard_normal(10)
        dm = DataFrame(arr, columns=['A'], index=np.arange(10))
        assert dm.values.ndim == 2
        arr = np.random.default_rng(2).standard_normal(0)
        dm = DataFrame(arr)
        assert dm.values.ndim == 2
        assert dm.values.ndim == 2
        dm = DataFrame(columns=['A', 'B'], index=np.arange(10))
        assert dm.values.shape == (10, 2)
        dm = DataFrame(columns=['A', 'B'])
        assert dm.values.shape == (0, 2)
        dm = DataFrame(index=np.arange(10))
        assert dm.values.shape == (10, 0)
        mat = np.array(['foo', 'bar'], dtype=object).reshape(2, 1)
        msg = "could not convert string to float: 'foo'"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, index=[0, 1], columns=[0], dtype=float)
        dm = DataFrame(DataFrame(float_frame._series))
        tm.assert_frame_equal(dm, float_frame)
        dm = DataFrame({'A': np.ones(10, dtype=int), 'B': np.ones(10, dtype
            =np.float64)}, index=np.arange(10))
        assert len(dm.columns) == 2
        assert dm.values.dtype == np.float64

    def func_35hus554(self):
        df = DataFrame([], index=[])
        expected = DataFrame(index=[])
        tm.assert_frame_equal(df, expected)
        df = DataFrame([], columns=['A', 'B'])
        expected = DataFrame({}, columns=['A', 'B'])
        tm.assert_frame_equal(df, expected)

        def func_pp61n8ox():
            yield from ()
        df = DataFrame(func_pp61n8ox(), columns=['A', 'B'])
        tm.assert_frame_equal(df, expected)

    def func_obm6vd79(self, using_infer_string):
        df = DataFrame(data=[[1, 'a'], [2, 'b']], columns=['num', 'str'])
        assert is_integer_dtype(df['num'])
        assert df['str'
            ].dtype == np.object_ if not using_infer_string else 'str'
        expected = DataFrame(np.arange(10))
        data = [np.array(x) for x in range(10)]
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def func_tuq9w7h4(self):
        ser = Series([1, 2])
        arr = np.array([None, None], dtype=object)
        arr[0] = ser
        arr[1] = ser * 2
        df = DataFrame(arr)
        expected = DataFrame(pd.array(arr))
        tm.assert_frame_equal(df, expected)
        assert df.shape == (2, 1)
        tm.assert_numpy_array_equal(df[0].values, arr)

    def func_5v96xln7(self):
        arrays = [list('abcd'), list('cdef')]
        result = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)
        mi = MultiIndex.from_arrays(arrays)
        expected = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=mi)
        tm.assert_frame_equal(result, expected)

    def func_ikwlqacy(self):
        arrays = [list('abc'), list('cde')]
        msg = '3 columns passed, passed data had 4'
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    def func_61dzby6c(self):
        arrays = [list('abcd'), list('cde')]
        msg = 'all arrays must be same length'
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    @pytest.mark.parametrize('data', [[[Timestamp('2021-01-01')]], [{'x':
        Timestamp('2021-01-01')}], {'x': [Timestamp('2021-01-01')]}, {'x':
        Timestamp('2021-01-01')}])
    def func_f6ylgvle(self, data):
        result = DataFrame(data, index=range(3), columns=['x'])
        expected = DataFrame({'x': [Timestamp('2021-01-01')] * 3})
        tm.assert_frame_equal(result, expected)

    def func_qgkf6lu3(self):


        class DummyContainer(abc.Sequence):

            def __init__(self, lst):
                self._lst = lst

            def __getitem__(self, n):
                return self._lst.__getitem__(n)

            def __len__(self):
                return self._lst.__len__()
        lst_containers = [DummyContainer([1, 'a']), DummyContainer([2, 'b'])]
        columns = ['num', 'str']
        result = DataFrame(lst_containers, columns=columns)
        expected = DataFrame([[1, 'a'], [2, 'b']], columns=columns)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def func_sk61zhms(self):
        result = DataFrame({'A': array.array('i', range(10))})
        expected = DataFrame({'A': list(range(10))})
        tm.assert_frame_equal(result, expected, check_dtype=False)
        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([array.array('i', range(10)), array.array('i',
            range(10))])
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def func_v38duqh4(self):
        result = DataFrame(range(10))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def func_1iks4ee2(self):
        result = DataFrame([range(10), range(10)])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def func_eczxrwpd(self):


        class Iter:

            def __iter__(self):
                for i in range(10):
                    yield [1, 2, 3]
        expected = DataFrame([[1, 2, 3]] * 10)
        result = DataFrame(Iter())
        tm.assert_frame_equal(result, expected)

    def func_ye0mawyo(self):
        result = DataFrame(iter(range(10)))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def func_cb4sdv3x(self):
        result = DataFrame([iter(range(10)), iter(range(10))])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def func_xv99tvst(self):
        gen1 = (i for i in range(10))
        gen2 = (i for i in range(10))
        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([gen1, gen2])
        tm.assert_frame_equal(result, expected)
        gen = ([i, 'a'] for i in range(10))
        result = DataFrame(gen)
        expected = DataFrame({(0): range(10), (1): 'a'})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def func_ep4mchnh(self):
        result = DataFrame([{}])
        expected = DataFrame(index=RangeIndex(1), columns=[])
        tm.assert_frame_equal(result, expected)

    def func_nxx8hdhi(self):
        nested1 = OrderedDict([('b', 1), ('a', 2)])
        nested2 = OrderedDict([('b', 2), ('a', 5)])
        data = OrderedDict([('col2', nested1), ('col1', nested2)])
        result = DataFrame(data)
        data = {'col2': [1, 2], 'col1': [2, 5]}
        expected = DataFrame(data=data, index=['b', 'a'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dict_type', [dict, OrderedDict])
    def func_xr8j2cbo(self, dict_type):
        expected = DataFrame([[2, 1]], columns=['b', 'a'])
        data = dict_type()
        data['b'] = [2]
        data['a'] = [1]
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)
        data = dict_type()
        data['b'] = 2
        data['a'] = 1
        result = DataFrame([data])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dict_type', [dict, OrderedDict])
    def func_6198285c(self, dict_type):
        row_one = dict_type()
        row_one['b'] = 2
        row_one['a'] = 1
        row_two = dict_type()
        row_two['a'] = 1
        row_two['b'] = 2
        row_three = {'b': 2, 'a': 1}
        expected = DataFrame([[2, 1], [2, 1]], columns=['b', 'a'])
        result = DataFrame([row_one, row_two])
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=['b', 'a'])
        result = DataFrame([row_one, row_two, row_three])
        tm.assert_frame_equal(result, expected)

    def func_7layzu3k(self):
        series = [Series(i, index=['b', 'a', 'c'], name=str(i)) for i in
            range(3)]
        result = DataFrame(series)
        expected = DataFrame({'b': [0, 1, 2], 'a': [0, 1, 2], 'c': [0, 1, 2
            ]}, columns=['b', 'a', 'c'], index=['0', '1', '2'])
        tm.assert_frame_equal(result, expected)

    def func_jqwgl2xi(self):


        class CustomDict(dict):
            pass
        d = {'a': 1.5, 'b': 3}
        data_custom = [CustomDict(d)]
        data = [d]
        result_custom = DataFrame(data_custom)
        result = DataFrame(data)
        tm.assert_frame_equal(result, result_custom)

    def func_2ii9jmbt(self):
        data = {'A': np.random.default_rng(2).standard_normal(10), 'B': np.
            random.default_rng(2).standard_normal(8)}
        with pytest.raises(ValueError, match=
            'All arrays must be of the same length'):
            DataFrame(data)

    def func_mguwokbg(self):
        idx = Index(range(3))
        df = DataFrame({'a': 0}, index=idx)
        expected = DataFrame({'a': [0, 0, 0]}, index=idx)
        tm.assert_frame_equal(df, expected, check_dtype=False)

    def func_yyo4r4ru(self, float_frame):
        df = DataFrame(float_frame['A'], index=float_frame.index, columns=['A']
            )
        df.copy()

    def func_qhc3hhuy(self):
        data = {}
        data['A'] = {'foo': 1, 'bar': 2, 'baz': 3}
        data['B'] = Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])
        result = DataFrame(data)
        assert result.index.is_monotonic_increasing
        with pytest.raises(ValueError, match='ambiguous ordering'):
            DataFrame({'A': ['a', 'b'], 'B': {'a': 'a', 'b': 'b'}})
        result = DataFrame({'A': ['a', 'b'], 'B': Series(['a', 'b'], index=
            ['a', 'b'])})
        expected = DataFrame({'A': ['a', 'b'], 'B': ['a', 'b']}, index=['a',
            'b'])
        tm.assert_frame_equal(result, expected)

    def func_axxkmi1m(self):
        data = [[1, 2], (3, 4)]
        result = DataFrame(data)
        expected = DataFrame([[1, 2], [3, 4]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('tuples,lists', [((), []), (((),), [[]]), (((),
        ()), [(), ()]), (((), ()), [[], []]), (([], []), [[], []]), (([1],
        [2]), [[1], [2]]), (([1, 2, 3], [4, 5, 6]), [[1, 2, 3], [4, 5, 6]])])
    def func_yl0io7d2(self, tuples, lists):
        result = DataFrame(tuples)
        expected = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def func_rotifwnu(self):
        result = DataFrame({'A': [(1, 2), (3, 4)]})
        expected = DataFrame({'A': Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def func_igjssz0v(self):
        named_tuple = namedtuple('Pandas', list('ab'))
        tuples = [named_tuple(1, 3), named_tuple(2, 4)]
        expected = DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
        result = DataFrame(tuples, columns=['y', 'z'])
        tm.assert_frame_equal(result, expected)

    def func_3wb18xjy(self):
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        data = [Point(0, 3), Point(1, 3)]
        expected = DataFrame({'x': [0, 1], 'y': [3, 3]})
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def func_v20gkk8c(self):
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        HLine = make_dataclass('HLine', [('x0', int), ('x1', int), ('y', int)])
        data = [Point(0, 3), HLine(1, 3, 3)]
        expected = DataFrame({'x': [0, np.nan], 'y': [3, 3], 'x0': [np.nan,
            1], 'x1': [np.nan, 3]})
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def func_cmzp2udq(self):
        Point = make_dataclass('Point', [('x', int), ('y', int)])
        msg = 'asdict() should be called on dataclass instances'
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {'x': 1, 'y': 0}])

    def func_ang4g08q(self):
        data = [{'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10}, {
            'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8}, {'Second': 
            6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13}]
        expected = DataFrame({'First': [1, 2, 3], 'Second': [4, 5, 6],
            'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None,
            14], 'XXX': [None, None, 13]})
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def func_jq6hnpji(self):
        a = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        df = DataFrame(a)
        assert df.columns[0] == 'x'
        tm.assert_index_equal(df.index, a.index)
        arr = np.random.default_rng(2).standard_normal(10)
        s = Series(arr, name='x')
        df = DataFrame(s)
        expected = DataFrame({'x': s})
        tm.assert_frame_equal(df, expected)
        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = DataFrame({(0): s})
        tm.assert_frame_equal(df, expected, check_column_type=False)
        msg = (
            'Shape of passed values is \\(10, 1\\), indices imply \\(10, 2\\)')
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])
        a = Series([], name='x', dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == 'x'
        s1 = Series(arr, name='x')
        df = DataFrame([s1, arr]).T
        expected = DataFrame({'x': s1, 'Unnamed 0': arr}, columns=['x',
            'Unnamed 0'])
        tm.assert_frame_equal(df, expected)
        df = DataFrame([arr, s1]).T
        expected = DataFrame({(1): s1, (0): arr}, columns=range(2))
        tm.assert_frame_equal(df, expected)

    def func_itzfgqbq(self):
        s0 = Series(range(5), name=0)
        s1 = Series(range(5), name=1)
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def func_2oiamatt(self):
        s1 = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
        s2 = Series([1, 2, 3], index=['a', 'b', 'c'])
        other_index = Index(['a', 'b'])
        df1 = DataFrame(s1, index=other_index)
        exp1 = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == 'x'
        tm.assert_frame_equal(df1, exp1)
        df2 = DataFrame(s2, index=other_index)
        exp2 = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize('name_in1,name_in2,name_in3,name_out', [('idx',
        'idx', 'idx', 'idx'), ('idx', 'idx', None, None), ('idx', None,
        None, None), ('idx1', 'idx2', None, None), ('idx1', 'idx1', 'idx2',
        None), ('idx1', 'idx2', 'idx3', None), (None, None, None, None)])
    def func_aab1pxmy(self, name_in1, name_in2, name_in3, name_out):
        indices = [Index(['a', 'b', 'c'], name=name_in1), Index(['b', 'c',
            'd'], name=name_in2), Index(['c', 'd', 'e'], name=name_in3)]
        series = {c: Series([0, 1, 2], index=i) for i, c in zip(indices, [
            'x', 'y', 'z'])}
        result = DataFrame(series)
        exp_ind = Index(['a', 'b', 'c', 'd', 'e'], name=name_out)
        expected = DataFrame({'x': [0, 1, 2, np.nan, np.nan], 'y': [np.nan,
            0, 1, 2, np.nan], 'z': [np.nan, np.nan, 0, 1, 2]}, index=exp_ind)
        tm.assert_frame_equal(result, expected)

    def func_scrbqpwq(self, float_frame):
        index = list(float_frame.index[:5])
        columns = list(float_frame.columns[:3])
        msg = 'Passing a BlockManager to DataFrame'
        with tm.assert_produces_warning(DeprecationWarning, match=msg,
            check_stacklevel=False):
            result = DataFrame(float_frame._mgr, index=index, columns=columns)
        tm.assert_index_equal(result.index, Index(index))
        tm.assert_index_equal(result.columns, Index(columns))

    def func_9yoxlpz9(self, float_frame):
        df = DataFrame({'A': float_frame['A'], 'B': list(float_frame['B'])},
            columns=['A', 'B'])
        tm.assert_frame_equal(df, float_frame.loc[:, ['A', 'B']])
        msg = 'does not match index length'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'A': float_frame['A'], 'B': list(float_frame['B'])[:-2]}
                )

    def func_4anjwiks(self):
        msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame([[np.nan, 1], [1, 0]], dtype=np.int64)

    def func_s54cliya(self):
        df = DataFrame([[8, 5]], columns=['a', 'a'])
        edf = DataFrame([[8, 5]])
        edf.columns = ['a', 'a']
        tm.assert_frame_equal(df, edf)
        idf = DataFrame.from_records([(8, 5)], columns=['a', 'a'])
        tm.assert_frame_equal(idf, edf)

    def func_eecbdz4m(self, using_infer_string):
        expected = DataFrame(index=[0, 1], columns=[0, 1], dtype=object)
        expected_str = DataFrame(index=[0, 1], columns=[0, 1], dtype=pd.
            StringDtype(na_value=np.nan))
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=str)
        if using_infer_string:
            tm.assert_frame_equal(df, expected_str)
        else:
            tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=np.str_)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype='U5')
        tm.assert_frame_equal(df, expected)

    def func_xmhn093r(self, nullable_string_dtype):
        expected = DataFrame(columns=['c1'], dtype=nullable_string_dtype)
        df = DataFrame(columns=['c1'], dtype=nullable_string_dtype)
        tm.assert_frame_equal(df, expected)

    def func_3167f72b(self):
        df = DataFrame(0.0, index=[1, 2, 3], columns=['a', 'b', 'c'])
        tm.assert_frame_equal(df, DataFrame(np.zeros(df.shape).astype(
            'float64'), df.index, df.columns))
        df = DataFrame(0, index=[1, 2, 3], columns=['a', 'b', 'c'])
        tm.assert_frame_equal(df, DataFrame(np.zeros(df.shape).astype(
            'int64'), df.index, df.columns))
        df = DataFrame('a', index=[1, 2], columns=['a', 'c'])
        tm.assert_frame_equal(df, DataFrame(np.array([['a', 'a'], ['a', 'a'
            ]], dtype=object), index=[1, 2], columns=['a', 'c']))
        msg = 'DataFrame constructor not properly called!'
        with pytest.raises(ValueError, match=msg):
            DataFrame('a', [1, 2])
        with pytest.raises(ValueError, match=msg):
            DataFrame('a', columns=['a', 'c'])
        msg = 'incompatible data and dtype'
        with pytest.raises(TypeError, match=msg):
            DataFrame('a', [1, 2], ['a', 'c'], float)

    def func_l1dei71x(self, using_infer_string):
        intname = np.dtype(int).name
        floatname = np.dtype(np.float64).name
        objectname = np.dtype(np.object_).name
        df = DataFrame({'A': 1, 'B': 'foo', 'C': 'bar', 'D': Timestamp(
            '20010101'), 'E': datetime(2001, 1, 2, 0, 0)}, index=np.arange(10))
        result = df.dtypes
        expected = Series([np.dtype('int64')] + [np.dtype(objectname) if 
            not using_infer_string else pd.StringDtype(na_value=np.nan)] * 
            2 + [np.dtype('M8[s]'), np.dtype('M8[us]')], index=list('ABCDE'))
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', floatname: np.array(
            1.0, dtype=floatname), intname: np.array(1, dtype=intname)},
            index=np.arange(10))
        result = df.dtypes
        expected = Series([np.dtype('float64')] + [np.dtype('int64')] + [np
            .dtype('object') if not using_infer_string else pd.StringDtype(
            na_value=np.nan)] + [np.dtype('float64')] + [np.dtype(intname)],
            index=['a', 'b', 'c', floatname, intname])
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', floatname: np.array([
            1.0] * 10, dtype=floatname), intname: np.array([1] * 10, dtype=
            intname)}, index=np.arange(10))
        result = df.dtypes
        expected = Series([np.dtype('float64')] + [np.dtype('int64')] + [np
            .dtype('object') if not using_infer_string else pd.StringDtype(
            na_value=np.nan)] + [np.dtype('float64')] + [np.dtype(intname)],
            index=['a', 'b', 'c', floatname, intname])
        tm.assert_series_equal(result, expected)

    def func_pyq5o82s(self):
        ind = date_range(start='2000-01-01', freq='D', periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        datetime_s = Series(datetimes)
        assert datetime_s.dtype == 'M8[us]'

    def func_f5qp71l5(self):
        ind = date_range(start='2000-01-01', freq='D', periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        dates = [ts.date() for ts in ind]
        df = DataFrame(datetimes, columns=['datetimes'])
        df['dates'] = dates
        result = df.dtypes
        expected = Series([np.dtype('datetime64[us]'), np.dtype('object')],
            index=['datetimes', 'dates'])
        tm.assert_series_equal(result, expected)

    def func_tv4x6sl9(self):
        dt = datetime(2012, 1, 1, tzinfo=zoneinfo.ZoneInfo('US/Eastern'))
        df = DataFrame({'End Date': dt}, index=[0])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(df.dtypes, Series({'End Date':
            'datetime64[us, US/Eastern]'}, dtype=object))
        df = DataFrame([{'End Date': dt}])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(df.dtypes, Series({'End Date':
            'datetime64[us, US/Eastern]'}, dtype=object))

    def func_u32effnr(self):
        dr = date_range('20130101', periods=3)
        df = DataFrame({'value': dr})
        assert df.iat[0, 0].tz is None
        dr = date_range('20130101', periods=3, tz='UTC')
        df = DataFrame({'value': dr})
        assert str(df.iat[0, 0].tz) == 'UTC'
        dr = date_range('20130101', periods=3, tz='US/Eastern')
        df = DataFrame({'value': dr})
        assert str(df.iat[0, 0].tz) == 'US/Eastern'

    @pytest.mark.xfail(using_string_dtype(), reason='TODO(infer_string)')
    def func_kquk9wd3(self):
        i = date_range('1/1/2011', periods=5, freq='10s', tz='US/Eastern')
        expected = DataFrame({'a': i.to_series().reset_index(drop=True)})
        df = DataFrame()
        df['a'] = i
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'a': i})
        tm.assert_frame_equal(df, expected)

    def func_ci7x2b4f(self):
        i = date_range('1/1/2011', periods=5, freq='10s', tz='US/Eastern')
        i_no_tz = date_range('1/1/2011', periods=5, freq='10s')
        df = DataFrame({'a': i, 'b': i_no_tz})
        expected = DataFrame({'a': i.to_series().reset_index(drop=True),
            'b': i_no_tz})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('arr', [np.array([None, None, None, None,
        datetime.now(), None]), np.array([None, None, datetime.now(), None]
        ), [[np.datetime64('NaT')], [None]], [[np.datetime64('NaT')], [pd.
        NaT]], [[None], [np.datetime64('NaT')]], [[None], [pd.NaT]], [[pd.
        NaT], [np.datetime64('NaT')]], [[pd.NaT], [None]]])
    def func_tw28ak8d(self, arr):
        result = DataFrame(arr).dtypes
        unit = 'ns'
        if isinstance(arr, np.ndarray):
            unit = 'us'
        elif not any(isinstance(x, np.datetime64) for y in arr for x in y):
            unit = 's'
        expected = Series([np.dtype(f'datetime64[{unit}]')])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('order', ['K', 'A', 'C', 'F'])
    @pytest.mark.parametrize('unit', ['M', 'D', 'h', 'm', 's', 'ms', 'us',
        'ns'])
    def func_7evxhxxc(self, order, unit):
        dtype = f'datetime64[{unit}]'
        na = np.array([['2015-01-01', '2015-01-02', '2015-01-03'], [
            '2017-01-01', '2017-01-02', '2017-02-03']], dtype=dtype, order=
            order)
        df = DataFrame(na)
        expected = DataFrame(na.astype('M8[ns]'))
        if unit in ['M', 'D', 'h', 'm']:
            with pytest.raises(TypeError, match='Cannot cast'):
                expected.astype(dtype)
            expected = expected.astype('datetime64[s]')
        else:
            expected = expected.astype(dtype=dtype)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('order', ['K', 'A', 'C', 'F'])
    @pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
    def func_3hi56qgm(self, order, unit):
        dtype = f'timedelta64[{unit}]'
        na = np.array([[np.timedelta64(1, 'D'), np.timedelta64(2, 'D')], [
            np.timedelta64(4, 'D'), np.timedelta64(5, 'D')]], dtype=dtype,
            order=order)
        df = DataFrame(na)
        if unit in ['D', 'h', 'm']:
            exp_unit = 's'
        else:
            exp_unit = unit
        exp_dtype = np.dtype(f'm8[{exp_unit}]')
        expected = DataFrame([[Timedelta(1, 'D'), Timedelta(2, 'D')], [
            Timedelta(4, 'D'), Timedelta(5, 'D')]], dtype=exp_dtype)
        tm.assert_frame_equal(df, expected)

    def func_km1cm3mw(self, using_infer_string):
        df = DataFrame([np.arange(5) for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype('int')] * 5)
        tm.assert_series_equal(result, expected)
        df = DataFrame([np.array(np.arange(5), dtype='int32') for x in
            range(5)])
        result = df.dtypes
        expected = Series([np.dtype('int32')] * 5)
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': [2 ** 31, 2 ** 31 + 1]})
        assert df.dtypes.iloc[0] == np.dtype('int64')
        df = DataFrame([1, 2])
        assert df.dtypes.iloc[0] == np.dtype('int64')
        df = DataFrame([1.0, 2.0])
        assert df.dtypes.iloc[0] == np.dtype('float64')
        df = DataFrame({'a': [1, 2]})
        assert df.dtypes.iloc[0] == np.dtype('int64')
        df = DataFrame({'a': [1.0, 2.0]})
        assert df.dtypes.iloc[0] == np.dtype('float64')
        df = DataFrame({'a': 1}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype('int64')
        df = DataFrame({'a': 1.0}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype('float64')
        df = DataFrame({'a': [1, 2, 4, 7], 'b': [1.2, 2.3, 5.1, 6.3], 'c':
            list('abcd'), 'd': [datetime(2000, 1, 1) for i in range(4)],
            'e': [1.0, 2, 4.0, 7]})
        result = df.dtypes
        expected = Series([np.dtype('int64'), np.dtype('float64'), np.dtype
            ('object') if not using_infer_string else pd.StringDtype(
            na_value=np.nan), np.dtype('datetime64[us]'), np.dtype(
            'float64')], index=list('abcde'))
        tm.assert_series_equal(result, expected)

    def func_yf2ihpln(self, float_frame):
        cop = DataFrame(float_frame, copy=True)
        cop['A'] = 5
        assert (cop['A'] == 5).all()
        assert not (float_frame['A'] == 5).all()

    def func_b5qsxchu(self, float_frame):
        orig = float_frame.copy()
        cop = DataFrame(float_frame)
        assert cop._mgr is not float_frame._mgr
        cop.index = np.arange(len(cop))
        tm.assert_frame_equal(float_frame, orig)

    def func_0px46tgq(self, float_frame):
        arr = float_frame.values.copy()
        df = DataFrame(arr)
        arr[5] = 5
        assert not (df.values[5] == 5).all()
        df = DataFrame(arr, copy=True)
        arr[6] = 6
        assert not (df.values[6] == 6).all()

    def func_3cjkyhml(self, float_frame):
        series = float_frame._series
        df = DataFrame({'A': series['A']}, copy=True)
        df.loc[df.index[0]:df.index[-1], 'A'] = 5
        assert not (series['A'] == 5).all()

    @pytest.mark.parametrize('df', [DataFrame([[1, 2, 3], [4, 5, 6]], index
        =[1, np.nan]), DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1.1, 2.2,
        np.nan]), DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=[np.nan, 
        1.1, 2.2, np.nan]), DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]],
        columns=[np.nan, 1.1, 2.2, np.nan]), DataFrame([[0.0, 1, 2, 3.0], [
        4, 5, 6, 7]], columns=[np.nan, 1, 2, 2])])
    def func_cjv1puzi(self, df):
        for i in range(len(df.columns)):
            df.iloc[:, i]
        indexer = np.arange(len(df.columns))[isna(df.columns)]
        if len(indexer) == 0:
            with pytest.raises(KeyError, match='^nan$'):
                df.loc[:, np.nan]
        elif len(indexer) == 1:
            tm.assert_series_equal(df.iloc[:, indexer[0]], df.loc[:, np.nan])
        else:
            tm.assert_frame_equal(df.iloc[:, indexer], df.loc[:, np.nan])

    def func_i4xb5y1h(self):
        d = DataFrame({'a': [np.nan, False]})
        assert d['a'].dtype == np.object_
        assert not d['a'][1]

    def func_p2e3ynhs(self):
        cat = Categorical(['A', 'B', 'C'])
        arr = np.array(cat).reshape(-1, 1)
        arr = np.broadcast_to(arr, (3, 4))
        result = DataFrame(arr, dtype=cat.dtype)
        expected = DataFrame({(0): cat, (1): cat, (2): cat, (3): cat})
        tm.assert_frame_equal(result, expected)

    def func_9rxdl74q(self):
        df = DataFrame({'A': list('abc')}, dtype='category')
        expected = Series(list('abc'), dtype='category', name='A')
        tm.assert_series_equal(df['A'], expected)
        s = Series(list('abc'), dtype='category')
        result = s.to_frame()
        expected = Series(list('abc'), dtype='category', name=0)
        tm.assert_series_equal(result[0], expected)
        result = s.to_frame(name='foo')
        expected = Series(list('abc'), dtype='category', name='foo')
        tm.assert_series_equal(result['foo'], expected)
        df = DataFrame(list('abc'), dtype='category')
        expected = Series(list('abc'), dtype='category', name=0)
        tm.assert_series_equal(df[0], expected)

    def func_p7khv8zb(self):
        cat = Categorical(list('abc'))
        df = DataFrame([cat])
        expected = DataFrame([cat.astype(object)])
        tm.assert_frame_equal(df, expected)

    def func_26t76sz9(self):
        df = DataFrame([Categorical(list('abc')), Categorical(list('abd'))])
        expected = DataFrame([['a', 'b', 'c'], ['a', 'b', 'd']])
        tm.assert_frame_equal(df, expected)

    def func_pkhybj48(self):
        df = DataFrame([Categorical(list('abc')), list('def')])
        expected = DataFrame([['a', 'b', 'c'], ['d', 'e', 'f']])
        tm.assert_frame_equal(df, expected)

    def func_7n6nfa6g(self):
        df = DataFrame([Categorical(list('abc')), Categorical(list('abdefg'))])
        expected = DataFrame([list('abc'), list('abdefg')])
        tm.assert_frame_equal(df, expected)

    def func_i53fssbn(self):
        items = [1, 2, 3, 1]
        exp = Series(items).astype('category')
        res = Series(items, dtype='category')
        tm.assert_series_equal(res, exp)
        items = ['a', 'b', 'c', 'a']
        exp = Series(items).astype('category')
        res = Series(items, dtype='category')
        tm.assert_series_equal(res, exp)
        index = date_range('20000101', periods=3)
        expected = Series(Categorical(values=[np.nan, np.nan, np.nan],
            categories=['a', 'b', 'c']))
        expected.index = index
        expected = DataFrame({'x': expected})
        df = DataFrame({'x': Series(['a', 'b', 'c'], dtype='category')},
            index=index)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('dtype', tm.ALL_NUMERIC_DTYPES + tm.
        DATETIME64_DTYPES + tm.TIMEDELTA64_DTYPES + tm.BOOL_DTYPES)
    def func_ik4qgwl7(self, dtype):
        data = DataFrame({'a': [1, 2]}, columns=['b'], dtype=dtype)
        assert data.b.dtype == dtype

    @pytest.mark.parametrize('dtype', tm.STRING_DTYPES + tm.BYTES_DTYPES +
        tm.OBJECT_DTYPES)
    def func_xln5xdwd(self, dtype):
        data = DataFrame({'a': [1, 2]}, columns=['b'], dtype=dtype)
        assert data.b.dtype.name == 'object'

    def func_h7tfy7ga(self):
        result = Series(name=0, dtype=object).to_frame().dtypes
        expected = Series({(0): object})
        tm.assert_series_equal(result, expected)
        result = DataFrame(Series(name=0, dtype=object)).dtypes
        tm.assert_series_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize('dtype', [None, 'uint8', 'category'])
    def func_0l7pyiyf(self, dtype):
        expected = DataFrame({'A': [0, 1, 2, 3, 4]}, dtype=dtype or 'int64')
        result = DataFrame(range(5), columns=['A'], dtype=dtype)
        tm.assert_frame_equal(result, expected)
        result = DataFrame({'A': range(5)}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def func_gbifj2ys(self):


        class List(list):
            pass
        expected = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = DataFrame(List([List([1, 2, 3]), List([4, 5, 6])]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('extension_arr', [Categorical(list('aabbc')),
        SparseArray([1, np.nan, np.nan, np.nan]), IntervalArray([Interval(0,
        1), Interval(1, 5)]), PeriodArray(pd.period_range(start='1/1/2017',
        end='1/1/2018', freq='M'))])
    def func_phu691ug(self, extension_arr):
        expected = DataFrame(Series(extension_arr))
        result = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def func_p3w3zyv2(self):
        v = date.today()
        tup = v, v
        result = DataFrame({tup: Series(range(3), index=range(3))}, columns
            =[tup])
        expected = DataFrame([0, 1, 2], columns=Index(Series([tup])))
        tm.assert_frame_equal(result, expected)

    def func_am93sb00(self):
        s1 = Series([39, 6, 4], index=CategoricalIndex(['female', 'male',
            'unknown']))
        s2 = Series([2, 152, 2, 242, 150], index=CategoricalIndex(['f',
            'female', 'm', 'male', 'unknown']))
        result = DataFrame([s1, s2])
        expected = DataFrame(np.array([[39, 6, 4, np.nan, np.nan], [152.0, 
            242.0, 150.0, 2.0, 2.0]]), columns=['female', 'male', 'unknown',
            'f', 'm'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        'ignore:invalid value encountered in cast:RuntimeWarning')
    def func_ss3bxfgj(self):
        ser = Series(range(100))
        ser1 = cut(ser, 10).value_counts().head(5)
        ser2 = cut(ser, 10).value_counts().tail(5)
        result = DataFrame({'1': ser1, '2': ser2})
        index = CategoricalIndex([Interval(-0.099, 9.9, closed='right'),
            Interval(9.9, 19.8, closed='right'), Interval(19.8, 29.7,
            closed='right'), Interval(29.7, 39.6, closed='right'), Interval
            (39.6, 49.5, closed='right'), Interval(49.5, 59.4, closed=
            'right'), Interval(59.4, 69.3, closed='right'), Interval(69.3, 
            79.2, closed='right'), Interval(79.2, 89.1, closed='right'),
            Interval(89.1, 99, closed='right')], ordered=True)
        expected = DataFrame({'1': [10] * 5 + [np.nan] * 5, '2': [np.nan] *
            5 + [10] * 5}, index=index)
        tm.assert_frame_equal(expected, result)

    def func_h6g6xdq5(self):
        dates = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr = np.array(dates, dtype=[('Date', 'M8[us]'), ('Forecasting',
            'M8[us]')])
        df = DataFrame(arr)
        assert df['Date'][0] == dates[0][0]
        assert df['Forecasting'][0] == dates[0][1]
        s = Series(arr['Date'])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def func_0iqdvdfv(self):


        class DatetimeSubclass(datetime):
            pass
        data = DataFrame({'datetime': [DatetimeSubclass(2020, 1, 1, 1, 1)]})
        assert data.datetime.dtype == 'datetime64[us]'

    def func_q18fr33h(self):
        dti = date_range('2016-01-01', periods=3, tz='US/Pacific')
        msg = (
            'Shape of passed values|Passed arrays should have the same length')
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def func_kawcvrh7(self):
        rng = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
        dates = np.asarray(rng)
        df = DataFrame({'A': np.random.default_rng(2).standard_normal(len(
            rng)), 'B': dates})
        assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))

    def func_taj7wf31(self):
        index_lists = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        multi = DataFrame(np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists])
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)
        multi = DataFrame(np.random.default_rng(2).standard_normal((4, 4)),
            columns=index_lists)
        assert isinstance(multi.columns, MultiIndex)

    @pytest.mark.parametrize('input_vals', [[1, 2], ['1', '2'], list(
        date_range('1/1/2011', periods=2, freq='h')), list(date_range(
        '1/1/2011', periods=2, freq='h', tz='US/Eastern')), [Interval(left=
        0, right=5)]])
    def func_hpfmknb7(self, input_vals, string_dtype):
        result = DataFrame({'A': input_vals}, dtype=string_dtype)
        expected = DataFrame({'A': input_vals}).astype({'A': string_dtype})
        tm.assert_frame_equal(result, expected)

    def func_46g0wtnn(self, string_dtype):
        result = DataFrame({'A': [1.0, 2.0, None]}, dtype=string_dtype)
        expected = DataFrame({'A': ['1.0', '2.0', None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('copy', [False, True])
    def func_zacgekae(self, copy, any_numeric_ea_dtype, any_numpy_dtype):
        a = np.array([1, 2], dtype=any_numpy_dtype)
        b = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ['S', 'U']:
            pytest.skip(
                f'{b.dtype} get cast, making the checks below more cumbersome')
        c = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig = c.copy()
        df = DataFrame({'a': a, 'b': b, 'c': c}, copy=copy)

        def func_p5hbqylr(obj):
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                return obj._ndarray.base
            else:
                raise TypeError

        def func_w6scyic3(c_only=False):
            assert sum(x.values is c for x in df._mgr.blocks) == 1
            if c_only:
                return
            assert sum(func_p5hbqylr(x.values) is a for x in df._mgr.blocks if
                isinstance(x.values.dtype, np.dtype)) == 1
            assert sum(func_p5hbqylr(x.values) is b for x in df._mgr.blocks if
                isinstance(x.values.dtype, np.dtype)) == 1
        if not copy:
            func_w6scyic3()
        should_raise = not lib.is_np_dtype(df.dtypes.iloc[0], 'fciuO')
        if should_raise:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            func_w6scyic3(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == 'M':
                assert a[0] == a.dtype.type(1, 'ns')
                assert b[0] == b.dtype.type(3, 'ns')
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]

    def func_326c4sor(self):
        ser = Series([1, 2, 3], dtype='Int64')
        df = DataFrame({'a': ser})
        assert not np.shares_memory(ser.values._data, df['a'].values._data)

    def func_on9diyyj(self):
        result = DataFrame(Series(1, name='foo'), columns=['bar'])
        expected = DataFrame(columns=['bar'])
        tm.assert_frame_equal(result, expected)

    def func_9yxpsvdn(self):
        result = DataFrame([[1, 2, 3], [4, 5, 6]], columns=[['A', 'A', 'A'],
            ['a', 'b', 'c']])
        expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=MultiIndex.
            from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')]))
        tm.assert_frame_equal(result, expected)

    def func_hivu6f0l(self):
        pi = pd.period_range('2016-04-05', periods=3)
        data = pi._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == pi.dtype).all()
        assert (df == pi).all().all()
        ii = pd.IntervalIndex.from_breaks([3, 4, 5, 6])
        data2 = ii._data.astype(object).reshape(1, -1)
        df2 = DataFrame(data2)
        assert df2.shape == (1, 3)
        assert (df2.dtypes == ii.dtype).all()
        assert (df2 == ii).all().all()
        data3 = np.r_[data, data2, data, data2].T
        df3 = DataFrame(data3)
        expected = DataFrame({(0): pi, (1): ii, (2): pi, (3): ii})
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize('col_a, col_b', [([[1], [2]], np.array([[1], [
        2]])), (np.array([[1], [2]]), [[1], [2]]), (np.array([[1], [2]]),
        np.array([[1], [2]]))])
    def func_jg2eycyy(self, col_a, col_b):
        msg = 'Per-column arrays must each be 1-dimensional'
        with pytest.raises(ValueError, match=msg):
            DataFrame({'a': col_a, 'b': col_b})

    def func_cibsrhbv(self):
        df = DataFrame(index=[1, 2, 3], columns=['a', 'b', 'c'], copy=False)
        assert not np.shares_memory(df['a']._values, df['b']._values)
        df.iloc[0, 0] = 0
        expected = DataFrame({'a': [0, np.nan, np.nan], 'b': [np.nan, np.
            nan, np.nan], 'c': [np.nan, np.nan, np.nan]}, index=[1, 2, 3],
            dtype=object)
        tm.assert_frame_equal(df, expected)

    def func_nagca9ee(self):
        msg = 'Shape of passed values is \\(0, 1\\), indices imply \\(0, 2\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(data=np.array([]), columns=['a', 'b'])

    def func_4rvbraqw(self):
        df = DataFrame(['1', '2', None], columns=['a'], dtype='str')
        expected = DataFrame({'a': ['1', '2', None]}, dtype='str')
        tm.assert_frame_equal(df, expected)

    def func_yw3kcj6u(self):
        dtype = pd.StringDtype(na_value=np.nan)
        expected = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(
            ['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': ['a', 'b']})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(
            ['a'], dtype=dtype), index=Index(['x', 'y'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': ['a', 'b']}, index=['x', 'y'])
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'a': ['a', 1]}, dtype='object', columns=Index
            (['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': ['a', 1]})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'a': ['a', 'b']}, dtype='object', columns=
            Index(['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': ['a', 'b']}, dtype='object')
        tm.assert_frame_equal(df, expected)

    def func_sov8gik9(self):
        dtype = pd.StringDtype(na_value=np.nan)
        expected = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(
            ['a'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df = DataFrame({'a': np.array(['a', 'b'])})
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({(0): ['a', 'b'], (1): ['c', 'd']}, dtype=dtype)
        with pd.option_context('future.infer_string', True):
            df = DataFrame(np.array([['a', 'c'], ['b', 'd']]))
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({'a': ['a', 'b'], 'b': ['c', 'd']}, dtype=
            dtype, columns=Index(['a', 'b'], dtype=dtype))
        with pd.option_context('future.infer_string', True):
            df = DataFrame(np.array([['a', 'c'], ['b', 'd']]), columns=['a',
                'b'])
        tm.assert_frame_equal(df, expected)

    def func_6x8emsbl(self):
        with pd.option_context('future.infer_string', True):
            df = DataFrame(np.array([['hello', 'goodbye'], ['hello', 'Hello']])
                )
        assert df._mgr.blocks[0].ndim == 2

    @pytest.mark.parametrize('klass', [Series, Index])
    def func_noc19nza(self, klass):
        obj = klass([Timestamp('2019-12-31')], dtype=object)
        result = DataFrame(obj, columns=['a'])
        assert result.dtypes.iloc[0] == np.object_
        result = DataFrame({'a': obj})
        assert result.dtypes.iloc[0] == np.object_

    def func_lhiifnpg(self):
        result = DataFrame({(0): [1], (1): [2]}).columns
        expected = RangeIndex(2)
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize('cons', [Series, Index, DatetimeIndex,
        DataFrame, pd.array, pd.to_datetime])
    def func_4cb8jeuz(self, cons):
        ts = Timestamp(2999, 1, 1)
        ts2 = ts.tz_localize('US/Pacific')
        obj = cons([ts])
        res_dtype = tm.get_dtype(obj)
        assert res_dtype == 'M8[us]', res_dtype
        obj2 = cons([ts2])
        res_dtype2 = tm.get_dtype(obj2)
        assert res_dtype2 == 'M8[us, US/Pacific]', res_dtype2

    def func_esqntjx0(self):
        result = DataFrame([None, 1], dtype='timedelta64[ns]')
        expected = DataFrame(['NaT', '0 days 00:00:00.000000001'], dtype=
            'timedelta64[ns]')
        tm.assert_frame_equal(result, expected)


class TestDataFrameConstructorIndexInference:

    def func_9lvzainh(self):
        rng1 = pd.period_range('1/1/1999', '1/1/2012', freq='M')
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)
        rng2 = pd.period_range('1/1/1980', '12/1/2001', freq='M')
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        df = DataFrame({'s1': s1, 's2': s2})
        exp = pd.period_range('1/1/1980', '1/1/2012', freq='M')
        tm.assert_index_equal(df.index, exp)

    def func_20wb63vx(self):
        dti = date_range('2016-01-01', periods=3)
        ser1 = Series(range(3), index=dti)
        ser2 = Series(range(3), index=dti.tz_localize('UTC'))
        ser3 = Series(range(3), index=dti.tz_localize('US/Central'))
        ser4 = Series(range(3))
        df1 = DataFrame({'A': ser2, 'B': ser3, 'C': ser4})
        exp_index = Index(list(ser2.index) + list(ser3.index) + list(ser4.
            index), dtype=object)
        tm.assert_index_equal(df1.index, exp_index)
        df2 = DataFrame({'A': ser2, 'C': ser4, 'B': ser3})
        exp_index3 = Index(list(ser2.index) + list(ser4.index) + list(ser3.
            index), dtype=object)
        tm.assert_index_equal(df2.index, exp_index3)
        df3 = DataFrame({'B': ser3, 'A': ser2, 'C': ser4})
        exp_index3 = Index(list(ser3.index) + list(ser2.index) + list(ser4.
            index), dtype=object)
        tm.assert_index_equal(df3.index, exp_index3)
        df4 = DataFrame({'C': ser4, 'B': ser3, 'A': ser2})
        exp_index4 = Index(list(ser4.index) + list(ser3.index) + list(ser2.
            index), dtype=object)
        tm.assert_index_equal(df4.index, exp_index4)
        msg = 'Cannot join tz-naive with tz-aware DatetimeIndex'
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'C': ser4, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'D': ser1, 'A': ser2, 'B': ser3})

    @pytest.mark.parametrize('key_val, col_vals, col_type', [['3', ['3',
        '4'], 'utf8'], [3, [3, 4], 'int8']])
    def func_3hlktz0s(self, key_val, col_vals, col_type):
        pa = pytest.importorskip('pyarrow')
        cols = pd.arrays.ArrowExtensionArray(pa.array(col_vals, type=pa.
            dictionary(pa.int8(), getattr(pa, col_type)())))
        result = DataFrame({key_val: [1, 2]}, columns=cols)
        expected = DataFrame([[1, np.nan], [2, np.nan]], columns=cols)
        expected.isetitem(1, expected.iloc[:, 1].astype(object))
        tm.assert_frame_equal(result, expected)


class TestDataFrameConstructorWithDtypeCoercion:

    def func_pq3www5r(self):
        arr = np.random.default_rng(2).standard_normal((10, 5))
        msg = 'Trying to coerce float values to integers'
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, dtype='i8')
        df = DataFrame(arr.round(), dtype='i8')
        assert (df.dtypes == 'i8').all()
        arr[0, 0] = np.nan
        msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype='i8')
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype='i8')
        msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr).astype('i8')
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0]).astype('i8')


class TestDataFrameConstructorWithDatetimeTZ:

    @pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
    def func_8c4irl7j(self, tz):
        dr = date_range('2011/1/1', '2012/1/1', freq='W-FRI')
        dr_tz = dr.tz_localize(tz)
        df = DataFrame({'A': 'foo', 'B': dr_tz}, index=dr)
        tz_expected = DatetimeTZDtype('ns', dr_tz.tzinfo)
        assert df['B'].dtype == tz_expected
        datetimes_naive = [ts.to_pydatetime() for ts in dr]
        datetimes_with_tz = [ts.to_pydatetime() for ts in dr_tz]
        df = DataFrame({'dr': dr})
        df['dr_tz'] = dr_tz
        df['datetimes_naive'] = datetimes_naive
        df['datetimes_with_tz'] = datetimes_with_tz
        result = df.dtypes
        expected = Series([np.dtype('datetime64[ns]'), DatetimeTZDtype(tz=
            tz), np.dtype('datetime64[us]'), DatetimeTZDtype(tz=tz, unit=
            'us')], index=['dr', 'dr_tz', 'datetimes_naive',
            'datetimes_with_tz'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('pydt', [True, False])
    def func_zpf8cqak(self, tz_aware_fixture, pydt):
        tz = tz_aware_fixture
        ts = Timestamp('2019', tz=tz)
        if pydt:
            ts = ts.to_pydatetime()
        msg = (
            'Cannot convert timezone-aware data to timezone-naive dtype. Use pd.Series\\(values\\).dt.tz_localize\\(None\\) instead.'
            )
        with pytest.raises(ValueError, match=msg):
            DataFrame({(0): [ts]}, dtype='datetime64[ns]')
        msg2 = 'Cannot unbox tzaware Timestamp to tznaive dtype'
        with pytest.raises(TypeError, match=msg2):
            DataFrame({(0): ts}, index=[0], dtype='datetime64[ns]')
        with pytest.raises(ValueError, match=msg):
            DataFrame([ts], dtype='datetime64[ns]')
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.array([ts], dtype=object), dtype='datetime64[ns]')
        with pytest.raises(TypeError, match=msg2):
            DataFrame(ts, index=[0], columns=[0], dtype='datetime64[ns]')
        with pytest.raises(ValueError, match=msg):
            DataFrame([Series([ts])], dtype='datetime64[ns]')
        with pytest.raises(ValueError, match=msg):
            DataFrame([[ts]], columns=[0], dtype='datetime64[ns]')

    def func_et6ofq4x(self):
        idx = Index(date_range('20130101', periods=3, tz='US/Eastern'),
            name='foo')
        dr = date_range('20130110', periods=3)
        df = DataFrame({'A': idx, 'B': dr})
        assert df['A'].dtype, 'M8[ns, US/Eastern'
        assert df['A'].name == 'A'
        tm.assert_series_equal(df['A'], Series(idx, name='A'))
        tm.assert_series_equal(df['B'], Series(dr, name='B'))

    def func_rq3d2egq(self):
        idx2 = date_range('20130101', periods=3, tz='US/Eastern', name='foo')
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2['foo'], Series(idx2, name='foo'))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2['foo'], Series(idx2, name='foo'))
        idx2 = date_range('20130101', periods=3, tz='US/Eastern')
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2[0], Series(idx2, name=0))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2[0], Series(idx2, name=0))

    def func_8khxkdda(self):
        dr = date_range('1/1/2012', periods=10)
        s = Series(dr, index=dr)
        DataFrame({'a': 'foo', 'b': s}, index=dr)
        DataFrame({'a': 'foo', 'b': s.values}, index=dr)

    def func_srud0x1w(self):
        dr = date_range('2011/1/1', '2012/1/1', freq='W-FRI')
        ts = Series(dr)
        d = DataFrame({'A': 'foo', 'B': ts}, index=dr)
        assert d['B'].isna().all()

    def func_snvsqbu4(self):
        dr = date_range(start='20130101T10:00:00', periods=3, freq='min',
            tz='US/Eastern')
        result = DataFrame(dr, columns=['timestamps'])
        expected = DataFrame({'timestamps': [Timestamp('20130101T10:00:00',
            tz='US/Eastern'), Timestamp('20130101T10:01:00', tz=
            'US/Eastern'), Timestamp('20130101T10:02:00', tz='US/Eastern')]
            }, dtype='M8[ns, US/Eastern]')
        tm.assert_frame_equal(result, expected)

    def func_3uwnopwz(self):
        columns = ['Nevada', 'Ohio']
        pop = {'Nevada': {(2001): 2.4, (2002): 2.9}, 'Ohio': {(2000): 1.5,
            (2001): 1.7, (2002): 3.6}}
        result = DataFrame(pop, index=[2001, 2002, 2003], columns=columns)
        expected = DataFrame([(2.4, 1.7), (2.9, 3.6), (np.nan, np.nan)],
            columns=columns, index=Index([2001, 2002, 2003]))
        tm.assert_frame_equal(result, expected)

    def func_edxqpmn1(self):
        dti = date_range('2016-04-05 04:30', periods=3, tz='UTC')
        data = dti._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == dti.dtype).all()
        assert (df == dti).all().all()

    def func_q4j4mra4(self):
        arr = np.array([[Timestamp('2013-01-01 00:00:00'), Timestamp(
            '2013-01-02 00:00:00'), Timestamp('2013-01-03 00:00:00')], [
            Timestamp('2013-01-01 00:00:00-0500', tz='US/Eastern'), pd.NaT,
            Timestamp('2013-01-03 00:00:00-0500', tz='US/Eastern')], [
            Timestamp('2013-01-01 00:00:00+0100', tz='CET'), pd.NaT,
            Timestamp('2013-01-03 00:00:00+0100', tz='CET')]], dtype=object).T
        res = DataFrame(arr, columns=['A', 'B', 'C'])
        expected_dtypes = ['datetime64[s]', 'datetime64[s, US/Eastern]',
            'datetime64[s, CET]']
        assert (res.dtypes == expected_dtypes).all()

    def func_liv0vkq3(self):
        array_dim2 = np.arange(10).reshape((5, 2))
        df = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
        expected = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('typ', [set, frozenset])
    def func_nyvvkvuc(self, typ):
        values = typ({1, 2, 3})
        msg = f"'{typ.__name__}' type is unordered"
        with pytest.raises(TypeError, match=msg):
            DataFrame({'a': values})
        with pytest.raises(TypeError, match=msg):
            Series(values)

    def func_llxiiwzi(self):
        arr = np.arange(0, 12, dtype='datetime64[ns]').reshape(4, 3)
        df = DataFrame(arr)
        assert all(isinstance(block.values, DatetimeArray) for block in df.
            _mgr.blocks)

    def func_lr7jc2er(self):
        arr = np.random.default_rng(2).standard_normal((10, 2))
        dtype = pd.array([2.0]).dtype
        msg = 'len\\(arrays\\) must match len\\(columns\\)'
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, columns=['foo'], dtype=dtype)
        arr2 = pd.array([2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr2, columns=['foo', 'bar'])

    def func_gzkcyrpk(self):
        data = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match='index cannot be a set'):
            DataFrame(data, index={'a', 'b'})
        with pytest.raises(ValueError, match='columns cannot be a set'):
            DataFrame(data, columns={'a', 'b', 'c'})

    def func_ex5ys1jy(self):
        result = DataFrame({'a': pd.NaT}, columns=['a'], index=range(2))
        expected = DataFrame({'a': Series([pd.NaT, pd.NaT])})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.skipif(not np_version_gt2, reason=
        'StringDType only available in numpy 2 and above')
    @pytest.mark.parametrize('data', [{'a': ['a', 'b', 'c'], 'b': [1.0, 2.0,
        3.0], 'c': ['d', 'e', 'f']}])
    def func_5mlq40l1(self, data):
        from numpy.dtypes import StringDType
        data['a'] = np.array(data['a'], dtype=StringDType())
        res = DataFrame(data)
        assert res['a'].dtype == np.object_
        assert (res['a'] == data['a']).all()


def func_134od1iq(obj):
    if isinstance(obj, Series):
        return obj.iloc[0]
    else:
        return obj.iloc[0, 0]


class TestFromScalar:

    @pytest.fixture(params=[list, dict, None])
    def func_qrheqcgi(self, request):
        return request.param

    @pytest.fixture
    def func_hwophxkj(self, frame_or_series, box):
        extra = {'index': range(2)}
        if frame_or_series is DataFrame:
            extra['columns'] = ['A']
        if box is None:
            return functools.partial(frame_or_series, **extra)
        elif box is dict:
            if frame_or_series is Series:
                return lambda x, **kwargs: frame_or_series({(0): x, (1): x},
                    **extra, **kwargs)
            else:
                return lambda x, **kwargs: frame_or_series({'A': x}, **
                    extra, **kwargs)
        elif frame_or_series is Series:
            return lambda x, **kwargs: frame_or_series([x, x], **extra, **
                kwargs)
        else:
            return lambda x, **kwargs: frame_or_series({'A': [x, x]}, **
                extra, **kwargs)

    @pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
    def func_8qdb3k83(self, dtype, constructor):
        obj = func_hwophxkj(pd.NaT, dtype=dtype)
        assert np.all(obj.dtypes == dtype)
        assert np.all(obj.isna())

    def func_i3d0ze28(self, constructor):
        td = Timedelta(1)
        obj = func_hwophxkj(td, dtype='m8[ns]')
        assert func_134od1iq(obj) == td

    def func_sntby4hu(self, constructor, fixed_now_ts):
        ts = fixed_now_ts + Timedelta(1)
        obj = func_hwophxkj(ts, dtype='M8[ns]')
        assert func_134od1iq(obj) == ts

    def func_lkwjgf86(self, constructor):
        td = Timedelta(1)
        td64 = td.to_timedelta64()
        obj = func_hwophxkj(td64, dtype=object)
        assert isinstance(func_134od1iq(obj), np.timedelta64)

    @pytest.mark.parametrize('cls', [np.datetime64, np.timedelta64])
    def func_s4rbdlnv(self, constructor, cls):
        scalar = cls('NaT', 'ns')
        dtype = {np.datetime64: 'm8[ns]', np.timedelta64: 'M8[ns]'}[cls]
        if cls is np.datetime64:
            msg1 = (
                "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
                )
        else:
            msg1 = "<class 'numpy.timedelta64'> is not convertible to datetime"
        msg = '|'.join(['Cannot cast', msg1])
        with pytest.raises(TypeError, match=msg):
            func_hwophxkj(scalar, dtype=dtype)
        scalar = cls(4, 'ns')
        with pytest.raises(TypeError, match=msg):
            func_hwophxkj(scalar, dtype=dtype)

    @pytest.mark.parametrize('cls', [datetime, np.datetime64])
    def func_lugrlfko(self, constructor, cls, request, box, frame_or_series):
        scalar = datetime(9999, 1, 1)
        exp_dtype = 'M8[us]'
        if cls is np.datetime64:
            scalar = np.datetime64(scalar, 'D')
            exp_dtype = 'M8[s]'
        result = func_hwophxkj(scalar)
        item = func_134od1iq(result)
        dtype = tm.get_dtype(result)
        assert type(item) is Timestamp
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype

    @pytest.mark.skip_ubsan
    def func_jrnzy3nb(self, constructor):
        scalar = np.datetime64(np.iinfo(np.int64).max, 'D')
        result = func_hwophxkj(scalar)
        item = func_134od1iq(result)
        assert type(item) is np.datetime64
        dtype = tm.get_dtype(result)
        assert dtype == object

    @pytest.mark.parametrize('cls', [timedelta, np.timedelta64])
    def func_rafq2pep(self, constructor, cls, request, box, frame_or_series):
        if box is list or frame_or_series is Series and box is dict:
            mark = pytest.mark.xfail(reason=
                'TimedeltaArray constructor has been updated to cast td64 to non-nano, but TimedeltaArray._from_sequence has not'
                , strict=True)
            request.applymarker(mark)
        scalar = datetime(9999, 1, 1) - datetime(1970, 1, 1)
        exp_dtype = 'm8[us]'
        if cls is np.timedelta64:
            scalar = np.timedelta64(scalar, 'D')
            exp_dtype = 'm8[s]'
        result = func_hwophxkj(scalar)
        item = func_134od1iq(result)
        dtype = tm.get_dtype(result)
        assert type(item) is Timedelta
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype

    @pytest.mark.skip_ubsan
    @pytest.mark.parametrize('cls', [np.datetime64, np.timedelta64])
    def func_bmjuhb5s(self, constructor, cls):
        scalar = cls(np.iinfo(np.int64).max, 'D')
        result = func_hwophxkj(scalar)
        item = func_134od1iq(result)
        assert type(item) is cls
        dtype = tm.get_dtype(result)
        assert dtype == object

    def func_ero5gtpn(self, constructor, box, frame_or_series):
        tz = 'US/Eastern'
        ts = Timestamp('2019', tz=tz)
        if box is None or frame_or_series is DataFrame and box is dict:
            msg = 'Cannot unbox tzaware Timestamp to tznaive dtype'
            err = TypeError
        else:
            msg = (
                'Cannot convert timezone-aware data to timezone-naive dtype. Use pd.Series\\(values\\).dt.tz_localize\\(None\\) instead.'
                )
            err = ValueError
        with pytest.raises(err, match=msg):
            func_hwophxkj(ts, dtype='M8[ns]')


class TestAllowNonNano:

    @pytest.fixture(params=[True, False])
    def func_6ksszotv(self, request):
        return request.param

    @pytest.fixture
    def func_bkrb10bi(self, as_td):
        values = np.arange(5).astype(np.int64).view('M8[s]')
        if as_td:
            values = values - values[0]
            return TimedeltaArray._simple_new(values, dtype=values.dtype)
        else:
            return DatetimeArray._simple_new(values, dtype=values.dtype)

    def func_q7eqtmp5(self, arr):
        idx = Index(arr)
        assert idx.dtype == arr.dtype

    def func_ujyzjxm3(self, arr, as_td):
        if as_td:
            idx = pd.TimedeltaIndex(arr)
        else:
            idx = DatetimeIndex(arr)
        assert idx.dtype == arr.dtype

    def func_oq1lt3nl(self, arr):
        ser = Series(arr)
        assert ser.dtype == arr.dtype

    def func_k2jtmkhg(self, arr):
        df = DataFrame(arr)
        assert df.dtypes[0] == arr.dtype

    def func_770x7oo0(self, arr):
        df = DataFrame({(0): arr})
        assert df.dtypes[0] == arr.dtype
