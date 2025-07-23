from datetime import datetime, timedelta
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import NA, Categorical, CategoricalDtype, DatetimeTZDtype, Index, Interval, NaT, Series, Timedelta, Timestamp, cut, date_range, to_datetime
import pandas._testing as tm
from typing import List, Union, Dict, Any

def rand_str(nchars: int) -> str:
    """
    Generate one random byte string.
    """
    RANDS_CHARS = np.array(list(string.ascii_letters + string.digits), dtype=(np.str_, 1))
    return ''.join(np.random.default_rng(2).choice(RANDS_CHARS, nchars))

class TestAstypeAPI:

    def test_astype_unitless_dt64_raises(self) -> None:
        ser = Series(['1970-01-01', '1970-01-01', '1970-01-01'], dtype='datetime64[ns]')
        df = ser.to_frame()
        msg = "Casting to unit-less dtype 'datetime64' is not supported"
        with pytest.raises(TypeError, match=msg):
            ser.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            df.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            ser.astype('datetime64')
        with pytest.raises(TypeError, match=msg):
            df.astype('datetime64')

    def test_arg_for_errors_in_astype(self) -> None:
        ser = Series([1, 2, 3])
        msg = "Expected value of kwarg 'errors' to be one of \\['raise', 'ignore'\\]\\. Supplied value is 'False'"
        with pytest.raises(ValueError, match=msg):
            ser.astype(np.float64, errors=False)
        ser.astype(np.int8, errors='raise')

    @pytest.mark.parametrize('dtype_class', [dict, Series])
    def test_astype_dict_like(self, dtype_class: Union[Dict[str, Any], Series]) -> None:
        ser = Series(range(0, 10, 2), name='abc')
        dt1 = dtype_class({'abc': str})
        result = ser.astype(dt1)
        expected = Series(['0', '2', '4', '6', '8'], name='abc', dtype='str')
        tm.assert_series_equal(result, expected)
        dt2 = dtype_class({'abc': 'float64'})
        result = ser.astype(dt2)
        expected = Series([0.0, 2.0, 4.0, 6.0, 8.0], dtype='float64', name='abc')
        tm.assert_series_equal(result, expected)
        dt3 = dtype_class({'abc': str, 'def': str})
        msg = 'Only the Series name can be used for the key in Series dtype mappings\\.'
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt3)
        dt4 = dtype_class({0: str})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt4)
        if dtype_class is Series:
            dt5 = dtype_class({}, dtype=object)
        else:
            dt5 = dtype_class({})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt5)

class TestAstype:

    @pytest.mark.parametrize('tz', [None, 'UTC', 'US/Pacific'])
    def test_astype_object_to_dt64_non_nano(self, tz: Union[None, str]) -> None:
        ts = Timestamp('2999-01-01')
        dtype = 'M8[us]'
        if tz is not None:
            dtype = f'M8[us, {tz}]'
        vals = [ts, '2999-01-02 03:04:05.678910', 2500]
        ser = Series(vals, dtype=object)
        result = ser.astype(dtype)
        pointwise = [vals[0].tz_localize(tz), Timestamp(vals[1], tz=tz), to_datetime(vals[2], unit='us', utc=True).tz_convert(tz)]
        exp_vals = [x.as_unit('us').asm8 for x in pointwise]
        exp_arr = np.array(exp_vals, dtype='M8[us]')
        expected = Series(exp_arr, dtype='M8[us]')
        if tz is not None:
            expected = expected.dt.tz_localize('UTC').dt.tz_convert(tz)
        tm.assert_series_equal(result, expected)

    def test_astype_mixed_object_to_dt64tz(self) -> None:
        ts = Timestamp('2016-01-04 05:06:07', tz='US/Pacific')
        ts2 = ts.tz_convert('Asia/Tokyo')
        ser = Series([ts, ts2], dtype=object)
        res = ser.astype('datetime64[ns, Europe/Brussels]')
        expected = Series([ts.tz_convert('Europe/Brussels'), ts2.tz_convert('Europe/Brussels')], dtype='datetime64[ns, Europe/Brussels]')
        tm.assert_series_equal(res, expected)

    @pytest.mark.parametrize('dtype', np.typecodes['All'])
    def test_astype_empty_constructor_equality(self, dtype: str) -> None:
        if dtype not in ('S', 'V', 'M', 'm'):
            init_empty = Series([], dtype=dtype)
            as_type_empty = Series([]).astype(dtype)
            tm.assert_series_equal(init_empty, as_type_empty)

    @pytest.mark.parametrize('dtype', [str, np.str_])
    @pytest.mark.parametrize('data', [[string.digits * 10, rand_str(63), rand_str(64), rand_str(1000)], [string.digits * 10, rand_str(63), rand_str(64), np.nan, 1.0]])
    def test_astype_str_map(self, dtype: Union[type, np.dtype], data: List[Union[str, float]], using_infer_string: bool) -> None:
        series = Series(data)
        using_string_dtype = using_infer_string and dtype is str
        result = series.astype(dtype)
        if using_string_dtype:
            expected = series.map(lambda val: str(val) if val is not np.nan else np.nan)
        else:
            expected = series.map(str)
            if using_infer_string:
                expected = expected.astype(object)
        tm.assert_series_equal(result, expected)

    def test_astype_float_to_period(self) -> None:
        result = Series([np.nan]).astype('period[D]')
        expected = Series([NaT], dtype='period[D]')
        tm.assert_series_equal(result, expected)

    def test_astype_no_pandas_dtype(self) -> None:
        ser = Series([1, 2], dtype='int64')
        result = ser.astype(ser.array.dtype)
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize('dtype', [np.datetime64, np.timedelta64])
    def test_astype_generic_timestamp_no_frequency(self, dtype: type, request: Any) -> None:
        data = [1]
        ser = Series(data)
        if np.dtype(dtype).name not in ['timedelta64', 'datetime64']:
            mark = pytest.mark.xfail(reason='GH#33890 Is assigned ns unit')
            request.applymarker(mark)
        msg = f"The '{dtype.__name__}' dtype has no unit\\. Please pass in '{dtype.__name__}\\[ns\\]' instead."
        with pytest.raises(ValueError, match=msg):
            ser.astype(dtype)

    def test_astype_dt64_to_str(self) -> None:
        dti = date_range('2012-01-01', periods=3)
        result = Series(dti).astype(str)
        expected = Series(['2012-01-01', '2012-01-02', '2012-01-03'], dtype='str')
        tm.assert_series_equal(result, expected)

    def test_astype_dt64tz_to_str(self) -> None:
        dti_tz = date_range('2012-01-01', periods=3, tz='US/Eastern')
        result = Series(dti_tz).astype(str)
        expected = Series(['2012-01-01 00:00:00-05:00', '2012-01-02 00:00:00-05:00', '2012-01-03 00:00:00-05:00'], dtype='str')
        tm.assert_series_equal(result, expected)

    def test_astype_datetime(self, unit: str) -> None:
        ser = Series(iNaT, dtype=f'M8[{unit}]', index=range(5))
        ser = ser.astype('O')
        assert ser.dtype == np.object_
        ser = Series([datetime(2001, 1, 2, 0, 0)])
        ser = ser.astype('O')
        assert ser.dtype == np.object_
        ser = Series([datetime(2001, 1, 2, 0, 0) for i in range(3)], dtype=f'M8[{unit}]')
        ser[1] = np.nan
        assert ser.dtype == f'M8[{unit}]'
        ser = ser.astype('O')
        assert ser.dtype == np.object_

    def test_astype_datetime64tz(self) -> None:
        ser = Series(date_range('20130101', periods=3, tz='US/Eastern'))
        result = ser.astype(object)
        expected = Series(ser.astype(object), dtype=object)
        tm.assert_series_equal(result, expected)
        result = Series(ser.values).dt.tz_localize('UTC').dt.tz_convert(ser.dt.tz)
        tm.assert_series_equal(result, ser)
        result = Series(ser.astype(object))
        expected = ser.astype(object)
        tm.assert_series_equal(result, expected)
        msg = 'Cannot use .astype to convert from timezone-naive'
        with pytest.raises(TypeError, match=msg):
            Series(ser.values).astype('datetime64[ns, US/Eastern]')
        with pytest.raises(TypeError, match=msg):
            Series(ser.values).astype(ser.dtype)
        result = ser.astype('datetime64[ns, CET]')
        expected = Series(date_range('20130101 06:00:00', periods=3, tz='CET'))
        tm.assert_series_equal(result, expected)

    def test_astype_str_cast_dt64(self) -> None:
        ts = Series([Timestamp('2010-01-04 00:00:00')])
        res = ts.astype(str)
        expected = Series(['2010-01-04'], dtype='str')
        tm.assert_series_equal(res, expected)
        ts = Series([Timestamp('2010-01-04 00:00:00', tz='US/Eastern')])
        res = ts.astype(str)
        expected = Series(['2010-01-04 00:00:00-05:00'], dtype='str')
        tm.assert_series_equal(res, expected)

    def test_astype_str_cast_td64(self) -> None:
        td = Series([Timedelta(1, unit='D')])
        ser = td.astype(str)
        expected = Series(['1 days'], dtype='str')
        tm.assert_series_equal(ser, expected)

    def test_dt64_series_astype_object(self) -> None:
        dt64ser = Series(date_range('20130101', periods=3))
        result = dt64ser.astype(object)
        assert isinstance(result.iloc[0], datetime)
        assert result.dtype == np.object_

    def test_td64_series_astype_object(self) -> None:
        tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='timedelta64[ns]')
        result = tdser.astype(object)
        assert isinstance(result.iloc[0], timedelta)
        assert result.dtype == np.object_

    @pytest.mark.parametrize('data, dtype', [(['x', 'y', 'z'], 'string[python]'), pytest.param(['x', 'y', 'z'], 'string[pyarrow]', marks=td.skip_if_no('pyarrow')), (['x', 'y', 'z'], 'category'), (3 * [Timestamp('2020-01-01', tz='UTC')], None), (3 * [Interval(0, 1)], None)])
    @pytest.mark.parametrize('errors', ['raise', 'ignore'])
    def test_astype_ignores_errors_for_extension_dtypes(self, data: List[Any], dtype: Union[str, None], errors: str) -> None:
        ser = Series(data, dtype=dtype)
        if errors == 'ignore':
            expected = ser
            result = ser.astype(float, errors='ignore')
            tm.assert_series_equal(result, expected)
        else:
            msg = '(Cannot cast)|(could not convert)'
            with pytest.raises((ValueError, TypeError), match=msg):
                ser.astype(float, errors=errors)

    def test_astype_from_float_to_str(self, any_float_dtype: np.dtype) -> None:
        ser = Series([0.1], dtype=any_float_dtype)
        result = ser.astype(str)
        expected = Series(['0.1'], dtype='str')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('value, string_value', [(None, 'None'), (np.nan, 'nan'), (NA, '<NA>')])
    def test_astype_to_str_preserves_na(self, value: Any, string_value: str, using_infer_string: bool) -> None:
        ser = Series(['a', 'b', value], dtype=object)
        result = ser.astype(str)
        expected = Series(['a', 'b', None if using_infer_string else string_value], dtype='str')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64', 'int32'])
    def test_astype(self, dtype: str) -> None:
        ser = Series(np.random.default_rng(2).standard_normal(5), name='foo')
        as_typed = ser.astype(dtype)
        assert as_typed.dtype == dtype
        assert as_typed.name == ser.name

    @pytest.mark.parametrize('value', [np.nan, np.inf])
    def test_astype_cast_nan_inf_int(self, any_int_numpy_dtype: np.dtype, value: float) -> None:
        msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        ser = Series([value])
        with pytest.raises(ValueError, match=msg):
            ser.astype(any_int_numpy_dtype)

    def test_astype_cast_object_int_fail(self, any_int_numpy_dtype: np.dtype) -> None:
        arr = Series(['car', 'house', 'tree', '1'])
        msg = "invalid literal for int\\(\\) with base 10: 'car'"
        with pytest.raises(ValueError, match=msg):
            arr.astype(any_int_numpy_dtype)

    def test_astype_float_to_uint_negatives_raise(self, float_numpy_dtype: np.dtype, any_unsigned_int_numpy_dtype: np.dtype) -> None:
        arr = np.arange(5).astype(float_numpy_dtype) - 3
        ser = Series(arr)
        msg = 'Cannot losslessly cast from .* to .*'
        with pytest.raises(ValueError, match=msg):
            ser.astype(any_unsigned_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            ser.to_frame().astype(any_unsigned_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            Index(ser).astype(any_unsigned_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            ser.array.astype(any_unsigned_int_numpy_dtype)

    def test_astype_cast_object_int(self) -> None:
        arr = Series(['1', '2', '3', '4'], dtype=object)
        result = arr.astype(int)
        tm.assert_series_equal(result, Series(np.arange(1, 5)))

    def test_astype_unicode(self, using_infer_string: bool) -> None:
        digits = string.digits
        test_series = [Series([digits * 10, rand_str(63), rand_str(64), rand_str(1000)]), Series(['データーサイエンス、お前はもう死んでいる'])]
        former_encoding = None
        if sys.getdefaultencoding() == 'utf-8':
            item = '野菜食べないとやばい'
            ser = Series([item.encode()])
            result = ser.astype(np.str_)
            expected = Series([item], dtype=object)
            tm.assert_series_equal(result, expected)
        for ser in test_series:
            res = ser.astype(np.str_)
            expec = ser.map(str)
            if using_infer_string:
                expec = expec.astype(object)
            tm.assert_series_equal(res, expec)
        if former_encoding is not None and former_encoding != 'utf-8':
            reload(sys)
            sys.setdefaultencoding(former_encoding)

    def test_astype_bytes(self) -> None:
        result = Series(['foo', 'bar', 'baz']).astype(bytes)
        assert result.dtypes == np.dtype('S3')

    def test_astype_nan_to_bool(self) -> None:
        ser = Series(np.nan, dtype='object')
        result = ser.astype('bool')
        expected = Series(True, dtype='bool')
        tm.assert_series_equal(result, expected)

    def test_astype_ea_to_datetimetzdtype(self, any_numeric_ea_dtype: np.dtype) -> None:
        ser = Series([4, 0, 9], dtype=any_numeric_ea_dtype)
        result = ser.astype(DatetimeTZDtype(tz='US/Pacific'))
        expected = Series({0: Timestamp('1969-12-31 16:00:00.000000004-08:00', tz='US/Pacific'), 1: Timestamp('1969-12-31 16:00:00.000000000-08:00', tz='US/Pacific'), 2: Timestamp('1969-12-31 16:00:00.000000009-08:00', tz='US/Pacific')})
        tm.assert_series_equal(result, expected)

    def test_astype_retain_attrs(self, any_numpy_dtype: np.dtype) -> None:
        ser = Series([0, 1, 2, 3])
        ser.attrs['Location'] = 'Michigan'
        result = ser.astype(any_numpy_dtype).attrs
        expected = ser.attrs
        tm.assert_dict_equal(expected, result)

class TestAstypeString:

    @pytest.mark.parametrize('data, dtype', [([True, NA], 'boolean'), (['A', NA], 'category'), (['2020-10-10', '2020-10-10'], 'datetime64[ns]'), (['2020-10-10', '2020-10-10', NaT], 'datetime64[ns]'), (['2012-01-01 00:00:00-05:00', NaT], 'datetime64[ns, US/Eastern]'), ([1, None], 'UInt16'), (['1/1/2021', '2/1/2021'], 'period[M]'), (['1/1/2021', '2/1/2021', NaT], 'period[M]'), (['1 Day', '59 Days', NaT], 'timedelta64[ns]')])
    def test_astype_string_to_extension_dtype_roundtrip(self, data: List[Any], dtype: str, request: Any, nullable_string_dtype: str) -> None:
        if dtype == 'boolean':
            mark = pytest.mark.xfail(reason='TODO StringArray.astype() with missing values #GH40566')
            request.applymarker(mark)
        ser = Series(data, dtype=dtype)
        result = ser.astype(nullable_string_dtype).astype(ser.dtype)
        tm.assert_series_equal(result, ser)

class TestAstypeCategorical:

    def test_astype_categorical_to_other(self) -> None:
        cat = Categorical([f'{i} - {i + 499}' for i in range(0, 10000, 500)])
        ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)
        expected = ser
        tm.assert_series_equal(ser.astype('category'), expected)
        tm.assert_series_equal(ser.astype(CategoricalDtype()), expected)
        msg = 'Cannot cast object|str dtype to float64'
        with pytest.raises(ValueError, match=msg):
            ser.astype('float64')
        cat = Series(Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']))
        exp = Series(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], dtype='str')
        tm.assert_series_equal(cat.astype('str'), exp)
        s2 = Series(Categorical(['1', '2', '3', '4']))
        exp2 = Series([1, 2, 3, 4]).astype('int')
        tm.assert_series_equal(s2.astype('int'), exp2)

        def cmp(a: Series, b: Series) -> None:
            tm.assert_almost_equal(np.sort(np.unique(a)), np.sort(np.unique(b)))
        expected = Series(np.array(ser.values), name='value_group')
        cmp(ser.astype('object'), expected)
        cmp(ser.astype(np.object_), expected)
        tm.assert_almost_equal(np.array(ser), np.array(ser.values))
        tm.assert_series_equal(ser.astype('category'), ser)
        tm.assert_series_equal(ser.astype(CategoricalDtype()), ser)
        roundtrip_expected = ser.cat.set_categories(ser.cat.categories.sort_values()).cat.remove_unused_categories()
        result = ser.astype('object').astype('category')
        tm.assert_series_equal(result, roundtrip_expected)
        result = ser.astype('object').astype(CategoricalDtype())
        tm.assert_series_equal(result, roundtrip_expected)

    def test_astype_categorical_invalid_conversions(self) -> None:
        cat = Categorical([f'{i} - {i + 499}' for i in range(0, 10000, 500)])
        ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)
        msg = "dtype '<class 'pandas.core.arrays.categorical.Categorical'>' not understood"
        with pytest.raises(TypeError, match=msg):
            ser.astype(Categorical)
        with pytest.raises(TypeError, match=msg):
            ser.astype('object').astype(Categorical)

    def test_astype_categoricaldtype(self) -> None:
        ser = Series(['a', 'b', 'a'])
        result = ser.astype(CategoricalDtype(['a', 'b'], ordered=True))
        expected = Series(Categorical(['a', 'b', 'a'], ordered=True))
        tm.assert_series_equal(result, expected)
        result = ser.astype(CategoricalDtype(['a', 'b'], ordered=False))
        expected = Series(Categorical(['a', 'b', 'a'], ordered=False))
        tm.assert_series_equal(result, expected)
        result = ser.astype(CategoricalDtype(['a', 'b', 'c'], ordered=False))
        expected = Series(Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c'], ordered=False))
        tm.assert_series_equal(result, expected)
        tm.assert_index_equal(result.cat.categories, Index(['a', 'b', 'c']))

    @pytest.mark.parametrize('name', [None, 'foo'])
    @pytest.mark.parametrize('dtype_ordered', [True, False])
    @pytest.mark.parametrize('series_ordered', [True, False])
    def test_astype_categorical_to_categorical(self, name: Union[None, str], dtype_ordered: bool, series_ordered: bool) -> None:
        s_data = list('abcaacbab')
        s_dtype = CategoricalDtype(list('bac'), ordered=series_ordered)
        ser = Series(s_data, dtype=s_dtype, name=name)
        dtype = CategoricalDtype(ordered=dtype_ordered)
        result = ser.astype(dtype)
        exp_dtype = CategoricalDtype(s_dtype.categories, dtype_ordered)
        expected = Series(s_data, name=name, dtype=exp_dtype)
        tm.assert_series_equal(result, expected)
        dtype = CategoricalDtype(list('adc'), dtype_ordered)
        result = ser.astype(dtype)
        expected = Series(s_data, name=name, dtype=dtype)
        tm.assert_series_equal(result, expected)
        if dtype_ordered is False:
            expected = ser
            result = ser.astype('category')
            tm.assert_series_equal(result, expected)

    def test_astype_bool_missing_to_categorical(self) -> None:
        ser = Series([True, False, np.nan])
        assert ser.dtypes == np.object_
        result = ser.astype(CategoricalDtype(categories=[True, False]))
        expected = Series(Categorical([True, False, np.nan], categories=[True, False]))
        tm.assert_series_equal(result, expected)

    def test_astype_categories_raises(self) -> None:
        ser = Series(['a', 'b', 'a'])
        with pytest.raises(TypeError, match='got an unexpected'):
            ser.astype('category', categories=['a', 'b'], ordered=True)

    @pytest.mark.parametrize('items', [['a', 'b', 'c', 'a'], [1, 2, 3, 1]])
    def test_astype_from_categorical(self, items: List[Union[str, int]]) -> None:
        ser = Series(items)
        exp = Series(Categorical(items))
        res = ser.astype('category')
        tm.assert_series_equal(res, exp)

    def test_astype_from_categorical_with_keywords(self) -> None:
        lst = ['a', 'b', 'c', 'a']
        ser = Series(lst)
        exp = Series(Categorical(lst, ordered=True))
        res = ser.astype(CategoricalDtype(None, ordered=True))
        tm.assert_series_equal(res, exp)
        exp = Series(Categorical(lst, categories=list('abcdef'), ordered=True))
        res = ser.astype(CategoricalDtype(list('abcdef'), ordered=True))
        tm.assert_series_equal(res, exp)

    def test_astype_timedelta64_with_np_nan(self) -> None:
        result = Series([Timedelta(1), np.nan], dtype='timedelta64[ns]')
        expected = Series([Timedelta(1), NaT], dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected)

    @td.skip_if_no('pyarrow')
    def test_astype_int_na_string(self) -> None:
        ser = Series([12, NA], dtype='Int64[pyarrow]')
        result = ser.astype('string[pyarrow]')
        expected = Series(['12', NA], dtype='string[pyarrow]')
        tm.assert_series_equal(result, expected)
