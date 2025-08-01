from datetime import datetime, timedelta
from importlib import reload
import string
import sys
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
    NA,
    Categorical,
    CategoricalDtype,
    DatetimeTZDtype,
    Index,
    Interval,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    to_datetime,
)
import pandas._testing as tm


def rand_str(nchars: int) -> str:
    """
    Generate one random byte string.
    """
    RANDS_CHARS: np.ndarray = np.array(
        list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
    )
    return ''.join(np.random.default_rng(2).choice(RANDS_CHARS, nchars))


class TestAstypeAPI:

    def test_astype_unitless_dt64_raises(self) -> None:
        ser: Series = Series(
            ['1970-01-01', '1970-01-01', '1970-01-01'], dtype='datetime64[ns]'
        )
        df: Series = ser.to_frame()
        msg: str = "Casting to unit-less dtype 'datetime64' is not supported"
        with pytest.raises(TypeError, match=msg):
            ser.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            df.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            ser.astype('datetime64')
        with pytest.raises(TypeError, match=msg):
            df.astype('datetime64')

    def test_arg_for_errors_in_astype(self) -> None:
        ser: Series = Series([1, 2, 3])
        msg: str = (
            "Expected value of kwarg 'errors' to be one of \\['raise', 'ignore'\\]\\. "
            "Supplied value is 'False'"
        )
        with pytest.raises(ValueError, match=msg):
            ser.astype(np.float64, errors=False)
        ser.astype(np.int8, errors='raise')

    @pytest.mark.parametrize('dtype_class', [dict, Series])
    def test_astype_dict_like(self, dtype_class: Type[Union[dict, Series]]) -> None:
        ser: Series = Series(range(0, 10, 2), name='abc')
        dt1: Union[dict, Series] = dtype_class({'abc': str})
        result: Series = ser.astype(dt1)
        expected: Series = Series(['0', '2', '4', '6', '8'], name='abc', dtype='str')
        tm.assert_series_equal(result, expected)

        dt2: Union[dict, Series] = dtype_class({'abc': 'float64'})
        result = ser.astype(dt2)
        expected = Series([0.0, 2.0, 4.0, 6.0, 8.0], dtype='float64', name='abc')
        tm.assert_series_equal(result, expected)

        dt3: Union[dict, Series] = dtype_class({'abc': str, 'def': str})
        msg: str = 'Only the Series name can be used for the key in Series dtype mappings\\.'
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt3)

        dt4: Union[dict, Series] = dtype_class({0: str})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt4)

        if dtype_class is Series:
            dt5: Union[dict, Series] = dtype_class({}, dtype=object)
        else:
            dt5 = dtype_class({})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt5)


class TestAstype:

    @pytest.mark.parametrize('tz', [None, 'UTC', 'US/Pacific'])
    def test_astype_object_to_dt64_non_nano(self, tz: Optional[str]) -> None:
        ts: Timestamp = Timestamp('2999-01-01')
        dtype: str = 'M8[us]' if tz is None else f'M8[us, {tz}]'
        vals: List[Union[Timestamp, str, int]] = [
            ts,
            '2999-01-02 03:04:05.678910',
            2500
        ]
        ser: Series = Series(vals, dtype=object)
        result: Series = ser.astype(dtype)

        pointwise: List[Timestamp] = [
            vals[0].tz_localize(tz) if isinstance(vals[0], Timestamp) else vals[0],
            Timestamp(vals[1], tz=tz),
            to_datetime(vals[2], unit='us', utc=True).tz_convert(tz)
        ]
        exp_vals: List[np.datetime64] = [x.as_unit('us').asm8 for x in pointwise]
        exp_arr: np.ndarray = np.array(exp_vals, dtype='M8[us]')
        expected: Series = Series(exp_arr, dtype='M8[us]')
        if tz is not None:
            expected = expected.dt.tz_localize('UTC').dt.tz_convert(tz)
        tm.assert_series_equal(result, expected)

    def test_astype_mixed_object_to_dt64tz(self) -> None:
        ts: Timestamp = Timestamp('2016-01-04 05:06:07', tz='US/Pacific')
        ts2: Timestamp = ts.tz_convert('Asia/Tokyo')
        ser: Series = Series([ts, ts2], dtype=object)
        res: Series = ser.astype('datetime64[ns, Europe/Brussels]')
        expected: Series = Series(
            [ts.tz_convert('Europe/Brussels'), ts2.tz_convert('Europe/Brussels')],
            dtype='datetime64[ns, Europe/Brussels]'
        )
        tm.assert_series_equal(res, expected)

    @pytest.mark.parametrize('dtype', list(np.typecodes['All']))
    def test_astype_empty_constructor_equality(self, dtype: str) -> None:
        if dtype not in {'S', 'V', 'M', 'm'}:
            init_empty: Series = Series([], dtype=dtype)
            as_type_empty: Series = Series([]).astype(dtype)
            tm.assert_series_equal(init_empty, as_type_empty)

    @pytest.mark.parametrize('dtype', [str, np.str_])
    @pytest.mark.parametrize(
        'data',
        [
            [string.digits * 10, rand_str(63), rand_str(64), rand_str(1000)],
            [string.digits * 10, rand_str(63), rand_str(64), np.nan, 1.0]
        ]
    )
    def test_astype_str_map(
        self,
        dtype: Type[Union[str, np.str_]],
        data: List[Any],
        using_infer_string: bool
    ) -> None:
        series: Series = Series(data)
        using_string_dtype: bool = using_infer_string and dtype is str
        result: Series = series.astype(dtype)
        if using_string_dtype:
            expected: Series = series.map(lambda val: str(val) if val is not np.nan else np.nan)
        else:
            expected = series.map(str)
            if using_infer_string:
                expected = expected.astype(object)
        tm.assert_series_equal(result, expected)

    def test_astype_float_to_period(self) -> None:
        result: Series = Series([np.nan]).astype('period[D]')
        expected: Series = Series([NaT], dtype='period[D]')
        tm.assert_series_equal(result, expected)

    def test_astype_no_pandas_dtype(self) -> None:
        ser: Series = Series([1, 2], dtype='int64')
        result: Series = ser.astype(ser.array.dtype)
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize('dtype', [np.datetime64, np.timedelta64])
    def test_astype_generic_timestamp_no_frequency(
        self,
        dtype: Type[Union[np.datetime64, np.timedelta64]],
        request: pytest.FixtureRequest
    ) -> None:
        data: List[int] = [1]
        ser: Series = Series(data)
        if np.dtype(dtype).name not in {'timedelta64', 'datetime64'}:
            mark = pytest.mark.xfail(reason='GH#33890 Is assigned ns unit')
            request.applymarker(mark)
        msg: str = (
            f"The '{dtype.__name__}' dtype has no unit\\. "
            f"Please pass in '{dtype.__name__}\\[ns\\]' instead."
        )
        with pytest.raises(ValueError, match=msg):
            ser.astype(dtype)

    def test_astype_dt64_to_str(self) -> None:
        dti: pd.DatetimeIndex = date_range('2012-01-01', periods=3)
        result: Series = Series(dti).astype(str)
        expected: Series = Series(['2012-01-01', '2012-01-02', '2012-01-03'], dtype='str')
        tm.assert_series_equal(result, expected)

    def test_astype_dt64tz_to_str(self) -> None:
        dti_tz: pd.DatetimeIndex = date_range('2012-01-01', periods=3, tz='US/Eastern')
        result: Series = Series(dti_tz).astype(str)
        expected: Series = Series(
            [
                '2012-01-01 00:00:00-05:00',
                '2012-01-02 00:00:00-05:00',
                '2012-01-03 00:00:00-05:00'
            ],
            dtype='str'
        )
        tm.assert_series_equal(result, expected)

    def test_astype_datetime(self, unit: str) -> None:
        ser: Series = Series(iNaT, dtype=f'M8[{unit}]', index=range(5))
        ser = ser.astype('O')
        assert ser.dtype == np.object_
        ser = Series([datetime(2001, 1, 2, 0, 0)])
        ser = ser.astype('O')
        assert ser.dtype == np.object_
        ser = Series(
            [datetime(2001, 1, 2, 0, 0) for _ in range(3)], dtype=f'M8[{unit}]'
        )
        ser[1] = np.nan
        assert ser.dtype == f'M8[{unit}]'
        ser = ser.astype('O')
        assert ser.dtype == np.object_

    def test_astype_datetime64tz(self) -> None:
        ser: Series = Series(date_range('20130101', periods=3, tz='US/Eastern'))
        result: Series = ser.astype(object)
        expected: Series = Series(ser.astype(object), dtype=object)
        tm.assert_series_equal(result, expected)

        result = Series(ser.values).dt.tz_localize('UTC').dt.tz_convert(ser.dt.tz)
        tm.assert_series_equal(result, ser)

        result = Series(ser.astype(object))
        expected = ser.astype(object)
        tm.assert_series_equal(result, expected)

        msg: str = 'Cannot use .astype to convert from timezone-naive'
        with pytest.raises(TypeError, match=msg):
            Series(ser.values).astype('datetime64[ns, US/Eastern]')
        with pytest.raises(TypeError, match=msg):
            Series(ser.values).astype(ser.dtype)

        result = ser.astype('datetime64[ns, CET]')
        expected = Series(date_range('20130101 06:00:00', periods=3, tz='CET'))
        tm.assert_series_equal(result, expected)

    def test_astype_str_cast_dt64(self) -> None:
        ts: Series = Series([Timestamp('2010-01-04 00:00:00')])
        res: Series = ts.astype(str)
        expected: Series = Series(['2010-01-04'], dtype='str')
        tm.assert_series_equal(res, expected)

        ts = Series([Timestamp('2010-01-04 00:00:00', tz='US/Eastern')])
        res = ts.astype(str)
        expected = Series(['2010-01-04 00:00:00-05:00'], dtype='str')
        tm.assert_series_equal(res, expected)

    def test_astype_str_cast_td64(self) -> None:
        td: Series = Series([Timedelta(1, unit='D')])
        ser: Series = td.astype(str)
        expected: Series = Series(['1 days'], dtype='str')
        tm.assert_series_equal(ser, expected)

    def test_dt64_series_astype_object(self) -> None:
        dt64ser: Series = Series(date_range('20130101', periods=3))
        result: Series = dt64ser.astype(object)
        assert isinstance(result.iloc[0], datetime)
        assert result.dtype == np.object_

    def test_td64_series_astype_object(self) -> None:
        tdser: Series = Series(['59 Days', '59 Days', 'NaT'], dtype='timedelta64[ns]')
        result: Series = tdser.astype(object)
        assert isinstance(result.iloc[0], timedelta)
        assert result.dtype == np.object_

    @pytest.mark.parametrize(
        'data, dtype',
        [
            (['x', 'y', 'z'], 'string[python]'),
            pytest.param(
                ['x', 'y', 'z'],
                'string[pyarrow]',
                marks=td.skip_if_no('pyarrow')
            ),
            (['x', 'y', 'z'], 'category'),
            (3 * [Timestamp('2020-01-01', tz='UTC')], None),
            (3 * [Interval(0, 1)], None)
        ]
    )
    @pytest.mark.parametrize('errors', ['raise', 'ignore'])
    def test_astype_ignores_errors_for_extension_dtypes(
        self,
        data: List[Any],
        dtype: Optional[str],
        errors: str
    ) -> None:
        ser: Series = Series(data, dtype=dtype)
        if errors == 'ignore':
            expected: Series = ser
            result: Series = ser.astype(float, errors='ignore')
            tm.assert_series_equal(result, expected)
        else:
            msg: str = '(Cannot cast)|(could not convert)'
            with pytest.raises((ValueError, TypeError), match=msg):
                ser.astype(float, errors=errors)

    def test_astype_from_float_to_str(self, any_float_dtype: str) -> None:
        ser: Series = Series([0.1], dtype=any_float_dtype)
        result: Series = ser.astype(str)
        expected: Series = Series(['0.1'], dtype='str')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'value, string_value',
        [
            (None, 'None'),
            (np.nan, 'nan'),
            (NA, '<NA>')
        ]
    )
    def test_astype_to_str_preserves_na(
        self,
        value: Any,
        string_value: str,
        using_infer_string: bool
    ) -> None:
        ser: Series = Series(['a', 'b', value], dtype=object)
        result: Series = ser.astype(str)
        expected: Series = Series(
            ['a', 'b', None if using_infer_string else string_value],
            dtype='str'
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64', 'int32'])
    def test_astype(self, dtype: str) -> None:
        ser: Series = Series(np.random.default_rng(2).standard_normal(5), name='foo')
        as_typed: Series = ser.astype(dtype)
        assert as_typed.dtype == dtype
        assert as_typed.name == ser.name

    @pytest.mark.parametrize('value', [np.nan, np.inf])
    def test_astype_cast_nan_inf_int(
        self,
        any_int_numpy_dtype: str,
        value: float
    ) -> None:
        msg: str = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        ser: Series = Series([value])
        with pytest.raises(ValueError, match=msg):
            ser.astype(any_int_numpy_dtype)

    def test_astype_cast_object_int_fail(self, any_int_numpy_dtype: str) -> None:
        arr: Series = Series(['car', 'house', 'tree', '1'])
        msg: str = "invalid literal for int\\(\\) with base 10: 'car'"
        with pytest.raises(ValueError, match=msg):
            arr.astype(any_int_numpy_dtype)

    def test_astype_float_to_uint_negatives_raise(
        self,
        float_numpy_dtype: str,
        any_unsigned_int_numpy_dtype: str
    ) -> None:
        arr: np.ndarray = np.arange(5).astype(float_numpy_dtype) - 3
        ser: Series = Series(arr)
        msg: str = 'Cannot losslessly cast from .* to .*'
        with pytest.raises(ValueError, match=msg):
            ser.astype(any_unsigned_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            ser.to_frame().astype(any_unsigned_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            Index(ser).astype(any_unsigned_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            ser.array.astype(any_unsigned_int_numpy_dtype)

    def test_astype_cast_object_int(self) -> None:
        arr: Series = Series(['1', '2', '3', '4'], dtype=object)
        result: Series = arr.astype(int)
        tm.assert_series_equal(result, Series(np.arange(1, 5)))

    def test_astype_unicode(self, using_infer_string: bool) -> None:
        digits: str = string.digits
        test_series: List[Series] = [
            Series([digits * 10, rand_str(63), rand_str(64), rand_str(1000)]),
            Series(['データーサイエンス、お前はもう死んでいる'])
        ]
        former_encoding = None
        if sys.getdefaultencoding() == 'utf-8':
            item: str = '野菜食べないとやばい'
            ser: Series = Series([item.encode()])
            result: Series = ser.astype(np.str_)
            expected: Series = Series([item], dtype=object)
            tm.assert_series_equal(result, expected)
        for ser in test_series:
            res: Series = ser.astype(np.str_)
            expec: Series = ser.map(str)
            if using_infer_string:
                expec = expec.astype(object)
            tm.assert_series_equal(res, expec)
        if former_encoding is not None and former_encoding != 'utf-8':
            reload(sys)
            sys.setdefaultencoding(former_encoding)

    def test_astype_bytes(self) -> None:
        result: Series = Series(['foo', 'bar', 'baz']).astype(bytes)
        assert result.dtypes == np.dtype('S3')

    def test_astype_nan_to_bool(self) -> None:
        ser: Series = Series(np.nan, dtype='object')
        result: Series = ser.astype('bool')
        expected: Series = Series(True, dtype='bool')
        tm.assert_series_equal(result, expected)

    def test_astype_ea_to_datetimetzdtype(
        self,
        any_numeric_ea_dtype: str
    ) -> None:
        ser: Series = Series([4, 0, 9], dtype=any_numeric_ea_dtype)
        result: Series = ser.astype(DatetimeTZDtype(tz='US/Pacific'))
        expected: Series = Series([
            Timestamp('1969-12-31 16:00:00.000000004-08:00', tz='US/Pacific'),
            Timestamp('1969-12-31 16:00:00.000000000-08:00', tz='US/Pacific'),
            Timestamp('1969-12-31 16:00:00.000000009-08:00', tz='US/Pacific')
        ])
        tm.assert_series_equal(result, expected)

    def test_astype_retain_attrs(
        self,
        any_numpy_dtype: str
    ) -> None:
        ser: Series = Series([0, 1, 2, 3])
        ser.attrs['Location'] = 'Michigan'
        result: Dict[str, Any] = ser.astype(any_numpy_dtype).attrs
        expected: Dict[str, Any] = ser.attrs
        tm.assert_dict_equal(expected, result)


class TestAstypeString:

    @pytest.mark.parametrize(
        'data, dtype',
        [
            (['x', 'y', 'z'], 'string[python]'),
            pytest.param(
                ['x', 'y', 'z'],
                'string[pyarrow]',
                marks=td.skip_if_no('pyarrow')
            ),
            (['x', 'y', 'z'], 'category'),
            (3 * [Timestamp('2020-01-01', tz='UTC')], None),
            (3 * [Interval(0, 1)], None)
        ]
    )
    @pytest.mark.parametrize('errors', ['raise', 'ignore'])
    def test_astype_string_to_extension_dtype_roundtrip(
        self,
        data: List[Any],
        dtype: Optional[str],
        request: pytest.FixtureRequest,
        nullable_string_dtype: str
    ) -> None:
        if dtype == 'boolean':
            mark = pytest.mark.xfail(
                reason='TODO StringArray.astype() with missing values #GH40566'
            )
            request.applymarker(mark)
        ser: Series = Series(data, dtype=dtype)
        result: Series = ser.astype(nullable_string_dtype).astype(ser.dtype)
        tm.assert_series_equal(result, ser)


class TestAstypeCategorical:

    def test_astype_categorical_to_other(self) -> None:
        cat: Categorical = Categorical([f'{i} - {i + 499}' for i in range(0, 10000, 500)])
        ser: Series = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)
        expected: Series = ser
        tm.assert_series_equal(ser.astype('category'), expected)
        tm.assert_series_equal(ser.astype(CategoricalDtype()), expected)
        msg: str = 'Cannot cast object|str dtype to float64'
        with pytest.raises(ValueError, match=msg):
            ser.astype('float64')

        cat_series: Series = Series(Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']))
        exp: Series = Series(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], dtype='str')
        tm.assert_series_equal(cat_series.astype('str'), exp)

        s2: Series = Series(Categorical(['1', '2', '3', '4']))
        exp2: Series = Series([1, 2, 3, 4]).astype('int')
        tm.assert_series_equal(s2.astype('int'), exp2)

        def cmp(a: Any, b: Any) -> None:
            tm.assert_almost_equal(
                np.sort(np.unique(a)),
                np.sort(np.unique(b))
            )
        
        expected = Series(np.array(ser.values), name='value_group')
        cmp(ser.astype('object'), expected)
        cmp(ser.astype(np.object_), expected)
        tm.assert_almost_equal(np.array(ser), np.array(ser.values))
        tm.assert_series_equal(ser.astype('category'), ser)
        tm.assert_series_equal(ser.astype(CategoricalDtype()), ser)
        roundtrip_expected: Series = ser.cat.set_categories(
            ser.cat.categories.sort_values()
        ).cat.remove_unused_categories()
        result: Series = ser.astype('object').astype('category')
        tm.assert_series_equal(result, roundtrip_expected)
        result = ser.astype('object').astype(CategoricalDtype())
        tm.assert_series_equal(result, roundtrip_expected)

    def test_astype_categorical_invalid_conversions(self) -> None:
        cat: Categorical = Categorical([f'{i} - {i + 499}' for i in range(0, 10000, 500)])
        ser: Series = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)
        msg: str = "dtype '<class 'pandas.core.arrays.categorical.Categorical'>' not understood"
        with pytest.raises(TypeError, match=msg):
            ser.astype(Categorical)
        with pytest.raises(TypeError, match=msg):
            ser.astype('object').astype(Categorical)

    def test_astype_categoricaldtype(self) -> None:
        ser: Series = Series(['a', 'b', 'a'])
        result: Series = ser.astype(CategoricalDtype(['a', 'b'], ordered=True))
        expected: Series = Series(Categorical(['a', 'b', 'a'], ordered=True))
        tm.assert_series_equal(result, expected)

        result = ser.astype(CategoricalDtype(['a', 'b'], ordered=False))
        expected = Series(Categorical(['a', 'b', 'a'], ordered=False))
        tm.assert_series_equal(result, expected)

        result = ser.astype(CategoricalDtype(['a', 'b', 'c'], ordered=False))
        expected = Series(
            Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c'], ordered=False)
        )
        tm.assert_series_equal(result, expected)
        tm.assert_index_equal(result.cat.categories, Index(['a', 'b', 'c']))

    @pytest.mark.parametrize('name', [None, 'foo'])
    @pytest.mark.parametrize('dtype_ordered', [True, False])
    @pytest.mark.parametrize('series_ordered', [True, False])
    def test_astype_categorical_to_categorical(
        self,
        name: Optional[str],
        dtype_ordered: bool,
        series_ordered: bool
    ) -> None:
        s_data: List[str] = list('abcaacbab')
        s_dtype: CategoricalDtype = CategoricalDtype(list('bac'), ordered=series_ordered)
        ser: Series = Series(s_data, dtype=s_dtype, name=name)
        dtype: CategoricalDtype = CategoricalDtype(ordered=dtype_ordered)
        result: Series = ser.astype(dtype)
        exp_dtype: CategoricalDtype = CategoricalDtype(s_dtype.categories, dtype_ordered)
        expected: Series = Series(s_data, name=name, dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

        dtype = CategoricalDtype(list('adc'), ordered=dtype_ordered)
        result = ser.astype(dtype)
        expected = Series(s_data, name=name, dtype=dtype)
        tm.assert_series_equal(result, expected)

        if not dtype_ordered:
            expected = ser
            result = ser.astype('category')
            tm.assert_series_equal(result, expected)

    def test_astype_bool_missing_to_categorical(self) -> None:
        ser: Series = Series([True, False, np.nan])
        assert ser.dtypes == np.object_
        result: Series = ser.astype(CategoricalDtype(categories=[True, False]))
        expected: Series = Series(
            Categorical([True, False, np.nan], categories=[True, False])
        )
        tm.assert_series_equal(result, expected)

    def test_astype_categories_raises(self) -> None:
        ser: Series = Series(['a', 'b', 'a'])
        with pytest.raises(TypeError, match='got an unexpected'):
            ser.astype('category', categories=['a', 'b'], ordered=True)

    @pytest.mark.parametrize('items', [['a', 'b', 'c', 'a'], [1, 2, 3, 1]])
    def test_astype_from_categorical(self, items: List[Union[str, int]]) -> None:
        ser: Series = Series(items)
        exp: Series = Series(Categorical(items))
        res: Series = ser.astype('category')
        tm.assert_series_equal(res, exp)

    def test_astype_from_categorical_with_keywords(self) -> None:
        lst: List[str] = ['a', 'b', 'c', 'a']
        ser: Series = Series(lst)
        exp: Series = Series(Categorical(lst, ordered=True))
        res: Series = ser.astype(CategoricalDtype(None, ordered=True))
        tm.assert_series_equal(res, exp)

        exp = Series(Categorical(lst, categories=list('abcdef'), ordered=True))
        res = ser.astype(CategoricalDtype(list('abcdef'), ordered=True))
        tm.assert_series_equal(res, exp)

    def test_astype_timedelta64_with_np_nan(self) -> None:
        result: Series = Series([Timedelta(1), np.nan], dtype='timedelta64[ns]')
        expected: Series = Series([Timedelta(1), NaT], dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected)

    @td.skip_if_no('pyarrow')
    def test_astype_int_na_string(self) -> None:
        ser: Series = Series([12, NA], dtype='Int64[pyarrow]')
        result: Series = ser.astype('string[pyarrow]')
        expected: Series = Series(['12', NA], dtype='string[pyarrow]')
        tm.assert_series_equal(result, expected)
