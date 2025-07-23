import re
import weakref
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_dtype_equal,
    is_interval_dtype,
    is_period_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    IntervalIndex,
    Series,
    SparseDtype,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


class Base:
    def test_hash(self, dtype: Union[CategoricalDtype, DatetimeTZDtype, PeriodDtype, IntervalDtype]) -> None:
        hash(dtype)

    def test_equality_invalid(self, dtype: Union[CategoricalDtype, DatetimeTZDtype, PeriodDtype, IntervalDtype]) -> None:
        assert not dtype == 'foo'
        assert not is_dtype_equal(dtype, np.int64)

    def test_numpy_informed(self, dtype: Union[CategoricalDtype, DatetimeTZDtype, PeriodDtype, IntervalDtype]) -> None:
        msg = '|'.join(['data type not understood', "Cannot interpret '.*' as a data type"])
        with pytest.raises(TypeError, match=msg):
            np.dtype(dtype)
        assert not dtype == np.str_
        assert not np.str_ == dtype

    def test_pickle(self, dtype: Union[CategoricalDtype, DatetimeTZDtype, PeriodDtype, IntervalDtype]) -> None:
        type(dtype).reset_cache()
        assert not len(dtype._cache_dtypes)
        result = tm.round_trip_pickle(dtype)
        if not isinstance(dtype, PeriodDtype):
            assert not len(dtype._cache_dtypes)
        assert result == dtype


class TestCategoricalDtype(Base):
    @pytest.fixture
    def dtype(self) -> CategoricalDtype:
        """
        Class level fixture of dtype for TestCategoricalDtype
        """
        return CategoricalDtype()

    def test_hash_vs_equality(self, dtype: CategoricalDtype) -> None:
        dtype2 = CategoricalDtype()
        assert dtype == dtype2
        assert dtype2 == dtype
        assert hash(dtype) == hash(dtype2)

    def test_equality(self, dtype: CategoricalDtype) -> None:
        assert dtype == 'category'
        assert is_dtype_equal(dtype, 'category')
        assert 'category' == dtype
        assert is_dtype_equal('category', dtype)
        assert dtype == CategoricalDtype()
        assert is_dtype_equal(dtype, CategoricalDtype())
        assert CategoricalDtype() == dtype
        assert is_dtype_equal(CategoricalDtype(), dtype)
        assert dtype != 'foo'
        assert not is_dtype_equal(dtype, 'foo')
        assert 'foo' != dtype
        assert not is_dtype_equal('foo', dtype)

    def test_construction_from_string(self, dtype: CategoricalDtype) -> None:
        result = CategoricalDtype.construct_from_string('category')
        assert is_dtype_equal(dtype, result)
        msg = "Cannot construct a 'CategoricalDtype' from 'foo'"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype.construct_from_string('foo')

    def test_constructor_invalid(self) -> None:
        msg = "Parameter 'categories' must be list-like"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype('category')

    dtype1 = CategoricalDtype(['a', 'b'], ordered=True)
    dtype2 = CategoricalDtype(['x', 'y'], ordered=False)
    c = Categorical([0, 1], dtype=dtype1)

    @pytest.mark.parametrize(
        'values, categories, ordered, dtype, expected',
        [
            [None, None, None, None, CategoricalDtype()],
            [None, ['a', 'b'], True, None, dtype1],
            [c, None, None, dtype2, dtype2],
            [c, ['x', 'y'], False, None, dtype2],
        ],
    )
    def test_from_values_or_dtype(
        self,
        values: Optional[Categorical],
        categories: Optional[List[str]],
        ordered: Optional[bool],
        dtype: Optional[CategoricalDtype],
        expected: CategoricalDtype,
    ) -> None:
        result = CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)
        assert result == expected

    @pytest.mark.parametrize(
        'values, categories, ordered, dtype',
        [
            [None, ['a', 'b'], True, dtype2],
            [None, ['a', 'b'], None, dtype2],
            [None, None, True, dtype2],
        ],
    )
    def test_from_values_or_dtype_raises(
        self,
        values: Optional[Categorical],
        categories: Optional[List[str]],
        ordered: Optional[bool],
        dtype: CategoricalDtype,
    ) -> None:
        msg = 'Cannot specify `categories` or `ordered` together with `dtype`.'
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)

    def test_from_values_or_dtype_invalid_dtype(self) -> None:
        msg = "Cannot not construct CategoricalDtype from <class 'object'>"
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(None, None, None, object)

    def test_is_dtype(self, dtype: CategoricalDtype) -> None:
        assert CategoricalDtype.is_dtype(dtype)
        assert CategoricalDtype.is_dtype('category')
        assert CategoricalDtype.is_dtype(CategoricalDtype())
        assert not CategoricalDtype.is_dtype('foo')
        assert not CategoricalDtype.is_dtype(np.float64)

    def test_basic(self, dtype: CategoricalDtype) -> None:
        msg = 'is_categorical_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_categorical_dtype(dtype)
            factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'])
            s = Series(factor, name='A')
            assert is_categorical_dtype(s.dtype)
            assert is_categorical_dtype(s)
            assert not is_categorical_dtype(np.dtype('float64'))

    def test_tuple_categories(self) -> None:
        categories = [(1, 'a'), (2, 'b'), (3, 'c')]
        result = CategoricalDtype(categories)
        assert all(result.categories == categories)

    @pytest.mark.parametrize(
        'categories, expected',
        [
            ([True, False], True),
            ([True, False, None], True),
            ([True, False, 'a', "b'"], False),
            ([0, 1], False),
        ],
    )
    def test_is_boolean(self, categories: List[Any], expected: bool) -> None:
        cat = Categorical(categories)
        assert cat.dtype._is_boolean is expected
        assert is_bool_dtype(cat) is expected
        assert is_bool_dtype(cat.dtype) is expected

    def test_dtype_specific_categorical_dtype(self) -> None:
        expected = 'datetime64[ns]'
        dti = DatetimeIndex([], dtype=expected)
        result = str(Categorical(dti).categories.dtype)
        assert result == expected

    def test_not_string(self) -> None:
        assert not is_string_dtype(CategoricalDtype())

    def test_repr_range_categories(self) -> None:
        rng = pd.Index(range(3))
        dtype = CategoricalDtype(categories=rng, ordered=False)
        result = repr(dtype)
        expected = 'CategoricalDtype(categories=range(0, 3), ordered=False, categories_dtype=int64)'
        assert result == expected

    def test_update_dtype(self) -> None:
        result = CategoricalDtype(['a']).update_dtype(Categorical(['b'], ordered=True))
        expected = CategoricalDtype(['b'], ordered=True)
        assert result == expected

    def test_repr(self) -> None:
        cat = Categorical(pd.Index([1, 2, 3], dtype='int32'))
        result = cat.dtype.__repr__()
        expected = 'CategoricalDtype(categories=[1, 2, 3], ordered=False, categories_dtype=int32)'
        assert result == expected


class TestDatetimeTZDtype(Base):
    @pytest.fixture
    def dtype(self) -> DatetimeTZDtype:
        """
        Class level fixture of dtype for TestDatetimeTZDtype
        """
        return DatetimeTZDtype('ns', 'US/Eastern')

    def test_alias_to_unit_raises(self) -> None:
        with pytest.raises(ValueError, match='Passing a dtype alias'):
            DatetimeTZDtype('datetime64[ns, US/Central]')

    def test_alias_to_unit_bad_alias_raises(self) -> None:
        with pytest.raises(TypeError, match=''):
            DatetimeTZDtype('this is a bad string')
        with pytest.raises(TypeError, match=''):
            DatetimeTZDtype('datetime64[ns, US/NotATZ]')

    def test_hash_vs_equality(self, dtype: DatetimeTZDtype) -> None:
        dtype2 = DatetimeTZDtype('ns', 'US/Eastern')
        dtype3 = DatetimeTZDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)
        dtype4 = DatetimeTZDtype('ns', 'US/Central')
        assert dtype2 != dtype4
        assert hash(dtype2) != hash(dtype4)

    def test_construction_non_nanosecond(self) -> None:
        res = DatetimeTZDtype('ms', 'US/Eastern')
        assert res.unit == 'ms'
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res.str == '|M8[ms]'
        assert str(res) == 'datetime64[ms, US/Eastern]'
        assert res.base == np.dtype('M8[ms]')

    def test_day_not_supported(self) -> None:
        msg = 'DatetimeTZDtype only supports s, ms, us, ns units'
        with pytest.raises(ValueError, match=msg):
            DatetimeTZDtype('D', 'US/Eastern')

    def test_subclass(self) -> None:
        a = DatetimeTZDtype.construct_from_string('datetime64[ns, US/Eastern]')
        b = DatetimeTZDtype.construct_from_string('datetime64[ns, CET]')
        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_compat(self, dtype: DatetimeTZDtype) -> None:
        msg = 'is_datetime64tz_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
            assert is_datetime64tz_dtype('datetime64[ns, US/Eastern]')
        assert is_datetime64_any_dtype(dtype)
        assert is_datetime64_any_dtype('datetime64[ns, US/Eastern]')
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_ns_dtype('datetime64[ns, US/Eastern]')
        assert not is_datetime64_dtype(dtype)
        assert not is_datetime64_dtype('datetime64[ns, US/Eastern]')

    def test_construction_from_string(self, dtype: DatetimeTZDtype) -> None:
        result = DatetimeTZDtype.construct_from_string('datetime64[ns, US/Eastern]')
        assert is_dtype_equal(dtype, result)

    @pytest.mark.parametrize(
        'string',
        ['foo', 'datetime64[ns, notatz]', 'datetime64[ps, UTC]', 'datetime64[ns, dateutil/invalid]'],
    )
    def test_construct_from_string_invalid_raises(self, string: str) -> None:
        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        with pytest.raises(TypeError, match=re.escape(msg)):
            DatetimeTZDtype.construct_from_string(string)

    def test_construct_from_string_wrong_type_raises(self) -> None:
        msg = "'construct_from_string' expects a string, got <class 'list'>"
        with pytest.raises(TypeError, match=msg):
            DatetimeTZDtype.construct_from_string(['datetime64[ns, notatz]'])

    def test_is_dtype(self, dtype: DatetimeTZDtype) -> None:
        assert not DatetimeTZDtype.is_dtype(None)
        assert DatetimeTZDtype.is_dtype(dtype)
        assert DatetimeTZDtype.is_dtype('datetime64[ns, US/Eastern]')
        assert DatetimeTZDtype.is_dtype('M8[ns, US/Eastern]')
        assert not DatetimeTZDtype.is_dtype('foo')
        assert DatetimeTZDtype.is_dtype(DatetimeTZDtype('ns', 'US/Pacific'))
        assert not DatetimeTZDtype.is_dtype(np.float64)

    def test_equality(self, dtype: DatetimeTZDtype) -> None:
        assert is_dtype_equal(dtype, 'datetime64[ns, US/Eastern]')
        assert is_dtype_equal(dtype, 'M8[ns, US/Eastern]')
        assert is_dtype_equal(dtype, DatetimeTZDtype('ns', 'US/Eastern'))
        assert not is_dtype_equal(dtype, 'foo')
        assert not is_dtype_equal(dtype, DatetimeTZDtype('ns', 'CET'))
        assert not is_dtype_equal(DatetimeTZDtype('ns', 'US/Eastern'), DatetimeTZDtype('ns', 'US/Pacific'))
        assert is_dtype_equal(np.dtype('M8[ns]'), 'datetime64[ns]')
        assert dtype == 'M8[ns, US/Eastern]'

    def test_basic(self, dtype: DatetimeTZDtype) -> None:
        msg = 'is_datetime64tz_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
        dr = date_range('20130101', periods=3, tz='US/Eastern')
        s = Series(dr, name='A')
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(s.dtype)
            assert is_datetime64tz_dtype(s)
            assert not is_datetime64tz_dtype(np.dtype('float64'))
            assert not is_datetime64tz_dtype(1.0)

    def test_dst(self) -> None:
        dr1 = date_range('2013-01-01', periods=3, tz='US/Eastern')
        s1 = Series(dr1, name='A')
        assert isinstance(s1.dtype, DatetimeTZDtype)
        dr2 = date_range('2013-08-01', periods=3, tz='US/Eastern')
        s2 = Series(dr2, name='A')
        assert isinstance(s2.dtype, DatetimeTZDtype)
        assert s1.dtype == s2.dtype

    @pytest.mark.parametrize('tz', ['UTC', 'US/Eastern'])
    @pytest.mark.parametrize('constructor', ['M8', 'datetime64'])
    def test_parser(self, tz: str, constructor: str) -> None:
        dtz_str = f'{constructor}[ns, {tz}]'
        result = DatetimeTZDtype.construct_from_string(dtz_str)
        expected = DatetimeTZDtype('ns', tz)
        assert result == expected

    def test_empty(self) -> None:
        with pytest.raises(TypeError, match="A 'tz' is required."):
            DatetimeTZDtype()

    def test_tz_standardize(self) -> None:
        pytz = pytest.importorskip('pytz')
        tz = pytz.timezone('US/Eastern')
        dr = date_range('2013-01-01', periods=3, tz=tz)
        dtype = DatetimeTZDtype('ns', dr.tz)
        assert dtype.tz == tz
        dtype = DatetimeTZDtype('ns', dr[0].tz)
        assert dtype.tz == tz


class TestPeriodDtype(Base):
    @pytest.fixture
    def dtype(self) -> PeriodDtype:
        """
        Class level fixture of dtype for TestPeriodDtype
        """
        return PeriodDtype('D')

    def test_hash_vs_equality(self, dtype: PeriodDtype) -> None:
        dtype2 = PeriodDtype('D')
        dtype3 = PeriodDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert dtype is not dtype2
        assert dtype2 is not dtype
        assert dtype3 is not dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

    def test_construction(self)