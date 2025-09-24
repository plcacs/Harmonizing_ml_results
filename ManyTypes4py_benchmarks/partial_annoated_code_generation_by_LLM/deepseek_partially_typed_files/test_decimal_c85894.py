from __future__ import annotations
import decimal
import operator
from typing import Any, Callable, List, Optional, Tuple, Type, Union, cast
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import DecimalArray, DecimalDtype, make_data, to_decimal
from pandas.core.dtypes.common import is_list_like
from pandas import Series, DataFrame
from pandas.core.groupby.generic import SeriesGroupBy

@pytest.fixture
def dtype() -> DecimalDtype:
    return DecimalDtype()

@pytest.fixture
def data() -> DecimalArray:
    return DecimalArray(make_data())

@pytest.fixture
def data_for_twos() -> DecimalArray:
    return DecimalArray([decimal.Decimal(2) for _ in range(100)])

@pytest.fixture
def data_missing() -> DecimalArray:
    return DecimalArray([decimal.Decimal('NaN'), decimal.Decimal(1)])

@pytest.fixture
def data_for_sorting() -> DecimalArray:
    return DecimalArray([decimal.Decimal('1'), decimal.Decimal('2'), decimal.Decimal('0')])

@pytest.fixture
def data_missing_for_sorting() -> DecimalArray:
    return DecimalArray([decimal.Decimal('1'), decimal.Decimal('NaN'), decimal.Decimal('0')])

@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]:
    return lambda x, y: x.is_nan() and y.is_nan()

@pytest.fixture
def data_for_grouping() -> DecimalArray:
    b: decimal.Decimal = decimal.Decimal('1.0')
    a: decimal.Decimal = decimal.Decimal('0.0')
    c: decimal.Decimal = decimal.Decimal('2.0')
    na: decimal.Decimal = decimal.Decimal('NaN')
    return DecimalArray([b, b, na, na, a, a, b, c])

class TestDecimalArray(base.ExtensionTests):

    def _get_expected_exception(self, op_name: str, obj: Any, other: Any) -> Optional[Type[Exception]]:
        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ['kurt', 'sem']:
            return False
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool) -> None:
        if op_name == 'count':
            return super().check_reduce(ser, op_name, skipna)
        else:
            result: Any = getattr(ser, op_name)(skipna=skipna)
            expected: Any = getattr(np.asarray(ser), op_name)()
            tm.assert_almost_equal(result, expected)

    def test_reduce_series_numeric(self, data: DecimalArray, all_numeric_reductions: str, skipna: bool, request: pytest.FixtureRequest) -> None:
        if all_numeric_reductions in ['kurt', 'skew', 'sem', 'median']:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    def test_reduce_frame(self, data: DecimalArray, all_numeric_reductions: str, skipna: bool, request: pytest.FixtureRequest) -> None:
        op_name: str = all_numeric_reductions
        if op_name in ['skew', 'median']:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def test_compare_scalar(self, data: DecimalArray, comparison_op: Callable[[Any, Any], bool]) -> None:
        ser: pd.Series = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0.5)

    def test_compare_array(self, data: DecimalArray, comparison_op: Callable[[Any, Any], bool]) -> None:
        ser: pd.Series = pd.Series(data)
        alter: np.ndarray = np.random.default_rng(2).choice([-1, 0, 1], len(data))
        other: pd.Series = pd.Series(data) * [decimal.Decimal(pow(2.0, i)) for i in alter]
        self._compare_other(ser, data, comparison_op, other)

    def test_arith_series_with_array(self, data: DecimalArray, all_arithmetic_operators: str) -> None:
        op_name: str = all_arithmetic_operators
        ser: pd.Series = pd.Series(data)
        context: decimal.Context = decimal.getcontext()
        divbyzerotrap: int = context.traps[decimal.DivisionByZero]
        invalidoptrap: int = context.traps[decimal.InvalidOperation]
        context.traps[decimal.DivisionByZero] = 0
        context.traps[decimal.InvalidOperation] = 0
        other: pd.Series = pd.Series([int(d * 100) for d in data])
        self.check_opname(ser, op_name, other)
        if 'mod' not in op_name:
            self.check_opname(ser, op_name, ser * 2)
        self.check_opname(ser, op_name, 0)
        self.check_opname(ser, op_name, 5)
        context.traps[decimal.DivisionByZero] = divbyzerotrap
        context.traps[decimal.InvalidOperation] = invalidoptrap

    def test_fillna_frame(self, data_missing: DecimalArray) -> None:
        msg: str = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
            super().test_fillna_frame(data_missing)

    def test_fillna_series(self, data_missing: DecimalArray) -> None:
        msg: str = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
            super().test_fillna_series(data_missing)

    def test_fillna_with_none(self, data_missing: DecimalArray) -> None:
        msg: str = 'conversion from NoneType to Decimal is not supported'
        with pytest.raises(TypeError, match=msg):
            super().test_fillna_with_none(data_missing)

    def test_fillna_limit_frame(self, data_missing: DecimalArray) -> None:
        msg: str = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
            super().test_fillna_limit_frame(data_missing)

    def test_fillna_limit_series(self, data_missing: DecimalArray) -> None:
        msg: str = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
            super().test_fillna_limit_series(data_missing)

    @pytest.mark.parametrize('dropna', [True, False])
    def test_value_counts(self, all_data: DecimalArray, dropna: bool) -> None:
        all_data = all_data[:10]
        other: Union[DecimalArray, np.ndarray]
        if dropna:
            other = np.array(all_data[~all_data.isna()])
        else:
            other = all_data
        vcs: pd.Series = pd.Series(all_data).value_counts(dropna=dropna)
        vcs_ex: pd.Series = pd.Series(other).value_counts(dropna=dropna)
        with decimal.localcontext() as ctx:
            ctx.traps[decimal.InvalidOperation] = False
            result: pd.Series = vcs.sort_index()
            expected: pd.Series = vcs_ex.sort_index()
        tm.assert_series_equal(result, expected)

    def test_series_repr(self, data: DecimalArray) -> None:
        ser: pd.Series = pd.Series(data)
        assert data.dtype.name in repr(ser)
        assert 'Decimal: ' in repr(ser)

    @pytest.mark.xfail(reason='Inconsistent array-vs-scalar behavior')
    @pytest.mark.parametrize('ufunc', [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data: DecimalArray, ufunc: Callable[[Any], Any]) -> None:
        super().test_unary_ufunc_dunder_equivalence(data, ufunc)

def test_take_na_value_other_decimal() -> None:
    arr: DecimalArray = DecimalArray([decimal.Decimal('1.0'), decimal.Decimal('2.0')])
    result: DecimalArray = arr.take([0, -1], allow_fill=True, fill_value=decimal.Decimal('-1.0'))
    expected: DecimalArray = DecimalArray([decimal.Decimal('1.0'), decimal.Decimal('-1.0')])
    tm.assert_extension_array_equal(result, expected)

def test_series_constructor_coerce_data_to_extension_dtype() -> None:
    dtype: DecimalDtype = DecimalDtype()
    ser: pd.Series = pd.Series([0, 1, 2], dtype=dtype)
    arr: DecimalArray = DecimalArray([decimal.Decimal(0), decimal.Decimal(1), decimal.Decimal(2)], dtype=dtype)
    exp: pd.Series = pd.Series(arr)
    tm.assert_series_equal(ser, exp)

def test_series_constructor_with_dtype() -> None:
    arr: DecimalArray = DecimalArray([decimal.Decimal('10.0')])
    result: pd.Series = pd.Series(arr, dtype=DecimalDtype())
    expected: pd.Series = pd.Series(arr)
    tm.assert_series_equal(result, expected)
    result = pd.Series(arr, dtype='int64')
    expected = pd.Series([10])
    tm.assert_series_equal(result, expected)

def test_dataframe_constructor_with_dtype() -> None:
    arr: DecimalArray = DecimalArray([decimal.Decimal('10.0')])
    result: pd.DataFrame = pd.DataFrame({'A': arr}, dtype=DecimalDtype())
    expected: pd.DataFrame = pd.DataFrame({'A': arr})
    tm.assert_frame_equal(result, expected)
    arr = DecimalArray([decimal.Decimal('10.0')])
    result = pd.DataFrame({'A': arr}, dtype='int64')
    expected = pd.DataFrame({'A': [10]})
    tm.assert_frame_equal(result, expected)

def test_astype_dispatches(frame_or_series: Callable) -> None:
    data: pd.Series = pd.Series(DecimalArray([decimal.Decimal(2)]), name='a')
    ctx: decimal.Context = decimal.Context()
    ctx.prec = 5
    data_converted: Union[pd.Series, pd.DataFrame] = frame_or_series(data)
    result: Union[pd.Series, pd.DataFrame] = data_converted.astype(DecimalDtype(ctx))
    if frame_or_series is pd.DataFrame:
        result = cast(pd.DataFrame, result)['a']
    assert result.dtype.context.prec == ctx.prec

class DecimalArrayWithoutFromSequence(DecimalArray):
    """Helper class for testing error handling in _from_sequence."""

    @classmethod
    def _from_sequence(cls, scalars: Any, *, dtype: Optional[DecimalDtype] = None, copy: bool = False) -> DecimalArrayWithoutFromSequence:
        raise KeyError('For the test')

class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):

    @classmethod
    def _create_arithmetic_method(cls, op: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
        return cls._create_method(op, coerce_to_dtype=False)

DecimalArrayWithoutCoercion._add_arithmetic_ops()

def test_combine_from_sequence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cls: Type[DecimalArrayWithoutFromSequence] = DecimalArrayWithoutFromSequence

    @classmethod
    def construct_array_type(cls) -> Type[DecimalArrayWithoutFromSequence]:
        return DecimalArrayWithoutFromSequence
    monkeypatch.setattr(DecimalDtype, 'construct_array_type', construct_array_type)
    arr: DecimalArrayWithoutFromSequence = cls([decimal.Decimal('1.0'), decimal.Decimal('2.0')])
    ser: pd.Series = pd.Series(arr)
    result: pd.Series = ser.combine(ser, operator.add)
    expected: pd.Series = pd.Series([decimal.Decimal('2.0'), decimal.Decimal('4.0')], dtype='object')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('class_', [DecimalArrayWithoutFromSequence, DecimalArrayWithoutCoercion])
def test_scalar_ops_from_sequence_raises(class_: Type[DecimalArray]) -> None:
    arr: DecimalArray = class_([decimal.Decimal('1.0'), decimal.Decimal('2.0')])
    result: np.ndarray = arr + arr
    expected: np.ndarray = np.array([decimal.Decimal('2.0'), decimal.Decimal('4.0')], dtype='object')
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize('reverse, expected_div, expected_mod', [(False, [0, 1, 1, 2], [1, 0, 1, 0]), (True, [2, 1, 0, 0], [0, 0, 2, 2])])
def test_divmod_array(reverse: bool, expected_div: List[int], expected_mod: List[int]) -> None:
    arr: DecimalArray = to_decimal([1, 2, 3, 4])
    div: DecimalArray
    mod: DecimalArray
    if reverse:
        (div, mod) = divmod(2, arr)
    else:
        (div, mod) = divmod(arr, 2)
    expected_div_converted: DecimalArray = to_decimal(expected_div)
    expected_mod_converted: DecimalArray = to_decimal(expected_mod)
    tm.assert_extension_array_equal(div, expected_div_converted)
    tm.assert_extension_array_equal(mod, expected_mod_converted)

def test_ufunc_fallback(data: DecimalArray) -> None:
    a: DecimalArray = data[:5]
    s: pd.Series = pd.Series(a, index=range(3, 8))
    result: pd.Series = np.abs(s)
    expected: pd.Series = pd.Series(np.abs(a), index=range(3, 8))
    tm.assert_series_equal(result, expected)

def test_array_ufunc() -> None:
    a: DecimalArray = to_decimal([1, 2, 3])
    result: DecimalArray = np.exp(a)
    expected: DecimalArray = to_decimal(np.exp(a._data))
    tm.assert_extension_array_equal(result, expected)

def test_array_ufunc_series() -> None:
    a: DecimalArray = to_decimal([1, 2, 3])
    s: pd.Series = pd.Series(a)
    result: pd.Series = np.exp(s)
    expected: pd.Series = pd.Series(to_decimal(np.exp(a._data)))
    tm.assert_series_equal(result, expected)

def test_array_ufunc_series_scalar_other() -> None:
    a: DecimalArray = to_decimal([1, 2, 3])
    s: pd.Series = pd.Series(a)
    result: pd.Series = np.add(s, decimal.Decimal(1))
    expected: pd.Series = pd.Series(np.add(a, decimal.Decimal(1)))
    tm.assert_series_equal(result, expected)

def test_array_ufunc_series_defer() -> None:
    a: DecimalArray = to_decimal([1, 2, 3])
    s: pd.Series = pd.Series(a)
    expected: pd.Series = pd.Series(to_decimal([2, 4, 6]))
    r1: pd.Series = np.add(s, a)
    r2: pd.Series = np.add(a, s)
    tm.assert_series_equal(r1, expected)
    tm.assert_series_equal(r2, expected)

def test_groupby_agg() -> None:
    data: List[decimal.Decimal] = make_data()[:5]
    df: pd.DataFrame = pd.DataFrame({'id1': [0, 0, 0, 1, 1], 'id2': [0, 1, 0, 1, 1], 'decimals': DecimalArray(data)})
    expected: pd.Series = pd.Series(to_decimal([data[0], data[3]]))
    result: pd.Series = df.groupby('id1')['decimals'].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df['decimals'].groupby(df['id1']).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    expected = pd.Series(to_decimal([data[0], data[1], data[3]]), index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 1)]))
    result = df.groupby(['id1', 'id2'])['decimals'].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df['decimals'].groupby([df['id1'], df['id2']]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    expected = pd.DataFrame({'id2': [0, 1], 'decimals': to_decimal([data[0], data[3]])})
    result = df.groupby('id1').agg(lambda x: x.iloc[0])
    tm.assert_frame_equal(result, expected, check