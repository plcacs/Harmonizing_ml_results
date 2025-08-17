from __future__ import annotations

import decimal
import operator
from typing import Any, Callable, Sequence, TypeVar, Union

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.common import Scalar
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
    make_data,
    to_decimal,
)


T = TypeVar("T")


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
    return DecimalArray([decimal.Decimal("NaN"), decimal.Decimal(1)])


@pytest.fixture
def data_for_sorting() -> DecimalArray:
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("2"), decimal.Decimal("0")]
    )


@pytest.fixture
def data_missing_for_sorting() -> DecimalArray:
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("NaN"), decimal.Decimal("0")]
    )


@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]:
    return lambda x, y: x.is_nan() and y.is_nan()


@pytest.fixture
def data_for_grouping() -> DecimalArray:
    b = decimal.Decimal("1.0")
    a = decimal.Decimal("0.0")
    c = decimal.Decimal("2.0")
    na = decimal.Decimal("NaN")
    return DecimalArray([b, b, na, na, a, a, b, c])


class TestDecimalArray(base.ExtensionTests):
    def _get_expected_exception(
        self, op_name: str, obj: Any, other: Any
    ) -> type[Exception] | tuple[type[Exception], ...] | None:
        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ["kurt", "sem"]:
            return False
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool) -> None:
        if op_name == "count":
            return super().check_reduce(ser, op_name, skipna)
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(np.asarray(ser), op_name)()
            tm.assert_almost_equal(result, expected)

    def test_reduce_series_numeric(
        self, data: DecimalArray, all_numeric_reductions: str, skipna: bool, request: pytest.FixtureRequest
    ) -> None:
        if all_numeric_reductions in ["kurt", "skew", "sem", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    def test_reduce_frame(
        self, data: DecimalArray, all_numeric_reductions: str, skipna: bool, request: pytest.FixtureRequest
    ) -> None:
        op_name = all_numeric_reductions
        if op_name in ["skew", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)

        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def test_compare_scalar(self, data: DecimalArray, comparison_op: Callable[[Any, Any], bool]) -> None:
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0.5)

    def test_compare_array(self, data: DecimalArray, comparison_op: Callable[[Any, Any], bool]) -> None:
        ser = pd.Series(data)

        alter = np.random.default_rng(2).choice([-1, 0, 1], len(data))
        # Randomly double, halve or keep same value
        other = pd.Series(data) * [decimal.Decimal(pow(2.0, i)) for i in alter]
        self._compare_other(ser, data, comparison_op, other)

    def test_arith_series_with_array(self, data: DecimalArray, all_arithmetic_operators: str) -> None:
        op_name = all_arithmetic_operators
        ser = pd.Series(data)

        context = decimal.getcontext()
        divbyzerotrap = context.traps[decimal.DivisionByZero]
        invalidoptrap = context.traps[decimal.InvalidOperation]
        context.traps[decimal.DivisionByZero] = 0
        context.traps[decimal.InvalidOperation] = 0

        # Decimal supports ops with int, but not float
        other = pd.Series([int(d * 100) for d in data])
        self.check_opname(ser, op_name, other)

        if "mod" not in op_name:
            self.check_opname(ser, op_name, ser * 2)

        self.check_opname(ser, op_name, 0)
        self.check_opname(ser, op_name, 5)
        context.traps[decimal.DivisionByZero] = divbyzerotrap
        context.traps[decimal.InvalidOperation] = invalidoptrap

    def test_fillna_frame(self, data_missing: DecimalArray) -> None:
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_frame(data_missing)

    def test_fillna_series(self, data_missing: DecimalArray) -> None:
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_series(data_missing)

    def test_fillna_with_none(self, data_missing: DecimalArray) -> None:
        # GH#57723
        # EAs that don't have special logic for None will raise, unlike pandas'
        # which interpret None as the NA value for the dtype.
        msg = "conversion from NoneType to Decimal is not supported"
        with pytest.raises(TypeError, match=msg):
            super().test_fillna_with_none(data_missing)

    def test_fillna_limit_frame(self, data_missing: DecimalArray) -> None:
        # GH#58001
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_limit_frame(data_missing)

    def test_fillna_limit_series(self, data_missing: DecimalArray) -> None:
        # GH#58001
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_limit_series(data_missing)

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data: DecimalArray, dropna: bool) -> None:
        all_data = all_data[:10]
        if dropna:
            other = np.array(all_data[~all_data.isna()])
        else:
            other = all_data

        vcs = pd.Series(all_data).value_counts(dropna=dropna)
        vcs_ex = pd.Series(other).value_counts(dropna=dropna)

        with decimal.localcontext() as ctx:
            # avoid raising when comparing Decimal("NAN") < Decimal(2)
            ctx.traps[decimal.InvalidOperation] = False

            result = vcs.sort_index()
            expected = vcs_ex.sort_index()

        tm.assert_series_equal(result, expected)

    def test_series_repr(self, data: DecimalArray) -> None:
        # Overriding this base test to explicitly test that
        # the custom _formatter is used
        ser = pd.Series(data)
        assert data.dtype.name in repr(ser)
        assert "Decimal: " in repr(ser)

    @pytest.mark.xfail(reason="Inconsistent array-vs-scalar behavior")
    @pytest.mark.parametrize("ufunc", [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data: DecimalArray, ufunc: Callable[[Any], Any]) -> None:
        super().test_unary_ufunc_dunder_equivalence(data, ufunc)


def test_take_na_value_other_decimal() -> None:
    arr = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result = arr.take([0, -1], allow_fill=True, fill_value=decimal.Decimal("-1.0"))
    expected = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("-1.0")])
    tm.assert_extension_array_equal(result, expected)


def test_series_constructor_coerce_data_to_extension_dtype() -> None:
    dtype = DecimalDtype()
    ser = pd.Series([0, 1, 2], dtype=dtype)

    arr = DecimalArray(
        [decimal.Decimal(0), decimal.Decimal(1), decimal.Decimal(2)],
        dtype=dtype,
    )
    exp = pd.Series(arr)
    tm.assert_series_equal(ser, exp)


def test_series_constructor_with_dtype() -> None:
    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.Series(arr, dtype=DecimalDtype())
    expected = pd.Series(arr)
    tm.assert_series_equal(result, expected)

    result = pd.Series(arr, dtype="int64")
    expected = pd.Series([10])
    tm.assert_series_equal(result, expected)


def test_dataframe_constructor_with_dtype() -> None:
    arr = DecimalArray([decimal.Decimal("10.0")])

    result = pd.DataFrame({"A": arr}, dtype=DecimalDtype())
    expected = pd.DataFrame({"A": arr})
    tm.assert_frame_equal(result, expected)

    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.DataFrame({"A": arr}, dtype="int64")
    expected = pd.DataFrame({"A": [10]})
    tm.assert_frame_equal(result, expected)


def test_astype_dispatches(frame_or_series: type[Union[pd.Series, pd.DataFrame]]) -> None:
    # This is a dtype-specific test that ensures Series[decimal].astype
    # gets all the way through to ExtensionArray.astype
    # Designing a reliable smoke test that works for arbitrary data types
    # is difficult.
    data = pd.Series(DecimalArray([decimal.Decimal(2)]), name="a")
    ctx = decimal.Context()
    ctx.prec = 5

    data = frame_or_series(data)

    result = data.astype(DecimalDtype(ctx))

    if frame_or_series is pd.DataFrame:
        result = result["a"]

    assert result.dtype.context.prec == ctx.prec


class DecimalArrayWithoutFromSequence(DecimalArray):
    """Helper class for testing error handling in _from_sequence."""

    @classmethod
    def _from_sequence(
        cls, scalars: Sequence[Scalar], *, dtype: DecimalDtype | None = None, copy: bool = False
    ) -> DecimalArrayWithoutFromSequence:
        raise KeyError("For the test")


class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):
    @classmethod
    def _create_arithmetic_method(cls, op: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
        return cls._create_method(op, coerce_to_dtype=False)


DecimalArrayWithoutCoercion._add_arithmetic_ops()


def test_combine_from_sequence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # https://github.com/pandas-dev/pandas/issues/22850
    cls = DecimalArrayWithoutFromSequence

    @classmethod
    def construct_array_type(cls) -> type[DecimalArrayWithoutFromSequence]:
        return DecimalArrayWithoutFromSequence

    monkeypatch.setattr(DecimalDtype, "construct_array_type", construct_array_type)

    arr = cls([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    ser = pd.Series(arr)
    result = ser.combine(ser, operator.add)

    # note: object dtype
    expected = pd.Series(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "class_", [DecimalArrayWithoutFromSequence, DecimalArrayWithoutCoercion]
)
def test_scalar_ops_from_sequence_raises(class_: type[DecimalArray]) -> None:
    # op(EA, EA) should return an EA, or an ndarray if it's not possible
    # to return an EA with the return values.
    arr = class_([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result = arr + arr
    expected = np.array(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "reverse, expected_div, expected_mod",
    [(False, [0, 1, 1, 2], [1, 0, 1, 0]), (True, [2, 1, 0, 0], [0, 0, 2, 2])],
)
def test_divmod_array(reverse: bool, expected_div: list[int], expected_mod: list[int]) -> None:
    # https://github.com/pandas-dev/pandas/issues/22930
    arr = to_decimal([1, 2, 3, 4])
    if reverse:
        div, mod = divmod(2, arr)
    else:
        div, mod = divmod(arr, 2)
    expected_div = to_decimal(expected_div)
    expected_mod = to_decimal(expected_mod)

    tm.assert_extension_array_equal(div, expected_div)
    tm.assert_extension_array_equal(mod, expected_mod)


def test_ufunc_fallback(data: DecimalArray) -> None:
    a = data[:5]
    s = pd.Series(a, index=range(3, 8))
    result = np.abs(s)
    expected = pd.Series(np.abs(a), index=range(3, 8))
    tm.assert_series_equal(result, expected)


def test_array_ufunc() -> None:
    a = to_decimal([1, 2, 3])
    result = np.exp(a)
    expected = to_decimal(np.exp(a._data))
    tm.assert_extension_array_equal(result, expected)


def test_array_ufunc_series() -> None:
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.exp(s)
    expected = pd.Series(to_decimal(np.exp(a._data)))
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_scalar_other() -> None:
    # check _HANDLED_TYPES
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.add(s, decimal.Decimal(1))
    expected = pd.Series(np.add(a, decimal.Decimal(1)))
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_defer() -> None:
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)

    expected = pd.Series(to_decimal([2, 4, 6]))
    r1 = np.add(s, a)
    r2 = np.add(a, s)

    tm.assert_series_equal(r1, expected)
    tm.assert_series_equal(r2, expected)


def test_groupby_agg() -> None:
    # Ensure that the result of agg is inferred to be decimal dtype
    # https://github.com/pandas-dev/pandas/issues/29141

    data = make_data()[:5]
    df = pd.DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data)}
    )

    # single key, selected column
    expected = pd.Series(to_decimal([data[0], data[3]]))
    result = df.groupby("id1")["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby(df["id1"]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)

    # multiple keys, selected column
    expected = pd.Series(
        to_decimal([data[0], data[1], data[3]]),
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 1)]),
    )
    result = df.groupby(["id1", "id2"])["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby([df["id1"], df["id2"]]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)

    # multiple columns
    expected =