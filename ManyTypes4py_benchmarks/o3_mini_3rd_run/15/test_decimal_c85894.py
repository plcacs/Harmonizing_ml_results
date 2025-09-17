from __future__ import annotations
import decimal
import operator
from typing import Any, Callable, List, Type, Union

import numpy as np
import pandas as pd
import pandas._testing as tm  # type: ignore
import pytest
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import DecimalArray, DecimalDtype, make_data, to_decimal


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
    return DecimalArray([decimal.Decimal("1"), decimal.Decimal("2"), decimal.Decimal("0")])


@pytest.fixture
def data_missing_for_sorting() -> DecimalArray:
    return DecimalArray([decimal.Decimal("1"), decimal.Decimal("NaN"), decimal.Decimal("0")])


@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]:
    return lambda x, y: x.is_nan() and y.is_nan()


class TestDecimalArray(base.ExtensionTests):
    def _get_expected_exception(self, op_name: str, obj: Any, other: Any) -> None:
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
        op_name: str = all_numeric_reductions
        if op_name in ["skew", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def test_compare_scalar(self, data: DecimalArray, comparison_op: Callable[[Any, Any], Any]) -> None:
        ser: pd.Series = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0.5)

    def test_compare_array(self, data: DecimalArray, comparison_op: Callable[[Any, Any], Any]) -> None:
        ser: pd.Series = pd.Series(data)
        alter: np.ndarray = np.random.default_rng(2).choice([-1, 0, 1], len(data))
        other: pd.Series = pd.Series(data) * [decimal.Decimal(pow(2.0, i)) for i in alter]
        self._compare_other(ser, data, comparison_op, other)

    def test_arith_series_with_array(self, data: DecimalArray, all_arithmetic_operators: str) -> None:
        op_name: str = all_arithmetic_operators
        ser: pd.Series = pd.Series(data)
        context: decimal.Context = decimal.getcontext()
        divbyzerotrap: Any = context.traps[decimal.DivisionByZero]
        invalidoptrap: Any = context.traps[decimal.InvalidOperation]
        context.traps[decimal.DivisionByZero] = 0
        context.traps[decimal.InvalidOperation] = 0
        other: pd.Series = pd.Series([int(d * 100) for d in data])
        self.check_opname(ser, op_name, other)
        if "mod" not in op_name:
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
        msg: str = "conversion from NoneType to Decimal is not supported"
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

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data: DecimalArray, dropna: bool) -> None:
        all_data = all_data[:10]
        if dropna:
            other: np.ndarray = np.array(all_data[~all_data.isna()])
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
        assert "Decimal: " in repr(ser)

    @pytest.mark.xfail(reason="Inconsistent array-vs-scalar behavior")
    @pytest.mark.parametrize("ufunc", [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data: DecimalArray, ufunc: Callable) -> None:
        super().test_unary_ufunc_dunder_equivalence(data, ufunc)


def test_take_na_value_other_decimal() -> None:
    arr: DecimalArray = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result: DecimalArray = arr.take([0, -1], allow_fill=True, fill_value=decimal.Decimal("-1.0"))
    expected: DecimalArray = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("-1.0")])
    tm.assert_extension_array_equal(result, expected)


def test_series_constructor_coerce_data_to_extension_dtype() -> None:
    dtype_inst: DecimalDtype = DecimalDtype()
    ser: pd.Series = pd.Series([0, 1, 2], dtype=dtype_inst)
    arr: DecimalArray = DecimalArray(
        [decimal.Decimal(0), decimal.Decimal(1), decimal.Decimal(2)], dtype=dtype_inst
    )
    exp: pd.Series = pd.Series(arr)
    tm.assert_series_equal(ser, exp)


def test_series_constructor_with_dtype() -> None:
    arr: DecimalArray = DecimalArray([decimal.Decimal("10.0")])
    result: pd.Series = pd.Series(arr, dtype=DecimalDtype())
    expected: pd.Series = pd.Series(arr)
    tm.assert_series_equal(result, expected)
    result = pd.Series(arr, dtype="int64")
    expected = pd.Series([10])
    tm.assert_series_equal(result, expected)


def test_dataframe_constructor_with_dtype() -> None:
    arr: DecimalArray = DecimalArray([decimal.Decimal("10.0")])
    result: pd.DataFrame = pd.DataFrame({"A": arr}, dtype=DecimalDtype())
    expected: pd.DataFrame = pd.DataFrame({"A": arr})
    tm.assert_frame_equal(result, expected)
    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.DataFrame({"A": arr}, dtype="int64")
    expected = pd.DataFrame({"A": [10]})
    tm.assert_frame_equal(result, expected)


def test_astype_dispatches(
    frame_or_series: Callable[[pd.Series], Union[pd.Series, pd.DataFrame]]
) -> None:
    data: pd.Series = pd.Series(DecimalArray([decimal.Decimal(2)]), name="a")
    ctx: decimal.Context = decimal.Context()
    ctx.prec = 5
    data = frame_or_series(data)
    result = data.astype(DecimalDtype(ctx))
    if frame_or_series is pd.DataFrame:
        result = result["a"]
    assert result.dtype.context.prec == ctx.prec


class DecimalArrayWithoutFromSequence(DecimalArray):
    """Helper class for testing error handling in _from_sequence."""

    @classmethod
    def _from_sequence(cls, scalars: List[decimal.Decimal], *, dtype: Any = None, copy: bool = False) -> Any:
        raise KeyError("For the test")


class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):
    @classmethod
    def _create_arithmetic_method(cls, op: Any) -> Callable:
        return cls._create_method(op, coerce_to_dtype=False)


DecimalArrayWithoutCoercion._add_arithmetic_ops()


def test_combine_from_sequence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cls: Type[DecimalArrayWithoutFromSequence] = DecimalArrayWithoutFromSequence

    @classmethod
    def construct_array_type(cls) -> Type[DecimalArrayWithoutFromSequence]:
        return DecimalArrayWithoutFromSequence

    monkeypatch.setattr(DecimalDtype, "construct_array_type", construct_array_type)
    arr: DecimalArrayWithoutFromSequence = cls([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    ser: pd.Series = pd.Series(arr)
    result: pd.Series = ser.combine(ser, operator.add)
    expected: pd.Series = pd.Series([decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("class_", [DecimalArrayWithoutFromSequence, DecimalArrayWithoutCoercion])
def test_scalar_ops_from_sequence_raises(class_: Type[DecimalArray]) -> None:
    arr: DecimalArray = class_([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result = arr + arr
    expected = np.array([decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "reverse, expected_div, expected_mod",
    [(False, [0, 1, 1, 2], [1, 0, 1, 0]), (True, [2, 1, 0, 0], [0, 0, 2, 2])],
)
def test_divmod_array(
    reverse: bool, expected_div: List[int], expected_mod: List[int]
) -> None:
    arr: DecimalArray = to_decimal([1, 2, 3, 4])
    if reverse:
        div, mod = divmod(2, arr)
    else:
        div, mod = divmod(arr, 2)
    expected_div_arr: DecimalArray = to_decimal(expected_div)
    expected_mod_arr: DecimalArray = to_decimal(expected_mod)
    tm.assert_extension_array_equal(div, expected_div_arr)
    tm.assert_extension_array_equal(mod, expected_mod_arr)


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
    data_list: List[decimal.Decimal] = make_data()[:5]
    df: pd.DataFrame = pd.DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data_list)}
    )
    expected: pd.Series = pd.Series(to_decimal([data_list[0], data_list[3]]))
    result: pd.Series = df.groupby("id1")["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby(df["id1"]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    expected = pd.Series(
        to_decimal([data_list[0], data_list[1], data_list[3]]),
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 1)]),
    )
    result = df.groupby(["id1", "id2"])["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby([df["id1"], df["id2"]]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    expected_df: pd.DataFrame = pd.DataFrame(
        {"id2": [0, 1], "decimals": to_decimal([data_list[0], data_list[3]])}
    )
    result_df: pd.DataFrame = df.groupby("id1").agg(lambda x: x.iloc[0])
    tm.assert_frame_equal(result_df, expected_df, check_names=False)


def test_groupby_agg_ea_method(monkeypatch: pytest.MonkeyPatch) -> None:
    def DecimalArray__my_sum(self: DecimalArray) -> Any:
        return np.sum(np.array(self))

    monkeypatch.setattr(DecimalArray, "my_sum", DecimalArray__my_sum, raising=False)
    data_list: List[decimal.Decimal] = make_data()[:5]
    df: pd.DataFrame = pd.DataFrame({"id": [0, 0, 0, 1, 1], "decimals": DecimalArray(data_list)})
    expected: pd.Series = pd.Series(to_decimal([data_list[0] + data_list[1] + data_list[2], data_list[3] + data_list[4]]))
    result: pd.Series = df.groupby("id")["decimals"].agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)
    s: pd.Series = pd.Series(DecimalArray(data_list))
    grouper: np.ndarray = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    result = s.groupby(grouper).agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)


def test_indexing_no_materialize(monkeypatch: pytest.MonkeyPatch) -> None:
    def DecimalArray__array__(self: DecimalArray, dtype: Any = None) -> Any:
        raise Exception("tried to convert a DecimalArray to a numpy array")

    monkeypatch.setattr(DecimalArray, "__array__", DecimalArray__array__, raising=False)
    data_list: List[decimal.Decimal] = make_data()
    s: pd.Series = pd.Series(DecimalArray(data_list))
    df: pd.DataFrame = pd.DataFrame({"a": s, "b": list(range(len(s)))})
    s[s > 0.5]
    df[s > 0.5]
    s.at[0]
    df.at[0, "a"]


def test_to_numpy_keyword() -> None:
    values: List[decimal.Decimal] = [decimal.Decimal("1.1111"), decimal.Decimal("2.2222")]
    expected: np.ndarray = np.array([decimal.Decimal("1.11"), decimal.Decimal("2.22")], dtype="object")
    a: DecimalArray = pd.array(values, dtype="decimal")
    result: np.ndarray = a.to_numpy(decimals=2)  # type: ignore
    tm.assert_numpy_array_equal(result, expected)
    result = pd.Series(a).to_numpy(decimals=2)  # type: ignore
    tm.assert_numpy_array_equal(result, expected)


def test_array_copy_on_write() -> None:
    df: pd.DataFrame = pd.DataFrame({"a": [decimal.Decimal(2), decimal.Decimal(3)]}, dtype="object")
    df2: pd.DataFrame = df.astype(DecimalDtype())
    df.iloc[0, 0] = 0
    expected: pd.DataFrame = pd.DataFrame({"a": [decimal.Decimal(2), decimal.Decimal(3)]}, dtype=DecimalDtype())
    tm.assert_equal(df2.values, expected.values)