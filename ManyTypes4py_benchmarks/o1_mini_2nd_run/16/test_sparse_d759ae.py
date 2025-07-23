"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.
"""
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base
from typing import Any, Callable, Generator, Iterable, Optional, Union

def make_data(fill_value: Union[int, float]) -> np.ndarray:
    rng = np.random.default_rng(2)
    if np.isnan(fill_value):
        data = rng.uniform(size=100)
    else:
        data = rng.integers(1, 100, size=100, dtype=int)
        if data[0] == data[1]:
            data[0] += 1
    data[2::3] = fill_value
    return data

@pytest.fixture
def dtype() -> SparseDtype:
    return SparseDtype()

@pytest.fixture(params=[0, np.nan])
def data(request: Any) -> SparseArray:
    """Length-100 PeriodArray for semantics test."""
    res = SparseArray(make_data(request.param), fill_value=request.param)
    return res

@pytest.fixture
def data_for_twos() -> SparseArray:
    return SparseArray(np.ones(100) * 2)

@pytest.fixture(params=[0, np.nan])
def data_missing(request: Any) -> SparseArray:
    """Length 2 array with [NA, Valid]"""
    return SparseArray([np.nan, 1], fill_value=request.param)

@pytest.fixture(params=[0, np.nan])
def data_repeated(request: Any) -> Callable[[int], Generator[SparseArray, None, None]]:
    """Return different versions of data for count times"""

    def gen(count: int) -> Generator[SparseArray, None, None]:
        for _ in range(count):
            yield SparseArray(make_data(request.param), fill_value=request.param)
    return gen

@pytest.fixture(params=[0, np.nan])
def data_for_sorting(request: Any) -> SparseArray:
    return SparseArray([2, 3, 1], fill_value=request.param)

@pytest.fixture(params=[0, np.nan])
def data_missing_for_sorting(request: Any) -> SparseArray:
    return SparseArray([2, np.nan, 1], fill_value=request.param)

@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]:
    return lambda left, right: pd.isna(left) and pd.isna(right)

@pytest.fixture(params=[0, np.nan])
def data_for_grouping(request: Any) -> SparseArray:
    return SparseArray([1, 1, np.nan, np.nan, 2, 2, 1, 3], fill_value=request.param)

@pytest.fixture(params=[0, np.nan])
def data_for_compare(request: Any) -> SparseArray:
    return SparseArray([0, 0, np.nan, -2, -1, 4, 2, 3, 0, 0], fill_value=request.param)

class TestSparseArray(base.ExtensionTests):

    def _supports_reduction(self, obj: Any, op_name: str) -> bool:
        return True

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_series_numeric(
        self,
        data: SparseArray,
        all_numeric_reductions: str,
        skipna: bool,
        request: Any
    ) -> None:
        if all_numeric_reductions in ['prod', 'median', 'var', 'std', 'sem', 'skew', 'kurt']:
            mark = pytest.mark.xfail(reason='This should be viable but is not implemented')
            request.node.add_marker(mark)
        elif (
            all_numeric_reductions in ['sum', 'max', 'min', 'mean']
            and data.dtype.kind == 'f'
            and (not skipna)
        ):
            mark = pytest.mark.xfail(reason='getting a non-nan float')
            request.node.add_marker(mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_frame(
        self,
        data: SparseArray,
        all_numeric_reductions: str,
        skipna: bool,
        request: Any
    ) -> None:
        if all_numeric_reductions in ['prod', 'median', 'var', 'std', 'sem', 'skew', 'kurt']:
            mark = pytest.mark.xfail(reason='This should be viable but is not implemented')
            request.node.add_marker(mark)
        elif (
            all_numeric_reductions in ['sum', 'max', 'min', 'mean']
            and data.dtype.kind == 'f'
            and (not skipna)
        ):
            mark = pytest.mark.xfail(reason='ExtensionArray NA mask are different')
            request.node.add_marker(mark)
        super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def _check_unsupported(self, data: SparseArray) -> None:
        if data.dtype == SparseDtype(int, 0):
            pytest.skip("Can't store nan in int array.")

    def test_concat_mixed_dtypes(self, data: SparseArray) -> None:
        df1 = pd.DataFrame({'A': data[:3]})
        df2 = pd.DataFrame({'A': [1, 2, 3]})
        df3 = pd.DataFrame({'A': ['a', 'b', 'c']}).astype('category')
        dfs = [df1, df2, df3]
        result = pd.concat(dfs)
        expected = pd.concat([x.apply(lambda s: np.asarray(s).astype(object)) for x in dfs])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'columns',
        [
            ['A', 'B'],
            pd.MultiIndex.from_tuples([('A', 'a'), ('A', 'b')], names=['outer', 'inner'])
        ]
    )
    @pytest.mark.parametrize('future_stack', [True, False])
    def test_stack(
        self,
        data: SparseArray,
        columns: Union[list, pd.MultiIndex],
        future_stack: bool
    ) -> None:
        super().test_stack(data, columns, future_stack)

    def test_concat_columns(self, data: SparseArray, na_value: Union[int, float]) -> None:
        self._check_unsupported(data)
        super().test_concat_columns(data, na_value)

    def test_concat_extension_arrays_copy_false(
        self,
        data: SparseArray,
        na_value: Union[int, float]
    ) -> None:
        self._check_unsupported(data)
        super().test_concat_extension_arrays_copy_false(data, na_value)

    def test_align(self(self, data: SparseArray, na_value: Union[int, float]) -> None:
        self._check_unsupported(data)
        super().test_align(data, na_value)

    def test_align_frame(self(self, data: SparseArray, na_value: Union[int, float]) -> None:
        self._check_unsupported(data)
        super().test_align_frame(data, na_value)

    def test_align_series_frame(
        self,
        data: SparseArray,
        na_value: Union[int, float]
    ) -> None:
        self._check_unsupported(data)
        super().test_align_series_frame(data, na_value)

    def test_merge(self, data: SparseArray, na_value: Union[int, float]) -> None:
        self._check_unsupported(data)
        super().test_merge(data, na_value)

    def test_get(self, data: SparseArray) -> None:
        ser = pd.Series(data, index=[2 * i for i in range(len(data))])
        if np.isnan(ser.values.fill_value):
            assert np.isnan(ser.get(4)) and np.isnan(ser.iloc[2])
        else:
            assert ser.get(4) == ser.iloc[2]
        assert ser.get(2) == ser.iloc[1]

    def test_reindex(self, data: SparseArray, na_value: Union[int, float]) -> None:
        self._check_unsupported(data)
        super().test_reindex(data, na_value)

    def test_isna(self, data_missing: SparseArray) -> None:
        sarr = SparseArray(data_missing)
        expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.fill_value))
        expected = SparseArray([True, False], dtype=expected_dtype)
        result = sarr.isna()
        tm.assert_sp_array_equal(result, expected)
        sarr = sarr.fillna(0)
        expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.fill_value))
        expected = SparseArray([False, False], fill_value=False, dtype=expected_dtype)
        tm.assert_equal(sarr.isna(), expected)

    def test_fillna_no_op_returns_copy(self, data: SparseArray, request: Any) -> None:
        super().test_fillna_no_op_returns_copy(data)

    @pytest.mark.xfail(reason='Unsupported')
    def test_fillna_series(self, data_missing: SparseArray) -> None:
        super().test_fillna_series()

    def test_fillna_frame(self, data_missing: SparseArray) -> None:
        fill_value = data_missing[1]
        result = pd.DataFrame({'A': data_missing, 'B': [1, 2]}).fillna(fill_value)
        if pd.isna(data_missing.fill_value):
            dtype = SparseDtype(data_missing.dtype, fill_value)
        else:
            dtype = data_missing.dtype
        expected = pd.DataFrame({
            'A': data_missing._from_sequence([fill_value, fill_value], dtype=dtype),
            'B': [1, 2]
        })
        tm.assert_frame_equal(result, expected)

    def test_fillna_limit_frame(self, data_missing: SparseArray) -> None) -> None:
        with pytest.raises(ValueError, match='limit must be None'):
            super().test_fillna_limit_frame(data_missing)

    def test_fillna_limit_series(self, data_missing: SparseArray) -> None:
        with pytest.raises(ValueError, match='limit must be None'):
            super().test_fillna_limit_frame(data_missing)
    _combine_le_expected_dtype: str = 'Sparse[bool]'

    def test_fillna_copy_frame(self, data_missing: SparseArray) -> None:
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({'A': arr}, copy=False)
        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)
        if hasattr(df._mgr, 'blocks'):
            assert df.values.base is result.values.base
        assert df.A._values.to_dense() is arr.to_dense()

    def test_fillna_copy_series(self, data_missing: SparseArray) -> None:
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr, copy=False)
        filled_val = ser[0]
        result = ser.fillna(filled_val)
        assert ser._values is result._values
        assert ser._values.to_dense() is arr.to_dense()

    @pytest.mark.xfail(reason='Not Applicable')
    def test_fillna_length_mismatch(self, data_missing: SparseArray) -> None:
        super().test_fillna_length_mismatch(data_missing)

    def test_where_series(self, data: SparseArray, na_value: Union[int, float]) -> None:
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]
        ser = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
        cond = np.array([True, True, False, False])
        result = ser.where(cond)
        new_dtype = SparseDtype('float', 0.0)
        expected = pd.Series(cls._from_sequence([a, a, na_value, na_value], dtype=new_dtype))
        tm.assert_series_equal(result, expected)
        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        cond = np.array([True, False, True, True])
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        tm.assert_series_equal(result, expected)

    def test_searchsorted(
        self,
        performance_warning: bool,
        data_for_sorting: SparseArray,
        as_series: bool
    ) -> None:
        with tm.assert_produces_warning(performance_warning, check_stacklevel=False):
            super().test_searchsorted(data_for_sorting, as_series)

    def test_shift_0_periods(self, data: SparseArray) -> None:
        result = data.shift(0)
        data._sparse_values[0] = data._sparse_values[1]
        assert result._sparse_values[0] != result._sparse_values[1]

    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def test_argmin_argmax_all_na(
        self,
        method: str,
        data: SparseArray,
        na_value: Union[int, float]
    ) -> None:
        self._check_unsupported(data)
        super().test_argmin_argmax_all_na(method, data, na_value)

    @pytest.mark.fails_arm_wheels
    @pytest.mark.parametrize('box', [pd.array, pd.Series, pd.DataFrame])
    def test_equals(
        self,
        data: SparseArray,
        na_value: Union[int, float],
        as_series: bool,
        box: Callable[..., Any]
    ) -> None:
        self._check_unsupported(data)
        super().test_equals(data, na_value, as_series, box)

    @pytest.mark.fails_arm_wheels
    def test_equals_same_data_different_object(self, data: SparseArray) -> None:
        super().test_equals_same_data_different_object(data)

    @pytest.mark.parametrize(
        'func, na_action, expected',
        [
            (
                lambda x: x,
                None,
                SparseArray([1.0, np.nan])
            ),
            (
                lambda x: x,
                'ignore',
                SparseArray([1.0, np.nan])
            ),
            (
                str,
                None,
                SparseArray(['1.0', 'nan'], fill_value='nan')
            ),
            (
                str,
                'ignore',
                SparseArray(['1.0', np.nan])
            )
        ]
    )
    def test_map(
        self,
        func: Callable[[Any], Any],
        na_action: Optional[str],
        expected: SparseArray
    ) -> None:
        data = SparseArray([1, np.nan])
        result = data.map(func, na_action=na_action)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def test_map_raises(
        self,
        data: SparseArray,
        na_action: Optional[str]
    ) -> None:
        msg = 'fill value in the sparse values not supported'
        with pytest.raises(ValueError, match=msg):
            data.map(lambda x: np.nan, na_action=na_action)

    @pytest.mark.xfail(raises=TypeError, reason='no sparse StringDtype')
    def test_astype_string(self, data: SparseArray, nullable_string_dtype: Any) -> None:
        super().test_astype_string(data)
    series_scalar_exc: Optional[Any] = None
    frame_scalar_exc: Optional[Any] = None
    divmod_exc: Optional[Any] = None
    series_array_exc: Optional[Any] = None

    def _skip_if_different_combine(self, data: SparseArray) -> None:
        if data.fill_value == 0:
            pytest.skip('Incorrected expected from Series.combine and tested elsewhere')

    def test_arith_series_with_scalar(
        self,
        data: SparseArray,
        all_arithmetic_operators: str
    ) -> None:
        self._skip_if_different_combine(data)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(
        self,
        data: SparseArray,
        all_arithmetic_operators: str
    ) -> None:
        self._skip_if_different_combine(data)
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(
        self,
        data: SparseArray,
        all_arithmetic_operators: str,
        request: Any
    ) -> None:
        if data.dtype.fill_value != 0:
            pass
        elif all_arithmetic_operators.strip('_') not in [
            'mul', 'rmul', 'floordiv', 'rfloordiv',
            'truediv', 'rtruediv', 'pow', 'mod', 'rmod'
        ]:
            mark = pytest.mark.xfail(reason='result dtype.fill_value mismatch')
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def _compare_other(
        self,
        ser: pd.Series,
        data_for_compare: SparseArray,
        comparison_op: Callable[[Any, Any], Any],
        other: Union[pd.Series, np.ndarray, SparseArray, int, float]
    ) -> None:
        op = comparison_op
        result = op(data_for_compare, other)
        if isinstance(other, pd.Series):
            assert isinstance(result, pd.Series)
            assert isinstance(result.dtype, SparseDtype)
        else:
            assert isinstance(result, SparseArray)
        assert result.dtype.subtype == np.bool_
        if isinstance(other, pd.Series):
            fill_value = op(data_for_compare.fill_value, other._values.fill_value)
            expected = SparseArray(
                op(data_for_compare.to_dense(), np.asarray(other)),
                fill_value=fill_value,
                dtype=np.bool_
            )
        else:
            fill_value = np.all(op(np.asarray(data_for_compare.fill_value), np.asarray(other)))
            expected = SparseArray(
                op(data_for_compare.to_dense(), np.asarray(other)),
                fill_value=fill_value,
                dtype=np.bool_
            )
        if isinstance(other, pd.Series):
            expected = pd.Series(expected)
        tm.assert_equal(result, expected)

    def test_scalar(
        self,
        data_for_compare: SparseArray,
        comparison_op: Callable[[Any, Any], Any]
    ) -> None:
        ser = pd.Series(data_for_compare)
        self._compare_other(ser, data_for_compare, comparison_op, 0)
        self._compare_other(ser, data_for_compare, comparison_op, 1)
        self._compare_other(ser, data_for_compare, comparison_op, -1)
        self._compare_other(ser, data_for_compare, comparison_op, np.nan)

    def test_array(
        self,
        data_for_compare: SparseArray,
        comparison_op: Callable[[Any, Any], Any],
        request: Any
    ) -> None:
        if data_for_compare.dtype.fill_value == 0 and comparison_op.__name__ in ['eq', 'ge', 'le']:
            mark = pytest.mark.xfail(reason='Wrong fill_value')
            request.applymarker(mark)
        arr = np.linspace(-4, 5, 10)
        ser = pd.Series(data_for_compare)
        self._compare_other(ser, data_for_compare, comparison_op, arr)

    def test_sparse_array(
        self,
        data_for_compare: SparseArray,
        comparison_op: Callable[[Any, Any], Any],
        request: Any
    ) -> None:
        if data_for_compare.dtype.fill_value == 0 and comparison_op.__name__ != 'gt':
            mark = pytest.mark.xfail(reason='Wrong fill_value')
            request.applymarker(mark)
        ser = pd.Series(data_for_compare)
        arr = data_for_compare + 1
        self._compare_other(ser, data_for_compare, comparison_op, arr)
        arr = data_for_compare * 2
        self._compare_other(ser, data_for_compare, comparison_op, arr)

    @pytest.mark.xfail(reason='Different repr')
    def test_array_repr(self, data: SparseArray, size: int) -> None:
        super().test_array_repr(data, size)

    @pytest.mark.xfail(reason='result does not match expected')
    @pytest.mark.parametrize('as_index', [True, False])
    def test_groupby_extension_agg(
        self,
        as_index: bool,
        data_for_grouping: SparseArray
    ) -> None:
        super().test_groupby_extension_agg(as_index, data_for_grouping)

def test_array_type_with_arg(dtype: SparseDtype) -> None:
    assert dtype.construct_array_type() is SparseArray
