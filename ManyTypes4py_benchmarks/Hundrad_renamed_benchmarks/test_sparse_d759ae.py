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


def func_5elvluar(fill_value):
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
def func_dg8vi7xj():
    return SparseDtype()


@pytest.fixture(params=[0, np.nan])
def func_dhlp426l(request):
    """Length-100 PeriodArray for semantics test."""
    res = SparseArray(func_5elvluar(request.param), fill_value=request.param)
    return res


@pytest.fixture
def func_6v9lybiv():
    return SparseArray(np.ones(100) * 2)


@pytest.fixture(params=[0, np.nan])
def func_4a5jeaww(request):
    """Length 2 array with [NA, Valid]"""
    return SparseArray([np.nan, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def func_f2ijz7df(request):
    """Return different versions of data for count times"""

    def func_61iejm25(count):
        for _ in range(count):
            yield SparseArray(func_5elvluar(request.param), fill_value=
                request.param)
    return gen


@pytest.fixture(params=[0, np.nan])
def func_6rnj6dau(request):
    return SparseArray([2, 3, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def func_1350epya(request):
    return SparseArray([2, np.nan, 1], fill_value=request.param)


@pytest.fixture
def func_91hgzgwh():
    return lambda left, right: pd.isna(left) and pd.isna(right)


@pytest.fixture(params=[0, np.nan])
def func_er8fuict(request):
    return SparseArray([1, 1, np.nan, np.nan, 2, 2, 1, 3], fill_value=
        request.param)


@pytest.fixture(params=[0, np.nan])
def func_zj8g5hl3(request):
    return SparseArray([0, 0, np.nan, -2, -1, 4, 2, 3, 0, 0], fill_value=
        request.param)


class TestSparseArray(base.ExtensionTests):

    def func_klgncd17(self, obj, op_name):
        return True

    @pytest.mark.parametrize('skipna', [True, False])
    def func_8wl6qn4x(self, data, all_numeric_reductions, skipna, request):
        if all_numeric_reductions in ['prod', 'median', 'var', 'std', 'sem',
            'skew', 'kurt']:
            mark = pytest.mark.xfail(reason=
                'This should be viable but is not implemented')
            request.node.add_marker(mark)
        elif all_numeric_reductions in ['sum', 'max', 'min', 'mean'
            ] and data.dtype.kind == 'f' and not skipna:
            mark = pytest.mark.xfail(reason='getting a non-nan float')
            request.node.add_marker(mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna
            )

    @pytest.mark.parametrize('skipna', [True, False])
    def func_pr2ao7i9(self, data, all_numeric_reductions, skipna, request):
        if all_numeric_reductions in ['prod', 'median', 'var', 'std', 'sem',
            'skew', 'kurt']:
            mark = pytest.mark.xfail(reason=
                'This should be viable but is not implemented')
            request.node.add_marker(mark)
        elif all_numeric_reductions in ['sum', 'max', 'min', 'mean'
            ] and data.dtype.kind == 'f' and not skipna:
            mark = pytest.mark.xfail(reason=
                'ExtensionArray NA mask are different')
            request.node.add_marker(mark)
        super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def func_n6pkw7zi(self, data):
        if data.dtype == SparseDtype(int, 0):
            pytest.skip("Can't store nan in int array.")

    def func_dob5uyxp(self, data):
        df1 = pd.DataFrame({'A': data[:3]})
        df2 = pd.DataFrame({'A': [1, 2, 3]})
        df3 = pd.DataFrame({'A': ['a', 'b', 'c']}).astype('category')
        dfs = [df1, df2, df3]
        result = pd.concat(dfs)
        expected = pd.concat([x.apply(lambda s: np.asarray(s).astype(object
            )) for x in dfs])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        'ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('columns', [['A', 'B'], pd.MultiIndex.
        from_tuples([('A', 'a'), ('A', 'b')], names=['outer', 'inner'])])
    @pytest.mark.parametrize('future_stack', [True, False])
    def func_o354njif(self, data, columns, future_stack):
        super().test_stack(data, columns, future_stack)

    def func_03qgc331(self, data, na_value):
        self._check_unsupported(data)
        super().test_concat_columns(data, na_value)

    def func_gjwwqo45(self, data, na_value):
        self._check_unsupported(data)
        super().test_concat_extension_arrays_copy_false(data, na_value)

    def func_5lencx54(self, data, na_value):
        self._check_unsupported(data)
        super().test_align(data, na_value)

    def func_4gouswat(self, data, na_value):
        self._check_unsupported(data)
        super().test_align_frame(data, na_value)

    def func_uiyt2v05(self, data, na_value):
        self._check_unsupported(data)
        super().test_align_series_frame(data, na_value)

    def func_vsagexbx(self, data, na_value):
        self._check_unsupported(data)
        super().test_merge(data, na_value)

    def func_2xtdwjjr(self, data):
        ser = pd.Series(data, index=[(2 * i) for i in range(len(data))])
        if np.isnan(ser.values.fill_value):
            assert np.isnan(ser.get(4)) and np.isnan(ser.iloc[2])
        else:
            assert ser.get(4) == ser.iloc[2]
        assert ser.get(2) == ser.iloc[1]

    def func_9msssk5a(self, data, na_value):
        self._check_unsupported(data)
        super().test_reindex(data, na_value)

    def func_fa9uo33f(self, data_missing):
        sarr = SparseArray(data_missing)
        expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.
            fill_value))
        expected = SparseArray([True, False], dtype=expected_dtype)
        result = sarr.isna()
        tm.assert_sp_array_equal(result, expected)
        sarr = sarr.fillna(0)
        expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.
            fill_value))
        expected = SparseArray([False, False], fill_value=False, dtype=
            expected_dtype)
        tm.assert_equal(sarr.isna(), expected)

    def func_g1nestnh(self, data, request):
        super().test_fillna_no_op_returns_copy(data)

    @pytest.mark.xfail(reason='Unsupported')
    def func_0nd61ja8(self, data_missing):
        super().test_fillna_series()

    def func_hh8aqsja(self, data_missing):
        fill_value = data_missing[1]
        result = pd.DataFrame({'A': data_missing, 'B': [1, 2]}).fillna(
            fill_value)
        if pd.isna(data_missing.fill_value):
            dtype = SparseDtype(data_missing.dtype, fill_value)
        else:
            dtype = data_missing.dtype
        expected = pd.DataFrame({'A': func_4a5jeaww._from_sequence([
            fill_value, fill_value], dtype=dtype), 'B': [1, 2]})
        tm.assert_frame_equal(result, expected)

    def func_qbb8vuek(self, data_missing):
        with pytest.raises(ValueError, match='limit must be None'):
            super().test_fillna_limit_frame(data_missing)

    def func_q5liqeyg(self, data_missing):
        with pytest.raises(ValueError, match='limit must be None'):
            super().test_fillna_limit_frame(data_missing)
    _combine_le_expected_dtype = 'Sparse[bool]'

    def func_7yz1xxbp(self, data_missing):
        arr = func_4a5jeaww.take([1, 1])
        df = pd.DataFrame({'A': arr}, copy=False)
        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)
        if hasattr(df._mgr, 'blocks'):
            assert df.values.base is result.values.base
        assert df.A._values.to_dense() is arr.to_dense()

    def func_ej7lzpug(self, data_missing):
        arr = func_4a5jeaww.take([1, 1])
        ser = pd.Series(arr, copy=False)
        filled_val = ser[0]
        result = ser.fillna(filled_val)
        assert ser._values is result._values
        assert ser._values.to_dense() is arr.to_dense()

    @pytest.mark.xfail(reason='Not Applicable')
    def func_kc9j7bll(self, data_missing):
        super().test_fillna_length_mismatch(data_missing)

    def func_lwkq0ufw(self, data, na_value):
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]
        ser = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
        cond = np.array([True, True, False, False])
        result = ser.where(cond)
        new_dtype = SparseDtype('float', 0.0)
        expected = pd.Series(cls._from_sequence([a, a, na_value, na_value],
            dtype=new_dtype))
        tm.assert_series_equal(result, expected)
        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        cond = np.array([True, False, True, True])
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype)
            )
        tm.assert_series_equal(result, expected)

    def func_y3kq1adz(self, performance_warning, data_for_sorting, as_series):
        with tm.assert_produces_warning(performance_warning,
            check_stacklevel=False):
            super().test_searchsorted(data_for_sorting, as_series)

    def func_dsa96ya9(self, data):
        result = func_dhlp426l.shift(0)
        data._sparse_values[0] = data._sparse_values[1]
        assert result._sparse_values[0] != result._sparse_values[1]

    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def func_xdofffe2(self, method, data, na_value):
        self._check_unsupported(data)
        super().test_argmin_argmax_all_na(method, data, na_value)

    @pytest.mark.fails_arm_wheels
    @pytest.mark.parametrize('box', [pd.array, pd.Series, pd.DataFrame])
    def func_961smidb(self, data, na_value, as_series, box):
        self._check_unsupported(data)
        super().test_equals(data, na_value, as_series, box)

    @pytest.mark.fails_arm_wheels
    def func_ueqwufen(self, data):
        super().test_equals_same_data_different_object(data)

    @pytest.mark.parametrize('func, na_action, expected', [(lambda x: x,
        None, SparseArray([1.0, np.nan])), (lambda x: x, 'ignore',
        SparseArray([1.0, np.nan])), (str, None, SparseArray(['1.0', 'nan'],
        fill_value='nan')), (str, 'ignore', SparseArray(['1.0', np.nan]))])
    def func_azcpj2c1(self, func, na_action, expected):
        data = SparseArray([1, np.nan])
        result = func_dhlp426l.map(func, na_action=na_action)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def func_bzen5rpv(self, data, na_action):
        msg = 'fill value in the sparse values not supported'
        with pytest.raises(ValueError, match=msg):
            func_dhlp426l.map(lambda x: np.nan, na_action=na_action)

    @pytest.mark.xfail(raises=TypeError, reason='no sparse StringDtype')
    def func_81ecbxqm(self, data, nullable_string_dtype):
        super().test_astype_string(data)
    series_scalar_exc = None
    frame_scalar_exc = None
    divmod_exc = None
    series_array_exc = None

    def func_mucpd3ge(self, data):
        if data.fill_value == 0:
            pytest.skip(
                'Incorrected expected from Series.combine and tested elsewhere'
                )

    def func_o4dm32ze(self, data, all_arithmetic_operators):
        self._skip_if_different_combine(data)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def func_8lu4hx68(self, data, all_arithmetic_operators):
        self._skip_if_different_combine(data)
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    def func_fmjtcv29(self, data, all_arithmetic_operators, request):
        if data.dtype.fill_value != 0:
            pass
        elif all_arithmetic_operators.strip('_') not in ['mul', 'rmul',
            'floordiv', 'rfloordiv', 'truediv', 'rtruediv', 'pow', 'mod',
            'rmod']:
            mark = pytest.mark.xfail(reason='result dtype.fill_value mismatch')
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def func_ndo0ydhc(self, ser, data_for_compare, comparison_op, other):
        op = comparison_op
        result = op(data_for_compare, other)
        if isinstance(other, pd.Series):
            assert isinstance(result, pd.Series)
            assert isinstance(result.dtype, SparseDtype)
        else:
            assert isinstance(result, SparseArray)
        assert result.dtype.subtype == np.bool_
        if isinstance(other, pd.Series):
            fill_value = op(data_for_compare.fill_value, other._values.
                fill_value)
            expected = SparseArray(op(func_zj8g5hl3.to_dense(), np.asarray(
                other)), fill_value=fill_value, dtype=np.bool_)
        else:
            fill_value = np.all(op(np.asarray(data_for_compare.fill_value),
                np.asarray(other)))
            expected = SparseArray(op(func_zj8g5hl3.to_dense(), np.asarray(
                other)), fill_value=fill_value, dtype=np.bool_)
        if isinstance(other, pd.Series):
            expected = pd.Series(expected)
        tm.assert_equal(result, expected)

    def func_kmd9zrxt(self, data_for_compare, comparison_op):
        ser = pd.Series(data_for_compare)
        self._compare_other(ser, data_for_compare, comparison_op, 0)
        self._compare_other(ser, data_for_compare, comparison_op, 1)
        self._compare_other(ser, data_for_compare, comparison_op, -1)
        self._compare_other(ser, data_for_compare, comparison_op, np.nan)

    def func_hcxwcxa9(self, data_for_compare, comparison_op, request):
        if (data_for_compare.dtype.fill_value == 0 and comparison_op.
            __name__ in ['eq', 'ge', 'le']):
            mark = pytest.mark.xfail(reason='Wrong fill_value')
            request.applymarker(mark)
        arr = np.linspace(-4, 5, 10)
        ser = pd.Series(data_for_compare)
        self._compare_other(ser, data_for_compare, comparison_op, arr)

    def func_oqgdnp1u(self, data_for_compare, comparison_op, request):
        if (data_for_compare.dtype.fill_value == 0 and comparison_op.
            __name__ != 'gt'):
            mark = pytest.mark.xfail(reason='Wrong fill_value')
            request.applymarker(mark)
        ser = pd.Series(data_for_compare)
        arr = data_for_compare + 1
        self._compare_other(ser, data_for_compare, comparison_op, arr)
        arr = data_for_compare * 2
        self._compare_other(ser, data_for_compare, comparison_op, arr)

    @pytest.mark.xfail(reason='Different repr')
    def func_21b2r291(self, data, size):
        super().test_array_repr(data, size)

    @pytest.mark.xfail(reason='result does not match expected')
    @pytest.mark.parametrize('as_index', [True, False])
    def func_0fx8wbte(self, as_index, data_for_grouping):
        super().test_groupby_extension_agg(as_index, data_for_grouping)


def func_bgve7pvq(dtype):
    assert func_dg8vi7xj.construct_array_type() is SparseArray
