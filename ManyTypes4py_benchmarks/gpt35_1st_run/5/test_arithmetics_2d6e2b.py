from typing import List, Union

def test_with_list(op: callable):
    arr: SparseArray = SparseArray([0, 1], fill_value=0)
    result: SparseArray = op(arr, [0, 1])
    expected: SparseArray = op(arr, SparseArray([0, 1]))
    tm.assert_sp_array_equal(result, expected)

def test_with_dataframe():
    arr: SparseArray = SparseArray([0, 1], fill_value=0)
    df: pd.DataFrame = pd.DataFrame([[1, 2], [3, 4])
    result = arr.__add__(df)
    assert result is NotImplemented

def test_with_zerodim_ndarray():
    arr: SparseArray = SparseArray([0, 1], fill_value=0)
    result: SparseArray = arr * np.array(2)
    expected: SparseArray = arr * 2
    tm.assert_sp_array_equal(result, expected)

def test_ufuncs(ufunc: callable, arr: SparseArray):
    result: SparseArray = ufunc(arr)
    fill_value = ufunc(arr.fill_value)
    expected: SparseArray = SparseArray(ufunc(np.asarray(arr)), fill_value=fill_value)
    tm.assert_sp_array_equal(result, expected)

def test_binary_ufuncs(ufunc: callable, a: SparseArray, b: np.ndarray):
    result: SparseArray = ufunc(a, b)
    expected = ufunc(np.asarray(a), np.asarray(b))
    assert isinstance(result, SparseArray)
    tm.assert_numpy_array_equal(np.asarray(result), expected)

def test_ndarray_inplace():
    sparray: SparseArray = SparseArray([0, 2, 0, 0])
    ndarray: np.ndarray = np.array([0, 1, 2, 3])
    ndarray += sparray
    expected: np.ndarray = np.array([0, 3, 2, 3])
    tm.assert_numpy_array_equal(ndarray, expected)

def test_sparray_inplace():
    sparray: SparseArray = SparseArray([0, 2, 0, 0])
    ndarray: np.ndarray = np.array([0, 1, 2, 3])
    sparray += ndarray
    expected: SparseArray = SparseArray([0, 3, 2, 3], fill_value=0)
    tm.assert_sp_array_equal(sparray, expected)

def test_mismatched_length_cmp_op(cons: callable):
    left: SparseArray = SparseArray([True, True])
    right: Union[List, np.ndarray, SparseArray] = cons([True, True, True])
    with pytest.raises(ValueError, match='operands have mismatched length'):
        left & right

def test_mismatched_length_arith_op(a: List[int], b: List[int], all_arithmetic_functions: callable):
    op = all_arithmetic_functions
    with pytest.raises(AssertionError, match=f'length mismatch: {len(a)} vs. {len(b)}'):
        op(SparseArray(a, fill_value=0), np.array(b))

def test_binary_operators(op: str, fill_value: Union[np.nan, int]):
    op = getattr(operator, op)
    data1: np.ndarray = np.random.default_rng(2).standard_normal(20)
    data2: np.ndarray = np.random.default_rng(2).standard_normal(20)
    data1[::2] = fill_value
    data2[::3] = fill_value
    first: SparseArray = SparseArray(data1, fill_value=fill_value)
    second: SparseArray = SparseArray(data2, fill_value=fill_value)
    with np.errstate(all='ignore'):
        res: SparseArray = op(first, second)
        exp: SparseArray = SparseArray(op(first.to_dense(), second.to_dense()), fill_value=first.fill_value)
        assert isinstance(res, SparseArray)
        tm.assert_almost_equal(res.to_dense(), exp.to_dense())
        res2: SparseArray = op(first, second.to_dense())
        assert isinstance(res2, SparseArray)
        tm.assert_sp_array_equal(res, res2)
        res3: SparseArray = op(first.to_dense(), second)
        assert isinstance(res3, SparseArray)
        tm.assert_sp_array_equal(res, res3)
        res4: SparseArray = op(first, 4)
        assert isinstance(res4, SparseArray)
        try:
            exp = op(first.to_dense(), 4)
            exp_fv = op(first.fill_value, 4)
        except ValueError:
            pass
        else:
            tm.assert_almost_equal(res4.fill_value, exp_fv)
            tm.assert_almost_equal(res4.to_dense(), exp)
