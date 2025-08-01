import string
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray

class TestSeriesAccessor:

    def test_to_dense(self) -> None:
        ser = pd.Series([0, 1, 0, 10], dtype='Sparse[int64]')
        result = ser.sparse.to_dense()
        expected = pd.Series([0, 1, 0, 10])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('attr', ['npoints', 'density', 'fill_value', 'sp_values'])
    def test_get_attributes(self, attr: str) -> None:
        arr = SparseArray([0, 1])
        ser = pd.Series(arr)
        result = getattr(ser.sparse, attr)
        expected = getattr(arr, attr)
        assert result == expected

    def test_from_coo(self) -> None:
        scipy_sparse = pytest.importorskip('scipy.sparse')
        row: List[int] = [0, 3, 1, 0]
        col: List[int] = [0, 3, 1, 2]
        data: List[int] = [4, 5, 7, 9]
        sp_array = scipy_sparse.coo_matrix((data, (row, col)))
        result = pd.Series.sparse.from_coo(sp_array)
        index = pd.MultiIndex.from_arrays([np.array([0, 0, 1, 3], dtype=np.int32), np.array([0, 2, 1, 3], dtype=np.int32)])
        expected = pd.Series([4, 9, 7, 5], index=index, dtype='Sparse[int]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('sort_labels, expected_rows, expected_cols, expected_values_pos', [
        (False, [('b', 2), ('a', 2), ('b', 1), ('a', 1)], [('z', 1), ('z', 2), ('x', 2), ('z', 0)], {1: (1, 0), 3: (3, 3)}),
        (True, [('a', 1), ('a', 2), ('b', 1), ('b', 2)], [('x', 2), ('z', 0), ('z', 1), ('z', 2)], {1: (1, 2), 3: (0, 1)})
    ])
    def test_to_coo(self, sort_labels: bool, expected_rows: List[Tuple[str, int]], 
                   expected_cols: List[Tuple[str, int]], 
                   expected_values_pos: Dict[int, Tuple[int, int]]) -> None:
        sp_sparse = pytest.importorskip('scipy.sparse')
        values = SparseArray([0, np.nan, 1, 0, None, 3], fill_value=0)
        index = pd.MultiIndex.from_tuples([('b', 2, 'z', 1), ('a', 2, 'z', 2), ('a', 2, 'z', 1), ('a', 2, 'x', 2), ('b', 1, 'z', 1), ('a', 1, 'z', 0)])
        ss = pd.Series(values, index=index)
        expected_A = np.zeros((4, 4))
        for value, (row, col) in expected_values_pos.items():
            expected_A[row, col] = value
        A, rows, cols = ss.sparse.to_coo(row_levels=(0, 1), column_levels=(2, 3), sort_labels=sort_labels)
        assert isinstance(A, sp_sparse.coo_matrix)
        tm.assert_numpy_array_equal(A.toarray(), expected_A)
        assert rows == expected_rows
        assert cols == expected_cols

    def test_non_sparse_raises(self) -> None:
        ser = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError, match='.sparse'):
            ser.sparse.density

class TestFrameAccessor:

    def test_accessor_raises(self) -> None:
        df = pd.DataFrame({'A': [0, 1]})
        with pytest.raises(AttributeError, match='sparse'):
            df.sparse

    @pytest.mark.parametrize('format', ['csc', 'csr', 'coo'])
    @pytest.mark.parametrize('labels', [None, list(string.ascii_letters[:10])])
    @pytest.mark.parametrize('dtype', [np.complex128, np.float64, np.int64, bool])
    def test_from_spmatrix(self, format: str, labels: Optional[List[str]], dtype: np.dtype) -> None:
        sp_sparse = pytest.importorskip('scipy.sparse')
        sp_dtype = SparseDtype(dtype)
        sp_mat = sp_sparse.eye(10, format=format, dtype=dtype)
        result = pd.DataFrame.sparse.from_spmatrix(sp_mat, index=labels, columns=labels)
        mat = np.eye(10, dtype=dtype)
        expected = pd.DataFrame(np.ma.array(mat, mask=mat == 0).filled(sp_dtype.fill_value), index=labels, columns=labels).astype(sp_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('format', ['csc', 'csr', 'coo'])
    @pytest.mark.parametrize('dtype', [np.int64, bool])
    def test_from_spmatrix_including_explicit_zero(self, format: str, dtype: np.dtype) -> None:
        sp_sparse = pytest.importorskip('scipy.sparse')
        sp_dtype = SparseDtype(dtype)
        sp_mat = sp_sparse.random(10, 2, density=0.5, format=format, dtype=dtype)
        sp_mat.data[0] = 0
        result = pd.DataFrame.sparse.from_spmatrix(sp_mat)
        mat = sp_mat.toarray()
        expected = pd.DataFrame(np.ma.array(mat, mask=mat == 0).filled(sp_dtype.fill_value)).astype(sp_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('columns', [['a', 'b'], pd.MultiIndex.from_product([['A'], ['a', 'b']]), ['a', 'a']])
    def test_from_spmatrix_columns(self, columns: Union[List[str], pd.MultiIndex]) -> None:
        sp_sparse = pytest.importorskip('scipy.sparse')
        sp_dtype = SparseDtype(np.float64)
        sp_mat = sp_sparse.random(10, 2, density=0.5)
        result = pd.DataFrame.sparse.from_spmatrix(sp_mat, columns=columns)
        mat = sp_mat.toarray()
        expected = pd.DataFrame(np.ma.array(mat, mask=mat == 0).filled(sp_dtype.fill_value), columns=columns).astype(sp_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('columns', [('A', 'B'), (1, 2), (1, pd.NA), (0.1, 0.2), ('x', 'x'), (0, 0)])
    @pytest.mark.parametrize('dtype', [np.complex128, np.float64, np.int64, bool])
    def test_to_coo(self, columns: Tuple[Any, Any], dtype: np.dtype) -> None:
        sp_sparse = pytest.importorskip('scipy.sparse')
        sp_dtype = SparseDtype(dtype)
        expected = sp_sparse.random(10, 2, density=0.5, format='coo', dtype=dtype)
        mat = expected.toarray()
        result = pd.DataFrame(np.ma.array(mat, mask=mat == 0).filled(sp_dtype.fill_value), columns=columns, dtype=sp_dtype).sparse.to_coo()
        assert (result != expected).nnz == 0

    def test_to_coo_midx_categorical(self) -> None:
        sp_sparse = pytest.importorskip('scipy.sparse')
        midx = pd.MultiIndex.from_arrays([pd.CategoricalIndex(list('ab'), name='x'), pd.CategoricalIndex([0, 1], name='y')])
        ser = pd.Series(1, index=midx, dtype='Sparse[int]')
        result = ser.sparse.to_coo(row_levels=['x'], column_levels=['y'])[0]
        expected = sp_sparse.coo_matrix((np.array([1, 1]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2))
        assert (result != expected).nnz == 0

    def test_to_dense(self) -> None:
        df = pd.DataFrame({
            'A': SparseArray([1, 0], dtype=SparseDtype('int64', 0)),
            'B': SparseArray([1, 0], dtype=SparseDtype('int64', 1)),
            'C': SparseArray([1.0, 0.0], dtype=SparseDtype('float64', 0.0))
        }, index=['b', 'a'])
        result = df.sparse.to_dense()
        expected = pd.DataFrame({'A': [1, 0], 'B': [1, 0], 'C': [1.0, 0.0]}, index=['b', 'a'])
        tm.assert_frame_equal(result, expected)

    def test_density(self) -> None:
        df = pd.DataFrame({
            'A': SparseArray([1, 0, 2, 1], fill_value=0),
            'B': SparseArray([0, 1, 1, 1], fill_value=0)
        })
        res = df.sparse.density
        expected = 0.75
        assert res == expected

    @pytest.mark.parametrize('dtype', ['int64', 'float64'])
    @pytest.mark.parametrize('dense_index', [True, False])
    def test_series_from_coo(self, dtype: str, dense_index: bool) -> None:
        sp_sparse = pytest.importorskip('scipy.sparse')
        A = sp_sparse.eye(3, format='coo', dtype=dtype)
        result = pd.Series.sparse.from_coo(A, dense_index=dense_index)
        index = pd.MultiIndex.from_tuples([np.array([0, 0], dtype=np.int32), np.array([1, 1], dtype=np.int32), np.array([2, 2], dtype=np.int32)])
        expected = pd.Series(SparseArray(np.array([1, 1, 1], dtype=dtype)), index=index)
        if dense_index:
            expected = expected.reindex(pd.MultiIndex.from_product(index.levels))
        tm.assert_series_equal(result, expected)

    def test_series_from_coo_incorrect_format_raises(self) -> None:
        sp_sparse = pytest.importorskip('scipy.sparse')
        m = sp_sparse.csr_matrix(np.array([[0, 1], [0, 0]]))
        with pytest.raises(TypeError, match='Expected coo_matrix. Got csr_matrix instead.'):
            pd.Series.sparse.from_coo(m)

    def test_with_column_named_sparse(self) -> None:
        df = pd.DataFrame({'sparse': pd.arrays.SparseArray([1, 2])})
        assert isinstance(df.sparse, pd.core.arrays.sparse.accessor.SparseFrameAccessor)

    def test_subclassing(self) -> None:
        df = tm.SubclassedDataFrame({'sparse': pd.arrays.SparseArray([1, 2])})
        assert isinstance(df.sparse.to_dense(), tm.SubclassedDataFrame)
