from pandas.arrays import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.arrays.sparse import SparseArray
from pandas.core.arrays.sparse.dtype import SparseDtype
import numpy as np
import pandas as pd
import pytest

def make_data(fill_value: float) -> np.ndarray:
    ...

def test_array_type_with_arg(dtype: ExtensionDtype) -> None:
    ...

@pytest.fixture
def dtype() -> SparseDtype:
    ...

@pytest.fixture(params=[0, np.nan])
def data(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture
def data_for_twos() -> SparseArray:
    ...

@pytest.fixture(params=[0, np.nan])
def data_missing(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture(params=[0, np.nan])
def data_repeated(request: pytest.FixtureRequest) -> Generator[SparseArray, None, None]:
    ...

@pytest.fixture(params=[0, np.nan])
def data_for_sorting(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture(params=[0, np.nan])
def data_missing_for_sorting(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]:
    ...

@pytest.fixture(params=[0, np.nan])
def data_for_grouping(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture(params=[0, np.nan])
def data_for_compare(request: pytest.FixtureRequest) -> SparseArray:
    ...

class TestSparseArray(base.ExtensionTests):

    def _supports_reduction(self, obj: ExtensionArray, op_name: str) -> bool:
        ...

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_series_numeric(self, data: SparseArray, all_numeric_reductions: str, skipna: bool, request: pytest.FixtureRequest) -> None:
        ...

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_frame(self, data: SparseArray, all_numeric_reductions: str, skipna: bool, request: pytest.FixtureRequest) -> None:
        ...

    def _check_unsupported(self, data: SparseArray) -> None:
        ...

    def test_concat_mixed_dtypes(self, data: SparseArray) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('columns', [['A', 'B'], pd.MultiIndex.from_tuples([('A', 'a'), ('A', 'b')], names=['outer', 'inner'])])
    @pytest.mark.parametrize('future_stack', [True, False])
    def test_stack(self, data: SparseArray, columns: List[str], future_stack: bool) -> None:
        ...

    def test_concat_columns(self, data: SparseArray, na_value: Any) -> None:
        ...

    def test_concat_extension_arrays_copy_false(self, data: SparseArray, na_value: Any) -> None:
        ...

    def test_align(self, data: SparseArray, na_value: Any) -> None:
        ...

    def test_align_frame(self, data: SparseArray, na_value: Any) -> None:
        ...

    def test_align_series_frame(self, data: SparseArray, na_value: Any) -> None:
        ...

    def test_merge(self, data: SparseArray, na_value: Any) -> None:
        ...

    def test_get(self, data: SparseArray) -> None:
        ...

    def test_reindex(self, data: SparseArray, na_value: Any) -> None:
        ...

    def test_isna(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_no_op_returns_copy(self, data: SparseArray) -> None:
        ...

    @pytest.mark.xfail(reason='Unsupported')
    def test_fillna_series(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_frame(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_limit_frame(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_limit_series(self, data_missing: SparseArray) -> None:
        ...

    _combine_le_expected_dtype: str = 'Sparse[bool]'

    def test_fillna_copy_frame(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_copy_series(self, data_missing: SparseArray) -> None:
        ...

    @pytest.mark.xfail(reason='Not Applicable')
    def test_fillna_length_mismatch(self, data_missing: SparseArray) -> None:
        ...

    def test_where_series(self, data: SparseArray, na_value: Any) -> None:
        ...

    def test_searchsorted(self, performance_warning: Any, data_for_sorting: SparseArray, as_series: bool) -> None:
        ...

    def test_shift_0_periods(self, data: SparseArray) -> None:
        ...

    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def test_argmin_argmax_all_na(self, method: str, data: SparseArray, na_value: Any) -> None:
        ...

    @pytest.mark.fails_arm_wheels
    @pytest.mark.parametrize('box', [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data: SparseArray, na_value: Any, as_series: bool, box: Type) -> None:
        ...

    @pytest.mark.fails_arm_wheels
    def test_equals_same_data_different_object(self, data: SparseArray) -> None:
        ...

    @pytest.mark.parametrize('func, na_action, expected', [(lambda x: x, None, SparseArray([1.0, np.nan])), (lambda x: x, 'ignore', SparseArray([1.0, np.nan])), (str, None, SparseArray(['1.0', 'nan'], fill_value='nan')), (str, 'ignore', SparseArray(['1.0', np.nan]))])
    def test_map(self, func: Callable, na_action: Optional[str], expected: SparseArray) -> None:
        ...

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def test_map_raises(self, data: SparseArray, na_action: Optional[str]) -> None:
        ...

    @pytest.mark.xfail(raises=TypeError, reason='no sparse StringDtype')
    def test_astype_string(self, data: SparseArray) -> None:
        ...

    series_scalar_exc: Optional[Type] = None
    frame_scalar_exc: Optional[Type] = None
    divmod_exc: Optional[Type] = None
    series_array_exc: Optional[Type] = None

    def _skip_if_different_combine(self, data: SparseArray) -> None:
        ...

    def test_arith_series_with_scalar(self, data: SparseArray, all_arithmetic_operators: str) -> None:
        ...

    def test_arith_series_with_array(self, data: SparseArray, all_arithmetic_operators: str) -> None:
        ...

    def test_arith_frame_with_scalar(self, data: SparseArray, all_arithmetic_operators: str, request: pytest.FixtureRequest) -> None:
        ...

    def _compare_other(self, ser: pd.Series, data_for_compare: SparseArray, comparison_op: Callable, other: Any) -> None:
        ...

    def test_scalar(self, data_for_compare: SparseArray, comparison_op: Callable) -> None:
        ...

    def test_array(self, data_for_compare: SparseArray, comparison_op: Callable, request: pytest.FixtureRequest) -> None:
        ...

    def test_sparse_array(self, data_for_compare: SparseArray, comparison_op: Callable, request: pytest.FixtureRequest) -> None:
        ...

    @pytest.mark.xfail(reason='Different repr')
    def test_array_repr(self, data: SparseArray, size: int) -> None:
        ...

    @pytest.mark.xfail(reason='result does not match expected')
    @pytest.mark.parametrize('as_index', [True, False])
    def test_groupby_extension_agg(self, as_index: bool, data_for_grouping: SparseArray) -> None:
        ...
