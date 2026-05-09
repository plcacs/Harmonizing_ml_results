import numpy as np
import pytest
from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
from pandas.util._test_decorators import skip_if_no
from pandas import DataFrame, Series, option_context, to_datetime
from pandas._testing import assert_series_equal, assert_frame_equal, assert_equal
from pandas.util.version import Version

pytestmark: list[pytest.Mark] = ...

method: pytest.Fixture[str] = ...
arithmetic_numba_supported_operators: pytest.Fixture[tuple[str, dict[str, Any]]] = ...
roll_frame: pytest.Fixture[DataFrame] = ...

@skip_if_no('numba')
@pytest.mark.filterwarnings('ignore')
class TestEngine:
    @pytest.mark.parametrize('jit', [True, False])
    def test_numba_vs_cython_apply(self, jit: bool, nogil: bool, parallel: bool, nopython: bool, center: bool, step: int | None) -> None:
        ...

    def test_apply_numba_with_kwargs(self, roll_frame: DataFrame) -> None:
        ...

    def test_numba_min_periods(self) -> None:
        ...

    @pytest.mark.parametrize('data', [DataFrame, Series])
    def test_numba_vs_cython_rolling_methods(self, data: DataFrame | Series, nogil: bool, parallel: bool, nopython: bool, arithmetic_numba_supported_operators: tuple[str, dict[str, Any]], step: int | None) -> None:
        ...

    @pytest.mark.parametrize('data', [DataFrame, Series])
    def test_numba_vs_cython_expanding_methods(self, data: DataFrame | Series, nogil: bool, parallel: bool, nopython: bool, arithmetic_numba_supported_operators: tuple[str, dict[str, Any]]) -> None:
        ...

    @pytest.mark.parametrize('jit', [True, False])
    def test_cache_apply(self, jit: bool, nogil: bool, parallel: bool, nopython: bool, step: int | None) -> None:
        ...

    @pytest.mark.parametrize('window,window_kwargs', [['rolling', {'window': 3, 'min_periods': 0}], ['expanding', {}]])
    def test_dont_cache_args(self, window: str, window_kwargs: dict[str, Any], nogil: bool, parallel: bool, nopython: bool, method: str) -> None:
        ...

    def test_dont_cache_engine_kwargs(self) -> None:
        ...

@skip_if_no('numba')
class TestEWM:
    @pytest.mark.parametrize('grouper', [lambda x: x, lambda x: x.groupby('A')], ids=['None', 'groupby'])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_invalid_engine(self, grouper: Callable[[DataFrame], DataFrame], method: str) -> None:
        ...

    @pytest.mark.parametrize('grouper', [lambda x: x, lambda x: x.groupby('A')], ids=['None', 'groupby'])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_invalid_engine_kwargs(self, grouper: Callable[[DataFrame], DataFrame], method: str) -> None:
        ...

    @pytest.mark.parametrize('grouper', ['None', 'groupby'])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_cython_vs_numba(self, grouper: str, method: str, nogil: bool, parallel: bool, nopython: bool, ignore_na: bool, adjust: bool) -> None:
        ...

    @pytest.mark.parametrize('grouper', ['None', 'groupby'])
    def test_cython_vs_numba_times(self, grouper: str, nogil: bool, parallel: bool, nopython: bool, ignore_na: bool) -> None:
        ...

@skip_if_no('numba')
def test_use_global_config() -> None:
    ...

@skip_if_no('numba')
def test_invalid_kwargs_nopython() -> None:
    ...

@skip_if_no('numba')
@pytest.mark.slow
@pytest.mark.filterwarnings('ignore')
class TestTableMethod:
    def test_table_series_valueerror(self) -> None:
        ...

    def test_table_method_rolling_methods(self, nogil: bool, parallel: bool, nopython: bool, arithmetic_numba_supported_operators: tuple[str, dict[str, Any]], step: int | None) -> None:
        ...

    def test_table_method_rolling_apply(self, nogil: bool, parallel: bool, nopython: bool, step: int | None) -> None:
        ...

    def test_table_method_rolling_apply_col_order(self) -> None:
        ...

    def test_table_method_rolling_weighted_mean(self, step: int | None) -> None:
        ...

    def test_table_method_expanding_apply(self, nogil: bool, parallel: bool, nopython: bool) -> None:
        ...

    def test_table_method_expanding_methods(self, nogil: bool, parallel: bool, nopython: bool, arithmetic_numba_supported_operators: tuple[str, dict[str, Any]]) -> None:
        ...

    @pytest.mark.parametrize('data', [np.eye(3), np.ones((2, 3)), np.ones((3, 2))])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_table_method_ewm(self, data: np.ndarray, method: str, nogil: bool, parallel: bool, nopython: bool) -> None:
        ...

@skip_if_no('numba')
def test_npfunc_no_warnings() -> None:
    ...