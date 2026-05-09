import numpy as np
import pytest
from pandas import Categorical, CategoricalDtype, DataFrame, Index, NaT, Series, date_range
from pandas.api.types import is_scalar

class TestCategoricalAnalytics:
    @pytest.mark.parametrize('aggregation', ['min', 'max'])
    def test_min_max_not_ordered_raises(self, aggregation: str) -> None:
        ...

    def test_min_max_ordered(self, index_or_series_or_array: Union[pd.Index, pd.Series, np.ndarray]) -> None:
        ...

    def test_min_max_ordered_empty(self, categories: Union[List[str], List[int], pd.Series], expected: Union[np.nan, NaT], aggregation: str) -> None:
        ...

    @pytest.mark.parametrize('values, categories', [(['a', 'b', 'c', np.nan], list('cba')), ([1, 2, 3, np.nan], [3, 2, 1])])
    @pytest.mark.parametrize('function', ['min', 'max'])
    def test_min_max_with_nan(self, values: List[Union[str, float]], categories: List[str], function: str, skipna: bool) -> None:
        ...

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_numeric_only_min_max_raises(self, method: str) -> None:
        ...

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_numpy_min_max_raises(self, method: str) -> None:
        ...

    @pytest.mark.parametrize('kwarg', ['axis', 'out', 'keepdims'])
    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_numpy_min_max_unsupported_kwargs_raises(self, method: str, kwarg: str) -> None:
        ...

    @pytest.mark.parametrize('method, expected', [('min', 'a'), ('max', 'c')])
    def test_numpy_min_max_axis_equals_none(self, method: str, expected: str) -> None:
        ...

    @pytest.mark.parametrize('values,categories,exp_mode', [([1, 1, 2, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5]), ([1, 1, 1, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5, 1]), ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1]), ([np.nan, np.nan, np.nan, 4, 5], [5, 4, 3, 2, 1], [5, 4]), ([np.nan, np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4]), ([np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4])])
    def test_mode(self, values: List[Any], categories: List[Any], exp_mode: List[Any]) -> None:
        ...

    def test_searchsorted(self, ordered: bool) -> None:
        ...

    def test_unique(self, ordered: bool) -> None:
        ...

    def test_unique_index_series(self, ordered: bool) -> None:
        ...

    def test_shift(self) -> None:
        ...

    def test_nbytes(self) -> None:
        ...

    def test_memory_usage(self, using_infer_string: bool) -> None:
        ...

    def test_map(self) -> None:
        ...

    @pytest.mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
    def test_validate_inplace_raises(self, value: Any) -> None:
        ...

    def test_quantile_empty(self, idx: pd.Index) -> None:
        ...