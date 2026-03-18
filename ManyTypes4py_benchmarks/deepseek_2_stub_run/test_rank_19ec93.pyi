```python
from typing import Any, Literal, Union
from datetime import datetime, timedelta
import numpy as np
import pandas._testing as tm
from pandas import DataFrame, Index, Series
from pandas._libs.algos import Infinity, NegInfinity

class TestRank:
    s: Series = ...
    df: DataFrame = ...
    results: dict[str, np.ndarray] = ...
    
    def test_rank(self, float_frame: Any) -> None: ...
    
    def test_rank2(self) -> None: ...
    
    def test_rank_does_not_mutate(self) -> None: ...
    
    def test_rank_mixed_frame(self, float_string_frame: Any) -> None: ...
    
    def test_rank_na_option(self, float_frame: Any) -> None: ...
    
    def test_rank_axis(self) -> None: ...
    
    @pytest.mark.parametrize('ax', [0, 1])
    def test_rank_methods_frame(self, ax: int, rank_method: str) -> None: ...
    
    @pytest.mark.parametrize('dtype', ['O', 'f8', 'i8'])
    def test_rank_descending(self, rank_method: str, dtype: str) -> None: ...
    
    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('dtype', [None, object])
    def test_rank_2d_tie_methods(self, rank_method: str, axis: int, dtype: Any) -> None: ...
    
    @pytest.mark.parametrize('rank_method,exp', [
        ('dense', [[1.0, 1.0, 1.0], [1.0, 0.5, 2.0 / 3], [1.0, 0.5, 1.0 / 3]]),
        ('min', [[1.0 / 3, 1.0, 1.0], [1.0 / 3, 1.0 / 3, 2.0 / 3], [1.0 / 3, 1.0 / 3, 1.0 / 3]]),
        ('max', [[1.0, 1.0, 1.0], [1.0, 2.0 / 3, 2.0 / 3], [1.0, 2.0 / 3, 1.0 / 3]]),
        ('average', [[2.0 / 3, 1.0, 1.0], [2.0 / 3, 0.5, 2.0 / 3], [2.0 / 3, 0.5, 1.0 / 3]]),
        ('first', [[1.0 / 3, 1.0, 1.0], [2.0 / 3, 1.0 / 3, 2.0 / 3], [3.0 / 3, 2.0 / 3, 1.0 / 3]])
    ])
    def test_rank_pct_true(self, rank_method: str, exp: list[list[float]]) -> None: ...
    
    @pytest.mark.single_cpu
    def test_pct_max_many_rows(self) -> None: ...
    
    @pytest.mark.parametrize('contents,dtype', [
        ([-np.inf, -50, -1, -1e-20, -1e-25, -1e-50, 0, 1e-40, 1e-20, 1e-10, 2, 40, np.inf], 'float64'),
        ([-np.inf, -50, -1, -1e-20, -1e-25, -1e-45, 0, 1e-40, 1e-20, 1e-10, 2, 40, np.inf], 'float32'),
        ([np.iinfo(np.uint8).min, 1, 2, 100, np.iinfo(np.uint8).max], 'uint8'),
        ([np.iinfo(np.int64).min, -100, 0, 1, 9999, 100000, 10000000000.0, np.iinfo(np.int64).max], 'int64'),
        ([NegInfinity(), '1', 'A', 'BA', 'Ba', 'C', Infinity()], 'object'),
        ([datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 5)], 'datetime64')
    ])
    def test_rank_inf_and_nan(self, contents: list[Any], dtype: str, frame_or_series: Any) -> None: ...
    
    def test_df_series_inf_nan_consistency(self) -> None: ...
    
    def test_rank_both_inf(self) -> None: ...
    
    @pytest.mark.parametrize('na_option,ascending,expected', [
        ('top', True, [3.0, 1.0, 2.0]),
        ('top', False, [2.0, 1.0, 3.0]),
        ('bottom', True, [2.0, 3.0, 1.0]),
        ('bottom', False, [1.0, 3.0, 2.0])
    ])
    def test_rank_inf_nans_na_option(self, frame_or_series: Any, rank_method: str, na_option: str, ascending: bool, expected: list[float]) -> None: ...
    
    @pytest.mark.parametrize('na_option,ascending,expected', [
        ('bottom', True, [1.0, 2.0, 4.0, 3.0]),
        ('bottom', False, [1.0, 2.0, 4.0, 3.0]),
        ('top', True, [2.0, 3.0, 1.0, 4.0]),
        ('top', False, [2.0, 3.0, 1.0, 4.0])
    ])
    def test_rank_object_first(self, frame_or_series: Any, na_option: str, ascending: bool, expected: list[float]) -> None: ...
    
    @pytest.mark.parametrize('data,expected', [
        ({'a': [1, 2, 'a'], 'b': [4, 5, 6]}, DataFrame({'b': [1.0, 2.0, 3.0]}, columns=Index(['b'], dtype=object))),
        ({'a': [1, 2, 'a']}, DataFrame(index=range(3), columns=[]))
    ])
    def test_rank_mixed_axis_zero(self, data: dict[str, list[Any]], expected: DataFrame) -> None: ...
    
    def test_rank_string_dtype(self, string_dtype_no_object: Any) -> None: ...
```