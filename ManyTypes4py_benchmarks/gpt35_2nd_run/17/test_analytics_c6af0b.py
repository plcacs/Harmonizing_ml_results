    def test_min_max_ordered(self, index_or_series_or_array: Any):
    def test_min_max_ordered_empty(self, categories: List, expected: Any, aggregation: str):
    def test_min_max_with_nan(self, values: List, categories: List, function: str, skipna: bool):
    def test_min_max_only_nan(self, function: str, skipna: bool):
    def test_numeric_only_min_max_raises(self, method: str):
    def test_numpy_min_max_raises(self, method: str):
    def test_numpy_min_max_unsupported_kwargs_raises(self, method: str, kwarg: str):
    def test_numpy_min_max_axis_equals_none(self, method: str, expected: Any):
    def test_mode(self, values: List, categories: List, exp_mode: List):
    def test_searchsorted(self, ordered: bool):
    def test_unique(self, ordered: bool):
    def test_unique_index_series(self, ordered: bool):
    def test_memory_usage(self, using_infer_string: bool):
    def test_validate_inplace_raises(self, value: Any):
