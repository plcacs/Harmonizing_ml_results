    def test_hash_pandas_object(self, data: pd.Series):
    def test_value_counts_default_dropna(self, data: pd.Series):
    def test_value_counts(self, all_data: pd.Series, dropna: bool):
    def test_value_counts_with_normalize(self, data: pd.Series):
    def test_count(self, data_missing: pd.Series):
    def test_series_count(self, data_missing: pd.Series):
    def test_apply_simple_series(self, data: pd.Series):
    def test_map(self, data_missing: pd.Series, na_action: str):
    def test_argsort(self, data_for_sorting: pd.Series):
    def test_argsort_missing_array(self, data_missing_for_sorting: pd.Series):
    def test_argsort_missing(self, data_missing_for_sorting: pd.Series):
    def test_argmin_argmax(self, data_for_sorting: pd.Series, data_missing_for_sorting: pd.Series, na_value: Dtype):
    def test_argmin_argmax_empty_array(self, method: str, data: pd.Series):
    def test_argmin_argmax_all_na(self, method: str, data: pd.Series, na_value: Dtype):
    def test_argreduce_series(self, data_missing_for_sorting: pd.Series, op_name: str, skipna: bool, expected: int):
    def test_argmax_argmin_no_skipna_notimplemented(self, data_missing_for_sorting: pd.Series):
    def test_nargsort(self, data_missing_for_sorting: pd.Series, na_position: str, expected: np.ndarray):
    def test_sort_values(self, data_for_sorting: pd.Series, ascending: bool, sort_by_key: Callable):
    def test_sort_values_missing(self, data_missing_for_sorting: pd.Series, ascending: bool, sort_by_key: Callable):
    def test_sort_values_frame(self, data_for_sorting: pd.Series, ascending: bool):
    def test_duplicated(self, data: pd.Series, keep: Union[str, bool]):
    def test_unique(self, data: pd.Series, box: Union[Type[pd.Series], Callable], method: Callable):
    def test_factorize(self, data_for_grouping: pd.Series):
    def test_factorize_equivalence(self, data_for_grouping: pd.Series):
    def test_factorize_empty(self, data: pd.Series):
    def test_fillna_limit_frame(self, data_missing: pd.Series):
    def test_fillna_limit_series(self, data_missing: pd.Series):
    def test_fillna_copy_frame(self, data_missing: pd.Series):
    def test_fillna_copy_series(self, data_missing: pd.Series):
    def test_fillna_length_mismatch(self, data_missing: pd.Series):
    def test_combine_le(self, data_repeated: Callable):
    def test_combine_add(self, data_repeated: Callable):
    def test_combine_first(self, data: pd.Series):
    def test_container_shift(self, data: pd.Series, frame: bool, periods: int, indices: List[int]):
    def test_shift_0_periods(self, data: pd.Series):
    def test_diff(self, data: pd.Series, periods: int):
    def test_shift_non_empty_array(self, data: pd.Series, periods: int, indices: List[int]):
    def test_shift_empty_array(self, data: pd.Series, periods: int):
    def test_shift_zero_copies(self, data: pd.Series):
    def test_shift_fill_value(self, data: pd.Series):
    def test_not_hashable(self, data: pd.Series):
    def test_hash_pandas_object_works(self, data: pd.Series, as_frame: bool):
    def test_searchsorted(self, data_for_sorting: Tuple[pd.Series, pd.Series, pd.Series], as_series: bool):
    def _test_searchsorted_bool_dtypes(self, data_for_sorting: pd.Series, as_series: bool):
    def test_where_series(self, data: pd.Series, na_value: Dtype, as_frame: bool):
    def test_repeat(self, data: pd.Series, repeats: Union[int, List[int]], as_series: bool, use_numpy: bool):
    def test_repeat_raises(self, data: pd.Series, repeats: Union[int, List[int]], kwargs: Dict[str, Any], error: Type[Exception], msg: str, use_numpy: bool):
    def test_delete(self, data: pd.Series):
    def test_insert(self, data: pd.Series):
    def test_insert_invalid(self, data: pd.Series, invalid_scalar: Any):
    def test_insert_invalid_loc(self, data: pd.Series):
    def test_equals(self, data: pd.Series, na_value: Dtype, as_series: bool, box: Union[Type[pd.Series], Callable]):
    def test_equals_same_data_different_object(self, data: pd.Series):
