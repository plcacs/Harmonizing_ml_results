from typing import Any, List, Tuple

def test_contains(self, data: JSONArray) -> None:
def test_from_dtype(self, data: JSONArray) -> None:
def test_series_constructor_no_data_with_index(self, dtype: JSONDtype, na_value: Any) -> None:
def test_series_constructor_scalar_na_with_index(self, dtype: JSONDtype, na_value: Any) -> None:
def test_series_constructor_scalar_with_index(self, data: JSONArray, dtype: JSONDtype) -> None:
def test_stack(self) -> None:
def test_unstack(self, data: JSONArray, index: Any) -> None:
def test_fillna_series(self) -> None:
def test_fillna_frame(self) -> None:
def test_fillna_with_none(self, data_missing: JSONArray) -> None:
def test_fillna_limit_frame(self, data_missing: JSONArray) -> None:
def test_fillna_limit_series(self, data_missing: JSONArray) -> None:
def test_ffill_limit_area(self, data_missing: JSONArray, limit_area: str, input_ilocs: List[int], expected_ilocs: List[int]) -> None:
def test_value_counts(self, all_data: Any, dropna: bool) -> None:
def test_value_counts_with_normalize(self, data: JSONArray) -> None:
def test_sort_values_frame(self) -> None:
def test_combine_le(self, data_repeated: Any) -> None:
def test_combine_first(self, data: JSONArray) -> None:
def test_where_series(self, data: JSONArray, na_value: Any) -> None:
def test_searchsorted(self, data_for_sorting: JSONArray) -> None:
def test_equals(self, data: JSONArray, na_value: Any, as_series: bool) -> None:
def test_fillna_copy_frame(self, data_missing: JSONArray) -> None:
def test_equals_same_data_different_object(self, data: JSONArray) -> None:
def test_astype_str(self) -> None:
def test_groupby_extension_transform(self) -> None:
def test_groupby_extension_apply(self) -> None:
def test_groupby_extension_agg(self) -> None:
def test_groupby_extension_no_sort(self) -> None:
def test_arith_frame_with_scalar(self, data: JSONArray, all_arithmetic_operators: Any, request: Any) -> None:
def test_compare_array(self, data: JSONArray, comparison_op: Any, request: Any) -> None:
def test_setitem_loc_scalar_mixed(self, data: JSONArray) -> None:
def test_setitem_loc_scalar_multiple_homogoneous(self, data: JSONArray) -> None:
def test_setitem_iloc_scalar_mixed(self, data: JSONArray) -> None:
def test_setitem_iloc_scalar_multiple_homogoneous(self, data: JSONArray) -> None:
def test_setitem_mask(self, data: JSONArray, mask: Any, box_in_series: bool, request: Any) -> None:
def test_setitem_mask_raises(self, data: JSONArray, box_in_series: bool, request: Any) -> None:
def test_setitem_mask_boolean_array_with_na(self, data: JSONArray, box_in_series: bool) -> None:
def test_setitem_integer_array(self, data: JSONArray, idx: Any, box_in_series: bool, request: Any) -> None:
def test_setitem_integer_with_missing_raises(self, data: JSONArray, idx: Any, box_in_series: bool, request: Any) -> None:
def test_setitem_scalar_key_sequence_raise(self, data: JSONArray) -> None:
def test_setitem_with_expansion_dataframe_column(self, data: JSONArray, full_indexer: bool, request: Any) -> None:
def test_setitem_frame_2d_values(self, data: JSONArray) -> None:
def test_setitem_mask_broadcast(self, data: JSONArray, setter: Any) -> None:
def test_setitem_slice(self, data: JSONArray, box_in_series: bool) -> None:
def test_setitem_loc_iloc_slice(self, data: JSONArray) -> None:
def test_setitem_slice_mismatch_length_raises(self, data: JSONArray) -> None:
def test_setitem_slice_array(self, data: JSONArray) -> None:
def test_setitem_invalid(self, data: JSONArray, invalid_scalar: Any) -> None:
def test_setitem_2d_values(self, data: JSONArray) -> None:
def test_EA_types(self, engine: str, data: JSONArray, request: Any) -> None:
