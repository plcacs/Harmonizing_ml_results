from typing import List, Dict, Union, Tuple

def test_groupby_dropna_multi_index_dataframe_nan_in_one_group(dropna: bool, tuples: List[List[str]], outputs: Dict[str, List[Union[float, int]]], nulls_fixture: str) -> None:
def test_groupby_dropna_multi_index_dataframe_nan_in_two_groups(dropna: bool, tuples: List[List[str]], outputs: Dict[str, List[Union[float, int]]], nulls_fixture: str, nulls_fixture2: str) -> None:
def test_groupby_dropna_normal_index_dataframe(dropna: bool, idx: List[str], outputs: Dict[str, List[Union[float, int]]]) -> None:
def test_groupby_dropna_series_level(dropna: bool, idx: List[str], expected: pd.Series) -> None:
def test_groupby_dropna_series_by(dropna: bool, expected: pd.Series) -> None:
def test_grouper_dropna_propagation(dropna: bool) -> None:
def test_groupby_dataframe_slice_then_transform(dropna: bool, index: pd.Index) -> None:
def test_groupby_dropna_multi_index_dataframe_agg(dropna: bool, tuples: List[List[str]], outputs: Dict[str, List[Union[float, int]]]) -> None:
def test_groupby_dropna_datetime_like_data(dropna: bool, values: List[Union[float, int]], datetime1: pd.Timestamp, datetime2: pd.Timestamp, unique_nulls_fixture: str, unique_nulls_fixture2: str) -> None:
def test_groupby_apply_with_dropna_for_multi_index(dropna: bool, data: Dict[str, List[str]], selected_data: Dict[str, List[Union[int, str]]], levels: List[str]) -> None:
def test_groupby_dropna_with_multiindex_input(input_index: List[str], keys: List[str], series: bool) -> None:
def test_groupby_nan_included() -> None:
def test_groupby_drop_nan_with_multi_index() -> None:
def test_no_sort_keep_na(sequence_index: int, dtype: Union[type, str], test_series: bool, as_index: bool) -> None:
def test_null_is_null_for_dtype(sort: bool, dtype: Union[type, None], nulls_fixture: str, nulls_fixture2: str, test_series: bool) -> None:
def test_categorical_reducers(reduction_func: str, observed: bool, sort: bool, as_index: bool, index_kind: str) -> None:
def test_categorical_transformers(transformation_func: str, observed: bool, sort: bool, as_index: bool) -> None:
def test_categorical_head_tail(method: str, observed: bool, sort: bool, as_index: bool) -> None:
def test_categorical_agg() -> None:
def test_categorical_transform() -> None:
