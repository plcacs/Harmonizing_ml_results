from typing import List, Tuple, Dict, Union

def test_apply_to_timedelta(by_row: bool):
def test_apply_listlike_reducer(string_series: Series, ops: List[callable], names: List[str], how: str, kwargs: Dict):
def test_apply_dictlike_reducer(string_series: Series, ops: Dict[str, callable], how: str, kwargs: Dict, by_row: bool):
def test_apply_listlike_transformer(string_series: Series, ops: List[callable], names: List[str], by_row: bool):
def test_apply_dictlike_transformer(string_series: Series, ops: Dict[str, callable], by_row: bool):
def test_apply_retains_column_name(by_row: bool):
def test_series_apply_unpack_nested_data():
