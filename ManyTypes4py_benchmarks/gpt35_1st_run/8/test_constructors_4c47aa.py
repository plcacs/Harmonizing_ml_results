from typing import List, Tuple, Union

class ConstructorTests:
    def test_constructor(self, constructor, breaks_and_expected_subtype: Tuple[List[Union[int, float]], Union[type, str]], closed: str, name: str):
    def test_constructor_dtype(self, constructor, breaks: List[int], subtype: str):
    def test_constructor_pass_closed(self, constructor, breaks: List[int]):
    def test_constructor_nan(self, constructor, breaks: List[float], closed: str):
    def test_constructor_empty(self, constructor, breaks: List[int], closed: str):
    def test_constructor_string(self, constructor, breaks: List[str]):
    def test_constructor_categorical_valid(self, constructor, cat_constructor):
    def test_generic_errors(self, constructor):
    def test_from_arrays_mismatched_datetimelike_resos(self, interval_cls):
    def test_from_breaks_mismatched_datetimelike_resos(self, interval_cls):
    def test_from_tuples_mismatched_datetimelike_resos(self, interval_cls):
    def test_override_inferred_closed(self, constructor, data, closed: str):
    def test_index_object_dtype(self, values_constructor):
