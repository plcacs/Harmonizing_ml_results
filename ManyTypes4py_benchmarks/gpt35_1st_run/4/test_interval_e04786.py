from typing import List, Tuple, Union

def create_categorical_intervals(left: List, right: List, closed: str = 'right') -> Categorical:
    return Categorical(IntervalIndex.from_arrays(left, right, closed))

def create_series_intervals(left: List, right: List, closed: str = 'right') -> Series:
    return Series(IntervalArray.from_arrays(left, right, closed))

def create_series_categorical_intervals(left: List, right: List, closed: str = 'right') -> Series:
    return Series(Categorical(IntervalIndex.from_arrays(left, right, closed))

class TestComparison:
    def elementwise_comparison(self, op, interval_array, other) -> np.array:
        ...

    def test_compare_scalar_interval(self, op, interval_array) -> None:
        ...

    def test_compare_scalar_interval_mixed_closed(self, op, closed, other_closed) -> None:
        ...

    def test_compare_scalar_na(self, op, interval_array, nulls_fixture, box_with_array) -> None:
        ...

    def test_compare_scalar_other(self, op, interval_array, other) -> None:
        ...

    def test_compare_list_like_interval(self, op, interval_array, interval_constructor) -> None:
        ...

    def test_compare_list_like_interval_mixed_closed(self, op, interval_constructor, closed, other_closed) -> None:
        ...

    def test_compare_list_like_object(self, op, interval_array, other) -> None:
        ...

    def test_compare_list_like_nan(self, op, interval_array, nulls_fixture) -> None:
        ...

    def test_compare_list_like_other(self, op, interval_array, other) -> None:
        ...

    def test_compare_length_mismatch_errors(self, op, other_constructor, length) -> None:
        ...

    def test_index_series_compat(self, op, constructor, expected_type, assert_func) -> None:
        ...

    def test_comparison_operations(self, scalars) -> None:
        ...
