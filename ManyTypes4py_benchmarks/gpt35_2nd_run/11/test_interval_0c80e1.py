from typing import Tuple

def left_right_dtypes(request) -> Tuple:
    return request.param

class TestAttributes:

    def test_is_empty(self, constructor, left, right, closed: str):
        ...

class TestMethods:

    def test_set_closed(self, closed: str, other_closed: str):
        ...

    def test_where_raises(self, other: Union[Interval, IntervalArray]):
        ...

    def test_shift(self):
        ...

    def test_shift_datetime(self):
        ...

class TestSetitem:

    def test_set_na(self, left_right_dtypes: Tuple):
        ...

    def test_setitem_mismatched_closed(self):
        ...

class TestReductions:

    def test_min_max_invalid_axis(self, left_right_dtypes: Tuple):
        ...

    def test_min_max(self, left_right_dtypes: Tuple, index_or_series_or_array):
        ...
