from datetime import date, datetime, time, timedelta
from typing import List, Tuple, Union

def test_get_slice_bounds_datetime_within(box: Union[date, datetime, Timestamp], side: str, tz_aware_fixture) -> int:
def test_get_slice_bounds_datetime_outside(box: Union[datetime, Timestamp], side: str, year: int, expected: int, tz_aware_fixture) -> None:
def test_slice_datetime_locs(box: Union[datetime, Timestamp], tz_aware_fixture) -> None:
def test_indexer_between_time() -> None:
def test_indexer_between_time_non_nano(unit: str) -> None:
