from typing import Any, Tuple
import numpy as np
import pytest
from pandas import Interval, IntervalIndex, Timedelta, Timestamp
import pandas._testing as tm
from pandas.core.arrays import IntervalArray

def start_shift(request: Any) -> Tuple[Any, Any]:
    ...

def test_overlaps_interval(self, constructor: Any, start_shift: Tuple[Any, Any], closed: Any, other_closed: Any) -> None:
    ...

def test_overlaps_interval_container(self, constructor: Any, other_constructor: Any) -> None:
    ...

def test_overlaps_na(self, constructor: Any, start_shift: Tuple[Any, Any]) -> None:
    ...

def test_overlaps_invalid_type(self, constructor: Any, other: Any) -> None:
    ...
