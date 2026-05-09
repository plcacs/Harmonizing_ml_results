import numpy as np
import pytest
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, isna, notna
import pandas._testing as tm

def test_doc_string() -> None:
    df = DataFrame({'B': [0, 1, 2, np.nan, 4]})
    df
    df.expanding(2).sum()

def test_constructor(frame_or_series: DataFrame | Series) -> None:
    c = frame_or_series(range(5)).expanding
    c(min_periods=1)

@pytest.mark.parametrize('w', [2.0, 'foo', np.array([2])])
def test_constructor_invalid(frame_or_series: DataFrame | Series, w: object) -> None:
    c = frame_or_series(range(5)).expanding
    msg = 'min_periods must be an integer'
    with pytest.raises(ValueError, match=msg):
        c(min_periods=w)

# ... (rest of the code remains the same)
