import numpy as np
import pytest
from pandas import DataFrame, Series, Timedelta, concat, date_range
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer

@pytest.fixture(params=['triang', 'blackman', 'hamming', 'bartlett', 'bohman', 'blackmanharris', 'nuttall', 'barthann'])
def win_types(request: str) -> str:
    return request.param

@pytest.fixture(params=['kaiser', 'gaussian', 'general_gaussian', 'exponential'])
def win_types_special(request: str) -> str:
    return request.param

def test_constructor(frame_or_series: Series) -> None:
    pytest.importorskip('scipy')
    c = frame_or_series(range(5)).rolling
    c(win_type='boxcar', window=2, min_periods=1)
    c(win_type='boxcar', window=2, min_periods=1, center=True)
    c(win_type='boxcar', window=2, min_periods=1, center=False)

@pytest.mark.parametrize('w', [2.0, 'foo', np.array([2])])
def test_invalid_constructor(frame_or_series: Series, w: object) -> None:
    pytest.importorskip('scipy')
    c = frame_or_series(range(5)).rolling
    with pytest.raises(ValueError, match='min_periods must be an integer'):
        c(win_type='boxcar', window=2, min_periods=w)
    with pytest.raises(ValueError, match='center must be a boolean'):
        c(win_type='boxcar', window=2, min_periods=1, center=w)

@pytest.mark.parametrize('wt', ['foobar', 1])
def test_invalid_constructor_wintype(frame_or_series: Series, wt: object) -> None:
    pytest.importorskip('scipy')
    c = frame_or_series(range(5)).rolling
    with pytest.raises(ValueError, match='Invalid win_type'):
        c(win_type=wt, window=2)

# ... and so on
