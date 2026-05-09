import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

@pytest.fixture(params=[True, False])
def by_blocks_fixture(request) -> bool:
    return request.param

def _assert_frame_equal_both(a: DataFrame, b: DataFrame, **kwargs) -> None:
    """
    Check that two DataFrame equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : DataFrame
        The first DataFrame to compare.
    b : DataFrame
        The second DataFrame to compare.
    kwargs : dict
        The arguments passed to `tm.assert_frame_equal`.
    """
    tm.assert_frame_equal(a, b, **kwargs)
    tm.assert_frame_equal(b, a, **kwargs)

@pytest.mark.parametrize('check_like', [True, False])
def test_frame_equal_row_order_mismatch(check_like: bool, frame_or_series: DataFrame) -> None:
    df1 = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
    df2 = DataFrame({'A': [3, 2, 1], 'B': [6, 5, 4]}, index=['c', 'b', 'a'])
    if not check_like:
        msg = f'{frame_or_series.__name__}.index are different'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, check_like=check_like, obj=frame_or_series.__name__)
    else:
        _assert_frame_equal_both(df1, df2, check_like=check_like, obj=frame_or_series.__name__)

# ... (rest of the code remains the same)
