import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Series
import pandas._testing as tm
from typing import List, Tuple

@pytest.fixture
def m() -> int:
    return 5

@pytest.fixture
def n() -> int:
    return 100

@pytest.fixture
def cols() -> List[str]:
    return ['jim', 'joe', 'jolie', 'joline', 'jolia']

@pytest.fixture
def vals(n: int) -> List[Tuple[int, str, pd.Timestamp, str, float]]:
    rng = np.random.default_rng(2)
    vals = [
        rng.integers(0, 10, n),
        rng.choice(list('abcdefghij'), n),
        rng.choice(pd.date_range('20141009', periods=10).tolist(), n),
        rng.choice(list('ZYXWVUTSRQ'), n),
        rng.standard_normal(n)
    ]
    vals = list(map(tuple, zip(*vals)))
    return vals

@pytest.fixture
def keys(n: int, m: int, vals: List[Tuple[int, str, pd.Timestamp, str, float]]) -> List[Tuple[int, str, pd.Timestamp, str]]:
    rng = np.random.default_rng(2)
    keys_part = [
        rng.integers(0, 11, m),
        rng.choice(list('abcdefghijk'), m),
        rng.choice(pd.date_range('20141009', periods=11).tolist(), m),
        rng.choice(list('ZYXWVUTSRQP'), m)
    ]
    keys = list(map(tuple, zip(*keys_part)))
    keys += [t[:-1] for t in vals[::n // m]]
    return keys

@pytest.fixture
def df(vals: List[Tuple[int, str, pd.Timestamp, str, float]], cols: List[str]) -> DataFrame:
    return DataFrame(vals, columns=cols)

@pytest.fixture
def a(df: DataFrame) -> DataFrame:
    return pd.concat([df, df])

@pytest.fixture
def b(df: DataFrame, cols: List[str]) -> DataFrame:
    return df.drop_duplicates(subset=cols[:-1])

@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
@pytest.mark.parametrize('lexsort_depth', list(range(5)))
@pytest.mark.parametrize('frame_fixture', ['a', 'b'])
def test_multiindex_get_loc(
    request: pytest.FixtureRequest,
    lexsort_depth: int,
    keys: List[Tuple[int, str, pd.Timestamp, str]],
    frame_fixture: str,
    cols: List[str]
) -> None:
    frame = request.getfixturevalue(frame_fixture)
    if lexsort_depth == 0:
        df = frame.copy(deep=False)
    else:
        df = frame.sort_values(by=cols[:lexsort_depth])
    mi = df.set_index(cols[:-1])
    assert not mi.index._lexsort_depth < lexsort_depth
    for key in keys:
        mask: np.ndarray = np.ones(len(df), dtype=bool)
        for i, k in enumerate(key):
            mask &= df.iloc[:, i] == k
            if not mask.any():
                assert key[:i + 1] not in mi.index
                continue
            assert key[:i + 1] in mi.index
            right = df[mask].copy(deep=False)
            if i + 1 != len(key):
                return_value = right.drop(cols[:i + 1], axis=1, inplace=True)
                assert return_value is None
                return_value = right.set_index(cols[i + 1:-1], inplace=True)
                assert return_value is None
                tm.assert_frame_equal(mi.loc[key[:i + 1]], right)
            else:
                return_value = right.set_index(cols[:-1], inplace=True)
                assert return_value is None
                if len(right) == 1:
                    right = Series(right['jolia'].values, name=right.index[0], index=['jolia'])
                    tm.assert_series_equal(mi.loc[key[:i + 1]], right)
                else:
                    tm.assert_frame_equal(mi.loc[key[:i + 1]], right)
