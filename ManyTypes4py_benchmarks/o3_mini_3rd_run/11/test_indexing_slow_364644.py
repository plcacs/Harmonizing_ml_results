from typing import List, Tuple, Any
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Series
import pandas._testing as tm

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
    ints = rng.integers(0, 10, n)
    choices1 = np.random.default_rng(2).choice(list('abcdefghij'), n)
    dates = np.random.default_rng(2).choice(pd.date_range('20141009', periods=10).tolist(), n)
    choices2 = np.random.default_rng(2).choice(list('ZYXWVUTSRQ'), n)
    normals = np.random.default_rng(2).standard_normal(n)
    vals_list = [ints, choices1, dates, choices2, normals]
    vals_tuples = list(map(tuple, zip(*vals_list)))
    return vals_tuples

@pytest.fixture
def keys(n: int, m: int, vals: List[Tuple[int, str, pd.Timestamp, str, float]]) -> List[Tuple[int, str, pd.Timestamp, str]]:
    rng_int = np.random.default_rng(2)
    rng_str = np.random.default_rng(2)
    rng_date = np.random.default_rng(2)
    rng_char = np.random.default_rng(2)
    key_parts = [
        rng_int.integers(0, 11, m),
        np.random.default_rng(2).choice(list('abcdefghijk'), m),
        np.random.default_rng(2).choice(pd.date_range('20141009', periods=11).tolist(), m),
        np.random.default_rng(2).choice(list('ZYXWVUTSRQP'), m)
    ]
    keys_list = list(map(tuple, zip(*key_parts)))
    # Append additional keys from vals
    extra_keys = [t[:-1] for t in vals[:: n // m]]
    return keys_list + extra_keys

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
def test_multiindex_get_loc(request: pytest.FixtureRequest, lexsort_depth: int, keys: List[Tuple[int, str, pd.Timestamp, str]], 
                              frame_fixture: str, cols: List[str]) -> None:
    frame: DataFrame = request.getfixturevalue(frame_fixture)
    if lexsort_depth == 0:
        df_local: DataFrame = frame.copy(deep=False)
    else:
        df_local = frame.sort_values(by=cols[:lexsort_depth])
    mi: DataFrame = df_local.set_index(cols[:-1])
    assert not mi.index._lexsort_depth < lexsort_depth
    for key in keys:
        mask: Any = np.ones(len(df_local), dtype=bool)
        for i, k in enumerate(key):
            mask &= df_local.iloc[:, i] == k
            if not mask.any():
                assert key[:i + 1] not in mi.index
                continue
            assert key[:i + 1] in mi.index
            right: DataFrame = df_local[mask].copy(deep=False)
            if i + 1 != len(key):
                ret_val = right.drop(cols[:i + 1], axis=1, inplace=True)
                assert ret_val is None
                ret_val = right.set_index(cols[i + 1:-1], inplace=True)
                assert ret_val is None
                tm.assert_frame_equal(mi.loc[key[:i + 1]], right)
            else:
                ret_val = right.set_index(cols[:-1], inplace=True)
                assert ret_val is None
                if len(right) == 1:
                    series_result: Series = Series(right['jolia'].values, name=right.index[0], index=['jolia'])
                    tm.assert_series_equal(mi.loc[key[:i + 1]], series_result)
                else:
                    tm.assert_frame_equal(mi.loc[key[:i + 1]], right)