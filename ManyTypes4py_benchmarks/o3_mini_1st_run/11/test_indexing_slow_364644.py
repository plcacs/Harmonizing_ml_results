import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Series
import pandas._testing as tm
from typing import List, Tuple, Any
from pytest import FixtureRequest

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
    col1 = rng.integers(0, 10, n)
    col2 = np.random.default_rng(2).choice(list('abcdefghij'), n)
    col3 = np.random.default_rng(2).choice(pd.date_range('20141009', periods=10).tolist(), n)
    col4 = np.random.default_rng(2).choice(list('ZYXWVUTSRQ'), n)
    col5 = np.random.default_rng(2).standard_normal(n)
    vals_list = [col1, col2, col3, col4, col5]
    vals_tuples = list(map(tuple, zip(*vals_list)))
    return vals_tuples

@pytest.fixture
def keys(n: int, m: int, vals: List[Tuple[int, str, pd.Timestamp, str, float]]) -> List[Tuple[int, str, pd.Timestamp, str]]:
    rng = np.random.default_rng(2)
    key_col1 = rng.integers(0, 11, m)
    key_col2 = np.random.default_rng(2).choice(list('abcdefghijk'), m)
    key_col3 = np.random.default_rng(2).choice(pd.date_range('20141009', periods=11).tolist(), m)
    key_col4 = np.random.default_rng(2).choice(list('ZYXWVUTSRQP'), m)
    keys_list = [key_col1, key_col2, key_col3, key_col4]
    keys_tuples = list(map(tuple, zip(*keys_list)))
    # Append additional keys from vals by dropping the last element of each tuple selected.
    keys_tuples += [t[:-1] for t in vals[:: n // m]]
    return keys_tuples

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
def test_multiindex_get_loc(request: FixtureRequest, lexsort_depth: int, keys: List[Tuple[Any, ...]], frame_fixture: str, cols: List[str]) -> None:
    frame: DataFrame = request.getfixturevalue(frame_fixture)
    if lexsort_depth == 0:
        df_local: DataFrame = frame.copy(deep=False)
    else:
        df_local = frame.sort_values(by=cols[:lexsort_depth])
    mi: DataFrame = df_local.set_index(cols[:-1])
    assert not mi.index._lexsort_depth < lexsort_depth
    for key in keys:
        mask = np.ones(len(df_local), dtype=bool)
        for i, k in enumerate(key):
            mask &= df_local.iloc[:, i] == k
            if not mask.any():
                assert key[:i + 1] not in mi.index
                continue
            assert key[:i + 1] in mi.index
            right: DataFrame = df_local[mask].copy(deep=False)
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
                    right_series: Series = Series(right['jolia'].values, name=right.index[0], index=['jolia'])
                    tm.assert_series_equal(mi.loc[key[:i + 1]], right_series)
                else:
                    tm.assert_frame_equal(mi.loc[key[:i + 1]], right)