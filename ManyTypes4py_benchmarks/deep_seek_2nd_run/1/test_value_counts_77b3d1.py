"""
these are systematically testing all of the args to value_counts
with different size combinations. This is to ensure stability of the sorting
and proper parameter handling
"""
import numpy as np
import pytest
from pandas import Categorical, CategoricalIndex, DataFrame, Grouper, Index, MultiIndex, Series, date_range, to_datetime
import pandas._testing as tm
from pandas.util.version import Version
from typing import Any, List, Tuple, Union, Optional, Dict, cast

def tests_value_counts_index_names_category_column() -> None:
    df = DataFrame({'gender': ['female'], 'country': ['US']})
    df['gender'] = df['gender'].astype('category')
    result = df.groupby('country')['gender'].value_counts()
    df_mi_expected = DataFrame([['US', 'female']], columns=['country', 'gender'])
    df_mi_expected['gender'] = df_mi_expected['gender'].astype('category')
    mi_expected = MultiIndex.from_frame(df_mi_expected)
    expected = Series([1], index=mi_expected, name='count')
    tm.assert_series_equal(result, expected)

def seed_df(seed_nans: bool, n: int, m: int) -> DataFrame:
    days = date_range('2015-08-24', periods=10)
    frame = DataFrame({'1st': np.random.default_rng(2).choice(list('abcd'), n), '2nd': np.random.default_rng(2).choice(days, n), '3rd': np.random.default_rng(2).integers(1, m + 1, n)})
    if seed_nans:
        frame['3rd'] = frame['3rd'].astype('float')
        frame.loc[1::11, '1st'] = np.nan
        frame.loc[3::17, '2nd'] = np.nan
        frame.loc[7::19, '3rd'] = np.nan
        frame.loc[8::19, '3rd'] = np.nan
        frame.loc[9::19, '3rd'] = np.nan
    return frame

@pytest.mark.slow
@pytest.mark.parametrize('seed_nans', [True, False])
@pytest.mark.parametrize('num_rows', [10, 50])
@pytest.mark.parametrize('max_int', [5, 20])
@pytest.mark.parametrize('keys', ['1st', '2nd', ['1st', '2nd']], ids=repr)
@pytest.mark.parametrize('bins', [None, [0, 5]], ids=repr)
@pytest.mark.parametrize('isort', [True, False])
@pytest.mark.parametrize('normalize, name', [(True, 'proportion'), (False, 'count')])
def test_series_groupby_value_counts(seed_nans: bool, num_rows: int, max_int: int, keys: Union[str, List[str]], bins: Optional[List[int]], isort: bool, normalize: bool, name: str, sort: bool, ascending: Optional[bool], dropna: bool) -> None:
    df = seed_df(seed_nans, num_rows, max_int)

    def rebuild_index(df: DataFrame) -> DataFrame:
        arr = list(map(df.index.get_level_values, range(df.index.nlevels)))
        df.index = MultiIndex.from_arrays(arr, names=df.index.names)
        return df
    kwargs = {'normalize': normalize, 'sort': sort, 'ascending': ascending, 'dropna': dropna, 'bins': bins}
    gr = df.groupby(keys, sort=isort)
    left = gr['3rd'].value_counts(**kwargs)
    gr = df.groupby(keys, sort=isort)
    right = gr['3rd'].apply(Series.value_counts, **kwargs)
    right.index.names = right.index.names[:-1] + ['3rd']
    right = right.rename(name)
    left, right = map(rebuild_index, (left, right))
    tm.assert_series_equal(left.sort_index(), right.sort_index())

@pytest.mark.parametrize('utc', [True, False])
def test_series_groupby_value_counts_with_grouper(utc: bool) -> None:
    df = DataFrame({'Timestamp': [1565083561, 1565083561 + 86400, 1565083561 + 86500, 1565083561 + 86400 * 2, 1565083561 + 86400 * 3, 1565083561 + 86500 * 3, 1565083561 + 86400 * 4], 'Food': ['apple', 'apple', 'banana', 'banana', 'orange', 'orange', 'pear']}).drop([3])
    df['Datetime'] = to_datetime(df['Timestamp'], utc=utc, unit='s')
    dfg = df.groupby(Grouper(freq='1D', key='Datetime'))
    result = dfg['Food'].value_counts().sort_index()
    expected = dfg['Food'].apply(Series.value_counts).sort_index()
    expected.index.names = result.index.names
    expected = expected.rename('count')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('columns', [['A', 'B'], ['A', 'B', 'C']])
def test_series_groupby_value_counts_empty(columns: List[str]) -> None:
    df = DataFrame(columns=columns)
    dfg = df.groupby(columns[:-1])
    result = dfg[columns[-1]].value_counts()
    expected = Series([], dtype=result.dtype, name='count')
    expected.index = MultiIndex.from_arrays([[]] * len(columns), names=columns)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('columns', [['A', 'B'], ['A', 'B', 'C']])
def test_series_groupby_value_counts_one_row(columns: List[str]) -> None:
    df = DataFrame(data=[range(len(columns))], columns=columns)
    dfg = df.groupby(columns[:-1])
    result = dfg[columns[-1]].value_counts()
    expected = df.value_counts()
    tm.assert_series_equal(result, expected)

def test_series_groupby_value_counts_on_categorical() -> None:
    s = Series(Categorical(['a'], categories=['a', 'b']))
    result = s.groupby([0]).value_counts()
    expected = Series(data=[1, 0], index=MultiIndex.from_arrays([np.array([0, 0]), CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=False, dtype='category')]), name='count')
    tm.assert_series_equal(result, expected)

def test_series_groupby_value_counts_no_sort() -> None:
    df = DataFrame({'gender': ['male', 'male', 'female', 'male', 'female', 'male'], 'education': ['low', 'medium', 'high', 'low', 'high', 'low'], 'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']})
    gb = df.groupby(['country', 'gender'], sort=False)['education']
    result = gb.value_counts(sort=False)
    index = MultiIndex(levels=[['US', 'FR'], ['male', 'female'], ['low', 'medium', 'high']], codes=[[0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 1, 2, 0, 2]], names=['country', 'gender', 'education'])
    expected = Series([1, 1, 1, 2, 1], index=index, name='count')
    tm.assert_series_equal(result, expected)

@pytest.fixture
def education_df() -> DataFrame:
    return DataFrame({'gender': ['male', 'male', 'female', 'male', 'female', 'male'], 'education': ['low', 'medium', 'high', 'low', 'high', 'low'], 'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']})

def test_bad_subset(education_df: DataFrame) -> None:
    gp = education_df.groupby('country')
    with pytest.raises(ValueError, match='subset'):
        gp.value_counts(subset=['country'])

def test_basic(education_df: DataFrame, request: Any) -> None:
    if Version(np.__version__) >= Version('1.25'):
        request.applymarker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
    result = education_df.groupby('country')[['gender', 'education']].value_counts(normalize=True)
    expected = Series(data=[0.5, 0.25, 0.25, 0.5, 0.5], index=MultiIndex.from_tuples([('FR', 'male', 'low'), ('FR', 'male', 'medium'), ('FR', 'female', 'high'), ('US', 'male', 'low'), ('US', 'female', 'high')], names=['country', 'gender', 'education']), name='proportion')
    tm.assert_series_equal(result, expected)

def _frame_value_counts(df: DataFrame, keys: List[str], normalize: bool, sort: bool, ascending: Optional[bool]) -> Series:
    return df[keys].value_counts(normalize=normalize, sort=sort, ascending=ascending)

@pytest.mark.parametrize('groupby', ['column', 'array', 'function'])
@pytest.mark.parametrize('normalize, name', [(True, 'proportion'), (False, 'count')])
@pytest.mark.parametrize('sort, ascending', [(False, None), (True, True), (True, False)])
@pytest.mark.parametrize('frame', [True, False])
def test_against_frame_and_seriesgroupby(education_df: DataFrame, groupby: str, normalize: bool, name: str, sort: bool, ascending: Optional[bool], as_index: bool, frame: bool, request: Any, using_infer_string: bool) -> None:
    if Version(np.__version__) >= Version('1.25') and frame and sort and normalize:
        request.applymarker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
    by = {'column': 'country', 'array': education_df['country'].values, 'function': lambda x: education_df['country'][x] == 'US'}[groupby]
    gp = education_df.groupby(by=by, as_index=as_index)
    result = gp[['gender', 'education']].value_counts(normalize=normalize, sort=sort, ascending=ascending)
    if frame:
        expected = gp.apply(_frame_value_counts, ['gender', 'education'], normalize, sort, ascending)
        if as_index:
            tm.assert_series_equal(result, expected)
        else:
            name = 'proportion' if normalize else 'count'
            expected = expected.reset_index().rename({0: name}, axis=1)
            if groupby in ['array', 'function'] and (not as_index and frame):
                expected.insert(loc=0, column='level_0', value=result['level_0'])
            else:
                expected.insert(loc=0, column='country', value=result['country'])
            tm.assert_frame_equal(result, expected)
    else:
        education_df['both'] = education_df['gender'] + '-' + education_df['education']
        expected = gp['both'].value_counts(normalize=normalize, sort=sort, ascending=ascending)
        expected.name = name
        if as_index:
            index_frame = expected.index.to_frame(index=False)
            index_frame['gender'] = index_frame['both'].str.split('-').str.get(0)
            index_frame['education'] = index_frame['both'].str.split('-').str.get(1)
            del index_frame['both']
            index_frame2 = index_frame.rename({0: None}, axis=1)
            expected.index = MultiIndex.from_frame(index_frame2)
            if index_frame2.columns.isna()[0]:
                expected.index.names = [None] + expected.index.names[1:]
            tm.assert_series_equal(result, expected)
        else:
            expected.insert(1, 'gender', expected['both'].str.split('-').str.get(0))
            expected.insert(2, 'education', expected['both'].str.split('-').str.get(1))
            if using_infer_string:
                expected = expected.astype({'gender': 'str', 'education': 'str'})
            del expected['both']
            tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('sort, ascending, expected_rows, expected_count, expected_group_size', [(False, None, [0, 1, 2, 3, 4], [1, 1, 1, 2, 1], [1, 3, 1, 3, 1]), (True, False, [3, 0, 1, 2, 4], [2, 1, 1, 1, 1], [3, 1, 3, 1, 1]), (True, True, [0, 1, 2, 4, 3], [1, 1, 1, 1, 2], [1, 3, 1, 1, 3])])
def test_compound(education_df: DataFrame, normalize: bool, sort: bool, ascending: Optional[bool], expected_rows: List[int], expected_count: List[int], expected_group_size: List[int], any_string_dtype: str, using_infer_string: bool) -> None:
    dtype = any_string_dtype
    education_df = education_df.astype(dtype)
    education_df.columns = education_df.columns.astype(dtype)
    gp = education_df.groupby(['country', 'gender'], as_index=False, sort=False)
    result = gp['education'].value_counts(normalize=normalize, sort=sort, ascending=ascending)
    expected = DataFrame()
    for column in ['country', 'gender', 'education']:
        expected[column] = [education_df[column][row] for row in expected_rows]
        expected = expected.astype(dtype)
        expected.columns = expected.columns.astype(dtype)
    if normalize:
        expected['proportion'] = expected_count
        expected['proportion'] /= expected_group_size
        if dtype == 'string[pyarrow]':
            expected['proportion'] = expected['proportion'].convert_dtypes()
    else:
        expected['count'] = expected_count
        if dtype == 'string[pyarrow]':
            expected['count'] = expected['count'].convert_dtypes()
    if using_infer_string and dtype == object:
        expected = expected.astype({'country': 'str', 'gender': 'str', 'education': 'str'})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('sort, ascending, normalize, name, expected_data, expected_index', [(False, None, False, 'count', [1, 2, 1], [(1, 1, 1), (2, 4, 6), (2, 0, 0)]), (True, True, False, 'count', [1, 1, 2], [(1, 1, 1), (2, 6, 4), (2, 0, 0)]), (True, False, False, 'count', [2, 1, 1], [(1, 1, 1), (4, 2, 6), (0, 2, 0)]), (True, False, True, 'proportion', [0.5, 0.25, 0.25], [(1, 1, 1), (4, 2, 6), (0, 2, 0)])])
def test_data_frame_value_counts(sort: bool, ascending: Optional[bool], normalize: bool, name: str, expected_data: List[int], expected_index: List[Tuple[int, int, int]]) -> None:
    animals_df = DataFrame({'key': [1, 1, 1, 1], 'num_legs': [2, 4, 4, 6], 'num_wings': [2, 0, 0, 0]}, index=['falcon', 'dog', 'cat', 'ant'])
    result_frame = animals_df.value_counts(sort=sort, ascending=ascending, normalize=normalize)
    expected = Series(data=expected_data, index=MultiIndex.from_arrays(expected_index, names=['key', 'num_legs', 'num_wings']), name=name)
    tm.assert_series_equal(result_frame, expected)
    result_frame_groupby = animals_df.groupby('key').value_counts(sort=sort, ascending=ascending, normalize=normalize)
    tm.assert_series_equal(result_frame_groupby, expected)

@pytest.mark.parametrize('group_dropna, count_dropna, expected_rows, expected_values', [(False, False, [0, 1, 3, 5, 6, 7, 8, 2, 4], [0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0]), (False, True, [0, 1, 3, 5, 2, 4], [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]), (True, False, [0, 1, 5, 6, 7, 8], [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]), (True, True, [0, 1, 5], [0.5, 0.5, 1.0])])
def test_dropna_combinations(group_dropna: bool, count_dropna: bool, expected_rows: List[int], expected_values: List[float], request: Any) -> None:
