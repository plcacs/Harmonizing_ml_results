#!/usr/bin/env python3
"""
these are systematically testing all of the args to value_counts
with different size combinations. This is to ensure stability of the sorting
and proper parameter handling
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas import Categorical, CategoricalIndex, DataFrame, Grouper, Index, MultiIndex, Series, date_range, to_datetime
from pandas.util.version import Version
import pandas._testing as tm


def tests_value_counts_index_names_category_column() -> None:
    df: DataFrame = DataFrame({'gender': ['female'], 'country': ['US']})
    df['gender'] = df['gender'].astype('category')
    result: Series = df.groupby('country')['gender'].value_counts()
    df_mi_expected: DataFrame = DataFrame([['US', 'female']], columns=['country', 'gender'])
    df_mi_expected['gender'] = df_mi_expected['gender'].astype('category')
    mi_expected: MultiIndex = MultiIndex.from_frame(df_mi_expected)
    expected: Series = Series([1], index=mi_expected, name='count')
    tm.assert_series_equal(result, expected)


def seed_df(seed_nans: bool, n: int, m: int) -> DataFrame:
    days = date_range('2015-08-24', periods=10)
    frame: DataFrame = DataFrame({
        '1st': np.random.default_rng(2).choice(list('abcd'), n),
        '2nd': np.random.default_rng(2).choice(days, n),
        '3rd': np.random.default_rng(2).integers(1, m + 1, n)
    })
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
@pytest.mark.parametrize('sort', [True, False])
@pytest.mark.parametrize('ascending', [True, None])
@pytest.mark.parametrize('dropna', [True, False])
def test_series_groupby_value_counts(seed_nans: bool, num_rows: int, max_int: int, keys: Union[str, List[str]],
                                     bins: Optional[List[int]], isort: bool, normalize: bool, name: str,
                                     sort: bool, ascending: Optional[bool], dropna: bool) -> None:
    df: DataFrame = seed_df(seed_nans, num_rows, max_int)

    def rebuild_index(df_inner: DataFrame) -> DataFrame:
        arr = list(map(df_inner.index.get_level_values, range(df_inner.index.nlevels)))
        df_inner.index = MultiIndex.from_arrays(arr, names=df_inner.index.names)
        return df_inner

    kwargs: dict[str, Any] = {'normalize': normalize, 'sort': sort, 'ascending': ascending, 'dropna': dropna, 'bins': bins}
    gr = df.groupby(keys, sort=isort)
    left: Series = gr['3rd'].value_counts(**kwargs)
    gr = df.groupby(keys, sort=isort)
    right: Series = gr['3rd'].apply(Series.value_counts, **kwargs)
    right.index.names = right.index.names[:-1] + ['3rd']
    right = right.rename(name)
    left = rebuild_index(left)
    right = rebuild_index(right)
    tm.assert_series_equal(left.sort_index(), right.sort_index())


@pytest.mark.parametrize('utc', [True, False])
def test_series_groupby_value_counts_with_grouper(utc: bool) -> None:
    df: DataFrame = DataFrame({
        'Timestamp': [1565083561, 1565083561 + 86400, 1565083561 + 86500,
                      1565083561 + 86400 * 2, 1565083561 + 86400 * 3, 1565083561 + 86500 * 3,
                      1565083561 + 86400 * 4],
        'Food': ['apple', 'apple', 'banana', 'banana', 'orange', 'orange', 'pear']
    }).drop([3])
    df['Datetime'] = to_datetime(df['Timestamp'], utc=utc, unit='s')
    dfg = df.groupby(Grouper(freq='1D', key='Datetime'))
    result: Series = dfg['Food'].value_counts().sort_index()
    expected: Series = dfg['Food'].apply(Series.value_counts).sort_index()
    expected.index.names = result.index.names
    expected = expected.rename('count')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('columns', [['A', 'B'], ['A', 'B', 'C']])
def test_series_groupby_value_counts_empty(columns: List[str]) -> None:
    df: DataFrame = DataFrame(columns=columns)
    dfg = df.groupby(columns[:-1])
    result: Series = dfg[columns[-1]].value_counts()
    expected: Series = Series([], dtype=result.dtype, name='count')
    expected.index = MultiIndex.from_arrays([[]] * len(columns), names=columns)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('columns', [['A', 'B'], ['A', 'B', 'C']])
def test_series_groupby_value_counts_one_row(columns: List[str]) -> None:
    df: DataFrame = DataFrame(data=[list(range(len(columns)))], columns=columns)
    dfg = df.groupby(columns[:-1])
    result: Series = dfg[columns[-1]].value_counts()
    expected: Series = df.value_counts()
    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_on_categorical() -> None:
    s: Series = Series(Categorical(['a'], categories=['a', 'b']))
    result: Series = s.groupby([0]).value_counts()
    expected: Series = Series(
        data=[1, 0],
        index=MultiIndex.from_arrays([
            np.array([0, 0]),
            CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=False, dtype='category')
        ]),
        name='count'
    )
    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_no_sort() -> None:
    df: DataFrame = DataFrame({
        'gender': ['male', 'male', 'female', 'male', 'female', 'male'],
        'education': ['low', 'medium', 'high', 'low', 'high', 'low'],
        'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']
    })
    gb = df.groupby(['country', 'gender'], sort=False)['education']
    result: Series = gb.value_counts(sort=False)
    index: MultiIndex = MultiIndex(
        levels=[['US', 'FR'], ['male', 'female'], ['low', 'medium', 'high']],
        codes=[[0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 1, 2, 0, 2]],
        names=['country', 'gender', 'education']
    )
    expected: Series = Series([1, 1, 1, 2, 1], index=index, name='count')
    tm.assert_series_equal(result, expected)


@pytest.fixture
def education_df() -> DataFrame:
    return DataFrame({
        'gender': ['male', 'male', 'female', 'male', 'female', 'male'],
        'education': ['low', 'medium', 'high', 'low', 'high', 'low'],
        'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']
    })


def test_bad_subset(education_df: DataFrame) -> None:
    gp = education_df.groupby('country')
    with pytest.raises(ValueError, match='subset'):
        gp.value_counts(subset=['country'])


def test_basic(education_df: DataFrame, request: pytest.FixtureRequest) -> None:
    if Version(np.__version__) >= Version('1.25'):
        request.applymarker(pytest.mark.xfail(
            reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions',
            strict=False))
    result: Series = education_df.groupby('country')[['gender', 'education']].value_counts(normalize=True)
    expected: Series = Series(
        data=[0.5, 0.25, 0.25],
        index=MultiIndex.from_tuples(
            [('FR', 'male', 'low'), ('FR', 'male', 'medium'), ('FR', 'female', 'high'),
             ('US', 'male', 'low'), ('US', 'female', 'high')],
            names=['country', 'gender', 'education']
        ),
        name='proportion'
    )
    tm.assert_series_equal(result, expected)


def _frame_value_counts(df: DataFrame, keys: Union[str, List[str]], normalize: bool, sort: bool,
                          ascending: Optional[bool]) -> Series:
    return df[keys].value_counts(normalize=normalize, sort=sort, ascending=ascending)


@pytest.mark.parametrize('groupby', ['column', 'array', 'function'])
@pytest.mark.parametrize('normalize, name', [(True, 'proportion'), (False, 'count')])
@pytest.mark.parametrize('sort, ascending', [(False, None), (True, True), (True, False)])
@pytest.mark.parametrize('frame', [True, False])
def test_against_frame_and_seriesgroupby(education_df: DataFrame, groupby: str, normalize: bool, name: str,
                                           sort: bool, ascending: Optional[bool], as_index: bool, frame: bool,
                                           request: pytest.FixtureRequest, using_infer_string: bool) -> None:
    if Version(np.__version__) >= Version('1.25') and frame and sort and normalize:
        request.applymarker(pytest.mark.xfail(
            reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions',
            strict=False))
    by: Union[str, Any] = {
        'column': 'country',
        'array': education_df['country'].values,
        'function': lambda x: education_df['country'][x] == 'US'
    }[groupby]
    gp = education_df.groupby(by=by, as_index=as_index)
    result: Any = gp[['gender', 'education']].value_counts(normalize=normalize, sort=sort, ascending=ascending)
    if frame:
        expected: Any = gp.apply(_frame_value_counts, ['gender', 'education'], normalize, sort, ascending)
        if as_index:
            tm.assert_series_equal(result, expected)
        else:
            name_ = 'proportion' if normalize else 'count'
            expected = expected.reset_index().rename({0: name_}, axis=1)
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
@pytest.mark.parametrize('sort, ascending, expected_rows, expected_count, expected_group_size',
                         [(False, None, [0, 1, 2, 3, 4], [1, 1, 1, 2, 1], [1, 3, 1, 3, 1]),
                          (True, False, [3, 0, 1, 2, 4], [2, 1, 1, 1, 1], [3, 1, 3, 1, 1]),
                          (True, True, [0, 1, 2, 4, 3], [1, 1, 1, 1, 2], [1, 3, 1, 1, 3])])
def test_compound(education_df: DataFrame, normalize: bool, sort: bool, ascending: Optional[bool],
                  expected_rows: List[int], expected_count: List[int], expected_group_size: List[int],
                  any_string_dtype: str, using_infer_string: bool) -> None:
    dtype: str = any_string_dtype
    education_df = education_df.astype(dtype)
    education_df.columns = education_df.columns.astype(dtype)
    gp = education_df.groupby(['country', 'gender'], as_index=False, sort=False)
    result: DataFrame = gp['education'].value_counts(normalize=normalize, sort=sort, ascending=ascending)
    expected: DataFrame = DataFrame()
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


@pytest.mark.parametrize('sort, ascending, normalize, name, expected_data, expected_index',
                         [(False, None, False, 'count', [1, 2, 1], [(1, 1, 1), (2, 4, 6), (2, 0, 0)]),
                          (True, True, False, 'count', [1, 1, 2], [(1, 1, 1), (2, 6, 4), (2, 0, 0)]),
                          (True, False, False, 'count', [2, 1, 1], [(1, 1, 1), (4, 2, 6), (0, 2, 0)]),
                          (True, False, True, 'proportion', [0.5, 0.25, 0.25], [(1, 1, 1), (4, 2, 6), (0, 2, 0)])])
def test_data_frame_value_counts(sort: bool, ascending: Optional[bool], normalize: bool, name: str,
                                   expected_data: List[Union[int, float]],
                                   expected_index: List[Tuple[Any, ...]]) -> None:
    animals_df: DataFrame = DataFrame({
        'key': [1, 1, 1, 1],
        'num_legs': [2, 4, 4, 6],
        'num_wings': [2, 0, 0, 0]
    }, index=['falcon', 'dog', 'cat', 'ant'])
    result_frame: Series = animals_df.value_counts(sort=sort, ascending=ascending, normalize=normalize)
    expected: Series = Series(
        data=expected_data,
        index=MultiIndex.from_arrays(expected_index, names=['key', 'num_legs', 'num_wings']),
        name=name
    )
    tm.assert_series_equal(result_frame, expected)
    result_frame_groupby: Series = animals_df.groupby('key').value_counts(sort=sort, ascending=ascending, normalize=normalize)
    tm.assert_series_equal(result_frame_groupby, expected)


@pytest.mark.parametrize('group_dropna, count_dropna, expected_rows, expected_values',
                         [(False, False, [0, 1, 3, 5, 6, 7, 8, 2, 4], [0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0]),
                          (False, True, [0, 1, 3, 5, 2, 4], [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
                          (True, False, [0, 1, 5, 6, 7, 8], [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]),
                          (True, True, [0, 1, 5], [0.5, 0.5, 1.0])])
def test_dropna_combinations(group_dropna: bool, count_dropna: bool, expected_rows: List[int],
                             expected_values: List[float], request: pytest.FixtureRequest) -> None:
    if Version(np.__version__) >= Version('1.25') and (not group_dropna):
        request.applymarker(pytest.mark.xfail(
            reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions',
            strict=False))
    nulls_df: DataFrame = DataFrame({
        'A': [1, 1, np.nan, 4, np.nan, 6, 6, 6, 6],
        'B': [1, 1, 3, np.nan, np.nan, 6, 6, 6, 6],
        'C': [1, 2, 3, 4, 5, 6, np.nan, 8, np.nan],
        'D': [1, 2, 3, 4, 5, 6, 7, np.nan, np.nan]
    })
    gp = nulls_df.groupby(['A', 'B'], dropna=group_dropna)
    result: Series = gp.value_counts(normalize=True, sort=True, dropna=count_dropna)
    columns: DataFrame = DataFrame()
    for column in nulls_df.columns:
        columns[column] = [nulls_df[column][row] for row in expected_rows]
    index: MultiIndex = MultiIndex.from_frame(columns)
    expected: Series = Series(data=expected_values, index=index, name='proportion')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('dropna, expected_data, expected_index',
                         [(True, [1, 1],
                           MultiIndex.from_arrays([(1, 1), ('John', 'Beth'), ('Smith', 'Louise')],
                                                   names=['key', 'first_name', 'middle_name'])),
                          (False, [1, 1, 1, 1],
                           MultiIndex(
                               levels=[Index([1]), Index(['Anne', 'Beth', 'John']),
                                       Index(['Louise', 'Smith', np.nan])],
                               codes=[[0, 0, 0, 0], [2, 0, 2, 1], [1, 2, 2, 0]],
                               names=['key', 'first_name', 'middle_name']))])
@pytest.mark.parametrize('normalize, name', [(False, 'count'), (True, 'proportion')])
def test_data_frame_value_counts_dropna(nulls_fixture: Any, dropna: bool, normalize: bool, name: str,
                                          expected_data: List[int], expected_index: MultiIndex) -> None:
    names_with_nulls_df: DataFrame = DataFrame({
        'key': [1, 1, 1, 1],
        'first_name': ['John', 'Anne', 'John', 'Beth'],
        'middle_name': ['Smith', nulls_fixture, nulls_fixture, 'Louise']
    })
    result_frame: Series = names_with_nulls_df.value_counts(dropna=dropna, normalize=normalize)
    expected: Series = Series(data=expected_data, index=expected_index, name=name)
    if normalize:
        expected = expected / float(len(expected_data))
    tm.assert_series_equal(result_frame, expected)
    result_frame_groupby: Series = names_with_nulls_df.groupby('key').value_counts(dropna=dropna, normalize=normalize)
    tm.assert_series_equal(result_frame_groupby, expected)


@pytest.mark.parametrize('observed', [False, True])
@pytest.mark.parametrize('normalize, name, expected_data',
                         [(False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
                          (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))])
def test_categorical_single_grouper_with_only_observed_categories(education_df: DataFrame, as_index: bool,
                                                                    observed: bool, normalize: bool, name: str,
                                                                    expected_data: np.ndarray,
                                                                    request: pytest.FixtureRequest) -> None:
    if Version(np.__version__) >= Version('1.25'):
        request.applymarker(pytest.mark.xfail(
            reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions',
            strict=False))
    gp = education_df.astype('category').groupby('country', as_index=as_index, observed=observed)
    result: Any = gp.value_counts(normalize=normalize)
    expected_index: MultiIndex = MultiIndex.from_tuples([
        ('FR', 'male', 'low'),
        ('FR', 'male', 'medium'),
        ('FR', 'female', 'high'),
        ('FR', 'male', 'high'),
        ('FR', 'female', 'low'),
        ('FR', 'female', 'medium'),
        ('US', 'male', 'low'),
        ('US', 'female', 'high'),
        ('US', 'male', 'medium'),
        ('US', 'male', 'high'),
        ('US', 'female', 'low'),
        ('US', 'female', 'medium')
    ], names=['country', 'gender', 'education'])
    expected_series: Series = Series(data=expected_data, index=expected_index, name=name)
    for i in range(3):
        expected_series.index = expected_series.index.set_levels(CategoricalIndex(expected_series.index.levels[i]), level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name='proportion' if normalize else 'count')
        tm.assert_frame_equal(result, expected)


def assert_categorical_single_grouper(education_df: DataFrame, as_index: bool, observed: bool,
                                        expected_index: List[Tuple[Any, Any, Any]], normalize: bool, name: str,
                                        expected_data: np.ndarray) -> None:
    education_df = education_df.copy().astype('category')
    education_df['country'] = education_df['country'].cat.add_categories(['ASIA'])
    gp = education_df.groupby('country', as_index=as_index, observed=observed)
    result: Any = gp.value_counts(normalize=normalize)
    expected_series: Series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(expected_index, names=['country', 'gender', 'education']),
        name=name
    )
    for i in range(3):
        index_level = CategoricalIndex(expected_series.index.levels[i])
        if i == 0:
            index_level = index_level.set_categories(education_df['country'].cat.categories)
        expected_series.index = expected_series.index.set_levels(index_level, level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected: DataFrame = expected_series.reset_index(name=name)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('normalize, name, expected_data',
                         [(False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
                          (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))])
def test_categorical_single_grouper_observed_true(education_df: DataFrame, as_index: bool, normalize: bool, name: str,
                                                  expected_data: np.ndarray, request: pytest.FixtureRequest) -> None:
    if Version(np.__version__) >= Version('1.25'):
        request.applymarker(pytest.mark.xfail(
            reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions',
            strict=False))
    expected_index: List[Tuple[Any, Any, Any]] = [
        ('FR', 'male', 'low'),
        ('FR', 'male', 'medium'),
        ('FR', 'female', 'high'),
        ('FR', 'male', 'high'),
        ('FR', 'female', 'low'),
        ('FR', 'female', 'medium'),
        ('US', 'male', 'low'),
        ('US', 'female', 'high'),
        ('US', 'male', 'medium'),
        ('US', 'male', 'high'),
        ('US', 'female', 'low'),
        ('US', 'female', 'medium')
    ]
    assert_categorical_single_grouper(education_df=education_df, as_index=as_index, observed=True,
                                        expected_index=expected_index, normalize=normalize, name=name,
                                        expected_data=expected_data)


@pytest.mark.parametrize('normalize, name, expected_data',
                         [(False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)),
                          (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))])
def test_categorical_single_grouper_observed_false(education_df: DataFrame, as_index: bool, normalize: bool, name: str,
                                                   expected_data: np.ndarray, request: pytest.FixtureRequest) -> None:
    if Version(np.__version__) >= Version('1.25'):
        request.applymarker(pytest.mark.xfail(
            reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions',
            strict=False))
    expected_index: List[Tuple[Any, Any, Any]] = [
        ('FR', 'male', 'low'),
        ('FR', 'male', 'medium'),
        ('FR', 'female', 'high'),
        ('FR', 'male', 'high'),
        ('FR', 'female', 'low'),
        ('FR', 'female', 'medium'),
        ('US', 'male', 'low'),
        ('US', 'female', 'high'),
        ('US', 'male', 'medium'),
        ('US', 'male', 'high'),
        ('US', 'female', 'low'),
        ('US', 'female', 'medium'),
        ('ASIA', 'male', 'low'),
        ('ASIA', 'male', 'medium'),
        ('ASIA', 'male', 'high'),
        ('ASIA', 'female', 'low'),
        ('ASIA', 'female', 'medium'),
        ('ASIA', 'female', 'high')
    ]
    assert_categorical_single_grouper(education_df=education_df, as_index=as_index, observed=False,
                                        expected_index=expected_index, normalize=normalize, name=name,
                                        expected_data=expected_data)


@pytest.mark.parametrize('observed, expected_index',
                         [(False, [
                             ('FR', 'high', 'female'),
                             ('FR', 'high', 'male'),
                             ('FR', 'low', 'male'),
                             ('FR', 'low', 'female'),
                             ('FR', 'medium', 'male'),
                             ('FR', 'medium', 'female'),
                             ('US', 'high', 'female'),
                             ('US', 'high', 'male'),
                             ('US', 'low', 'male'),
                             ('US', 'low', 'female'),
                             ('US', 'medium', 'male'),
                             ('US', 'medium', 'female')
                         ]),
                          (True, [
                              ('FR', 'high', 'female'),
                              ('FR', 'low', 'male'),
                              ('FR', 'medium', 'male'),
                              ('US', 'high', 'female'),
                              ('US', 'low', 'male')
                          ])])
@pytest.mark.parametrize('normalize, name, expected_data',
                         [(False, 'count', np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64)),
                          (True, 'proportion', np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]))])
def test_categorical_multiple_groupers(education_df: DataFrame, as_index: bool, observed: bool,
                                         expected_index: List[Tuple[Any, Any, Any]], normalize: bool, name: str,
                                         expected_data: np.ndarray) -> None:
    education_df = education_df.copy()
    education_df['country'] = education_df['country'].astype('category')
    education_df['education'] = education_df['education'].astype('category')
    gp = education_df.groupby(['country', 'education'], as_index=as_index, observed=observed)
    result: Any = gp.value_counts(normalize=normalize)
    if observed:
        expected_series: Series = Series(
            data=expected_data[expected_data > 0.0] if observed else expected_data,
            index=MultiIndex.from_tuples(expected_index, names=['country', 'education', 'gender']),
            name=name
        )
    else:
        expected_series = Series(
            data=expected_data,
            index=MultiIndex.from_tuples(expected_index, names=['country', 'education', 'gender']),
            name=name
        )
    for i in range(2):
        expected_series.index = expected_series.index.set_levels(CategoricalIndex(expected_series.index.levels[i]), level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected_df: DataFrame = expected_series.reset_index(name='proportion' if normalize else 'count')
        tm.assert_frame_equal(result, expected_df)


@pytest.mark.parametrize('observed', [False, True])
@pytest.mark.parametrize('normalize, name, expected_data',
                         [(False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
                          (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))])
def test_categorical_non_groupers(education_df: DataFrame, as_index: bool, observed: bool,
                                  normalize: bool, name: str, expected_data: np.ndarray,
                                  request: pytest.FixtureRequest) -> None:
    if Version(np.__version__) >= Version('1.25'):
        request.applymarker(pytest.mark.xfail(
            reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions',
            strict=False))
    education_df = education_df.copy()
    education_df['gender'] = education_df['gender'].astype('category')
    education_df['education'] = education_df['education'].astype('category')
    gp = education_df.groupby('country', as_index=as_index, observed=observed)
    result: Any = gp.value_counts(normalize=normalize)
    expected_index: List[Tuple[Any, Any, Any]] = [
        ('FR', 'male', 'low'),
        ('FR', 'male', 'medium'),
        ('FR', 'female', 'high'),
        ('FR', 'male', 'high'),
        ('FR', 'female', 'low'),
        ('FR', 'female', 'medium'),
        ('US', 'male', 'low'),
        ('US', 'female', 'high'),
        ('US', 'male', 'medium'),
        ('US', 'male', 'high'),
        ('US', 'female', 'low'),
        ('US', 'female', 'medium')
    ]
    expected_series: Series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(expected_index, names=['country', 'gender', 'education']),
        name=name
    )
    for i in range(1, 3):
        expected_series.index = expected_series.index.set_levels(CategoricalIndex(expected_series.index.levels[i]), level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected_df = expected_series.reset_index(name='proportion' if normalize else 'count')
        tm.assert_frame_equal(result, expected_df)


@pytest.mark.parametrize('normalize, expected_label, expected_values',
                         [(False, 'count', [1, 1, 1]),
                          (True, 'proportion', [0.5, 0.5, 1.0])])
def test_mixed_groupings(normalize: bool, expected_label: str, expected_values: List[float]) -> None:
    df: DataFrame = DataFrame({'A': [1, 2, 1], 'B': [1, 2, 3]})
    gp = df.groupby([[4, 5, 4], 'A', lambda i: 7 if i == 1 else 8], as_index=False)
    result: DataFrame = gp.value_counts(sort=True, normalize=normalize)
    expected: DataFrame = DataFrame({
        'level_0': np.array([4, 4, 5], dtype=int),
        'A': [1, 1, 2],
        'level_2': [8, 8, 7],
        'B': [1, 3, 2],
        expected_label: expected_values
    })
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('test, columns, expected_names',
                         [('repeat', list('abbde'), ['a', None, 'd', 'b', 'b', 'e']),
                          ('level', list('abcd') + ['level_1'], ['a', None, 'd', 'b', 'c', 'level_1'])])
def test_column_label_duplicates(test: str, columns: List[str], expected_names: List[Optional[str]], as_index: bool) -> None:
    df: DataFrame = DataFrame([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], columns=columns)
    expected_data: List[Tuple[Any, ...]] = [(1, 0, 7, 3, 5, 9), (2, 1, 8, 4, 6, 10)]
    keys: List[Any] = ['a', np.array([0, 1], dtype=np.int64), 'd']
    result: Any = df.groupby(keys, as_index=as_index).value_counts()
    if as_index:
        expected: Series = Series(data=(1, 1), index=MultiIndex.from_tuples(expected_data, names=expected_names), name='count')
        tm.assert_series_equal(result, expected)
    else:
        expected_data_df = [list(row) + [1] for row in expected_data]
        expected_columns: List[str] = list(expected_names)
        expected_columns[1] = 'level_1'
        expected_columns.append('count')
        expected = DataFrame(expected_data_df, columns=expected_columns)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('normalize, expected_label', [(False, 'count'), (True, 'proportion')])
def test_result_label_duplicates(normalize: bool, expected_label: str) -> None:
    gb = DataFrame([[1, 2, 3]], columns=['a', 'b', expected_label]).groupby('a', as_index=False)
    msg: str = f"Column label '{expected_label}' is duplicate of result column"
    with pytest.raises(ValueError, match=msg):
        gb.value_counts(normalize=normalize)


def test_ambiguous_grouping() -> None:
    df: DataFrame = DataFrame({'a': [1, 1]})
    gb = df.groupby(np.array([1, 1], dtype=np.int64))
    result: Series = gb.value_counts()
    expected: Series = Series([2], index=MultiIndex.from_tuples([[1, 1]], names=[None, 'a']), name='count')
    tm.assert_series_equal(result, expected)


def test_subset_overlaps_gb_key_raises() -> None:
    df: DataFrame = DataFrame({'c1': ['a', 'b', 'c'], 'c2': ['x', 'y', 'y']}, index=[0, 1, 1])
    msg: str = "Keys {'c1'} in subset cannot be in the groupby column keys."
    with pytest.raises(ValueError, match=msg):
        df.groupby('c1').value_counts(subset=['c1'])


def test_subset_doesnt_exist_in_frame() -> None:
    df: DataFrame = DataFrame({'c1': ['a', 'b', 'c'], 'c2': ['x', 'y', 'y']}, index=[0, 1, 1])
    msg: str = "Keys {'c3'} in subset do not exist in the DataFrame."
    with pytest.raises(ValueError, match=msg):
        df.groupby('c1').value_counts(subset=['c3'])


def test_subset() -> None:
    df: DataFrame = DataFrame({'c1': ['a', 'b', 'c'], 'c2': ['x', 'y', 'y']}, index=[0, 1, 1])
    result: Series = df.groupby(level=0).value_counts(subset=['c2'])
    expected: Series = Series([1, 2], index=MultiIndex.from_arrays([[0, 1], ['x', 'y']], names=[None, 'c2']), name='count')
    tm.assert_series_equal(result, expected)


def test_subset_duplicate_columns() -> None:
    df: DataFrame = DataFrame([['a', 'x', 'x'], ['b', 'y', 'y'], ['b', 'y', 'y']], index=[0, 1, 1], columns=['c1', 'c2', 'c2'])
    result: Series = df.groupby(level=0).value_counts(subset=['c2'])
    expected: Series = Series([1, 2], index=MultiIndex.from_arrays([[0, 1], ['x', 'y'], ['x', 'y']], names=[None, 'c2', 'c2']), name='count')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('utc', [True, False])
def test_value_counts_time_grouper(utc: bool, unit: str) -> None:
    df: DataFrame = DataFrame({
        'Timestamp': [1565083561, 1565083561 + 86400, 1565083561 + 86500,
                      1565083561 + 86400 * 2, 1565083561 + 86400 * 3, 1565083561 + 86500 * 3,
                      1565083561 + 86400 * 4],
        'Food': ['apple', 'apple', 'banana', 'banana', 'orange', 'orange', 'pear']
    }).drop([3])
    df['Datetime'] = to_datetime(df['Timestamp'], utc=utc, unit='s').dt.as_unit(unit)
    gb = df.groupby(Grouper(freq='1D', key='Datetime'))
    result: Series = gb.value_counts()
    dates = to_datetime(['2019-08-06', '2019-08-07', '2019-08-09', '2019-08-10'], utc=utc).as_unit(unit)
    timestamps = df['Timestamp'].unique()
    index: MultiIndex = MultiIndex(
        levels=[dates, timestamps, ['apple', 'banana', 'orange', 'pear']],
        codes=[[0, 1, 1, 2, 2, 3], list(range(6)), [0, 0, 1, 2, 2, 3]],
        names=['Datetime', 'Timestamp', 'Food']
    )
    expected: Series = Series(1, index=index, name='count')
    tm.assert_series_equal(result, expected)


def test_value_counts_integer_columns() -> None:
    df: DataFrame = DataFrame({
        1: ['a', 'a', 'a'],
        2: ['a', 'a', 'd'],
        3: ['a', 'b', 'c']
    })
    gp = df.groupby([1, 2], as_index=False, sort=False)
    result: DataFrame = gp[3].value_counts()
    expected: DataFrame = DataFrame({1: ['a', 'a', 'a'], 2: ['a', 'a', 'd'], 3: ['a', 'b', 'c'], 'count': 1})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('vc_sort', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
def test_value_counts_sort(sort: bool, vc_sort: bool, normalize: bool) -> None:
    df: DataFrame = DataFrame({'a': [2, 1, 1, 1], 0: [3, 4, 3, 3]})
    gb = df.groupby('a', sort=sort)
    result: Series = gb.value_counts(sort=vc_sort, normalize=normalize)
    if normalize:
        values = [2 / 3, 1 / 3, 1.0]
    else:
        values = [2, 1, 1]
    index: MultiIndex = MultiIndex(
        levels=[[1, 2], [3, 4]],
        codes=[[0, 0, 1], [0, 1, 0]],
        names=['a', 0]
    )
    expected: Series = Series(values, index=index, name='proportion' if normalize else 'count')
    if sort and vc_sort:
        taker = [0, 1, 2]
    elif sort and (not vc_sort):
        taker = [1, 0, 2]
    elif not sort and vc_sort:
        taker = [0, 2, 1]
    else:
        taker = [2, 1, 0]
    expected = expected.take(taker)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('vc_sort', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
def test_value_counts_sort_categorical(sort: bool, vc_sort: bool, normalize: bool) -> None:
    df: DataFrame = DataFrame({'a': [2, 1, 1, 1], 0: [3, 4, 3, 3]}, dtype='category')
    gb = df.groupby('a', sort=sort, observed=True)
    result: Series = gb.value_counts(sort=vc_sort, normalize=normalize)
    if normalize:
        values = [2 / 3, 1 / 3, 1.0, 0.0]
    else:
        values = [2, 1, 1, 0]
    name: str = 'proportion' if normalize else 'count'
    expected: Series = DataFrame({
        'a': Categorical([1, 1, 2, 2]),
        0: Categorical([3, 4, 3, 4]),
        name: values
    }).set_index(['a', 0])[name]
    if sort and vc_sort:
        taker = [0, 1, 2, 3]
    elif sort and (not vc_sort):
        taker = [0, 1, 2, 3]
    elif not sort and vc_sort:
        taker = [0, 2, 1, 3]
    else:
        taker = [2, 1, 0, 3]
    expected = expected.take(taker)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('groupby_sort', [True, False])
def test_value_counts_all_na(sort: bool, dropna: bool, groupby_sort: bool) -> None:
    df: DataFrame = DataFrame({'a': [2, 1, 1], 'b': np.nan})
    gb = df.groupby('a', sort=groupby_sort)
    result: Series = gb.value_counts(sort=sort, dropna=dropna)
    kwargs = {'levels': [[1, 2], [np.nan]], 'names': ['a', 'b']}
    if dropna:
        data: List[int] = []
        index: MultiIndex = MultiIndex(codes=[[], []], **kwargs)
    elif not groupby_sort and (not sort):
        data = [1, 2]
        index = MultiIndex(codes=[[1, 0], [0, 0]], **kwargs)
    else:
        data = [2, 1]
        index = MultiIndex(codes=[[0, 1], [0, 0]], **kwargs)
    expected: Series = Series(data, index=index, dtype='int64', name='count')
    tm.assert_series_equal(result, expected)