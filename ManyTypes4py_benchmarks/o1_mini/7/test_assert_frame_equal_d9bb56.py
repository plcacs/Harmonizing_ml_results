import pytest
import pandas as pd
from pandas import DataFrame, Series
import pandas._testing as tm
from typing import Any, Dict, Tuple, Union, Type

@pytest.fixture(params=[True, False])
def by_blocks_fixture(request: Any) -> bool:
    return request.param

def _assert_frame_equal_both(a: DataFrame, b: DataFrame, **kwargs: Any) -> None:
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
def test_frame_equal_row_order_mismatch(check_like: bool, frame_or_series: Type[Union[DataFrame, Series]]) -> None:
    df1 = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
    df2 = DataFrame({'A': [3, 2, 1], 'B': [6, 5, 4]}, index=['c', 'b', 'a'])
    if not check_like:
        msg: str = f'{frame_or_series.__name__}.index are different'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, check_like=check_like, obj=frame_or_series.__name__)
    else:
        _assert_frame_equal_both(df1, df2, check_like=check_like, obj=frame_or_series.__name__)

@pytest.mark.parametrize(
    'df1,df2',
    [
        ({'A': [1, 2, 3]}, {'A': [1, 2, 3, 4]}),
        ({'A': [1, 2, 3], 'B': [4, 5, 6]}, {'A': [1, 2, 3]})
    ]
)
def test_frame_equal_shape_mismatch(
    df1: Dict[str, Any],
    df2: Dict[str, Any],
    frame_or_series: Type[Union[DataFrame, Series]]
) -> None:
    df1_df: DataFrame = DataFrame(df1)
    df2_df: DataFrame = DataFrame(df2)
    msg: str = f'{frame_or_series.__name__} are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1_df, df2_df, obj=frame_or_series.__name__)

@pytest.mark.parametrize(
    'df1,df2,msg',
    [
        (
            DataFrame.from_records({'a': [1, 2], 'c': ['l1', 'l2']}, index=['a']),
            DataFrame.from_records({'a': [1.0, 2.0], 'c': ['l1', 'l2']}, index=['a']),
            r'DataFrame\.index are different'
        ),
        (
            DataFrame.from_records({'a': [1, 2], 'b': [2.1, 1.5], 'c': ['l1', 'l2']}, index=['a', 'b']),
            DataFrame.from_records({'a': [1.0, 2.0], 'b': [2.1, 1.5], 'c': ['l1', 'l2']}, index=['a', 'b']),
            r'DataFrame\.index level \[0\] are different'
        )
    ]
)
def test_frame_equal_index_dtype_mismatch(
    df1: DataFrame,
    df2: DataFrame,
    msg: str,
    check_index_type: bool
) -> None:
    kwargs: Dict[str, Any] = {'check_index_type': check_index_type}
    if check_index_type:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        tm.assert_frame_equal(df1, df2, **kwargs)

def test_empty_dtypes(check_dtype: bool) -> None:
    columns: Tuple[str, ...] = ('col1', 'col2')
    df1: DataFrame = DataFrame(columns=columns)
    df2: DataFrame = DataFrame(columns=columns)
    kwargs: Dict[str, Any] = {'check_dtype': check_dtype}
    df1['col1'] = df1['col1'].astype('int64')
    if check_dtype:
        msg: str = 'Attributes of DataFrame\\..* are different'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        tm.assert_frame_equal(df1, df2, **kwargs)

@pytest.mark.parametrize('check_like', [True, False])
def test_frame_equal_index_mismatch(
    check_like: bool,
    frame_or_series: Type[Union[DataFrame, Series]],
    using_infer_string: bool
) -> None:
    if using_infer_string:
        dtype: str = 'str'
    else:
        dtype: str = 'object'
    msg: str = (
        f"{frame_or_series.__name__}\\.index are different\n\n"
        f"{frame_or_series.__name__}\\.index values are different \\(33\\.33333 %\\)\n"
        f"\\[left\\]:  Index\\(\\['a', 'b', 'c'\\], dtype='{dtype}'\\)\n"
        f"\\[right\\]: Index\\(\\['a', 'b', 'd'\\], dtype='{dtype}'\\)\n"
        "At positional index 2, first diff: c != d"
    )
    df1: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
    df2: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'd'])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_like=check_like, obj=frame_or_series.__name__)

@pytest.mark.parametrize('check_like', [True, False])
def test_frame_equal_columns_mismatch(
    check_like: bool,
    frame_or_series: Type[Union[DataFrame, Series]],
    using_infer_string: bool
) -> None:
    if using_infer_string:
        dtype: str = 'str'
    else:
        dtype: str = 'object'
    msg: str = (
        f"{frame_or_series.__name__}\\.columns are different\n\n"
        f"{frame_or_series.__name__}\\.columns values are different \\(50\\.0 %\\)\n"
        f"\\[left\\]:  Index\\(\\['A', 'B'\\], dtype='{dtype}'\\)\n"
        f"\\[right\\]: Index\\(\\['A', 'b'\\], dtype='{dtype}'\\)"
    )
    df1: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
    df2: DataFrame = DataFrame({'A': [1, 2, 3], 'b': [4, 5, 6]}, index=['a', 'b', 'c'])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_like=check_like, obj=frame_or_series.__name__)

def test_frame_equal_block_mismatch(
    by_blocks_fixture: bool,
    frame_or_series: Type[Union[DataFrame, Series]]
) -> None:
    obj: str = frame_or_series.__name__
    msg: str = (
        f'{obj}\\.iloc\\[:, 1\\] \\(column name="B"\\) are different\n\n'
        f'{obj}\\.iloc\\[:, 1\\] \\(column name="B"\\) values are different \\(33\\.33333 %\\)\n'
        "\\[index\\]: \\[0, 1, 2\\]\n"
        "\\[left\\]:  \\[4, 5, 6\\]\n"
        "\\[right\\]: \\[4, 5, 7\\]"
    )
    df1: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 7]})
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, by_blocks=by_blocks_fixture, obj=obj)

@pytest.mark.parametrize(
    'df1,df2,msg',
    [
        (
            {'A': ['á', 'à', 'ä'], 'E': ['é', 'è', 'ë']},
            {'A': ['á', 'à', 'ä'], 'E': ['é', 'è', 'e̊']},
            (
                '{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) are different\n\n'
                '{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) values are different \\(33\\.33333 %\\)\n'
                '\\[index\\]: \\[0, 1, 2\\]\n'
                '\\[left\\]:  \\[é, è, ë\\]\n'
                '\\[right\\]: \\[é, è, e̊\\]'
            )
        ),
        (
            {'A': ['á', 'à', 'ä'], 'E': ['é', 'è', 'ë']},
            {'A': ['a', 'a', 'a'], 'E': ['e', 'e', 'e']},
            (
                '{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) are different\n\n'
                '{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) values are different \\(100\\.0 %\\)\n'
                '\\[index\\]: \\[0, 1, 2\\]\n'
                '\\[left\\]:  \\[á, à, ä\\]\n'
                '\\[right\\]: \\[a, a, a\\]'
            )
        )
    ]
)
def test_frame_equal_unicode(
    df1: Dict[str, Any],
    df2: Dict[str, Any],
    msg: str,
    by_blocks_fixture: bool,
    frame_or_series: Type[Union[DataFrame, Series]]
) -> None:
    df1_df: DataFrame = DataFrame(df1)
    df2_df: DataFrame = DataFrame(df2)
    msg_formatted: str = msg.format(obj=frame_or_series.__name__)
    with pytest.raises(AssertionError, match=msg_formatted):
        tm.assert_frame_equal(df1_df, df2_df, by_blocks=by_blocks_fixture, obj=frame_or_series.__name__)

def test_assert_frame_equal_extension_dtype_mismatch() -> None:
    left: DataFrame = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
    right: DataFrame = left.astype(int)
    msg: str = (
        'Attributes of DataFrame\\.iloc\\[:, 0\\] \\(column name="a"\\) are different\n\n'
        'Attribute "dtype" are different\n\\[left\\]:  Int64\n\\[right\\]: int[32|64]'
    )
    tm.assert_frame_equal(left, right, check_dtype=False)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)

def test_assert_frame_equal_interval_dtype_mismatch() -> None:
    left: DataFrame = DataFrame({'a': [pd.Interval(0, 1)]}, dtype='interval')
    right: DataFrame = left.astype(object)
    msg: str = (
        'Attributes of DataFrame\\.iloc\\[:, 0\\] \\(column name="a"\\) are different\n\n'
        'Attribute "dtype" are different\n\\[left\\]:  interval\\[int64, right\\]\n\\[right\\]: object'
    )
    tm.assert_frame_equal(left, right, check_dtype=False)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)

def test_assert_frame_equal_ignore_extension_dtype_mismatch() -> None:
    left: DataFrame = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
    right: DataFrame = DataFrame({'a': [1, 2, 3]}, dtype='Int32')
    tm.assert_frame_equal(left, right, check_dtype=False)

def test_assert_frame_equal_ignore_extension_dtype_mismatch_cross_class() -> None:
    left: DataFrame = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
    right: DataFrame = DataFrame({'a': [1, 2, 3]}, dtype='int64')
    tm.assert_frame_equal(left, right, check_dtype=False)

@pytest.mark.parametrize('dtype', ['timedelta64[ns]', 'datetime64[ns, UTC]', 'Period[D]'])
def test_assert_frame_equal_datetime_like_dtype_mismatch(dtype: str) -> None:
    df1: DataFrame = DataFrame({'a': []}, dtype=dtype)
    df2: DataFrame = DataFrame({'a': []})
    tm.assert_frame_equal(df1, df2, check_dtype=False)

def test_allows_duplicate_labels() -> None:
    left: DataFrame = DataFrame()
    right: DataFrame = DataFrame().set_flags(allows_duplicate_labels=False)
    tm.assert_frame_equal(left, left)
    tm.assert_frame_equal(right, right)
    tm.assert_frame_equal(left, right, check_flags=False)
    tm.assert_frame_equal(right, left, check_flags=False)
    with pytest.raises(AssertionError, match='<Flags'):
        tm.assert_frame_equal(left, right)
    with pytest.raises(AssertionError, match='<Flags'):
        tm.assert_frame_equal(left, right)

def test_assert_frame_equal_columns_mixed_dtype() -> None:
    df: DataFrame = DataFrame([[0, 1, 2]], columns=['foo', 'bar', 42], index=[1, 'test', 2])
    tm.assert_frame_equal(df, df, check_like=True)

def test_frame_equal_extension_dtype(
    frame_or_series: Type[Union[DataFrame, Series]],
    any_numeric_ea_dtype: Any
) -> None:
    obj: Union[DataFrame, Series] = frame_or_series([1, 2], dtype=any_numeric_ea_dtype)
    tm.assert_equal(obj, obj, check_exact=True)

@pytest.mark.parametrize('indexer', [(0, 1), (1, 0)])
def test_frame_equal_mixed_dtypes(
    frame_or_series: Type[Union[DataFrame, Series]],
    any_numeric_ea_dtype: Any,
    indexer: Tuple[int, int]
) -> None:
    dtypes: Tuple[Any, ...] = (any_numeric_ea_dtype, 'int64')
    obj1: Union[DataFrame, Series] = frame_or_series([1, 2], dtype=dtypes[indexer[0]])
    obj2: Union[DataFrame, Series] = frame_or_series([1, 2], dtype=dtypes[indexer[1]])
    tm.assert_equal(obj1, obj2, check_exact=True, check_dtype=False)

def test_assert_frame_equal_check_like_different_indexes() -> None:
    df1: DataFrame = DataFrame(index=pd.Index([], dtype='object'))
    df2: DataFrame = DataFrame(index=pd.RangeIndex(start=0, stop=0, step=1))
    with pytest.raises(AssertionError, match='DataFrame.index are different'):
        tm.assert_frame_equal(df1, df2, check_like=True)

def test_assert_frame_equal_checking_allow_dups_flag() -> None:
    left: DataFrame = DataFrame([[1, 2], [3, 4]])
    left.flags.allows_duplicate_labels = False
    right: DataFrame = DataFrame([[1, 2], [3, 4]])
    right.flags.allows_duplicate_labels = True
    tm.assert_frame_equal(left, right, check_flags=False)
    with pytest.raises(AssertionError, match='allows_duplicate_labels'):
        tm.assert_frame_equal(left, right, check_flags=True)

def test_assert_frame_equal_check_like_categorical_midx() -> None:
    left: DataFrame = DataFrame(
        [[1], [2], [3]],
        index=pd.MultiIndex.from_arrays([
            pd.Categorical(['a', 'b', 'c']),
            pd.Categorical(['a', 'b', 'c'])
        ])
    )
    right: DataFrame = DataFrame(
        [[3], [2], [1]],
        index=pd.MultiIndex.from_arrays([
            pd.Categorical(['c', 'b', 'a']),
            pd.Categorical(['c', 'b', 'a'])
        ])
    )
    tm.assert_frame_equal(left, right, check_like=True)

def test_assert_frame_equal_ea_column_definition_in_exception_mask() -> None:
    df1: DataFrame = DataFrame({'a': pd.Series([pd.NA, 1], dtype='Int64')})
    df2: DataFrame = DataFrame({'a': pd.Series([1, 1], dtype='Int64')})
    msg: str = 'DataFrame.iloc\\[:, 0\\] \\(column name="a"\\) NA mask values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)

def test_assert_frame_equal_ea_column_definition_in_exception() -> None:
    df1: DataFrame = DataFrame({'a': pd.Series([pd.NA, 1], dtype='Int64')})
    df2: DataFrame = DataFrame({'a': pd.Series([pd.NA, 2], dtype='Int64')})
    msg: str = 'DataFrame.iloc\\[:, 0\\] \\(column name="a"\\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_exact=True)

def test_assert_frame_equal_ts_column() -> None:
    df1: DataFrame = DataFrame({
        'a': [pd.Timestamp('2019-12-31'), pd.Timestamp('2020-12-31')]
    })
    df2: DataFrame = DataFrame({
        'a': [pd.Timestamp('2020-12-31'), pd.Timestamp('2020-12-31')]
    })
    msg: str = 'DataFrame.iloc\\[:, 0\\] \\(column name="a"\\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)

def test_assert_frame_equal_set() -> None:
    df1: DataFrame = DataFrame({'set_column': [{1, 2, 3}, {4, 5, 6}]})
    df2: DataFrame = DataFrame({'set_column': [{1, 2, 3}, {4, 5, 6}]})
    tm.assert_frame_equal(df1, df2)

def test_assert_frame_equal_set_mismatch() -> None:
    df1: DataFrame = DataFrame({'set_column': [{1, 2, 3}, {4, 5, 6}]})
    df2: DataFrame = DataFrame({'set_column': [{1, 2, 3}, {4, 5, 7}]})
    msg: str = 'DataFrame.iloc\\[:, 0\\] \\(column name="set_column"\\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)
