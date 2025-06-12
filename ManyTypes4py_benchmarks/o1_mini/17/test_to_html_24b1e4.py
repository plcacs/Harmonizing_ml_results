from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, get_option, option_context
import pandas.io.formats.format as fmt

lorem_ipsum: str = (
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore '
    'et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip '
    'ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu '
    'fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt '
    'mollit anim id est laborum.'
)

def expected_html(datapath: Any, name: str) -> str:
    """
    Read HTML file from formats data directory.

    Parameters
    ----------
    datapath : pytest fixture
        The datapath fixture injected into a test by pytest.
    name : str
        The name of the HTML file without the suffix.

    Returns
    -------
    str : contents of HTML file.
    """
    filename: str = '.'.join([name, 'html'])
    filepath: str = datapath('io', 'formats', 'data', 'html', filename)
    with open(filepath, encoding='utf-8') as f:
        html: str = f.read()
    return html.rstrip()

@pytest.fixture(params=['mixed', 'empty'])
def biggie_df_fixture(request: pytest.FixtureRequest) -> DataFrame:
    """Fixture for a big mixed Dataframe and an empty Dataframe"""
    if request.param == 'mixed':
        df: DataFrame = DataFrame({
            'A': np.random.default_rng(2).standard_normal(200),
            'B': Index([f'{i}?!' for i in range(200)])
        }, index=np.arange(200))
        df.loc[:20, 'A'] = np.nan
        df.loc[:20, 'B'] = np.nan
        return df
    elif request.param == 'empty':
        df: DataFrame = DataFrame(index=np.arange(200))
        return df

@pytest.fixture(params=fmt.VALID_JUSTIFY_PARAMETERS)
def justify(request: pytest.FixtureRequest) -> str:
    return request.param

@pytest.mark.parametrize('col_space', [30, 50])
def test_to_html_with_col_space(col_space: int) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    result: str = df.to_html(col_space=col_space)
    hdrs: List[str] = [x for x in result.split('\n') if re.search('<th[>\\s]', x)]
    assert len(hdrs) > 0
    for h in hdrs:
        assert 'min-width' in h
        assert str(col_space) in h

def test_to_html_with_column_specific_col_space_raises() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).random(size=(3, 3)), columns=['a', 'b', 'c'])
    msg: str = r'Col_space length\(\d+\) should match DataFrame number of columns\(\d+\)'
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space=[30, 40])
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space=[30, 40, 50, 60])
    msg = 'unknown column'
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space={'a': 'foo', 'b': 23, 'd': 34})

def test_to_html_with_column_specific_col_space() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).random(size=(3, 3)), columns=['a', 'b', 'c'])
    result: str = df.to_html(col_space={'a': '2em', 'b': 23})
    hdrs: List[str] = [x for x in result.split('\n') if re.search('<th[>\\s]', x)]
    assert 'min-width: 2em;">a</th>' in hdrs[1]
    assert 'min-width: 23px;">b</th>' in hdrs[2]
    assert '<th>c</th>' in hdrs[3]
    result = df.to_html(col_space=['1em', 2, 3])
    hdrs = [x for x in result.split('\n') if re.search('<th[>\\s]', x)]
    assert 'min-width: 1em;">a</th>' in hdrs[1]
    assert 'min-width: 2px;">b</th>' in hdrs[2]
    assert 'min-width: 3px;">c</th>' in hdrs[3]

def test_to_html_with_empty_string_label() -> None:
    data: Dict[str, List[Union[str, int]]] = {'c1': ['a', 'b'], 'c2': ['a', ''], 'data': [1, 2]}
    df: DataFrame = DataFrame(data).set_index(['c1', 'c2'])
    result: str = df.to_html()
    assert 'rowspan' not in result

@pytest.mark.parametrize(
    'df_data,expected',
    [
        ({'σ': np.arange(10.0)}, 'unicode_1'),
        ({'A': ['σ']}, 'unicode_2')
    ]
)
def test_to_html_unicode(
    df_data: Dict[str, Union[List[float], List[str]]],
    expected: str,
    datapath: Any
) -> None:
    df: DataFrame = DataFrame(df_data)
    expected_content: str = expected_html(datapath, expected)
    result: str = df.to_html()
    assert result == expected_content

def test_to_html_encoding(float_frame: DataFrame, tmp_path: Any) -> None:
    path: Any = tmp_path / 'test.html'
    float_frame.to_html(path, encoding='gbk')
    with open(str(path), encoding='gbk') as f:
        assert float_frame.to_html() == f.read()

def test_to_html_decimal(datapath: Any) -> None:
    df: DataFrame = DataFrame({'A': [6.0, 3.1, 2.2]})
    result: str = df.to_html(decimal=',')
    expected: str = expected_html(datapath, 'gh12031_expected_output')
    assert result == expected

@pytest.mark.parametrize(
    'kwargs,string,expected',
    [
        ({}, "<type 'str'>", 'escaped'),
        ({'escape': False}, '<b>bold</b>', 'escape_disabled')
    ]
)
def test_to_html_escaped(
    kwargs: Dict[str, Any],
    string: str,
    expected: str,
    datapath: Any
) -> None:
    a: str = 'str<ing1 &amp;'
    b: str = 'stri>ng2 &amp;'
    test_dict: Dict[str, Dict[str, str]] = {
        'co<l1': {a: string, b: string},
        'co>l2': {a: string, b: string}
    }
    result: str = DataFrame(test_dict).to_html(**kwargs)
    expected_content: str = expected_html(datapath, expected)
    assert result == expected_content

@pytest.mark.parametrize('index_is_named', [True, False])
def test_to_html_multiindex_index_false(index_is_named: bool, datapath: Any) -> None:
    df: DataFrame = DataFrame({
        'a': range(2),
        'b': range(3, 5),
        'c': range(5, 7),
        'd': range(3, 5)
    })
    df.columns = MultiIndex.from_product([['a', 'b'], ['c', 'd']])
    if index_is_named:
        df.index = Index(df.index.values, name='idx')
    result: str = df.to_html(index=False)
    expected: str = expected_html(datapath, 'gh8452_expected_output')
    assert result == expected

@pytest.mark.parametrize(
    'multi_sparse,expected',
    [
        (False, 'multiindex_sparsify_false_multi_sparse_1'),
        (False, 'multiindex_sparsify_false_multi_sparse_2'),
        (True, 'multiindex_sparsify_1'),
        (True, 'multiindex_sparsify_2')
    ]
)
def test_to_html_multiindex_sparsify(
    multi_sparse: bool,
    expected: str,
    datapath: Any
) -> None:
    index: MultiIndex = MultiIndex.from_arrays([[0, 0, 1, 1], [0, 1, 0, 1]], names=['foo', None])
    df: DataFrame = DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]], index=index)
    if expected.endswith('2'):
        df.columns = index[::2]
    with option_context('display.multi_sparse', multi_sparse):
        result: str = df.to_html()
    expected_content: str = expected_html(datapath, expected)
    assert result == expected_content

@pytest.mark.parametrize(
    'max_rows,expected',
    [
        (60, 'gh14882_expected_output_1'),
        (56, 'gh14882_expected_output_2')
    ]
)
def test_to_html_multiindex_odd_even_truncate(
    max_rows: int,
    expected: str,
    datapath: Any
) -> None:
    index: MultiIndex = MultiIndex.from_product(
        [[100, 200, 300], [10, 20, 30], [1, 2, 3, 4, 5, 6, 7]],
        names=['a', 'b', 'c']
    )
    df: DataFrame = DataFrame({'n': range(len(index))}, index=index)
    result: str = df.to_html(max_rows=max_rows)
    expected_content: str = expected_html(datapath, expected)
    assert result == expected_content

@pytest.mark.parametrize(
    'df,formatters,expected',
    [
        (
            DataFrame(
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                columns=Index(['foo', None], dtype=object),
                index=np.arange(4)
            ),
            {'__index__': lambda x: 'abcd'[x]},
            'index_formatter'
        ),
        (
            DataFrame({'months': [datetime(2016, 1, 1), datetime(2016, 2, 2)]}),
            {'months': lambda x: x.strftime('%Y-%m')},
            'datetime64_monthformatter'
        ),
        (
            DataFrame({'hod': pd.to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f')}),
            {'hod': lambda x: x.strftime('%H:%M')},
            'datetime64_hourformatter'
        ),
        (
            DataFrame({
                'i': pd.Series([1, 2], dtype='int64'),
                'f': pd.Series([1, 2], dtype='float64'),
                'I': pd.Series([1, 2], dtype='Int64'),
                's': pd.Series([1, 2], dtype='string'),
                'b': pd.Series([True, False], dtype='boolean'),
                'c': pd.Series(['a', 'b'], dtype=pd.CategoricalDtype(['a', 'b'])),
                'o': pd.Series([1, '2'], dtype=object)
            }),
            [lambda x: 'formatted'] * 7,
            'various_dtypes_formatted'
        )
    ]
)
def test_to_html_formatters(
    df: DataFrame,
    formatters: Union[Dict[str, Callable[[Any], Any]], List[Optional[Callable[[Any], Any]]]],
    expected: str,
    datapath: Any
) -> None:
    expected_content: str = expected_html(datapath, expected)
    result: str = df.to_html(formatters=formatters)
    assert result == expected_content

def test_to_html_regression_GH6098() -> None:
    df: DataFrame = DataFrame({
        'clé1': ['a', 'a', 'b', 'b', 'a'],
        'clé2': ['1er', '2ème', '1er', '2ème', '1er'],
        'données1': np.random.default_rng(2).standard_normal(5),
        'données2': np.random.default_rng(2).standard_normal(5)
    })
    df.pivot_table(index=['clé1'], columns=['clé2'])._repr_html_()

def test_to_html_truncate(datapath: Any) -> None:
    index: pd.DatetimeIndex = pd.date_range(start='20010101', freq='D', periods=20)
    df: DataFrame = DataFrame(index=index, columns=range(20))
    result: str = df.to_html(max_rows=8, max_cols=4)
    expected: str = expected_html(datapath, 'truncate')
    assert result == expected

@pytest.mark.parametrize('size', [1, 5])
def test_html_invalid_formatters_arg_raises(size: int) -> None:
    df: DataFrame = DataFrame(columns=['a', 'b', 'c'])
    msg: str = f'Formatters length({size}) should match DataFrame number of columns(3)'
    with pytest.raises(ValueError, match=re.escape(msg)):
        df.to_html(formatters=['{}'.format] * size)

def test_to_html_truncate_formatter(datapath: Any) -> None:
    data: List[Dict[str, int]] = [
        {'A': 1, 'B': 2, 'C': 3, 'D': 4},
        {'A': 5, 'B': 6, 'C': 7, 'D': 8},
        {'A': 9, 'B': 10, 'C': 11, 'D': 12},
        {'A': 13, 'B': 14, 'C': 15, 'D': 16}
    ]
    df: DataFrame = DataFrame(data)
    fmt: Callable[[Any], str] = lambda x: str(x) + '_mod'
    formatters: List[Optional[Callable[[Any], str]]] = [fmt, fmt, None, None]
    result: str = df.to_html(formatters=formatters, max_cols=3)
    expected: str = expected_html(datapath, 'truncate_formatter')
    assert result == expected

@pytest.mark.parametrize(
    'sparsify,expected',
    [
        (True, 'truncate_multi_index'),
        (False, 'truncate_multi_index_sparse_off')
    ]
)
def test_to_html_truncate_multi_index(
    sparsify: bool,
    expected: str,
    datapath: Any
) -> None:
    arrays: List[List[Union[str, int]]] = [
        ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
        ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']
    ]
    df: DataFrame = DataFrame(index=arrays, columns=arrays)
    result: str = df.to_html(max_rows=7, max_cols=7, sparsify=sparsify)
    expected_content: str = expected_html(datapath, expected)
    assert result == expected_content

@pytest.mark.parametrize(
    'option,result,expected',
    [
        (None, lambda df: df.to_html(), '1'),
        (None, lambda df: df.to_html(border=2), '2'),
        (2, lambda df: df.to_html(), '2'),
        (2, lambda df: df._repr_html_(), '2')
    ]
)
def test_to_html_border(
    option: Optional[int],
    result: Callable[[DataFrame], str],
    expected: str
) -> None:
    df: DataFrame = DataFrame({'A': [1, 2]})
    if option is None:
        html_result: str = result(df)
    else:
        with option_context('display.html.border', option):
            html_result = result(df)
    expected_attr: str = f'border="{expected}"'
    assert expected_attr in html_result

@pytest.mark.parametrize('biggie_df_fixture', ['mixed'], indirect=True)
def test_to_html(biggie_df_fixture: DataFrame) -> None:
    df: DataFrame = biggie_df_fixture
    s: str = df.to_html()
    buf: StringIO = StringIO()
    retval: Optional[str] = df.to_html(buf=buf)
    assert retval is None
    assert buf.getvalue() == s
    assert isinstance(s, str)
    df.to_html(columns=['B', 'A'], col_space=17)
    df.to_html(columns=['B', 'A'], formatters={'A': lambda x: f'{x:.1f}'})
    df.to_html(columns=['B', 'A'], float_format=str)
    df.to_html(columns=['B', 'A'], col_space=12, float_format=str)

@pytest.mark.parametrize('biggie_df_fixture', ['empty'], indirect=True)
def test_to_html_empty_dataframe(biggie_df_fixture: DataFrame) -> None:
    df: DataFrame = biggie_df_fixture
    df.to_html()

def test_to_html_filename(biggie_df_fixture: DataFrame, tmpdir: Any) -> None:
    df: DataFrame = biggie_df_fixture
    expected: str = df.to_html()
    path: Any = tmpdir.join('test.html')
    df.to_html(path)
    result: str = path.read()
    assert result == expected

def test_to_html_with_no_bold() -> None:
    df: DataFrame = DataFrame({'x': np.random.default_rng(2).standard_normal(5)})
    html: str = df.to_html(bold_rows=False)
    result: str = html[html.find('</thead>'):]
    assert '<strong' not in result

def test_to_html_columns_arg(float_frame: DataFrame) -> None:
    result: str = float_frame.to_html(columns=['A'])
    assert '<th>B</th>' not in result

@pytest.mark.parametrize(
    'columns,justify,expected',
    [
        (
            MultiIndex.from_arrays(
                [np.arange(2).repeat(2), np.mod(range(4), 2)],
                names=['CL0', 'CL1']
            ),
            'left',
            'multiindex_1'
        ),
        (
            MultiIndex.from_arrays(
                [np.arange(4), np.mod(range(4), 2)]
            ),
            'right',
            'multiindex_2'
        )
    ]
)
def test_to_html_multiindex(
    columns: MultiIndex,
    justify: str,
    expected: str,
    datapath: Any
) -> None:
    df: DataFrame = DataFrame([list('abcd'), list('efgh')], columns=columns)
    result: str = df.to_html(justify=justify)
    expected_content: str = expected_html(datapath, expected)
    assert result == expected_content

def test_to_html_justify(justify: str, datapath: Any) -> None:
    df: DataFrame = DataFrame({
        'A': [6, 30000, 2],
        'B': [1, 2, 70000],
        'C': [223442, 0, 1]
    }, columns=['A', 'B', 'C'])
    result: str = df.to_html(justify=justify)
    expected: str = expected_html(datapath, 'justify').format(justify=justify)
    assert result == expected

@pytest.mark.parametrize('justify', ['super-right', 'small-left', 'noinherit', 'tiny', 'pandas'])
def test_to_html_invalid_justify(justify: str) -> None:
    df: DataFrame = DataFrame()
    msg: str = 'Invalid value for justify parameter'
    with pytest.raises(ValueError, match=msg):
        df.to_html(justify=justify)

class TestHTMLIndex:

    @pytest.fixture
    def df(self) -> DataFrame:
        index: List[str] = ['foo', 'bar', 'baz']
        df: DataFrame = DataFrame({
            'A': [1, 2, 3],
            'B': [1.2, 3.4, 5.6],
            'C': ['one', 'two', np.nan]
        }, columns=['A', 'B', 'C'], index=index)
        return df

    @pytest.fixture
    def expected_without_index(self, datapath: Any) -> str:
        return expected_html(datapath, 'index_2')

    def test_to_html_flat_index_without_name(
        self,
        datapath: Any,
        df: DataFrame,
        expected_without_index: str
    ) -> None:
        expected_with_index: str = expected_html(datapath, 'index_1')
        assert df.to_html() == expected_with_index
        result: str = df.to_html(index=False)
        for i in df.index:
            assert i not in result
        assert result == expected_without_index

    def test_to_html_flat_index_with_name(
        self,
        datapath: Any,
        df: DataFrame,
        expected_without_index: str
    ) -> None:
        df.index = Index(['foo', 'bar', 'baz'], name='idx')
        expected_with_index: str = expected_html(datapath, 'index_3')
        assert df.to_html() == expected_with_index
        assert df.to_html(index=False) == expected_without_index

    def test_to_html_multiindex_without_names(
        self,
        datapath: Any,
        df: DataFrame,
        expected_without_index: str
    ) -> None:
        tuples: List[Tuple[str, str]] = [('foo', 'car'), ('foo', 'bike'), ('bar', 'car')]
        df.index = MultiIndex.from_tuples(tuples)
        expected_with_index: str = expected_html(datapath, 'index_4')
        assert df.to_html() == expected_with_index
        result: str = df.to_html(index=False)
        for i in ['foo', 'bar', 'car', 'bike']:
            assert i not in result
        assert result == expected_without_index

    def test_to_html_multiindex_with_names(
        self,
        datapath: Any,
        df: DataFrame,
        expected_without_index: str
    ) -> None:
        tuples: List[Tuple[str, str]] = [('foo', 'car'), ('foo', 'bike'), ('bar', 'car')]
        df.index = MultiIndex.from_tuples(tuples, names=['idx1', 'idx2'])
        expected_with_index: str = expected_html(datapath, 'index_5')
        assert df.to_html() == expected_with_index
        assert df.to_html(index=False) == expected_without_index

@pytest.mark.parametrize(
    'classes',
    [
        'sortable draggable',
        ['sortable', 'draggable']
    ]
)
def test_to_html_with_classes(
    classes: Union[str, List[str]],
    datapath: Any
) -> None:
    df: DataFrame = DataFrame()
    expected: str = expected_html(datapath, 'with_classes')
    result: str = df.to_html(classes=classes)
    assert result == expected

def test_to_html_no_index_max_rows(datapath: Any) -> None:
    df: DataFrame = DataFrame({'A': [1, 2, 3, 4]})
    result: str = df.to_html(index=False, max_rows=1)
    expected: str = expected_html(datapath, 'gh14998_expected_output')
    assert result == expected

def test_to_html_multiindex_max_cols(datapath: Any) -> None:
    index: MultiIndex = MultiIndex(
        levels=[['ba', 'bb', 'bc'], ['ca', 'cb', 'cc']],
        codes=[[0, 1, 2], [0, 1, 2]],
        names=['b', 'c']
    )
    columns: MultiIndex = MultiIndex(
        levels=[['d'], ['aa', 'ab', 'ac']],
        codes=[[0, 0, 0], [0, 1, 2]],
        names=[None, 'a']
    )
    data: np.ndarray = np.array([[1.0, np.nan, np.nan], [np.nan, 2.0, np.nan], [np.nan, np.nan, 3.0]])
    df: DataFrame = DataFrame(data, index, columns)
    result: str = df.to_html(max_cols=2)
    expected: str = expected_html(datapath, 'gh6131_expected_output')
    assert result == expected

def test_to_html_multi_indexes_index_false(datapath: Any) -> None:
    df: DataFrame = DataFrame({
        'a': range(10),
        'b': range(10, 20),
        'c': range(10, 20),
        'd': range(10, 20)
    })
    df.columns = MultiIndex.from_product([['a', 'b'], ['c', 'd']])
    df.index = MultiIndex.from_product([['a', 'b'], ['c', 'd', 'e', 'f', 'g']])
    result: str = df.to_html(index=False)
    expected: str = expected_html(datapath, 'gh22579_expected_output')
    assert result == expected

@pytest.mark.parametrize(
    'index_names,header,index,column_index,column_type,row_index,row_type',
    [
        (True, True, True, Index([0, 1]), 'unnamed_standard', Index([0, 1]), 'unnamed_standard'),
        (False, True, True, Index([0, 1], name='columns.name'), 'named_standard', Index([0, 1]), 'unnamed_standard'),
        (True, True, True, MultiIndex.from_product([['a'], ['b', 'c']]), 'unnamed_multi', MultiIndex.from_product([['a'], ['b', 'c']]), 'unnamed_multi'),
        (
            False,
            True,
            True,
            MultiIndex.from_product([['a'], ['b', 'c']], names=['columns.name.0', 'columns.name.1']),
            'named_multi',
            MultiIndex.from_product([['a'], ['b', 'c']]),
            'unnamed_multi'
        ),
    ]
)
@pytest.mark.parametrize(
    'row_index, row_type, column_index, column_type',
    [
        (Index([0, 1]), 'unnamed_standard'),
        (Index([0, 1], name='index.name'), 'named_standard'),
        (MultiIndex.from_product([['a'], ['b', 'c']]), 'unnamed_multi'),
        (MultiIndex.from_product([['a'], ['b', 'c']], names=['index.name.0', 'index.name.1']), 'named_multi')
    ]
)
def test_to_html_basic_alignment(
    datapath: Any,
    row_index: Index,
    row_type: str,
    column_index: Union[Index, MultiIndex],
    column_type: str,
    index: bool,
    header: bool,
    index_names: bool
) -> None:
    df: DataFrame = DataFrame(np.zeros((2, 2), dtype=int), index=row_index, columns=column_index)
    result: str = df.to_html(index=index, header=header, index_names=index_names)
    if not index:
        current_row_type: str = 'none'
    elif not index_names and row_type.startswith('named'):
        current_row_type = 'un' + row_type
    else:
        current_row_type = row_type
    if not header:
        current_column_type: str = 'none'
    elif not index_names and column_type.startswith('named'):
        current_column_type = 'un' + column_type
    else:
        current_column_type = column_type
    filename: str = f'index_{current_row_type}_columns_{current_column_type}'
    expected: str = expected_html(datapath, filename)
    assert result == expected

@pytest.mark.parametrize(
    'index_names,header,index,column_index,column_type,row_index,row_type',
    []
)
@pytest.mark.parametrize(
    'column_index, column_type',
    [
        (Index(np.arange(8)), 'unnamed_standard'),
        (Index(np.arange(8), name='columns.name'), 'named_standard'),
        (MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']]), 'unnamed_multi'),
        (MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']], names=['foo', None, 'baz']), 'named_multi')
    ]
)
@pytest.mark.parametrize(
    'row_index, row_type',
    [
        (Index(np.arange(8)), 'unnamed_standard'),
        (Index(np.arange(8), name='index.name'), 'named_standard'),
        (MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']]), 'unnamed_multi'),
        (MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']], names=['foo', None, 'baz']), 'named_multi')
    ]
)
def test_to_html_alignment_with_truncation(
    datapath: Any,
    row_index: Index,
    row_type: str,
    column_index: Union[Index, MultiIndex],
    column_type: str,
    index: bool,
    header: bool,
    index_names: bool
) -> None:
    df: DataFrame = DataFrame(np.arange(64).reshape(8, 8), index=row_index, columns=column_index)
    result: str = df.to_html(max_rows=4, max_cols=4, index=index, header=header, index_names=index_names)
    if not index:
        current_row_type: str = 'none'
    elif not index_names and row_type.startswith('named'):
        current_row_type = 'un' + row_type
    else:
        current_row_type = row_type
    if not header:
        current_column_type: str = 'none'
    elif not index_names and column_type.startswith('named'):
        current_column_type = 'un' + column_type
    else:
        current_column_type = column_type
    filename: str = f'trunc_df_index_{current_row_type}_columns_{current_column_type}'
    expected: str = expected_html(datapath, filename)
    assert result == expected

@pytest.mark.parametrize(
    'index',
    [
        False,
        0
    ]
)
def test_to_html_truncation_index_false_max_rows(
    datapath: Any,
    index: Union[bool, int]
) -> None:
    data: List[List[float]] = [
        [1.764052, 0.400157],
        [0.978738, 2.240893],
        [1.867558, -0.977278],
        [0.950088, -0.151357],
        [-0.103219, 0.410599]
    ]
    df: DataFrame = DataFrame(data)
    result: str = df.to_html(max_rows=4, index=index)
    expected: str = expected_html(datapath, 'gh15019_expected_output')
    assert result == expected

@pytest.mark.parametrize(
    'index,col_index_named,expected_output',
    [
        (False, False, 'gh22783_expected_output'),
        (False, True, 'gh22783_named_columns_index')
    ]
)
@pytest.mark.parametrize(
    'index',
    [
        False,
        0
    ]
)
def test_to_html_truncation_index_false_max_cols(
    datapath: Any,
    index: Union[bool, int],
    col_index_named: bool,
    expected_output: str
) -> None:
    data: List[List[Union[float, str]]] = [
        [1.764052, 0.400157, 0.978738, 2.240893, 1.867558],
        [-0.977278, 0.950088, -0.151357, -0.103219, 0.410599]
    ]
    df: DataFrame = DataFrame(data)
    if col_index_named:
        df.columns.rename('columns.name', inplace=True)
    result: str = df.to_html(max_cols=4, index=index)
    expected: str = expected_html(datapath, expected_output)
    assert result == expected

@pytest.mark.parametrize('notebook', [True, False])
def test_to_html_notebook_has_style(notebook: bool) -> None:
    df: DataFrame = DataFrame({'A': [1, 2, 3]})
    result: str = df.to_html(notebook=notebook)
    if notebook:
        assert 'tbody tr th:only-of-type' in result
        assert 'vertical-align: middle;' in result
        assert 'thead th' in result
    else:
        assert 'tbody tr th:only-of-type' not in result
        assert 'vertical-align: middle;' not in result
        assert 'thead th' not in result

def test_to_html_with_index_names_false() -> None:
    df: DataFrame = DataFrame({
        'A': [1, 2]
    }, index=Index(['a', 'b'], name='myindexname'))
    result: str = df.to_html(index_names=False)
    assert 'myindexname' not in result

def test_to_html_with_id() -> None:
    df: DataFrame = DataFrame({
        'A': [1, 2]
    }, index=Index(['a', 'b'], name='myindexname'))
    result: str = df.to_html(index_names=False, table_id='TEST_ID')
    assert ' id="TEST_ID"' in result

@pytest.mark.parametrize(
    'value,float_format,expected',
    [
        (0.19999, '%.3f', 'gh21625_expected_output'),
        (100.0, '%.0f', 'gh22270_expected_output')
    ]
)
def test_to_html_float_format_no_fixed_width(
    value: float,
    float_format: str,
    expected: str,
    datapath: Any
) -> None:
    df: DataFrame = DataFrame({'x': [value]})
    expected_content: str = expected_html(datapath, expected)
    result: str = df.to_html(float_format=float_format)
    assert result == expected_content

@pytest.mark.parametrize(
    'render_links,expected',
    [
        (True, 'render_links_true'),
        (False, 'render_links_false')
    ]
)
def test_to_html_render_links(
    render_links: bool,
    expected: str,
    datapath: Any
) -> None:
    data: List[List[Union[int, str]]] = [
        [0, 'https://pandas.pydata.org/?q1=a&q2=b', 'pydata.org'],
        [0, 'www.pydata.org', 'pydata.org']
    ]
    df: DataFrame = DataFrame(data, columns=Index(['foo', 'bar', None], dtype=object))
    result: str = df.to_html(render_links=render_links)
    expected_content: str = expected_html(datapath, expected)
    assert result == expected_content

@pytest.mark.parametrize(
    'method,expected',
    [
        ('to_html', lambda x: lorem_ipsum),
        ('_repr_html_', lambda x: lorem_ipsum[:x - 4] + '...')
    ]
)
@pytest.mark.parametrize('max_colwidth', [10, 20, 50, 100])
def test_ignore_display_max_colwidth(
    method: str,
    expected: Callable[[int], str],
    max_colwidth: int
) -> None:
    df: DataFrame = DataFrame([lorem_ipsum])
    with option_context('display.max_colwidth', max_colwidth):
        result: str = getattr(df, method)()
    expected_content: str = expected(max_colwidth)
    assert expected_content in result

@pytest.mark.parametrize('classes', [True, 0])
def test_to_html_invalid_classes_type(classes: Union[bool, int]) -> None:
    df: DataFrame = DataFrame()
    msg: str = 'classes must be a string, list, or tuple'
    with pytest.raises(TypeError, match=msg):
        df.to_html(classes=classes)

def test_to_html_round_column_headers() -> None:
    df: DataFrame = DataFrame([1], columns=[0.55555])
    with option_context('display.precision', 3):
        html: str = df.to_html(notebook=False)
        notebook: str = df.to_html(notebook=True)
    assert '0.55555' in html
    assert '0.556' in notebook

@pytest.mark.parametrize(
    'unit',
    [
        '100px',
        '10%',
        '5em',
        150
    ]
)
def test_to_html_with_col_space_units(unit: Union[str, int]) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    result: str = df.to_html(col_space=unit)
    result_split: List[str] = result.split('tbody')[0]
    hdrs: List[str] = [x for x in result_split.split('\n') if re.search('<th[>\\s]', x)]
    if isinstance(unit, int):
        unit_str: str = f'{unit}px'
    else:
        unit_str = unit
    for h in hdrs:
        expected: str = f'<th style="min-width: {unit_str};">'
        assert expected in h

class TestReprHTML:

    def test_html_repr_min_rows_default(self, datapath: Any) -> None:
        df: DataFrame = DataFrame({'a': range(20)})
        result: str = df._repr_html_()
        expected: str = expected_html(datapath, 'html_repr_min_rows_default_no_truncation')
        assert result == expected
        df = DataFrame({'a': range(61)})
        result = df._repr_html_()
        expected = expected_html(datapath, 'html_repr_min_rows_default_truncated')
        assert result == expected

    @pytest.mark.parametrize(
        'max_rows,min_rows,expected',
        [
            (10, 4, 'html_repr_max_rows_10_min_rows_4'),
            (12, None, 'html_repr_max_rows_12_min_rows_None'),
            (10, 12, 'html_repr_max_rows_10_min_rows_12'),
            (None, 12, 'html_repr_max_rows_None_min_rows_12')
        ]
    )
    def test_html_repr_min_rows(
        self,
        datapath: Any,
        max_rows: Optional[int],
        min_rows: Optional[int],
        expected: str
    ) -> None:
        df: DataFrame = DataFrame({'a': range(61)})
        expected_content: str = expected_html(datapath, expected)
        with option_context('display.max_rows', max_rows, 'display.min_rows', min_rows):
            result: str = df._repr_html_()
        assert result == expected_content

    def test_repr_html_ipython_config(self, ip: Any) -> None:
        code: str = textwrap.dedent('''
            from pandas import DataFrame
            df = DataFrame({"A": [1, 2]})
            df._repr_html_()

            cfg = get_ipython().config
            cfg['IPKernelApp']['parent_appname']
            df._repr_html_()
            ''')
        result = ip.run_cell(code, silent=True)
        assert not result.error_in_exec

    def test_info_repr_html(self) -> None:
        max_rows: int = 60
        max_cols: int = 20
        h, w = (max_rows + 1, max_cols - 1)
        df: DataFrame = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert '&lt;class' not in df._repr_html_()
        with option_context('display.large_repr', 'info'):
            assert '&lt;class' in df._repr_html_()
        h, w = (max_rows - 1, max_cols + 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert '<class' not in df._repr_html_()
        with option_context('display.large_repr', 'info', 'display.max_columns', max_cols):
            assert '&lt;class' in df._repr_html_()

    def test_fake_qtconsole_repr_html(self, float_frame: DataFrame) -> None:
        df: DataFrame = float_frame

        def get_ipython() -> Dict[str, Any]:
            return {'config': {'KernelApp': {'parent_appname': 'ipython-qtconsole'}}}
        
        repstr: Optional[str] = df._repr_html_()
        assert repstr is not None
        with option_context('display.max_rows', 5, 'display.max_columns', 2):
            repstr = df._repr_html_()
        assert 'class' in repstr

    def test_repr_html(self, float_frame: DataFrame) -> None:
        df: DataFrame = float_frame
        df._repr_html_()
        with option_context('display.max_rows', 1, 'display.max_columns', 1):
            df._repr_html_()
        with option_context('display.notebook_repr_html', False):
            df._repr_html_()
        df = DataFrame([[1, 2], [3, 4]])
        with option_context('display.show_dimensions', True):
            assert '2 rows' in df._repr_html_()
        with option_context('display.show_dimensions', False):
            assert '2 rows' not in df._repr_html_()

    def test_repr_html_mathjax(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]])
        assert 'tex2jax_ignore' not in df._repr_html_()
        assert 'mathjax_ignore' not in df._repr_html_()
        with option_context('display.html.use_mathjax', False):
            assert 'tex2jax_ignore' in df._repr_html_()
            assert 'mathjax_ignore' in df._repr_html_()

    def test_repr_html_wide(self) -> None:
        max_cols: int = 20
        df: DataFrame = DataFrame([['a' * 25] * (max_cols - 1)] * 10)
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' not in df._repr_html_()
        wide_df: DataFrame = DataFrame([['a' * 25] * (max_cols + 1)] * 10)
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' in wide_df._repr_html_()

    def test_repr_html_wide_multiindex_cols(self) -> None:
        max_cols: int = 20
        mcols: MultiIndex = MultiIndex.from_product(
            [np.arange(max_cols // 2), ['foo', 'bar']],
            names=['first', 'second']
        )
        df: DataFrame = DataFrame([['a' * 25] * len(mcols)] * 10, columns=mcols)
        reg_repr: str = df._repr_html_()
        assert '...' not in reg_repr
        mcols = MultiIndex.from_product(
            (np.arange(1 + max_cols // 2), ['foo', 'bar']),
            names=['first', 'second']
        )
        df = DataFrame([['a' * 25] * len(mcols)] * 10, columns=mcols)
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' in df._repr_html_()

    def test_repr_html_long(self) -> None:
        with option_context('display.max_rows', 60):
            max_rows: int = get_option('display.max_rows')
            h: int = max_rows - 1
            df: DataFrame = DataFrame({'A': np.arange(1, 1 + h), 'B': np.arange(41, 41 + h)})
            reg_repr: str = df._repr_html_()
            assert '..' not in reg_repr
            assert str(41 + max_rows // 2) in reg_repr
            h = max_rows + 1
            df = DataFrame({'A': np.arange(1, 1 + h), 'B': np.arange(41, 41 + h)})
            long_repr: str = df._repr_html_()
            assert '..' in long_repr
            assert str(41 + max_rows // 2) not in long_repr
            assert f'{h} rows ' in long_repr
            assert '2 columns' in long_repr

    def test_repr_html_float(self) -> None:
        with option_context('display.max_rows', 60):
            max_rows: int = get_option('display.max_rows')
            h: int = max_rows - 1
            df: DataFrame = DataFrame({
                'idx': np.linspace(-10, 10, h),
                'A': np.arange(1, 1 + h),
                'B': np.arange(41, 41 + h)
            }).set_index('idx')
            reg_repr: str = df._repr_html_()
            assert '..' not in reg_repr
            assert f'<td>{40 + h}</td>' in reg_repr
            h = max_rows + 1
            df = DataFrame({
                'idx': np.linspace(-10, 10, h),
                'A': np.arange(1, 1 + h),
                'B': np.arange(41, 41 + h)
            }).set_index('idx')
            long_repr: str = df._repr_html_()
            assert '..' in long_repr
            assert '<td>31</td>' not in long_repr
            assert f'{h} rows ' in long_repr
            assert '2 columns' in long_repr

    def test_repr_html_long_multiindex(self) -> None:
        max_rows: int = 60
        max_L1: int = max_rows // 2
        tuples: List[Tuple[int, str]] = list(itertools.product(np.arange(max_L1), ['foo', 'bar']))
        idx: MultiIndex = MultiIndex.from_tuples(tuples, names=['first', 'second'])
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((max_L1 * 2, 2)),
            index=idx,
            columns=['A', 'B']
        )
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            reg_repr: str = df._repr_html_()
        assert '...' not in reg_repr
        tuples = list(itertools.product(np.arange(max_L1 + 1), ['foo', 'bar']))
        idx = MultiIndex.from_tuples(tuples, names=['first', 'second'])
        df = DataFrame(
            np.random.default_rng(2).standard_normal(((max_L1 + 1) * 2, 2)),
            index=idx,
            columns=['A', 'B']
        )
        long_repr: str = df._repr_html_()
        assert '...' in long_repr

    def test_repr_html_long_and_wide(self) -> None:
        max_cols: int = 20
        max_rows: int = 60
        h, w = (max_rows - 1, max_cols - 1)
        df: DataFrame = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' not in df._repr_html_()
        h, w = (max_rows + 1, max_cols + 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        with option_context('display.max_rows', 60, 'display.max_columns', 20):
            assert '...' in df._repr_html_()

def test_to_html_multilevel(multiindex_year_month_day_dataframe_random_data: DataFrame) -> None:
    ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
    ymd.columns.name = 'foo'
    ymd.to_html()
    ymd.T.to_html()

@pytest.mark.parametrize(
    'na_rep',
    ['NaN', 'Ted']
)
def test_to_html_na_rep_and_float_format(
    na_rep: str,
    datapath: Any
) -> None:
    df: DataFrame = DataFrame([
        ['A', 1.2225],
        ['A', None]
    ], columns=['Group', 'Data'])
    result: str = df.to_html(na_rep=na_rep, float_format='{:.2f}'.format)
    expected: str = expected_html(datapath, 'gh13828_expected_output')
    expected = expected.format(na_rep=na_rep)
    assert result == expected

def test_to_html_na_rep_non_scalar_data(datapath: Any) -> None:
    df: DataFrame = DataFrame([{'a': 1, 'b': [1, 2, 3]}])
    result: str = df.to_html(na_rep='-')
    expected: str = expected_html(datapath, 'gh47103_expected_output')
    assert result == expected

def test_to_html_float_format_object_col(datapath: Any) -> None:
    df: DataFrame = DataFrame(data={'x': [1000.0, 'test']})
    result: str = df.to_html(float_format=lambda x: f'{x:,.0f}')
    expected: str = expected_html(datapath, 'gh40024_expected_output')
    assert result == expected

def test_to_html_multiindex_col_with_colspace() -> None:
    df: DataFrame = DataFrame([[1, 2]])
    df.columns = MultiIndex.from_tuples([(1, 1), (2, 1)])
    result: str = df.to_html(col_space=100)
    expected: str = (
        '<table border="1" class="dataframe">\n'
        '  <thead>\n'
        '    <tr>\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '      <th style="min-width: 100px;">2</th>\n'
        '    </tr>\n'
        '    <tr>\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '    </tr>\n'
        '  </thead>\n'
        '  <tbody>\n'
        '    <tr>\n'
        '      <th>0</th>\n'
        '      <td>1</td>\n'
        '      <td>2</td>\n'
        '    </tr>\n'
        '  </tbody>\n'
        '</table>'
    )
    assert result == expected

def test_to_html_tuple_col_with_colspace() -> None:
    df: DataFrame = DataFrame({('a', 'b'): [1], 'b': [2]})
    result: str = df.to_html(col_space=100)
    expected: str = (
        '<table border="1" class="dataframe">\n'
        '  <thead>\n'
        '    <tr style="text-align: right;">\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">(a, b)</th>\n'
        '      <th style="min-width: 100px;">b</th>\n'
        '    </tr>\n'
        '  </thead>\n'
        '  <tbody>\n'
        '    <tr>\n'
        '      <th>0</th>\n'
        '      <td>1</td>\n'
        '      <td>2</td>\n'
        '    </tr>\n'
        '  </tbody>\n'
        '</table>'
    )
    assert result == expected

def test_to_html_empty_complex_array() -> None:
    df: DataFrame = DataFrame({'x': np.array([], dtype='complex')})
    result: str = df.to_html(col_space=100)
    expected: str = (
        '<table border="1" class="dataframe">\n'
        '  <thead>\n'
        '    <tr style="text-align: right;">\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">x</th>\n'
        '    </tr>\n'
        '  </thead>\n'
        '  <tbody>\n'
        '  </tbody>\n'
        '</table>'
    )
    assert result == expected
