from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, get_option, option_context
import pandas.io.formats.format as fmt

lorem_ipsum: str = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

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
    filepath: Any = datapath('io', 'formats', 'data', 'html', filename)
    with open(filepath, encoding='utf-8') as f:
        html: str = f.read()
    return html.rstrip()

@pytest.fixture(params=['mixed', 'empty'])
def biggie_df_fixture(request: pytest.FixtureRequest) -> DataFrame:
    """Fixture for a big mixed Dataframe and an empty Dataframe"""
    if request.param == 'mixed':
        df: DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(200), 'B': Index([f'{i}?!' for i in range(200)])}, index=np.arange(200))
        df.loc[:20, 'A'] = np.nan
        df.loc[:20, 'B'] = np.nan
        return df
    elif request.param == 'empty':
        df = DataFrame(index=np.arange(200))
        return df

@pytest.fixture(params=fmt.VALID_JUSTIFY_PARAMETERS)
def justify(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)

@pytest.mark.parametrize('col_space', [30, 50])
def test_to_html_with_col_space(col_space: int) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    result: str = df.to_html(col_space=col_space)
    hdrs: List[str] = [x for x in result.split('\\n') if re.search('<th[>\\s]', x)]
    assert len(hdrs) > 0
    for h in hdrs:
        assert 'min-width' in h
        assert str(col_space) in h

def test_to_html_with_column_specific_col_space_raises() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).random(size=(3, 3)), columns=['a', 'b', 'c'])
    msg: str = 'Col_space length\\(\\d+\\) should match DataFrame number of columns\\(\\d+\\)'
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

@pytest.mark.parametrize('df_data,expected', [({'σ': np.arange(10.0)}, 'unicode_1'), ({'A': ['σ']}, 'unicode_2')])
def test_to_html_unicode(df_data: Dict[str, Any], expected: str, datapath: Any) -> None:
    df: DataFrame = DataFrame(df_data)
    expected_html_str: str = expected_html(datapath, expected)
    result: str = df.to_html()
    assert result == expected_html_str

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

@pytest.mark.parametrize('kwargs,string,expected', [({}, "<type 'str'>", 'escaped'), ({'escape': False}, '<b>bold</b>', 'escape_disabled')])
def test_to_html_escaped(kwargs: Dict[str, Any], string: str, expected: str, datapath: Any) -> None:
    a: str = 'str<ing1 &amp;'
    b: str = 'stri>ng2 &amp;'
    test_dict: Dict[str, Dict[str, str]] = {'co<l1': {a: string, b: string}, 'co>l2': {a: string, b: string}}
    result: str = DataFrame(test_dict).to_html(**kwargs)
    expected_html_str: str = expected_html(datapath, expected)
    assert result == expected_html_str

@pytest.mark.parametrize('index_is_named', [True, False])
def test_to_html_multiindex_index_false(index_is_named: bool, datapath: Any) -> None:
    df: DataFrame = DataFrame({'a': range(2), 'b': range(3, 5), 'c': range(5, 7), 'd': range(3, 5)})
    df.columns = MultiIndex.from_product([['a', 'b'], ['c', 'd']])
    if index_is_named:
        df.index = Index(df.index.values, name='idx')
    result: str = df.to_html(index=False)
    expected: str = expected_html(datapath, 'gh8452_expected_output')
    assert result == expected

@pytest.mark.parametrize('multi_sparse,expected', [(False, 'multiindex_sparsify_false_multi_sparse_1'), (False, 'multiindex_sparsify_false_multi_sparse_2'), (True, 'multiindex_sparsify_1'), (True, 'multiindex_sparsify_2')])
def test_to_html_multiindex_sparsify(multi_sparse: bool, expected: str, datapath: Any) -> None:
    index: MultiIndex = MultiIndex.from_arrays([[0, 0, 1, 1], [0, 1, 0, 1]], names=['foo', None])
    df: DataFrame = DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]], index=index)
    if expected.endswith('2'):
        df.columns = index[::2]
    with option_context('display.multi_sparse', multi_sparse):
        result: str = df.to_html()
    expected_html_str: str = expected_html(datapath, expected)
    assert result == expected_html_str

@pytest.mark.parametrize('max_rows,expected', [(60, 'gh14882_expected_output_1'), (56, 'gh14882_expected_output_2')])
def test_to_html_multiindex_odd_even_truncate(max_rows: int, expected: str, datapath: Any) -> None:
    index: MultiIndex = MultiIndex.from_product([[100, 200, 300], [10, 20, 30], [1, 2, 3, 4, 5, 6, 7]], names=['a', 'b', 'c'])
    df: DataFrame = DataFrame({'n': range(len(index))}, index=index)
    result: str = df.to_html(max_rows=max_rows)
    expected_html_str: str = expected_html(datapath, expected)
    assert result == expected_html_str

@pytest.mark.parametrize('df,formatters,expected', [(DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]], columns=Index(['foo', None], dtype=object), index=np.arange(4)), {'__index__': lambda x: 'abcd'[x]}, 'index_formatter'), (DataFrame({'months': [datetime(2016, 1, 1), datetime(2016, 2, 2)]}), {'months': lambda x: x.strftime('%Y-%m')}, 'datetime64_monthformatter'), (DataFrame({'hod': pd.to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f')}), {'hod': lambda x: x.strftime('%H:%M')}, 'datetime64_hourformatter'), (DataFrame({'i': pd.Series([1, 2], dtype='int64'), 'f': pd.Series([1, 2], dtype='float64'), 'I': pd.Series([1, 2], dtype='Int64'), 's': pd.Series([1, 2], dtype='string'), 'b': pd.Series([True, False], dtype='boolean'), 'c': pd.Series(['a', 'b'], dtype=pd.CategoricalDtype(['a', 'b'])), 'o': pd.Series([1, '2'], dtype=object)}), [lambda x: 'formatted'] * 7, 'various_dtypes_formatted')])
def test_to_html_formatters(df: DataFrame, formatters: Any, expected: str, datapath: Any) -> None:
    expected_html_str: str = expected_html(datapath, expected)
    result: str = df.to_html(formatters=formatters)
    assert result == expected_html_str

def test_to_html_regression_GH6098() -> None:
    df: DataFrame = DataFrame({'clé1': ['a', 'a', 'b', 'b', 'a'], 'clé2': ['1er', '2ème', '1er', '2ème', '1er'], 'données1': np.random.default_rng(2).standard_normal(5), 'données2': np.random.default_rng(2).standard_normal(5)})
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
    msg: str = 'Formatters length({}) should match DataFrame number of columns(3)'
    with pytest.raises(ValueError, match=re.escape(msg.format(size))):
        df.to_html(formatters=['{}'.format] * size)

def test_to_html_truncate_formatter(datapath: Any) -> None:
    data: List[Dict[str, int]] = [{'A': 1, 'B': 2, 'C': 3, 'D': 4}, {'A': 5, 'B': 6, 'C': 7, 'D': 8}, {'A': 9, 'B': 10, 'C': 11, 'D': 12}, {'A': 13, 'B': 14, 'C': 15, 'D': 16}]
    df: DataFrame = DataFrame(data)
    fmt_func = lambda x: str(x) + '_mod'
    formatters: List[Optional[Any]] = [fmt_func, fmt_func, None, None]
    result: str = df.to_html(formatters=formatters, max_cols=3)
    expected: str = expected_html(datapath, 'truncate_formatter')
    assert result == expected

@pytest.mark.parametrize('sparsify,expected', [(True, 'truncate_multi_index'), (False, 'truncate_multi_index_sparse_off')])
def test_to_html_truncate_multi_index(sparsify: bool, expected: str, datapath: Any) -> None:
    arrays: List[List[str]] = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
    df: DataFrame = DataFrame(index=arrays, columns=arrays)
    result: str = df.to_html(max_rows=7, max_cols=7, sparsify=sparsify)
    expected_html_str: str = expected_html(datapath, expected)
    assert result == expected_html_str

@pytest.mark.parametrize('option,result,expected', [(None, lambda df: df.to_html(), '1'), (None, lambda df: df.to_html(border=2), '2'), (2, lambda df: df.to_html(), '2'), (2, lambda df: df._repr_html_(), '2')])
def test_to_html_border(option: Optional[int], result: Any, expected: str) -> None:
    df: DataFrame = DataFrame({'A': [1, 2]})
    if option is None:
        result_str: str = result(df)
    else:
        with option_context('display.html.border', option):
            result_str = result(df)
    expected_str: str = f'border="{expected}"'
    assert expected_str in result_str

@pytest.mark.parametrize('biggie_df_fixture', ['mixed'], indirect=True)
def test_to_html(biggie_df_fixture: DataFrame) -> None:
    df: DataFrame = biggie_df_fixture
    s: str = df.to_html()
    buf: StringIO = StringIO()
    retval: None = df.to_html(buf=buf)
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
    result: str = html[html.find('</thead>')]
    assert '<strong' not in result

def test_to_html_columns_arg(float_frame: DataFrame) -> None:
    result: str = float_frame.to_html(columns=['A'])
    assert '<th>B</th>' not in result

@pytest.mark.parametrize('columns,justify,expected', [(MultiIndex.from_arrays([np.arange(2).repeat(2), np.mod(range(4), 2)], names=['CL0', 'CL1']), 'left', 'multiindex_