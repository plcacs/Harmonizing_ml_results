import numpy as np
import pytest
from pandas import NA, DataFrame, IndexSlice, MultiIndex, NaT, Timestamp, option_context
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape

@pytest.fixture
def df() -> DataFrame:
    return DataFrame(data=[[0, -0.609], [1, -1.228]], columns=['A', 'B'], index=['x', 'y'])

@pytest.fixture
def styler(df: DataFrame) -> Styler:
    return Styler(df, uuid_len=0)

@pytest.fixture
def df_multi() -> DataFrame:
    return DataFrame(data=np.arange(16).reshape(4, 4), columns=MultiIndex.from_product([['A', 'B'], ['a', 'b']]), index=MultiIndex.from_product([['X', 'Y'], ['x', 'y']])).rename_axis(['0_0', '0_1'], axis=0).rename_axis(['1_0', '1_1'], axis=1)

@pytest.fixture
def styler_multi(df_multi: DataFrame) -> Styler:
    return Styler(df_multi, uuid_len=0)

def test_display_format(styler: Styler) -> None:
    ctx: Dict[str, Any] = styler.format('{:0.1f}')._translate(True, True)
    assert all((['display_value' in c for c in row] for row in ctx['body']))
    assert all(([len(c['display_value']) <= 3 for c in row[1:]] for row in ctx['body']))
    assert len(ctx['body'][0][1]['display_value'].lstrip('-')) <= 3

@pytest.mark.parametrize('index', [True, False])
@pytest.mark.parametrize('columns', [True, False])
def test_display_format_index(styler: Styler, index: bool, columns: bool) -> None:
    exp_index: List[str] = ['x', 'y']
    if index:
        styler.format_index(lambda v: v.upper(), axis=0)
        exp_index = ['X', 'Y']
    exp_columns: List[str] = ['A', 'B']
    if columns:
        styler.format_index('*{}*', axis=1)
        exp_columns = ['*A*', '*B*']
    ctx: Dict[str, Any] = styler._translate(True, True)
    for r, row in enumerate(ctx['body']):
        assert row[0]['display_value'] == exp_index[r]
    for c, col in enumerate(ctx['head'][1:]):
        assert col['display_value'] == exp_columns[c]

def test_format_dict(styler: Styler) -> None:
    ctx: Dict[str, Any] = styler.format({'A': '{:0.1f}', 'B': '{0:.2%}'})._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '0.0'
    assert ctx['body'][0][2]['display_value'] == '-60.90%'

def test_format_index_dict(styler: Styler) -> None:
    ctx: Dict[str, Any] = styler.format_index({0: lambda v: v.upper()})._translate(True, True)
    for i, val in enumerate(['X', 'Y']):
        assert ctx['body'][i][0]['display_value'] == val

def test_format_string(styler: Styler) -> None:
    ctx: Dict[str, Any] = styler.format('{:.2f}')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '0.00'
    assert ctx['body'][0][2]['display_value'] == '-0.61'
    assert ctx['body'][1][1]['display_value'] == '1.00'
    assert ctx['body'][1][2]['display_value'] == '-1.23'

def test_format_callable(styler: Styler) -> None:
    ctx: Dict[str, Any] = styler.format(lambda v: 'neg' if v < 0 else 'pos')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == 'pos'
    assert ctx['body'][0][2]['display_value'] == 'neg'
    assert ctx['body'][1][1]['display_value'] == 'pos'
    assert ctx['body'][1][2]['display_value'] == 'neg'

def test_format_with_na_rep() -> None:
    df: DataFrame = DataFrame([[None, None], [1.1, 1.2]], columns=['A', 'B'])
    ctx: Dict[str, Any] = df.style.format(None, na_rep='-')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '-'
    assert ctx['body'][0][2]['display_value'] == '-'
    ctx = df.style.format('{:.2%}', na_rep='-')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '-'
    assert ctx['body'][0][2]['display_value'] == '-'
    assert ctx['body'][1][1]['display_value'] == '110.00%'
    assert ctx['body'][1][2]['display_value'] == '120.00%'
    ctx = df.style.format('{:.2%}', na_rep='-', subset=['B'])._translate(True, True)
    assert ctx['body'][0][2]['display_value'] == '-'
    assert ctx['body'][1][2]['display_value'] == '120.00%'

def test_format_index_with_na_rep() -> None:
    df: DataFrame = DataFrame([[1, 2, 3, 4, 5]], columns=['A', None, np.nan, NaT, NA])
    ctx: Dict[str, Any] = df.style.format_index(None, na_rep='--', axis=1)._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'A'
    for i in [2, 3, 4, 5]:
        assert ctx['head'][0][i]['display_value'] == '--'

def test_format_non_numeric_na() -> None:
    df: DataFrame = DataFrame({'object': [None, np.nan, 'foo'], 'datetime': [None, NaT, Timestamp('20120101')]})
    ctx: Dict[str, Any] = df.style.format(None, na_rep='-')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '-'
    assert ctx['body'][0][2]['display_value'] == '-'
    assert ctx['body'][1][1]['display_value'] == '-'
    assert ctx['body'][1][2]['display_value'] == '-'

@pytest.mark.parametrize('func, attr, kwargs', [('format', '_display_funcs', {}), ('format_index', '_display_funcs_index', {'axis': 0}), ('format_index', '_display_funcs_columns', {'axis': 1})])
def test_format_clear(styler: Styler, func: str, attr: str, kwargs: Dict[str, Any]) -> None:
    assert (0, 0) not in getattr(styler, attr)
    getattr(styler, func)('{:.2f}', **kwargs)
    assert (0, 0) in getattr(styler, attr)
    getattr(styler, func)(**kwargs)
    assert (0, 0) not in getattr(styler, attr)

@pytest.mark.parametrize('escape, exp', [('html', '&lt;&gt;&amp;&#34;%$#_{}~^\\~ ^ \\ '), ('latex', '<>\\&"\\%\\$\\#\\_\\{\\}\\textasciitilde \\textasciicircum \\textbackslash \\textasciitilde \\space \\textasciicircum \\space \\textbackslash \\space ')])
def test_format_escape_html(escape: str, exp: str) -> None:
    chars: str = '<>&"%$#_{}~^\\~ ^ \\ '
    df: DataFrame = DataFrame([[chars]])
    s: Styler = Styler(df, uuid_len=0).format('&{0}&', escape=None)
    expected: str = f'<td id="T__row0_col0" class="data row0 col0" >&{chars}&</td>'
    assert expected in s.to_html()
    s = Styler(df, uuid_len=0).format('&{0}&', escape=escape)
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{exp}&</td>'
    assert expected in s.to_html()
    styler: Styler = Styler(DataFrame(columns=[chars]), uuid_len=0)
    styler.format_index('&{0}&', escape=None, axis=1)
    assert styler._translate(True, True)['head'][0][1]['display_value'] == f'&{chars}&'
    styler.format_index('&{0}&', escape=escape, axis=1)
    assert styler._translate(True, True)['head'][0][1]['display_value'] == f'&{exp}&'

@pytest.mark.parametrize('chars, expected', [('$ \\$&%#_{}~^\\ $ &%#_{}~^\\ $', ''.join(['$ \\$&%#_{}~^\\ $ ', '\\&\\%\\#\\_\\{\\}\\textasciitilde \\textasciicircum ', '\\textbackslash \\space \\$'])), ('\\( &%#_{}~^\\ \\) &%#_{}~^\\ \\(', ''.join(['\\( &%#_{}~^\\ \\) ', '\\&\\%\\#\\_\\{\\}\\textasciitilde \\textasciicircum ', '\\textbackslash \\space \\textbackslash ('])), ('$\\&%#_{}^\\$', '\\$\\textbackslash \\&\\%\\#\\_\\{\\}\\textasciicircum \\textbackslash \\$'), ('$ \\frac{1}{2} $ \\( \\frac{1}{2} \\)', ''.join(['$ \\frac{1}{2} $', ' \\textbackslash ( \\textbackslash frac\\{1\\}\\{2\\} \\textbackslash )']))])
def test_format_escape_latex_math(chars: str, expected: str) -> None:
    df: DataFrame = DataFrame([[chars]])
    s: Styler = df.style.format('{0}', escape='latex-math')
    assert s._translate(True, True)['body'][0][1]['display_value'] == expected

def test_format_escape_na_rep() -> None:
    df: DataFrame = DataFrame([['<>&"', None]])
    s: Styler = Styler(df, uuid_len=0).format('X&{0}>X', escape='html', na_rep='&')
    ex: str = '<td id="T__row0_col0" class="data row0 col0" >X&&lt;&gt;&amp;&#34;>X</td>'
    expected2: str = '<td id="T__row0_col1" class="data row0 col1" >&</td>'
    assert ex in s.to_html()
    assert expected2 in s.to_html()
    df = DataFrame(columns=['<>&"', None])
    styler: Styler = Styler(df, uuid_len=0)
    styler.format_index('X&{0}>X', escape='html', na_rep='&', axis=1)
    ctx: Dict[str, Any] = styler._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'X&&lt;&gt;&amp;&#34;>X'
    assert ctx['head'][0][2]['display_value'] == '&'

def test_format_escape_floats(styler: Styler) -> None:
    s: Styler = styler.format('{:.1f}', escape='html')
    for expected in ['>0.0<', '>1.0<', '>-1.2<', '>-0.6<']:
        assert expected in s.to_html()
    s = styler.format(precision=1, escape='html')
    for expected in ['>0<', '>1<', '>-1.2<', '>-0.6<']:
        assert expected in s.to_html()

@pytest.mark.parametrize('formatter', [5, True, [2.0]])
@pytest.mark.parametrize('func', ['format', 'format_index'])
def test_format_raises(styler: Styler, formatter: Any, func: str) -> None:
    with pytest.raises(TypeError, match='expected str or callable'):
        getattr(styler, func)(formatter)

@pytest.mark.parametrize('precision, expected', [(1, ['1.0', '2.0', '3.2', '4.6']), (2, ['1.00', '2.01', '3.21', '4.57']), (3, ['1.000', '2.009', '3.212', '4.566'])])
def test_format_with_precision(precision: int, expected: List[str]) -> None:
    df: DataFrame = DataFrame([[1.0, 2.009, 3.2121, 4.566]], columns=[1.0, 2.009, 3.2121, 4.566])
    styler: Styler = Styler(df)
    styler.format(precision=precision)
    styler.format_index(precision=precision, axis=1)
    ctx: Dict[str, Any] = styler._translate(True, True)
    for col, exp in enumerate(expected):
        assert ctx['body'][0][col + 1]['display_value'] == exp
        assert ctx['head'][0][col + 1]['display_value'] == exp

@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('level, expected', [(0, ['X', 'X', '_', '_']), ('zero', ['X', 'X', '_', '_']), (1, ['_', '_', 'X', 'X']), ('one', ['_', '_', 'X', 'X']), ([0, 1], ['X', 'X', 'X', 'X']), ([0, 'zero'], ['X', 'X', '_', '_']), ([0, 'one'], ['X', 'X', 'X', 'X']), (['one', 'zero'], ['X', 'X', 'X', 'X'])])
def test_format_index_level(axis: int, level: Union[int, str, List[Union[int, str]]], expected: List[str]) -> None:
    midx: MultiIndex = MultiIndex.from_arrays([['_', '_'], ['_', '_']], names=['zero', 'one'])
    df: DataFrame = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx
    styler: Styler = df.style.format_index(lambda v: 'X', level=level, axis=axis)
    ctx: Dict[str, Any] = styler._translate(True, True)
    if axis == 0:
        result: List[str] = [ctx['body'][s][0]['display_value'] for s in range(2)]
        result += [ctx['body'][s][1]['display_value'] for s in range(2)]
    else:
        result = [ctx['head'][0][s + 1]['display_value'] for s in range(2)]
        result += [ctx['head'][1][s + 1]['display_value'] for s in range(2)]
    assert expected == result

def test_format_subset() -> None:
    df: DataFrame = DataFrame([[0.1234, 0.1234], [1.1234, 1.1234]], columns=['a', 'b'])
    ctx: Dict[str, Any] = df.style.format({'a': '{:0.1f}', 'b': '{0:.2%}'}, subset=IndexSlice[0, :])._translate(True, True)
    expected: str = '0.1'
    raw_11: str = '1.123400'
    assert ctx['body'][0][1]['display_value'] == expected
    assert ctx['body'][1][1]['display_value'] == raw_11
    assert ctx['body'][0][2]['display_value'] == '12.34%'
    ctx = df.style.format('{:0.1f}', subset=IndexSlice[0, :])._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == expected
    assert ctx['body'][1][1]['display_value'] == raw_11
    ctx = df.style.format('{:0.1f}', subset=IndexSlice['a'])._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == expected
    assert ctx['body'][0][2]['display_value'] == '0.123400'
    ctx = df.style.format('{:0.1f}', subset=IndexSlice[0, 'a'])._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == expected
    assert ctx['body'][1][1]['display_value'] == raw_11
    ctx = df.style.format('{:0.1f}', subset=IndexSlice[[0, 1], ['a']])._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == expected
    assert ctx['body'][1][1]['display_value'] == '1.1'
    assert ctx['body'][0][2]['display_value'] == '0.123400'
    assert ctx['body'][1][2]['display_value'] == raw_11

@pytest.mark.parametrize('formatter', [None, '{