from __future__ import annotations

import numpy as np
import pytest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pandas import NA, DataFrame, IndexSlice, MultiIndex, NaT, Timestamp, option_context
pytest.importorskip('jinja2')
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
    return DataFrame(
        data=np.arange(16).reshape(4, 4),
        columns=MultiIndex.from_product([['A', 'B'], ['a', 'b']]),
        index=MultiIndex.from_product([['X', 'Y'], ['x', 'y']]),
    ).rename_axis(['0_0', '0_1'], axis=0).rename_axis(['1_0', '1_1'], axis=1)


@pytest.fixture
def styler_multi(df_multi: DataFrame) -> Styler:
    return Styler(df_multi, uuid_len=0)


def test_display_format(styler: Styler) -> None:
    ctx = styler.format('{:0.1f}')._translate(True, True)
    assert all((['display_value' in c for c in row] for row in ctx['body']))
    assert all(([len(c['display_value']) <= 3 for c in row[1:]] for row in ctx['body']))
    assert len(ctx['body'][0][1]['display_value'].lstrip('-')) <= 3


@pytest.mark.parametrize('index', [True, False])
@pytest.mark.parametrize('columns', [True, False])
def test_display_format_index(styler: Styler, index: bool, columns: bool) -> None:
    exp_index = ['x', 'y']
    if index:
        styler.format_index(lambda v: v.upper(), axis=0)
        exp_index = ['X', 'Y']
    exp_columns = ['A', 'B']
    if columns:
        styler.format_index('*{}*', axis=1)
        exp_columns = ['*A*', '*B*']
    ctx = styler._translate(True, True)
    for r, row in enumerate(ctx['body']):
        assert row[0]['display_value'] == exp_index[r]
    for c, col in enumerate(ctx['head'][1:]):
        assert col['display_value'] == exp_columns[c]


def test_format_dict(styler: Styler) -> None:
    ctx = styler.format({'A': '{:0.1f}', 'B': '{0:.2%}'})._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '0.0'
    assert ctx['body'][0][2]['display_value'] == '-60.90%'


def test_format_index_dict(styler: Styler) -> None:
    ctx = styler.format_index({0: lambda v: v.upper()})._translate(True, True)
    for i, val in enumerate(['X', 'Y']):
        assert ctx['body'][i][0]['display_value'] == val


def test_format_string(styler: Styler) -> None:
    ctx = styler.format('{:.2f}')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '0.00'
    assert ctx['body'][0][2]['display_value'] == '-0.61'
    assert ctx['body'][1][1]['display_value'] == '1.00'
    assert ctx['body'][1][2]['display_value'] == '-1.23'


def test_format_callable(styler: Styler) -> None:
    ctx = styler.format(lambda v: 'neg' if v < 0 else 'pos')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == 'pos'
    assert ctx['body'][0][2]['display_value'] == 'neg'
    assert ctx['body'][1][1]['display_value'] == 'pos'
    assert ctx['body'][1][2]['display_value'] == 'neg'


def test_format_with_na_rep() -> None:
    df = DataFrame([[None, None], [1.1, 1.2]], columns=['A', 'B'])
    ctx = df.style.format(None, na_rep='-')._translate(True, True)
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
    df = DataFrame([[1, 2, 3, 4, 5]], columns=['A', None, np.nan, NaT, NA])
    ctx = df.style.format_index(None, na_rep='--', axis=1)._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'A'
    for i in [2, 3, 4, 5]:
        assert ctx['head'][0][i]['display_value'] == '--'


def test_format_non_numeric_na() -> None:
    df = DataFrame({'object': [None, np.nan, 'foo'], 'datetime': [None, NaT, Timestamp('20120101')]})
    ctx = df.style.format(None, na_rep='-')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '-'
    assert ctx['body'][0][2]['display_value'] == '-'
    assert ctx['body'][1][1]['display_value'] == '-'
    assert ctx['body'][1][2]['display_value'] == '-'


@pytest.mark.parametrize(
    'func, attr, kwargs',
    [
        ('format', '_display_funcs', {}),
        ('format_index', '_display_funcs_index', {'axis': 0}),
        ('format_index', '_display_funcs_columns', {'axis': 1}),
    ],
)
def test_format_clear(styler: Styler, func: str, attr: str, kwargs: Dict[str, Any]) -> None:
    assert (0, 0) not in getattr(styler, attr)
    getattr(styler, func)('{:.2f}', **kwargs)
    assert (0, 0) in getattr(styler, attr)
    getattr(styler, func)(**kwargs)
    assert (0, 0) not in getattr(styler, attr)


@pytest.mark.parametrize(
    'escape, exp',
    [
        ('html', '&lt;&gt;&amp;&#34;%$#_{}~^\\~ ^ \\ '),
        ('latex', '<>\\&"\\%\\$\\#\\_\\{\\}\\textasciitilde \\textasciicircum \\textbackslash \\textasciitilde \\space \\textasciicircum \\space \\textbackslash \\space '),
    ],
)
def test_format_escape_html(escape: str, exp: str) -> None:
    chars = '<>&"%$#_{}~^\\~ ^ \\ '
    df = DataFrame([[chars]])
    s = Styler(df, uuid_len=0).format('&{0}&', escape=None)
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{chars}&</td>'
    assert expected in s.to_html()
    s = Styler(df, uuid_len=0).format('&{0}&', escape=escape)
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{exp}&</td>'
    assert expected in s.to_html()
    styler = Styler(DataFrame(columns=[chars]), uuid_len=0)
    styler.format_index('&{0}&', escape=None, axis=1)
    assert styler._translate(True, True)['head'][0][1]['display_value'] == f'&{chars}&'
    styler.format_index('&{0}&', escape=escape, axis=1)
    assert styler._translate(True, True)['head'][0][1]['display_value'] == f'&{exp}&'


@pytest.mark.parametrize(
    'chars, expected',
    [
        (
            '$ \\$&%#_{}~^\\ $ &%#_{}~^\\ $',
            ''.join(['$ \\$&%#_{}~^\\ $ ', '\\&\\%\\#\\_\\{\\}\\textasciitilde \\textasciicircum ', '\\textbackslash \\space \\$']),
        ),
        (
            '\\( &%#_{}~^\\ \\) &%#_{}~^\\ \\(',
            ''.join(['\\( &%#_{}~^\\ \\) ', '\\&\\%\\#\\_\\{\\}\\textasciitilde \\textasciicircum ', '\\textbackslash \\space \\textbackslash (']),
        ),
        (
            '$\\&%#_{}^\\$',
            '\\$\\textbackslash \\&\\%\\#\\_\\{\\}\\textasciitilde \\textasciicircum \\textbackslash \\$',
        ),
        (
            '$ \\frac{1}{2} $ \\( \\frac{1}{2} \\)',
            ''.join(['$ \\frac{1}{2} $', ' \\textbackslash ( \\textbackslash frac\\{1\\}\\{2\\} \\textbackslash )']),
        ),
    ],
)
def test_format_escape_latex_math(chars: str, expected: str) -> None:
    df = DataFrame([[chars]])
    s = df.style.format('{0}', escape='latex-math')
    assert s._translate(True, True)['body'][0][1]['display_value'] == expected


def test_format_escape_na_rep() -> None:
    df = DataFrame([['<>&"', None]])
    s = Styler(df, uuid_len=0).format('X&{0}>X', escape='html', na_rep='&')
    ex = '<td id="T__row0_col0" class="data row0 col0" >X&&lt;&gt;&amp;&#34;>X</td>'
    expected2 = '<td id="T__row0_col1" class="data row0 col1" >&</td>'
    assert ex in s.to_html()
    assert expected2 in s.to_html()
    df = DataFrame(columns=['<>&"', None])
    styler = Styler(df, uuid_len=0)
    styler.format_index('X&{0}>X', escape='html', na_rep='&', axis=1)
    ctx = styler._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'X&&lt;&gt;&amp;&#34;>X'
    assert ctx['head'][0][2]['display_value'] == '&'


def test_format_escape_floats(styler: Styler) -> None:
    s = styler.format('{:.1f}', escape='html')
    for expected in ['>0.0<', '>1.0<', '>-1.2<', '>-0.6<']:
        assert expected in s.to_html()
    s = styler.format(precision=1, escape='html')
    for expected in ['>0<', '>1<', '>-1.2<', '>-0.6<']:
        assert expected in s.to_html()


@pytest.mark.parametrize('formatter', [5, True, [2.0]])
@pytest.mark.parametrize('func', ['format', 'format_index'])
def test_format_raises(styler: Styler, formatter: object, func: str) -> None:
    with pytest.raises(TypeError, match='expected str or callable'):
        getattr(styler, func)(formatter)


@pytest.mark.parametrize(
    'precision, expected',
    [
        (1, ['1.0', '2.0', '3.2', '4.6']),
        (2, ['1.00', '2.01', '3.21', '4.57']),
        (3, ['1.000', '2.009', '3.212', '4.566']),
    ],
)
def test_format_with_precision(precision: int, expected: List[str]) -> None:
    df = DataFrame([[1.0, 2.009, 3.2121, 4.566]], columns=[1.0, 2.009, 3.2121, 4.566])
    styler = Styler(df)
    styler.format(precision=precision)
    styler.format_index(precision=precision, axis=1)
    ctx = styler._translate(True, True)
    for col, exp in enumerate(expected):
        assert ctx['body'][0][col + 1]['display_value'] == exp
        assert ctx['head'][0][col + 1]['display_value'] == exp


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize(
    'level, expected',
    [
        (0, ['X', 'X', '_', '_']),
        ('zero', ['X', 'X', '_', '_']),
        (1, ['_', '_', 'X', 'X']),
        ('one', ['_', '_', 'X', 'X']),
        ([0, 1], ['X', 'X', 'X', 'X']),
        ([0, 'zero'], ['X', 'X', '_', '_']),
        ([0, 'one'], ['X', 'X', 'X', 'X']),
        (['one', 'zero'], ['X', 'X', 'X', 'X']),
    ],
)
def test_format_index_level(
    axis: int, level: Union[int, str, List[Union[int, str]]], expected: List[str]
) -> None:
    midx = MultiIndex.from_arrays([['_', '_'], ['_', '_']], names=['zero', 'one'])
    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx
    styler = df.style.format_index(lambda v: 'X', level=level, axis=axis)
    ctx = styler._translate(True, True)
    if axis == 0:
        result = [ctx['body'][s][0]['display_value'] for s in range(2)]
        result += [ctx['body'][s][1]['display_value'] for s in range(2)]
    else:
        result = [ctx['head'][0][s + 1]['display_value'] for s in range(2)]
        result += [ctx['head'][1][s + 1]['display_value'] for s in range(2)]
    assert expected == result


def test_format_subset() -> None:
    df = DataFrame([[0.1234, 0.1234], [1.1234, 1.1234]], columns=['a', 'b'])
    ctx = df.style.format({'a': '{:0.1f}', 'b': '{0:.2%}'}, subset=IndexSlice[0, :])._translate(True, True)
    expected = '0.1'
    raw_11 = '1.123400'
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


@pytest.mark.parametrize('formatter', [None, '{:,.1f}'])
@pytest.mark.parametrize('decimal', ['.', '*'])
@pytest.mark.parametrize('precision', [None, 2])
@pytest.mark.parametrize('func, col', [('format', 1), ('format_index', 0)])
def test_format_thousands(
    formatter: Optional[str],
    decimal: str,
    precision: Optional[int],
    func: str,
    col: int,
) -> None:
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    result = getattr(styler, func)(thousands='_', formatter=formatter, decimal=decimal, precision=precision)._translate(True, True)
    assert '1_000_000' in result['body'][0][col]['display_value']
    styler = DataFrame([[1000000]], index=[1000000]).style
    result = getattr(styler, func)(thousands='_', formatter=formatter, decimal=decimal, precision=precision)._translate(True, True)
    assert '1_000_000' in result['body'][0][col]['display_value']
    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    result = getattr(styler, func)(thousands='_', formatter=formatter, decimal=decimal, precision=precision)._translate(True, True)
    assert '1_000_000' in result['body'][0][col]['display_value']


@pytest.mark.parametrize('formatter', [None, '{:,.4f}'])
@pytest.mark.parametrize('thousands', [None, ',', '*'])
@pytest.mark.parametrize('precision', [None, 4])
@pytest.mark.parametrize('func, col', [('format', 1), ('format_index', 0)])
def test_format_decimal(
    formatter: Optional[str],
    thousands: Optional[str],
    precision: Optional[int],
    func: str,
    col: int,
) -> None:
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    result = getattr(styler, func)(decimal='_', formatter=formatter, thousands=thousands, precision=precision)._translate(True, True)
    assert '000_123' in result['body'][0][col]['display_value']
    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    result = getattr(styler, func)(decimal='_', formatter=formatter, thousands=thousands, precision=precision)._translate(True, True)
    assert '000_123' in result['body'][0][col]['display_value']


def test_str_escape_error() -> None:
    msg = "`escape` only permitted in {'html', 'latex', 'latex-math'}, got "
    with pytest.raises(ValueError, match=msg):
        _str_escape('text', 'bad_escape')
    with pytest.raises(ValueError, match=msg):
        _str_escape('text', [])
    _str_escape(2.0, 'bad_escape')


def test_long_int_formatting() -> None:
    df = DataFrame(data=[[1234567890123456789]], columns=['test'])
    styler = df.style
    ctx = styler._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '1234567890123456789'
    styler = df.style.format(thousands='_')
    ctx = styler._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '1_234_567_890_123_456_789'


def test_format_options() -> None:
    df = DataFrame({'int': [2000, 1], 'float': [1.009, None], 'str': ['&<', '&~']})
    ctx = df.style._translate(True, True)
    assert ctx['body'][1][2]['display_value'] == 'nan'
    with option_context('styler.format.na_rep', 'MISSING'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][1][2]['display_value'] == 'MISSING'
    assert ctx['body'][0][2]['display_value'] == '1.009000'
    with option_context('styler.format.decimal', '_'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][2]['display_value'] == '1_009000'
    with option_context('styler.format.precision', 2):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][2]['display_value'] == '1.01'
    assert ctx['body'][0][1]['display_value'] == '2000'
    with option_context('styler.format.thousands', '_'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][1]['display_value'] == '2_000'
    assert ctx['body'][0][3]['display_value'] == '&<'
    assert ctx['body'][1][3]['display_value'] == '&~'
    with option_context('styler.format.escape', 'html'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][3]['display_value'] == '&amp;&lt;'
    with option_context('styler.format.escape', 'latex'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][1][3]['display_value'] == '\\&\\textasciitilde '
    with option_context('styler.format.escape', 'latex-math'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][1][3]['display_value'] == '\\&\\textasciitilde '
    with option_context('styler.format.formatter', {'int': '{:,.2f}'}):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][1]['display_value'] == '2,000.00'


def test_precision_zero(df: DataFrame) -> None:
    styler = Styler(df, precision=0)
    ctx = styler._translate(True, True)
    assert ctx['body'][0][2]['display_value'] == '-1'
    assert ctx['body'][1][2]['display_value'] == '-1'


@pytest.mark.parametrize(
    'formatter, exp',
    [
        (lambda x: f'{x:.3f}', '9.000'),
        ('{:.2f}', '9.00'),
        ({0: '{:.1f}'}, '9.0'),
        (None, '9'),
    ],
)
def test_formatter_options_validator(
    formatter: Union[Callable[[Any], str], str, Dict[int, str], None],
    exp: str,
) -> None:
    df = DataFrame([[9]])
    with option_context('styler.format.formatter', formatter):
        assert f' {exp} ' in df.style.to_latex()


def test_formatter_options_raises() -> None:
    msg = 'Value must be an instance of'
    with pytest.raises(ValueError, match=msg):
        with option_context('styler.format.formatter', ['bad', 'type']):
            DataFrame().style.to_latex()


def test_1level_multiindex() -> None:
    midx = MultiIndex.from_product([[1, 2]], names=[''])
    df = DataFrame(-1, index=midx, columns=[0, 1])
    ctx = df.style._translate(True, True)
    assert ctx['body'][0][0]['display_value'] == '1'
    assert ctx['body'][0][0]['is_visible'] is True
    assert ctx['body'][1][0]['display_value'] == '2'
    assert ctx['body'][1][0]['is_visible'] is True


def test_boolean_format() -> None:
    df = DataFrame([[True, False]])
    ctx = df.style._translate(True, True)
    assert ctx['body'][0][1]['display_value'] is True
    assert ctx['body'][0][2]['display_value'] is False


@pytest.mark.parametrize('hide, labels', [(False, [1, 2]), (True, [1, 2, 3, 4])])
def test_relabel_raise_length(styler_multi: Styler, hide: bool, labels: List[int]) -> None:
    if hide:
        styler_multi.hide(axis=0, subset=[('X', 'x'), ('Y', 'y')])
    with pytest.raises(ValueError, match='``labels`` must be of length equal'):
        styler_multi.relabel_index(labels=labels)


def test_relabel_index(styler_multi: Styler) -> None:
    labels: List[Tuple[int, int]] = [(1, 2), (3, 4)]
    styler_multi.hide(axis=0, subset=[('X', 'x'), ('Y', 'y')])
    styler_multi.relabel_index(labels=labels)
    ctx = styler_multi._translate(True, True)
    assert {'value': 'X', 'display_value': 1}.items() <= ctx['body'][0][0].items()
    assert {'value': 'y', 'display_value': 2}.items() <= ctx['body'][0][1].items()
    assert {'value': 'Y', 'display_value': 3}.items() <= ctx['body'][1][0].items()
    assert {'value': 'x', 'display_value': 4}.items() <= ctx['body'][1][1].items()


def test_relabel_columns(styler_multi: Styler) -> None:
    labels: List[Tuple[int, int]] = [(1, 2), (3, 4)]
    styler_multi.hide(axis=1, subset=[('A', 'a'), ('B', 'b')])
    styler_multi.relabel_index(axis=1, labels=labels)
    ctx = styler_multi._translate(True, True)
    assert {'value': 'A', 'display_value': 1}.items() <= ctx['head'][0][3].items()
    assert {'value': 'B', 'display_value': 3}.items() <= ctx['head'][0][4].items()
    assert {'value': 'b', 'display_value': 2}.items() <= ctx['head'][1][3].items()
    assert {'value': 'a', 'display_value': 4}.items() <= ctx['head'][1][4].items()


def test_relabel_roundtrip(styler: Styler) -> None:
    styler.relabel_index(['{}', '{}'])
    ctx = styler._translate(True, True)
    assert {'value': 'x', 'display_value': 'x'}.items() <= ctx['body'][0][0].items()
    assert {'value': 'y', 'display_value': 'y'}.items() <= ctx['body'][1][0].items()


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize(
    'level, expected',
    [
        (0, ['X', 'one']),
        ('zero', ['X', 'one']),
        (1, ['zero', 'X']),
        ('one', ['zero', 'X']),
        ([0, 1], ['X', 'X']),
        ([0, 'zero'], ['X', 'one']),
        ([0, 'one'], ['X', 'X']),
        (['one', 'zero'], ['X', 'X']),
    ],
)
def test_format_index_names_level(
    axis: int, level: Union[int, str, List[Union[int, str]]], expected: List[str]
) -> None:
    midx = MultiIndex.from_arrays([['_', '_'], ['_', '_']], names=['zero', 'one'])
    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx
    styler = df.style.format_index_names(lambda v: 'X', level=level, axis=axis)
    ctx = styler._translate(True, True)
    if axis == 0:
        result = [ctx['head'][1][s]['display_value'] for s in range(2)]
    else:
        result = [ctx['head'][s][0]['display_value'] for s in range(2)]
    assert expected == result


@pytest.mark.parametrize('attr, kwargs', [('_display_funcs_index_names', {'axis': 0}), ('_display_funcs_column_names', {'axis': 1})])
def test_format_index_names_clear(styler: Styler, attr: str, kwargs: Dict[str, Any]) -> None:
    assert 0 not in getattr(styler, attr)
    styler.format_index_names('{:.2f}', **kwargs)
    assert 0 in getattr(styler, attr)
    styler.format_index_names(**kwargs)
    assert 0 not in getattr(styler, attr)


@pytest.mark.parametrize('axis', [0, 1])
def test_format_index_names_callable(styler_multi: Styler, axis: int) -> None:
    ctx = styler_multi.format_index_names(lambda v: v.replace('_', 'A'), axis=axis)._translate(True, True)
    result = [
        ctx['head'][2][0]['display_value'],
        ctx['head'][2][1]['display_value'],
        ctx['head'][0][1]['display_value'],
        ctx['head'][1][1]['display_value'],
    ]
    if axis == 0:
        expected = ['0A0', '0A1', '1_0', '1_1']
    else:
        expected = ['0_0', '0_1', '1A0', '1A1']
    assert result == expected


def test_format_index_names_dict(styler_multi: Styler) -> None:
    ctx = styler_multi.format_index_names({'0_0': '{:<<5}'}).format_index_names({'1_1': '{:>>4}'}, axis=1)._translate(True, True)
    assert ctx['head'][2][0]['display_value'] == '0_0<<'
    assert ctx['head'][1][1]['display_value'] == '>1_1'


def test_format_index_names_with_hidden_levels(styler_multi: Styler) -> None:
    ctx = styler_multi._translate(True, True)
    full_head_height = len(ctx['head'])
    full_head_width = len(ctx['head'][0])
    assert full_head_height == 3
    assert full_head_width == 6
    ctx = (
        styler_multi.hide(axis=0, level=1)
        .hide(axis=1, level=1)
        .format_index_names('{:>>4}', axis=1)
        .format_index_names('{:!<5}')
        ._translate(True, True)
    )
    assert len(ctx['head']) == full_head_height - 1
    assert len(ctx['head'][0]) == full_head_width - 1
    assert ctx['head'][0][0]['display_value'] == '>1_0'
    assert ctx['head'][1][0]['display_value'] == '0_0!!'