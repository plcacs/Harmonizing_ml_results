from textwrap import dedent
import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series, option_context
pytest.importorskip('jinja2')
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
    _parse_latex_cell_styles,
    _parse_latex_css_conversion,
    _parse_latex_header_span,
    _parse_latex_table_styles,
    _parse_latex_table_wrapping,
)


@pytest.fixture
def df() -> DataFrame:
    return DataFrame({'A': [0, 1], 'B': [-0.61, -1.22], 'C': Series(['ab', 'cd'], dtype=object)})


@pytest.fixture
def df_ext() -> DataFrame:
    return DataFrame({'A': [0, 1, 2], 'B': [-0.61, -1.22, -2.22], 'C': ['ab', 'cd', 'de']})


@pytest.fixture
def styler(df: DataFrame) -> Styler:
    return Styler(df, uuid_len=0, precision=2)


def test_minimal_latex_tabular(styler: Styler) -> None:
    expected = dedent('        \\begin{tabular}{lrrl}\n         & A & B & C \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    assert styler.to_latex() == expected


def test_tabular_hrules(styler: Styler) -> None:
    expected = dedent('        \\begin{tabular}{lrrl}\n        \\toprule\n         & A & B & C \\\\\n        \\midrule\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\bottomrule\n        \\end{tabular}\n        ')
    assert styler.to_latex(hrules=True) == expected


def test_tabular_custom_hrules(styler: Styler) -> None:
    styler.set_table_styles([{'selector': 'toprule', 'props': ':hline'}, {'selector': 'bottomrule', 'props': ':otherline'}])
    expected = dedent('        \\begin{tabular}{lrrl}\n        \\hline\n         & A & B & C \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\otherline\n        \\end{tabular}\n        ')
    assert styler.to_latex() == expected


def test_column_format(styler: Styler) -> None:
    styler.set_table_styles([{'selector': 'column_format', 'props': ':cccc'}])
    assert '\\begin{tabular}{rrrr}' in styler.to_latex(column_format='rrrr')
    styler.set_table_styles([{'selector': 'column_format', 'props': ':r|r|cc'}])
    assert '\\begin{tabular}{r|r|cc}' in styler.to_latex()


def test_siunitx_cols(styler: Styler) -> None:
    expected = dedent('        \\begin{tabular}{lSSl}\n        {} & {A} & {B} & {C} \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    assert styler.to_latex(siunitx=True) == expected


def test_position(styler: Styler) -> None:
    assert '\\begin{table}[h!]' in styler.to_latex(position='h!')
    assert '\\end{table}' in styler.to_latex(position='h!')
    styler.set_table_styles([{'selector': 'position', 'props': ':b!'}])
    assert '\\begin{table}[b!]' in styler.to_latex()
    assert '\\end{table}' in styler.to_latex()


@pytest.mark.parametrize('env', [None, 'longtable'])
def test_label(styler: Styler, env: str) -> None:
    assert '\n\\label{text}' in styler.to_latex(label='text', environment=env)
    styler.set_table_styles([{'selector': 'label', 'props': ':{more §text}'}])
    assert '\n\\label{more :text}' in styler.to_latex(environment=env)


def test_position_float_raises(styler: Styler) -> None:
    msg = "`position_float` should be one of 'raggedright', 'raggedleft', 'centering',"
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(position_float='bad_string')
    msg = "`position_float` cannot be used in 'longtable' `environment`"
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(position_float='centering', environment='longtable')


@pytest.mark.parametrize('label', [(None, ''), ('text', '\\label{text}')])
@pytest.mark.parametrize('position', [(None, ''), ('h!', '{table}[h!]')])
@pytest.mark.parametrize('caption', [(None, ''), ('text', '\\caption{text}')])
@pytest.mark.parametrize('column_format', [(None, ''), ('rcrl', '{tabular}{rcrl}')])
@pytest.mark.parametrize('position_float', [(None, ''), ('centering', '\\centering')])
def test_kwargs_combinations(
    styler: Styler, label: tuple[str, str], position: tuple[str, str], caption: tuple[str, str], column_format: tuple[str, str], position_float: tuple[str, str]
) -> None:
    result = styler.to_latex(
        label=label[0], position=position[0], caption=caption[0], column_format=column_format[0], position_float=position_float[0]
    )
    assert label[1] in result
    assert position[1] in result
    assert caption[1] in result
    assert column_format[1] in result
    assert position_float[1] in result


def test_custom_table_styles(styler: Styler) -> None:
    styler.set_table_styles([{'selector': 'mycommand', 'props': ':{myoptions}'}, {'selector': 'mycommand2', 'props': ':{myoptions2}'}])
    expected = dedent('        \\begin{table}\n        \\mycommand{myoptions}\n        \\mycommand2{myoptions2}\n        ')
    assert expected in styler.to_latex()


def test_cell_styling(styler: Styler) -> None:
    styler.highlight_max(props='itshape:;Huge:--wrap;')
    expected = dedent('        \\begin{tabular}{lrrl}\n         & A & B & C \\\\\n        0 & 0 & \\itshape {\\Huge -0.61} & ab \\\\\n        1 & \\itshape {\\Huge 1} & -1.22 & \\itshape {\\Huge cd} \\\\\n        \\end{tabular}\n        ')
    assert expected == styler.to_latex()


def test_multiindex_columns(df: DataFrame) -> None:
    cidx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df.columns = cidx
    expected = dedent('        \\begin{tabular}{lrrl}\n         & \\multicolumn{2}{r}{A} & B \\\\\n         & a & b & c \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    s = df.style.format(precision=2)
    assert expected == s.to_latex()
    expected = dedent('        \\begin{tabular}{lrrl}\n         & A & A & B \\\\\n         & a & b & c \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    s = df.style.format(precision=2)
    assert expected == s.to_latex(sparse_columns=False)


def test_multiindex_row(df_ext: DataFrame) -> None:
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index = ridx
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n         & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex()
    assert expected == result
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        A & a & 0 & -0.61 & ab \\\\\n        A & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    result = styler.to_latex(sparse_index=False)
    assert expected == result


def test_multirow_naive(df_ext: DataFrame) -> None:
    ridx = MultiIndex.from_tuples([('X', 'x'), ('X', 'y'), ('Y', 'z')])
    df_ext.index = ridx
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        X & x & 0 & -0.61 & ab \\\\\n         & y & 1 & -1.22 & cd \\\\\n        Y & z & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex(multirow_align='naive')
    assert expected == result


def test_multiindex_row_and_col(df_ext: DataFrame) -> None:
    cidx = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & \\multicolumn{2}{l}{Z} & Y \\\\\n         &  & a & b & c \\\\\n        \\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n         & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex(multirow_align='b', multicol_align='l')
    assert result == expected
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & Z & Z & Y \\\\\n         &  & a & b & c \\\\\n        A & a & 0 & -0.61 & ab \\\\\n        A & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    result = styler.to_latex(sparse_index=False, sparse_columns=False)
    assert result == expected


@pytest.mark.parametrize('multicol_align, siunitx, header', [('naive-l', False, ' & A & &'), ('naive-r', False, ' & & & A'), ('naive-l', True, '{} & {A} & {} & {}'), ('naive-r', True, '{} & {} & {} & {A}')])
def test_multicol_naive(
    df: DataFrame, multicol_align: str, siunitx: bool, header: str
) -> None:
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')])
    df.columns = ridx
    level1 = ' & a & b & c' if not siunitx else '{} & {a} & {b} & {c}'
    col_format = 'lrrl' if not siunitx else 'lSSl'
    expected = dedent(f'        \\begin{{tabular}}{{{col_format}}}\n        {header} \\\\\n        {level1} \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{{tabular}}\n        ')
    styler = df.style.format(precision=2)
    result = styler.to_latex(multicol_align=multicol_align, siunitx=siunitx)
    assert expected == result


def test_multi_options(df_ext: DataFrame) -> None:
    cidx = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler = df_ext.style.format(precision=2)
    expected = dedent('     &  & \\multicolumn{2}{r}{Z} & Y \\\\\n     &  & a & b & c \\\\\n    \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n    ')
    result = styler.to_latex()
    assert expected in result
    with option_context('styler.latex.multicol_align', 'l'):
        assert ' &  & \\multicolumn{2}{l}{Z} & Y \\\\' in styler.to_latex()
    with option_context('styler.latex.multirow_align', 'b'):
        assert '\\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\' in styler.to_latex()


def test_multiindex_columns_hidden() -> None:
    df = DataFrame([[1, 2, 3, 4]])
    df.columns = MultiIndex.from_tuples([('A', 1), ('A', 2), ('A', 3), ('B', 1)])
    s = df.style
    assert '{tabular}{lrrrr}' in s.to_latex()
    s.set_table_styles([])
    s.hide([('A', 2)], axis='columns')
    assert '{tabular}{lrrr}' in s.to_latex()


@pytest.mark.parametrize('option, value', [('styler.sparse.index', True), ('styler.sparse.index', False), ('styler.sparse.columns', True), ('styler.sparse.columns', False)])
def test_sparse_options(df_ext: DataFrame, option: str, value: bool) -> None:
    cidx = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler = df_ext.style
    latex1 = styler.to_latex()
    with option_context(option, value):
        latex2 = styler.to_latex()
    assert (latex1 == latex2) is value


def test_hidden_index(styler: Styler) -> None:
    styler.hide(axis='index')
    expected = dedent('        \\begin{tabular}{rrl}\n        A & B & C \\\\\n        0 & -0.61 & ab \\\\\n        1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    assert styler.to_latex() == expected


@pytest.mark.parametrize('environment', ['table', 'figure*', None])
def test_comprehensive(df_ext: DataFrame, environment: str) -> None:
    cidx = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    stlr = df_ext.style
    stlr.set_caption('mycap')
    stlr.set_table_styles([{'selector': 'label', 'props': ':{fig§item}'}, {'selector': 'position', 'props': ':h!'}, {'selector': 'position_float', 'props': ':centering'}, {'selector': 'column_format', 'props': ':rlrlr'}, {'selector': 'toprule', 'props': ':toprule'}, {'selector': 'midrule', 'props': ':midrule'}, {'selector': 'bottomrule', 'props': ':bottomrule'}, {'selector': 'rowcolors', 'props': ':{3}{pink}{}'}])
    stlr.highlight_max(axis=0, props='textbf:--rwrap;cellcolor:[rgb]{1,1,0.6}--rwrap')
    stlr.highlight_max(axis=None, props='Huge:--wrap;', subset=[('Z', 'a'), ('Z', 'b')])
    expected = '\\begin{table}[h!]\n\\centering\n\\caption{mycap}\n\\label{fig:item}\n\\rowcolors{3}{pink}{}\n\\begin{tabular}{rlrlr}\n\\toprule\n &  & \\multicolumn{2}{r}{Z} & Y \\\\\n &  & a & b & c \\\\\n\\midrule\n\\multirow[c]{2}{*}{A} & a & 0 & \\textbf{\\cellcolor[rgb]{1,1,0.6}{-0.61}} & ab \\\\\n & b & 1 & -1.22 & cd \\\\\nB & c & \\textbf{\\cellcolor[rgb]{1,1,0.6}{{\\Huge 2}}} & -2.22 & \\textbf{\\cellcolor[rgb]{1,1,0.6}{de}} \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'.replace('table', environment if environment else 'table')
    result = stlr.format(precision=2).to_latex(environment=environment)
    assert result == expected


def test_environment_option(styler: Styler) -> None:
    with option_context('styler.render.repr', 'latex'):
        assert '\\begin{tabular}' in styler._repr_latex_()[:15]
        assert styler._repr_html_() is None


@pytest.mark.parametrize('option', ['hrules'])
def test_bool_options(styler: Styler, option: str) -> None:
    with option_context(f'styler.latex.{option}', False):
        latex_false = styler.to_latex()
    with option_context(f'styler.latex.{option}', True):
        latex_true = styler.to_latex()
    assert latex_false != latex_true


def test_siunitx_basic_headers(styler: Styler) -> None:
    assert '{} & {A} & {B} & {C} \\\\' in styler.to_latex(siunitx=True)
    assert ' & A & B & C \\\\' in styler.to_latex()


@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_css_convert_apply_index(styler: Styler, axis: str) -> None:
    styler.map_index(lambda x: 'font-weight: bold;', axis=axis)
    for label in getattr(styler, axis):
        assert f'\\bfseries {label}' in styler.to_latex(convert_css=True)


def test_hide_index_latex(styler: Styler) -> None:
    styler.hide([0], axis=0)
    result = styler.to_latex()
    expected = dedent('    \\begin{tabular}{lrrl}\n     & A & B & C \\\\\n    1 & 1 & -1.22 & cd \\\\\n    \\end{tabular}\n    ')
    assert expected == result


def test_latex_hiding_index_columns_multiindex_alignment() -> None:
    midx = MultiIndex.from_product([['i0', 'j0'], ['i1'], ['i2', 'j2']], names=['i-0', 'i-1', 'i-2'])
    cidx = MultiIndex.from_product([['c0'], ['c1', 'd1'], ['c2', 'd2']], names=['c-0', 'c-1', 'c-2'])
    df = DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=cidx)
    styler = Styler(df, uuid_len=0)
    styler.hide(level=1, axis=0).hide(level=0, axis=1)
    styler.hide([('i0', 'i1', 'i2')], axis=0)
    styler.hide([('c0', 'c1', 'c2')], axis=1)
    styler.map(lambda x: 'color:{red};' if x == 5 else '')
    styler.map_index(lambda x: 'color:{blue};' if 'j' in x else '')
    result = styler.to_latex()
    expected = dedent('        \\begin{tabular}{llrrr}\n         & c-1 & c1 & \\multicolumn{2}{r}{d1} \\\\\n         & c-2 & d2 & c2 & d2 \\\\\n        i-0 & i-2 &  &  &  \\\\\n        i0 & \\color{blue} j2 & \\color{red} 5 & 6 & 7 \\\\\n        \\multirow[c]{2}{*}{\\color{blue} j0} & i2 & 9 & 10 & 11 \\\\\n        \\color{blue}  & \\color{blue} j2 & 13 & 14 & 15 \\\\\n        \\end{tabular}\n        ')
    assert result == expected


def test_rendered_links() -> None:
    df = DataFrame(['text www.domain.com text'])
    result = df.style.format(hyperlinks='latex').to_latex()
    assert 'text \\href{www.domain.com}{www.domain.com} text' in result


def test_apply_index_hidden_levels() -> None:
    styler = DataFrame([[1]], index=MultiIndex.from_tuples([(0, 1)], names=['l0', 'l1']), columns=MultiIndex.from_tuples([(0, 1)], names=['c0', 'c1'])).style
    styler.hide(level=1)
    styler.map_index(lambda v: 'color: red;', level=0, axis=1)
    result = styler.to_latex(convert_css=True)
    expected = dedent('        \\begin{tabular}{lr}\n        c0 & \\color{red} 0 \\\\\n        c1 & 1 \\\\\n        l0 &  \\\\\n        0 & 1 \\\\\n        \\end{tabular}\n        ')
    assert result == expected


@pytest.mark.parametrize('clines', ['bad', 'index', 'skip-last', 'all', 'data'])
def test_clines_validation(clines: str, styler: Styler) -> None:
    msg = f'`clines` value of {clines} is invalid.'
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(clines=clines)


@pytest.mark.parametrize('clines, exp', [('all;index', '\n\\cline{1-1}'), ('all;data', '\n\\cline{1-2}'), ('skip-last;index', ''), ('skip-last;data', ''), (None, '')])
@pytest.mark.parametrize('env', ['table'])
def test_clines_index(clines: str, exp: str, env: str) -> None:
    df = DataFrame([[1], [2], [3], [4]])
    result = df.style.to_latex(clines=clines, environment=env)
    expected = f'0 & 1 \\\\{exp}\n1 & 2 \\\\{exp}\n2 & 3 \\\\{exp}\n3 & 4 \\\\{exp}\n'
    assert expected in result


@pytest.mark.parametrize('clines, expected', [(None, dedent('            \\begin{tabular}{l}\n            \\end{tabular}\n            ')), ('skip-last;index', dedent('            \\begin{tabular}{l}\n            \\end{tabular}\n            ')), ('skip-last;data', dedent('            \\begin{tabular}{ll}\n             & \\\\\n            \\end{tabular}\n            ')), ('all;index', dedent('            \\begin{tabular}{l}\n            \\cline{2-2}\n            \\cline{1-1} \\cline{2-2}\n            \\end{tabular}\n            ')), ('all;data', dedent('            \\begin{tabular}{ll}\n            \\cline{2-3}\n            \\cline{1-2} \\cline{2-3}\n            \\end{tabular}\n            '))])
@pytest.mark.parametrize('env', ['table'])
def test_clines_multiindex(clines: str, expected: str, env: str) -> None:
    midx = MultiIndex.from_product([['A', '-', 'B'], [0], ['X', 'Y']])
    df = DataFrame([[1], [2], [99], [99], [3], [4]], index=midx)
    styler = df.style
    styler.hide([('-', 0, 'X'), ('-', 0, 'Y')])
    styler.hide(level=1)
    result = styler.to_latex(clines=clines, environment=env)
    assert expected in result


def test_col_format_len(styler: Styler) -> None:
    result = styler.to_latex(environment='longtable', column_format='lrr{10cm}')
    expected = '\\multicolumn{4}{r}{Continued on next page} \\\\'
    assert expected in result


def test_concat(styler: Styler) -> None:
    result = styler.concat(styler.data.agg(['sum']).style).to_latex()
    expected = dedent('    \\begin{tabular}{lrrl}\n     & A & B & C \\\\\n    0 & 0 & -0.61 & ab \\\\\n    1 & 1 & -1.22 & cd \\\\\n    sum & 1 & -1.830000 & abcd \\\\\n    \\end{tabular}\n    ')
    assert result == expected


def test_concat_recursion() -> None:
    styler1 = DataFrame([[1], [9]]).style.hide([1]).highlight_min(color='red')
    styler2 = DataFrame([[9], [2]]).style.hide([0]).highlight_min(color='green')
    styler3 = DataFrame([[3], [9]]).style.hide([1]).highlight_min(color='blue')
    result = styler1.concat(styler2.concat(styler3)).to_latex(convert_css=True)
    expected = dedent('    \\begin{tabular}{lr}\n     & 0 \\\\\n    0 & {\\cellcolor{red}} 1 \\\\\n    1 & {\\cellcolor{green}} 2 \\\\\n    0 & {\\cellcolor{blue}} 3 \\\\\n    \\end{tabular}\n    ')
    assert result == expected


def test_concat_chain() -> None:
    styler1 = DataFrame([[1], [9]]).style.hide([1]).highlight_min(color='red')
    styler2 = DataFrame([[9], [2]]).style.hide([0]).highlight_min(color='green')
    styler3 = DataFrame([[3], [9]]).style.hide([1]).highlight_min(color='blue')
    result = styler1.concat(styler2).concat(styler3).to_latex(convert_css=True)
    expected = dedent('    \\begin{tabular}{lr}\n     & 0 \\\\\n    0 & {\\cellcolor{red}} 1 \\\\\n    1 & {\\cellcolor{green}} 2 \\\\\n    0 & {\\cellcolor{blue}} 3 \\\\\n    \\end{tabular}\n    ')
    assert result == expected


@pytest.mark.parametrize('columns, expected', [(None, dedent('            \\begin{tabular}{l}\n            \\end{tabular}\n            ')), (['a', 'b', 'c'], dedent('            \\begin{tabular}{llll}\n             & a & b & c \\\\\n            \\end{tabular}\n            '))])
@pytest.mark.parametrize('clines', [None, 'all;data', 'all;index', 'skip-last;data', 'skip-last;index'])
def test_empty_clines(columns: list[str], expected: str, clines: str) -> None:
    df = DataFrame(columns=columns)
    result = df.style.to_latex(clines=clines)
    assert result == expected
