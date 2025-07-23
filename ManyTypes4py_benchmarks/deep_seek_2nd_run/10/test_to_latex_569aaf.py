from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series, option_context
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
    expected: str = dedent('        \\begin{tabular}{lrrl}\n         & A & B & C \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    assert styler.to_latex() == expected

def test_tabular_hrules(styler: Styler) -> None:
    expected: str = dedent('        \\begin{tabular}{lrrl}\n        \\toprule\n         & A & B & C \\\\\n        \\midrule\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\bottomrule\n        \\end{tabular}\n        ')
    assert styler.to_latex(hrules=True) == expected

def test_tabular_custom_hrules(styler: Styler) -> None:
    styler.set_table_styles([{'selector': 'toprule', 'props': ':hline'}, {'selector': 'bottomrule', 'props': ':otherline'}])
    expected: str = dedent('        \\begin{tabular}{lrrl}\n        \\hline\n         & A & B & C \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\otherline\n        \\end{tabular}\n        ')
    assert styler.to_latex() == expected

def test_column_format(styler: Styler) -> None:
    styler.set_table_styles([{'selector': 'column_format', 'props': ':cccc'}])
    assert '\\begin{tabular}{rrrr}' in styler.to_latex(column_format='rrrr')
    styler.set_table_styles([{'selector': 'column_format', 'props': ':r|r|cc'}])
    assert '\\begin{tabular}{r|r|cc}' in styler.to_latex()

def test_siunitx_cols(styler: Styler) -> None:
    expected: str = dedent('        \\begin{tabular}{lSSl}\n        {} & {A} & {B} & {C} \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    assert styler.to_latex(siunitx=True) == expected

def test_position(styler: Styler) -> None:
    assert '\\begin{table}[h!]' in styler.to_latex(position='h!')
    assert '\\end{table}' in styler.to_latex(position='h!')
    styler.set_table_styles([{'selector': 'position', 'props': ':b!'}])
    assert '\\begin{table}[b!]' in styler.to_latex()
    assert '\\end{table}' in styler.to_latex()

@pytest.mark.parametrize('env', [None, 'longtable'])
def test_label(styler: Styler, env: Optional[str]) -> None:
    assert '\n\\label{text}' in styler.to_latex(label='text', environment=env)
    styler.set_table_styles([{'selector': 'label', 'props': ':{more §text}'}])
    assert '\n\\label{more :text}' in styler.to_latex(environment=env)

def test_position_float_raises(styler: Styler) -> None:
    msg: str = "`position_float` should be one of 'raggedright', 'raggedleft', 'centering',"
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
    styler: Styler,
    label: Tuple[Optional[str], str],
    position: Tuple[Optional[str], str],
    caption: Tuple[Optional[str], str],
    column_format: Tuple[Optional[str], str],
    position_float: Tuple[Optional[str], str],
) -> None:
    result: str = styler.to_latex(
        label=label[0],
        position=position[0],
        caption=caption[0],
        column_format=column_format[0],
        position_float=position_float[0],
    )
    assert label[1] in result
    assert position[1] in result
    assert caption[1] in result
    assert column_format[1] in result
    assert position_float[1] in result

def test_custom_table_styles(styler: Styler) -> None:
    styler.set_table_styles([{'selector': 'mycommand', 'props': ':{myoptions}'}, {'selector': 'mycommand2', 'props': ':{myoptions2}'}])
    expected: str = dedent('        \\begin{table}\n        \\mycommand{myoptions}\n        \\mycommand2{myoptions2}\n        ')
    assert expected in styler.to_latex()

def test_cell_styling(styler: Styler) -> None:
    styler.highlight_max(props='itshape:;Huge:--wrap;')
    expected: str = dedent('        \\begin{tabular}{lrrl}\n         & A & B & C \\\\\n        0 & 0 & \\itshape {\\Huge -0.61} & ab \\\\\n        1 & \\itshape {\\Huge 1} & -1.22 & \\itshape {\\Huge cd} \\\\\n        \\end{tabular}\n        ')
    assert expected == styler.to_latex()

def test_multiindex_columns(df: DataFrame) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df.columns = cidx
    expected: str = dedent('        \\begin{tabular}{lrrl}\n         & \\multicolumn{2}{r}{A} & B \\\\\n         & a & b & c \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    s: Styler = df.style.format(precision=2)
    assert expected == s.to_latex()
    expected = dedent('        \\begin{tabular}{lrrl}\n         & A & A & B \\\\\n         & a & b & c \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    s = df.style.format(precision=2)
    assert expected == s.to_latex(sparse_columns=False)

def test_multiindex_row(df_ext: DataFrame) -> None:
    ridx: MultiIndex = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index = ridx
    expected: str = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n         & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler: Styler = df_ext.style.format(precision=2)
    result: str = styler.to_latex()
    assert expected == result
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        A & a & 0 & -0.61 & ab \\\\\n        A & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    result = styler.to_latex(sparse_index=False)
    assert expected == result

def test_multirow_naive(df_ext: DataFrame) -> None:
    ridx: MultiIndex = MultiIndex.from_tuples([('X', 'x'), ('X', 'y'), ('Y', 'z')])
    df_ext.index = ridx
    expected: str = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        X & x & 0 & -0.61 & ab \\\\\n         & y & 1 & -1.22 & cd \\\\\n        Y & z & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler: Styler = df_ext.style.format(precision=2)
    result: str = styler.to_latex(multirow_align='naive')
    assert expected == result

def test_multiindex_row_and_col(df_ext: DataFrame) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx: MultiIndex = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    expected: str = dedent('        \\begin{tabular}{llrrl}\n         &  & \\multicolumn{2}{l}{Z} & Y \\\\\n         &  & a & b & c \\\\\n        \\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n         & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler: Styler = df_ext.style.format(precision=2)
    result: str = styler.to_latex(multirow_align='b', multicol_align='l')
    assert result == expected
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & Z & Z & Y \\\\\n         &  & a & b & c \\\\\n        A & a & 0 & -0.61 & ab \\\\\n        A & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    result = styler.to_latex(sparse_index=False, sparse_columns=False)
    assert result == expected

@pytest.mark.parametrize('multicol_align, siunitx, header', [('naive-l', False, ' & A & &'), ('naive-r', False, ' & & & A'), ('naive-l', True, '{} & {A} & {} & {}'), ('naive-r', True, '{} & {} & {} & {A}')])
def test_multicol_naive(
    df: DataFrame,
    multicol_align: str,
    siunitx: bool,
    header: str,
) -> None:
    ridx: MultiIndex = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')])
    df.columns = ridx
    level1: str = ' & a & b & c' if not siunitx else '{} & {a} & {b} & {c}'
    col_format: str = 'lrrl' if not siunitx else 'lSSl'
    expected: str = dedent(f'        \\begin{{tabular}}{{{col_format}}}\n        {header} \\\\\n        {level1} \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{{tabular}}\n        ')
    styler: Styler = df.style.format(precision=2)
    result: str = styler.to_latex(multicol_align=multicol_align, siunitx=siunitx)
    assert expected == result

def test_multi_options(df_ext: DataFrame) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx: MultiIndex = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler: Styler = df_ext.style.format(precision=2)
    expected: str = dedent('     &  & \\multicolumn{2}{r}{Z} & Y \\\\\n     &  & a & b & c \\\\\n    \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n    ')
    result: str = styler.to_latex()
    assert expected in result
    with option_context('styler.latex.multicol_align', 'l'):
        assert ' &  & \\multicolumn{2}{l}{Z} & Y \\\\' in styler.to_latex()
    with option_context('styler.latex.multirow_align', 'b'):
        assert '\\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\' in styler.to_latex()

def test_multiindex_columns_hidden() -> None:
    df: DataFrame = DataFrame([[1, 2, 3, 4]])
    df.columns = MultiIndex.from_tuples([('A', 1), ('A', 2), ('A', 3), ('B', 1)])
    s: Styler = df.style
    assert '{tabular}{lrrrr}' in s.to_latex()
    s.set_table_styles([])
    s.hide([('A', 2)], axis='columns')
    assert '{tabular}{lrrr}' in s.to_latex()

@pytest.mark.parametrize('option, value', [('styler.sparse.index', True), ('styler.sparse.index', False), ('styler.sparse.columns', True), ('styler.sparse.columns', False)])
def test_sparse_options(df_ext: DataFrame, option: str, value: bool) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx: MultiIndex = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler: Styler = df_ext.style
    latex1: str = styler.to_latex()
    with option_context(option, value):
        latex2: str = styler.to_latex()
    assert (latex1 == latex2) is value

def test_hidden_index(styler: Styler) -> None:
    styler.hide(axis='index')
    expected: str = dedent('        \\begin{tabular}{rrl}\n        A & B & C \\\\\n        0 & -0.61 & ab \\\\\n        1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    assert styler.to_latex() == expected

@pytest.mark.parametrize('environment', ['table', 'figure*', None])
def test_comprehensive(df_ext: DataFrame, environment: Optional[str]) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx: MultiIndex = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    stlr: Styler = df