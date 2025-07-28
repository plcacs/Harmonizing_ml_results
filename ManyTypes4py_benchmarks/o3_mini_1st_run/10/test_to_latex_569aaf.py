from textwrap import dedent
import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series, option_context
from typing import Any, List, Tuple, Optional, Union
pytest.importorskip("jinja2")
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
    return DataFrame({"A": [0, 1], "B": [-0.61, -1.22], "C": Series(["ab", "cd"], dtype=object)})


@pytest.fixture
def df_ext() -> DataFrame:
    return DataFrame({"A": [0, 1, 2], "B": [-0.61, -1.22, -2.22], "C": ["ab", "cd", "de"]})


@pytest.fixture
def styler(df: DataFrame) -> Styler:
    return Styler(df, uuid_len=0, precision=2)


def test_minimal_latex_tabular(styler: Styler) -> None:
    expected: str = dedent(
        "        \\begin{tabular}{lrrl}\n         & A & B & C \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        "
    )
    assert styler.to_latex() == expected


def test_tabular_hrules(styler: Styler) -> None:
    expected: str = dedent(
        "        \\begin{tabular}{lrrl}\n        \\toprule\n         & A & B & C \\\\\n        \\midrule\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\bottomrule\n        \\end{tabular}\n        "
    )
    assert styler.to_latex(hrules=True) == expected


def test_tabular_custom_hrules(styler: Styler) -> None:
    styler.set_table_styles(
        [
            {"selector": "toprule", "props": ":hline"},
            {"selector": "bottomrule", "props": ":otherline"},
        ]
    )
    expected: str = dedent(
        "        \\begin{tabular}{lrrl}\n        \\hline\n         & A & B & C \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\otherline\n        \\end{tabular}\n        "
    )
    assert styler.to_latex() == expected


def test_column_format(styler: Styler) -> None:
    styler.set_table_styles([{"selector": "column_format", "props": ":cccc"}])
    assert "\\begin{tabular}{rrrr}" in styler.to_latex(column_format="rrrr")
    styler.set_table_styles([{"selector": "column_format", "props": ":r|r|cc"}])
    assert "\\begin{tabular}{r|r|cc}" in styler.to_latex()


def test_siunitx_cols(styler: Styler) -> None:
    expected: str = dedent(
        "        \\begin{tabular}{lSSl}\n        {} & {A} & {B} & {C} \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        "
    )
    assert styler.to_latex(siunitx=True) == expected


def test_position(styler: Styler) -> None:
    assert "\\begin{table}[h!]" in styler.to_latex(position="h!")
    assert "\\end{table}" in styler.to_latex(position="h!")
    styler.set_table_styles([{"selector": "position", "props": ":b!"}])
    assert "\\begin{table}[b!]" in styler.to_latex()
    assert "\\end{table}" in styler.to_latex()


@pytest.mark.parametrize("env", [None, "longtable"])
def test_label(styler: Styler, env: Optional[str]) -> None:
    assert "\n\\label{text}" in styler.to_latex(label="text", environment=env)
    styler.set_table_styles([{"selector": "label", "props": ":{more §text}"}])
    assert "\n\\label{more :text}" in styler.to_latex(environment=env)


def test_position_float_raises(styler: Styler) -> None:
    msg: str = "`position_float` should be one of 'raggedright', 'raggedleft', 'centering',"
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(position_float="bad_string")
    msg = "`position_float` cannot be used in 'longtable' `environment`"
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(position_float="centering", environment="longtable")


@pytest.mark.parametrize("label", [(None, ""), ("text", "\\label{text}")])
@pytest.mark.parametrize("position", [(None, ""), ("h!", "{table}[h!]")])
@pytest.mark.parametrize("caption", [(None, ""), ("text", "\\caption{text}")])
@pytest.mark.parametrize("column_format", [(None, ""), ("rcrl", "{tabular}{rcrl}")])
@pytest.mark.parametrize("position_float", [(None, ""), ("centering", "\\centering")])
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
    styler.set_table_styles(
        [
            {"selector": "mycommand", "props": ":{myoptions}"},
            {"selector": "mycommand2", "props": ":{myoptions2}"},
        ]
    )
    expected: str = dedent(
        "        \\begin{table}\n        \\mycommand{myoptions}\n        \\mycommand2{myoptions2}\n        "
    )
    assert expected in styler.to_latex()


def test_cell_styling(styler: Styler) -> None:
    styler.highlight_max(props="itshape:;Huge:--wrap;")
    expected: str = dedent(
        "        \\begin{tabular}{lrrl}\n         & A & B & C \\\\\n        0 & 0 & \\itshape {\\Huge -0.61} & ab \\\\\n        1 & \\itshape {\\Huge 1} & -1.22 & \\itshape {\\Huge cd} \\\\\n        \\end{tabular}\n        "
    )
    assert expected == styler.to_latex()


def test_multiindex_columns(df: DataFrame) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df.columns = cidx
    expected: str = dedent(
        "        \\begin{tabular}{lrrl}\n         & \\multicolumn{2}{r}{A} & B \\\\\n         & a & b & c \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        "
    )
    s = df.style.format(precision=2)
    assert expected == s.to_latex()
    expected = dedent(
        "        \\begin{tabular}{lrrl}\n         & A & A & B \\\\\n         & a & b & c \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        "
    )
    s = df.style.format(precision=2)
    assert expected == s.to_latex(sparse_columns=False)


def test_multiindex_row(df_ext: DataFrame) -> None:
    ridx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index = ridx
    expected: str = dedent(
        "        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n         & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        "
    )
    styler_obj: Styler = df_ext.style.format(precision=2)
    result: str = styler_obj.to_latex()
    assert expected == result
    expected = dedent(
        "        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        A & a & 0 & -0.61 & ab \\\\\n        A & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        "
    )
    result = styler_obj.to_latex(sparse_index=False)
    assert expected == result


def test_multirow_naive(df_ext: DataFrame) -> None:
    ridx: MultiIndex = MultiIndex.from_tuples([("X", "x"), ("X", "y"), ("Y", "z")])
    df_ext.index = ridx
    expected: str = dedent(
        "        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        X & x & 0 & -0.61 & ab \\\\\n         & y & 1 & -1.22 & cd \\\\\n        Y & z & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        "
    )
    styler_obj: Styler = df_ext.style.format(precision=2)
    result: str = styler_obj.to_latex(multirow_align="naive")
    assert expected == result


def test_multiindex_row_and_col(df_ext: DataFrame) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = (ridx, cidx)
    expected: str = dedent(
        "        \\begin{tabular}{llrrl}\n         &  & \\multicolumn{2}{l}{Z} & Y \\\\\n         &  & a & b & c \\\\\n        \\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n         & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        "
    )
    styler_obj: Styler = df_ext.style.format(precision=2)
    result: str = styler_obj.to_latex(multirow_align="b", multicol_align="l")
    assert result == expected
    expected = dedent(
        "        \\begin{tabular}{llrrl}\n         &  & Z & Z & Y \\\\\n         &  & a & b & c \\\\\n        A & a & 0 & -0.61 & ab \\\\\n        A & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        "
    )
    result = styler_obj.to_latex(sparse_index=False, sparse_columns=False)
    assert result == expected


@pytest.mark.parametrize(
    "multicol_align, siunitx, header",
    [
        ("naive-l", False, " & A & &"),
        ("naive-r", False, " & & & A"),
        ("naive-l", True, "{} & {A} & {} & {}"),
        ("naive-r", True, "{} & {} & {} & {A}"),
    ],
)
def test_multicol_naive(
    df: DataFrame, multicol_align: str, siunitx: bool, header: str
) -> None:
    ridx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")])
    df.columns = ridx
    level1: str = " & a & b & c" if not siunitx else "{} & {a} & {b} & {c}"
    col_format: str = "lrrl" if not siunitx else "lSSl"
    expected: str = dedent(
        f"        \\begin{{tabular}}{{{col_format}}}\n        {header} \\\\\n        {level1} \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{{tabular}}\n        "
    )
    styler_obj: Styler = df.style.format(precision=2)
    result: str = styler_obj.to_latex(multicol_align=multicol_align, siunitx=siunitx)
    assert expected == result


def test_multi_options(df_ext: DataFrame) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler_obj: Styler = df_ext.style.format(precision=2)
    expected: str = dedent(
        "     &  & \\multicolumn{2}{r}{Z} & Y \\\\\n     &  & a & b & c \\\\\n    \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n    "
    )
    result: str = styler_obj.to_latex()
    assert expected in result
    with option_context("styler.latex.multicol_align", "l"):
        assert " &  & \\multicolumn{2}{l}{Z} & Y \\\\" in styler_obj.to_latex()
    with option_context("styler.latex.multirow_align", "b"):
        assert "\\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\" in styler_obj.to_latex()


def test_multiindex_columns_hidden() -> None:
    df: DataFrame = DataFrame([[1, 2, 3, 4]])
    df.columns = MultiIndex.from_tuples([("A", 1), ("A", 2), ("A", 3), ("B", 1)])
    s: Styler = df.style
    assert "{tabular}{lrrrr}" in s.to_latex()
    s.set_table_styles([])
    s.hide([("A", 2)], axis="columns")
    assert "{tabular}{lrrr}" in s.to_latex()


@pytest.mark.parametrize("option, value", [("styler.sparse.index", True), ("styler.sparse.index", False), ("styler.sparse.columns", True), ("styler.sparse.columns", False)])
def test_sparse_options(df_ext: DataFrame, option: str, value: bool) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler_obj: Styler = df_ext.style
    latex1: str = styler_obj.to_latex()
    with option_context(option, value):
        latex2: str = styler_obj.to_latex()
    assert (latex1 == latex2) is value


def test_hidden_index(styler: Styler) -> None:
    styler.hide(axis="index")
    expected: str = dedent(
        "        \\begin{tabular}{rrl}\n        A & B & C \\\\\n        0 & -0.61 & ab \\\\\n        1 & -1.22 & cd \\\\\n        \\end{tabular}\n        "
    )
    assert styler.to_latex() == expected


@pytest.mark.parametrize("environment", ["table", "figure*", None])
def test_comprehensive(df_ext: DataFrame, environment: Optional[str]) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = (ridx, cidx)
    stlr: Styler = df_ext.style
    stlr.set_caption("mycap")
    stlr.set_table_styles(
        [
            {"selector": "label", "props": ":{fig§item}"},
            {"selector": "position", "props": ":h!"},
            {"selector": "position_float", "props": ":centering"},
            {"selector": "column_format", "props": ":rlrlr"},
            {"selector": "toprule", "props": ":toprule"},
            {"selector": "midrule", "props": ":midrule"},
            {"selector": "bottomrule", "props": ":bottomrule"},
            {"selector": "rowcolors", "props": ":{3}{pink}{}"},
        ]
    )
    stlr.highlight_max(axis=0, props="textbf:--rwrap;cellcolor:[rgb]{1,1,0.6}--rwrap")
    stlr.highlight_max(axis=None, props="Huge:--wrap;", subset=[("Z", "a"), ("Z", "b")])
    expected: str = (
        "\\begin{table}[h!]\n\\centering\n\\caption{mycap}\n\\label{fig:item}\n"
        "\\rowcolors{3}{pink}{}\n\\begin{tabular}{rlrlr}\n\\toprule\n"
        " &  & \\multicolumn{2}{r}{Z} & Y \\\\\n &  & a & b & c \\\\\n\\midrule\n"
        "\\multirow[c]{2}{*}{A} & a & 0 & \\textbf{\\cellcolor[rgb]{1,1,0.6}{-0.61}} & ab \\\\\n"
        " & b & 1 & -1.22 & cd \\\\\nB & c & \\textbf{\\cellcolor[rgb]{1,1,0.6}{{\\Huge 2}}} & -2.22 & \\textbf{\\cellcolor[rgb]{1,1,0.6}{de}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    ).replace("table", environment if environment else "table")
    result: str = stlr.format(precision=2).to_latex(environment=environment)
    assert result == expected


def test_environment_option(styler: Styler) -> None:
    with option_context("styler.latex.environment", "bar-env"):
        assert "\\begin{bar-env}" in styler.to_latex()
        assert "\\begin{foo-env}" in styler.to_latex(environment="foo-env")


def test_parse_latex_table_styles(styler: Styler) -> None:
    styler.set_table_styles(
        [
            {"selector": "foo", "props": [("attr", "value")]},
            {"selector": "bar", "props": [("attr", "overwritten")]},
            {"selector": "bar", "props": [("attr", "baz"), ("attr2", "ignored")]},
            {"selector": "label", "props": [("", "{fig§item}")]},
        ]
    )
    assert _parse_latex_table_styles(styler.table_styles, "bar") == "baz"
    assert _parse_latex_table_styles(styler.table_styles, "label") == "{fig:item}"


def test_parse_latex_cell_styles_basic() -> None:
    cell_style: List[Tuple[str, str]] = [("itshape", "--rwrap"), ("cellcolor", "[rgb]{0,1,1}--rwrap")]
    expected: str = "\\itshape{\\cellcolor[rgb]{0,1,1}{text}}"
    assert _parse_latex_cell_styles(cell_style, "text") == expected


@pytest.mark.parametrize(
    "wrap_arg, expected",
    [
        ("", "\\<command><options> <display_value>"),
        ("--wrap", "{\\<command><options> <display_value>}"),
        ("--nowrap", "\\<command><options> <display_value>"),
        ("--lwrap", "{\\<command><options>} <display_value>"),
        ("--dwrap", "{\\<command><options>}{<display_value>}"),
        ("--rwrap", "\\<command><options>{<display_value>}"),
    ],
)
def test_parse_latex_cell_styles_braces(wrap_arg: str, expected: str) -> None:
    cell_style: List[Tuple[str, str]] = [("<command>", f"<options>{wrap_arg}")]
    assert _parse_latex_cell_styles(cell_style, "<display_value>") == expected


def test_parse_latex_header_span() -> None:
    cell: dict = {"attributes": 'colspan="3"', "display_value": "text", "cellstyle": []}
    expected: str = "\\multicolumn{3}{Y}{text}"
    assert _parse_latex_header_span(cell, "X", "Y") == expected
    cell = {"attributes": 'rowspan="5"', "display_value": "text", "cellstyle": []}
    expected = "\\multirow[X]{5}{*}{text}"
    assert _parse_latex_header_span(cell, "X", "Y") == expected
    cell = {"display_value": "text", "cellstyle": []}
    assert _parse_latex_header_span(cell, "X", "Y") == "text"
    cell = {"display_value": "text", "cellstyle": [("bfseries", "--rwrap")]}
    assert _parse_latex_header_span(cell, "X", "Y") == "\\bfseries{text}"


def test_parse_latex_table_wrapping(styler: Styler) -> None:
    styler.set_table_styles(
        [
            {"selector": "toprule", "props": ":value"},
            {"selector": "bottomrule", "props": ":value"},
            {"selector": "midrule", "props": ":value"},
            {"selector": "column_format", "props": ":value"},
        ]
    )
    assert _parse_latex_table_wrapping(styler.table_styles, styler.caption) is False
    assert _parse_latex_table_wrapping(styler.table_styles, "some caption") is True
    styler.set_table_styles([{"selector": "not-ignored", "props": ":value"}], overwrite=False)
    assert _parse_latex_table_wrapping(styler.table_styles, None) is True


def test_short_caption(styler: Styler) -> None:
    result: str = styler.to_latex(caption=("full cap", "short cap"))
    assert "\\caption[short cap]{full cap}" in result


@pytest.mark.parametrize(
    "css, expected",
    [
        ([("color", "red")], [("color", "{red}")]),
        ([("color", "rgb(128, 128, 128 )")], [("color", "[rgb]{0.502, 0.502, 0.502}")]),
        ([("color", "rgb(128, 50%, 25% )")], [("color", "[rgb]{0.502, 0.500, 0.250}")]),
        ([("color", "rgba(128,128,128,1)")], [("color", "[rgb]{0.502, 0.502, 0.502}")]),
        ([("color", "#FF00FF")], [("color", "[HTML]{FF00FF}")]),
        ([("color", "#F0F")], [("color", "[HTML]{FF00FF}")]),
        ([("font-weight", "bold")], [("bfseries", "")]),
        ([("font-weight", "bolder")], [("bfseries", "")]),
        ([("font-weight", "normal")], []),
        ([("background-color", "red")], [("cellcolor", "{red}--lwrap")]),
        ([("background-color", "#FF00FF")], [("cellcolor", "[HTML]{FF00FF}--lwrap")]),
        ([("font-style", "italic")], [("itshape", "")]),
        ([("font-style", "oblique")], [("slshape", "")]),
        ([("font-style", "normal")], []),
        ([("color", "red /*--dwrap*/")], [("color", "{red}--dwrap")]),
        ([("background-color", "red /* --dwrap */")], [("cellcolor", "{red}--dwrap")]),
    ],
)
def test_parse_latex_css_conversion(css: List[Tuple[str, str]], expected: List[Tuple[str, str]]) -> None:
    result: List[Tuple[str, str]] = _parse_latex_css_conversion(css)
    assert result == expected


@pytest.mark.parametrize("env, inner_env", [(None, "tabular"), ("table", "tabular"), ("longtable", "longtable")])
@pytest.mark.parametrize("convert, exp", [(True, "bfseries"), (False, "font-weightbold")])
def test_parse_latex_css_convert_minimal(
    styler: Styler, env: Optional[str], inner_env: str, convert: bool, exp: str
) -> None:
    styler.highlight_max(props="font-weight:bold;")
    result: str = styler.to_latex(convert_css=convert, environment=env)
    expected: str = dedent(
        f"        0 & 0 & \\{exp} -0.61 & ab \\\\\n        1 & \\{exp} 1 & -1.22 & \\{exp} cd \\\\\n        \\end{{{inner_env}}}\n    "
    )
    assert expected in result


def test_parse_latex_css_conversion_option() -> None:
    css: List[Tuple[str, str]] = [("command", "option--latex--wrap")]
    expected: List[Tuple[str, str]] = [("command", "option--wrap")]
    result: List[Tuple[str, str]] = _parse_latex_css_conversion(css)
    assert result == expected


def test_styler_object_after_render(styler: Styler) -> None:
    pre_render: Styler = styler._copy(deepcopy=True)
    styler.to_latex(column_format="rllr", position="h", position_float="centering", hrules=True, label="my lab", caption="my cap")
    assert pre_render.table_styles == styler.table_styles
    assert pre_render.caption == styler.caption


def test_longtable_comprehensive(styler: Styler) -> None:
    result: str = styler.to_latex(environment="longtable", hrules=True, label="fig:A", caption=("full", "short"))
    expected: str = dedent(
        "        \\begin{longtable}{lrrl}\n        \\caption[short]{full} \\label{fig:A} \\\\\n        \\toprule\n         & A & B & C \\\\\n        \\midrule\n        \\endfirsthead\n        \\caption[]{full} \\\\\n        \\toprule\n         & A & B & C \\\\\n        \\midrule\n        \\endhead\n        \\midrule\n        \\multicolumn{4}{r}{Continued on next page} \\\\\n        \\midrule\n        \\endfoot\n        \\bottomrule\n        \\endlastfoot\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{longtable}\n    "
    )
    assert result == expected


def test_longtable_minimal(styler: Styler) -> None:
    result: str = styler.to_latex(environment="longtable")
    expected: str = dedent(
        "        \\begin{longtable}{lrrl}\n         & A & B & C \\\\\n        \\endfirsthead\n         & A & B & C \\\\\n        \\endhead\n        \\multicolumn{4}{r}{Continued on next page} \\\\\n        \\endfoot\n        \\endlastfoot\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{longtable}\n    "
    )
    assert result == expected


@pytest.mark.parametrize("sparse, exp, siunitx", [(True, "{} & \\multicolumn{2}{r}{A} & {B}", True), (False, "{} & {A} & {A} & {B}", True), (True, " & \\multicolumn{2}{r}{A} & B", False), (False, " & A & A & B", False)])
def test_longtable_multiindex_columns(
    df: DataFrame, sparse: bool, exp: str, siunitx: bool
) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df.columns = cidx
    with_si: str = "{} & {a} & {b} & {c} \\\\"
    without_si: str = " & a & b & c \\\\"
    expected: str = dedent(
        f"        \\begin{{longtable}}{{l{('SS' if siunitx else 'rr')}l}}\n        {exp} \\\\\n        {(with_si if siunitx else without_si)}\n        \\endfirsthead\n        {exp} \\\\\n        {(with_si if siunitx else without_si)}\n        \\endhead\n        "
    )
    result: str = df.style.to_latex(environment="longtable", sparse_columns=sparse, siunitx=siunitx)
    assert expected in result


@pytest.mark.parametrize("caption, cap_exp", [("full", ("{full}", "")), (("full", "short"), ("{full}", "[short]"))])
@pytest.mark.parametrize("label, lab_exp", [(None, ""), ("tab:A", " \\label{tab:A}")])
def test_longtable_caption_label(
    styler: Styler, caption: Union[str, Tuple[str, str]], cap_exp: Tuple[str, str], label: Optional[str], lab_exp: str
) -> None:
    cap_exp1: str = f"\\caption{cap_exp[1]}{cap_exp[0]}"
    cap_exp2: str = f"\\caption[]{cap_exp[0]}"
    expected: str = dedent(
        f"        {cap_exp1}{lab_exp} \\\\\n         & A & B & C \\\\\n        \\endfirsthead\n        {cap_exp2} \\\\\n        "
    )
    assert expected in styler.to_latex(environment="longtable", caption=caption, label=label)


@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("columns, siunitx", [(True, True), (True, False), (False, False)])
def test_apply_map_header_render_mi(
    df_ext: DataFrame, index: bool, columns: bool, siunitx: bool
) -> None:
    cidx: MultiIndex = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx: MultiIndex = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler_obj: Styler = df_ext.style
    func = lambda v: "bfseries: --rwrap" if ("A" in v or "Z" in v or "c" in v) else None
    if index:
        styler_obj.map_index(func, axis="index")
    if columns:
        styler_obj.map_index(func, axis="columns")
    result: str = styler_obj.to_latex(siunitx=siunitx)
    expected_index: str = dedent(
        "    \\multirow[c]{2}{*}{\\bfseries{A}} & a & 0 & -0.610000 & ab \\\\\n    \\bfseries{} & b & 1 & -1.220000 & cd \\\\\n    B & \\bfseries{c} & 2 & -2.220000 & de \\\\\n    "
    )
    assert (expected_index in result) is index
    exp_cols_si: str = dedent(
        "    {} & {} & \\multicolumn{2}{r}{\\bfseries{Z}} & {Y} \\\\\n    {} & {} & {a} & {b} & {\\bfseries{c}} \\\\\n    "
    )
    exp_cols_no_si: str = " &  & \\multicolumn{2}{r}{\\bfseries{Z}} & Y \\\\\n &  & a & b & \\bfseries{c} \\\\\n"
    assert ((exp_cols_si if siunitx else exp_cols_no_si) in result) is columns


def test_repr_option(styler: Styler) -> None:
    assert "<style" in styler._repr_html_()[:6]
    assert styler._repr_latex_() is None
    with option_context("styler.render.repr", "latex"):
        assert "\\begin{tabular}" in styler._repr_latex_()[:15]
        assert styler._repr_html_() is None


@pytest.mark.parametrize("option", ["hrules"])
def test_bool_options(styler: Styler, option: str) -> None:
    with option_context(f"styler.latex.{option}", False):
        latex_false: str = styler.to_latex()
    with option_context(f"styler.latex.{option}", True):
        latex_true: str = styler.to_latex()
    assert latex_false != latex_true


def test_siunitx_basic_headers(styler: Styler) -> None:
    assert "{} & {A} & {B} & {C} \\\\" in styler.to_latex(siunitx=True)
    assert " & A & B & C \\\\" in styler.to_latex()


@pytest.mark.parametrize("axis", ["index", "columns"])
def test_css_convert_apply_index(styler: Styler, axis: str) -> None:
    styler.map_index(lambda x: "font-weight: bold;", axis=axis)
    for label in getattr(styler, axis):
        assert f"\\bfseries {label}" in styler.to_latex(convert_css=True)


def test_hide_index_latex(styler: Styler) -> None:
    styler.hide([0], axis=0)
    result: str = styler.to_latex()
    expected: str = dedent(
        "    \\begin{tabular}{lrrl}\n     & A & B & C \\\\\n    1 & 1 & -1.22 & cd \\\\\n    \\end{tabular}\n    "
    )
    assert expected == result


def test_latex_hiding_index_columns_multiindex_alignment() -> None:
    midx: MultiIndex = MultiIndex.from_product([["i0", "j0"], ["i1"], ["i2", "j2"]], names=["i-0", "i-1", "i-2"])
    cidx: MultiIndex = MultiIndex.from_product([["c0"], ["c1", "d1"], ["c2", "d2"]], names=["c-0", "c-1", "c-2"])
    df: DataFrame = DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=cidx)
    styler_obj: Styler = Styler(df, uuid_len=0)
    styler_obj.hide(level=1, axis=0).hide(level=0, axis=1)
    styler_obj.hide([("i0", "i1", "i2")], axis=0)
    styler_obj.hide([("c0", "c1", "c2")], axis=1)
    styler_obj.map(lambda x: "color:{red};" if x == 5 else "")
    styler_obj.map_index(lambda x: "color:{blue};" if "j" in x else "")
    result: str = styler_obj.to_latex()
    expected: str = dedent(
        "        \\begin{tabular}{llrrr}\n         & c-1 & c1 & \\multicolumn{2}{r}{d1} \\\\\n         & c-2 & d2 & c2 & d2 \\\\\n        i-0 & i-2 &  &  &  \\\\\n        i0 & \\color{blue} j2 & \\color{red} 5 & 6 & 7 \\\\\n        \\multirow[c]{2}{*}{\\color{blue} j0} & i2 & 9 & 10 & 11 \\\\\n        \\color{blue}  & \\color{blue} j2 & 13 & 14 & 15 \\\\\n        \\end{tabular}\n        "
    )
    assert result == expected


def test_rendered_links() -> None:
    df: DataFrame = DataFrame(["text www.domain.com text"])
    result: str = df.style.format(hyperlinks="latex").to_latex()
    assert "text \\href{www.domain.com}{www.domain.com} text" in result


def test_apply_index_hidden_levels() -> None:
    styler_obj: Styler = DataFrame(
        [[1]],
        index=MultiIndex.from_tuples([(0, 1)], names=["l0", "l1"]),
        columns=MultiIndex.from_tuples([(0, 1)], names=["c0", "c1"]),
    ).style
    styler_obj.hide(level=1)
    styler_obj.map_index(lambda v: "color: red;", level=0, axis="columns")
    result: str = styler_obj.to_latex(convert_css=True)
    expected: str = dedent(
        "        \\begin{tabular}{lr}\n        c0 & \\color{red} 0 \\\\\n        c1 & 1 \\\\\n        l0 &  \\\\\n        0 & 1 \\\\\n        \\end{tabular}\n        "
    )
    assert result == expected


@pytest.mark.parametrize("clines", ["bad", "index", "skip-last", "all", "data"])
def test_clines_validation(clines: str, styler: Styler) -> None:
    msg: str = f"`clines` value of {clines} is invalid."
    with pytest.raises(ValueError, match=msg):
        styler.to_latex(clines=clines)


@pytest.mark.parametrize("clines, exp", [("all;index", "\n\\cline{1-1}"), ("all;data", "\n\\cline{1-2}"), ("skip-last;index", ""), ("skip-last;data", ""), (None, "")])
@pytest.mark.parametrize("env", ["table", "longtable"])
def test_clines_index(clines: Optional[str], exp: str, env: str) -> None:
    df: DataFrame = DataFrame([[1], [2], [3], [4]])
    result: str = df.style.to_latex(clines=clines, environment=env)
    expected: str = f"0 & 1 \\\\{exp}\n1 & 2 \\\\{exp}\n2 & 3 \\\\{exp}\n3 & 4 \\\\{exp}\n"
    assert expected in result


@pytest.mark.parametrize(
    "clines, expected",
    [
        (None, dedent("            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n             & Y & 2 \\\\\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n             & Y & 4 \\\\\n            ")),
        ("skip-last;index", dedent("            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n             & Y & 2 \\\\\n            \\cline{1-2}\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n             & Y & 4 \\\\\n            \\cline{1-2}\n            ")),
        ("skip-last;data", dedent("            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n             & Y & 2 \\\\\n            \\cline{1-3}\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n             & Y & 4 \\\\\n            \\cline{1-3}\n            ")),
        (
            "all;index",
            dedent("            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n            \\cline{2-2}\n             & Y & 2 \\\\\n            \\cline{1-2} \\cline{2-2}\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n            \\cline{2-2}\n             & Y & 4 \\\\\n            \\cline{1-2} \\cline{2-2}\n            "),
        ),
        (
            "all;data",
            dedent("            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n            \\cline{2-3}\n             & Y & 2 \\\\\n            \\cline{1-3} \\cline{2-3}\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n            \\cline{2-3}\n             & Y & 4 \\\\\n            \\cline{1-3} \\cline{2-3}\n            "),
        ),
    ],
)
@pytest.mark.parametrize("env", ["table"])
def test_clines_multiindex(clines: Optional[str], expected: str, env: str) -> None:
    midx: MultiIndex = MultiIndex.from_product([["A", "-", "B"], [0], ["X", "Y"]])
    df: DataFrame = DataFrame([[1], [2], [99], [99], [3], [4]], index=midx)
    styler_obj: Styler = df.style
    styler_obj.hide([("-", 0, "X"), ("-", 0, "Y")])
    styler_obj.hide(level=1)
    result: str = styler_obj.to_latex(clines=clines, environment=env)
    assert expected in result


def test_col_format_len(styler: Styler) -> None:
    result: str = styler.to_latex(environment="longtable", column_format="lrr{10cm}")
    expected: str = "\\multicolumn{4}{r}{Continued on next page} \\\\"
    assert expected in result


def test_concat(styler: Styler) -> None:
    result: str = styler.concat(styler.data.agg(["sum"]).style).to_latex()
    expected: str = dedent(
        "    \\begin{tabular}{lrrl}\n     & A & B & C \\\\\n    0 & 0 & -0.61 & ab \\\\\n    1 & 1 & -1.22 & cd \\\\\n    sum & 1 & -1.830000 & abcd \\\\\n    \\end{tabular}\n    "
    )
    assert result == expected


def test_concat_recursion() -> None:
    styler1: Styler = DataFrame([[1], [9]]).style.hide([1]).highlight_min(color="red")
    styler2: Styler = DataFrame([[9], [2]]).style.hide([0]).highlight_min(color="green")
    styler3: Styler = DataFrame([[3], [9]]).style.hide([1]).highlight_min(color="blue")
    result: str = styler1.concat(styler2.concat(styler3)).to_latex(convert_css=True)
    expected: str = dedent(
        "    \\begin{tabular}{lr}\n     & 0 \\\\\n    0 & {\\cellcolor{red}} 1 \\\\\n    1 & {\\cellcolor{green}} 2 \\\\\n    0 & {\\cellcolor{blue}} 3 \\\\\n    \\end{tabular}\n    "
    )
    assert result == expected


def test_concat_chain() -> None:
    styler1: Styler = DataFrame([[1], [9]]).style.hide([1]).highlight_min(color="red")
    styler2: Styler = DataFrame([[9], [2]]).style.hide([0]).highlight_min(color="green")
    styler3: Styler = DataFrame([[3], [9]]).style.hide([1]).highlight_min(color="blue")
    result: str = styler1.concat(styler2).concat(styler3).to_latex(convert_css=True)
    expected: str = dedent(
        "    \\begin{tabular}{lr}\n     & 0 \\\\\n    0 & {\\cellcolor{red}} 1 \\\\\n    1 & {\\cellcolor{green}} 2 \\\\\n    0 & {\\cellcolor{blue}} 3 \\\\\n    \\end{tabular}\n    "
    )
    assert result == expected


@pytest.mark.parametrize("columns, expected", [(None, dedent("            \\begin{tabular}{l}\n            \\end{tabular}\n            ")), (["a", "b", "c"], dedent("            \\begin{tabular}{llll}\n             & a & b & c \\\\\n            \\end{tabular}\n            "))])
@pytest.mark.parametrize("clines", [None, "all;data", "all;index", "skip-last;data", "skip-last;index"])
def test_empty_clines(columns: Optional[List[str]], expected: str, clines: Optional[str]) -> None:
    df: DataFrame = DataFrame(columns=columns)
    result: str = df.style.to_latex(clines=clines)
    assert result == expected