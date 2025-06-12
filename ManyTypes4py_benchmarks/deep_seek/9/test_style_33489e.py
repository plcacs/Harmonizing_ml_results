import contextlib
import copy
import re
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pytest
from pandas import DataFrame, IndexSlice, MultiIndex, Series, option_context
import pandas._testing as tm
jinja2 = pytest.importorskip('jinja2')
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _get_level_lengths, _get_trimming_maximums, maybe_convert_css_to_tuples, non_reducing_slice

@pytest.fixture
def mi_df() -> DataFrame:
    return DataFrame([[1, 2], [3, 4]], index=MultiIndex.from_product([['i0'], ['i1_a', 'i1_b']]), columns=MultiIndex.from_product([['c0'], ['c1_a', 'c1_b']]), dtype=int)

@pytest.fixture
def mi_styler(mi_df: DataFrame) -> Styler:
    return Styler(mi_df, uuid_len=0)

@pytest.fixture
def mi_styler_comp(mi_styler: Styler) -> Styler:
    mi_styler = mi_styler._copy(deepcopy=True)
    mi_styler.css = {**mi_styler.css, 'row': 'ROW', 'col': 'COL'}
    mi_styler.uuid_len = 5
    mi_styler.uuid = 'abcde'
    mi_styler.set_caption('capt')
    mi_styler.set_table_styles([{'selector': 'a', 'props': 'a:v;'}])
    mi_styler.hide(axis='columns')
    mi_styler.hide([('c0', 'c1_a')], axis='columns', names=True)
    mi_styler.hide(axis='index')
    mi_styler.hide([('i0', 'i1_a')], axis='index', names=True)
    mi_styler.set_table_attributes('class="box"')
    other = mi_styler.data.agg(['mean'])
    other.index = MultiIndex.from_product([[''], other.index])
    mi_styler.concat(other.style)
    mi_styler.format(na_rep='MISSING', precision=3)
    mi_styler.format_index(precision=2, axis=0)
    mi_styler.format_index(precision=4, axis=1)
    mi_styler.highlight_max(axis=None)
    mi_styler.map_index(lambda x: 'color: white;', axis=0)
    mi_styler.map_index(lambda x: 'color: black;', axis=1)
    mi_styler.set_td_classes(DataFrame([['a', 'b'], ['a', 'c']], index=mi_styler.index, columns=mi_styler.columns))
    mi_styler.set_tooltips(DataFrame([['a2', 'b2'], ['a2', 'c2']], index=mi_styler.index, columns=mi_styler.columns))
    mi_styler.format_index_names(escape='html', axis=0)
    mi_styler.format_index_names(escape='html', axis=1)
    return mi_styler

@pytest.fixture
def blank_value() -> str:
    return '&nbsp;'

@pytest.fixture
def df() -> DataFrame:
    df = DataFrame({'A': [0, 1], 'B': np.random.default_rng(2).standard_normal(2)})
    return df

@pytest.fixture
def styler(df: DataFrame) -> Styler:
    df = DataFrame({'A': [0, 1], 'B': np.random.default_rng(2).standard_normal(2)})
    return Styler(df)

@pytest.mark.parametrize('sparse_columns, exp_cols', [(True, [{'is_visible': True, 'attributes': 'colspan="2"', 'value': 'c0'}, {'is_visible': False, 'attributes': '', 'value': 'c0'}]), (False, [{'is_visible': True, 'attributes': '', 'value': 'c0'}, {'is_visible': True, 'attributes': '', 'value': 'c0'}])])
def test_mi_styler_sparsify_columns(mi_styler: Styler, sparse_columns: bool, exp_cols: List[Dict[str, Any]]) -> None:
    exp_l1_c0 = {'is_visible': True, 'attributes': '', 'display_value': 'c1_a'}
    exp_l1_c1 = {'is_visible': True, 'attributes': '', 'display_value': 'c1_b'}
    ctx = mi_styler._translate(True, sparse_columns)
    assert exp_cols[0].items() <= ctx['head'][0][2].items()
    assert exp_cols[1].items() <= ctx['head'][0][3].items()
    assert exp_l1_c0.items() <= ctx['head'][1][2].items()
    assert exp_l1_c1.items() <= ctx['head'][1][3].items()

@pytest.mark.parametrize('sparse_index, exp_rows', [(True, [{'is_visible': True, 'attributes': 'rowspan="2"', 'value': 'i0'}, {'is_visible': False, 'attributes': '', 'value': 'i0'}]), (False, [{'is_visible': True, 'attributes': '', 'value': 'i0'}, {'is_visible': True, 'attributes': '', 'value': 'i0'}])])
def test_mi_styler_sparsify_index(mi_styler: Styler, sparse_index: bool, exp_rows: List[Dict[str, Any]]) -> None:
    exp_l1_r0 = {'is_visible': True, 'attributes': '', 'display_value': 'i1_a'}
    exp_l1_r1 = {'is_visible': True, 'attributes': '', 'display_value': 'i1_b'}
    ctx = mi_styler._translate(sparse_index, True)
    assert exp_rows[0].items() <= ctx['body'][0][0].items()
    assert exp_rows[1].items() <= ctx['body'][1][0].items()
    assert exp_l1_r0.items() <= ctx['body'][0][1].items()
    assert exp_l1_r1.items() <= ctx['body'][1][1].items()

def test_mi_styler_sparsify_options(mi_styler: Styler) -> None:
    with option_context('styler.sparse.index', False):
        html1 = mi_styler.to_html()
    with option_context('styler.sparse.index', True):
        html2 = mi_styler.to_html()
    assert html1 != html2
    with option_context('styler.sparse.columns', False):
        html1 = mi_styler.to_html()
    with option_context('styler.sparse.columns', True):
        html2 = mi_styler.to_html()
    assert html1 != html2

@pytest.mark.parametrize('rn, cn, max_els, max_rows, max_cols, exp_rn, exp_cn', [(100, 100, 100, None, None, 12, 6), (1000, 3, 750, None, None, 250, 3), (4, 1000, 500, None, None, 4, 125), (1000, 3, 750, 10, None, 10, 3), (4, 1000, 500, None, 5, 4, 5), (100, 100, 700, 50, 50, 25, 25)])
def test_trimming_maximum(rn: int, cn: int, max_els: int, max_rows: Optional[int], max_cols: Optional[int], exp_rn: int, exp_cn: int) -> None:
    rn, cn = _get_trimming_maximums(rn, cn, max_els, max_rows, max_cols, scaling_factor=0.5)
    assert (rn, cn) == (exp_rn, exp_cn)

@pytest.mark.parametrize('option, val', [('styler.render.max_elements', 6), ('styler.render.max_rows', 3)])
def test_render_trimming_rows(option: str, val: int) -> None:
    df = DataFrame(np.arange(120).reshape(60, 2))
    with option_context(option, val):
        ctx = df.style._translate(True, True)
    assert len(ctx['head'][0]) == 3
    assert len(ctx['body']) == 4
    assert len(ctx['body'][0]) == 3

@pytest.mark.parametrize('option, val', [('styler.render.max_elements', 6), ('styler.render.max_columns', 2)])
def test_render_trimming_cols(option: str, val: int) -> None:
    df = DataFrame(np.arange(30).reshape(3, 10))
    with option_context(option, val):
        ctx = df.style._translate(True, True)
    assert len(ctx['head'][0]) == 4
    assert len(ctx['body']) == 3
    assert len(ctx['body'][0]) == 4

def test_render_trimming_mi() -> None:
    midx = MultiIndex.from_product([[1, 2], [1, 2, 3]])
    df = DataFrame(np.arange(36).reshape(6, 6), columns=midx, index=midx)
    with option_context('styler.render.max_elements', 4):
        ctx = df.style._translate(True, True)
    assert len(ctx['body'][0]) == 5
    assert {'attributes': 'rowspan="2"'}.items() <= ctx['body'][0][0].items()
    assert {'class': 'data row0 col_trim'}.items() <= ctx['body'][0][4].items()
    assert {'class': 'data row_trim col_trim'}.items() <= ctx['body'][2][4].items()
    assert len(ctx['body']) == 3

def test_render_empty_mi() -> None:
    df = DataFrame(index=MultiIndex.from_product([['A'], [0, 1]], names=[None, 'one']))
    expected = dedent('    >\n      <thead>\n        <tr>\n          <th class="index_name level0" >&nbsp;</th>\n          <th class="index_name level1" >one</th>\n        </tr>\n      </thead>\n    ')
    assert expected in df.style.to_html()

@pytest.mark.parametrize('comprehensive', [True, False])
@pytest.mark.parametrize('render', [True, False])
@pytest.mark.parametrize('deepcopy', [True, False])
def test_copy(comprehensive: bool, render: bool, deepcopy: bool, mi_styler: Styler, mi_styler_comp: Styler) -> None:
    styler = mi_styler_comp if comprehensive else mi_styler
    styler.uuid_len = 5
    s2 = copy.deepcopy(styler) if deepcopy else copy.copy(styler)
    assert s2 is not styler
    if render:
        styler.to_html()
    excl = ['cellstyle_map', 'cellstyle_map_columns', 'cellstyle_map_index', 'template_latex', 'template_html', 'template_html_style', 'template_html_table']
    if not deepcopy:
        for attr in [a for a in styler.__dict__ if not callable(a) and a not in excl]:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))
    else:
        shallow = ['data', 'columns', 'index', 'uuid_len', 'uuid', 'caption', 'cell_ids', 'hide_index_', 'hide_columns_', 'hide_index_names', 'hide_column_names', 'table_attributes']
        for attr in shallow:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))
        for attr in [a for a in styler.__dict__ if not callable(a) and a not in excl and (a not in shallow)]:
            if getattr(s2, attr) is None:
                assert id(getattr(s2, attr)) == id(getattr(styler, attr))
            else:
                assert id(getattr(s2, attr)) != id(getattr(styler, attr))

@pytest.mark.parametrize('deepcopy', [True, False])
def test_inherited_copy(mi_styler: Styler, deepcopy: bool) -> None:

    class CustomStyler(Styler):
        pass
    custom_styler = CustomStyler(mi_styler.data)
    custom_styler_copy = copy.deepcopy(custom_styler) if deepcopy else copy.copy(custom_styler)
    assert isinstance(custom_styler_copy, CustomStyler)

def test_clear(mi_styler_comp: Styler) -> None:
    styler = mi_styler_comp
    styler._compute()
    clean_copy = Styler(styler.data, uuid=styler.uuid)
    excl = ['data', 'index', 'columns', 'uuid', 'uuid_len', 'cell_ids', 'cellstyle_map', 'cellstyle_map_columns', 'cellstyle_map_index', 'template_latex', 'template_html', 'template_html_style', 'template_html_table']
    for attr in [a for a in styler.__dict__ if not (callable(a) or a in excl)]:
        res = getattr(styler, attr) == getattr(clean_copy, attr)
        if hasattr(res, '__iter__') and len(res) > 0:
            assert not all(res)
        elif hasattr(res, '__iter__') and len(res) == 0:
            pass
        else:
            assert not res
    styler.clear()
    for attr in [a for a in styler.__dict__ if not callable(a)]:
        res = getattr(styler, attr) == getattr(clean_copy, attr)
        assert all(res) if hasattr(res, '__iter__') else res

def test_export(mi_styler_comp: Styler, mi_styler: Styler) -> None:
    exp_attrs = ['_todo', 'hide_index_', 'hide_index_names', 'hide_columns_', 'hide_column_names', 'table_attributes', 'table_styles', 'css']
    for attr in exp_attrs:
        check = getattr(mi_styler, attr) == getattr(mi_styler_comp, attr)
        assert not (all(check) if hasattr(check, '__iter__') and len(check) > 0 else check)
    export = mi_styler_comp.export()
    used = mi_styler.use(export)
    for attr in exp_attrs:
        check = getattr(used, attr) == getattr(mi_styler_comp, attr)
        assert all(check) if hasattr(check, '__iter__') and len(check) > 0 else check
    used.to_html()

def test_hide_raises(mi_styler: Styler) -> None:
    msg = '`subset` and `level` cannot be passed simultaneously'
    with pytest.raises(ValueError, match=msg):
        mi_styler.hide(axis='index', subset='something', level='something else')
    msg = '`level` must be of type `int`, `str` or list of such'
    with pytest.raises(ValueError, match=msg):
        mi_styler.hide(axis='index', level={'bad': 1, 'type': 2})

@pytest.mark.parametrize('level', [1, 'one', [1], ['one']])
def test_hide_index_level(mi_styler: Styler, level: Union[int, str, List[Union[int, str]]]) -> None:
    mi_styler.index.names, mi_styler.columns.names = (['zero', 'one'], ['zero', 'one'])
    ctx = mi_styler.hide(axis='index', level=level)._translate(False, True)
    assert len(ctx['head'][0]) == 3
    assert len(ctx['head'][1]) == 3
    assert len(ctx['head'][2]) == 4
    assert ctx['head'][2][0]['is_visible']
    assert not ctx['head'][2][1]['is_visible']
    assert ctx['body'][0][0]['is_visible']
    assert not ctx['body'][0][1]['is_visible']
    assert ctx['body'][1][0]['is_visible']
    assert not ctx['body'][1][1]['is_visible']

@pytest.mark.parametrize('level', [1, 'one', [1], ['one']])
@pytest.mark.parametrize('names', [True, False])
def test_hide_columns_level(mi_styler: Styler, level: Union[int, str, List[Union[int, str]]], names: bool) -> None:
    mi_styler.columns.names = ['zero', 'one']
    if names:
        mi_styler.index.names = ['zero', 'one']
    ctx = mi_styler.hide(axis='columns', level=level)._translate(True, False)
    assert len(ctx['head']) == (2 if names else 1)

@pytest.mark.parametrize('method', ['map', 'apply'])
@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_apply_map_header(method: str, axis: str) -> None:
    df = DataFrame({'A': [0, 0], 'B': [1, 1]}, index=['C', 'D'])
    func = {'apply': lambda s: ['attr: val' if 'A' in v or 'C' in v else '' for v in s], 'map': lambda v: 'attr: val' if 'A' in v or 'C' in v else ''}
    result = getattr(df.style, f'{method}_index')(func[method], axis=axis)
    assert len(result._todo) == 1
    assert len(getattr(result, f'ctx_{axis}')) == 0
    result._compute()
    expected = {(0, 0): [('attr', 'val')]}
    assert getattr(result, f'ctx_{axis}') == expected

@pytest.mark.parametrize('method', ['apply', 'map'])
@pytest.mark.parametrize('axis', ['