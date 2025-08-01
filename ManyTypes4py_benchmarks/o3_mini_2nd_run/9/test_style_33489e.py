from typing import Any, Dict, List, Tuple, Union, Optional, ContextManager
import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import DataFrame, IndexSlice, MultiIndex, Series, option_context
import pandas._testing as tm
jinja2 = pytest.importorskip('jinja2')
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _get_level_lengths, _get_trimming_maximums, maybe_convert_css_to_tuples, non_reducing_slice

@pytest.fixture
def mi_df() -> DataFrame:
    return DataFrame(
        [[1, 2], [3, 4]],
        index=MultiIndex.from_product([['i0'], ['i1_a', 'i1_b']]),
        columns=MultiIndex.from_product([['c0'], ['c1_a', 'c1_b']]),
        dtype=int
    )

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
    other: DataFrame = mi_styler.data.agg(['mean'])
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
    df_obj = DataFrame({'A': [0, 1], 'B': np.random.default_rng(2).standard_normal(2)})
    return df_obj

@pytest.fixture
def styler(df: DataFrame) -> Styler:
    df_obj = DataFrame({'A': [0, 1], 'B': np.random.default_rng(2).standard_normal(2)})
    return Styler(df_obj)

@pytest.mark.parametrize('sparse_columns, exp_cols', [
    (True, [{'is_visible': True, 'attributes': 'colspan="2"', 'value': 'c0'},
            {'is_visible': False, 'attributes': '', 'value': 'c0'}]),
    (False, [{'is_visible': True, 'attributes': '', 'value': 'c0'},
             {'is_visible': True, 'attributes': '', 'value': 'c0'}])
])
def test_mi_styler_sparsify_columns(mi_styler: Styler, sparse_columns: bool, exp_cols: List[Dict[str, Any]]) -> None:
    exp_l1_c0: Dict[str, Any] = {'is_visible': True, 'attributes': '', 'display_value': 'c1_a'}
    exp_l1_c1: Dict[str, Any] = {'is_visible': True, 'attributes': '', 'display_value': 'c1_b'}
    ctx: Dict[str, Any] = mi_styler._translate(True, sparse_columns)
    assert exp_cols[0].items() <= ctx['head'][0][2].items()
    assert exp_cols[1].items() <= ctx['head'][0][3].items()
    assert exp_l1_c0.items() <= ctx['head'][1][2].items()
    assert exp_l1_c1.items() <= ctx['head'][1][3].items()

@pytest.mark.parametrize('sparse_index, exp_rows', [
    (True, [{'is_visible': True, 'attributes': 'rowspan="2"', 'value': 'i0'},
            {'is_visible': False, 'attributes': '', 'value': 'i0'}]),
    (False, [{'is_visible': True, 'attributes': '', 'value': 'i0'},
             {'is_visible': True, 'attributes': '', 'value': 'i0'}])
])
def test_mi_styler_sparsify_index(mi_styler: Styler, sparse_index: bool, exp_rows: List[Dict[str, Any]]) -> None:
    exp_l1_r0: Dict[str, Any] = {'is_visible': True, 'attributes': '', 'display_value': 'i1_a'}
    exp_l1_r1: Dict[str, Any] = {'is_visible': True, 'attributes': '', 'display_value': 'i1_b'}
    ctx: Dict[str, Any] = mi_styler._translate(sparse_index, True)
    assert exp_rows[0].items() <= ctx['body'][0][0].items()
    assert exp_rows[1].items() <= ctx['body'][1][0].items()
    assert exp_l1_r0.items() <= ctx['body'][0][1].items()
    assert exp_l1_r1.items() <= ctx['body'][1][1].items()

def test_mi_styler_sparsify_options(mi_styler: Styler) -> None:
    with option_context('styler.sparse.index', False):
        html1: str = mi_styler.to_html()
    with option_context('styler.sparse.index', True):
        html2: str = mi_styler.to_html()
    assert html1 != html2
    with option_context('styler.sparse.columns', False):
        html1 = mi_styler.to_html()
    with option_context('styler.sparse.columns', True):
        html2 = mi_styler.to_html()
    assert html1 != html2

@pytest.mark.parametrize('rn, cn, max_els, max_rows, max_cols, exp_rn, exp_cn', [
    (100, 100, 100, None, None, 12, 6),
    (1000, 3, 750, None, None, 250, 3),
    (4, 1000, 500, None, None, 4, 125),
    (1000, 3, 750, 10, None, 10, 3),
    (4, 1000, 500, None, 5, 4, 5),
    (100, 100, 700, 50, 50, 25, 25)
])
def test_trimming_maximum(rn: int, cn: int, max_els: int, max_rows: Optional[int], max_cols: Optional[int], exp_rn: int, exp_cn: int) -> None:
    rn_out, cn_out = _get_trimming_maximums(rn, cn, max_els, max_rows, max_cols, scaling_factor=0.5)
    assert (rn_out, cn_out) == (exp_rn, exp_cn)

@pytest.mark.parametrize('option, val', [
    ('styler.render.max_elements', 6),
    ('styler.render.max_rows', 3)
])
def test_render_trimming_rows(option: str, val: int) -> None:
    df_obj: DataFrame = DataFrame(np.arange(120).reshape(60, 2))
    with option_context(option, val):
        ctx: Dict[str, Any] = df_obj.style._translate(True, True)
    assert len(ctx['head'][0]) == 3
    assert len(ctx['body']) == 4
    assert len(ctx['body'][0]) == 3

@pytest.mark.parametrize('option, val', [
    ('styler.render.max_elements', 6),
    ('styler.render.max_columns', 2)
])
def test_render_trimming_cols(option: str, val: int) -> None:
    df_obj: DataFrame = DataFrame(np.arange(30).reshape(3, 10))
    with option_context(option, val):
        ctx: Dict[str, Any] = df_obj.style._translate(True, True)
    assert len(ctx['head'][0]) == 4
    assert len(ctx['body']) == 3
    assert len(ctx['body'][0]) == 4

def test_render_trimming_mi() -> None:
    midx: MultiIndex = MultiIndex.from_product([[1, 2], [1, 2, 3]])
    df_obj: DataFrame = DataFrame(np.arange(36).reshape(6, 6), columns=midx, index=midx)
    with option_context('styler.render.max_elements', 4):
        ctx: Dict[str, Any] = df_obj.style._translate(True, True)
    assert len(ctx['body'][0]) == 5
    assert {'attributes': 'rowspan="2"'}.items() <= ctx['body'][0][0].items()
    assert {'class': 'data row0 col_trim'}.items() <= ctx['body'][0][4].items()
    assert {'class': 'data row_trim col_trim'}.items() <= ctx['body'][2][4].items()
    assert len(ctx['body']) == 3

def test_render_empty_mi() -> None:
    df_obj: DataFrame = DataFrame(index=MultiIndex.from_product([['A'], [0, 1]], names=[None, 'one']))
    expected: str = dedent('    >\n      <thead>\n        <tr>\n          <th class="index_name level0" >&nbsp;</th>\n          <th class="index_name level1" >one</th>\n        </tr>\n      </thead>\n    ')
    assert expected in df_obj.style.to_html()

@pytest.mark.parametrize('comprehensive', [True, False])
@pytest.mark.parametrize('render', [True, False])
@pytest.mark.parametrize('deepcopy', [True, False])
def test_copy(comprehensive: bool, render: bool, deepcopy: bool, mi_styler: Styler, mi_styler_comp: Styler) -> None:
    styler: Styler = mi_styler_comp if comprehensive else mi_styler
    styler.uuid_len = 5
    s2: Styler = copy.deepcopy(styler) if deepcopy else copy.copy(styler)
    assert s2 is not styler
    if render:
        styler.to_html()
    excl: List[str] = ['cellstyle_map', 'cellstyle_map_columns', 'cellstyle_map_index', 'template_latex', 'template_html', 'template_html_style', 'template_html_table']
    if not deepcopy:
        for attr in [a for a in styler.__dict__ if not callable(a) and a not in excl]:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))
    else:
        shallow: List[str] = ['data', 'columns', 'index', 'uuid_len', 'uuid', 'caption', 'cell_ids', 'hide_index_', 'hide_columns_', 'hide_index_names', 'hide_column_names', 'table_attributes']
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
    custom_styler: Styler = CustomStyler(mi_styler.data)
    custom_styler_copy: Styler = copy.deepcopy(custom_styler) if deepcopy else copy.copy(custom_styler)
    assert isinstance(custom_styler_copy, CustomStyler)

def test_clear(mi_styler_comp: Styler) -> None:
    styler: Styler = mi_styler_comp
    styler._compute()
    clean_copy: Styler = Styler(styler.data, uuid=styler.uuid)
    excl: List[str] = ['data', 'index', 'columns', 'uuid', 'uuid_len', 'cell_ids', 'cellstyle_map', 'cellstyle_map_columns', 'cellstyle_map_index', 'template_latex', 'template_html', 'template_html_style', 'template_html_table']
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
    exp_attrs: List[str] = ['_todo', 'hide_index_', 'hide_index_names', 'hide_columns_', 'hide_column_names', 'table_attributes', 'table_styles', 'css']
    for attr in exp_attrs:
        check = getattr(mi_styler, attr) == getattr(mi_styler_comp, attr)
        assert not (all(check) if hasattr(check, '__iter__') and len(check) > 0 else check)
    export: Dict[str, Any] = mi_styler_comp.export()
    used: Styler = mi_styler.use(export)
    for attr in exp_attrs:
        check = getattr(used, attr) == getattr(mi_styler_comp, attr)
        assert all(check) if hasattr(check, '__iter__') and len(check) > 0 else check
    used.to_html()

def test_hide_raises(mi_styler: Styler) -> None:
    msg1: str = '`subset` and `level` cannot be passed simultaneously'
    with pytest.raises(ValueError, match=msg1):
        mi_styler.hide(axis='index', subset='something', level='something else')
    msg2: str = '`level` must be of type `int`, `str` or list of such'
    with pytest.raises(ValueError, match=msg2):
        mi_styler.hide(axis='index', level={'bad': 1, 'type': 2})

@pytest.mark.parametrize('level', [1, 'one', [1], ['one']])
def test_hide_index_level(mi_styler: Styler, level: Union[int, str, List[Union[int, str]]]) -> None:
    mi_styler.index.names, mi_styler.columns.names = (['zero', 'one'], ['zero', 'one'])
    ctx: Dict[str, Any] = mi_styler.hide(axis='index', level=level)._translate(False, True)
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
    ctx: Dict[str, Any] = mi_styler.hide(axis='columns', level=level)._translate(True, False)
    assert len(ctx['head']) == (2 if names else 1)

@pytest.mark.parametrize('method', ['map', 'apply'])
@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_apply_map_header(method: str, axis: str) -> None:
    df_obj: DataFrame = DataFrame({'A': [0, 0], 'B': [1, 1]}, index=['C', 'D'])
    func: Dict[str, Any] = {
        'apply': lambda s: ['attr: val' if 'A' in v or 'C' in v else '' for v in s],
        'map': lambda v: 'attr: val' if 'A' in v or 'C' in v else ''
    }
    result: Styler = getattr(df_obj.style, f'{method}_index')(func[method], axis=axis)
    assert len(result._todo) == 1
    assert len(getattr(result, f'ctx_{axis}')) == 0
    result._compute()
    expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {(0, 0): [('attr', 'val')]}
    assert getattr(result, f'ctx_{axis}') == expected

@pytest.mark.parametrize('method', ['apply', 'map'])
@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_apply_map_header_mi(mi_styler: Styler, method: str, axis: str) -> None:
    func: Dict[str, Any] = {
        'apply': lambda s: ['attr: val;' if 'b' in v else '' for v in s],
        'map': lambda v: 'attr: val' if 'b' in v else ''
    }
    result: Styler = getattr(mi_styler, f'{method}_index')(func[method], axis=axis)._compute()
    expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {(1, 1): [('attr', 'val')]}
    assert getattr(result, f'ctx_{axis}') == expected

def test_apply_map_header_raises(mi_styler: Styler) -> None:
    with pytest.raises(ValueError, match='No axis named bad for object type DataFrame'):
        mi_styler.map_index(lambda v: 'attr: val;', axis='bad')._compute()

class TestStyler:
    def test_init_non_pandas(self) -> None:
        msg: str = '``data`` must be a Series or DataFrame'
        with pytest.raises(TypeError, match=msg):
            Styler([1, 2, 3])

    def test_init_series(self) -> None:
        result: Styler = Styler(Series([1, 2]))
        assert result.data.ndim == 2

    def test_repr_html_ok(self, styler: Styler) -> None:
        styler._repr_html_()

    def test_repr_html_mathjax(self, styler: Styler) -> None:
        assert 'tex2jax_ignore' not in styler._repr_html_()
        assert 'mathjax_ignore' not in styler._repr_html_()
        with option_context('styler.html.mathjax', False):
            assert 'tex2jax_ignore' in styler._repr_html_()
            assert 'mathjax_ignore' in styler._repr_html_()

    def test_update_ctx(self, styler: Styler) -> None:
        styler._update_ctx(DataFrame({'A': ['color: red', 'color: blue']}))
        expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {(0, 0): [('color', 'red')], (1, 0): [('color', 'blue')]}
        assert styler.ctx == expected

    def test_update_ctx_flatten_multi_and_trailing_semi(self, styler: Styler) -> None:
        attrs: DataFrame = DataFrame({'A': ['color: red; foo: bar', 'color:blue ; foo: baz;']})
        styler._update_ctx(attrs)
        expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {
            (0, 0): [('color', 'red'), ('foo', 'bar')],
            (1, 0): [('color', 'blue'), ('foo', 'baz')]
        }
        assert styler.ctx == expected

    def test_render(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [0, 1]})
        style_func = lambda x: Series(['color: red', 'color: blue'], name=x.name)
        s: Styler = Styler(df_obj, uuid='AB').apply(style_func)
        s.to_html()

    def test_multiple_render(self, df: DataFrame) -> None:
        s: Styler = Styler(df, uuid_len=0).map(lambda x: 'color: red;', subset=['A'])
        s.to_html()
        assert '<style type="text/css">\n#T__row0_col0, #T__row1_col0 {\n  color: red;\n}\n</style>' in s.to_html()

    def test_render_empty_dfs(self) -> None:
        empty_df: DataFrame = DataFrame()
        es: Styler = Styler(empty_df)
        es.to_html()
        DataFrame(columns=['a']).style.to_html()
        DataFrame(index=['a']).style.to_html()

    def test_render_double(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [0, 1]})
        style_func = lambda x: Series(['color: red; border: 1px', 'color: blue; border: 2px'], name=x.name)
        s: Styler = Styler(df_obj, uuid='AB').apply(style_func)
        s.to_html()

    def test_set_properties(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [0, 1]})
        result: Dict[Tuple[int, int], List[Tuple[str, str]]] = df_obj.style.set_properties(color='white', size='10px')._compute().ctx
        v: List[Tuple[str, str]] = [('color', 'white'), ('size', '10px')]
        expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {(0, 0): v, (1, 0): v}
        assert result.keys() == expected.keys()
        for v1, v2 in zip(result.values(), expected.values()):
            assert sorted(v1) == sorted(v2)

    def test_set_properties_subset(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [0, 1]})
        result: Dict[Tuple[int, int], List[Tuple[str, str]]] = df_obj.style.set_properties(subset=IndexSlice[0, 'A'], color='white')._compute().ctx
        expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {(0, 0): [('color', 'white')]}
        assert result == expected

    def test_empty_index_name_doesnt_display(self, blank_value: str) -> None:
        df_obj: DataFrame = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result: Dict[str, Any] = df_obj.style._translate(True, True)
        assert len(result['head']) == 1
        expected: Dict[str, Any] = {'class': 'blank level0', 'type': 'th', 'value': blank_value, 'is_visible': True, 'display_value': blank_value}
        assert expected.items() <= result['head'][0][0].items()

    def test_index_name(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result: Dict[str, Any] = df_obj.set_index('A').style._translate(True, True)
        expected: Dict[str, Any] = {'class': 'index_name level0', 'type': 'th', 'value': 'A', 'is_visible': True, 'display_value': 'A'}
        assert expected.items() <= result['head'][1][0].items()

    def test_numeric_columns(self) -> None:
        df_obj: DataFrame = DataFrame({0: [1, 2, 3]})
        df_obj.style._translate(True, True)

    def test_apply_axis(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [0, 0], 'B': [1, 1]})
        f = lambda x: [f'val: {x.max()}' for v in x]
        result: Styler = df_obj.style.apply(f, axis=1)
        assert len(result._todo) == 1
        assert len(result.ctx) == 0
        result._compute()
        expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {
            (0, 0): [('val', '1')],
            (0, 1): [('val', '1')],
            (1, 0): [('val', '1')],
            (1, 1): [('val', '1')]
        }
        assert result.ctx == expected
        result = df_obj.style.apply(f, axis=0)
        expected = {(0, 0): [('val', '0')], (0, 1): [('val', '1')], (1, 0): [('val', '0')], (1, 1): [('val', '1')]}
        result._compute()
        assert result.ctx == expected
        result = df_obj.style.apply(f)
        result._compute()
        assert result.ctx == expected

    @pytest.mark.parametrize('axis', [0, 1])
    def test_apply_series_return(self, axis: int) -> None:
        df_obj: DataFrame = DataFrame([[1, 2], [3, 4]], index=['X', 'Y'], columns=['X', 'Y'])
        func = lambda s: Series(['color: red;'], index=['Y'])
        result: Styler = df_obj.style.apply(func, axis=axis)._compute()
        assert result.ctx[(1, 1)] == [('color', 'red')]
        assert result.ctx[(1 - axis, axis)] == [('color', 'red')]
        func = lambda s: Series(['color: red;', 'color: blue;'], index=['Y', 'X'])
        result = df_obj.style.apply(func, axis=axis)._compute()
        assert result.ctx[(0, 0)] == [('color', 'blue')]
        assert result.ctx[(1, 1)] == [('color', 'red')]
        assert result.ctx[(1 - axis, axis)] == [('color', 'red')]
        assert result.ctx[(axis, 1 - axis)] == [('color', 'blue')]

    @pytest.mark.parametrize('index', [False, True])
    @pytest.mark.parametrize('columns', [False, True])
    def test_apply_dataframe_return(self, index: bool, columns: bool) -> None:
        df_obj: DataFrame = DataFrame([[1, 2], [3, 4]], index=['X', 'Y'], columns=['X', 'Y'])
        idxs: List[str] = ['X', 'Y'] if index else ['Y']
        cols: List[str] = ['X', 'Y'] if columns else ['Y']
        df_styles: DataFrame = DataFrame('color: red;', index=idxs, columns=cols)
        result: Styler = df_obj.style.apply(lambda x: df_styles, axis=None)._compute()
        assert result.ctx[(1, 1)] == [('color', 'red')]
        assert (result.ctx[(0, 1)] == [('color', 'red')]) is index
        assert (result.ctx[(1, 0)] == [('color', 'red')]) is columns
        assert (result.ctx[(0, 0)] == [('color', 'red')]) is (index and columns)

    @pytest.mark.parametrize('slice_', [IndexSlice[:], IndexSlice[:, ['A']], IndexSlice[[1], :], IndexSlice[[1], ['A']], IndexSlice[:2, ['A', 'B']]])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_apply_subset(self, slice_: Any, axis: int, df: DataFrame) -> None:
        def h(x: Any, color: str = 'bar') -> Series:
            return Series(f'color: {color}', index=x.index, name=x.name)
        result: Styler = df.style.apply(h, axis=axis, subset=slice_, color='baz')._compute()
        expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {
            (r, c): [('color', 'baz')]
            for r, row in enumerate(df.index)
            for c, col in enumerate(df.columns)
            if row in df.loc[slice_].index and col in df.loc[slice_].columns
        }
        assert result.ctx == expected

    @pytest.mark.parametrize('slice_', [IndexSlice[:], IndexSlice[:, ['A']], IndexSlice[[1], :], IndexSlice[[1], ['A']], IndexSlice[:2, ['A', 'B']]])
    def test_map_subset(self, slice_: Any, df: DataFrame) -> None:
        result: Styler = df.style.map(lambda x: 'color:baz;', subset=slice_)._compute()
        expected: Dict[Tuple[int, int], List[Tuple[str, str]]] = {
            (r, c): [('color', 'baz')]
            for r, row in enumerate(df.index)
            for c, col in enumerate(df.columns)
            if row in df.loc[slice_].index and col in df.loc[slice_].columns
        }
        assert result.ctx == expected

    @pytest.mark.parametrize('slice_', [IndexSlice[:, IndexSlice['x', 'A']], IndexSlice[:, IndexSlice[:, 'A']], IndexSlice[:, IndexSlice[:, ['A', 'C']]], IndexSlice[IndexSlice['a', 1], :], IndexSlice[IndexSlice[:, 1], :], IndexSlice[IndexSlice[:, [1, 3]], :], IndexSlice[:, ('x', 'A')], IndexSlice[('a', 1), :]])
    def test_map_subset_multiindex(self, slice_: Any) -> None:
        if isinstance(slice_[-1], tuple) and isinstance(slice_[-1][-1], list) and ('C' in slice_[-1][-1]):
            ctx_manager: ContextManager = pytest.raises(KeyError, match='C')
        elif isinstance(slice_[0], tuple) and isinstance(slice_[0][1], list) and (3 in slice_[0][1]):
            ctx_manager = pytest.raises(KeyError, match='3')
        else:
            ctx_manager = contextlib.nullcontext()
        idx: MultiIndex = MultiIndex.from_product([['a', 'b'], [1, 2]])
        col: MultiIndex = MultiIndex.from_product([['x', 'y'], ['A', 'B']])
        df_obj: DataFrame = DataFrame(np.random.default_rng(2).random((4, 4)), columns=col, index=idx)
        with ctx_manager:
            df_obj.style.map(lambda x: 'color: red;', subset=slice_).to_html()

    def test_map_subset_multiindex_code(self) -> None:
        codes: np.ndarray = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        columns: MultiIndex = MultiIndex(levels=[['a', 'b'], ['%', '#']], codes=codes, names=['', ''])
        df_obj: DataFrame = DataFrame([[1, -1, 1, 1], [-1, 1, 1, 1]], index=['hello', 'world'], columns=columns)
        pct_subset: Any = IndexSlice[:, IndexSlice[:, '%':'%']]
        def color_negative_red(val: Any) -> str:
            color: str = 'red' if val < 0 else 'black'
            return f'color: {color}'
        df_obj.loc[pct_subset]
        df_obj.style.map(color_negative_red, subset=pct_subset)

    @pytest.mark.parametrize('stylefunc', ['background_gradient', 'bar', 'text_gradient'])
    def test_subset_for_boolean_cols(self, stylefunc: str) -> None:
        df_obj: DataFrame = DataFrame([[1, 2], [3, 4]], columns=[False, True])
        styled: Styler = getattr(df_obj.style, stylefunc)()
        styled._compute()
        assert set(styled.ctx) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_empty(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [1, 0]})
        s: Styler = df_obj.style
        s.ctx = {(0, 0): [('color', 'red')], (1, 0): [('', '')]}
        result: List[Dict[str, Any]] = s._translate(True, True)['cellstyle']
        expected: List[Dict[str, Any]] = [
            {'props': [('color', 'red')], 'selectors': ['row0_col0']},
            {'props': [('', '')], 'selectors': ['row1_col0']}
        ]
        assert result == expected

    def test_duplicate(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [1, 0]})
        s: Styler = df_obj.style
        s.ctx = {(0, 0): [('color', 'red')], (1, 0): [('color', 'red')]}
        result: List[Dict[str, Any]] = s._translate(True, True)['cellstyle']
        expected: List[Dict[str, Any]] = [{'props': [('color', 'red')], 'selectors': ['row0_col0', 'row1_col0']}]
        assert result == expected

    def test_init_with_na_rep(self) -> None:
        df_obj: DataFrame = DataFrame([[None, None], [1.1, 1.2]], columns=['A', 'B'])
        ctx: Dict[str, Any] = Styler(df_obj, na_rep='NA')._translate(True, True)
        assert ctx['body'][0][1]['display_value'] == 'NA'
        assert ctx['body'][0][2]['display_value'] == 'NA'

    def test_caption(self, df: DataFrame) -> None:
        styler_obj: Styler = Styler(df, caption='foo')
        result: str = styler_obj.to_html()
        assert all(['caption' in result, 'foo' in result])
        styler_obj = df.style
        result = styler_obj.set_caption('baz')
        assert styler_obj is result
        assert styler_obj.caption == 'baz'

    def test_uuid(self, df: DataFrame) -> None:
        styler_obj: Styler = Styler(df, uuid='abc123')
        result: str = styler_obj.to_html()
        assert 'abc123' in result
        styler_obj = df.style
        result = styler_obj.set_uuid('aaa')
        assert result is styler_obj
        assert result.uuid == 'aaa'

    def test_unique_id(self) -> None:
        df_obj: DataFrame = DataFrame({'a': [1, 3, 5, 6], 'b': [2, 4, 12, 21]})
        result: str = df_obj.style.to_html(uuid='test')
        assert 'test' in result
        ids: List[str] = re.findall('id="(.*?)"', result)
        assert np.unique(ids).size == len(ids)

    def test_table_styles(self, df: DataFrame) -> None:
        style: List[Dict[str, Any]] = [{'selector': 'th', 'props': [('foo', 'bar')]}]
        styler_obj: Styler = Styler(df, table_styles=style)
        result: str = ' '.join(styler_obj.to_html().split())
        assert 'th { foo: bar; }' in result
        styler_obj = df.style
        result = styler_obj.set_table_styles(style)
        assert styler_obj is result
        assert styler_obj.table_styles == style
        style = [{'selector': 'th', 'props': 'foo:bar;'}]
        styler_obj = df.style.set_table_styles(style)
        result = ' '.join(styler_obj.to_html().split())
        assert 'th { foo: bar; }' in result

    def test_table_styles_multiple(self, df: DataFrame) -> None:
        ctx: List[Dict[str, Any]] = df.style.set_table_styles(
            [{'selector': 'th,td', 'props': 'color:red;'}, {'selector': 'tr', 'props': 'color:green;'}]
        )._translate(True, True)['table_styles']
        assert ctx == [
            {'selector': 'th', 'props': [('color', 'red')]},
            {'selector': 'td', 'props': [('color', 'red')]},
            {'selector': 'tr', 'props': [('color', 'green')]}
        ]

    def test_table_styles_dict_multiple_selectors(self, df: DataFrame) -> None:
        result: List[Dict[str, Any]] = df.style.set_table_styles(
            {'B': [{'selector': 'th,td', 'props': [('border-left', '2px solid black')]}]}
        )._translate(True, True)['table_styles']
        expected: List[Dict[str, Any]] = [
            {'selector': 'th.col1', 'props': [('border-left', '2px solid black')]},
            {'selector': 'td.col1', 'props': [('border-left', '2px solid black')]}
        ]
        assert result == expected

    def test_maybe_convert_css_to_tuples(self) -> None:
        expected: List[Tuple[str, str]] = [('a', 'b'), ('c', 'd e')]
        assert maybe_convert_css_to_tuples('a:b;c:d e;') == expected
        assert maybe_convert_css_to_tuples('a: b ;c:  d e  ') == expected
        expected = []
        assert maybe_convert_css_to_tuples('') == expected
        expected = [('a', 'b'), ('c', "url('data:123')")]
        assert maybe_convert_css_to_tuples("a:b;c: url('data:123');") == expected
        expected = [('a', ''), ('c', '')]
        assert maybe_convert_css_to_tuples('a:;c: ') == expected

    def test_maybe_convert_css_to_tuples_err(self) -> None:
        msg: str = "Styles supplied as string must follow CSS rule formats, for example 'attr: val;'. 'err' was given."
        with pytest.raises(ValueError, match=msg):
            maybe_convert_css_to_tuples('err')

    def test_table_attributes(self, df: DataFrame) -> None:
        attributes: str = 'class="foo" data-bar'
        styler_obj: Styler = Styler(df, table_attributes=attributes)
        result: str = styler_obj.to_html()
        assert 'class="foo" data-bar' in result
        result = df.style.set_table_attributes(attributes).to_html()
        assert 'class="foo" data-bar' in result

    def test_apply_none(self) -> None:
        def f(x: Any) -> DataFrame:
            return DataFrame(np.where(x == x.max(), 'color: red', ''), index=x.index, columns=x.columns)
        result: Styler = DataFrame([[1, 2], [3, 4]]).style.apply(f, axis=None)._compute()
        assert result.ctx[(1, 1)] == [('color', 'red')]

    def test_trim(self, df: DataFrame) -> None:
        result: str = df.style.to_html()
        assert result.count('#') == 0
        result = df.style.highlight_max().to_html()
        assert result.count('#') == len(df.columns)

    def test_export(self, df: DataFrame, styler: Styler) -> None:
        f = lambda x: 'color: red' if x > 0 else 'color: blue'
        g = lambda x, z: f'color: {z}'
        style1: Styler = styler
        style1.map(f).map(g, z='b').highlight_max()._compute()
        result_export: Dict[str, Any] = style1.export()
        style2: Styler = df.style
        style2.use(result_export)
        assert style1._todo == style2._todo
        style2.to_html()

    def test_bad_apply_shape(self) -> None:
        df_obj: DataFrame = DataFrame([[1, 2], [3, 4]], index=['A', 'B'], columns=['X', 'Y'])
        msg: str = 'resulted in the apply method collapsing to a Series.'
        with pytest.raises(ValueError, match=msg):
            df_obj.style._apply(lambda x: 'x')
        msg2: str = 'created invalid {} labels'
        with pytest.raises(ValueError, match=msg2.format('index')):
            df_obj.style._apply(lambda x: [''])
        with pytest.raises(ValueError, match=msg2.format('index')):
            df_obj.style._apply(lambda x: ['', '', '', ''])
        with pytest.raises(ValueError, match=msg2.format('index')):
            df_obj.style._apply(lambda x: Series(['a:v;', ''], index=['A', 'C']), axis=0)
        with pytest.raises(ValueError, match=msg2.format('columns')):
            df_obj.style._apply(lambda x: ['', '', ''], axis=1)
        with pytest.raises(ValueError, match=msg2.format('columns')):
            df_obj.style._apply(lambda x: Series(['a:v;', ''], index=['X', 'Z']), axis=1)
        msg3: str = 'returned ndarray with wrong shape'
        with pytest.raises(ValueError, match=msg3):
            df_obj.style._apply(lambda x: np.array([[''], ['']]), axis=None)

    def test_apply_bad_return(self) -> None:
        def f(x: Any) -> str:
            return ''
        df_obj: DataFrame = DataFrame([[1, 2], [3, 4]])
        msg: str = 'must return a DataFrame or ndarray when passed to `Styler.apply` with axis=None'
        with pytest.raises(TypeError, match=msg):
            df_obj.style._apply(f, axis=None)

    @pytest.mark.parametrize('axis', ['index', 'columns'])
    def test_apply_bad_labels(self, axis: str) -> None:
        def f(x: Any) -> DataFrame:
            return DataFrame(**{axis: ['bad', 'labels']})
        df_obj: DataFrame = DataFrame([[1, 2], [3, 4]])
        msg: str = f'created invalid {axis} labels.'
        with pytest.raises(ValueError, match=msg):
            df_obj.style._apply(f, axis=None)

    def test_get_level_lengths(self) -> None:
        index: MultiIndex = MultiIndex.from_product([['a', 'b'], [0, 1, 2]])
        expected: Dict[Any, int] = {(0, 0): 3, (0, 3): 3, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1}
        result: Dict[Any, int] = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)
        expected = {(0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1, (0, 5): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1}
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_get_level_lengths_un_sorted(self) -> None:
        index: MultiIndex = MultiIndex.from_arrays([[1, 1, 2, 1], ['a', 'b', 'b', 'd']])
        expected: Dict[Any, int] = {(0, 0): 2, (0, 2): 1, (0, 3): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1}
        result: Dict[Any, int] = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)
        expected = {(0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1}
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_mi_sparse_index_names(self, blank_value: str) -> None:
        df_obj: DataFrame = DataFrame({'A': [1, 2]}, index=MultiIndex.from_arrays([['a', 'a'], [0, 1]], names=['idx_level_0', 'idx_level_1']))
        result: Dict[str, Any] = df_obj.style._translate(True, True)
        head: List[Dict[str, Any]] = result['head'][1]
        expected: List[Dict[str, Any]] = [
            {'class': 'index_name level0', 'display_value': 'idx_level_0', 'is_visible': True},
            {'class': 'index_name level1', 'display_value': 'idx_level_1', 'is_visible': True},
            {'class': 'blank col0', 'display_value': blank_value, 'is_visible': True}
        ]
        for i, expected_dict in enumerate(expected):
            assert expected_dict.items() <= head[i].items()

    def test_mi_sparse_column_names(self, blank_value: str) -> None:
        df_obj: DataFrame = DataFrame(
            np.arange(16).reshape(4, 4),
            index=MultiIndex.from_arrays([['a', 'a', 'b', 'a'], [0, 1, 1, 2]], names=['idx_level_0', 'idx_level_1']),
            columns=MultiIndex.from_arrays([['C1', 'C1', 'C2', 'C2'], [1, 0, 1, 0]], names=['colnam_0', 'colnam_1'])
        )
        result: Dict[str, Any] = Styler(df_obj, cell_ids=False)._translate(True, True)
        for level in [0, 1]:
            head: List[Dict[str, Any]] = result['head'][level]
            expected: List[Dict[str, Any]] = [
                {'class': 'blank', 'display_value': blank_value, 'is_visible': True},
                {'class': f'index_name level{level}', 'display_value': f'colnam_{level}', 'is_visible': True}
            ]
            for i, expected_dict in enumerate(expected):
                assert expected_dict.items() <= head[i].items()

    def test_hide_column_headers(self, df: DataFrame, styler: Styler) -> None:
        ctx: Dict[str, Any] = styler.hide(axis='columns')._translate(True, True)
        assert len(ctx['head']) == 0
        df.index.name = 'some_name'
        ctx = df.style.hide(axis='columns')._translate(True, True)
        assert len(ctx['head']) == 1

    def test_hide_single_index(self, df: DataFrame) -> None:
        ctx: Dict[str, Any] = df.style._translate(True, True)
        assert ctx['body'][0][0]['is_visible']
        assert ctx['head'][0][0]['is_visible']
        ctx2: Dict[str, Any] = df.style.hide(axis='index')._translate(True, True)
        assert not ctx2['body'][0][0]['is_visible']
        assert not ctx2['head'][0][0]['is_visible']
        ctx3: Dict[str, Any] = df.set_index('A').style._translate(True, True)
        assert ctx3['body'][0][0]['is_visible']
        assert len(ctx3['head']) == 2
        assert ctx3['head'][0][0]['is_visible']
        ctx4: Dict[str, Any] = df.set_index('A').style.hide(axis='index')._translate(True, True)
        assert not ctx4['body'][0][0]['is_visible']
        assert len(ctx4['head']) == 1
        assert not ctx4['head'][0][0]['is_visible']

    def test_hide_multiindex(self) -> None:
        df_obj: DataFrame = DataFrame({'A': [1, 2], 'B': [1, 2]},
                                       index=MultiIndex.from_arrays([['a', 'a'], [0, 1]], names=['idx_level_0', 'idx_level_1']))
        ctx1: Dict[str, Any] = df_obj.style._translate(True, True)
        assert ctx1['body'][0][0]['is_visible']
        assert ctx1['body'][0][1]['is_visible']
        assert len(ctx1['head'][0]) == 4
        ctx2: Dict[str, Any] = df_obj.style.hide(axis='index')._translate(True, True)
        assert not ctx2['body'][0][0]['is_visible']
        assert not ctx2['body'][0][1]['is_visible']
        assert len(ctx2['head'][0]) == 3
        assert not ctx2['head'][0][0]['is_visible']

    def test_hide_columns_single_level(self, df: DataFrame) -> None:
        ctx: Dict[str, Any] = df.style._translate(True, True)
        assert ctx['head'][0][1]['is_visible']
        assert ctx['head'][0][1]['display_value'] == 'A'
        assert ctx['head'][0][2]['is_visible']
        assert ctx['head'][0][2]['display_value'] == 'B'
        assert ctx['body'][0][1]['is_visible']
        assert ctx['body'][1][2]['is_visible']
        ctx = df.style.hide('A', axis='columns')._translate(True, True)
        assert not ctx['head'][0][1]['is_visible']
        assert not ctx['body'][0][1]['is_visible']
        assert ctx['body'][1][2]['is_visible']
        ctx = df.style.hide(['A', 'B'], axis='columns')._translate(True, True)
        assert not ctx['head'][0][1]['is_visible']
        assert not ctx['head'][0][2]['is_visible']
        assert not ctx['body'][0][1]['is_visible']
        assert not ctx['body'][1][2]['is_visible']

    def test_hide_columns_index_mult_levels(self) -> None:
        i1: MultiIndex = MultiIndex.from_arrays([['a', 'a'], [0, 1]], names=['idx_level_0', 'idx_level_1'])
        i2: MultiIndex = MultiIndex.from_arrays([['b', 'b'], [0, 1]], names=['col_level_0', 'col_level_1'])
        df_obj: DataFrame = DataFrame([[1, 2], [3, 4]], index=i1, columns=i2)
        ctx: Dict[str, Any] = df_obj.style._translate(True, True)
        assert ctx['head'][0][2]['is_visible']
        assert ctx['head'][1][2]['is_visible']
        assert ctx['head'][1][3]['display_value'] == '1'
        assert ctx['body'][0][0]['is_visible']
        assert ctx['body'][1][2]['is_visible']
        assert ctx['body'][1][2]['display_value'] == '3'
        assert ctx['body'][1][3]['is_visible']
        assert ctx['body'][1][3]['display_value'] == '4'
        ctx = df_obj.style.hide('b', axis='columns')._translate(True, True)
        assert not ctx['head'][0][2]['is_visible']
        assert not ctx['head'][1][2]['is_visible']
        assert not ctx['body'][1][2]['is_visible']
        assert ctx['body'][0][0]['is_visible']
        ctx = df_obj.style.hide([('b', 0)], axis='columns')._translate(True, True)
        assert not ctx['head'][0][2]['is_visible']
        assert ctx['head'][0][3]['is_visible']
        assert not ctx['head'][1][2]['is_visible']
        assert not ctx['body'][1][2]['is_visible']
        assert ctx['body'][1][3]['is_visible']
        assert ctx['body'][1][3]['display_value'] == '4'
        ctx = df_obj.style.hide([('b', 1)], axis=1).hide(axis=0)._translate(True, True)
        assert not ctx['body'][0][0]['is_visible']
        assert len(ctx['head'][0]) == 3
        assert ctx['head'][0][1]['is_visible']
        assert ctx['head'][1][1]['is_visible']
        assert not ctx['head'][1][2]['is_visible']
        assert not ctx['body'][1][3]['is_visible']
        assert ctx['body'][1][2]['is_visible']
        assert ctx['body'][1][2]['display_value'] == '3'
        ctx = df_obj.style.hide('a', axis='index')._translate(True, True)
        assert ctx['body'] == []
        ctx = df_obj.style.hide(('a', 0), axis='index')._translate(True, True)
        for i in [0, 1, 2, 3]:
            assert 'row1' in ctx['body'][0][i]['class']
            assert ctx['body'][0][i]['is_visible']

    def test_pipe(self, df: DataFrame) -> None:
        def set_caption_from_template(styler: Styler, a: Any, b: Any) -> Styler:
            return styler.set_caption(f'Dataframe with a = {a} and b = {b}')
        styler_obj: Styler = df.style.pipe(set_caption_from_template, 'A', b='B')
        assert 'Dataframe with a = A and b = B' in styler_obj.to_html()

        def f(a: Any, b: Any, styler: Styler) -> Tuple[Any, Any, Styler]:
            return (a, b, styler)
        styler_obj = df.style
        result = styler_obj.pipe((f, 'styler'), a=1, b=2)
        assert result == (1, 2, styler_obj)

    def test_no_cell_ids(self) -> None:
        df_obj: DataFrame = DataFrame(data=[[0]])
        styler_obj: Styler = Styler(df_obj, uuid='_', cell_ids=False)
        styler_obj.to_html()
        s: str = styler_obj.to_html()
        assert s.find('<td class="data row0 col0" >') != -1

    @pytest.mark.parametrize('classes', [
        DataFrame(data=[['', 'test-class'], [np.nan, None]], columns=['A', 'B'], index=['a', 'b']),
        DataFrame(data=[['test-class']], columns=['B'], index=['a']),
        DataFrame(data=[['test-class', 'unused']], columns=['B', 'C'], index=['a'])
    ])
    def test_set_data_classes(self, classes: DataFrame) -> None:
        df_obj: DataFrame = DataFrame(data=[[0, 1], [2, 3]], columns=['A', 'B'], index=['a', 'b'])
        s: str = Styler(df_obj, uuid_len=0, cell_ids=False).set_td_classes(classes).to_html()
        assert '<td class="data row0 col0" >0</td>' in s
        assert '<td class="data row0 col1 test-class" >1</td>' in s
        assert '<td class="data row1 col0" >2</td>' in s
        assert '<td class="data row1 col1" >3</td>' in s
        s = Styler(df_obj, uuid_len=0, cell_ids=True).set_td_classes(classes).to_html()
        assert '<td id="T__row0_col0" class="data row0 col0" >0</td>' in s
        assert '<td id="T__row0_col1" class="data row0 col1 test-class" >1</td>' in s
        assert '<td id="T__row1_col0" class="data row1 col0" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >3</td>' in s

    def test_set_data_classes_reindex(self) -> None:
        df_obj: DataFrame = DataFrame(data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=[0, 1, 2], index=[0, 1, 2])
        classes: DataFrame = DataFrame(data=[['mi', 'ma'], ['mu', 'mo']], columns=[0, 2], index=[0, 2])
        s: str = Styler(df_obj, uuid_len=0).set_td_classes(classes).to_html()
        assert '<td id="T__row0_col0" class="data row0 col0 mi" >0</td>' in s
        assert '<td id="T__row0_col2" class="data row0 col2 ma" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >4</td>' in s
        assert '<td id="T__row2_col0" class="data row2 col0 mu" >6</td>' in s
        assert '<td id="T__row2_col2" class="data row2 col2 mo" >8</td>' in s

    def test_chaining_table_styles(self) -> None:
        df_obj: DataFrame = DataFrame(data=[[0, 1], [1, 2]], columns=['A', 'B'])
        styler_obj: Styler = df_obj.style.set_table_styles(
            [{'selector': '', 'props': [('background-color', 'yellow')]}]
        ).set_table_styles(
            [{'selector': '.col0', 'props': [('background-color', 'blue')]}], overwrite=False
        )
        assert len(styler_obj.table_styles) == 2

    def test_column_and_row_styling(self) -> None:
        df_obj: DataFrame = DataFrame(data=[[0, 1], [1, 2]], columns=['A', 'B'])
        s: Styler = Styler(df_obj, uuid_len=0)
        s = s.set_table_styles({'A': [{'selector': '', 'props': [('color', 'blue')]}]})
        assert '#T_ .col0 {\n  color: blue;\n}' in s.to_html()
        s = s.set_table_styles({0: [{'selector': '', 'props': [('color', 'blue')]}]}, axis=1)
        assert '#T_ .row0 {\n  color: blue;\n}' in s.to_html()

    @pytest.mark.parametrize('len_', [1, 5, 32, 33, 100])
    def test_uuid_len(self, len_: int) -> None:
        df_obj: DataFrame = DataFrame(data=[['A']])
        s: str = Styler(df_obj, uuid_len=len_, cell_ids=False).to_html()
        strt: int = s.find('id="T_')
        end: int = s[strt + 6:].find('"')
        if len_ > 32:
            assert end == 32
        else:
            assert end == len_

    @pytest.mark.parametrize('len_', [-2, 'bad', None])
    def test_uuid_len_raises(self, len_: Any) -> None:
        df_obj: DataFrame = DataFrame(data=[['A']])
        msg: str = '``uuid_len`` must be an integer in range \\[0, 32\\].'
        with pytest.raises(TypeError, match=msg):
            Styler(df_obj, uuid_len=len_, cell_ids=False).to_html()

    @pytest.mark.parametrize('slc', [
        IndexSlice[:, :], IndexSlice[:, 1], IndexSlice[1, :],
        IndexSlice[[1], [1]], IndexSlice[1, [1]], IndexSlice[[1], 1],
        IndexSlice[1], IndexSlice[1, 1], slice(None, None, None), [0, 1],
        np.array([0, 1]), Series([0, 1])
    ])
    def test_non_reducing_slice(self, slc: Any) -> None:
        df_obj: DataFrame = DataFrame([[0, 1], [2, 3]])
        tslice_: Any = non_reducing_slice(slc)
        assert isinstance(df_obj.loc[tslice_], DataFrame)

    @pytest.mark.parametrize('box', [list, Series, np.array])
    def test_list_slice(self, box: Any) -> None:
        subset: Any = box(['A'])
        df_obj: DataFrame = DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['A', 'B'])
        expected: Any = IndexSlice[:, ['A']]
        result: Any = non_reducing_slice(subset)
        tm.assert_frame_equal(df_obj.loc[result], df_obj.loc[expected])

    def test_non_reducing_slice_on_multiindex(self) -> None:
        dic: Dict[Tuple[Any, Any], List[int]] = {('a', 'd'): [1, 4], ('a', 'c'): [2, 3], ('b', 'c'): [3, 2], ('b', 'd'): [4, 1]}
        df_obj: DataFrame = DataFrame(dic, index=[0, 1])
        idx = IndexSlice
        slice_: Any = idx[:, idx['b', 'd']]
        tslice_: Any = non_reducing_slice(slice_)
        result: DataFrame = df_obj.loc[tslice_]
        expected: DataFrame = DataFrame({('b', 'd'): [4, 1]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('slice_', [
        IndexSlice[:, :], IndexSlice[:, IndexSlice[['a']]], IndexSlice[:, IndexSlice[['a'], ['c']]],
        IndexSlice[:, IndexSlice['a', 'c', :]], IndexSlice[:, IndexSlice['a', :, 'e']],
        IndexSlice[:, IndexSlice[:, 'c', 'e']], IndexSlice[:, IndexSlice['a', ['c', 'd'], :]],
        IndexSlice[:, IndexSlice['a', ['c', 'd', '-'], :]], IndexSlice[:, IndexSlice['a', ['c', 'd', '-'], 'e']],
        IndexSlice[IndexSlice[['U']], :], IndexSlice[IndexSlice[['U'], ['W']], :],
        IndexSlice[IndexSlice['U', 'W', :], :], IndexSlice[IndexSlice['U', :, 'Y'], :],
        IndexSlice[IndexSlice[:, 'W', 'Y'], :], IndexSlice[IndexSlice[:, 'W', ['Y', 'Z']], :],
        IndexSlice[IndexSlice[:, 'W', ['Y', 'Z', '-']], :], IndexSlice[IndexSlice['U', 'W', ['Y', 'Z', '-']], :],
        IndexSlice[IndexSlice[:, 'W', 'Y'], IndexSlice['a', 'c', :]]
    ])
    def test_non_reducing_multi_slice_on_multiindex(self, slice_: Any) -> None:
        cols: MultiIndex = MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']])
        idxs: MultiIndex = MultiIndex.from_product([['U', 'V'], ['W', 'X'], ['Y', 'Z']])
        df_obj: DataFrame = DataFrame(np.arange(64).reshape(8, 8), columns=cols, index=idxs)
        for lvl in [0, 1]:
            key = slice_[lvl]
            if isinstance(key, tuple):
                for subkey in key:
                    if isinstance(subkey, list) and '-' in subkey:
                        with pytest.raises(KeyError, match='-'):
                            df_obj.loc[slice_]
                        return
        expected: DataFrame = df_obj.loc[slice_]
        result: DataFrame = df_obj.loc[non_reducing_slice(slice_)]
        tm.assert_frame_equal(result, expected)

def test_hidden_index_names(mi_df: DataFrame) -> None:
    mi_df.index.names = ['Lev0', 'Lev1']
    mi_styler: Styler = mi_df.style
    ctx: Dict[str, Any] = mi_styler._translate(True, True)
    assert len(ctx['head']) == 3
    mi_styler.hide(axis='index', names=True)
    ctx = mi_styler._translate(True, True)
    assert len(ctx['head']) == 2
    for i in range(4):
        assert ctx['body'][0][i]['is_visible']
    mi_styler.hide(axis='index', level=1)
    ctx = mi_styler._translate(True, True)
    assert len(ctx['head']) == 2
    assert ctx['body'][0][0]['is_visible'] is True
    assert ctx['body'][0][1]['is_visible'] is False

def test_hidden_column_names(mi_df: DataFrame) -> None:
    mi_df.columns.names = ['Lev0', 'Lev1']
    mi_styler: Styler = mi_df.style
    ctx: Dict[str, Any] = mi_styler._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'Lev0'
    assert ctx['head'][1][1]['display_value'] == 'Lev1'
    mi_styler.hide(names=True, axis='columns')
    ctx = mi_styler._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == '&nbsp;'
    assert ctx['head'][1][1]['display_value'] == '&nbsp;'
    mi_styler.hide(level=0, axis='columns')
    ctx = mi_styler._translate(True, True)
    assert len(ctx['head']) == 1
    assert ctx['head'][0][1]['display_value'] == '&nbsp;'

@pytest.mark.parametrize('caption', [1, ('a', 'b', 'c'), (1, 's')])
def test_caption_raises(mi_styler: Styler, caption: Any) -> None:
    msg: str = '`caption` must be either a string or 2-tuple of strings.'
    with pytest.raises(ValueError, match=msg):
        mi_styler.set_caption(caption)

def test_hiding_headers_over_index_no_sparsify() -> None:
    midx: MultiIndex = MultiIndex.from_product([[1, 2], ['a', 'a', 'b']])
    df_obj: DataFrame = DataFrame(9, index=midx, columns=[0])
    ctx: Dict[str, Any] = df_obj.style._translate(False, False)
    assert len(ctx['body']) == 6
    ctx = df_obj.style.hide((1, 'a'), axis=0)._translate(False, False)
    assert len(ctx['body']) == 4
    assert 'row2' in ctx['body'][0][0]['class']

def test_hiding_headers_over_columns_no_sparsify() -> None:
    midx: MultiIndex = MultiIndex.from_product([[1, 2], ['a', 'a', 'b']])
    df_obj: DataFrame = DataFrame(9, columns=midx, index=[0])
    ctx: Dict[str, Any] = df_obj.style._translate(False, False)
    for ix in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        assert ctx['head'][ix[0]][ix[1]]['is_visible'] is True
    ctx = df_obj.style.hide((1, 'a'), axis='columns')._translate(False, False)
    for ix in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        assert ctx['head'][ix[0]][ix[1]]['is_visible'] is False

def test_get_level_lengths_mi_hidden() -> None:
    index: MultiIndex = MultiIndex.from_arrays([[1, 1, 1, 2, 2, 2], ['a', 'a', 'b', 'a', 'a', 'b']])
    expected: Dict[Any, int] = {(0, 2): 1, (0, 3): 1, (0, 4): 1, (0, 5): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1}
    result: Dict[Any, int] = _get_level_lengths(index, sparsify=False, max_index=100, hidden_elements=[0, 1, 0, 1])
    tm.assert_dict_equal(result, expected)

def test_row_trimming_hide_index() -> None:
    df_obj: DataFrame = DataFrame([[1], [2], [3], [4], [5]])
    with option_context('styler.render.max_rows', 2):
        ctx: Dict[str, Any] = df_obj.style.hide([0, 1], axis='index')._translate(True, True)
    assert len(ctx['body']) == 3
    for r, val in enumerate(['3', '4', '...']):
        assert ctx['body'][r][1]['display_value'] == val

def test_row_trimming_hide_index_mi() -> None:
    df_obj: DataFrame = DataFrame([[1], [2], [3], [4], [5]])
    df_obj.index = MultiIndex.from_product([[0], [0, 1, 2, 3, 4]])
    with option_context('styler.render.max_rows', 2):
        ctx: Dict[str, Any] = df_obj.style.hide([(0, 0), (0, 1)], axis='index')._translate(True, True)
    assert len(ctx['body']) == 3
    assert {'value': 0, 'attributes': 'rowspan="2"', 'is_visible': True}.items() <= ctx['body'][0][0].items()
    assert {'value': 0, 'attributes': '', 'is_visible': False}.items() <= ctx['body'][1][0].items()
    assert {'value': '...', 'is_visible': True}.items() <= ctx['body'][2][0].items()
    for r, val in enumerate(['2', '3', '...']):
        assert ctx['body'][r][1]['display_value'] == val
    for r, val in enumerate(['3', '4', '...']):
        assert ctx['body'][r][2]['display_value'] == val

def test_col_trimming_hide_columns() -> None:
    df_obj: DataFrame = DataFrame([[1, 2, 3, 4, 5]])
    with option_context('styler.render.max_columns', 2):
        ctx: Dict[str, Any] = df_obj.style.hide([0, 1], axis='columns')._translate(True, True)
    assert len(ctx['head'][0]) == 6
    for c, vals in enumerate([(1, False), (2, True), (3, True), ('...', True)]):
        assert ctx['head'][0][c + 2]['value'] == vals[0]
        assert ctx['head'][0][c + 2]['is_visible'] == vals[1]
    assert len(ctx['body'][0]) == 6

def test_no_empty_apply(mi_styler: Styler) -> None:
    mi_styler.apply(lambda s: ['a:v;'] * 2, subset=[False, False])
    mi_styler._compute()

@pytest.mark.parametrize('format', ['html', 'latex', 'string'])
def test_output_buffer(mi_styler: Styler, format: str) -> None:
    with tm.ensure_clean(f'delete_me.{format}') as f:
        getattr(mi_styler, f'to_{format}')(f)