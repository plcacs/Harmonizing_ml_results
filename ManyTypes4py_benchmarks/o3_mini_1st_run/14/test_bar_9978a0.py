from __future__ import annotations
import io
import numpy as np
import pytest
from pandas import NA, DataFrame, read_csv
from typing import List, Tuple, Optional, Union, Callable, Dict, Any

# Helper type for style bar formatting
BarList = List[Tuple[str, str]]
AlignType = Union[str, float, Callable[[List[float]], float]]

pytest.importorskip('jinja2')

def bar_grad(a: Optional[str] = None, b: Optional[str] = None, c: Optional[str] = None, d: Optional[str] = None) -> BarList:
    """Used in multiple tests to simplify formatting of expected result"""
    ret: BarList = [('width', '10em')]
    if all((x is None for x in [a, b, c, d])):
        return ret
    return ret + [('background', f'linear-gradient(90deg,{",".join([x for x in [a, b, c, d] if x])})')]

def no_bar() -> BarList:
    return bar_grad()

def bar_to(x: float, color: str = '#d65f5f') -> BarList:
    return bar_grad(f' {color} {x:.1f}%', f' transparent {x:.1f}%')

def bar_from_to(x: float, y: float, color: str = '#d65f5f') -> BarList:
    return bar_grad(f' transparent {x:.1f}%', f' {color} {x:.1f}%', f' {color} {y:.1f}%', f' transparent {y:.1f}%')

@pytest.fixture
def df_pos() -> DataFrame:
    return DataFrame([[1], [2], [3]])

@pytest.fixture
def df_neg() -> DataFrame:
    return DataFrame([[-1], [-2], [-3]])

@pytest.fixture
def df_mix() -> DataFrame:
    return DataFrame([[-3], [1], [2]])

@pytest.mark.parametrize(
    'align, exp',
    [
        ('left', [no_bar(), bar_to(50), bar_to(100)]),
        ('right', [bar_to(100), bar_from_to(50, 100), no_bar()]),
        ('mid', [bar_to(33.33), bar_to(66.66), bar_to(100)]),
        ('zero', [bar_from_to(50, 66.7), bar_from_to(50, 83.3), bar_from_to(50, 100)]),
        ('mean', [bar_to(50), no_bar(), bar_from_to(50, 100)]),
        (2.0, [bar_to(50), no_bar(), bar_from_to(50, 100)]),
        (np.median, [bar_to(50), no_bar(), bar_from_to(50, 100)]),
    ],
)
def test_align_positive_cases(df_pos: DataFrame, align: AlignType, exp: List[BarList]) -> None:
    result: Dict[Tuple[int, int], BarList] = df_pos.style.bar(align=align)._compute().ctx
    expected: Dict[Tuple[int, int], BarList] = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected

@pytest.mark.parametrize(
    'align, exp',
    [
        ('left', [bar_to(100), bar_to(50), no_bar()]),
        ('right', [no_bar(), bar_from_to(50, 100), bar_to(100)]),
        ('mid', [bar_from_to(66.66, 100), bar_from_to(33.33, 100), bar_to(100)]),
        ('zero', [bar_from_to(33.33, 50), bar_from_to(16.66, 50), bar_to(50)]),
        ('mean', [bar_from_to(50, 100), no_bar(), bar_to(50)]),
        (-2.0, [bar_from_to(50, 100), no_bar(), bar_to(50)]),
        (np.median, [bar_from_to(50, 100), no_bar(), bar_to(50)]),
    ],
)
def test_align_negative_cases(df_neg: DataFrame, align: AlignType, exp: List[BarList]) -> None:
    result: Dict[Tuple[int, int], BarList] = df_neg.style.bar(align=align)._compute().ctx
    expected: Dict[Tuple[int, int], BarList] = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected

@pytest.mark.parametrize(
    'align, exp',
    [
        ('left', [no_bar(), bar_to(80), bar_to(100)]),
        ('right', [bar_to(100), bar_from_to(80, 100), no_bar()]),
        ('mid', [bar_to(60), bar_from_to(60, 80), bar_from_to(60, 100)]),
        ('zero', [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        ('mean', [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        (-0.0, [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        (np.nanmedian, [bar_to(50), no_bar(), bar_from_to(50, 62.5)]),
    ],
)
@pytest.mark.parametrize('nans', [True, False])
def test_align_mixed_cases(df_mix: DataFrame, align: AlignType, exp: List[BarList], nans: bool) -> None:
    expected: Dict[Tuple[int, int], BarList] = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    if nans:
        df_mix.loc[3, :] = np.nan
        expected.update({(3, 0): no_bar()})
    result: Dict[Tuple[int, int], BarList] = df_mix.style.bar(align=align)._compute().ctx
    assert result == expected

@pytest.mark.parametrize(
    'align, exp',
    [
        (
            'left',
            {
                'index': [[no_bar(), no_bar()], [bar_to(100), bar_to(100)]],
                'columns': [[no_bar(), bar_to(100)], [no_bar(), bar_to(100)]],
                'none': [[no_bar(), bar_to(33.33)], [bar_to(66.66), bar_to(100)]],
            },
        ),
        (
            'mid',
            {
                'index': [[bar_to(33.33), bar_to(50)], [bar_to(100), bar_to(100)]],
                'columns': [[bar_to(50), bar_to(100)], [bar_to(75), bar_to(100)]],
                'none': [[bar_to(25), bar_to(50)], [bar_to(75), bar_to(100)]],
            },
        ),
        (
            'zero',
            {
                'index': [[bar_from_to(50, 66.66), bar_from_to(50, 75)], [bar_from_to(50, 100), bar_from_to(50, 100)]],
                'columns': [[bar_from_to(50, 75), bar_from_to(50, 100)], [bar_from_to(50, 87.5), bar_from_to(50, 100)]],
                'none': [[bar_from_to(50, 62.5), bar_from_to(50, 75)], [bar_from_to(50, 87.5), bar_from_to(50, 100)]],
            },
        ),
        (
            2,
            {
                'index': [[bar_to(50), no_bar()], [bar_from_to(50, 100), bar_from_to(50, 100)]],
                'columns': [[bar_to(50), no_bar()], [bar_from_to(50, 75), bar_from_to(50, 100)]],
                'none': [[bar_from_to(25, 50), no_bar()], [bar_from_to(50, 75), bar_from_to(50, 100)]],
            },
        ),
    ],
)
@pytest.mark.parametrize('axis', ['index', 'columns', 'none'])
def test_align_axis(align: AlignType, exp: Dict[str, List[List[BarList]]], axis: str) -> None:
    data: DataFrame = DataFrame([[1, 2], [3, 4]])
    result: Dict[Tuple[int, int], BarList] = data.style.bar(align=align, axis=None if axis == 'none' else axis)._compute().ctx
    expected: Dict[Tuple[int, int], BarList] = {
        (0, 0): exp[axis][0][0],
        (0, 1): exp[axis][0][1],
        (1, 0): exp[axis][1][0],
        (1, 1): exp[axis][1][1],
    }
    assert result == expected

@pytest.mark.parametrize('values, vmin, vmax', [('positive', 1.5, 2.5), ('negative', -2.5, -1.5), ('mixed', -2.5, 1.5)])
@pytest.mark.parametrize('nullify', [None, 'vmin', 'vmax'])
@pytest.mark.parametrize('align', ['left', 'right', 'zero', 'mid'])
def test_vmin_vmax_clipping(
    df_pos: DataFrame,
    df_neg: DataFrame,
    df_mix: DataFrame,
    values: str,
    vmin: float,
    vmax: float,
    nullify: Optional[str],
    align: str,
) -> None:
    if align == 'mid':
        if values == 'positive':
            align = 'left'
        elif values == 'negative':
            align = 'right'
    df: DataFrame = {'positive': df_pos, 'negative': df_neg, 'mixed': df_mix}[values]
    vmin = None if nullify == 'vmin' else vmin
    vmax = None if nullify == 'vmax' else vmax
    clip_df: DataFrame = df.where(df <= (vmax if vmax is not None else 999), other=vmax)
    clip_df = clip_df.where(clip_df >= (vmin if vmin is not None else -999), other=vmin)
    result: Dict[Tuple[int, int], BarList] = df.style.bar(align=align, vmin=vmin, vmax=vmax, color=['red', 'green'])._compute().ctx
    expected: Dict[Tuple[int, int], BarList] = clip_df.style.bar(align=align, color=['red', 'green'])._compute().ctx
    assert result == expected

@pytest.mark.parametrize('values, vmin, vmax', [('positive', 0.5, 4.5), ('negative', -4.5, -0.5), ('mixed', -4.5, 4.5)])
@pytest.mark.parametrize('nullify', [None, 'vmin', 'vmax'])
@pytest.mark.parametrize('align', ['left', 'right', 'zero', 'mid'])
def test_vmin_vmax_widening(
    df_pos: DataFrame,
    df_neg: DataFrame,
    df_mix: DataFrame,
    values: str,
    vmin: float,
    vmax: float,
    nullify: Optional[str],
    align: str,
) -> None:
    if align == 'mid':
        if values == 'positive':
            align = 'left'
        elif values == 'negative':
            align = 'right'
    df: DataFrame = {'positive': df_pos, 'negative': df_neg, 'mixed': df_mix}[values]
    vmin = None if nullify == 'vmin' else vmin
    vmax = None if nullify == 'vmax' else vmax
    expand_df: DataFrame = df.copy()
    expand_df.loc[3, :], expand_df.loc[4, :] = (vmin, vmax)
    result: Dict[Tuple[int, int], BarList] = df.style.bar(align=align, vmin=vmin, vmax=vmax, color=['red', 'green'])._compute().ctx
    expected: Dict[Tuple[int, int], BarList] = expand_df.style.bar(align=align, color=['red', 'green'])._compute().ctx
    assert result.items() <= expected.items()

def test_numerics() -> None:
    data: DataFrame = DataFrame([[1, 'a'], [2, 'b']])
    result: Dict[Tuple[int, int], BarList] = data.style.bar()._compute().ctx
    assert (0, 1) not in result
    assert (1, 1) not in result

@pytest.mark.parametrize(
    'align, exp',
    [
        ('left', [no_bar(), bar_to(100, 'green')]),
        ('right', [bar_to(100, 'red'), no_bar()]),
        ('mid', [bar_to(25, 'red'), bar_from_to(25, 100, 'green')]),
        ('zero', [bar_from_to(33.33, 50, 'red'), bar_from_to(50, 100, 'green')]),
    ],
)
def test_colors_mixed(align: str, exp: List[BarList]) -> None:
    data: DataFrame = DataFrame([[-1], [3]])
    result: Dict[Tuple[int, int], BarList] = data.style.bar(align=align, color=['red', 'green'])._compute().ctx
    assert result == {(0, 0): exp[0], (1, 0): exp[1]}

def test_bar_align_height() -> None:
    data: DataFrame = DataFrame([[1], [2]])
    result: Dict[Tuple[int, int], BarList] = data.style.bar(align='left', height=50)._compute().ctx
    bg_s: str = 'linear-gradient(90deg, #d65f5f 100.0%, transparent 100.0%) no-repeat center'
    expected: Dict[Tuple[int, int], BarList] = {
        (0, 0): [('width', '10em')],
        (1, 0): [('width', '10em'), ('background', bg_s), ('background-size', '100% 50.0%')]
    }
    assert result == expected

def test_bar_value_error_raises() -> None:
    df: DataFrame = DataFrame({'A': [-100, -60, -30, -20]})
    msg: str = "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(align='poorly', color=['#d65f5f', '#5fba7d']).to_html()
    msg = '`width` must be a value in \\[0, 100\\]'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(width=200).to_html()
    msg = '`height` must be a value in \\[0, 100\\]'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(height=200).to_html()

def test_bar_color_and_cmap_error_raises() -> None:
    df: DataFrame = DataFrame({'A': [1, 2, 3, 4]})
    msg: str = '`color` and `cmap` cannot both be given'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color='#d65f5f', cmap='viridis').to_html()

def test_bar_invalid_color_type_error_raises() -> None:
    df: DataFrame = DataFrame({'A': [1, 2, 3, 4]})
    msg: str = "`color` must be string or list or tuple of 2 strings,\\(eg: color=\\['#d65f5f', '#5fba7d'\\]\\)"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=123).to_html()
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=['#d65f5f', '#5fba7d', '#abcdef']).to_html()

def test_styler_bar_with_NA_values() -> None:
    df1: DataFrame = DataFrame({'A': [1, 2, NA, 4]})
    df2: DataFrame = DataFrame([[NA, NA], [NA, NA]])
    expected_substring: str = 'style type='
    html_output1: str = df1.style.bar(subset='A').to_html()
    html_output2: str = df2.style.bar(align='left', axis=None).to_html()
    assert expected_substring in html_output1
    assert expected_substring in html_output2

def test_style_bar_with_pyarrow_NA_values() -> None:
    pytest.importorskip('pyarrow')
    data: str = (
        'name,age,test1,test2,teacher\n'
        '        Adam,15,95.0,80,Ashby\n'
        '        Bob,16,81.0,82,Ashby\n'
        '        Dave,16,89.0,84,Jones\n'
        '        Fred,15,,88,Jones'
    )
    df: DataFrame = read_csv(io.StringIO(data), dtype_backend='pyarrow')
    expected_substring: str = 'style type='
    html_output: str = df.style.bar(subset='test1').to_html()
    assert expected_substring in html_output
