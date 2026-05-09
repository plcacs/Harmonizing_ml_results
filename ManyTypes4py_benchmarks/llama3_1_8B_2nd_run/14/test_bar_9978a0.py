import io
import numpy as np
import pytest
from pandas import NA, DataFrame, read_csv
pytest.importorskip('jinja2')

def bar_grad(a: str | None = None, b: str | None = None, c: str | None = None, d: str | None = None) -> list[tuple[str, str]]:
    """Used in multiple tests to simplify formatting of expected result"""
    ret: list[tuple[str, str]] = [('width', '10em')]
    if all((x is None for x in [a, b, c, d])):
        return ret
    return ret + [('background', f'linear-gradient(90deg,{','.join([x for x in [a, b, c, d] if x])})')]

def no_bar() -> list[tuple[str, str]]:
    return bar_grad()

def bar_to(x: float, color: str = '#d65f5f') -> list[tuple[str, str]]:
    return bar_grad(f' {color} {x:.1f}%', f' transparent {x:.1f}%')

def bar_from_to(x: float, y: float, color: str = '#d65f5f') -> list[tuple[str, str]]:
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

@pytest.mark.parametrize('align: str', ['left', 'right', 'mid', 'mean', 'zero', '2.0', np.median])
def test_align_positive_cases(df_pos: DataFrame, align: str):
    result = df_pos.style.bar(align=align)._compute().ctx
    expected = {(0, 0): [no_bar(), bar_to(50), bar_to(100)], (1, 0): [no_bar(), bar_to(100), bar_to(100)], (2, 0): [no_bar(), bar_to(100), bar_to(100)]}
    assert result == expected

@pytest.mark.parametrize('align: str', ['left', 'right', 'mid', 'mean', 'zero', '-2.0', np.median])
def test_align_negative_cases(df_neg: DataFrame, align: str):
    result = df_neg.style.bar(align=align)._compute().ctx
    expected = {(0, 0): [bar_to(100), bar_to(50), no_bar()], (1, 0): [bar_to(100), bar_to(50), no_bar()], (2, 0): [bar_to(100), bar_to(50), no_bar()]}
    assert result == expected

@pytest.mark.parametrize('align: str', ['left', 'right', 'mid', 'mean', 'zero', '-0.0', np.nanmedian])
@pytest.mark.parametrize('nans: bool', [True, False])
def test_align_mixed_cases(df_mix: DataFrame, align: str, nans: bool):
    expected = {(0, 0): [no_bar(), bar_to(80), bar_to(100)], (1, 0): [bar_to(100), bar_from_to(80, 100), no_bar()], (2, 0): [bar_to(60), bar_from_to(60, 80), bar_from_to(60, 100)]}
    if nans:
        df_mix.loc[3, :] = np.nan
        expected.update({(3, 0): no_bar()})
    result = df_mix.style.bar(align=align)._compute().ctx
    assert result == expected

@pytest.mark.parametrize('align: str', ['left', 'right', 'mid', 'zero', 'mean', '-0.0', np.nanmedian])
def test_align_axis(align: str):
    data = DataFrame([[1, 2], [3, 4]])
    result = data.style.bar(align=align, axis=None)._compute().ctx
    expected = {(0, 0): [no_bar(), bar_to(100)], (0, 1): [bar_to(100), no_bar()], (1, 0): [bar_from_to(66.66, 100), bar_from_to(33.33, 100)], (1, 1): [bar_from_to(66.66, 100), bar_from_to(33.33, 100)]}
    assert result == expected

@pytest.mark.parametrize('values: str', ['positive', 'negative', 'mixed'])
@pytest.mark.parametrize('nullify: str | None', [None, 'vmin', 'vmax'])
@pytest.mark.parametrize('align: str', ['left', 'right', 'zero', 'mid'])
def test_vmin_vmax_clipping(df_pos: DataFrame, df_neg: DataFrame, df_mix: DataFrame, values: str, nullify: str | None, align: str):
    if align == 'mid':
        if values == 'positive':
            align = 'left'
        elif values == 'negative':
            align = 'right'
    df = {'positive': df_pos, 'negative': df_neg, 'mixed': df_mix}[values]
    vmin = None if nullify == 'vmin' else None
    vmax = None if nullify == 'vmax' else None
    clip_df = df.where(df <= (vmax if vmax else 999), other=vmax)
    clip_df = clip_df.where(clip_df >= (vmin if vmin else -999), other=vmin)
    result = df.style.bar(align=align, vmin=vmin, vmax=vmax, color=['red', 'green'])._compute().ctx
    expected = clip_df.style.bar(align=align, color=['red', 'green'])._compute().ctx
    assert result == expected

@pytest.mark.parametrize('values: str', ['positive', 'negative', 'mixed'])
@pytest.mark.parametrize('nullify: str | None', [None, 'vmin', 'vmax'])
@pytest.mark.parametrize('align: str', ['left', 'right', 'zero', 'mid'])
def test_vmin_vmax_widening(df_pos: DataFrame, df_neg: DataFrame, df_mix: DataFrame, values: str, nullify: str | None, align: str):
    if align == 'mid':
        if values == 'positive':
            align = 'left'
        elif values == 'negative':
            align = 'right'
    df = {'positive': df_pos, 'negative': df_neg, 'mixed': df_mix}[values]
    vmin = None if nullify == 'vmin' else None
    vmax = None if nullify == 'vmax' else None
    expand_df = df.copy()
    expand_df.loc[3, :], expand_df.loc[4, :] = (vmin, vmax)
    result = df.style.bar(align=align, vmin=vmin, vmax=vmax, color=['red', 'green'])._compute().ctx
    expected = expand_df.style.bar(align=align, color=['red', 'green'])._compute().ctx
    assert result.items() <= expected.items()

def test_numerics():
    data = DataFrame([[1, 'a'], [2, 'b']])
    result = data.style.bar()._compute().ctx
    assert (0, 1) not in result
    assert (1, 1) not in result

@pytest.mark.parametrize('align: str', ['left', 'right', 'mid', 'zero'])
def test_colors_mixed(align: str):
    data = DataFrame([[-1], [3]])
    result = data.style.bar(align=align, color=['red', 'green'])._compute().ctx
    assert result == {(0, 0): [no_bar(), bar_to(100, 'green')], (1, 0): [bar_to(100, 'red'), no_bar()]}

def test_bar_align_height():
    data = DataFrame([[1], [2]])
    result = data.style.bar(align='left', height=50)._compute().ctx
    bg_s = 'linear-gradient(90deg, #d65f5f 100.0%, transparent 100.0%) no-repeat center'
    expected = {(0, 0): [('width', '10em')], (1, 0): [('width', '10em'), ('background', bg_s), ('background-size', '100% 50.0%')]}
    assert result == expected

def test_bar_value_error_raises():
    df = DataFrame({'A': [-100, -60, -30, -20]})
    msg = "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(align='poorly', color=['#d65f5f', '#5fba7d']).to_html()
    msg = '`width` must be a value in \\[0, 100\\]'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(width=200).to_html()
    msg = '`height` must be a value in \\[0, 100\\]'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(height=200).to_html()

def test_bar_color_and_cmap_error_raises():
    df = DataFrame({'A': [1, 2, 3, 4]})
    msg = '`color` and `cmap` cannot both be given'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color='#d65f5f', cmap='viridis').to_html()

def test_bar_invalid_color_type_error_raises():
    df = DataFrame({'A': [1, 2, 3, 4]})
    msg = "`color` must be string or list or tuple of 2 strings,\\(eg: color=\\['#d65f5f', '#5fba7d'\\]\\)"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=123).to_html()
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=['#d65f5f', '#5fba7d', '#abcdef']).to_html()

def test_styler_bar_with_NA_values():
    df1 = DataFrame({'A': [1, 2, NA, 4]})
    df2 = DataFrame([[NA, NA], [NA, NA]])
    expected_substring = 'style type='
    html_output1 = df1.style.bar(subset='A').to_html()
    html_output2 = df2.style.bar(align='left', axis=None).to_html()
    assert expected_substring in html_output1
    assert expected_substring in html_output2

def test_style_bar_with_pyarrow_NA_values():
    pytest.importorskip('pyarrow')
    data = 'name,age,test1,test2,teacher\n        Adam,15,95.0,80,Ashby\n        Bob,16,81.0,82,Ashby\n        Dave,16,89.0,84,Jones\n        Fred,15,,88,Jones'
    df = read_csv(io.StringIO(data), dtype_backend='pyarrow')
    expected_substring = 'style type='
    html_output = df.style.bar(subset='test1').to_html()
    assert expected_substring in html_output
