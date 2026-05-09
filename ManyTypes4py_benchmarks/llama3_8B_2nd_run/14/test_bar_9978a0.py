import io
import numpy as np
import pytest
from pandas import NA, DataFrame, read_csv
pytest.importorskip('jinja2')

def bar_grad(a: str | None, b: str | None, c: str | None, d: str | None) -> list[tuple[str, str]]:
    """Used in multiple tests to simplify formatting of expected result"""
    ret: list[tuple[str, str]] = [('width', '10em')]
    if all((x is None for x in [a, b, c, d])):
        return ret
    return ret + [('background', f'linear-gradient(90deg,{",".join([x for x in [a, b, c, d] if x])})')]

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

@pytest.mark.parametrize('align', [str])
@pytest.mark.parametrize('exp', [list[tuple[str, str]]])
def test_align_positive_cases(df_pos: DataFrame, align: str, exp: list[tuple[str, str]]) -> None:
    result = df_pos.style.bar(align=align)._compute().ctx
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected

@pytest.mark.parametrize('align', [str])
@pytest.mark.parametrize('exp', [list[tuple[str, str]]])
def test_align_negative_cases(df_neg: DataFrame, align: str, exp: list[tuple[str, str]]) -> None:
    result = df_neg.style.bar(align=align)._compute().ctx
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected

# ... and so on for the rest of the functions and tests
