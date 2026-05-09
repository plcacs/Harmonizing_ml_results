import numpy as np
import pytest
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
    return DataFrame(data=np.arange(16).reshape(4, 4), columns=MultiIndex.from_product([['A', 'B'], ['a', 'b']]), index=MultiIndex.from_product([['X', 'Y'], ['x', 'y']])).rename_axis(['0_0', '0_1'], axis=0).rename_axis(['1_0', '1_1'], axis=1)

@pytest.fixture
def styler_multi(df_multi: DataFrame) -> Styler:
    return Styler(df_multi, uuid_len=0)

def test_display_format(styler: Styler) -> None:
    ...

def test_display_format_index(styler: Styler, index: bool, columns: bool) -> None:
    ...

def test_format_dict(styler: Styler) -> None:
    ...

def test_format_index_dict(styler: Styler) -> None:
    ...

def test_format_string(styler: Styler) -> None:
    ...

def test_format_callable(styler: Styler) -> None:
    ...

def test_format_with_na_rep(styler: Styler) -> None:
    ...

def test_format_escape_html(styler: Styler, escape: str, exp: str) -> None:
    ...

def test_format_escape_latex_math(styler: Styler, chars: str, expected: str) -> None:
    ...

def test_format_escape_na_rep(styler: Styler) -> None:
    ...

def test_format_escape_floats(styler: Styler) -> None:
    ...

def test_format_thousands(styler: Styler, formatter: str | None, decimal: str, precision: int | None) -> None:
    ...

def test_format_decimal(styler: Styler, formatter: str | None, thousands: str | None, precision: int | None) -> None:
    ...

def test_formatter_options_validator(styler: Styler, formatter: callable, exp: str) -> None:
    ...

def test_formatter_options_raises(styler: Styler) -> None:
    ...

def test_relabel_raise_length(styler_multi: Styler, hide: bool, labels: list) -> None:
    ...

def test_relabel_index(styler_multi: Styler, labels: list) -> None:
    ...

def test_relabel_columns(styler_multi: Styler, labels: list) -> None:
    ...

def test_relabel_roundtrip(styler: Styler) -> None:
    ...

def test_format_index_names_level(styler: Styler, axis: int, level: int, expected: list) -> None:
    ...

def test_format_index_names_clear(styler: Styler, attr: str, kwargs: dict) -> None:
    ...

def test_format_index_names_callable(styler_multi: Styler, axis: int) -> None:
    ...

def test_format_index_names_dict(styler_multi: Styler) -> None:
    ...

def test_format_index_names_with_hidden_levels(styler_multi: Styler) -> None:
    ...
