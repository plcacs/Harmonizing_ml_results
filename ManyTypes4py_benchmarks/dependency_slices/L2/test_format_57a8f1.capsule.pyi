from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, nan

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import IndexSlice
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas.io.formats.style ===
class Styler(StylerRenderer):
    def __init__(self, data: DataFrame | Series, precision: int | None = ..., table_styles: CSSStyles | None = ..., uuid: str | None = ..., caption: str | tuple | list | None = ..., table_attributes: str | None = ..., cell_ids: bool = ..., na_rep: str | None = ..., uuid_len: int = ..., decimal: str | None = ..., thousands: str | None = ..., escape: str | None = ..., formatter: ExtFormatter | None = ...) -> None: ...

# === Internal dependency: pandas.io.formats.style_render ===
def _str_escape(x, escape) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises