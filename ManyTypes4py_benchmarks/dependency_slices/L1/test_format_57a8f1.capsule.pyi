# === Third-party dependency: numpy ===
# Used symbols: arange, nan

# === Internal dependency: pandas ===
from pandas._config import option_context
from pandas.core.api import NA
from pandas.core.api import MultiIndex
from pandas.core.api import IndexSlice
from pandas.core.api import NaT
from pandas.core.api import Timestamp
from pandas.core.api import DataFrame

# === Internal dependency: pandas.io.formats.style ===
class Styler(StylerRenderer):
    def __init__(self, data, precision=..., table_styles=..., uuid=..., caption=..., table_attributes=..., cell_ids=..., na_rep=..., uuid_len=..., decimal=..., thousands=..., escape=..., formatter=...): ...

# === Internal dependency: pandas.io.formats.style_render ===
def _str_escape(x, escape): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises