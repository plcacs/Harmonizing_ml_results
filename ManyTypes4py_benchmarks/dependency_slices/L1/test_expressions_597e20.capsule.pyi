# === Third-party dependency: numpy ===
# Used symbols: abs, arange, bool_, empty, random, shape, where

# === Internal dependency: pandas ===
from pandas._config import option_context

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.api ===
from pandas.core.frame import DataFrame

# === Internal dependency: pandas.core.computation.expressions ===
def set_numexpr_threads(n=...): ...
def _can_use_numexpr(op, op_str, a, b, dtype_check): ...
def evaluate(op, a, b, use_numexpr=...): ...
def where(cond, a, b, use_numexpr=...): ...
def set_test_mode(v=...): ...
def get_test_result(): ...
from pandas.core.computation.check import NUMEXPR_INSTALLED
USE_NUMEXPR = NUMEXPR_INSTALLED

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises