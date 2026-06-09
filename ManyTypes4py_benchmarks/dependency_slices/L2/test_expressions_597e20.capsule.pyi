from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: abs, arange, bool_, empty, random, shape, where

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.api ===
# re-export: from pandas.core.frame import DataFrame

# === Internal dependency: pandas.core.computation.check ===
NUMEXPR_INSTALLED: Any

# === Internal dependency: pandas.core.computation.expressions ===
def set_numexpr_threads(n = ...) -> None: ...
def _can_use_numexpr(op, op_str, a, b, dtype_check) -> bool: ...
def evaluate(op, a, b, use_numexpr: bool = ...) -> Any: ...
def where(cond, a, b, use_numexpr: bool = ...) -> Any: ...
def set_test_mode(v: bool = ...) -> None: ...
def get_test_result() -> list[bool]: ...
# re-export: from pandas.core.computation.check import NUMEXPR_INSTALLED
USE_NUMEXPR = NUMEXPR_INSTALLED

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises