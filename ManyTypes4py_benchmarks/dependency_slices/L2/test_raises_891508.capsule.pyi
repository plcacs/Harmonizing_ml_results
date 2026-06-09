from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: array, mean, sum

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Grouper
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning

# === Internal dependency: pandas.tests.groupby ===
def get_groupby_method_args(name, obj) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises