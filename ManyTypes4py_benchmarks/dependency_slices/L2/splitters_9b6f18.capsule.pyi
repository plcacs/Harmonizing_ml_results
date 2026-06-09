from typing import Any

# === Internal dependency: fklearn.common_docstrings ===
splitter_return_docstring: str

# === Internal dependency: fklearn.types ===
LogType: Any
SplitterReturnType: Any

# === Third-party dependency: numpy ===
# Used symbols: array, ceil, max, min, split

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Index, Series, date_range, period_range

# === Third-party dependency: sklearn.model_selection ===
# Used symbols: GroupKFold, KFold, StratifiedKFold

# === Third-party dependency: sklearn.utils ===
# Used symbols: check_random_state

# === Third-party dependency: toolz ===
# Used symbols: curry

# === Third-party dependency: toolz.curried ===
# re-export: from toolz import curry
# re-export: from toolz import pipe
accumulate: curry
assoc: curry
filter: curry
map: curry
partial: curry