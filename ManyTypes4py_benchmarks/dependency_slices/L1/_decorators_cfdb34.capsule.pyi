from typing import Any

# === Internal dependency: pandas._typing ===
T = TypeVar(...)
FuncType = Callable[..., Any]
F = TypeVar(...)

# === Internal dependency: pandas.util._exceptions ===
def find_stack_level(): ...