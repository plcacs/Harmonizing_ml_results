from typing import Any

# === Internal dependency: fklearn.types ===
LogType: Any
ValidatorReturnType: Any

# === Third-party dependency: joblib ===
# Used symbols: Parallel, delayed

# === Third-party dependency: toolz ===
# Used symbols: compose, curry

# === Third-party dependency: toolz.curried ===
# re-export: from toolz import curry
# re-export: from toolz import first
# re-export: from toolz import pipe
assoc: curry
dissoc: curry
map: curry
partial: curry

# === Third-party dependency: toolz.functoolz ===
def identity(x) -> Any: ...

# === Third-party dependency: tqdm ===
# Used symbols: tqdm