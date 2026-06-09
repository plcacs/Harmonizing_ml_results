from typing import Any

# === Internal dependency: chalice.compat ===
def pip_import_string() -> Any: ...

# === Internal dependency: chalice.constants ===
MISSING_DEPENDENCIES_TEMPLATE: str

# === Internal dependency: chalice.utils ===
class OSUtils(object):
    ...