# === Third-party dependency: packaging.version ===
class Version(_BaseVersion):
    def __init__(self, version: str) -> None: ...
    def local(self) -> str | None: ...
    def public(self) -> str: ...

# === Internal dependency: prefect ===
__version__ = _build_info.__version__
__version_info__ = cast(...)

# === Internal dependency: prefect.types._datetime ===
def now(tz=...): ...

# === Internal dependency: prefect.utilities.importtools ===
def lazy_import(name, error_on_import=..., help_message=...): ...

# === Internal dependency: prefect.utilities.slugify ===
from slugify import slugify