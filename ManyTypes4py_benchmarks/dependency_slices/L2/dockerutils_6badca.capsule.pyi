from typing import Any

# === Third-party dependency: packaging.version ===
class Version(_BaseVersion):
    def __init__(self, version: str) -> None: ...
    def local(self) -> str | None: ...
    def public(self) -> str: ...

# === Internal dependency: prefect ===
__version_info__: 'VersionInfo'
__version__: Any

# === Internal dependency: prefect.types._datetime ===
def now(tz: str | Any = ...) -> datetime.datetime: ...

# === Internal dependency: prefect.utilities.importtools ===
def lazy_import(name: str, error_on_import: bool = ..., help_message: Optional[str] = ...) -> ModuleType: ...

# === Internal dependency: prefect.utilities.slugify ===
# re-export: from slugify import slugify