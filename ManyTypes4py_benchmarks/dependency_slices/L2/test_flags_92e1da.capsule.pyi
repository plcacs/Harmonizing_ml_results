from typing import Any

# === Third-party dependency: click ===
# Used symbols: Context

# === Third-party dependency: dbt.cli.exceptions ===
class DbtUsageException(Exception): ...

# === Third-party dependency: dbt.cli.flags ===
class Flags:
    def __init__(self, ctx: Optional[Context] = ..., project_flags: Optional[ProjectFlags] = ...) -> None: ...

# === Third-party dependency: dbt.cli.main ===
def cli(ctx, **kwargs) -> Any: ...

# === Third-party dependency: dbt.cli.types ===
class Command(Enum): ...

# === Third-party dependency: dbt.contracts.project ===
class ProjectFlags(ExtensibleDbtClassMixin):
    ...

# === Third-party dependency: dbt.tests.util ===
def rm_file(*paths) -> None: ...
def write_file(contents, *paths) -> Any: ...

# === Third-party dependency: dbt_common.exceptions ===
# Used symbols: DbtInternalError

# === Third-party dependency: dbt_common.helper_types ===
class WarnErrorOptions(IncludeExclude): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises