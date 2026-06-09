from typing import Any

# === Third-party dependency: dbt.exceptions ===
class ParsingError(DbtRuntimeError): ...

# === Third-party dependency: dbt.tests.util ===
def run_dbt(args: Optional[List[str]] = ..., expect_pass: bool = ...) -> Any: ...
def write_file(contents, *paths) -> Any: ...

# === Third-party dependency: dbt_common.exceptions ===
# Used symbols: CompilationError

# === Third-party dependency: pytest ===
# Used symbols: fixture, raises

# === Internal dependency: tests.functional.adapter.hooks.fixtures ===
macros__before_and_after: str
models__hooks: str
models__hooks_configured: str
models__hooks_error: str
models__hooks_kwargs: str
models__hooked: str
models__post: str
models__pre: str
snapshots__test_snapshot: str
properties__seed_models: str
properties__test_snapshot_models: str
properties__model_hooks: str
properties__model_hooks_list: str
seeds__example_seed_csv: str