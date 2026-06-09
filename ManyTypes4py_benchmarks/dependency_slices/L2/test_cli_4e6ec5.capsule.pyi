# === Third-party dependency: libcst ===
# Used symbols: parse_module

# === Third-party dependency: libcst.codemod.visitors ===
# Used symbols: ImportItem

# === Internal dependency: monkeytype.cli ===
def display_sample_count(traces: List[CallTrace], stderr: IO[str]) -> None: ...
class HandlerError(Exception): ...
def get_newly_imported_items(stub_module: Module, source_module: Module) -> List[ImportItem]: ...
def apply_stub_using_libcst(stub: str, source: str, overwrite_existing_annotations: bool, confine_new_imports_in_type_checking_block: bool = ...) -> str: ...
def main(argv: List[str], stdout: IO[str], stderr: IO[str]) -> int: ...

# === Internal dependency: monkeytype.config ===
class DefaultConfig(Config):
    ...

# === Internal dependency: monkeytype.db.sqlite ===
def create_call_trace_table(conn: sqlite3.Connection, table: str = ...) -> None: ...
class SQLiteStore(CallTraceStore): ...

# === Internal dependency: monkeytype.exceptions ===
class MonkeyTypeError(Exception): ...

# === Internal dependency: monkeytype.tracing ===
class CallTrace: ...

# === Internal dependency: monkeytype.typing ===
NoneType: type

# === Unresolved dependency: my_test_module ===
# Used unresolved symbols: my_test_function

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.test_tracing ===
# re-export: from monkeytype.tracing import trace_calls

# === Internal dependency: tests.testmodule ===
class Foo: ...