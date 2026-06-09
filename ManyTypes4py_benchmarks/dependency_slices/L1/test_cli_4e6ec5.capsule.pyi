# === Third-party dependency: libcst ===
# Used symbols: parse_module

# === Third-party dependency: libcst.codemod.visitors ===
# Used symbols: ImportItem

# === Internal dependency: monkeytype.cli ===
def display_sample_count(traces, stderr): ...
class HandlerError(Exception): ...
def get_newly_imported_items(stub_module, source_module): ...
def apply_stub_using_libcst(stub, source, overwrite_existing_annotations, confine_new_imports_in_type_checking_block=...): ...
def main(argv, stdout, stderr): ...

# === Internal dependency: monkeytype.config ===
class DefaultConfig(Config):
    ...

# === Internal dependency: monkeytype.db.sqlite ===
def create_call_trace_table(conn, table=...): ...
class SQLiteStore(CallTraceStore): ...

# === Internal dependency: monkeytype.exceptions ===
class MonkeyTypeError(Exception): ...

# === Internal dependency: monkeytype.tracing ===
class CallTrace: ...

# === Internal dependency: monkeytype.typing ===
NoneType = Ellipsis

# === Unresolved dependency: my_test_module ===
# Used unresolved symbols: my_test_function

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.test_tracing ===
from monkeytype.tracing import trace_calls

# === Internal dependency: tests.testmodule ===
class Foo: ...