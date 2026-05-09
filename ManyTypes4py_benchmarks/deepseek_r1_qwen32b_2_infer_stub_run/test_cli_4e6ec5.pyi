from contextlib import contextmanager
from io import StringIO
from os import PathLike
from sqlite3 import Connection
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from unittest.mock import Mock

from libcst import Module
from monkeytype.cli import (
    CallTrace,
    CliContext,
    CliError,
    Config,
    DisplayOptions,
    MonkeyTypeError,
    StubWriter,
)
from monkeytype.db.sqlite import SQLiteStore
from monkeytype.typing import NoneType

class LoudContextConfig(Config):
    @contextmanager
    def cli_context(self, command: str) -> Iterator[None]:
        ...

def test_apply_stub_init() -> None:
    ...

def test_apply_stub_file_with_spaces() -> None:
    ...

def test_apply_stub_using_libcst(
    stub: str,
    source: str,
    overwrite_existing_annotations: bool = ...,
    confine_new_imports_in_type_checking_block: bool = ...,
) -> str:
    ...

def test_apply_stub_using_libcst__exception() -> None:
    ...

def test_apply_stub_using_libcst__overwrite_existing_annotations() -> None:
    ...

def test_apply_stub_using_libcst__confine_new_imports_in_type_checking_block() -> None:
    ...

def test_get_newly_imported_items() -> None:
    ...

def func_foo() -> None:
    ...

def func(a: int, b: str) -> None:
    ...

def func2(a: int, b: int) -> None:
    ...

def func_anno(a: int, b: int) -> int:
    ...

def func_anno2(a: str, b: str) -> None:
    ...

def super_long_function_with_long_params(
    long_param1: str,
    long_param2: str,
    long_param3: int,
    long_param4: str,
    long_param5: int,
) -> None:
    ...

def test_generate_stub() -> None:
    ...

def test_print_stub_ignore_existing_annotations() -> None:
    ...

def test_get_diff() -> None:
    ...

def test_get_diff2() -> None:
    ...

def test_no_traces() -> None:
    ...

def test_display_list_of_modules() -> None:
    ...

def test_display_list_of_modules_no_modules() -> None:
    ...

def test_display_sample_count() -> None:
    ...

def test_display_sample_count_from_cli() -> None:
    ...

def test_quiet_failed_traces() -> None:
    ...

def test_verbose_failed_traces() -> None:
    ...

def test_cli_context_manager_activated() -> None:
    ...

def test_pathlike_parameter() -> None:
    ...

def test_toplevel_filename_parameter() -> None:
    ...