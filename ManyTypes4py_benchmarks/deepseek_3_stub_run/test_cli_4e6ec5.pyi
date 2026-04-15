from contextlib import contextmanager
from typing import Iterator
from unittest import mock
import io
import os
import os.path
import pytest
import sqlite3
import sys
import tempfile
import textwrap
from libcst import parse_module
from libcst.codemod.visitors import ImportItem
from monkeytype import cli
from monkeytype.config import DefaultConfig
from monkeytype.db.sqlite import SQLiteStore
from monkeytype.exceptions import MonkeyTypeError
from monkeytype.tracing import CallTrace
from monkeytype.typing import NoneType
from .testmodule import Foo
from .test_tracing import trace_calls

def func_foo() -> None: ...

def func(a: int, b: str) -> None: ...

def func2(a: int, b: int) -> None: ...

def func_anno(a: int, b: str) -> None: ...

def func_anno2(a: int, b: str) -> None: ...

def super_long_function_with_long_params(
    long_param1: str,
    long_param2: str,
    long_param3: str,
    long_param4: str,
    long_param5: str
) -> None: ...

class LoudContextConfig(DefaultConfig):
    @contextmanager
    def cli_context(self, command: str) -> Iterator[None]: ...

@pytest.fixture
def store_data() -> Iterator[tuple[SQLiteStore, tempfile._TemporaryFileWrapper]]: ...

@pytest.fixture
def store(store_data: tuple[SQLiteStore, tempfile._TemporaryFileWrapper]) -> Iterator[SQLiteStore]: ...

@pytest.fixture
def db_file(store_data: tuple[SQLiteStore, tempfile._TemporaryFileWrapper]) -> Iterator[tempfile._TemporaryFileWrapper]: ...

@pytest.fixture
def stdout() -> io.StringIO: ...

@pytest.fixture
def stderr() -> io.StringIO: ...

def test_generate_stub(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_print_stub_ignore_existing_annotations(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_get_diff(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_get_diff2(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

@pytest.mark.parametrize('arg, error', [(func.__module__, f'No traces found for module {func.__module__}\n'), (func.__module__ + ':foo', f'No traces found for specifier {func.__module__}:foo\n')])
def test_no_traces(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO,
    arg: str,
    error: str
) -> None: ...

def test_display_list_of_modules(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_display_list_of_modules_no_modules(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_display_sample_count(stderr: io.StringIO) -> None: ...

def test_display_sample_count_from_cli(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_quiet_failed_traces(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_verbose_failed_traces(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_cli_context_manager_activated(
    capsys: pytest.CaptureFixture[str],
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_pathlike_parameter(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    capsys: pytest.CaptureFixture[str],
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_toplevel_filename_parameter(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

@pytest.mark.usefixtures('collector')
def test_apply_stub_init(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO,
    collector: object
) -> None: ...

def test_apply_stub_file_with_spaces(
    store: SQLiteStore,
    db_file: tempfile._TemporaryFileWrapper,
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_apply_stub_using_libcst() -> None: ...

def test_apply_stub_using_libcst__exception(
    stdout: io.StringIO,
    stderr: io.StringIO
) -> None: ...

def test_apply_stub_using_libcst__overwrite_existing_annotations() -> None: ...

def test_apply_stub_using_libcst__confine_new_imports_in_type_checking_block() -> None: ...

def test_get_newly_imported_items() -> set[ImportItem]: ...