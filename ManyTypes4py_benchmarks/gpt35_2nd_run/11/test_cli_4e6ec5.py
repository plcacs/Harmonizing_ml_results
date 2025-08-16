from contextlib import contextmanager
import io
import os
import os.path
import sqlite3
import sys
import tempfile
from typing import Iterator
from unittest import mock
from libcst import parse_module
from libcst.codemod.visitors import ImportItem
from monkeytype import cli
from monkeytype.config import DefaultConfig
from monkeytype.db.sqlite import create_call_trace_table, SQLiteStore
from monkeytype.exceptions import MonkeyTypeError
from monkeytype.tracing import CallTrace
from monkeytype.typing import NoneType
from .testmodule import Foo
from .test_tracing import trace_calls

def func_foo() -> None:
    Foo(arg1='string', arg2=1)

def func(a: int, b: str) -> None:
    pass

def func2(a: int, b: int) -> None:
    pass

def func_anno(a: int, b: int) -> int:
    pass

def func_anno2(a: str, b: str) -> None:
    pass

def super_long_function_with_long_params(long_param1: str, long_param2: str, long_param3: int, long_param4: str, long_param5: int) -> None:
    pass

class LoudContextConfig(DefaultConfig):

    @contextmanager
    def cli_context(self, command: str) -> Iterator[None]:
        print(f'IN SETUP: {command}')
        yield
        print(f'IN TEARDOWN: {command}')

def test_generate_stub(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(func2, {'a': int, 'b': int}, NoneType)]
    store.add(traces)
    ret = cli.main(['stub', func.__module__], stdout, stderr)
    expected = 'def func(a: int, b: str) -> None: ...\n\n\ndef func2(a: int, b: int) -> None: ...\n'
    assert stdout.getvalue() == expected
    assert stderr.getvalue() == ''
    assert ret == 0

# Add type annotations to the remaining test functions
