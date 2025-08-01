#!/usr/bin/env python3
from contextlib import contextmanager
import io
import os
import os.path
import pytest
import sqlite3
import sys
import tempfile
import textwrap
from typing import Any, Iterator, Tuple, List
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


def func(a: object, b: object) -> None:
    pass


def func2(a: object, b: object) -> None:
    pass


def func_anno(a: object, b: object) -> None:
    pass


def func_anno2(a: object, b: object) -> None:
    pass


def super_long_function_with_long_params(
    long_param1: object,
    long_param2: object,
    long_param3: object,
    long_param4: object,
    long_param5: object
) -> None:
    pass


class LoudContextConfig(DefaultConfig):

    @contextmanager
    def cli_context(self, command: str) -> Iterator[None]:
        print(f'IN SETUP: {command}')
        yield
        print(f'IN TEARDOWN: {command}')


@pytest.fixture
def store_data() -> Iterator[Tuple[SQLiteStore, tempfile._TemporaryFileWrapper]]:
    with tempfile.NamedTemporaryFile(prefix='monkeytype_tests') as db_file:
        conn = sqlite3.connect(db_file.name)
        create_call_trace_table(conn)
        with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
            yield (SQLiteStore(conn), db_file)


@pytest.fixture
def store(store_data: Tuple[SQLiteStore, tempfile._TemporaryFileWrapper]) -> Iterator[SQLiteStore]:
    store_obj, _ = store_data
    yield store_obj


@pytest.fixture
def db_file(store_data: Tuple[SQLiteStore, tempfile._TemporaryFileWrapper]) -> Iterator[tempfile._TemporaryFileWrapper]:
    _, db_file_obj = store_data
    yield db_file_obj


@pytest.fixture
def stdout() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def stderr() -> io.StringIO:
    return io.StringIO()


def test_generate_stub(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [
        CallTrace(func, {'a': int, 'b': str}, NoneType),
        CallTrace(func2, {'a': int, 'b': int}, NoneType)
    ]
    store.add(traces)
    ret: int = cli.main(['stub', func.__module__], stdout, stderr)
    expected: str = 'def func(a: int, b: str) -> None: ...\n\n\ndef func2(a: int, b: int) -> None: ...\n'
    assert stdout.getvalue() == expected
    assert stderr.getvalue() == ''
    assert ret == 0


def test_print_stub_ignore_existing_annotations(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [CallTrace(func_anno, {'a': int, 'b': int}, int)]
    store.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret: int = cli.main(['stub', func.__module__, '--ignore-existing-annotations'], stdout, stderr)
    expected: str = 'def func_anno(a: int, b: int) -> int: ...\n'
    assert stdout.getvalue() == expected
    assert stderr.getvalue() == ''
    assert ret == 0


def test_get_diff(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [
        CallTrace(func_anno, {'a': int, 'b': int}, int),
        CallTrace(func_anno2, {'a': str, 'b': str}, None)
    ]
    store.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret: int = cli.main(['stub', func.__module__, '--diff'], stdout, stderr)
    expected: str = ('- def func_anno(a: int, b: str) -> None: ...\n'
                     '?                          ^ -     ^^ ^\n'
                     '+ def func_anno(a: int, b: int) -> int: ...\n'
                     '?                          ^^      ^ ^\n')
    assert stdout.getvalue() == expected
    assert stderr.getvalue() == ''
    assert ret == 0


def test_get_diff2(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [
        CallTrace(super_long_function_with_long_params, {'long_param1': str, 'long_param2': str, 'long_param3': int, 'long_param4': str, 'long_param5': int}, None),
        CallTrace(func_anno, {'a': int, 'b': int}, int)
    ]
    store.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret: int = cli.main(['stub', func.__module__, '--diff'], stdout, stderr)
    expected: str = ('- def func_anno(a: int, b: str) -> None: ...\n'
                     '?                          ^ -     ^^ ^\n'
                     '+ def func_anno(a: int, b: int) -> int: ...\n'
                     '?                          ^^      ^ ^\n'
                     '\n'
                     '\n'
                     '  def super_long_function_with_long_params(\n'
                     '      long_param1: str,\n'
                     '      long_param2: str,\n'
                     '-     long_param3: str,\n'
                     '?                  ^ -\n'
                     '+     long_param3: int,\n'
                     '?                  ^^\n'
                     '      long_param4: str,\n'
                     '-     long_param5: str\n'
                     '?                  ^ -\n'
                     '+     long_param5: int\n'
                     '?                  ^^\n'
                     '  ) -> None: ...\n')
    assert stdout.getvalue() == expected
    assert stderr.getvalue() == ''
    assert ret == 0


@pytest.mark.parametrize('arg, error', [
    (func.__module__, f'No traces found for module {func.__module__}\n'),
    (func.__module__ + ':foo', f'No traces found for specifier {func.__module__}:foo\n')
])
def test_no_traces(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO, arg: str, error: str) -> None:
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret: int = cli.main(['stub', arg], stdout, stderr)
    assert stderr.getvalue() == error
    assert stdout.getvalue() == ''
    assert ret == 0


def test_display_list_of_modules(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [CallTrace(func, {'a': int, 'b': str}, NoneType)]
    store.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret: int = cli.main(['list-modules'], stdout, stderr)
    expected: str = ''
    assert stderr.getvalue() == expected
    expected = 'tests.test_cli\n'
    assert stdout.getvalue() == expected
    assert ret == 0


def test_display_list_of_modules_no_modules(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret: int = cli.main(['list-modules'], stdout, stderr)
    expected: str = ''
    assert stderr.getvalue() == expected
    expected = '\n'
    assert stdout.getvalue() == expected
    assert ret == 0


def test_display_sample_count(stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [
        CallTrace(func, {'a': int, 'b': str}, NoneType),
        CallTrace(func, {'a': str, 'b': str}, NoneType),
        CallTrace(func2, {'a': str, 'b': int}, NoneType),
        CallTrace(func2, {'a': int, 'b': str}, NoneType),
        CallTrace(func2, {'a': str, 'b': int}, NoneType)
    ]
    cli.display_sample_count(traces, stderr)
    expected: str = ('Annotation for tests.test_cli.func based on 2 call trace(s).\n'
                     'Annotation for tests.test_cli.func2 based on 3 call trace(s).\n')
    assert stderr.getvalue() == expected


def test_display_sample_count_from_cli(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [
        CallTrace(func, {'a': int, 'b': str}, NoneType),
        CallTrace(func2, {'a': int, 'b': int}, NoneType)
    ]
    store.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret: int = cli.main(['stub', func.__module__, '--sample-count'], stdout, stderr)
    expected: str = ('Annotation for tests.test_cli.func based on 1 call trace(s).\n'
                     'Annotation for tests.test_cli.func2 based on 1 call trace(s).\n')
    assert stderr.getvalue() == expected
    assert ret == 0


def test_quiet_failed_traces(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [
        CallTrace(func, {'a': int, 'b': str}, NoneType),
        CallTrace(func2, {'a': int, 'b': int}, NoneType)
    ]
    store.add(traces)
    with mock.patch('monkeytype.encoding.CallTraceRow.to_trace', side_effect=MonkeyTypeError('the-trace')):
        ret: int = cli.main(['stub', func.__module__], stdout, stderr)
    assert '2 traces failed to decode' in stderr.getvalue()
    assert ret == 0


def test_verbose_failed_traces(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces: List[CallTrace] = [
        CallTrace(func, {'a': int, 'b': str}, NoneType),
        CallTrace(func2, {'a': int, 'b': int}, NoneType)
    ]
    store.add(traces)
    with mock.patch('monkeytype.encoding.CallTraceRow.to_trace', side_effect=MonkeyTypeError('the-trace')):
        ret: int = cli.main(['-v', 'stub', func.__module__], stdout, stderr)
    assert 'WARNING: Failed decoding trace: the-trace' in stderr.getvalue()
    assert ret == 0


def test_cli_context_manager_activated(capsys: Any, stdout: io.StringIO, stderr: io.StringIO) -> None:
    ret: int = cli.main(['-c', f'{__name__}:LoudContextConfig()', 'stub', 'some.module'], stdout, stderr)
    out, err = capsys.readouterr()
    assert out == 'IN SETUP: stub\nIN TEARDOWN: stub\n'
    assert err == ''
    assert ret == 0


def test_pathlike_parameter(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, capsys: Any, stdout: io.StringIO, stderr: io.StringIO) -> None:
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        with pytest.raises(SystemExit):
            cli.main(['stub', 'test/foo.py:bar'], stdout, stderr)
        out, err = capsys.readouterr()
        assert 'test/foo.py does not look like a valid Python import path' in err


def test_toplevel_filename_parameter(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    filename: str = 'foo.py'
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        orig_exists = os.path.exists

        def side_effect(x: str) -> bool:
            return True if x == filename else orig_exists(x)
        with mock.patch('os.path.exists', side_effect=side_effect) as mock_exists:
            ret: int = cli.main(['stub', filename], stdout, stderr)
            mock_exists.assert_called_with(filename)
        err_msg: str = (f"No traces found for {filename}; did you pass a filename instead of a module name? "
                        f"Maybe try just '{os.path.splitext(filename)[0]}'.\n")
        assert stderr.getvalue() == err_msg
        assert stdout.getvalue() == ''
        assert ret == 0


@pytest.mark.usefixtures('collector')
def test_apply_stub_init(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO, collector: Any) -> None:
    """Regression test for applying stubs to testmodule/__init__.py style module layout"""
    with trace_calls(collector, max_typed_dict_size=0):
        func_foo()
    store.add(collector.traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret: int = cli.main(['apply', Foo.__module__], stdout, stderr)
    assert ret == 0
    assert 'def __init__(self, arg1: str, arg2: int) -> None:' in stdout.getvalue()


def test_apply_stub_file_with_spaces(store: SQLiteStore, db_file: tempfile._TemporaryFileWrapper, stdout: io.StringIO, stderr: io.StringIO) -> None:
    """Regression test for applying a stub to a filename containing spaces"""
    src: str = '\ndef my_test_function(a, b):\n  return a + b\n'
    with tempfile.TemporaryDirectory(prefix='monkey type') as tempdir:
        module: str = 'my_test_module'
        src_path: str = os.path.join(tempdir, module + '.py')
        with open(src_path, 'w+') as f:
            f.write(src)
        with mock.patch('sys.path', sys.path + [tempdir]):
            import my_test_module as mtm  # type: ignore
            traces: List[CallTrace] = [CallTrace(mtm.my_test_function, {'a': int, 'b': str}, NoneType)]
            store.add(traces)
            with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
                ret: int = cli.main(['apply', 'my_test_module'], stdout, stderr)
    assert ret == 0
    assert 'warning:' not in stdout.getvalue()


def test_apply_stub_using_libcst() -> None:
    source: str = (
        '\n        def my_test_function(a, b):\n          return True\n\n'
        '        def has_return_type(a, b) -> bool:\n          return True\n\n'
        '        def uses_forward_ref(d):\n          return None\n\n'
        '        def no_stub(a):\n          return True\n\n'
        '        def uses_union(d):\n          return None\n'
    )
    stub: str = (
        "\n        from mypy_extensions import TypedDict\n        from typing import Union\n        def my_test_function(a: int, b: str) -> bool: ...\n\n"
        "        def has_return_type(a: int, b: int) -> bool: ...\n\n"
        "        def uses_forward_ref(d: 'Foo') -> None: ...\n\n"
        "        def uses_union(d: Union[int, bool]) -> None: ...\n\n"
        "        class Foo: ...\n\n"
        "        class Movie(TypedDict):\n          name: str\n          year: int\n"
    )
    expected: str = (
        "\n        from mypy_extensions import TypedDict\n        from typing import Union\n\n"
        "        class Foo: ...\n\n"
        "        class Movie(TypedDict):\n          name: str\n          year: int\n\n"
        "        def my_test_function(a: int, b: str) -> bool:\n          return True\n\n"
        "        def has_return_type(a: int, b: int) -> bool:\n          return True\n\n"
        "        def uses_forward_ref(d: 'Foo') -> None:\n          return None\n\n"
        "        def no_stub(a):\n          return True\n\n"
        "        def uses_union(d: Union[int, bool]) -> None:\n          return None\n"
    )
    result: str = cli.apply_stub_using_libcst(textwrap.dedent(stub), textwrap.dedent(source), overwrite_existing_annotations=False)
    assert result == textwrap.dedent(expected)


def test_apply_stub_using_libcst__exception(stdout: io.StringIO, stderr: io.StringIO) -> None:
    erroneous_source: str = '\n        def my_test_function(\n    '
    stub: str = '\n        def my_test_function(a: int, b: str) -> bool: ...\n    '
    with pytest.raises(cli.HandlerError):
        cli.apply_stub_using_libcst(textwrap.dedent(stub), textwrap.dedent(erroneous_source), overwrite_existing_annotations=False)


def test_apply_stub_using_libcst__overwrite_existing_annotations() -> None:
    source: str = '\n        def has_annotations(x: int) -> str:\n          return 1 in x\n    '
    stub: str = '\n        from typing import List\n        def has_annotations(x: List[int]) -> bool: ...\n    '
    expected: str = '\n        from typing import List\n\n        def has_annotations(x: List[int]) -> bool:\n          return 1 in x\n    '
    result: str = cli.apply_stub_using_libcst(textwrap.dedent(stub), textwrap.dedent(source), overwrite_existing_annotations=True)
    assert result == textwrap.dedent(expected)


def test_apply_stub_using_libcst__confine_new_imports_in_type_checking_block() -> None:
    source: str = '\n        def spoof(x):\n            return x.get_some_object()\n    '
    stub: str = (
        '\n        from some.module import (\n            AnotherObject,\n            SomeObject,\n        )\n\n'
        '        def spoof(x: AnotherObject) -> SomeObject: ...\n    '
    )
    expected: str = (
        '\n        from __future__ import annotations\n        from typing import TYPE_CHECKING\n\n'
        '        if TYPE_CHECKING:\n            from some.module import AnotherObject, SomeObject\n\n'
        '        def spoof(x: AnotherObject) -> SomeObject:\n            return x.get_some_object()\n    '
    )
    result: str = cli.apply_stub_using_libcst(
        textwrap.dedent(stub),
        textwrap.dedent(source),
        overwrite_existing_annotations=True,
        confine_new_imports_in_type_checking_block=True
    )
    assert result == textwrap.dedent(expected)


def test_get_newly_imported_items() -> None:
    source: str = '\n        import q\n        from x import Y\n    '
    stub: str = (
        '\n        from a import (\n            B,\n            C,\n        )\n        import d\n        import q, w, e\n        from x import (\n            Y,\n            Z,\n        )\n        import z as t\n    '
    )
    expected: set[ImportItem] = {
        ImportItem('a', 'B'),
        ImportItem('a', 'C'),
        ImportItem('d'),
        ImportItem('w'),
        ImportItem('e'),
        ImportItem('x', 'Z'),
        ImportItem('z', None, 't')
    }
    result_set: set[ImportItem] = set(cli.get_newly_imported_items(parse_module(textwrap.dedent(stub)), parse_module(textwrap.dedent(source))))
    assert expected == result_set
