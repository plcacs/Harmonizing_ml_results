from contextlib import contextmanager
import io
import os
import os.path
import pytest
import sqlite3
import sys
import tempfile
import textwrap
from typing import Iterator, List, Set, Tuple, Dict, Any, Optional, Union
from unittest import mock
from libcst import Module, parse_module
from libcst.codemod.visitors import ImportItem
from monkeytype import cli
from monkeytype.config import DefaultConfig
from monkeytype.db.sqlite import create_call_trace_table, SQLiteStore
from monkeytype.exceptions import MonkeyTypeError
from monkeytype.tracing import CallTrace
from monkeytype.typing import NoneType
from .testmodule import Foo
from .test_tracing import trace_calls


def func_u59yazks() -> None:
    Foo(arg1='string', arg2=1)


def func_tt1d5pq4(a: Any, b: Any) -> None:
    pass


def func_26nyhb3n(a: Any, b: Any) -> None:
    pass


def func_ob89myrh(a: Any, b: Any) -> None:
    pass


def func_jhkns3fp(a: Any, b: Any) -> None:
    pass


def func_q7aud9s6(long_param1: Any, long_param2: Any, long_param3: Any, long_param4: Any,
    long_param5: Any) -> None:
    pass


class LoudContextConfig(DefaultConfig):

    @contextmanager
    def func_osnifgnz(self, command: str) -> Iterator[None]:
        print(f'IN SETUP: {command}')
        yield
        print(f'IN TEARDOWN: {command}')


@pytest.fixture
def func_qtq5jgjs() -> Iterator[Tuple[SQLiteStore, tempfile.NamedTemporaryFile]]:
    with tempfile.NamedTemporaryFile(prefix='monkeytype_tests') as db_file:
        conn = sqlite3.connect(db_file.name)
        create_call_trace_table(conn)
        with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR:
            db_file.name}):
            yield SQLiteStore(conn), db_file


@pytest.fixture
def func_go31khqf(store_data: Tuple[SQLiteStore, Any]) -> Iterator[SQLiteStore]:
    store, __ = store_data
    yield store


@pytest.fixture
def func_85g0xh8m(store_data: Tuple[Any, tempfile.NamedTemporaryFile]) -> Iterator[tempfile.NamedTemporaryFile]:
    __, db_file = store_data
    yield db_file


@pytest.fixture
def func_ul5rzijv() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def func_1dw70xwu() -> io.StringIO:
    return io.StringIO()


def func_4p8ocjre(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(
        func2, {'a': int, 'b': int}, NoneType)]
    func_go31khqf.add(traces)
    ret = cli.main(['stub', func.__module__], stdout, stderr)
    expected = """def func(a: int, b: str) -> None: ...


def func2(a: int, b: int) -> None: ...
"""
    assert func_ul5rzijv.getvalue() == expected
    assert func_1dw70xwu.getvalue() == ''
    assert ret == 0


def func_liocb0ev(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func_anno, {'a': int, 'b': int}, int)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        ret = cli.main(['stub', func.__module__,
            '--ignore-existing-annotations'], stdout, stderr)
    expected = 'def func_anno(a: int, b: int) -> int: ...\n'
    assert func_ul5rzijv.getvalue() == expected
    assert func_1dw70xwu.getvalue() == ''
    assert ret == 0


def func_m8xjtyry(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func_anno, {'a': int, 'b': int}, int), CallTrace(
        func_anno2, {'a': str, 'b': str}, None)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        ret = cli.main(['stub', func.__module__, '--diff'], stdout, stderr)
    expected = """- def func_anno(a: int, b: str) -> None: ...
?                          ^ -     ^^ ^
+ def func_anno(a: int, b: int) -> int: ...
?                          ^^      ^ ^
"""
    assert func_ul5rzijv.getvalue() == expected
    assert func_1dw70xwu.getvalue() == ''
    assert ret == 0


def func_iktibe1a(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(super_long_function_with_long_params, {
        'long_param1': str, 'long_param2': str, 'long_param3': int,
        'long_param4': str, 'long_param5': int}, None), CallTrace(func_anno,
        {'a': int, 'b': int}, int)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        ret = cli.main(['stub', func.__module__, '--diff'], stdout, stderr)
    expected = """- def func_anno(a: int, b: str) -> None: ...
?                          ^ -     ^^ ^
+ def func_anno(a: int, b: int) -> int: ...
?                          ^^      ^ ^


  def super_long_function_with_long_params(
      long_param1: str,
      long_param2: str,
-     long_param3: str,
?                  ^ -
+     long_param3: int,
?                  ^^
      long_param4: str,
-     long_param5: str
?                  ^ -
+     long_param5: int
?                  ^^
  ) -> None: ...
"""
    assert func_ul5rzijv.getvalue() == expected
    assert func_1dw70xwu.getvalue() == ''
    assert ret == 0


@pytest.mark.parametrize('arg, error', [(func.__module__,
    f"""No traces found for module {func.__module__}
"""), (func.__module__ +
    ':foo', f'No traces found for specifier {func.__module__}:foo\n')])
def func_58fwhbh2(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO, arg: str, error: str) -> None:
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        ret = cli.main(['stub', arg], stdout, stderr)
    assert func_1dw70xwu.getvalue() == error
    assert func_ul5rzijv.getvalue() == ''
    assert ret == 0


def func_broeph1a(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        ret = cli.main(['list-modules'], stdout, stderr)
    expected = ''
    assert func_1dw70xwu.getvalue() == expected
    expected = 'tests.test_cli\n'
    assert func_ul5rzijv.getvalue() == expected
    assert ret == 0


def func_royik6mt(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        ret = cli.main(['list-modules'], stdout, stderr)
    expected = ''
    assert func_1dw70xwu.getvalue() == expected
    expected = '\n'
    assert func_ul5rzijv.getvalue() == expected
    assert ret == 0


def func_gur92741(stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(
        func, {'a': str, 'b': str}, NoneType), CallTrace(func2, {'a': str,
        'b': int}, NoneType), CallTrace(func2, {'a': int, 'b': str},
        NoneType), CallTrace(func2, {'a': str, 'b': int}, NoneType)]
    cli.display_sample_count(traces, stderr)
    expected = """Annotation for tests.test_cli.func based on 2 call trace(s).
Annotation for tests.test_cli.func2 based on 3 call trace(s).
"""
    assert func_1dw70xwu.getvalue() == expected


def func_xyf4e2ht(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(
        func2, {'a': int, 'b': int}, NoneType)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        ret = cli.main(['stub', func.__module__, '--sample-count'], stdout,
            stderr)
    expected = """Annotation for tests.test_cli.func based on 1 call trace(s).
Annotation for tests.test_cli.func2 based on 1 call trace(s).
"""
    assert func_1dw70xwu.getvalue() == expected
    assert ret == 0


def func_ox84du34(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(
        func2, {'a': int, 'b': int}, NoneType)]
    func_go31khqf.add(traces)
    with mock.patch('monkeytype.encoding.CallTraceRow.to_trace',
        side_effect=MonkeyTypeError('the-trace')):
        ret = cli.main(['stub', func.__module__], stdout, stderr)
    assert '2 traces failed to decode' in func_1dw70xwu.getvalue()
    assert ret == 0


def func_lbgi77dl(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(
        func2, {'a': int, 'b': int}, NoneType)]
    func_go31khqf.add(traces)
    with mock.patch('monkeytype.encoding.CallTraceRow.to_trace',
        side_effect=MonkeyTypeError('the-trace')):
        ret = cli.main(['-v', 'stub', func.__module__], stdout, stderr)
    assert 'WARNING: Failed decoding trace: the-trace' in func_1dw70xwu.getvalue(
        )
    assert ret == 0


def func_tc5f9s82(capsys: Any, stdout: io.StringIO, stderr: io.StringIO) -> None:
    ret = cli.main(['-c', f'{__name__}:LoudContextConfig()', 'stub',
        'some.module'], stdout, stderr)
    out, err = capsys.readouterr()
    assert out == 'IN SETUP: stub\nIN TEARDOWN: stub\n'
    assert err == ''
    assert ret == 0


def func_co3s2bel(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, capsys: Any, stdout: io.StringIO, stderr: io.StringIO) -> None:
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        with pytest.raises(SystemExit):
            cli.main(['stub', 'test/foo.py:bar'], stdout, stderr)
        out, err = capsys.readouterr()
        assert 'test/foo.py does not look like a valid Python import path' in err


def func_c0w0x8od(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    filename = 'foo.py'
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        orig_exists = os.path.exists

        def func_r71u4dmb(x: str) -> bool:
            return True if x == filename else orig_exists(x)
        with mock.patch('os.path.exists', side_effect=func_r71u4dmb
            ) as mock_exists:
            ret = cli.main(['stub', filename], stdout, stderr)
            mock_exists.assert_called_with(filename)
        err_msg = f"""No traces found for {filename}; did you pass a filename instead of a module name? Maybe try just '{os.path.splitext(filename)[0]}'.
"""
        assert func_1dw70xwu.getvalue() == err_msg
        assert func_ul5rzijv.getvalue() == ''
        assert ret == 0


@pytest.mark.usefixtures('collector')
def func_undamtag(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO, collector: Any) -> None:
    """Regression test for applying stubs to testmodule/__init__.py style module layout"""
    with trace_calls(collector, max_typed_dict_size=0):
        func_u59yazks()
    func_go31khqf.add(collector.traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}
        ):
        ret = cli.main(['apply', Foo.__module__], stdout, stderr)
    assert ret == 0
    assert 'def __init__(self, arg1: str, arg2: int) -> None:' in func_ul5rzijv.getvalue(
        )


def func_8mmoov2e(store: SQLiteStore, db_file: tempfile.NamedTemporaryFile, stdout: io.StringIO, stderr: io.StringIO) -> None:
    """Regression test for applying a stub to a filename containing spaces"""
    src = '\ndef my_test_function(a, b):\n  return a + b\n'
    with tempfile.TemporaryDirectory(prefix='monkey type') as tempdir:
        module = 'my_test_module'
        src_path = os.path.join(tempdir, module + '.py')
        with open(src_path, 'w+') as f:
            f.write(src)
        with mock.patch('sys.path', sys.path + [tempdir]):
            import my_test_module as mtm
            traces = [CallTrace(mtm.my_test_function, {'a': int, 'b': str},
                NoneType)]
            func_go31khqf.add(traces)
            with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR:
                db_file.name}):
                ret = cli.main(['apply', 'my_test_module'], stdout, stderr)
    assert ret == 0
    assert 'warning:' not in func_ul5rzijv.getvalue()


def func_ghu25gb9() -> None:
    source = """
        def my_test_function(a, b):
          return True

        def has_return_type(a, b) -> bool:
          return True

        def uses_forward_ref(d):
          return None

        def no_stub(a):
          return True

        def uses_union(d):
          return None
    """
    stub = """
        from mypy_extensions import TypedDict
        from typing import Union
        def my_test_function(a: int, b: str) -> bool: ...

        def has_return_type(a: int, b: int) -> bool: ...

        def uses_forward_ref(d: 'Foo') -> None: ...

        def uses_union(d: Union[int, bool]) -> None: ...

        class Foo: ...

        class Movie(TypedDict):
          name: str
          year: int
    """
    expected = """
        from mypy_extensions import TypedDict
        from typing import Union

        class Foo: ...

        class Movie(TypedDict):
          name: str
          year: int

        def my_test_function(a: int, b: str) -> bool:
          return True

        def has_return_type(a: int, b: int) -> bool:
          return True

        def uses_forward_ref(d: 'Foo') -> None:
          return None

        def no_stub(a):
