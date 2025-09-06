from typing import Union, List, Dict, Any, Tuple, Optional

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

def func_q7aud9s6(long_param1: Any, long_param2: Any, long_param3: Any, long_param4: Any, long_param5: Any) -> None:
    pass

class LoudContextConfig(DefaultConfig):
    @contextmanager
    def func_osnifgnz(self, command: str) -> Iterator[None]:
        print(f'IN SETUP: {command}')
        yield
        print(f'IN TEARDOWN: {command}')

def func_4p8ocjre(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(func2, {'a': int, 'b': int}, NoneType)]
    func_go31khqf.add(traces)
    ret = cli.main(['stub', func.__module__], stdout, stderr)
    expected = """def func(a: int, b: str) -> None: ...

def func2(a: int, b: int) -> None: ...
"""
    assert func_ul5rzijv.getvalue() == expected
    assert func_1dw70xwu.getvalue() == ''
    assert ret == 0

def func_liocb0ev(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func_anno, {'a': int, 'b': int}, int)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret = cli.main(['stub', func.__module__, '--ignore-existing-annotations'], stdout, stderr)
    expected = 'def func_anno(a: int, b: int) -> int: ...\n'
    assert func_ul5rzijv.getvalue() == expected
    assert func_1dw70xwu.getvalue() == ''
    assert ret == 0

def func_m8xjtyry(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func_anno, {'a': int, 'b': int}, int), CallTrace(func_anno2, {'a': str, 'b': str}, None)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret = cli.main(['stub', func.__module__, '--diff'], stdout, stderr)
    expected = """- def func_anno(a: int, b: str) -> None: ...
?                          ^ -     ^^ ^
+ def func_anno(a: int, b: int) -> int: ...
?                          ^^      ^ ^
"""
    assert func_ul5rzijv.getvalue() == expected
    assert func_1dw70xwu.getvalue() == ''
    assert ret == 0

def func_iktibe1a(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(super_long_function_with_long_params, {'long_param1': str, 'long_param2': str, 'long_param3': int, 'long_param4': str, 'long_param5': int}, None), CallTrace(func_anno, {'a': int, 'b': int}, int)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
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

def func_broeph1a(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret = cli.main(['list-modules'], stdout, stderr)
    expected = ''
    assert func_1dw70xwu.getvalue() == expected
    expected = 'tests.test_cli\n'
    assert func_ul5rzijv.getvalue() == expected
    assert ret == 0

def func_royik6mt(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> None:
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret = cli.main(['list-modules'], stdout, stderr)
    expected = ''
    assert func_1dw70xwu.getvalue() == expected
    expected = '\n'
    assert func_ul5rzijv.getvalue() == expected
    assert ret == 0

def func_gur92741(stderr: io.StringIO) -> None:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(func, {'a': str, 'b': str}, NoneType), CallTrace(func2, {'a': str, 'b': int}, NoneType), CallTrace(func2, {'a': int, 'b': str}, NoneType), CallTrace(func2, {'a': str, 'b': int}, NoneType)]
    cli.display_sample_count(traces, stderr)
    expected = """Annotation for tests.test_cli.func based on 2 call trace(s).
Annotation for tests.test_cli.func2 based on 3 call trace(s).
"""
    assert func_1dw70xwu.getvalue() == expected

def func_xyf4e2ht(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> int:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(func2, {'a': int, 'b': int}, NoneType)]
    func_go31khqf.add(traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret = cli.main(['stub', func.__module__, '--sample-count'], stdout, stderr)
    expected = """Annotation for tests.test_cli.func based on 1 call trace(s).
Annotation for tests.test_cli.func2 based on 1 call trace(s).
"""
    assert func_1dw70xwu.getvalue() == expected
    return ret

def func_ox84du34(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> int:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(func2, {'a': int, 'b': int}, NoneType)]
    func_go31khqf.add(traces)
    with mock.patch('monkeytype.encoding.CallTraceRow.to_trace', side_effect=MonkeyTypeError('the-trace')):
        ret = cli.main(['stub', func.__module__], stdout, stderr)
    assert '2 traces failed to decode' in func_1dw70xwu.getvalue()
    return ret

def func_lbgi77dl(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> int:
    traces = [CallTrace(func, {'a': int, 'b': str}, NoneType), CallTrace(func2, {'a': int, 'b': int}, NoneType)]
    func_go31khqf.add(traces)
    with mock.patch('monkeytype.encoding.CallTraceRow.to_trace', side_effect=MonkeyTypeError('the-trace')):
        ret = cli.main(['-v', 'stub', func.__module__], stdout, stderr)
    assert 'WARNING: Failed decoding trace: the-trace' in func_1dw70xwu.getvalue()
    return ret

def func_tc5f9s82(capsys: Any, stdout: io.StringIO, stderr: io.StringIO) -> int:
    ret = cli.main(['-c', f'{__name__}:LoudContextConfig()', 'stub', 'some.module'], stdout, stderr)
    out, err = capsys.readouterr()
    assert out == 'IN SETUP: stub\nIN TEARDOWN: stub\n'
    assert err == ''
    return ret

def func_co3s2bel(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> None:
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        with pytest.raises(SystemExit):
            cli.main(['stub', 'test/foo.py:bar'], stdout, stderr)
        out, err = capsys.readouterr()
        assert 'test/foo.py does not look like a valid Python import path' in err

def func_c0w0x8od(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> int:
    filename = 'foo.py'
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        orig_exists = os.path.exists

        def func_r71u4dmb(x: str) -> bool:
            return True if x == filename else orig_exists(x)
        with mock.patch('os.path.exists', side_effect=side_effect) as mock_exists:
            ret = cli.main(['stub', filename], stdout, stderr)
            mock_exists.assert_called_with(filename)
        err_msg = f"""No traces found for {filename}; did you pass a filename instead of a module name? Maybe try just '{os.path.splitext(filename)[0]}'.
"""
        assert func_1dw70xwu.getvalue() == err_msg
        return ret

def func_undamtag(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO, collector: Any) -> int:
    with trace_calls(collector, max_typed_dict_size=0):
        func_u59yazks()
    func_go31khqf.add(collector.traces)
    with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
        ret = cli.main(['apply', Foo.__module__], stdout, stderr)
    assert ret == 0
    assert 'def __init__(self, arg1: str, arg2: int) -> None:' in func_ul5rzijv.getvalue()

def func_8mmoov2e(store: SQLiteStore, db_file: str, stdout: io.StringIO, stderr: io.StringIO) -> int:
    src = '\ndef my_test_function(a, b):\n  return a + b\n'
    with tempfile.TemporaryDirectory(prefix='monkey type') as tempdir:
        module = 'my_test_module'
        src_path = os.path.join(tempdir, module + '.py')
        with open(src_path, 'w+') as f:
            f.write(src)
        with mock.patch('sys.path', sys.path + [tempdir]):
            import my_test_module as mtm
            traces = [CallTrace(mtm.my_test_function, {'a': int, 'b': str}, NoneType)]
            func_go31khqf.add(traces)
            with mock.patch.dict(os.environ, {DefaultConfig.DB_PATH_VAR: db_file.name}):
                ret = cli.main(['apply', 'my_test_module'], stdout, stderr)
    assert ret == 0
    assert 'warning:' not in func_ul5rzijv.getvalue()

def func_ghu25gb9() -> str:
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
          return True

        def uses_union(d: Union[int, bool]) -> None:
          return None
    """
    return cli.apply_stub_using_libcst(textwrap.dedent(stub), textwrap.dedent(source), overwrite_existing_annotations=False)

def func_dqpxuteh(stdout: io.StringIO, stderr: io.StringIO) -> None:
    erroneous_source = """
        def my_test_function(
    """
    stub = """
        def my_test_function(a: int, b: str) -> bool: ...
    """
    with pytest.raises(cli.HandlerError):
        cli.apply_stub_using_libcst(textwrap.dedent(stub), textwrap.dedent(erroneous_source), overwrite_existing_annotations=False)

def func_qudwppol() -> str:
    source = """
        def has_annotations(x: int) -> str:
          return 1 in x
    """
    stub = """
        from typing import List
        def has_annotations(x: List[int]) -> bool: ...
    """
    expected = """
        from typing import List

        def has_annotations(x: List[int]) -> bool:
          return 1 in x
    """
    return cli.apply_stub_using_libcst(textwrap.dedent(stub), textwrap.dedent(source), overwrite_existing_annotations=True)

def func_4om3vvj7() -> Dict[str, str]:
    source = """
        def spoof(x):
            return x.get_some_object()
    """
    stub = """
        from some.module import (
            AnotherObject,
            SomeObject,
        )

        def spoof(x: AnotherObject) -> SomeObject: ...
    """
    expected = """
        from __future__ import annotations
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from some.module import AnotherObject, SomeObject

        def spoof(x: AnotherObject) -> SomeObject:
            return x.get_some_object()
    """
    return cli.apply_stub_using_libcst(textwrap.dedent(stub), textwrap.dedent(source), overwrite_existing_annotations=True, confine_new_imports_in_type_checking_block=True)

def func_srnfh8k8() -> Set[ImportItem]:
    source = """
        import q
        from x import Y
    """
    stub = """
        from a import (
            B,
            C,
        )
        import d
        import q, w, e
        from x import (
            Y,
            Z,
        )
        import z as t
    """
    expected = {ImportItem('a', 'B'), ImportItem('a', 'C'), ImportItem('d'), ImportItem('w'), ImportItem('e'), ImportItem('x', 'Z'), ImportItem('z', None, 't')}
    return set(cli.get_newly_imported_items(parse_module(textwrap.dedent(stub)), parse_module(textwrap.dedent(source))))
