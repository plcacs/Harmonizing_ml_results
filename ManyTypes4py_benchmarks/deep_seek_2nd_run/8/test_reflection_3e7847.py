import sys
from copy import deepcopy
from datetime import time
from functools import partial, wraps
from inspect import Parameter, Signature, signature
from textwrap import dedent
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from unittest.mock import MagicMock, Mock, NonCallableMagicMock, NonCallableMock
import pytest
from pytest import raises
from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning
from hypothesis.internal import reflection
from hypothesis.internal.reflection import convert_keyword_arguments, convert_positional_arguments, define_function_signature, function_digest, get_pretty_function_description, get_signature, is_first_param_referenced_in_function, is_mock, proxies, repr_call, required_args, source_exec_as_module
from hypothesis.strategies._internal.lazy import LazyStrategy

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def do_conversion_test(f: Callable[..., T], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    result = f(*args, **kwargs)
    cargs, ckwargs = convert_keyword_arguments(f, args, kwargs)
    assert result == f(*cargs, **ckwargs)
    cargs2, ckwargs2 = convert_positional_arguments(f, args, kwargs)
    assert result == f(*cargs2, **ckwargs2)

def test_simple_conversion() -> None:
    def foo(a: int, b: int, c: int) -> Tuple[int, int, int]:
        return (a, b, c)
    assert convert_keyword_arguments(foo, (1, 2, 3), {}) == ((1, 2, 3), {})
    assert convert_keyword_arguments(foo, (), {'a': 3, 'b': 2, 'c': 1}) == ((3, 2, 1), {})
    do_conversion_test(foo, (1, 0), {'c': 2})
    do_conversion_test(foo, (1,), {'c': 2, 'b': 'foo'})

def test_leaves_unknown_kwargs_in_dict() -> None:
    def bar(x: int, **kwargs: Any) -> None:
        pass
    assert convert_keyword_arguments(bar, (1,), {'foo': 'hi'}) == ((1,), {'foo': 'hi'})
    assert convert_keyword_arguments(bar, (), {'x': 1, 'foo': 'hi'}) == ((1,), {'foo': 'hi'})
    do_conversion_test(bar, (1,), {})
    do_conversion_test(bar, (), {'x': 1, 'y': 1})

def test_errors_on_bad_kwargs() -> None:
    def bar() -> None:
        pass
    with raises(TypeError):
        convert_keyword_arguments(bar, (), {'foo': 1})

def test_passes_varargs_correctly() -> None:
    def foo(*args: Any) -> None:
        pass
    assert convert_keyword_arguments(foo, (1, 2, 3), {}) == ((1, 2, 3), {})
    do_conversion_test(foo, (1, 2, 3), {})

def test_errors_if_keyword_precedes_positional() -> None:
    def foo(x: int, y: int) -> None:
        pass
    with raises(TypeError):
        convert_keyword_arguments(foo, (1,), {'x': 2})

def test_errors_if_not_enough_args() -> None:
    def foo(a: int, b: int, c: int, d: int = 1) -> None:
        pass
    with raises(TypeError):
        convert_keyword_arguments(foo, (1, 2), {'d': 4})

def test_errors_on_extra_kwargs() -> None:
    def foo(a: int) -> None:
        pass
    with raises(TypeError, match='keyword'):
        convert_keyword_arguments(foo, (1,), {'b': 1})
    with raises(TypeError, match='keyword'):
        convert_keyword_arguments(foo, (1,), {'b': 1, 'c': 2})

def test_positional_errors_if_too_many_args() -> None:
    def foo(a: int) -> None:
        pass
    with raises(TypeError, match='too many positional arguments'):
        convert_positional_arguments(foo, (1, 2), {})

def test_positional_errors_if_too_few_args() -> None:
    def foo(a: int, b: int, c: int) -> None:
        pass
    with raises(TypeError):
        convert_positional_arguments(foo, (1, 2), {})

def test_positional_does_not_error_if_extra_args_are_kwargs() -> None:
    def foo(a: int, b: int, c: int) -> None:
        pass
    convert_positional_arguments(foo, (1, 2), {'c': 3})

def test_positional_errors_if_given_bad_kwargs() -> None:
    def foo(a: int) -> None:
        pass
    with raises(TypeError, match="missing a required argument: 'a'"):
        convert_positional_arguments(foo, (), {'b': 1})

def test_positional_errors_if_given_duplicate_kwargs() -> None:
    def foo(a: int) -> None:
        pass
    with raises(TypeError, match='multiple values'):
        convert_positional_arguments(foo, (2,), {'a': 1})

def test_names_of_functions_are_pretty() -> None:
    assert get_pretty_function_description(test_names_of_functions_are_pretty) == 'test_names_of_functions_are_pretty'

class Foo:
    @classmethod
    def bar(cls) -> None:
        pass

    def baz(cls) -> None:
        pass

    def __repr__(self) -> str:
        return 'SoNotFoo()'

def test_class_names_are_not_included_in_class_method_prettiness() -> None:
    assert get_pretty_function_description(Foo.bar) == 'bar'

def test_repr_is_included_in_bound_method_prettiness() -> None:
    assert get_pretty_function_description(Foo().baz) == 'SoNotFoo().baz'

def test_class_is_not_included_in_unbound_method() -> None:
    assert get_pretty_function_description(Foo.baz) == 'baz'

def test_does_not_error_on_confused_sources() -> None:
    def ed(f: Callable[..., Any], *args: Any) -> Callable[..., Any]:
        return f
    x = ed(lambda x, y: (x * y).conjugate() == x.conjugate() * y.conjugate(), complex, complex)
    get_pretty_function_description(x)

def test_digests_are_reasonably_unique() -> None:
    assert function_digest(test_simple_conversion) != function_digest(test_does_not_error_on_confused_sources)

def test_digest_returns_the_same_value_for_two_calls() -> None:
    assert function_digest(test_simple_conversion) == function_digest(test_simple_conversion)

def test_can_digest_a_built_in_function() -> None:
    import math
    assert function_digest(math.isnan) != function_digest(range)

def test_can_digest_a_unicode_lambda() -> None:
    function_digest(lambda x: '☃' in str(x))

def test_can_digest_a_function_with_no_name() -> None:
    def foo(x: int, y: int) -> None:
        pass
    function_digest(partial(foo, 1))

def test_arg_string_is_in_order() -> None:
    def foo(c: int, a: int, b: int, f: int, a1: int) -> None:
        pass
    assert repr_call(foo, (1, 2, 3, 4, 5), {}) == 'foo(c=1, a=2, b=3, f=4, a1=5)'
    assert repr_call(foo, (1, 2), {'b': 3, 'f': 4, 'a1': 5}) == 'foo(c=1, a=2, b=3, f=4, a1=5)'

def test_varkwargs_are_sorted_and_after_real_kwargs() -> None:
    def foo(d: int, e: int, f: int, **kwargs: Any) -> None:
        pass
    assert repr_call(foo, (), {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}) == 'foo(d=4, e=5, f=6, a=1, b=2, c=3)'

def test_varargs_come_without_equals() -> None:
    def foo(a: int, *args: Any) -> None:
        pass
    assert repr_call(foo, (1, 2, 3, 4), {}) == 'foo(2, 3, 4, a=1)'

def test_can_mix_varargs_and_varkwargs() -> None:
    def foo(*args: Any, **kwargs: Any) -> None:
        pass
    assert repr_call(foo, (1, 2, 3), {'c': 7}) == 'foo(1, 2, 3, c=7)'

def test_arg_string_does_not_include_unprovided_defaults() -> None:
    def foo(a: int, b: int, c: int = 9, d: int = 10) -> None:
        pass
    assert repr_call(foo, (1,), {'b': 1, 'd': 11}) == 'foo(a=1, b=1, d=11)'

def universal_acceptor(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    return (args, kwargs)

def has_one_arg(hello: Any) -> None:
    pass

def has_two_args(hello: Any, world: Any) -> None:
    pass

def has_a_default(x: Any, y: Any, z: Any = 1) -> None:
    pass

def has_varargs(*args: Any) -> None:
    pass

def has_kwargs(**kwargs: Any) -> None:
    pass

@pytest.mark.parametrize('f', [has_one_arg, has_two_args, has_varargs, has_kwargs])
def test_copying_preserves_signature(f: Callable[..., Any]) -> None:
    af = get_signature(f)
    t = define_function_signature('foo', 'docstring', af)(universal_acceptor)
    at = get_signature(t)
    assert af == at

def test_name_does_not_clash_with_function_names() -> None:
    def f() -> None:
        pass

    @define_function_signature('f', 'A docstring for f', signature(f))
    def g() -> None:
        pass
    g()

def test_copying_sets_name() -> None:
    f = define_function_signature('hello_world', 'A docstring for hello_world', signature(has_two_args))(universal_acceptor)
    assert f.__name__ == 'hello_world'

def test_copying_sets_docstring() -> None:
    f = define_function_signature('foo', 'A docstring for foo', signature(has_two_args))(universal_acceptor)
    assert f.__doc__ == 'A docstring for foo'

def test_uses_defaults() -> None:
    f = define_function_signature('foo', 'A docstring for foo', signature(has_a_default))(universal_acceptor)
    assert f(3, 2) == ((3, 2, 1), {})

def test_uses_varargs() -> None:
    f = define_function_signature('foo', 'A docstring for foo', signature(has_varargs))(universal_acceptor)
    assert f(1, 2) == ((1, 2), {})

DEFINE_FOO_FUNCTION: str = '\ndef foo(x):\n    return x\n'

def test_exec_as_module_execs() -> None:
    m = source_exec_as_module(DEFINE_FOO_FUNCTION)
    assert m.foo(1) == 1

def test_exec_as_module_caches() -> None:
    assert source_exec_as_module(DEFINE_FOO_FUNCTION) is source_exec_as_module(DEFINE_FOO_FUNCTION)

def test_exec_leaves_sys_path_unchanged() -> None:
    old_path = deepcopy(sys.path)
    source_exec_as_module('hello_world = 42')
    assert sys.path == old_path

def test_define_function_signature_works_with_conflicts() -> None:
    def accepts_everything(*args: Any, **kwargs: Any) -> None:
        pass
    define_function_signature('hello', 'A docstring for hello', Signature(parameters=[Parameter('f', Parameter.POSITIONAL_OR_KEYWORD)]))(accepts_everything)(1)
    define_function_signature('hello', 'A docstring for hello', Signature(parameters=[Parameter('f', Parameter.VAR_POSITIONAL)]))(accepts_everything)(1)
    define_function_signature('hello', 'A docstring for hello', Signature(parameters=[Parameter('f', Parameter.VAR_KEYWORD)]))(accepts_everything)()
    define_function_signature('hello', 'A docstring for hello', Signature(parameters=[Parameter('f', Parameter.POSITIONAL_OR_KEYWORD), Parameter('f_3', Parameter.POSITIONAL_OR_KEYWORD), Parameter('f_1', Parameter.VAR_POSITIONAL), Parameter('f_2', Parameter.VAR_KEYWORD)]))(accepts_everything)(1, 2)

def test_define_function_signature_validates_function_name() -> None:
    define_function_signature('hello_world', None, Signature())
    with raises(ValueError):
        define_function_signature('hello world', None, Signature())

class Container:
    def funcy(self) -> None:
        pass

def test_can_proxy_functions_with_mixed_args_and_varargs() -> None:
    def foo(a: int, *args: Any) -> Tuple[int, Tuple[Any, ...]]:
        return (a, args)

    @proxies(foo)
    def bar(*args: Any, **kwargs: Any) -> Tuple[int, Tuple[Any, ...]]:
        return foo(*args, **kwargs)
    assert bar(1, 2) == (1, (2,))

def test_can_delegate_to_a_function_with_no_positional_args() -> None:
    def foo(a: int, b: int) -> Tuple[int, int]:
        return (a, b)

    @proxies(foo)
    def bar(**kwargs: Any) -> Tuple[int, int]:
        return foo(**kwargs)
    assert bar(2, 1) == (2, 1)

@pytest.mark.parametrize('func,args,expected', [(lambda: None, (), None), (lambda a: a ** 2, (2,), 4), (lambda *a: a, [1, 2, 3], (1, 2, 3))])
def test_can_proxy_lambdas(func: Callable[..., Any], args: Tuple[Any, ...], expected: Any) -> None:
    @proxies(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    assert wrapped.__name__ == '<lambda>'
    assert wrapped(*args) == expected

class Snowman:
    def __repr__(self) -> str:
        return '☃'

class BittySnowman:
    def __repr__(self) -> str:
        return '☃'

def test_can_handle_unicode_repr() -> None:
    def foo(x: Any) -> None:
        pass
    assert repr_call(foo, [Snowman()], {}) == 'foo(x=☃)'
    assert repr_call(foo, [], {'x': Snowman()}) == 'foo(x=☃)'

class NoRepr:
    pass

def test_can_handle_repr_on_type() -> None:
    def foo(x: Any) -> None:
        pass
    assert repr_call(foo, [Snowman], {}) == 'foo(x=Snowman)'
    assert repr_call(foo, [NoRepr], {}) == 'foo(x=NoRepr)'

def test_can_handle_repr_of_none() -> None:
    def foo(x: Any) -> None:
        pass
    assert repr_call(foo, [None], {}) == 'foo(x=None)'
    assert repr_call(foo, [], {'x': None}) == 'foo(x=None)'

def test_kwargs_appear_in_arg_string() -> None:
    def varargs(*args: Any, **kwargs: Any) -> None:
        pass
    assert 'x=1' in repr_call(varargs, (), {'x': 1})

def test_is_mock_with_negative_cases() -> None:
    assert not is_mock(None)
    assert not is_mock(1234)
    assert not is_mock(is_mock)
    assert not is_mock(BittySnowman())
    assert not is_mock('foobar')
    assert not is_mock(Mock(spec=BittySnowman))
    assert not is_mock(MagicMock(spec=BittySnowman))

def test_is_mock_with_positive_cases() -> None:
    assert is_mock(Mock())
    assert is_mock(MagicMock())
    assert is_mock(NonCallableMock())
    assert is_mock(NonCallableMagicMock())

class Target:
    def __init__(self, a: Any, b: Any) -> None:
        pass

    def method(self, a: Any, b: Any) -> None:
        pass

@pytest.mark.parametrize('target', [Target, Target(1, 2).method])
@pytest.mark.param