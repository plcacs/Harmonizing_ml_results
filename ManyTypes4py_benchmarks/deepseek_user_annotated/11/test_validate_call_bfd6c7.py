import asyncio
import inspect
import re
import sys
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import pytest
from pydantic_core import ArgsKwargs
from typing_extensions import Annotated, Literal, Required, TypedDict, Unpack

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    PydanticInvalidForJsonSchema,
    PydanticUserError,
    Strict,
    TypeAdapter,
    ValidationError,
    validate_call,
    with_config,
)


def test_wrap() -> None:
    @validate_call
    def foo_bar(a: int, b: int) -> str:
        """This is the foo_bar method."""
        return f'{a}, {b}'

    assert foo_bar.__doc__ == 'This is the foo_bar method.'
    assert foo_bar.__name__ == 'foo_bar'
    assert foo_bar.__module__ == 'tests.test_validate_call'
    assert foo_bar.__qualname__ == 'test_wrap.<locals>.foo_bar'
    assert callable(foo_bar.raw_function)
    assert repr(inspect.signature(foo_bar)) == '<Signature (a: int, b: int)>'


def test_func_type() -> None:
    def f(x: int) -> None: ...

    class A:
        def m(self, x: int) -> None: ...

    for func in (f, lambda x: None, A.m, A().m):
        assert validate_call(func).__name__ == func.__name__
        assert validate_call(func).__qualname__ == func.__qualname__
        assert validate_call(partial(func)).__name__ == f'partial({func.__name__})'
        assert validate_call(partial(func)).__qualname__ == f'partial({func.__qualname__})'

    with pytest.raises(
        PydanticUserError,
        match=(f'Partial of `{list}` is invalid because the type of `{list}` is not supported by `validate_call`'),
    ):
        validate_call(partial(list))

    with pytest.raises(
        PydanticUserError,
        match=('`validate_call` should be applied to one of the following: function, method, partial, or lambda'),
    ):
        validate_call([])


def validate_bare_none() -> None:
    @validate_call
    def func(f: None) -> None:
        return f

    assert func(f=None) is None


def test_validate_class() -> None:
    class A:
        @validate_call
        def __new__(cls, x: int) -> 'A':
            return super().__new__(cls)

        @validate_call
        def __init__(self, x: int) -> None:
            self.x = x

    class M(type): ...

    for cls in (A, int, type, Exception, M):
        with pytest.raises(
            PydanticUserError,
            match=re.escape(
                '`validate_call` should be applied to functions, not classes (put `@validate_call` on top of `__init__` or `__new__` instead)'
            ),
        ):
            validate_call(cls)

    assert A('5').x == 5


def test_validate_custom_callable() -> None:
    class A:
        def __call__(self, x: int) -> int:
            return x

    with pytest.raises(
        PydanticUserError,
        match=re.escape(
            '`validate_call` should be applied to functions, not instances or other callables. Use `validate_call` explicitly on `__call__` instead.'
        ),
    ):
        validate_call(A())

    a = A()
    assert validate_call(a.__call__)('5') == 5  # Note: dunder methods cannot be overridden at instance level

    class B:
        @validate_call
        def __call__(self, x: int) -> int:
            return x

    assert B()('5') == 5


def test_invalid_signature() -> None:
    # Builtins functions not supported:
    with pytest.raises(PydanticUserError, match=(f'Input built-in function `{breakpoint}` is not supported')):
        validate_call(breakpoint)

    class A:
        def f() -> None: ...

    # A method require at least one positional arg (i.e. `self`), so the signature is invalid
    func = A().f
    with pytest.raises(PydanticUserError, match=(f"Input function `{func}` doesn't have a valid signature")):
        validate_call(func)


@pytest.mark.parametrize('decorator', [staticmethod, classmethod])
def test_classmethod_order_error(decorator: Callable) -> None:
    name = decorator.__name__
    with pytest.raises(
        PydanticUserError,
        match=re.escape(f'The `@{name}` decorator should be applied after `@validate_call` (put `@{name}` on top)'),
    ):

        class A:
            @validate_call
            @decorator
            def method(self, x: int) -> None:
                pass


def test_args() -> None:
    @validate_call
    def foo(a: int, b: int) -> str:
        return f'{a}, {b}'

    assert foo(1, 2) == '1, 2'
    assert foo(*[1, 2]) == '1, 2'
    assert foo(*(1, 2)) == '1, 2'
    assert foo(*[1], 2) == '1, 2'
    assert foo(a=1, b=2) == '1, 2'
    assert foo(1, b=2) == '1, 2'
    assert foo(b=2, a=1) == '1, 2'

    with pytest.raises(ValidationError) as exc_info:
        foo()
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'missing_argument', 'loc': ('a',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())},
        {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())},
    ]

    with pytest.raises(ValidationError) as exc_info:
        foo(1, 'x')
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'int_parsing',
            'loc': (1,),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'x',
        }
    ]

    with pytest.raises(ValidationError, match=r'2\s+Unexpected positional argument'):
        foo(1, 2, 3)

    with pytest.raises(ValidationError, match=r'apple\s+Unexpected keyword argument'):
        foo(1, 2, apple=3)

    with pytest.raises(ValidationError, match=r'a\s+Got multiple values for argument'):
        foo(1, 2, a=3)

    with pytest.raises(ValidationError) as exc_info:
        foo(1, 2, a=3, b=4)
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'multiple_argument_values', 'loc': ('a',), 'msg': 'Got multiple values for argument', 'input': 3},
        {'type': 'multiple_argument_values', 'loc': ('b',), 'msg': 'Got multiple values for argument', 'input': 4},
    ]


def test_optional() -> None:
    @validate_call
    def foo_bar(a: Optional[int] = None) -> str:
        return f'a={a}'

    assert foo_bar() == 'a=None'
    assert foo_bar(1) == 'a=1'
    with pytest.raises(ValidationError) as exc_info:
        foo_bar(None)

    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_type', 'loc': (0,), 'msg': 'Input should be a valid integer', 'input': None}
    ]


def test_kwargs() -> None:
    @validate_call
    def foo(*, a: int, b: int) -> int:
        return a + b

    assert foo(a=1, b=3) == 4

    with pytest.raises(ValidationError) as exc_info:
        foo(a=1, b='x')

    assert exc_info.value.errors(include_url=False) == [
        {
            'input': 'x',
            'loc': ('b',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'type': 'int_parsing',
        }
    ]

    with pytest.raises(ValidationError) as exc_info:
        foo(1, 'x')
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'missing_keyword_only_argument',
            'loc': ('a',),
            'msg': 'Missing required keyword only argument',
            'input': ArgsKwargs((1, 'x')),
        },
        {
            'type': 'missing_keyword_only_argument',
            'loc': ('b',),
            'msg': 'Missing required keyword only argument',
            'input': ArgsKwargs((1, 'x')),
        },
        {'type': 'unexpected_positional_argument', 'loc': (0,), 'msg': 'Unexpected positional argument', 'input': 1},
        {'type': 'unexpected_positional_argument', 'loc': (1,), 'msg': 'Unexpected positional argument', 'input': 'x'},
    ]


def test_untyped() -> None:
    @validate_call
    def foo(a: Any, b: Any, c: str = 'x', *, d: str = 'y') -> str:
        return ', '.join(str(arg) for arg in [a, b, c, d])

    assert foo(1, 2) == '1, 2, x, y'
    assert foo(1, {'x': 2}, c='3', d='4') == "1, {'x': 2}, 3, 4"


@pytest.mark.parametrize('validated', (True, False))
def test_var_args_kwargs(validated: bool) -> None:
    def foo(a: Any, b: Any, *args: Any, d: int = 3, **kwargs: Any) -> str:
        return f'a={a!r}, b={b!r}, args={args!r}, d={d!r}, kwargs={kwargs!r}'

    if validated:
        foo = validate_call(foo)

    assert foo(1, 2) == 'a=1, b=2, args=(), d=3, kwargs={}'
    assert foo(1, 2, 3, d=4) == 'a=1, b=2, args=(3,), d=4, kwargs={}'
    assert foo(*[1, 2, 3], d=4) == 'a=1, b=2, args=(3,), d=4, kwargs={}'
    assert foo(1, 2, args=(10, 11)) == "a=1, b=2, args=(), d=3, kwargs={'args': (10, 11)}"
    assert foo(1, 2, 3, args=(10, 11)) == "a=1, b=2, args=(3,), d=3, kwargs={'args': (10, 11)}"
    assert foo(1, 2, 3, e=10) == "a=1, b=2, args=(3,), d=3, kwargs={'e': 10}"
    assert foo(1, 2, kwargs=4) == "a=1, b=2, args=(), d=3, kwargs={'kwargs': 4}"
    assert foo(1, 2, kwargs=4, e=5) == "a=1, b=2, args=(), d=3, kwargs={'kwargs': 4, 'e': 5}"


def test_unpacked_typed_dict_kwargs_invalid_type() -> None:
    with pytest.raises(PydanticUserError) as exc:

        @validate_call
        def foo(**kwargs: Unpack[int]) -> None:
            pass

    assert exc.value.code == 'unpack-typed-dict'


def test_unpacked_typed_dict_kwargs_overlaps() -> None:
    class TD(TypedDict, total=False):
        a: int
        b: int
        c: int

    with pytest.raises(PydanticUserError) as exc:

        @validate_call
        def foo(a: int, b: int, **kwargs: Unpack[TD]) -> None:
            pass

    assert exc.value.code == 'overlapping-unpack-typed-dict'
    assert exc.value.message == "Typed dictionary 'TD' overlaps with parameters 'a', 'b'"

    # Works for a pos-only argument
    @validate_call
    def foo(a: int, /, **kwargs: Unpack[TD]) -> None:
        pass

    foo(1, a=1)


def test_unpacked_typed_dict_kwargs() -> None:
    @with_config({'strict': True})
    class TD(TypedDict, total=False):
        a: int
        b: Required[str]

    @validate_call
    def foo1(**kwargs: Unpack[TD]) -> None:
        pass

    @validate_call
    def foo2(**kwargs: 'Unpack[TD]') -> None:
        pass

    for foo in (foo1, foo2):
        foo(a=1, b='test')
        foo(b='test')

        with pytest.raises(ValidationError) as exc:
            foo(a='1')

        assert exc.value.errors()[0]['type'] == 'int_type'
        assert exc.value.errors()[0]['loc'] == ('a',)
        assert exc.value.errors()[1]['type'] == 'missing'
        assert exc.value.errors()[1]['loc'] == ('b',)

        # Make sure that when called without any arguments,
        # empty kwargs are still validated against the typed dict:
        with pytest.raises(ValidationError) as exc:
            foo()

        assert exc.value.errors()[0]['type'] == 'missing'
        assert exc.value.errors()[0]['loc'] == ('b',)


def test_unpacked_typed_dict_kwargs_functional_syntax() -> None:
    TD = TypedDict('TD', {'in': int, 'x-y': int})

    @validate_call
    def foo(**kwargs: Unpack[TD]) -> None:
        pass

    foo(**{'in': 1, 'x-y': 2})

    with pytest.raises(ValidationError) as exc:
        foo(**{'in': 'not_an_int', 'x-y': 1})

    assert exc.value.errors()[0]['type'] == 'int_parsing'
    assert exc.value.errors()[0]['loc'] == ('in',)


def test_field_can_provide_factory() -> None:
    @validate_call
    def foo(a: int, b: int = Field(default_factory=lambda: 99), *args: int) -> int:
        """mypy is happy with this"""
        return a + b + sum(args)

    assert foo(3) == 102
    assert foo(1, 2, 3) == 6


def test_annotated_field_can_provide_factory() -> None:
    @validate_call
    def foo2(a: int, b: 'Annotated[int, Field(default_factory=lambda: 99)]', *args: int) -> int:
        """mypy reports Incompatible default for argument "b" if we don't supply ANY as default"""
        return a + b + sum(args)

    assert foo2(1) == 100


def test_positional_only(create_module: Callable) -> None:
    module = create_module(
        # language=Python
        """
from pydantic import validate_call

@validate_call
def foo(a: Any, b: Any, /, c: Any = None) -> str:
    return f'{a}, {b}, {c}'
"""
    )
    assert module.foo(1, 2) == '1, 2, None'
    assert module.foo(1, 2, 44) == '1, 2, 44'
    assert module.foo(1, 2, c=44) == '1, 2, 44'
    with pytest.raises(ValidationError) as exc_info:
        module.foo(1, b=2)
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'missing_positional_only_argument',
            'loc': (1,),
            'msg': 'Missing required positional only argument',
            'input': ArgsKwargs((1,), {'b': 2}),
        },
        {'type': 'unexpected_keyword_argument', 'loc': ('b',), 'msg': 'Unexpected keyword argument', 'input': 2},
    ]

    with pytest.raises(ValidationError) as exc_info:
        module.foo(a=1, b=2)
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'missing_positional_only_argument',
            'loc': (0,),
            'msg': 'Missing required positional only argument',
            'input': ArgsKwargs((), {'a': 1, 'b': 2}),
        },
        {
            'type': 'missing_positional_only_argument',
            'loc': (1,),
            'msg': 'Missing required positional only argument',
            'input': ArgsKwargs((), {'a': 1, 'b': 2}),
