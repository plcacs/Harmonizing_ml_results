import asyncio
import inspect
import re
import sys
from datetime import datetime, timezone
from functools import partial
from typing import Annotated, Any, Literal, Union
import pytest
from pydantic_core import ArgsKwargs
from typing_extensions import Required, TypedDict, Unpack
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, PydanticInvalidForJsonSchema, PydanticUserError, Strict, TypeAdapter, ValidationError, validate_call, with_config

def test_wrap() -> None:

    @validate_call
    def foo_bar(a: int, b: int) -> str:
        """This is the foo_bar method."""
        return f'{a}, {b}'

def test_func_type() -> None:

    def f(x: Any) -> None:
        ...

    class A:

        def m(self, x: Any) -> None:
            ...
    for func in (f, lambda x: None, A.m, A().m):
        assert validate_call(func).__name__ == func.__name__
        assert validate_call(func).__qualname__ == func.__qualname__
        assert validate_call(partial(func)).__name__ == f'partial({func.__name__})'
        assert validate_call(partial(func)).__qualname__ == f'partial({func.__qualname__})'

def validate_bare_none() -> None:

    @validate_call
    def func(f: Any) -> Any:
        return f
    assert func(f=None) is None

def test_validate_class() -> None:

    class A:

        @validate_call
        def __new__(cls, x: int) -> Any:
            return super().__new__(cls)

        @validate_call
        def __init__(self, x: int) -> None:
            self.x = x

    class M(type):
        ...
    for cls in (A, int, type, Exception, M):
        with pytest.raises(PydanticUserError, match=re.escape('`validate_call` should be applied to functions, not classes (put `@validate_call` on top of `__init__` or `__new__` instead)')):
            validate_call(cls)
    assert A('5').x == 5

def test_validate_custom_callable() -> None:

    class A:

        def __call__(self, x: str) -> int:
            return int(x)
    with pytest.raises(PydanticUserError, match=re.escape('`validate_call` should be applied to functions, not instances or other callables. Use `validate_call` explicitly on `__call__` instead.')):
        validate_call(A())
    a = A()
    assert validate_call(a.__call__)('5') == 5

    class B:

        @validate_call
        def __call__(self, x: str) -> int:
            return int(x)
    assert B()('5') == 5

def test_invalid_signature() -> None:
    with pytest.raises(PydanticUserError, match=f'Input built-in function `{breakpoint}` is not supported'):
        validate_call(breakpoint)

    class A:

        def f() -> None:
            ...
    func = A().f
    with pytest.raises(PydanticUserError, match=f"Input function `{func}` doesn't have a valid signature"):
        validate_call(func)

@pytest.mark.parametrize('decorator', [staticmethod, classmethod])
def test_classmethod_order_error(decorator: Any) -> None:
    name = decorator.__name__
    with pytest.raises(PydanticUserError, match=re.escape(f'The `@{name}` decorator should be applied after `@validate_call` (put `@{name}` on top)')):

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
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('a',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())}, {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())}]
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 'x')
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': (1,), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}]
    with pytest.raises(ValidationError, match='2\\s+Unexpected positional argument'):
        foo(1, 2, 3)
    with pytest.raises(ValidationError, match='apple\\s+Unexpected keyword argument'):
        foo(1, 2, apple=3)
    with pytest.raises(ValidationError, match='a\\s+Got multiple values for argument'):
        foo(1, 2, a=3, b=4)
