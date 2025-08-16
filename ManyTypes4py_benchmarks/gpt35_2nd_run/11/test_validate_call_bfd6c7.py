from typing import Annotated, Any, Literal, Union
import pytest
from pydantic import BaseModel, Field, validate_call

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

def test_validate_custom_callable() -> None:

    class A:

        def __call__(self, x: str) -> int:
            return int(x)
    with pytest.raises(PydanticUserError):
        validate_call(A())
    a = A()
    assert validate_call(a.__call__)('5') == 5

    class B:

        @validate_call
        def __call__(self, x: str) -> int:
            return int(x)
    assert B()('5') == 5

def test_invalid_signature() -> None:
    with pytest.raises(PydanticUserError):
        validate_call(breakpoint)

    class A:

        def f(self) -> None:
            ...
    func = A().f
    with pytest.raises(PydanticUserError):
        validate_call(func)

@pytest.mark.parametrize('decorator', [staticmethod, classmethod])
def test_classmethod_order_error(decorator: Any) -> None:
    name = decorator.__name__
    with pytest.raises(PydanticUserError):
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

def test_optional() -> None:

    @validate_call
    def foo_bar(a: int = None) -> str:
        return f'a={a}'
    assert foo_bar() == 'a=None'
    assert foo_bar(1) == 'a=1'
    with pytest.raises(ValidationError):
        foo_bar(None)

def test_kwargs() -> None:

    @validate_call
    def foo(*, a: int, b: int) -> int:
        return a + b
    assert foo(a=1, b=3) == 4
    with pytest.raises(ValidationError):
        foo(a=1, b='x')
    with pytest.raises(ValidationError):
        foo(1, 'x')

def test_untyped() -> None:

    @validate_call
    def foo(a, b, c='x', *, d='y') -> str:
        return ', '.join((str(arg) for arg in [a, b, c, d]))
    assert foo(1, 2) == '1, 2, x, y'
    assert foo(1, {'x': 2}, c='3', d='4') == "1, {'x': 2}, 3, 4"

@pytest.mark.parametrize('validated', (True, False))
def test_var_args_kwargs(validated: bool) -> None:

    def foo(a, b, *args, d=3, **kwargs) -> str:
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
    with pytest.raises(PydanticUserError):
        @validate_call
        def foo(**kwargs) -> None:
            pass

def test_unpacked_typed_dict_kwargs_overlaps() -> None:

    class TD(TypedDict, total=False):
        pass
    with pytest.raises(PydanticUserError):
        @validate_call
        def foo(a, b, **kwargs) -> None:
            pass

    @validate_call
    def foo(a, /, **kwargs) -> None:
        pass
    foo(1, a=1)

def test_unpacked_typed_dict_kwargs() -> None:

    @validate_call
    def foo1(**kwargs) -> None:
        pass

    @validate_call
    def foo2(**kwargs) -> None:
        pass
    for foo in (foo1, foo2):
        foo(a=1, b='test')
        foo(b='test')
        with pytest.raises(ValidationError):
            foo(a='1')
        with pytest.raises(ValidationError):
            foo()
