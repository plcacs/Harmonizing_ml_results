import asyncio
import inspect
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Set, Tuple, Type
import pytest
from dirty_equals import IsInstance
from pydantic import BaseModel, Field, PydanticDeprecatedSince20, ValidationError
from pydantic.deprecated.decorator import ValidatedFunction
from pydantic.deprecated.decorator import validate_arguments as validate_arguments_deprecated
from pydantic.errors import PydanticUserError

def validate_arguments(*args: Any, **kwargs: Any) -> Any:
    with pytest.warns(PydanticDeprecatedSince20, match='^The `validate_arguments` method is deprecated; use `validate_call`'):
        return validate_arguments_deprecated(*args, **kwargs)

def test_args() -> None:

    @validate_arguments
    def foo(a: int, b: int) -> str:
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert foo(*[1, 2]) == '1, 2'
    assert foo(*(1, 2)) == '1, 2'
    assert foo(*[1], 2) == '1, 2'
    with pytest.raises(ValidationError) as exc_info:
        foo()  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {'input': {}, 'loc': ('a',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {}, 'loc': ('b',), 'msg': 'Field required', 'type': 'missing'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 'x')  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'x', 'loc': ('b',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}
    ]
    with pytest.raises(TypeError, match='2 positional arguments expected but 3 given'):
        foo(1, 2, 3)  # type: ignore
    with pytest.raises(TypeError, match="unexpected keyword argument: 'apple'"):
        foo(1, 2, apple=3)  # type: ignore
    with pytest.raises(TypeError, match="multiple values for argument: 'a'"):
        foo(1, 2, a=3)  # type: ignore
    with pytest.raises(TypeError, match="multiple values for arguments: 'a', 'b'"):
        foo(1, 2, a=3, b=4)  # type: ignore

def test_wrap() -> None:

    @validate_arguments
    def foo_bar(a: int, b: int) -> str:
        """This is the foo_bar method."""
        return f'{a}, {b}'
    assert foo_bar.__doc__ == 'This is the foo_bar method.'
    assert foo_bar.__name__ == 'foo_bar'
    assert foo_bar.__module__ == 'tests.test_deprecated_validate_arguments'
    assert foo_bar.__qualname__ == 'test_wrap.<locals>.foo_bar'
    assert isinstance(foo_bar.vd, ValidatedFunction)
    assert callable(foo_bar.raw_function)
    assert foo_bar.vd.arg_mapping == {0: 'a', 1: 'b'}
    assert foo_bar.vd.positional_only_args == set()  # type: Set[Any]
    assert issubclass(foo_bar.model, BaseModel)
    assert foo_bar.model.model_fields.keys() == {'a', 'b', 'args', 'kwargs', 'v__duplicate_kwargs'}
    assert foo_bar.model.__name__ == 'FooBar'
    assert foo_bar.model.model_json_schema()['title'] == 'FooBar'
    assert repr(inspect.signature(foo_bar)) == '<Signature (a: int, b: int)>'

def test_kwargs() -> None:

    @validate_arguments
    def foo(*, a: int, b: int) -> int:
        return a + b
    assert foo.model.model_fields.keys() == {'a', 'b', 'args', 'kwargs'}
    assert foo(a=1, b=3) == 4
    with pytest.raises(ValidationError) as exc_info:
        foo(a=1, b='x')  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'x', 'loc': ('b',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}
    ]
    with pytest.raises(TypeError, match='0 positional arguments expected but 2 given'):
        foo(1, 'x')  # type: ignore

def test_untyped() -> None:

    @validate_arguments
    def foo(a: Any, b: Any, c: str = 'x', *, d: str = 'y') -> str:
        return ', '.join((str(arg) for arg in [a, b, c, d]))
    assert foo(1, 2) == '1, 2, x, y'
    assert foo(1, {'x': 2}, c='3', d='4') == "1, {'x': 2}, 3, 4"

@pytest.mark.parametrize('validated', (True, False))
def test_var_args_kwargs(validated: bool) -> None:

    def foo(a: int, b: int, *args: Any, d: int = 3, **kwargs: Any) -> str:
        return f'a={a!r}, b={b!r}, args={args!r}, d={d!r}, kwargs={kwargs!r}'
    if validated:
        foo = validate_arguments(foo)
    assert foo(1, 2) == 'a=1, b=2, args=(), d=3, kwargs={}'
    assert foo(1, 2, 3, d=4) == 'a=1, b=2, args=(3,), d=4, kwargs={}'
    assert foo(*[1, 2, 3], d=4) == 'a=1, b=2, args=(3,), d=4, kwargs={}'
    assert foo(1, 2, args=(10, 11)) == "a=1, b=2, args=(), d=3, kwargs={'args': (10, 11)}"
    assert foo(1, 2, 3, args=(10, 11)) == "a=1, b=2, args=(3,), d=3, kwargs={'args': (10, 11)}"
    assert foo(1, 2, 3, e=10) == "a=1, b=2, args=(3,), d=3, kwargs={'e': 10}"
    assert foo(1, 2, kwargs=4) == "a=1, b=2, args=(), d=3, kwargs={'kwargs': 4}"
    assert foo(1, 2, kwargs=4, e=5) == "a=1, b=2, args=(), d=3, kwargs={'kwargs': 4, 'e': 5}"

def test_field_can_provide_factory() -> None:

    @validate_arguments
    def foo(a: int, b: int = Field(default_factory=lambda: 99), *args: int) -> int:
        """mypy is happy with this"""
        return a + b + sum(args)
    assert foo(3) == 102
    assert foo(1, 2, 3) == 6

def test_positional_only(create_module: Callable[[str], Any]) -> None:
    with pytest.warns(PydanticDeprecatedSince20):
        module = create_module(
            "\nfrom pydantic.deprecated.decorator import validate_arguments\n\n@validate_arguments\ndef foo(a, b, /, c=None):\n    return f'{a}, {b}, {c}'\n"
        )
    assert module.foo(1, 2) == '1, 2, None'
    assert module.foo(1, 2, 44) == '1, 2, 44'
    assert module.foo(1, 2, c=44) == '1, 2, 44'
    with pytest.raises(TypeError, match="positional-only argument passed as keyword argument: 'b'"):
        module.foo(1, b=2)
    with pytest.raises(TypeError, match="positional-only arguments passed as keyword arguments: 'a', 'b'"):
        module.foo(a=1, b=2)

def test_args_name() -> None:

    @validate_arguments
    def foo(args: Any, kwargs: Any) -> str:
        return f'args={args!r}, kwargs={kwargs!r}'
    assert foo.model.model_fields.keys() == {'args', 'kwargs', 'v__args', 'v__kwargs', 'v__duplicate_kwargs'}
    assert foo(1, 2) == 'args=1, kwargs=2'
    with pytest.raises(TypeError, match="unexpected keyword argument: 'apple'"):
        foo(1, 2, apple=4)  # type: ignore
    with pytest.raises(TypeError, match="unexpected keyword arguments: 'apple', 'banana'"):
        foo(1, 2, apple=4, banana=5)  # type: ignore
    with pytest.raises(TypeError, match='2 positional arguments expected but 3 given'):
        foo(1, 2, 3)  # type: ignore

def test_v_args() -> None:
    with pytest.raises(PydanticUserError, match='"v__args", "v__kwargs", "v__positional_only" and "v__duplicate_kwargs" are not permitted'):

        @validate_arguments
        def foo1(v__args: Any) -> None:
            pass  # pragma: no cover
    with pytest.raises(PydanticUserError, match='"v__args", "v__kwargs", "v__positional_only" and "v__duplicate_kwargs" are not permitted'):

        @validate_arguments
        def foo2(v__kwargs: Any) -> None:
            pass  # pragma: no cover
    with pytest.raises(PydanticUserError, match='"v__args", "v__kwargs", "v__positional_only" and "v__duplicate_kwargs" are not permitted'):

        @validate_arguments
        def foo3(v__positional_only: Any) -> None:
            pass  # pragma: no cover
    with pytest.raises(PydanticUserError, match='"v__args", "v__kwargs", "v__positional_only" and "v__duplicate_kwargs" are not permitted'):

        @validate_arguments
        def foo4(v__duplicate_kwargs: Any) -> None:
            pass  # pragma: no cover

def test_async() -> None:

    @validate_arguments
    async def foo(a: int, b: int) -> str:
        return f'a={a} b={b}'

    async def run() -> None:
        v: str = await foo(1, 2)
        assert v == 'a=1 b=2'
    asyncio.run(run())
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(foo('x', 0))  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {'input': {'a': 'x'}, 'loc': ('b',), 'msg': 'Field required', 'type': 'missing'}
    ]

def test_string_annotation() -> None:

    @validate_arguments
    def foo(a: list[int], b: str) -> str:
        return f'a={a!r} b={b!r}'
    assert foo([1, 2, 3], '/')  # type: ignore
    with pytest.raises(ValidationError) as exc_info:
        foo(['x'], '/')  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'x', 'loc': ('a', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'},
        {'input': {'a': ['x']}, 'loc': ('b',), 'msg': 'Field required', 'type': 'missing'}
    ]

def test_item_method() -> None:
    class X:
        def __init__(self, v: int) -> None:
            self.v: int = v

        @validate_arguments
        def foo(self, a: int, b: int) -> str:
            assert self.v == a
            return f'{a}, {b}'
    x: X = X(4)
    assert x.foo(4, 2) == '4, 2'
    assert x.foo(*[4, 2]) == '4, 2'
    with pytest.raises(ValidationError) as exc_info:
        x.foo()  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {'input': {'self': IsInstance(X)}, 'loc': ('a',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {'self': IsInstance(X)}, 'loc': ('b',), 'msg': 'Field required', 'type': 'missing'}
    ]

def test_class_method() -> None:
    class X:
        @classmethod
        @validate_arguments
        def foo(cls, a: int, b: int) -> str:
            assert cls == X
            return f'{a}, {b}'
    x: Type[X] = X
    assert x.foo(4, 2) == '4, 2'
    assert x.foo(*[4, 2]) == '4, 2'
    with pytest.raises(ValidationError) as exc_info:
        x.foo()  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {'input': {'cls': X}, 'loc': ('a',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {'cls': X}, 'loc': ('b',), 'msg': 'Field required', 'type': 'missing'}
    ]

def test_config_title() -> None:

    @validate_arguments
    def foo(a: int, b: int) -> str:
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert foo(1, b=2) == '1, 2'
    assert foo.model.model_json_schema()['title'] == 'Testing'

def test_config_title_cls() -> None:

    class Config:
        title: str = 'Testing'

    @validate_arguments(config={'title': 'Testing'})
    def foo(a: int, b: int) -> str:
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert foo(1, b=2) == '1, 2'
    assert foo.model.model_json_schema()['title'] == 'Testing'

def test_config_fields() -> None:
    with pytest.raises(PydanticUserError, match='Setting the "alias_generator" property on custom Config for @'):

        @validate_arguments
        def foo(a: int, b: int) -> str:
            return f'{a}, {b}'
        # The error is expected during decoration

def test_config_arbitrary_types_allowed() -> None:

    class EggBox:
        def __str__(self) -> str:
            return 'EggBox()'

    @validate_arguments(config={'arbitrary_types_allowed': True})
    def foo(a: int, b: EggBox) -> str:
        return f'{a}, {b}'
    assert foo(1, EggBox()) == '1, EggBox()'
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 2)  # type: ignore
    assert exc_info.value.errors(include_url=False) == [
        {'ctx': {'class': 'test_config_arbitrary_types_allowed.<locals>.EggBox'}, 'input': 2, 'loc': ('b',), 'msg': 'Input should be an instance of test_config_arbitrary_types_allowed.<locals>.EggBox', 'type': 'is_instance_of'}
    ]

def test_validate(mocker: Any) -> None:
    stub: Any = mocker.stub(name='on_something_stub')

    @validate_arguments
    def func(s: str, count: int, *, separator: bytes = b'') -> None:
        stub(s, count, separator)
    func.validate('qwe', 2)
    with pytest.raises(ValidationError):
        func.validate(['qwe'], 2)  # type: ignore
    stub.assert_not_called()

def test_use_of_alias() -> None:

    @validate_arguments
    def foo(c: int = Field(default_factory=lambda: 20), a: int = Field(default_factory=lambda: 10, alias='b')) -> int:
        return a + c
    assert foo(b=10) == 30

def test_populate_by_name() -> None:

    @validate_arguments(config={'populate_by_name': True})
    def foo(a: int, c: int) -> int:
        return a + c
    assert foo(a=10, d=1) == 11  # type: ignore
    assert foo(b=10, c=1) == 11  # type: ignore
    assert foo(a=10, c=1) == 11
