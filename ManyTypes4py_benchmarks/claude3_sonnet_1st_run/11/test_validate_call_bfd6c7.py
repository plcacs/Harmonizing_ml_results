import asyncio
import inspect
import re
import sys
from datetime import datetime, timezone
from functools import partial
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Type, TypeVar, Union, cast, get_type_hints
import pytest
from pydantic_core import ArgsKwargs
from typing_extensions import Required, TypedDict, Unpack
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, PydanticInvalidForJsonSchema, PydanticUserError, Strict, TypeAdapter, ValidationError, validate_call, with_config

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
    with pytest.raises(PydanticUserError, match=f'Partial of `{list}` is invalid because the type of `{list}` is not supported by `validate_call`'):
        validate_call(partial(list))
    with pytest.raises(PydanticUserError, match='`validate_call` should be applied to one of the following: function, method, partial, or lambda'):
        validate_call([])

def validate_bare_none() -> None:

    @validate_call
    def func(f: Optional[int] = None) -> Optional[int]:
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

    class M(type):
        ...
    for cls in (A, int, type, Exception, M):
        with pytest.raises(PydanticUserError, match=re.escape('`validate_call` should be applied to functions, not classes (put `@validate_call` on top of `__init__` or `__new__` instead)')):
            validate_call(cls)
    assert A('5').x == 5

def test_validate_custom_callable() -> None:

    class A:

        def __call__(self, x: int) -> int:
            return x
    with pytest.raises(PydanticUserError, match=re.escape('`validate_call` should be applied to functions, not instances or other callables. Use `validate_call` explicitly on `__call__` instead.')):
        validate_call(A())
    a = A()
    assert validate_call(a.__call__)('5') == 5

    class B:

        @validate_call
        def __call__(self, x: int) -> int:
            return x
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
def test_classmethod_order_error(decorator: Callable) -> None:
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
        foo(1, 2, a=3)
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 2, a=3, b=4)
    assert exc_info.value.errors(include_url=False) == [{'type': 'multiple_argument_values', 'loc': ('a',), 'msg': 'Got multiple values for argument', 'input': 3}, {'type': 'multiple_argument_values', 'loc': ('b',), 'msg': 'Got multiple values for argument', 'input': 4}]

def test_optional() -> None:

    @validate_call
    def foo_bar(a: Optional[int] = None) -> str:
        return f'a={a}'
    assert foo_bar() == 'a=None'
    assert foo_bar(1) == 'a=1'
    with pytest.raises(ValidationError) as exc_info:
        foo_bar(None)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': (0,), 'msg': 'Input should be a valid integer', 'input': None}]

def test_kwargs() -> None:

    @validate_call
    def foo(*, a: int, b: int) -> int:
        return a + b
    assert foo(a=1, b=3) == 4
    with pytest.raises(ValidationError) as exc_info:
        foo(a=1, b='x')
    assert exc_info.value.errors(include_url=False) == [{'input': 'x', 'loc': ('b',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 'x')
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_keyword_only_argument', 'loc': ('a',), 'msg': 'Missing required keyword only argument', 'input': ArgsKwargs((1, 'x'))}, {'type': 'missing_keyword_only_argument', 'loc': ('b',), 'msg': 'Missing required keyword only argument', 'input': ArgsKwargs((1, 'x'))}, {'type': 'unexpected_positional_argument', 'loc': (0,), 'msg': 'Unexpected positional argument', 'input': 1}, {'type': 'unexpected_positional_argument', 'loc': (1,), 'msg': 'Unexpected positional argument', 'input': 'x'}]

def test_untyped() -> None:

    @validate_call
    def foo(a: Any, b: Any, c: str = 'x', *, d: str = 'y') -> str:
        return ', '.join((str(arg) for arg in [a, b, c, d]))
    assert foo(1, 2) == '1, 2, x, y'
    assert foo(1, {'x': 2}, c='3', d='4') == "1, {'x': 2}, 3, 4"

@pytest.mark.parametrize('validated', (True, False))
def test_var_args_kwargs(validated: bool) -> None:

    def foo(a: int, b: int, *args: int, d: int = 3, **kwargs: Any) -> str:
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
    with pytest.raises(PydanticUserError) as exc:

        @validate_call
        def foo(a: int, b: int, **kwargs: Unpack[TD]) -> None:
            pass
    assert exc.value.code == 'overlapping-unpack-typed-dict'
    assert exc.value.message == "Typed dictionary 'TD' overlaps with parameters 'a', 'b'"

    @validate_call
    def foo(a: int, /, **kwargs: Unpack[TD]) -> None:
        pass
    foo(1, a=1)

def test_unpacked_typed_dict_kwargs() -> None:

    @with_config({'strict': True})
    class TD(TypedDict, total=False):
        a: int
        b: str

    @validate_call
    def foo1(**kwargs: Unpack[TD]) -> None:
        pass

    @validate_call
    def foo2(**kwargs: Unpack[TD]) -> None:
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
    def foo2(a: int, b: Annotated[int, Field(default_factory=lambda: 99)] = ..., *args: int) -> int:
        """mypy reports Incompatible default for argument "b" if we don't supply ANY as default"""
        return a + b + sum(args)
    assert foo2(1) == 100

def test_positional_only(create_module: Callable) -> None:
    module = create_module("\nfrom pydantic import validate_call\n\n@validate_call\ndef foo(a: int, b: int, /, c: Optional[int] = None) -> str:\n    return f'{a}, {b}, {c}'\n")
    assert module.foo(1, 2) == '1, 2, None'
    assert module.foo(1, 2, 44) == '1, 2, 44'
    assert module.foo(1, 2, c=44) == '1, 2, 44'
    with pytest.raises(ValidationError) as exc_info:
        module.foo(1, b=2)
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_positional_only_argument', 'loc': (1,), 'msg': 'Missing required positional only argument', 'input': ArgsKwargs((1,), {'b': 2})}, {'type': 'unexpected_keyword_argument', 'loc': ('b',), 'msg': 'Unexpected keyword argument', 'input': 2}]
    with pytest.raises(ValidationError) as exc_info:
        module.foo(a=1, b=2)
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_positional_only_argument', 'loc': (0,), 'msg': 'Missing required positional only argument', 'input': ArgsKwargs((), {'a': 1, 'b': 2})}, {'type': 'missing_positional_only_argument', 'loc': (1,), 'msg': 'Missing required positional only argument', 'input': ArgsKwargs((), {'a': 1, 'b': 2})}, {'type': 'unexpected_keyword_argument', 'loc': ('a',), 'msg': 'Unexpected keyword argument', 'input': 1}, {'type': 'unexpected_keyword_argument', 'loc': ('b',), 'msg': 'Unexpected keyword argument', 'input': 2}]

def test_args_name() -> None:

    @validate_call
    def foo(args: int, kwargs: int) -> str:
        return f'args={args!r}, kwargs={kwargs!r}'
    assert foo(1, 2) == 'args=1, kwargs=2'
    with pytest.raises(ValidationError, match='apple\\s+Unexpected keyword argument'):
        foo(1, 2, apple=4)
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 2, apple=4, banana=5)
    assert exc_info.value.errors(include_url=False) == [{'type': 'unexpected_keyword_argument', 'loc': ('apple',), 'msg': 'Unexpected keyword argument', 'input': 4}, {'type': 'unexpected_keyword_argument', 'loc': ('banana',), 'msg': 'Unexpected keyword argument', 'input': 5}]
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 2, 3)
    assert exc_info.value.errors(include_url=False) == [{'type': 'unexpected_positional_argument', 'loc': (2,), 'msg': 'Unexpected positional argument', 'input': 3}]

def test_async() -> None:

    @validate_call
    async def foo(a: int, b: int) -> str:
        return f'a={a} b={b}'

    async def run() -> None:
        v = await foo(1, 2)
        assert v == 'a=1 b=2'
    assert inspect.iscoroutinefunction(foo) is True
    asyncio.run(run())
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(foo('x'))
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs(('x',))}]

def test_string_annotation() -> None:

    @validate_call
    def foo(a: 'List[int]', b: 'float') -> str:
        return f'a={a!r} b={b!r}'
    assert foo([1, 2, 3], 22) == 'a=[1, 2, 3] b=22.0'
    with pytest.raises(ValidationError) as exc_info:
        foo(['x'])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': (0, 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}, {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs((['x'],))}]

def test_local_annotation() -> None:
    ListInt = list[int]

    @validate_call
    def foo(a: ListInt) -> str:
        return f'a={a!r}'
    assert foo([1, 2, 3]) == 'a=[1, 2, 3]'
    with pytest.raises(ValidationError) as exc_info:
        foo(['x'])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': (0, 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}]

def test_item_method() -> None:

    class X:

        def __init__(self, v: int) -> None:
            self.v = v

        @validate_call
        def foo(self, a: int, b: int) -> str:
            assert self.v == a
            return f'{a}, {b}'
    x = X(4)
    assert x.foo(4, 2) == '4, 2'
    assert x.foo(*[4, 2]) == '4, 2'
    with pytest.raises(ValidationError) as exc_info:
        x.foo()
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('a',), 'msg': 'Missing required argument', 'input': ArgsKwargs((x,))}, {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs((x,))}]

def test_class_method() -> None:

    class X:

        @classmethod
        @validate_call
        def foo(cls, a: int, b: int) -> str:
            assert cls == X
            return f'{a}, {b}'
    x = X()
    assert x.foo(4, 2) == '4, 2'
    assert x.foo(*[4, 2]) == '4, 2'
    with pytest.raises(ValidationError) as exc_info:
        x.foo()
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('a',), 'msg': 'Missing required argument', 'input': ArgsKwargs((X,))}, {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs((X,))}]

def test_json_schema() -> None:

    @validate_call
    def foo(a: int, b: Optional[int] = None) -> str:
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert foo(1, b=2) == '1, 2'
    assert foo(1) == '1, None'
    assert TypeAdapter(foo).json_schema() == {'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'default': None, 'title': 'B', 'type': 'integer'}}, 'required': ['a'], 'additionalProperties': False}

    @validate_call
    def foo(a: int, /, b: int) -> str:
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert TypeAdapter(foo).json_schema() == {'maxItems': 2, 'minItems': 2, 'prefixItems': [{'title': 'A', 'type': 'integer'}, {'title': 'B', 'type': 'integer'}], 'type': 'array'}

    @validate_call
    def foo(a: int, /, *, b: int, c: int) -> str:
        return f'{a}, {b}, {c}'
    assert foo(1, b=2, c=3) == '1, 2, 3'
    with pytest.raises(PydanticInvalidForJsonSchema, match='Unable to generate JSON schema for arguments validator with positional-only and keyword-only arguments'):
        TypeAdapter(foo).json_schema()

    @validate_call
    def foo(*numbers: int) -> int:
        return sum(numbers)
    assert foo(1, 2, 3) == 6
    assert TypeAdapter(foo).json_schema() == {'items': {'type': 'integer'}, 'type': 'array'}

    @validate_call
    def foo(a: int, *numbers: int) -> int:
        return a + sum(numbers)
    assert foo(1, 2, 3) == 6
    assert TypeAdapter(foo).json_schema() == {'items': {'type': 'integer'}, 'prefixItems': [{'title': 'A', 'type': 'integer'}], 'minItems': 1, 'type': 'array'}

    @validate_call
    def foo(**scores: int) -> str:
        return ', '.join((f'{k}={v}' for k, v in sorted(scores.items())))
    assert foo(a=1, b=2) == 'a=1, b=2'
    assert TypeAdapter(foo).json_schema() == {'additionalProperties': {'type': 'integer'}, 'properties': {}, 'type': 'object'}

    @validate_call
    def foo(a: int) -> int:
        return a
    assert foo(1) == 1
    assert TypeAdapter(foo).json_schema() == {'additionalProperties': False, 'properties': {'A': {'title': 'A', 'type': 'integer'}}, 'required': ['A'], 'type': 'object'}

def test_alias_generator() -> None:

    @validate_call(config=dict(alias_generator=lambda x: x * 2))
    def foo(a: int, b: int) -> str:
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert foo(aa=1, bb=2) == '1, 2'

def test_config_arbitrary_types_allowed() -> None:

    class EggBox:

        def __str__(self) -> str:
            return 'EggBox()'

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def foo(a: int, b: EggBox) -> str:
        return f'{a}, {b}'
    assert foo(1, EggBox()) == '1, EggBox()'
    with pytest.raises(ValidationError) as exc_info:
        assert foo(1, 2) == '1, 2'
    assert exc_info.value.errors(include_url=False) == [{'type': 'is_instance_of', 'loc': (1,), 'msg': 'Input should be an instance of test_config_arbitrary_types_allowed.<locals>.EggBox', 'input': 2, 'ctx': {'class': 'test_config_arbitrary_types_allowed.<locals>.EggBox'}}]

def test_config_strict() -> None:

    @validate_call(config=dict(strict=True))
    def foo(a: int, b: list[str]) -> str:
        return f'{a}, {b[0]}'
    assert foo(1, ['bar', 'foobar']) == '1, bar'
    with pytest.raises(ValidationError) as exc_info:
        foo('foo', ('bar', 'foobar'))
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': (0,), 'msg': 'Input should be a valid integer', 'input': 'foo'}, {'type': 'list_type', 'loc': (1,), 'msg': 'Input should be a valid list', 'input': ('bar', 'foobar')}]

def test_annotated_num() -> None:

    @validate_call
    def f(a: Annotated[int, Field(gt=0, lt=10)]) -> int:
        return a
    assert f(5) == 5
    with pytest.raises(ValidationError) as exc_info:
        f(0)
    assert exc_info.value.errors(include_url=False) == [{'type': 'greater_than', 'loc': (0,), 'msg': 'Input should be greater than 0', 'input': 0, 'ctx': {'gt': 0}}]
    with pytest.raises(ValidationError) as exc_info:
        f(10)
    assert exc_info.value.errors(include_url=False) == [{'type': 'less_than', 'loc': (0,), 'msg': 'Input should be less than 10', 'input': 10, 'ctx': {'lt': 10}}]

def test_annotated_discriminator() -> None:

    class Cat(BaseModel):
        type: Literal['cat'] = 'cat'

    class Dog(BaseModel):
        type: Literal['dog'] = 'dog'
        bark: str = 'woof'
    Pet = Annotated[Union[Cat, Dog], Field(discriminator='type')]

    @validate_call
    def f(pet: Pet) -> Union[Cat, Dog]:
        return pet
    with pytest.raises(ValidationError) as exc_info:
        f({'food': 'fish'})
    assert exc_info.value.errors(include_url=False) == [{'type': 'union_tag_not_found', 'loc': (0,), 'msg': "Unable to extract tag using discriminator 'type'", 'input': {'food': 'fish'}, 'ctx': {'discriminator': "'type'"}}]
    with pytest.raises(ValidationError) as exc_info:
        f({'type': 'dog', 'food': 'fish'})
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing', 'loc': (0, 'dog', 'bark'), 'msg': 'Field required', 'input': {'type': 'dog', 'food': 'fish'}}]

def test_annotated_validator() -> None:

    @validate_call
    def f(x: Annotated[int, BeforeValidator(lambda x: int(x) + 12)]) -> int:
        return x
    assert f('1') == 13

def test_annotated_strict() -> None:

    @validate_call
    def f1(x: Annotated[int, Strict()]) -> int:
        return x

    @validate_call
    def f2(x: Annotated[int, Strict]) -> int:
        return x
    for f in (f1, f2):
        assert f(1) == 1
        with pytest.raises(ValidationError) as exc_info:
            f('1')
        assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': (0,), 'msg': 'Input should be a valid integer', 'input': '1'}]

def test_annotated_use_of_alias() -> None:

    @validate_call
    def foo(a: Annotated[int, Field(alias='b')], c: int, d: Annotated[int, Field(alias='')]) -> int:
        return a + c + d
    assert foo(**{'b': 10, 'c': 12, '': 1}) == 23
    with pytest.raises(ValidationError) as exc_info:
        assert foo(a=10, c=12, d=1) == 10
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs((), {'a': 10, 'c': 12, 'd': 1})}, {'type': 'missing_argument', 'loc': ('',), 'msg': 'Missing required argument', 'input': ArgsKwargs((), {'a': 10, 'c': 12, 'd': 1})}, {'type': 'unexpected_keyword_argument', 'loc': ('a',), 'msg': 'Unexpected keyword argument', 'input': 10}, {'type': 'unexpected_keyword_argument', 'loc': ('d',), 'msg': 'Unexpected keyword argument', 'input': 1}]

def test_use_of_alias() -> None:

    @validate_call
    def foo(c: int = Field(default_factory=lambda: 20), a: int = Field(default_factory=lambda: 10, alias='b')) -> int:
        return a + c
    assert foo(b=10) == 30

def test_populate_by_name() -> None:

    @validate_call(config=dict(populate_by_name=True))
    def foo(a: Annotated[int, Field(alias='b')], c: Annotated[int, Field(alias='d')]) -> int:
        return a + c
    assert foo(b=10, d=1) == 11
    assert foo(a=10, d=1) == 11
    assert foo(b=10, c=1) == 11
    assert foo(a=10, c=1) == 11

def test_validate_return() -> None:

    @validate_call(config=dict(validate_return=True))
    def foo(a: int, b: int) -> int:
        return a + b
    assert foo(1, 2) == 3

def test_validate_all() -> None:

    @validate_call(config=dict(validate_default=True))
    def foo(dt: datetime = Field(default_factory=lambda: 946684800)) -> datetime:
        return dt
    assert foo() == datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert foo(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)

def test_validate_all_positional(create_module: Callable) -> None:
    module = create_module('\nfrom datetime import datetime\n\nfrom pydantic import Field, validate_call\n\n@validate_call(config=dict(validate_default=True))\ndef foo(dt: datetime = Field(default_factory=lambda: 946684800), /):\n    return dt\n')
    assert module.foo() == datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert module.foo(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)

def test_partial() -> None:

    def my_wrapped_function(a: int, b: int, c: int) -> int:
        return a + b + c
    my_partial_function = partial(my_wrapped_function, c=3)
    f = validate_call(my_partial_function)
    assert f(1, 2) == 6

def test_validator_init() -> None:

    class Foo:

        @validate_call
        def __init__(self, a: int, b: int) -> None:
            self.v = a + b
    assert Foo(1, 2).v == 3
    assert Foo(1, '2').v == 3
    with pytest.raises(ValidationError, match="type=int_parsing, input_value='x', input_type=str"):
        Foo(1, 'x')

def test_positional_and_keyword_with_same_name(create_module: Callable) -> None:
    module = create_module('\nfrom pydantic import validate_call\n\n@validate_call\ndef f(a: int, /, **kwargs):\n    return a, kwargs\n')
    assert module.f(1, a=2) == (1, {'a': 2})

def test_model_as_arg() -> None:

    class Model1(TypedDict):
        x: int

    class Model2(BaseModel):
        y: int

    @validate_call(validate_return=True)
    def f1(m1: Model1, m2: Model2) -> tuple[Model1, Dict[str, Any]]:
        return (m1, m2.model_dump())
    res = f1({'x': '1'}, {'y': '2'})
    assert res == ({'x': 1}, {'y': 2})

def test_do_not_call_repr_on_validate_call() -> None:

    class Class:

        @validate_call
        def __init__(self, number: int) -> None:
            ...

        def __repr__(self) -> str:
            assert False
    Class(50)

def test_methods_are_not_rebound() -> None:

    class Thing:

        def __init__(self, x: int) -> None:
            self.x = x

        def a(self, x: Any) -> Any:
            return x + self.x
        c = validate_call(a)
    thing = Thing(1)
    assert thing.a == thing.a
    assert thing.c == thing.c
    assert Thing.c == Thing.c
    assert Thing.c(thing, '2') == 3
    assert Thing(2).c('3') == 5

def test_basemodel_method() -> None:

    class Foo(BaseModel):

        @classmethod
        @validate_call
        def test(cls, x: int) -> tuple[Type[Foo], int]:
            return (cls, x)
    assert Foo.test('1') == (Foo, 1)

    class Bar(BaseModel):

        @validate_call
        def test(self, x: int) -> tuple[Bar, int]:
            return (self, x)
    bar = Bar()
    assert bar.test('1') == (bar, 1)

def test_dynamic_method_decoration() -> None:

    class Foo:

        def bar(self, value: str) -> str:
            return f'bar-{value}'
    Foo.bar = validate_call(Foo.bar)
    assert Foo.bar
    foo = Foo()
    assert foo.bar('test') == 'bar-test'

def test_async_func() -> None:

    @validate_call(validate_return=True)
    async def foo(a: int) -> int:
        return a
    res = asyncio.run(foo(1))
    assert res == 1
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(foo('x'))
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': (), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}]

def test_validate_call_with_slots() -> None:

    class ClassWithSlots:
        __slots__ = {}

        @validate_call(validate_return=True)
        def some_instance_method(self, x: str) -> str:
            return x

        @classmethod
        @validate_call(validate_return=True)
        def some_class_method(cls, x: str) -> str:
            return x

        @staticmethod
        @validate_call(validate_return=True)
        def some_static_method(x: str) -> str:
            return x
    c = ClassWithSlots()
    assert c.some_instance_method(x='potato') == 'potato'
    assert c.some_class_method(x='pepper') == 'pepper'
    assert c.some_static_method(x='onion') == 'onion'
    assert c.some_instance_method == c.some_instance_method
    assert c.some_class_method == c.some_class_method
    assert c.some_static_method == c.some_static_method

def test_eval_type_backport() -> None:

    @validate_call
    def foo(bar: list[Union[int, str]]) -> list[Union[int, str]]:
        return bar
    assert foo([1, '2']) == [1, '2']
    with pytest.raises(ValidationError) as exc_info:
        foo('not a list')
    assert exc_info.value.errors(include_url=False) == [{'type': 'list_type', 'loc': (0,), 'msg': 'Input should be a valid list', 'input': 'not a list'}]
    with pytest.raises(ValidationError) as exc_info:
        foo([{'not a str or int'}])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': (0, 0, 'int'), 'msg': 'Input should be a valid integer', 'input': {'not a str or int'}}, {'type': 'string_type', 'loc': (0, 0, 'str'), 'msg': 'Input should be a valid string', 'input': {'not a str or int'}}]

def test_eval_namespace_basic(create_module: Callable) -> None:
    module = create_module("\nfrom __future__ import annotations\nfrom typing import TypeVar\nfrom pydantic import validate_call\n\nT = TypeVar('T', bound=int)\n\n@validate_call\ndef f(x: T): ...\n\ndef g():\n    MyList = list\n\n    @validate_call\n    def h(x: MyList[int]): ...\n    return h\n")
    f = module.f
    f(1)
    with pytest.raises(ValidationError) as exc_info:
        f('x')
    assert exc_info.value.errors(include_url=False) == [{'input': 'x', 'loc': (0,), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]
    h = module.g()
    with pytest.raises(ValidationError) as exc_info:
        h('not a list')
    assert exc_info.value.errors(include_url=False) == [{'input': 'not a list', 'loc': (0,), 'msg': 'Input should be a valid list', 'type': 'list_type'}]

@pytest.mark.skipif(sys.version_info < (3, 12), reason='requires Python 3.12+ for PEP 695 syntax with generics')
def test_validate_call_with_pep_695_syntax(create_module: Callable) -> None:
    """Note: validate_call still doesn't work properly with generics, see https://github.com/pydantic/pydantic/issues/7796.

    This test is just to ensure that the syntax is accepted and doesn't raise a NameError."""
    module = create_module("\nfrom typing import Iterable\nfrom pydantic import validate_call\n\n@validate_call\ndef find_max_no_validate_return[T](args: 'Iterable[T]') -> T:\n    return sorted(args, reverse=True)[0]\n\n@validate_call(validate_return=True)\ndef find_max_validate_return[T](args: 'Iterable[T]') -> T:\n    return sorted(args, reverse=True)[0]\n        ")
    functions = [module.find_max_no_validate_return, module.find_max_validate_return]
    for find_max in functions:
        assert len(find_max.__type_params__) == 1
        assert find_max([1, 2, 10, 5]) == 10
        with pytest.raises(ValidationError):
            find_max(1)

@pytest.mark.skipif(sys.version_info < (3, 12), reason='requires Python 3.12+ for PEP 695 syntax with generics')
def test_pep695_with_class(create_module: Callable) -> None:
    """Primarily to ensure that the syntax is accepted and doesn't raise a `NameError` with `T`.
    The validation is not expected to work properly when parameterized at this point."""
    for import_annotations in ('from __future__ import annotations', ''):
        module = create_module(f'\n{import_annotations}\nfrom pydantic import validate_call\n\nclass A[T]:\n    @validate_call(validate_return=True)\n    def f(self, a: T) -> T:\n        return str(a)\n            ')
        A = module.A
        a = A[int]()
        assert a.f(1) == '1'
        assert a.f('1') == '1'

@pytest.mark.skipif(sys.version_info < (3, 12), reason='requires Python 3.12+ for PEP 695 syntax with generics')
def test_pep695_with_nested_scopes(create_module: Callable) -> None:
    """Nested scopes generally cannot be caught by `parent_frame_namespace`,
    so currently this test is expected to fail.
    """
    module = create_module('\nfrom __future__ import annotations\nfrom pydantic import validate_call\n\nclass A[T]:\n    def g(self):\n        @validate_call(validate_return=True)\n        def inner(a: T) -> T: ...\n\n    def h[S](self):\n        @validate_call(validate_return=True)\n        def inner(a: T) -> S: ...\n        ')
    A = module.A
    a = A[int]()
    with pytest.raises(NameError):
        a.g()
    with pytest.raises(NameError):
        a.h()
    with pytest.raises(NameError):
        create_module('\nfrom __future__ import annotations\nfrom pydantic import validate_call\n\nclass A[T]:\n    class B:\n        @validate_call(validate_return=True)\n        def f(a: T) -> T: ...\n\n    class C[S]:\n        @validate_call(validate_return=True)\n        def f(a: T) -> S: ...\n            ')

class M0(BaseModel):
    pass
M: Type[BaseModel] = M0

def test_uses_local_ns() -> None:

    class M1(BaseModel):
        z: int
    M: Type[BaseModel] = M1

    def foo() -> None:

        class M2(BaseModel):
            z: int
        M: Type[BaseModel] = M2

        @validate_call(validate_return=True)
        def bar(m: M) -> M:
            return m
        assert bar({'z': 1}) == M2(z=1)
    foo()
