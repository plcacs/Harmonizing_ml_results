"""
New tests for v2 of serialization logic.
"""

import json
import re
import sys
from enum import Enum
from functools import partial, partialmethod
from re import Pattern
from typing import Any, Callable, ClassVar, Optional, Union, Dict, List, TypeVar, cast

import pytest
from pydantic_core import PydanticSerializationError, core_schema, to_jsonable_python
from typing_extensions import TypedDict, Annotated

from pydantic import (
    BaseModel,
    Field,
    FieldSerializationInfo,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    computed_field,
    errors,
    field_serializer,
    model_serializer,
)
from pydantic.config import ConfigDict
from pydantic.functional_serializers import PlainSerializer, WrapSerializer


def test_serialize_extra_allow() -> None:
    class Model(BaseModel):
        x: int
        model_config = ConfigDict(extra='allow')

    m = Model(x=1, y=2)
    assert m.y == 2
    assert m.model_dump() == {'x': 1, 'y': 2}
    assert json.loads(m.model_dump_json()) == {'x': 1, 'y': 2}


def test_serialize_extra_allow_subclass_1() -> None:
    class Parent(BaseModel):
        x: int

    class Child(Parent):
        model_config = ConfigDict(extra='allow')

    class Model(BaseModel):
        inner: Parent

    m = Model(inner=Child(x=1, y=2))
    assert m.inner.y == 2
    assert m.model_dump() == {'inner': {'x': 1}}
    assert json.loads(m.model_dump_json()) == {'inner': {'x': 1}}


def test_serialize_extra_allow_subclass_2() -> None:
    class Parent(BaseModel):
        x: int
        model_config = ConfigDict(extra='allow')

    class Child(Parent):
        y: int

    class Model(BaseModel):
        inner: Parent

    m = Model(inner=Child(x=1, y=2))
    assert m.inner.y == 2
    assert m.model_dump() == {'inner': {'x': 1}}
    assert json.loads(m.model_dump_json()) == {'inner': {'x': 1}}

    m = Model(inner=Parent(x=1, y=2))
    assert m.inner.y == 2
    assert m.model_dump() == {'inner': {'x': 1, 'y': 2}}
    assert json.loads(m.model_dump_json()) == {'inner': {'x': 1, 'y': 2}}


def test_serializer_annotated_plain_always():
    FancyInt = Annotated[int, PlainSerializer(lambda x: f'{x:,}', return_type=str)]

    class MyModel(BaseModel):
        x: FancyInt

    assert MyModel(x=1234).model_dump() == {'x': '1,234'}
    assert MyModel(x=1234).model_dump(mode='json') == {'x': '1,234'}
    assert MyModel(x=1234).model_dump_json() == '{"x":"1,234"}'


def test_serializer_annotated_plain_json():
    FancyInt = Annotated[int, PlainSerializer(lambda x: f'{x:,}', return_type=str, when_used='json')]

    class MyModel(BaseModel):
        x: FancyInt

    assert MyModel(x=1234).model_dump() == {'x': 1234}
    assert MyModel(x=1234).model_dump(mode='json') == {'x': '1,234'}
    assert MyModel(x=1234).model_dump_json() == '{"x":"1,234"}'


def test_serializer_annotated_wrap_always():
    def ser_wrap(v: Any, nxt: SerializerFunctionWrapHandler) -> str:
        return f'{nxt(v + 1):,}'

    FancyInt = Annotated[int, WrapSerializer(ser_wrap, return_type=str)]

    class MyModel(BaseModel):
        x: FancyInt

    assert MyModel(x=1234).model_dump() == {'x': '1,235'}
    assert MyModel(x=1234).model_dump(mode='json') == {'x': '1,235'}
    assert MyModel(x=1234).model_dump_json() == '{"x":"1,235"}'


def test_serializer_annotated_wrap_json():
    def ser_wrap(v: Any, nxt: SerializerFunctionWrapHandler) -> str:
        return f'{nxt(v + 1):,}'

    FancyInt = Annotated[int, WrapSerializer(ser_wrap, when_used='json')]

    class MyModel(BaseModel):
        x: FancyInt

    assert MyModel(x=1234).model_dump() == {'x': 1234}
    assert MyModel(x=1234).model_dump(mode='json') == {'x': '1,235'}
    assert MyModel(x=1234).model_dump_json() == '{"x":"1,235"}'


@pytest.mark.parametrize(
    'serializer, func',
    [
        (PlainSerializer, lambda v: f'{v + 1:,}'),
        (WrapSerializer, lambda v, nxt: f'{nxt(v + 1):,}'),
    ],
)
def test_serializer_annotated_typing_cache(serializer: Any, func: Any) -> None:
    FancyInt = Annotated[int, serializer(func)]

    class FancyIntModel(BaseModel):
        x: Optional[FancyInt]

    assert FancyIntModel(x=1234).model_dump() == {'x': '1,235'}


def test_serialize_decorator_always():
    class MyModel(BaseModel):
        x: Optional[int]

        @field_serializer('x')
        def customise_x_serialization(v: Any, _info: Any) -> str:
            return f'{v:,}'

    assert MyModel(x=1234).model_dump() == {'x': '1,234'}
    assert MyModel(x=1234).model_dump(mode='json') == {'x': '1,234'}
    assert MyModel(x=1234).model_dump_json() == '{"x":"1,234"}'
    m = MyModel(x=None)
    error_msg = (
        'Error calling function `customise_x_serialization`: '
        'TypeError: unsupported format string passed to NoneType.__format__'
    )
    with pytest.raises(PydanticSerializationError, match=error_msg):
        m.model_dump()
    with pytest.raises(PydanticSerializationError, match=error_msg):
        m.model_dump_json()


def test_serialize_decorator_json():
    class MyModel(BaseModel):
        x: int

        @field_serializer('x', when_used='json')
        def customise_x_serialization(v: Any) -> str:
            return f'{v:,}'

    assert MyModel(x=1234).model_dump() == {'x': 1234}
    assert MyModel(x=1234).model_dump(mode='json') == {'x': '1,234'}
    assert MyModel(x=1234).model_dump_json() == '{"x":"1,234"}'


def test_serialize_decorator_unless_none():
    class MyModel(BaseModel):
        x: Optional[int]

        @field_serializer('x', when_used='unless-none')
        def customise_x_serialization(v: Any) -> str:
            return f'{v:,}'

    assert MyModel(x=1234).model_dump() == {'x': '1,234'}
    assert MyModel(x=None).model_dump() == {'x': None}
    assert MyModel(x=1234).model_dump(mode='json') == {'x': '1,234'}
    assert MyModel(x=None).model_dump(mode='json') == {'x': None}
    assert MyModel(x=1234).model_dump_json() == '{"x":"1,234"}'
    assert MyModel(x=None).model_dump_json() == '{"x":null}'


def test_annotated_customisation():
    def parse_int(s: str, _: Any) -> int:
        return int(s.replace(',', ''))

    class CommaFriendlyIntLogic:
        @classmethod
        def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> core_schema.CoreSchema:
            return core_schema.with_info_before_validator_function(
                parse_int,
                core_schema.int_schema(),
                serialization=core_schema.format_ser_schema(',', when_used='unless-none'),
            )

    CommaFriendlyInt = Annotated[int, CommaFriendlyIntLogic]

    class MyModel(BaseModel):
        x: CommaFriendlyInt

    m = MyModel(x='1,000')
    assert m.x == 1000
    assert m.model_dump(mode='json') == {'x': '1,000'}
    assert m.model_dump_json() == '{"x":"1,000"}'


def test_serialize_valid_signatures():
    def ser_plain(v: Any, info: SerializationInfo) -> Any:
        return f'{v:,}'

    def ser_plain_no_info(v: Any, unrelated_arg: int = 1, other_unrelated_arg: int = 2) -> Any:
        return f'{v:,}'

    def ser_wrap(v: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
        return f'{nxt(v):,}'

    class MyModel(BaseModel):
        f1: int
        f2: int
        f3: int
        f4: int
        f5: int

        @field_serializer('f1')
        def ser_f1(self, v: Any, info: FieldSerializationInfo) -> Any:
            assert self.f1 == 1_000
            assert v == 1_000
            assert info.field_name == 'f1'
            return f'{v:,}'

        @field_serializer('f2', mode='wrap')
        def ser_f2(self, v: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo) -> Any:
            assert self.f2 == 2_000
            assert v == 2_000
            assert info.field_name == 'f2'
            return f'{nxt(v):,}'

        ser_f3 = field_serializer('f3')(ser_plain)
        ser_f4 = field_serializer('f4')(ser_plain_no_info)
        ser_f5 = field_serializer('f5', mode='wrap')(ser_wrap)

    m = MyModel(**{f'f{x}': x * 1_000 for x in range(1, 9)})

    assert m.model_dump() == {
        'f1': '1,000',
        'f2': '2,000',
        'f3': '3,000',
        'f4': '4,000',
        'f5': '5,000',
    }
    assert m.model_dump_json() == '{"f1":"1,000","f2":"2,000","f3":"3,000","f4":"4,000","f5":"5,000"}'


def test_invalid_signature_no_params() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature'):

        class _(BaseModel):
            x: int

            @field_serializer('x')
            def no_args() -> Any: ...


def test_invalid_signature_single_params() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature'):

        class _(BaseModel):
            x: int

            @field_serializer('x')
            def no_args(self) -> Any: ...


def test_invalid_signature_too_many_params_1() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature'):

        class _(BaseModel):
            x: int

            @field_serializer('x')
            def no_args(self, value: Any, nxt: Any, info: Any, extra_param: Any) -> Any: ...


def test_invalid_signature_too_many_params_2() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature'):

        class _(BaseModel):
            x: int

            @field_serializer('x')
            @staticmethod
            def no_args(not_self: Any, value: Any, nxt: Any, info: Any) -> Any: ...


def test_invalid_signature_bad_plain_signature() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature for'):

        class _(BaseModel):
            x: int

            @field_serializer('x', mode='plain')
            def no_args(self, value: Any, nxt: Any, info: Any) -> Any: ...


def test_serialize_ignore_info_plain():
    class MyModel(BaseModel):
        x: int

        @field_serializer('x')
        def ser_x(v: Any) -> str:
            return f'{v:,}'

    assert MyModel(x=1234).model_dump() == {'x': '1,234'}


def test_serialize_ignore_info_wrap():
    class MyModel(BaseModel):
        x: int

        @field_serializer('x', mode='wrap')
        def ser_x(v: Any, handler: SerializerFunctionWrapHandler) -> str:
            return f'{handler(v):,}'

    assert MyModel(x=1234).model_dump() == {'x': '1,234'}


def test_serialize_decorator_self_info():
    class MyModel(BaseModel):
        x: Optional[int]

        @field_serializer('x')
        def customise_x_serialization(self, v: Any, info: SerializationInfo) -> str:
            return f'{info.mode}:{v:,}'

    assert MyModel(x=1234).model_dump() == {'x': 'python:1,234'}
    assert MyModel(x=1234).model_dump(mode='foobar') == {'x': 'foobar:1,234'}


def test_serialize_decorator_self_no_info():
    class MyModel(BaseModel):
        x: Optional[int]

        @field_serializer('x')
        def customise_x_serialization(self, v: Any) -> str:
            return f'{v:,}'

    assert MyModel(x=1234).model_dump() == {'x': '1,234'}


def test_model_serializer_plain():
    class MyModel(BaseModel):
        a: int
        b: bytes

        @model_serializer
        def _serialize(self) -> Union[str, Dict[str, Any]]:
            if self.b == b'custom':
                return f'MyModel(a={self.a!r}, b={self.b!r})'
            else:
                return self.__dict__

    m = MyModel(a=1, b='boom')
    assert m.model_dump() == {'a': 1, 'b': b'boom'}
    assert m.model_dump(mode='json') == {'a': 1, 'b': 'boom'}
    assert m.model_dump_json() == '{"a":1,"b":"boom"}'

    assert m.model_dump(exclude={'a'}) == {'a': 1, 'b': b'boom'}
    assert m.model_dump(mode='json', exclude={'a'}) == {'a': 1, 'b': 'boom'}
    assert m.model_dump_json(exclude={'a'}) == '{"a":1,"b":"boom"}'

    m = MyModel(a=1, b='custom')
    assert m.model_dump() == "MyModel(a=1, b=b'custom')"
    assert m.model_dump(mode='json') == "MyModel(a=1, b=b'custom')"
    assert m.model_dump_json() == '"MyModel(a=1, b=b\'custom\')"'


def test_model_serializer_plain_info():
    class MyModel(BaseModel):
        a: int
        b: bytes

        @model_serializer
        def _serialize(self, info: SerializationInfo) -> Dict[str, Any]:
            if info.exclude:
                return {k: v for k, v in self.__dict__.items() if k not in info.exclude}
            else:
                return self.__dict__

    m = MyModel(a=1, b='boom')
    assert m.model_dump() == {'a': 1, 'b': b'boom'}
    assert m.model_dump(mode='json') == {'a': 1, 'b': 'boom'}
    assert m.model_dump_json() == '{"a":1,"b":"boom"}'

    assert m.model_dump(exclude={'a'}) == {'b': b'boom'}
    assert m.model_dump(mode='json', exclude={'a'}) == {'b': 'boom'}
    assert m.model_dump_json(exclude={'a'}) == '{"b":"boom"}'


def test_model_serializer_wrap():
    class MyModel(BaseModel):
        a: int
        b: bytes
        c: bytes = Field(exclude=True)

        @model_serializer(mode='wrap')
        def _serialize(self, handler: Callable[['MyModel'], Dict[str, Any]]) -> Dict[str, Any]:
            d = handler(self)
            d['extra'] = 42
            return d

    m = MyModel(a=1, b='boom', c='excluded')
    assert m.model_dump() == {'a': 1, 'b': b'boom', 'extra': 42}
    assert m.model_dump(mode='json') == {'a': 1, 'b': '