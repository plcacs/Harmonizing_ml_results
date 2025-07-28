#!/usr/bin/env python
"""
New tests for v2 of serialization logic.
"""
import json
import re
import sys
from enum import Enum
from functools import partial, partialmethod
from re import Pattern
from typing import Any, Callable, Dict, Optional, Union, Literal
import pytest
from pydantic_core import PydanticSerializationError, core_schema
from pydantic_core.core_schema import CoreSchema
from pydantic import BaseModel, Field, FieldSerializationInfo, SerializationInfo, PlainSerializer, WrapSerializer, computed_field, errors, field_serializer, model_serializer, ConfigDict
from typing_extensions import TypedDict
from pydantic import TypeAdapter

def test_serialize_extra_allow() -> None:
    class Model(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')
    m = Model(x=1, y=2)
    assert m.y == 2
    assert m.model_dump() == {'x': 1, 'y': 2}
    assert json.loads(m.model_dump_json()) == {'x': 1, 'y': 2}

def test_serialize_extra_allow_subclass_1() -> None:
    class Parent(BaseModel):
        pass

    class Child(Parent):
        model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')

    class Model(BaseModel):
        pass
    m = Model(inner=Child(x=1, y=2))
    assert m.inner.y == 2
    assert m.model_dump() == {'inner': {'x': 1}}
    assert json.loads(m.model_dump_json()) == {'inner': {'x': 1}}

def test_serialize_extra_allow_subclass_2() -> None:
    class Parent(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')

    class Child(Parent):
        pass

    class Model(BaseModel):
        pass
    m = Model(inner=Child(x=1, y=2))
    assert m.inner.y == 2
    assert m.model_dump() == {'inner': {'x': 1}}
    assert json.loads(m.model_dump_json()) == {'inner': {'x': 1}}
    m = Model(inner=Parent(x=1, y=2))
    assert m.inner.y == 2
    assert m.model_dump() == {'inner': {'x': 1, 'y': 2}}
    assert json.loads(m.model_dump_json()) == {'inner': {'x': 1, 'y': 2}}

def test_serializer_annotated_plain_always() -> None:
    FancyInt = Annotated[int, PlainSerializer(lambda x: f'{x:,}', return_type=str)]

    class MyModel(BaseModel):
        x: FancyInt
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': '1,234'}
    assert m.model_dump(mode='json') == {'x': '1,234'}
    assert m.model_dump_json() == '{"x":"1,234"}'

def test_serializer_annotated_plain_json() -> None:
    FancyInt = Annotated[int, PlainSerializer(lambda x: f'{x:,}', return_type=str, when_used='json')]

    class MyModel(BaseModel):
        x: FancyInt
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': 1234}
    assert m.model_dump(mode='json') == {'x': '1,234'}
    assert m.model_dump_json() == '{"x":"1,234"}'

def test_serializer_annotated_wrap_always() -> None:
    def ser_wrap(v: int, nxt: Callable[[int], Any]) -> str:
        return f'{nxt(v + 1):,}'
    FancyInt = Annotated[int, WrapSerializer(ser_wrap, return_type=str)]

    class MyModel(BaseModel):
        x: FancyInt
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': '1,235'}
    assert m.model_dump(mode='json') == {'x': '1,235'}
    assert m.model_dump_json() == '{"x":"1,235"}'

def test_serializer_annotated_wrap_json() -> None:
    def ser_wrap(v: int, nxt: Callable[[int], Any]) -> str:
        return f'{nxt(v + 1):,}'
    FancyInt = Annotated[int, WrapSerializer(ser_wrap, when_used='json')]

    class MyModel(BaseModel):
        x: FancyInt
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': 1234}
    assert m.model_dump(mode='json') == {'x': '1,235'}
    assert m.model_dump_json() == '{"x":"1,235"}'

@pytest.mark.parametrize('serializer, func', [
    (PlainSerializer, lambda v: f'{v + 1:,}'),
    (WrapSerializer, lambda v, nxt: f'{nxt(v + 1):,}')
])
def test_serializer_annotated_typing_cache(serializer: Any, func: Callable) -> None:
    FancyInt = Annotated[int, serializer(func)]

    class FancyIntModel(BaseModel):
        x: FancyInt
    m = FancyIntModel(x=1234)
    assert m.model_dump() == {'x': '1,235'}

def test_serialize_decorator_always() -> None:
    class MyModel(BaseModel):
        x: Optional[int]

        @field_serializer('x')
        def customise_x_serialization(self, v: int, _info: FieldSerializationInfo) -> str:
            return f'{v:,}'
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': '1,234'}
    assert m.model_dump(mode='json') == {'x': '1,234'}
    assert m.model_dump_json() == '{"x":"1,234"}'
    m = MyModel(x=None)
    error_msg = 'Error calling function `customise_x_serialization`: TypeError: unsupported format string passed to NoneType.__format__'
    with pytest.raises(PydanticSerializationError, match=error_msg):
        m.model_dump()
    with pytest.raises(PydanticSerializationError, match=error_msg):
        m.model_dump_json()

def test_serialize_decorator_json() -> None:
    class MyModel(BaseModel):
        x: int

        @field_serializer('x', when_used='json')
        def customise_x_serialization(self, v: int) -> str:
            return f'{v:,}'
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': 1234}
    assert m.model_dump(mode='json') == {'x': '1,234'}
    assert m.model_dump_json() == '{"x":"1,234"}'

def test_serialize_decorator_unless_none() -> None:
    class MyModel(BaseModel):
        x: Optional[int]

        @field_serializer('x', when_used='unless-none')
        def customise_x_serialization(self, v: Optional[int]) -> Optional[str]:
            if v is None:
                return None
            return f'{v:,}'
    m1 = MyModel(x=1234)
    m2 = MyModel(x=None)
    assert m1.model_dump() == {'x': '1,234'}
    assert m2.model_dump() == {'x': None}
    assert m1.model_dump(mode='json') == {'x': '1,234'}
    assert m2.model_dump(mode='json') == {'x': None}
    assert m1.model_dump_json() == '{"x":"1,234"}'
    assert m2.model_dump_json() == '{"x":null}'

def test_annotated_customisation() -> None:
    def parse_int(s: str, _: Any) -> int:
        return int(s.replace(',', ''))

    class CommaFriendlyIntLogic:
        @classmethod
        def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> CoreSchema:
            return core_schema.with_info_before_validator_function(
                parse_int,
                core_schema.int_schema(),
                serialization=core_schema.format_ser_schema(',', when_used='unless-none')
            )
    CommaFriendlyInt = Annotated[int, CommaFriendlyIntLogic]

    class MyModel(BaseModel):
        x: Union[int, str, CommaFriendlyInt]
    m = MyModel(x='1,000')
    assert m.x == 1000
    assert m.model_dump(mode='json') == {'x': '1,000'}
    assert m.model_dump_json() == '{"x":"1,000"}'

def test_serialize_valid_signatures() -> None:
    def ser_plain(v: int, info: FieldSerializationInfo) -> str:
        return f'{v:,}'

    def ser_plain_no_info(v: int, unrelated_arg: int = 1, other_unrelated_arg: int = 2) -> str:
        return f'{v:,}'

    def ser_wrap(v: int, nxt: Callable[[int], Any], info: FieldSerializationInfo) -> str:
        return f'{nxt(v):,}'

    class MyModel(BaseModel):
        f1: int
        f2: int
        f3: int
        f4: int
        f5: int

        @field_serializer('f1')
        def ser_f1(self, v: int, info: FieldSerializationInfo) -> str:
            assert self.f1 == 1000
            assert v == 1000
            assert info.field_name == 'f1'
            return f'{v:,}'

        @field_serializer('f2', mode='wrap')
        def ser_f2(self, v: int, nxt: Callable[[int], Any], info: FieldSerializationInfo) -> str:
            assert self.f2 == 2000
            assert v == 2000
            assert info.field_name == 'f2'
            return f'{nxt(v):,}'
        ser_f3 = field_serializer('f3')(ser_plain)
        ser_f4 = field_serializer('f4')(ser_plain_no_info)
        ser_f5 = field_serializer('f5', mode='wrap')(ser_wrap)
    m = MyModel(**{f'f{x}': x * 1000 for x in range(1, 6)})
    assert m.model_dump() == {
        'f1': '1,000',
        'f2': '2,000',
        'f3': '3,000',
        'f4': '4,000',
        'f5': '5,000'
    }
    assert m.model_dump_json() == '{"f1":"1,000","f2":"2,000","f3":"3,000","f4":"4,000","f5":"5,000"}'

def test_invalid_signature_no_params() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature'):
        class Model(BaseModel):
            @field_serializer('x')
            def no_args() -> Any:
                ...

def test_invalid_signature_single_params() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature'):
        class Model(BaseModel):
            @field_serializer('x')
            def no_args(self) -> Any:
                ...

def test_invalid_signature_too_many_params_1() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature'):
        class Model(BaseModel):
            @field_serializer('x')
            def no_args(self, value: Any, nxt: Any, info: Any, extra_param: Any) -> Any:
                ...

def test_invalid_signature_too_many_params_2() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature'):
        class Model(BaseModel):
            @field_serializer('x')
            @staticmethod
            def no_args(not_self: Any, value: Any, nxt: Any, info: Any) -> Any:
                ...

def test_invalid_signature_bad_plain_signature() -> None:
    with pytest.raises(TypeError, match='Unrecognized field_serializer function signature for'):
        class Model(BaseModel):
            @field_serializer('x', mode='plain')
            def no_args(self, value: Any, nxt: Any, info: Any) -> Any:
                ...

def test_serialize_ignore_info_plain() -> None:
    class MyModel(BaseModel):
        x: int

        @field_serializer('x')
        def ser_x(v: int) -> str:
            return f'{v:,}'
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': '1,234'}

def test_serialize_ignore_info_wrap() -> None:
    class MyModel(BaseModel):
        x: int

        @field_serializer('x', mode='wrap')
        def ser_x(v: int, handler: Callable[[int], Any]) -> str:
            return f'{handler(v):,}'
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': '1,234'}

def test_serialize_decorator_self_info() -> None:
    class MyModel(BaseModel):
        x: int

        @field_serializer('x')
        def customise_x_serialization(self, v: int, info: FieldSerializationInfo) -> str:
            return f'{info.mode}:{v:,}'
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': 'python:1,234'}
    assert m.model_dump(mode='foobar') == {'x': 'foobar:1,234'}

def test_serialize_decorator_self_no_info() -> None:
    class MyModel(BaseModel):
        x: int

        @field_serializer('x')
        def customise_x_serialization(self, v: int) -> str:
            return f'{v:,}'
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': '1,234'}

def test_model_serializer_plain() -> None:
    class MyModel(BaseModel):
        a: int
        b: Union[str, bytes]

        @model_serializer
        def _serialize(self) -> Any:
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

def test_model_serializer_plain_info() -> None:
    class MyModel(BaseModel):
        a: int
        b: Union[str, bytes]

        @model_serializer
        def _serialize(self, info: SerializationInfo) -> Any:
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

def test_model_serializer_wrap() -> None:
    class MyModel(BaseModel):
        a: int
        b: Union[str, bytes]
        c: Any = Field(exclude=True)

        @model_serializer(mode='wrap')
        def _serialize(self, handler: Callable[[Any], Dict[str, Any]]) -> Dict[str, Any]:
            d = handler(self)
            d['extra'] = 42
            return d
    m = MyModel(a=1, b='boom', c='excluded')
    assert m.model_dump() == {'a': 1, 'b': b'boom', 'extra': 42}
    assert m.model_dump(mode='json') == {'a': 1, 'b': 'boom', 'extra': 42}
    assert m.model_dump_json() == '{"a":1,"b":"boom","extra":42}'
    assert m.model_dump(exclude={'a'}) == {'b': b'boom', 'extra': 42}
    assert m.model_dump(mode='json', exclude={'a'}) == {'b': 'boom', 'extra': 42}
    assert m.model_dump_json(exclude={'a'}) == '{"b":"boom","extra":42}'

def test_model_serializer_wrap_info() -> None:
    class MyModel(BaseModel):
        a: int
        b: Union[str, bytes]
        c: Any = Field(exclude=True)

        @model_serializer(mode='wrap')
        def _serialize(self, handler: Callable[[Any], Dict[str, Any]], info: SerializationInfo) -> Dict[str, Any]:
            d = handler(self)
            d['info'] = f'mode={info.mode} exclude={info.exclude}'
            return d
    m = MyModel(a=1, b='boom', c='excluded')
    assert m.model_dump() == {'a': 1, 'b': b'boom', 'info': 'mode=python exclude=None'}
    assert m.model_dump(mode='json') == {'a': 1, 'b': 'boom', 'info': 'mode=json exclude=None'}
    assert m.model_dump_json() == '{"a":1,"b":"boom","info":"mode=json exclude=None"}'
    assert m.model_dump(exclude={'a'}) == {'b': b'boom', 'info': "mode=python exclude={'a'}"}
    assert m.model_dump(mode='json', exclude={'a'}) == {'b': 'boom', 'info': "mode=json exclude={'a'}"}
    assert m.model_dump_json(exclude={'a'}) == '{"b":"boom","info":"mode=json exclude={\'a\'}"}'

def test_model_serializer_plain_json_return_type() -> None:
    class MyModel(BaseModel):
        a: int

        @model_serializer(when_used='json')
        def _serialize(self) -> Any:
            if self.a == 666:
                return self.a
            else:
                return f'MyModel(a={self.a!r})'
    m = MyModel(a=1)
    assert m.model_dump() == {'a': 1}
    assert m.model_dump(mode='json') == 'MyModel(a=1)'
    assert m.model_dump_json() == '"MyModel(a=1)"'
    m = MyModel(a=666)
    assert m.model_dump() == {'a': 666}
    with pytest.warns(UserWarning, match='Expected `str` but got `int` with value `666` - serialized value may not be as expected'):
        assert m.model_dump(mode='json') == 666
    with pytest.warns(UserWarning, match='Expected `str` but got `int` with value `666` - serialized value may not be as expected'):
        assert m.model_dump_json() == '666'

def test_model_serializer_wrong_args() -> None:
    m_pattern = r"Unrecognized model_serializer function signature for <.+MyModel\._serialize at 0x\w+> with `mode=plain`:\(self, x, y, z\)"
    with pytest.raises(TypeError, match=m_pattern):
        class MyModel(BaseModel):
            a: int
            @model_serializer
            def _serialize(self, x: Any, y: Any, z: Any) -> Any:
                return self

def test_model_serializer_no_self() -> None:
    with pytest.raises(TypeError, match='`@model_serializer` must be applied to instance methods'):
        class MyModel(BaseModel):
            a: int
            @model_serializer
            def _serialize(slf, x: Any, y: Any, z: Any) -> Any:
                return slf

def test_model_serializer_classmethod() -> None:
    with pytest.raises(TypeError, match='`@model_serializer` must be applied to instance methods'):
        class MyModel(BaseModel):
            a: int
            @model_serializer
            @classmethod
            def _serialize(cls, x: Any, y: Any, z: Any) -> Any:
                return cls

def test_field_multiple_serializer() -> None:
    m_pattern = "Multiple field serializer functions were defined for field 'x', this is not allowed."
    with pytest.raises(TypeError, match=m_pattern):
        class MyModel(BaseModel):
            x: int
            @field_serializer('x', 'y')
            def serializer1(v: Any) -> Any:
                return f'{v:,}'

            @field_serializer('x')
            def serializer2(v: Any) -> Any:
                return v

def test_field_multiple_serializer_subclass() -> None:
    class MyModel(BaseModel):
        x: int
        @field_serializer('x')
        def serializer1(v: Any) -> Any:
            return f'{v:,}'

    class MySubModel(MyModel):
        @field_serializer('x')
        def serializer1(v: Any) -> Any:
            return f'{v}'
    assert MyModel(x=1234).model_dump() == {'x': '1,234'}
    assert MySubModel(x=1234).model_dump() == {'x': '1234'}

def test_serialize_all_fields() -> None:
    class MyModel(BaseModel):
        x: int
        @field_serializer('*')
        @classmethod
        def serialize_all(cls, v: Any) -> Any:
            return v * 2
    m = MyModel(x=10)
    assert m.model_dump() == {'x': 20}

def int_ser_func_without_info1(v: int, expected: int) -> str:
    return f'{v:,}'

def int_ser_func_without_info2(v: int, *, expected: int) -> str:
    return f'{v:,}'

def int_ser_func_with_info1(v: int, info: FieldSerializationInfo, expected: int) -> str:
    return f'{v:,}'

def int_ser_func_with_info2(v: int, info: FieldSerializationInfo, *, expected: int) -> str:
    return f'{v:,}'

def int_ser_instance_method_without_info1(self: Any, v: int, *, expected: int) -> str:
    assert self.x == v
    return f'{v:,}'

def int_ser_instance_method_without_info2(self: Any, v: int, expected: int) -> str:
    assert self.x == v
    return f'{v:,}'

def int_ser_instance_method_with_info1(self: Any, v: int, info: FieldSerializationInfo, expected: int) -> str:
    assert self.x == v
    return f'{v:,}'

def int_ser_instance_method_with_info2(self: Any, v: int, info: FieldSerializationInfo, *, expected: int) -> str:
    assert self.x == v
    return f'{v:,}'

@pytest.mark.parametrize('func', [
    int_ser_func_with_info1, int_ser_func_with_info2, int_ser_func_without_info1, int_ser_func_without_info2,
    int_ser_instance_method_with_info1, int_ser_instance_method_with_info2, int_ser_instance_method_without_info1, int_ser_instance_method_without_info2
])
def test_serialize_partial(func: Callable[..., str]) -> None:
    class MyModel(BaseModel):
        x: int
        ser = field_serializer('x', return_type=str)(partial(func, expected=1234))
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': '1,234'}

@pytest.mark.parametrize('func', [
    int_ser_func_with_info1, int_ser_func_with_info2, int_ser_func_without_info1, int_ser_func_without_info2,
    int_ser_instance_method_with_info1, int_ser_instance_method_with_info2, int_ser_instance_method_without_info1, int_ser_instance_method_without_info2
])
def test_serialize_partialmethod(func: Callable[..., str]) -> None:
    class MyModel(BaseModel):
        x: int
        ser = field_serializer('x', return_type=str)(partialmethod(func, expected=1234))
    m = MyModel(x=1234)
    assert m.model_dump() == {'x': '1,234'}

def test_serializer_allow_reuse_inheritance_override() -> None:
    class Parent(BaseModel):
        x: int

        @field_serializer('x')
        def ser_x(self, _v: Any, _info: FieldSerializationInfo) -> str:
            return 'parent_encoder'

    class Child1(Parent):
        @field_serializer('x')
        def ser_x(self, _v: Any, _info: FieldSerializationInfo) -> str:
            return 'child1_encoder' + ' ' + super().ser_x(_v, _info)
    p = Parent(x=1)
    c = Child1(x=1)
    assert p.model_dump_json() == '{"x":"parent_encoder"}'
    assert c.model_dump_json() == '{"x":"child1_encoder parent_encoder"}'
    msg = "Multiple field serializer functions were defined for field 'x', this is not allowed."
    with pytest.raises(TypeError, match=msg):
        class Model(Parent):
            @field_serializer('x')
            def ser_x_other(self, _v: Any) -> str:
                return 'err'
    with pytest.raises(TypeError, match=msg):
        class Model(BaseModel):
            x: int
            @field_serializer('x')
            def ser_x(self, _v: Any) -> str:
                return 'parent_encoder'
            @field_serializer('x')
            def other_func_name(self, _v: Any) -> str:
                return 'parent_encoder'

def test_serializer_allow_reuse_same_field() -> None:
    with pytest.warns(UserWarning, match='`ser_x` overrides an existing Pydantic `@field_serializer` decorator'):
        class Model(BaseModel):
            x: int
            @field_serializer('x')
            def ser_x(self, _v: Any) -> str:
                return 'ser_1'
            @field_serializer('x')
            def ser_x(self, _v: Any) -> str:
                return 'ser_2'
        m = Model(x=1)
        assert m.model_dump() == {'x': 'ser_2'}

def test_serializer_allow_reuse_different_field_1() -> None:
    with pytest.warns(UserWarning, match='`ser` overrides an existing Pydantic `@field_serializer` decorator'):
        class Model(BaseModel):
            x: int
            y: int
            @field_serializer('x')
            def ser(self, _v: Any) -> Any:
                return 'x'
            @field_serializer('y')
            def ser(self, _v: Any) -> Any:
                return 'y'
    m = Model(x=1, y=2)
    assert m.model_dump() == {'x': 1, 'y': 'y'}

def test_serializer_allow_reuse_different_field_2() -> None:
    with pytest.warns(UserWarning, match='`ser_x` overrides an existing Pydantic `@field_serializer` decorator'):
        def ser(self, _v: Any, _info: FieldSerializationInfo) -> str:
            return 'ser'
        class Model(BaseModel):
            x: int
            y: int
            @field_serializer('x')
            def ser_x(self, _v: Any) -> str:
                return 'ser_x'
            ser_x = field_serializer('y')(ser)
    m = Model(x=1, y=2)
    assert m.model_dump() == {'x': 1, 'y': 'ser'}

def test_serializer_allow_reuse_different_field_3() -> None:
    with pytest.warns(UserWarning, match='`ser_x` overrides an existing Pydantic `@field_serializer` decorator'):
        def ser1(self, _v: Any, _info: FieldSerializationInfo) -> str:
            return 'ser1'
        def ser2(self, _v: Any, _info: FieldSerializationInfo) -> str:
            return 'ser2'
        class Model(BaseModel):
            x: int
            y: int
            ser_x = field_serializer('x')(ser1)
            ser_x = field_serializer('y')(ser2)
    m = Model(x=1, y=2)
    assert m.model_dump() == {'x': 1, 'y': 'ser2'}

def test_serializer_allow_reuse_different_field_4() -> None:
    def ser(self, _v: Any, _info: FieldSerializationInfo) -> str:
        return f'{_v:,}'
    class Model(BaseModel):
        x: int
        y: int
        ser_x = field_serializer('x')(ser)
        not_ser_x = field_serializer('y')(ser)
    m = Model(x=1000, y=2000)
    assert m.model_dump() == {'x': '1,000', 'y': '2,000'}

def test_serialize_any_model() -> None:
    class Model(BaseModel):
        m: str

        @field_serializer('m')
        def ser_m(self, v: str, _info: FieldSerializationInfo) -> str:
            return f'custom:{v}'

    class AnyModel(BaseModel):
        x: Any
    m = Model(m='test')
    assert m.model_dump() == {'m': 'custom:test'}
    any_instance = AnyModel(x=m)
    assert to_jsonable_python(any_instance) == {'x': {'m': 'custom:test'}}
    assert any_instance.model_dump() == {'x': {'m': 'custom:test'}}

def test_invalid_field() -> None:
    msg = (r"Decorators defined with incorrect fields: tests\.test_serialize\.test_invalid_field\.<locals>\.Model:\d+\.customise_b_serialization "
           r"\(use check_fields=False if you're inheriting from the model and intended this\)")
    with pytest.raises(errors.PydanticUserError, match=msg):
        class Model(BaseModel):
            b: Any
            @field_serializer('b')
            def customise_b_serialization(v: Any) -> Any:
                return v

def test_serialize_with_extra() -> None:
    class Inner(BaseModel):
        a: str = 'a'

    class Outer(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')
        inner: Inner = Field(default_factory=Inner)
    m = Outer.model_validate({})
    assert m.model_dump() == {'inner': {'a': 'a'}}

def test_model_serializer_nested_models() -> None:
    class Model(BaseModel):
        x: int
        inner: Optional[Any]

        @model_serializer(mode='wrap')
        def ser_model(self, handler: Callable[[Any], Dict[str, Any]]) -> Dict[str, Any]:
            inner_data = handler(self)
            inner_data['x'] += 1
            return inner_data
    m1 = Model(x=0, inner=None)
    assert m1.model_dump() == {'x': 1, 'inner': None}
    m2 = Model(x=2, inner=Model(x=1, inner=Model(x=0, inner=None)))
    assert m2.model_dump() == {'x': 3, 'inner': {'x': 2, 'inner': {'x': 1, 'inner': None}}}

def test_pattern_serialize() -> None:
    ta = TypeAdapter(Pattern[str])
    pattern: Pattern[str] = re.compile('^regex$')
    assert ta.dump_python(pattern) == pattern
    assert ta.dump_python(pattern, mode='json') == '^regex$'
    assert ta.dump_json(pattern) == b'"^regex$"'

def test_custom_return_schema() -> None:
    class Model(BaseModel):
        x: Any

        @field_serializer('x', return_type=str)
        def ser_model(self, v: Any) -> str:
            return repr(v)
    return_serializer = re.search(r'return_serializer: *\w+', repr(Model.__pydantic_serializer__))
    assert return_serializer is not None
    assert return_serializer.group(0) == 'return_serializer: Str'

def test_clear_return_schema() -> None:
    class Model(BaseModel):
        x: Any

        @field_serializer('x', return_type=Any)
        def ser_model(self, v: Any) -> str:
            return repr(v)
    return_serializer = re.search(r'return_serializer: *\w+', repr(Model.__pydantic_serializer__))
    assert return_serializer is not None
    assert return_serializer.group(0) == 'return_serializer: Any'

def test_serializer_return_type_model() -> None:
    """https://github.com/pydantic/pydantic/issues/10443"""
    class Sub(BaseModel):
        pass

    class Model(BaseModel):
        sub: Sub
    assert Model(sub=Sub()).model_dump() == {'sub': {}}

def test_type_adapter_dump_json() -> None:
    class Model(TypedDict):
        x: int
        y: float

        @model_serializer(mode='plain')
        def ser_model(self) -> Dict[str, Union[int, float]]:
            return {'x': self['x'] * 2, 'y': self['y'] * 3}
    ta = TypeAdapter(Model)
    instance = Model({'x': 1, 'y': 2.5})
    assert ta.dump_json(instance) == b'{"x":2,"y":7.5}'

def test_type_adapter_dump_with_context() -> None:
    class Model(TypedDict):
        x: int
        y: float

        @model_serializer(mode='wrap')
        def _serialize(self, handler: Callable[[Any], Dict[str, Any]], info: SerializationInfo) -> Dict[str, Any]:
            data = handler(self)
            if info.context and info.context.get('mode') == 'x-only':
                data.pop('y')
            return data
    ta = TypeAdapter(Model)
    instance = Model({'x': 1, 'y': 2.5})
    assert ta.dump_json(instance, context={'mode': 'x-only'}) == b'{"x":1}'

@pytest.mark.parametrize('as_annotation', [True, False])
@pytest.mark.parametrize('mode', ['plain', 'wrap'])
def test_forward_ref_for_serializers(as_annotation: bool, mode: Literal['plain', 'wrap']) -> None:
    if mode == 'plain':
        def ser_model_func(v: int) -> Any:
            return OtherModel(y=v + 1)

        def ser_model_method(self, v: int) -> Any:
            return ser_model_func(v)
        annotation = PlainSerializer(ser_model_func)
    else:
        def ser_model_func(v: int, handler: Callable[[int], Any]) -> Any:
            return OtherModel(y=v + 1)

        def ser_model_method(self, v: int, handler: Callable[[int], Any]) -> Any:
            return ser_model_func(v, handler)
        annotation = WrapSerializer(ser_model_func)

    class Model(BaseModel):
        x: int
        if as_annotation:
            __annotations__['x'] = Annotated[int, annotation]  # type: ignore
        else:
            pass
            # if not as_annotation, assign serializer via decorator below
            # Will assign after class definition
    if not as_annotation:
        setattr(Model, 'ser_model', field_serializer('x', mode=mode)(ser_model_method))
    class OtherModel(BaseModel):
        y: int
    Model.model_rebuild(_types_namespace={'SomeOtherModel': OtherModel})
    m = Model(x=1)
    assert m.model_dump() == {'x': {'y': 2}}
    schema = Model.model_json_schema(mode='serialization')
    assert schema == {
        '$defs': {
            'OtherModel': {
                'properties': {'y': {'title': 'Y', 'type': 'integer'}},
                'required': ['y'],
                'title': 'OtherModel',
                'type': 'object'
            }
        },
        'properties': {'x': {'$ref': '#/$defs/OtherModel', 'title': 'X'}},
        'required': ['x'],
        'title': 'Model',
        'type': 'object'
    }

def test_forward_ref_for_computed_fields() -> None:
    class Model(BaseModel):
        x: int

        @computed_field
        @property
        def two_x(self) -> int:
            return self.x * 2
    Model.model_rebuild(_types_namespace={'IntAlias': int})
    expected_schema = {
        'properties': {
            'two_x': {'readOnly': True, 'title': 'Two X', 'type': 'integer'},
            'x': {'title': 'X', 'type': 'integer'}
        },
        'required': ['x', 'two_x'],
        'title': 'Model',
        'type': 'object'
    }
    assert Model.model_json_schema(mode='serialization') == expected_schema
    m = Model(x=1)
    assert m.model_dump() == {'two_x': 2, 'x': 1}

def test_computed_field_custom_serializer() -> None:
    class Model(BaseModel):
        x: int

        @computed_field
        @property
        def two_x(self) -> int:
            return self.x * 2

        @field_serializer('two_x', when_used='json')
        def ser_two_x(self, v: int) -> str:
            return f'The double of x is {v}'
    m = Model(x=1)
    assert m.model_dump() == {'two_x': 2, 'x': 1}
    assert json.loads(m.model_dump_json()) == {'two_x': 'The double of x is 2', 'x': 1}

def test_annotated_computed_field_custom_serializer() -> None:
    class Model(BaseModel):
        x: int

        @computed_field
        @property
        def two_x(self) -> int:
            return self.x * 2

        @computed_field
        @property
        def triple_x(self) -> int:
            return self.two_x * 3

        @computed_field
        @property
        def quadruple_x_plus_one(self) -> int:
            return self.two_x * 2 + 1
    m = Model(x=1)
    assert m.x == 1
    assert m.two_x == 2
    assert m.triple_x == 6
    assert m.quadruple_x_plus_one == 5
    dumped = m.model_dump()
    assert dumped == {'x': 1, 'two_x': 'The double of x is 2', 'triple_x': 'The triple of x is 6', 'quadruple_x_plus_one': 5}
    json_dumped = json.loads(m.model_dump_json())
    assert json_dumped == {'x': 1, 'two_x': 'The double of x is 2', 'triple_x': 'The triple of x is 6', 'quadruple_x_plus_one': 5}
    expected_schema = {
        'properties': {
            'x': {'title': 'X', 'type': 'integer'},
            'two_x': {'readOnly': True, 'title': 'Two X', 'type': 'string'},
            'triple_x': {'readOnly': True, 'title': 'Triple X', 'type': 'string'},
            'quadruple_x_plus_one': {'readOnly': True, 'title': 'Quadruple X Plus One', 'type': 'integer'}
        },
        'required': ['x', 'two_x', 'triple_x', 'quadruple_x_plus_one'],
        'title': 'Model',
        'type': 'object'
    }
    assert Model.model_json_schema(mode='serialization') == expected_schema

@pytest.mark.skipif(sys.version_info >= (3, 13), reason='@computed_field @classmethod @property only works in 3.9-3.12')
def test_forward_ref_for_classmethod_computed_fields() -> None:
    class Model(BaseModel):
        y: int = 4

        @computed_field
        @classmethod
        @property
        def two_y(cls) -> int:
            return cls.y * 2
    Model.model_rebuild(_types_namespace={'IntAlias': int})
    expected_schema = {
        'properties': {
            'two_y': {'readOnly': True, 'title': 'Two Y', 'type': 'integer'}
        },
        'required': ['two_y'],
        'title': 'Model',
        'type': 'object'
    }
    assert Model.model_json_schema(mode='serialization') == expected_schema
    assert Model().model_dump() == {'two_y': 8}

def test_enum_as_dict_key() -> None:
    class MyEnum(Enum):
        A = 'a'
        B = 'b'

    class MyModel(BaseModel):
        foo: Dict[MyEnum, str]
        bar: MyEnum
    m = MyModel(foo={MyEnum.A: 'hello'}, bar=MyEnum.B)
    assert m.model_dump_json() == '{"foo":{"a":"hello"},"bar":"b"}'

def test_subclass_support_unions() -> None:
    class Pet(BaseModel):
        name: str

    class Dog(Pet):
        breed: str

    class Kid(BaseModel):
        name: str

    class Home(BaseModel):
        little_guys: list[Pet]

    class Shelter(BaseModel):
        pets: list[Dog]
    h1 = Home(little_guys=[Pet(name='spot'), Pet(name='buddy')])
    assert h1.model_dump() == {'little_guys': [{'name': 'spot'}, {'name': 'buddy'}]}
    h2 = Home(little_guys=[Dog(name='fluffy', breed='lab'), Dog(name='patches', breed='boxer')])
    assert h2.model_dump() == {'little_guys': [{'name': 'fluffy'}, {'name': 'patches'}]}
    s = Shelter(pets=[Dog(name='fluffy', breed='lab'), Dog(name='patches', breed='boxer')])
    assert s.model_dump() == {'pets': [{'name': 'fluffy'}, {'name': 'patches'}]}

def test_subclass_support_unions_with_forward_ref() -> None:
    class Bar(BaseModel):
        bar_id: int

    class Baz(Bar):
        baz_id: int

    class Foo(BaseModel):
        items: list[Union[Baz, 'Foo']]  # forward reference
    Foo.model_rebuild()
    foo = Foo(items=[Baz(bar_id=1, baz_id=2), Baz(bar_id=3, baz_id=4)])
    assert foo.model_dump() == {'items': [{'bar_id': 1}, {'bar_id': 3}]}
    foo_recursive = Foo(items=[Foo(items=[Baz(bar_id=42, baz_id=99)])])
    assert foo_recursive.model_dump() == {'items': [{'items': [{'bar_id': 42}]}]}

def test_serialize_python_context() -> None:
    contexts = [None, None, {'foo': 'bar'}]

    class Model(BaseModel):
        x: int

        @field_serializer('x')
        def serialize_x(self, v: int, info: FieldSerializationInfo) -> int:
            assert info.context == contexts.pop(0)
            return v
    m = Model.model_construct(**{'x': 1})
    m.model_dump()
    m.model_dump(context=None)
    m.model_dump(context={'foo': 'bar'})
    assert contexts == []

def test_serialize_json_context() -> None:
    contexts = [None, None, {'foo': 'bar'}]

    class Model(BaseModel):
        x: int

        @field_serializer('x')
        def serialize_x(self, v: int, info: FieldSerializationInfo) -> int:
            assert info.context == contexts.pop(0)
            return v
    m = Model.model_construct(**{'x': 1})
    m.model_dump_json()
    m.model_dump_json(context=None)
    m.model_dump_json(context={'foo': 'bar'})
    assert contexts == []

def test_plain_serializer_with_std_type() -> None:
    """Ensure that a plain serializer can be used with a standard type constructor, rather than having to use lambda x: std_type(x)."""
    class MyModel(BaseModel):
        x: float

        @field_serializer('x', return_type=float)
        def ser_x(self, v: Any) -> float:
            return float(v)
    m = MyModel(x=1)
    assert m.model_dump() == {'x': 1.0}
    assert m.model_dump_json() == '{"x":1.0}'
    assert m.model_json_schema(mode='serialization') == {
        'properties': {'x': {'title': 'X', 'type': 'number'}},
        'required': ['x'],
        'title': 'MyModel',
        'type': 'object'
    }

@pytest.mark.xfail(reason='Waiting for union serialization fixes via https://github.com/pydantic/pydantic/issues/9688.')
def smart_union_serialization() -> None:
    """Initially reported via https://github.com/pydantic/pydantic/issues/9417, effectively a round tripping problem with type consistency."""
    class FloatThenInt(BaseModel):
        value: int = Field(union_mode='smart')

    class IntThenFloat(BaseModel):
        value: int = Field(union_mode='smart')
    float_then_int = FloatThenInt(value=100)
    assert type(json.loads(float_then_int.model_dump_json())['value']) is int
    int_then_float = IntThenFloat(value=100)
    assert type(json.loads(int_then_float.model_dump_json())['value']) is int

def test_serialize_with_custom_ser() -> None:
    class Item(BaseModel):
        id: int

        @model_serializer
        def dump(self) -> Dict[str, int]:
            return {'id': self.id}

    class ItemContainer(BaseModel):
        item_or_items: Any
    items = [Item(id=i) for i in range(5)]
    container = ItemContainer(item_or_items=items)
    assert container.model_dump_json() == '{"item_or_items":[{"id":0},{"id":1},{"id":2},{"id":3},{"id":4}]}'

def test_field_serializers_use_enum_ref() -> None:
    """See https://github.com/pydantic/pydantic/issues/9394 for the original issue."""
    class MyEnum(Enum):
        A = 'a'
        B = 'b'

    class MyModel(BaseModel):
        @computed_field
        @property
        def computed_a_or_b(self) -> MyEnum:
            return MyEnum.B

        @field_serializer('computed_a_or_b')
        def serialize_my_enum(self, a_or_b: MyEnum) -> str:
            return a_or_b.value
    m = MyModel()
    assert m.model_dump()['computed_a_or_b'] == 'b'

def test_serialization_fallback() -> None:
    class Arbitrary:
        value: int = 1

    def fallback(v: Any) -> Any:
        if isinstance(v, Arbitrary):
            return v.value
    ta = TypeAdapter(Any)
    assert ta.dump_python(Arbitrary(), fallback=fallback) == 1
