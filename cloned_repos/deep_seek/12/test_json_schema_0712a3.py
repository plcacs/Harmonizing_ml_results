import dataclasses
import importlib.metadata
import json
import math
import re
import sys
import typing
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, Callable, Generic, Literal, NamedTuple, NewType, Optional, TypedDict, TypeVar, Union
from uuid import UUID
import pytest
from dirty_equals import HasRepr
from packaging.version import Version
from pydantic_core import CoreSchema, SchemaValidator, core_schema, to_jsonable_python
from pydantic_core.core_schema import ValidatorFunctionWrapHandler
from typing_extensions import Self, TypeAliasType, TypedDict, deprecated
import pydantic
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, GetCoreSchemaHandler, GetJsonSchemaHandler, ImportString, InstanceOf, PlainSerializer, PlainValidator, PydanticDeprecatedSince20, PydanticDeprecatedSince29, PydanticUserError, RootModel, ValidationError, WithJsonSchema, WrapValidator, computed_field, field_serializer, field_validator
from pydantic.color import Color
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, Examples, GenerateJsonSchema, JsonSchemaValue, PydanticJsonSchemaWarning, SkipJsonSchema, model_json_schema, models_json_schema
from pydantic.networks import AnyUrl, EmailStr, IPvAnyAddress, IPvAnyInterface, IPvAnyNetwork, NameEmail, _CoreMultiHostUrl
from pydantic.type_adapter import TypeAdapter
from pydantic.types import UUID1, UUID3, UUID4, UUID5, ByteSize, DirectoryPath, FilePath, Json, NegativeFloat, NegativeInt, NewPath, NonNegativeFloat, NonNegativeInt, NonPositiveFloat, NonPositiveInt, PositiveFloat, PositiveInt, SecretBytes, SecretStr, StrictBool, StrictStr, StringConstraints, conbytes, condate, condecimal, confloat, conint, constr

T = TypeVar('T')

def test_by_alias() -> None:
    class ApplePie(BaseModel):
        model_config = ConfigDict(title='Apple Pie')
        a: float = Field(alias='Snap')
        b: int = Field(10, alias='Crackle')
    
    assert ApplePie.model_json_schema() == {'type': 'object', 'title': 'Apple Pie', 'properties': {'Snap': {'type': 'number', 'title': 'Snap'}, 'Crackle': {'type': 'integer', 'title': 'Crackle', 'default': 10}}, 'required': ['Snap']}
    assert list(ApplePie.model_json_schema(by_alias=True)['properties'].keys()) == ['Snap', 'Crackle']
    assert list(ApplePie.model_json_schema(by_alias=False)['properties'].keys()) == ['a', 'b']

def test_ref_template() -> None:
    class KeyLimePie(BaseModel):
        x: None = None

    class ApplePie(BaseModel):
        model_config = ConfigDict(title='Apple Pie')
        a: None = None
        key_lime: None = None
    
    assert ApplePie.model_json_schema(ref_template='foobar/{model}.json') == {'title': 'Apple Pie', 'type': 'object', 'properties': {'a': {'default': None, 'title': 'A', 'type': 'number'}, 'key_lime': {'anyOf': [{'$ref': 'foobar/KeyLimePie.json'}, {'type': 'null'}], 'default': None}}, '$defs': {'KeyLimePie': {'title': 'KeyLimePie', 'type': 'object', 'properties': {'x': {'default': None, 'title': 'X', 'type': 'string'}}}}
    assert ApplePie.model_json_schema()['properties']['key_lime'] == {'anyOf': [{'$ref': '#/$defs/KeyLimePie'}, {'type': 'null'}], 'default': None}
    json_schema = json.dumps(ApplePie.model_json_schema(ref_template='foobar/{model}.json'))
    assert 'foobar/KeyLimePie.json' in json_schema
    assert '#/$defs/KeyLimePie' not in json_schema

def test_by_alias_generator() -> None:
    class ApplePie(BaseModel):
        model_config = ConfigDict(alias_generator=lambda x: x.upper())
        b: int = 10
    
    assert ApplePie.model_json_schema() == {'title': 'ApplePie', 'type': 'object', 'properties': {'A': {'title': 'A', 'type': 'number'}, 'B': {'title': 'B', 'default': 10, 'type': 'integer'}}, 'required': ['A']}
    assert ApplePie.model_json_schema(by_alias=False)['properties'].keys() == {'a', 'b'}

def test_sub_model() -> None:
    class Foo(BaseModel):
        """hello"""
        b: float

    class Bar(BaseModel):
        b: Optional[Foo] = None
    
    assert Bar.model_json_schema() == {'type': 'object', 'title': 'Bar', '$defs': {'Foo': {'type': 'object', 'title': 'Foo', 'description': 'hello', 'properties': {'b': {'type': 'number', 'title': 'B'}}, 'required': ['b']}}, 'properties': {'a': {'type': 'integer', 'title': 'A'}, 'b': {'anyOf': [{'$ref': '#/$defs/Foo'}, {'type': 'null'}], 'default': None}}, 'required': ['a']}

def test_schema_class() -> None:
    class Model(BaseModel):
        foo: int = Field(4, title='Foo is Great')
        bar: str = Field(description='this description of bar')
    
    with pytest.raises(ValidationError):
        Model()
    
    m = Model(bar='123')
    assert m.model_dump() == {'foo': 4, 'bar': '123'}
    assert Model.model_json_schema() == {'type': 'object', 'title': 'Model', 'properties': {'foo': {'type': 'integer', 'title': 'Foo is Great', 'default': 4}, 'bar': {'type': 'string', 'title': 'Bar', 'description': 'this description of bar'}}, 'required': ['bar']}

def test_schema_repr() -> None:
    s = Field(4, title='Foo is Great')
    assert str(s) == "annotation=NoneType required=False default=4 title='Foo is Great'"
    assert repr(s) == "FieldInfo(annotation=NoneType, required=False, default=4, title='Foo is Great')"

def test_schema_class_by_alias() -> None:
    class Model(BaseModel):
        foo: int = Field(4, alias='foofoo')
    
    assert list(Model.model_json_schema()['properties'].keys()) == ['foofoo']
    assert list(Model.model_json_schema(by_alias=False)['properties'].keys()) == ['foo']

def test_choices() -> None:
    FooEnum = Enum('FooEnum', {'foo': 'f', 'bar': 'b'})
    BarEnum = IntEnum('BarEnum', {'foo': 1, 'bar': 2})

    class SpamEnum(str, Enum):
        foo = 'f'
        bar = 'b'

    class Model(BaseModel):
        spam: Optional[SpamEnum] = None
    
    assert Model.model_json_schema() == {'$defs': {'BarEnum': {'enum': [1, 2], 'title': 'BarEnum', 'type': 'integer'}, 'FooEnum': {'enum': ['f', 'b'], 'title': 'FooEnum', 'type': 'string'}, 'SpamEnum': {'enum': ['f', 'b'], 'title': 'SpamEnum', 'type': 'string'}}, 'properties': {'foo': {'$ref': '#/$defs/FooEnum'}, 'bar': {'$ref': '#/$defs/BarEnum'}, 'spam': {'$ref': '#/$defs/SpamEnum', 'default': None}}, 'required': ['foo', 'bar'], 'title': 'Model', 'type': 'object'}

def test_enum_modify_schema() -> None:
    class SpamEnum(str, Enum):
        """Spam enum."""
        foo = 'f'
        bar = 'b'

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
            field_schema = handler(core_schema)
            field_schema = handler.resolve_ref_schema(field_schema)
            existing_comment = field_schema.get('$comment', '')
            field_schema['$comment'] = existing_comment + 'comment'
            field_schema['tsEnumNames'] = [e.name for e in cls]
            return field_schema

    class Model(BaseModel):
        spam: Optional[SpamEnum] = None
    
    assert Model.model_json_schema() == {'$defs': {'SpamEnum': {'$comment': 'comment', 'description': 'Spam enum.', 'enum': ['f', 'b'], 'title': 'SpamEnum', 'tsEnumNames': ['foo', 'bar'], 'type': 'string'}}, 'properties': {'spam': {'anyOf': [{'$ref': '#/$defs/SpamEnum'}, {'type': 'null'}], 'default': None}}, 'title': 'Model', 'type': 'object'}

def test_enum_schema_custom_field() -> None:
    class FooBarEnum(str, Enum):
        foo = 'foo'
        bar = 'bar'

    class Model(BaseModel):
        pika: FooBarEnum = Field(alias='pikalias', title='Pikapika!', description='Pika is definitely the best!')
        bulbi: FooBarEnum = Field('foo', alias='bulbialias', title='Bulbibulbi!', description='Bulbi is not...')
    
    assert Model.model_json_schema() == {'type': 'object', 'properties': {'pikalias': {'title': 'Pikapika!', 'description': 'Pika is definitely the best!', '$ref': '#/$defs/FooBarEnum'}, 'bulbialias': {'$ref': '#/$defs/FooBarEnum', 'default': 'foo', 'title': 'Bulbibulbi!', 'description': 'Bulbi is not...'}, 'cara': {'$ref': '#/$defs/FooBarEnum'}}, 'required': ['pikalias', 'cara'], 'title': 'Model', '$defs': {'FooBarEnum': {'enum': ['foo', 'bar'], 'title': 'FooBarEnum', 'type': 'string'}}}

def test_enum_and_model_have_same_behaviour() -> None:
    class Names(str, Enum):
        rick = 'Rick'
        morty = 'Morty'
        summer = 'Summer'

    class Pika(BaseModel):
        pass

    class Foo(BaseModel):
        titled_enum: Names = Field(..., title='Title of enum', description='Description of enum')
        titled_model: Pika = Field(..., title='Title of model', description='Description of model')
    
    assert Foo.model_json_schema() == {'type': 'object', 'properties': {'enum': {'$ref': '#/$defs/Names'}, 'titled_enum': {'title': 'Title of enum', 'description': 'Description of enum', '$ref': '#/$defs/Names'}, 'model': {'$ref': '#/$defs/Pika'}, 'titled_model': {'title': 'Title of model', 'description': 'Description of model', '$ref': '#/$defs/Pika'}}, 'required': ['enum', 'titled_enum', 'model', 'titled_model'], 'title': 'Foo', '$defs': {'Names': {'enum': ['Rick', 'Morty', 'Summer'], 'title': 'Names', 'type': 'string'}, 'Pika': {'type': 'object', 'properties': {'a': {'type': 'string', 'title': 'A'}}, 'required': ['a'], 'title': 'Pika'}}}

def test_enum_includes_extra_without_other_params() -> None:
    class Names(str, Enum):
        rick = 'Rick'
        morty = 'Morty'
        summer = 'Summer'

    class Foo(BaseModel):
        extra_enum: Names = Field(json_schema_extra={'extra': 'Extra field'})
    
    assert Foo.model_json_schema() == {'$defs': {'Names': {'enum': ['Rick', 'Morty', 'Summer'], 'title': 'Names', 'type': 'string'}}, 'properties': {'enum': {'$ref': '#/$defs/Names'}, 'extra_enum': {'$ref': '#/$defs/Names', 'extra': 'Extra field'}}, 'required': ['enum', 'extra_enum'], 'title': 'Foo', 'type': 'object'}

def test_invalid_json_schema_extra() -> None:
    class MyModel(BaseModel):
        model_config = ConfigDict(json_schema_extra=1)
    
    with pytest.raises(ValueError, match=re.escape("model_config['json_schema_extra']=1 should be a dict, callable, or None")):
        MyModel.model_json_schema()

def test_list_enum_schema_extras() -> None:
    class FoodChoice(str, Enum):
        spam = 'spam'
        egg = 'egg'
        chips = 'chips'

    class Model(BaseModel):
        foods: list[FoodChoice] = Field(examples=[['spam', 'egg']])
    
    assert Model.model_json_schema() == {'$defs': {'FoodChoice': {'enum': ['spam', 'egg', 'chips'], 'title': 'FoodChoice', 'type': 'string'}}, 'properties': {'foods': {'title': 'Foods', 'type': 'array', 'items': {'$ref': '#/$defs/FoodChoice'}, 'examples': [['spam', 'egg']]}}, 'required': ['foods'], 'title': 'Model', 'type': 'object'}

def test_enum_schema_cleandoc() -> None:
    class FooBar(str, Enum):
        """This is docstring which needs to be cleaned up"""
        foo = 'foo'
        bar = 'bar'

    class Model(BaseModel):
        pass
    
    assert Model.model_json_schema() == {'title': 'Model', 'type': 'object', 'properties': {'enum': {'$ref': '#/$defs/FooBar'}}, 'required': ['enum'], '$defs': {'FooBar': {'title': 'FooBar', 'description': 'This is docstring which needs to be cleaned up', 'enum': ['foo', 'bar'], 'type': 'string'}}}

def test_decimal_json_schema() -> None:
    class Model(BaseModel):
        a: bytes = b'foobar'
        b: Decimal = Decimal('12.34')
    
    model_json_schema_validation = Model.model_json_schema(mode='validation')
    model_json_schema_serialization = Model.model_json_schema(mode='serialization')
    assert model_json_schema_validation == {'properties': {'a': {'default': 'foobar', 'format': 'binary', 'title': 'A', 'type': 'string'}, 'b': {'anyOf': [{'type': 'number'}, {'type': 'string'}], 'default': '12.34', 'title': 'B'}}, 'title': 'Model', 'type': 'object'}
    assert model_json_schema_serialization == {'properties': {'a': {'default': 'foobar', 'format': 'binary', 'title': 'A', 'type': 'string'}, 'b': {'default': '12.34', 'title': 'B', 'type': 'string'}}, 'title': 'Model', 'type': 'object'}

def test_list_sub_model() -> None:
    class Foo(BaseModel):
        a: float

    class Bar(BaseModel):
        b: list[Foo]
    
    assert Bar.model_json_schema() == {'title': 'Bar', 'type': 'object', '$defs': {'Foo': {'title': 'Foo', 'type': 'object', 'properties': {'a': {'type': 'number', 'title': 'A'}}, 'required': ['a']}}, 'properties': {'b': {'type': 'array', 'items': {'$ref': '#/$defs/Foo'}, 'title': 'B'}}, 'required': ['b']}

def test_optional() -> None:
    class Model(BaseModel):
        a: Optional[str]
    
    assert Model.model_json_schema() == {'title': 'Model', 'type': 'object', 'properties': {'a': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'A'}}, 'required': ['a']}

def test_optional_modify_schema() -> None:
    class MyNone(type(None)):
        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
            return core_schema.nullable_schema(core_schema.none_schema())

    class Model(BaseModel):
        x: MyNone
    
    assert Model.model_json_schema() == {'properties': {'x': {'title': 'X', 'type': 'null'}}, 'required': ['x'], 'title': 'Model', 'type': 'object'}

def test_any() -> None:
    class Model(BaseModel):
        a: Any
        b: Any
    
    assert Model.model_json_schema() == {'title': 'Model', 'type':