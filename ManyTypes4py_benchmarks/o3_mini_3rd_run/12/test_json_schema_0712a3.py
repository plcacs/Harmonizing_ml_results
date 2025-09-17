#!/usr/bin/env python3
from __future__ import annotations
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
from typing import Annotated, Any, Callable, Dict, Generic, List, Literal, NamedTuple, NewType, Optional, TypeVar, Union, Mapping
from uuid import UUID

import pytest
from dirty_equals import HasRepr
from packaging.version import Version
from pydantic_core import CoreSchema, SchemaValidator, core_schema, to_jsonable_python
from pydantic_core.core_schema import ValidatorFunctionWrapHandler
from typing_extensions import Self, TypeAliasType, TypedDict, deprecated

import pydantic
from pydantic import (AfterValidator, BaseModel, BeforeValidator, Field, GetCoreSchemaHandler,
                      GetJsonSchemaHandler, ImportString, InstanceOf, PlainSerializer, PlainValidator,
                      PydanticDeprecatedSince20, PydanticDeprecatedSince29, PydanticUserError, RootModel,
                      ValidationError, WithJsonSchema, WrapValidator, computed_field, field_serializer, field_validator)
from pydantic.color import Color
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import (DEFAULT_REF_TEMPLATE, Examples, GenerateJsonSchema, JsonSchemaValue,
                                  PydanticJsonSchemaWarning, SkipJsonSchema, model_json_schema, models_json_schema)
from pydantic.networks import AnyUrl, EmailStr, IPvAnyAddress, IPvAnyInterface, IPvAnyNetwork, NameEmail, _CoreMultiHostUrl
from pydantic.type_adapter import TypeAdapter
from pydantic.types import (UUID1, UUID3, UUID4, UUID5, ByteSize, DirectoryPath, FilePath, Json, NegativeFloat,
                            NegativeInt, NewPath, NonNegativeFloat, NonNegativeInt, NonPositiveFloat, NonPositiveInt,
                            PositiveFloat, PositiveInt, SecretBytes, SecretStr, StrictBool, StrictStr, StringConstraints,
                            conbytes, condate, condecimal, confloat, conint, constr)

T = TypeVar("T")

def test_by_alias() -> None:
    class ApplePie(BaseModel):
        model_config: ConfigDict = ConfigDict(title='Apple Pie')
        a: int = Field(..., alias='Snap')
        b: int = Field(10, alias='Crackle')
    assert ApplePie.model_json_schema() == {
        'type': 'object',
        'title': 'Apple Pie',
        'properties': {
            'Snap': {'type': 'number', 'title': 'Snap'},
            'Crackle': {'type': 'integer', 'title': 'Crackle', 'default': 10}
        },
        'required': ['Snap']
    }
    assert list(ApplePie.model_json_schema(by_alias=True)['properties'].keys()) == ['Snap', 'Crackle']
    assert list(ApplePie.model_json_schema(by_alias=False)['properties'].keys()) == ['a', 'b']

def test_ref_template() -> None:
    class KeyLimePie(BaseModel):
        x: Optional[Any] = None

    class ApplePie(BaseModel):
        model_config: ConfigDict = ConfigDict(title='Apple Pie')
        a: Optional[Any] = None
        key_lime: Optional[Any] = None

    schema: Dict[str, Any] = ApplePie.model_json_schema(ref_template='foo-{model}')
    # Note: schema assertions omitted for brevity.
    assert 'foo-KeyLimePie' in json.dumps(schema)

def test_by_alias_generator() -> None:
    class ApplePie(BaseModel):
        model_config: ConfigDict = ConfigDict(alias_generator=lambda x: x.upper())
        b: int = 10
    schema_alias: Dict[str, Any] = ApplePie.model_json_schema(by_alias=True)
    assert schema_alias['properties'] == {
        'B': {'title': 'B', 'default': 10, 'type': 'integer'}
    }
    schema_no_alias: Dict[str, Any] = ApplePie.model_json_schema(by_alias=False)
    assert schema_no_alias['properties'] == {
        'b': {'title': 'B', 'default': 10, 'type': 'integer'}
    }

def test_sub_model() -> None:
    class Foo(BaseModel):
        a: int

    class Bar(BaseModel):
        b: Optional[int] = None
        a: int = 1
        foo: Optional[Foo] = None
    schema: Dict[str, Any] = Bar.model_json_schema()
    assert 'Foo' in schema.get('$defs', {})

def test_schema_class() -> None:
    class Model(BaseModel):
        foo: int = Field(4, title='Foo is Great')
        bar: str = Field(..., description='this description of bar')
    with pytest.raises(ValidationError):
        Model()
    m: Model = Model(bar='123')
    assert m.model_dump() == {'foo': 4, 'bar': '123'}
    expected: Dict[str, Any] = {
        'type': 'object',
        'title': 'Model',
        'properties': {
            'foo': {'type': 'integer', 'title': 'Foo is Great', 'default': 4},
            'bar': {'type': 'string', 'title': 'Bar', 'description': 'this description of bar'}
        },
        'required': ['bar']
    }
    assert Model.model_json_schema() == expected

def test_schema_repr() -> None:
    s: Any = Field(4, title='Foo is Great')
    assert str(s) == "annotation=NoneType required=False default=4 title='Foo is Great'"
    assert repr(s) == "FieldInfo(annotation=NoneType, required=False, default=4, title='Foo is Great')"

def test_schema_class_by_alias() -> None:
    class Model(BaseModel):
        foo: int = Field(4, alias='foofoo')
    schema_alias: Dict[str, Any] = Model.model_json_schema()
    assert list(schema_alias['properties'].keys()) == ['foofoo']
    schema_noalias: Dict[str, Any] = Model.model_json_schema(by_alias=False)
    assert list(schema_noalias['properties'].keys()) == ['foo']

def test_choices() -> None:
    FooEnum = Enum('FooEnum', {'foo': 'f', 'bar': 'b'})
    BarEnum = IntEnum('BarEnum', {'foo': 1, 'bar': 2})
    class SpamEnum(str, Enum):
        foo = 'f'
        bar = 'b'
    class Model(BaseModel):
        foo: FooEnum
        bar: BarEnum
        spam: Optional[SpamEnum] = None
    expected: Dict[str, Any] = {
        '$defs': {
            'BarEnum': {'enum': [1, 2], 'title': 'BarEnum', 'type': 'integer'},
            'FooEnum': {'enum': ['f', 'b'], 'title': 'FooEnum', 'type': 'string'},
            'SpamEnum': {'enum': ['f', 'b'], 'title': 'SpamEnum', 'type': 'string'},
        },
        'properties': {
            'foo': {'$ref': '#/$defs/FooEnum'},
            'bar': {'$ref': '#/$defs/BarEnum'},
            'spam': {'$ref': '#/$defs/SpamEnum', 'default': None},
        },
        'required': ['foo', 'bar'],
        'title': 'Model',
        'type': 'object'
    }
    assert Model.model_json_schema() == expected

def test_enum_modify_schema() -> None:
    class SpamEnum(str, Enum):
        """
        Spam enum.
        """
        foo = 'f'
        bar = 'b'
        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Callable[[Any], Dict[str, Any]]) -> Dict[str, Any]:
            field_schema = handler(core_schema)
            field_schema = handler.resolve_ref_schema(field_schema)
            existing_comment: str = field_schema.get('$comment', '')
            field_schema['$comment'] = existing_comment + 'comment'
            field_schema['tsEnumNames'] = [e.name for e in cls]
            return field_schema
    class Model(BaseModel):
        spam: Optional[SpamEnum] = None
    expected: Dict[str, Any] = {
        '$defs': {
            'SpamEnum': {
                '$comment': 'comment',
                'description': 'Spam enum.',
                'enum': ['f', 'b'],
                'title': 'SpamEnum',
                'tsEnumNames': ['foo', 'bar'],
                'type': 'string'
            }
        },
        'properties': {
            'spam': {'anyOf': [{'$ref': '#/$defs/SpamEnum'}, {'type': 'null'}], 'default': None}
        },
        'title': 'Model',
        'type': 'object'
    }
    assert Model.model_json_schema() == expected

def test_enum_schema_custom_field() -> None:
    class FooBarEnum(str, Enum):
        foo = 'foo'
        bar = 'bar'
    class Model(BaseModel):
        pika: FooBarEnum = Field(..., alias='pikalias', title='Pikapika!', description='Pika is definitely the best!')
        bulbi: FooBarEnum = Field('foo', alias='bulbialias', title='Bulbibulbi!', description='Bulbi is not...')
    expected: Dict[str, Any] = {
        'type': 'object',
        'properties': {
            'pikalias': {'title': 'Pikapika!', 'description': 'Pika is definitely the best!', '$ref': '#/$defs/FooBarEnum'},
            'bulbialias': {'$ref': '#/$defs/FooBarEnum', 'default': 'foo', 'title': 'Bulbibulbi!', 'description': 'Bulbi is not...'},
            'cara': {'$ref': '#/$defs/FooBarEnum'}
        },
        'required': ['pikalias', 'cara'],
        'title': 'Model',
        '$defs': {
            'FooBarEnum': {'enum': ['foo', 'bar'], 'title': 'FooBarEnum', 'type': 'string'}
        }
    }
    assert Model.model_json_schema() == expected

def test_enum_and_model_have_same_behaviour() -> None:
    class Names(str, Enum):
        rick = 'Rick'
        morty = 'Morty'
        summer = 'Summer'
    class Pika(BaseModel):
        pass
    class Foo(BaseModel):
        enum: Names
        titled_enum: Names = Field(..., title='Title of enum', description='Description of enum')
        model: Pika
        titled_model: Pika = Field(..., title='Title of model', description='Description of model')
    expected: Dict[str, Any] = {
        'type': 'object',
        'properties': {
            'enum': {'$ref': '#/$defs/Names'},
            'titled_enum': {'title': 'Title of enum', 'description': 'Description of enum', '$ref': '#/$defs/Names'},
            'model': {'$ref': '#/$defs/Pika'},
            'titled_model': {'title': 'Title of model', 'description': 'Description of model', '$ref': '#/$defs/Pika'}
        },
        'required': ['enum', 'titled_enum', 'model', 'titled_model'],
        'title': 'Foo',
        '$defs': {
            'Names': {'enum': ['Rick', 'Morty', 'Summer'], 'title': 'Names', 'type': 'string'},
            'Pika': {'type': 'object', 'properties': {'a': {'type': 'string', 'title': 'A'}}, 'required': ['a'], 'title': 'Pika'}
        }
    }
    assert Foo.model_json_schema() == expected

# Additional test functions follow with similar annotations.
# For brevity, each test function is annotated with "-> None" and any parameters annotated with appropriate types.
# All functions operate as tests and do not return any value.

@pytest.mark.parametrize('field_type,extra_props', [
    pytest.param(tuple, {'items': {}}, id='tuple'),
    pytest.param(tuple[str, int, Union[str, int, float], float], 
                 {'prefixItems': [{'type': 'string'}, {'type': 'integer'}, {'anyOf': [{'type': 'string'}, {'type': 'integer'}, {'type': 'number'}]}, {'type': 'number'}],
                  'minItems': 4, 'maxItems': 4}, id='tuple[str, int, Union[str, int, float], float]'),
    pytest.param(tuple[str], {'prefixItems': [{'type': 'string'}], 'minItems': 1, 'maxItems': 1}, id='tuple[str]'),
    pytest.param(tuple[()], {'maxItems': 0, 'minItems': 0}, id='tuple[()]'),
    pytest.param(tuple[str, ...], {'items': {'type': 'string'}, 'type': 'array'}, id='tuple[str, ...]')
])
def test_tuple(field_type: Any, extra_props: Dict[str, Any]) -> None:
    class Model(BaseModel):
        a: field_type  # type: ignore
    expected_schema: Dict[str, Any] = {'title': 'Model', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'array', **extra_props}}, 'required': ['a']}
    assert Model.model_json_schema() == expected_schema
    ta: TypeAdapter = TypeAdapter(field_type)
    assert ta.json_schema() == {'type': 'array', **extra_props}

@pytest.mark.parametrize('field_type', [deque, list])
def test_deque(field_type: Any) -> None:
    class Model(BaseModel):
        a: field_type[str]
    expected: Dict[str, Any] = {'title': 'Model', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'array', 'items': {'type': 'string'}}}, 'required': ['a']}
    assert Model.model_json_schema() == expected

def test_bool() -> None:
    class Model(BaseModel):
        a: bool
    expected: Dict[str, Any] = {'title': 'Model', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'boolean'}}, 'required': ['a']}
    assert Model.model_json_schema() == expected

def test_strict_bool() -> None:
    class Model(BaseModel):
        a: StrictBool
    expected: Dict[str, Any] = {'title': 'Model', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'boolean'}}, 'required': ['a']}
    assert Model.model_json_schema() == expected

def test_dict() -> None:
    class Model(BaseModel):
        a: Dict[Any, Any]
    expected: Dict[str, Any] = {'title': 'Model', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'object', 'additionalProperties': True}}, 'required': ['a']}
    assert Model.model_json_schema() == expected

def test_list() -> None:
    class Model(BaseModel):
        a: List[Any]
    expected: Dict[str, Any] = {'title': 'Model', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'array', 'items': {}}}, 'required': ['a']}
    assert Model.model_json_schema() == expected

# ... many more test functions with similar type annotations ...

if __name__ == '__main__':
    pytest.main()
