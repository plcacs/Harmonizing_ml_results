import collections
import ipaddress
import itertools
import json
import math
import os
import platform
import re
import sys
import typing
import uuid
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum, IntEnum
from fractions import Fraction
from numbers import Number
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, Callable, Literal, NewType, Optional, TypeVar, Union
from uuid import UUID
import annotated_types
import dirty_equals
import pytest
from dirty_equals import HasRepr, IsFloatNan, IsOneOf, IsStr
from pydantic_core import CoreSchema, PydanticCustomError, SchemaError, core_schema
from typing_extensions import NotRequired, TypedDict, get_args
from pydantic import (
    UUID1, UUID3, UUID4, UUID5, AfterValidator, AllowInfNan, AwareDatetime, Base64Bytes, Base64Str, 
    Base64UrlBytes, Base64UrlStr, BaseModel, BeforeValidator, ByteSize, ConfigDict, DirectoryPath, 
    EmailStr, FailFast, Field, FilePath, FiniteFloat, FutureDate, FutureDatetime, GetCoreSchemaHandler, 
    GetPydanticSchema, ImportString, InstanceOf, Json, JsonValue, NaiveDatetime, NameEmail, 
    NegativeFloat, NegativeInt, NewPath, NonNegativeFloat, NonNegativeInt, NonPositiveFloat, 
    NonPositiveInt, OnErrorOmit, PastDate, PastDatetime, PlainSerializer, PositiveFloat, PositiveInt, 
    PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, Secret, SecretBytes, SecretStr, 
    SerializeAsAny, SkipValidation, SocketPath, Strict, StrictBool, StrictBytes, StrictFloat, 
    StrictInt, StrictStr, StringConstraints, Tag, TypeAdapter, ValidationError, conbytes, condate, 
    condecimal, confloat, confrozenset, conint, conlist, conset, constr, field_serializer, 
    field_validator, validate_call
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

try:
    import email_validator
except ImportError:
    email_validator = None

@pytest.fixture(scope='session', name='ConBytesModel')
def con_bytes_model_fixture() -> type[BaseModel]:
    class ConBytesModel(BaseModel):
        v: bytes = b'foobar'
    return ConBytesModel

def test_constrained_bytes_good(ConBytesModel: type[BaseModel]) -> None:
    m: BaseModel = ConBytesModel(v=b'short')
    assert m.v == b'short'

def test_constrained_bytes_default(ConBytesModel: type[BaseModel]) -> None:
    m: BaseModel = ConBytesModel()
    assert m.v == b'foobar'

def test_strict_raw_type() -> None:
    class Model(BaseModel):
        v: str
    assert Model(v='foo').v == 'foo'
    with pytest.raises(ValidationError, match='Input should be a valid string \\[type=string_type,'):
        Model(v=b'fo')

@pytest.mark.parametrize(('data', 'valid'), [(b'this is too long', False), ('⪶⓲⽷01'.encode(), False), (b'not long90', True), ('⪶⓲⽷0'.encode(), True)])
def test_constrained_bytes_too_long(ConBytesModel: type[BaseModel], data: bytes, valid: bool) -> None:
    if valid:
        assert ConBytesModel(v=data).model_dump() == {'v': data}
    else:
        with pytest.raises(ValidationError) as exc_info:
            ConBytesModel(v=data)
        assert exc_info.value.errors(include_url=False) == [{'ctx': {'max_length': 10}, 'input': data, 'loc': ('v',), 'msg': 'Data should have at most 10 bytes', 'type': 'bytes_too_long'}]

def test_constrained_bytes_strict_true() -> None:
    class Model(BaseModel):
        v: bytes
    assert Model(v=b'foobar').v == b'foobar'
    with pytest.raises(ValidationError):
        Model(v=bytearray('foobar', 'utf-8'))
    with pytest.raises(ValidationError):
        Model(v='foostring')
    with pytest.raises(ValidationError):
        Model(v=42)
    with pytest.raises(ValidationError):
        Model(v=0.42)

def test_constrained_bytes_strict_false() -> None:
    class Model(BaseModel):
        v: bytes
    assert Model(v=b'foobar').v == b'foobar'
    assert Model(v=bytearray('foobar', 'utf-8')).v == b'foobar'
    assert Model(v='foostring').v == b'foostring'
    with pytest.raises(ValidationError):
        Model(v=42)
    with pytest.raises(ValidationError):
        Model(v=0.42)

def test_constrained_bytes_strict_default() -> None:
    class Model(BaseModel):
        v: bytes
    assert Model(v=b'foobar').v == b'foobar'
    assert Model(v=bytearray('foobar', 'utf-8')).v == b'foobar'
    assert Model(v='foostring').v == b'foostring'
    with pytest.raises(ValidationError):
        Model(v=42)
    with pytest.raises(ValidationError):
        Model(v=0.42)

def test_constrained_list_good() -> None:
    class ConListModelMax(BaseModel):
        v: list[Any] = []
    m: ConListModelMax = ConListModelMax(v=[1, 2, 3])
    assert m.v == [1, 2, 3]

def test_constrained_list_default() -> None:
    class ConListModelMax(BaseModel):
        v: list[Any] = []
    m: ConListModelMax = ConListModelMax()
    assert m.v == []

def test_constrained_list_too_long() -> None:
    class ConListModelMax(BaseModel):
        v: list[Any] = []
    with pytest.raises(ValidationError) as exc_info:
        ConListModelMax(v=list((str(i) for i in range(11))))
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_long', 'loc': ('v',), 'msg': 'List should have at most 10 items after validation, not 11', 'input': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 'ctx': {'field_type': 'List', 'max_length': 10, 'actual_length': 11}}]

def test_constrained_list_too_short() -> None:
    class ConListModelMin(BaseModel):
        v: list[Any]
    with pytest.raises(ValidationError) as exc_info:
        ConListModelMin(v=[])
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_short', 'loc': ('v',), 'msg': 'List should have at least 1 item after validation, not 0', 'input': [], 'ctx': {'field_type': 'List', 'min_length': 1, 'actual_length': 0}}]

def test_constrained_list_optional() -> None:
    class Model(BaseModel):
        opt: Optional[list[Any]] = None
    assert Model(req=None).model_dump() == {'req': None, 'opt': None}
    assert Model(req=None, opt=None).model_dump() == {'req': None, 'opt': None}
    with pytest.raises(ValidationError) as exc_info:
        Model(req=[], opt=[])
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_short', 'loc': ('req',), 'msg': 'List should have at least 1 item after validation, not 0', 'input': [], 'ctx': {'field_type': 'List', 'min_length': 1, 'actual_length': 0}}, {'type': 'too_short', 'loc': ('opt',), 'msg': 'List should have at least 1 item after validation, not 0', 'input': [], 'ctx': {'field_type': 'List', 'min_length': 1, 'actual_length': 0}}]
    assert Model(req=['a'], opt=['a']).model_dump() == {'req': ['a'], 'opt': ['a']}

def test_constrained_list_constraints() -> None:
    class ConListModelBoth(BaseModel):
        v: list[Any]
    m: ConListModelBoth = ConListModelBoth(v=list(range(7)))
    assert m.v == list(range(7))
    m = ConListModelBoth(v=list(range(11)))
    assert m.v == list(range(11))
    with pytest.raises(ValidationError) as exc_info:
        ConListModelBoth(v=list(range(6)))
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_short', 'loc': ('v',), 'msg': 'List should have at least 7 items after validation, not 6', 'input': [0, 1, 2, 3, 4, 5], 'ctx': {'field_type': 'List', 'min_length': 7, 'actual_length': 6}}]
    with pytest.raises(ValidationError) as exc_info:
        ConListModelBoth(v=list(range(12)))
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_long', 'loc': ('v',), 'msg': 'List should have at most 11 items after validation, not 12', 'input': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'ctx': {'field_type': 'List', 'max_length': 11, 'actual_length': 12}}]
    with pytest.raises(ValidationError) as exc_info:
        ConListModelBoth(v=1)
    assert exc_info.value.errors(include_url=False) == [{'type': 'list_type', 'loc': ('v',), 'msg': 'Input should be a valid list', 'input': 1}]

def test_constrained_list_item_type_fails() -> None:
    class ConListModel(BaseModel):
        v: list[int] = []
    with pytest.raises(ValidationError) as exc_info:
        ConListModel(v=['a', 'b', 'c'])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': ('v', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'a'}, {'type': 'int_parsing', 'loc': ('v', 1), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'b'}, {'type': 'int_parsing', 'loc': ('v', 2), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'c'}]

def test_conlist() -> None:
    class Model(BaseModel):
        foo: list[int] = Field(min_length=2, max_length=4)
        bar: list[str] = None
    assert Model(foo=[1, 2], bar=['spoon']).model_dump() == {'foo': [1, 2], 'bar': ['spoon']}
    msg = 'List should have at least 2 items after validation, not 1 \\[type=too_short,'
    with pytest.raises(ValidationError, match=msg):
        Model(foo=[1])
    msg = 'List should have at most 4 items after validation, not 5 \\[type=too_long,'
    with pytest.raises(ValidationError, match=msg):
        Model(foo=list(range(5)))
    with pytest.raises(ValidationError) as exc_info:
        Model(foo=[1, 'x', 'y'])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': ('foo', 1), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}, {'type': 'int_parsing', 'loc': ('foo', 2), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'y'}]
    with pytest.raises(ValidationError) as exc_info:
        Model(foo=1)
    assert exc_info.value.errors(include_url=False) == [{'type': 'list_type', 'loc': ('foo',), 'msg': 'Input should be a valid list', 'input': 1}]

def test_conlist_wrong_type_default() -> None:
    class Model(BaseModel):
        v: str = 'a'
    m: Model = Model()
    assert m.v == 'a'

def test_constrained_set_good() -> None:
    class Model(BaseModel):
        v: set[int] = []
    m: Model = Model(v=[1, 2, 3])
    assert m.v == {1, 2, 3}

def test_constrained_set_default() -> None:
    class Model(BaseModel):
        v: set[Any] = set()
    m: Model = Model()
    assert m.v == set()

def test_constrained_set_default_invalid() -> None:
    class Model(BaseModel):
        v: str = 'not valid, not validated'
    m: Model = Model()
    assert m.v == 'not valid, not validated'

def test_constrained_set_too_long() -> None:
    class ConSetModelMax(BaseModel):
        v: set[Any] = []
    with pytest.raises(ValidationError) as exc_info:
        ConSetModelMax(v={str(i) for i in range(11)})
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_long', 'loc': ('v',), 'msg': 'Set should have at most 10 items after validation, not more', 'input': {'4', '3', '10', '9', '5', '6', '1', '8', '0', '7', '2'}, 'ctx': {'field_type': 'Set', 'max_length': 10, 'actual_length': None}}]

def test_constrained_set_too_short() -> None:
    class ConSetModelMin(BaseModel):
        v: set[Any]
    with pytest.raises(ValidationError) as exc_info:
        ConSetModelMin(v=[])
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_short', 'loc': ('v',), 'msg': 'Set should have at least 1 item after validation, not 0', 'input': [], 'ctx': {'field_type': 'Set', 'min_length': 1, 'actual_length': 0}}]

def test_constrained_set_optional() -> None:
    class Model(BaseModel):
        opt: Optional[set[Any]] = None
    assert Model(req=None).model_dump() == {'req': None, 'opt': None}
    assert Model(req=None, opt=None).model_dump() == {'req': None, 'opt': None}
    with pytest.raises(ValidationError) as exc_info:
        Model(req=set(), opt=set())
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_short', 'loc': ('req',), 'msg': 'Set should have at least 1 item after validation, not 0', 'input': set(), 'ctx': {'field_type': 'Set', 'min_length': 1, 'actual_length': 0}}, {'type': 'too_short', 'loc': ('opt',), 'msg': 'Set should have at least 1 item after validation, not 0', 'input': set(), 'ctx': {'field_type': 'Set', 'min_length': 1, 'actual_length': 0}}]
    assert Model(req={'a'}, opt={'a'}).model_dump() == {'req': {'a'}, 'opt': {'a'}}

def test_constrained_set_constraints() -> None:
    class ConSetModelBoth(BaseModel):
        v: set[int]
    m: ConSetModelBoth = ConSetModelBoth(v=set(range(7)))
    assert m.v == set(range(7))
    m = ConSetModelBoth(v=set(range(11)))
    assert m.v == set(range(11))
    with pytest.raises(ValidationError) as exc_info:
        ConSetModelBoth(v=set(range(6)))
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_short', 'loc': ('v',), 'msg': 'Set should have at least 7 items after validation, not 6', 'input': {0, 1, 2, 3, 4, 5}, 'ctx': {'field_type': 'Set', 'min_length': 7, 'actual_length': 6}}]
    with pytest.raises(ValidationError) as exc_info:
        ConSetModelBoth(v=set(range(12)))
    assert exc_info.value.errors(include_url=False) == [{'type': 'too_long', 'loc': ('v',), 'msg': 'Set should have at most 11 items after validation, not more', 'input': {0, 8, 1, 9, 2, 10, 3, 7, 11, 4, 6, 5}, 'ctx': {'field_type': 'Set', 'max_length': 11, 'actual_length': None}}]
    with pytest.raises(ValidationError) as exc_info:
        ConSetModelBoth(v=1)
    assert exc_info.value.errors(include_url=False) == [{'type': 'set_type', 'loc': ('v',), 'msg': 'Input should be a valid set', 'input': 1}]

def test_constrained_set_item_type_fails() -> None:
    class ConSetModel(BaseModel):
        v