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
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    NewType,
    Optional,
    TypeVar,
    Union,
)
from uuid import UUID

import annotated_types
import dirty_equals
import pytest
from dirty_equals import HasRepr, IsFloatNan, IsOneOf, IsStr
from pydantic_core import (
    CoreSchema,
    PydanticCustomError,
    SchemaError,
    core_schema,
)
from typing_extensions import NotRequired, TypedDict, get_args

from pydantic import (
    UUID1,
    UUID3,
    UUID4,
    UUID5,
    AfterValidator,
    AllowInfNan,
    AwareDatetime,
    Base64Bytes,
    Base64Str,
    Base64UrlBytes,
    Base64UrlStr,
    BaseModel,
    BeforeValidator,
    ByteSize,
    ConfigDict,
    DirectoryPath,
    EmailStr,
    FailFast,
    Field,
    FilePath,
    FiniteFloat,
    FutureDate,
    FutureDatetime,
    GetCoreSchemaHandler,
    GetPydanticSchema,
    ImportString,
    InstanceOf,
    Json,
    JsonValue,
    NaiveDatetime,
    NameEmail,
    NegativeFloat,
    NegativeInt,
    NewPath,
    NonNegativeFloat,
    NonNegativeInt,
    NonPositiveFloat,
    NonPositiveInt,
    OnErrorOmit,
    PastDate,
    PastDatetime,
    PlainSerializer,
    PositiveFloat,
    PositiveInt,
    PydanticInvalidForJsonSchema,
    PydanticSchemaGenerationError,
    Secret,
    SecretBytes,
    SecretStr,
    SerializeAsAny,
    SkipValidation,
    SocketPath,
    Strict,
    StrictBool,
    StrictBytes,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    Tag,
    TypeAdapter,
    ValidationError,
    conbytes,
    condate,
    condecimal,
    confloat,
    confrozenset,
    conint,
    conlist,
    conset,
    constr,
    field_serializer,
    field_validator,
    validate_call,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

try:
    import email_validator
except ImportError:
    email_validator = None


# TODO add back tests for Iterator


@pytest.fixture(scope='session', name='ConBytesModel')
def con_bytes_model_fixture() -> typing.Generator[type[BaseModel], None, None]:
    class ConBytesModel(BaseModel):
        v: conbytes(max_length=10) = b'foobar'

    return ConBytesModel


def test_constrained_bytes_good(ConBytesModel: type[BaseModel]) -> None:
    m = ConBytesModel(v=b'short')
    assert m.v == b'short'


def test_constrained_bytes_default(ConBytesModel: type[BaseModel]) -> None:
    m = ConBytesModel()
    assert m.v == b'foobar'


@pytest.mark.parametrize(
    ('data', 'valid'),
    [(b'this is too long', False), ('⪶⓲⽷01'.encode(), False), (b'not long90', True), ('⪶⓲⽷0'.encode(), True)],
)
def test_constrained_bytes_too_long(ConBytesModel: type[BaseModel], data: bytes, valid: bool) -> None:
    if valid:
        assert ConBytesModel(v=data).model_dump() == {'v': data}
    else:
        with pytest.raises(ValidationError) as exc_info:
            ConBytesModel(v=data)
        # insert_assert(exc_info.value.errors(include_url=False))
        assert exc_info.value.errors(include_url=False) == [
            {
                'ctx': {'max_length': 10},
                'input': data,
                'loc': ('v',),
                'msg': 'Data should have at most 10 bytes',
                'type': 'bytes_too_long',
            }
        ]


def test_constrained_bytes_strict_true() -> None:
    class Model(BaseModel):
        v: conbytes(strict=True)

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
        v: conbytes(strict=False)

    assert Model(v=b'foobar').v == b'foobar'
    assert Model(v=bytearray('foobar', 'utf-8')).v == b'foobar'
    assert Model(v='foostring').v == b'foostring'

    with pytest.raises(ValidationError):
        Model(v=42)

    with pytest.raises(ValidationError):
        Model(v=0.42)


def test_constrained_bytes_strict_default() -> None:
    class Model(BaseModel):
        v: conbytes()

    assert Model(v=b'foobar').v == b'foobar'
    assert Model(v=bytearray('foobar', 'utf-8')).v == b'foobar'
    assert Model(v='foostring').v == b'foostring'

    with pytest.raises(ValidationError):
        Model(v=42)

    with pytest.raises(ValidationError):
        Model(v=0.42)


def test_constrained_list_good() -> None:
    class ConListModelMax(BaseModel):
        v: conlist(int) = []

    m = ConListModelMax(v=[1, 2, 3])
    assert m.v == [1, 2, 3]


def test_constrained_list_default() -> None:
    class ConListModelMax(BaseModel):
        v: conlist(int) = []

    m = ConListModelMax()
    assert m.v == []


def test_constrained_list_too_long() -> None:
    class ConListModelMax(BaseModel):
        v: conlist(int, max_length=10) = []

    with pytest.raises(ValidationError) as exc_info:
        ConListModelMax(v=list(str(i) for i in range(11)))
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'too_long',
            'loc': ('v',),
            'msg': 'List should have at most 10 items after validation, not 11',
            'input': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'ctx': {'field_type': 'List', 'max_length': 10, 'actual_length': 11},
        }
    ]


def test_constrained_list_too_short() -> None:
    class ConListModelMin(BaseModel):
        v: conlist(int, min_length=1)

    with pytest.raises(ValidationError) as exc_info:
        ConListModelMin(v=[])
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'too_short',
            'loc': ('v',),
            'msg': 'List should have at least 1 item after validation, not 0',
            'input': [],
            'ctx': {'field_type': 'List', 'min_length': 1, 'actual_length': 0},
        }
    ]


def test_constrained_list_optional() -> None:
    class Model(BaseModel):
        req: Optional[conlist(str, min_length=1)]
        opt: Optional[conlist(str, min_length=1)] = None

    assert Model(req=None).model_dump() == {'req': None, 'opt': None}
    assert Model(req=None, opt=None).model_dump() == {'req': None, 'opt': None}

    with pytest.raises(ValidationError) as exc_info:
        Model(req=[], opt=[])
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'too_short',
            'loc': ('req',),
            'msg': 'List should have at least 1 item after validation, not 0',
            'input': [],
            'ctx': {'field_type': 'List', 'min_length': 1, 'actual_length': 0},
        },
        {
            'type': 'too_short',
            'loc': ('opt',),
            'msg': 'List should have at least 1 item after validation, not 0',
            'input': [],
            'ctx': {'field_type': 'List', 'min_length': 1, 'actual_length': 0},
        },
    ]

    assert Model(req=['a'], opt=['a']).model_dump() == {'req': ['a'], 'opt': ['a']}


def test_constrained_list_constraints() -> None:
    class ConListModelBoth(BaseModel):
        v: conlist(int, min_length=7, max_length=11)

    m = ConListModelBoth(v=list(range(7)))
    assert m.v == list(range(7))

    m = ConListModelBoth(v=list(range(11)))
    assert m.v == list(range(11))

    with pytest.raises(ValidationError) as exc_info:
        ConListModelBoth(v=list(range(6)))
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'too_short',
            'loc': ('v',),
            'msg': 'List should have at least 7 items after validation, not 6',
            'input': [0, 1, 2, 3, 4, 5],
            'ctx': {'field_type': 'List', 'min_length': 7, 'actual_length': 6},
        }
    ]

    with pytest.raises(ValidationError) as exc_info:
        ConListModelBoth(v=list(range(12)))
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'too_long',
            'loc': ('v',),
            'msg': 'List should have at most 11 items after validation, not 12',
            'input': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'ctx': {'field_type': 'List', 'max_length': 11, 'actual_length': 12},
        }
    ]

    with pytest.raises(ValidationError) as exc_info:
        ConListModelBoth(v=1)
    # insert_assert(exc_info.value.errors(include_url=False))
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'list_type',