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
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)
from uuid import UUID

import pytest
from dirty_equals import HasRepr
from packaging.version import Version
from pydantic_core import CoreSchema, SchemaValidator, core_schema, to_jsonable_python
from pydantic_core.core_schema import ValidatorFunctionWrapHandler
from typing_extensions import Self, TypeAliasType, TypedDict, deprecated

import pydantic
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ImportString,
    InstanceOf,
    PlainSerializer,
    PlainValidator,
    PydanticDeprecatedSince20,
    PydanticDeprecatedSince29,
    PydanticUserError,
    RootModel,
    ValidationError,
    WithJsonSchema,
    WrapValidator,
    computed_field,
    field_serializer,
    field_validator,
)
from pydantic.color import Color
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    Examples,
    GenerateJsonSchema,
    JsonSchemaValue,
    PydanticJsonSchemaWarning,
    SkipJsonSchema,
    model_json_schema,
    models_json_schema,
)
from pydantic.networks import (
    AnyUrl,
    EmailStr,
    IPvAnyAddress,
    IPvAnyInterface,
    IPvAnyNetwork,
    NameEmail,
    _CoreMultiHostUrl,
)
from pydantic.type_adapter import TypeAdapter
from pydantic.types import (
    UUID1,
    UUID3,
    UUID4,
    UUID5,
    ByteSize,
    DirectoryPath,
    FilePath,
    Json,
    NegativeFloat,
    NegativeInt,
    NewPath,
    NonNegativeFloat,
    NonNegativeInt,
    NonPositiveFloat,
    NonPositiveInt,
    PositiveFloat,
    PositiveInt,
    SecretBytes,
    SecretStr,
    StrictBool,
    StrictStr,
    StringConstraints,
    conbytes,
    condate,
    condecimal,
    confloat,
    conint,
    constr,
)

try:
    import email_validator
except ImportError:
    email_validator = None

T = TypeVar('T')

AnnBool = Annotated[
    bool,
    WithJsonSchema({}),
]
