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
