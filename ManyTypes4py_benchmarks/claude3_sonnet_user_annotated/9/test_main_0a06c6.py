import json
import platform
import re
import sys
import warnings
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from functools import cache, cached_property, partial
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    get_type_hints,
)
from uuid import UUID, uuid4

import pytest
from pydantic_core import CoreSchema, core_schema

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    PrivateAttr,
    PydanticDeprecatedSince211,
    PydanticUndefinedAnnotation,
    PydanticUserError,
    SecretStr,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    constr,
    field_validator,
)
from pydantic._internal._generate_schema import GenerateSchema
from pydantic._internal._mock_val_ser import MockCoreSchema
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.v1 import BaseModel as BaseModelV1
