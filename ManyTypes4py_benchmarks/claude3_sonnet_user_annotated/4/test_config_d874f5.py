import json
import re
import sys
from collections.abc import Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from decimal import Decimal
from inspect import signature
from typing import Annotated, Any, NamedTuple, Optional, Union, Dict, List, Type, Callable, TypeVar, cast

import pytest
from dirty_equals import HasRepr, IsPartialDict
from pydantic_core import SchemaError, SchemaSerializer, SchemaValidator

from pydantic import (
    BaseConfig,
    BaseModel,
    Field,
    PrivateAttr,
    PydanticDeprecatedSince20,
    PydanticSchemaGenerationError,
    ValidationError,
    create_model,
    field_validator,
    validate_call,
    with_config,
)
from pydantic._internal._config import ConfigWrapper, config_defaults
from pydantic._internal._generate_schema import GenerateSchema
from pydantic._internal._mock_val_ser import MockValSer
from pydantic._internal._typing_extra import get_type_hints
from pydantic.config import ConfigDict, JsonValue
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.dataclasses import rebuild_dataclass
from pydantic.errors import PydanticUserError
from pydantic.fields import ComputedFieldInfo, FieldInfo
from pydantic.type_adapter import TypeAdapter
from pydantic.warnings import PydanticDeprecatedSince210, PydanticDeprecationWarning

from .conftest import CallCounter
