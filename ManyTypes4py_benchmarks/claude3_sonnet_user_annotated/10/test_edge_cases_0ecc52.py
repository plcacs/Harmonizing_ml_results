import functools
import importlib.util
import re
import sys
import typing
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Annotated,
    Any,
    Callable,
    ForwardRef,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import pytest
from dirty_equals import HasRepr, IsStr
from pydantic_core import ErrorDetails, InitErrorDetails, PydanticSerializationError, PydanticUndefined, core_schema
from typing_extensions import TypeAliasType, TypedDict, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    PrivateAttr,
    PydanticDeprecatedSince20,
    PydanticSchemaGenerationError,
    PydanticUserError,
    RootModel,
    TypeAdapter,
    ValidationError,
    constr,
    errors,
    field_validator,
    model_validator,
    root_validator,
    validator,
)
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import Field, computed_field
from pydantic.functional_serializers import (
    field_serializer,
    model_serializer,
)
