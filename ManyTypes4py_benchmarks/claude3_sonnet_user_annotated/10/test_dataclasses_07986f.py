import dataclasses
import inspect
import pickle
import re
import sys
import traceback
from collections.abc import Hashable
from dataclasses import InitVar
from datetime import date, datetime
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    Dict,
    List,
    Type,
    Set,
    cast,
    overload,
)

import pytest
from dirty_equals import HasRepr
from pydantic_core import ArgsKwargs, SchemaValidator

import pydantic
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PydanticDeprecatedSince20,
    PydanticUndefinedAnnotation,
    PydanticUserError,
    RootModel,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
    with_config,
)
from pydantic._internal._mock_val_ser import MockValSer
from pydantic.dataclasses import is_pydantic_dataclass, rebuild_dataclass
from pydantic.fields import Field, FieldInfo
from pydantic.json_schema import model_json_schema
