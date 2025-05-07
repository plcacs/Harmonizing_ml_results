"""This module contains related classes and functions for serialization."""
from __future__ import annotations
import dataclasses
from functools import partial, partialmethod
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    Type,
    TypeVar,
    overload,
)
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core.core_schema import (
    SerializationInfo,
    SerializerFunctionWrapHandler,
    WhenUsed,
)
from typing_extensions import TypeAlias
from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler

if TYPE_CHECKING:
    from functools import partial, partialmethod
    from typing import Generic, Union

@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class PlainSerializer:
    """Plain serializers use a function to modify the output of serialization.

    This is particularly helpful when you want to customize the serialization for annotated types.
    Consider an input of `list`, which will be serialized into a space-delimited string.

    