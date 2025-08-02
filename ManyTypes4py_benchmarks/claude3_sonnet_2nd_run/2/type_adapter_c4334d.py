"""Type adapter specification."""
from __future__ import annotations as _annotations
import sys
from collections.abc import Callable, Iterable
from dataclasses import is_dataclass
from types import FrameType
from typing import Any, Dict, Generic, Literal, Mapping, Optional, Tuple, Type, TypeVar, Union, cast, final, overload
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing_extensions import ParamSpec, is_typeddict
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel, IncEx
from ._internal import _config, _generate_schema, _mock_val_ser, _namespace_utils, _repr, _typing_extra, _utils
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaKeyT, JsonSchemaMode, JsonSchemaValue
from .plugin._schema_validator import PluggableSchemaValidator, create_schema_validator
T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')
TypeAdapterT = TypeVar('TypeAdapterT', bound='TypeAdapter')

def _getattr_no_parents(obj: Any, attribute: str) -> Any:
    """Returns the attribute value without attempting to look up attributes from parent types."""
    if hasattr(obj, '__dict__'):
        try:
            return obj.__dict__[attribute]
        except KeyError:
            pass
    slots = getattr(obj, '__slots__', None)
    if slots is not None and attribute in slots:
        return getattr(obj, attribute)
    else:
        raise AttributeError(attribute)

def _type_has_config(type_: Any) -> bool:
    """Returns whether the type has config."""
    type_ = _typing_extra.annotated_type(type_) or type_
    try:
        return issubclass(type_, BaseModel) or is_dataclass(type_) or is_typeddict(type_)
    except TypeError:
        return False

@final
class TypeAdapter(Generic[T]):
    """!!! abstract "Usage Documentation"
        [`TypeAdapter`](../concepts/type_adapter.md)

    Type adapters provide a flexible way to perform validation and serialization based on a Python type.

    A `TypeAdapter` instance exposes some of the functionality from `BaseModel` instance methods
    for types that do not have such methods (such as dataclasses, primitive types, and more).

    **Note:** `TypeAdapter` instances are not types, and cannot be used as type annotations for fields.

    Args:
        type: The type associated with the `TypeAdapter`.
        config: Configuration for the `TypeAdapter`, should be a dictionary conforming to
            [`ConfigDict`][pydantic.config.ConfigDict].

            !!! note
                You cannot provide a configuration when instantiating a `TypeAdapter` if the type you're using
                has its own config that cannot be overridden (ex: `BaseModel`, `TypedDict`, and `dataclass`). A
                [`type-adapter-config-unused`](../errors/usage_errors.md#type-adapter-config-unused) error will
                be raised in this case.
        _parent_depth: Depth at which to search for the [parent frame][frame-objects]. This frame is used when
            resolving forward annotations during schema building, by looking for the globals and locals of this
            frame. Defaults to 2, which will result in the frame where the `TypeAdapter` was instantiated.

            !!! note
                This parameter is named with an underscore to suggest its private nature and discourage use.
                It may be deprecated in a minor version, so we only recommend using it if you're comfortable
                with potential change in behavior/support. It's default value is 2 because internally,
                the `TypeAdapter` class makes another call to fetch the frame.
        module: The module that passes to plugin if provided.

    Attributes:
        core_schema: The core schema for the type.
        validator: The schema validator for the type.
        serializer: The schema serializer for the type.
        pydantic_complete: Whether the core schema for the type is successfully built.

    ??? tip "Compatibility with `mypy`"
        Depending on the type used, `mypy` might raise an error when instantiating a `TypeAdapter`. As a workaround, you can explicitly
        annotate your variable:

        