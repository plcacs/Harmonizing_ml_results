from __future__ import annotations
import sys
from collections.abc import Callable, Iterable
from dataclasses import is_dataclass
from types import FrameType
from typing import Any, Generic, Literal, TypeVar, cast, final, overload
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

def _getattr_no_parents(obj, attribute) -> Any:
    ...

def _type_has_config(type_) -> bool:
    ...

@final
class TypeAdapter(Generic[T]):
    def __init__(self, type: Any, *, config: ConfigDict = ..., _parent_depth: int = ..., module: str = ...) -> None:
        ...

    def _fetch_parent_frame(self) -> FrameType:
        ...

    def _init_core_attrs(self, ns_resolver, force, raise_errors=False) -> bool:
        ...

    @property
    def _defer_build(self) -> bool:
        ...

    @property
    def _model_config(self) -> Any:
        ...

    def __repr__(self) -> str:
        ...

    def rebuild(self, *, force: bool = ..., raise_errors: bool = ..., _parent_namespace_depth: int = ..., _types_namespace: Any = ...) -> bool:
        ...

    def validate_python(self, object, /, *, strict: bool = ..., from_attributes: Any = ..., context: Any = ..., experimental_allow_partial: bool = ...) -> Any:
        ...

    def validate_json(self, data, /, *, strict: bool = ..., context: Any = ..., experimental_allow_partial: bool = ...) -> Any:
        ...

    def validate_strings(self, obj, /, *, strict: bool = ..., context: Any = ..., experimental_allow_partial: bool = ...) -> Any:
        ...

    def get_default_value(self, *, strict: bool = ..., context: Any = ...) -> Some:
        ...

    def dump_python(self, instance, /, *, mode: str = ..., include: Any = ..., exclude: Any = ..., by_alias: bool = ..., exclude_unset: bool = ..., exclude_defaults: bool = ..., exclude_none: bool = ..., round_trip: bool = ..., warnings: Any = ..., fallback: Any = ..., serialize_as_any: bool = ..., context: Any = ...) -> Any:
        ...

    def dump_json(self, instance, /, *, indent: int = ..., include: Any = ..., exclude: Any = ..., by_alias: bool = ..., exclude_unset: bool = ..., exclude_defaults: bool = ..., exclude_none: bool = ..., round_trip: bool = ..., warnings: Any = ..., fallback: Any = ..., serialize_as_any: bool = ..., context: Any = ...) -> Any:
        ...

    def json_schema(self, *, by_alias: bool = ..., ref_template: str = ..., schema_generator: GenerateJsonSchema = ..., mode: str = ...) -> Any:
        ...

    @staticmethod
    def json_schemas(inputs, /, *, by_alias: bool = ..., title: str = ..., description: str = ..., ref_template: str = ..., schema_generator: GenerateJsonSchema = ...) -> Any:
        ...
